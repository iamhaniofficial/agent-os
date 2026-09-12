"""Issue #1580: a CSV cell over ``csv.field_size_limit()`` must not escape
``read_spreadsheet`` as a raw ``_csv.Error``.

The limit is process-global, so the fix raises it only as far as the file's
own length, only for the duration of the parse, and puts it back afterwards
-- another CSV consumer in the same process must never observe a changed
default once the read is over.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path

import pytest

from agentos.tools.builtin.filesystem import _read_delimited_rows, read_spreadsheet
from agentos.tools.types import ToolError

_DEFAULT_LIMIT = 131072


@pytest.fixture(autouse=True)
def _default_field_limit():
    """Pin the process-wide limit so the restore assertions are meaningful."""
    previous = csv.field_size_limit(_DEFAULT_LIMIT)
    yield
    csv.field_size_limit(previous)


def _write_large_field_csv(path: Path, cell_len: int) -> str:
    payload = "A" * cell_len
    path.write_text("id,payload\n1," + payload + "\n", encoding="utf-8")
    return payload


def test_field_over_the_default_limit_is_parsed(tmp_path: Path) -> None:
    csv_file = tmp_path / "large_field.csv"
    payload = _write_large_field_csv(csv_file, _DEFAULT_LIMIT + 1)

    [(_, rows, total)] = _read_delimited_rows(csv_file, ",")

    assert total == 2
    assert rows[1] == ["id", "payload"]
    assert rows[2] == ["1", payload]


def test_quoted_multiline_field_over_the_limit_is_parsed(tmp_path: Path) -> None:
    """Embedded JSON / log blobs are the realistic trigger: quoted, with
    newlines and delimiters inside."""
    csv_file = tmp_path / "blob.csv"
    body = ('{"k": "v",\n' * ((_DEFAULT_LIMIT // 10) + 1)) + "}"
    quoted = body.replace('"', '""')
    csv_file.write_text('id,blob\n1,"' + quoted + '"\n', encoding="utf-8")

    [(_, rows, total)] = _read_delimited_rows(csv_file, ",")

    assert total == 2
    assert rows[2] == ["1", body]


@pytest.mark.asyncio
async def test_read_spreadsheet_tool_returns_the_large_cell(tmp_path: Path) -> None:
    """The exact reproduction from the report, through the tool boundary."""
    csv_file = tmp_path / "large_field.csv"
    payload = _write_large_field_csv(csv_file, _DEFAULT_LIMIT + 1)

    out = await read_spreadsheet(str(csv_file))

    assert "Sheet: large_field.csv (2 rows x 2 columns)" in out
    assert f"2\t1\t{payload}" in out


def test_process_wide_limit_is_restored_after_the_read(tmp_path: Path) -> None:
    csv_file = tmp_path / "large_field.csv"
    _write_large_field_csv(csv_file, _DEFAULT_LIMIT + 1)

    _read_delimited_rows(csv_file, ",")

    assert csv.field_size_limit() == _DEFAULT_LIMIT


def test_process_wide_limit_is_restored_after_a_parse_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_file = tmp_path / "large_field.csv"
    _write_large_field_csv(csv_file, _DEFAULT_LIMIT + 1)

    def _boom(*_args: object, **_kwargs: object):
        raise csv.Error("boom")

    monkeypatch.setattr(csv, "reader", _boom)

    with pytest.raises(ToolError, match="Cannot parse large_field.csv"):
        _read_delimited_rows(csv_file, ",")
    assert csv.field_size_limit() == _DEFAULT_LIMIT


def test_small_files_never_touch_the_process_wide_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Below the default there is nothing to raise; the common path never
    mutates the process-wide value."""
    csv_file = tmp_path / "small.csv"
    csv_file.write_text("a,b\n1,2\n", encoding="utf-8")
    calls: list[int] = []
    real = csv.field_size_limit

    def _spy(*args: int) -> int:
        if args:
            calls.append(args[0])
        return real(*args)

    monkeypatch.setattr(csv, "field_size_limit", _spy)

    [(_, rows, _)] = _read_delimited_rows(csv_file, ",")

    assert rows[2] == ["1", "2"]
    assert calls == []


def test_limit_is_capped_at_the_file_length_not_sys_maxsize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_file = tmp_path / "large_field.csv"
    _write_large_field_csv(csv_file, _DEFAULT_LIMIT + 1)
    size = len(csv_file.read_text(encoding="utf-8-sig"))
    seen: list[int] = []
    real = csv.field_size_limit

    def _spy(*args: int) -> int:
        if args:
            seen.append(args[0])
        return real(*args)

    monkeypatch.setattr(csv, "field_size_limit", _spy)

    _read_delimited_rows(csv_file, ",")

    assert seen, "expected the limit to be raised for an over-limit field"
    assert max(seen) <= size + 1


def test_concurrent_large_reads_restore_the_default(tmp_path: Path) -> None:
    """Executor threads read spreadsheets concurrently; two raise/restore
    pairs racing each other must not leave the limit raised or drop it
    under a parse still in flight."""
    files = []
    for i in range(4):
        f = tmp_path / f"large_{i}.csv"
        _write_large_field_csv(f, _DEFAULT_LIMIT + 1 + i)
        files.append(f)
    errors: list[BaseException] = []

    def _worker(path: Path) -> None:
        try:
            for _ in range(5):
                [(_, rows, _)] = _read_delimited_rows(path, ",")
                assert len(rows[2][1]) > _DEFAULT_LIMIT
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(f,)) for f in files]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert csv.field_size_limit() == _DEFAULT_LIMIT
