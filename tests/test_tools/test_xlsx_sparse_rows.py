"""Regression tests for issue #1149: sparse .xlsx rows keep their row numbers."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from agentos.tools.builtin import filesystem as fs
from agentos.tools.builtin.filesystem import (
    _format_spreadsheet,
    _read_xlsx_sheets,
    _read_xlsx_worksheet,
    _xlsx_row_index,
)

_SHEET_HEAD = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    "<sheetData>"
)
_SHEET_TAIL = "</sheetData></worksheet>"


def _row(number: str, column: str, text: str) -> str:
    return (
        f'<row r="{number}"><c r="{column}{number}" t="inlineStr"><is><t>{text}</t></is></c></row>'
    )


def _worksheet(*rows: str) -> bytes:
    return (_SHEET_HEAD + "".join(rows) + _SHEET_TAIL).encode("utf-8")


def _write_workbook(path: Path, worksheet: bytes) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
            ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>'
            "</workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"'
            ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>'
            "</Relationships>",
        )
        zf.writestr("xl/worksheets/sheet1.xml", worksheet)
    return path


# ── Row padding ─────────────────────────────────────────────────────────


def test_omitted_rows_are_padded_to_their_declared_row_number() -> None:
    rows = _read_xlsx_worksheet(_worksheet(_row("1", "A", "Header"), _row("5", "A", "Data")), [])

    assert rows == [["Header"], [], [], [], ["Data"]]


def test_dense_sheet_is_unchanged() -> None:
    rows = _read_xlsx_worksheet(
        _worksheet(_row("1", "A", "a"), _row("2", "A", "b"), _row("3", "A", "c")), []
    )

    assert rows == [["a"], ["b"], ["c"]]


def test_leading_gap_before_the_first_populated_row_is_padded() -> None:
    rows = _read_xlsx_worksheet(_worksheet(_row("3", "A", "Data")), [])

    assert rows == [[], [], ["Data"]]


def test_rows_without_a_usable_reference_fall_back_to_positional_append() -> None:
    worksheet = _worksheet(
        '<row><c r="A1" t="inlineStr"><is><t>first</t></is></c></row>',
        '<row r="not-a-number"><c r="A2" t="inlineStr"><is><t>second</t></is></c></row>',
    )

    assert _read_xlsx_worksheet(worksheet, []) == [["first"], ["second"]]


def test_out_of_range_row_reference_does_not_inflate_the_sheet() -> None:
    """A tiny file must not be able to allocate an absurd number of rows."""
    worksheet = _worksheet(
        _row("1", "A", "Header"),
        '<row r="99999999999"><c r="A2" t="inlineStr"><is><t>Data</t></is></c></row>',
    )

    assert _read_xlsx_worksheet(worksheet, []) == [["Header"], ["Data"]]


def test_row_index_bounds() -> None:
    assert _xlsx_row_index("1") == 1
    assert _xlsx_row_index("1048576") == 1_048_576
    assert _xlsx_row_index("1048577") is None
    assert _xlsx_row_index("0") is None
    assert _xlsx_row_index("-2") is None
    assert _xlsx_row_index("") is None
    assert _xlsx_row_index(" 3 ") is None


# ── Reported row numbers and pagination ─────────────────────────────────


def test_formatted_output_reports_the_real_row_numbers(tmp_path: Path) -> None:
    rows = _read_xlsx_worksheet(_worksheet(_row("1", "A", "Header"), _row("5", "A", "Data")), [])

    output = _format_spreadsheet(
        path=tmp_path / "test.xlsx", sheets=[("Sheet1", rows)], offset=1, limit=10
    )

    assert "Sheet: Sheet1 (5 rows x 1 columns)" in output
    assert "1\tHeader" in output
    assert "5\tData" in output
    assert "2\tData" not in output


def test_offset_reaches_a_row_past_a_gap(tmp_path: Path) -> None:
    rows = _read_xlsx_worksheet(_worksheet(_row("1", "A", "Header"), _row("5", "A", "Data")), [])

    output = _format_spreadsheet(
        path=tmp_path / "test.xlsx", sheets=[("Sheet1", rows)], offset=5, limit=10
    )

    assert "5\tData" in output


# ── End to end through the workbook reader ──────────────────────────────


@pytest.mark.asyncio
async def test_read_spreadsheet_keeps_row_numbers_for_a_sparse_workbook(tmp_path: Path) -> None:
    workbook = _write_workbook(
        tmp_path / "sparse.xlsx",
        _worksheet(_row("1", "A", "Header"), _row("5", "A", "Data")),
    )

    [(name, rows)] = _read_xlsx_sheets(workbook)
    assert name == "Sheet1"
    assert rows == [["Header"], [], [], [], ["Data"]]

    output = await fs.read_spreadsheet(str(workbook), offset=5)
    assert "5\tData" in output
