"""``git-diff`` bundled script is byte-exact on non-UTF-8 code pages (#1834).

The agent runs this script through the shell tool, so stdout is a pipe and
Python picks the locale code page (cp936, cp1252, plain ASCII under ``LC_ALL=C``)
for it. The script decoded git's bytes with that code page and re-encoded them
with it on the way out, so any non-ASCII hunk died with ``UnicodeEncodeError``
or ``UnicodeDecodeError`` and exit 1 -- no diff, no ``NO_DIFF``. The contract
is that the diff reaches stdout unmodified, so git's bytes are now passed
through untouched.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "src" / "agentos" / "skills" / "bundled" / "git-diff" / "scripts"

_CJK_EMOJI = "你好世界 \U0001f389\n"


def _git_diff_module() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    try:
        import git_diff  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return git_diff


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", ".")
    _git(tmp_path, "config", "user.email", "t@t.t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "f.txt").write_bytes(b"hello\n")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def _code_page_stdout(encoding: str) -> io.TextIOWrapper:
    """A piped stdout whose text layer cannot encode CJK or emoji."""
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding, newline="")


def _stdout_bytes(stream: io.TextIOWrapper) -> bytes:
    stream.flush()
    return stream.buffer.getvalue()  # type: ignore[attr-defined]


@pytest.mark.parametrize("code_page", ["cp936", "cp1252", "ascii"])
def test_non_ascii_diff_reaches_stdout_on_any_code_page(
    repo: Path, monkeypatch: pytest.MonkeyPatch, code_page: str
) -> None:
    git_diff = _git_diff_module()
    (repo / "f.txt").write_text(_CJK_EMOJI, encoding="utf-8")
    _git(repo, "add", "f.txt")
    stdout = _code_page_stdout(code_page)
    monkeypatch.setattr(sys, "stdout", stdout)

    rc = git_diff.main(["--cwd", str(repo)])

    assert rc == 0
    out = _stdout_bytes(stdout)
    assert b"+" + _CJK_EMOJI.encode("utf-8") in out
    assert out.startswith(b"diff --git")


def test_git_output_is_not_decoded_with_the_locale(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Latin-1 file is not UTF-8; its bytes must still pass through verbatim.

    Decoding git's output with ``text=True`` used the locale code page, which
    raised inside ``subprocess`` under ``LC_ALL=C`` before the script's own
    code ran -- and would mangle any non-UTF-8 file under a UTF-8 locale.
    """
    git_diff = _git_diff_module()
    (repo / "f.txt").write_bytes(b"caf\xe9\n")
    _git(repo, "add", "f.txt")
    stdout = _code_page_stdout("ascii")
    monkeypatch.setattr(sys, "stdout", stdout)

    rc = git_diff.main(["--cwd", str(repo)])

    assert rc == 0
    assert b"+caf\xe9\n" in _stdout_bytes(stdout)


def test_no_diff_sentinel_is_written_on_a_code_page_stdout(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_diff = _git_diff_module()
    stdout = _code_page_stdout("cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)

    rc = git_diff.main(["--cwd", str(repo)])

    assert rc == 0
    assert _stdout_bytes(stdout) == b"NO_DIFF"


def test_git_error_reaches_stderr_on_a_code_page_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_diff = _git_diff_module()
    stderr = _code_page_stdout("ascii")
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(sys, "stdout", _code_page_stdout("ascii"))

    rc = git_diff.main(["--cwd", str(tmp_path), "--mode", "cached"])

    assert rc != 0
    assert _stdout_bytes(stderr)


def test_text_only_stdout_falls_back_to_escapes_instead_of_crashing(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stdout with no ``buffer`` (a ``StringIO`` harness) still gets the diff."""
    git_diff = _git_diff_module()
    (repo / "f.txt").write_text(_CJK_EMOJI, encoding="utf-8")
    _git(repo, "add", "f.txt")
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)

    rc = git_diff.main(["--cwd", str(repo)])

    assert rc == 0
    assert "+你好世界" in stdout.getvalue()
