"""``grep_search`` / ``glob_search`` raise on a path that does not exist (#1802).

Both tools used to resolve the base path and search it without checking it
was there: ``Path.glob`` / ``Path.rglob`` on a missing directory simply yield
nothing, so a mistyped ``path`` produced ``No matches for '<pattern>'`` --
which reads, to the agent, as "this code does not exist anywhere". Every
other filesystem tool (``read_file``, ``list_dir``, ``edit_file``) raises
``FileNotFoundError("Path not found: ...")``; these tests pin the search
tools to the same contract.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from agentos.tools.builtin import filesystem as fs
from agentos.tools.types import CallerKind, ToolContext, current_tool_context


@contextmanager
def _tool_context(workspace: Path) -> Iterator[None]:
    token = current_tool_context.set(
        ToolContext(
            caller_kind=CallerKind.CLI,
            channel_kind="cli",
            channel_id="cli:test",
            workspace_dir=str(workspace),
            workspace_strict=True,
        )
    )
    try:
        yield
    finally:
        current_tool_context.reset(token)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("def handle_auth():\n    pass\n")
    return tmp_path


@pytest.mark.asyncio
async def test_grep_search_missing_directory_raises(repo: Path) -> None:
    missing = repo / "src" / "nonexistent_folder"
    with _tool_context(repo), pytest.raises(FileNotFoundError, match="Path not found"):
        await fs.grep_search("handle_auth", path=str(missing))


@pytest.mark.asyncio
async def test_grep_search_missing_file_raises(repo: Path) -> None:
    missing = repo / "src" / "nope.py"
    with _tool_context(repo), pytest.raises(FileNotFoundError, match="Path not found"):
        await fs.grep_search("handle_auth", path=str(missing))


@pytest.mark.asyncio
async def test_glob_search_missing_directory_raises(repo: Path) -> None:
    missing = repo / "src" / "nonexistent_folder"
    with _tool_context(repo), pytest.raises(FileNotFoundError, match="Path not found"):
        await fs.glob_search("*.py", path=str(missing))


@pytest.mark.asyncio
async def test_error_names_the_path_as_given(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The message echoes the caller's spelling, not the resolved absolute path."""
    monkeypatch.chdir(repo)
    with _tool_context(repo), pytest.raises(FileNotFoundError) as excinfo:
        await fs.grep_search("handle_auth", path="src/nonexistent_folder")
    assert str(excinfo.value) == "Path not found: src/nonexistent_folder"


@pytest.mark.asyncio
async def test_existing_paths_still_search_and_report_no_matches(repo: Path) -> None:
    with _tool_context(repo):
        grep_out = await fs.grep_search("handle_auth", path=str(repo / "src"))
        assert grep_out.endswith("auth.py:1: def handle_auth():")
        assert await fs.grep_search("does_not_appear", path=str(repo / "src")) == (
            "No matches for 'does_not_appear'"
        )
        glob_out = await fs.glob_search("*.py", path=str(repo / "src"))
        assert glob_out == str(repo / "src" / "auth.py")
        no_glob = await fs.glob_search("*.rs", path=str(repo / "src"))
        assert no_glob.startswith("No files matched pattern '*.rs'")


@pytest.mark.asyncio
async def test_default_base_is_not_affected(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no ``path`` the base is the workspace/cwd, which always exists."""
    monkeypatch.chdir(repo)
    with _tool_context(repo):
        assert "auth.py" in await fs.grep_search("handle_auth")
        assert "auth.py" in await fs.glob_search("**/*.py")
