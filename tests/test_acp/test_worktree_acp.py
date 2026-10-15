"""Managed git worktrees (real git, temporary repositories, no network)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from agentos.acp.worktree import (
    WorktreeDirtyError,
    WorktreeError,
    create_worktree,
    find_worktree,
    list_worktrees,
    remove_worktree,
    repo_fingerprint,
    validate_worktree_name,
    worktrees_root,
)

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(cwd: Path, *args: str) -> str:
    env = {**os.environ, **_GIT_ENV}
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key, value in _GIT_ENV.items():
        monkeypatch.setenv(key, value)
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "a.txt").write_text("a\n")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-q", "-m", "init")
    return root


async def test_create_list_find_remove(repo: Path, tmp_path: Path) -> None:
    root = worktrees_root(tmp_path / "state")
    info = await create_worktree(repo / ".", name="feature/x", root=root)
    path = Path(info.path)
    assert path.is_dir() and (path / "a.txt").exists()
    assert path.parent.name == repo_fingerprint(repo) and path.name == "feature-x"
    assert info.branch == "feature/x" and info.base_ref == "HEAD"
    assert _git(path, "rev-parse", "--abbrev-ref", "HEAD") == "feature/x"
    meta = json.loads((path.parent / "feature-x.json").read_text())
    assert meta["repo_root"] == str(repo.resolve())

    assert [w.name for w in list_worktrees(root)] == ["feature/x"]
    assert find_worktree(root, "feature/x") == info
    assert find_worktree(root, info.path) == info

    await remove_worktree(info)
    assert not path.exists()
    assert list_worktrees(root) == []


async def test_subdirectory_resolves_to_repo_and_base_ref(repo: Path, tmp_path: Path) -> None:
    (repo / "pkg").mkdir()
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-q", "-m", "two")
    info = await create_worktree(repo / "pkg", name="old", root=tmp_path / "wt", base_ref=first)
    assert info.repo_root == str(repo.resolve())
    assert not (Path(info.path) / "b.txt").exists()


async def test_dirty_worktree_is_kept_unless_forced(repo: Path, tmp_path: Path) -> None:
    info = await create_worktree(repo, name="dirty", root=tmp_path / "wt")
    (Path(info.path) / "new.txt").write_text("uncommitted\n")
    with pytest.raises(WorktreeDirtyError):
        await remove_worktree(info)
    assert Path(info.path).exists()
    await remove_worktree(info, force=True, delete_branch=True)
    assert not Path(info.path).exists()
    assert "dirty" not in _git(repo, "branch", "--list", "dirty")


async def test_cap_evicts_merged_first_and_never_dirty(repo: Path, tmp_path: Path) -> None:
    root = tmp_path / "wt"
    merged = await create_worktree(repo, name="merged", root=root)
    unmerged = await create_worktree(repo, name="unmerged", root=root)
    (Path(unmerged.path) / "c.txt").write_text("c\n")
    _git(Path(unmerged.path), "add", "c.txt")
    _git(Path(unmerged.path), "commit", "-q", "-m", "ahead")
    # "merged" points at main's HEAD -> merged; "unmerged" is ahead -> not.
    third = await create_worktree(repo, name="third", root=root, cap=2)
    names = {w.name for w in list_worktrees(root)}
    assert names == {"unmerged", "third"}
    assert not Path(merged.path).exists()

    (Path(third.path) / "x.txt").write_text("dirty\n")
    (Path(unmerged.path) / "y.txt").write_text("dirty\n")
    with pytest.raises(WorktreeError, match="cap"):
        await create_worktree(repo, name="fourth", root=root, cap=2)


async def test_errors(repo: Path, tmp_path: Path) -> None:
    for bad in ("", "../x", "-x", "a..b", "x.lock", "a b", "x/"):
        with pytest.raises(WorktreeError):
            validate_worktree_name(bad)
    assert validate_worktree_name("acp-1234/fix_v2.1") == "acp-1234/fix_v2.1"
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(WorktreeError, match="Not inside a git repository"):
        await create_worktree(plain, name="x", root=tmp_path / "wt")
    await create_worktree(repo, name="dup", root=tmp_path / "wt")
    with pytest.raises(WorktreeError, match="already exists"):
        await create_worktree(repo, name="dup", root=tmp_path / "wt")
    with pytest.raises(WorktreeError, match="Invalid base ref"):
        await create_worktree(repo, name="y", root=tmp_path / "wt", base_ref="--upload-pack=x")
    with pytest.raises(WorktreeError, match="git worktree add failed"):
        await create_worktree(repo, name="z", root=tmp_path / "wt", base_ref="no-such-ref")


async def test_spawn_with_worktree(repo: Path, tmp_path: Path) -> None:
    from acp_s_harness import acp_rig

    from agentos.tools.builtin import sessions as sessions_tool

    async with acp_rig(tmp_path, allowed_cwds=[str(repo)]) as rig:
        out = json.loads(
            await sessions_tool.sessions_spawn(
                task="refactor",
                runtime="acp",
                acp_agent="claude",
                cwd=str(repo),
                worktree=True,
            )
        )
        wt = out["worktree"]
        assert wt["branch"].startswith("acp-") and Path(wt["path"]).is_dir()
        assert out["cwd"] == wt["path"]
        assert Path(wt["path"]).is_relative_to(tmp_path / "state" / "acp" / "worktrees")
        await rig.task_runtime.wait(out["task_id"], timeout=5)
        spec = rig.runtime.calls[0][1]
        assert spec.cwd == wt["path"]
        row = await rig.sessions.get_session(out["session_key"])
        assert row.origin["worktree"]["repo_root"] == str(repo.resolve())


async def test_failed_spawn_removes_its_worktree(repo: Path, tmp_path: Path) -> None:
    from acp_s_harness import acp_rig

    from agentos.tools.builtin import sessions as sessions_tool
    from agentos.tools.types import ToolError

    async def _no_children(_mgr, _agent):
        return {"max_children_per_session": 0}

    async with acp_rig(tmp_path, allowed_cwds=[str(repo)]):
        original = sessions_tool._resolve_subagent_policy
        sessions_tool._resolve_subagent_policy = _no_children  # type: ignore[assignment]
        try:
            with pytest.raises(ToolError, match="Max active children"):
                await sessions_tool.sessions_spawn(
                    task="x",
                    runtime="acp",
                    acp_agent="claude",
                    cwd=str(repo),
                    worktree=True,
                    worktree_name="gone",
                )
        finally:
            sessions_tool._resolve_subagent_policy = original  # type: ignore[assignment]
        assert list_worktrees(worktrees_root(tmp_path / "state" / "acp")) == []
        assert _git(repo, "branch", "--list", "gone") == ""
