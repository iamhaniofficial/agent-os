"""Managed git worktrees for ACP coding sessions.

``sessions_spawn(runtime="acp", worktree=true)`` gives the harness its own
checkout so it never edits the operator's working tree. Layout (modelled on
OpenClaw's ``docs/concepts/managed-worktrees.md``, MIT):

    <acp state>/worktrees/<repo-fingerprint>/<name>/          the checkout
    <acp state>/worktrees/<repo-fingerprint>/<name>.json      our metadata

``repo-fingerprint`` is the first 12 hex chars of sha256(realpath(repo root)),
so two clones of the same project never collide. A worktree with uncommitted
changes is never removed unless ``force`` is passed (no stash juggling). The
total is capped (4096 by default); when full, the oldest worktree whose
branch is already merged goes first, then the oldest clean one.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("agentos.acp.worktree")

DEFAULT_WORKTREE_CAP = 4096
_GIT_TIMEOUT_S = 60.0
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")


class WorktreeError(RuntimeError):
    code = "ACP_WORKTREE_FAILED"


class WorktreeDirtyError(WorktreeError):
    code = "ACP_WORKTREE_DIRTY"


@dataclass(slots=True)
class WorktreeInfo:
    name: str
    branch: str
    path: str
    repo_root: str
    fingerprint: str
    base_ref: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def worktrees_root(acp_state_dir: str | os.PathLike[str]) -> Path:
    return Path(acp_state_dir) / "worktrees"


def repo_fingerprint(repo_root: str | os.PathLike[str]) -> str:
    real = os.path.realpath(os.fspath(repo_root))
    return hashlib.sha256(real.encode("utf-8")).hexdigest()[:12]


def validate_worktree_name(name: str) -> str:
    """Return ``name`` if it is a safe branch + directory name, else raise."""
    value = (name or "").strip()
    if (
        not _NAME_RE.match(value)
        or ".." in value
        or value.endswith((".lock", "/", "."))
        or "//" in value
        or "@{" in value
    ):
        raise WorktreeError(
            f"Invalid worktree name {name!r}: use letters, digits, '.', '_', '-', '/' "
            "(max 100 chars, no '..')"
        )
    return value


def _dir_name(name: str) -> str:
    return name.replace("/", "-")


async def _git(*args: str, cwd: str | os.PathLike[str]) -> tuple[int, str, str]:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=os.fspath(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=_GIT_TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise WorktreeError(f"git {' '.join(args[:2])} timed out") from None
    return (
        proc.returncode if proc.returncode is not None else -1,
        out.decode("utf-8", "replace").strip(),
        err.decode("utf-8", "replace").strip(),
    )


async def repo_root(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if not p.is_dir():
        raise WorktreeError(f"Not a directory: {p}")
    rc, out, err = await _git("rev-parse", "--show-toplevel", cwd=p)
    if rc != 0 or not out:
        raise WorktreeError(f"Not inside a git repository: {p} ({err or 'rev-parse failed'})")
    return Path(out).resolve()


def _meta_path(info_path: Path) -> Path:
    return info_path.parent / f"{info_path.name}.json"


def _write_meta(info: WorktreeInfo) -> None:
    meta = _meta_path(Path(info.path))
    tmp = meta.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, meta)


def _read_meta(meta: Path) -> WorktreeInfo | None:
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        return WorktreeInfo(**{f.name: data[f.name] for f in fields(WorktreeInfo)})
    except (OSError, ValueError, KeyError, TypeError):
        return None


def list_worktrees(root: str | os.PathLike[str]) -> list[WorktreeInfo]:
    base = Path(root)
    if not base.is_dir():
        return []
    found: list[WorktreeInfo] = []
    for fp_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for meta in sorted(fp_dir.glob("*.json")):
            info = _read_meta(meta)
            if info is not None:
                found.append(info)
    return found


def find_worktree(root: str | os.PathLike[str], name_or_path: str) -> WorktreeInfo | None:
    target = name_or_path.strip()
    for info in list_worktrees(root):
        if target in (info.name, info.path, info.branch):
            return info
    return None


async def is_dirty(path: str | os.PathLike[str]) -> bool:
    rc, out, _err = await _git("status", "--porcelain", "--untracked-files=normal", cwd=path)
    if rc != 0:
        # Unknown state: treat as dirty so nothing gets thrown away.
        return True
    return bool(out.strip())


async def _merged_branches(repo: str) -> set[str]:
    rc, out, _err = await _git("branch", "--merged", "--format=%(refname:short)", cwd=repo)
    if rc != 0:
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


async def remove_worktree(
    info: WorktreeInfo, *, force: bool = False, delete_branch: bool = False
) -> None:
    path = Path(info.path)
    if path.exists() and not force and await is_dirty(path):
        raise WorktreeDirtyError(
            f"Worktree {info.name} has uncommitted changes; commit them or pass force"
        )
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(path))
    rc, _out, err = await _git(*args, cwd=info.repo_root)
    if rc != 0:
        if not force and path.exists():
            raise WorktreeError(f"git worktree remove failed: {err}")
        shutil.rmtree(path, ignore_errors=True)
        await _git("worktree", "prune", cwd=info.repo_root)
    if delete_branch:
        await _git("branch", "-D" if force else "-d", info.branch, cwd=info.repo_root)
    with_meta = _meta_path(path)
    with_meta.unlink(missing_ok=True)
    log.info("acp.worktree.removed", name=info.name, path=str(path), forced=force)


async def _evict(root: Path, *, cap: int) -> None:
    infos = list_worktrees(root)
    excess = len(infos) - cap + 1
    if excess <= 0:
        return
    merged_by_repo: dict[str, set[str]] = {}
    ranked: list[tuple[int, float, WorktreeInfo]] = []
    for info in infos:
        if info.repo_root not in merged_by_repo:
            merged_by_repo[info.repo_root] = await _merged_branches(info.repo_root)
        merged = info.branch in merged_by_repo[info.repo_root]
        ranked.append((0 if merged else 1, info.created_at, info))
    ranked.sort(key=lambda item: (item[0], item[1]))
    for _rank, _created, info in ranked:
        if excess <= 0:
            break
        if Path(info.path).exists() and await is_dirty(info.path):
            continue
        try:
            await remove_worktree(info, force=False)
        except WorktreeError:
            continue
        excess -= 1
    if excess > 0:
        raise WorktreeError(
            f"Managed worktree cap ({cap}) reached and every worktree has uncommitted changes"
        )


async def create_worktree(
    repo: str | os.PathLike[str],
    *,
    name: str,
    root: str | os.PathLike[str],
    base_ref: str | None = None,
    cap: int = DEFAULT_WORKTREE_CAP,
) -> WorktreeInfo:
    """``git worktree add -b <name> <path> <base_ref>`` under the managed root."""
    branch = validate_worktree_name(name)
    if base_ref is not None and (not base_ref.strip() or base_ref.strip().startswith("-")):
        raise WorktreeError(f"Invalid base ref {base_ref!r}")
    ref = (base_ref or "HEAD").strip()
    top = await repo_root(repo)
    fingerprint = repo_fingerprint(top)
    base = Path(root)
    path = base / fingerprint / _dir_name(branch)
    if path.exists() or _meta_path(path).exists():
        raise WorktreeError(f"Worktree {branch!r} already exists at {path}")
    await _evict(base, cap=cap)
    path.parent.mkdir(parents=True, exist_ok=True)
    rc, _out, err = await _git("worktree", "add", "-b", branch, str(path), ref, cwd=top)
    if rc != 0:
        raise WorktreeError(f"git worktree add failed: {err}")
    info = WorktreeInfo(
        name=branch,
        branch=branch,
        path=str(path.resolve()),
        repo_root=str(top),
        fingerprint=fingerprint,
        base_ref=ref,
        created_at=time.time(),
    )
    _write_meta(info)
    log.info("acp.worktree.created", name=branch, path=info.path, base_ref=ref)
    return info
