#!/usr/bin/env python3
"""Direct shell wrapper around ``git diff`` — meta-skill entrypoint.

Returns the diff text on stdout, the literal ``NO_DIFF`` when the
diff is empty, and exits non-zero with the git error on stderr when
git itself fails (not a repo, missing binary, etc.).

Used by workflows that need repository diffs while skipping a full
sub-Agent loop just to call ``git diff``.

git's output is passed through as bytes and written to the binary stdout
buffer: stdout is a pipe when the agent runs this, so Python would otherwise
decode and re-encode the diff with the locale code page (cp936, cp1252,
ASCII under ``LC_ALL=C``) and any non-ASCII hunk would die with a
``UnicodeEncodeError`` instead of reaching the caller.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import TextIO

_VALID_MODES = {
    "cached_fallback_worktree",
    "cached",
    "worktree",
    "staged_files",
}


def _run_git(args: list[str], cwd: Path) -> tuple[int, bytes, bytes]:
    # Bytes in, bytes out: ``text=True`` would decode with the locale code
    # page, which raises inside subprocess under a C/ASCII locale and would
    # mangle a diff of any file that is not in that code page.
    proc = subprocess.run(  # noqa: S603 — argv is constructed from a static allowlist
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _write_bytes(stream: TextIO, data: bytes) -> None:
    """Write ``data`` verbatim, bypassing the stream's text encoder.

    Falls back to the text layer with ``backslashreplace`` when the stream has
    no binary buffer (a ``StringIO`` harness), so nothing is silently lost.
    """
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        try:
            buffer.write(data)
            buffer.flush()
            return
        except (AttributeError, OSError, ValueError):
            pass
    stream.write(data.decode("utf-8", errors="backslashreplace"))
    stream.flush()


def _diff_for_mode(mode: str, cwd: Path) -> tuple[int, bytes, bytes]:
    if mode == "cached_fallback_worktree":
        rc, out, err = _run_git(["diff", "--cached", "HEAD"], cwd)
        if rc != 0:
            return rc, out, err
        if out.strip():
            return 0, out, err
        return _run_git(["diff", "HEAD"], cwd)
    if mode == "cached":
        return _run_git(["diff", "--cached", "HEAD"], cwd)
    if mode == "worktree":
        return _run_git(["diff", "HEAD"], cwd)
    if mode == "staged_files":
        return _run_git(["diff", "--cached", "--name-only"], cwd)
    raise ValueError(f"unsupported mode {mode!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="cached_fallback_worktree")
    parser.add_argument("--cwd", default=".")
    args = parser.parse_args(argv)

    if args.mode not in _VALID_MODES:
        print(
            f"unsupported mode {args.mode!r}; valid: {sorted(_VALID_MODES)!r}",
            file=sys.stderr,
        )
        return 2

    cwd = Path(args.cwd).expanduser().resolve()
    if not cwd.is_dir():
        print(f"cwd does not exist: {cwd}", file=sys.stderr)
        return 2

    try:
        rc, out, err = _diff_for_mode(args.mode, cwd)
    except FileNotFoundError as exc:
        print(f"git binary not found: {exc}", file=sys.stderr)
        return 1

    if rc != 0:
        _write_bytes(sys.stderr, err)
        return rc

    _write_bytes(sys.stdout, out if out.strip() else b"NO_DIFF")
    return 0


if __name__ == "__main__":
    sys.exit(main())
