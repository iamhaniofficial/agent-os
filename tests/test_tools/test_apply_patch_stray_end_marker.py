"""A stray ``*** End Patch`` before ``*** Begin Patch`` must not empty the patch (#2084).

``_parse_patch`` used to locate the two markers with independent searches from
the top of the text. A preamble that quoted ``*** End Patch`` therefore paired
the opening marker with a line *above* it, the body slice came back empty, and
``apply_patch`` reported "no changes" while silently dropping every operation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from agentos.tools.builtin import patch as patch_tool
from agentos.tools.builtin.patch import AddFile, DeleteFile, UpdateFile, _parse_patch
from agentos.tools.types import ToolContext, current_tool_context


def _original_async(fn: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
    return fn.__wrapped__.__wrapped__  # type: ignore[attr-defined, no-any-return]


_PATCH = """*** Begin Patch
*** Add File: sample.txt
+hello world
*** End Patch"""


def test_stray_end_marker_in_preamble_is_ignored() -> None:
    ops = _parse_patch("*** End Patch\n" + _PATCH)

    assert ops == [AddFile(path="sample.txt", content="hello world")]


def test_preamble_discussing_both_markers_is_ignored() -> None:
    preamble = "Every patch ends with the line\n*** End Patch\nand nothing else; here is mine:\n"
    ops = _parse_patch(preamble + _PATCH)

    assert ops == [AddFile(path="sample.txt", content="hello world")]


def test_stray_end_marker_does_not_truncate_multi_op_patch() -> None:
    patch_text = """*** End Patch
*** Begin Patch
*** Add File: a.txt
+a
*** Update File: b.txt
@@@ -1,1 +1,1 @@@
-old
+new
*** Delete File: c.txt
*** End Patch"""

    ops = _parse_patch(patch_text)

    assert [type(op) for op in ops] == [AddFile, UpdateFile, DeleteFile]


def test_first_end_marker_after_begin_closes_the_patch() -> None:
    ops = _parse_patch(_PATCH + "\n*** Add File: after.txt\n+dropped\n*** End Patch")

    assert ops == [AddFile(path="sample.txt", content="hello world")]


def test_end_marker_only_before_begin_is_reported_missing() -> None:
    with pytest.raises(ValueError, match=r"Missing '\*\*\* End Patch' marker"):
        _parse_patch("*** End Patch\n*** Begin Patch\n*** Add File: x.txt\n+x")


def test_missing_end_marker_is_still_reported() -> None:
    with pytest.raises(ValueError, match=r"Missing '\*\*\* End Patch' marker"):
        _parse_patch("*** Begin Patch\n*** Add File: x.txt\n+x")


@pytest.mark.asyncio
async def test_apply_patch_writes_the_file_despite_stray_end_marker(tmp_path: Path) -> None:
    token = current_tool_context.set(ToolContext(workspace_dir=str(tmp_path)))
    apply_patch = _original_async(patch_tool.apply_patch)
    try:
        result = await apply_patch("*** End Patch\n" + _PATCH)
    finally:
        current_tool_context.reset(token)

    assert result == "Applied patch: 1 file(s) added"
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "hello world"
