"""Tests for memory context fencing (Task B2).

Ported case-for-case from hermes-agent ``memory_manager.py`` (MIT) fencing
semantics: ``sanitize_context``, ``build_memory_context_block``, and the
``StreamingContextScrubber`` state machine (split-tag chunk boundaries,
unterminated-span discard, partial-tag tail holdback, block-boundary gating).
"""
from __future__ import annotations

from agentos.memory.providers.fencing import (
    StreamingContextScrubber,
    build_memory_context_block,
    sanitize_context,
)

OPEN = "<memory-context>"
CLOSE = "</memory-context>"


# -- build_memory_context_block -----------------------------------------------


def test_block_wraps_with_fence_and_system_note() -> None:
    block = build_memory_context_block("user likes tea")
    assert block.startswith(OPEN)
    assert block.endswith(CLOSE)
    assert "[System note:" in block
    assert "NOT new user input" in block
    assert "user likes tea" in block


def test_block_empty_input_returns_empty() -> None:
    assert build_memory_context_block("") == ""
    assert build_memory_context_block("   \n  ") == ""


def test_block_strips_stray_fence_tags_then_refences() -> None:
    """Provider output carrying stray fence tags (not a full pair) is stripped
    of those tags then wrapped exactly once — no nesting, content preserved.

    NOTE: a *fully* pre-wrapped block (a complete open+close pair) is removed
    in its entirety by ``sanitize_context`` (faithful hermes behavior — the
    block regex eats the inner content too); that case is covered by
    ``test_sanitize_strips_full_block``. Here we exercise the common
    "leaked a stray tag" case where real content survives.
    """
    dirty = f"real fact {OPEN} more real content"
    block = build_memory_context_block(dirty)
    # Exactly one open and one close tag survive (no nesting).
    assert block.count(OPEN) == 1
    assert block.count(CLOSE) == 1
    assert "real fact" in block
    assert "more real content" in block


# -- sanitize_context ---------------------------------------------------------


def test_sanitize_strips_full_block() -> None:
    text = f"before {OPEN}\nsecret\n{CLOSE} after"
    clean = sanitize_context(text)
    assert "secret" not in clean
    assert OPEN not in clean
    assert CLOSE not in clean
    assert "before" in clean and "after" in clean


def test_sanitize_strips_stray_tags() -> None:
    assert OPEN not in sanitize_context(f"a {OPEN} b")
    assert CLOSE not in sanitize_context(f"a {CLOSE} b")


def test_sanitize_strips_nested_tags() -> None:
    nested = f"{OPEN}\nouter {OPEN}\ninner\n{CLOSE} tail\n{CLOSE}"
    clean = sanitize_context(nested)
    assert OPEN not in clean
    assert CLOSE not in clean


# -- StreamingContextScrubber -------------------------------------------------


def _feed_all(scrubber: StreamingContextScrubber, chunks: list[str]) -> str:
    out = "".join(scrubber.feed(c) for c in chunks)
    return out + scrubber.flush()


def test_scrubber_passthrough_plain_text() -> None:
    s = StreamingContextScrubber()
    assert _feed_all(s, ["hello ", "world"]) == "hello world"


def test_scrubber_open_and_close_in_one_chunk() -> None:
    s = StreamingContextScrubber()
    out = _feed_all(s, [f"pre\n{OPEN}\nsecret\n{CLOSE}\npost"])
    assert "secret" not in out
    assert "pre" in out and "post" in out


def test_scrubber_tag_split_across_chunks_leaks_nothing() -> None:
    """Open tag in one chunk, closed in a later chunk: payload never leaks."""
    s = StreamingContextScrubber()
    chunks = [
        "intro\n",
        "<memory-",
        "context>\nsecret payload",
        " continues",
        "\n</memory",
        "-context>\ntail",
    ]
    out = _feed_all(s, chunks)
    assert "secret payload" not in out
    assert "continues" not in out
    assert "intro" in out
    assert "tail" in out
    assert OPEN not in out and CLOSE not in out


def test_scrubber_unterminated_span_discarded_on_flush() -> None:
    """A span opened but never closed drops its content on flush (fail safe)."""
    s = StreamingContextScrubber()
    out = "".join(s.feed(c) for c in ["ok\n", OPEN, "\nleaked?"])
    out += s.flush()
    assert "leaked?" not in out
    assert "ok" in out


def test_scrubber_partial_tag_tail_held_then_emitted_when_not_a_tag() -> None:
    """A trailing fragment that looks like a tag start is held, then emitted
    verbatim once flush proves it was not a real tag."""
    s = StreamingContextScrubber()
    # Ends with "<memory" which could begin the open tag; must be held back.
    first = s.feed("line\n<memory")
    assert "<memory" not in first  # held
    tail = s.flush()
    assert tail == "<memory"


def test_scrubber_partial_tag_completed_next_chunk() -> None:
    s = StreamingContextScrubber()
    out = s.feed("line\n<memo")
    out += s.feed("ry-context>\nsecret\n</memory-context>\ndone")
    out += s.flush()
    assert "secret" not in out
    assert "done" in out
    assert "line" in out


def test_scrubber_block_boundary_gating_inline_tag_not_treated_as_fence() -> None:
    """An open tag that is NOT at a block boundary (text on same line before it,
    or no newline after it) is not treated as a memory-context fence."""
    s = StreamingContextScrubber()
    # Inline mention with text before it on the same line -> not a block opener.
    out = _feed_all(s, ["see <memory-context> inline mention here\n"])
    assert "inline mention here" in out


def test_scrubber_reset_clears_state() -> None:
    s = StreamingContextScrubber()
    s.feed(f"{OPEN}\nunclosed")  # enters span
    s.reset()
    # After reset, plain text passes straight through.
    assert _feed_all(s, ["fresh text"]) == "fresh text"
