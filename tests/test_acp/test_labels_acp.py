"""Human labels for harness steps: activity in acp.status and tool row names."""

from __future__ import annotations

import pytest

from agentos.acp.labels import MAX_LABEL_CHARS, humanize_tool_name, tool_label
from agentos.acp.projector import AcpProjector, ToolResultEvent, ToolUseStartEvent
from agentos.acp.types import AcpEvent


@pytest.fixture(autouse=True)
def _home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/Users/x")


@pytest.mark.parametrize(
    ("title", "kind", "raw_input", "tool_name", "label"),
    [
        # execute: the command's first line, from rawInput first
        (
            "git status",
            "execute",
            {"command": "git status", "description": "x"},
            None,
            "Run git status",
        ),
        ("Terminal", "execute", None, None, "Running a command"),
        ("Terminal", "execute", {"command": "pytest -q\necho done"}, None, "Run pytest -q"),
        ("git log", "execute", None, None, "Run git log"),
        ("Run make", "execute", None, None, "Run make"),
        (
            None,
            "execute",
            {"command": ["bash", "-lc", "ls -la /Users/x/app"]},
            None,
            "Run ls -la ~/app",
        ),
        # read / edit: the path, home shortened
        (
            "Read /Users/x/app/a.py (1 - 50)",
            "read",
            {"file_path": "/Users/x/app/a.py"},
            None,
            "Read ~/app/a.py",
        ),
        (
            "Write /Users/x/app/new.py",
            "edit",
            {"file_path": "/Users/x/app/new.py"},
            None,
            "Write ~/app/new.py",
        ),
        ("Edit /tmp/b.py", "edit", {"file_path": "/tmp/b.py"}, None, "Edit /tmp/b.py"),
        ("Preparing file…", "edit", None, None, "Editing a file"),
        ("Read File", "read", None, None, "Reading a file"),
        # search / fetch
        ('grep -n "TODO" src', "search", {"pattern": "TODO", "path": "src"}, None, "Search TODO"),
        (
            "Fetch https://docs.python.org/3/x",
            "fetch",
            {"url": "https://docs.python.org/3/x"},
            None,
            "Fetch docs.python.org",
        ),
        ('Search "acp spec"', "fetch", {"query": "acp spec"}, None, "Search acp spec"),
        # think
        ("Explore the codebase", "think", {"description": "Explore"}, None, "Thinking"),
        # harness-internal tools are humanized, descriptive titles kept
        ("ToolSearch", "other", {"query": "select:Read"}, "ToolSearch", "Looking up tools"),
        ("TodoWrite", "other", None, None, "Updating the plan"),
        ("Load skill: pdf", "other", None, "Skill", "Load skill: pdf"),
        ("mcp__github__create_issue", "other", None, None, "github: create issue"),
        ("ask_user", "other", None, None, "Ask user"),
        (None, None, None, None, "Working"),
    ],
)
def test_tool_label(title, kind, raw_input, tool_name, label) -> None:
    assert tool_label(title, kind, raw_input, tool_name=tool_name) == label


def test_label_is_one_line_and_capped() -> None:
    label = tool_label(None, "execute", {"command": "echo " + "x" * 300})
    assert len(label) == MAX_LABEL_CHARS and label.endswith("…") and "\n" not in label


def test_humanized_names_never_collide_with_snake_case_tool_names() -> None:
    for name in ("sessions_spawn", "exit_plan_mode", "web_search", "publish_artifact"):
        assert humanize_tool_name(name) != name


def test_projector_names_tool_rows_with_the_label_and_refines_it() -> None:
    p = AcpProjector(stream_delivery="live")
    start = p.feed(
        AcpEvent(kind="tool_call", tool_call_id="t", tool_kind="execute", title="Terminal")
    )
    assert isinstance(start[0], ToolUseStartEvent) and start[0].tool_name == "Running a command"
    # Claude sends the command once the input is complete.
    p.feed(
        AcpEvent(
            kind="tool_call_update",
            tool_call_id="t",
            tool_kind="execute",
            title="git status",
            raw_input={"command": "git status"},
        )
    )
    done = p.feed(AcpEvent(kind="tool_call_update", tool_call_id="t", status="completed"))
    assert isinstance(done[0], ToolResultEvent) and done[0].tool_name == "Run git status"
    assert done[0].arguments is not None and done[0].arguments["acp_kind"] == "execute"
    segments = p.transcript_segments()
    assert segments[0]["name"] == "Run git status" and segments[1]["name"] == "Run git status"


def test_text_runs_split_by_tool_calls_become_paragraphs() -> None:
    """2026-10-19: "Now the DCA mandate dumps.The two list dumps were ...":
    Claude Code sends no message ids, so narration glued together."""
    p = AcpProjector(stream_delivery="live")
    out = []
    for event in (
        AcpEvent(kind="text_delta", text="Now the DCA mandate dumps."),
        AcpEvent(kind="tool_call", tool_call_id="a", tool_kind="read", title="Read x"),
        AcpEvent(kind="tool_call_update", tool_call_id="a", status="completed"),
        AcpEvent(kind="text_delta", text="The two list dumps were truncated."),
        AcpEvent(kind="text_delta", text=" Checking totals."),
        AcpEvent(kind="tool_call", tool_call_id="b", tool_kind="search", title="grep"),
        AcpEvent(kind="text_delta", text="Line one\n"),
    ):
        out.extend(p.feed(event))
    assert p.text == (
        "Now the DCA mandate dumps.\n\nThe two list dumps were truncated. Checking totals."
        "\n\nLine one\n"
    )
    # The live stream carries the same breaks.
    streamed = "".join(e.text for e in out if e.kind == "text_delta")
    assert streamed == p.text
