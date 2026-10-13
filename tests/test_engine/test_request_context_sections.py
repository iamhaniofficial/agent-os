"""Per-turn request context rides with the user turn, diffed per section.

The request context is persisted with the turn that carried it (the
transcript is append-only), so only the ``## <title>`` sections that changed
since their latest copy are sent again, and session-frozen content
(workspace files, project knowledge, router control) lives in the cached
system prompt instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentos.engine import Agent, AgentConfig
from agentos.engine.runtime import TurnRunner
from agentos.engine.transcript import request_context_delta, split_context_sections
from agentos.provider import ChatConfig, Message
from agentos.provider import DoneEvent as ProviderDone
from agentos.provider import TextDeltaEvent as ProviderText

FROZEN = "## Channel Rendering\n\nUse plain text, no tables."


def _context(recall: str, *, internal_event: bool = False) -> str:
    sections = [FROZEN, f"## Memory Context\n\n{recall}"]
    if internal_event:
        sections.append("## Internal Event Mode\n\nThe next input is an internal event.")
    return "\n\n".join(sections)


class _Provider:
    provider_name = "fake"

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def chat(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        config: ChatConfig | None = None,
    ) -> AsyncIterator[Any]:
        self.calls.append(list(messages))
        return self._stream()

    async def _stream(self) -> AsyncIterator[Any]:
        yield ProviderText(text="ok")
        yield ProviderDone(stop_reason="end_turn", input_tokens=1, output_tokens=1)

    async def list_models(self) -> list[Any]:
        return []


def test_sections_split_only_on_known_titles() -> None:
    recall = "remembered:\n\n## Not A Section\n\nstill recall"
    sections = split_context_sections(
        f"[Compacted Session Summaries]\nold\n\n{FROZEN}\n\n## Memory Context\n\n{recall}"
    )

    assert [title for title, _ in sections] == ["", "Channel Rendering", "Memory Context"]
    assert sections[2][1].endswith("still recall")


@pytest.mark.asyncio
async def test_changed_recall_persists_only_the_recall_section_again() -> None:
    provider = _Provider()
    agent = Agent(provider=provider, config=AgentConfig(max_iterations=1))

    agent.config.request_context_prompt = _context("recall for turn one")
    _ = [event async for event in agent.run_turn("first")]
    agent.config.request_context_prompt = _context("recall for turn two")
    _ = [event async for event in agent.run_turn("second")]

    first_input = str(provider.calls[0][-1].content)
    second_input = str(provider.calls[1][-1].content)
    assert "Use plain text, no tables." in first_input
    assert "recall for turn one" in first_input
    # Turn two carries the changed recall section only.
    assert "recall for turn two" in second_input
    assert "## Channel Rendering" not in second_input
    assert "recall for turn one" not in second_input
    # Nothing older was rewritten.
    assert provider.calls[1][: len(provider.calls[0])] == provider.calls[0]
    persisted = "".join(str(message.content) for message in agent._history)
    assert persisted.count("Use plain text, no tables.") == 1


@pytest.mark.asyncio
async def test_heartbeat_user_alternation_does_not_re_append_unchanged_sections() -> None:
    provider = _Provider()
    agent = Agent(provider=provider, config=AgentConfig(max_iterations=1))

    for index in range(4):
        heartbeat = index % 2 == 0
        agent.config.request_context_prompt = _context("same recall", internal_event=heartbeat)
        _ = [event async for event in agent.run_turn(f"turn {index}")]

    persisted = "".join(str(message.content) for message in agent._history)
    assert persisted.count("Use plain text, no tables.") == 1
    assert persisted.count("same recall") == 1
    # The internal-event section comes and goes with a one-line marker.
    third_input = str(provider.calls[1][-1].content)
    assert "## Internal Event Mode\n\n[No longer applies" in third_input


def test_no_change_means_no_request_context_block() -> None:
    history = [
        Message(
            role="user",
            content="hi\n\n"
            + str(request_context_delta([], _context("r")))
            + "\n\n[Runtime context for this turn]\nnow",
        )
    ]

    assert request_context_delta(history, _context("r")) is None


def test_session_frozen_parts_join_the_cached_system_prompt(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("AGENTS.md body", encoding="utf-8")
    runner = TurnRunner(
        provider_selector=MagicMock(),
        config=SimpleNamespace(
            agent_name=None,
            tools=SimpleNamespace(profile=None),
            safety=SimpleNamespace(injection_scan_mode="off", wrap_untrusted_workspace=False),
            subagents=SimpleNamespace(prompt_compact=False),
            memory=SimpleNamespace(inject_limit=4000),
            heartbeat_prompt=None,
        ),
    )
    runner._resolve_bootstrap_workspace_dir = lambda _agent_id: tmp_path
    runner._resolve_memory_source_dir = lambda _agent_id: tmp_path

    assembled = runner._assemble_prompt(
        "main",
        [],
        session_key="agent:main:main",
        extra_context={
            "Project Knowledge": "project facts",
            "Memory Context": "per-turn recall",
            "Internal Event Mode": "internal",
        },
    )

    assert isinstance(assembled, tuple)
    base, dynamic = assembled
    assert "AGENTS.md body" in base
    assert "project facts" in base
    assert "per-turn recall" not in base
    assert "## Memory Context" in dynamic
    assert "## Internal Event Mode" in dynamic
    assert "AGENTS.md body" not in dynamic
    assert "project facts" not in dynamic
