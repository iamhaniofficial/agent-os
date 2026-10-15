"""The router context reads a window of the newest transcript rows.

It must give exactly what the old whole-transcript read gave: for random
transcripts (seeded), with a window small enough that the fallback to the full
read is exercised too, and with the current user row excluded or not.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

import agentos.engine.runtime as runtime_module
from agentos.engine.runtime import TurnRunner
from agentos.session.manager import SessionManager
from agentos.session.storage import SessionStorage

KEY = "agent:main:webchat:router-window"


@pytest.fixture
async def manager() -> AsyncIterator[SessionManager]:
    storage = SessionStorage(":memory:")
    await storage.connect()
    yield SessionManager(storage, inject_time_prefix=False)
    await storage.close()


def _runner(manager: SessionManager) -> TurnRunner:
    return TurnRunner(
        provider_selector=None,
        config=SimpleNamespace(workspace_dir=".", memory=SimpleNamespace(source="workspace")),
        session_manager=manager,
    )


async def _full_read_context(runner: TurnRunner, manager: SessionManager, exclude: bool):
    entries = await manager.get_transcript(KEY)
    context, _ = runner._router_context_from_entries(list(entries), exclude_last_user=exclude)
    return context


@pytest.mark.parametrize("seed", range(12))
async def test_windowed_context_matches_the_full_read(
    manager: SessionManager, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    rng = random.Random(seed)
    monkeypatch.setattr(runtime_module, "_ROUTER_CONTEXT_WINDOW_ENTRIES", rng.choice([3, 5, 8]))
    await manager.create(KEY, agent_id="main")
    runner = _runner(manager)
    for i in range(rng.randint(0, 30)):
        role = rng.choice(["user", "assistant", "assistant", "tool", "system"])
        content = rng.choice(["", "   ", f"{role} text {i}"])
        token_count = rng.choice([None, 0, 7])
        await manager.append_message(KEY, role, content, token_count=token_count)
        for exclude in (False, True):
            got = await runner._router_previous_assistant_context(KEY, exclude_last_user=exclude)
            assert got == await _full_read_context(runner, manager, exclude)


async def test_long_session_is_served_from_the_window(
    manager: SessionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    await manager.create(KEY, agent_id="main")
    for i in range(40):
        await manager.append_message(KEY, "user", f"question {i}")
        await manager.append_message(KEY, "assistant", f"answer {i}", token_count=3)
    calls: list[str] = []
    real_get_transcript = manager.get_transcript

    async def spy(*args, **kwargs):
        calls.append("full")
        return await real_get_transcript(*args, **kwargs)

    monkeypatch.setattr(manager, "get_transcript", spy)
    runner = _runner(manager)
    context = await runner._router_previous_assistant_context(KEY)
    assert calls == []
    assert context == {
        "history_user_texts": ["question 36", "question 37", "question 38", "question 39"],
        "prev_assistant_text": "answer 39",
        "prev_assistant_usage": {"output_tokens": 3},
    }


async def test_missing_session_still_yields_an_empty_context(manager: SessionManager) -> None:
    assert await _runner(manager)._router_previous_assistant_context("agent:main:nope") == {}
