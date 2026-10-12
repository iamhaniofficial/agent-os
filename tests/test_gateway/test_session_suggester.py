"""Suggesting the user's next prompt once a turn has finished.

Same harness as ``test_session_titler``: a stub auxiliary provider and a real
``SessionManager``, so eligibility is decided from a real transcript.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from agentos.gateway import session_suggester as suggester_mod
from agentos.gateway.session_suggester import (
    MAX_SUGGESTION_CHARS,
    SessionSuggester,
    clean_suggestion,
    reset_suggesters,
    suggester_for,
)
from agentos.gateway.session_titler import fast_model_hint
from agentos.provider import auxiliary as aux
from agentos.provider.types import DoneEvent, TextDeltaEvent
from agentos.session.manager import SessionManager
from agentos.session.storage import SessionStorage

KEY = "agent:main:webchat:abc12345"

_ENV_KEYS = (
    "AGENTOS_SUGGEST_NEXT_MODEL",
    "AGENTOS_SUGGEST_NEXT_PROVIDER",
    "AGENTOS_AUXILIARY_MODEL",
    "AGENTOS_LLM_MODEL",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    reset_suggesters()
    yield
    reset_suggesters()


@pytest_asyncio.fixture
async def manager():
    storage = SessionStorage(":memory:")
    await storage.connect()
    mgr = SessionManager(storage, inject_time_prefix=False)
    yield mgr
    await storage.close()


def _stub_client(monkeypatch: pytest.MonkeyPatch, text: str, *, gate=None) -> list[dict]:
    """Auxiliary client whose provider answers ``text``; ``gate`` holds it open."""
    calls: list[dict] = []

    class _Provider:
        async def chat(self, *, messages, config=None, tools=None):
            calls.append({"messages": messages, "config": config})
            if gate is not None:
                await gate.wait()
            yield TextDeltaEvent(text=text)
            yield DoneEvent(input_tokens=10, output_tokens=5)

    client = aux.AuxiliaryClient(provider_factory=lambda **_kw: _Provider())
    monkeypatch.setattr(suggester_mod, "get_auxiliary_client", lambda: client)
    return calls


def _failing_client(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    class _Client:
        async def complete(self, **_kwargs):
            calls.append(1)
            raise aux.AuxiliaryError("no credentials", task="suggest_next")

    monkeypatch.setattr(suggester_mod, "get_auxiliary_client", lambda: _Client())
    return calls


async def _finished_turn(manager, user: str = "Write a parser for the log format", reply=None):
    await manager.create(KEY)
    await manager.append_message(KEY, "user", user)
    return await manager.append_message(
        KEY, "assistant", reply or "Done — `parse_log()` is in parser.py."
    )


# ── clean_suggestion ─────────────────────────────────────────────────────────


def test_clean_suggestion_strips_model_noise():
    assert clean_suggestion('Suggestion: "Now add tests for it"') == "Now add tests for it"
    assert clean_suggestion("Next: deploy it to staging\nbecause …") == "deploy it to staging"
    assert clean_suggestion("\n\n  Viết test cho parser  \n") == "Viết test cho parser"


def test_clean_suggestion_rejects_none_and_empty():
    assert clean_suggestion("NONE") is None
    assert clean_suggestion("none.") is None
    assert clean_suggestion("   ") is None
    assert clean_suggestion("") is None


def test_clean_suggestion_cuts_long_text_at_a_word_boundary():
    raw = "Please " + "refactor the parser module " * 20
    cleaned = clean_suggestion(raw)
    assert cleaned is not None
    assert len(cleaned) <= MAX_SUGGESTION_CHARS
    assert raw.startswith(cleaned)
    assert not cleaned.endswith(" ")
    assert raw[len(cleaned)] == " "


# ── suggester ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_suggests_from_the_last_exchange(manager, monkeypatch):
    calls = _stub_client(monkeypatch, "Now write tests for parse_log")
    await _finished_turn(manager)

    text = await SessionSuggester(manager).suggest(KEY)

    assert text == "Now write tests for parse_log"
    prompt = calls[0]["messages"][0].content
    assert "Write a parser for the log format" in prompt
    assert "parse_log()" in prompt
    config = calls[0]["config"]
    assert config.max_tokens >= 256
    assert "next message" in (config.system or "").lower()


@pytest.mark.asyncio
async def test_no_model_call_after_an_errored_turn(manager, monkeypatch):
    calls = _stub_client(monkeypatch, "anything")
    await manager.create(KEY)
    await manager.append_message(KEY, "user", "Do the thing")
    await manager.append_message(KEY, "system", "Error: provider exploded")

    assert await SessionSuggester(manager).suggest(KEY) is None
    assert calls == []


@pytest.mark.asyncio
async def test_no_model_call_without_an_assistant_reply(manager, monkeypatch):
    calls = _stub_client(monkeypatch, "anything")
    await manager.create(KEY)
    await manager.append_message(KEY, "user", "Do the thing")

    suggester = SessionSuggester(manager, settle_seconds=0.0)
    assert await suggester.suggest(KEY) is None
    assert await suggester.suggest("agent:main:webchat:missing") is None
    assert calls == []


@pytest.mark.asyncio
async def test_tool_entries_after_the_reply_do_not_hide_it(manager, monkeypatch):
    _stub_client(monkeypatch, "Ship it")
    await _finished_turn(manager)
    await manager.append_message(KEY, "tool", "{}", tool_call_id="t1")

    assert await SessionSuggester(manager).suggest(KEY) == "Ship it"


@pytest.mark.asyncio
async def test_one_model_call_per_finished_turn(manager, monkeypatch):
    calls = _stub_client(monkeypatch, "Now add tests")
    await _finished_turn(manager)
    suggester = SessionSuggester(manager)

    assert await suggester.suggest(KEY) == "Now add tests"
    assert await suggester.suggest(KEY) == "Now add tests"
    assert len(calls) == 1

    await manager.append_message(KEY, "user", "ok add tests")
    await manager.append_message(KEY, "assistant", "Added 4 tests.")
    assert await suggester.suggest(KEY) == "Now add tests"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_model_call(manager, monkeypatch):
    gate = asyncio.Event()
    calls = _stub_client(monkeypatch, "Deploy it", gate=gate)
    await _finished_turn(manager)
    suggester = SessionSuggester(manager)

    first = asyncio.create_task(suggester.suggest(KEY))
    second = asyncio.create_task(suggester.suggest(KEY))
    await asyncio.sleep(0)
    gate.set()

    assert await asyncio.gather(first, second) == ["Deploy it", "Deploy it"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_model_failure_is_none_and_not_retried(manager, monkeypatch):
    calls = _failing_client(monkeypatch)
    await _finished_turn(manager)
    suggester = SessionSuggester(manager)

    assert await suggester.suggest(KEY) is None
    assert await suggester.suggest(KEY) is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_model_saying_none_gives_no_suggestion(manager, monkeypatch):
    _stub_client(monkeypatch, "NONE")
    await _finished_turn(manager)

    assert await SessionSuggester(manager).suggest(KEY) is None


@pytest.mark.asyncio
async def test_hint_reaches_the_auxiliary_call(manager, monkeypatch):
    seen: list[dict] = []

    class _Client:
        async def complete(self, **kwargs):
            seen.append(kwargs)
            raise aux.AuxiliaryError("stop", task="suggest_next")

    monkeypatch.setattr(suggester_mod, "get_auxiliary_client", lambda: _Client())
    await _finished_turn(manager)

    await SessionSuggester(manager, hint=("opencap", "fast-1"), timeout=7.0).suggest(KEY)

    assert seen[0]["task"] == "suggest_next"
    assert seen[0]["preferred_provider"] == "opencap"
    assert seen[0]["preferred_model"] == "fast-1"
    assert seen[0]["timeout"] == 7.0


def test_suggester_for_is_per_manager():
    a, b = object(), object()
    assert suggester_for(a) is suggester_for(a)
    assert suggester_for(a) is not suggester_for(b)


def test_fast_model_hint_yields_to_the_suggest_next_env_pin(monkeypatch: pytest.MonkeyPatch):
    config = {"agentos_router": {"tiers": {"c0": {"model": "fast"}}}}
    monkeypatch.setenv("AGENTOS_SUGGEST_NEXT_MODEL", "pinned")
    assert fast_model_hint(config, task="suggest_next") == ("", "")
    # The title pin is a different scope.
    assert fast_model_hint(config) == ("", "fast")


@pytest.mark.asyncio
async def test_waits_for_a_reply_persisted_just_after_done(manager, monkeypatch):
    # The turn runner yields DoneEvent before TurnFinalizerStage persists the
    # reply, so a client that asks the moment it sees ``done`` can land first.
    _stub_client(monkeypatch, "Now add tests")
    await manager.create(KEY)
    await manager.append_message(KEY, "user", "Write a parser")
    suggester = SessionSuggester(manager, settle_seconds=2.0)

    asking = asyncio.create_task(suggester.suggest(KEY))
    await asyncio.sleep(0.15)
    await manager.append_message(KEY, "assistant", "Done, see parser.py")

    assert await asking == "Now add tests"


@pytest.mark.asyncio
async def test_gives_up_when_no_reply_arrives(manager, monkeypatch):
    calls = _stub_client(monkeypatch, "anything")
    await manager.create(KEY)
    await manager.append_message(KEY, "user", "Do the thing")

    assert await SessionSuggester(manager, settle_seconds=0.2).suggest(KEY) is None
    assert calls == []
