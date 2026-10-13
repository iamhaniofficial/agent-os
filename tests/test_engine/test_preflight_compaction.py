"""Preflight compaction through the TurnRunner (Feature D, one CompactionService).

The before-turn DB compaction, the T3-upgrade compaction and the emergency
request-scoped trim were replaced by the agent's single ``CompactionService``,
which checks the live request before every provider call -- the first call
of a turn is the "preflight". These tests drive real turns through the
runtime with a real ``SessionManager`` and cover what the old preflight tests
guarded:

- compaction triggers above ``preflight_compact_ratio`` and not below it;
- a durable checkpoint is recorded before any row is archived, and a failing
  checkpoint prevents the destructive rewrite;
- cron/subagent sessions (canonical keys included) compact without one;
- the compacted transcript is persisted (``reset`` record + durable summary)
  and the next turn replays it without duplicating the summary;
- the summarizer runs through the session's ``LLMProvider``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentos.engine.runtime import TurnRunner
from agentos.engine.types import DoneEvent
from agentos.gateway.config import AgentOSRouterConfig, AttachmentsConfig, GatewayConfig
from agentos.provider import DoneEvent as ProviderDone
from agentos.provider import Message
from agentos.provider import TextDeltaEvent as ProviderText
from agentos.session.manager import SessionManager
from agentos.session.storage import SessionStorage
from agentos.tools.registry import ToolRegistry
from agentos.tools.types import CallerKind, ToolContext

KEY = "agent:main:webchat:preflight"
SUMMARY = "SUMMARY: the user asked about deployments on host-17"


class _Provider:
    """Answers turns; answers summarizer calls with a fixed summary."""

    provider_name = "test"

    def __init__(self) -> None:
        self.model = "test/model"
        self.turn_requests: list[list[Message]] = []
        self.summarizer_requests: list[list[Message]] = []

    def chat(self, messages: list[Message], tools=None, config=None) -> AsyncIterator[Any]:
        system = str(getattr(config, "system", "") or "")
        if system.startswith("You are a conversation compactor"):
            self.summarizer_requests.append(list(messages))
            return self._stream(SUMMARY)
        self.turn_requests.append(list(messages))
        return self._stream(f"answer {len(self.turn_requests)}")

    async def _stream(self, text: str) -> AsyncIterator[Any]:
        yield ProviderText(text=text)
        yield ProviderDone(stop_reason="end_turn", input_tokens=10, output_tokens=1)

    async def list_models(self) -> list[Any]:
        return []


class _SelectorClone:
    current_config = SimpleNamespace(model="test/model")

    def __init__(self, provider: _Provider) -> None:
        self.provider = provider

    def override_model(self, model: str) -> None:
        self.current_config = SimpleNamespace(model=model)

    def resolve(self) -> _Provider:
        return self.provider


class _Selector:
    def __init__(self, provider: _Provider) -> None:
        self.provider = provider

    def clone(self) -> _SelectorClone:
        return _SelectorClone(self.provider)


class _Catalog:
    def __init__(self, context_window: int) -> None:
        self.context_window = context_window

    def resolve_max_tokens(self, model_id, user_override=0, provider_name=""):  # noqa: ARG002
        return 1024

    def resolve_context_window(self, model_id, provider_name=""):  # noqa: ARG002
        return self.context_window

    def get_capabilities(self, model_id, provider_name="", base_url=""):  # noqa: ARG002
        return None


@pytest.fixture(autouse=True)
def _auxiliary_client(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Route the summarizer's auxiliary client to the test provider.

    The in-turn summarizer runs as the auxiliary side task ``compaction``
    (cheap model, no thinking), not on the turn's own model.
    """
    from agentos.provider import auxiliary

    built: list[dict[str, Any]] = []

    def _factory(**kwargs: Any) -> Any:
        built.append(kwargs)
        return _AUX_PROVIDER["provider"]

    monkeypatch.setattr(
        auxiliary,
        "_client",
        auxiliary.AuxiliaryClient(
            config={"provider": "opencap", "model": "cheap-model"},
            provider_factory=_factory,
        ),
    )
    return built


_AUX_PROVIDER: dict[str, Any] = {}


@pytest.fixture
async def storages() -> AsyncIterator[list[SessionStorage]]:
    opened: list[SessionStorage] = []
    yield opened
    for storage in opened:
        await storage.close()


async def _manager(
    tmp_path: Path,
    storages: list[SessionStorage],
    *,
    checkpoint: bool = True,
) -> SessionManager:
    storage = SessionStorage(":memory:")
    await storage.connect()
    storages.append(storage)
    return SessionManager(
        storage,
        inject_time_prefix=False,
        checkpoint_workspace_dir=tmp_path if checkpoint else None,
    )


def _runner(
    provider: _Provider,
    manager: SessionManager,
    tmp_path: Path,
    *,
    ratio: float = 0.5,
    context_window: int = 20_000,
) -> TurnRunner:
    return TurnRunner(
        provider_selector=_Selector(provider),
        tool_registry=ToolRegistry(),
        session_manager=manager,
        model_catalog=_Catalog(context_window),
        config=GatewayConfig(
            attachments=AttachmentsConfig(media_root=str(tmp_path / "media")),
            agentos_router=AgentOSRouterConfig(enabled=False),
            preflight_compact_ratio=ratio,
        ),
    )


async def _seed(manager: SessionManager, key: str, *, rounds: int = 6) -> None:
    await manager.create(key)
    for index in range(rounds):
        await manager.append_message(key, "user", f"old question {index} " + "q" * 6000)
        await manager.append_message(key, "assistant", f"old answer {index} " + "a" * 6000)


async def _turn(runner: TurnRunner, manager: SessionManager, key: str, text: str, tmp_path: Path):
    await manager.append_message(key, "user", text)
    return [
        event
        async for event in runner.run(
            text,
            key,
            tool_context=ToolContext(caller_kind=CallerKind.WEB, workspace_dir=str(tmp_path)),
            history_has_persisted_user=True,
            no_memory_capture=True,
        )
    ]


def _dump(messages: list[Message]) -> list[str]:
    return [json.dumps(message.model_dump(mode="json"), sort_keys=True) for message in messages]


@pytest.mark.asyncio
async def test_preflight_under_threshold_does_not_compact(
    tmp_path: Path, storages: list[SessionStorage]
) -> None:
    manager = await _manager(tmp_path, storages)
    provider = _Provider()
    _AUX_PROVIDER["provider"] = provider
    runner = _runner(provider, manager, tmp_path, ratio=0.9, context_window=1_000_000)
    await _seed(manager, KEY, rounds=1)

    events = await _turn(runner, manager, KEY, "what now?", tmp_path)

    assert any(isinstance(event, DoneEvent) for event in events)
    assert provider.summarizer_requests == []
    assert str(provider.turn_requests[0][0].content).startswith("old question 0")
    assert await manager.get_summaries(KEY) == []


@pytest.mark.asyncio
async def test_preflight_compacts_before_the_first_call_and_persists_it(
    tmp_path: Path,
    storages: list[SessionStorage],
    _auxiliary_client: list[dict[str, Any]],
) -> None:
    manager = await _manager(tmp_path, storages)
    provider = _Provider()
    _AUX_PROVIDER["provider"] = provider
    runner = _runner(provider, manager, tmp_path)
    await _seed(manager, KEY)
    rows_before = len(await manager.get_transcript(KEY))

    first = await _turn(runner, manager, KEY, "deploy status?", tmp_path)

    assert any(isinstance(event, DoneEvent) for event in first)
    # The summarizer ran as the auxiliary side task on the cheap model, over
    # the old rows (in chunks that fit the model window).
    assert provider.summarizer_requests
    from agentos.gateway.session_titler import fast_model_hint

    cheapest_tier = fast_model_hint(runner._config, task="compaction")
    assert cheapest_tier[1]
    assert {kwargs["model"] for kwargs in _auxiliary_client} == {cheapest_tier[1]}
    assert "old question 0" in str(provider.summarizer_requests[0][0].content)
    summarizer_calls = len(provider.summarizer_requests)
    # The very first turn request already carries the single-summary shape.
    sent = provider.turn_requests[0]
    assert str(sent[0].content).startswith("[Context summary]\n")
    assert SUMMARY in str(sent[0].content)
    assert sent[1].content == "Understood. Continuing from summary."
    assert str(sent[2].content).startswith("deploy status?")
    assert len(sent) == 3

    # Durable side: checkpoint receipt, summary row, archived transcript rows.
    receipts = await manager.storage.list_memory_durable_receipts(
        session_key=KEY, status="checkpoint_saved", limit=5
    )
    assert receipts and receipts[0].scope == "checkpoint"
    assert (tmp_path / receipts[0].source_path).exists()
    summaries = await manager.get_summaries(KEY)
    assert len(summaries) == 1
    assert SUMMARY in summaries[0].summary_text
    rows = await manager.get_transcript(KEY)
    assert len(rows) < rows_before
    record = rows[-1].provider_messages
    assert isinstance(record, dict)
    assert record["mode"] == "reset"
    assert record["messages"][0]["content"] == sent[0].content

    second = await _turn(runner, manager, KEY, "and the logs?", tmp_path)

    assert any(isinstance(event, DoneEvent) for event in second)
    # No second compaction; the reload replays the compacted transcript and
    # the durable summary is not repeated as request context.
    assert len(provider.summarizer_requests) == summarizer_calls
    resent = provider.turn_requests[1]
    assert _dump(resent)[: len(sent)] == _dump(sent)
    assert resent[len(sent)].role == "assistant"
    payload = json.dumps([message.model_dump(mode="json") for message in resent])
    assert payload.count("[Context summary]") == 1
    assert "[Compacted Session Summaries]" not in payload


@pytest.mark.asyncio
async def test_canonical_subagent_session_compacts_without_a_checkpoint(
    tmp_path: Path, storages: list[SessionStorage]
) -> None:
    key = "agent:main:subagent:run-7"
    manager = await _manager(tmp_path, storages)
    provider = _Provider()
    _AUX_PROVIDER["provider"] = provider
    runner = _runner(provider, manager, tmp_path)
    await _seed(manager, key)

    events = await _turn(runner, manager, key, "subtask", tmp_path)

    assert any(isinstance(event, DoneEvent) for event in events)
    assert str(provider.turn_requests[0][0].content).startswith("[Context summary]\n")
    receipts = await manager.storage.list_memory_durable_receipts(session_key=key, limit=5)
    assert receipts == []


@pytest.mark.asyncio
async def test_checkpoint_failure_prevents_the_destructive_rewrite(
    tmp_path: Path, storages: list[SessionStorage]
) -> None:
    # No checkpoint workspace: record_memory_checkpoint raises.
    manager = await _manager(tmp_path, storages, checkpoint=False)
    provider = _Provider()
    _AUX_PROVIDER["provider"] = provider
    runner = _runner(provider, manager, tmp_path)
    await _seed(manager, KEY)
    rows_before = len(await manager.get_transcript(KEY))

    events = await _turn(runner, manager, KEY, "deploy status?", tmp_path)

    # The turn still answers from the compacted request ...
    assert any(isinstance(event, DoneEvent) for event in events)
    assert str(provider.turn_requests[0][0].content).startswith("[Context summary]\n")
    # ... but no row was archived and no durable summary was written.
    rows = await manager.get_transcript(KEY)
    assert len(rows) == rows_before + 2  # only this turn's user + assistant rows were added
    assert await manager.get_summaries(KEY) == []
    failed = await manager.storage.list_memory_durable_receipts(
        session_key=KEY, status="checkpoint_failed", limit=5
    )
    assert failed
