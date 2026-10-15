"""Proactive (between-turns) compaction and estimator calibration.

Real turns through the runtime with a real ``SessionManager``: a turn that
ends with the history near the trigger schedules one background job that
compacts it after the turn, the next turn's preflight is then a no-op, and a
turn that starts first makes the job persist nothing.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentos.engine.agent import Agent
from agentos.engine.runtime import TurnRunner
from agentos.engine.types import DoneEvent
from agentos.gateway.config import (
    AgentOSRouterConfig,
    AttachmentsConfig,
    CompactionLlmConfig,
    GatewayConfig,
)
from agentos.provider import DoneEvent as ProviderDone
from agentos.provider import Message
from agentos.provider import TextDeltaEvent as ProviderText
from agentos.session.manager import SessionManager
from agentos.session.storage import SessionStorage
from agentos.tools.registry import ToolRegistry
from agentos.tools.types import CallerKind, ToolContext

KEY = "agent:main:webchat:proactive"
SUMMARY = "SUMMARY: the user asked about deployments on host-17"


class _Provider:
    """Answers turns; answers summarizer calls with a fixed summary.

    *input_ratio* makes every turn's usage report that share of the agent's
    own request estimate (the calibration signal). *summary_gate*, when set,
    holds the summarizer until it is released.
    """

    provider_name = "test"

    def __init__(self, *, input_ratio: float | None = None) -> None:
        self.model = "test/model"
        self.turn_requests: list[list[Message]] = []
        self.summarizer_requests: list[list[Message]] = []
        self.input_ratio = input_ratio
        self.summary_gate: asyncio.Event | None = None
        self.summary_started = asyncio.Event()

    def chat(self, messages: list[Message], tools=None, config=None) -> AsyncIterator[Any]:
        system = str(getattr(config, "system", "") or "")
        if system.startswith("You are a conversation compactor"):
            self.summarizer_requests.append(list(messages))
            return self._summary()
        self.turn_requests.append(list(messages))
        return self._stream(f"answer {len(self.turn_requests)}")

    async def _summary(self) -> AsyncIterator[Any]:
        self.summary_started.set()
        if self.summary_gate is not None:
            await self.summary_gate.wait()
        yield ProviderText(text=SUMMARY)
        yield ProviderDone(stop_reason="end_turn", input_tokens=10, output_tokens=1)

    async def _stream(self, text: str) -> AsyncIterator[Any]:
        input_tokens = 10
        if self.input_ratio is not None:
            input_tokens = int(_LAST_ESTIMATE["tokens"] * self.input_ratio)
        yield ProviderText(text=text)
        yield ProviderDone(stop_reason="end_turn", input_tokens=input_tokens, output_tokens=1)

    async def list_models(self) -> list[Any]:
        return []


_LAST_ESTIMATE: dict[str, int] = {"tokens": 0}
_ESTIMATES: list[int] = []
_AUX_PROVIDER: dict[str, Any] = {}


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
def _auxiliary_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the summarizer's auxiliary client to the test provider."""
    from agentos.provider import auxiliary

    monkeypatch.setattr(
        auxiliary,
        "_client",
        auxiliary.AuxiliaryClient(
            config={"provider": "opencap", "model": "cheap-model"},
            provider_factory=lambda **_kwargs: _AUX_PROVIDER["provider"],
        ),
    )


@pytest.fixture(autouse=True)
def _record_estimates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose the agent's raw request estimate to the fake provider."""
    original = Agent._estimate_live_request_tokens

    def _recording(self: Agent, *args: Any, **kwargs: Any) -> int:
        value = original(self, *args, **kwargs)
        _LAST_ESTIMATE["tokens"] = value
        _ESTIMATES.append(value)
        return value

    monkeypatch.setattr(Agent, "_estimate_live_request_tokens", _recording)


@pytest.fixture
async def storages() -> AsyncIterator[list[SessionStorage]]:
    opened: list[SessionStorage] = []
    yield opened
    for storage in opened:
        await storage.close()


async def _manager(tmp_path: Path, storages: list[SessionStorage]) -> SessionManager:
    storage = SessionStorage(":memory:")
    await storage.connect()
    storages.append(storage)
    return SessionManager(storage, inject_time_prefix=False, checkpoint_workspace_dir=tmp_path)


def _runner(
    provider: _Provider,
    manager: SessionManager,
    tmp_path: Path,
    *,
    ratio: float = 0.85,
    context_window: int = 40_000,
    **compaction: Any,
) -> TurnRunner:
    _AUX_PROVIDER["provider"] = provider
    return TurnRunner(
        provider_selector=_Selector(provider),
        tool_registry=ToolRegistry(),
        session_manager=manager,
        model_catalog=_Catalog(context_window),
        config=GatewayConfig(
            attachments=AttachmentsConfig(media_root=str(tmp_path / "media")),
            agentos_router=AgentOSRouterConfig(enabled=False),
            preflight_compact_ratio=ratio,
            compaction=CompactionLlmConfig(**compaction),
        ),
    )


async def _seed(manager: SessionManager, key: str, *, rounds: int, chars: int = 4500) -> None:
    if await manager.get_session(key) is None:
        await manager.create(key)
    for index in range(rounds):
        await manager.append_message(key, "user", f"old question {index} " + "q" * chars)
        await manager.append_message(key, "assistant", f"old answer {index} " + "a" * chars)


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


def _scheduler(runner: TurnRunner) -> Any:
    return runner._proactive_compaction


# ---------------------------------------------------------------------------
# Proactive compaction
# ---------------------------------------------------------------------------


async def test_job_compacts_between_turns_and_the_next_preflight_is_a_noop(
    tmp_path: Path, storages: list[SessionStorage]
) -> None:
    manager = await _manager(tmp_path, storages)
    provider = _Provider()
    runner = _runner(provider, manager, tmp_path)
    # ~6k of a 7.2k trigger (the provider request budget): under the
    # preflight trigger, over 0.70 of it.
    await _seed(manager, KEY, rounds=2)

    first = await _turn(runner, manager, KEY, "deploy status?", tmp_path)

    assert any(isinstance(event, DoneEvent) for event in first)
    # The turn itself never waited for a summarizer.
    assert str(provider.turn_requests[0][0].content).startswith("old question 0")
    await _scheduler(runner).wait_idle(KEY)
    assert len(provider.summarizer_requests) == 1
    assert "old question 0" in str(provider.summarizer_requests[0][0].content)

    # Persisted like the in-turn path: checkpoint, summary, archived rows and
    # a reset record that carries the summary exchange plus the kept turn.
    receipts = await manager.storage.list_memory_durable_receipts(
        session_key=KEY, status="checkpoint_saved", limit=5
    )
    assert receipts and receipts[0].scope == "checkpoint"
    summaries = await manager.get_summaries(KEY)
    assert len(summaries) == 1 and SUMMARY in summaries[0].summary_text
    rows = await manager.get_transcript(KEY)
    assert [row.role for row in rows] == ["user", "assistant"]
    assert rows[0].content == "deploy status?"
    record = rows[-1].provider_messages
    assert isinstance(record, dict) and record["mode"] == "reset"
    assert record["messages"][0]["content"].startswith("[Context summary]\n")

    second = await _turn(runner, manager, KEY, "and the logs?", tmp_path)

    assert any(isinstance(event, DoneEvent) for event in second)
    await _scheduler(runner).wait_idle(KEY)
    # No compaction during or after the second turn.
    assert len(provider.summarizer_requests) == 1
    sent = provider.turn_requests[1]
    assert str(sent[0].content) == f"[Context summary]\n{SUMMARY}"
    assert sent[1].content == "Understood. Continuing from summary."
    assert str(sent[2].content).startswith("deploy status?")
    assert sent[3].role == "assistant"
    assert str(sent[4].content).startswith("and the logs?")
    assert len(sent) == 5
    payload = json.dumps([message.model_dump(mode="json") for message in sent])
    assert payload.count("[Context summary]") == 1


async def test_under_the_proactive_ratio_nothing_is_scheduled(
    tmp_path: Path, storages: list[SessionStorage]
) -> None:
    manager = await _manager(tmp_path, storages)
    provider = _Provider()
    runner = _runner(provider, manager, tmp_path, context_window=1_000_000)
    await _seed(manager, KEY, rounds=2)

    await _turn(runner, manager, KEY, "deploy status?", tmp_path)
    await _scheduler(runner).wait_idle()

    assert provider.summarizer_requests == []
    assert await manager.get_summaries(KEY) == []
    # Idle sessions leave no scheduler state behind.
    assert _scheduler(runner)._sessions == {}


async def test_config_switch_turns_the_job_off(
    tmp_path: Path, storages: list[SessionStorage]
) -> None:
    manager = await _manager(tmp_path, storages)
    provider = _Provider()
    runner = _runner(provider, manager, tmp_path, proactive=False)
    await _seed(manager, KEY, rounds=2)

    await _turn(runner, manager, KEY, "deploy status?", tmp_path)
    await _scheduler(runner).wait_idle()

    assert provider.summarizer_requests == []
    assert await manager.get_summaries(KEY) == []


async def test_ephemeral_sessions_get_no_job(
    tmp_path: Path, storages: list[SessionStorage]
) -> None:
    key = "agent:main:subagent:run-9"
    manager = await _manager(tmp_path, storages)
    provider = _Provider()
    runner = _runner(provider, manager, tmp_path)
    await _seed(manager, key, rounds=2)

    await _turn(runner, manager, key, "subtask", tmp_path)
    await _scheduler(runner).wait_idle()

    assert provider.summarizer_requests == []


async def test_a_turn_that_starts_first_makes_the_job_persist_nothing(
    tmp_path: Path, storages: list[SessionStorage]
) -> None:
    manager = await _manager(tmp_path, storages)
    provider = _Provider()
    provider.summary_gate = asyncio.Event()
    runner = _runner(provider, manager, tmp_path)
    await _seed(manager, KEY, rounds=2)

    await _turn(runner, manager, KEY, "deploy status?", tmp_path)
    await asyncio.wait_for(provider.summary_started.wait(), timeout=2)
    rows_before = len(await manager.get_transcript(KEY))
    job = _scheduler(runner)._sessions[KEY].task
    assert job is not None and not job.done()

    # The next turn starts while the job is still summarizing.
    second = await _turn(runner, manager, KEY, "and the logs?", tmp_path)
    provider.summary_gate.set()
    await asyncio.wait_for(asyncio.gather(job, return_exceptions=True), timeout=2)

    assert any(isinstance(event, DoneEvent) for event in second)
    assert job.done()
    assert await manager.get_summaries(KEY) == []
    # Both turns' rows are intact: nothing was rewritten under the turn.
    rows = await manager.get_transcript(KEY)
    assert len(rows) == rows_before + 2
    assert str(provider.turn_requests[1][0].content).startswith("old question 0")


async def test_stale_generation_skips_the_persist(
    tmp_path: Path, storages: list[SessionStorage]
) -> None:
    """A turn that bumps the generation after summarizing wins over the job."""
    manager = await _manager(tmp_path, storages)
    provider = _Provider()
    provider.summary_gate = asyncio.Event()
    runner = _runner(provider, manager, tmp_path)
    await _seed(manager, KEY, rounds=2)
    await _turn(runner, manager, KEY, "deploy status?", tmp_path)
    await asyncio.wait_for(provider.summary_started.wait(), timeout=2)
    scheduler = _scheduler(runner)
    state = scheduler._sessions[KEY]
    job = state.task

    # Simulate a turn that loaded history between summarizer and persist,
    # without the cancellation the real hook sends.
    state.generation += 1
    provider.summary_gate.set()
    await asyncio.wait_for(asyncio.gather(job, return_exceptions=True), timeout=2)

    assert len(provider.summarizer_requests) == 1
    assert await manager.get_summaries(KEY) == []


# ---------------------------------------------------------------------------
# Estimator calibration
# ---------------------------------------------------------------------------


async def _calibrated_run(
    tmp_path: Path, storages: list[SessionStorage], *, calibrate: bool
) -> _Provider:
    manager = await _manager(tmp_path, storages)
    # Usage reports 60% of the agent's estimate, as providers do.
    provider = _Provider(input_ratio=0.6)
    runner = _runner(provider, manager, tmp_path, proactive=False, estimate_calibration=calibrate)
    await _seed(manager, KEY, rounds=1, chars=400)
    await _turn(runner, manager, KEY, "first", tmp_path)
    await _turn(runner, manager, KEY, "second", tmp_path)
    # Grow the history past the raw 7.2k trigger, under it once scaled by 0.6.
    await _seed(manager, KEY, rounds=3)
    _ESTIMATES.clear()
    await _turn(runner, manager, KEY, "third", tmp_path)
    assert 7_200 < _ESTIMATES[0] < 7_200 / 0.6
    return provider


async def test_calibrated_estimate_no_longer_triggers_at_the_raw_threshold(
    tmp_path: Path, storages: list[SessionStorage]
) -> None:
    provider = await _calibrated_run(tmp_path, storages, calibrate=True)

    assert provider.summarizer_requests == []
    assert str(provider.turn_requests[-1][0].content).startswith("old question 0")


async def test_without_calibration_the_raw_estimate_still_compacts(
    tmp_path: Path, storages: list[SessionStorage]
) -> None:
    provider = await _calibrated_run(tmp_path, storages, calibrate=False)

    assert len(provider.summarizer_requests) == 1
    assert str(provider.turn_requests[-1][0].content).startswith("[Context summary]")
