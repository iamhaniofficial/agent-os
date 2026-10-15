"""Run one model iteration's tool calls and stream each result as it finishes.

``ToolBatchExecutor`` takes the tool_use blocks of a single assistant message
and runs them under the concurrency each tool declares on its ``ToolSpec``:

* ``parallel`` calls run together, at most ``max_concurrency`` at a time;
* ``keyed`` calls run like ``parallel`` ones, except that calls sharing a
  concurrency key run one after another;
* ``exclusive`` calls (the default for a tool that declares nothing) run
  alone: every earlier call finishes first and every later call waits.

Admission is strictly in the order the model emitted the calls, so an
``exclusive`` call still acts as a barrier exactly like the old mutex tools.

Each result is delivered (artifact events, ``ToolResultEvent``, router-control
replay) as soon as its call finishes rather than after the whole batch. A call
that comes back asking for operator approval is delivered at once too, and the
wait for that approval runs in the background alongside every other call;
when the approval is resolved the call is retried with its ``approval_id``.
Approvals are therefore awaited concurrently instead of one by one, while what
gets approved is still decided by the tool itself.

The pairing guarantee holds whatever order calls finish in, and on timeout,
cancellation or exception: ``outcomes`` has exactly one final result per
tool_use, in the original call order, ready to be appended to the transcript.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import structlog

from agentos.engine.types import (
    AgentEvent,
    ArtifactEvent,
    RunHeartbeatEvent,
    ToolCall,
    ToolResult,
    ToolResultEvent,
    WarningEvent,
)
from agentos.execution_status import ExecutionStatusValue, runtime_execution_status
from agentos.provider import ContentBlockToolResult
from agentos.router_control import router_control_replay_event_from_payload
from agentos.tools.types import ToolConcurrency, ToolSpec

_log = structlog.get_logger(__name__)

DEFAULT_MAX_TOOL_CONCURRENCY = 8


# ---------------------------------------------------------------------------
# Concurrency policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolConcurrencyPolicy:
    """How one concrete call may overlap with the other calls of its batch."""

    mode: ToolConcurrency
    key: Hashable | None = None


EXCLUSIVE_POLICY = ToolConcurrencyPolicy(mode="exclusive")
PARALLEL_POLICY = ToolConcurrencyPolicy(mode="parallel")


def resolve_tool_concurrency(
    spec: ToolSpec | None,
    arguments: Mapping[str, Any] | None,
    *,
    session_key: str | None = None,
) -> ToolConcurrencyPolicy:
    """Return the policy for one call of the tool described by *spec*.

    An unknown tool, an unknown mode, or a ``keyed`` tool whose key cannot be
    derived runs ``exclusive``: the safe choice, and what such a call did
    before tools declared their concurrency.
    """

    if spec is None:
        return EXCLUSIVE_POLICY
    mode = getattr(spec, "concurrency", "exclusive")
    if mode == "parallel":
        return PARALLEL_POLICY
    if mode != "keyed":
        return EXCLUSIVE_POLICY
    key_fn = getattr(spec, "concurrency_key", None)
    if key_fn is None:
        return ToolConcurrencyPolicy(mode="keyed", key=("tool", spec.name))
    try:
        key = key_fn(arguments or {}, session_key)
    except Exception:  # noqa: BLE001 - a broken key function must not lose the call
        _log.warning("tool_batch.concurrency_key_failed", tool=spec.name, exc_info=True)
        return EXCLUSIVE_POLICY
    if key is None:
        return EXCLUSIVE_POLICY
    return ToolConcurrencyPolicy(mode="keyed", key=key)


def registry_spec_lookup(registry: Any | None) -> Callable[[str], ToolSpec | None]:
    """Find a tool's spec in *registry*, then in the default registry.

    The fallback matters for agents built without a registry (embedded use and
    tests): builtins register on the default registry, so their declared
    concurrency still applies.
    """

    def _lookup(name: str) -> ToolSpec | None:
        from agentos.tools.registry import get_default_registry  # noqa: PLC0415

        for candidate in (registry, get_default_registry()):
            getter = getattr(candidate, "get", None)
            if getter is None:
                continue
            registered = getter(name)
            spec = getattr(registered, "spec", None)
            if isinstance(spec, ToolSpec):
                return spec
        return None

    return _lookup


# ---------------------------------------------------------------------------
# Ordered admission gate
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _Slot:
    exclusive: bool
    future: asyncio.Future[None]
    released: bool = False


class _OrderedGate:
    """FIFO reader/writer gate.

    Shared holders run together; an exclusive holder runs alone. Requests are
    admitted strictly in the order they were reserved, so a shared request
    behind a waiting exclusive one does not overtake it.
    """

    def __init__(self) -> None:
        self._queue: deque[_Slot] = deque()
        self._shared_active = 0
        self._exclusive_active = False

    def reserve(self, *, exclusive: bool) -> _Slot:
        slot = _Slot(exclusive=exclusive, future=asyncio.get_running_loop().create_future())
        self._queue.append(slot)
        self._admit()
        return slot

    def release(self, slot: _Slot) -> None:
        """Give *slot* back. Idempotent; also abandons a slot never admitted."""

        if slot.released:
            return
        slot.released = True
        future = slot.future
        if future.done() and not future.cancelled():
            if slot.exclusive:
                self._exclusive_active = False
            else:
                self._shared_active -= 1
        elif not future.done():
            future.cancel()
        self._admit()

    def _admit(self) -> None:
        while self._queue:
            slot = self._queue[0]
            if slot.future.done():
                # Abandoned before admission.
                self._queue.popleft()
                continue
            if self._exclusive_active:
                return
            if slot.exclusive:
                if self._shared_active:
                    return
                self._exclusive_active = True
            else:
                self._shared_active += 1
            self._queue.popleft()
            slot.future.set_result(None)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolBatchOutcome:
    """Final result of one tool_use: the call that produced it and its projection."""

    call: ToolCall
    result: ToolResult
    projected: ToolResult


@dataclass(eq=False)
class _Job:
    index: int
    call: ToolCall
    is_approval_retry: bool = False


@dataclass
class _BatchClock:
    deadline: float
    total_deadline: float | None
    now: Callable[[], float]

    def remaining(self) -> float:
        current = self.now()
        remaining = max(0.0, self.deadline - current)
        if self.total_deadline is not None:
            remaining = min(remaining, max(0.0, self.total_deadline - current))
        return remaining

    def cap(self, timeout: float) -> float:
        return max(0.001, min(timeout, self.remaining()))


def _error_result(
    call: ToolCall, content: str, status: ExecutionStatusValue, reason: str
) -> ToolResult:
    return ToolResult(
        tool_use_id=call.tool_use_id,
        tool_name=call.tool_name,
        content=content,
        is_error=True,
        execution_status=runtime_execution_status(status, reason=reason),
    )


def _timeout_result(call: ToolCall, seconds: float) -> ToolResult:
    return ToolResult(
        tool_use_id=call.tool_use_id,
        tool_name=call.tool_name,
        content=f"Tool '{call.tool_name}' timed out after {seconds}s",
        is_error=True,
        execution_status=runtime_execution_status(
            "timeout",
            reason="runtime_timeout",
            timed_out=True,
        ),
    )


@dataclass
class ToolBatchExecutor:
    """Execute one iteration's tool calls; see the module docstring.

    The agent supplies the pieces that stay agent-specific (dispatch, result
    projection, approval waiting, logging) so the scheduling, timeout and
    pairing rules live here and can be tested without a provider.
    """

    execute: Callable[[ToolCall], Awaitable[ToolResult]]
    project: Callable[..., Awaitable[ToolResult]]
    timeout_for: Callable[[ToolCall], float]
    spec_for: Callable[[str], ToolSpec | None]
    deadline: float
    iteration_timeout: float
    total_deadline: float | None = None
    session_key: str | None = None
    max_concurrency: int = DEFAULT_MAX_TOOL_CONCURRENCY
    heartbeat_interval: float = 15.0
    approval_wait_timeout: float = 180.0
    pending_approval: Callable[[str], dict[str, Any] | None] = lambda _content: None
    wait_for_approval: Callable[..., Awaitable[Any]] | None = None
    artifact_event: Callable[[dict[str, Any]], ArtifactEvent] = lambda a: ArtifactEvent(**a)
    pending_warnings: list[WarningEvent] = field(default_factory=list)
    preflight_results: Mapping[str, ToolResult] = field(default_factory=dict)
    call_log: Callable[..., None] = lambda _kind, **_payload: None
    iteration: int = 0
    clock: Callable[[], float] | None = None

    _outcomes: dict[int, ToolBatchOutcome] = field(default_factory=dict, init=False)
    _calls: list[ToolCall] = field(default_factory=list, init=False)

    # -- public results -------------------------------------------------

    @property
    def outcomes(self) -> list[ToolBatchOutcome]:
        """One final outcome per tool_use, in the original call order."""

        return [self._outcomes[index] for index in range(len(self._calls))]

    @property
    def executed_results(self) -> list[ToolResult]:
        return [outcome.result for outcome in self.outcomes]

    def result_blocks(self) -> list[ContentBlockToolResult]:
        return [
            ContentBlockToolResult(
                tool_use_id=outcome.projected.tool_use_id,
                content=outcome.projected.content,
                is_error=outcome.projected.is_error,
                execution_status=outcome.projected.execution_status,
            )
            for outcome in self.outcomes
        ]

    def policy_for(self, call: ToolCall) -> ToolConcurrencyPolicy:
        return resolve_tool_concurrency(
            self.spec_for(call.tool_name),
            call.arguments,
            session_key=self.session_key,
        )

    # -- execution ------------------------------------------------------

    async def run(self, tool_calls: Sequence[ToolCall]) -> AsyncIterator[AgentEvent]:
        """Run every call; yield events as results arrive. Fills ``outcomes``."""

        self._calls = list(tool_calls)
        self._outcomes = {}
        if not self._calls:
            return
        clock = _BatchClock(
            deadline=self.deadline,
            total_deadline=self.total_deadline,
            now=self.clock or asyncio.get_running_loop().time,
        )
        gate = _OrderedGate()
        semaphore = asyncio.Semaphore(max(1, int(self.max_concurrency)))
        key_locks: dict[Hashable, asyncio.Lock] = {}
        tasks: dict[asyncio.Task[ToolResult], _Job] = {}

        async def _run_one(call: ToolCall) -> ToolResult:
            started = time.monotonic()
            self.call_log(
                "tool_request",
                iteration=self.iteration,
                tool_use_id=call.tool_use_id,
                name=call.tool_name,
                arguments=call.arguments,
            )
            tool_timeout = clock.cap(self.timeout_for(call))
            preflight = self.preflight_results.get(call.tool_use_id)
            if preflight is not None:
                result = preflight
            elif clock.remaining() <= 0:
                # Admitted only because an earlier call's own timeout fired at
                # the batch deadline: starting it now would run a tool the
                # batch has already given up on. It times out unstarted, like
                # every other call still queued at the deadline.
                result = _timeout_result(call, self.iteration_timeout)
            else:
                try:
                    result = await asyncio.wait_for(self.execute(call), timeout=tool_timeout)
                except TimeoutError:
                    result = _timeout_result(call, tool_timeout)
            self.call_log(
                "tool_response",
                iteration=self.iteration,
                tool_use_id=result.tool_use_id,
                name=result.tool_name,
                result=result.content,
                result_chars=len(result.content),
                is_error=result.is_error,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            return result

        async def _run_admitted(
            call: ToolCall, policy: ToolConcurrencyPolicy, slot: _Slot
        ) -> ToolResult:
            try:
                await slot.future
                if policy.mode == "exclusive":
                    return await _run_one(call)
                key_lock = (
                    key_locks.setdefault(policy.key, asyncio.Lock())
                    if policy.key is not None
                    else None
                )
                if key_lock is None:
                    async with semaphore:
                        return await _run_one(call)
                async with key_lock, semaphore:
                    return await _run_one(call)
            finally:
                gate.release(slot)

        def _schedule(job: _Job) -> None:
            policy = self.policy_for(job.call)
            slot = gate.reserve(exclusive=policy.mode == "exclusive")
            task = asyncio.create_task(_run_admitted(job.call, policy, slot))
            # A task cancelled before it first runs never reaches its
            # ``finally``; give its slot back so the gate cannot stall.
            task.add_done_callback(functools.partial(_release_slot, gate, slot))
            tasks[task] = job

        async def _approve_then_retry(job: _Job, payload: dict[str, Any]) -> ToolResult:
            if self.wait_for_approval is not None:
                await self.wait_for_approval(
                    payload,
                    timeout=clock.cap(self.approval_wait_timeout),
                )
            policy = self.policy_for(job.call)
            slot = gate.reserve(exclusive=policy.mode == "exclusive")
            return await _run_admitted(job.call, policy, slot)

        def _schedule_retry(job: _Job, payload: dict[str, Any]) -> None:
            retry_arguments = dict(job.call.arguments)
            retry_arguments["approval_id"] = payload["approval_id"]
            retry_call = ToolCall(
                tool_use_id=job.call.tool_use_id,
                tool_name=job.call.tool_name,
                arguments=retry_arguments,
                synthetic_from_text=job.call.synthetic_from_text,
                origin_trace=job.call.origin_trace,
                thought_signature=job.call.thought_signature,
            )
            retry_job = _Job(index=job.index, call=retry_call, is_approval_retry=True)
            tasks[asyncio.create_task(_approve_then_retry(retry_job, payload))] = retry_job

        for index, call in enumerate(self._calls):
            _schedule(_Job(index=index, call=call))

        interval = float(self.heartbeat_interval)
        started = time.monotonic()
        last_event_at = started
        try:
            while tasks:
                remaining = clock.remaining()
                if remaining <= 0:
                    timed_out = sorted(tasks.items(), key=lambda item: item[1].index)
                    tasks.clear()
                    await _cancel_and_wait([task for task, _job in timed_out])
                    for _task, job in timed_out:
                        result = _timeout_result(job.call, self.iteration_timeout)
                        for event in await self._finish(job, result):
                            yield event
                    break
                wait_timeout = remaining if interval <= 0 else min(interval, remaining)
                done, _pending = await asyncio.wait(
                    tasks,
                    timeout=max(0.001, wait_timeout),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    if clock.remaining() <= 0:
                        continue
                    now = time.monotonic()
                    yield RunHeartbeatEvent(
                        phase="tool",
                        elapsed_ms=int((now - started) * 1000),
                        idle_ms=int((now - last_event_at) * 1000),
                        message="Tool still running",
                    )
                    continue

                last_event_at = time.monotonic()
                finished = sorted(((task, tasks.pop(task)) for task in done), key=_job_index)
                for task, job in finished:
                    result = _task_result(task, job.call)
                    payload = (
                        None
                        if job.is_approval_retry or job.call.arguments.get("approval_id")
                        else self.pending_approval(result.content)
                    )
                    if payload is None:
                        for event in await self._finish(job, result):
                            yield event
                        continue
                    events, _projected = await self._render(job.call, result)
                    for event in events:
                        yield event
                    # Start waiting only once the approval prompt has been
                    # delivered; a consumer that answers it inline (CLI) has
                    # resolved it by the time the generator resumes.
                    _schedule_retry(job, payload)
        finally:
            leftover = list(tasks)
            tasks.clear()
            await _cancel_and_wait(leftover)

        for index, call in enumerate(self._calls):
            if index not in self._outcomes:  # pragma: no cover - defensive pairing guard
                result = _error_result(
                    call, f"Tool '{call.tool_name}' was cancelled", "cancelled", "cancelled"
                )
                self._outcomes[index] = ToolBatchOutcome(call=call, result=result, projected=result)

    async def _render(
        self, call: ToolCall, result: ToolResult
    ) -> tuple[list[AgentEvent], ToolResult]:
        """Build the events delivering *result*, in the order the UI expects."""

        events: list[AgentEvent] = [self.artifact_event(a) for a in result.artifacts]
        projected = await self.project(result, tool_call=call)
        events.append(
            ToolResultEvent(
                tool_use_id=projected.tool_use_id,
                tool_name=projected.tool_name,
                result=projected.content,
                is_error=projected.is_error,
                arguments=call.arguments,
                execution_status=projected.execution_status,
                thought_signature=call.thought_signature,
            )
        )
        replay_event = router_control_replay_event_from_payload(result.content)
        if replay_event is not None:
            events.append(replay_event)
        return events, projected

    async def _finish(self, job: _Job, result: ToolResult) -> list[AgentEvent]:
        """Deliver a final result and record it as the call's outcome."""

        events, projected = await self._render(job.call, result)
        self._outcomes[job.index] = ToolBatchOutcome(
            call=job.call,
            result=result,
            projected=projected,
        )
        while self.pending_warnings:
            events.append(self.pending_warnings.pop(0))
        return events


def _release_slot(gate: _OrderedGate, slot: _Slot, _task: asyncio.Task[ToolResult]) -> None:
    gate.release(slot)


def _job_index(item: tuple[asyncio.Task[ToolResult], _Job]) -> int:
    return item[1].index


def _task_result(task: asyncio.Task[ToolResult], call: ToolCall) -> ToolResult:
    try:
        return task.result()
    except asyncio.CancelledError:
        return _error_result(
            call, f"Tool '{call.tool_name}' was cancelled", "cancelled", "cancelled"
        )
    except Exception as exc:  # noqa: BLE001
        return _error_result(
            call, f"Tool '{call.tool_name}' raised: {exc}", "error", "runtime_error"
        )


async def _cancel_and_wait(tasks: list[asyncio.Task[ToolResult]]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.wait(tasks)
