"""In-memory ``AcpRuntime`` fake for the session-manager / spawn / dispatch tests.

Implements the protocol in ``agentos.acp.types`` without any subprocess, so
the manager, projector and gateway wiring can be tested deterministically.
A turn is driven by a :class:`Script`: a list of events to emit, then either
a result, an exception, or a block on ``release`` (until released or
cancelled).
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from agentos.acp.types import (
    AcpEvent,
    AcpSessionHandle,
    AcpSessionSpec,
    AcpTurn,
    AcpTurnResult,
    AcpUsage,
)


@dataclass
class Script:
    events: list[AcpEvent] = field(default_factory=list)
    result: AcpTurnResult | None = None
    raise_exc: BaseException | None = None
    block: bool = False  # wait for ``release`` (or cancel) after the events


def text_script(text: str = "done", *, cost: float | None = None, **usage: int) -> Script:
    return Script(
        events=[AcpEvent(kind="text_delta", text=text)],
        result=AcpTurnResult(
            stop_reason="end_turn",
            text=text,
            usage=AcpUsage(cost_usd=cost, **usage),
        ),
    )


class FakeRuntime:
    def __init__(self, scripts: list[Script] | Callable[[str], Script] | None = None) -> None:
        self._scripts = scripts if scripts is not None else []
        self._ids = itertools.count(1)
        self.calls: list[tuple[str, Any]] = []
        self.handles: list[AcpSessionHandle] = []
        self.closed: list[tuple[str, bool]] = []
        self.cancelled: list[str] = []
        self.turn_cancels = 0
        self.release = asyncio.Event()
        self.turn_started = asyncio.Event()
        self.config_options: list[dict[str, Any]] = [{"id": "model"}]
        self.modes: list[dict[str, Any]] = [{"id": "default"}, {"id": "plan"}]

    def _next_script(self, text: str) -> Script:
        if callable(self._scripts):
            return self._scripts(text)
        if self._scripts:
            return self._scripts.pop(0)
        return text_script(f"echo: {text}")

    async def ensure_session(self, spec: AcpSessionSpec, *, session_key: str) -> AcpSessionHandle:
        n = next(self._ids)
        self.calls.append(("ensure_session", spec))
        handle = AcpSessionHandle(
            record_id=spec.extra.get("record_id") or f"rec-{n}",
            agent_id=spec.agent_id,
            cwd=spec.cwd,
            argv=["fake", spec.agent_id],
            argv_hash="hash",
            acp_session_id=spec.resume_session_id or f"sess-{n}",
            pid=1000 + n,
            mode=spec.mode,
            config_options=[dict(o) for o in self.config_options],
            available_modes=[dict(m) for m in self.modes],
            status="idle",
        )
        self.handles.append(handle)
        return handle

    async def start_turn(
        self,
        handle: AcpSessionHandle,
        *,
        text: str,
        attachments: list[dict[str, Any]] | None = None,
        session_key: str,
    ) -> AcpTurn:
        self.calls.append(("start_turn", text))
        script = self._next_script(text)
        cancelled = asyncio.Event()
        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[AcpTurnResult] = loop.create_future()
        self.turn_started.set()

        async def _events() -> AsyncIterator[AcpEvent]:
            try:
                for event in script.events:
                    if cancelled.is_set():
                        break
                    yield event
                    await asyncio.sleep(0)
                if script.block and not cancelled.is_set():
                    waiter = asyncio.ensure_future(self.release.wait())
                    canceller = asyncio.ensure_future(cancelled.wait())
                    try:
                        await asyncio.wait({waiter, canceller}, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        waiter.cancel()
                        canceller.cancel()
                if result_future.done():
                    return
                if cancelled.is_set():
                    result_future.set_result(AcpTurnResult(stop_reason="cancelled"))
                elif script.raise_exc is not None:
                    result_future.set_exception(script.raise_exc)
                else:
                    result_future.set_result(script.result or AcpTurnResult(stop_reason="end_turn"))
            finally:
                if not result_future.done():
                    result_future.set_result(AcpTurnResult(stop_reason="cancelled"))

        async def _cancel() -> None:
            self.turn_cancels += 1
            cancelled.set()

        async def _close_stream() -> None:
            return None

        return AcpTurn(
            events=_events(),
            result=result_future,
            cancel=_cancel,
            close_stream=_close_stream,
        )

    async def cancel(self, handle: AcpSessionHandle, *, reason: str = "") -> None:
        self.cancelled.append(handle.record_id)

    async def close(self, handle: AcpSessionHandle, *, discard: bool = False) -> None:
        handle.status = "closed"
        self.closed.append((handle.record_id, discard))

    async def status(self, handle: AcpSessionHandle) -> AcpSessionHandle:
        return handle

    async def set_mode(self, handle: AcpSessionHandle, mode_id: str) -> None:
        self.calls.append(("set_mode", mode_id))

    async def set_config_option(self, handle: AcpSessionHandle, key: str, value: Any) -> None:
        self.calls.append(("set_config_option", (key, value)))

    async def doctor(self, agent_id: str) -> dict[str, Any]:
        return {"agent_id": agent_id, "probe": "ok", "version": "fake-1"}
