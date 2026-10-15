"""Shared HTTP transport for the streaming LLM providers.

Four pieces every streaming provider uses:

- :class:`PooledHttpClient` keeps one ``httpx.AsyncClient`` per provider
  instance (and event loop), so consecutive requests of a turn reuse the
  TCP/TLS connection instead of paying a handshake per tool-loop iteration.
- :func:`stream_timeouts` splits connect / read / write / pool timeouts
  instead of one float for all four.
- :func:`iter_lines_with_idle_timeout` is the stream-idle watchdog: once data
  has started to flow, a gap longer than the idle budget ends the stream with
  :class:`StreamIdleTimeoutError` instead of hanging until the engine's much
  longer iteration timeout.
- :func:`guard_tool_stream` holds back each tool call's events until the call
  is complete and retries, once, a stream that ends inside a tool call.
"""

from __future__ import annotations

import asyncio
import os
import weakref
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import structlog

from agentos.env import trust_env as _trust_env

from .types import (
    DoneEvent,
    ErrorEvent,
    ProviderHeartbeatEvent,
    StreamEvent,
    TextDeltaEvent,
    ToolUseDeltaEvent,
    ToolUseEndEvent,
    ToolUseStartEvent,
)

log = structlog.get_logger(__name__)

#: Default seconds without a streamed data event before the stream is
#: abandoned. Overridden by ``ChatConfig.stream_idle_timeout`` or
#: ``AGENTOS_LLM_STREAM_IDLE_TIMEOUT_SECONDS``.
DEFAULT_STREAM_IDLE_TIMEOUT_S = 120.0
_IDLE_TIMEOUT_ENV = "AGENTOS_LLM_STREAM_IDLE_TIMEOUT_SECONDS"
_CONNECT_TIMEOUT_ENV = "AGENTOS_LLM_STREAM_CONNECT_TIMEOUT_SECONDS"
_WRITE_TIMEOUT_ENV = "AGENTOS_LLM_STREAM_WRITE_TIMEOUT_SECONDS"
_DEFAULT_CONNECT_TIMEOUT_S = 12.0
_DEFAULT_WRITE_TIMEOUT_S = 30.0
_DEFAULT_POOL_TIMEOUT_S = 30.0

#: Error codes this module emits. ``failures.classify_provider_error`` maps the
#: first to a retriable transport failure and the other two to a surfaced one.
STREAM_IDLE_TIMEOUT_CODE = "stream_idle_timeout"
INCOMPLETE_TOOL_STREAM_CODE = "incomplete_tool_stream"
STREAM_INTERRUPTED_AFTER_OUTPUT_CODE = "stream_interrupted_after_output"

#: Same cadence as the engine's own tool-argument heartbeat.
_TOOL_ARGUMENT_HEARTBEAT_CHARS = 4096


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        log.warning("provider.transport.invalid_env_float", env=name, raw=raw)
        return None


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


def stream_timeouts(
    timeout: float,
    *,
    connect_default: float = _DEFAULT_CONNECT_TIMEOUT_S,
    honour_connect_env: bool = True,
) -> httpx.Timeout:
    """Split timeouts for one streamed provider request.

    ``timeout`` (``ChatConfig.timeout``, i.e. the agent's request timeout) stays
    the per-read budget, so a slow first token behaves exactly as before.
    Connect is short (a host that does not accept a connection in seconds will
    not accept it in two minutes); ``AGENTOS_LLM_STREAM_CONNECT_TIMEOUT_SECONDS``
    overrides it. Write is per chunk sent, so 30 s covers multi-megabyte
    attachment uploads; ``AGENTOS_LLM_STREAM_WRITE_TIMEOUT_SECONDS`` overrides
    it. Every phase is capped at ``timeout``.
    """
    ceiling = max(float(timeout), 1.0)
    connect = _env_float(_CONNECT_TIMEOUT_ENV) if honour_connect_env else None
    if connect is None or connect <= 0:
        connect = connect_default
    write = _env_float(_WRITE_TIMEOUT_ENV)
    if write is None or write <= 0:
        write = _DEFAULT_WRITE_TIMEOUT_S
    return httpx.Timeout(
        float(timeout),
        connect=min(connect, ceiling),
        write=min(write, ceiling),
        pool=min(_DEFAULT_POOL_TIMEOUT_S, ceiling),
    )


def resolve_stream_idle_timeout(configured: float | None) -> float | None:
    """Return the stream-idle budget in seconds, or ``None`` when disabled."""
    value = configured
    if value is None:
        value = _env_float(_IDLE_TIMEOUT_ENV)
    if value is None:
        value = DEFAULT_STREAM_IDLE_TIMEOUT_S
    return float(value) if value > 0 else None


# ---------------------------------------------------------------------------
# Stream-idle watchdog
# ---------------------------------------------------------------------------


class StreamIdleTimeoutError(Exception):
    """No streamed data event arrived within the idle budget."""

    def __init__(self, idle_seconds: float) -> None:
        super().__init__(f"no streamed data for {idle_seconds:g}s")
        self.idle_seconds = idle_seconds


def is_sse_data_line(line: str) -> bool:
    """True for an SSE ``data:`` line; comments and ``event:`` lines are not data.

    Gateways (OpenRouter) send ``: keep-alive`` comments while an upstream is
    stalled; counting them as progress would keep a dead stream open forever.
    """
    return line.startswith("data:")


def is_nonempty_line(line: str) -> bool:
    """Liveness test for NDJSON streams (Ollama): every non-empty line is data."""
    return bool(line.strip())


async def iter_lines_with_idle_timeout(
    lines: AsyncIterator[str],
    idle_timeout: float | None,
    *,
    is_data: Callable[[str], bool] = is_sse_data_line,
) -> AsyncIterator[str]:
    """Yield ``lines``, raising :class:`StreamIdleTimeoutError` on a stall.

    The watchdog arms on the first data line, not on the response headers: the
    wait for a first token (prompt prefill, a reasoning model thinking without
    streaming it, a gateway queue) stays governed by the read timeout and the
    engine's iteration timeout, as before. After that, the time between two
    data lines may not exceed ``idle_timeout``; non-data lines (SSE comments,
    blank separators) do not reset the clock.
    """
    iterator = lines.__aiter__()
    loop = asyncio.get_running_loop()
    deadline: float | None = None
    while True:
        try:
            if deadline is None or idle_timeout is None:
                line = await iterator.__anext__()
            else:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise StreamIdleTimeoutError(idle_timeout)
                try:
                    async with asyncio.timeout(remaining):
                        line = await iterator.__anext__()
                except TimeoutError:
                    raise StreamIdleTimeoutError(idle_timeout) from None
        except StopAsyncIteration:
            return
        if idle_timeout is not None and is_data(line):
            deadline = loop.time() + idle_timeout
        yield line


def stream_idle_error_event(exc: StreamIdleTimeoutError) -> ErrorEvent:
    return ErrorEvent(
        message=(
            f"Stream idle timeout: the provider sent no data for {exc.idle_seconds:g}s "
            "after the response had started"
        ),
        code=STREAM_IDLE_TIMEOUT_CODE,
    )


# ---------------------------------------------------------------------------
# Pooled client
# ---------------------------------------------------------------------------

_PENDING_CLOSES: set[asyncio.Task[Any]] = set()


def _spawn_close(client: Any) -> None:
    close = getattr(client, "aclose", None)
    if close is None:
        return
    try:
        task = asyncio.get_running_loop().create_task(close())
    except RuntimeError:
        return
    _PENDING_CLOSES.add(task)
    task.add_done_callback(_PENDING_CLOSES.discard)


def _close_soon(client: Any, loop: asyncio.AbstractEventLoop | None) -> None:
    """Close ``client`` on its own loop without awaiting (finalizer-safe)."""
    if loop is None or loop.is_closed():
        return
    try:
        loop.call_soon_threadsafe(_spawn_close, client)
    except RuntimeError:
        # The loop closed between the check and the call.
        return


class PooledHttpClient:
    """One reusable ``httpx.AsyncClient`` for a provider instance.

    The client is created lazily on first use, inside the running event loop,
    with the same ``proxy`` and ``trust_env`` handling the per-request clients
    had. An ``AsyncClient`` is safe to share between tasks of one loop; it is
    bound to that loop, so a call from another loop (a second ``asyncio.run``,
    a worker thread) gets its own client and the old one is closed on its own
    loop. A change of ``AGENTOS_TRUST_ENV`` also rebuilds the client.

    ``aclose()`` closes it; a provider dropped without ``aclose()`` still
    closes its client on its loop when it is garbage-collected.
    """

    def __init__(self, *, proxy: str | None = None) -> None:
        self._proxy = proxy or None
        self._client: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._trust_env: bool | None = None
        self._finalizer: weakref.finalize[Any, Any] | None = None
        #: How many clients this pool has built; tests assert reuse with it.
        self.clients_created = 0

    def get(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        trust_env = _trust_env()
        client = self._client
        if (
            client is not None
            and self._loop is loop
            and self._trust_env == trust_env
            and not getattr(client, "is_closed", False)
        ):
            return client  # type: ignore[no-any-return]
        self._release()
        client = httpx.AsyncClient(trust_env=trust_env, proxy=self._proxy)
        self._client = client
        self._loop = loop
        self._trust_env = trust_env
        self._finalizer = weakref.finalize(self, _close_soon, client, loop)
        self.clients_created += 1
        return client  # type: ignore[no-any-return]

    def _release(self) -> None:
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        client, loop = self._client, self._loop
        self._client = None
        self._loop = None
        if client is not None:
            _close_soon(client, loop)

    async def aclose(self) -> None:
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        client, loop = self._client, self._loop
        self._client = None
        self._loop = None
        if client is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        close = getattr(client, "aclose", None)
        if running is loop and close is not None:
            await close()
        else:
            _close_soon(client, loop)


async def aclose_provider(provider: Any) -> None:
    """Close ``provider``'s pooled client if it has one; never raises.

    For callers that build a provider for a short-lived event loop (one
    ``asyncio.run`` per call): a client left open there keeps its sockets until
    garbage collection, after the loop that owns them is closed, which leaks
    them with a ``ResourceWarning``.
    """
    close = getattr(provider, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception:  # noqa: BLE001 - cleanup must never fail the caller
        log.debug("provider.transport.aclose_failed", exc_info=True)


# ---------------------------------------------------------------------------
# Tool-call stream guard
# ---------------------------------------------------------------------------


async def guard_tool_stream(
    make_stream: Callable[[], AsyncIterator[StreamEvent]],
    *,
    provider: str,
    model: str,
    max_incomplete_tool_retries: int = 1,
) -> AsyncIterator[StreamEvent]:
    """Wrap one provider attempt so a stream cut inside a tool call can retry.

    A tool call's start and argument fragments are held until its
    ``ToolUseEndEvent`` arrives, then released together; text, thinking,
    heartbeats, done and errors pass straight through. A heartbeat names the
    tool while its arguments stream, so the caller still sees progress.

    When the stream ends with a tool call still open (no end, no done, no
    error), nothing about that call has reached the caller yet, so the attempt
    is re-run once, provided no text and no complete tool call was emitted
    either: re-running then would duplicate output the caller already has. In
    that case, or when the retry is cut too, an ``ErrorEvent`` is emitted
    instead of silently ending: ``incomplete_tool_stream`` when nothing was
    shown, ``stream_interrupted_after_output`` otherwise; neither is retried by
    the engine. A ``stream_idle_timeout`` after visible output is reported as
    ``stream_interrupted_after_output`` for the same reason.
    """
    attempt = 0
    while True:
        held: dict[str, list[StreamEvent]] = {}
        held_names: dict[str, str] = {}
        held_chars: dict[str, int] = {}
        emitted_output = False
        finished = False
        stream = make_stream()
        try:
            async for event in stream:
                if isinstance(event, ToolUseStartEvent):
                    held[event.tool_use_id] = [event]
                    held_names[event.tool_use_id] = event.tool_name
                    held_chars[event.tool_use_id] = 0
                    yield ProviderHeartbeatEvent(
                        phase="llm_tool_arguments",
                        message=f"Receiving {event.tool_name} arguments",
                    )
                    continue
                if isinstance(event, ToolUseDeltaEvent) and event.tool_use_id in held:
                    held[event.tool_use_id].append(event)
                    before = held_chars[event.tool_use_id]
                    after = before + len(event.json_fragment)
                    held_chars[event.tool_use_id] = after
                    if after // _TOOL_ARGUMENT_HEARTBEAT_CHARS > (
                        before // _TOOL_ARGUMENT_HEARTBEAT_CHARS
                    ):
                        yield ProviderHeartbeatEvent(
                            phase="llm_tool_arguments",
                            message=f"Receiving {held_names[event.tool_use_id]} arguments",
                        )
                    continue
                if isinstance(event, ToolUseEndEvent):
                    for buffered in held.pop(event.tool_use_id, ()):
                        yield buffered
                    held_names.pop(event.tool_use_id, None)
                    held_chars.pop(event.tool_use_id, None)
                    emitted_output = True
                    yield event
                    continue
                if isinstance(event, DoneEvent):
                    # A done with a call still open is a provider quirk the
                    # engine already reports (incomplete tools); release the
                    # held events so it sees exactly what it saw before.
                    for tool_use_id in list(held):
                        for buffered in held.pop(tool_use_id):
                            yield buffered
                    finished = True
                    yield event
                    return
                if isinstance(event, ErrorEvent):
                    finished = True
                    if event.code == STREAM_IDLE_TIMEOUT_CODE and emitted_output:
                        event = ErrorEvent(
                            message=f"{event.message}; partial output was already streamed",
                            code=STREAM_INTERRUPTED_AFTER_OUTPUT_CODE,
                            retry_after_seconds=event.retry_after_seconds,
                        )
                    yield event
                    return
                if isinstance(event, TextDeltaEvent) and event.text:
                    emitted_output = True
                yield event
        finally:
            closer = getattr(stream, "aclose", None)
            if closer is not None:
                await closer()

        if finished or not held:
            return
        open_tools = [held_names.get(tool_use_id, "") for tool_use_id in held]
        if not emitted_output and attempt < max_incomplete_tool_retries:
            attempt += 1
            log.warning(
                "provider.incomplete_tool_stream_retry",
                provider=provider,
                model=model,
                attempt=attempt,
                open_tools=open_tools,
            )
            yield ProviderHeartbeatEvent(
                phase="llm_retry",
                message="The provider stream ended inside a tool call; retrying once.",
            )
            continue
        log.warning(
            "provider.incomplete_tool_stream",
            provider=provider,
            model=model,
            attempts=attempt + 1,
            open_tools=open_tools,
            emitted_output=emitted_output,
        )
        # Tool names stay in the log above, never in the error text: the
        # engine scans error messages for words such as "thinking" to decide
        # on fallbacks, and a tool name (``sequentialthinking``) is model- or
        # MCP-supplied text that must not steer that.
        if emitted_output:
            yield ErrorEvent(
                message=(
                    "The provider stream ended inside a tool call after "
                    "partial output was already streamed"
                ),
                code=STREAM_INTERRUPTED_AFTER_OUTPUT_CODE,
            )
        else:
            yield ErrorEvent(
                message=(
                    f"The provider stream ended inside a tool call ({attempt + 1} attempt(s))"
                ),
                code=INCOMPLETE_TOOL_STREAM_CODE,
            )
        return
