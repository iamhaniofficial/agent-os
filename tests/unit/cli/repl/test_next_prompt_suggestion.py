"""Suggested next prompt in the terminal chat (`chat.suggestNext`).

Ghost text over an empty input; Tab takes it; Enter sends it. The gateway
runtime asks for it in the background after each turn and hands it to the
output handle.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.input.base import DummyInput
from prompt_toolkit.layout.processors import TransformationInput
from prompt_toolkit.output import DummyOutput

from agentos.cli.repl.session_state import ChatSessionState
from agentos.cli.repl.stream import TurnResult, UsageSummary
from agentos.cli.tui.terminal.app import ChatApplication, _build_key_bindings
from agentos.cli.tui.terminal.surface import TerminalOutputHandle
from agentos.engine.commands import Surface

# --------------------------------------------------------------------------- #
# ChatApplication                                                             #
# --------------------------------------------------------------------------- #


def _chat_app() -> ChatApplication:
    return ChatApplication(
        surface=Surface.CLI_GATEWAY,
        toolbar_context={"model": None, "session_id": None, "suppress": None, "status": None},
        bottom_toolbar=lambda: "",
        style=None,
        input=DummyInput(),
        output=DummyOutput(),
    )


def _ghost_text(chat_app: ChatApplication, text: str = "") -> str:
    processor = chat_app._next_prompt_processor
    document = Document(text, len(text))
    ti = TransformationInput(
        buffer_control=cast(Any, None),
        document=document,
        lineno=0,
        source_to_display=lambda i: i,
        fragments=[("", text)],
        width=80,
        height=1,
    )
    fragments = processor.apply_transformation(ti).fragments
    return "".join(part[1] for part in fragments[1:])


def _tab_binding():
    bindings = [b for b in _build_key_bindings().bindings if tuple(b.keys) == ("c-i",)]
    assert len(bindings) == 1, f"expected one tab binding, got {len(bindings)}"
    return bindings[0]


class _FakeEvent:
    def __init__(self, chat_app: ChatApplication) -> None:
        self.app = chat_app.application
        self.current_buffer = chat_app._buffer


def test_ghost_text_shows_over_an_empty_input_only() -> None:
    chat_app = _chat_app()
    assert _ghost_text(chat_app) == ""

    chat_app.set_next_prompt_suggestion("Now add tests")
    assert "Now add tests" in _ghost_text(chat_app)
    assert "Tab" in _ghost_text(chat_app)
    assert _ghost_text(chat_app, "typed") == ""


def test_tab_takes_the_suggestion_into_the_buffer() -> None:
    chat_app = _chat_app()
    chat_app.set_next_prompt_suggestion("Now add tests")
    binding = _tab_binding()

    assert chat_app.next_prompt_suggestion_ready() is True
    binding.handler(_FakeEvent(chat_app))

    assert chat_app._buffer.text == "Now add tests"
    assert chat_app._buffer.cursor_position == len("Now add tests")
    # With text in the buffer the binding steps aside for completion.
    assert chat_app.next_prompt_suggestion_ready() is False


def test_tab_binding_is_filtered_to_a_ready_suggestion() -> None:
    chat_app = _chat_app()
    assert chat_app.next_prompt_suggestion_ready() is False

    chat_app.set_next_prompt_suggestion("Ship it")
    chat_app._buffer.text = "/he"
    assert chat_app.next_prompt_suggestion_ready() is False

    chat_app._buffer.text = ""
    chat_app.set_next_prompt_suggestion(None)
    assert chat_app.next_prompt_suggestion_ready() is False
    # The binding carries a filter, so Tab falls through to completion.
    assert _tab_binding().filter is not None


def test_submitting_retires_the_suggestion() -> None:
    chat_app = _chat_app()
    chat_app.set_next_prompt_suggestion("Ship it")
    chat_app._buffer.text = "Ship it"
    chat_app._buffer.validate_and_handle()

    assert chat_app.next_prompt_suggestion() is None


def test_output_handle_forwards_the_suggestion() -> None:
    chat_app = _chat_app()
    handle = TerminalOutputHandle(chat_app, approval_surface=Surface.CLI_GATEWAY)

    handle.set_suggestion("Deploy it")
    assert chat_app.next_prompt_suggestion() == "Deploy it"
    handle.set_suggestion(None)
    assert chat_app.next_prompt_suggestion() is None


def test_production_chain_reaches_the_chat_app() -> None:
    """The handle the runtime gets is TerminalSurface.output_handle over the session handle."""
    from agentos.cli.tui.terminal.prompt import InteractiveSessionHandle
    from agentos.cli.tui.terminal.surface import TerminalSurface

    chat_app = _chat_app()
    surface = TerminalSurface(InteractiveSessionHandle(chat_app))

    surface.output_handle.set_suggestion("Deploy it")
    assert chat_app.next_prompt_suggestion() == "Deploy it"


def test_output_handle_tolerates_a_source_without_suggestions() -> None:
    class _Source:
        async def write_through(self, payload: str) -> None: ...

        def stream_output(self):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        async def clear_screen(self) -> None: ...

    TerminalOutputHandle(_Source(), approval_surface=Surface.CLI_GATEWAY).set_suggestion("x")


# --------------------------------------------------------------------------- #
# Gateway runtime                                                             #
# --------------------------------------------------------------------------- #


class _Output:
    approval_surface = Surface.CLI_GATEWAY

    def __init__(self) -> None:
        self.suggestions: list[str | None] = []

    def set_suggestion(self, text: str | None) -> None:
        self.suggestions.append(text)


def _client_class(answer: Any, *, gate: asyncio.Event | None = None):
    class _Client:
        instances: list[Any] = []

        def __init__(self) -> None:
            self.calls: list[tuple[str, Any]] = []
            self.closed = False
            _Client.instances.append(self)

        async def connect(self, url: str, *, token: str | None = None) -> None: ...

        async def create_session(self, model: str | None = None) -> str:
            return "agent:main:new"

        async def resolve_session(self, key: str) -> dict[str, str]:
            return {}

        async def call(self, method: str, params: dict | None = None) -> Any:
            self.calls.append((method, params))
            if gate is not None:
                await gate.wait()
            if isinstance(answer, Exception):
                raise answer
            return answer

        async def close(self) -> None:
            self.closed = True

    return _Client


async def _run(
    monkeypatch: pytest.MonkeyPatch,
    client_cls: Any,
    output: Any,
    script,  # type: ignore[no-untyped-def]
) -> None:
    from agentos.cli.repl import gateway_runtime

    monkeypatch.setattr("agentos.cli.gateway_client.GatewayClient", client_cls)

    async def fake_stream_response(*_a: Any, **_kw: Any) -> TurnResult:
        return TurnResult(text="done", usage=UsageSummary(), model_after=None)

    async def fake_slash(cmd: str, state: ChatSessionState, *_a: Any, **_kw: Any) -> bool:
        return True

    async def fake_loop(*, scope, dispatch, abort_active_turn=None) -> None:  # type: ignore[no-untyped-def]
        await script(dispatch)

    deps = gateway_runtime.GatewayRuntimeDependencies(
        stream_response=fake_stream_response,
        handle_slash_command=fake_slash,
        run_input_loop=fake_loop,
        get_tui_output=lambda _scope: output,
        is_exit_command=lambda value: value.strip() == "/exit",
        notify=lambda _notice: None,
    )
    await gateway_runtime.run_gateway_chat(model=None, session_id=None, deps=deps)


@pytest.mark.asyncio
async def test_runtime_asks_after_a_turn_and_shows_the_answer(monkeypatch) -> None:
    client_cls = _client_class({"text": "Now add tests"})
    output = _Output()

    async def script(dispatch) -> None:  # type: ignore[no-untyped-def]
        assert await dispatch("write a parser") is True
        for _ in range(5):
            await asyncio.sleep(0)

    await _run(monkeypatch, client_cls, output, script)

    client = client_cls.instances[-1]
    assert ("chat.suggestNext", {"sessionKey": "agent:main:new"}) in client.calls
    assert output.suggestions == [None, "Now add tests"]


@pytest.mark.asyncio
async def test_runtime_drops_an_answer_overtaken_by_the_next_input(monkeypatch) -> None:
    gate = asyncio.Event()
    client_cls = _client_class({"text": "stale"}, gate=gate)
    output = _Output()

    async def script(dispatch) -> None:  # type: ignore[no-untyped-def]
        assert await dispatch("first") is True
        await asyncio.sleep(0)
        assert await dispatch("/status") is True
        gate.set()
        for _ in range(5):
            await asyncio.sleep(0)

    await _run(monkeypatch, client_cls, output, script)

    assert "stale" not in output.suggestions


@pytest.mark.asyncio
async def test_runtime_swallows_errors_and_skips_outputs_without_suggestions(monkeypatch) -> None:
    output = _Output()

    async def script(dispatch) -> None:  # type: ignore[no-untyped-def]
        assert await dispatch("hello") is True
        for _ in range(5):
            await asyncio.sleep(0)

    await _run(monkeypatch, _client_class(RuntimeError("Unknown method")), output, script)
    assert output.suggestions == [None]

    plain_cls = _client_class({"text": "unused"})
    await _run(monkeypatch, plain_cls, object(), script)
    assert plain_cls.instances[-1].calls == []


@pytest.mark.asyncio
async def test_runtime_cancels_a_pending_ask_before_closing(monkeypatch) -> None:
    gate = asyncio.Event()
    client_cls = _client_class({"text": "late"}, gate=gate)
    output = _Output()

    async def script(dispatch) -> None:  # type: ignore[no-untyped-def]
        assert await dispatch("hello") is True
        await asyncio.sleep(0)

    await _run(monkeypatch, client_cls, output, script)

    assert client_cls.instances[-1].closed is True
    gate.set()
    await asyncio.sleep(0)
    assert "late" not in output.suggestions
