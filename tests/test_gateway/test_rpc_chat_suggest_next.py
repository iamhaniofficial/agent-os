"""chat.suggestNext RPC + the [sessions].suggest_next gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import pytest_asyncio

from agentos.gateway import session_suggester as suggester_mod
from agentos.gateway.config import SessionsConfig
from agentos.gateway.rpc import RpcContext
from agentos.gateway.rpc_chat import _handle_chat_suggest_next
from agentos.gateway.session_suggester import reset_suggesters
from agentos.provider import auxiliary as aux
from agentos.provider.types import DoneEvent, TextDeltaEvent
from agentos.session.manager import SessionManager
from agentos.session.storage import SessionStorage

KEY = "agent:main:webchat:abc12345"


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    for name in ("AGENTOS_SUGGEST_NEXT_MODEL", "AGENTOS_SUGGEST_NEXT_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    reset_suggesters()
    yield
    reset_suggesters()


@pytest_asyncio.fixture
async def manager():
    storage = SessionStorage(":memory:")
    await storage.connect()
    mgr = SessionManager(storage, inject_time_prefix=False)
    await mgr.create(KEY)
    await mgr.append_message(KEY, "user", "Write a parser")
    await mgr.append_message(KEY, "assistant", "Done, see parser.py")
    yield mgr
    await storage.close()


def _stub_client(monkeypatch: pytest.MonkeyPatch, text: str) -> list[dict]:
    calls: list[dict] = []

    class _Provider:
        async def chat(self, *, messages, config=None, tools=None):
            calls.append({"messages": messages})
            yield TextDeltaEvent(text=text)
            yield DoneEvent(input_tokens=1, output_tokens=1)

    client = aux.AuxiliaryClient(provider_factory=lambda **_kw: _Provider())
    monkeypatch.setattr(suggester_mod, "get_auxiliary_client", lambda: client)
    return calls


def _ctx(mgr, **sessions) -> RpcContext:
    config = SimpleNamespace(sessions=SessionsConfig(**sessions), agentos_router=None)
    return RpcContext(conn_id="test", session_manager=mgr, config=config)


@pytest.mark.asyncio
async def test_returns_the_cleaned_suggestion(manager, monkeypatch):
    _stub_client(monkeypatch, '"Now add tests"')

    result = await _handle_chat_suggest_next({"sessionKey": KEY}, _ctx(manager))

    assert result == {"text": "Now add tests"}


@pytest.mark.asyncio
async def test_disabled_flag_returns_null_without_a_model_call(manager, monkeypatch):
    calls = _stub_client(monkeypatch, "Now add tests")

    result = await _handle_chat_suggest_next({"sessionKey": KEY}, _ctx(manager, suggest_next=False))

    assert result == {"text": None}
    assert calls == []


@pytest.mark.asyncio
async def test_no_session_manager_returns_null():
    result = await _handle_chat_suggest_next({"sessionKey": KEY}, RpcContext(conn_id="t"))

    assert result == {"text": None}


@pytest.mark.asyncio
async def test_key_is_canonicalised_like_chat_send(manager, monkeypatch):
    _stub_client(monkeypatch, "Ship it")

    # chat.send maps a legacy ``sess-`` id onto the agent-scoped WebChat key.
    result = await _handle_chat_suggest_next({"sessionKey": "sess-abc12345"}, _ctx(manager))

    assert result == {"text": "Ship it"}


def test_sessions_config_defaults_and_env(monkeypatch: pytest.MonkeyPatch):
    assert SessionsConfig().suggest_next is True
    assert SessionsConfig().suggest_next_timeout_seconds == 15.0
    monkeypatch.setenv("AGENTOS_SESSIONS_SUGGEST_NEXT", "false")
    assert SessionsConfig().suggest_next is False
    with pytest.raises(ValueError):
        SessionsConfig(suggest_next_timeout_seconds=0)
