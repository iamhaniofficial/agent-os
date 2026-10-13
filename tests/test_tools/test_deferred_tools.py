"""Deferred tool loading: tool_search lists the long tail and loads it on demand."""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

import agentos.tools.builtin  # noqa: F401 - registers the builtin tools
from agentos.engine.context_breakdown import tool_payload_chars
from agentos.engine.runtime import TurnRunner
from agentos.gateway.config import ToolsConfig
from agentos.identity.prompt import assemble_system_prompt
from agentos.identity.types import AgentIdentity, AgentProfile
from agentos.session.manager import SessionManager
from agentos.session.storage import SessionStorage
from agentos.tool_boundary import ToolCall, ToolResult
from agentos.tools import deferred
from agentos.tools.deferred import (
    DEFERRED_BUILTIN_TOOLS,
    TOOL_SEARCH_NAME,
    DeferredToolSurface,
    bind_agent_tools,
    deferred_tool_names,
    hydrate_loaded_tools,
    persist_loaded_with,
    prompt_tool_names,
    surface_for_turn,
)
from agentos.tools.dispatch import build_tool_handler
from agentos.tools.registry import ToolRegistry, get_default_registry
from agentos.tools.types import ToolContext, ToolSpec, current_tool_context

SESSION = "agent:main:deferred-test"

#: The default request must carry at least this many fewer tool schemas than
#: the full surface. 30 builtins are deferred today (tool_search adds one back).
MIN_SCHEMAS_SAVED = 25


async def _ok(**_kwargs: Any) -> str:
    return "ran"


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for name in ("read_file", "exec_command", "web_search"):
        registry.register(ToolSpec(name=name, description=f"{name} tool.", parameters={}), _ok)
    registry.register(
        ToolSpec(
            name="voice_maker",
            description="Synthesize speech from text. Returns an audio file.",
            parameters={"text": {"type": "string"}},
            required=["text"],
            defer_loading=True,
            execution_timeout_seconds=300,
        ),
        _ok,
    )
    registry.register(
        ToolSpec(
            name="project_admin",
            description="Rename or move projects.",
            parameters={},
            defer_loading=True,
        ),
        _ok,
    )
    search = get_default_registry().get(TOOL_SEARCH_NAME)
    assert search is not None
    registry.register(search.spec, search.handler)
    return registry


def _ctx(**overrides: Any) -> ToolContext:
    return ToolContext(session_key=SESSION, **overrides)


@pytest.fixture(autouse=True)
def _clean_session_state() -> Iterator[None]:
    deferred.forget_session(SESSION)
    yield
    deferred.forget_session(SESSION)


def _names(definitions: list[Any]) -> list[str]:
    return [d.name for d in definitions]


# ---------------------------------------------------------------------------
# What the request carries
# ---------------------------------------------------------------------------


def test_deferred_tools_are_left_out_and_tool_search_lists_them() -> None:
    registry = _registry()

    full = _names(registry.to_tool_definitions(_ctx()))
    request = registry.request_tool_definitions(_ctx())
    names = _names(request)

    assert "voice_maker" in full and "project_admin" in full
    assert TOOL_SEARCH_NAME not in full  # only offered when something is deferred
    assert names == ["exec_command", "read_file", TOOL_SEARCH_NAME, "web_search"]
    search = next(d for d in request if d.name == TOOL_SEARCH_NAME)
    assert "- voice_maker: Synthesize speech from text." in search.description
    assert "- project_admin: Rename or move projects." in search.description


def test_disabling_deferral_sends_every_tool() -> None:
    registry = _registry()

    names = _names(registry.request_tool_definitions(_ctx(), defer_loading=False))

    assert names == _names(registry.to_tool_definitions(_ctx()))
    assert TOOL_SEARCH_NAME not in names
    assert (
        surface_for_turn(
            registry, _ctx(), config=SimpleNamespace(tools=ToolsConfig(defer_loading=False))
        )
        is None
    )


def test_turn_runner_build_tools_honours_the_switch() -> None:
    registry = _registry()

    def build(config: object) -> list[str]:
        runner = TurnRunner(
            provider_selector=None,
            tool_registry=registry,
            session_manager=object(),
            config=config,
        )
        defs, handler = runner._build_tools(_ctx())
        assert handler is not None
        return _names(defs)

    on = build(SimpleNamespace(tools=ToolsConfig()))
    off = build(SimpleNamespace(tools=ToolsConfig(defer_loading=False)))

    assert ToolsConfig().defer_loading is True
    assert "voice_maker" not in on and TOOL_SEARCH_NAME in on
    assert "voice_maker" in off and TOOL_SEARCH_NAME not in off


def test_an_allowlisted_surface_defers_nothing() -> None:
    registry = _registry()
    ctx = _ctx(allowed_tools={"read_file", "voice_maker"})

    assert registry.deferred_tools(ctx) == []
    assert _names(registry.request_tool_definitions(ctx)) == ["read_file", "voice_maker"]


def test_a_surfaced_tool_is_loaded_up_front() -> None:
    registry = _registry()
    ctx = _ctx(surfaced_tools={"voice_maker"})

    assert "voice_maker" in _names(registry.request_tool_definitions(ctx))
    assert [rt.spec.name for rt in registry.deferred_tools(ctx)] == ["project_admin"]


def test_denying_tool_search_turns_deferral_off() -> None:
    registry = _registry()
    ctx = _ctx(denied_tools={TOOL_SEARCH_NAME})

    assert registry.deferred_tools(ctx) == []
    assert "voice_maker" in _names(registry.request_tool_definitions(ctx))


def test_default_request_is_much_smaller_than_the_full_surface() -> None:
    """Measurement: what deferral saves on the stock builtin set."""

    registry = get_default_registry()
    ctx = _ctx()

    full = registry.to_tool_definitions(ctx)
    request = registry.request_tool_definitions(ctx)
    full_chars = sum(tool_payload_chars(d) for d in full)
    request_chars = sum(tool_payload_chars(d) for d in request)

    assert len(full) - len(request) >= MIN_SCHEMAS_SAVED
    # The tool_search catalog is far cheaper than the schemas it stands for.
    assert full_chars - request_chars >= 10_000


def test_builtin_registration_sites_match_the_documented_set() -> None:
    registry = get_default_registry()
    flagged = {rt.spec.name for rt in registry.all_tools() if rt.spec.defer_loading}

    assert flagged == DEFERRED_BUILTIN_TOOLS & set(registry.list_names())
    # Never defer what a typical turn, a prompt section, or a bundled skill
    # needs without a search first.
    for name in (
        "read_file",
        "write_file",
        "edit_file",
        "apply_patch",
        "exec_command",
        "web_search",
        "web_fetch",
        "ask_user",
        "sessions_spawn",
        "skill_view",
        "image_generate",
        "publish_artifact",
        "cron",
        TOOL_SEARCH_NAME,
    ):
        rt = registry.get(name)
        if rt is not None:
            assert not rt.spec.defer_loading, name


# ---------------------------------------------------------------------------
# Searching and loading
# ---------------------------------------------------------------------------


async def test_search_loads_a_tool_into_the_next_request_and_the_running_agent() -> None:
    registry = _registry()
    surface = DeferredToolSurface.build(registry, _ctx())
    assert surface is not None
    live = surface.tool_definitions()
    by_name = {d.name: d for d in live}
    agent = SimpleNamespace(tool_definitions=live, _tool_definition_by_name=by_name)
    bind_agent_tools(surface.guard(_handler_returning("x")), agent)

    matches = surface.search("speech audio")
    newly = await surface.load(rt.spec.name for rt in matches)

    assert newly == ["voice_maker"]
    # Appended in place: the agent's next provider call carries it.
    assert live[-1].name == "voice_maker"
    assert by_name["voice_maker"].execution_timeout_seconds == 300
    # The next turn rebuilds the same order: always-loaded first, then loads.
    rebuilt = DeferredToolSurface.build(registry, _ctx())
    assert rebuilt is not None
    assert _names(rebuilt.tool_definitions()) == _names(live)


async def test_tool_search_tool_loads_by_name_and_says_so() -> None:
    registry = _registry()
    ctx = _ctx()
    surface = DeferredToolSurface.build(registry, ctx)
    assert surface is not None
    handler = surface.guard(build_tool_handler(registry, ctx))

    result = await handler(
        ToolCall(
            tool_use_id="t1",
            tool_name=TOOL_SEARCH_NAME,
            arguments={"names": ["voice_maker", "read_file"]},
        )
    )

    payload = json.loads(result.content)
    assert not result.is_error
    assert payload["loaded"] == ["voice_maker"]
    assert payload["not_deferred"] == ["read_file"]
    assert "now available" in payload["note"]
    assert "voice_maker" in _names(surface.tool_definitions())


async def test_calling_an_unloaded_deferred_tool_says_to_load_it_first() -> None:
    registry = _registry()
    ctx = _ctx()
    surface = DeferredToolSurface.build(registry, ctx)
    assert surface is not None
    handler = surface.guard(build_tool_handler(registry, ctx))
    call = ToolCall(tool_use_id="t2", tool_name="voice_maker", arguments={"text": "hi"})

    blocked = await handler(call)

    envelope = json.loads(blocked.content)
    assert blocked.is_error
    assert envelope["error_class"] == "ToolNotLoaded"
    assert 'tool_search(names=["voice_maker"])' in envelope["user_message"]
    assert "not found" not in envelope["user_message"].lower()

    await surface.load(["voice_maker"])
    ran = await handler(call)
    assert not ran.is_error
    assert "ran" in ran.content


async def test_a_policy_forbidden_tool_cannot_be_listed_or_loaded() -> None:
    registry = _registry()
    ctx = _ctx(denied_tools={"voice_maker"})
    surface = DeferredToolSurface.build(registry, ctx)
    assert surface is not None

    assert "voice_maker" not in surface.deferred
    search = next(d for d in surface.tool_definitions() if d.name == TOOL_SEARCH_NAME)
    assert "voice_maker" not in search.description
    assert surface.search("speech", ["voice_maker"]) == []
    assert await surface.load(["voice_maker"]) == []
    assert "voice_maker" not in _names(surface.tool_definitions())

    handler = surface.guard(build_tool_handler(registry, ctx))
    result = await handler(
        ToolCall(tool_use_id="t3", tool_name=TOOL_SEARCH_NAME, arguments={"names": ["voice_maker"]})
    )
    assert json.loads(result.content)["loaded"] == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


async def test_loaded_tools_survive_a_reload_and_a_reset_clears_them() -> None:
    storage = SessionStorage(":memory:")
    await storage.connect()
    try:
        manager = SessionManager(storage)
        registry = _registry()
        ctx = _ctx(artifact_session_id="sid-1")
        surface = surface_for_turn(registry, ctx, session_manager=manager)
        assert surface is not None
        await surface.load(["project_admin"])
        await surface.load(["voice_maker"])

        # A gateway restart: nothing in memory, only the session store.
        deferred.forget_session(SESSION)
        assert await hydrate_loaded_tools(manager, SESSION) == ["project_admin", "voice_maker"]
        reloaded = surface_for_turn(registry, ctx, session_manager=manager)
        assert reloaded is not None
        assert _names(reloaded.tool_definitions())[-2:] == ["project_admin", "voice_maker"]

        # "New chat" invalidates every context state of the session.
        await manager.invalidate_context_states(SESSION, reason="session_reset")
        assert await hydrate_loaded_tools(manager, SESSION) == []
        fresh = surface_for_turn(registry, ctx, session_manager=manager)
        assert fresh is not None
        assert "voice_maker" not in _names(fresh.tool_definitions())
    finally:
        await storage.close()


async def test_persist_callback_needs_a_store() -> None:
    assert persist_loaded_with(object(), SESSION, None) is None


# ---------------------------------------------------------------------------
# Prompt and skills see deferred tools as available
# ---------------------------------------------------------------------------


def test_prompt_lists_only_loaded_tools_but_keeps_tool_dependent_sections() -> None:
    registry = get_default_registry()
    surface = DeferredToolSurface.build(registry, _ctx())
    assert surface is not None
    request = surface.tool_definitions()

    tools, deferred_names = prompt_tool_names(request, SESSION)
    profile = AgentProfile(agent_id="main", identity=AgentIdentity(name="Test"))
    prompt = assemble_system_prompt(profile, tools=tools, deferred_tools=deferred_names)
    listing = prompt.split("## Available Tools", 1)[1].split("##", 1)[0]

    assert "- read_file" in listing
    assert "- tts" not in listing
    assert "`tool_search`" in listing
    assert set(deferred_names) <= set(tools)
    assert deferred_tool_names(SESSION) == deferred_names

    # Without deferral the prompt is exactly what it was.
    plain = assemble_system_prompt(profile, tools=["read_file", "tts"])
    assert plain == assemble_system_prompt(profile, tools=["read_file", "tts"], deferred_tools=[])
    assert "tool_search" not in plain


async def test_skill_view_loads_the_deferred_tools_a_skill_requires() -> None:
    from agentos.tools.builtin.skill_tools import _load_required_tools_note

    registry = _registry()
    ctx = _ctx()
    surface = DeferredToolSurface.build(registry, ctx)
    assert surface is not None
    skill = SimpleNamespace(requires_tools=["voice_maker", "read_file"])

    token = current_tool_context.set(ctx)
    try:
        note = await _load_required_tools_note(skill)
        again = await _load_required_tools_note(skill)
    finally:
        current_tool_context.reset(token)

    assert note == "[Loaded tools this skill uses: voice_maker]\n\n"
    assert again == ""
    assert surface.loaded == ["voice_maker"]


def _handler_returning(content: str) -> Any:
    async def _handler(call: ToolCall) -> ToolResult:
        return ToolResult(tool_use_id=call.tool_use_id, tool_name=call.tool_name, content=content)

    return _handler
