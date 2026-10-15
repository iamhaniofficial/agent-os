"""sessions_spawn(runtime="acp") gates, cwd confinement, resume authorization,
subagents list/kill/steer for ACP children, and the ``acp`` tool."""

from __future__ import annotations

import json

import pytest
from acp_s_fakes import FakeRuntime, Script
from acp_s_harness import PARENT_KEY, acp_rig

from agentos.session.keys import (
    allows_private_memory_prompt_injection,
    build_acp_session_key,
    is_acp_key,
    is_child_session_key,
)
from agentos.tools.builtin import acp_tools
from agentos.tools.builtin import agents as agents_tool
from agentos.tools.builtin import sessions as sessions_tool
from agentos.tools.types import (
    CRON_AGENT_DENY,
    SUBAGENT_TOOL_DENY,
    ToolError,
    current_tool_context,
)


async def _spawn(**kwargs) -> dict:
    return json.loads(await sessions_tool.sessions_spawn(runtime="acp", **kwargs))


def test_acp_key_helpers() -> None:
    key = build_acp_session_key("Main", "ABCD1234")
    assert key == "agent:main:acp:abcd1234"
    assert is_acp_key(key) and is_child_session_key(key)
    assert not is_acp_key("agent:main:subagent:abcd1234")
    assert not is_acp_key("agent:main:acp:x:y")
    assert allows_private_memory_prompt_injection(key) is False


def test_acp_tool_is_denied_to_subagents_and_cron() -> None:
    assert "acp" in SUBAGENT_TOOL_DENY and "acp" in CRON_AGENT_DENY


async def test_spawn_creates_row_and_enqueues_acp_run(tmp_path) -> None:
    async with acp_rig(tmp_path) as rig:
        out = await _spawn(task="write tests", acp_agent="codex", harness_mode="read-only")
        key = out["session_key"]
        assert is_acp_key(key) and key.startswith("agent:main:acp:")
        assert out["runtime"] == "acp" and out["mode"] == "oneshot"
        row = await rig.sessions.get_session(key)
        assert row.spawned_by == PARENT_KEY and row.spawn_depth == 1
        assert row.origin["kind"] == "acp"
        assert row.origin["agent_id"] == "codex"
        assert row.origin["cwd"] == str(rig.workspace.resolve())
        assert row.origin["parent_task_id"] == "task-parent"
        assert row.origin["harness_mode"] == "read-only"
        tasks = await rig.task_runtime.list(session_key=key)
        assert tasks[0].run_kind == "acp"
        await rig.task_runtime.wait(out["task_id"], timeout=5)
        spec = rig.runtime.calls[0][1]
        assert (spec.agent_id, spec.harness_mode, spec.mode) == ("codex", "read-only", "oneshot")
        # sessions.send-style persisted user row: not duplicated by dispatch
        transcript = await rig.sessions.read_recent_transcript(key, 10)
        assert [r["role"] for r in transcript].count("user") == 1


async def test_spawn_defaults_to_config_default_agent(tmp_path) -> None:
    async with acp_rig(tmp_path, default_agent="gemini") as _rig:
        out = await _spawn(task="x")
        assert out["acp_agent"] == "gemini"


async def test_invalid_runtime_and_mode(tmp_path) -> None:
    async with acp_rig(tmp_path):
        with pytest.raises(ToolError, match="runtime must be"):
            await sessions_tool.sessions_spawn(task="x", runtime="docker")
        with pytest.raises(ToolError, match="mode must be"):
            await _spawn(task="x", acp_agent="claude", mode="forever")


async def test_policy_gates(tmp_path) -> None:
    async with acp_rig(tmp_path, allowed_agents=["claude"]):
        with pytest.raises(ToolError, match="allowed_agents"):
            await _spawn(task="x", acp_agent="codex")
        with pytest.raises(ToolError, match="Unknown ACP harness"):
            await _spawn(task="x", acp_agent="no-such-harness")
    async with acp_rig(tmp_path / "off", enabled=False):
        with pytest.raises(ToolError, match="disabled"):
            await _spawn(task="x", acp_agent="claude")


async def test_override_only_harness_is_spawnable(tmp_path) -> None:
    async with acp_rig(tmp_path, agents={"mybot": {"command": "/bin/mybot"}}):
        out = await _spawn(task="x", acp_agent="mybot")
        assert out["acp_agent"] == "mybot"


async def test_manager_unavailable_refuses_spawn(tmp_path) -> None:
    from agentos.acp.manager import AcpSessionManager, set_acp_manager

    async with acp_rig(tmp_path):
        set_acp_manager(AcpSessionManager(None, unavailable_reason="runtime missing"))
        with pytest.raises(ToolError, match="ACP unavailable: runtime missing"):
            await _spawn(task="x", acp_agent="claude")


async def test_cwd_confinement(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    project = tmp_path / "projects" / "app"
    project.mkdir(parents=True)
    async with acp_rig(tmp_path, allowed_cwds=[str(tmp_path / "projects")]) as rig:
        (rig.workspace / "sub").mkdir()
        rel = await _spawn(task="x", acp_agent="claude", cwd="sub")
        assert rel["cwd"] == str((rig.workspace / "sub").resolve())
        ok = await _spawn(task="x", acp_agent="claude", cwd=str(project))
        assert ok["cwd"] == str(project.resolve())
        with pytest.raises(ToolError, match="outside the agent workspace"):
            await _spawn(task="x", acp_agent="claude", cwd=str(outside))
        with pytest.raises(ToolError, match="not an existing directory"):
            await _spawn(task="x", acp_agent="claude", cwd=str(tmp_path / "missing"))
        # A symlink inside the workspace pointing outside is still outside.
        (rig.workspace / "escape").symlink_to(outside)
        with pytest.raises(ToolError, match="outside"):
            await _spawn(task="x", acp_agent="claude", cwd="escape")


async def test_depth_and_max_children_gates(tmp_path) -> None:
    from agentos.tools.types import CallerKind, ToolContext

    async with acp_rig(tmp_path) as rig:
        token = current_tool_context.set(
            ToolContext(
                caller_kind=CallerKind.AGENT,
                subagent_depth=sessions_tool._MAX_SPAWN_DEPTH,
                agent_id="main",
                session_key=PARENT_KEY,
                task_id="t",
                workspace_dir=str(rig.workspace),
            )
        )
        try:
            with pytest.raises(ToolError, match="Max spawn depth"):
                await _spawn(task="x", acp_agent="claude")
        finally:
            current_tool_context.reset(token)

        async def _policy(_mgr, _agent):
            return {"max_children_per_session": 1}

        original = sessions_tool._resolve_subagent_policy
        sessions_tool._resolve_subagent_policy = _policy  # type: ignore[assignment]
        try:
            rig.runtime._scripts = [Script(block=True)]
            await _spawn(task="first", acp_agent="claude", mode="persistent")
            with pytest.raises(ToolError, match="Max active children"):
                await _spawn(task="second", acp_agent="claude")
        finally:
            sessions_tool._resolve_subagent_policy = original  # type: ignore[assignment]
            rig.runtime.release.set()


async def test_resume_authorization(tmp_path) -> None:
    async with acp_rig(tmp_path) as rig:
        out = await _spawn(task="x", acp_agent="claude", mode="persistent")
        await rig.task_runtime.wait(out["task_id"], timeout=5)
        recorded = (await rig.sessions.get_session(out["session_key"])).origin["acp_session_id"]
        assert recorded == "sess-1"

        resumed = await _spawn(task="continue", acp_agent="claude", resume_session_id=recorded)
        await rig.task_runtime.wait(resumed["task_id"], timeout=5)
        spec = [c[1] for c in rig.runtime.calls if c[0] == "ensure_session"][-1]
        assert spec.resume_session_id == "sess-1"
        assert spec.extra["record_id"] == "rec-1"

        with pytest.raises(ToolError, match="resume_forbidden"):
            await _spawn(task="x", acp_agent="claude", resume_session_id="sess-not-mine")
        with pytest.raises(ToolError, match="different harness"):
            await _spawn(task="x", acp_agent="codex", resume_session_id=recorded)


async def test_worktree_requires_flag(tmp_path) -> None:
    async with acp_rig(tmp_path):
        with pytest.raises(ToolError, match="worktree=true"):
            await _spawn(task="x", acp_agent="claude", worktree_name="feature")


async def test_subagents_list_kill_and_steer_for_acp(tmp_path) -> None:
    runtime = FakeRuntime([Script(block=True), Script(block=True)])
    async with acp_rig(tmp_path, runtime) as rig:
        out = await _spawn(task="long", acp_agent="claude", mode="persistent")
        key = out["session_key"]
        listed = json.loads(await agents_tool.subagents(action="list"))["subagents"]
        assert listed[0]["runtime"] == "acp" and listed[0]["acp_agent"] == "claude"

        steer = json.loads(
            await agents_tool.subagents(action="steer", session_key=key, message="pivot")
        )
        assert steer["runtime"] == "acp" and steer["mode"] == "steer"

        killed = json.loads(await agents_tool.subagents(action="kill", session_key=key))
        assert killed == {
            "action": "kill",
            "session_key": key,
            "status": "killed",
            "runtime": "acp",
        }
        row = await rig.sessions.get_session(key)
        assert row.status == "killed" and row.origin["closed"] is True
        assert runtime.closed  # harness closed


async def test_acp_tool_actions(tmp_path) -> None:
    runtime = FakeRuntime()
    runtime.config_options = [{"id": "model"}, {"id": "effort"}]
    async with acp_rig(tmp_path, runtime) as rig:
        agents = json.loads(await acp_tools.acp(action="agents"))
        ids = {a["agent_id"] for a in agents["agents"]}
        assert {"claude", "codex", "gemini"} <= ids
        assert agents["default_agent"] == "claude"

        doctor = json.loads(await acp_tools.acp(action="doctor", agent="claude"))
        assert doctor["probe"] == "ok"

        out = await _spawn(task="x", acp_agent="claude", mode="persistent")
        key = out["session_key"]
        await rig.task_runtime.wait(out["task_id"], timeout=5)

        status = json.loads(await acp_tools.acp(action="status", session_key=key))
        assert status["live"] is True and status["record_id"] == "rec-1"
        assert status["last_turn"]["stop_reason"] == "end_turn"

        sessions = json.loads(await acp_tools.acp(action="sessions"))
        assert sessions["live"][0]["session_key"] == key
        assert sessions["children"][0]["acp_agent"] == "claude"

        mode = json.loads(await acp_tools.acp(action="set_mode", session_key=key, value="plan"))
        assert mode["mode"] == "plan"
        model = json.loads(await acp_tools.acp(action="set_model", session_key=key, value="opus"))
        assert model["option"] == "model"
        thinking = json.loads(
            await acp_tools.acp(
                action="set_model", session_key=key, value="high", option="thinking"
            )
        )
        assert thinking["option"] == "effort"
        with pytest.raises(ToolError, match="ACP_OPTION_UNSUPPORTED"):
            await acp_tools.acp(
                action="set_model", session_key=key, value="1", option="temperature"
            )

        cancel = json.loads(await acp_tools.acp(action="cancel", session_key=key))
        assert cancel["cancelled"] is False
        closed = json.loads(await acp_tools.acp(action="close", session_key=key))
        assert closed["closed"] is True and closed["was_live"] is True

        with pytest.raises(ToolError, match="not an ACP session"):
            await acp_tools.acp(action="status", session_key=PARENT_KEY)


async def test_acp_tool_refuses_other_agents_sessions(tmp_path) -> None:
    async with acp_rig(tmp_path) as rig:
        other = "agent:ops:acp:deadbeef"
        await rig.sessions.create(
            other,
            agent_id="ops",
            spawned_by="agent:ops:main",
            origin={"kind": "acp", "agent_id": "claude", "cwd": str(rig.workspace)},
        )
        with pytest.raises(ToolError, match="another agent"):
            await acp_tools.acp(action="close", session_key=other)
