"""AcpRuntimeImpl end to end against the stdlib fake ACP agent."""

from __future__ import annotations

import asyncio
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from acp_runtime_helpers import FAKE_AGENT, collect, fake_config, make_runtime, wait_until

from agentos.acp import process as proc_mod
from agentos.acp.runtime import AcpRuntimeImpl, usage_from_mapping
from agentos.acp.types import (
    AcpAgentNotAllowedError,
    AcpError,
    AcpRuntime,
    AcpSessionInitError,
    AcpSessionSpec,
    AcpTurnError,
    AcpTurnTimeoutError,
    ElicitationRequest,
    ElicitationResponse,
    PermissionPromptUnavailableError,
    PermissionRequest,
)
from agentos.gateway.config import AcpAgentOverride

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")

KEY = "agent:main:acp:0badf00d"


def _spec(tmp_path: Path, script: str = "echo", **kw: Any) -> AcpSessionSpec:
    env = {"FAKE_ACP_SCRIPT": script, **kw.pop("env", {})}
    return AcpSessionSpec(agent_id="fake", cwd=str(tmp_path), env=env, **kw)


@pytest.fixture
async def runtimes():
    made: list[AcpRuntimeImpl] = []

    def factory(tmp_path: Path, *args: Any, **kwargs: Any) -> AcpRuntimeImpl:
        rt = make_runtime(tmp_path, *args, **kwargs)
        made.append(rt)
        return rt

    yield factory
    for rt in made:
        await rt.aclose()


async def test_runtime_satisfies_protocol(tmp_path: Path, runtimes) -> None:
    rt: AcpRuntime = runtimes(tmp_path)
    assert isinstance(rt, AcpRuntimeImpl)


async def test_echo_turn_events_result_and_persisted_record(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path), session_key=KEY)
    assert (
        h.status == "idle" and h.pid and h.acp_session_id and h.acp_session_id.startswith("fake-")
    )
    assert h.agent_capabilities["_agent_info"]["version"] == "9.9.9"
    assert [m["id"] for m in h.available_modes] == ["default", "plan", "acceptEdits"]
    assert proc_mod.read_lease(rt.state_dir, h.lease_id or "") is not None

    turn = await rt.start_turn(h, text="hello world", session_key=KEY)
    events = await collect(turn)
    result = await turn.result
    kinds = [e.kind for e in events]
    assert kinds[-1] == "done" and events[-1].stop_reason == "end_turn"
    assert "text_delta" in kinds and "usage" in kinds
    text = "".join(e.text for e in events if e.kind == "text_delta")
    assert text == "echo: hello world"
    assert result.stop_reason == "end_turn" and result.text == "echo: hello world"
    assert result.usage.input_tokens == 10 and result.usage.output_tokens == 5
    assert result.usage.total_tokens == 15
    assert result.usage.cost_usd == pytest.approx(0.01)
    assert result.usage.context_used == 100 and result.usage.context_size == 1000
    assert result.usage.model == "fake-small"
    assert result.tool_calls == 0
    rec = rt.store.load(h.record_id)
    assert (
        rec is not None and rec.session_key == KEY and rec.handle.acp_session_id == h.acp_session_id
    )
    assert (await rt.status(h)).status == "idle"


async def test_cost_is_reported_per_turn_not_cumulative(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path), session_key=KEY)
    costs = []
    for i in range(3):
        turn = await rt.start_turn(h, text=f"t{i}", session_key=KEY)
        await collect(turn)
        costs.append((await turn.result).usage.cost_usd)
    assert costs == [pytest.approx(0.01)] * 3
    turn = await rt.start_turn(h, text="script:tool_then_text go", session_key=KEY)
    await collect(turn)
    assert (await turn.result).usage.cost_usd is None  # no usage_update this turn


async def test_tool_calls_and_plan_and_meta_quota(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path, "tool_then_text"), session_key=KEY)
    turn = await rt.start_turn(h, text="read it", session_key=KEY)
    events = await collect(turn)
    tools = [e for e in events if e.kind in ("tool_call", "tool_call_update")]
    assert [e.status for e in tools] == ["pending", "in_progress", "completed"]
    assert all(e.title == "Read README.md" and e.tool_kind == "read" for e in tools)
    assert tools[-1].raw_output == {"content": "hello"} and tools[-1].text == "hello"
    result = await turn.result
    assert result.tool_calls == 1 and result.text == "done"

    turn = await rt.start_turn(h, text="script:plan_usage x", session_key=KEY)
    events = await collect(turn)
    by_kind = {e.kind: e for e in events}
    assert by_kind["plan"].text.startswith("[x] Read code")
    assert by_kind["thought_delta"].text == "thinking..."
    assert any(e.kind == "status" and e.title == "Fix the bug" for e in events)
    result = await turn.result
    assert result.usage.input_tokens == 120 and result.usage.output_tokens == 30
    assert result.usage.total_tokens == 150 and result.usage.cost_usd == pytest.approx(0.25)


async def test_startup_command_blob_never_becomes_text(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path), session_key=KEY)
    turn = await rt.start_turn(h, text="x", session_key=KEY)
    events = await collect(turn)
    assert all("x" * 200 not in e.text for e in events)
    assert all(e.tag != "available_commands_update" for e in events)
    assert (await turn.result).text == "echo: x"


class _Perm:
    def __init__(self, decision: Any = "allow_once", exc: Exception | None = None) -> None:
        self.decision, self.exc = decision, exc
        self.requests: list[PermissionRequest] = []

    async def decide(self, request: PermissionRequest) -> Any:
        self.requests.append(request)
        if self.exc:
            raise self.exc
        return self.decision


@pytest.mark.parametrize(
    ("decision", "text"),
    [
        ("allow_once", "permission: allowed (a1)"),
        ("reject_once", "permission: rejected (r1)"),
        ("cancel", "permission: cancelled"),
    ],
)
async def test_permission_bridge_decides(tmp_path: Path, runtimes, decision, text) -> None:
    bridge = _Perm(decision)
    rt = runtimes(tmp_path, permission_bridge=bridge)
    h = await rt.ensure_session(_spec(tmp_path, "permission_write"), session_key=KEY)
    turn = await rt.start_turn(h, text="write", session_key=KEY)
    await collect(turn)
    result = await turn.result
    assert result.text == text
    assert result.stop_reason == ("cancelled" if decision == "cancel" else "end_turn")
    req = bridge.requests[0]
    assert req.session_key == KEY and req.acp_session_id == h.acp_session_id
    assert req.title == "Write hello.txt" and req.tool_kind == "edit"
    assert req.raw_input == {"path": "hello.txt", "content": "hi"}
    assert req.cwd == str(tmp_path) and req.agent_id == "fake"


async def test_permission_unavailable_fails_turn_with_code(tmp_path: Path, runtimes) -> None:
    bridge = _Perm(exc=PermissionPromptUnavailableError("no operator surface"))
    rt = runtimes(tmp_path, permission_bridge=bridge)
    h = await rt.ensure_session(_spec(tmp_path, "permission_write"), session_key=KEY)
    turn = await rt.start_turn(h, text="write", session_key=KEY)
    events = await collect(turn)
    assert events[-1].kind == "error"
    assert events[-1].error and events[-1].error.startswith("ACP_PERMISSION_PROMPT_UNAVAILABLE")
    with pytest.raises(PermissionPromptUnavailableError):
        await turn.result
    # the harness answered cancel and stays usable
    turn = await rt.start_turn(h, text="script:echo again", session_key=KEY)
    await collect(turn)
    assert (await turn.result).text == "echo: again"


class _Elicit:
    def __init__(self) -> None:
        self.requests: list[ElicitationRequest] = []

    async def ask(self, request: ElicitationRequest) -> ElicitationResponse:
        self.requests.append(request)
        return ElicitationResponse(action="accept", content={"color": "teal"})


async def test_elicitation_bridge(tmp_path: Path, runtimes) -> None:
    bridge = _Elicit()
    rt = runtimes(tmp_path, elicitation_bridge=bridge)
    h = await rt.ensure_session(_spec(tmp_path, "elicit_form"), session_key=KEY)
    turn = await rt.start_turn(h, text="ask", session_key=KEY)
    await collect(turn)
    assert (await turn.result).text == 'elicit: accept {"color": "teal"}'
    req = bridge.requests[0]
    assert req.mode == "form" and req.message == "Pick a color" and req.session_key == KEY


async def test_elicitation_without_bridge_declines(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path, "elicit_form"), session_key=KEY)
    turn = await rt.start_turn(h, text="ask", session_key=KEY)
    await collect(turn)
    assert (await turn.result).text.startswith("elicit: ")
    assert "accept" not in (await turn.result).text


async def test_cancel_returns_cancelled_and_session_survives(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path, "slow"), session_key=KEY)
    turn = await rt.start_turn(h, text="go", session_key=KEY)
    first = await asyncio.wait_for(turn.events.__anext__(), 10)
    assert first.text == "working"
    assert (await rt.status(h)).status == "busy"
    await turn.cancel()
    await turn.cancel()  # idempotent
    result = await asyncio.wait_for(turn.result, 10)
    assert result.stop_reason == "cancelled"
    rest = await collect(turn)
    assert rest[-1].kind == "done" and rest[-1].stop_reason == "cancelled"
    assert (await rt.status(h)).status == "idle"
    turn = await rt.start_turn(h, text="script:echo still here", session_key=KEY)
    await collect(turn)
    assert (await turn.result).text == "echo: still here"


async def test_runtime_cancel_without_turn_is_noop(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path), session_key=KEY)
    await rt.cancel(h, reason="nothing running")
    assert (await rt.status(h)).status == "idle"


async def test_cancel_escalates_to_kill_when_harness_ignores_it(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path, cancel_wait_s=0.2, kill_grace_s=0.1)
    h = await rt.ensure_session(_spec(tmp_path, "stubborn"), session_key=KEY)
    pid = h.pid
    turn = await rt.start_turn(h, text="go", session_key=KEY)
    await asyncio.wait_for(turn.events.__anext__(), 10)
    await rt.cancel(h, reason="stop")
    result = await asyncio.wait_for(turn.result, 10)
    assert result.stop_reason == "cancelled"
    assert await wait_until(lambda: not proc_mod.pid_alive(pid), timeout=3)
    status = await rt.status(h)
    assert status.status == "closed" and status.pid is None
    assert proc_mod.list_leases(rt.state_dir) == []


async def test_turn_timeout_raises_and_keeps_persistent_session(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path, "slow", turn_timeout_s=0.3), session_key=KEY)
    turn = await rt.start_turn(h, text="go", session_key=KEY)
    events = await collect(turn)
    assert events[-1].kind == "error" and events[-1].error.startswith("ACP_TURN_TIMEOUT")
    with pytest.raises(AcpTurnTimeoutError):
        await turn.result
    assert (await rt.status(h)).status == "idle"


async def test_turn_timeout_from_config_closes_oneshot(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path, fake_config(turn_timeout_s=1))
    h = await rt.ensure_session(_spec(tmp_path, "slow", mode="oneshot"), session_key=KEY)
    pid = h.pid
    turn = await rt.start_turn(h, text="go", session_key=KEY)
    with pytest.raises(AcpTurnTimeoutError):
        await asyncio.wait_for(turn.result, 10)
    assert (await rt.status(h)).status == "closed"
    assert await wait_until(lambda: not proc_mod.pid_alive(pid), timeout=3)


async def test_crash_mid_turn_is_an_error_with_exit_code(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path, "crash_mid_turn"), session_key=KEY)
    turn = await rt.start_turn(h, text="boom", session_key=KEY)
    events = await collect(turn)
    assert events[-1].kind == "error" and events[-1].error.startswith("ACP_PROCESS_EXITED")
    with pytest.raises(AcpTurnError) as ei:
        await turn.result
    assert ei.value.code == "ACP_PROCESS_EXITED"
    assert (await rt.status(h)).status == "closed"
    with pytest.raises(AcpTurnError) as ei2:
        await rt.start_turn(h, text="again", session_key=KEY)
    assert ei2.value.code == "ACP_SESSION_NOT_RUNNING"
    # ensure_session for the same key brings a fresh process up
    h2 = await rt.ensure_session(_spec(tmp_path), session_key=KEY)
    assert h2.pid and h2.pid != h.pid


async def test_oneshot_closes_after_first_turn(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path, mode="oneshot"), session_key=KEY)
    pid, lease = h.pid, h.lease_id
    turn = await rt.start_turn(h, text="once", session_key=KEY)
    await collect(turn)
    assert (await turn.result).text == "echo: once"
    assert (await rt.status(h)).status == "closed"
    assert not proc_mod.pid_alive(pid) or await wait_until(
        lambda: not proc_mod.pid_alive(pid), timeout=3
    )
    assert proc_mod.read_lease(rt.state_dir, lease or "") is None
    rec = rt.store.load(h.record_id)
    assert rec is not None and rec.closed


async def test_one_turn_at_a_time(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path, "slow"), session_key=KEY)
    first = await rt.start_turn(h, text="first", session_key=KEY)
    await asyncio.wait_for(first.events.__anext__(), 10)
    second_task = asyncio.create_task(rt.start_turn(h, text="script:echo second", session_key=KEY))
    await asyncio.sleep(0.05)
    assert not second_task.done()
    await first.cancel()
    assert (await first.result).stop_reason == "cancelled"
    second = await asyncio.wait_for(second_task, 10)
    await collect(second)
    assert (await second.result).text == "echo: second"


async def test_close_stream_drops_events(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path, "slow"), session_key=KEY)
    turn = await rt.start_turn(h, text="go", session_key=KEY)
    await asyncio.wait_for(turn.events.__anext__(), 10)
    await turn.close_stream()
    assert await collect(turn) == []
    await turn.cancel()
    assert (await turn.result).stop_reason == "cancelled"


async def test_resume_uses_session_load(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path, "load_session_ok"), session_key=KEY)
    sid, record_id = h.acp_session_id, h.record_id
    await rt.close(h)
    assert rt.store.load(record_id).closed

    rt2 = runtimes(tmp_path)
    h2 = await rt2.ensure_session(
        _spec(tmp_path, "load_session_ok", resume_session_id=sid), session_key="other-key"
    )
    assert h2.acp_session_id == sid and h2.record_id == record_id
    assert h2.status == "idle"
    turn = await rt2.start_turn(h2, text="hi again", session_key="other-key")
    events = await collect(turn)
    # the history replay during session/load is not part of the turn
    assert all("old answer" not in e.text for e in events)
    assert (await turn.result).text == f"resumed:{sid}: hi again"


async def test_same_session_key_resumes_without_explicit_id(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path, "load_session_ok"), session_key=KEY)
    sid = h.acp_session_id
    await rt.aclose()
    rt2 = runtimes(tmp_path)
    h2 = await rt2.ensure_session(_spec(tmp_path, "load_session_ok"), session_key=KEY)
    assert h2.acp_session_id == sid


async def test_resume_falls_back_to_new_session_when_load_fails(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path, "load_session_fail"), session_key=KEY)
    sid = h.acp_session_id
    await rt.close(h)
    h2 = await rt.ensure_session(
        _spec(tmp_path, "load_session_fail", resume_session_id=sid), session_key=KEY
    )
    assert h2.acp_session_id and h2.acp_session_id != sid
    turn = await rt.start_turn(h2, text="fresh", session_key=KEY)
    await collect(turn)
    assert (await turn.result).text == "echo: fresh"


async def test_resume_requires_matching_cwd(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    h = await rt.ensure_session(_spec(a, "load_session_ok"), session_key=KEY)
    sid = h.acp_session_id
    await rt.close(h)
    h2 = await rt.ensure_session(
        _spec(b, "load_session_ok", resume_session_id=sid), session_key="k2"
    )
    assert h2.acp_session_id != sid and h2.record_id != h.record_id


async def test_ensure_session_is_idempotent_per_key(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h1 = await rt.ensure_session(_spec(tmp_path), session_key=KEY)
    h2 = await rt.ensure_session(_spec(tmp_path), session_key=KEY)
    assert h1 is h2 and len(rt.live_handles()) == 1
    h3 = await rt.ensure_session(_spec(tmp_path), session_key="agent:main:acp:other")
    assert h3.record_id != h1.record_id and len(rt.live_handles()) == 2


async def test_max_concurrent(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path, fake_config(max_concurrent=1))
    await rt.ensure_session(_spec(tmp_path), session_key="k1")
    with pytest.raises(AcpSessionInitError) as ei:
        await rt.ensure_session(_spec(tmp_path), session_key="k2")
    assert ei.value.code == "ACP_CAPACITY"


async def test_init_timeout_cleans_up(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path, init_timeout_s=0.3)
    with pytest.raises(AcpSessionInitError) as ei:
        await rt.ensure_session(_spec(tmp_path, "hang_init"), session_key=KEY)
    assert "timed out" in str(ei.value)
    assert proc_mod.list_leases(rt.state_dir) == []
    assert rt.live_handles() == []


async def test_init_failure_reports_stderr_tail(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    with pytest.raises(AcpSessionInitError) as ei:
        await rt.ensure_session(_spec(tmp_path, "fail_init"), session_key=KEY)
    assert ei.value.code == "ACP_SESSION_INIT_FAILED"
    assert "fatal: cannot start" in str(ei.value)
    with pytest.raises(AcpSessionInitError):
        await rt.ensure_session(_spec(tmp_path, "fail_new_session"), session_key=KEY)
    assert proc_mod.list_leases(rt.state_dir) == []


async def test_launch_checks(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path, which=lambda _cmd: None)
    with pytest.raises(AcpSessionInitError) as ei:
        await rt.ensure_session(
            AcpSessionSpec(agent_id="gemini", cwd=str(tmp_path)), session_key=KEY
        )
    assert ei.value.code == "ACP_HARNESS_MISSING" and ei.value.details["missing"] == ["gemini"]

    with pytest.raises(AcpSessionInitError) as ei:
        await rt.ensure_session(AcpSessionSpec(agent_id="nope", cwd=str(tmp_path)), session_key=KEY)
    assert ei.value.code == "ACP_UNKNOWN_AGENT"

    with pytest.raises(AcpSessionInitError) as ei:
        await rt.ensure_session(_spec(tmp_path / "missing-dir"), session_key=KEY)
    assert ei.value.code == "ACP_CWD_MISSING"

    rt2 = runtimes(tmp_path, fake_config(allowed_agents=["claude"]))
    with pytest.raises(AcpAgentNotAllowedError):
        await rt2.ensure_session(_spec(tmp_path), session_key=KEY)

    def which(cmd: str) -> str | None:
        return "/usr/bin/claude" if cmd == "claude" else None

    rt3 = runtimes(tmp_path, fake_config(install_missing_adapters=False), which=which)
    with pytest.raises(AcpSessionInitError) as ei:
        await rt3.ensure_session(
            AcpSessionSpec(agent_id="claude", cwd=str(tmp_path)), session_key=KEY
        )
    assert ei.value.code == "ACP_ADAPTER_MISSING"


async def test_env_is_scrubbed_and_overrides_merged(tmp_path: Path, runtimes) -> None:
    base = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "AGENTOS_GATEWAY_TOKEN": "gw-secret",
        "SLACK_BOT_TOKEN": "xoxb",
        "OPENAI_API_KEY": "sk-keep",
    }
    cfg = fake_config(
        agents={
            "fake": AcpAgentOverride(
                command=sys.executable, args=[FAKE_AGENT], env={"FAKE_ACP_EXTRA": "from-config"}
            )
        }
    )
    rt = runtimes(tmp_path, cfg, base_env=base)
    h = await rt.ensure_session(_spec(tmp_path, "env"), session_key=KEY)
    turn = await rt.start_turn(h, text="env", session_key=KEY)
    await collect(turn)
    payload = json.loads((await turn.result).text)
    assert "AGENTOS_GATEWAY_TOKEN" not in payload["keys"]
    assert "SLACK_BOT_TOKEN" not in payload["keys"]
    assert "OPENAI_API_KEY" in payload["keys"]
    assert payload["lease"] == h.lease_id
    assert payload["extra"] == "from-config"
    assert 1 <= int(payload["tokio"]) <= 8
    assert payload["codex_home"] is None


async def test_codex_gets_isolated_home(tmp_path: Path, runtimes) -> None:
    user_home = tmp_path / "user-codex"
    user_home.mkdir()
    (user_home / "auth.json").write_text("{}")
    (user_home / "config.toml").write_text('model = "gpt-x"\napproval_policy = "never"\n')
    cfg = fake_config(agents={"codex": AcpAgentOverride(command=sys.executable, args=[FAKE_AGENT])})
    rt = runtimes(tmp_path, cfg, user_codex_home=user_home)
    work = tmp_path / "repo"
    work.mkdir()
    spec = AcpSessionSpec(agent_id="codex", cwd=str(work), env={"FAKE_ACP_SCRIPT": "env"})
    h = await rt.ensure_session(spec, session_key=KEY)
    turn = await rt.start_turn(h, text="env", session_key=KEY)
    await collect(turn)
    payload = json.loads((await turn.result).text)
    home = Path(payload["codex_home"])
    assert home == rt.state_dir / "codex-home"
    cfg_toml = tomllib.loads((home / "config.toml").read_text())
    assert cfg_toml["model"] == "gpt-x" and "approval_policy" not in cfg_toml
    assert cfg_toml["projects"][str(work)] == {"trust_level": "trusted"}
    assert (home / "auth.json").exists()


async def test_harness_mode_and_model_applied_at_start(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(
        _spec(tmp_path, harness_mode="plan", model="fake-large"), session_key=KEY
    )
    assert h.current_mode_id == "plan"
    model = next(o for o in h.config_options if o["id"] == "model")
    assert model["currentValue"] == "fake-large"
    turn = await rt.start_turn(h, text="x", session_key=KEY)
    await collect(turn)
    assert (await turn.result).usage.model == "fake-large"


async def test_unknown_harness_mode_is_ignored(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path, harness_mode="yolo"), session_key=KEY)
    assert h.current_mode_id == "default" and h.status == "idle"


async def test_set_mode_and_config_option(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path), session_key=KEY)
    await rt.set_mode(h, "acceptEdits")
    assert h.current_mode_id == "acceptEdits"
    assert rt.store.get(h.record_id).current_mode_id == "acceptEdits"
    with pytest.raises(AcpError) as ei:
        await rt.set_mode(h, "nope")
    assert ei.value.code == "ACP_MODE_UNAVAILABLE"
    await rt.set_config_option(h, "model", "fake-large")
    assert next(o for o in h.config_options if o["id"] == "model")["currentValue"] == "fake-large"
    with pytest.raises(AcpError) as ei:
        await rt.set_config_option(h, "temperature", "1")
    assert ei.value.code == "ACP_CONFIG_OPTION_UNAVAILABLE"
    await rt.close(h)
    with pytest.raises(AcpError) as ei:
        await rt.set_mode(h, "plan")
    assert ei.value.code == "ACP_SESSION_NOT_RUNNING"


async def test_close_keeps_or_discards_record(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(_spec(tmp_path), session_key="k1")
    pid = h.pid
    await rt.close(h)
    assert h.status == "closed" and h.pid is None
    assert await wait_until(lambda: not proc_mod.pid_alive(pid), timeout=3)
    assert rt.store.load(h.record_id).closed
    assert (await rt.status(h)).status == "closed"

    h2 = await rt.ensure_session(_spec(tmp_path), session_key="k2")
    await rt.close(h2, discard=True)
    assert rt.store.load(h2.record_id) is None
    assert proc_mod.list_leases(rt.state_dir) == []
    await rt.close(h2, discard=True)  # idempotent on a gone session


async def test_stderr_is_captured_redacted(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    h = await rt.ensure_session(
        _spec(tmp_path, env={"FAKE_ACP_STDERR_SECRET": "1"}), session_key=KEY
    )
    log = rt.state_dir / "logs" / f"{h.record_id}.stderr.log"
    assert await wait_until(lambda: log.exists() and "fake-acp starting" in log.read_text())
    assert "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789" not in log.read_text()


async def test_doctor(tmp_path: Path, runtimes) -> None:
    rt = runtimes(tmp_path)
    info = await rt.doctor("fake")
    assert info["agent_id"] == "fake" and info["source"] == "override"
    assert info["adapter_installed"] is True and "probe" not in info
    info = await rt.doctor("fake", probe=True)
    assert info["probe"] == "ok" and info["version"] == "9.9.9"
    assert proc_mod.list_leases(rt.state_dir) == []

    rt2 = runtimes(tmp_path, which=lambda _cmd: None)
    info = await rt2.doctor("claude")
    assert info["source"] == "package-exec" and info["missing_commands"] == ["claude"]
    assert info["adapter_installed"] is False
    info = await rt2.doctor("claude", probe=True)
    assert info["probe"] == "error" and "claude" in info["error"]
    info = await rt2.doctor("nope")
    assert "error" in info


async def test_reap_stale_kills_previous_gateway_harness(tmp_path: Path, runtimes) -> None:
    old = runtimes(tmp_path, gateway_instance_id="gw-old")
    h = await old.ensure_session(_spec(tmp_path), session_key=KEY)
    pid = h.pid
    new = runtimes(tmp_path, gateway_instance_id="gw-new")
    # the old runtime still owns its lease, so it skips it
    assert (await old.reap_stale())["terminated"] == []
    result = await new.reap_stale()
    assert pid in result["terminated"]
    assert await wait_until(lambda: old._live[h.record_id].proc.returncode is not None, 5)
    assert not proc_mod.pid_alive(pid) or await wait_until(lambda: not proc_mod.pid_alive(pid), 3)
    assert proc_mod.list_leases(tmp_path / "acp") == []


def test_usage_from_mapping_variants() -> None:
    u = usage_from_mapping({"inputTokens": 3, "outputTokens": 4})
    assert u is not None and u.total_tokens == 7
    u = usage_from_mapping({"token_count": {"input_tokens": 1, "output_tokens": 2}})
    assert u is not None and (u.input_tokens, u.output_tokens) == (1, 2)
    u = usage_from_mapping(
        {"input_tokens": 1, "cache_read_input_tokens": 9, "reasoning_output_tokens": 2}
    )
    assert u is not None and u.cached_read_tokens == 9 and u.thought_tokens == 2
    assert usage_from_mapping({"nothing": 1}) is None
    assert usage_from_mapping(None) is None
