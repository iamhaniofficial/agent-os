"""Every surface that can create a cron ``script`` job agrees on the rules.

A script job runs a file unattended with nothing in the loop to review it, so
the interesting assertions here are the refusals: who may create one, which
paths are accepted, and which session targets make sense.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import typer

from agentos.cli import cron_cmd
from agentos.gateway.rpc_cron import _build_payload, _handler_key_for_payload_kind
from agentos.scheduler.ops import SchedulerOps
from agentos.scheduler.payloads import SCRIPT_KIND, make_script_payload
from agentos.scheduler.persistence import JobStore
from agentos.scheduler.types import (
    CronJob,
    DeliveryConfig,
    ScheduleKind,
    SessionTarget,
)
from agentos.tools.builtin.control import cron as cron_tool
from agentos.tools.envelope import build_tool_failure_envelope
from agentos.tools.types import (
    CallerKind,
    InteractionMode,
    SafeToolError,
    ToolContext,
    ToolError,
    current_tool_context,
)


@pytest.fixture
def agentos_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTOS_STATE_DIR", str(tmp_path))
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    return tmp_path


@contextmanager
def _with_ctx(ctx: ToolContext):
    token = current_tool_context.set(ctx)
    try:
        yield
    finally:
        current_tool_context.reset(token)


def _ctx(caller_kind: CallerKind, session_key: str = "agent:main:cli:control") -> ToolContext:
    return ToolContext(
        caller_kind=caller_kind,
        interaction_mode=InteractionMode.INTERACTIVE,
        session_key=session_key,
        agent_id="main",
        channel_kind="feishu" if caller_kind is CallerKind.CHANNEL else "",
        channel_id="oc_chat_001" if caller_kind is CallerKind.CHANNEL else "",
        source_kind="channel" if caller_kind is CallerKind.CHANNEL else "cli",
    )


class _FakeScheduler:
    def __init__(self) -> None:
        self.add_calls: list[dict[str, Any]] = []

    async def list_jobs(self):
        return []

    async def add_job(self, **kwargs):
        self.add_calls.append(kwargs)
        return CronJob(
            id="job-1",
            name=kwargs["name"],
            handler_key=kwargs["handler_key"],
            payload=kwargs["payload"],
            session_target=kwargs["session_target"],
            session_key=kwargs["session_key"],
            delivery=kwargs.get("delivery") or DeliveryConfig(),
        )

    async def update_job(self, job_id, **patch):
        return None


@pytest.fixture
def fake_scheduler(monkeypatch):
    import agentos.tools.builtin.control as control_mod

    scheduler = _FakeScheduler()
    monkeypatch.setattr(control_mod, "_scheduler", scheduler)
    return scheduler


_SCHEDULE = {"kind": "every", "every_seconds": 300}


# ── the cron tool ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_creates_a_script_job_for_a_cli_caller(agentos_home, fake_scheduler):
    with _with_ctx(_ctx(CallerKind.CLI)):
        raw = await cron_tool(
            action="add",
            schedule=_SCHEDULE,
            job_kind="script",
            script="watch.sh",
            workdir=str(agentos_home),
        )

    result = json.loads(raw)
    assert result["payload_kind"] == SCRIPT_KIND
    assert result["script"] == "watch.sh"
    call = fake_scheduler.add_calls[-1]
    assert call["handler_key"] == "script_run"
    assert call["payload"]["script"] == "watch.sh"
    assert call["payload"]["workdir"] == str(agentos_home)
    assert call["name"] == "watch.sh"


@pytest.mark.asyncio
async def test_tool_refuses_a_script_job_from_a_channel(agentos_home, fake_scheduler):
    """A chat message must not be able to schedule unattended execution."""
    with _with_ctx(_ctx(CallerKind.CHANNEL, session_key="agent:main:feishu:oc_chat_001")):
        with pytest.raises(ToolError, match="interactive CLI or Web caller"):
            await cron_tool(
                action="add",
                schedule=_SCHEDULE,
                job_kind="script",
                script="watch.sh",
            )

    assert fake_scheduler.add_calls == []


@pytest.mark.asyncio
async def test_tool_refuses_a_pre_run_script_from_a_channel(agentos_home, fake_scheduler):
    """Same gate: a pre-run collector is unattended execution too."""
    with _with_ctx(_ctx(CallerKind.CHANNEL, session_key="agent:main:feishu:oc_chat_001")):
        with pytest.raises(ToolError, match="interactive CLI or Web caller"):
            await cron_tool(
                action="add",
                schedule=_SCHEDULE,
                job_kind="agent_turn",
                task="summarize",
                script="watch.py",
            )

    assert fake_scheduler.add_calls == []


@pytest.mark.asyncio
async def test_tool_attaches_a_pre_run_script_to_an_agent_turn(agentos_home, fake_scheduler):
    with _with_ctx(_ctx(CallerKind.CLI)):
        await cron_tool(
            action="add",
            schedule=_SCHEDULE,
            job_kind="agent_turn",
            task="summarize anything urgent",
            script="watch.py",
            script_args=["--repo", "owner/name"],
        )

    call = fake_scheduler.add_calls[-1]
    assert call["handler_key"] == "agent_run"
    assert call["payload"]["script"] == "watch.py"
    assert call["payload"]["args"] == ["--repo", "owner/name"]


@pytest.mark.asyncio
async def test_tool_refuses_a_script_on_a_reminder(agentos_home, fake_scheduler):
    with _with_ctx(_ctx(CallerKind.CLI)):
        with pytest.raises(ToolError, match="only used by job_kind"):
            await cron_tool(
                action="add",
                schedule=_SCHEDULE,
                job_kind="reminder",
                task="stand up",
                script="watch.py",
            )


@pytest.mark.asyncio
async def test_tool_requires_a_script_for_script_jobs(agentos_home, fake_scheduler):
    with _with_ctx(_ctx(CallerKind.CLI)):
        with pytest.raises(ToolError, match="requires 'script'"):
            await cron_tool(action="add", schedule=_SCHEDULE, job_kind="script")


@pytest.mark.asyncio
async def test_tool_rejects_an_absolute_script_path(agentos_home, fake_scheduler):
    with _with_ctx(_ctx(CallerKind.CLI)):
        with pytest.raises(ToolError, match="must be relative"):
            await cron_tool(
                action="add",
                schedule=_SCHEDULE,
                job_kind="script",
                script="/etc/passwd",
            )


@pytest.mark.asyncio
async def test_tool_rejects_script_jobs_on_the_main_session(agentos_home, fake_scheduler):
    with _with_ctx(_ctx(CallerKind.CLI)):
        with pytest.raises(ToolError, match="cannot use session_target=main"):
            await cron_tool(
                action="add",
                schedule=_SCHEDULE,
                job_kind="script",
                script="watch.sh",
                session_target="main",
            )


@pytest.mark.asyncio
async def test_tool_still_requires_task_for_other_kinds(agentos_home, fake_scheduler):
    with _with_ctx(_ctx(CallerKind.CLI)):
        with pytest.raises(ToolError, match="'task' required"):
            await cron_tool(action="add", schedule=_SCHEDULE, job_kind="reminder")


@pytest.mark.asyncio
async def test_tool_rejects_elevation_on_a_script_job(agentos_home, fake_scheduler):
    """A script job has no agent turn, so elevation has nothing to govern.

    ``SchedulerOps`` refuses this too, but only as a bare ``ValueError`` from
    several layers down; the tool has to name the offending field itself.
    """
    with _with_ctx(_ctx(CallerKind.CLI)):
        with pytest.raises(SafeToolError, match="cannot carry tool_policy.elevated"):
            await cron_tool(
                action="add",
                schedule=_SCHEDULE,
                job_kind="script",
                script="watch.sh",
                tool_policy={"profile": "minimal", "elevated": "bypass"},
            )
    assert fake_scheduler.add_calls == []


@pytest.mark.asyncio
async def test_tool_accepts_a_script_job_without_a_tool_policy(agentos_home, fake_scheduler):
    """The same call minus the elevation is the shape that was meant."""
    with _with_ctx(_ctx(CallerKind.CLI)):
        raw = await cron_tool(
            action="add",
            schedule=_SCHEDULE,
            job_kind="script",
            script="watch.sh",
        )

    assert json.loads(raw)["payload_kind"] == SCRIPT_KIND
    assert fake_scheduler.add_calls[-1]["handler_key"] == "script_run"


@pytest.mark.asyncio
async def test_script_job_refusals_reach_the_model_intact(agentos_home, fake_scheduler):
    """The refusal has to survive the envelope, not just the raise.

    A refusal sanitised to "failed with an internal error" is what sent one
    session into seven identical retries: the model was never told which field
    was wrong. Plain ``ToolError`` messages now reach the model too (#3642).
    """
    cases = [
        (
            {"job_kind": "script", "script": "watch.sh", "tool_policy": {"elevated": "bypass"}},
            "tool_policy.elevated",
        ),
        ({"job_kind": "script"}, "requires 'script'"),
        ({"job_kind": "script", "script": "watch.sh", "session_target": "main"}, "session_target"),
    ]
    for kwargs, expected in cases:
        with _with_ctx(_ctx(CallerKind.CLI)):
            try:
                await cron_tool(action="add", schedule=_SCHEDULE, **kwargs)
            except ToolError as exc:
                envelope = build_tool_failure_envelope(exc, "cron")
            else:  # pragma: no cover - the call must refuse
                pytest.fail(f"cron accepted {kwargs!r}")

        assert expected in envelope["user_message"]
        assert "internal error" not in envelope["user_message"]


@pytest.mark.asyncio
async def test_missing_schedule_names_the_missing_field(agentos_home, fake_scheduler):
    """``schedule`` is required for add, and the model must be told so."""
    with _with_ctx(_ctx(CallerKind.CLI)):
        try:
            await cron_tool(action="add", job_kind="script", script="watch.sh")
        except ToolError as exc:
            envelope = build_tool_failure_envelope(exc, "cron")
        else:  # pragma: no cover - the call must refuse
            pytest.fail("cron accepted an add without a schedule")

    assert "'schedule' required for add" in envelope["user_message"]


# ── the RPC surface ─────────────────────────────────────────────────────────


def test_rpc_builds_a_script_payload(agentos_home):
    kind, payload = _build_payload(
        {"payloadKind": SCRIPT_KIND, "script": "watch.sh", "workdir": "/tmp"},
        SessionTarget.ISOLATED,
    )

    assert kind == SCRIPT_KIND
    assert payload == {
        "kind": SCRIPT_KIND,
        "script": "watch.sh",
        "workdir": "/tmp",
        "agent_id": "main",
    }


def test_rpc_maps_script_kind_to_the_script_handler():
    assert _handler_key_for_payload_kind(SCRIPT_KIND) == "script_run"


def test_rpc_requires_a_script(agentos_home):
    with pytest.raises(ValueError, match="script is required"):
        _build_payload({"payloadKind": SCRIPT_KIND}, SessionTarget.ISOLATED)


def test_rpc_rejects_an_absolute_script(agentos_home):
    with pytest.raises(ValueError, match="must be relative"):
        _build_payload(
            {"payloadKind": SCRIPT_KIND, "script": "/etc/passwd"},
            SessionTarget.ISOLATED,
        )


def test_rpc_rejects_script_jobs_on_main(agentos_home):
    with pytest.raises(ValueError, match="cannot use sessionTarget='main'"):
        _build_payload(
            {"payloadKind": SCRIPT_KIND, "script": "watch.sh"},
            SessionTarget.MAIN,
        )


def test_rpc_requires_a_script_even_on_update(agentos_home):
    """The update merge carries the current script, so empty means "none left"."""
    with pytest.raises(ValueError, match="script is required"):
        _build_payload(
            {"payloadKind": SCRIPT_KIND, "script": ""},
            SessionTarget.ISOLATED,
            require_text=False,
        )


def test_rpc_refuses_a_script_on_a_kind_that_cannot_run_one(agentos_home):
    """Dropping it silently would make `update --script` on a reminder a no-op."""
    with pytest.raises(ValueError, match="only used by payloadKind='script'"):
        _build_payload(
            {"payloadKind": "reminder", "text": "ping", "script": "watch.sh"},
            SessionTarget.ISOLATED,
        )


def test_rpc_attaches_a_pre_run_script_to_an_agent_turn(agentos_home):
    kind, payload = _build_payload(
        {
            "payloadKind": "agent_turn",
            "text": "Summarize anything urgent",
            "script": "watch.py",
            "scriptArgs": ["--repo", "owner/name"],
        },
        SessionTarget.ISOLATED,
    )

    assert kind == "agent_turn"
    assert payload["script"] == "watch.py"
    assert payload["args"] == ["--repo", "owner/name"]


def test_rpc_validates_a_pre_run_script_path(agentos_home):
    with pytest.raises(ValueError, match="must be relative"):
        _build_payload(
            {"payloadKind": "agent_turn", "text": "go", "script": "/etc/passwd"},
            SessionTarget.ISOLATED,
        )


def test_rpc_splits_script_args_given_as_one_string(agentos_home):
    """What a single text input in the Web UI sends."""
    _, payload = _build_payload(
        {
            "payloadKind": SCRIPT_KIND,
            "script": "watch.py",
            "scriptArgs": '--name "my feed" --limit 5',
        },
        SessionTarget.ISOLATED,
    )

    assert payload["args"] == ["--name", "my feed", "--limit", "5"]


def test_rpc_rejects_unparseable_script_args(agentos_home):
    with pytest.raises(ValueError, match="could not be parsed"):
        _build_payload(
            {"payloadKind": SCRIPT_KIND, "script": "watch.py", "scriptArgs": "--name 'unclosed"},
            SessionTarget.ISOLATED,
        )


# ── the RPC update path ─────────────────────────────────────────────────────


class _UpdateScheduler:
    def __init__(self, job: CronJob) -> None:
        self.job = job
        self.updated: dict[str, Any] | None = None

    async def get_job(self, job_id: str) -> CronJob | None:
        return self.job if self.job.id == job_id else None

    async def update_job(self, job_id: str, **patch: Any) -> CronJob:
        self.updated = patch
        for key, value in patch.items():
            setattr(self.job, key, value)
        return self.job


def _stored_script_job() -> CronJob:
    return CronJob(
        id="watchdog",
        name="Watchdog",
        handler_key="script_run",
        payload=make_script_payload("watch.sh", "main", "/srv"),
        session_target=SessionTarget.ISOLATED,
    )


@pytest.mark.asyncio
async def test_rpc_update_keeps_the_script_across_an_unrelated_patch(agentos_home):
    from agentos.gateway.rpc import RpcContext
    from agentos.gateway.rpc_cron import _handle_cron_update

    scheduler = _UpdateScheduler(_stored_script_job())

    await _handle_cron_update(
        {"id": "watchdog", "agentId": "main"},
        RpcContext(conn_id="test", cron_scheduler=scheduler),
    )

    assert scheduler.updated is not None
    assert scheduler.updated["handler_key"] == "script_run"
    assert scheduler.updated["payload"]["script"] == "watch.sh"
    assert scheduler.updated["payload"]["workdir"] == "/srv"


@pytest.mark.asyncio
async def test_rpc_update_converts_a_script_job_to_a_reminder(agentos_home):
    """The old script path must not leak in as the reminder's text."""
    from agentos.gateway.rpc import RpcContext
    from agentos.gateway.rpc_cron import _handle_cron_update

    scheduler = _UpdateScheduler(_stored_script_job())

    await _handle_cron_update(
        {"id": "watchdog", "payloadKind": "reminder", "text": "stand up"},
        RpcContext(conn_id="test", cron_scheduler=scheduler),
    )

    assert scheduler.updated is not None
    assert scheduler.updated["handler_key"] == "static_message"
    assert scheduler.updated["payload"] == {
        "kind": "reminder",
        "text": "stand up",
        "agent_id": "main",
    }


@pytest.mark.asyncio
async def test_rpc_update_converts_a_reminder_to_a_script_job(agentos_home):
    from agentos.gateway.rpc import RpcContext
    from agentos.gateway.rpc_cron import _handle_cron_update

    scheduler = _UpdateScheduler(
        CronJob(
            id="watchdog",
            name="Watchdog",
            handler_key="static_message",
            payload={"kind": "reminder", "text": "stand up", "agent_id": "main"},
            session_target=SessionTarget.ISOLATED,
        )
    )

    await _handle_cron_update(
        {"id": "watchdog", "payloadKind": SCRIPT_KIND, "script": "watch.sh"},
        RpcContext(conn_id="test", cron_scheduler=scheduler),
    )

    assert scheduler.updated is not None
    assert scheduler.updated["handler_key"] == "script_run"
    assert scheduler.updated["payload"]["script"] == "watch.sh"


@pytest.mark.asyncio
async def test_rpc_update_drops_a_pre_run_script_when_cleared(agentos_home):
    """An explicit empty script is how the UI and CLI say "no collector"."""
    from agentos.gateway.rpc import RpcContext
    from agentos.gateway.rpc_cron import _handle_cron_update
    from agentos.scheduler.payloads import make_agent_turn_payload

    scheduler = _UpdateScheduler(
        CronJob(
            id="triage",
            name="Triage",
            handler_key="agent_run",
            payload=make_agent_turn_payload("summarize", "main", "watch.py", "/srv"),
            session_target=SessionTarget.ISOLATED,
        )
    )

    await _handle_cron_update(
        {"id": "triage", "script": ""},
        RpcContext(conn_id="test", cron_scheduler=scheduler),
    )

    assert scheduler.updated is not None
    assert "script" not in scheduler.updated["payload"]


@pytest.mark.asyncio
async def test_rpc_update_keeps_a_pre_run_script_across_an_unrelated_patch(agentos_home):
    from agentos.gateway.rpc import RpcContext
    from agentos.gateway.rpc_cron import _handle_cron_update
    from agentos.scheduler.payloads import make_agent_turn_payload

    scheduler = _UpdateScheduler(
        CronJob(
            id="triage",
            name="Triage",
            handler_key="agent_run",
            payload=make_agent_turn_payload("summarize", "main", "watch.py", "/srv", ["--x"]),
            session_target=SessionTarget.ISOLATED,
        )
    )

    await _handle_cron_update(
        {"id": "triage", "text": "summarize harder"},
        RpcContext(conn_id="test", cron_scheduler=scheduler),
    )

    assert scheduler.updated is not None
    assert scheduler.updated["payload"]["script"] == "watch.py"
    assert scheduler.updated["payload"]["args"] == ["--x"]
    assert scheduler.updated["payload"]["task"] == "summarize harder"


def test_cli_update_can_clear_the_script(agentos_home, stub_gateway):
    _cli_update(job_id="job-1", script="")

    _, params = stub_gateway[-1]
    assert params["script"] == ""


@pytest.mark.asyncio
async def test_rpc_update_refuses_a_script_on_a_reminder(agentos_home):
    from agentos.gateway.rpc import RpcContext
    from agentos.gateway.rpc_cron import _handle_cron_update

    scheduler = _UpdateScheduler(
        CronJob(
            id="watchdog",
            name="Watchdog",
            handler_key="static_message",
            payload={"kind": "reminder", "text": "stand up", "agent_id": "main"},
            session_target=SessionTarget.ISOLATED,
        )
    )

    with pytest.raises(ValueError, match="only used by payloadKind='script'"):
        await _handle_cron_update(
            {"id": "watchdog", "script": "watch.sh"},
            RpcContext(conn_id="test", cron_scheduler=scheduler),
        )

    assert scheduler.updated is None


# ── the CLI ─────────────────────────────────────────────────────────────────


@pytest.fixture
def stub_gateway(monkeypatch):
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Client:
        async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            calls.append((method, params))
            return {"id": "job-1"}

    def _runner(fn, **kwargs):
        import asyncio

        return asyncio.run(fn(_Client()))

    monkeypatch.setattr(cron_cmd, "run_gateway_sync", _runner)
    monkeypatch.setattr(cron_cmd, "confirm_or_exit", lambda *a, **kw: None)
    monkeypatch.setattr(cron_cmd, "_emit_success", lambda *a, **kw: None)
    return calls


def _cli_add(**overrides: Any) -> None:
    """Call cron_add the way typer would — every option filled in."""
    params: dict[str, Any] = {
        "expression": None,
        "cron": None,
        "every": None,
        "at": None,
        "text": None,
        "script": None,
        "script_arg": None,
        "workdir": None,
        "name": None,
        "agent": None,
        "job_kind": "auto",
        "session_target": "isolated",
        "timeout": None,
        "tz": None,
        "wake": None,
        "exact": False,
        "jitter": None,
        "announce": False,
        "no_deliver": False,
        "channel": None,
        "to": None,
        "account": None,
        "best_effort_deliver": False,
        "webhook_url": None,
        "webhook_token": None,
        "webhook_token_env": None,
        "webhook_token_file": None,
        "failure_mode": None,
        "failure_channel": None,
        "failure_to": None,
        "failure_account": None,
        "failure_webhook_url": None,
        "failure_webhook_token": None,
        "failure_webhook_token_env": None,
        "failure_webhook_token_file": None,
        "elevated": None,
        "elevated_mode": None,
        "tool_policy": None,
        "json_output": False,
    }
    cron_cmd.cron_add(**{**params, **overrides})


def _cli_update(**overrides: Any) -> None:
    params: dict[str, Any] = {
        "job_id": "job-1",
        "expression": None,
        "cron": None,
        "every": None,
        "at": None,
        "text": None,
        "job_kind": None,
        "script": None,
        "script_arg": None,
        "workdir": None,
        "name": None,
        "enabled": None,
        "timeout": None,
        "tz": None,
        "wake": None,
        "failure_mode": None,
        "failure_channel": None,
        "failure_to": None,
        "failure_account": None,
        "failure_webhook_url": None,
        "failure_webhook_token": None,
        "failure_webhook_token_env": None,
        "failure_webhook_token_file": None,
        "elevated": None,
        "elevated_mode": None,
        "tool_policy": None,
        "json_output": False,
    }
    cron_cmd.cron_update(**{**params, **overrides})


def test_cli_script_flag_implies_the_script_kind(agentos_home, stub_gateway):
    _cli_add(every="5m", script="watch.sh", name="Memory watchdog")

    method, params = stub_gateway[-1]
    assert method == "cron.add"
    assert params["payloadKind"] == SCRIPT_KIND
    assert params["script"] == "watch.sh"


def test_cli_script_kind_requires_a_script(agentos_home, stub_gateway):
    with pytest.raises(typer.BadParameter, match="requires --script"):
        _cli_add(every="5m", job_kind="script")


def test_cli_rejects_a_script_on_a_kind_that_cannot_run_one(agentos_home, stub_gateway):
    with pytest.raises(typer.BadParameter, match="only used by script and agent_turn"):
        _cli_add(every="5m", job_kind="reminder", text="ping", script="watch.sh")


def test_cli_rejects_an_absolute_script_path(agentos_home, stub_gateway):
    with pytest.raises(typer.BadParameter, match="must be relative"):
        _cli_add(every="5m", job_kind="script", script="/etc/passwd")


def test_cli_rejects_script_jobs_on_main(agentos_home, stub_gateway):
    with pytest.raises(typer.BadParameter, match="cannot use --session-target main"):
        _cli_add(every="5m", job_kind="script", script="watch.sh", session_target="main")


def test_cli_still_requires_text_for_reminders(agentos_home, stub_gateway):
    with pytest.raises(typer.BadParameter, match="--text is required"):
        _cli_add(every="5m", job_kind="reminder")


def test_cli_update_patches_the_script(agentos_home, stub_gateway):
    _cli_update(job_id="job-1", script="other.sh")

    method, params = stub_gateway[-1]
    assert method == "cron.update"
    assert params["script"] == "other.sh"


def test_cli_update_converts_a_job_to_a_script_job(agentos_home, stub_gateway):
    """hermes' `cron edit --no-agent --script x`, in AgentOS terms."""
    _cli_update(job_id="job-1", job_kind="script", script="watch.sh")

    _, params = stub_gateway[-1]
    assert params["payloadKind"] == "script"
    assert params["script"] == "watch.sh"


def test_cli_update_converts_a_script_job_back(agentos_home, stub_gateway):
    """hermes' `cron edit --agent`."""
    _cli_update(job_id="job-1", job_kind="agent_turn", text="do the thing")

    _, params = stub_gateway[-1]
    assert params["payloadKind"] == "agent_turn"
    assert params["text"] == "do the thing"


def test_cli_update_rejects_auto_as_a_target_kind(agentos_home, stub_gateway):
    with pytest.raises(typer.BadParameter, match="must name a kind"):
        _cli_update(job_id="job-1", job_kind="auto")


def test_cli_attaches_a_pre_run_script_to_an_agent_turn(agentos_home, stub_gateway):
    _cli_add(
        every="10m",
        job_kind="agent_turn",
        text="Summarize anything urgent",
        script="watch.py",
        script_arg=["--repo", "owner/name"],
    )

    _, params = stub_gateway[-1]
    assert params["payloadKind"] == "agent_turn"
    assert params["script"] == "watch.py"
    assert params["scriptArgs"] == ["--repo", "owner/name"]


def test_cli_rejects_script_args_without_a_script(agentos_home, stub_gateway):
    with pytest.raises(typer.BadParameter, match="require --script"):
        _cli_add(every="10m", job_kind="agent_turn", text="go", script_arg=["--x"])


# ── ops floor ───────────────────────────────────────────────────────────────


async def test_ops_rejects_elevation_on_a_script_job(tmp_path: Path) -> None:
    """Elevation only means something for a job that runs an agent turn."""
    store = JobStore(str(tmp_path / "cron.db"))
    await store.open()
    try:
        ops = SchedulerOps(store)
        with pytest.raises(ValueError, match="agent_turn"):
            await ops.add(
                name="watchdog",
                handler_key="script_run",
                payload=make_script_payload("watch.sh"),
                session_target=SessionTarget.ISOLATED,
                schedule_kind=ScheduleKind.EVERY,
                schedule_value="300",
                tool_policy={"elevated": "bypass"},
            )
    finally:
        await store.close()


def test_every_contract_handler_key_is_registered_at_boot() -> None:
    """A handler key the payload contract can emit but boot never registers is
    a job that persists fine and then has nowhere to run."""
    import re

    from agentos.gateway import boot
    from agentos.scheduler.payloads import _KNOWN_HANDLER_KEYS

    source = Path(boot.__file__).read_text(encoding="utf-8")
    registered = set(re.findall(r'register_handler\(\s*"([a-z_]+)"', source))

    assert _KNOWN_HANDLER_KEYS <= registered


async def test_ops_update_can_remove_a_pre_run_script(tmp_path: Path) -> None:
    """A normalized payload replaces, it does not merge.

    Merging made an optional key unremovable: clearing a job's pre-run script
    sends a payload with no ``script``, and the old value came straight back.
    """
    from agentos.scheduler.payloads import make_agent_turn_payload

    store = JobStore(str(tmp_path / "cron.db"))
    await store.open()
    try:
        ops = SchedulerOps(store)
        job = await ops.add(
            name="triage",
            handler_key="agent_run",
            payload=make_agent_turn_payload("summarize", "main", "watch.py", "/srv"),
            session_target=SessionTarget.ISOLATED,
            schedule_kind=ScheduleKind.EVERY,
            schedule_value="300",
        )
        assert job.payload["script"] == "watch.py"

        updated = await ops.update(
            job.id,
            payload=make_agent_turn_payload("summarize", "main"),
        )
    finally:
        await store.close()

    assert updated is not None
    assert "script" not in updated.payload
    assert "workdir" not in updated.payload


async def test_ops_update_still_merges_a_partial_payload(tmp_path: Path) -> None:
    """Legacy callers that patch one field keep the rest of the payload."""
    from agentos.scheduler.payloads import make_agent_turn_payload

    store = JobStore(str(tmp_path / "cron.db"))
    await store.open()
    try:
        ops = SchedulerOps(store)
        job = await ops.add(
            name="triage",
            handler_key="agent_run",
            payload=make_agent_turn_payload("summarize", "main", "watch.py"),
            session_target=SessionTarget.ISOLATED,
            schedule_kind=ScheduleKind.EVERY,
            schedule_value="300",
        )

        updated = await ops.update(job.id, payload={"task": "summarize harder"})
    finally:
        await store.close()

    assert updated is not None
    assert updated.payload["task"] == "summarize harder"
    assert updated.payload["script"] == "watch.py"


async def test_ops_round_trips_a_script_job(tmp_path: Path) -> None:
    store = JobStore(str(tmp_path / "cron.db"))
    await store.open()
    try:
        ops = SchedulerOps(store)
        job = await ops.add(
            name="watchdog",
            handler_key="script_run",
            payload=make_script_payload("watch.sh", "main", "/tmp"),
            session_target=SessionTarget.ISOLATED,
            schedule_kind=ScheduleKind.EVERY,
            schedule_value="300",
        )
        reloaded = await store.get(job.id)
    finally:
        await store.close()

    assert reloaded is not None
    assert reloaded.handler_key == "script_run"
    assert reloaded.payload["script"] == "watch.sh"
    assert reloaded.payload["workdir"] == "/tmp"
