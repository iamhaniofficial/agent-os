"""``agentos acp`` CLI: registry listing, doctor, local run, sessions, cancel/close.

Runs fully offline: the gateway client and the ACP runtime are in-memory
fakes, ``shutil.which`` is pinned, and the store lives under ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentos.acp.types import (
    AcpEvent,
    AcpSessionHandle,
    AcpSessionSpec,
    AcpTurn,
    AcpTurnResult,
    AcpUsage,
    PermissionOptionInfo,
    PermissionRequest,
)
from agentos.cli import acp_cmd
from agentos.cli.main import app

runner = CliRunner()

ON_PATH = {"npx", "claude", "codex", "uvx", "my-agent"}


def _which(cmd: str) -> str | None:
    return f"/usr/bin/{cmd}" if cmd in ON_PATH else None


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("shutil.which", _which)
    monkeypatch.setenv("AGENTOS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("AGENTOS_GATEWAY_CONFIG_PATH", raising=False)
    monkeypatch.delenv("AGENTOS_GATEWAY_URL", raising=False)
    monkeypatch.setattr(
        "agentos.cli.gateway_client.GatewayClient", OfflineGatewayClient, raising=True
    )


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    state = tmp_path / "acp-state"
    path = tmp_path / "agentos.toml"
    path.write_text(
        "[acp]\n"
        f'state_dir = "{state}"\n'
        'allowed_agents = ["claude", "codex", "mine"]\n'
        "[acp.agents.mine]\n"
        'command = "my-agent"\n'
        'args = ["--acp"]\n',
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------
# Gateway fakes
# --------------------------------------------------------------------------


class OfflineGatewayClient:
    calls: list[tuple[str, Any]] = []

    async def connect(self, url: str, *, token: Any = None) -> None:
        raise SystemExit("gateway offline")

    async def close(self) -> None:
        return None

    async def call(self, method: str, params: dict | None = None) -> Any:  # pragma: no cover
        raise AssertionError("offline client must not be called")


class OnlineGatewayClient:
    calls: list[tuple[str, Any]] = []
    payloads: dict[str, Any] = {}
    server_version = None

    async def connect(self, url: str, *, token: Any = None) -> None:
        return None

    async def close(self) -> None:
        return None

    async def call(self, method: str, params: dict | None = None) -> Any:
        type(self).calls.append((method, params or {}))
        payload = type(self).payloads.get(method)
        if isinstance(payload, Exception):
            raise payload
        return payload if payload is not None else {}


@pytest.fixture
def online(monkeypatch: pytest.MonkeyPatch) -> type[OnlineGatewayClient]:
    OnlineGatewayClient.calls = []
    OnlineGatewayClient.payloads = {}
    monkeypatch.setattr("agentos.cli.gateway_client.GatewayClient", OnlineGatewayClient)
    monkeypatch.setattr("agentos.cli.gateway_rpc._apply_version_skew_policy", lambda *a, **k: None)
    monkeypatch.setattr("agentos.cli.gateway_rpc._maybe_emit_update_notice", lambda **k: None)
    return OnlineGatewayClient


def _rpc_error(code: str) -> Exception:
    from agentos.cli.gateway_client import GatewayRPCError

    return GatewayRPCError("acp", code=code, message=f"{code} from gateway")


# --------------------------------------------------------------------------
# Runtime fake (AcpRuntime protocol, in memory)
# --------------------------------------------------------------------------


class FakeRuntime:
    def __init__(
        self,
        *,
        permission_bridge: Any,
        elicitation_bridge: Any = None,
        config: Any = None,
        state_dir: Path | None = None,
    ) -> None:
        self.bridge = permission_bridge
        self.config = config
        self.state_dir = state_dir
        self.specs: list[AcpSessionSpec] = []
        self.closed: list[tuple[str, bool]] = []
        self.prompts: list[str] = []
        self.permission_kinds = ["read", "edit"]

    async def ensure_session(self, spec: AcpSessionSpec, *, session_key: str) -> AcpSessionHandle:
        self.specs.append(spec)
        assert session_key.startswith("agent:cli:acp:")
        return AcpSessionHandle(
            record_id="rec1",
            agent_id=spec.agent_id,
            cwd=spec.cwd,
            argv=["fake"],
            argv_hash="h",
            acp_session_id="sess-123",
            mode=spec.mode,
        )

    async def start_turn(
        self,
        handle: AcpSessionHandle,
        *,
        text: str,
        attachments: list[dict[str, Any]] | None = None,
        session_key: str,
    ) -> AcpTurn:
        self.prompts.append(text)
        decisions = []
        for kind in self.permission_kinds:
            decisions.append(
                await self.bridge.decide(
                    PermissionRequest(
                        session_key=session_key,
                        acp_session_id="sess-123",
                        agent_id=handle.agent_id,
                        tool_call_id="t1",
                        title=f"{kind} something",
                        tool_kind=kind,
                        raw_input=None,
                        options=[PermissionOptionInfo("a", "Allow", "allow_once")],
                        cwd=handle.cwd,
                    )
                )
            )
        self.decisions = decisions

        async def events() -> AsyncIterator[AcpEvent]:
            yield AcpEvent(kind="text_delta", text="Hello ")
            yield AcpEvent(
                kind="tool_call", tool_call_id="t1", title="Read README", tool_kind="read"
            )
            yield AcpEvent(kind="text_delta", text="world")
            yield AcpEvent(kind="done", stop_reason="end_turn")

        async def result() -> AcpTurnResult:
            return AcpTurnResult(
                stop_reason="end_turn",
                text="Hello world",
                usage=AcpUsage(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.0123),
                duration_ms=1500,
                tool_calls=1,
            )

        async def noop() -> None:
            return None

        return AcpTurn(events=events(), result=result(), cancel=noop, close_stream=noop)

    async def cancel(self, handle: AcpSessionHandle, *, reason: str = "") -> None:
        return None

    async def close(self, handle: AcpSessionHandle, *, discard: bool = False) -> None:
        self.closed.append((handle.record_id, discard))

    async def status(self, handle: AcpSessionHandle) -> AcpSessionHandle:
        return handle

    async def set_mode(self, handle: AcpSessionHandle, mode_id: str) -> None:
        return None

    async def set_config_option(self, handle: AcpSessionHandle, key: str, value: Any) -> None:
        return None

    async def doctor(self, agent_id: str, *, probe: bool = False) -> dict[str, Any]:
        return {"agent_id": agent_id, "probe": "ok", "version": "9.9.9"}


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> list[FakeRuntime]:
    made: list[FakeRuntime] = []

    def factory(**kwargs: Any) -> FakeRuntime:
        rt = FakeRuntime(**kwargs)
        made.append(rt)
        return rt

    monkeypatch.setattr(acp_cmd, "_runtime_factory", lambda: factory)
    return made


# --------------------------------------------------------------------------
# list
# --------------------------------------------------------------------------


def test_list_local_json_has_every_builtin_and_override(config_file: Path) -> None:
    result = runner.invoke(app, ["acp", "list", "--config", str(config_file), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    rows = {r["agent_id"]: r for r in payload["agents"]}
    assert len(rows) == 27  # 26 built-ins + the configured "mine"
    assert rows["mine"]["source"] == "override"
    assert rows["mine"]["argv"] == ["my-agent", "--acp"]
    assert rows["mine"]["override"] is True
    assert rows["claude"]["default"] is True
    assert rows["claude"]["source"] == "package-exec"
    assert rows["gemini"]["missing_commands"] == ["gemini"]
    assert rows["gemini"]["allowed"] is False


def test_list_table_renders_without_gateway(config_file: Path) -> None:
    result = runner.invoke(app, ["acp", "list", "--config", str(config_file)])

    assert result.exit_code == 0, result.output
    assert "ACP harnesses" in result.output
    assert "claude *" in result.output
    assert "Gemini CLI" in result.output


def test_list_prefers_running_gateway(config_file: Path, online) -> None:
    online.payloads["acp.agents"] = {
        "enabled": True,
        "default_agent": "codex",
        "permission_mode": "ask",
        "agents": [{"agent_id": "codex", "name": "Codex", "source": "installed"}],
    }

    result = runner.invoke(app, ["acp", "list", "--config", str(config_file), "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["default_agent"] == "codex"
    assert online.calls == [("acp.agents", {})]


def test_list_falls_back_when_gateway_predates_acp(config_file: Path, online) -> None:
    online.payloads["acp.agents"] = _rpc_error("METHOD_NOT_FOUND")

    result = runner.invoke(app, ["acp", "list", "--config", str(config_file), "--json"])

    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)["agents"]) == 27


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def test_doctor_ok_for_installed_harness(config_file: Path) -> None:
    result = runner.invoke(app, ["acp", "doctor", "claude", "--config", str(config_file)])

    assert result.exit_code == 0, result.output
    assert "claude-agent-acp" in result.output


def test_doctor_reports_missing_cli_and_exits_1(config_file: Path) -> None:
    result = runner.invoke(
        app, ["acp", "doctor", "gemini-cli", "--config", str(config_file), "--json"]
    )

    assert result.exit_code == 1
    row = json.loads(result.stdout)["agents"][0]
    assert row["agent_id"] == "gemini"
    assert row["ok"] is False
    assert any("gemini" in p for p in row["problems"])


def test_doctor_unknown_agent_exits_2(config_file: Path) -> None:
    result = runner.invoke(app, ["acp", "doctor", "nope", "--config", str(config_file)])

    assert result.exit_code == 2
    assert "unknown ACP agent" in result.output


def test_doctor_probe_uses_runtime(config_file: Path, fake_runtime) -> None:
    result = runner.invoke(
        app, ["acp", "doctor", "codex", "--probe", "--config", str(config_file), "--json"]
    )

    assert result.exit_code == 0, result.output
    row = json.loads(result.stdout)["agents"][0]
    assert row["probe"] == {"agent_id": "codex", "probe": "ok", "version": "9.9.9"}


def test_doctor_probe_without_runtime_is_a_clear_error(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acp_cmd, "_runtime_factory", lambda: None)

    result = runner.invoke(app, ["acp", "doctor", "codex", "--probe", "--config", str(config_file)])

    assert result.exit_code == 1
    assert "ACP runtime unavailable in this build" in result.output


def test_doctor_uses_gateway_payload(config_file: Path, online) -> None:
    online.payloads["acp.doctor"] = {
        "agents": [{"agent_id": "codex", "ok": True, "argv": ["codex-acp"], "source": "installed"}]
    }

    result = runner.invoke(app, ["acp", "doctor", "codex", "--config", str(config_file)])

    assert result.exit_code == 0, result.output
    assert online.calls == [("acp.doctor", {"probe": False, "agent": "codex"})]
    assert "codex-acp" in result.output


def test_probe_timeout_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    class Slow:
        async def doctor(self, agent_id: str, *, probe: bool = False) -> dict[str, Any]:
            await asyncio.sleep(1)
            return {}

    monkeypatch.setattr(acp_cmd, "PROBE_TIMEOUT_S", 0.01)

    out = asyncio.run(acp_cmd._probe(Slow(), "claude"))

    assert out["probe"] == "error"
    assert "timed out" in out["error"]


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def test_run_streams_text_and_prints_stats(config_file: Path, fake_runtime, tmp_path) -> None:
    result = runner.invoke(
        app,
        ["acp", "run", "claude", "say hi", "--cwd", str(tmp_path), "--config", str(config_file)],
    )

    assert result.exit_code == 0, result.output
    assert "Hello world" in result.stdout
    assert "Stats: claude end_turn" in result.output
    assert "tokens 15 (in 10 / out 5)" in result.output
    assert "cost $0.0123" in result.output
    assert "--resume sess-123" in result.output
    rt = fake_runtime[0]
    assert rt.prompts == ["say hi"]
    spec = rt.specs[0]
    assert spec.agent_id == "claude"
    assert spec.cwd == str(tmp_path.resolve())
    assert spec.mode == "persistent"
    # approve-reads: reads allowed, edits denied, never allow_always
    assert rt.decisions == ["allow_once", "reject_once"]
    assert rt.closed == [("rec1", False)]
    assert rt.state_dir == tmp_path / "acp-state"


def test_run_oneshot_json_discards_and_reports(config_file: Path, fake_runtime, tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "acp",
            "run",
            "claude-code",
            "go",
            "--cwd",
            str(tmp_path),
            "--oneshot",
            "--mode",
            "plan",
            "--permissions",
            "approve-all",
            "--json",
            "--config",
            str(config_file),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["agent_id"] == "claude"
    assert payload["acp_session_id"] == "sess-123"
    assert payload["result"]["stop_reason"] == "end_turn"
    assert payload["result"]["usage"]["cost_usd"] == 0.0123
    rt = fake_runtime[0]
    assert rt.specs[0].mode == "oneshot"
    assert rt.specs[0].harness_mode == "plan"
    assert rt.decisions == ["allow_once", "allow_once"]
    assert rt.closed == [("rec1", True)]


def test_run_deny_all(config_file: Path, fake_runtime, tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "acp",
            "run",
            "codex",
            "go",
            "--cwd",
            str(tmp_path),
            "--permissions",
            "deny-all",
            "--config",
            str(config_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_runtime[0].decisions == ["reject_once", "reject_once"]


def test_run_rejects_unknown_permission_mode(config_file: Path, fake_runtime) -> None:
    result = runner.invoke(
        app, ["acp", "run", "claude", "go", "--permissions", "ask", "--config", str(config_file)]
    )

    assert result.exit_code == 2
    assert not fake_runtime


def test_run_missing_harness_cli_fails_before_spawn(config_file: Path, fake_runtime) -> None:
    result = runner.invoke(app, ["acp", "run", "gemini", "go", "--config", str(config_file)])

    assert result.exit_code == 1
    assert "gemini" in result.output
    assert not fake_runtime


def test_run_without_runtime_is_a_clear_error(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acp_cmd, "_runtime_factory", lambda: None)

    result = runner.invoke(app, ["acp", "run", "claude", "go", "--config", str(config_file)])

    assert result.exit_code == 1
    assert "ACP runtime unavailable in this build" in result.output


def test_run_refuses_when_acp_disabled(tmp_path: Path, fake_runtime) -> None:
    cfg = tmp_path / "off.toml"
    cfg.write_text("[acp]\nenabled = false\n", encoding="utf-8")

    result = runner.invoke(app, ["acp", "run", "claude", "go", "--config", str(cfg)])

    assert result.exit_code == 1
    assert "disabled" in result.output


def test_run_maps_acp_errors(config_file: Path, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from agentos.acp.types import AcpSessionInitError

    class Broken(FakeRuntime):
        async def ensure_session(self, spec, *, session_key):
            raise AcpSessionInitError("adapter crashed", code="ACP_HARNESS_MISSING")

    monkeypatch.setattr(acp_cmd, "_runtime_factory", lambda: Broken)

    result = runner.invoke(
        app,
        [
            "acp",
            "run",
            "claude",
            "go",
            "--cwd",
            str(tmp_path),
            "--config",
            str(config_file),
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "ACP_HARNESS_MISSING"


def test_stats_line_without_cost_or_tokens() -> None:
    line = acp_cmd.stats_line("codex", AcpTurnResult(stop_reason="end_turn", duration_ms=312_000))

    assert line == "Stats: codex end_turn • runtime 5m12s"


# --------------------------------------------------------------------------
# sessions / cancel / close
# --------------------------------------------------------------------------


def test_sessions_reads_store_when_gateway_is_down(config_file: Path, tmp_path: Path) -> None:
    folder = tmp_path / "acp-state" / "sessions"
    folder.mkdir(parents=True)
    (folder / "r-old.json").write_text(
        json.dumps({"record_id": "r-old", "agent_id": "codex", "cwd": "/a", "last_used_at": 1}),
        encoding="utf-8",
    )
    (folder / "r-new.json").write_text(
        json.dumps(
            {
                "agent_id": "claude",
                "cwd": "/b",
                "status": "idle",
                "last_used_at": 5,
                "mode": "persistent",
            }
        ),
        encoding="utf-8",
    )
    (folder / "junk.json").write_text("{not json", encoding="utf-8")

    result = runner.invoke(app, ["acp", "sessions", "--config", str(config_file), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["source"] == "store"
    assert [r["record_id"] for r in payload["sessions"]] == ["r-new", "r-old"]


def test_sessions_from_gateway(config_file: Path, online) -> None:
    online.payloads["acp.sessions"] = {
        "sessions": [{"session_key": "agent:main:acp:abcd1234", "agent_id": "claude"}],
        "count": 1,
    }

    result = runner.invoke(app, ["acp", "sessions", "--config", str(config_file)])

    assert result.exit_code == 0, result.output
    assert "agent:main:acp:abcd1234" in result.output
    assert "ACP sessions (gateway)" in result.output


def test_sessions_falls_back_when_manager_missing(config_file: Path, online) -> None:
    online.payloads["acp.sessions"] = _rpc_error("ACP_UNAVAILABLE")

    result = runner.invoke(app, ["acp", "sessions", "--config", str(config_file), "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["source"] == "store"


def test_cancel_and_close_call_gateway(online) -> None:
    online.payloads["acp.cancel"] = {"session_key": "agent:main:acp:1", "cancelled": True}
    online.payloads["acp.close"] = {"session_key": "agent:main:acp:1", "closed": True}

    first = runner.invoke(app, ["acp", "cancel", "agent:main:acp:1"])
    second = runner.invoke(app, ["acp", "close", "agent:main:acp:1", "--discard", "--json"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert ("acp.cancel", {"session_key": "agent:main:acp:1"}) in online.calls
    assert ("acp.close", {"session_key": "agent:main:acp:1", "discard": True}) in online.calls
    assert json.loads(second.stdout)["closed"] is True


def test_cancel_needs_a_gateway() -> None:
    result = runner.invoke(app, ["acp", "cancel", "agent:main:acp:1"])

    assert result.exit_code == 1
