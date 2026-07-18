"""AgentRegistry persistence must route through the override-aware gateway
writer, so an agents.* mutation never freezes a one-off CLI ``--listen`` /
``--port`` / ``--debug`` (recorded in the process-global runtime-override map)
into config.toml. The agent change itself must still persist.
"""

from __future__ import annotations

import tomllib

import pytest
import tomli_w

from agentos.agents.registry import AgentRegistry
from agentos.gateway.config import GatewayConfig
from agentos.gateway.config_persist import set_runtime_overrides


@pytest.fixture(autouse=True)
def _reset_overrides():
    """The override map is process-global; isolate every test."""
    set_runtime_overrides(None)
    yield
    set_runtime_overrides(None)


def _seed_disk(path, data: dict) -> None:
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def _read(path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


@pytest.mark.asyncio
async def test_create_agent_does_not_freeze_cli_bind_overrides(tmp_path):
    """AgentRegistry.create_agent with a runtime config (public bind + debug that
    run_gateway injected in memory) must NOT freeze host/port/debug into the file
    — the runtime-override map restores the on-disk originals — while the new
    agent IS persisted."""
    cfg_path = tmp_path / "config.toml"
    _seed_disk(cfg_path, {"host": "127.0.0.1", "port": 18791, "debug": False})

    # run_gateway recorded the on-disk originals before overriding in memory.
    set_runtime_overrides({"host": "127.0.0.1", "port": 18791, "debug": False})
    cfg = GatewayConfig(
        host="0.0.0.0",
        port=19999,
        debug=True,
        config_path=str(cfg_path),
    )
    registry = AgentRegistry(cfg, persist_changes=True)

    await registry.create_agent(agent_id="bot", name="Bot")

    saved = _read(cfg_path)
    # Bind posture NOT frozen to the transient runtime values.
    assert saved.get("host", "127.0.0.1") == "127.0.0.1"
    assert saved.get("port", 18791) == 18791
    assert saved.get("debug", False) is False
    # The agent change WAS persisted.
    assert [a["id"] for a in saved.get("agents", [])] == ["bot"]


@pytest.mark.asyncio
async def test_update_agent_does_not_freeze_cli_bind_overrides(tmp_path):
    cfg_path = tmp_path / "config.toml"
    _seed_disk(
        cfg_path,
        {
            "host": "127.0.0.1",
            "agents": [{"id": "bot", "name": "Bot"}],
        },
    )

    set_runtime_overrides({"host": "127.0.0.1", "port": 18791, "debug": False})
    cfg = GatewayConfig.load(str(cfg_path))
    cfg = cfg.model_copy(update={"host": "0.0.0.0", "debug": True})
    cfg.config_path = str(cfg_path)
    registry = AgentRegistry(cfg, persist_changes=True)

    await registry.update_agent("bot", name="Renamed")

    saved = _read(cfg_path)
    assert saved.get("host", "127.0.0.1") == "127.0.0.1"  # NOT frozen to 0.0.0.0
    assert saved.get("debug", False) is False
    assert saved["agents"][0]["name"] == "Renamed"  # the edit persisted


@pytest.mark.asyncio
async def test_delete_agent_does_not_freeze_cli_bind_overrides(tmp_path):
    cfg_path = tmp_path / "config.toml"
    _seed_disk(
        cfg_path,
        {
            "host": "127.0.0.1",
            "agents": [{"id": "bot", "name": "Bot"}, {"id": "ops", "name": "Ops"}],
        },
    )

    set_runtime_overrides({"host": "127.0.0.1", "port": 18791, "debug": False})
    cfg = GatewayConfig.load(str(cfg_path))
    cfg = cfg.model_copy(update={"host": "0.0.0.0"})
    cfg.config_path = str(cfg_path)
    registry = AgentRegistry(cfg, persist_changes=True)

    await registry.delete_agent("bot")

    saved = _read(cfg_path)
    assert saved.get("host", "127.0.0.1") == "127.0.0.1"  # NOT frozen
    assert [a["id"] for a in saved.get("agents", [])] == ["ops"]  # delete persisted
