"""``[gateway_control] chat_restart``: default, strictness, round-trip, env."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentos.gateway.config import GatewayConfig, GatewayControlConfig


def test_chat_restart_defaults_off() -> None:
    assert GatewayConfig().gateway_control.chat_restart is False


def test_gateway_control_forbids_extra() -> None:
    with pytest.raises(ValidationError):
        GatewayControlConfig(chat_restart=True, bogus=1)  # type: ignore[call-arg]


def test_chat_restart_toml_round_trip(tmp_path: Path) -> None:
    cfg = tmp_path / "agentos.toml"
    cfg.write_text("[gateway_control]\nchat_restart = true\n", encoding="utf-8")
    loaded = GatewayConfig.load_from_toml(cfg)
    assert loaded.gateway_control.chat_restart is True


def test_chat_restart_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTOS_GATEWAY_GATEWAY_CONTROL__CHAT_RESTART", "true")
    assert GatewayConfig().gateway_control.chat_restart is True


def test_gateway_control_in_public_dict() -> None:
    public = GatewayConfig().to_public_dict()
    assert public.get("gateway_control") == {"chat_restart": False}
