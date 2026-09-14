"""``gateway(action="config_get")`` regression tests for #1892.

A key whose configured value is ``None`` is present in the config; only a
key that is genuinely absent should raise "Config key not found".
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from agentos.tools.builtin import control
from agentos.tools.types import ToolError


class _StaticConfig:
    """Minimal stand-in for the gateway config object the tool navigates."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def to_toml_dict(self) -> dict[str, Any]:
        return self._data


@pytest.fixture
def gateway_config() -> Iterator[None]:
    previous = control._gateway_config
    control.set_gateway_config(
        _StaticConfig(
            {
                "gateway": {"host": "127.0.0.1", "port": 8000, "auth_token": None},
                "llm": {"params": {"temperature": 0.0, "seed": None}},
                "flag": False,
                "empty": "",
                "zero": 0,
            }
        )
    )
    try:
        yield
    finally:
        control.set_gateway_config(previous)


async def _config_get(key: str) -> Any:
    return json.loads(await control.gateway(action="config_get", key=key))


@pytest.mark.asyncio
@pytest.mark.usefixtures("gateway_config")
async def test_config_get_returns_a_present_key_whose_value_is_none() -> None:
    assert await _config_get("gateway.auth_token") == {
        "action": "config_get",
        "key": "gateway.auth_token",
        "value": None,
    }


@pytest.mark.asyncio
@pytest.mark.usefixtures("gateway_config")
async def test_config_get_returns_a_nested_none_value() -> None:
    assert (await _config_get("llm.params.seed"))["value"] is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("gateway_config")
@pytest.mark.parametrize(
    ("key", "expected"),
    [("gateway.host", "127.0.0.1"), ("flag", False), ("empty", ""), ("zero", 0)],
)
async def test_config_get_returns_falsy_values_verbatim(key: str, expected: Any) -> None:
    assert (await _config_get(key))["value"] == expected


@pytest.mark.asyncio
@pytest.mark.usefixtures("gateway_config")
@pytest.mark.parametrize(
    "key",
    [
        "gateway.missing",  # absent leaf
        "nope.host",  # absent section
        "gateway.port.sub",  # descending through a scalar
        "gateway.auth_token.sub",  # descending through a None leaf
    ],
)
async def test_config_get_still_rejects_an_absent_key(key: str) -> None:
    with pytest.raises(ToolError, match=f"Config key not found: {key}"):
        await control.gateway(action="config_get", key=key)
