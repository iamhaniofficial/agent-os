"""Shared helpers for the ACP runtime tests (fake agent wiring)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from agentos.acp.runtime import AcpRuntimeImpl
from agentos.acp.types import AcpEvent, AcpTurn
from agentos.gateway.config import AcpAgentOverride, AcpConfig

FAKE_AGENT = str(Path(__file__).with_name("fake_agent.py"))


def fake_config(**kwargs: Any) -> AcpConfig:
    agents = kwargs.pop("agents", None) or {
        "fake": AcpAgentOverride(command=sys.executable, args=[FAKE_AGENT])
    }
    return AcpConfig(agents=agents, **kwargs)


def make_runtime(tmp_path: Path, config: AcpConfig | None = None, **kwargs: Any) -> AcpRuntimeImpl:
    base_env = kwargs.pop("base_env", None) or {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
    }
    kwargs.setdefault("cancel_wait_s", 0.3)
    kwargs.setdefault("kill_grace_s", 0.2)
    kwargs.setdefault("close_wait_s", 1.0)
    kwargs.setdefault("init_timeout_s", 15.0)
    return AcpRuntimeImpl(
        config or fake_config(),
        state_dir=tmp_path / "acp",
        base_env=base_env,
        **kwargs,
    )


async def collect(turn: AcpTurn, timeout: float = 15.0) -> list[AcpEvent]:
    async def _drain() -> list[AcpEvent]:
        return [ev async for ev in turn.events]

    return await asyncio.wait_for(_drain(), timeout)


async def wait_until(predicate: Any, timeout: float = 5.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return bool(predicate())
