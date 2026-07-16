"""Tests for the memory-provider registry (Task B3).

Offline and deterministic: no ``mem0ai``, no network. The mem0 factory path
is exercised for real (the provider module imports without the ``mem0ai``
extra) and via monkeypatch to simulate the "module exists but raises
ImportError" (extra missing at ``initialize``) and generic-failure paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from agentos.memory.providers import registry
from agentos.memory.providers.base import MemoryProvider
from agentos.memory.providers.registry import create_provider


class _FakeProvider(MemoryProvider):
    """Minimal available provider for the success-path test."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        return True

    async def initialize(self, session_id: str, **kwargs: Any) -> None:  # pragma: no cover
        return None


def test_create_provider_unknown_name_returns_none() -> None:
    result = create_provider(
        "nope-not-a-provider",
        memory_config=None,
        agent_state_dir=Path("/tmp/agent"),
    )
    assert result is None


def test_create_provider_mem0_module_imports_cleanly() -> None:
    """B5 landed: ``mem0_provider`` now imports without the ``mem0ai`` dep.

    The provider imports ``mem0ai`` only inside ``initialize`` (never at module
    import time), so the lazy import in ``_make_mem0`` succeeds even with the
    extra absent. Construction with a valid config returns a real provider —
    the ImportError-on-missing-module path from B3 no longer applies. (A bare
    ``object()`` config still degrades to None via the generic-error path,
    proving the registry never crashes boot.)
    """
    from agentos.gateway.config import MemoryConfig

    cfg = MemoryConfig()
    cfg.provider.name = "mem0"
    result = create_provider(
        "mem0",
        memory_config=cfg,
        agent_state_dir=Path("/tmp/agent"),
    )
    assert result is not None
    assert result.name == "mem0"

    # A malformed config never crashes boot — it degrades to None.
    assert (
        create_provider("mem0", memory_config=object(), agent_state_dir=Path("/tmp/agent"))
        is None
    )


def test_create_provider_mem0_import_error_from_factory(monkeypatch: Any) -> None:
    """Future path: provider module exists but its dependency is missing."""

    def _raising_factory(*, memory_config: Any, agent_state_dir: Path) -> MemoryProvider:
        raise ImportError("No module named 'mem0ai'")

    monkeypatch.setitem(registry._FACTORIES, "mem0", _raising_factory)
    result = create_provider(
        "mem0",
        memory_config=object(),
        agent_state_dir=Path("/tmp/agent"),
    )
    assert result is None


def test_create_provider_generic_error_returns_none(monkeypatch: Any) -> None:
    """A non-ImportError construction failure also degrades to None."""

    def _raising_factory(*, memory_config: Any, agent_state_dir: Path) -> MemoryProvider:
        raise RuntimeError("boom")

    monkeypatch.setitem(registry._FACTORIES, "mem0", _raising_factory)
    result = create_provider(
        "mem0",
        memory_config=object(),
        agent_state_dir=Path("/tmp/agent"),
    )
    assert result is None


def test_create_provider_success_path(monkeypatch: Any) -> None:
    """A working factory returns the constructed provider."""

    def _ok_factory(*, memory_config: Any, agent_state_dir: Path) -> MemoryProvider:
        return _FakeProvider(memory_config=memory_config, agent_state_dir=agent_state_dir)

    monkeypatch.setitem(registry._FACTORIES, "mem0", _ok_factory)
    result = create_provider(
        "mem0",
        memory_config=object(),
        agent_state_dir=Path("/tmp/agent"),
    )
    assert isinstance(result, _FakeProvider)
