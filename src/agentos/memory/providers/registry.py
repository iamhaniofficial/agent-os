"""Memory-provider registry — name → provider factory (Plan B, Task B3).

Maps a configured provider name (``config.memory.provider.name``) to a
constructed :class:`MemoryProvider`. Failures never crash boot: an unknown
name or a missing optional dependency logs an actionable warning and returns
``None`` so the gateway degrades to no external provider.

The mem0 provider (Task B5) and its ``mem0ai`` dependency are imported lazily
INSIDE the factory. Nothing here imports the provider implementation or
``mem0ai`` at module import time — importing this module is cheap and safe
even when the ``mem0`` extra is not installed.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agentos.memory.providers.base import MemoryProvider

logger = structlog.get_logger(__name__)

# pip target for the mem0 optional dependency group.
_MEM0_EXTRA = "pip install 'use-agent-os[mem0]'"


def _make_mem0(
    *, memory_config: Any, agent_state_dir: Path
) -> MemoryProvider:
    """Construct the mem0 provider (Task B5).

    Lazily imports ``agentos.memory.providers.mem0_provider``. That module
    imports ``mem0ai`` only inside ``initialize`` (never at import time), so
    this import succeeds whether or not the optional ``mem0`` extra is present;
    a genuinely missing/broken provider module raises :class:`ImportError`,
    which the caller routes to the actionable-warning path.
    """
    from agentos.memory.providers.mem0_provider import Mem0Provider

    provider: MemoryProvider = Mem0Provider(
        memory_config=memory_config, agent_state_dir=agent_state_dir
    )
    return provider


# name → factory. Extend as new providers land.
_FACTORIES: dict[str, Callable[..., MemoryProvider]] = {
    "mem0": _make_mem0,
}


def create_provider(
    name: str,
    *,
    memory_config: Any,
    agent_state_dir: Path,
) -> MemoryProvider | None:
    """Construct the memory provider named ``name``, or ``None`` on failure.

    * Unknown ``name`` → warning naming the known providers, ``None``.
    * ``ImportError`` from the factory (optional extra missing, e.g. the
      ``mem0`` extra not installed, or the provider module not yet present) →
      actionable warning naming ``pip install 'use-agent-os[mem0]'``, ``None``.
    * Any other construction error → warning, ``None``.

    The gateway boots regardless of the outcome.
    """
    factory = _FACTORIES.get(name)
    if factory is None:
        logger.warning(
            "memory_provider.unknown_name",
            requested=name,
            known=sorted(_FACTORIES),
        )
        return None

    try:
        return factory(memory_config=memory_config, agent_state_dir=agent_state_dir)
    except ImportError as exc:
        logger.warning(
            "memory_provider.extra_not_installed",
            provider=name,
            install=_MEM0_EXTRA,
            error=str(exc),
        )
        return None
    except Exception as exc:  # noqa: BLE001 — never let provider setup crash boot
        logger.warning(
            "memory_provider.create_failed",
            provider=name,
            error=str(exc),
        )
        return None
