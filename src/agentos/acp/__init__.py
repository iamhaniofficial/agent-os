"""ACP (Agent Client Protocol) consumer: drive external coding harnesses.

AgentOS acts as the ACP *client*; each harness (Claude Code, Codex, Gemini
CLI, Cursor, ...) runs as an ACP *agent* subprocess speaking JSON-RPC over
stdio. The design mirrors OpenClaw's ``acpx`` runtime (MIT, see
``THIRD_PARTY_NOTICES.md``): one normalized event stream, a permission
bridge, a session store with resume handles, and process leases so no
harness outlives the gateway.

Layout (each stream of the initial build owns one area):

- ``types``      shared contract: events, specs, handles, protocols
- ``registry``   built-in harness definitions (argv per agent id)
- ``runtime``    ``AcpRuntime`` implementation over the ``acp`` SDK
- ``process``    subprocess launch, env scrubbing, leases, reaper
- ``store``      persisted session records / resume handles
- ``permissions`` permission + elicitation bridge to ``ApprovalQueue``
- ``manager``    ``AcpSessionManager``: actor per session, turn driving
- ``projector``  ACP events -> AgentOS engine events / transcript rows
"""

from agentos.acp.types import (
    AcpEvent,
    AcpSessionHandle,
    AcpSessionSpec,
    AcpTurn,
    AcpTurnResult,
    PermissionDecision,
    PermissionRequest,
)

__all__ = [
    "AcpEvent",
    "AcpSessionHandle",
    "AcpSessionSpec",
    "AcpTurn",
    "AcpTurnResult",
    "PermissionDecision",
    "PermissionRequest",
]
