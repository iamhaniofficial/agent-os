"""Agent-facing environment tools.

``skill_list`` has always been able to tell the model *that* a skill needs
``BASE_RPC_URL``; it could not do anything about it. These two tools close
that gap, with the asymmetry the risk warrants:

* ``env_list`` is exposed by default and returns names and set/unset state.
  It carries no values at all — not even masked ones. Diagnosing "the skill
  is unavailable because X is missing" needs the name; it never needs the
  secret, and there is no reason to put one in a transcript.
* ``env_set`` is hidden by default and gated behind the approval queue. It
  writes a credential to disk on the operator's machine, so a human resolves
  it the same way they resolve a patch or a warnlisted command.

There is deliberately no reveal tool. A model that can read back stored
credentials is one prompt injection away from exfiltrating them, and nothing
an agent legitimately does requires it.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from agentos import env_catalog, env_policy, env_store
from agentos.tools.registry import tool
from agentos.tools.types import ToolError

log = structlog.get_logger(__name__)

_ENV_APPROVAL_NAMESPACE = "exec"
_ENV_APPROVAL_TOOL = "env_set"


def _approval_params(name: str) -> dict[str, Any]:
    """Return the approval record for setting *name* — describing, not carrying.

    The value is not part of the record. An approval prompt is rendered in a
    UI and stored in a queue; putting the secret in it would spread the
    credential to two more places for no benefit, since what the operator is
    deciding is whether this variable may be written at all.
    """
    return {
        "toolName": _ENV_APPROVAL_TOOL,
        "command": f"env_set {name}",
        "args": {"name": name},
        "warning": f"Writes {name} to ~/.agentos/.env and applies it to the running gateway.",
        "mode": "env",
    }


def _approval_envelope(status: str, approval_id: str, name: str, message: str) -> str:
    return json.dumps(
        {
            "status": status,
            "approval_id": approval_id,
            "name": name,
            "command": f"env_set {name}",
            "message": message,
        }
    )


def _gate_env_set(name: str, approval_id: str | None) -> str | None:
    """Return an envelope when the write needs (or fails) approval, else ``None``."""
    from agentos.gateway.approval_queue import get_approval_queue

    queue = get_approval_queue()
    params = _approval_params(name)

    if approval_id is not None:
        try:
            entry = queue.get(approval_id)
        except KeyError as exc:
            raise ToolError(str(exc)) from exc
        if entry.namespace != _ENV_APPROVAL_NAMESPACE:
            raise ToolError(f"Approval does not belong to exec namespace: {approval_id}")
        if entry.params.get("toolName") != _ENV_APPROVAL_TOOL:
            raise ToolError(f"Approval does not belong to env_set: {approval_id}")
        # Bind the approval to the variable it was granted for, so an approval
        # for a harmless name cannot be replayed to write a different one.
        if entry.params.get("args") != params["args"]:
            raise ToolError("Approval does not match the requested variable")
        if entry.consumed:
            raise ToolError(f"Approval already consumed: {approval_id}")
        if not entry.resolved:
            return _approval_envelope(
                "approval_pending", approval_id, name, "Approval is still pending."
            )
        if not entry.approved:
            return _approval_envelope("approval_denied", approval_id, name, "Approval was denied.")
        try:
            queue.consume(approval_id)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return None

    settings = queue.get_settings()
    new_id = queue.request(namespace=_ENV_APPROVAL_NAMESPACE, params=params)
    if settings.mode == "auto-approve":
        queue.resolve(new_id, True)
        queue.consume(new_id)
        return None
    if settings.mode == "auto-deny":
        queue.resolve(new_id, False)
        return _approval_envelope(
            "approval_denied", new_id, name, "Denied by the active approval policy."
        )
    return _approval_envelope(
        "approval_required",
        new_id,
        name,
        "Resolve this approval via exec.approval.resolve and retry with the returned approval_id.",
    )


@tool(
    name="env_list",
    description=(
        "List environment variables AgentOS knows about and whether each is set. "
        "Returns names and status only — never values."
    ),
    params={
        "missing_only": {
            "type": "boolean",
            "description": "Only return variables that are not set",
        },
    },
    defer_loading=True,
)
async def env_list(missing_only: bool = False) -> str:
    catalog = env_catalog.build_catalog(present_names=set(env_store.read_env_file()))
    rows = []
    for name, spec in sorted(catalog.items()):
        entry = env_store.resolve_entry(name, secret=spec.secret)
        if missing_only and entry.is_set:
            continue
        rows.append(
            {
                "name": name,
                "is_set": entry.is_set,
                "description": spec.description,
                "url": spec.url,
                "category": spec.category,
                "needed_by": spec.owner,
                "writable": entry.writable,
            }
        )
    return json.dumps({"count": len(rows), "vars": rows})


@tool(
    name="env_set",
    description=(
        "Set an environment variable in ~/.agentos/.env. Requires operator approval. "
        "Use when a skill or provider reports a missing variable and the user has "
        "supplied the value."
    ),
    params={
        "name": {"type": "string", "description": "Variable name"},
        "value": {"type": "string", "description": "Value to store"},
        "approval_id": {
            "type": "string",
            "description": "Approval id returned by a previous call",
        },
    },
    required=["name", "value"],
    exposed_by_default=False,
)
async def env_set(name: str, value: str, approval_id: str | None = None) -> str:
    # Policy first: a denylisted name must be refused outright rather than
    # queued for an operator who might wave it through without reading it.
    try:
        env_policy.assert_writable(name)
        env_policy.sanitize_value(name, value)
    except env_policy.EnvPolicyError as exc:
        raise ToolError(str(exc)) from exc

    gate = _gate_env_set(name, approval_id)
    if gate is not None:
        return gate

    entry = env_store.set_env_var(name, value)
    spec = env_catalog.describe(name, env_catalog.build_catalog())
    log.info("env.tool_set", key=name)
    return json.dumps(
        {
            "status": "ok",
            "name": entry.name,
            "is_set": entry.is_set,
            "restart_required": spec.restart_required,
            "note": (
                "Restart the gateway for this to take full effect."
                if spec.restart_required
                else "Applied to the running gateway."
            ),
        }
    )
