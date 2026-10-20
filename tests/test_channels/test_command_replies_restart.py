"""The channel reply for ``/restart``: what the sender reads before the drain."""

from __future__ import annotations

from agentos.channels.command_replies import format_channel_success_reply


def _reply(payload: object) -> str | None:
    return format_channel_success_reply(name="restart", method="gateway.restart", payload=payload)


def test_safe_restart_names_the_drain_window() -> None:
    text = _reply({"accepted": True, "mode": "safe", "drainTimeoutSeconds": 30.0})
    assert text == (
        "Restarting gateway… draining active turns (up to 30s). I'll confirm here when it's back."
    )


def test_force_restart_says_so() -> None:
    text = _reply({"accepted": True, "mode": "force", "drainTimeoutSeconds": 2.0})
    assert text == "Restarting gateway now (force). I'll confirm here when it's back."


def test_pending_restart_is_reported() -> None:
    assert _reply({"accepted": False, "alreadyPending": True}) == (
        "A gateway restart is already in progress."
    )


def test_unexpected_payload_falls_back_to_the_generic_line() -> None:
    assert _reply({"accepted": False}) is None
    assert _reply("nope") is None
