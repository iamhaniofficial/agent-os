"""Address an outbound message to a resolved channel target.

One rule for every out-of-turn sender (the scheduler's delivery chain, the
"gateway restarted" announcement): a thread id names a conversation *inside*
the chat, so it is what the message is addressed to, with the chat carried
beside it in ``metadata["channel"]`` — the same rule
``channel_dispatch._route_envelope_reply_message`` applies to in-turn replies.
Dropping it delivered a job scheduled in a Telegram forum topic to the
group's General topic, and one scheduled in a Discord thread to the parent
channel.
"""

from __future__ import annotations

from agentos.channels.types import OutgoingMessage


def build_outgoing_for_target(
    *,
    channel_type: str,
    to: str,
    thread_id: str = "",
    text: str,
    configured_recipient: bool = False,
    bare_reply_to: str = "cron",
) -> OutgoingMessage:
    """Build the ``OutgoingMessage`` for a target ``resolve_delivery_target`` returned.

    ``channel_type`` is the adapter's type (``slack``, ``telegram``, ...),
    ``to`` the chat or channel id and ``thread_id`` the thread inside it.

    Slack resolves its destination from ``metadata``: a thread goes to
    ``reply_to=<thread_ts>`` with the channel beside it; a bare channel post
    carries the channel in metadata with ``thread_ts`` explicitly ``None``,
    and ``reply_to`` is only a marker (``bare_reply_to``) the adapter does not
    treat as a timestamp.

    ``configured_recipient`` says ``to`` is an address someone configured
    rather than the key of the conversation a job came from. The email
    adapter needs to know: a thread key is a Message-ID with a mailbox's
    shape, so it refuses to read a recipient off ``reply_to`` and only mails
    an address it is handed as ``metadata["to"]``.
    """
    if channel_type == "slack":
        if thread_id:
            return OutgoingMessage(content=text, reply_to=thread_id, metadata={"channel": to})
        return OutgoingMessage(
            content=text,
            reply_to=bare_reply_to,
            metadata={"channel": to, "thread_ts": None},
        )
    if channel_type == "email" and configured_recipient and to:
        return OutgoingMessage(content=text, reply_to=to, metadata={"to": to})
    thread_metadata = {"channel": to} if thread_id and to else {}
    return OutgoingMessage(
        content=text,
        reply_to=thread_id or to or None,
        metadata=thread_metadata,
    )


__all__ = ["build_outgoing_for_target"]
