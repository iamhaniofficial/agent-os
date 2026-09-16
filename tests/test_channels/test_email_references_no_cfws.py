"""Threading ids with no whitespace between them are still separate ids (#2080).

RFC 5322 3.6.4 makes the CFWS between two ``msg-id`` tokens optional, so
``<root@x><m2@x>`` is a well-formed ``References`` header that several clients
emit. ``str.split()`` saw one token and ``.strip("<>")`` only peeled the outer
brackets, so the run collapsed into a single bogus id: the thread key was no
longer the root, the same mail thread landed in a second session, and the
outbound chain went back on the wire mangled.
"""

from __future__ import annotations

from email.message import EmailMessage
from email.parser import BytesParser
from email.policy import default as email_policy
from typing import Any

import pytest

from agentos.channels.email import (
    EmailChannel,
    EmailChannelConfig,
    _merge_references,
    _message_ids,
    thread_key_for,
)


def _config(**overrides: Any) -> EmailChannelConfig:
    base: dict[str, Any] = {
        "name": "inbox",
        "imap_host": "imap.example.com",
        "imap_username": "agent@example.com",
        "smtp_host": "smtp.example.com",
        "from_address": "agent@example.com",
        "allowed_senders": ["owner@example.com"],
    }
    base.update(overrides)
    return EmailChannelConfig(**base)


def _raw(*, message_id: str, headers: dict[str, str]) -> EmailMessage:
    message = EmailMessage()
    message["From"] = "owner@example.com"
    message["To"] = "agent@example.com"
    message["Subject"] = "Re: deploy"
    message["Message-ID"] = f"<{message_id}>"
    for key, value in headers.items():
        message[key] = value
    message.set_content("ping")
    parsed = BytesParser(policy=email_policy).parsebytes(message.as_bytes())
    assert isinstance(parsed, EmailMessage)
    return parsed


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<root@example.com><m2@example.com>", ["root@example.com", "m2@example.com"]),
        ("<a@x><b@x><c@x>", ["a@x", "b@x", "c@x"]),
        ("<a@x><b@x> <c@x>", ["a@x", "b@x", "c@x"]),
        ("<a@x>(reply)<b@x>", ["a@x", "b@x"]),
        ("<a@x>\r\n\t<b@x><c@x>", ["a@x", "b@x", "c@x"]),
    ],
)
def test_ids_without_cfws_between_them_are_split(raw: str, expected: list[str]) -> None:
    assert _message_ids(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<a@x> <b@x>", ["a@x", "b@x"]),
        ("a@x b@x", ["a@x", "b@x"]),
        ("a@x <b@x>", ["a@x", "b@x"]),
        ("(from Outlook) <a@x> <b@x>", ["a@x", "b@x"]),
        ("<a@x> (from Outlook)", ["a@x"]),
        ("", []),
        (None, []),
        ("   ", []),
    ],
)
def test_spaced_bare_and_commented_ids_are_unchanged(raw: Any, expected: list[str]) -> None:
    assert _message_ids(raw) == expected


def test_thread_key_is_the_root_whether_or_not_ids_are_spaced() -> None:
    spaced = _raw(
        message_id="m3@example.com",
        headers={"References": "<root@example.com> <m2@example.com>"},
    )
    unspaced = _raw(
        message_id="m3@example.com",
        headers={"References": "<root@example.com><m2@example.com>"},
    )

    assert thread_key_for(unspaced) == thread_key_for(spaced) == "root@example.com"


def test_unspaced_references_land_in_the_same_session_thread() -> None:
    channel = EmailChannel(config=_config())
    spaced = channel._to_incoming(
        _raw(
            message_id="m3@example.com",
            headers={"References": "<root@example.com> <m2@example.com>"},
        )
    )
    unspaced = channel._to_incoming(
        _raw(
            message_id="m4@example.com",
            headers={"References": "<root@example.com><m2@example.com><m3@example.com>"},
        )
    )

    assert spaced is not None and unspaced is not None
    assert unspaced.metadata["native_thread_id"] == spaced.metadata["native_thread_id"]
    assert unspaced.metadata["native_thread_id"] == "root@example.com"


def test_outbound_chain_is_well_formed_from_an_unspaced_parent() -> None:
    parsed = _raw(
        message_id="m3@example.com",
        headers={"References": "<root@example.com><m2@example.com>"},
    )

    assert _merge_references(parsed, "m3@example.com") == (
        "<root@example.com> <m2@example.com> <m3@example.com>"
    )


def test_unspaced_in_reply_to_fallback() -> None:
    parsed = _raw(
        message_id="m3@example.com",
        headers={"In-Reply-To": "<root@example.com><m2@example.com>"},
    )

    assert thread_key_for(parsed) == "root@example.com"
    assert _merge_references(parsed, "m3@example.com") == (
        "<root@example.com> <m2@example.com> <m3@example.com>"
    )
