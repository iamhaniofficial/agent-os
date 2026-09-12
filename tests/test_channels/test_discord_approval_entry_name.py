"""Issue #1600: the Discord approval binding must compare the sessionKey's
channel segment against the entry's configured name, not the literal
``"discord"``.

Session keys embed the *entry name* (``ChannelManager._build_session_key`` ->
``build_group_key(channel=entry.name, ...)``), and entry names are unique, so
any multi-account Discord setup has at least one entry not called
``"discord"`` -- for which every Approve/Deny click was logged as
``discord.component_mismatch`` and the approval never resolved. Telegram and
MS Teams already compare against ``self.config.name``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentos.channels.discord import DiscordChannel, DiscordChannelConfig
from agentos.channels.registry import build_managed_channel
from agentos.gateway.approval_queue import get_approval_queue, reset_approval_queue
from agentos.gateway.config import DiscordChannelEntry
from agentos.session.keys import build_direct_key, build_group_key

CHANNEL_ID = "chan123"
USER_ID = "usr123"


@pytest.fixture(autouse=True)
def _clean_approval_queue():
    reset_approval_queue()
    yield
    reset_approval_queue()


def _channel(name: str) -> tuple[DiscordChannel, list[tuple[str, Any]]]:
    channel = DiscordChannel(DiscordChannelConfig(token="test-token", name=name))
    channel.policy = replace(channel.policy, allowlist=frozenset(), allowlist_enabled=False)
    post_calls: list[tuple[str, Any]] = []

    async def fake_post(path, json=None, headers=None, **kwargs):
        post_calls.append((path, json))

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"id": "999"}

        return FakeResp()

    channel._get_client = lambda: AsyncMock(post=fake_post)
    return channel, post_calls


def _click(approval_id: str, *, action: str = "approve", channel_id: str = CHANNEL_ID) -> dict:
    return {
        "type": 3,
        "id": "int123",
        "token": "token123",
        "channel_id": channel_id,
        "user": {"id": USER_ID},
        "message": {"content": "Approval requested"},
        "data": {"custom_id": f"{action}:{approval_id}"},
    }


def _pending(session_key: str) -> str:
    return get_approval_queue().request(
        "exec", {"argv": ["rm", "-rf"], "action_kind": "exec", "sessionKey": session_key}
    )


@pytest.mark.asyncio
async def test_approval_resolves_for_a_non_default_entry_name() -> None:
    queue = get_approval_queue()
    approval_id = _pending(
        build_group_key(agent_id="main", channel="discord-support", peer_id=CHANNEL_ID)
    )
    channel, post_calls = _channel("discord-support")

    await channel._handle_discord_component_interaction(_click(approval_id))

    entry = queue.get(approval_id)
    assert entry.resolved is True
    assert entry.approved is True
    assert post_calls and post_calls[0][1]["type"] == 7  # UPDATE_MESSAGE
    assert "Approved" in post_calls[0][1]["data"]["content"]


@pytest.mark.asyncio
async def test_direct_message_approval_resolves_for_a_non_default_entry_name() -> None:
    queue = get_approval_queue()
    approval_id = _pending(
        build_direct_key(agent_id="main", channel="discord-support", peer_id=USER_ID)
    )
    channel, _ = _channel("discord-support")

    await channel._handle_discord_component_interaction(_click(approval_id, action="deny"))

    entry = queue.get(approval_id)
    assert entry.resolved is True
    assert entry.approved is False


@pytest.mark.asyncio
async def test_default_entry_name_keeps_working() -> None:
    queue = get_approval_queue()
    approval_id = _pending(build_group_key(agent_id="main", channel="discord", peer_id=CHANNEL_ID))
    channel, _ = _channel("discord")

    await channel._handle_discord_component_interaction(_click(approval_id))

    assert queue.get(approval_id).resolved is True


@pytest.mark.asyncio
async def test_click_from_another_entry_is_still_rejected() -> None:
    """The binding is tightened, not loosened: an approval raised on the
    ``discord-ops`` entry cannot be resolved by a click on ``discord-support``,
    even from the same channel id."""
    queue = get_approval_queue()
    approval_id = _pending(
        build_group_key(agent_id="main", channel="discord-ops", peer_id=CHANNEL_ID)
    )
    channel, _ = _channel("discord-support")

    await channel._handle_discord_component_interaction(_click(approval_id))

    assert queue.get(approval_id).resolved is False


@pytest.mark.asyncio
async def test_click_from_another_channel_is_still_rejected() -> None:
    queue = get_approval_queue()
    approval_id = _pending(
        build_group_key(agent_id="main", channel="discord-support", peer_id=CHANNEL_ID)
    )
    channel, _ = _channel("discord-support")

    await channel._handle_discord_component_interaction(
        _click(approval_id, channel_id="other-chan")
    )

    assert queue.get(approval_id).resolved is False


def test_registry_hands_the_entry_name_to_the_adapter() -> None:
    """The literal was only half the defect: the generic builder copies
    ``entry.name`` into the config only when the config declares ``name``,
    so without the field the adapter could never learn its own entry name."""
    entry = DiscordChannelEntry(name="discord-support", token="test-token")

    adapter = build_managed_channel(entry)

    assert isinstance(adapter, DiscordChannel)
    assert adapter.config.name == "discord-support"


def test_config_name_defaults_to_discord() -> None:
    assert DiscordChannelConfig(token="test-token").name == "discord"
