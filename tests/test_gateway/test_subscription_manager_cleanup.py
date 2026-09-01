"""SubscriptionManager must not leave empty subscriber sets behind.

Regression tests for #609: `_message_subs` kept a `session_key -> set()` entry
alive after the last subscriber went away, so a long-running gateway grew one
dead dict entry per session key it ever served. The topic path already pruned
itself; the message path now matches it.
"""

from __future__ import annotations

from agentos.gateway.websocket import SubscriptionManager


def test_unsubscribe_messages_drops_the_key_once_the_last_subscriber_leaves() -> None:
    mgr = SubscriptionManager()
    mgr.subscribe_messages("conn-1", "sess-a")

    mgr.unsubscribe_messages("conn-1", "sess-a")

    assert mgr._message_subs == {}
    assert mgr.get_message_subscribers("sess-a") == set()


def test_unsubscribe_messages_keeps_the_key_while_others_remain() -> None:
    mgr = SubscriptionManager()
    mgr.subscribe_messages("conn-1", "sess-a")
    mgr.subscribe_messages("conn-2", "sess-a")

    mgr.unsubscribe_messages("conn-1", "sess-a")

    assert mgr.get_message_subscribers("sess-a") == {"conn-2"}
    assert set(mgr._message_subs) == {"sess-a"}


def test_unsubscribe_messages_only_touches_the_named_session() -> None:
    mgr = SubscriptionManager()
    mgr.subscribe_messages("conn-1", "sess-a")
    mgr.subscribe_messages("conn-1", "sess-b")

    mgr.unsubscribe_messages("conn-1", "sess-a")

    assert set(mgr._message_subs) == {"sess-b"}
    assert mgr.get_message_subscribers("sess-b") == {"conn-1"}


def test_unsubscribe_messages_for_an_unknown_key_creates_nothing() -> None:
    mgr = SubscriptionManager()

    mgr.unsubscribe_messages("conn-1", "never-subscribed")

    assert mgr._message_subs == {}


def test_unsubscribe_messages_by_a_non_subscriber_leaves_the_set_intact() -> None:
    mgr = SubscriptionManager()
    mgr.subscribe_messages("conn-1", "sess-a")

    mgr.unsubscribe_messages("conn-2", "sess-a")

    assert mgr.get_message_subscribers("sess-a") == {"conn-1"}


def test_repeated_unsubscribe_is_idempotent() -> None:
    mgr = SubscriptionManager()
    mgr.subscribe_messages("conn-1", "sess-a")

    mgr.unsubscribe_messages("conn-1", "sess-a")
    mgr.unsubscribe_messages("conn-1", "sess-a")

    assert mgr._message_subs == {}


def test_resubscribe_after_cleanup_works() -> None:
    mgr = SubscriptionManager()
    mgr.subscribe_messages("conn-1", "sess-a")
    mgr.unsubscribe_messages("conn-1", "sess-a")

    mgr.subscribe_messages("conn-1", "sess-a")

    assert mgr.get_message_subscribers("sess-a") == {"conn-1"}


def test_remove_connection_prunes_emptied_message_subscriptions() -> None:
    mgr = SubscriptionManager()
    for key in ("sess-a", "sess-b", "sess-c"):
        mgr.subscribe_messages("conn-1", key)

    mgr.remove_connection("conn-1")

    assert mgr._message_subs == {}


def test_remove_connection_keeps_sessions_with_other_subscribers() -> None:
    mgr = SubscriptionManager()
    mgr.subscribe_messages("conn-1", "shared")
    mgr.subscribe_messages("conn-2", "shared")
    mgr.subscribe_messages("conn-1", "solo")

    mgr.remove_connection("conn-1")

    assert set(mgr._message_subs) == {"shared"}
    assert mgr.get_message_subscribers("shared") == {"conn-2"}


def test_remove_connection_still_clears_session_and_topic_subscriptions() -> None:
    mgr = SubscriptionManager()
    mgr.subscribe_sessions("conn-1")
    mgr.subscribe_messages("conn-1", "sess-a")
    mgr.subscribe_topic("conn-1", "cron")

    mgr.remove_connection("conn-1")

    assert mgr.get_session_subscribers() == set()
    assert mgr._message_subs == {}
    assert mgr._topic_subs == {}


def test_churning_many_sessions_leaves_no_residue() -> None:
    """The leak in #609 showed up as unbounded growth across a connection churn."""
    mgr = SubscriptionManager()

    for i in range(200):
        conn_id = f"conn-{i}"
        mgr.subscribe_messages(conn_id, f"sess-{i}")
        mgr.subscribe_topic(conn_id, f"topic-{i}")
        if i % 2:
            mgr.unsubscribe_messages(conn_id, f"sess-{i}")
            mgr.unsubscribe_topic(conn_id, f"topic-{i}")
        else:
            mgr.remove_connection(conn_id)

    assert mgr._message_subs == {}
    assert mgr._topic_subs == {}


def test_message_and_topic_paths_prune_alike() -> None:
    """Drift guard: the two dict-backed paths must stay symmetric (#609)."""
    mgr = SubscriptionManager()
    mgr.subscribe_messages("conn-1", "sess-a")
    mgr.subscribe_topic("conn-1", "sess-a")

    mgr.unsubscribe_messages("conn-1", "sess-a")
    mgr.unsubscribe_topic("conn-1", "sess-a")
    assert list(mgr._message_subs) == list(mgr._topic_subs) == []

    mgr.subscribe_messages("conn-1", "sess-a")
    mgr.subscribe_topic("conn-1", "sess-a")

    mgr.remove_connection("conn-1")
    assert list(mgr._message_subs) == list(mgr._topic_subs) == []
