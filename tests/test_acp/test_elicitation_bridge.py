"""ApprovalQueueElicitationBridge: form / URL accept, decline, timeout, limits."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from acp_perm_helpers import make_queue, next_pending

from agentos.acp.elicitation import (
    ACP_ELICITATION_NAMESPACE,
    ApprovalQueueElicitationBridge,
    UnsupportedElicitationError,
    batch_fields,
    compile_form,
    decode_answer,
    validate_url,
)
from agentos.acp.types import ElicitationRequest

SESSION = "agent:main:acp:0badc0de"


def _req(mode: str = "form", schema=None, url=None, message="Pick options") -> ElicitationRequest:
    return ElicitationRequest(
        session_key=SESSION,
        acp_session_id="sess-1",
        agent_id="codex",
        message=message,
        mode=mode,
        schema=schema,
        url=url,
    )


FORM = {
    "type": "object",
    "properties": {
        "branch": {"type": "string", "title": "Branch name"},
        "push": {"type": "boolean", "title": "Push after commit"},
        "count": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["branch"],
}


async def test_form_accept_with_payload(tmp_path) -> None:
    queue = make_queue(tmp_path)
    bridge = ApprovalQueueElicitationBridge(queue=queue)
    task = asyncio.create_task(bridge.ask(_req(schema=FORM)))
    row = await next_pending(queue, ACP_ELICITATION_NAMESPACE)
    params = row["params"]
    assert params["mode"] == "form"
    assert params["displayTitle"] == "Codex asks: Pick options"
    assert [q["id"] for q in params["questions"]] == ["branch", "push", "count"]
    assert params["questions"][1]["choices"] == [
        {"value": "true", "label": "True"},
        {"value": "false", "label": "False"},
    ]
    queue.resolve(
        row["id"],
        True,
        payload={"content": {"branch": "feat/x", "push": "yes", "count": "3"}},
    )
    response = await task
    assert response.action == "accept"
    assert response.content == {"branch": "feat/x", "push": True, "count": 3}
    # The answers do not linger in the queue file.
    assert queue.get(row["id"]).payload is None
    queue.close()


async def test_form_denied_declines(tmp_path) -> None:
    queue = make_queue(tmp_path)
    bridge = ApprovalQueueElicitationBridge(queue=queue)
    task = asyncio.create_task(bridge.ask(_req(schema=FORM)))
    row = await next_pending(queue, ACP_ELICITATION_NAMESPACE)
    queue.resolve(row["id"], False)
    assert (await task).action == "decline"
    queue.close()


async def test_form_missing_required_answer_declines(tmp_path) -> None:
    queue = make_queue(tmp_path)
    bridge = ApprovalQueueElicitationBridge(queue=queue)
    task = asyncio.create_task(bridge.ask(_req(schema=FORM)))
    row = await next_pending(queue, ACP_ELICITATION_NAMESPACE)
    queue.resolve(row["id"], True, payload={"content": {"push": True}})
    assert (await task).action == "decline"
    queue.close()


async def test_form_timeout_declines(tmp_path) -> None:
    queue = make_queue(tmp_path)
    bridge = ApprovalQueueElicitationBridge(queue=queue, timeout_s=0.05)
    response = await bridge.ask(_req(schema=FORM))
    assert response.action == "decline"
    assert queue.list_pending(ACP_ELICITATION_NAMESPACE) == []
    queue.close()


async def test_default_timeout_is_fifteen_minutes(tmp_path) -> None:
    queue = make_queue(tmp_path)
    bridge = ApprovalQueueElicitationBridge(queue=queue)
    task = asyncio.create_task(bridge.ask(_req(mode="url", url="https://example.com/auth")))
    row = await next_pending(queue, ACP_ELICITATION_NAMESPACE)
    assert 890 <= row["expires_at"] - row["created_at"] <= 900
    queue.resolve(row["id"], False)
    await task
    queue.close()


async def test_batches_of_three_and_secret_alone(tmp_path, caplog) -> None:
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"},
            "c": {"type": "string"},
            "d": {"type": "string"},
            "token": {"type": "string", "_meta": {"codex": {"isSecret": True}}},
        },
    }
    queue = make_queue(tmp_path)
    bridge = ApprovalQueueElicitationBridge(queue=queue)
    caplog.set_level(logging.DEBUG)
    task = asyncio.create_task(bridge.ask(_req(schema=schema)))
    seen: set[str] = set()
    rows = []
    answers = [
        {"a": "1", "b": "2", "c": "3"},
        {"d": "4"},
        {"token": "s3cr3t-value-never-logged"},
    ]
    for answer in answers:
        row = await next_pending(queue, ACP_ELICITATION_NAMESPACE, seen=seen)
        rows.append(row)
        queue.resolve(row["id"], True, payload={"content": answer})
    response = await task
    assert [len(r["params"]["questions"]) for r in rows] == [3, 1, 1]
    assert rows[2]["params"]["questions"][0]["secret"] is True
    assert rows[0]["params"]["displayTitle"].endswith("(1/3)")
    assert response.action == "accept"
    assert response.content == {
        "a": "1",
        "b": "2",
        "c": "3",
        "d": "4",
        "token": "s3cr3t-value-never-logged",
    }
    assert "s3cr3t-value-never-logged" not in caplog.text
    for row in rows:
        assert "s3cr3t" not in json.dumps(queue.status(row["id"]))
    queue.close()


async def test_url_accept_and_decline(tmp_path) -> None:
    queue = make_queue(tmp_path)
    bridge = ApprovalQueueElicitationBridge(queue=queue)
    task = asyncio.create_task(
        bridge.ask(_req(mode="url", url="https://example.com/login", message="Sign in"))
    )
    row = await next_pending(queue, ACP_ELICITATION_NAMESPACE)
    assert row["params"]["url"] == "https://example.com/login"
    assert row["params"]["options"] == ["Continue", "Decline"]
    assert "https://example.com/login" in row["params"]["displayTitle"]
    queue.resolve(row["id"], True)
    response = await task
    assert response.action == "accept" and response.content is None

    task = asyncio.create_task(bridge.ask(_req(mode="url", url="https://example.com/x")))
    row = await next_pending(queue, ACP_ELICITATION_NAMESPACE, seen={row["id"]})
    queue.resolve(row["id"], False)
    assert (await task).action == "decline"
    queue.close()


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "file:///etc/passwd",
        "https://user:pw@example.com/",
        " https://example.com/",
        "https://example.com/\x07",
        "https://" + "a" * 3000,
    ],
)
async def test_unsafe_urls_are_declined_without_asking(tmp_path, url) -> None:
    queue = make_queue(tmp_path)
    bridge = ApprovalQueueElicitationBridge(queue=queue)
    assert (await bridge.ask(_req(mode="url", url=url))).action == "decline"
    assert queue.list_pending() == []
    queue.close()


async def test_unsupported_mode_and_unattended_decline(tmp_path) -> None:
    queue = make_queue(tmp_path)
    bridge = ApprovalQueueElicitationBridge(queue=queue)
    assert (await bridge.ask(_req(mode="other"))).action == "decline"
    unattended = ApprovalQueueElicitationBridge(queue=queue, is_unattended=lambda key: True)
    assert (await unattended.ask(_req(schema=FORM))).action == "decline"
    assert queue.list_pending() == []
    queue.close()


async def test_empty_form_is_a_confirmation(tmp_path) -> None:
    queue = make_queue(tmp_path)
    bridge = ApprovalQueueElicitationBridge(queue=queue)
    task = asyncio.create_task(bridge.ask(_req(schema={"type": "object", "properties": {}})))
    row = await next_pending(queue, ACP_ELICITATION_NAMESPACE)
    assert row["params"]["questions"] == []
    queue.resolve(row["id"], True)
    response = await task
    assert response.action == "accept" and response.content == {}
    queue.close()


async def test_cancel_denies_pending_row(tmp_path) -> None:
    queue = make_queue(tmp_path)
    bridge = ApprovalQueueElicitationBridge(queue=queue)
    task = asyncio.create_task(bridge.ask(_req(schema=FORM)))
    row = await next_pending(queue, ACP_ELICITATION_NAMESPACE)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert queue.get(row["id"]).resolved is True
    queue.close()


# -- compiler limits ----------------------------------------------------------


def test_choice_limit_is_four() -> None:
    ok = {"type": "object", "properties": {"x": {"type": "string", "enum": list("abcd")}}}
    assert compile_form(ok)[0].type == "choice"
    too_many = {"type": "object", "properties": {"x": {"type": "string", "enum": list("abcde")}}}
    with pytest.raises(UnsupportedElicitationError):
        compile_form(too_many)


def test_titled_and_multi_select_choices() -> None:
    schema = {
        "type": "object",
        "properties": {
            "env": {
                "type": "string",
                "oneOf": [{"const": "prod", "title": "Production"}, {"const": "dev"}],
            },
            "tags": {
                "type": "array",
                "items": {"type": "string", "enum": ["a", "b"]},
                "maxItems": 2,
            },
        },
    }
    env, tags = compile_form(schema)
    assert env.choices == [
        {"value": "prod", "label": "Production"},
        {"value": "dev", "label": "dev"},
    ]
    assert decode_answer(env, "Production") == (True, "prod")
    assert decode_answer(tags, ["a", "b"]) == (True, ["a", "b"])
    with pytest.raises(ValueError):
        decode_answer(tags, ["c"])


def test_field_limit_and_bad_schemas() -> None:
    many = {"type": "object", "properties": {f"f{i}": {"type": "string"} for i in range(13)}}
    with pytest.raises(UnsupportedElicitationError):
        compile_form(many)
    with pytest.raises(UnsupportedElicitationError):
        compile_form({"type": "array"})
    with pytest.raises(UnsupportedElicitationError):
        compile_form({"type": "object", "properties": {"x": {"type": "object"}}})
    with pytest.raises(UnsupportedElicitationError):
        compile_form({"type": "object", "properties": {"x": {"type": "string"}}, "required": ["y"]})


def test_decode_number_bounds_and_secret_safe_errors() -> None:
    (count,) = compile_form(
        {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 3}}}
    )
    assert decode_answer(count, 2) == (True, 2)
    with pytest.raises(ValueError, match="maximum"):
        decode_answer(count, 9)
    with pytest.raises(ValueError, match="whole"):
        decode_answer(count, "1.5")
    (secret,) = compile_form(
        {
            "type": "object",
            "properties": {"pw": {"type": "string", "format": "password", "minLength": 20}},
        }
    )
    assert secret.secret is True
    with pytest.raises(ValueError) as exc:
        decode_answer(secret, "hunter2")
    assert "hunter2" not in str(exc.value)


def test_batching_rule() -> None:
    fields = compile_form(
        {
            "type": "object",
            "properties": {
                "s": {"type": "string", "format": "password"},
                **{k: {"type": "string"} for k in "abcd"},
            },
        }
    )
    assert [[f.field_id for f in b] for b in batch_fields(fields)] == [
        ["s"],
        ["a", "b", "c"],
        ["d"],
    ]


def test_validate_url_accepts_plain_https() -> None:
    assert validate_url("https://example.com/a?b=c") == "https://example.com/a?b=c"
