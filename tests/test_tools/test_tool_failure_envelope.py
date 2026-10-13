from __future__ import annotations

import json

import httpx
import pytest

from agentos.engine.types import ToolCall
from agentos.tools.dispatch import build_tool_handler
from agentos.tools.envelope import build_tool_failure_envelope
from agentos.tools.registry import ToolRegistry
from agentos.tools.types import (
    CallerKind,
    SafeToolError,
    ToolContext,
    ToolError,
    ToolSpec,
    current_tool_context,
)


def test_type_error_tool_failure_is_model_retriable_without_raw_traceback() -> None:
    envelope = build_tool_failure_envelope(
        TypeError("secret raw argument detail"),
        "write_file",
    )

    assert envelope["status"] == "error"
    assert envelope["tool"] == "write_file"
    assert envelope["error_class"] == "TypeError"
    assert envelope["retry_allowed"] is True
    assert "tool schema" in envelope["user_message"]
    assert "secret raw argument detail" not in envelope["user_message"]


def test_httpx_timeout_exception_is_retriable_with_specific_message() -> None:
    envelope = build_tool_failure_envelope(
        httpx.TimeoutException("secret endpoint"),
        "web_fetch",
    )

    assert envelope["error_class"] == "TimeoutException"
    assert envelope["retry_allowed"] is True
    assert "timed out" in envelope["user_message"]
    assert "secret endpoint" not in envelope["user_message"]


def test_httpx_connect_error_is_retriable_with_specific_message() -> None:
    envelope = build_tool_failure_envelope(
        httpx.ConnectError("secret host"),
        "web_fetch",
    )

    assert envelope["error_class"] == "ConnectError"
    assert envelope["retry_allowed"] is True
    assert "could not connect" in envelope["user_message"]
    assert "secret host" not in envelope["user_message"]


def test_json_decode_error_has_specific_message() -> None:
    envelope = build_tool_failure_envelope(
        json.JSONDecodeError("secret payload", "not-json", 0),
        "web_fetch",
    )

    assert envelope["error_class"] == "JSONDecodeError"
    assert envelope["retry_allowed"] is False
    assert "invalid response payload" in envelope["user_message"]
    assert "secret payload" not in envelope["user_message"]


def test_policy_denial_envelope_has_exactly_five_keys() -> None:
    envelope = build_tool_failure_envelope(
        PermissionError("raw denied detail"),
        "exec_command",
        policy_denial=True,
        error_class_override="PolicyDenied",
        user_message_override="Blocked by policy.",
    )

    assert set(envelope) == {
        "status",
        "tool",
        "error_class",
        "user_message",
        "retry_allowed",
    }
    assert envelope["status"] == "error"
    assert envelope["tool"] == "exec_command"
    assert envelope["error_class"] == "PolicyDenied"
    assert envelope["user_message"] == "Blocked by policy."
    assert envelope["retry_allowed"] is False


def test_safe_tool_error_instance_message_preserves_five_key_shape() -> None:
    envelope = build_tool_failure_envelope(
        SafeToolError("PDF file not found: input.pdf (resolved=/workspace/input.pdf)", "secret"),
        "pdf",
    )

    assert set(envelope) == {
        "status",
        "tool",
        "error_class",
        "user_message",
        "retry_allowed",
    }
    assert envelope["status"] == "error"
    assert envelope["tool"] == "pdf"
    assert envelope["error_class"] == "SafeToolError"
    assert "PDF file not found" in envelope["user_message"]
    assert "secret" not in envelope["user_message"]


def test_image_attachment_path_safe_tool_error_is_not_generic_internal_error() -> None:
    envelope = build_tool_failure_envelope(
        SafeToolError(
            "Image path is not accessible by the image tool: ab367.png. "
            "Pass a real local file path or HTTP(S) URL. If this is a chat attachment, "
            "answer from the attached image directly instead of calling the image tool.",
            "resolved=/secret/workspace/ab367.png",
        ),
        "image",
    )

    assert set(envelope) == {
        "status",
        "tool",
        "error_class",
        "user_message",
        "retry_allowed",
    }
    assert envelope["tool"] == "image"
    assert envelope["error_class"] == "SafeToolError"
    assert envelope["retry_allowed"] is False
    assert "not accessible by the image tool" in envelope["user_message"]
    assert "chat attachment" in envelope["user_message"]
    assert "internal error" not in envelope["user_message"]
    assert "secret" not in envelope["user_message"]


# ── ToolError messages reach the model (#3642) ──────────────────────────────


def test_tool_error_message_reaches_the_model() -> None:
    envelope = build_tool_failure_envelope(ToolError("file not found: /tmp/x"), "read_file")

    assert set(envelope) == {"status", "tool", "error_class", "user_message", "retry_allowed"}
    assert envelope["error_class"] == "ToolError"
    assert envelope["user_message"] == "file not found: /tmp/x"
    assert envelope["retry_allowed"] is False


@pytest.mark.parametrize(
    ("raw", "secret"),
    [
        ("upstream rejected key sk-ant-api03-abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrst"),
        ("request failed: Bearer abc123def456ghi789jkl0", "abc123def456ghi789jkl0"),
        ("request failed: Authorization: Bearer abc123def456ghi789jkl0", "def456ghi789"),
        ("cannot reach postgres://agent:hunter2-pass@db.internal/app", "hunter2-pass"),
        ("bad config OPENAI_API_KEY=Zq81mXw0pLr7Tk2vBn5c", "Zq81mXw0pLr7Tk2vBn5c"),
    ],
)
def test_tool_error_message_is_redacted(raw: str, secret: str) -> None:
    envelope = build_tool_failure_envelope(ToolError(raw), "http_request")

    assert secret not in envelope["user_message"]
    assert "internal error" not in envelope["user_message"]
    # The non-secret part of the message survives so the model can act on it.
    assert envelope["user_message"].split()[0] == raw.split()[0]


def test_bearer_in_prose_is_not_masked() -> None:
    envelope = build_tool_failure_envelope(
        ToolError("Bearer authentication is not supported by this endpoint"), "web_fetch"
    )

    assert envelope["user_message"] == "Bearer authentication is not supported by this endpoint"


def test_plain_runtime_error_still_renders_the_generic_line() -> None:
    envelope = build_tool_failure_envelope(RuntimeError("secret operator detail"), "exec")

    assert envelope["error_class"] == "RuntimeError"
    assert envelope["user_message"] == "The tool 'exec' failed with an internal error."
    assert "secret" not in envelope["user_message"]


def test_empty_tool_error_falls_back_to_the_generic_line() -> None:
    envelope = build_tool_failure_envelope(ToolError(), "exec")

    assert envelope["user_message"] == "The tool 'exec' failed with an internal error."


def test_safe_tool_error_still_renders_only_its_user_message() -> None:
    envelope = build_tool_failure_envelope(
        SafeToolError("schedule must be an object", "raw secret detail"), "cron"
    )

    assert envelope["error_class"] == "SafeToolError"
    assert envelope["user_message"] == "schedule must be an object"


def test_tool_error_message_is_length_capped() -> None:
    envelope = build_tool_failure_envelope(ToolError("x" * 10_000), "read_file")

    assert len(envelope["user_message"]) == 2000
    assert envelope["user_message"].endswith("...[truncated]")


def test_tool_error_message_newlines_are_collapsed() -> None:
    envelope = build_tool_failure_envelope(
        ToolError("unknown command 'foo'.\nTry one of: list, add"), "skill"
    )

    assert envelope["user_message"] == "unknown command 'foo'. Try one of: list, add"


@pytest.mark.asyncio
async def test_dispatched_tool_error_reaches_the_tool_result() -> None:
    registry = ToolRegistry()

    async def _handler() -> str:
        raise ToolError("path outside workspace: ../etc/passwd")

    registry.register(ToolSpec(name="probe_tool", description="probe", parameters={}), _handler)
    handler = build_tool_handler(registry)
    token = current_tool_context.set(
        ToolContext(caller_kind=CallerKind.CLI, channel_kind="cli", channel_id="cli:test")
    )
    try:
        result = await handler(ToolCall(tool_use_id="tc-1", tool_name="probe_tool", arguments={}))
    finally:
        current_tool_context.reset(token)

    envelope = json.loads(result.content)
    assert result.is_error is True
    assert envelope["error_class"] == "ToolError"
    assert envelope["user_message"] == "path outside workspace: ../etc/passwd"
