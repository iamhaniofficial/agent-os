"""EstimateCalibration: per-(provider, model) EMA, clamped, bounded."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentos.engine import Agent, AgentConfig
from agentos.engine.estimate_calibration import (
    MAX_FACTOR,
    MIN_FACTOR,
    EstimateCalibration,
    request_has_media,
)
from agentos.provider import DoneEvent as ProviderDone
from agentos.provider import Message
from agentos.provider import TextDeltaEvent as ProviderText
from agentos.provider.types import (
    ContentBlockImage,
    ContentBlockText,
    ContentBlockToolResult,
)


def test_factor_is_one_until_two_observations() -> None:
    calibration = EstimateCalibration()
    assert calibration.factor("openrouter", "m") == 1.0
    assert calibration.observe(
        "openrouter", "m", estimated_tokens=10_000, provider_input_tokens=8_000
    )
    assert calibration.factor("openrouter", "m") == 1.0
    calibration.observe("openrouter", "m", estimated_tokens=10_000, provider_input_tokens=8_000)
    assert calibration.factor("openrouter", "m") == pytest.approx(0.8)


def test_ema_moves_towards_new_samples() -> None:
    calibration = EstimateCalibration(alpha=0.5)
    calibration.observe("p", "m", estimated_tokens=10_000, provider_input_tokens=8_000)
    calibration.observe("p", "m", estimated_tokens=10_000, provider_input_tokens=10_000)
    assert calibration.factor("p", "m") == pytest.approx(0.9)


def test_factor_is_clamped() -> None:
    low = EstimateCalibration()
    high = EstimateCalibration()
    for _ in range(3):
        low.observe("p", "m", estimated_tokens=10_000, provider_input_tokens=3_000)
        high.observe("p", "m", estimated_tokens=10_000, provider_input_tokens=20_000)
    assert low.factor("p", "m") == MIN_FACTOR
    assert high.factor("p", "m") == MAX_FACTOR
    # A 0.5 floor let one bad stretch halve the estimate; 0.7 is the floor now.
    assert MIN_FACTOR == 0.7


@pytest.mark.parametrize(
    ("estimated", "reported"),
    [
        (10_000, 0),  # provider reports no usage (some Ollama builds)
        (100, 60),  # too small: fixed overhead dominates
        (100_000, 1_000),  # outlier: images counted as base64 text
        (1_000, 10_000),  # outlier: the provider saw a different request
    ],
)
def test_unusable_samples_are_ignored(estimated: int, reported: int) -> None:
    calibration = EstimateCalibration()
    assert not calibration.observe(
        "p", "m", estimated_tokens=estimated, provider_input_tokens=reported
    )
    assert calibration.observations("p", "m") == 0


def test_keys_are_per_provider_and_model_and_bounded() -> None:
    calibration = EstimateCalibration(max_keys=2)
    for model in ("a", "b", "c"):
        for _ in range(2):
            calibration.observe("p", model, estimated_tokens=10_000, provider_input_tokens=8_000)
    assert len(calibration) == 2
    assert calibration.factor("p", "a") == 1.0  # evicted, least recently used
    assert calibration.factor("P", "C") == pytest.approx(0.8)  # keys are case-insensitive
    assert calibration.factor("other", "c") == 1.0


def test_estimated_usage_and_media_requests_are_not_samples() -> None:
    calibration = EstimateCalibration()
    assert not calibration.observe(
        "opencap", "m", estimated_tokens=10_000, provider_input_tokens=8_000, usage_estimated=True
    )
    assert not calibration.observe(
        "opencap", "m", estimated_tokens=10_000, provider_input_tokens=8_000, has_media=True
    )
    assert calibration.observations("opencap", "m") == 0


def _image() -> ContentBlockImage:
    return ContentBlockImage(media_type="image/png", data="aGVsbG8=")


def test_request_has_media_finds_images_anywhere_in_the_request() -> None:
    text_only = [Message(role="user", content="hi"), Message(role="assistant", content="ok")]
    assert not request_has_media(text_only)
    assert request_has_media(
        [Message(role="user", content=[ContentBlockText(text="look"), _image()])]
    )
    in_tool_result = Message(
        role="user",
        content=[ContentBlockToolResult(tool_use_id="t1", content=[_image()])],
    )
    assert request_has_media([*text_only, in_tool_result])
    assert request_has_media([{"role": "user", "content": [{"type": "document"}]}])
    assert not request_has_media(
        [Message(role="user", content=[ContentBlockToolResult(tool_use_id="t", content="x")])]
    )


class _UsageProvider:
    """Answers every call; reports a fixed input count, optionally estimated."""

    provider_name = "calibration-test"

    def __init__(self, *, input_tokens: int, usage_estimated: bool) -> None:
        self.input_tokens = input_tokens
        self.usage_estimated = usage_estimated

    def chat(self, messages: Any, tools: Any = None, config: Any = None) -> AsyncIterator[Any]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Any]:
        yield ProviderText(text="ok")
        yield ProviderDone(
            stop_reason="stop",
            input_tokens=self.input_tokens,
            output_tokens=1,
            usage_estimated=self.usage_estimated,
        )

    async def list_models(self) -> list[Any]:
        return []


async def _observations_after_turn(
    *, usage_estimated: bool, extra_messages: list[Message] | None = None
) -> int:
    calibration = EstimateCalibration()
    provider = _UsageProvider(input_tokens=2_000, usage_estimated=usage_estimated)
    agent = Agent(provider=provider, config=AgentConfig(model_id="m"))
    agent.estimate_calibration = calibration
    _ = [event async for event in agent.run_turn("x" * 8_000, extra_messages=extra_messages)]
    return calibration.observations("calibration-test", "m")


@pytest.mark.asyncio
async def test_agent_feeds_real_text_only_usage_to_the_calibration() -> None:
    assert await _observations_after_turn(usage_estimated=False) == 1


@pytest.mark.asyncio
async def test_agent_skips_estimated_usage() -> None:
    assert await _observations_after_turn(usage_estimated=True) == 0


@pytest.mark.asyncio
async def test_agent_skips_requests_that_carry_an_image() -> None:
    # Current-turn uploads ride in extra_messages (history images are stripped).
    upload = [Message(role="user", content=[ContentBlockText(text="what is this?"), _image()])]
    assert await _observations_after_turn(usage_estimated=False, extra_messages=upload) == 0
