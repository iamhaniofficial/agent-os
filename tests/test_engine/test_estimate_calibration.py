"""EstimateCalibration: per-(provider, model) EMA, clamped, bounded."""

from __future__ import annotations

import pytest

from agentos.engine.estimate_calibration import (
    MAX_FACTOR,
    MIN_FACTOR,
    EstimateCalibration,
)


def test_factor_is_one_until_two_observations() -> None:
    calibration = EstimateCalibration()
    assert calibration.factor("openrouter", "m") == 1.0
    assert calibration.observe(
        "openrouter", "m", estimated_tokens=10_000, provider_input_tokens=6_000
    )
    assert calibration.factor("openrouter", "m") == 1.0
    calibration.observe("openrouter", "m", estimated_tokens=10_000, provider_input_tokens=6_000)
    assert calibration.factor("openrouter", "m") == pytest.approx(0.6)


def test_ema_moves_towards_new_samples() -> None:
    calibration = EstimateCalibration(alpha=0.5)
    calibration.observe("p", "m", estimated_tokens=10_000, provider_input_tokens=6_000)
    calibration.observe("p", "m", estimated_tokens=10_000, provider_input_tokens=8_000)
    assert calibration.factor("p", "m") == pytest.approx(0.7)


def test_factor_is_clamped() -> None:
    low = EstimateCalibration()
    high = EstimateCalibration()
    for _ in range(3):
        low.observe("p", "m", estimated_tokens=10_000, provider_input_tokens=3_000)
        high.observe("p", "m", estimated_tokens=10_000, provider_input_tokens=20_000)
    assert low.factor("p", "m") == MIN_FACTOR
    assert high.factor("p", "m") == MAX_FACTOR


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
            calibration.observe("p", model, estimated_tokens=10_000, provider_input_tokens=6_000)
    assert len(calibration) == 2
    assert calibration.factor("p", "a") == 1.0  # evicted, least recently used
    assert calibration.factor("P", "C") == pytest.approx(0.6)  # keys are case-insensitive
    assert calibration.factor("other", "c") == 1.0
