"""Read a provider's "come back in N seconds" hint off an HTTP error response.

A 429 that says ``Retry-After: 2`` is backpressure, not an outage: the right
move is to wait two seconds and ask the same model again. Without the hint the
runtime can only guess, and its guess was to fail over to another model (losing
the prompt cache and changing the cost) and count the 429 toward opening the
provider's circuit breaker.

The parsers here are deliberately forgiving: a header that cannot be read is
treated as absent, never as an error, so a malformed hint degrades to the old
blind-backoff behaviour instead of breaking a turn.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

#: Longest wait any hint is allowed to ask for. Anything past this is clamped;
#: the engine applies its own (smaller) cap before deciding to wait at all.
MAX_RETRY_AFTER_SECONDS = 300.0

#: Default for ``AgentConfig.rate_limit_retry_after_cap_seconds``: a hint at or
#: under this many seconds is waited out on the same provider instead of
#: failing over, and does not count toward that provider's circuit breaker.
DEFAULT_RETRY_AFTER_CAP_SECONDS = 30.0

# OpenAI's ``x-ratelimit-reset-*`` durations: "1s", "6m0s", "20ms", "1h2m3.5s".
_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DURATION_UNITS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}

# (reset header, remaining header) pairs. A reset only matters for a limit that
# is actually spent; when the remaining header is absent we take the reset.
_OPENAI_RESET_HEADERS = (
    ("x-ratelimit-reset-requests", "x-ratelimit-remaining-requests"),
    ("x-ratelimit-reset-tokens", "x-ratelimit-remaining-tokens"),
)
_ANTHROPIC_RESET_HEADERS = (
    ("anthropic-ratelimit-requests-reset", "anthropic-ratelimit-requests-remaining"),
    ("anthropic-ratelimit-tokens-reset", "anthropic-ratelimit-tokens-remaining"),
    ("anthropic-ratelimit-input-tokens-reset", "anthropic-ratelimit-input-tokens-remaining"),
    ("anthropic-ratelimit-output-tokens-reset", "anthropic-ratelimit-output-tokens-remaining"),
)


def _clamp(seconds: float) -> float:
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup that works for ``httpx.Headers`` and dicts."""
    value = headers.get(name)
    if value is None:
        lowered = name.lower()
        for key, candidate in headers.items():
            if key.lower() == lowered:
                value = candidate
                break
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _now(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(UTC)


def _seconds_until(moment: datetime, now: datetime | None) -> float:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (moment - _now(now)).total_seconds()


def parse_retry_after_value(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse one ``Retry-After`` value (delta-seconds or an HTTP-date).

    Returns seconds clamped to ``[0, MAX_RETRY_AFTER_SECONDS]``, or ``None``
    when *value* is missing or unreadable.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        pass
    else:
        if seconds != seconds:  # NaN
            return None
        return _clamp(seconds)
    try:
        moment = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if moment is None:
        return None
    return _clamp(_seconds_until(moment, now))


def parse_duration(value: str | None) -> float | None:
    """Parse an OpenAI-style duration (``6m0s``, ``20ms``, ``1.5s``) to seconds."""
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        return None
    try:
        return _clamp(float(text))
    except ValueError:
        pass
    position = 0
    total = 0.0
    for match in _DURATION_PART_RE.finditer(text):
        if match.start() != position:
            return None
        total += float(match.group(1)) * _DURATION_UNITS[match.group(2)]
        position = match.end()
    if position == 0 or position != len(text):
        return None
    return _clamp(total)


def _parse_timestamp(value: str | None, *, now: datetime | None) -> float | None:
    """Parse an RFC 3339 reset timestamp (Anthropic) to seconds from *now*."""
    if value is None:
        return None
    try:
        moment = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return _clamp(_seconds_until(moment, now))


def _spent(headers: Mapping[str, str], remaining_header: str) -> bool:
    remaining = _header(headers, remaining_header)
    if remaining is None:
        return True
    try:
        return float(remaining) <= 0
    except ValueError:
        return True


def _longest_reset(
    headers: Mapping[str, str],
    pairs: tuple[tuple[str, str], ...],
    parse: Callable[[str | None], float | None],
) -> float | None:
    resets = [
        seconds
        for reset_header, remaining_header in pairs
        if _spent(headers, remaining_header)
        and (seconds := parse(_header(headers, reset_header))) is not None
    ]
    return max(resets) if resets else None


def parse_retry_after(
    headers: Mapping[str, str] | None,
    *,
    now: datetime | None = None,
) -> float | None:
    """Return the provider's retry hint in seconds, or ``None`` when it gave none.

    Precedence: ``retry-after-ms``, then ``Retry-After`` (delta-seconds or
    HTTP-date), then the vendor reset headers for whichever limit is spent —
    OpenAI's ``x-ratelimit-reset-*`` durations and Anthropic's
    ``anthropic-ratelimit-*-reset`` timestamps. The result is clamped to
    ``[0, MAX_RETRY_AFTER_SECONDS]``.
    """
    if not headers:
        return None
    try:
        millis = _header(headers, "retry-after-ms")
        if millis is not None:
            try:
                return _clamp(float(millis) / 1000.0)
            except ValueError:
                pass
        hinted = parse_retry_after_value(_header(headers, "retry-after"), now=now)
        if hinted is not None:
            return hinted
        openai_reset = _longest_reset(headers, _OPENAI_RESET_HEADERS, parse_duration)
        if openai_reset is not None:
            return openai_reset
        return _longest_reset(
            headers,
            _ANTHROPIC_RESET_HEADERS,
            lambda value: _parse_timestamp(value, now=now),
        )
    except Exception:  # noqa: BLE001 - a bad hint must never break the error path
        return None


def retry_after_from_response(response: Any) -> float | None:
    """:func:`parse_retry_after` on ``response.headers``, tolerating stand-ins.

    Test doubles and some transports hand back response objects with no
    ``headers`` attribute; those simply carry no hint.
    """
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    return parse_retry_after(headers)


__all__ = [
    "DEFAULT_RETRY_AFTER_CAP_SECONDS",
    "MAX_RETRY_AFTER_SECONDS",
    "parse_duration",
    "parse_retry_after",
    "parse_retry_after_value",
    "retry_after_from_response",
]
