"""Calibrate the request-size estimate against what providers report.

The compaction trigger runs on ``len(json.dumps(request)) // 4``, which
over-counts by 1.6-2x against the input tokens providers bill for (JSON
quoting and escaping, base64-free tool schemas, tokenizers that pack more
than four chars per token). Compaction then fires far below the real limit
and drops context for nothing.

:class:`EstimateCalibration` keeps, per ``(provider, model)``, an
exponential moving average of ``provider_input_tokens / our_estimate`` from
the usage of every provider call. The factor it hands back is clamped to
``[MIN_FACTOR, MAX_FACTOR]`` and only applies after ``MIN_OBSERVATIONS``
samples; until then it is exactly ``1.0`` (uncalibrated, the old behaviour).
The store is in memory, owned by the runtime and bounded in size.

Samples that do not measure the estimator are skipped: usage the provider
marked ``"estimated": true`` (OpenCAP's trailing chunk) is the gateway's own
guess, and a request carrying image or document blocks counts their base64
text in the estimate while the provider bills them by size or tiles, which
drags the ratio down for every later text-only request.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

MIN_FACTOR = 0.7
MAX_FACTOR = 1.25
MIN_OBSERVATIONS = 2
EMA_ALPHA = 0.3
# A single observation outside this band is not a tokenizer difference but a
# request the estimate does not describe (an image counted as its base64
# text, a provider that rewrote the payload); it is ignored.
_MIN_SANE_RATIO = 0.2
_MAX_SANE_RATIO = 3.0
# Below this size the fixed per-request overhead dominates the ratio.
_MIN_ESTIMATE_TOKENS = 256
DEFAULT_MAX_KEYS = 64
# Content blocks whose payload is base64 the estimate counts as text.
_MEDIA_BLOCK_TYPES = frozenset({"image", "document"})


def _block_type(block: Any) -> Any:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _blocks_have_media(blocks: Iterable[Any]) -> bool:
    for block in blocks:
        kind = _block_type(block)
        if kind in _MEDIA_BLOCK_TYPES or kind == "image_url":
            return True
        if kind == "tool_result":
            nested = block.get("content") if isinstance(block, dict) else block.content
            if isinstance(nested, list) and _blocks_have_media(nested):
                return True
    return False


def request_has_media(messages: Iterable[Any]) -> bool:
    """True when any message carries an image or document block (also in tool results)."""
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else message.content
        if isinstance(content, list) and _blocks_have_media(content):
            return True
    return False


@dataclass
class _Entry:
    ratio: float
    observations: int


class EstimateCalibration:
    """Bounded per-(provider, model) EMA of provider tokens / estimated tokens."""

    def __init__(self, *, max_keys: int = DEFAULT_MAX_KEYS, alpha: float = EMA_ALPHA) -> None:
        self._entries: OrderedDict[tuple[str, str], _Entry] = OrderedDict()
        self._max_keys = max(1, max_keys)
        self._alpha = min(1.0, max(0.01, alpha))

    @staticmethod
    def key(provider: str | None, model: str | None) -> tuple[str, str]:
        return ((provider or "").strip().lower(), (model or "").strip().lower())

    def observe(
        self,
        provider: str | None,
        model: str | None,
        *,
        estimated_tokens: int,
        provider_input_tokens: int,
        usage_estimated: bool = False,
        has_media: bool = False,
    ) -> bool:
        """Record one provider call; ``False`` when the sample was ignored.

        *usage_estimated*: the provider marked its usage as an estimate.
        *has_media*: the request carried image or document blocks.
        """
        if usage_estimated or has_media:
            return False
        if estimated_tokens < _MIN_ESTIMATE_TOKENS or provider_input_tokens <= 0:
            return False
        ratio = provider_input_tokens / estimated_tokens
        if not _MIN_SANE_RATIO <= ratio <= _MAX_SANE_RATIO:
            return False
        key = self.key(provider, model)
        entry = self._entries.get(key)
        if entry is None:
            entry = _Entry(ratio=ratio, observations=1)
            self._entries[key] = entry
        else:
            entry.ratio = entry.ratio + self._alpha * (ratio - entry.ratio)
            entry.observations += 1
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_keys:
            self._entries.popitem(last=False)
        return True

    def factor(self, provider: str | None, model: str | None) -> float:
        """The multiplier for an estimate of this provider/model (1.0 = none)."""
        entry = self._entries.get(self.key(provider, model))
        if entry is None or entry.observations < MIN_OBSERVATIONS:
            return 1.0
        return min(MAX_FACTOR, max(MIN_FACTOR, entry.ratio))

    def observations(self, provider: str | None, model: str | None) -> int:
        entry = self._entries.get(self.key(provider, model))
        return entry.observations if entry is not None else 0

    def __len__(self) -> int:
        return len(self._entries)
