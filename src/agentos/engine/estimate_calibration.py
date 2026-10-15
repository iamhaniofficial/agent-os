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
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

MIN_FACTOR = 0.5
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
    ) -> bool:
        """Record one provider call; ``False`` when the sample was ignored."""
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
