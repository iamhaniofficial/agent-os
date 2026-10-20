"""FIFO lot accounting and portfolio aggregation. Pure, unit-tested.

Amounts are integers in the token's base units; money is float USD. A lot is
a quantity acquired at one cost **per raw unit** (``cost_usd_per_raw``), so
the math never needs the token's decimals until a number is displayed.
Selling consumes lots oldest-first and realises proceeds minus cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Context, Decimal, InvalidOperation


@dataclass
class Lot:
    lot_id: int | None
    amount_raw: int
    cost_usd_per_raw: float
    acquired_at: float

    @property
    def cost_usd(self) -> float:
        return max(0, self.amount_raw) * self.cost_usd_per_raw


@dataclass
class Consumption:
    lot_id: int | None
    amount_raw: int
    cost_usd: float


@dataclass
class SellResult:
    consumed: list[Consumption] = field(default_factory=list)
    cost_usd: float = 0.0
    proceeds_usd: float = 0.0
    # Sold more than the lots held (history gap on an imported wallet).
    unmatched_raw: int = 0

    @property
    def pnl_usd(self) -> float:
        return self.proceeds_usd - self.cost_usd


# Decimal arithmetic in the default context keeps 28 significant digits, and
# base-unit amounts routinely carry more: ten billion tokens at 18 decimals is
# already 29 digits, and a uint256 has up to 78. In the default context such an
# amount is rounded (sometimes up, past what the wallet holds), and quantize()
# raises InvalidOperation when the result needs more digits than it keeps. The
# helpers below size the context to the operands so these conversions are exact.


def _shift(value: Decimal, places: int) -> Decimal:
    """``value * 10**places``, exactly: scaleb() moves only the exponent."""
    return value.scaleb(places, Context(prec=max(1, len(value.as_tuple().digits))))


def _truncate(value: Decimal, places: int) -> Decimal:
    """``value`` cut (toward zero) to ``places`` decimal places, exactly."""
    digits = max(1, value.adjusted() + 1 + places)
    return value.quantize(
        Decimal(1).scaleb(-places), rounding=ROUND_DOWN, context=Context(prec=digits)
    )


def to_human(amount_raw: int, decimals: int) -> Decimal:
    if decimals <= 0:
        return Decimal(amount_raw)
    return _shift(Decimal(amount_raw), -decimals)


def to_raw(amount: str | Decimal | float | int, decimals: int) -> int:
    """Human amount to base units, rounding down.

    Anything that is not a finite, non-negative number is a ``ValueError`` —
    the one error the callers turn into "invalid input". ``Decimal`` would
    otherwise accept ``"NaN"`` and ``"Infinity"`` and blow up later, in the
    comparison or the ``int()``, with errors nobody maps.
    """
    try:
        value = Decimal(str(amount))
    except InvalidOperation as exc:
        raise ValueError(f"amount is not a number: {amount!r}") from exc
    if not value.is_finite():
        raise ValueError(f"amount is not a finite number: {amount!r}")
    if value < 0:
        raise ValueError("amount must be positive")
    scaled = _shift(value, decimals)
    return int(scaled.to_integral_value(rounding=ROUND_DOWN))


def format_amount(amount_raw: int, decimals: int, *, max_places: int = 8) -> str:
    """Human decimal string without exponent noise or trailing zeros."""
    human = to_human(amount_raw, decimals)
    places = min(decimals, max_places)
    if places > 0:
        human = _truncate(human, places)
    text = format(human, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def format_ratio(numerator: Decimal, denominator: Decimal, *, places: int = 12) -> str | None:
    """One unit of the denominator, priced in the numerator, as a decimal string.

    Every amount this API returns is a decimal string in human units. A rate is
    an amount like any other, and a float both broke that rule for its
    consumers and lost digits on tokens whose price sits many places below the
    decimal point. ``None`` when there is nothing to divide by.
    """
    if denominator <= 0:
        return None
    # Wide enough for every whole digit of the quotient plus ``places`` and a
    # guard digit, rounding down so the guard digit never carries upward.
    digits = max(28, numerator.adjusted() - denominator.adjusted() + places + 3)
    quotient = Context(prec=digits, rounding=ROUND_DOWN).divide(numerator, denominator)
    value = _truncate(quotient, places)
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def per_raw(cost_usd_per_token: float, decimals: int) -> float:
    """Cost per raw unit from a cost per whole token."""
    return float(cost_usd_per_token) / float(10**decimals)


def per_token(cost_usd_per_raw: float, decimals: int) -> float:
    return float(cost_usd_per_raw) * float(10**decimals)


def sell_fifo(lots: list[Lot], amount_raw: int, proceeds_usd: float | None) -> SellResult:
    """Consume ``amount_raw`` from ``lots`` (mutated in place, oldest first)."""
    result = SellResult()
    remaining = int(amount_raw)
    if remaining <= 0:
        return result
    total = remaining
    for lot in sorted(lots, key=lambda entry: entry.acquired_at):
        if remaining <= 0:
            break
        if lot.amount_raw <= 0:
            continue
        take = min(lot.amount_raw, remaining)
        cost = take * lot.cost_usd_per_raw
        lot.amount_raw -= take
        remaining -= take
        result.consumed.append(Consumption(lot.lot_id, take, cost))
        result.cost_usd += cost
    result.unmatched_raw = remaining
    if proceeds_usd is not None and total:
        matched = total - remaining
        result.proceeds_usd = proceeds_usd * (matched / total)
    return result


@dataclass
class HoldingPnl:
    amount_raw: int
    cost_usd: float
    realized_usd: float
    price_usd: float | None
    decimals: int

    @property
    def amount(self) -> float:
        return float(to_human(self.amount_raw, self.decimals))

    @property
    def value_usd(self) -> float | None:
        if self.price_usd is None:
            return None
        return self.amount * self.price_usd

    @property
    def avg_cost_usd(self) -> float | None:
        if self.amount_raw <= 0 or self.cost_usd <= 0:
            return None
        return self.cost_usd / self.amount if self.amount > 0 else None

    @property
    def unrealized_usd(self) -> float | None:
        """Value minus cost whenever there is a value; an airdrop at cost 0 is all gain."""
        value = self.value_usd
        if value is None:
            return None
        return value - self.cost_usd

    @property
    def unrealized_pct(self) -> float | None:
        """Gain over cost; undefined (``None``) when nothing was paid."""
        unrealized = self.unrealized_usd
        if unrealized is None or self.cost_usd <= 0:
            return None
        return unrealized / self.cost_usd * 100.0


def holding_from_lots(
    lots: list[Lot], *, decimals: int, price_usd: float | None, realized_usd: float
) -> HoldingPnl:
    return HoldingPnl(
        amount_raw=sum(max(0, lot.amount_raw) for lot in lots),
        cost_usd=sum(lot.cost_usd for lot in lots),
        realized_usd=realized_usd,
        price_usd=price_usd,
        decimals=decimals,
    )
