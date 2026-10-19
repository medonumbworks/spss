"""Per-asset and market aggregation of scored messages.

* Messages younger than ``window_hours`` are *current*; those ``window_hours``..``2*window_hours``
  old only count as ``mentionsPrev``.
* Weight of a message = recency x engagement, where recency halves every 12 h and engagement is
  ``1 + log10(1 + views_or_score)``.
* ``composite``, per-model ``scores`` and the bullish/bearish/neutral shares are weighted.
* ``mentions`` counts messages (unweighted).
* Samples (max 5 per asset) come only from public sources and are display-safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

from sentiment_hub.common import Message, is_http_url, iso_z, truncate_display
from sentiment_hub.mentions import MentionMatcher
from sentiment_hub.nlp import ScoredMessage

HALF_LIFE_HOURS = 12.0
# Paid "trending" channels sell their slots: their posts still count as mentions but barely move sentiment.
PROMO_WEIGHT = 0.2
PROMO_CATEGORIES = frozenset({"dex_trending"})
MAX_SAMPLES = 5
SAMPLE_CHARS = 280


def message_weight(ts: datetime, engagement: float, now: datetime) -> float:
    age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
    recency = 0.5 ** (age_hours / HALF_LIFE_HOURS)
    return recency * (1.0 + math.log10(1.0 + max(0.0, float(engagement or 0.0))))


@dataclass
class _Acc:
    count: int = 0
    weight: float = 0.0
    weighted_composite: float = 0.0
    bullish: float = 0.0
    bearish: float = 0.0
    neutral: float = 0.0
    model_weight: dict[str, float] = field(default_factory=dict)
    model_sum: dict[str, float] = field(default_factory=dict)

    def add(self, sm: ScoredMessage, w: float) -> None:
        self.count += 1
        self.weight += w
        self.weighted_composite += w * sm.composite
        if sm.label == "bullish":
            self.bullish += w
        elif sm.label == "bearish":
            self.bearish += w
        else:
            self.neutral += w
        for name, value in sm.scores.items():
            self.model_weight[name] = self.model_weight.get(name, 0.0) + w
            self.model_sum[name] = self.model_sum.get(name, 0.0) + w * value

    def composite(self) -> float:
        return self.weighted_composite / self.weight if self.weight > 0 else 0.0

    def shares(self) -> tuple[float, float, float]:
        if self.weight <= 0:
            return 0.0, 0.0, 0.0
        return self.bullish / self.weight, self.bearish / self.weight, self.neutral / self.weight

    def model_scores(self, models: Sequence[str]) -> dict[str, float]:
        return {
            m: self.model_sum[m] / self.model_weight[m]
            for m in models
            if self.model_weight.get(m, 0.0) > 0
        }


@dataclass
class Aggregation:
    market: dict
    assets: list[dict]


def _sample(sm: ScoredMessage) -> dict:
    m = sm.message
    return {
        "text": truncate_display(m.text, SAMPLE_CHARS),
        "score": sm.composite,
        "source": m.source,
        "community": m.community or "",
        "url": m.url or "",
        "ts": iso_z(m.ts),
    }


def pick_samples(items: Iterable[tuple[ScoredMessage, float]], limit: int = MAX_SAMPLES) -> list[dict]:
    """Strongest public messages: weight x (0.25 + |composite|), one per distinct text."""
    candidates = [
        (sm, w)
        for sm, w in items
        if sm.message.public and is_http_url(sm.message.url) and sm.message.community and sm.message.text.strip()
    ]
    candidates.sort(key=lambda x: (-(x[1] * (0.25 + abs(x[0].composite))), x[0].message.key))
    out: list[dict] = []
    seen: set[str] = set()
    for sm, _ in candidates:
        key = sm.message.text.strip().lower()[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(_sample(sm))
        if len(out) >= limit:
            break
    return out


def aggregate(
    scored: Sequence[ScoredMessage],
    previous: Sequence[Message],
    matchers: Sequence[MentionMatcher],
    now: datetime,
    models: Sequence[str],
    promo_communities: frozenset[str] = frozenset(),
) -> Aggregation:
    """``promo_communities``: lower-cased handles of paid-promotion channels (``PROMO_WEIGHT``)."""
    market = _Acc()
    per_asset: dict[str, _Acc] = {}
    per_asset_source: dict[str, dict[str, _Acc]] = {}
    per_asset_items: dict[str, list[tuple[ScoredMessage, float]]] = {}

    for sm in scored:
        w = message_weight(sm.message.ts, sm.message.engagement, now)
        if sm.message.community and sm.message.community.lower() in promo_communities:
            w *= PROMO_WEIGHT
        market.add(sm, w)
        folded = sm.message.text.casefold()
        for matcher in matchers:
            if not matcher.matches(sm.message.text, folded):
                continue
            sym = matcher.symbol
            per_asset.setdefault(sym, _Acc()).add(sm, w)
            per_asset_source.setdefault(sym, {}).setdefault(sm.message.source, _Acc()).add(sm, w)
            per_asset_items.setdefault(sym, []).append((sm, w))

    prev_counts: dict[str, int] = {}
    active = [m for m in matchers if m.symbol in per_asset]
    for msg in previous:
        folded = msg.text.casefold()
        for matcher in active:
            if matcher.matches(msg.text, folded):
                prev_counts[matcher.symbol] = prev_counts.get(matcher.symbol, 0) + 1

    assets: list[dict] = []
    for sym, acc in per_asset.items():
        if acc.count <= 0:
            continue
        bull, bear, neu = acc.shares()
        assets.append(
            {
                "symbol": sym,
                "mentions": acc.count,
                "mentionsPrev": prev_counts.get(sym, 0),
                "composite": acc.composite(),
                "scores": acc.model_scores(models),
                "bullish": bull,
                "bearish": bear,
                "neutral": neu,
                "bySource": {
                    src: {"mentions": sacc.count, "composite": sacc.composite()}
                    for src, sacc in per_asset_source[sym].items()
                    if sacc.count > 0
                },
                "samples": pick_samples(per_asset_items.get(sym, [])),
                "_weight": acc.weight,
            }
        )
    assets.sort(key=lambda a: (-a["mentions"], -a["_weight"], a["symbol"]))
    for a in assets:
        del a["_weight"]

    bull, bear, neu = market.shares()
    market_out = {
        "mentions": market.count,
        "composite": market.composite(),
        "bullish": bull,
        "bearish": bear,
        "neutral": neu,
    }
    return Aggregation(market=market_out, assets=assets)
