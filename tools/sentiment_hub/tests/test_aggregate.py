from __future__ import annotations

import math
from datetime import timedelta

import pytest

from sentiment_hub.aggregate import aggregate, message_weight, pick_samples
from sentiment_hub.assets import Asset
from sentiment_hub.common import Message, utf16_len
from sentiment_hub.mentions import build_matchers
from sentiment_hub.nlp import ScoredMessage, composite_score, label_for

ASSETS = [Asset("BTC", "Bitcoin"), Asset("ETH", "Ethereum"), Asset("SOL", "Solana"), Asset("DOGE", "Doge")]


def scored(text, now, hours_ago, engagement=0.0, source="telegram_public", public=True, i=None, **scores):
    i = i if i is not None else abs(hash((text, hours_ago, source))) % 10**8
    m = Message(
        source=source,
        key=f"{source}:{i}",
        text=text,
        ts=now - timedelta(hours=hours_ago),
        url=f"https://t.me/chan/{i}" if public else None,
        community="chan" if public else None,
        public=public,
        engagement=engagement,
    )
    c = composite_score(scores)
    return ScoredMessage(message=m, scores=scores, composite=c, label=label_for(c))


def test_weight_formula(now):
    assert message_weight(now, 0, now) == pytest.approx(1.0)
    assert message_weight(now - timedelta(hours=12), 0, now) == pytest.approx(0.5)  # half-life 12 h
    assert message_weight(now - timedelta(hours=24), 0, now) == pytest.approx(0.25)
    assert message_weight(now, 9, now) == pytest.approx(2.0)  # 1 + log10(1 + 9)
    assert message_weight(now - timedelta(hours=24), 99, now) == pytest.approx(0.25 * 3.0)
    assert message_weight(now, 1200, now) == pytest.approx(1 + math.log10(1201))
    assert message_weight(now, -50, now) == pytest.approx(1.0)  # negative engagement clamps to 0
    assert message_weight(now + timedelta(hours=1), 0, now) == pytest.approx(1.0)  # clock skew -> age 0


def test_aggregate_math(now):
    a = scored("Bitcoin rallies", now, 0, engagement=9, vader=0.8, finbert=0.6, cryptobert=0.4)  # w=2
    b = scored("BTC dumps hard", now, 12, engagement=0, vader=-0.6, finbert=-0.2, cryptobert=-0.8)  # w=0.5
    c = scored("Ethereum is flat", now, 0, engagement=0, source="rss", vader=0.0)  # w=1, vader only
    d = scored("no asset here", now, 1, vader=0.5)  # market only
    prev = [
        Message("rss", "p1", "Bitcoin yesterday", now - timedelta(hours=30)),
        Message("rss", "p2", "BTC and $ETH yesterday", now - timedelta(hours=40)),
        Message("rss", "p3", "Doge yesterday only", now - timedelta(hours=30)),
    ]
    agg = aggregate([a, b, c, d], prev, build_matchers(ASSETS), now, ["vader", "finbert", "cryptobert"])

    assert [x["symbol"] for x in agg.assets] == ["BTC", "ETH"]  # DOGE has only previous mentions
    btc, eth = agg.assets
    assert btc["mentions"] == 2 and btc["mentionsPrev"] == 2
    assert eth["mentions"] == 1 and eth["mentionsPrev"] == 1
    wa, wb = 2.0, 0.5
    assert btc["composite"] == pytest.approx((wa * a.composite + wb * b.composite) / (wa + wb))
    assert btc["scores"]["vader"] == pytest.approx((wa * 0.8 + wb * -0.6) / 2.5)
    assert btc["scores"]["cryptobert"] == pytest.approx((wa * 0.4 + wb * -0.8) / 2.5)
    assert a.label == "bullish" and b.label == "bearish"
    assert btc["bullish"] == pytest.approx(0.8) and btc["bearish"] == pytest.approx(0.2) and btc["neutral"] == 0
    assert btc["bySource"] == {"telegram_public": {"mentions": 2, "composite": pytest.approx(btc["composite"])}}
    assert eth["scores"] == {"vader": 0.0}  # models without scores for this asset are left out
    assert eth["neutral"] == pytest.approx(1.0)
    assert eth["bySource"]["rss"]["mentions"] == 1

    market = agg.market
    assert market["mentions"] == 4
    wd = 0.5 ** (1 / 12)
    total = wa + wb + 1.0 + wd
    assert market["composite"] == pytest.approx((wa * a.composite + wb * b.composite + 0.0 + wd * 0.5) / total)
    assert market["bullish"] + market["bearish"] + market["neutral"] == pytest.approx(1.0)
    assert market["bullish"] == pytest.approx((wa + wd) / total)


def test_assets_sorted_by_mentions(now):
    items = [scored("Solana", now, 0, i=1, vader=0.1)] + [scored("Bitcoin", now, h, i=10 + h, vader=0.1) for h in range(3)]
    items += [scored("Ethereum", now, h, i=20 + h, vader=0.1) for h in range(2)]
    agg = aggregate(items, [], build_matchers(ASSETS), now, ["vader"])
    assert [(x["symbol"], x["mentions"]) for x in agg.assets] == [("BTC", 3), ("ETH", 2), ("SOL", 1)]


def test_samples_public_only_limited_and_safe(now):
    long_text = "Bitcoin " + "🚀" * 200 + " end"
    items = [
        scored("Bitcoin private alpha: buy now", now, 0, engagement=10**6, source="telegram_account", public=False, i=1, vader=0.9),
        scored(long_text, now, 0, i=2, vader=0.7),
        scored("Bitcoin dup text", now, 0, i=3, vader=0.5),
        scored("bitcoin DUP text", now, 0, i=4, vader=0.5),
    ] + [scored(f"Bitcoin note {k}", now, k, i=10 + k, vader=0.2) for k in range(6)]
    agg = aggregate(items, [], build_matchers(ASSETS), now, ["vader"])
    (btc,) = agg.assets
    assert btc["mentions"] == 10  # the private message still counts
    assert btc["bySource"]["telegram_account"]["mentions"] == 1
    samples = btc["samples"]
    assert len(samples) == 5
    texts = [s["text"] for s in samples]
    assert not any("private alpha" in t for t in texts)
    assert sum("dup text" in t.lower() for t in texts) == 1
    for s in samples:
        assert utf16_len(s["text"]) <= 280 and len(s["text"]) <= 280
        assert s["url"].startswith("https://t.me/") and s["community"] == "chan" and s["ts"].endswith("Z")
        assert s["source"] == "telegram_public"
    assert any(t.endswith("…") for t in texts)  # the long one was cut


def test_pick_samples_strongest_first(now):
    weak = scored("Bitcoin meh", now, 0, i=1, vader=0.0)
    strong = scored("Bitcoin amazing", now, 0, i=2, vader=0.9)
    bear = scored("Bitcoin awful", now, 0, i=3, vader=-0.8)
    out = pick_samples([(weak, 1.0), (strong, 1.0), (bear, 1.0)], limit=2)
    assert [s["text"] for s in out] == ["Bitcoin amazing", "Bitcoin awful"]


def test_empty_input(now):
    agg = aggregate([], [], build_matchers(ASSETS), now, ["vader"])
    assert agg.assets == []
    assert agg.market == {"mentions": 0, "composite": 0.0, "bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
