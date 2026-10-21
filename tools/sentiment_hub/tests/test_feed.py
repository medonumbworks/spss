from __future__ import annotations

import copy
import json
from datetime import timedelta

import pytest

from sentiment_hub.aggregate import aggregate
from sentiment_hub.assets import Asset
from sentiment_hub.common import Message, SourceResult
from sentiment_hub.feed import SCHEMA, build_feed, dumps, rnd, validate_feed, write_feed
from sentiment_hub.mentions import build_matchers
from sentiment_hub.nlp import ScoredMessage, composite_score, label_for

ASSETS = [Asset("BTC", "Bitcoin"), Asset("ETH", "Ethereum"), Asset("SOL", "Solana")]


def _scored(text, now, hours, source="telegram_public", public=True, i=0, **scores):
    m = Message(source, f"{source}:{i}", text, now - timedelta(hours=hours),
                url=f"https://t.me/c/{i}" if public else None, community="c" if public else None, public=public, engagement=100)
    c = composite_score(scores)
    return ScoredMessage(m, scores, c, label_for(c))


def sources(now):
    return [
        SourceResult("telegram_public", "ok", [Message("telegram_public", "k", "t", now)] * 3, "80/87 channels"),
        SourceResult("telegram_account", "disabled", detail="no TELEGRAM_SESSION secret"),
        SourceResult("rss", "ok", [Message("rss", "r", "t", now)], "6/6 feeds"),
        SourceResult("reddit", "unauthorized", detail="0/4 subreddits (4 HTTP 403)"),
    ]


@pytest.fixture
def sample_feed(now):
    items = [
        _scored("Bitcoin to new highs", now, 1, i=1, vader=0.712345, finbert=0.3, cryptobert=0.55555),
        _scored("BTC dumps", now, 3, i=2, vader=-0.5, finbert=-0.4, cryptobert=-0.6),
        _scored("Ethereum upgrade went fine", now, 2, source="rss", i=3, vader=0.3, finbert=0.1, cryptobert=0.0),
        _scored("Solana and Bitcoin in a private chat", now, 2, source="telegram_account", public=False, i=4, vader=0.2, finbert=0.2, cryptobert=0.2),
    ]
    agg = aggregate(items, [Message("rss", "p", "Bitcoin earlier", now - timedelta(hours=30))], build_matchers(ASSETS), now,
                    ["vader", "finbert", "cryptobert"])
    return build_feed(now, 24, ["vader", "finbert", "cryptobert"], sources(now), agg.market, agg.assets)


def test_feed_matches_contract(sample_feed):
    assert validate_feed(sample_feed) == []
    assert list(sample_feed) == ["schema", "generatedAt", "windowHours", "models", "sources", "market", "assets"]
    assert sample_feed["schema"] == SCHEMA == "cryptopulse.sentiment-feed/v1"
    assert sample_feed["generatedAt"] == "2026-10-19T10:00:00Z"
    assert sample_feed["windowHours"] == 24
    assert sample_feed["models"] == ["vader", "finbert", "cryptobert"]
    assert [s["name"] for s in sample_feed["sources"]] == ["telegram_public", "telegram_account", "rss", "reddit"]
    assert sample_feed["sources"][0] == {"name": "telegram_public", "status": "ok", "items": 3, "detail": "80/87 channels"}
    assert sample_feed["market"]["mentions"] == 4

    btc = sample_feed["assets"][0]
    assert list(btc) == ["symbol", "mentions", "mentionsPrev", "baseline7d", "spike", "composite", "scores", "bullish", "bearish", "neutral", "bySource", "samples"]
    assert btc["symbol"] == "BTC" and btc["mentions"] == 3 and btc["mentionsPrev"] == 1
    assert btc["baseline7d"] is None and btc["spike"] is None  # no mention history was attached
    assert list(btc["scores"]) == ["vader", "finbert", "cryptobert"]
    assert list(btc["bySource"]) == ["telegram_public", "telegram_account"]
    assert list(btc["samples"][0]) == ["text", "score", "source", "community", "url", "ts"]
    # the private chat contributes to the numbers but never to samples
    assert btc["bySource"]["telegram_account"]["mentions"] == 1
    assert all("private chat" not in s["text"] for a in sample_feed["assets"] for s in a["samples"])
    sol = next(a for a in sample_feed["assets"] if a["symbol"] == "SOL")
    assert sol["samples"] == []


def test_baseline_and_spike_in_the_feed(now):
    items = [
        _scored("Bitcoin to new highs", now, 1, i=1, vader=0.7),
        _scored("Bitcoin dumps", now, 2, i=2, vader=-0.5),
        _scored("Ethereum upgrade went fine", now, 2, source="rss", i=3, vader=0.3),
        _scored("Solana is back", now, 3, source="rss", i=4, vader=0.1),
    ]
    agg = aggregate(items, [], build_matchers(ASSETS), now, ["vader"])
    # what hub.run attaches from history.baselines: BTC has a baseline, ETH's is too small
    # for a meaningful ratio, SOL never appeared in the old entries
    for a in agg.assets:
        a["baseline7d"] = {"BTC": 4 / 3, "ETH": 0.5, "SOL": 0.0}[a["symbol"]]
    feed = build_feed(now, 24, ["vader"], sources(now), agg.market, agg.assets)
    assert validate_feed(feed) == []
    by_symbol = {a["symbol"]: a for a in feed["assets"]}
    assert by_symbol["BTC"]["baseline7d"] == 1.33 and by_symbol["BTC"]["spike"] == 1.5  # 2 / (4/3)
    assert by_symbol["ETH"]["baseline7d"] == 0.5 and by_symbol["ETH"]["spike"] is None  # baseline < 1
    assert by_symbol["SOL"]["baseline7d"] == 0.0 and by_symbol["SOL"]["spike"] is None


def test_numbers_rounded_to_4_decimals(sample_feed):
    def walk(x):
        if isinstance(x, float):
            assert x == round(x, 4)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(sample_feed)
    assert rnd(0.123456) == 0.1235 and rnd(-0.00001) == 0.0 and str(rnd(-0.00001)) == "0.0"
    assert rnd(float("nan")) == 0.0


def test_json_roundtrip_and_atomic_write(sample_feed, tmp_path):
    out = write_feed(sample_feed, tmp_path / "sub" / "feed.json")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == sample_feed
    assert validate_feed(loaded) == []
    assert [p.name for p in out.parent.iterdir()] == ["feed.json"]  # no temp files left
    assert "NaN" not in dumps(sample_feed)


def test_empty_feed_is_valid(now):
    failed = [SourceResult(n, "unavailable", detail="blocked") for n in ("telegram_public", "rss", "reddit")]
    failed.insert(1, SourceResult("telegram_account", "disabled", detail="no TELEGRAM_SESSION secret"))
    feed = build_feed(now, 24, [], failed, {"mentions": 0, "composite": 0.0, "bullish": 0.0, "bearish": 0.0, "neutral": 0.0}, [])
    assert validate_feed(feed) == []
    assert feed["assets"] == [] and feed["models"] == []


def test_missing_sources_are_reported_as_disabled(now):
    feed = build_feed(now, 24, ["vader"], [], {"mentions": 0, "composite": 0, "bullish": 0, "bearish": 0, "neutral": 0}, [])
    assert [s["status"] for s in feed["sources"]] == ["disabled"] * 4
    assert validate_feed(feed) == []


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda f: f.update(schema="v2"), "schema"),
        (lambda f: f.pop("market"), "missing"),
        (lambda f: f.update(extra=1), "unexpected"),
        (lambda f: f["sources"][0].update(status="broken"), "status"),
        (lambda f: f["sources"].pop(), "sources"),
        (lambda f: f["models"].append("gpt"), "models"),
        (lambda f: f.update(generatedAt="2026-10-19 10:00"), "generatedAt"),
        (lambda f: f["assets"].reverse(), "sorted"),
        (lambda f: f["assets"][0].update(composite=1.5), "outside"),
        (lambda f: f["assets"][0].update(composite=0.123456), "decimals"),
        (lambda f: f["assets"][0].update(baseline7d="2.5"), "baseline7d"),
        (lambda f: f["assets"][0].update(spike=-0.5), "spike"),
        (lambda f: f["assets"][0].update(spike=1.234), "2 decimals"),
        (lambda f: f["assets"][0].pop("baseline7d"), "missing"),
        (lambda f: f["assets"][0].update(bullish=0.9), "shares"),
        (lambda f: f["assets"][0].update(mentions=0), "mentions"),
        (lambda f: f["assets"][0]["samples"][0].update(text="x" * 281), "text"),
        (lambda f: f["assets"][0]["samples"][0].update(url=None), "url"),
        (lambda f: f["assets"][0]["samples"].extend([f["assets"][0]["samples"][0]] * 5), "at most"),
        (lambda f: f["assets"][0]["scores"].update(gpt=0.1), "scores"),
        (lambda f: f["assets"][0]["bySource"]["telegram_public"].update(mentions=99), "sum"),
        (lambda f: f["assets"].append(copy.deepcopy(f["assets"][-1])), "duplicate"),
        (lambda f: f["market"].update(mentions=True), "market.mentions"),
        (lambda f: f["assets"][0]["samples"][0].update(url="/news/relative"), "http(s) URL"),
        (lambda f: f.update(schema=f.pop("schema")), "out of order"),
        (lambda f: f["assets"][0].update(symbol=f["assets"][0].pop("symbol")), "out of order"),
        (lambda f: f["assets"][0].update(symbol="币安人生"), "symbol"),
        (lambda f: f["assets"][0].update(symbol="BTC\n"), "symbol"),
    ],
)
def test_validator_rejects(sample_feed, mutate, expected):
    feed = copy.deepcopy(sample_feed)
    mutate(feed)
    errors = validate_feed(feed)
    assert errors, "mutation should be rejected"
    assert any(expected in e for e in errors), errors
