"""Memecoins beyond Bitpanda and cashtag discovery."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from conftest import FakeHttp

from sentiment_hub import memecoins
from sentiment_hub.assets import Asset
from sentiment_hub.common import Message
from sentiment_hub.mentions import MentionMatcher

NOW = datetime(2026, 10, 19, 12, tzinfo=timezone.utc)

ROWS = [
    {"id": "dogecoin", "symbol": "doge", "name": "Dogecoin", "market_cap_rank": 9},
    {"id": "shiba-inu", "symbol": "shib", "name": "Shiba Inu", "market_cap_rank": 30},
    {"id": "dogwifcoin", "symbol": "wif", "name": "dogwifhat", "market_cap_rank": 90},
    {"id": "usd1-meme", "symbol": "usd1", "name": "Not a meme"},
    {"symbol": "noid"},
]


def msg(text: str, community: str, n: int = 0) -> Message:
    return Message(source="telegram_public", key=f"tg:{community}/{n}", text=text, ts=NOW, community=community, public=True)


def test_fetch_memecoins_parses_and_skips_non_tokens():
    http = FakeHttp().add(memecoins.MEME_URL, json.dumps(ROWS))
    assets, note = memecoins.fetch_memecoins(http)
    assert [a.symbol for a in assets] == ["DOGE", "SHIB", "WIF"]
    assert assets[1].coingecko_id == "shiba-inu" and assets[1].rank == 30
    assert note == "3 memecoins"


def test_fetch_memecoins_retries_one_429_then_gives_up_quietly():
    sleeps: list[float] = []
    http = FakeHttp()
    http.routes[memecoins.MEME_URL] = [
        FakeHttp().add("x", "", status=429).routes["x"],
        FakeHttp().add("x", json.dumps(ROWS[:1])).routes["x"],
    ]
    assets, _ = memecoins.fetch_memecoins(http, sleep=sleeps.append)
    assert [a.symbol for a in assets] == ["DOGE"] and sleeps == [memecoins.RETRY_AFTER_429_S]
    down, note = memecoins.fetch_memecoins(FakeHttp())
    assert down == [] and note.startswith("unavailable")
    refused, note = memecoins.fetch_memecoins(FakeHttp().add(memecoins.MEME_URL, "", status=500))
    assert refused == [] and note == "HTTP 500"


def test_bitpanda_wins_a_symbol_and_an_id():
    base = [Asset("DOGE", "Doge", "dogecoin"), Asset("PEPE", "Pepe", "pepe")]
    extra = [Asset("DOGE", "Dogecoin", "dogecoin"), Asset("PEPE2", "Pepe again", "pepe"), Asset("SHIB", "Shiba Inu", "shiba-inu")]
    assert [a.symbol for a in memecoins.merge_assets(base, extra)] == ["DOGE", "PEPE", "SHIB"]


def test_discover_cashtags_needs_several_communities():
    messages = [
        msg("$MOODY is sending, $MOODY again", "alpha", 1),
        msg("aped into $MOODY", "beta", 2),
        msg("$MOODY looks heavy", "beta", 3),
        msg("$LONELY only here", "alpha", 4),
        msg("$LONELY again", "alpha", 5),
        msg("$LONELY and more", "alpha", 6),
        msg("$BTC to $100k, $USDT on the side, $SOL $SOL", "gamma", 7),
        msg("$BTC $SOL", "delta", 8),
        msg("$BTC $SOL", "epsilon", 9),
    ]
    found = memecoins.discover_cashtags(messages, known={"BTC"})
    assert found == ["MOODY"], "one community, quote assets, known coins and prices are not new tokens"


def test_cashtag_only_matcher_ignores_words():
    m = MentionMatcher(Asset("MOODY", "MOODY"), cashtag_only=True)
    assert m.matches("$moody breaks out")
    assert not m.matches("MOODY breaks out")
    assert not m.matches("feeling moody today")
