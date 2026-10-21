from __future__ import annotations

import json

from conftest import FakeHttp, fixture_bytes

from sentiment_hub import reddit


def test_parse_listing(now):
    payload = json.loads(fixture_bytes("reddit_cryptocurrency_new.json"))
    posts = reddit.parse_listing(payload, now, lookback_hours=48)
    assert [p.key for p in posts] == ["reddit:1abc002", "reddit:1abc003"]  # stickied and old posts skipped
    eth, sol = posts
    assert eth.text == "ETH looks incredibly strong here, staking demand is insane Not financial advice but I'm buying more Ethereum."
    assert eth.url == "https://www.reddit.com/r/CryptoCurrency/comments/1abc002/eth_looks_incredibly_strong/"
    assert eth.community == "r/CryptoCurrency"
    assert eth.engagement == 99
    assert sol.text == "Got rekt on a SOL long, this market is brutal"  # [removed] body dropped
    assert sol.engagement == 0  # negative score clamped


def test_collect_records_blocked_status(now):
    subs = ("CryptoCurrency", "CryptoMarkets", "altcoin", "SatoshiStreetBets")
    http = FakeHttp()
    http.add(reddit.listing_url("CryptoCurrency"), fixture_bytes("reddit_cryptocurrency_new.json"))
    http.add(reddit.listing_url("CryptoMarkets"), b"<html>blocked</html>", status=403)
    http.add(reddit.listing_url("altcoin"), b"{}", status=429)
    http.add(reddit.listing_url("SatoshiStreetBets"), b"<html>not json</html>", status=200)
    result = reddit.collect(http, now, subreddits=subs, pause_s=0)
    assert result.status == "ok"
    assert len(result.items) == 2
    assert result.detail == "1/4 subreddits (1 HTTP 403, 1 HTTP 429, 1 not JSON)"
    assert http.calls[0] == "https://www.reddit.com/r/CryptoCurrency/new.json?limit=100&raw_json=1"


def test_collect_all_blocked(now):
    http = FakeHttp(default=None)
    for sub in reddit.SUBREDDITS:
        http.add(reddit.listing_url(sub), b"", status=403)
    result = reddit.collect(http, now, pause_s=0)
    assert result.status == "unauthorized"
    assert result.items == []
    http = FakeHttp()
    for sub in reddit.SUBREDDITS:
        http.add(reddit.listing_url(sub), b"", status=429)
    assert reddit.collect(http, now, pause_s=0).status == "rate_limited"
