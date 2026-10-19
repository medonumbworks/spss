"""End-to-end runs of the hub without network: everything failing, and a full run on fixtures."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import timedelta

from conftest import FakeHttp, NOW, REPO_ROOT, fixture_bytes, fixture_path

from sentiment_hub import hub, reddit, rss
from sentiment_hub.common import Message, SourceResult
from sentiment_hub.feed import validate_feed
from sentiment_hub.net import NetError

HUB = REPO_ROOT / "tools" / "sentiment_hub" / "hub.py"


def args(*extra: str, out="feed.json"):
    return hub.build_parser().parse_args(
        ["--out", str(out), "--assets", str(fixture_path("assets_small.json")), "--channels", str(fixture_path("channels_small.json")),
         "--lexicon", str(fixture_path("lexicon_small.tsv")), *extra]
    )


class DownHttp:
    """Every request fails like a blocked host."""

    def __init__(self):
        self.calls = []

    def limiter_for(self, url):
        return None

    def get(self, url, headers=None, allow_redirects=True):
        self.calls.append(url)
        raise NetError("ProxyError: blocked")


def test_everything_down_still_writes_a_valid_feed(tmp_path, monkeypatch):
    monkeypatch.setattr(reddit.time, "sleep", lambda s: None)
    tg, news = DownHttp(), DownHttp()
    feed, results, notes = hub.run(args("--models", "vader,finbert,cryptobert"), now=NOW, tg_http=tg, news_http=news, account_env={})
    assert validate_feed(feed) == []
    assert feed["assets"] == []
    assert feed["models"] == []  # nothing was scored, so no scorer ran
    assert feed["market"] == {"mentions": 0, "composite": 0.0, "bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
    statuses = {s["name"]: (s["status"], s["items"]) for s in feed["sources"]}
    assert statuses == {
        "telegram_public": ("unavailable", 0),
        "telegram_account": ("disabled", 0),
        "rss": ("unavailable", 0),
        "reddit": ("unavailable", 0),
    }
    assert feed["sources"][1]["detail"] == "no TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_SESSION secrets"
    assert len(tg.calls) == 4  # every valid catalogue handle was tried (cointelegraph, solana, gone_channel, group)
    assert len(news.calls) == len(rss.FEEDS) + len(reddit.SUBREDDITS)


def test_one_source_crashing_does_not_fail_the_run(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("unexpected bug")

    monkeypatch.setattr(hub.rss, "collect", boom)
    feed, results, _ = hub.run(args("--no-telegram", "--no-reddit"), now=NOW, news_http=DownHttp())
    assert results["rss"].status == "error" and results["rss"].detail == "RuntimeError"
    assert validate_feed(feed) == []


def _fixture_http():
    tg = FakeHttp()
    tg.add("https://t.me/s/cointelegraph", fixture_bytes("tme_cointelegraph_page1.html"))
    tg.add("https://t.me/s/cointelegraph?before=9001", fixture_bytes("tme_cointelegraph_page2.html"))
    tg.add("https://t.me/s/bitcoin_traders_chat", fixture_bytes("tme_group_landing.html"), final_url="https://t.me/bitcoin_traders_chat")
    tg.add("https://t.me/s/gone_channel", b"", status=404)
    news = FakeHttp()
    news.add(rss.FEEDS[0][1], fixture_bytes("rss_cointelegraph.xml"))
    news.add(rss.FEEDS[2][1], fixture_bytes("atom_decrypt.xml"))
    news.add(reddit.listing_url("CryptoCurrency"), fixture_bytes("reddit_cryptocurrency_new.json"))
    for sub in reddit.SUBREDDITS[1:]:
        news.add(reddit.listing_url(sub), b"", status=403)
    return tg, news


def fake_pipeline_factory(task, model=None, tokenizer=None, device=None, **kwargs):
    """Stands in for transformers.pipeline: no model download, deterministic scores."""
    table = {
        "ProsusAI/finbert": [{"label": "positive", "score": 0.5}, {"label": "negative", "score": 0.2}, {"label": "neutral", "score": 0.3}],
        "ElKulako/cryptobert": [{"label": "Bearish", "score": 0.1}, {"label": "Neutral", "score": 0.3}, {"label": "Bullish", "score": 0.6}],
    }[model]

    def pipe(inputs, **kw):
        assert kw["batch_size"] == 32 and kw["max_length"] == 128 and kw["truncation"] is True
        return [list(table) for _ in inputs]

    return pipe


def test_full_run_on_fixtures(monkeypatch, tmp_path):
    monkeypatch.setattr(reddit.time, "sleep", lambda s: None)
    tg, news = _fixture_http()
    feed, results, notes = hub.run(args(), now=NOW, tg_http=tg, news_http=news, account_env={}, pipeline_factory=fake_pipeline_factory)
    assert validate_feed(feed) == [], validate_feed(feed)
    assert feed["models"] == ["vader", "finbert", "cryptobert"]
    src = {s["name"]: s for s in feed["sources"]}
    assert src["telegram_public"]["status"] == "ok" and src["telegram_public"]["detail"].startswith("1/4 channels")
    assert src["telegram_public"]["items"] == 8
    assert src["rss"]["status"] == "ok" and src["reddit"]["status"] == "ok"
    assert src["reddit"]["detail"] == "1/4 subreddits (3 HTTP 403)"
    # current window: 5 Telegram posts, 4 headlines, 2 Reddit posts
    assert feed["market"]["mentions"] == 11
    mentions = [a["mentions"] for a in feed["assets"]]
    assert mentions == sorted(mentions, reverse=True)
    by_symbol = {a["symbol"]: a for a in feed["assets"]}
    assert {"BTC", "ETH", "SOL", "DOGE"} <= set(by_symbol)
    btc = by_symbol["BTC"]
    assert btc["mentions"] == 3
    assert btc["mentionsPrev"] == 1  # "BTC dominance climbs" is 32 h old; the 53 h old post is outside the lookback
    assert set(btc["scores"]) == {"vader", "finbert", "cryptobert"}
    assert btc["scores"]["finbert"] == 0.3 and btc["scores"]["cryptobert"] == 0.5
    assert set(btc["bySource"]) == {"telegram_public", "rss"}
    assert all(s["url"].startswith("https://") for a in feed["assets"] for s in a["samples"])
    sol = by_symbol["SOL"]
    assert any("Solana is back online" in s["text"] for s in sol["samples"])
    out = tmp_path / "feed.json"
    from sentiment_hub.feed import write_feed

    write_feed(feed, out)
    assert json.loads(out.read_text())["schema"] == "cryptopulse.sentiment-feed/v1"


def test_public_post_seen_by_both_telegram_sources_counts_once():
    ts = NOW - timedelta(hours=1)
    pub = SourceResult("telegram_public", "ok", [Message("telegram_public", "tg:cointelegraph/1", "Bitcoin up", ts, "https://t.me/cointelegraph/1", "cointelegraph")])
    acc = SourceResult("telegram_account", "ok", [
        Message("telegram_account", "tg:cointelegraph/1", "Bitcoin up", ts, "https://t.me/cointelegraph/1", "cointelegraph"),
        Message("telegram_account", "tgp:77/5", "spam", ts, None, None, public=False),
        Message("telegram_account", "tgp:77/6", "spam", ts, None, None, public=False),  # same text, same private chat
        Message("telegram_account", "tgp:78/6", "spam", ts, None, None, public=False),  # other private chat: kept
    ])
    merged, dropped = hub.merge_messages({"telegram_public": pub, "telegram_account": acc}, NOW, 48)
    assert [m.key for m in merged] == ["tg:cointelegraph/1", "tgp:77/5", "tgp:78/6"]
    assert dropped == 2


def test_split_windows():
    msgs = [Message("rss", str(h), "x", NOW - timedelta(hours=h)) for h in (0, 23.9, 24, 30, 47.9, 48)]
    current, previous = hub.split_windows(msgs, NOW, 24)
    assert [m.key for m in current] == ["0", "23.9"]
    assert [m.key for m in previous] == ["24", "30", "47.9"]


def test_cli_with_all_sources_disabled(tmp_path):
    """Real subprocess, no network needed: proves the script entry point and exit code."""
    out = tmp_path / "feed.json"
    proc = subprocess.run(
        [sys.executable, str(HUB), "--out", str(out), "--no-telegram", "--no-rss", "--no-reddit",
         "--assets", str(fixture_path("assets_small.json")), "--models", "vader"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    feed = json.loads(out.read_text(encoding="utf-8"))
    assert validate_feed(feed) == []
    assert [s["status"] for s in feed["sources"]] == ["disabled"] * 4
    last = proc.stderr.strip().splitlines()[-1]
    assert last.startswith("sentiment-hub ") and "telegram_public disabled" in last


def test_cli_rejects_unknown_model(tmp_path):
    proc = subprocess.run([sys.executable, str(HUB), "--out", str(tmp_path / "f.json"), "--models", "gpt"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2 and "unknown model" in proc.stderr
