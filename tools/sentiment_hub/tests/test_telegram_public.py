from __future__ import annotations

from datetime import timedelta

import pytest
from conftest import FakeHttp, fixture_bytes

from sentiment_hub import telegram_public as tp
from sentiment_hub.assets import ChannelEntry
from sentiment_hub.net import HttpResponse, RateLimiter

PAGE1 = "https://t.me/s/cointelegraph"
PAGE2 = "https://t.me/s/cointelegraph?before=9001"


@pytest.mark.parametrize(
    "raw,expected",
    [("523", 523), ("1.2K", 1200), ("15.7K", 15700), ("12K", 12000), ("3.4M", 3_400_000), ("1,234", 1234),
     ("2,5K", 2500), ("1B", 1_000_000_000), ("", 0), (None, 0), ("n/a", 0)],
)
def test_parse_views(raw, expected):
    assert tp.parse_views(raw) == expected


def test_parse_preview_page():
    page = tp.parse_preview(fixture_bytes("tme_cointelegraph_page1.html").decode())
    assert page.is_preview
    assert page.before == 9001
    assert [p.post_id for p in page.posts] == [9001, 9002, 9003, 9004, 9005]
    by_id = {p.post_id: p for p in page.posts}

    first = by_id[9001]
    assert first.data_post == "cointelegraph/9001"
    assert first.views == 1200
    assert first.ts.isoformat() == "2026-10-19T09:12:00+00:00"
    # <b>, <br/>, emoji markup and links become plain text separated by spaces
    assert first.text.startswith("JUST IN: Bitcoin breaks $112,000 as ETF inflows hit a 3-month high 🚀")
    assert "bull run" in first.text
    assert "cointelegraph.com/news/bitcoin-112k-etf-inflows" in first.text
    assert "<" not in first.text and "  " not in first.text

    photo = by_id[9002]
    assert photo.text == ""  # photo without caption
    assert photo.views == 3_400_000

    reply = by_id[9003]
    assert reply.text.startswith("UPDATE: Solana is back online")
    assert "outage reported by validators" not in reply.text  # quoted reply text is not the message
    assert reply.views == 523

    assert by_id[9004].text.startswith("$ETH whales accumulated")
    assert by_id[9004].views == 15700
    assert by_id[9005].views == 987


def test_parse_non_preview_page():
    page = tp.parse_preview(fixture_bytes("tme_group_landing.html").decode())
    assert not page.is_preview
    assert page.posts == []


def _channel_http():
    http = FakeHttp()
    http.add(PAGE1, fixture_bytes("tme_cointelegraph_page1.html"))
    http.add(PAGE2, fixture_bytes("tme_cointelegraph_page2.html"))
    return http


def test_fetch_channel_paginates_until_window_covered(now):
    http = _channel_http()
    result = tp.fetch_channel(http, "cointelegraph", now, lookback_hours=48)
    assert http.calls == [PAGE1, PAGE2]  # page 2 reaches past 48 h -> stop
    assert result.status == "ok"
    assert result.pages == 2
    ids = [m.url.rsplit("/", 1)[1] for m in result.messages]
    # 9002 has no text, 8996 is older than 48 h
    assert ids == ["8997", "8998", "8999", "9000", "9001", "9003", "9004", "9005"]
    first = next(m for m in result.messages if m.url.endswith("/9001"))
    assert first.url == "https://t.me/cointelegraph/9001"
    assert first.key == "tg:cointelegraph/9001"
    assert first.community == "cointelegraph"
    assert first.engagement == 1200
    assert first.public and first.source == "telegram_public"


def test_fetch_channel_stops_after_one_page_when_window_is_short(now):
    http = _channel_http()
    result = tp.fetch_channel(http, "cointelegraph", now, lookback_hours=5)
    assert http.calls == [PAGE1]
    assert [m.url.rsplit("/", 1)[1] for m in result.messages] == ["9001", "9003"]


def test_fetch_channel_max_pages(now):
    """A very busy channel: every page is fresh, so pagination stops at max_pages."""

    def page(url: str) -> HttpResponse:
        before = int(url.split("before=")[1]) if "before=" in url else 10_000
        posts = "".join(
            f'<div class="tgme_widget_message" data-post="busy/{i}"><div class="tgme_widget_message_text js-message_text">msg {i}</div>'
            f'<a class="tgme_widget_message_date"><time datetime="{(now - timedelta(minutes=10_000 - i)).isoformat()}"></time></a></div>'
            for i in range(before - 20, before)
        )
        html = f'<section class="tgme_channel_history">{posts}</section>'
        return HttpResponse(200, url, html.encode())

    http = FakeHttp(default=page)
    result = tp.fetch_channel(http, "busy", now, lookback_hours=48, max_pages=5)
    assert len(http.calls) == 5
    assert result.pages == 5
    assert len(result.messages) == 100
    assert http.calls[1] == "https://t.me/s/busy?before=9980"


def test_group_redirect_is_skipped(now):
    http = FakeHttp()
    http.add("https://t.me/s/bitcoin_traders_chat", fixture_bytes("tme_group_landing.html"), final_url="https://t.me/bitcoin_traders_chat")
    result = tp.fetch_channel(http, "bitcoin_traders_chat", now)
    assert result.status == "no_preview"
    assert result.messages == []


def test_404_is_skipped(now):
    http = FakeHttp().add("https://t.me/s/gone_channel", b"Not Found", status=404)
    assert tp.fetch_channel(http, "gone_channel", now).status == "unavailable"


def test_429_is_retried_once(now, monkeypatch):
    monkeypatch.setattr(tp.time, "sleep", lambda s: None)
    http = FakeHttp()
    http.routes[PAGE1] = [
        HttpResponse(429, PAGE1, b"Too Many Requests", headers={"Retry-After": "2"}),
        HttpResponse(200, PAGE1, fixture_bytes("tme_cointelegraph_page1.html")),
    ]
    result = tp.fetch_channel(http, "cointelegraph", now, lookback_hours=5)
    assert result.status == "ok"
    assert http.calls[:2] == [PAGE1, PAGE1]


def test_collect_iterates_all_entries(now):
    http = _channel_http()
    http.add("https://t.me/s/bitcoin_traders_chat", fixture_bytes("tme_group_landing.html"), final_url="https://t.me/bitcoin_traders_chat")
    http.add("https://t.me/s/gone_channel", b"", status=404)
    # t.me/s/solana is not routed -> NetError -> unavailable
    entries = [
        ChannelEntry("cointelegraph", kind="channel", confidence=0.9),
        ChannelEntry("bitcoin_traders_chat", kind="group", confidence=0.6),
        ChannelEntry("@gone_channel", confidence=0.4),
        ChannelEntry("https://t.me/solana", confidence=0.8),
        ChannelEntry("CoinTelegraph", confidence=0.1),  # duplicate (case-insensitive)
    ]
    result = tp.collect(entries, http, now, lookback_hours=48, concurrency=4)
    assert result.name == "telegram_public"
    assert result.status == "ok"
    assert len(result.items) == 8
    assert result.detail == "1/3 channels (2 unavailable)"  # the group is left to the account source
    assert result.meta["covered_handles"] == {"cointelegraph"}
    assert result.meta["per_channel"] == {
        "cointelegraph": "ok",
        "gone_channel": "unavailable",
        "solana": "unavailable",
    }
    assert sorted(set(http.calls)) == sorted(
        {PAGE1, PAGE2, "https://t.me/s/gone_channel", "https://t.me/s/solana"}
    )


def test_collect_all_failing(now):
    result = tp.collect([ChannelEntry("cointelegraph"), ChannelEntry("solana")], FakeHttp(), now)
    assert result.status == "unavailable"
    assert result.items == []
    assert result.detail.startswith("0/2 channels")


def test_collect_time_budget(now, monkeypatch):
    http = _channel_http()
    result = tp.collect([ChannelEntry("cointelegraph")], http, now, budget_s=-1)
    assert http.calls == []
    assert result.meta["per_channel"] == {"cointelegraph": "skipped"}


def test_rate_limiter_spaces_requests():
    clock = [100.0]
    slept: list[float] = []

    def sleep(s):
        slept.append(s)
        clock[0] += s

    limiter = RateLimiter(1.0, clock=lambda: clock[0], sleep=sleep)
    limiter.wait()
    limiter.wait()
    clock[0] += 0.25
    limiter.wait()
    assert slept == [1.0, 0.75]
    limiter.penalize(10)
    limiter.wait()
    assert slept[-1] == pytest.approx(10.0)
