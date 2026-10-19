from __future__ import annotations

from conftest import FakeHttp, fixture_bytes

from sentiment_hub import rss


def test_parse_rss2():
    items = rss.parse_feed(fixture_bytes("rss_cointelegraph.xml"))
    assert [i.title for i in items] == [
        "Bitcoin price rallies as ETF inflows surge to record",
        "Solana DEX volume drops 30% as memecoin mania cools",
        "Three-day-old story about Cardano governance",
        "Undated sponsored post",
    ]
    first = items[0]
    assert first.link == "https://cointelegraph.com/news/bitcoin-price-rallies-etf-inflows"
    # HTML stripped, entities decoded, whitespace collapsed
    assert first.summary == "Spot Bitcoin ETFs saw $1.2 billion of net inflows on Friday, the strongest day since March."
    assert first.ts.isoformat() == "2026-10-19T08:45:00+00:00"
    assert items[1].summary == "Traders are rotating out of SOL memecoins & into stablecoins."
    assert items[1].ts.isoformat() == "2026-10-18T21:10:00+00:00"
    assert items[3].ts is None


def test_parse_atom():
    items = rss.parse_feed(fixture_bytes("atom_decrypt.xml"))
    assert len(items) == 2
    eth, doge = items
    assert eth.title == "Ethereum staking hits new high"
    assert eth.link == "https://decrypt.co/300001/ethereum-staking-new-high"  # rel=alternate, not the enclosure
    assert eth.summary == "More than 35M ETH is now staked, a bullish sign for the network."
    assert eth.ts.isoformat() == "2026-10-19T07:00:00+00:00"  # published wins over updated
    assert doge.link == "https://decrypt.co/300002/dogecoin-volatility"
    assert doge.summary == "Doge options open interest doubled overnight."
    assert doge.ts.isoformat() == "2026-10-18T21:15:00+00:00"  # converted to UTC
    assert doge.guid == "tag:decrypt.co,2026:post-2"


def test_parse_rss1_rdf():
    xml = b"""<?xml version="1.0"?>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns="http://purl.org/rss/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <channel rdf:about="https://example.org/"><title>Example</title></channel>
      <item rdf:about="https://example.org/a"><title>Cardano upgrade ships</title><link>https://example.org/a</link>
        <dc:date>2026-10-19T06:00:00Z</dc:date><description>Smooth hard fork.</description></item>
    </rdf:RDF>"""
    (item,) = rss.parse_feed(xml)
    assert item.title == "Cardano upgrade ships"
    assert item.link == "https://example.org/a"
    assert item.ts.isoformat() == "2026-10-19T06:00:00+00:00"


def test_parse_html_entities_outside_cdata():
    (item,) = rss.parse_feed(fixture_bytes("rss_broken_entities.xml"))
    assert item.title == "XRP & Stellar rally — AT&T-style bare ampersand"
    assert item.summary == "Ripple’s XRP gained 12% … analysts say R&D matters."


def test_item_text_joins_title_and_summary():
    item = rss.FeedItem("Bitcoin rallies", "https://x", "ETF demand is strong.", None, None)
    assert rss.item_text(item) == "Bitcoin rallies. ETF demand is strong."
    item = rss.FeedItem("Is Bitcoin back?", "https://x", "Maybe.", None, None)
    assert rss.item_text(item) == "Is Bitcoin back? Maybe."
    assert rss.item_text(rss.FeedItem("Only a title", None, "", None, None)) == "Only a title"


def test_collect_mixed_results(now):
    feeds = (
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
        ("Decrypt", "https://decrypt.co/feed"),
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("The Block", "https://www.theblock.co/rss.xml"),
        ("CryptoSlate", "https://cryptoslate.com/feed/"),
        ("Bitcoin Magazine", "https://bitcoinmagazine.com/.rss/full/"),
    )
    http = FakeHttp()
    http.add(feeds[0][1], fixture_bytes("rss_cointelegraph.xml"))
    http.add(feeds[1][1], fixture_bytes("atom_decrypt.xml"))
    http.add(feeds[2][1], b"<html>Access denied</html>", status=403)
    http.add(feeds[3][1], b"<rss><channel><item><title>broken", status=200)
    http.add(feeds[4][1], fixture_bytes("rss_broken_entities.xml"))
    # Bitcoin Magazine not routed -> network error
    result = rss.collect(http, now, lookback_hours=48, feeds=feeds)
    assert result.status == "ok"
    titles = [m.text.split(".")[0] for m in result.items]
    assert len(result.items) == 5  # 2 CoinTelegraph (one too old, one undated) + 2 Decrypt + 1 CryptoSlate
    assert "Three-day-old story about Cardano governance" not in " ".join(titles)
    assert result.detail == "3/6 feeds (1 error, 1 unauthorized, 1 unavailable); 1 undated items skipped"
    first = result.items[0]
    assert first.source == "rss" and first.community == "CoinTelegraph" and first.public
    assert first.url == "https://cointelegraph.com/news/bitcoin-price-rallies-etf-inflows"
    assert first.text.startswith("Bitcoin price rallies as ETF inflows surge to record. Spot Bitcoin ETFs saw")


def test_collect_all_down(now):
    result = rss.collect(FakeHttp(), now)
    assert result.status == "unavailable"
    assert result.items == []
    assert result.detail.startswith("0/6 feeds")
    assert len(rss.FEEDS) == 6


def test_links_are_made_absolute_and_non_http_links_dropped(now):
    xml = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>
    <item><title>Relative link</title><link>/news/relative</link><pubDate>Mon, 19 Oct 2026 09:00:00 GMT</pubDate></item>
    <item><title>Protocol-relative link</title><link>//cdn.example.org/a</link><pubDate>Mon, 19 Oct 2026 09:01:00 GMT</pubDate></item>
    <item><title>Tag id as link</title><link>tag:example.org,2026:7</link><pubDate>Mon, 19 Oct 2026 09:02:00 GMT</pubDate></item>
    <item><title>Upper-case scheme</title><link>HTTPS://Example.org/Up</link><pubDate>Mon, 19 Oct 2026 09:03:00 GMT</pubDate></item>
    </channel></rss>"""
    feeds = (("Example", "https://example.org/feed/rss"),)
    http = FakeHttp()
    http.add(feeds[0][1], xml)
    result = rss.collect(http, now, feeds=feeds)
    assert [m.url for m in result.items] == [
        "https://example.org/news/relative",
        "https://cdn.example.org/a",
        None,
        "https://Example.org/Up",
    ]
    assert rss.absolute_link("  ", feeds[0][1]) is None
    assert rss.absolute_link("javascript:alert(1)", feeds[0][1]) is None
