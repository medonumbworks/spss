package dev.cryptopulse.core.sources

import dev.cryptopulse.core.sources.feed.SentimentFeedApi
import dev.cryptopulse.core.sources.feed.telegramBuzz
import dev.cryptopulse.core.testing.Fixtures
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class BuzzTest {

    @Test
    fun fixtureFeedRanksSpikingAssetFirstAndSkipsZeroMentionAssets() {
        val buzz = SentimentFeedApi.parse(Fixtures.read("sentiment_feed.json")).telegramBuzz()
        // SOL spikes (2.67x its 7-day baseline) so it outranks BTC's larger raw mention count;
        // TON has the buzz fields but zero mentions, so it never makes the list.
        assertEquals(listOf("SOL", "BTC"), buzz.map { it.symbol })
        val sol = buzz.first()
        assertEquals(2.67, sol.spike)
        assertEquals(45.0, sol.baseline7d)
        assertEquals(60, sol.mentionsPrev)
        assertEquals(1, sol.samples.size, "the fixture's SOL sample has a url, so it is kept")
        assertNull(buzz[1].spike, "old-feed asset falls back to null for the UI's prev-window display")
    }

    @Test
    fun ordersBySpikeDescNullsLastThenMentionsThenSymbol() {
        val doc = SentimentFeedApi.parse(
            """{"schema":"cryptopulse.sentiment-feed/v1","generatedAt":"2026-10-18T00:00:00Z","windowHours":24,"models":["vader"],"sources":[],"assets":[
            {"symbol":"AAA","mentions":10,"mentionsPrev":4,"baseline7d":8.0,"spike":1.25,"composite":0.1,"samples":[]},
            {"symbol":"ZZZ","mentions":5,"mentionsPrev":1,"baseline7d":1.0,"spike":5.0,"composite":0.2,"samples":[]},
            {"symbol":"CCC","mentions":50,"mentionsPrev":40,"composite":0.0,"samples":[]},
            {"symbol":"BBB","mentions":50,"mentionsPrev":45,"composite":0.0,"samples":[]},
            {"symbol":"DDD","mentions":80,"mentionsPrev":70,"composite":0.0,"samples":[]}]}""",
        )
        // ZZZ's spike beats AAA's despite fewer mentions; the null-spike rest rank by mentions,
        // and the BBB/CCC tie breaks alphabetically so the order is stable between refreshes.
        assertEquals(listOf("ZZZ", "AAA", "DDD", "BBB", "CCC"), doc.telegramBuzz().map { it.symbol })
        assertEquals(listOf("ZZZ", "AAA"), doc.telegramBuzz(limit = 2).map { it.symbol })
    }

    @Test
    fun keepsOnlySamplesThatLinkToAPost() {
        val doc = SentimentFeedApi.parse(
            """{"schema":"cryptopulse.sentiment-feed/v1","generatedAt":"2026-10-18T00:00:00Z","windowHours":24,"models":["vader"],"sources":[],"assets":[
            {"symbol":"SOL","mentions":3,"mentionsPrev":1,"composite":0.4,"samples":[
              {"text":"Solana rallies","score":0.6,"source":"telegram_public","community":"cointelegraph","url":"https://t.me/cointelegraph/51001","ts":"2026-10-17T13:00:00Z"},
              {"text":"sol looking strong","score":0.3,"source":"telegram_account"}]}]}""",
        )
        val samples = doc.telegramBuzz().single().samples
        assertEquals(listOf("https://t.me/cointelegraph/51001"), samples.map { it.url })
    }
}
