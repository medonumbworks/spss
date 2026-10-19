package dev.cryptopulse.core.sources

import dev.cryptopulse.core.FIXTURE_NOW_MS
import dev.cryptopulse.core.http.RateLimiter
import dev.cryptopulse.core.sources.polymarket.PolymarketApi
import dev.cryptopulse.core.testHttp
import dev.cryptopulse.core.testing.FakeTransport
import dev.cryptopulse.core.testing.Fixtures
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Polymarket gamma parsers, against events captured live (tools/api_samples). */
class PolymarketTest {
    // The Polymarket capture ran on 2026-10-19 ~17:50Z, so the up-or-down tests pin "now" to that
    // day for the 48h window to contain its markets (its dates parse to epochs after FIXTURE_NOW_MS).
    private val captureNowMs = 1_792_431_000_000L // 2026-10-19T17:30:00Z

    private val markets = PolymarketApi.parseEvents(Fixtures.read("spec/polymarket_events.json"))

    @Test
    fun marketsAreParsedFromStringifiedArraysAndRankedByVolume() {
        assertEquals(11, markets.size)
        val btc150k = markets.first { it.slug == "will-bitcoin-reach-150k-in-october-2026" }
        assertEquals("Will Bitcoin reach \$150,000 in October?", btc150k.question)
        assertEquals("Yes", btc150k.outcome, "first entry of the stringified outcomes array")
        assertEquals(0.001, btc150k.probability, "first entry of the stringified outcomePrices array")
        assertEquals("https://polymarket.com/event/what-price-will-bitcoin-hit-in-october-2026", btc150k.url)
        assertEquals(1_793_505_600_000L, btc150k.endDateMs, "2026-11-01T04:00:00Z")
        assertEquals(1196.24, btc150k.volume24hUsd)
        assertEquals(102306.2531, btc150k.liquidityUsd)
        // ranked by 24h volume; the (non-crypto) Fed market is the biggest in the fixture
        assertTrue(markets.first().question.contains("Fed interest rates"))
        assertTrue(markets.zipWithNext().all { (a, b) -> (a.volume24hUsd ?: 0.0) >= (b.volume24hUsd ?: 0.0) })
    }

    @Test
    fun closedAndBarelyTradedMarketsAreDropped() {
        // the fixture carries two "$82,500" markets: one closed (already resolved inside a still-open
        // event) and one with $32 of 24h volume; both must be gone
        assertTrue(markets.none { it.question == "Will Bitcoin reach \$82,500 in October?" })
    }

    @Test
    fun forAssetMatchesTickerOrNameAsWholeWords() {
        val btc = PolymarketApi.forAsset(markets, "BTC", "Bitcoin")
        assertEquals(5, btc.size, "seven Bitcoin markets in the fixture, capped at five")
        assertTrue(btc.all { it.question.contains("Bitcoin") })
        assertEquals("Will the price of Bitcoin be above \$86,000 on October 20?", btc.first().question)
        assertTrue(btc.zipWithNext().all { (a, b) -> (a.volume24hUsd ?: 0.0) >= (b.volume24hUsd ?: 0.0) })
        assertEquals(2, PolymarketApi.forAsset(markets, "ETH", "Ethereum").size)
        // "XRP" appears only as the ticker; the Fed market names no coin at all
        assertEquals(listOf("Will XRP reach \$4.00 in October?"), PolymarketApi.forAsset(markets, "XRP", "XRP").map { it.question })
        assertTrue(PolymarketApi.forAsset(markets, "SOL", "Solana").isEmpty())
        // whole words only: the ticker must not hit the inside of "Ethereum"
        assertTrue(PolymarketApi.forAsset(markets, "ETH", "ETH").isEmpty())
        // "Up" is a word in every up-or-down question; a two-letter ticker must not ride on that
        assertTrue(PolymarketApi.forAsset(markets, "UP", "Uptober").isEmpty())
    }

    @Test
    fun cashtagsLetShortTickersMatchExplicitly() {
        // no captured question carries a cashtag, so minimally edit a real one: the daily
        // up-or-down market's coin name becomes a "$UP" cashtag, which a two-letter ticker may match
        val body = Fixtures.read("spec/polymarket_events.json")
            .replace("Bitcoin Up or Down on October 20?", "\$UP coin Up or Down on October 20?")
        val edited = PolymarketApi.parseEvents(body)
        val tagged = edited.first { it.slug == "bitcoin-up-or-down-on-october-20-2026" }
        assertEquals(listOf("UP"), tagged.symbols)
        assertEquals(listOf(tagged), PolymarketApi.forAsset(edited, "UP", "Uptober"))
    }

    @Test
    fun upDownProbabilityPrefersTheDailyMarket() {
        // both the daily (Oct 20) and the hourly (Oct 19, 1PM ET) BTC windows close within 48h of
        // the capture; the daily one wins and its first outcome ("Up") prices at 0.505
        assertEquals(0.505, PolymarketApi.upDownProbability(markets, "BTC", "Bitcoin", captureNowMs))
        // without the daily market the nearest intraday window still answers
        val hourlyOnly = markets.filterNot { it.slug == "bitcoin-up-or-down-on-october-20-2026" }
        assertEquals(0.265, PolymarketApi.upDownProbability(hourlyOnly, "BTC", "Bitcoin", captureNowMs))
        // a market whose first outcome is "Down" is inverted so the answer is always P(up)
        val flipped = markets.map { if (it.slug == "bitcoin-up-or-down-on-october-20-2026") it.copy(outcome = "Down") else it }
        assertEquals(0.495, PolymarketApi.upDownProbability(flipped, "BTC", "Bitcoin", captureNowMs)!!, 1e-9)
        // no Ethereum up-or-down market was captured
        assertNull(PolymarketApi.upDownProbability(markets, "ETH", "Ethereum", captureNowMs))
        // well before the capture every candidate ends further than 48h away
        assertNull(PolymarketApi.upDownProbability(markets, "BTC", "Bitcoin", FIXTURE_NOW_MS))
    }

    @Test
    fun cryptoMarketsHitTheTaggedEventsEndpoint() = runTest {
        val t = FakeTransport().onUrlContaining("/events?tag_slug=crypto", body = Fixtures.read("spec/polymarket_events.json"))
        val r = PolymarketApi(testHttp(t)).cryptoMarkets()
        assertEquals(11, (r as SourceResult.Success).value.size)
        assertEquals(1, t.countRequests("gamma-api.polymarket.com"))
    }

    @Test
    fun gammaHostHasItsOwnRateBudget() {
        val rate = RateLimiter().rateFor("gamma-api.polymarket.com")
        assertEquals(30.0, rate.requestsPerMinute)
        assertEquals(5, rate.burst)
    }

    @Test
    fun aTickerThatIsAnEverydayWordNeverMatchesProse() {
        // "Will the price of Bitcoin be above $86,000..." must not become a THE market
        assertTrue(PolymarketApi.forAsset(markets, "THE", "The Token").isEmpty())
        assertTrue(PolymarketApi.forAsset(markets, "ONE", "Harmony").isEmpty(), "AMBIGUOUS_SYMBOLS are cashtag/name-only")
        assertTrue(PolymarketApi.forAsset(markets, "BTC", "Bitcoin").isNotEmpty(), "real tickers still match")
    }
}
