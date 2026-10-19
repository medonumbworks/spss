package dev.cryptopulse.mcp

import dev.cryptopulse.core.CryptoPulse
import dev.cryptopulse.core.analysis.SentimentScorer
import dev.cryptopulse.core.analysis.TextSentiment
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.http.Rate
import dev.cryptopulse.core.http.RateLimiter
import dev.cryptopulse.core.testing.FakeTransport
import dev.cryptopulse.core.testing.Fixtures
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.Logger
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Test
import java.time.Instant
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Shape-level checks of the dashboard export against the `cryptopulse.dashboard/v1` contract in
 * web/app.js. Fixture values drift when payloads are re-captured, so the assertions pin field
 * names, ranges and invariants, not concrete numbers.
 */
class ExportTest {
    private val clock = Clock.fixed(1_760_713_200_000L)

    private val transport = FakeTransport()
        // same captured payloads as McpServerTest, plus Polymarket's gamma API
        .onUrlContaining("gamma-api", body = Fixtures.read("spec/polymarket_events.json"))
        .onUrlContaining("category=meme-token", body = Fixtures.read("spec/coingecko_meme_markets.json"))
        .onUrlContaining("networks/trending_pools", body = Fixtures.read("spec/geckoterminal_trending.json"))
        .onUrlContaining("networks/new_pools", body = Fixtures.read("spec/geckoterminal_new_pools.json"))
        .onUrlContaining("/token-boosts/top/v1", body = Fixtures.read("spec/dexscreener_boosts_top.json"))
        .onUrlContaining("/tokens/v1/solana/", body = Fixtures.read("spec/dexscreener_tokens_boosted.json"))
        .onUrlContaining("gopluslabs.io/api/v1/solana/", body = Fixtures.read("spec/goplus_sol.json"))
        .onUrlContaining("api.bitpanda.com/v1/ticker", body = Fixtures.read("bitpanda_ticker.json"))
        .onUrlContaining("/coins/list", body = Fixtures.read("coingecko_coins_list.json"))
        .onUrlContaining("/coins/markets", body = Fixtures.read("coingecko_markets.json"))
        .onUrlContaining("/market_chart", body = Fixtures.read("coingecko_market_chart_btc.json"))
        .onUrlContaining("/global", body = Fixtures.read("coingecko_global.json"))
        .onUrlContaining("/search/trending", body = Fixtures.read("coingecko_trending.json"))
        .onUrlContaining("api.coingecko.com/api/v3/coins/", body = Fixtures.read("coingecko_coin_btc.json"))
        .onUrlContaining("alternative.me", body = Fixtures.read("fear_greed.json"))
        .onUrlContaining("premiumIndex", body = Fixtures.read("binance_premium.json"))
        .onUrlContaining("sentiment-feed/feed.json", body = Fixtures.read("sentiment_feed.json"))
        .onUrlContaining("reddit.com", body = Fixtures.read("reddit_search.json"))
        .onUrlContaining("stocktwits.com", body = Fixtures.read("stocktwits_sol.json"))
        .onUrlContaining("t.me/s/", body = Fixtures.read("telegram_preview.html"))
        .onUrlContaining("news.google.com", body = Fixtures.read("rss_cointelegraph.xml"))
        .onUrlContaining("rss", body = Fixtures.read("rss_cointelegraph.xml"))
        .onUrlContaining("feed", body = Fixtures.read("atom_feed.xml"))

    private val app = CryptoPulse.create(
        AppConfig(), transport, null,
        scorer = SentimentScorer { TextSentiment(0.0) },
        clock = clock, logger = Logger.NONE,
        rateLimiter = RateLimiter(emptyMap(), Rate(1e6, 1_000_000), clock, sleeper = {}),
    )

    @Test
    fun exportMatchesTheDashboardContract() = runBlocking {
        val out = Json.parseToJsonElement(Export.build(app)).jsonObject

        assertEquals("cryptopulse.dashboard/v1", out.getValue("schema").jsonPrimitive.content)
        assertEquals("EUR", out.getValue("currency").jsonPrimitive.content)
        assertNotNull(Instant.parse(out.getValue("generatedAt").jsonPrimitive.content))
        assertNotNull(out.getValue("fearGreed").jsonObject["value"]?.jsonPrimitive?.int)

        val rows = out.getValue("rows").jsonArray.map { it.jsonObject }
        val btc = rows.first { it["symbol"]?.jsonPrimitive?.content == "BTC" }
        assertEquals("BITPANDA", btc.getValue("tier").jsonPrimitive.content)
        assertTrue(assertNotNull(btc["price"]?.jsonPrimitive?.doubleOrNull) > 0.0, btc.toString())
        btc["up24"]?.jsonPrimitive?.doubleOrNull?.let { assertTrue(it in 0.0..1.0, btc.toString()) }
        // DEX rows carry their chain, and a row without a call says why
        val dex = rows.filter { it["tier"]?.jsonPrimitive?.content == "DEX" }
        assertTrue(dex.isNotEmpty() && dex.all { it["chain"] != null }, dex.toString())
        val noCall = rows.filter { it["noCall"] != null }
        assertTrue(noCall.isNotEmpty())
        assertTrue(noCall.all { it["up24"] == null || it["up7"] == null }, noCall.toString())

        val buzz = out.getValue("buzz").jsonArray.map { it.jsonObject }
        assertTrue(buzz.isNotEmpty())
        val samples = buzz.flatMap { it.getValue("samples").jsonArray }.map { it.jsonObject }
        assertTrue(samples.isNotEmpty() && samples.all { it["url"] != null && it["text"] != null }, samples.toString())
        assertTrue(buzz.all { it.getValue("samples").jsonArray.size <= 3 })
        // the row join: the buzz leader's mentions show up on its market row too
        val loudest = buzz.maxBy { it.getValue("mentions").jsonPrimitive.int }
        rows.firstOrNull { it["symbol"] == loudest["symbol"] && it["tier"]?.jsonPrimitive?.content == "BITPANDA" }?.let {
            assertEquals(loudest.getValue("mentions").jsonPrimitive.int, it.getValue("mentions").jsonPrimitive.int)
        }

        val h24 = out.getValue("accuracy").jsonObject.getValue("24h").jsonObject
        assertTrue(h24.getValue("pending").jsonPrimitive.int >= 0, h24.toString())
        assertTrue(h24.getValue("evaluated").jsonPrimitive.int >= 0, h24.toString())

        val pm = out.getValue("polymarket").jsonArray.map { it.jsonObject }
        assertTrue(pm.isNotEmpty() && pm.size <= 20, pm.toString())
        pm.forEach { m ->
            assertTrue(assertNotNull(m["question"]?.jsonPrimitive?.content).isNotBlank(), m.toString())
            assertTrue(assertNotNull(m["url"]?.jsonPrimitive?.content).startsWith("https://polymarket.com/"), m.toString())
            assertTrue(assertNotNull(m["probability"]?.jsonPrimitive?.doubleOrNull) in 0.0..1.0, m.toString())
            assertTrue(m.getValue("symbols").jsonArray.isNotEmpty(), m.toString())
        }

        val alerts = out.getValue("alerts").jsonArray.map { it.jsonObject }
        alerts.forEach { a ->
            assertNotNull(a["symbol"]?.jsonPrimitive?.content, a.toString())
            assertTrue(assertNotNull(a["probabilityUp"]?.jsonPrimitive?.doubleOrNull) in 0.0..1.0, a.toString())
        }

        assertTrue(out.getValue("notes").jsonArray.size >= 2, "cadence and disclaimer notes expected")
    }
}
