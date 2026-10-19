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
import io.ktor.client.HttpClient
import io.ktor.client.plugins.sse.SSE
import io.ktor.server.cio.CIO
import io.ktor.server.engine.embeddedServer
import io.modelcontextprotocol.kotlin.sdk.client.Client
import io.modelcontextprotocol.kotlin.sdk.client.mcpStreamableHttp
import io.modelcontextprotocol.kotlin.sdk.server.mcpStreamableHttp
import io.modelcontextprotocol.kotlin.sdk.types.TextContent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import io.ktor.client.engine.cio.CIO as ClientCIO

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class McpServerTest {
    private val now = 1_760_713_200_000L
    private val clock = Clock.fixed(now)

    private val transport = FakeTransport()
        // memecoins and DEX radar (payloads captured live; routes before the generic /coins/markets one)
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
        scorer = SentimentScorer { t -> TextSentiment(if ("surge" in t.lowercase() || "bullish" in t.lowercase()) 0.5 else 0.0) },
        clock = clock, logger = Logger.NONE,
        rateLimiter = RateLimiter(emptyMap(), Rate(1e6, 1_000_000), clock, sleeper = {}),
    )
    private val tools = CryptoPulseTools(app, CoroutineScope(SupervisorJob() + Dispatchers.IO))
    private val server = embeddedServer(CIO, host = "127.0.0.1", port = 0) { mcpStreamableHttp { tools.createServer() } }
    private val http = HttpClient(ClientCIO) { install(SSE) }
    private lateinit var client: Client

    @BeforeAll
    fun start() = runBlocking {
        server.start(wait = false)
        val port = server.engine.resolvedConnectors().single().port
        client = http.mcpStreamableHttp(url = "http://127.0.0.1:$port/mcp")
    }

    @AfterAll
    fun stop() = runBlocking {
        client.close()
        http.close()
        server.stop(200, 500)
    }

    private fun text(r: io.modelcontextprotocol.kotlin.sdk.types.CallToolResult) = r.content.filterIsInstance<TextContent>().joinToString("\n") { it.text }

    @Test
    fun listsAllTools() = runBlocking {
        val names = client.listTools().tools.map { it.name }.toSet()
        assertEquals(
            setOf(
                "market_overview", "speculative_radar", "get_forecast", "get_sentiment", "find_opportunities", "get_prices", "list_bitpanda_assets",
                "market_mood", "telegram_sources", "daily_digest",
            ),
            names,
        )
        assertEquals("cryptopulse", client.serverVersion?.name)
        assertEquals(listOf("crypto_briefing"), client.listPrompts().prompts.map { it.name })
    }

    @Test
    fun overviewForecastAndOpportunities() = runBlocking {
        val overview = client.callTool("market_overview", mapOf("limit" to 10, "sort" to "market_cap"))
        val t = text(overview)
        assertTrue(t.contains("| 1 | Bitcoin (BTC) |"), t)
        assertNotNull(overview.structuredContent)

        val forecast = client.callTool("get_forecast", mapOf("symbol" to "sol"))
        val ft = text(forecast)
        assertTrue(ft.contains("Next 24h") && ft.contains("Next 7d") && ft.contains("Is this the bottom?"), ft)
        assertEquals(true, forecast.structuredContent?.containsKey("h24"))

        val opp = client.callTool("find_opportunities", mapOf("horizon" to "24h", "min_probability" to 0.0, "limit" to 5))
        assertTrue(text(opp).contains("chance of rising"))

        val prices = client.callTool("get_prices", mapOf("symbols" to "BTC, eth,XYZ"))
        assertTrue(text(prices).contains("72,529.49 EUR") && text(prices).contains("XYZ: not in the list"), text(prices))
    }

    @Test
    fun speculativeRadarAndTierFilters() = runBlocking {
        val radar = client.callTool("speculative_radar", mapOf("tier" to "dex", "limit" to 10))
        val t = text(radar)
        assertTrue(t.contains("Not on Bitpanda; information only"), t)
        assertTrue(t.contains("TORCH") && t.contains("robinhood"), t)
        assertTrue(t.contains("Unverified") || t.contains("Too new"), "no probability without checks: $t")
        val memes = client.callTool("market_overview", mapOf("tier" to "meme", "limit" to 20))
        assertTrue(text(memes).contains("SHIB") && text(memes).contains("Memecoin"), text(memes))
        val bitpandaOnly = client.callTool("market_overview", mapOf("tier" to "bitpanda", "limit" to 600))
        assertTrue(!text(bitpandaOnly).contains("TORCH"), text(bitpandaOnly))
        val torch = client.callTool("get_forecast", mapOf("symbol" to "TORCH", "deep" to false))
        val ft = text(torch)
        assertTrue(ft.contains("Not on Bitpanda: DEX radar") && ft.contains("No forecast"), ft)
        // blocked: no "50% chance" or "Score 50/100" placeholders, in the text or the structured result
        assertTrue(ft.contains("### Next 24h: no forecast — ") && ft.contains("### Next 7d: no forecast — "), ft)
        assertTrue(!ft.contains("chance of going up") && !ft.contains("Is this the bottom?") && !ft.contains("+ Safety red flags"), ft)
        val s = assertNotNull(torch.structuredContent)
        assertTrue(s.getValue("h24").jsonObject["probabilityUp"].let { it == null || it == JsonNull }, s.toString())
        assertEquals(true, s.getValue("risk").jsonObject["blocked"]?.jsonPrimitive?.booleanOrNull, s.toString())
        val badTier = client.callTool("market_overview", mapOf("tier" to "stocks"))
        assertEquals(true, badTier.isError)
    }

    @Test
    fun sentimentMoodDigestAndErrors() = runBlocking {
        val s = client.callTool("get_sentiment", mapOf("symbol" to "SOL"))
        assertTrue(text(s).contains("sentiment hub"), text(s))
        val mood = client.callTool("market_mood", emptyMap())
        assertTrue(text(mood).contains("Fear & Greed: 22"), text(mood))
        val digest = client.callTool("daily_digest", mapOf("lang" to "hr"))
        assertTrue(text(digest).contains("Najveća šansa rasta u 24h"), text(digest))
        val tg = client.callTool("telegram_sources", mapOf("kind" to "channel"))
        assertTrue(text(tg).contains("Telegram directory"), text(tg))
        val bad = client.callTool("get_forecast", mapOf("symbol" to "NOPE"))
        assertEquals(true, bad.isError)
        assertTrue(text(bad).contains("is not in the list"), text(bad))
    }
}
