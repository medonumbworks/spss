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
import io.modelcontextprotocol.kotlin.sdk.types.CallToolResult
import io.modelcontextprotocol.kotlin.sdk.types.TextContent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import io.ktor.client.engine.cio.CIO as ClientCIO

/**
 * The captured DEX payloads about two weeks after their pools opened: SI276 (13 days, not blocked)
 * gets a 24h call but has too little history for a 7-day one; TORCH, EMPIREX and CATONICE stay
 * blocked. Without the Bitpanda ticker and futures there is no USD rate, so DEX prices stay in USD
 * while the snapshot is in EUR.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class NoCallRowsTest {
    private val clock = Clock.fixed(1_760_713_200_000L + 8_833L * 3_600_000L)

    private val transport = FakeTransport()
        .onUrlContaining("category=meme-token", body = Fixtures.read("spec/coingecko_meme_markets.json"))
        .onUrlContaining("networks/trending_pools", body = Fixtures.read("spec/geckoterminal_trending.json"))
        .onUrlContaining("networks/new_pools", body = Fixtures.read("spec/geckoterminal_new_pools.json"))
        .onUrlContaining("/token-boosts/top/v1", body = Fixtures.read("spec/dexscreener_boosts_top.json"))
        .onUrlContaining("/tokens/v1/solana/", body = Fixtures.read("spec/dexscreener_tokens_boosted.json"))
        .onUrlContaining("gopluslabs.io/api/v1/solana/", body = Fixtures.read("spec/goplus_sol.json"))
        .onUrlContaining("/coins/list", body = Fixtures.read("coingecko_coins_list.json"))
        .onUrlContaining("/coins/markets", body = Fixtures.read("coingecko_markets.json"))
        .onUrlContaining("/market_chart", body = Fixtures.read("coingecko_market_chart_btc.json"))
        .onUrlContaining("/global", body = Fixtures.read("coingecko_global.json"))
        .onUrlContaining("/search/trending", body = Fixtures.read("coingecko_trending.json"))
        .onUrlContaining("api.coingecko.com/api/v3/coins/", body = Fixtures.read("coingecko_coin_btc.json"))
        .onUrlContaining("alternative.me", body = Fixtures.read("fear_greed.json"))
        .onUrlContaining("sentiment-feed/feed.json", body = Fixtures.read("sentiment_feed.json"))

    private val app = CryptoPulse.create(
        AppConfig(), transport, null,
        scorer = SentimentScorer { TextSentiment(0.0) },
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

    private fun text(r: CallToolResult) = r.content.filterIsInstance<TextContent>().joinToString("\n") { it.text }

    /** One column of the Markdown table in [text], top to bottom. */
    private fun column(text: String, header: String): List<String> {
        val lines = text.lines().filter { it.startsWith("|") }
        val i = lines.first().split('|').map { it.trim() }.indexOf(header)
        assertTrue(i > 0, "no column $header in $text")
        return lines.drop(2).map { it.split('|')[i].trim() }
    }

    private fun hasValue(cell: String) = cell.endsWith("%") || cell.toIntOrNull() != null

    @Test
    fun forecastShowsNoNumberForAHorizonWithoutACall() = runBlocking {
        val r = client.callTool("get_forecast", mapOf("symbol" to "SI276", "deep" to false))
        val t = text(r)
        assertTrue(Regex("### Next 24h: . \\d+% chance of going up").containsMatchIn(t), t)
        assertTrue(t.contains("### Next 7d: no forecast — "), t)
        assertFalse(Regex("### Next 7d: [^\\n]*chance").containsMatchIn(t), t)
        val s = assertNotNull(r.structuredContent)
        assertNotNull(s.getValue("h24").jsonObject["probabilityUp"]?.jsonPrimitive?.doubleOrNull, s.toString())
        assertTrue(s.getValue("d7").jsonObject["probabilityUp"].let { it == null || it == JsonNull }, s.toString())
        assertEquals(false, s.getValue("risk").jsonObject["blocked"]?.jsonPrimitive?.booleanOrNull, s.toString())
    }

    @Test
    fun opportunitiesSkipHorizonsWithoutACall() = runBlocking {
        val week = text(client.callTool("find_opportunities", mapOf("tier" to "dex", "horizon" to "7d", "min_probability" to 0.0)))
        assertFalse(week.contains("SI276"), "too young for a 7-day call: $week")
        val day = text(client.callTool("find_opportunities", mapOf("tier" to "dex", "horizon" to "24h", "min_probability" to 0.0)))
        assertTrue(day.contains("SI276") && !day.contains("TORCH") && !day.contains("CATONICE"), day)
    }

    @Test
    fun rankingsPutRowsWithoutAForecastLast() = runBlocking {
        for ((sort, header) in listOf("up_24h" to "Up 24h", "up_7d" to "Up 7d", "bottom_score" to "Bottom")) {
            val col = column(text(client.callTool("market_overview", mapOf("tier" to "all", "sort" to sort, "limit" to 600))), header)
            val firstBlank = col.indexOfFirst { !hasValue(it) }
            assertTrue(firstBlank > 0 && col.drop(firstBlank).none { hasValue(it) }, "$sort: $col")
        }
        // SI276's 49% beats the 0.5 placeholders of the blocked pools
        val dex = text(client.callTool("market_overview", mapOf("tier" to "dex", "sort" to "up_24h")))
        assertTrue(column(dex, "Coin").first().contains("(SI276)"), dex)
        val radar = text(client.callTool("speculative_radar", mapOf("tier" to "dex", "sort" to "up_24h")))
        assertTrue(column(radar, "Token").first().contains("(SI276)"), radar)
        val withCall = text(client.callTool("speculative_radar", mapOf("tier" to "dex", "hide_blocked" to true)))
        assertEquals(listOf(true), column(withCall, "Token").map { "(SI276)" in it }, withCall)
    }

    /** market.currency of each structured row, in table order. */
    private fun currencies(r: CallToolResult) =
        assertNotNull(r.structuredContent).getValue("rows").jsonArray.map { it.jsonObject["market"]?.jsonObject?.get("currency")?.jsonPrimitive?.content }

    @Test
    fun pricesCarryTheirOwnCurrency() = runBlocking {
        // with these routes DEX prices stay in USD; every price not in EUR must say so
        for ((tool, header) in listOf("market_overview" to "Price (EUR)", "speculative_radar" to "Price")) {
            val r = client.callTool(tool, mapOf("tier" to "dex"))
            val cells = column(text(r), header)
            assertTrue(cells.any { it != "—" }, text(r))
            currencies(r).zip(cells).filter { it.second != "—" }.forEach { (cur, cell) -> assertEquals(cur != "EUR", cell.endsWith(" $cur"), "$tool, $cur: $cell") }
        }
        val cur = currencies(client.callTool("market_overview", mapOf("tier" to "dex", "query" to "SI276"))).single()
        val prices = text(client.callTool("get_prices", mapOf("symbols" to "SI276,BTC")))
        assertTrue(Regex("\\(SI276\\) \\[DEX radar, not on Bitpanda]: [0-9.,]+ $cur · ").containsMatchIn(prices) && prices.contains("72,529.49 EUR"), prices)
    }
}
