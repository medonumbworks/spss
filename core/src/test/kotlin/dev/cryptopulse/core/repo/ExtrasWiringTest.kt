package dev.cryptopulse.core.repo

import dev.cryptopulse.core.CryptoPulse
import dev.cryptopulse.core.FIXTURE_NOW_MS
import dev.cryptopulse.core.analysis.journal.ForecastAlerts
import dev.cryptopulse.core.analysis.SentimentScorer
import dev.cryptopulse.core.analysis.TextSentiment
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.fastLimiter
import dev.cryptopulse.core.http.InMemoryDiskStore
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.Direction
import dev.cryptopulse.core.model.Forecast
import dev.cryptopulse.core.model.BottomAssessment
import dev.cryptopulse.core.model.HorizonForecast
import dev.cryptopulse.core.model.Horizon
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.testing.FakeTransport
import dev.cryptopulse.core.testing.Fixtures
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.Logger
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** The journal, alerts, buzz, pump and Polymarket wiring added on top of the Wave-1 components. */
class ExtrasWiringTest {
    private val clock = Clock.fixed(FIXTURE_NOW_MS)
    private val scorer = SentimentScorer { TextSentiment(0.0) }

    /** BTC pumps +25% in the markets fixture and its Telegram mentions spike 8x in the feed. */
    private fun transport(): FakeTransport {
        // both 24 h keys of the bitcoin entry: the parser prefers the in_currency one
        var markets = Fixtures.read("coingecko_markets.json")
        for (key in listOf("\"price_change_percentage_24h\": ", "\"price_change_percentage_24h_in_currency\": ")) {
            markets = Regex("(\"id\": \"bitcoin\".*?" + Regex.escape(key) + ")(-?[0-9.eE+]+)", RegexOption.DOT_MATCHES_ALL)
                .replace(markets) { m -> m.groupValues[1] + "25.0" }
        }
        val feed = Fixtures.read("sentiment_feed.json")
            .replace("{\"symbol\":\"BTC\",\"mentions\":300,", "{\"symbol\":\"BTC\",\"mentions\":300,\"spike\":8.0,")
        return FakeTransport()
            .onUrlContaining("api.bitpanda.com/v1/ticker", body = Fixtures.read("bitpanda_ticker.json"))
            .onUrlContaining("/coins/list", body = Fixtures.read("coingecko_coins_list.json"))
            .onUrlContaining("/coins/markets", body = markets)
            .onUrlContaining("/global", body = Fixtures.read("coingecko_global.json"))
            .onUrlContaining("/search/trending", body = Fixtures.read("coingecko_trending.json"))
            .onUrlContaining("alternative.me", body = Fixtures.read("fear_greed.json"))
            .onUrlContaining("premiumIndex", body = Fixtures.read("binance_premium.json"))
            .onUrlContaining("sentiment-feed/feed.json", body = feed)
            .onUrlContaining("gamma-api.polymarket.com", body = Fixtures.read("spec/polymarket_events.json"))
    }

    private fun app(store: InMemoryDiskStore? = null) = CryptoPulse.create(
        // only the Bitpanda tier matters here; the speculative tiers have their own tests
        AppConfig(showMemecoins = false, showTrending = false, showDexRadar = false),
        transport(), store, scorer, clock = clock, logger = Logger.NONE, rateLimiter = fastLimiter(clock),
    )

    @Test
    fun refreshFeedsTheJournalAndBuzzAndPump() = runTest {
        val cp = app(InMemoryDiskStore())
        val snap = cp.repository.refresh()

        val stats = cp.repository.journalStats()
        assertTrue(stats.total > 0, "every refresh logs the issued forecasts")
        assertTrue((stats.perHorizon.getValue(Horizon.H24.label).pending) > 0)

        val buzz = cp.repository.telegramBuzz()
        assertTrue(buzz.isNotEmpty())
        assertEquals("BTC", buzz.first().symbol, "the spiking coin ranks first: ${buzz.take(3)}")
        assertEquals(8.0, buzz.first().spike)

        val btc = snap.rows.first { it.asset.symbol == "BTC" }
        val pump = btc.pump
        assertTrue(pump != null && pump.reasons.any { "Telegram mentions" in it }, "hype + 25% pump warns: $pump")
        assertTrue(snap.rows.first { it.asset.symbol == "ETH" }.pump == null, "a quiet coin stays quiet")
    }

    @Test
    fun alertsFireOnceGroupTheCrossingAndSurviveARestart() = runTest {
        val store = InMemoryDiskStore()
        val cp = app(store)
        cp.repository.refresh()
        // threshold 1%: every Bitpanda coin with a real 24 h call crosses on the first check
        val first = cp.repository.alertCrossings(thresholdPct = 1)
        assertTrue(first.isNotEmpty())
        assertTrue(first.all { it.probabilityUp >= 0.01 && it.price != null })
        assertEquals(emptyList(), cp.repository.alertCrossings(thresholdPct = 1), "no repeat while a coin stays above")
        // the above-set is persisted: a restarted app does not re-notify
        val cp2 = app(store)
        cp2.repository.refresh()
        assertEquals(emptyList(), cp2.repository.alertCrossings(thresholdPct = 1))
        // a changed threshold starts a fresh set
        assertTrue(cp2.repository.alertCrossings(thresholdPct = 2).isNotEmpty())
    }

    @Test
    fun detailCarriesTheCoinsPredictionMarkets() = runTest {
        val cp = app()
        cp.repository.refresh()
        val d = cp.repository.detail("BTC")
        assertTrue(d.predictionMarkets.isNotEmpty(), "the fixture has Bitcoin markets")
        assertTrue(d.predictionMarkets.all { "bitcoin" in it.question.lowercase() || "BTC" in it.symbols })
        val eth = cp.repository.detail("ETH")
        assertTrue(eth.predictionMarkets.none { "bitcoin" in it.question.lowercase() }, "another coin's markets never leak in")
    }

    @Test
    fun alertsIgnoreBlockedAndNonBitpandaRows() {
        fun f(p: Double) = Forecast(
            symbol = "X", currency = "EUR", price = 1.0,
            h24 = HorizonForecast(Horizon.H24, p, Direction.UP, 0.5),
            d7 = HorizonForecast(Horizon.D7, p, Direction.UP, 0.5),
            bottom = BottomAssessment(50, "x"), summary = "", computedAtMs = 0, modelVersion = "t",
        )
        val rows = listOf(
            AssetRow(Asset("AAA", "Aaa"), MarketData("AAA", currency = "EUR", price = 1.0), f(0.61)),
            AssetRow(Asset("MEM", "Meme", tier = Tier.MEME), MarketData("MEM", currency = "EUR", price = 1.0), f(0.9)),
            AssetRow(Asset("LOW", "Low"), MarketData("LOW", currency = "EUR", price = 1.0), f(0.52)),
        )
        val alerts = ForecastAlerts(InMemoryDiskStore(), clock)
        assertEquals(listOf("AAA"), alerts.crossings(rows, 55).map { it.symbol }, "only Bitpanda coins above the threshold alert")
        alerts.crossings(emptyList(), 55)
        assertTrue(alerts.crossings(rows, 55).isEmpty(), "an empty refresh must not re-arm old alerts")
    }
}
