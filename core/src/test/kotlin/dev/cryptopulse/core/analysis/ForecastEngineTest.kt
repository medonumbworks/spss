package dev.cryptopulse.core.analysis

import dev.cryptopulse.core.FIXTURE_NOW_MS
import dev.cryptopulse.core.analysis.forecast.Backtester
import dev.cryptopulse.core.analysis.forecast.DefaultForecastEngine
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.DerivativesSnapshot
import dev.cryptopulse.core.model.Direction
import dev.cryptopulse.core.model.FearGreed
import dev.cryptopulse.core.model.MarketContext
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.SentimentSummary
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ForecastEngineTest {
    private val engine = DefaultForecastEngine()

    private fun series(n: Int, start: Double, drift: Double, wiggle: Double = 0.004) =
        (0 until n).map { i -> start * exp(drift * i + wiggle * sin(i / 3.0)) }

    private fun input(symbol: String, hourly: List<Double>, daily: List<Double> = emptyList(), ctx: MarketContext? = null, sentiment: SentimentSummary? = null, deriv: DerivativesSnapshot? = null) =
        ForecastInput(
            asset = Asset(symbol, symbol),
            market = MarketData(symbol = symbol, currency = "EUR", price = hourly.last(), sparkline7d = hourly, change7dPct = (hourly.last() / hourly.first() - 1) * 100, athChangePct = -60.0),
            hourlyCloses = hourly,
            dailyCloses = daily,
            context = ctx,
            sentiment = sentiment,
            derivatives = deriv,
            nowMs = FIXTURE_NOW_MS,
        )

    @Test
    fun probabilitiesStayHumbleAndExplained() {
        val f = engine.forecast(input("SOL", series(168, 100.0, 0.002)))
        for (h in listOf(f.h24, f.d7)) {
            assertTrue(h.probabilityUp in 0.15..0.85, "p=${h.probabilityUp}")
            assertTrue(h.confidence in 0.05..0.9)
            assertTrue(h.signals.isNotEmpty())
            assertTrue(h.expectedLow!! < f.price!! && h.expectedHigh!! > f.price!!)
        }
        assertTrue(f.summary.startsWith("SOL:"))
        assertEquals(DefaultForecastEngine.MODEL_VERSION, f.modelVersion)
    }

    @Test
    fun crowdedLongsAndGreedLowerTheOdds() {
        val hourly = series(168, 100.0, 0.0005)
        val calm = engine.forecast(input("ETH", hourly))
        val greedy = engine.forecast(
            input(
                "ETH", hourly,
                ctx = MarketContext(fearGreed = FearGreed(90, "Extreme Greed", 0), btcTrendScore = -0.6),
                deriv = DerivativesSnapshot("ETH", fundingRate = 0.0015, longShortRatio = 3.0),
            ),
        )
        assertTrue(greedy.d7.probabilityUp < calm.d7.probabilityUp, "${greedy.d7.probabilityUp} vs ${calm.d7.probabilityUp}")
        assertTrue(greedy.h24.signals.any { it.name == "funding" && it.contribution < 0 })
    }

    @Test
    fun bullishSentimentRaisesTheOdds() {
        val hourly = series(168, 100.0, 0.0)
        val base = engine.forecast(input("PEPE", hourly))
        val bull = engine.forecast(input("PEPE", hourly, sentiment = SentimentSummary("PEPE", 0.6, 0.9, "Very bullish", totalSamples = 200)))
        assertTrue(bull.h24.probabilityUp > base.h24.probabilityUp)
    }

    @Test
    fun bottomScoreHighAfterSellOffAndLowNearHighs() {
        val crash = series(300, 100.0, -0.004, 0.01) // long decline into the lows
        val f = engine.forecast(input("ADA", crash.takeLast(168), daily = series(365, 1.0, -0.003, 0.02), ctx = MarketContext(fearGreed = FearGreed(15, "Extreme Fear", 0))))
        assertTrue(f.bottom.score >= 60, "score=${f.bottom.score} ${f.bottom.reasons}")
        assertTrue(f.bottom.reasons.isNotEmpty())
        val rally = engine.forecast(input("ADA", series(168, 100.0, 0.004), daily = series(365, 1.0, 0.004), ctx = MarketContext(fearGreed = FearGreed(85, "Extreme Greed", 0))))
        assertTrue(rally.bottom.score <= 35, "score=${rally.bottom.score}")
        assertTrue(f.d7.backtest != null && f.d7.backtest!!.samples > 100)
    }

    @Test
    fun stablecoinsAreFlat() {
        val f = engine.forecast(input("USDC", List(168) { 0.8898 }))
        assertEquals(Direction.FLAT, f.h24.direction)
        assertEquals(0.5, f.d7.probabilityUp)
        assertEquals("Pegged", f.bottom.label)
    }

    @Test
    fun backtestReportsChanceBaselineAndIndependentSamples() {
        val bt = Backtester.shortTerm(series(168, 100.0, 0.001, 0.02))!!
        assertTrue(bt.samples > 30)
        assertTrue(bt.hitRate in 0.0..1.0 && bt.baselineHitRate in 0.0..1.0)
        assertEquals(bt.samples * 2 / 24, bt.effectiveSamples, "overlapping 24h windows every 2h")
    }

    @Test
    fun randomWalkBacktestDoesNotDistortTheTechnicalWeight() {
        // with no edge, most runs must land in the neutral band instead of being cut or boosted
        val rnd = Random(7)
        var neutral = 0
        repeat(60) {
            var p = 100.0
            val daily = List(365) { p *= exp(rnd.nextDouble(-0.04, 0.04)); p }
            val bt = Backtester.mediumTermDaily(daily)!!
            val z = (bt.hitRate - bt.baselineHitRate) / sqrt(0.25 / bt.effectiveSamples)
            if (z > -1.0 && z < 1.0) neutral++
        }
        assertTrue(neutral >= 30, "neutral=$neutral/60")
    }

    @Test
    fun bottomScoreNeedsPriceHistory() {
        val f = engine.forecast(
            ForecastInput(
                asset = Asset("NEW", "New coin"),
                market = MarketData(symbol = "NEW", currency = "EUR", price = 1.0),
                hourlyCloses = emptyList(),
                context = MarketContext(fearGreed = FearGreed(15, "Extreme Fear", 0)),
                nowMs = FIXTURE_NOW_MS,
            ),
        )
        assertEquals(50, f.bottom.score)
        assertEquals(DefaultForecastEngine.NO_HISTORY, f.bottom.label)
        assertEquals(0.0, f.d7.signals.single { it.name == "bottom_score" }.contribution)
    }

    @Test
    fun weekMoveIsUsedWithAnHourlySparklineOnly() {
        for (n in listOf(167, 168)) {
            val f = engine.forecast(input("DOT", series(n, 10.0, 0.001)))
            assertTrue(f.d7.signals.any { it.name == "move_7d" }, "n=$n: ${f.d7.signals.map { it.name }}")
            assertEquals(null, f.bottom.drawdownFrom30dHighPct, "no 30-day high without daily data")
        }
    }

    @Test
    fun usdStablecoinQuotedInEuroIsStillPegged() {
        // EUR/USD noise (about 0.4 % over a week) on top of a perfect USD peg
        val hourly = List(168) { i -> 0.92 * (1 + 0.006 * sin(i / 20.0)) }
        val f = engine.forecast(input("XUSD", hourly).copy(asset = Asset("XUSD", "Example Dollar")))
        assertEquals("Pegged", f.bottom.label)
        val coin = engine.forecast(input("CALM", hourly).copy(asset = Asset("CALM", "Calm Network")))
        assertTrue(coin.bottom.label != "Pegged", "no peg hint, no peg")
    }

    @Test
    fun shortHistoryStillProducesAForecast() {
        val f = engine.forecast(input("NEW", series(10, 1.0, 0.01)))
        assertTrue(f.h24.probabilityUp in 0.15..0.85)
        assertTrue(f.coverage.notes.isNotEmpty())
    }
}
