package dev.cryptopulse.mcp

import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.BottomAssessment
import dev.cryptopulse.core.model.Direction
import dev.cryptopulse.core.model.Forecast
import dev.cryptopulse.core.model.Horizon
import dev.cryptopulse.core.model.HorizonForecast
import dev.cryptopulse.core.model.ListingSource
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.MarketSnapshot
import dev.cryptopulse.core.model.Signal
import dev.cryptopulse.core.model.Tier
import org.junit.jupiter.api.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class DigestTest {
    private val now = 1_760_713_200_000L

    private fun called(h: Horizon, p: Double) = HorizonForecast(h, p, if (p > 0.5) Direction.UP else Direction.DOWN, 0.4, signals = listOf(Signal("rsi14", "RSI", 50.0, 0.1, "x")))

    /** What DefaultForecastEngine.lowMoveGate leaves: a 0.5 placeholder behind a "low_move" signal. */
    private fun calm(h: Horizon) = HorizonForecast(h, 0.5, Direction.FLAT, 0.0, signals = listOf(Signal("low_move", "Small expected move", 0.7, 0.0, "Typical move is only 0.70%")))

    private fun row(sym: String, h24: HorizonForecast, d7: HorizonForecast, tier: Tier = Tier.BITPANDA, bottom: Int = 50) = AssetRow(
        Asset(sym, sym, tier = tier),
        MarketData(sym, currency = "EUR", price = 1.0, change24hPct = 1.0),
        Forecast(sym, "EUR", 1.0, h24, d7, BottomAssessment(bottom, "Near its recent lows"), summary = "", computedAtMs = now, modelVersion = "t"),
    )

    @Test
    fun aCoinWithoutACallIsNeverRankedByItsPlaceholder() {
        // every called coin is likely up, so a 0.5 placeholder would top "highest odds to fall"
        val rows = listOf(
            row("AAA", called(Horizon.H24, 0.62), called(Horizon.D7, 0.6)),
            row("BBB", called(Horizon.H24, 0.58), called(Horizon.D7, 0.57)),
            row("TRX", calm(Horizon.H24), calm(Horizon.D7), bottom = 80),
            row("HOT", called(Horizon.H24, 0.55), called(Horizon.D7, 0.52), tier = Tier.TRENDING),
        )
        val snap = MarketSnapshot("EUR", rows, universeSource = ListingSource.BITPANDA_LEGACY_TICKER, universeSize = rows.size, fetchedAtMs = now)
        for (lang in listOf("en", "hr")) {
            val text = Digest.build(snap, lang)
            assertFalse(Regex("• TRX 50%").containsMatchIn(text), text)
            assertTrue(text.contains("• TRX 80/100, 7d —"), "the bottom line shows no 7-day number: $text")
            assertFalse(text.contains("DEX radar"), "only trending coins beyond Bitpanda: $text")
        }
        assertTrue(Digest.build(snap, "en").contains("Trending, not on Bitpanda"))
        assertTrue(Digest.build(snap, "hr").contains("U trendu, nisu na Bitpandi"))
    }
}
