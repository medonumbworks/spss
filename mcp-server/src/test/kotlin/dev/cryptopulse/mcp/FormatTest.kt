package dev.cryptopulse.mcp

import dev.cryptopulse.core.analysis.forecast.DefaultForecastEngine
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.BottomAssessment
import dev.cryptopulse.core.model.DexInfo
import dev.cryptopulse.core.model.DexWindow
import dev.cryptopulse.core.model.Direction
import dev.cryptopulse.core.model.Forecast
import dev.cryptopulse.core.model.Horizon
import dev.cryptopulse.core.model.HorizonForecast
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.RiskAssessment
import dev.cryptopulse.core.model.Signal
import dev.cryptopulse.core.model.Tier
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class FormatTest {
    private val now = 1_760_713_200_000L

    private fun called(h: Horizon, p: Double, signals: List<Signal>) =
        HorizonForecast(h, p, if (p > 0.5) Direction.UP else Direction.DOWN, 0.3, 0.9, 1.1, 1.0, signals)

    /** What DefaultForecastEngine.historyGate returns for a pool too young for this horizon. */
    private fun gated(h: Horizon, note: String) =
        HorizonForecast(h, 0.5, Direction.FLAT, 0.0, signals = listOf(Signal("history", "Not enough history", 50.0, 0.0, note), Signal("base_rate", "Base rate", null, 0.0, "Brand-new DEX tokens usually fall")))

    private fun forecast(h24: HorizonForecast, d7: HorizonForecast, bottom: BottomAssessment) =
        Forecast("MOON", "USD", 0.01, h24, d7, bottom, summary = "Moon.", computedAtMs = now, modelVersion = "test")

    private fun dexRow(symbol: String, currency: String, risk: RiskAssessment?, f: Forecast? = null, volumeUsd: Double? = 12_000.0) = AssetRow(
        Asset(symbol, symbol, tier = Tier.DEX),
        MarketData(symbol, currency = currency, price = 0.0088, change24hPct = 5.0),
        f,
        dex = DexInfo("solana", "mint$symbol", liquidityUsd = 40_000.0, volumeUsd = DexWindow(h24 = volumeUsd), pairCreatedAtMs = now - 50 * 3_600_000L),
        risk = risk,
    )

    @Test
    fun gatedHorizonShowsNoNumberAndZeroDriversAreNeutral() {
        val h24 = called(Horizon.H24, 0.6, listOf(
            Signal("momentum_24h", "24h momentum", 4.0, 0.2, "Rising"),
            Signal("rsi14", "RSI", 70.0, -0.1, "Overbought"),
            Signal("sentiment", "Social sentiment", null, 0.0, "No posts"),
        ))
        val d7 = gated(Horizon.D7, "Not enough history: the pool is 4 days old; a 7d call needs 3 weeks of trading")
        val text = Format.forecast("Moon", forecast(h24, d7, BottomAssessment(40, "Mid-range")), dexRow("MOON", "USD", RiskAssessment(60, "High risk")))

        assertTrue(text.contains("### Next 24h: ▲ 60% chance of going up"), text)
        assertTrue(text.contains("- + 24h momentum") && text.contains("- − RSI") && text.contains("- · Social sentiment"), text)
        assertTrue(text.contains("### Next 7d: no forecast — Not enough history: the pool is 4 days old"), text)
        assertFalse(text.contains("Next 7d: ■"), text)
        // the gated horizon's zeroed drivers are not listed
        assertFalse(text.contains("Brand-new DEX tokens usually fall"), text)
        assertEquals(1, Regex("Likely range").findAll(text).count(), text)
        assertTrue(text.contains("### Is this the bottom? Score 40/100 — Mid-range"), text)
    }

    @Test
    fun blockedTokenGetsNoProbabilityAndNoBottomScore() {
        val note = "No forecast (too new): Pool is only 5 h old"
        fun none(h: Horizon) = HorizonForecast(h, 0.5, Direction.FLAT, 0.0, signals = listOf(Signal("risk_blocked", "Safety red flags", 80.0, 0.0, note)))
        val f = forecast(none(Horizon.H24), none(Horizon.D7), BottomAssessment(50, DefaultForecastEngine.NOT_ASSESSED, reasons = listOf(note)))
        for (row in listOf(dexRow("MOON", "USD", RiskAssessment(80, "Too new", listOf("Pool is only 5 h old"), blocked = true)), null)) {
            val text = Format.forecast("Moon", f, row)
            assertFalse(text.contains("chance of going up") || text.contains("50%"), text)
            assertFalse(text.contains("Is this the bottom?") || text.contains("/100 — "), text)
            assertFalse(text.contains("Main drivers") || text.contains("+ Safety red flags"), text)
            assertTrue(text.contains("### Next 24h: no forecast — $note") && text.contains("### Next 7d: no forecast — $note"), text)
        }
    }

    @Test
    fun pricesInAnotherCurrencyAreLabelled() {
        val usd = dexRow("USDQ", "USD", RiskAssessment(60, "High risk"))
        val eur = dexRow("EURQ", "EUR", RiskAssessment(60, "High risk"))
        val overview = Format.overviewTable(listOf(usd, eur), "EUR")
        assertTrue(overview.contains("| Price (EUR) |") && overview.contains("| 0.008800 USD |"), overview)
        assertTrue(overview.contains("| 0.008800 |"), overview)
        val radar = Format.radarTable(listOf(usd, eur), "EUR", now)
        assertTrue(radar.contains("| 0.008800 USD |") && radar.contains("| 0.008800 |"), radar)
    }

    @Test
    fun coinGeckoVolumeKeepsItsCurrency() {
        val meme = AssetRow(Asset("WOOF", "Woof", tier = Tier.MEME), MarketData("WOOF", currency = "EUR", price = 0.5, volume24h = 12_000.0))
        val radar = Format.radarTable(listOf(meme, dexRow("MOON", "EUR", RiskAssessment(60, "High risk"))), "EUR", now)
        assertTrue(radar.contains("| 12K EUR |"), radar)
        assertTrue(radar.contains("| \$12K |"), "pool volume is USD: $radar")
        assertEquals("\$12K", Format.usd(12_000.0))
        assertEquals("12K CHF", Format.money(12_000.0, "CHF"))
        assertEquals("—", Format.money(null, "EUR"))
    }

    @Test
    fun placeholderBottomScoresAreHiddenInTheOverview() {
        val flat = HorizonForecast(Horizon.H24, 0.5, Direction.FLAT, 0.9)
        val pegged = AssetRow(
            Asset("USDQ", "Quoted Dollar"),
            MarketData("USDQ", currency = "EUR", price = 0.92),
            Forecast("USDQ", "EUR", 0.92, flat, flat.copy(horizon = Horizon.D7), BottomAssessment(50, "Pegged"), summary = "", computedAtMs = now, modelVersion = "test"),
        )
        val table = Format.overviewTable(listOf(pegged), "EUR")
        assertFalse(table.contains("| 50 |"), table)
    }
}
