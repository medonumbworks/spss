package dev.cryptopulse.app.ui

import dev.cryptopulse.core.analysis.ForecastInput
import dev.cryptopulse.core.analysis.forecast.DefaultForecastEngine
import dev.cryptopulse.core.analysis.risk.RiskScorer
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.DexInfo
import dev.cryptopulse.core.model.DexTxns
import dev.cryptopulse.core.model.DexWindow
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.RiskAssessment
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.model.TokenSafety
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Red-flagged and too-young tokens get no "chance of going up" on the detail screen or in the list. */
class NoCallTest {
    private val now = 1_792_411_200_000L
    private val day = 86_400_000L
    private val engine = DefaultForecastEngine()
    private val clean = TokenSafety(source = "goplus", honeypot = false, buyTaxPct = 0.0, sellTaxPct = 0.0, mintable = false, freezable = false, openSource = true)

    /** A DEX pool [ageDays] old, forecast by the real engine. */
    private fun dexRow(sym: String, ageDays: Double, safety: TokenSafety): AssetRow {
        val asset = Asset(sym, "$sym Token", tier = Tier.DEX, chainId = "solana", contract = "Mint$sym")
        val dex = DexInfo(
            chainId = "solana", tokenAddress = "Mint$sym", liquidityUsd = 400_000.0, marketCapUsd = 8_000_000.0,
            priceChangePct = DexWindow(h1 = 2.0, h6 = 5.0, h24 = 12.0), volumeUsd = DexWindow(h1 = 40_000.0, h24 = 600_000.0),
            txnsH6 = DexTxns(900, 700), txnsH24 = DexTxns(2700, 2100), pairCreatedAtMs = now - (ageDays * day).toLong(),
        )
        // a gentle climb with about a 2 % daily swing: calmer pools get no call at all (DefaultForecastEngine.LOW_MOVE_PCT)
        val hourly = (0 until minOf(168, (ageDays * 24).toInt())).map { 0.01 * (1 + 0.001 * it) * (1 + 0.02 * sin(it / 3.0)) }
        val market = MarketData(sym, currency = "EUR", price = hourly.last(), marketCap = 8e6, change24hPct = 12.0, sparkline7d = hourly)
        val risk = RiskScorer.assess(Tier.DEX, market, dex, safety, now)
        val forecast = engine.forecast(ForecastInput(asset, market, hourly, dex = dex, risk = risk, nowMs = now))
        return AssetRow(asset, market, forecast = forecast, dex = dex, safety = safety, risk = risk)
    }

    @Test
    fun honeypotGetsReasonsInsteadOfFiftyPercent() {
        val row = dexRow("TRAP", 40.0, clean.copy(honeypot = true))
        val f = row.forecast!!
        assertTrue(row.risk!!.blocked)
        val reason = assertNotNull(noCallReason(row, f.h24))
        assertTrue(reason.startsWith("No forecast"), reason)
        assertNotNull(noCallReason(row, f.d7))
        assertNull(row.upOrNull(f.d7), "no 7-day rebound chance either")
        assertEquals(DefaultForecastEngine.NOT_ASSESSED, f.bottom.label, "the bottom card hides its placeholder score")
    }

    @Test
    fun youngPoolSaysNotEnoughHistory() {
        val row = dexRow("BABY", 2.0, clean)
        val f = row.forecast!!
        assertFalse(row.risk!!.blocked, "${row.risk}")
        assertTrue(noCallReason(row, f.h24)!!.startsWith("Not enough history"), noCallReason(row, f.h24))
        assertTrue(noCallReason(row, f.d7)!!.startsWith("Not enough history"), noCallReason(row, f.d7))
        assertNull(row.upOrNull(f.d7))
    }

    @Test
    fun establishedPoolKeepsItsNumber() {
        val row = dexRow("OLD", 40.0, clean)
        assertFalse(row.risk!!.blocked, "${row.risk}")
        assertNull(noCallReason(row, row.forecast!!.h24))
        assertNull(noCallReason(row, row.forecast!!.d7))
        // a risk block that arrives after the forecast still hides the number
        val flagged = row.copy(risk = RiskAssessment(100, RiskScorer.LABEL_RED_FLAG, blocked = true))
        assertEquals("No forecast (red flags)", noCallReason(flagged, flagged.forecast!!.h24))
    }

    @Test
    fun listNeverRanksAPlaceholder() {
        val old = dexRow("OLD", 40.0, clean)
        val young = dexRow("BABY", 2.0, clean).let { r -> r.copy(forecast = r.forecast!!.copy(h24 = r.forecast!!.h24.copy(probabilityUp = 0.9))) }
        val trap = dexRow("TRAP", 40.0, clean.copy(honeypot = true))
        val rows = listOf(young, trap, old)
        assertEquals(listOf("OLD", "BABY", "TRAP"), ListOptions.apply(rows, "", ListFilter.ALL, ListSort.UP_24H).map { it.asset.symbol })
        assertTrue(ListOptions.apply(rows, "", ListFilter.UP_24H, ListSort.MARKET_CAP).none { it.asset.symbol == "BABY" }, "a no-call horizon never counts as likely up")
    }
}
