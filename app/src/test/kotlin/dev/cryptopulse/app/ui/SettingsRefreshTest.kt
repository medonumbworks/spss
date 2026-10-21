package dev.cryptopulse.app.ui

import dev.cryptopulse.core.config.AppConfig
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SettingsRefreshTest {
    private val base = AppConfig()

    @Test
    fun listSettingsReloadAtOnce() {
        assertTrue(needsRefresh(base, base.copy(dexMinLiquidityUsd = 200_000.0)), "the minimum-liquidity slider")
        assertTrue(needsRefresh(base, base.copy(dexMinVolume24hUsd = 50_000.0)))
        assertTrue(needsRefresh(base, base.copy(dexRadarMax = 50)))
        assertTrue(needsRefresh(base, base.copy(memecoinCount = 100)))
        assertTrue(needsRefresh(base, base.copy(showTrending = !base.showTrending)))
        assertTrue(needsRefresh(base, base.copy(safetyChecks = false)))
        assertTrue(needsRefresh(base, base.copy(currency = "USD")))
    }

    @Test
    fun otherSettingsWaitForTheNextRefresh() {
        assertFalse(needsRefresh(base, base))
        assertFalse(needsRefresh(base, base.copy(backgroundSentimentTopN = base.backgroundSentimentTopN + 10)))
    }
}
