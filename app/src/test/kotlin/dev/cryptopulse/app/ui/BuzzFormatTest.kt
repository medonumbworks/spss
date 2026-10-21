package dev.cryptopulse.app.ui

import kotlin.test.Test
import kotlin.test.assertEquals

/** The buzz row's trend label, pure and locale-stable. */
class BuzzFormatTest {
    @Test
    fun spikeWinsOverThePreviousWindow() {
        assertEquals("×2.1 vs 7-day norm", buzzTrendLabel(spike = 2.1, mentionsPrev = 12))
    }

    @Test
    fun spikeRoundsToOneDecimal() {
        assertEquals("×0.5 vs 7-day norm", buzzTrendLabel(spike = 0.49, mentionsPrev = null))
        assertEquals("×12.0 vs 7-day norm", buzzTrendLabel(spike = 12.0, mentionsPrev = null))
    }

    @Test
    fun withoutABaselineTheLabelFallsBackToThePreviousWindow() {
        assertEquals("prev window: 7", buzzTrendLabel(spike = null, mentionsPrev = 7))
        assertEquals("prev window: 0", buzzTrendLabel(spike = null, mentionsPrev = null))
    }
}
