package dev.cryptopulse.core.analysis

import dev.cryptopulse.core.analysis.indicators.Indicators
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class IndicatorsTest {
    @Test
    fun rsiExtremesAndFlat() {
        val up = (1..40).map { it.toDouble() }
        assertEquals(100.0, Indicators.rsi(up)!!, 1e-9)
        val down = up.reversed()
        assertEquals(0.0, Indicators.rsi(down)!!, 1e-9)
        assertEquals(50.0, Indicators.rsi(List(30) { 5.0 })!!, 1e-9)
        assertNull(Indicators.rsi(listOf(1.0, 2.0)))
    }

    @Test
    fun rsiMatchesWilderReference() {
        // StockCharts' Wilder example (rounded closes): gains 3.34, losses 1.40 over 14 changes -> RS 2.3857 -> RSI 70.464.
        val closes = listOf(44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28)
        assertEquals(70.464, Indicators.rsi(closes)!!, 0.001)
        // one more step uses Wilder smoothing: change +0.00 -> avgGain 0.2216, avgLoss 0.0929
        val next = Indicators.rsi(closes + 46.28)!!
        assertEquals(100.0 - 100.0 / (1 + (3.34 / 14 * 13 / 14) / (1.40 / 14 * 13 / 14)), next, 0.001)
    }

    @Test
    fun emaConvergesAndBollingerBounds() {
        val flat = List(50) { 10.0 }
        assertEquals(10.0, Indicators.lastEma(flat, 12)!!, 1e-9)
        assertEquals(0.5, Indicators.bollingerPercentB(flat)!!, 1e-9)
        val spike = List(30) { 10.0 + (it % 2) * 0.1 } + 20.0
        assertTrue(Indicators.bollingerPercentB(spike)!! > 1.0)
    }

    @Test
    fun changesDistancesResample() {
        val s = listOf(100.0, 90.0, 80.0, 88.0)
        assertEquals(10.0, Indicators.distanceToMinPct(s)!!, 1e-9)
        assertEquals(-12.0, Indicators.drawdownFromMaxPct(s)!!, 1e-9)
        assertEquals(-12.0, Indicators.pctChange(s, 3)!!, 1e-9)
        val hourly = (1..10).map { it.toDouble() }
        assertEquals(listOf(2.0, 6.0, 10.0), Indicators.resample(hourly, 4))
        assertEquals(listOf(4.0, 8.0), Indicators.resample((1..8).map { it.toDouble() }, 4))
        val macd = Indicators.macd((1..60).map { it.toDouble() })!!
        assertTrue(macd.macd > 0)
    }
}
