package dev.cryptopulse.app.work

import dev.cryptopulse.core.analysis.journal.ForecastAlert
import kotlin.test.Test
import kotlin.test.assertEquals

/** The notification wording is pure string building, tested without Android. */
class AlertTextTest {
    private fun alert(symbol: String, p: Double) = ForecastAlert(symbol, "$symbol Coin", p, 0.6, 100.0, "EUR")

    @Test
    fun titleUsesTheSingularForOneCoin() {
        assertEquals("1 coin crossed 50% up-chance (24h)", AlertText.title(1, 50))
        assertEquals("3 coins crossed 65% up-chance (24h)", AlertText.title(3, 65))
    }

    @Test
    fun summaryListsSymbolsWithRoundedPercentages() {
        val alerts = listOf(alert("BTC", 0.57), alert("SOL", 0.561))
        assertEquals("BTC 57%, SOL 56%", AlertText.summary(alerts))
    }

    @Test
    fun collapsedSummaryStopsAtSixCoinsWithAnEllipsis() {
        val alerts = (1..8).map { alert("C$it", 0.50 + it / 100.0) }
        val collapsed = AlertText.summary(alerts)
        assertEquals("C1 51%, C2 52%, C3 53%, C4 54%, C5 55%, C6 56%, …", collapsed)
        // the BigText expansion carries every coin and no ellipsis
        val full = AlertText.summary(alerts, limit = alerts.size)
        assertEquals("C1 51%, C2 52%, C3 53%, C4 54%, C5 55%, C6 56%, C7 57%, C8 58%", full)
    }

    @Test
    fun summaryOfExactlySixCoinsHasNoEllipsis() {
        val alerts = (1..6).map { alert("C$it", 0.6) }
        assertEquals(5, AlertText.summary(alerts).count { it == ',' })
    }
}
