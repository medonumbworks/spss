package dev.cryptopulse.core.analysis.journal

import dev.cryptopulse.core.FIXTURE_NOW_MS
import dev.cryptopulse.core.http.InMemoryDiskStore
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.BottomAssessment
import dev.cryptopulse.core.model.Direction
import dev.cryptopulse.core.model.Forecast
import dev.cryptopulse.core.model.Horizon
import dev.cryptopulse.core.model.HorizonForecast
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.Signal
import dev.cryptopulse.core.util.Clock
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/** Recording strides, hit scoring, the unevaluable grace window, persistence, trimming and stats math. */
class ForecastJournalTest {
    private val hour = 3_600_000L
    private val now = longArrayOf(FIXTURE_NOW_MS)
    private val clock = Clock { now[0] }
    private val store = InMemoryDiskStore()
    private val journal = ForecastJournal(store, clock)

    private fun hf(h: Horizon, p: Double?) = HorizonForecast(
        horizon = h,
        probabilityUp = p ?: 0.5,
        direction = if ((p ?: 0.5) > 0.5) Direction.UP else if ((p ?: 0.5) < 0.5) Direction.DOWN else Direction.FLAT,
        confidence = 0.5,
        // null p = no call: the first signal name marks the horizon as not callable
        signals = if (p == null) listOf(Signal("history", "History", contribution = 0.0, description = "too little history")) else emptyList(),
    )

    /** A market row with a price and a forecast; a null probability means that horizon has no call. */
    private fun row(symbol: String, price: Double, p24: Double?, p7d: Double?, currency: String = "EUR") = AssetRow(
        asset = Asset(symbol, symbol),
        market = MarketData(symbol, currency = currency, price = price),
        forecast = Forecast(
            symbol = symbol,
            currency = currency,
            price = price,
            h24 = hf(Horizon.H24, p24),
            d7 = hf(Horizon.D7, p7d),
            bottom = BottomAssessment(50, "Mid-range"),
            summary = "",
            computedAtMs = now[0],
            modelVersion = "test-model-1",
        ),
    )

    @Test
    fun recordsOnlyCalledHorizonsOncePerStride() {
        val rows = listOf(
            row("BTC", 100.0, p24 = 0.62, p7d = 0.58),
            row("NEW", 1.0, p24 = 0.55, p7d = null), // no 7d call: never logged
            row("NOPRICE", 2.0, 0.6, 0.6).copy(market = null),
            row("NOFC", 2.0, 0.6, 0.6).copy(forecast = null),
        )
        assertEquals(3, journal.record(rows, "EUR"), "BTC both horizons + NEW 24h")
        assertEquals(0, journal.record(rows, "EUR"), "same moment: everything within its stride")
        now[0] += 6 * hour
        assertEquals(2, journal.record(rows, "EUR"), "6 h later only the 24 h entries stride again")
        now[0] += 18 * hour
        assertEquals(3, journal.record(rows, "EUR"), "24 h after the first record the 7 d entry strides too")
        assertEquals(8, journal.stats().total)
    }

    @Test
    fun evaluateScoresUpAndDownCallsBothWays() {
        journal.record(
            listOf(
                row("UPHIT", 100.0, 0.7, null), row("UPMISS", 100.0, 0.7, null),
                row("DOWNHIT", 100.0, 0.3, null), row("DOWNMISS", 100.0, 0.3, null),
            ),
            "EUR",
        )
        val prices = mapOf("UPHIT" to 110.0, "UPMISS" to 90.0, "DOWNHIT" to 90.0, "DOWNMISS" to 110.0)
        assertEquals(0, journal.evaluate(prices, "EUR"), "nothing is due yet")
        now[0] += 24 * hour
        assertEquals(4, journal.evaluate(prices, "EUR"))
        val s = journal.stats().perHorizon["24h"]!!
        assertEquals(0, s.pending)
        assertEquals(4, s.evaluated)
        assertEquals(2, s.hits)
        assertEquals(0.5, s.hitRate)
        assertEquals(0.5, s.upShare)
    }

    @Test
    fun aCoinFlipCallIsPricedButNeverScored() {
        journal.record(listOf(row("FLIP", 100.0, 0.5, null)), "EUR")
        now[0] += 24 * hour
        assertEquals(0, journal.evaluate(mapOf("FLIP" to 120.0), "EUR"), "p = 0.5 called no direction")
        val s = journal.stats().perHorizon["24h"]!!
        assertEquals(0, s.pending, "the entry got its realised price, it no longer waits")
        assertEquals(0, s.evaluated, "but without a direction there is nothing to count as evaluated")
        assertNull(s.hitRate)
        assertEquals(1, journal.stats().total)
        assertEquals(0, journal.evaluate(mapOf("FLIP" to 120.0), "EUR"), "it is not re-scored either")
    }

    @Test
    fun unevaluableEntriesAreDroppedOnlyAfterTheGraceWindow() {
        journal.record(listOf(row("GONE", 100.0, 0.7, null), row("LATE", 100.0, 0.3, null)), "EUR")
        now[0] += 24 * hour + 1
        // due, but no usable price: a missing symbol and a price quoted in another currency both wait
        assertEquals(0, journal.evaluate(mapOf("LATE" to 150.0), "USD"))
        assertEquals(2, journal.stats().perHorizon["24h"]!!.pending)
        now[0] += 12 * hour // past due + half the horizon
        assertEquals(1, journal.evaluate(mapOf("LATE" to 150.0), "EUR"), "a late price still scores")
        assertEquals(0, journal.stats().perHorizon["24h"]!!.pending, "GONE was given up on")
        assertEquals(1, journal.stats().total)
        assertEquals(0, journal.stats().perHorizon["24h"]!!.hits, "LATE called down and the price went up: a miss")
    }

    @Test
    fun aNewJournalPicksUpWhereTheOldOneLeftOff() {
        journal.record(listOf(row("BTC", 100.0, 0.7, 0.6)), "EUR")
        val reloaded = ForecastJournal(store, clock)
        assertEquals(0, reloaded.record(listOf(row("BTC", 100.0, 0.7, 0.6)), "EUR"), "the stride holds across restarts")
        assertEquals(2, reloaded.stats().total)
        now[0] += 24 * hour
        assertEquals(1, reloaded.evaluate(mapOf("BTC" to 110.0), "EUR"), "only the 24 h entry is due")
        assertEquals(1, ForecastJournal(store, clock).stats().perHorizon["24h"]!!.hits, "the verdict survives another reload")
    }

    @Test
    fun aCorruptBlobStartsTheJournalEmpty() {
        store.write(ForecastJournal.STORE_KEY, "{not json")
        assertEquals(0, journal.stats().total)
        assertEquals(1, journal.record(listOf(row("BTC", 100.0, 0.7, null)), "EUR"), "and it is writable again")
    }

    @Test
    fun trimDropsOldestEvaluatedEntriesBeforePendingOnes() {
        val journal = ForecastJournal(store, clock, maxEntries = 2)
        journal.record(listOf(row("A", 100.0, 0.7, null)), "EUR")
        now[0] += 24 * hour
        assertEquals(1, journal.evaluate(mapOf("A" to 110.0), "EUR"))
        journal.record(listOf(row("B", 100.0, 0.7, null)), "EUR") // T1
        now[0] += 7 * hour
        journal.record(listOf(row("C", 100.0, 0.7, null)), "EUR") // over the cap: scored A goes first
        assertEquals(2, journal.stats().total)
        assertEquals(0, journal.stats().perHorizon["24h"]!!.evaluated, "A is gone")
        now[0] += 7 * hour
        journal.record(listOf(row("D", 100.0, 0.7, null)), "EUR") // over again: nothing scored, oldest pending B goes
        assertEquals(2, journal.stats().total)
        now[0] += 10 * hour + 1 // B would be due now (24 h after T1)
        assertEquals(0, journal.evaluate(mapOf("B" to 200.0), "EUR"), "B was trimmed, there is nothing to score")
        now[0] += 7 * hour // C due (24 h after its record)
        assertEquals(1, journal.evaluate(mapOf("C" to 200.0), "EUR"), "C survived the trims")
    }

    @Test
    fun statsMathOnAHandBuiltSet() {
        // 0.8 up call that went up, 0.6 up call that went down, 0.3 down call that went down, one still pending
        journal.record(
            listOf(row("A", 100.0, 0.8, null), row("B", 100.0, 0.6, null), row("C", 100.0, 0.3, null), row("P", 100.0, 0.7, null)),
            "EUR",
        )
        now[0] += 24 * hour
        assertEquals(3, journal.evaluate(mapOf("A" to 120.0, "B" to 80.0, "C" to 90.0), "EUR"))
        val s = journal.stats().perHorizon["24h"]!!
        assertEquals(1, s.pending)
        assertEquals(3, s.evaluated)
        assertEquals(2, s.hits)
        assertEquals(2.0 / 3.0, s.hitRate!!, 1e-9)
        assertEquals((0.8 + 0.6 + 0.3) / 3.0, s.avgProbability!!, 1e-9)
        // outcomes 1, 0, 0: mean of (0.8-1)^2, 0.6^2, 0.3^2
        assertEquals((0.04 + 0.36 + 0.09) / 3.0, s.brier!!, 1e-9)
        assertEquals(1.0 / 3.0, s.upShare!!, 1e-9)
        val d7 = journal.stats().perHorizon["7d"]!!
        assertEquals(0, d7.evaluated)
        assertNull(d7.hitRate)
        assertEquals(4, journal.stats().total)
    }
}
