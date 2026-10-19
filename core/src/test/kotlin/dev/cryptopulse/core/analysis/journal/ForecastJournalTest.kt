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
import dev.cryptopulse.core.model.Tier
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

    /** A bare price quote for evaluation: no forecast needed to score what is already logged. */
    private fun priced(symbol: String, price: Double, currency: String = "EUR") =
        AssetRow(Asset(symbol, symbol), MarketData(symbol, currency = currency, price = price))

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
        assertEquals(3, journal.record(rows), "BTC both horizons + NEW 24h")
        assertEquals(0, journal.record(rows), "same moment: everything within its stride")
        now[0] += 6 * hour
        assertEquals(2, journal.record(rows), "6 h later only the 24 h entries stride again")
        now[0] += 18 * hour
        assertEquals(3, journal.record(rows), "24 h after the first record the 7 d entry strides too")
        assertEquals(8, journal.stats().total)
    }

    @Test
    fun evaluateScoresUpAndDownCallsBothWays() {
        journal.record(
            listOf(
                row("UPHIT", 100.0, 0.7, null), row("UPMISS", 100.0, 0.7, null),
                row("DOWNHIT", 100.0, 0.3, null), row("DOWNMISS", 100.0, 0.3, null),
            ),
        )
        val prices = listOf(priced("UPHIT", 110.0), priced("UPMISS", 90.0), priced("DOWNHIT", 90.0), priced("DOWNMISS", 110.0))
        assertEquals(0, journal.evaluate(prices), "nothing is due yet")
        now[0] += 24 * hour
        assertEquals(4, journal.evaluate(prices))
        val s = journal.stats().perHorizon["24h"]!!
        assertEquals(0, s.pending)
        assertEquals(4, s.evaluated)
        assertEquals(2, s.hits)
        assertEquals(0.5, s.hitRate)
        assertEquals(0.5, s.upShare)
    }

    @Test
    fun aCoinFlipCallIsPricedButNeverScored() {
        journal.record(listOf(row("FLIP", 100.0, 0.5, null)))
        now[0] += 24 * hour
        assertEquals(0, journal.evaluate(listOf(priced("FLIP", 120.0))), "p = 0.5 called no direction")
        val s = journal.stats().perHorizon["24h"]!!
        assertEquals(0, s.pending, "the entry got its realised price, it no longer waits")
        assertEquals(0, s.evaluated, "but without a direction there is nothing to count as evaluated")
        assertNull(s.hitRate)
        assertEquals(1, journal.stats().total)
        assertEquals(0, journal.evaluate(listOf(priced("FLIP", 120.0))), "it is not re-scored either")
    }

    @Test
    fun everythingIsDroppedHalfAHorizonPastItsDueDate() {
        journal.record(listOf(row("GONE", 100.0, 0.7, null), row("LATE", 100.0, 0.3, null)))
        now[0] += 24 * hour + 1
        // due, but no usable price: a missing symbol and a price quoted in another currency both wait
        assertEquals(0, journal.evaluate(listOf(priced("LATE", 150.0, currency = "USD"))))
        assertEquals(2, journal.stats().perHorizon["24h"]!!.pending)
        now[0] += 12 * hour // past due + half the horizon
        assertEquals(0, journal.evaluate(listOf(priced("LATE", 150.0))), "a price seen half a horizon late measures a different horizon")
        assertEquals(0, journal.stats().total, "both entries were given up on, not scored")
    }

    @Test
    fun entriesFollowTheirTokenAndItsOwnCurrency() {
        // a DEX row stays quoted in USD inside an EUR snapshot, and its ticker is not unique
        val mintA = Asset("XY", "Xy One", tier = Tier.DEX, chainId = "solana", contract = "MintA")
        journal.record(listOf(AssetRow(mintA, MarketData("XY", currency = "USD", price = 100.0), row("XY", 100.0, 0.7, null).forecast)))
        now[0] += 24 * hour
        val mintB = Asset("XY", "Xy Two", tier = Tier.DEX, chainId = "solana", contract = "MintB")
        assertEquals(0, journal.evaluate(listOf(AssetRow(mintB, MarketData("XY", currency = "USD", price = 200.0)))), "another token took the ticker")
        assertEquals(0, journal.evaluate(listOf(AssetRow(mintA, MarketData("XY", currency = "EUR", price = 95.0)))), "scored in its own currency only")
        assertEquals(1, journal.evaluate(listOf(AssetRow(mintA, MarketData("XY", currency = "USD", price = 130.0)))))
        assertEquals(1, journal.stats().perHorizon["24h"]!!.hits)
    }

    @Test
    fun aNewJournalPicksUpWhereTheOldOneLeftOff() {
        journal.record(listOf(row("BTC", 100.0, 0.7, 0.6)))
        val reloaded = ForecastJournal(store, clock)
        assertEquals(0, reloaded.record(listOf(row("BTC", 100.0, 0.7, 0.6))), "the stride holds across restarts")
        assertEquals(2, reloaded.stats().total)
        now[0] += 24 * hour
        assertEquals(1, reloaded.evaluate(listOf(priced("BTC", 110.0))), "only the 24 h entry is due")
        assertEquals(1, ForecastJournal(store, clock).stats().perHorizon["24h"]!!.hits, "the verdict survives another reload")
    }

    @Test
    fun aCorruptBlobStartsTheJournalEmpty() {
        store.write(ForecastJournal.STORE_KEY, "{not json")
        assertEquals(0, journal.stats().total)
        assertEquals(1, journal.record(listOf(row("BTC", 100.0, 0.7, null))), "and it is writable again")
    }

    @Test
    fun trimDropsOldestEvaluatedEntriesBeforePendingOnes() {
        val journal = ForecastJournal(store, clock, maxEntries = 2)
        journal.record(listOf(row("A", 100.0, 0.7, null)))
        now[0] += 24 * hour
        assertEquals(1, journal.evaluate(listOf(priced("A", 110.0))))
        journal.record(listOf(row("B", 100.0, 0.7, null))) // T1
        now[0] += 7 * hour
        journal.record(listOf(row("C", 100.0, 0.7, null))) // over the cap: scored A goes first
        assertEquals(2, journal.stats().total)
        assertEquals(0, journal.stats().perHorizon["24h"]!!.evaluated, "A is gone")
        now[0] += 7 * hour
        journal.record(listOf(row("D", 100.0, 0.7, null))) // over again: nothing scored, oldest pending B goes
        assertEquals(2, journal.stats().total)
        now[0] += 10 * hour + 1 // B would be due now (24 h after T1)
        assertEquals(0, journal.evaluate(listOf(priced("B", 200.0))), "B was trimmed, there is nothing to score")
        now[0] += 7 * hour // C due (24 h after its record)
        assertEquals(1, journal.evaluate(listOf(priced("C", 200.0))), "C survived the trims")
    }

    @Test
    fun statsMathOnAHandBuiltSet() {
        // 0.8 up call that went up, 0.6 up call that went down, 0.3 down call that went down, one still pending
        journal.record(
            listOf(row("A", 100.0, 0.8, null), row("B", 100.0, 0.6, null), row("C", 100.0, 0.3, null), row("P", 100.0, 0.7, null)),
        )
        now[0] += 24 * hour
        assertEquals(3, journal.evaluate(listOf(priced("A", 120.0), priced("B", 80.0), priced("C", 90.0))))
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
