package dev.cryptopulse.core.analysis.journal

import dev.cryptopulse.core.http.DiskStore
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.Forecast
import dev.cryptopulse.core.model.Horizon
import dev.cryptopulse.core.model.HorizonForecast
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.util.Clock
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json

/** One issued forecast, frozen at issue time so it can be scored honestly against the realised price. */
@Serializable
data class JournalEntry(
    val symbol: String,
    /** [dev.cryptopulse.core.model.Asset.tokenKey]: beyond Bitpanda a ticker can move to another token. */
    val token: String = "",
    val tier: Tier,
    val horizon: Horizon,
    /** P(up) exactly as the model said it, including the calibration of the day. */
    val probabilityUp: Double,
    /** Price at issue time in [currency]; the entry is only ever scored against the same currency. */
    val price: Double,
    val currency: String,
    val atMs: Long,
    /** When the horizon ends and the entry becomes due for scoring. */
    val dueMs: Long,
    /** Model version that issued the forecast, so recalibrations can be told apart in the stats. */
    val modelVersion: String,
    /** Typical move (1 sigma, %) over the horizon when the call was made; half of it is the noise band. */
    val movePct: Double? = null,
    val evalPrice: Double? = null,
    val evalAtMs: Long? = null,
    /**
     * True when the called direction matched the realised one. Null while the entry is pending,
     * and also after scoring when [probabilityUp] was exactly 0.5 (no direction was called).
     */
    val hit: Boolean? = null,
)

/**
 * Realised accuracy of one horizon. "Evaluated" counts only entries with a directional verdict
 * ([JournalEntry.hit] != null): a scored 50:50 call is neither pending nor evaluated, it only
 * raises [JournalStats.total]. All rates are null when nothing was evaluated yet.
 */
@Serializable
data class HorizonRecord(
    /** Recorded entries still waiting for their due date or a usable price. */
    val pending: Int,
    val evaluated: Int,
    val hits: Int,
    val hitRate: Double?,
    /** Mean issued P(up) of the evaluated entries, to compare against [upShare]. */
    val avgProbability: Double?,
    /** Mean (p - outcome)^2 with outcome 1.0 when the price ended higher; 0.25 is coin-flip. */
    val brier: Double?,
    /** Share of evaluated entries whose price actually ended higher. */
    val upShare: Double?,
    /** Priced calls whose move stayed inside the noise band (half the typical move): not scored. */
    val neutral: Int = 0,
)

/** [perHorizon] is keyed by [Horizon.label] ("24h", "7d"); [total] counts every stored entry. */
@Serializable
data class JournalStats(val perHorizon: Map<String, HorizonRecord>, val total: Int)

/**
 * Logs every issued forecast and later measures the realised hit rate per horizon - the backtest
 * only replays history, this journal scores what the model actually told the user.
 *
 * [record] appends one entry per called horizon of each row, at most once per stride (6 h for 24 h
 * forecasts, 24 h for 7 d ones) so overlapping snapshots do not flood the journal with near-copies.
 * [evaluate] scores due entries against current prices and drops entries that stayed unevaluable
 * (price gone or currency switched) for half a horizon past their due date.
 *
 * Entries persist as JSON under [STORE_KEY]; a missing or corrupt blob simply starts the journal
 * empty. The journal is capped at [maxEntries] by dropping the oldest scored entries first and
 * the oldest pending ones only when that is not enough. Mutating methods are synchronized: the
 * repository calls them from one coroutine, but a cheap lock keeps misuse harmless.
 */
class ForecastJournal(
    private val store: DiskStore?,
    private val clock: Clock = Clock.SYSTEM,
    private val maxEntries: Int = 20_000,
) {
    /** Loaded lazily so construction never touches the disk; guarded by [lock]. */
    private var entries: MutableList<JournalEntry>? = null
    private val lock = Any()

    /**
     * Logs the called horizons of [rows] (price known, forecast present, [AssetRow.upOrNull] != null;
     * risk-blocked and no-call horizons are never logged). A (symbol, horizon) pair whose newest entry
     * is younger than the horizon's stride is skipped. Returns how many entries were added.
     */
    fun record(rows: List<AssetRow>): Int = synchronized(lock) {
        val list = loaded()
        val nowMs = clock.nowMs()
        // newest issue time per (token, horizon) so the stride holds across restarts too
        val newest = HashMap<String, Long>()
        for (e in list) {
            val key = e.token + "|" + e.horizon.name
            if (e.atMs > (newest[key] ?: Long.MIN_VALUE)) newest[key] = e.atMs
        }
        var added = 0
        for (row in rows) {
            val market = row.market ?: continue
            val price = market.price ?: continue
            val forecast = row.forecast ?: continue
            for (horizon in Horizon.entries) {
                val p = row.upOrNull(forecast.at(horizon)) ?: continue
                val key = row.asset.tokenKey + "|" + horizon.name
                val last = newest[key]
                if (last != null && nowMs - last < strideMs(horizon)) continue
                list += JournalEntry(
                    symbol = row.asset.symbol,
                    token = row.asset.tokenKey,
                    tier = row.asset.tier,
                    horizon = horizon,
                    probabilityUp = p,
                    price = price,
                    // the row's own currency: a DEX row can stay quoted in USD inside an EUR snapshot
                    currency = market.currency,
                    atMs = nowMs,
                    dueMs = nowMs + horizon.hours * HOUR_MS,
                    modelVersion = forecast.modelVersion,
                    movePct = forecast.at(horizon).let { h ->
                        val lo = h.expectedLow; val hi = h.expectedHigh
                        if (lo != null && hi != null && price > 0) (hi - lo) / 2 / price * 100 else null
                    },
                )
                newest[key] = nowMs
                added++
            }
        }
        if (added > 0) {
            trim(list)
            persist(list)
        }
        added
    }

    /**
     * Scores every pending entry whose due date has passed against the same token's current price, in
     * the entry's own currency. A 50:50 call gets its realised price but hit = null and does not count
     * as evaluated. An entry more than half a horizon past its due date is dropped, scorable or not: a
     * price seen that late measures a different horizon. Returns how many got a directional verdict.
     */
    fun evaluate(rows: List<AssetRow>): Int = synchronized(lock) {
        val list = loaded()
        val nowMs = clock.nowMs()
        val byToken = HashMap<String, MarketData>()
        // calls are only made on market data with a price history; a Bitpanda-only quote (other source,
        // with its spread) or a row kept from an older refresh would score them against the wrong price
        for (row in rows) row.market?.takeIf { m ->
            m.price != null && m.source != BITPANDA_ONLY_SOURCE && (m.lastUpdatedMs == null || nowMs - m.lastUpdatedMs <= MAX_QUOTE_AGE_MS)
        }?.let { byToken[row.asset.tokenKey] = it }
        var scored = 0
        var changed = false
        val iter = list.listIterator()
        while (iter.hasNext()) {
            val e = iter.next()
            if (e.evalAtMs != null || e.dueMs > nowMs) continue
            if (nowMs > e.dueMs + e.horizon.hours * HOUR_MS / 2) {
                // too late to score fairly, and unevaluable entries (token gone, currency switched) end here too
                iter.remove()
                changed = true
                continue
            }
            // entries from before the token field match any token with their ticker
            val m = byToken[e.token] ?: if (e.token.isEmpty()) byToken.entries.firstOrNull { it.key.contains("|" + e.symbol + "|") || it.key.endsWith("|" + e.symbol) }?.value else null
            val price = m?.price
            if (price != null && e.currency == m.currency) {
                // a move inside half the typical move is noise: neither a hit nor a miss
                val movedPct = kotlin.math.abs(price / e.price - 1) * 100
                val noise = e.movePct != null && movedPct < e.movePct * NOISE_BAND
                val hit = if (e.probabilityUp == 0.5 || noise) null else (price > e.price) == (e.probabilityUp > 0.5)
                iter.set(e.copy(evalPrice = price, evalAtMs = nowMs, hit = hit))
                if (hit != null) scored++
                changed = true
            }
        }
        if (changed) persist(list)
        scored
    }

    /** Realised accuracy per horizon; see [HorizonRecord] for what counts as evaluated. */
    fun stats(): JournalStats = synchronized(lock) {
        val list = loaded()
        val perHorizon = LinkedHashMap<String, HorizonRecord>()
        for (horizon in Horizon.entries) {
            val all = list.filter { it.horizon == horizon }
            val done = all.filter { it.hit != null }
            val n = done.size
            val hits = done.count { it.hit == true }
            val ups = done.count { it.evalPrice!! > it.price }
            perHorizon[horizon.label] = HorizonRecord(
                pending = all.count { it.evalAtMs == null },
                neutral = all.count { it.evalAtMs != null && it.hit == null && it.probabilityUp != 0.5 },
                evaluated = n,
                hits = hits,
                hitRate = if (n == 0) null else hits.toDouble() / n,
                avgProbability = if (n == 0) null else done.sumOf { it.probabilityUp } / n,
                brier = if (n == 0) null else done.sumOf { e ->
                    val outcome = if (e.evalPrice!! > e.price) 1.0 else 0.0
                    (e.probabilityUp - outcome) * (e.probabilityUp - outcome)
                } / n,
                upShare = if (n == 0) null else ups.toDouble() / n,
            )
        }
        JournalStats(perHorizon, list.size)
    }

    // ------------------------------------------------------------------------------------------

    /** Scored entries have served their purpose: drop them oldest first, pending ones only after that. */
    private fun trim(list: MutableList<JournalEntry>) {
        var excess = list.size - maxEntries
        if (excess <= 0) return
        val drop = HashSet<Int>()
        for (wantScored in listOf(true, false)) {
            // entries are appended in time order, so a forward scan finds the oldest first
            for (i in list.indices) {
                if (excess == 0) break
                if ((list[i].evalAtMs != null) == wantScored) {
                    drop += i
                    excess--
                }
            }
        }
        val kept = list.filterIndexedTo(ArrayList(maxEntries)) { i, _ -> i !in drop }
        list.clear()
        list += kept
    }

    private fun loaded(): MutableList<JournalEntry> {
        entries?.let { return it }
        val raw = store?.read(STORE_KEY)
        val list: MutableList<JournalEntry> = if (raw == null) {
            ArrayList()
        } else {
            try {
                json.decodeFromString(serializer, raw).toMutableList()
            } catch (_: Exception) {
                ArrayList() // corrupt blob: better an empty journal than a crash loop
            }
        }
        entries = list
        return list
    }

    private fun persist(list: List<JournalEntry>) {
        val s = store ?: return
        try {
            s.write(STORE_KEY, json.encodeToString(serializer, list))
        } catch (_: Exception) {
            // disk is best effort, like the HTTP cache
        }
    }

    companion object {
        /** A realised move under this share of the typical move counts as noise, not as a hit or a miss. */
        const val NOISE_BAND = 0.5

        const val STORE_KEY = "forecast_journal"
        private const val HOUR_MS = 3_600_000L
        /** MarketData.source of a row that has only the Bitpanda ticker price. */
        private const val BITPANDA_ONLY_SOURCE = "bitpanda"
        /** A quote older than this (a row reused while CoinGecko fails) does not score a forecast. */
        private const val MAX_QUOTE_AGE_MS = 2 * HOUR_MS

        private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
        private val serializer = ListSerializer(JournalEntry.serializer())

        /** Minimum age of the newest (symbol, horizon) entry before another one is logged. */
        private fun strideMs(horizon: Horizon): Long = when (horizon) {
            Horizon.H24 -> 6 * HOUR_MS
            Horizon.D7 -> 24 * HOUR_MS
        }

        private fun Forecast.at(horizon: Horizon): HorizonForecast = when (horizon) {
            Horizon.H24 -> h24
            Horizon.D7 -> d7
        }
    }
}
