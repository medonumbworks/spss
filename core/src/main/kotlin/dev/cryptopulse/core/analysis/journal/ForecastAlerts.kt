package dev.cryptopulse.core.analysis.journal

import dev.cryptopulse.core.http.DiskStore
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.util.Clock
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** A Bitpanda coin whose 24 h chance of rising just crossed the user's threshold. */
@Serializable
data class ForecastAlert(
    val symbol: String,
    val name: String,
    val probabilityUp: Double,
    val confidence: Double,
    val price: Double?,
    val currency: String?,
)

/**
 * Crossing detector behind the "notify me when a Bitpanda coin's predicted rise is over X%" option.
 * Only a *crossing* fires: a coin already above the threshold at the last check stays quiet, so an
 * hourly check does not repeat the same names. The set of coins currently above is persisted (key
 * [STORE_KEY]) and resets when the threshold changes.
 */
class ForecastAlerts(
    private val store: DiskStore?,
    private val clock: Clock = Clock.SYSTEM,
) {
    @Serializable
    private data class State(val thresholdPct: Int, val above: Map<String, Long>)

    private val lock = Any()
    private var state: State? = null
    private var loaded = false

    /**
     * Bitpanda coins whose 24 h up-probability is at least [thresholdPct] percent now and was not at
     * the previous call. Blocked and history-gated forecasts never alert ([AssetRow.upOrNull]).
     */
    fun crossings(rows: List<AssetRow>, thresholdPct: Int): List<ForecastAlert> = synchronized(lock) {
        val threshold = thresholdPct.coerceIn(1, 99) / 100.0
        val previous = load().takeIf { it?.thresholdPct == thresholdPct }?.above ?: emptyMap()
        val now = clock.nowMs()
        val above = LinkedHashMap<String, Long>()
        val fresh = ArrayList<ForecastAlert>()
        for (row in rows) {
            if (row.asset.tier != Tier.BITPANDA) continue
            val f = row.forecast ?: continue
            val p = row.upOrNull(f.h24)?.takeIf { it >= threshold } ?: continue
            above[row.asset.symbol] = previous[row.asset.symbol] ?: now
            if (row.asset.symbol !in previous) {
                fresh += ForecastAlert(row.asset.symbol, row.asset.name, p, f.h24.confidence, row.market?.price, row.market?.currency)
            }
        }
        // rows can be empty while a refresh fails: keep the old state rather than re-alerting later
        if (rows.isNotEmpty()) save(State(thresholdPct, above))
        fresh.sortedByDescending { it.probabilityUp }
    }

    private fun load(): State? {
        if (!loaded) {
            loaded = true
            state = store?.read(STORE_KEY)?.let { runCatching { json.decodeFromString(State.serializer(), it) }.getOrNull() }
        }
        return state
    }

    private fun save(s: State) {
        state = s
        try {
            store?.write(STORE_KEY, json.encodeToString(State.serializer(), s))
        } catch (_: Exception) {
            // alerts are a convenience; a failed write only risks one repeated notification
        }
    }

    companion object {
        const val STORE_KEY = "forecast_alerts"
        private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    }
}
