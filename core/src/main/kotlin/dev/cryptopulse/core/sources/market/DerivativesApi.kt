package dev.cryptopulse.core.sources.market

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.model.DerivativesSnapshot
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.sources.DerivativesSource
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.dbl
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.str

/**
 * Crowd positioning from perpetual futures (public, keyless): Binance USDⓈ-M first, Bybit as fallback
 * (Binance answers HTTP 451 in some countries). Funding > 0 means longs pay shorts (crowded longs).
 */
class DerivativesApi(
    private val http: Http,
    private val clock: Clock = Clock.SYSTEM,
) : DerivativesSource {

    override suspend fun fundingRates(): SourceResult<Map<String, DerivativesSnapshot>> {
        val binance = http.getText("https://fapi.binance.com/fapi/v1/premiumIndex", JSON, ttlMs = 10 * 60_000L, staleOkMs = 6 * 3_600_000L, retries = 1)
        if (binance is HttpResult.Ok) {
            try {
                val map = parseBinancePremium(binance.body, clock.nowMs(), binanceIntervals())
                if (map.isNotEmpty()) return SourceResult.Success(map, binance.fromCache, binance.ageMs)
            } catch (_: Exception) {
                // fall through to Bybit
            }
        }
        val bybit = http.getText("https://api.bybit.com/v5/market/tickers?category=linear", JSON, ttlMs = 10 * 60_000L, staleOkMs = 6 * 3_600_000L, retries = 1)
        return when (bybit) {
            is HttpResult.Ok -> try {
                val map = parseBybitTickers(bybit.body, clock.nowMs())
                if (map.isEmpty()) SourceResult.Failure(SourceStatus.EMPTY, "Derivatives: no funding data") else SourceResult.Success(map, bybit.fromCache, bybit.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("Bybit funding", e)
            }
            is HttpResult.Failed -> {
                val b = binance as? HttpResult.Failed
                SourceResult.failureOf(bybit, "Derivatives (Binance: ${b?.reason ?: "parse error"}; Bybit)")
            }
        }
    }

    /** Contracts that settle funding more often than every 8 h (only those are listed by Binance). */
    private suspend fun binanceIntervals(): Map<String, Int> {
        val r = http.getText("https://fapi.binance.com/fapi/v1/fundingInfo", JSON, ttlMs = 12 * 3_600_000L, staleOkMs = 7 * 86_400_000L, retries = 0)
        return if (r is HttpResult.Ok) runCatching { parseBinanceFundingInfo(r.body) }.getOrDefault(emptyMap()) else emptyMap()
    }

    override suspend fun longShortRatio(baseSymbol: String): SourceResult<Double?> {
        val sym = baseSymbol.uppercase().filter { it.isLetterOrDigit() } + "USDT"
        val url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=$sym&period=1h&limit=1"
        return when (val r = http.getText(url, JSON, ttlMs = 15 * 60_000L, retries = 0)) {
            is HttpResult.Failed -> if (r.code == 400) SourceResult.Success(null) else SourceResult.failureOf(r, "Long/short ratio")
            is HttpResult.Ok -> try {
                SourceResult.Success(JsonUtil.parse(r.body).arr?.lastOrNull()?.get("longShortRatio").dbl)
            } catch (e: Exception) {
                SourceResult.parseError("Long/short ratio", e)
            }
        }
    }

    companion object {
        private val JSON = mapOf("Accept" to "application/json")
        private val MULTIPLIER_PREFIX = Regex("^(1000000|100000|10000|1000|100|1M|1K)(?=[A-Z])")

        /** "1000PEPEUSDT" -> "PEPE", "BTCUSDT" -> "BTC"; non-USDT pairs are ignored. */
        fun baseOf(pair: String): String? {
            if (!pair.endsWith("USDT")) return null
            val base = pair.removeSuffix("USDT")
            return MULTIPLIER_PREFIX.replace(base, "").takeIf { it.isNotEmpty() }
        }

        /** Rates are quoted per settlement; the model expects a rate per 8 h. */
        fun per8h(rate: Double, intervalHours: Double?): Double =
            if (intervalHours == null || intervalHours <= 0) rate else rate * 8.0 / intervalHours

        /** `/fapi/v1/fundingInfo`: pair -> funding interval in hours. */
        fun parseBinanceFundingInfo(body: String): Map<String, Int> {
            val out = HashMap<String, Int>()
            JsonUtil.parse(body).arr.orEmpty().forEach { el ->
                val pair = el["symbol"].str ?: return@forEach
                val hours = el["fundingIntervalHours"].dbl?.toInt() ?: return@forEach
                if (hours > 0) out[pair] = hours
            }
            return out
        }

        fun parseBinancePremium(body: String, nowMs: Long, intervals: Map<String, Int> = emptyMap()): Map<String, DerivativesSnapshot> {
            val out = HashMap<String, DerivativesSnapshot>()
            JsonUtil.parse(body).arr.orEmpty().forEach { el ->
                val pair = el["symbol"].str ?: return@forEach
                val base = baseOf(pair) ?: return@forEach
                val rate = per8h(el["lastFundingRate"].dbl ?: return@forEach, intervals[pair]?.toDouble())
                // keep the plain contract when both "PEPE" and "1000PEPE" exist
                if (out.containsKey(base) && (el["symbol"].str ?: "").first().isDigit()) return@forEach
                out[base] = DerivativesSnapshot(base, rate, el["markPrice"].dbl, null, null, "binance", nowMs)
            }
            return out
        }

        fun parseBybitTickers(body: String, nowMs: Long): Map<String, DerivativesSnapshot> {
            val out = HashMap<String, DerivativesSnapshot>()
            JsonUtil.parse(body)["result"]["list"].arr.orEmpty().forEach { el ->
                val base = baseOf(el["symbol"].str ?: return@forEach) ?: return@forEach
                // Bybit lists the settlement interval on each ticker ("fundingIntervalHour")
                val rate = per8h(el["fundingRate"].dbl ?: return@forEach, el["fundingIntervalHour"].dbl)
                if (out.containsKey(base) && (el["symbol"].str ?: "").first().isDigit()) return@forEach
                out[base] = DerivativesSnapshot(base, rate, el["markPrice"].dbl, null, null, "bybit", nowMs)
            }
            return out
        }
    }
}
