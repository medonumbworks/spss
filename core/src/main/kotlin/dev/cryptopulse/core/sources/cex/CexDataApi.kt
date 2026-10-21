package dev.cryptopulse.core.sources.cex

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.model.DerivativesSnapshot
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.sources.dex.Candle
import dev.cryptopulse.core.sources.market.DerivativesApi
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.dbl
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.lng
import dev.cryptopulse.core.util.str

/**
 * Free exchange data that is not geo-blocked:
 *  - Binance's public market-data mirror `data-api.binance.vision` (spot klines without the 451 that
 *    api.binance.com returns in some regions): price history for the hundreds of coins Binance lists,
 *    used when CoinGecko's free tier refuses history;
 *  - OKX public funding rates, a third fallback after Binance and Bybit futures.
 */
class CexDataApi(
    private val http: Http,
    private val clock: Clock = Clock.SYSTEM,
    private val binanceBase: String = "https://data-api.binance.vision/api/v3",
    private val okxBase: String = "https://www.okx.com/api/v5",
) {
    /** Candles of `<SYMBOL>USDT` on Binance spot; HTTP 400 (unknown pair) is an empty list. */
    suspend fun binanceKlines(symbol: String, interval: String, limit: Int): SourceResult<List<Candle>> {
        val pair = symbol.uppercase().filter { it.isLetterOrDigit() } + "USDT"
        val url = "$binanceBase/klines?symbol=$pair&interval=$interval&limit=${limit.coerceIn(1, 1000)}"
        val ttl = if (interval.endsWith("d")) 3 * 3_600_000L else 15 * 60_000L
        return when (val r = http.getText(url, JSON, ttlMs = ttl, staleOkMs = 12 * 3_600_000L, retries = 1)) {
            is HttpResult.Failed -> if (r.code == 400) SourceResult.Success(emptyList()) else SourceResult.failureOf(r, "Binance klines")
            is HttpResult.Ok -> try {
                SourceResult.Success(parseKlines(r.body), r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("Binance klines", e)
            }
        }
    }

    /** Current OKX perpetual funding for `<SYMBOL>-USDT-SWAP`, normalised to 8 hours. */
    suspend fun okxFunding(symbol: String): SourceResult<DerivativesSnapshot?> {
        val inst = symbol.uppercase().filter { it.isLetterOrDigit() } + "-USDT-SWAP"
        return when (val r = http.getText("$okxBase/public/funding-rate?instId=$inst", JSON, ttlMs = 10 * 60_000L, staleOkMs = 3_600_000L, retries = 0)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "OKX funding")
            is HttpResult.Ok -> try {
                val root = JsonUtil.parse(r.body)
                if (root["code"].str != "0") {
                    // 51001 = instrument does not exist: the coin simply has no OKX perpetual
                    SourceResult.Success(null)
                } else {
                    SourceResult.Success(parseOkxFunding(r.body, symbol.uppercase(), clock.nowMs()))
                }
            } catch (e: Exception) {
                SourceResult.parseError("OKX funding", e)
            }
        }
    }

    companion object {
        private val JSON = mapOf("Accept" to "application/json")

        /** Binance kline rows: [openTime, open, high, low, close, volume, closeTime, ...]. */
        fun parseKlines(body: String): List<Candle> =
            JsonUtil.parse(body).arr.orEmpty().mapNotNull { row ->
                val c = row.arr ?: return@mapNotNull null
                Candle(
                    timeMs = c.getOrNull(0).lng ?: return@mapNotNull null,
                    open = c.getOrNull(1).dbl ?: return@mapNotNull null,
                    high = c.getOrNull(2).dbl ?: return@mapNotNull null,
                    low = c.getOrNull(3).dbl ?: return@mapNotNull null,
                    close = c.getOrNull(4).dbl ?: return@mapNotNull null,
                    volume = c.getOrNull(5).dbl ?: 0.0,
                )
            }.sortedBy { it.timeMs }

        fun parseOkxFunding(body: String, base: String, nowMs: Long): DerivativesSnapshot? {
            val d = JsonUtil.parse(body)["data"].arr?.firstOrNull() ?: return null
            val rate = d["fundingRate"].dbl ?: return null
            val t = d["fundingTime"].lng
            val next = d["nextFundingTime"].lng
            val intervalH = if (t != null && next != null && next > t) (next - t) / 3_600_000.0 else null
            return DerivativesSnapshot(base, DerivativesApi.per8h(rate, intervalH), null, null, null, "okx", nowMs)
        }
    }
}
