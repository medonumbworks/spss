package dev.cryptopulse.core.sources.dex

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.model.DexInfo
import dev.cryptopulse.core.model.DexTxns
import dev.cryptopulse.core.model.DexWindow
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.util.Dates
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.dbl
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.int
import dev.cryptopulse.core.util.lng
import dev.cryptopulse.core.util.str
import kotlinx.serialization.Serializable

/** One OHLCV candle (USD). */
@Serializable
data class Candle(val timeMs: Long, val open: Double, val high: Double, val low: Double, val close: Double, val volume: Double)

/** A pool from GeckoTerminal's trending or new lists, with its base token. */
@Serializable
data class GtPool(
    /** GeckoTerminal network id ("solana", "eth", "base", "bsc", "robinhood", ...). */
    val network: String,
    val name: String,
    val baseSymbol: String,
    val baseName: String,
    val baseCoingeckoId: String? = null,
    val imageUrl: String? = null,
    /** Same fields as DexScreener, with the chain id translated to DexScreener's naming. */
    val info: DexInfo,
)

/**
 * GeckoTerminal public API v2 (keyless, about 30 calls/min): on-chain pools across networks.
 * https://www.geckoterminal.com/dex-api
 */
class GeckoTerminalApi(
    private val http: Http,
    private val base: String = "https://api.geckoterminal.com/api/v2",
) {
    /** Trending pools across all networks (20 per page). */
    suspend fun trendingPools(page: Int = 1, duration: String = "24h"): SourceResult<List<GtPool>> =
        pools("$base/networks/trending_pools?include=base_token&page=${page.coerceIn(1, 10)}&duration=$duration", "gt_trending", 10 * 60_000L)

    /** The newest pools across all networks (20 per page); almost all of them fail, filter hard. */
    suspend fun newPools(page: Int = 1): SourceResult<List<GtPool>> =
        pools("$base/networks/new_pools?include=base_token&page=${page.coerceIn(1, 10)}", "gt_new", 5 * 60_000L)

    /**
     * Hourly candles of one pool, oldest first (at most 1000; 168 = one week). [token]: the token's
     * contract, so the candles are its own price even when it is the pool's quote side; if GeckoTerminal
     * rejects it (HTTP 400/404/422), the pool's default (base token) candles are read instead.
     */
    suspend fun hourlyCandles(network: String, poolAddress: String, limit: Int = 168, token: String? = null): SourceResult<List<Candle>> {
        val plain = "$base/networks/$network/pools/$poolAddress/ohlcv/hour?aggregate=1&limit=${limit.coerceIn(1, 1000)}&currency=usd"
        var r = http.getText(if (token == null) plain else "$plain&token=$token", JSON, ttlMs = 15 * 60_000L, staleOkMs = 6 * 3_600_000L, retries = 1)
        if (token != null && r is HttpResult.Failed && r.code in REJECTED) {
            r = http.getText(plain, JSON, ttlMs = 15 * 60_000L, staleOkMs = 6 * 3_600_000L, retries = 1)
        }
        return when (r) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "GeckoTerminal candles")
            is HttpResult.Ok -> try {
                SourceResult.Success(parseCandles(r.body), r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("GeckoTerminal candles", e)
            }
        }
    }

    private suspend fun pools(url: String, foundBy: String, ttl: Long): SourceResult<List<GtPool>> =
        when (val r = http.getText(url, JSON, ttlMs = ttl, staleOkMs = 2 * 3_600_000L, retries = 1)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "GeckoTerminal")
            is HttpResult.Ok -> try {
                SourceResult.Success(parsePools(r.body, foundBy), r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("GeckoTerminal", e)
            }
        }

    companion object {
        private val REJECTED = setOf(400, 404, 422)

        private val JSON = mapOf("Accept" to "application/json")

        /** GeckoTerminal network ids that DexScreener (and the rest of the app) spell differently. */
        private val TO_DEXSCREENER = mapOf(
            "eth" to "ethereum", "polygon_pos" to "polygon", "avax" to "avalanche", "cro" to "cronos", "ftm" to "fantom",
            "sui-network" to "sui", "xdai" to "gnosischain", "one" to "harmony", "polygon_zkevm" to "polygonzkevm",
            "arbitrum_nova" to "arbitrumnova",
        )

        fun dexScreenerChain(network: String): String = TO_DEXSCREENER[network] ?: network

        fun geckoTerminalNetwork(chainId: String): String = TO_DEXSCREENER.entries.firstOrNull { it.value == chainId }?.key ?: chainId

        fun parsePools(body: String, foundBy: String): List<GtPool> {
            val root = JsonUtil.parse(body)
            val tokens = root["included"].arr.orEmpty().filter { it["type"].str == "token" }.associateBy { it["id"].str }
            return root["data"].arr.orEmpty().mapNotNull { p ->
                val a = p["attributes"]
                val network = p["relationships"]["network"]["data"]["id"].str ?: p["id"].str?.substringBefore('_') ?: return@mapNotNull null
                val baseId = p["relationships"]["base_token"]["data"]["id"].str ?: return@mapNotNull null
                val token = tokens[baseId]?.get("attributes")
                val address = token["address"].str ?: baseId.substringAfter('_')
                val symbol = token["symbol"].str?.trim()?.removePrefix("$")?.takeIf { it.isNotEmpty() }
                    ?: a["name"].str?.substringBefore('/')?.trim()?.takeIf { it.isNotEmpty() } ?: return@mapNotNull null
                fun tx(key: String) = a["transactions"][key].let { DexTxns(it["buys"].int ?: 0, it["sells"].int ?: 0, it["buyers"].int, it["sellers"].int) }
                val chain = dexScreenerChain(network)
                val poolAddress = a["address"].str
                GtPool(
                    network = network,
                    name = a["name"].str.orEmpty(),
                    baseSymbol = symbol,
                    baseName = token["name"].str?.trim().orEmpty().ifEmpty { symbol },
                    baseCoingeckoId = token["coingecko_coin_id"].str,
                    imageUrl = token["image_url"].str?.takeIf { it.startsWith("http") },
                    info = DexInfo(
                        chainId = chain,
                        tokenAddress = address,
                        pairAddress = poolAddress,
                        dexId = p["relationships"]["dex"]["data"]["id"].str,
                        url = poolAddress?.let { "https://www.geckoterminal.com/$network/pools/$it" },
                        priceUsd = a["base_token_price_usd"].dbl,
                        liquidityUsd = a["reserve_in_usd"].dbl,
                        fdvUsd = a["fdv_usd"].dbl,
                        marketCapUsd = a["market_cap_usd"].dbl,
                        priceChangePct = a["price_change_percentage"].let { DexWindow(it["m5"].dbl, it["h1"].dbl, it["h6"].dbl, it["h24"].dbl) },
                        volumeUsd = a["volume_usd"].let { DexWindow(it["m5"].dbl, it["h1"].dbl, it["h6"].dbl, it["h24"].dbl) },
                        txnsH1 = tx("h1"),
                        txnsH6 = tx("h6"),
                        txnsH24 = tx("h24"),
                        pairCreatedAtMs = Dates.parseToMs(a["pool_created_at"].str),
                        foundBy = listOf(foundBy),
                    ),
                )
            }
        }

        fun parseCandles(body: String): List<Candle> =
            JsonUtil.parse(body)["data"]["attributes"]["ohlcv_list"].arr.orEmpty().mapNotNull { row ->
                val c = row.arr ?: return@mapNotNull null
                val t = c.getOrNull(0).lng ?: return@mapNotNull null
                Candle(
                    timeMs = if (t < 100_000_000_000L) t * 1000 else t,
                    open = c.getOrNull(1).dbl ?: return@mapNotNull null,
                    high = c.getOrNull(2).dbl ?: return@mapNotNull null,
                    low = c.getOrNull(3).dbl ?: return@mapNotNull null,
                    close = c.getOrNull(4).dbl ?: return@mapNotNull null,
                    volume = c.getOrNull(5).dbl ?: 0.0,
                )
            }.sortedBy { it.timeMs }
    }
}
