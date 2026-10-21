package dev.cryptopulse.core.sources.dex

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.model.DexInfo
import dev.cryptopulse.core.model.DexTxns
import dev.cryptopulse.core.model.DexWindow
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.dbl
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.int
import dev.cryptopulse.core.util.lng
import dev.cryptopulse.core.util.obj
import dev.cryptopulse.core.util.str
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement

/** A token DexScreener promotes: paid boost, new profile or community takeover. */
@Serializable
data class DexPromo(
    val chainId: String,
    val tokenAddress: String,
    /** "ds_boost" (top or latest boosts), "ds_profile", "ds_takeover". */
    val kind: String,
    val boostAmount: Double? = null,
    val description: String? = null,
)

/** One DexScreener pair, with the token it trades (base token). */
@Serializable
data class DexPair(
    val baseAddress: String,
    val baseSymbol: String,
    val baseName: String,
    val imageUrl: String? = null,
    val info: DexInfo,
)

/**
 * DexScreener public API (keyless). Documented limits: 60 requests/min for the boost, profile and
 * takeover lists, 300/min for pair and token lookups. https://docs.dexscreener.com/api/reference
 */
class DexScreenerApi(
    private val http: Http,
    private val base: String = "https://api.dexscreener.com",
) {
    /** Tokens with the most active paid boosts, the latest boosts, new profiles and takeovers, deduplicated. */
    suspend fun promoted(): SourceResult<List<DexPromo>> {
        val calls = listOf(
            "/token-boosts/top/v1" to "ds_boost",
            "/token-boosts/latest/v1" to "ds_boost",
            "/token-profiles/latest/v1" to "ds_profile",
            "/community-takeovers/latest/v1" to "ds_takeover",
        )
        val out = LinkedHashMap<String, DexPromo>()
        var lastFailure: SourceResult.Failure? = null
        var anyOk = false
        for ((path, kind) in calls) {
            when (val r = http.getText(base + path, JSON, ttlMs = 10 * 60_000L, staleOkMs = 2 * 3_600_000L, retries = 1)) {
                is HttpResult.Failed -> lastFailure = SourceResult.failureOf(r, "DexScreener $kind")
                is HttpResult.Ok -> try {
                    parsePromos(r.body, kind).forEach { p ->
                        val key = p.chainId + ":" + p.tokenAddress
                        val prev = out[key]
                        out[key] = if (prev == null) p else prev.copy(boostAmount = maxOf(prev.boostAmount ?: 0.0, p.boostAmount ?: 0.0).takeIf { it > 0 })
                    }
                    anyOk = true
                } catch (e: Exception) {
                    lastFailure = SourceResult.parseError("DexScreener $kind", e)
                }
            }
        }
        return if (anyOk) SourceResult.Success(out.values.toList()) else lastFailure ?: SourceResult.Success(emptyList())
    }

    /** `GET /tokens/v1/{chainId}/{addresses}`: up to 30 addresses per call; the most liquid pair per token. */
    suspend fun tokens(chainId: String, addresses: List<String>): SourceResult<List<DexPair>> {
        if (addresses.isEmpty()) return SourceResult.Success(emptyList())
        val pairs = ArrayList<DexPair>()
        var lastFailure: SourceResult.Failure? = null
        for (chunk in addresses.distinct().chunked(MAX_ADDRESSES)) {
            val url = "$base/tokens/v1/$chainId/${chunk.joinToString(",")}"
            when (val r = http.getText(url, JSON, ttlMs = 2 * 60_000L, staleOkMs = 30 * 60_000L, retries = 1)) {
                is HttpResult.Failed -> lastFailure = SourceResult.failureOf(r, "DexScreener tokens")
                is HttpResult.Ok -> try {
                    pairs += parsePairs(JsonUtil.parse(r.body).arr.orEmpty())
                } catch (e: Exception) {
                    lastFailure = SourceResult.parseError("DexScreener tokens", e)
                }
            }
        }
        if (pairs.isEmpty() && lastFailure != null) return lastFailure
        return SourceResult.Success(mostLiquidPerToken(pairs))
    }

    /** `GET /latest/dex/search?q=` (pairs matching a name, symbol or address). */
    suspend fun search(query: String): SourceResult<List<DexPair>> {
        val url = "$base/latest/dex/search?q=" + java.net.URLEncoder.encode(query, "UTF-8")
        return when (val r = http.getText(url, JSON, ttlMs = 5 * 60_000L, retries = 1)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "DexScreener search")
            is HttpResult.Ok -> try {
                SourceResult.Success(mostLiquidPerToken(parsePairs(JsonUtil.parse(r.body)["pairs"].arr.orEmpty())))
            } catch (e: Exception) {
                SourceResult.parseError("DexScreener search", e)
            }
        }
    }

    companion object {
        const val MAX_ADDRESSES = 30
        private val JSON = mapOf("Accept" to "application/json")

        fun parsePromos(body: String, kind: String): List<DexPromo> =
            JsonUtil.parse(body).arr.orEmpty().mapNotNull { el ->
                val chain = el["chainId"].str ?: return@mapNotNull null
                val address = el["tokenAddress"].str ?: return@mapNotNull null
                DexPromo(chain, address, kind, el["totalAmount"].dbl ?: el["amount"].dbl, el["description"].str?.take(280))
            }

        fun parsePairs(items: List<JsonElement>): List<DexPair> = items.mapNotNull { parsePair(it) }

        private fun window(o: JsonElement?) = DexWindow(o["m5"].dbl, o["h1"].dbl, o["h6"].dbl, o["h24"].dbl)
        private fun txns(o: JsonElement?) = DexTxns(o["buys"].int ?: 0, o["sells"].int ?: 0)

        private fun parsePair(el: JsonElement): DexPair? {
            val o = el.obj ?: return null
            val chain = o["chainId"].str ?: return null
            val baseToken = o["baseToken"]
            val address = baseToken["address"].str ?: return null
            val symbol = baseToken["symbol"].str?.trim()?.removePrefix("$")?.takeIf { it.isNotEmpty() } ?: return null
            val info = o["info"]
            val socials = info["socials"].arr.orEmpty().mapNotNull { s ->
                val type = s["type"].str ?: return@mapNotNull null
                val url = s["url"].str ?: return@mapNotNull null
                type.lowercase() to url
            }.toMap()
            return DexPair(
                baseAddress = address,
                baseSymbol = symbol,
                baseName = baseToken["name"].str?.trim().orEmpty().ifEmpty { symbol },
                imageUrl = info["imageUrl"].str,
                info = DexInfo(
                    chainId = chain,
                    tokenAddress = address,
                    pairAddress = o["pairAddress"].str,
                    dexId = o["dexId"].str,
                    url = o["url"].str,
                    quoteSymbol = o["quoteToken"]["symbol"].str,
                    priceUsd = o["priceUsd"].dbl,
                    liquidityUsd = o["liquidity"]["usd"].dbl,
                    fdvUsd = o["fdv"].dbl,
                    marketCapUsd = o["marketCap"].dbl,
                    priceChangePct = window(o["priceChange"]),
                    volumeUsd = window(o["volume"]),
                    txnsH1 = txns(o["txns"]["h1"]),
                    txnsH6 = txns(o["txns"]["h6"]),
                    txnsH24 = txns(o["txns"]["h24"]),
                    pairCreatedAtMs = o["pairCreatedAt"].lng,
                    boostsActive = o["boosts"]["active"].int ?: 0,
                    websites = info["websites"].arr.orEmpty().mapNotNull { it["url"].str },
                    socials = socials,
                ),
            )
        }

        /** A token trades in many pools; the deepest one has the most reliable price. */
        fun mostLiquidPerToken(pairs: List<DexPair>): List<DexPair> =
            pairs.groupBy { it.info.chainId + ":" + it.baseAddress.lowercase() }
                .values.map { group -> group.maxBy { it.info.liquidityUsd ?: 0.0 } }
    }
}
