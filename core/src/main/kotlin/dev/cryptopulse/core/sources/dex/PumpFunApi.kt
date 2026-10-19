package dev.cryptopulse.core.sources.dex

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.bool
import dev.cryptopulse.core.util.dbl
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.int
import dev.cryptopulse.core.util.lng
import dev.cryptopulse.core.util.str
import kotlinx.serialization.Serializable

/** A pump.fun token (Solana). */
@Serializable
data class PumpCoin(
    val mint: String,
    val symbol: String,
    val name: String,
    val usdMarketCap: Double? = null,
    /** Left the bonding curve for a real pool. Most pump.fun tokens never do. */
    val graduated: Boolean = false,
    val createdMs: Long? = null,
    val lastTradeMs: Long? = null,
    val replyCount: Int? = null,
    val twitter: String? = null,
    val telegram: String? = null,
    val website: String? = null,
    val imageUri: String? = null,
)

/**
 * pump.fun's own (unofficial, undocumented) frontend API. It changes without notice and may sit
 * behind Cloudflare, so every failure is treated as "source unavailable", never as an error.
 */
class PumpFunApi(
    private val http: Http,
    private val base: String = "https://frontend-api-v3.pump.fun",
) {
    /** Largest pump.fun tokens by market cap (banned and NSFW tokens excluded). */
    suspend fun topByMarketCap(limit: Int = 50): SourceResult<List<PumpCoin>> =
        fetch("$base/coins?offset=0&limit=${limit.coerceIn(1, 50)}&sort=market_cap&order=DESC&includeNsfw=false", 10 * 60_000L)

    /** Tokens with a live stream right now (pump.fun's own "hot" list). */
    suspend fun currentlyLive(limit: Int = 30): SourceResult<List<PumpCoin>> =
        fetch("$base/coins/currently-live?limit=${limit.coerceIn(1, 50)}&offset=0&includeNsfw=false", 5 * 60_000L)

    private suspend fun fetch(url: String, ttl: Long): SourceResult<List<PumpCoin>> =
        when (val r = http.getText(url, mapOf("Accept" to "application/json"), ttlMs = ttl, staleOkMs = 2 * 3_600_000L, retries = 0)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "pump.fun")
            is HttpResult.Ok -> try {
                SourceResult.Success(parse(r.body), r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("pump.fun", e)
            }
        }

    companion object {
        fun parse(body: String): List<PumpCoin> =
            JsonUtil.parse(body).arr.orEmpty().mapNotNull { c ->
                if (c["is_banned"].bool == true || c["nsfw"].bool == true) return@mapNotNull null
                val mint = c["mint"].str ?: return@mapNotNull null
                val symbol = c["symbol"].str?.trim()?.removePrefix("$")?.takeIf { it.isNotEmpty() } ?: return@mapNotNull null
                PumpCoin(
                    mint = mint,
                    symbol = symbol,
                    name = c["name"].str?.trim().orEmpty().ifEmpty { symbol },
                    usdMarketCap = c["usd_market_cap"].dbl,
                    graduated = c["complete"].bool == true,
                    createdMs = c["created_timestamp"].lng,
                    lastTradeMs = c["last_trade_timestamp"].lng,
                    replyCount = c["reply_count"].int,
                    twitter = c["twitter"].str?.takeIf { it.startsWith("http") },
                    telegram = c["telegram"].str?.takeIf { it.startsWith("http") },
                    website = c["website"].str?.takeIf { it.startsWith("http") },
                    imageUri = c["image_uri"].str,
                )
            }
    }
}
