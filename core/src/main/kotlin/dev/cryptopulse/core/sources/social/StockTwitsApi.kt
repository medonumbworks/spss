package dev.cryptopulse.core.sources.social

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.model.SentimentSource
import dev.cryptopulse.core.sources.SocialPost
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.sources.StockTwitsSource
import dev.cryptopulse.core.util.Dates
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.int
import dev.cryptopulse.core.util.obj
import dev.cryptopulse.core.util.str

/** StockTwits public symbol streams. Crypto symbols use the `.X` suffix (BTC.X); users label posts Bullish/Bearish. */
class StockTwitsApi(
    private val http: Http,
    private val base: String = "https://api.stocktwits.com/api/2",
    private val ttlMs: Long = 20 * 60_000L,
) : StockTwitsSource {

    override suspend fun symbolStream(symbol: String): SourceResult<List<SocialPost>> {
        val clean = symbol.uppercase().filter { it.isLetterOrDigit() }
        if (clean.isEmpty()) return SourceResult.Success(emptyList())
        val url = "$base/streams/symbol/$clean.X.json"
        return when (val r = http.getText(url, mapOf("Accept" to "application/json"), ttlMs = ttlMs, retries = 1)) {
            is HttpResult.Failed -> if (r.code == 404) SourceResult.Success(emptyList()) else SourceResult.failureOf(r, "StockTwits")
            is HttpResult.Ok -> try {
                SourceResult.Success(parseStream(r.body, "$clean.X"), r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("StockTwits", e)
            }
        }
    }

    companion object {
        fun parseStream(body: String, stream: String): List<SocialPost> =
            JsonUtil.parse(body)["messages"].arr.orEmpty().mapNotNull { m ->
                val o = m.obj ?: return@mapNotNull null
                val id = o["id"].str ?: return@mapNotNull null
                val created = Dates.parseToMs(o["created_at"].str) ?: return@mapNotNull null
                SocialPost(
                    source = SentimentSource.STOCKTWITS,
                    id = id,
                    title = (o["body"].str ?: "").take(1000),
                    url = "https://stocktwits.com/message/$id",
                    author = o["user"]["username"].str,
                    createdMs = created,
                    score = o["likes"]["total"].int ?: 0,
                    community = stream,
                    explicitSentiment = o["entities"]["sentiment"]["basic"].str,
                )
            }
    }
}
