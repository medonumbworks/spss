package dev.cryptopulse.core.sources.feed

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.util.Dates
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.dbl
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.int
import dev.cryptopulse.core.util.obj
import dev.cryptopulse.core.util.str
import kotlinx.serialization.Serializable

@Serializable
data class FeedSourceStatus(val name: String, val status: String, val items: Int = 0, val detail: String? = null)

@Serializable
data class FeedSample(
    val text: String,
    val score: Double,
    val source: String,
    val community: String? = null,
    val url: String? = null,
    val tsMs: Long? = null,
)

@Serializable
data class FeedAsset(
    val symbol: String,
    val mentions: Int,
    val mentionsPrev: Int? = null,
    /** Average mentions per window over the hub's trailing 7 days; null until the hub publishes it. */
    val baseline7d: Double? = null,
    /** mentions vs [baseline7d] as published by the hub; null in feeds from older hubs. */
    val spike: Double? = null,
    val composite: Double,
    val scores: Map<String, Double> = emptyMap(),
    val bullish: Double = 0.0,
    val bearish: Double = 0.0,
    val neutral: Double = 0.0,
    /** source name -> (mentions, composite) */
    val bySource: Map<String, Pair<Int, Double>> = emptyMap(),
    val samples: List<FeedSample> = emptyList(),
)

@Serializable
data class FeedMarket(val mentions: Int, val composite: Double, val bullish: Double, val bearish: Double, val neutral: Double)

/** Parsed `cryptopulse.sentiment-feed/v1` document published by tools/sentiment_hub. */
@Serializable
data class SentimentFeedDoc(
    val generatedAtMs: Long,
    val windowHours: Int,
    val models: List<String>,
    val sources: List<FeedSourceStatus>,
    val market: FeedMarket?,
    val assets: Map<String, FeedAsset>,
) {
    fun ageMs(nowMs: Long): Long = (nowMs - generatedAtMs).coerceAtLeast(0)

    /** Sum of all assets' mentions in the current and in the previous window. */
    val mentionTotals: Pair<Int, Int> get() = assets.values.sumOf { it.mentions } to assets.values.sumOf { it.mentionsPrev ?: 0 }
    val telegramAccountConnected: Boolean get() = sources.any { it.name == "telegram_account" && it.status == "ok" }
}

/**
 * Downloads the free cloud sentiment feed (one small JSON for all assets) produced hourly by the
 * GitHub Actions workflow `.github/workflows/sentiment-hub.yml`.
 */
class SentimentFeedApi(
    private val http: Http,
    private val urlProvider: () -> String,
    private val ttlMs: Long = 20 * 60_000L,
) {
    suspend fun fetch(): SourceResult<SentimentFeedDoc> {
        val url = urlProvider().trim()
        if (url.isEmpty()) return SourceResult.disabled("Sentiment hub")
        return when (val r = http.getText(url, mapOf("Accept" to "application/json"), ttlMs = ttlMs, staleOkMs = 24 * 3_600_000L, retries = 1)) {
            is HttpResult.Failed -> if (r.code == 404) {
                SourceResult.Failure(SourceStatus.EMPTY, "Sentiment hub has not published a feed yet (run the 'sentiment-hub' GitHub workflow)", 404)
            } else SourceResult.failureOf(r, "Sentiment hub")
            is HttpResult.Ok -> try {
                SourceResult.Success(parse(r.body), r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("Sentiment hub", e)
            }
        }
    }

    companion object {
        const val SCHEMA = "cryptopulse.sentiment-feed/v1"
        const val DEFAULT_URL = "https://raw.githubusercontent.com/medonumbworks/spss/sentiment-feed/feed.json"

        fun parse(body: String): SentimentFeedDoc {
            val root = JsonUtil.parse(body)
            val schema = root["schema"].str
            require(schema == null || schema.startsWith("cryptopulse.sentiment-feed/")) { "Unexpected feed schema: $schema" }
            val assets = root["assets"].arr.orEmpty().mapNotNull { a ->
                val sym = a["symbol"].str?.uppercase() ?: return@mapNotNull null
                val bySource = a["bySource"].obj?.mapNotNull { (k, v) ->
                    val m = v["mentions"].int ?: return@mapNotNull null
                    k to (m to (v["composite"].dbl ?: 0.0))
                }?.toMap().orEmpty()
                FeedAsset(
                    symbol = sym,
                    mentions = a["mentions"].int ?: 0,
                    mentionsPrev = a["mentionsPrev"].int,
                    baseline7d = a["baseline7d"].dbl,
                    spike = a["spike"].dbl,
                    composite = (a["composite"].dbl ?: 0.0).coerceIn(-1.0, 1.0),
                    scores = a["scores"].obj?.mapNotNull { (k, v) -> v.dbl?.let { k to it } }?.toMap().orEmpty(),
                    bullish = a["bullish"].dbl ?: 0.0,
                    bearish = a["bearish"].dbl ?: 0.0,
                    neutral = a["neutral"].dbl ?: 0.0,
                    bySource = bySource,
                    samples = a["samples"].arr.orEmpty().mapNotNull { s ->
                        FeedSample(
                            text = s["text"].str ?: return@mapNotNull null,
                            score = s["score"].dbl ?: 0.0,
                            source = s["source"].str ?: "hub",
                            community = s["community"].str,
                            url = s["url"].str,
                            tsMs = Dates.parseToMs(s["ts"].str),
                        )
                    },
                )
            }.associateBy { it.symbol }
            val market = root["market"].obj?.let { m ->
                FeedMarket(m["mentions"].int ?: 0, m["composite"].dbl ?: 0.0, m["bullish"].dbl ?: 0.0, m["bearish"].dbl ?: 0.0, m["neutral"].dbl ?: 0.0)
            }
            return SentimentFeedDoc(
                generatedAtMs = Dates.parseToMs(root["generatedAt"].str) ?: 0L,
                windowHours = root["windowHours"].int ?: 24,
                models = root["models"].arr.orEmpty().mapNotNull { it.str },
                sources = root["sources"].arr.orEmpty().mapNotNull { s ->
                    FeedSourceStatus(s["name"].str ?: return@mapNotNull null, s["status"].str ?: "unknown", s["items"].int ?: 0, s["detail"].str)
                },
                market = market,
                assets = assets,
            )
        }
    }
}
