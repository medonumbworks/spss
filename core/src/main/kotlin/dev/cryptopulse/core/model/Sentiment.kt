package dev.cryptopulse.core.model

import kotlinx.serialization.Serializable

@Serializable
enum class SentimentSource {
    REDDIT,
    STOCKTWITS,
    TELEGRAM,
    NEWS,
    FEAR_GREED,
    COINGECKO_VOTES,
    DERIVATIVES,
    /**
     * Free cloud sentiment hub (GitHub Actions, see tools/sentiment_hub): 100+ public Telegram channels,
     * the user's own Telegram account when connected, news feeds; scored with VADER, FinBERT and CryptoBERT.
     */
    SENTIMENT_HUB,
}

@Serializable
enum class SourceStatus {
    OK,
    EMPTY,
    RATE_LIMITED,
    UNAUTHORIZED,
    UNAVAILABLE,
    GEO_BLOCKED,
    PARSE_ERROR,
    DISABLED,
}

/** One scored text (post, message, headline) or one scored data point. */
@Serializable
data class SentimentSample(
    val source: SentimentSource,
    /** Short, display-safe text (title or first ~200 chars). */
    val text: String,
    /** -1 (very bearish) .. +1 (very bullish). */
    val score: Double,
    /** Relative weight used in aggregation (recency x engagement x explicitness). */
    val weight: Double = 1.0,
    /** "Bullish"/"Bearish" when the platform provides an explicit label (StockTwits). */
    val explicitLabel: String? = null,
    val url: String? = null,
    val author: String? = null,
    /** Community/channel/feed name. */
    val community: String? = null,
    val timestampMs: Long? = null,
    /** Raw engagement (upvotes, likes, views). */
    val engagement: Double = 0.0,
)

@Serializable
data class SourceSentiment(
    val source: SentimentSource,
    val status: SourceStatus,
    val sampleCount: Int = 0,
    /** Weighted composite -1..1. */
    val composite: Double = 0.0,
    val bullishShare: Double = 0.0,
    val bearishShare: Double = 0.0,
    val neutralShare: Double = 0.0,
    /** Mentions per day over the sampled window; 0 when unknown. */
    val buzzPerDay: Double = 0.0,
    val topSamples: List<SentimentSample> = emptyList(),
    val note: String? = null,
    /** Mentions now / mentions in the previous window (attention change), when the source knows it. */
    val buzzRatio: Double? = null,
    /** Per-model composites (e.g. "vader", "finbert", "cryptobert") when the source reports them. */
    val modelScores: Map<String, Double> = emptyMap(),
)

@Serializable
data class SentimentSummary(
    val symbol: String,
    /** Weighted composite -1..1 across all sources. */
    val composite: Double,
    /** 0..1: grows with sample count and agreement between sources. */
    val confidence: Double,
    val label: String,
    val perSource: List<SourceSentiment> = emptyList(),
    val totalSamples: Int = 0,
    val computedAtMs: Long = 0,
)
