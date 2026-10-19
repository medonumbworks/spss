package dev.cryptopulse.core.analysis

import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.DerivativesSnapshot
import dev.cryptopulse.core.model.Forecast
import dev.cryptopulse.core.model.MarketContext
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.SentimentSample
import dev.cryptopulse.core.model.SentimentSource
import dev.cryptopulse.core.model.SentimentSummary
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.sources.NewsItem
import dev.cryptopulse.core.sources.SocialPost

/** Everything the forecast engine may use for one asset. Missing inputs are allowed (coverage shrinks). */
data class ForecastInput(
    val asset: Asset,
    val market: MarketData?,
    /** Hourly closes, oldest first (7-day sparkline: ~168 points). */
    val hourlyCloses: List<Double>,
    /** Daily closes, oldest first (ideally 90+ points; may be derived from the sparkline when absent). */
    val dailyCloses: List<Double> = emptyList(),
    val dailyVolumes: List<Double> = emptyList(),
    val context: MarketContext? = null,
    val sentiment: SentimentSummary? = null,
    val derivatives: DerivativesSnapshot? = null,
    /** CoinGecko community vote share (0..100) that the coin will go up. */
    val communityVotesUpPct: Double? = null,
    val nowMs: Long,
)

/** Produces 24h/7d up-down probabilities, a bottom assessment and explanations. Implemented by `DefaultForecastEngine`. */
interface Forecaster {
    fun forecast(input: ForecastInput): Forecast
}

/** Lexicon-based scoring of one text. Implemented by `LexiconSentimentScorer`. */
data class TextSentiment(
    /** -1..1 */
    val score: Double,
    val matchedPositive: List<String> = emptyList(),
    val matchedNegative: List<String> = emptyList(),
    /** How many sentiment-bearing tokens were found; 0 means neutral/unknown. */
    val hits: Int = 0,
)

interface SentimentScorer {
    fun score(text: String): TextSentiment
}

/** Turns posts/news into samples and samples into a per-asset summary. Implemented by `DefaultSentimentAggregator`. */
interface SentimentAggregator {
    fun sampleFromPost(post: SocialPost, nowMs: Long, asset: Asset? = null): SentimentSample
    fun sampleFromNews(item: NewsItem, nowMs: Long, asset: Asset? = null): SentimentSample

    /**
     * @param statuses status of every source that was attempted (OK/EMPTY/RATE_LIMITED/...), so the
     *                 UI can show "Reddit: rate limited" instead of pretending there was no chatter.
     */
    fun summarize(
        symbol: String,
        samples: List<SentimentSample>,
        statuses: Map<SentimentSource, SourceStatus>,
        nowMs: Long,
    ): SentimentSummary
}
