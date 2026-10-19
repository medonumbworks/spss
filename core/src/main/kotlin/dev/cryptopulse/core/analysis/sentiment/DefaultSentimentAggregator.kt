package dev.cryptopulse.core.analysis.sentiment

import dev.cryptopulse.core.analysis.SentimentAggregator
import dev.cryptopulse.core.analysis.SentimentScorer
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.SentimentSample
import dev.cryptopulse.core.model.SentimentSource
import dev.cryptopulse.core.model.SentimentSummary
import dev.cryptopulse.core.model.SourceSentiment
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.sources.NewsItem
import dev.cryptopulse.core.sources.SocialPost
import dev.cryptopulse.core.util.Text
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Scores posts/headlines with a [SentimentScorer] (VADER + crypto slang on device) and aggregates them:
 *  - sample weight = recency (half-life 24 h) x engagement (log scale) x explicitness (StockTwits labels)
 *  - per-source composite = weighted mean; overall = sources weighted by reliability x sqrt(sample count)
 *  - confidence grows with sample count and agreement between sources.
 */
class DefaultSentimentAggregator(
    private val scorer: SentimentScorer,
    private val halfLifeHours: Double = 24.0,
) : SentimentAggregator {

    override fun sampleFromPost(post: SocialPost, nowMs: Long, asset: Asset?): SentimentSample {
        val text = (post.title + " " + post.body).trim().take(MAX_SCORED_CHARS)
        val lexical = scorer.score(text).score
        val explicit = when (post.explicitSentiment?.lowercase()) {
            "bullish" -> 0.6
            "bearish" -> -0.6
            else -> null
        }
        val score = if (explicit != null) (0.6 * explicit + 0.4 * lexical) else lexical
        val engagement = (post.score.coerceAtLeast(0) + 2 * post.comments.coerceAtLeast(0)).toDouble()
        val engagementWeight = when (post.source) {
            SentimentSource.TELEGRAM -> 1.0 + 0.25 * log10(1.0 + engagement) // views are large numbers
            else -> 1.0 + 0.5 * log10(1.0 + engagement)
        }
        val ratioPenalty = post.upvoteRatio?.let { 0.5 + 0.5 * it } ?: 1.0
        val weight = recency(post.createdMs, nowMs) * engagementWeight * ratioPenalty * (if (explicit != null) 1.3 else 1.0)
        return SentimentSample(
            source = post.source,
            text = Text.truncate(post.title.ifBlank { post.body }.replace(Regex("\\s+"), " ").trim(), 220),
            score = score.coerceIn(-1.0, 1.0),
            weight = weight,
            explicitLabel = post.explicitSentiment,
            url = post.url,
            author = post.author,
            community = post.community,
            timestampMs = post.createdMs,
            engagement = engagement,
        )
    }

    override fun sampleFromNews(item: NewsItem, nowMs: Long, asset: Asset?): SentimentSample {
        // Headlines carry most of the signal; summaries are often boilerplate.
        val titleScore = scorer.score(item.title).score
        val summaryScore = item.summary?.let { scorer.score(it.take(MAX_SCORED_CHARS)).score }
        val score = if (summaryScore == null) titleScore else 0.7 * titleScore + 0.3 * summaryScore
        val ts = item.publishedMs ?: nowMs
        return SentimentSample(
            source = SentimentSource.NEWS,
            text = Text.truncate(item.title, 220),
            score = score.coerceIn(-1.0, 1.0),
            weight = recency(ts, nowMs) * 1.2,
            url = item.url,
            community = item.sourceName,
            timestampMs = item.publishedMs,
        )
    }

    override fun summarize(
        symbol: String,
        samples: List<SentimentSample>,
        statuses: Map<SentimentSource, SourceStatus>,
        nowMs: Long,
        preAggregated: List<SourceSentiment>,
    ): SentimentSummary {
        val bySource = samples.groupBy { it.source }
        val perSource = ArrayList<SourceSentiment>()
        val sourceKeys = (statuses.keys + bySource.keys).filter { k -> preAggregated.none { it.source == k } }.distinct()
        for (src in sourceKeys) {
            val list = bySource[src].orEmpty()
            val status = statuses[src] ?: if (list.isEmpty()) SourceStatus.EMPTY else SourceStatus.OK
            if (list.isEmpty()) {
                perSource += SourceSentiment(src, if (status == SourceStatus.OK) SourceStatus.EMPTY else status)
                continue
            }
            val wsum = list.sumOf { it.weight }.takeIf { it > 0 } ?: list.size.toDouble()
            val composite = list.sumOf { it.score * it.weight } / wsum
            val bull = list.filter { it.score >= LABEL_THRESHOLD }.sumOf { it.weight } / wsum
            val bear = list.filter { it.score <= -LABEL_THRESHOLD }.sumOf { it.weight } / wsum
            val times = list.mapNotNull { it.timestampMs }
            val spanDays = if (times.size >= 2) max(1.0, (nowMs - times.min()) / DAY_MS) else 7.0
            perSource += SourceSentiment(
                source = src,
                status = SourceStatus.OK,
                sampleCount = list.size,
                composite = round4(composite),
                bullishShare = round4(bull),
                bearishShare = round4(bear),
                neutralShare = round4((1 - bull - bear).coerceAtLeast(0.0)),
                buzzPerDay = round4(list.size / min(spanDays, 7.0)),
                topSamples = list.sortedByDescending { abs(it.score) * it.weight }.take(TOP_SAMPLES),
            )
        }
        perSource += preAggregated

        val contributing = perSource.filter { it.sampleCount > 0 }
        val total = contributing.sumOf { it.sampleCount }
        if (contributing.isEmpty()) {
            return SentimentSummary(symbol, 0.0, 0.0, "No data", perSource.sortedBy { it.source.ordinal }, 0, nowMs)
        }
        val weights = contributing.map { (RELIABILITY[it.source] ?: 1.0) * sqrt(it.sampleCount.toDouble()) }
        val wsum = weights.sum()
        val composite = contributing.indices.sumOf { contributing[it].composite * weights[it] } / wsum
        // agreement: 1 when all sources share a sign/size, lower when they disagree
        val spread = if (contributing.size >= 2) {
            val mean = composite
            sqrt(contributing.indices.sumOf { weights[it] * (contributing[it].composite - mean).let { d -> d * d } } / wsum)
        } else 0.0
        val agreement = (1.0 - min(1.0, spread / 0.5))
        val volume = 1.0 - exp(-total / 30.0)
        val confidence = (volume * (0.6 + 0.4 * agreement)).coerceIn(0.0, 1.0)
        return SentimentSummary(
            symbol = symbol,
            composite = round4(composite.coerceIn(-1.0, 1.0)),
            confidence = round4(confidence),
            label = label(composite, total),
            perSource = perSource.sortedBy { it.source.ordinal },
            totalSamples = total,
            computedAtMs = nowMs,
        )
    }

    private fun recency(tsMs: Long, nowMs: Long): Double {
        val ageH = max(0.0, (nowMs - tsMs) / 3_600_000.0)
        return exp(-ln(2.0) * ageH / halfLifeHours).coerceAtLeast(0.02)
    }

    companion object {
        const val LABEL_THRESHOLD = 0.1
        const val TOP_SAMPLES = 6
        private const val MAX_SCORED_CHARS = 600
        private const val DAY_MS = 86_400_000.0

        /** Relative trust per source (news and explicit-label sources are less noisy). */
        val RELIABILITY: Map<SentimentSource, Double> = mapOf(
            SentimentSource.SENTIMENT_HUB to 1.3,
            SentimentSource.NEWS to 1.1,
            SentimentSource.STOCKTWITS to 1.0,
            SentimentSource.REDDIT to 1.0,
            SentimentSource.TELEGRAM to 0.9,
        )

        fun label(composite: Double, samples: Int): String = when {
            samples == 0 -> "No data"
            composite >= 0.25 -> "Very bullish"
            composite >= 0.08 -> "Bullish"
            composite <= -0.25 -> "Very bearish"
            composite <= -0.08 -> "Bearish"
            else -> "Neutral"
        }

        private fun round4(x: Double) = Math.round(x * 10_000.0) / 10_000.0
    }
}
