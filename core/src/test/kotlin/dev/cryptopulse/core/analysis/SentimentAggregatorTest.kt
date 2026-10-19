package dev.cryptopulse.core.analysis

import dev.cryptopulse.core.FIXTURE_NOW_MS
import dev.cryptopulse.core.analysis.sentiment.DefaultSentimentAggregator
import dev.cryptopulse.core.model.SentimentSource
import dev.cryptopulse.core.model.SourceSentiment
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.sources.NewsItem
import dev.cryptopulse.core.sources.SocialPost
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SentimentAggregatorTest {
    /** Keyword stub so this test does not depend on the VADER port. */
    private val scorer = SentimentScorer { text ->
        val t = text.lowercase()
        TextSentiment(
            when {
                "moon" in t || "bullish" in t -> 0.8
                "scam" in t || "dump" in t -> -0.8
                else -> 0.0
            },
        )
    }
    private val agg = DefaultSentimentAggregator(scorer)
    private val hour = 3_600_000L

    private fun post(src: SentimentSource, text: String, ageH: Long, score: Int = 0, label: String? = null) =
        SocialPost(src, text + ageH, text, createdMs = FIXTURE_NOW_MS - ageH * hour, score = score, explicitSentiment = label)

    @Test
    fun recentAndPopularPostsWeighMore() {
        val fresh = agg.sampleFromPost(post(SentimentSource.REDDIT, "to the moon", 1, 500), FIXTURE_NOW_MS)
        val old = agg.sampleFromPost(post(SentimentSource.REDDIT, "to the moon", 72, 0), FIXTURE_NOW_MS)
        assertTrue(fresh.weight > old.weight * 4)
    }

    @Test
    fun explicitStocktwitsLabelDominates() {
        val s = agg.sampleFromPost(post(SentimentSource.STOCKTWITS, "what do you think", 1, label = "Bearish"), FIXTURE_NOW_MS)
        assertTrue(s.score < -0.3)
    }

    @Test
    fun summaryCombinesSourcesAndHub() {
        val samples = listOf(
            agg.sampleFromPost(post(SentimentSource.REDDIT, "bullish", 2), FIXTURE_NOW_MS),
            agg.sampleFromPost(post(SentimentSource.REDDIT, "bullish", 3), FIXTURE_NOW_MS),
            agg.sampleFromPost(post(SentimentSource.TELEGRAM, "scam", 1), FIXTURE_NOW_MS),
            agg.sampleFromNews(NewsItem("Coin to the moon", publishedMs = FIXTURE_NOW_MS - hour), FIXTURE_NOW_MS),
        )
        val hub = SourceSentiment(SentimentSource.SENTIMENT_HUB, SourceStatus.OK, sampleCount = 100, composite = 0.4)
        val s = agg.summarize("SOL", samples, mapOf(SentimentSource.STOCKTWITS to SourceStatus.RATE_LIMITED), FIXTURE_NOW_MS, listOf(hub))
        assertEquals(104, s.totalSamples)
        assertTrue(s.composite > 0.2, "composite=${s.composite}")
        assertTrue(s.confidence > 0.5)
        assertEquals(SourceStatus.RATE_LIMITED, s.perSource.first { it.source == SentimentSource.STOCKTWITS }.status)
        assertTrue(s.label.contains("ullish"))
    }

    @Test
    fun noSamplesMeansNoData() {
        val s = agg.summarize("X", emptyList(), mapOf(SentimentSource.REDDIT to SourceStatus.UNAVAILABLE), FIXTURE_NOW_MS)
        assertEquals("No data", s.label)
        assertEquals(0.0, s.confidence)
    }
}
