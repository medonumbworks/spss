package dev.cryptopulse.core.sources.feed

import kotlinx.serialization.Serializable

/** One row of the Telegram buzz ranking; samples keep only posts we can link to. */
@Serializable
data class BuzzEntry(
    val symbol: String,
    val mentions: Int,
    val mentionsPrev: Int?,
    val baseline7d: Double?,
    val spike: Double?,
    val composite: Double,
    val samples: List<FeedSample>,
)

/**
 * Ranks the feed's assets by how loudly Telegram is buzzing about them. This backs the app's
 * Telegram tab and the MCP `telegram_buzz` tool. `spike` is mentions vs the 7-day baseline the
 * hub publishes, falling back to null (the UI then shows mentions vs the previous window).
 *
 * Only assets with mentions are listed, and samples are filtered to those with a url so every
 * row can link to its posts. Order: spike descending with nulls last, then mentions descending,
 * then symbol.
 */
fun SentimentFeedDoc.telegramBuzz(limit: Int = 50): List<BuzzEntry> = assets.values
    .filter { it.mentions > 0 }
    .map { a ->
        BuzzEntry(
            symbol = a.symbol,
            mentions = a.mentions,
            mentionsPrev = a.mentionsPrev,
            baseline7d = a.baseline7d,
            spike = a.spike,
            composite = a.composite,
            samples = a.samples.filter { it.url != null },
        )
    }
    .sortedWith(compareBy<BuzzEntry, Double?>(nullsLast(reverseOrder())) { it.spike }.thenByDescending { it.mentions }.thenBy { it.symbol })
    .take(limit)
