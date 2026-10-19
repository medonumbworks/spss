package dev.cryptopulse.core.repo

import dev.cryptopulse.core.analysis.SentimentAggregator
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.SentimentSample
import dev.cryptopulse.core.model.SentimentSource
import dev.cryptopulse.core.model.SentimentSummary
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.sources.NewsItem
import dev.cryptopulse.core.sources.NewsSource
import dev.cryptopulse.core.sources.RedditSource
import dev.cryptopulse.core.sources.SocialPost
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.sources.StockTwitsSource
import dev.cryptopulse.core.sources.TelegramSource
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.Logger
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.ConcurrentHashMap

/**
 * Collects social chatter and news for one asset (or the whole market) from every enabled free
 * source and turns it into a [SentimentSummary]. Shared feeds (Telegram channels, RSS) are fetched
 * once per TTL and filtered locally so per-asset cost stays at ~3 requests.
 */
class SentimentService(
    private val reddit: RedditSource,
    private val stockTwits: StockTwitsSource,
    private val telegram: TelegramSource,
    private val news: NewsSource,
    private val aggregator: SentimentAggregator,
    private val configProvider: () -> AppConfig,
    private val clock: Clock = Clock.SYSTEM,
    private val logger: Logger = Logger.STDERR,
) {
    private class Cached<T>(val value: T, val atMs: Long)

    private val assetCache = ConcurrentHashMap<String, Cached<SentimentSummary>>()
    private var globalCache: Cached<SentimentSummary>? = null
    private var telegramCache: Cached<Pair<List<SocialPost>, Map<String, SourceStatus>>>? = null
    private var rssCache: Cached<Pair<List<NewsItem>, SourceStatus>>? = null
    private val globalMutex = Mutex()
    private val sharedMutex = Mutex()

    fun cached(symbol: String): SentimentSummary? = assetCache[symbol]?.value

    suspend fun forAsset(asset: Asset, force: Boolean = false): SentimentSummary {
        val config = configProvider()
        val now = clock.nowMs()
        assetCache[asset.symbol]?.let { if (!force && now - it.atMs < config.sentimentTtlMs) return it.value }

        val matcher = MentionMatcher(asset)
        val samples = ArrayList<SentimentSample>()
        val statuses = LinkedHashMap<SentimentSource, SourceStatus>()

        if (config.redditEnabled) {
            val subs = config.subreddits.joinToString(" OR ") { "subreddit:$it" }
            val query = "(${matcher.searchQuery}) ($subs)"
            when (val r = reddit.search(query = query, sort = "new", timeRange = "week", limit = 100)) {
                is SourceResult.Success -> {
                    val posts = r.value.filter { matcher.matches(it.title + " " + it.body) }
                    statuses[SentimentSource.REDDIT] = if (posts.isEmpty()) SourceStatus.EMPTY else SourceStatus.OK
                    posts.forEach { samples += aggregator.sampleFromPost(it, now, asset) }
                }
                is SourceResult.Failure -> statuses[SentimentSource.REDDIT] = r.status.also { logger.debug(TAG, "reddit ${asset.symbol}: ${r.message}") }
            }
        } else statuses[SentimentSource.REDDIT] = SourceStatus.DISABLED

        if (config.stocktwitsEnabled) {
            when (val r = stockTwits.symbolStream(asset.symbol)) {
                is SourceResult.Success -> {
                    statuses[SentimentSource.STOCKTWITS] = if (r.value.isEmpty()) SourceStatus.EMPTY else SourceStatus.OK
                    r.value.forEach { samples += aggregator.sampleFromPost(it, now, asset) }
                }
                is SourceResult.Failure -> statuses[SentimentSource.STOCKTWITS] = r.status.also { logger.debug(TAG, "stocktwits ${asset.symbol}: ${r.message}") }
            }
        } else statuses[SentimentSource.STOCKTWITS] = SourceStatus.DISABLED

        if (config.telegramEnabled) {
            val (posts, channelStatuses) = telegramPosts(config, force)
            val matched = posts.filter { matcher.matches(it.title + " " + it.body) }
            statuses[SentimentSource.TELEGRAM] = when {
                matched.isNotEmpty() -> SourceStatus.OK
                channelStatuses.values.any { it == SourceStatus.OK } -> SourceStatus.EMPTY
                channelStatuses.values.firstOrNull() != null -> channelStatuses.values.first()
                else -> SourceStatus.UNAVAILABLE
            }
            matched.forEach { samples += aggregator.sampleFromPost(it, now, asset) }
        } else statuses[SentimentSource.TELEGRAM] = SourceStatus.DISABLED

        if (config.newsEnabled) {
            var status = SourceStatus.EMPTY
            when (val r = news.googleNews("${matcher.searchQuery} crypto")) {
                is SourceResult.Success -> {
                    val items = r.value.filter { matcher.matches(it.title + " " + (it.summary ?: "")) }
                    if (items.isNotEmpty()) status = SourceStatus.OK
                    items.forEach { samples += aggregator.sampleFromNews(it, now, asset) }
                }
                is SourceResult.Failure -> status = r.status.also { logger.debug(TAG, "news ${asset.symbol}: ${r.message}") }
            }
            val (feedItems, feedStatus) = rssItems(config, force)
            val matchedFeed = feedItems.filter { matcher.matches(it.title + " " + (it.summary ?: "")) }
            if (matchedFeed.isNotEmpty()) status = SourceStatus.OK
            else if (status == SourceStatus.EMPTY && feedStatus != SourceStatus.OK && feedStatus != SourceStatus.EMPTY) status = feedStatus
            matchedFeed.forEach { samples += aggregator.sampleFromNews(it, now, asset) }
            statuses[SentimentSource.NEWS] = status
        } else statuses[SentimentSource.NEWS] = SourceStatus.DISABLED

        val summary = aggregator.summarize(asset.symbol, dedupe(samples), statuses, now)
        assetCache[asset.symbol] = Cached(summary, now)
        return summary
    }

    /** Market-wide mood: r/CryptoCurrency front page, crypto news feeds and Telegram channels. */
    suspend fun globalMood(force: Boolean = false): SentimentSummary = globalMutex.withLock {
        val config = configProvider()
        val now = clock.nowMs()
        globalCache?.let { if (!force && now - it.atMs < config.sentimentTtlMs) return@withLock it.value }
        val samples = ArrayList<SentimentSample>()
        val statuses = LinkedHashMap<SentimentSource, SourceStatus>()
        if (config.redditEnabled) {
            when (val r = reddit.hot("CryptoCurrency", 100)) {
                is SourceResult.Success -> {
                    statuses[SentimentSource.REDDIT] = if (r.value.isEmpty()) SourceStatus.EMPTY else SourceStatus.OK
                    r.value.forEach { samples += aggregator.sampleFromPost(it, now, null) }
                }
                is SourceResult.Failure -> statuses[SentimentSource.REDDIT] = r.status
            }
        } else statuses[SentimentSource.REDDIT] = SourceStatus.DISABLED
        if (config.newsEnabled) {
            val (items, status) = rssItems(config, force)
            statuses[SentimentSource.NEWS] = status
            items.forEach { samples += aggregator.sampleFromNews(it, now, null) }
        } else statuses[SentimentSource.NEWS] = SourceStatus.DISABLED
        if (config.telegramEnabled) {
            val (posts, channelStatuses) = telegramPosts(config, force)
            statuses[SentimentSource.TELEGRAM] = when {
                posts.isNotEmpty() -> SourceStatus.OK
                channelStatuses.isEmpty() -> SourceStatus.UNAVAILABLE
                else -> channelStatuses.values.first()
            }
            posts.forEach { samples += aggregator.sampleFromPost(it, now, null) }
        } else statuses[SentimentSource.TELEGRAM] = SourceStatus.DISABLED
        statuses[SentimentSource.STOCKTWITS] = SourceStatus.DISABLED
        val summary = aggregator.summarize(GLOBAL_SYMBOL, dedupe(samples), statuses, now)
        globalCache = Cached(summary, now)
        summary
    }

    private suspend fun telegramPosts(config: AppConfig, force: Boolean): Pair<List<SocialPost>, Map<String, SourceStatus>> = sharedMutex.withLock {
        val now = clock.nowMs()
        telegramCache?.let { if (!force && now - it.atMs < config.sentimentTtlMs) return@withLock it.value }
        val posts = ArrayList<SocialPost>()
        val statuses = LinkedHashMap<String, SourceStatus>()
        for (channel in config.telegramChannels) {
            when (val r = telegram.channelMessages(channel)) {
                is SourceResult.Success -> { statuses[channel] = if (r.value.isEmpty()) SourceStatus.EMPTY else SourceStatus.OK; posts += r.value }
                is SourceResult.Failure -> { statuses[channel] = r.status; logger.debug(TAG, "telegram $channel: ${r.message}") }
            }
        }
        val value = posts.toList() to statuses.toMap()
        telegramCache = Cached(value, now)
        value
    }

    private suspend fun rssItems(config: AppConfig, force: Boolean): Pair<List<NewsItem>, SourceStatus> = sharedMutex.withLock {
        val now = clock.nowMs()
        rssCache?.let { if (!force && now - it.atMs < config.sentimentTtlMs) return@withLock it.value }
        val items = ArrayList<NewsItem>()
        var status = SourceStatus.UNAVAILABLE
        var anyOk = false
        for ((name, url) in DEFAULT_FEEDS) {
            when (val r = news.rssFeed(url, name)) {
                is SourceResult.Success -> { anyOk = true; items += r.value }
                is SourceResult.Failure -> { status = r.status; logger.debug(TAG, "rss $name: ${r.message}") }
            }
        }
        if (anyOk) status = if (items.isEmpty()) SourceStatus.EMPTY else SourceStatus.OK
        val value = items.toList() to status
        rssCache = Cached(value, now)
        value
    }

    private fun dedupe(samples: List<SentimentSample>): List<SentimentSample> {
        val seen = HashSet<String>()
        return samples.filter { s -> seen.add((s.url ?: "") + "|" + s.text.lowercase().take(80)) }
    }

    companion object {
        const val TAG = "Sentiment"
        const val GLOBAL_SYMBOL = "MARKET"

        /** Free crypto news feeds (name -> URL). */
        val DEFAULT_FEEDS: List<Pair<String, String>> = listOf(
            "CoinTelegraph" to "https://cointelegraph.com/rss",
            "CoinDesk" to "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "Decrypt" to "https://decrypt.co/feed",
        )
    }
}
