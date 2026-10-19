package dev.cryptopulse.core.repo

import dev.cryptopulse.core.analysis.SentimentAggregator
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.SentimentSample
import dev.cryptopulse.core.model.SentimentSource
import dev.cryptopulse.core.model.SentimentSummary
import dev.cryptopulse.core.model.SourceSentiment
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.sources.NewsItem
import dev.cryptopulse.core.sources.NewsSource
import dev.cryptopulse.core.sources.RedditSource
import dev.cryptopulse.core.sources.SocialPost
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.sources.StockTwitsSource
import dev.cryptopulse.core.sources.TelegramSource
import dev.cryptopulse.core.sources.feed.FeedAsset
import dev.cryptopulse.core.sources.feed.SentimentFeedApi
import dev.cryptopulse.core.sources.feed.SentimentFeedDoc
import dev.cryptopulse.core.telegram.TelegramDirectory
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.Logger
import dev.cryptopulse.core.util.Text
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.ConcurrentHashMap

/**
 * Collects social chatter and news for one asset (or the whole market) from every enabled free source
 * and turns it into a [SentimentSummary]:
 *  - cloud hub feed (one request for all assets; FinBERT/CryptoBERT/VADER over 100+ Telegram channels,
 *    the user's own Telegram account when connected, and news),
 *  - on-device: Reddit search, StockTwits, public Telegram channels (t.me/s), Google News + crypto RSS,
 *    all scored with VADER + crypto slang.
 * Shared feeds (Telegram channels, RSS) are fetched once per TTL and filtered locally per asset.
 */
class SentimentService(
    private val reddit: RedditSource,
    private val stockTwits: StockTwitsSource,
    private val telegram: TelegramSource,
    private val news: NewsSource,
    private val aggregator: SentimentAggregator,
    private val configProvider: () -> AppConfig,
    private val feedApi: SentimentFeedApi? = null,
    private val directory: TelegramDirectory = TelegramDirectory.loadBundled(),
    private val clock: Clock = Clock.SYSTEM,
    private val logger: Logger = Logger.STDERR,
) {
    private class Cached<T>(val value: T, val atMs: Long)

    private val assetCache = ConcurrentHashMap<String, Cached<SentimentSummary>>()
    private var globalCache: Cached<SentimentSummary>? = null
    private var telegramCache: Cached<Pair<List<SocialPost>, Map<String, SourceStatus>>>? = null
    private var rssCache: Cached<Pair<List<NewsItem>, SourceStatus>>? = null
    private var feedCache: Cached<SentimentFeedDoc?>? = null
    private var feedStatus: SourceStatus = SourceStatus.DISABLED
    private var feedMessage: String? = null
    private val globalMutex = Mutex()
    private val sharedMutex = Mutex()
    private val feedMutex = Mutex()

    /** Full (on-device + hub) summary computed earlier for this symbol, if any. */
    fun cached(symbol: String): SentimentSummary? = assetCache[symbol]?.value

    fun cachedGlobalMood(): SentimentSummary? = globalCache?.value

    val telegramDirectory: TelegramDirectory get() = directory

    /** Status of the cloud hub feed after the last [feed] call, for the UI ("hub not published yet", ...). */
    fun feedState(): Pair<SourceStatus, String?> = feedStatus to feedMessage

    /** Cloud hub feed (cached ~20 min). Never throws; null when disabled or unavailable. */
    suspend fun feed(force: Boolean = false): SentimentFeedDoc? = feedMutex.withLock {
        val api = feedApi ?: return@withLock null
        val now = clock.nowMs()
        feedCache?.let { if (!force && now - it.atMs < FEED_TTL_MS) return@withLock it.value }
        val doc = when (val r = api.fetch()) {
            is SourceResult.Success -> { feedStatus = SourceStatus.OK; feedMessage = null; r.value }
            is SourceResult.Failure -> {
                feedStatus = r.status
                feedMessage = r.message
                logger.debug(TAG, "feed: ${r.message}")
                feedCache?.value
            }
        }
        feedCache = Cached(doc, now)
        doc
    }

    /** Fast path for the market list: hub data only, no per-asset network calls. */
    fun fromFeed(asset: Asset, doc: SentimentFeedDoc?): SentimentSummary? {
        val fa = doc?.assets?.get(asset.symbol) ?: return null
        val now = clock.nowMs()
        return aggregator.summarize(asset.symbol, emptyList(), emptyMap(), now, listOf(hubSource(fa, doc)))
    }

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
            val matched = posts.filter { matcher.matches(it.title + " " + it.body) }.toMutableList()
            // official/project channels of this asset: everything they post is about the asset
            for (entry in directory.channelsFor(asset.symbol).take(3)) {
                when (val r = telegram.channelMessages(entry.handle)) {
                    is SourceResult.Success -> matched += r.value.filter { now - it.createdMs < WEEK_MS }
                    is SourceResult.Failure -> logger.debug(TAG, "telegram project ${entry.handle}: ${r.message}")
                }
            }
            statuses[SentimentSource.TELEGRAM] = when {
                matched.isNotEmpty() -> SourceStatus.OK
                channelStatuses.values.any { it == SourceStatus.OK } -> SourceStatus.EMPTY
                channelStatuses.values.firstOrNull() != null -> channelStatuses.values.first()
                else -> SourceStatus.UNAVAILABLE
            }
            matched.distinctBy { it.id }.forEach { samples += aggregator.sampleFromPost(it, now, asset) }
        } else statuses[SentimentSource.TELEGRAM] = SourceStatus.DISABLED

        if (config.newsEnabled) {
            var status = SourceStatus.EMPTY
            when (val r = news.googleNews("${matcher.searchQuery} crypto")) {
                is SourceResult.Success -> {
                    val items = r.value.filter { matcher.matches(it.title + " " + (it.summary ?: "")) }
                    if (items.isNotEmpty()) status = SourceStatus.OK
                    items.take(40).forEach { samples += aggregator.sampleFromNews(it, now, asset) }
                }
                is SourceResult.Failure -> status = r.status.also { logger.debug(TAG, "news ${asset.symbol}: ${r.message}") }
            }
            val (feedItems, feedStatusRss) = rssItems(config, force)
            val matchedFeed = feedItems.filter { matcher.matches(it.title + " " + (it.summary ?: "")) }
            if (matchedFeed.isNotEmpty()) status = SourceStatus.OK
            else if (status == SourceStatus.EMPTY && feedStatusRss != SourceStatus.OK && feedStatusRss != SourceStatus.EMPTY) status = feedStatusRss
            matchedFeed.forEach { samples += aggregator.sampleFromNews(it, now, asset) }
            statuses[SentimentSource.NEWS] = status
        } else statuses[SentimentSource.NEWS] = SourceStatus.DISABLED

        val hub = hubFor(asset.symbol)
        val summary = aggregator.summarize(asset.symbol, dedupe(samples), statuses, now, listOfNotNull(hub))
        assetCache[asset.symbol] = Cached(summary, now)
        return summary
    }

    /** Market-wide mood: r/CryptoCurrency front page, crypto news feeds, Telegram channels and the hub. */
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
        val hubMarket = feed()?.let { doc ->
            doc.market?.takeIf { it.mentions > 0 }?.let { m ->
                SourceSentiment(
                    source = SentimentSource.SENTIMENT_HUB,
                    status = SourceStatus.OK,
                    sampleCount = m.mentions,
                    composite = m.composite,
                    bullishShare = m.bullish,
                    bearishShare = m.bearish,
                    neutralShare = m.neutral,
                    buzzPerDay = m.mentions * 24.0 / doc.windowHours.coerceAtLeast(1),
                    note = hubNote(doc),
                )
            }
        }
        val summary = aggregator.summarize(GLOBAL_SYMBOL, dedupe(samples), statuses, now, listOfNotNull(hubMarket))
        globalCache = Cached(summary, now)
        summary
    }

    /** Channels the phone scans: user overrides first, then the curated directory. */
    fun channelsToScan(config: AppConfig): List<String> {
        val max = if (config.telegramScanAll) Int.MAX_VALUE else config.telegramMaxChannels.coerceAtLeast(0)
        val fromDirectory = directory.publicChannels().map { it.handle }
        return (config.telegramChannels + fromDirectory).distinctBy { it.lowercase() }.take(max)
    }

    private suspend fun hubFor(symbol: String): SourceSentiment? {
        val doc = try {
            feed()
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            null
        }
        val fa = doc?.assets?.get(symbol) ?: return null
        return hubSource(fa, doc)
    }

    private fun hubSource(fa: FeedAsset, doc: SentimentFeedDoc): SourceSentiment {
        val samples = fa.samples.map { s ->
            SentimentSample(
                source = SentimentSource.SENTIMENT_HUB,
                text = Text.truncate(s.text, 220),
                score = s.score.coerceIn(-1.0, 1.0),
                url = s.url,
                community = s.community ?: s.source,
                timestampMs = s.tsMs,
            )
        }
        val buzz = fa.mentionsPrev?.takeIf { it > 0 }?.let { fa.mentions.toDouble() / it }
        return SourceSentiment(
            source = SentimentSource.SENTIMENT_HUB,
            status = SourceStatus.OK,
            sampleCount = fa.mentions,
            composite = fa.composite,
            bullishShare = fa.bullish,
            bearishShare = fa.bearish,
            neutralShare = fa.neutral,
            buzzPerDay = fa.mentions * 24.0 / doc.windowHours.coerceAtLeast(1),
            topSamples = samples,
            note = hubNote(doc),
            buzzRatio = buzz,
            modelScores = fa.scores,
        )
    }

    private fun hubNote(doc: SentimentFeedDoc): String {
        val ageMin = doc.ageMs(clock.nowMs()) / 60_000
        val tg = doc.sources.filter { it.name.startsWith("telegram") && it.status == "ok" }.joinToString { "${it.name.replace('_', ' ')}: ${it.detail ?: it.items}" }
        return buildString {
            append("Cloud hub, ")
            append(if (ageMin < 120) "$ageMin min old" else "${ageMin / 60} h old")
            if (doc.models.isNotEmpty()) append(", models: ").append(doc.models.joinToString())
            if (tg.isNotEmpty()) append(", ").append(tg)
        }
    }

    private suspend fun telegramPosts(config: AppConfig, force: Boolean): Pair<List<SocialPost>, Map<String, SourceStatus>> = sharedMutex.withLock {
        val now = clock.nowMs()
        telegramCache?.let { if (!force && now - it.atMs < config.sentimentTtlMs) return@withLock it.value }
        val posts = ArrayList<SocialPost>()
        val statuses = LinkedHashMap<String, SourceStatus>()
        for (channel in channelsToScan(config)) {
            when (val r = telegram.channelMessages(channel)) {
                is SourceResult.Success -> {
                    val recent = r.value.filter { now - it.createdMs < WEEK_MS }
                    statuses[channel] = if (recent.isEmpty()) SourceStatus.EMPTY else SourceStatus.OK
                    posts += recent
                }
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
        private const val FEED_TTL_MS = 20 * 60_000L
        private const val WEEK_MS = 7 * 86_400_000L

        /** Free crypto news feeds (name -> URL). */
        val DEFAULT_FEEDS: List<Pair<String, String>> = listOf(
            "CoinTelegraph" to "https://cointelegraph.com/rss",
            "CoinDesk" to "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "Decrypt" to "https://decrypt.co/feed",
            "Bitcoin Magazine" to "https://bitcoinmagazine.com/.rss/full/",
        )
    }
}
