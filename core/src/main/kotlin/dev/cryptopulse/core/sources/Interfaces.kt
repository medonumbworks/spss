package dev.cryptopulse.core.sources

import dev.cryptopulse.core.model.CoinCommunity
import dev.cryptopulse.core.model.DerivativesSnapshot
import dev.cryptopulse.core.model.FearGreed
import dev.cryptopulse.core.model.GlobalStats
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.PricePoint
import dev.cryptopulse.core.model.SentimentSource
import dev.cryptopulse.core.model.TrendingCoin
import kotlinx.serialization.Serializable

// ---------------------------------------------------------------------------------------------
// Bitpanda
// ---------------------------------------------------------------------------------------------

/** One symbol of the legacy ticker: prices keyed by fiat code ("EUR", "USD", "CHF", "GBP", "TRY", ...). */
@Serializable
data class BitpandaTickerEntry(val symbol: String, val prices: Map<String, Double>)

@Serializable
data class BitpandaLegacyTicker(val entries: List<BitpandaTickerEntry>, val fetchedAtMs: Long)

/** Asset from `GET https://api.public.bitpanda.com/assets` (new Public API). */
@Serializable
data class BitpandaPublicAsset(
    val id: String,
    val symbol: String,
    val name: String? = null,
    /** "CRYPTO", "COMMODITY", "STOCK", "ETF", ... as returned by the API. */
    val type: String? = null,
    val group: String? = null,
    val isin: String? = null,
)

@Serializable
data class BitpandaPublicTicker(val assetId: String, val price: Double, val currency: String, val timestamp: String? = null)

interface BitpandaSource {
    /** Legacy `GET https://api.bitpanda.com/v1/ticker`: every asset with prices in several fiat currencies. */
    suspend fun legacyTicker(apiKey: String? = null): SourceResult<BitpandaLegacyTicker>

    /** New Public API asset list, follows the cursor until the end. `type` filters e.g. "CRYPTO". */
    suspend fun publicAssets(apiKey: String? = null, type: String? = "CRYPTO"): SourceResult<List<BitpandaPublicAsset>>

    /** New Public API ticker for one asset UUID. */
    suspend fun publicTicker(assetId: String, apiKey: String? = null): SourceResult<BitpandaPublicTicker>
}

// ---------------------------------------------------------------------------------------------
// CoinGecko
// ---------------------------------------------------------------------------------------------

@Serializable
data class MarketChart(
    /** Oldest first. */
    val prices: List<PricePoint>,
    val volumes: List<PricePoint> = emptyList(),
    val marketCaps: List<PricePoint> = emptyList(),
)

interface CoinGeckoSource {
    /**
     * `GET /coins/markets`. When [ids] is null the page is ordered by market cap (used to map symbols);
     * otherwise only those ids are returned (max [perPage] per call).
     */
    suspend fun markets(
        vsCurrency: String,
        page: Int = 1,
        perPage: Int = 250,
        ids: List<String>? = null,
        sparkline: Boolean = true,
    ): SourceResult<List<MarketData>>

    /** `GET /coins/{id}/market_chart?vs_currency=..&days=..` (daily granularity for days > 90, hourly for 2..90). */
    suspend fun marketChart(id: String, vsCurrency: String, days: Int): SourceResult<MarketChart>

    /** `GET /coins/{id}` with community data and sentiment votes. */
    suspend fun coinCommunity(id: String): SourceResult<CoinCommunity>

    /** `GET /search/trending`. */
    suspend fun trending(): SourceResult<List<TrendingCoin>>

    /** `GET /global`. */
    suspend fun global(): SourceResult<GlobalStats>
}

// ---------------------------------------------------------------------------------------------
// Social and news
// ---------------------------------------------------------------------------------------------

/** A post/message from Reddit, StockTwits or Telegram, normalized. */
@Serializable
data class SocialPost(
    val source: SentimentSource,
    val id: String,
    val title: String,
    val body: String = "",
    val url: String? = null,
    val author: String? = null,
    val createdMs: Long,
    /** Upvotes / likes / views, whatever the platform exposes. */
    val score: Int = 0,
    val comments: Int = 0,
    /** Subreddit, channel handle or stream name. */
    val community: String? = null,
    /** "Bullish" / "Bearish" when the platform labels it (StockTwits). */
    val explicitSentiment: String? = null,
    /** Upvote ratio when known (Reddit). */
    val upvoteRatio: Double? = null,
)

@Serializable
data class NewsItem(
    val title: String,
    val summary: String? = null,
    val url: String? = null,
    val sourceName: String? = null,
    val publishedMs: Long? = null,
)

interface RedditSource {
    /** Site-wide or subreddit search (`/search.json` or `/r/{sub}/search.json?restrict_sr=1`). */
    suspend fun search(
        query: String,
        subreddit: String? = null,
        sort: String = "new",
        timeRange: String = "week",
        limit: Int = 100,
    ): SourceResult<List<SocialPost>>

    /** `/r/{sub}/hot.json`, used for the global crypto mood. */
    suspend fun hot(subreddit: String, limit: Int = 100): SourceResult<List<SocialPost>>
}

interface StockTwitsSource {
    /** `/api/2/streams/symbol/{SYMBOL}.X.json` (crypto symbols carry the `.X` suffix). */
    suspend fun symbolStream(symbol: String): SourceResult<List<SocialPost>>
}

interface TelegramSource {
    /** Public channel preview `https://t.me/s/{channel}` (HTML), newest ~20 messages. */
    suspend fun channelMessages(channel: String): SourceResult<List<SocialPost>>
}

interface NewsSource {
    /** Google News RSS search, e.g. `"Solana" crypto`. */
    suspend fun googleNews(query: String, lang: String = "en-US", country: String = "US"): SourceResult<List<NewsItem>>

    /** Any RSS/Atom feed (CoinTelegraph, CoinDesk, Decrypt, ...). */
    suspend fun rssFeed(feedUrl: String, sourceName: String): SourceResult<List<NewsItem>>
}

interface FearGreedSource {
    /** alternative.me Fear & Greed with history (oldest first). */
    suspend fun fetch(days: Int = 30): SourceResult<FearGreed>
}

interface DerivativesSource {
    /** Funding rates for all perpetual markets in one call, keyed by base symbol ("BTC"). */
    suspend fun fundingRates(): SourceResult<Map<String, DerivativesSnapshot>>

    /** Long/short account ratio for one base symbol, when the venue exposes it. */
    suspend fun longShortRatio(baseSymbol: String): SourceResult<Double?>
}
