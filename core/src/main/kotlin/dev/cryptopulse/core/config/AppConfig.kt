package dev.cryptopulse.core.config

import kotlinx.serialization.Serializable

/**
 * Runtime configuration. Everything has a sensible default so the app works with zero input;
 * optional keys only unlock higher rate limits or the authenticated Bitpanda endpoints.
 */
@Serializable
data class AppConfig(
    /** Quote currency for prices: EUR (Bitpanda's home currency), USD, CHF, GBP. */
    val currency: String = "EUR",
    /** Optional CoinGecko Demo key (x_cg_demo_api_key) for 30 req/min instead of ~5-10. */
    val coingeckoDemoKey: String? = null,
    /** Optional Bitpanda API key for api.public.bitpanda.com when keyless access is refused. */
    val bitpandaApiKey: String? = null,
    val redditEnabled: Boolean = true,
    val stocktwitsEnabled: Boolean = true,
    val telegramEnabled: Boolean = true,
    val newsEnabled: Boolean = true,
    val derivativesEnabled: Boolean = true,
    val fearGreedEnabled: Boolean = true,
    /**
     * Extra/override public Telegram channels scanned via t.me/s/<channel>. Empty = use the bundled
     * directory `telegram/channels.json` (hundreds of curated channels).
     */
    val telegramChannels: List<String> = emptyList(),
    /** How many directory channels the phone scans itself (the cloud hub scans all of them). */
    val telegramMaxChannels: Int = 40,
    /** Scan every channel in the directory on the device too (slower, uses more data). */
    val telegramScanAll: Boolean = false,
    /** Free cloud sentiment feed published by the GitHub Actions hub (tools/sentiment_hub). Empty = off. */
    val sentimentFeedUrl: String = DEFAULT_SENTIMENT_FEED_URL,
    /** Subreddits searched for per-asset sentiment (in addition to a site-wide search). */
    val subreddits: List<String> = DEFAULT_SUBREDDITS,
    /** How many top-market-cap assets get per-asset sentiment in the background after each refresh. */
    val backgroundSentimentTopN: Int = 25,
    val marketsTtlMs: Long = 3 * 60_000L,
    val universeTtlMs: Long = 24 * 60 * 60_000L,
    val chartTtlMs: Long = 15 * 60_000L,
    val sentimentTtlMs: Long = 30 * 60_000L,
    val contextTtlMs: Long = 10 * 60_000L,
    /** Max assets per CoinGecko `/coins/markets` request. */
    val coingeckoPageSize: Int = 250,
    /** How many CoinGecko market pages (250 each) to scan when mapping symbols to ids. */
    val coingeckoMappingPages: Int = 4,
) {
    companion object {
        const val DEFAULT_SENTIMENT_FEED_URL: String = "https://raw.githubusercontent.com/medonumbworks/spss/sentiment-feed/feed.json"
        val DEFAULT_SUBREDDITS: List<String> = listOf(
            "CryptoCurrency",
            "CryptoMarkets",
            "altcoin",
            "SatoshiStreetBets",
        )
        val SUPPORTED_CURRENCIES: List<String> = listOf("EUR", "USD", "CHF", "GBP")
    }
}
