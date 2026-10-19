package dev.cryptopulse.core

import dev.cryptopulse.core.analysis.Forecaster
import dev.cryptopulse.core.analysis.SentimentScorer
import dev.cryptopulse.core.analysis.forecast.DefaultForecastEngine
import dev.cryptopulse.core.analysis.sentiment.DefaultSentimentAggregator
import dev.cryptopulse.core.analysis.sentiment.LexiconSentimentScorer
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.http.DiskStore
import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpCache
import dev.cryptopulse.core.http.HttpTransport
import dev.cryptopulse.core.http.OkHttpTransport
import dev.cryptopulse.core.http.RateLimiter
import dev.cryptopulse.core.repo.MarketRepository
import dev.cryptopulse.core.repo.SentimentService
import dev.cryptopulse.core.sources.bitpanda.BitpandaApi
import dev.cryptopulse.core.sources.coingecko.CoinGeckoApi
import dev.cryptopulse.core.sources.feed.SentimentFeedApi
import dev.cryptopulse.core.sources.market.DerivativesApi
import dev.cryptopulse.core.sources.market.FearGreedApi
import dev.cryptopulse.core.sources.social.NewsFeeds
import dev.cryptopulse.core.sources.social.RedditApi
import dev.cryptopulse.core.sources.social.StockTwitsApi
import dev.cryptopulse.core.sources.social.TelegramWeb
import dev.cryptopulse.core.telegram.TelegramDirectory
import dev.cryptopulse.core.universe.BundledSnapshot
import dev.cryptopulse.core.universe.DefaultAssetUniverse
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.Logger

/** Wires every free data source, the sentiment pipeline and the forecast engine together. Used by the app and the MCP server. */
class CryptoPulse private constructor(
    val repository: MarketRepository,
    val sentiment: SentimentService,
    val http: Http,
    private val configHolder: ConfigHolder,
) {
    val config: AppConfig get() = configHolder.config

    fun updateConfig(config: AppConfig) {
        configHolder.config = config
        repository.updateConfig(config)
    }

    internal class ConfigHolder(@Volatile var config: AppConfig)

    companion object {
        /**
         * @param scorer on-device text scorer; defaults to VADER with the crypto slang lexicon.
         * @param transport network layer; tests and the offline demo pass a fake.
         * @param store optional persistence (Android cache dir, a temp dir for the MCP server).
         */
        fun create(
            config: AppConfig = AppConfig(),
            transport: HttpTransport = OkHttpTransport(),
            store: DiskStore? = null,
            scorer: SentimentScorer = defaultScorer(),
            forecaster: Forecaster = DefaultForecastEngine(),
            clock: Clock = Clock.SYSTEM,
            logger: Logger = Logger.STDERR,
            rateLimiter: RateLimiter = RateLimiter(clock = clock),
            telegramDirectory: TelegramDirectory = TelegramDirectory.loadBundled(),
        ): CryptoPulse {
            val holder = ConfigHolder(config)
            val http = Http(transport, rateLimiter, HttpCache(store, clock), clock, logger)
            val coinGecko = CoinGeckoApi(http, demoKeyProvider = { holder.config.coingeckoDemoKey }, marketsTtlMs = config.marketsTtlMs)
            val bitpanda = BitpandaApi(http, clock)
            val universe = DefaultAssetUniverse(bitpanda, coinGecko, { BundledSnapshot.load() }, store, clock, logger)
            val sentiment = SentimentService(
                reddit = RedditApi(http),
                stockTwits = StockTwitsApi(http),
                telegram = TelegramWeb(http),
                news = NewsFeeds(http),
                aggregator = DefaultSentimentAggregator(scorer),
                configProvider = { holder.config },
                feedApi = SentimentFeedApi(http, { holder.config.sentimentFeedUrl }),
                directory = telegramDirectory,
                clock = clock,
                logger = logger,
            )
            val repository = MarketRepository(
                universe = universe,
                coinGecko = coinGecko,
                fearGreed = FearGreedApi(http),
                derivatives = DerivativesApi(http, clock),
                sentiment = sentiment,
                forecaster = forecaster,
                store = store,
                clock = clock,
                logger = logger,
                initialConfig = config,
            )
            return CryptoPulse(repository, sentiment, http, holder)
        }

        /** VADER (vaderSentiment 3.3.2 port) with the crypto slang lexicon; loaded once and thread-safe. */
        fun defaultScorer(): SentimentScorer = LexiconSentimentScorer()
    }
}
