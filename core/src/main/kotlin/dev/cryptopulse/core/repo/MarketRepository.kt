package dev.cryptopulse.core.repo

import dev.cryptopulse.core.analysis.ForecastInput
import dev.cryptopulse.core.analysis.Forecaster
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.http.DiskStore
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.AssetDetail
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.CoinCommunity
import dev.cryptopulse.core.analysis.risk.RiskScorer
import dev.cryptopulse.core.model.DerivativesSnapshot
import dev.cryptopulse.core.model.DexInfo
import dev.cryptopulse.core.model.Forecast
import dev.cryptopulse.core.model.MarketContext
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.ListingSource
import dev.cryptopulse.core.model.MarketSnapshot
import dev.cryptopulse.core.model.PricePoint
import dev.cryptopulse.core.model.RefreshStatus
import dev.cryptopulse.core.model.RiskAssessment
import dev.cryptopulse.core.model.SentimentSummary
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.sources.CoinGeckoSource
import dev.cryptopulse.core.sources.DerivativesSource
import dev.cryptopulse.core.sources.FearGreedSource
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.sources.cex.CexDataApi
import dev.cryptopulse.core.sources.dex.GeckoTerminalApi
import dev.cryptopulse.core.sources.safety.TokenSafetyApi
import dev.cryptopulse.core.universe.AssetUniverse
import dev.cryptopulse.core.universe.UniverseResult
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.Logger
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.Json
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.tanh

/**
 * Single entry point for the app and the MCP server:
 *  - [refresh] builds the full market list (Bitpanda universe -> CoinGecko markets -> context -> forecasts),
 *  - [detail] loads everything for one asset (daily history, community votes, derivatives, per-asset sentiment),
 *  - [enrichSentimentForTop] adds per-asset sentiment for the largest assets in the background.
 *
 * All methods are safe to call concurrently; refreshes are serialized. Results are exposed as
 * [StateFlow]s and the last snapshot is persisted so a cold start shows data immediately.
 */
class MarketRepository(
    private val universe: AssetUniverse,
    private val coinGecko: CoinGeckoSource,
    private val fearGreed: FearGreedSource,
    private val derivatives: DerivativesSource,
    private val sentiment: SentimentService,
    private val forecaster: Forecaster,
    private val store: DiskStore? = null,
    private val clock: Clock = Clock.SYSTEM,
    private val logger: Logger = Logger.STDERR,
    initialConfig: AppConfig = AppConfig(),
    /** Memecoins, trending coins and DEX tokens beyond Bitpanda; null = Bitpanda only. */
    private val speculative: SpeculativeService? = null,
    /** Binance mirror klines and OKX funding: history/funding fallbacks for the detail view. */
    private val cex: CexDataApi? = null,
    private val geckoTerminal: GeckoTerminalApi? = null,
    private val safetyApi: TokenSafetyApi? = null,
) {
    @Volatile
    var config: AppConfig = initialConfig
        private set

    private val _snapshot = MutableStateFlow<MarketSnapshot?>(null)
    val snapshot: StateFlow<MarketSnapshot?> = _snapshot.asStateFlow()

    private val _status = MutableStateFlow<RefreshStatus>(RefreshStatus.Idle)
    val status: StateFlow<RefreshStatus> = _status.asStateFlow()

    private val refreshMutex = Mutex()
    private val detailMutex = Mutex()

    @Volatile
    private var lastUniverse: UniverseResult? = null

    @Volatile
    private var lastContext: MarketContext? = null

    /** USD -> quote currency, for prices that only exist in USD (DEX pools). */
    @Volatile
    private var lastUsdToQuote: Double? = null

    @Volatile
    private var fundingCache: Pair<Long, Map<String, DerivativesSnapshot>>? = null

    private val detailCache = ConcurrentHashMap<String, AssetDetail>()

    init {
        // keep the persisted market context so forecasts made before the first network refresh use it
        loadPersisted()?.let { _snapshot.value = it; lastContext = it.context }
    }

    fun updateConfig(newConfig: AppConfig) {
        val currencyChanged = newConfig.currency != config.currency
        config = newConfig
        if (currencyChanged) {
            detailCache.clear()
            lastContext = null
        }
    }

    fun row(symbol: String): AssetRow? = _snapshot.value?.rows?.firstOrNull { it.asset.symbol == symbol }

    /** True when the current snapshot is older than the configured TTL or missing. */
    fun isStale(): Boolean {
        val snap = _snapshot.value ?: return true
        return snap.currency != config.currency || clock.nowMs() - snap.fetchedAtMs >= config.marketsTtlMs
    }

    suspend fun refresh(force: Boolean = false): MarketSnapshot = refreshMutex.withLock {
        val cfg = config
        val current = _snapshot.value
        if (!force && current != null && current.currency == cfg.currency && clock.nowMs() - current.fetchedAtMs < cfg.marketsTtlMs) {
            return@withLock current
        }
        try {
            setStatus("Loading Bitpanda asset list", 0.05f)
            val uni = universe.load(cfg, force)
            lastUniverse = uni
            val warnings = ArrayList(uni.warnings)

            setStatus("Fetching market data for ${uni.assets.size} assets", 0.2f)
            val markets = fetchMarkets(uni, cfg, current, warnings)

            setStatus("Reading market mood", 0.55f)
            val context = loadContext(markets, cfg, force, warnings)
            val funding = loadFunding(cfg, force, warnings)

            setStatus("Reading the cloud sentiment hub", 0.7f)
            val feed = if (cfg.sentimentFeedUrl.isNotBlank()) sentiment.feed(force) else null
            sentiment.feedState().let { (st, msg) -> if (feed == null && msg != null && st != SourceStatus.DISABLED) warnings += msg }

            setStatus("Computing forecasts", 0.8f)
            val nowMs = clock.nowMs()
            val visible = visibleAssets(uni, markets, warnings)
            val previousRows = current?.rows?.associateBy { it.asset.symbol }.orEmpty()
            val bitpandaRows = visible.map { asset ->
                val market = markets[asset.symbol]
                val sent = sentiment.cached(asset.symbol) ?: sentiment.fromFeed(asset, feed) ?: previousRows[asset.symbol]?.sentiment
                val forecast = market?.let { safeForecast(asset, it, emptyList(), emptyList(), context, sent, funding[asset.symbol], null, nowMs) }
                AssetRow(asset, market, forecast, sent)
            }

            val usdToQuote = usdRate(uni, cfg, markets, funding)
            lastUsdToQuote = usdToQuote
            val spec = speculative?.let {
                setStatus("Finding memecoins and DEX tokens", 0.85f)
                try {
                    it.load(cfg, bitpandaRows, context.trendingCoingeckoIds, usdToQuote, current?.rows.orEmpty())
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    logger.warn(TAG, "speculative tiers failed", e)
                    warnings += "Memecoins/DEX radar: ${e.message}"
                    null
                }
            }
            spec?.let { warnings += it.warnings }
            val trendingIds = context.trendingCoingeckoIds.toSet()
            val tagged = bitpandaRows.map { row -> withTags(row, spec?.memeIds.orEmpty(), spec?.memeSymbols.orEmpty(), trendingIds, spec?.dexTrendingIds.orEmpty()) }
            val extra = spec?.rows.orEmpty().map { row ->
                val risk = RiskScorer.assess(row.asset.tier, row.market, row.dex, row.safety, nowMs)
                val sent = sentiment.cached(row.asset.symbol) ?: sentiment.fromFeed(row.asset, feed) ?: previousRows[row.asset.symbol]?.sentiment
                val forecast = row.market?.let { safeForecast(row.asset, it, emptyList(), emptyList(), context, sent, funding[row.asset.symbol], null, nowMs, row.dex, risk) }
                row.copy(sentiment = sent, forecast = forecast, risk = risk)
            }
            val rows = (tagged + extra).sortedWith(ROW_ORDER)

            val snap = MarketSnapshot(
                currency = cfg.currency,
                rows = rows,
                context = context,
                universeSource = uni.source,
                universeSize = visible.size,
                fetchedAtMs = nowMs,
                warnings = warnings.distinct(),
            )
            _snapshot.value = snap
            persist(snap)
            _status.value = RefreshStatus.Idle
            snap
        } catch (e: CancellationException) {
            _status.value = RefreshStatus.Idle
            throw e
        } catch (e: Exception) {
            logger.error(TAG, "refresh failed", e)
            _status.value = RefreshStatus.Error(e.message ?: "Refresh failed")
            current ?: throw e
        }
    }

    suspend fun detail(symbol: String, force: Boolean = false): AssetDetail {
        val cfg = config
        detailCache[symbol]?.let { if (!force && clock.nowMs() - it.fetchedAtMs < cfg.chartTtlMs) return it }
        // history is fetched in the snapshot's currency; never mix it with rows quoted in another one
        val snap = _snapshot.value?.takeIf { it.currency == cfg.currency } ?: refresh()
        val row = snap.rows.firstOrNull { it.asset.symbol == symbol }
            ?: throw IllegalArgumentException("Unknown asset: $symbol")
        return detailMutex.withLock {
            detailCache[symbol]?.let { if (!force && clock.nowMs() - it.fetchedAtMs < cfg.chartTtlMs) return@withLock it }
            val warnings = ArrayList<String>()
            val vs = snap.currency.lowercase()
            var daily: List<PricePoint> = emptyList()
            var dailyVolumes: List<Double> = emptyList()
            var community: CoinCommunity? = null
            val id = row.asset.coingeckoId
            if (id != null) {
                when (val r = coinGecko.marketChart(id, vs, DETAIL_HISTORY_DAYS)) {
                    is SourceResult.Success -> {
                        daily = toDaily(r.value.prices)
                        dailyVolumes = toDaily(r.value.volumes).map { it.value }
                    }
                    is SourceResult.Failure -> warnings += "History: ${r.message}"
                }
                when (val r = coinGecko.coinCommunity(id)) {
                    is SourceResult.Success -> community = r.value
                    is SourceResult.Failure -> warnings += "Community data: ${r.message}"
                }
            } else if (row.asset.tier != Tier.DEX) {
                warnings += "No CoinGecko match for ${row.asset.symbol}; history and community data unavailable"
            }
            if (daily.isEmpty() && row.asset.tier != Tier.DEX && cex != null) {
                // CoinGecko's free tier often refuses history; Binance's public mirror is not rate-limited as hard
                binanceDaily(row)?.let { daily = it; warnings += "Daily history from Binance (CoinGecko did not answer)" }
            }
            var dexRow = row
            if (row.asset.tier == Tier.DEX) dexRow = refreshDex(row, warnings)
            val telegramHandle = dexRow.dex?.socials?.get("telegram")?.let { TELEGRAM_LINK.find(it)?.groupValues?.get(1) }
            val sent: SentimentSummary? = try {
                sentiment.forAsset(row.asset, force, listOfNotNull(telegramHandle))
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                warnings += "Sentiment: ${e.message}"
                row.sentiment
            }
            val funding = loadFunding(cfg, false, warnings)[symbol]
                ?: if (cfg.derivativesEnabled && row.asset.tier != Tier.DEX) cex?.okxFunding(symbol)?.valueOrNull else null
            val deriv = if (funding != null && cfg.derivativesEnabled && funding.source != "okx") {
                val ls = derivatives.longShortRatio(symbol).valueOrNull
                funding.copy(longShortRatio = ls ?: funding.longShortRatio)
            } else funding
            // the network calls above can take minutes; a refresh may have replaced the row meanwhile
            val now = clock.nowMs()
            val latestInList = _snapshot.value?.takeIf { it.currency == snap.currency }?.rows?.firstOrNull { it.asset.symbol == symbol } ?: row
            val latest = if (row.asset.tier == Tier.DEX) latestInList.copy(market = dexRow.market, dex = dexRow.dex, safety = dexRow.safety) else latestInList
            val hourly = hourlyPoints(latest.market, now)
            val risk = RiskScorer.assess(latest.asset.tier, latest.market, latest.dex, latest.safety, now)
            val forecast = latest.market?.let {
                safeForecast(latest.asset, it, daily.map { p -> p.value }, dailyVolumes, lastContext, sent, deriv, community?.sentimentVotesUpPct, now, latest.dex, risk)
            }
            val newRow = latest.copy(forecast = forecast ?: latest.forecast, sentiment = sent ?: latest.sentiment, risk = risk)
            updateRow(newRow, snap.currency)
            val detail = AssetDetail(
                row = newRow,
                daily = daily,
                hourly = hourly,
                community = community,
                derivatives = deriv,
                bitpandaPrices = lastUniverse?.bitpandaPrices?.get(symbol) ?: emptyMap(),
                fetchedAtMs = now,
                warnings = warnings,
            )
            if (config.currency == snap.currency) detailCache[symbol] = detail
            detail
        }
    }

    /**
     * Background sentiment collection (slow on purpose because of free-tier rate limits):
     *  1. market-wide mood (r/CryptoCurrency, crypto RSS, public Telegram channels, cloud hub),
     *  2. per-asset sentiment for the [n] largest assets, updating rows one by one so the UI fills in.
     * Forecasts of updated rows are recomputed with the new sentiment.
     */
    suspend fun enrichSentiment(n: Int = config.backgroundSentimentTopN, onProgress: (done: Int, total: Int) -> Unit = { _, _ -> }) {
        val mood = try {
            sentiment.globalMood()
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            logger.warn(TAG, "global mood failed", e)
            null
        }
        if (mood != null) {
            lastContext = lastContext?.copy(globalSentiment = mood)
            _snapshot.update { snap -> snap?.copy(context = snap.context?.copy(globalSentiment = mood) ?: lastContext) }
        }
        val snap = _snapshot.value ?: return
        val targets = snap.rows.filter { it.market != null && it.asset.symbol !in DefaultStablecoins }.take(n)
        targets.forEachIndexed { index, row ->
            val s = try {
                sentiment.forAsset(row.asset)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                logger.warn(TAG, "sentiment for ${row.asset.symbol} failed", e)
                null
            }
            if (s != null) {
                val latest = row(row.asset.symbol) ?: row
                val fc = latest.market?.let {
                    safeForecast(latest.asset, it, emptyList(), emptyList(), lastContext, s, fundingCache?.second?.get(latest.asset.symbol), null, clock.nowMs(), latest.dex, latest.risk)
                }
                updateRow(latest.copy(sentiment = s, forecast = fc ?: latest.forecast))
            }
            onProgress(index + 1, targets.size)
        }
        _snapshot.value?.let { persist(it) }
    }

    /**
     * In bundled-snapshot mode, unverified research candidates are hidden when their market cap is above the
     * smallest verified asset (the snapshot is the contiguous top of Bitpanda's list, so a bigger coin that is
     * missing from it is not on Bitpanda). Live Bitpanda lists are shown as they are.
     */
    private fun visibleAssets(uni: UniverseResult, markets: Map<String, MarketData>, warnings: MutableList<String>): List<Asset> {
        if (uni.source != ListingSource.BUNDLED_SNAPSHOT) return uni.assets
        val verifiedCaps = uni.assets.filter { it.verified }.mapNotNull { markets[it.symbol]?.marketCap?.takeIf { c -> c > 0 } }
        if (verifiedCaps.size < 20) return uni.assets
        val threshold = verifiedCaps.min() * 1.15
        val hidden = uni.assets.filter { a -> !a.verified && (markets[a.symbol]?.marketCap ?: 0.0) > threshold }
        if (hidden.isNotEmpty()) warnings += "Hid ${hidden.size} unconfirmed coins whose market cap shows they are not on Bitpanda"
        val hiddenSymbols = hidden.mapTo(HashSet()) { it.symbol }
        return uni.assets.filter { it.symbol !in hiddenSymbols }
    }

    // ------------------------------------------------------------------------------------------

    private suspend fun fetchMarkets(
        uni: UniverseResult,
        cfg: AppConfig,
        previous: MarketSnapshot?,
        warnings: MutableList<String>,
    ): Map<String, MarketData> {
        val vs = cfg.currency.lowercase()
        // Largest coins first: when CoinGecko's free tier answers 429 part-way, the coins people
        // actually look at already have their data. Order = Bitpanda snapshot rank, then last market cap.
        val lastCap = previous?.rows?.associate { it.asset.symbol to (it.market?.marketCap ?: 0.0) }.orEmpty()
        val ordered = uni.assets.sortedWith(compareBy<Asset> { it.snapshotRank ?: Int.MAX_VALUE }.thenByDescending { lastCap[it.symbol] ?: 0.0 })
        val idToSymbol = LinkedHashMap<String, String>()
        ordered.forEach { a -> a.coingeckoId?.let { idToSymbol.putIfAbsent(it, a.symbol) } }
        val result = HashMap<String, MarketData>()
        var failures = 0
        var staleAgeMs = 0L
        val chunks = idToSymbol.keys.toList().chunked(cfg.coingeckoPageSize.coerceIn(1, 250))
        chunks.forEachIndexed { i, chunk ->
            setStatus("Fetching market data (${i + 1}/${chunks.size})", 0.2f + 0.3f * i / chunks.size.coerceAtLeast(1))
            // smaller coins change rank slowly: refresh their pages less often to stay inside the free tier
            val ttl = if (i == 0) cfg.marketsTtlMs else cfg.marketsTtlMs * 4
            when (val r = coinGecko.markets(vs, page = 1, perPage = chunk.size.coerceAtLeast(1), ids = chunk, sparkline = true, ttlMs = ttl)) {
                is SourceResult.Success -> {
                    // Http hands back an old body (up to 6 h) when CoinGecko fails; say so instead of showing it as fresh
                    if (r.fromCache && r.ageMs > ttl) staleAgeMs = maxOf(staleAgeMs, r.ageMs)
                    r.value.forEach { md ->
                        val sym = md.coingeckoId?.let { idToSymbol[it] } ?: return@forEach
                        result[sym] = md.copy(symbol = sym, currency = cfg.currency)
                    }
                }
                is SourceResult.Failure -> {
                    failures++
                    warnings += "CoinGecko market data: ${r.message}"
                    logger.warn(TAG, "markets chunk ${i + 1} failed: ${r.message}")
                }
            }
        }
        if (staleAgeMs > 0) warnings += "CoinGecko is not answering; market data is ${staleAgeMs / 60_000} min old"
        if (failures > 0 && previous != null && previous.currency == cfg.currency) {
            var reused = 0
            previous.rows.forEach { row ->
                val m = row.market ?: return@forEach
                if (!result.containsKey(row.asset.symbol) && m.coingeckoId != null) {
                    result[row.asset.symbol] = m
                    reused++
                }
            }
            if (reused > 0) warnings += "Showing cached market data for $reused assets"
        }
        // Bitpanda ticker prices: fill gaps and keep as reference price.
        uni.assets.forEach { a ->
            val bp = uni.bitpandaPrices[a.symbol]?.get(cfg.currency)
            val existing = result[a.symbol]
            if (existing != null) {
                if (bp != null) result[a.symbol] = existing.copy(bitpandaPrice = bp)
            } else if (bp != null) {
                result[a.symbol] = MarketData(
                    symbol = a.symbol,
                    coingeckoId = a.coingeckoId,
                    name = a.name,
                    currency = cfg.currency,
                    price = bp,
                    bitpandaPrice = bp,
                    lastUpdatedMs = uni.fetchedAtMs,
                    source = "bitpanda",
                )
            }
        }
        if (result.isEmpty()) {
            throw IllegalStateException(warnings.lastOrNull() ?: "No market data available")
        }
        return result
    }

    private suspend fun loadContext(
        markets: Map<String, MarketData>,
        cfg: AppConfig,
        force: Boolean,
        warnings: MutableList<String>,
    ): MarketContext {
        val now = clock.nowMs()
        val btc = markets["BTC"]
        val btcTrend = btc?.let { btcTrendScore(it) }
        val cached = lastContext
        if (!force && cached != null && now - cached.fetchedAtMs < cfg.contextTtlMs) {
            return cached.copy(btcChange24hPct = btc?.change24hPct, btcChange7dPct = btc?.change7dPct, btcTrendScore = btcTrend)
                .also { lastContext = it }
        }
        val fg = if (cfg.fearGreedEnabled) {
            when (val r = fearGreed.fetch(30)) {
                is SourceResult.Success -> r.value
                is SourceResult.Failure -> { warnings += "Fear & Greed: ${r.message}"; cached?.fearGreed }
            }
        } else null
        val global = when (val r = coinGecko.global()) {
            is SourceResult.Success -> r.value
            is SourceResult.Failure -> cached?.global
        }
        val trending = when (val r = coinGecko.trending()) {
            is SourceResult.Success -> r.value.map { it.coingeckoId }
            is SourceResult.Failure -> cached?.trendingCoingeckoIds ?: emptyList()
        }
        val globalSent = sentiment.cachedGlobalMood() ?: cached?.globalSentiment
        val ctx = MarketContext(
            fearGreed = fg,
            global = global,
            btcChange24hPct = btc?.change24hPct,
            btcChange7dPct = btc?.change7dPct,
            btcTrendScore = btcTrend,
            globalSentiment = globalSent,
            trendingCoingeckoIds = trending,
            fetchedAtMs = now,
        )
        lastContext = ctx
        return ctx
    }

    private suspend fun loadFunding(cfg: AppConfig, force: Boolean, warnings: MutableList<String>): Map<String, DerivativesSnapshot> {
        if (!cfg.derivativesEnabled) return emptyMap()
        val now = clock.nowMs()
        fundingCache?.let { if (!force && now - it.first < cfg.contextTtlMs) return it.second }
        return when (val r = derivatives.fundingRates()) {
            is SourceResult.Success -> r.value.also { fundingCache = now to it }
            is SourceResult.Failure -> {
                warnings += "Derivatives: ${r.message}"
                fundingCache?.second ?: emptyMap()
            }
        }
    }

    private fun safeForecast(
        asset: Asset,
        market: MarketData,
        daily: List<Double>,
        dailyVolumes: List<Double>,
        context: MarketContext?,
        sent: SentimentSummary?,
        deriv: DerivativesSnapshot?,
        votesUp: Double?,
        nowMs: Long,
        dex: DexInfo? = null,
        risk: RiskAssessment? = null,
    ): Forecast? = try {
        forecaster.forecast(
            ForecastInput(
                asset = asset,
                market = market,
                hourlyCloses = market.sparkline7d,
                dailyCloses = daily,
                dailyVolumes = dailyVolumes,
                context = context,
                sentiment = sent,
                derivatives = deriv,
                communityVotesUpPct = votesUp,
                dex = dex,
                risk = risk,
                nowMs = nowMs,
            ),
        )
    } catch (e: Exception) {
        logger.warn(TAG, "forecast failed for ${asset.symbol}", e)
        null
    }

    /** USD -> quote currency from the Bitpanda ticker (BTC in both), else CoinGecko BTC vs the futures mark price. */
    private fun usdRate(uni: UniverseResult, cfg: AppConfig, markets: Map<String, MarketData>, funding: Map<String, DerivativesSnapshot>): Double? {
        if (cfg.currency.equals("USD", ignoreCase = true)) return 1.0
        uni.bitpandaPrices["BTC"]?.let { p ->
            val q = p[cfg.currency]
            val u = p["USD"]
            if (q != null && u != null && u > 0) return q / u
        }
        val btc = markets["BTC"]?.price
        val mark = funding["BTC"]?.markPrice
        return if (btc != null && mark != null && mark > 0) btc / mark else null
    }

    private fun withTags(row: AssetRow, memeIds: Set<String>, memeSymbols: Set<String>, trendingIds: Set<String>, dexTrendingIds: Set<String>): AssetRow {
        val id = row.asset.coingeckoId
        val add = buildList {
            if ((id != null && id in memeIds) || row.asset.symbol in memeSymbols) add("meme")
            if (id != null && id in trendingIds) add("trending")
            if (id != null && id in dexTrendingIds) add("dex_trending")
        }
        if (add.isEmpty()) return row
        return row.copy(asset = row.asset.copy(tags = (row.asset.tags + add).distinct()))
    }

    /** One year of daily closes from Binance's public mirror, scaled to the row's quote-currency price. */
    private suspend fun binanceDaily(row: AssetRow): List<PricePoint>? {
        val price = row.market?.price ?: return null
        val candles = cex?.binanceKlines(row.asset.symbol, "1d", DETAIL_HISTORY_DAYS)?.valueOrNull?.takeIf { it.size >= 30 } ?: return null
        val last = candles.last().close.takeIf { it > 0 } ?: return null
        val k = price / last
        // a ticker Binance lists for a different coin would be off by orders of magnitude
        if (k !in 0.5..2.0 && row.market.currency.equals("USD", ignoreCase = true)) return null
        return candles.map { PricePoint(it.timeMs, it.close * k) }
    }

    /** DEX token detail: a week of hourly candles from GeckoTerminal and fresh contract checks. */
    private suspend fun refreshDex(row: AssetRow, warnings: MutableList<String>): AssetRow {
        val dex = row.dex ?: return row
        var market = row.market
        val pool = dex.pairAddress
        if (geckoTerminal != null && pool != null) {
            when (val r = geckoTerminal.hourlyCandles(GeckoTerminalApi.geckoTerminalNetwork(dex.chainId), pool, 168)) {
                is SourceResult.Success -> if (r.value.size >= 24 && market != null) {
                    val k = (market.price ?: 0.0).let { p -> if (p > 0 && r.value.last().close > 0) p / r.value.last().close else lastUsdToQuote ?: 1.0 }
                    market = market.copy(sparkline7d = r.value.map { it.close * k })
                }
                is SourceResult.Failure -> warnings += "Price history: ${r.message}"
            }
        }
        var safety = row.safety
        val address = row.asset.contract
        if (config.safetyChecks && safetyApi != null && address != null) {
            safetyApi.check(dex.chainId, address).valueOrNull?.let { safety = it }
            if (address.startsWith("0x")) {
                safetyApi.simulate(dex.chainId, address).valueOrNull?.let { sim ->
                    safety = safety?.copy(
                        honeypot = sim.honeypot ?: safety?.honeypot,
                        buyTaxPct = sim.buyTaxPct ?: safety?.buyTaxPct,
                        sellTaxPct = sim.sellTaxPct ?: safety?.sellTaxPct,
                    ) ?: sim
                }
            }
            if (safety == null) warnings += "No free contract check available for ${dex.chainId}"
        }
        return row.copy(market = market, safety = safety)
    }

    /** Replaces one row, unless the snapshot has meanwhile switched to another quote currency. */
    private fun updateRow(row: AssetRow, currency: String? = null) {
        _snapshot.update { snap ->
            if (snap == null || (currency != null && snap.currency != currency)) snap
            else snap.copy(rows = snap.rows.map { if (it.asset.symbol == row.asset.symbol) row else it })
        }
    }

    private fun setStatus(step: String, progress: Float) {
        _status.value = RefreshStatus.Refreshing(step, progress.coerceIn(0f, 1f))
    }

    private fun persist(snap: MarketSnapshot) {
        val s = store ?: return
        try {
            s.write(SNAPSHOT_KEY, json.encodeToString(MarketSnapshot.serializer(), snap))
        } catch (e: Exception) {
            logger.warn(TAG, "persist failed", e)
        }
    }

    private fun loadPersisted(): MarketSnapshot? {
        val raw = store?.read(SNAPSHOT_KEY) ?: return null
        return try {
            json.decodeFromString(MarketSnapshot.serializer(), raw)
        } catch (e: Exception) {
            logger.warn(TAG, "ignoring unreadable persisted snapshot", e)
            null
        }
    }

    companion object {
        const val TAG = "MarketRepository"
        const val SNAPSHOT_KEY = "market_snapshot_v1"
        const val DETAIL_HISTORY_DAYS = 365
        private const val HOUR_MS = 3_600_000L
        private val TELEGRAM_LINK = Regex("""(?:https?://)?(?:t\.me|telegram\.me)/(?:s/)?([A-Za-z][A-Za-z0-9_]{3,31})(?:[/?#]|$)""")
        private const val DAY_MS = 86_400_000L

        private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
        private val DefaultStablecoins = dev.cryptopulse.core.analysis.forecast.DefaultForecastEngine.STABLECOINS

        val ROW_ORDER: Comparator<AssetRow> = compareBy<AssetRow> { it.market?.marketCap == null }
            .thenByDescending { it.market?.marketCap ?: 0.0 }
            .thenBy { it.asset.snapshotRank ?: Int.MAX_VALUE }
            .thenBy { it.asset.symbol }

        /** -1..1 trend score from BTC's own momentum; altcoins are strongly correlated with it. */
        fun btcTrendScore(btc: MarketData): Double {
            val m = 0.5 * (btc.change24hPct ?: 0.0) + 0.3 * (btc.change7dPct ?: 0.0) + 0.2 * (btc.change30dPct ?: 0.0)
            return tanh(m / 8.0)
        }

        /** Last point of every UTC day, oldest first. Works for hourly or daily input. */
        fun toDaily(points: List<PricePoint>): List<PricePoint> {
            if (points.isEmpty()) return emptyList()
            val byDay = LinkedHashMap<Long, PricePoint>()
            // CoinGecko's daily samples sit at 00:00 UTC and hold the previous day's close, followed by
            // one live sample; key each sample by the day it closes so the live one does not replace it
            for (p in points.sortedBy { it.timeMs }) byDay[(p.timeMs - 1).floorDiv(DAY_MS)] = p
            return byDay.values.toList()
        }

        /** Rebuilds timestamps for the 7-day hourly sparkline (last point = last update or now). */
        fun hourlyPoints(market: MarketData?, nowMs: Long): List<PricePoint> {
            val closes = market?.sparkline7d ?: return emptyList()
            if (closes.isEmpty()) return emptyList()
            val end = market.lastUpdatedMs ?: nowMs
            val n = closes.size
            return closes.mapIndexed { i, v -> PricePoint(end - (n - 1 - i) * HOUR_MS, v) }
        }
    }
}
