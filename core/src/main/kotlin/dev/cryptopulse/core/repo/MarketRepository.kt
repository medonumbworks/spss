package dev.cryptopulse.core.repo

import dev.cryptopulse.core.analysis.ForecastInput
import dev.cryptopulse.core.analysis.Forecaster
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.http.DiskStore
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.AssetDetail
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.CoinCommunity
import dev.cryptopulse.core.analysis.journal.ForecastAlert
import dev.cryptopulse.core.analysis.journal.ForecastAlerts
import dev.cryptopulse.core.analysis.journal.ForecastJournal
import dev.cryptopulse.core.analysis.journal.JournalStats
import dev.cryptopulse.core.analysis.risk.PumpDetector
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
import dev.cryptopulse.core.model.TokenSafety
import dev.cryptopulse.core.sources.CoinGeckoSource
import dev.cryptopulse.core.sources.DerivativesSource
import dev.cryptopulse.core.sources.FearGreedSource
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.sources.cex.CexDataApi
import dev.cryptopulse.core.sources.dex.Candle
import dev.cryptopulse.core.sources.dex.GeckoTerminalApi
import dev.cryptopulse.core.sources.feed.BuzzEntry
import dev.cryptopulse.core.sources.feed.telegramBuzz
import dev.cryptopulse.core.sources.polymarket.PolymarketApi
import dev.cryptopulse.core.sources.polymarket.PredictionMarket
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
    /** Accumulating state (forecast journal, alert crossings); falls back to [store]. On Android this
     *  should live in filesDir: the cache dir is wiped under storage pressure. */
    durableStore: DiskStore? = null,
    private val clock: Clock = Clock.SYSTEM,
    private val logger: Logger = Logger.STDERR,
    initialConfig: AppConfig = AppConfig(),
    /** Memecoins, trending coins and DEX tokens beyond Bitpanda; null = Bitpanda only. */
    private val speculative: SpeculativeService? = null,
    /** Binance mirror klines and OKX funding: history/funding fallbacks for the detail view. */
    private val cex: CexDataApi? = null,
    private val geckoTerminal: GeckoTerminalApi? = null,
    private val safetyApi: TokenSafetyApi? = null,
    /** Polymarket prediction markets; null = feature off. */
    private val polymarket: PolymarketApi? = null,
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

    /** USD -> quote currency (currency to rate), for prices that only exist in USD (DEX pools). */
    @Volatile
    private var lastUsdToQuote: Pair<String, Double>? = null

    @Volatile
    private var fundingCache: Pair<Long, Map<String, DerivativesSnapshot>>? = null

    /** Keyed by [identity]: after a refresh a DEX ticker can belong to another token. */
    private val detailCache = ConcurrentHashMap<String, AssetDetail>()

    /**
     * Contract checks made when a DEX token was opened, by [identity]. A refresh that was already running
     * would otherwise overwrite them with its older list rows; they are added to the next rows for
     * [SpeculativeService.KEEP_SAFETY_MS].
     */
    private val openedChecks = ConcurrentHashMap<String, TokenSafety>()

    /** Logs every issued forecast and scores it once its horizon has passed. */
    private val journal = ForecastJournal(durableStore ?: store, clock)

    /** Crossing detector for the "chance of rising above X%" notifications. */
    private val alerts = ForecastAlerts(durableStore ?: store, clock)

    /** Polymarket markets of the last refresh, for forecasts recomputed outside a refresh. */
    @Volatile
    private var lastPolymarket: List<PredictionMarket> = emptyList()

    init {
        // keep the persisted market context so forecasts made before the first network refresh use it
        loadPersisted()?.let { snap ->
            _snapshot.value = snap
            lastContext = snap.context
            lastUsdToQuote = snap.usdToQuote?.let { snap.currency to it }
        }
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

            val predictionMarkets = if (cfg.polymarketEnabled && polymarket != null) {
                when (val r = polymarket.cryptoMarkets()) {
                    is SourceResult.Success -> r.value
                    is SourceResult.Failure -> { warnings += r.message; emptyList() }
                }
            } else emptyList()
            lastPolymarket = predictionMarkets

            setStatus("Computing forecasts", 0.8f)
            val nowMs = clock.nowMs()
            val visible = visibleAssets(uni, markets, warnings)
            val previousRows = current?.rows?.associateBy { it.asset.symbol }.orEmpty()
            val bitpandaRows = visible.map { asset ->
                val market = markets[asset.symbol]
                val sent = sentiment.cached(asset.symbol) ?: sentiment.fromFeed(asset, feed) ?: previousRows[asset.symbol]?.sentiment
                val fa = feed?.assets?.get(asset.symbol)
                val forecast = market?.let {
                    safeForecast(asset, it, emptyList(), emptyList(), context, sent, funding[asset.symbol], null, nowMs, polymarketUp24 = upDown(predictionMarkets, asset, nowMs))
                }
                AssetRow(asset, market, forecast, sent, pump = PumpDetector.assess(market, null, fa?.mentions, fa?.spike, nowMs))
            }

            val usdToQuote = usdRate(uni, cfg, markets, funding)
            lastUsdToQuote = usdToQuote?.let { cfg.currency to it }
            val spec = speculative?.let {
                setStatus("Finding memecoins and DEX tokens", 0.85f)
                try {
                    it.load(cfg, bitpandaRows, context.trendingCoingeckoIds, usdToQuote, current?.rows.orEmpty())
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    logger.warn(TAG, "speculative tiers failed", e)
                    warnings += "Coins beyond Bitpanda: ${e.message}"
                    null
                }
            }
            spec?.let { warnings += it.warnings }
            val trendingIds = context.trendingCoingeckoIds.toSet()
            val tagged = bitpandaRows.map { row -> withTags(row, spec?.memeIds.orEmpty(), spec?.memeSymbols.orEmpty(), trendingIds, spec?.dexTrendingIds.orEmpty()) }
            val extra = spec?.rows.orEmpty()
                // rows kept from an earlier snapshot must be quoted like this one (DEX pools may stay in USD)
                .filter { r -> r.market?.currency.let { it == null || it == cfg.currency || (r.asset.tier == Tier.DEX && it == "USD") } }
                .map { raw ->
                    val opened = openedChecks[identity(raw.asset)]?.takeIf { nowMs - it.checkedAtMs < SpeculativeService.KEEP_SAFETY_MS }
                    val row = if (opened == null) raw else raw.copy(safety = raw.safety?.combinedWith(opened) ?: opened)
                    val risk = RiskScorer.assess(row.asset.tier, row.market, row.dex, row.safety, nowMs)
                    val sent = sentiment.cached(row.asset.symbol) ?: sentiment.fromFeed(row.asset, feed) ?: previousRows[row.asset.symbol]?.sentiment
                    val forecast = row.market?.let { safeForecast(row.asset, it, emptyList(), emptyList(), context, sent, fundingFor(row.asset, funding), null, nowMs, row.dex, risk) }
                    val fa = feed?.assets?.get(row.asset.symbol)
                    row.copy(sentiment = sent, forecast = forecast, risk = risk, pump = PumpDetector.assess(row.market, row.dex, fa?.mentions, fa?.spike, nowMs))
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
                usdToQuote = usdToQuote,
            )
            openedChecks.entries.removeIf { nowMs - it.value.checkedAtMs >= SpeculativeService.KEEP_SAFETY_MS }
            _snapshot.value = snap
            persist(snap)
            try {
                // score the forecasts that came due at today's prices, then log the fresh calls
                journal.evaluate(snap.rows)
                journal.record(snap.rows)
            } catch (e: Exception) {
                logger.warn(TAG, "forecast journal failed", e)
            }
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
        // history is fetched in the snapshot's currency; never mix it with rows quoted in another one
        val snap = _snapshot.value?.takeIf { it.currency == cfg.currency } ?: refresh()
        val listed = snap.rows.firstOrNull { it.asset.symbol == symbol }
            ?: throw IllegalArgumentException("Unknown asset: $symbol")
        detailCache[identity(listed.asset)]?.let { if (!force && clock.nowMs() - it.fetchedAtMs < cfg.chartTtlMs) return it }
        return detailMutex.withLock {
            // waiting for another detail can take a while: start from the row as the list has it now
            val row = _snapshot.value?.takeIf { it.currency == snap.currency }?.rows?.firstOrNull { it.asset.symbol == symbol } ?: listed
            val key = identity(row.asset)
            detailCache[key]?.let { if (!force && clock.nowMs() - it.fetchedAtMs < cfg.chartTtlMs) return@withLock it }
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
            val dexDetail = if (row.asset.tier == Tier.DEX) loadDex(row, warnings) else null
            val polymarkets = if (cfg.polymarketEnabled && polymarket != null) polymarket.cryptoMarkets().valueOrNull.orEmpty() else emptyList()
            val relatedMarkets = if (polymarkets.isEmpty()) emptyList() else PolymarketApi.forAsset(polymarkets, row.asset.symbol, row.asset.name)
            val sent: SentimentSummary? = try {
                sentiment.forAsset(row.asset, force, listOfNotNull(telegramHandle(row)))
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                warnings += "Sentiment: ${e.message}"
                row.sentiment
            }
            // a DEX ticker is not the listed coin of the same name: no CEX perpetual data for it
            val funding = if (row.asset.tier == Tier.DEX) null else {
                loadFunding(cfg, false, warnings)[symbol] ?: if (cfg.derivativesEnabled) cex?.okxFunding(symbol)?.valueOrNull else null
            }
            val deriv = if (funding != null && cfg.derivativesEnabled && funding.source != "okx") {
                val ls = derivatives.longShortRatio(symbol).valueOrNull
                funding.copy(longShortRatio = ls ?: funding.longShortRatio)
            } else funding
            // the network calls above can take minutes; a refresh may have replaced the row meanwhile, or given
            // the ticker to another token: only ever merge into the same token
            val now = clock.nowMs()
            val inList = _snapshot.value?.takeIf { it.currency == snap.currency }?.rows
                ?.firstOrNull { it.asset.symbol == symbol }?.takeIf { identity(it.asset) == key }
            // the list's newer market and pool data stay; the detail adds the candle chart and contract checks
            val latest = (inList ?: row).let { r -> dexDetail?.let { withDex(r, it, warnings) } ?: r }
            if (latest.asset.tier == Tier.DEX) latest.safety?.let { openedChecks[key] = it }
            val hourly = hourlyPoints(latest.market, now)
            val risk = RiskScorer.assess(latest.asset.tier, latest.market, latest.dex, latest.safety, now)
            val forecast = latest.market?.let {
                safeForecast(latest.asset, it, daily.map { p -> p.value }, dailyVolumes, lastContext, sent, deriv, community?.sentimentVotesUpPct, now, latest.dex, risk,
                    polymarketUp24 = upDown(polymarkets, latest.asset, now))
            }
            val newRow = latest.copy(forecast = forecast ?: latest.forecast, sentiment = sent ?: latest.sentiment, risk = risk)
            if (inList != null) {
                updateRow(newRow, snap.currency)
                // contract findings must survive an app restart before the next refresh
                if (newRow.asset.tier == Tier.DEX) _snapshot.value?.let { persist(it) }
            }
            val detail = AssetDetail(
                row = newRow,
                daily = daily,
                hourly = hourly,
                community = community,
                derivatives = deriv,
                bitpandaPrices = lastUniverse?.bitpandaPrices?.get(symbol) ?: emptyMap(),
                fetchedAtMs = now,
                warnings = warnings,
                predictionMarkets = relatedMarkets,
            )
            if (inList != null && config.currency == snap.currency) detailCache[key] = detail
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
                sentiment.forAsset(row.asset, extraChannels = listOfNotNull(telegramHandle(row)))
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                logger.warn(TAG, "sentiment for ${row.asset.symbol} failed", e)
                null
            }
            // skipped when the list has meanwhile given the ticker to another token
            val latest = row(row.asset.symbol)?.takeIf { identity(it.asset) == identity(row.asset) }
            if (s != null && latest != null) {
                val fc = latest.market?.let {
                    safeForecast(latest.asset, it, emptyList(), emptyList(), lastContext, s, fundingFor(latest.asset, fundingCache?.second), null, clock.nowMs(), latest.dex, latest.risk,
                        polymarketUp24 = upDown(lastPolymarket, latest.asset, clock.nowMs()))
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
            return cached.copy(btcChange24hPct = btc?.change24hPct, btcChange7dPct = btc?.change7dPct, btcTrendScore = btcTrend,
                btcHourly = btc?.sparkline7d.orEmpty())
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
            btcHourly = btc?.sparkline7d.orEmpty(),
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
        polymarketUp24: Double? = null,
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
                polymarketUp24 = polymarketUp24,
                risk = risk,
                nowMs = nowMs,
            ),
        )
    } catch (e: Exception) {
        logger.warn(TAG, "forecast failed for ${asset.symbol}", e)
        null
    }

    /**
     * USD -> quote currency from the Bitpanda ticker (BTC in both), else a USD stablecoin's quote price,
     * else CoinGecko BTC vs the futures mark price.
     */
    private fun usdRate(uni: UniverseResult, cfg: AppConfig, markets: Map<String, MarketData>, funding: Map<String, DerivativesSnapshot>): Double? {
        if (cfg.currency.equals("USD", ignoreCase = true)) return 1.0
        uni.bitpandaPrices["BTC"]?.let { p ->
            val q = p[cfg.currency]
            val u = p["USD"]
            if (q != null && u != null && u > 0) return q / u
        }
        // the Public API and the bundled snapshot have no Bitpanda prices, and futures can be off or blocked
        USD_STABLECOINS.firstNotNullOfOrNull { markets[it]?.price?.takeIf { p -> p > 0 } }?.let { return it }
        val btc = markets["BTC"]?.price
        val mark = funding["BTC"]?.markPrice
        return if (btc != null && mark != null && mark > 0) btc / mark else null
    }

    /** Realized accuracy of every forecast the engine issued, per horizon (the forecast journal). */
    fun journalStats(): JournalStats = journal.stats()

    /** Bitpanda coins whose 24 h chance of rising crossed [thresholdPct] since the last check. */
    fun alertCrossings(thresholdPct: Int = config.alertThresholdPct): List<ForecastAlert> =
        alerts.crossings(_snapshot.value?.rows.orEmpty(), thresholdPct)

    /**
     * Most-talked-about coins in the hub's Telegram window, spikes vs the 7-day baseline first. Only
     * coins in the current list: the hub also counts memecoins and new cashtags, which the app does
     * not show.
     */
    suspend fun telegramBuzz(limit: Int = 50, force: Boolean = false): List<BuzzEntry> {
        val feed = (if (config.sentimentFeedUrl.isNotBlank()) sentiment.feed(force) else null) ?: return emptyList()
        val listed = _snapshot.value?.rows?.mapTo(HashSet()) { it.asset.symbol }
        return feed.telegramBuzz(Int.MAX_VALUE).filter { listed == null || it.symbol in listed }.take(limit)
    }

    /** Polymarket's daily up-or-down probability for one coin, or null without such a market. */
    private fun upDown(markets: List<PredictionMarket>, asset: Asset, nowMs: Long): Double? =
        if (markets.isEmpty()) null else PolymarketApi.upDownProbability(markets, asset.symbol, asset.name, nowMs)

    /** USD -> [currency] as of the last refresh; null when unknown. */
    private fun usdTo(currency: String): Double? =
        if (currency.equals("USD", ignoreCase = true)) 1.0 else lastUsdToQuote?.takeIf { it.first.equals(currency, ignoreCase = true) }?.second

    /** CEX perpetual data for a ticker; never for DEX tokens, whose tickers are re-used by unrelated coins. */
    private fun fundingFor(asset: Asset, funding: Map<String, DerivativesSnapshot>?): DerivativesSnapshot? =
        if (asset.tier == Tier.DEX) null else funding?.get(asset.symbol)

    /**
     * Which token a row is. Beyond Bitpanda a ticker is not unique: DEX tokens are told apart by chain and
     * contract, memecoins and trending coins by their CoinGecko id.
     */
    private fun identity(a: Asset): String = a.tokenKey

    /** The project's own public Telegram channel from its DEX profile. */
    private fun telegramHandle(row: AssetRow): String? = row.dex?.socials?.get("telegram")?.let { TELEGRAM_LINK.find(it)?.groupValues?.get(1) }

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
        // a ticker Binance lists for a different coin is off by orders of magnitude, the right coin only by the
        // USD rate (when that is unknown: every supported currency is within 0.5..2 of USD)
        if (k / (usdTo(row.market.currency) ?: 1.0) !in 0.5..2.0) return null
        return candles.map { PricePoint(it.timeMs, it.close * k) }
    }

    /** What the detail view adds for a DEX token: hourly USD closes and fresh contract checks. */
    private class DexDetail(val closesUsd: List<Double>, val closesAtMs: Long, val goplus: TokenSafety?, val simulation: TokenSafety?)

    /** DEX token detail: a week of hourly candles from GeckoTerminal and fresh contract checks. */
    private suspend fun loadDex(row: AssetRow, warnings: MutableList<String>): DexDetail? {
        val dex = row.dex ?: return null
        var closes = emptyList<Double>()
        var closesAt = 0L
        val pool = dex.pairAddress
        if (geckoTerminal != null && pool != null) {
            when (val r = geckoTerminal.hourlyCandles(GeckoTerminalApi.geckoTerminalNetwork(dex.chainId), pool, 168, token = row.asset.contract)) {
                is SourceResult.Success -> {
                    closesAt = clock.nowMs() - r.ageMs
                    closes = hourlyGrid(r.value, closesAt)
                }
                is SourceResult.Failure -> warnings += "Price history: ${r.message}"
            }
        }
        var goplus: TokenSafety? = null
        var simulation: TokenSafety? = null
        val address = row.asset.contract
        if (config.safetyChecks && safetyApi != null && address != null) {
            goplus = safetyApi.check(dex.chainId, address).valueOrNull
            if (address.startsWith("0x")) {
                // a run that could not simulate (no verdict, no taxes) checked nothing
                simulation = safetyApi.simulate(dex.chainId, address).valueOrNull?.takeIf { it.honeypot != null || it.buyTaxPct != null || it.sellTaxPct != null }
            }
            if (goplus == null && simulation == null && row.safety == null) warnings += "No free contract check available for ${dex.chainId}"
        }
        return DexDetail(closes, closesAt, goplus, simulation)
    }

    /** [row] (the list's newest version of the token) with the detail's candle chart and contract checks. */
    private fun withDex(row: AssetRow, d: DexDetail, warnings: MutableList<String>): AssetRow {
        var market = row.market
        val lastUsd = d.closesUsd.lastOrNull()
        val fxKnown = market?.let { usdTo(it.currency) }
        // candles far from the pool's own price are another token's (or broken): no chart rather than a wrong one
        val implausible = fxKnown != null && lastUsd != null && market?.price?.let { p -> p > 0 && (lastUsd * fxKnown / p) !in 0.5..2.0 } == true
        if (implausible) warnings += "Price history did not match the pool price; chart left out"
        if (market != null && !implausible && d.closesUsd.size >= 24 && lastUsd != null && lastUsd > 0) {
            // USD candles are converted at the exchange rate: scaling them to the row's price would shift the
            // whole chart by every move since the last refresh
            val fx = fxKnown ?: market.price?.takeIf { it > 0 }?.let { it / lastUsd } ?: 1.0
            val newer = d.closesAtMs > (market.lastUpdatedMs ?: 0L)
            market = market.copy(
                sparkline7d = d.closesUsd.map { it * fx },
                price = if (newer) lastUsd * fx else market.price,
                lastUpdatedMs = if (newer) d.closesAtMs else market.lastUpdatedMs,
            )
        }
        // the worse answer wins: a clean honeypot.is simulation never clears a honeypot or a tax GoPlus found
        val checked = d.goplus ?: row.safety
        val safety = d.simulation?.let { sim -> checked?.combinedWith(sim) ?: sim } ?: checked
        return row.copy(market = market, safety = safety)
    }

    /** Replaces the row of the same token, unless the snapshot has meanwhile switched to another quote currency. */
    private fun updateRow(row: AssetRow, currency: String? = null) {
        val id = identity(row.asset)
        _snapshot.update { snap ->
            if (snap == null || (currency != null && snap.currency != currency)) snap
            else snap.copy(rows = snap.rows.map { if (identity(it.asset) == id) row else it })
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
            val snap = json.decodeFromString(MarketSnapshot.serializer(), raw)
            // a tier switched off since (memecoins, DEX radar) must not reappear on a cold start
            val off = buildSet {
                if (!config.showMemecoins) add(Tier.MEME)
                if (!config.showTrending) add(Tier.TRENDING)
                if (!config.showDexRadar) add(Tier.DEX)
            }
            if (off.isEmpty()) snap else snap.copy(rows = snap.rows.filter { it.asset.tier !in off })
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
        private val USD_STABLECOINS = listOf("USDT", "USDC")

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

        /**
         * Closes on an hourly grid over the last [hours] hours up to [atMs], oldest first. GeckoTerminal leaves
         * out hours without trades: those keep the last close. A younger pool starts at its first candle.
         */
        fun hourlyGrid(candles: List<Candle>, atMs: Long, hours: Int = 168): List<Double> {
            if (candles.isEmpty()) return emptyList()
            val sorted = candles.sortedBy { it.timeMs }
            // a candle ahead of the device clock still ends the grid
            val end = maxOf(atMs, sorted.last().timeMs).floorDiv(HOUR_MS) * HOUR_MS
            val out = ArrayList<Double>(hours)
            var i = 0
            var last: Double? = null
            for (h in hours - 1 downTo 0) {
                val start = end - h * HOUR_MS
                while (i < sorted.size && sorted[i].timeMs < start + HOUR_MS) last = sorted[i++].close
                last?.let { out += it }
            }
            return out
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
