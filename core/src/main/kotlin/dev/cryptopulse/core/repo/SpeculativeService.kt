package dev.cryptopulse.core.repo

import dev.cryptopulse.core.analysis.risk.RiskScorer
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.DexInfo
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.model.TokenSafety
import dev.cryptopulse.core.sources.CoinGeckoSource
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.sources.dex.DexScreenerApi
import dev.cryptopulse.core.sources.dex.GeckoTerminalApi
import dev.cryptopulse.core.sources.dex.PumpFunApi
import dev.cryptopulse.core.sources.paprika.CoinpaprikaApi
import dev.cryptopulse.core.sources.safety.TokenSafetyApi
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.Logger
import dev.cryptopulse.core.util.Text

/**
 * Everything beyond Bitpanda's list, for information only:
 *  - MEME: the largest memecoins by market cap (CoinGecko "Meme" category, or Coinpaprika's "meme-coin"
 *    tag when CoinGecko refuses) that Bitpanda lacks,
 *  - TRENDING: CoinGecko's trending coins that Bitpanda lacks,
 *  - DEX: tokens that trade only on decentralised exchanges, found through GeckoTerminal trending
 *    and new pools, DexScreener boosts/profiles/community takeovers and pump.fun.
 *
 * Bitpanda always wins a ticker: a DEX token named like a listed coin is left out (most such
 * tokens are copycats). Dust (tiny pools, no volume) is filtered; contract checks run for the most
 * active DEX tokens.
 */
class SpeculativeService(
    private val coinGecko: CoinGeckoSource,
    private val dexScreener: DexScreenerApi,
    private val geckoTerminal: GeckoTerminalApi,
    private val pumpFun: PumpFunApi,
    private val safety: TokenSafetyApi,
    private val clock: Clock = Clock.SYSTEM,
    private val logger: Logger = Logger.STDERR,
    private val paprika: CoinpaprikaApi? = null,
) {
    data class Result(
        /** New rows (no forecast yet), memecoins and trending coins first, then DEX tokens. */
        val rows: List<AssetRow>,
        /** CoinGecko ids in the Meme category, including coins that Bitpanda lists. */
        val memeIds: Set<String>,
        /** CoinGecko ids whose pools trend on DEXes right now. */
        val dexTrendingIds: Set<String>,
        val warnings: List<String>,
        /** Bitpanda tickers that Coinpaprika tags as memecoins (only filled when CoinGecko refused). */
        val memeSymbols: Set<String> = emptySet(),
    )

    /**
     * @param taken rows already in the list (Bitpanda); their tickers and CoinGecko ids are reserved.
     * @param usdToQuote multiply a USD price by this to get the quote currency (null = unknown).
     */
    suspend fun load(cfg: AppConfig, taken: List<AssetRow>, trendingIds: List<String>, usdToQuote: Double?, previous: List<AssetRow>): Result {
        val warnings = ArrayList<String>()
        val symbols = taken.mapTo(HashSet()) { it.asset.symbol }
        val ids = taken.mapNotNullTo(HashSet()) { it.asset.coingeckoId }
        val rows = ArrayList<AssetRow>()
        val memeIds = HashSet<String>()
        val memeSymbols = HashSet<String>()

        if (cfg.showMemecoins) {
            val memes = memecoins(cfg, warnings)
            if (memes == null) {
                val fallback = paprikaMemecoins(cfg, taken)
                if (fallback != null) {
                    memeSymbols += fallback.second
                    for (t in fallback.first) {
                        if (t.symbol in symbols) continue
                        rows += AssetRow(Asset(t.symbol, t.name, null, tier = Tier.MEME, tags = listOf("meme")), t.market)
                        symbols += t.symbol
                    }
                } else {
                    // nobody answered: keep the last known memecoin rows rather than dropping the tier
                    previous.filter { it.asset.tier == Tier.MEME && it.asset.symbol !in symbols }.forEach { rows += it.copy(forecast = null); symbols += it.asset.symbol }
                }
            } else {
                memes.forEach { m -> m.coingeckoId?.let { memeIds += it } }
                for (m in memes) {
                    val id = m.coingeckoId ?: continue
                    val sym = Text.normalizeSymbol(m.symbol)
                    if (id in ids || sym.isEmpty() || sym in symbols || sym in QUOTE_SYMBOLS) continue
                    rows += AssetRow(Asset(sym, m.name ?: sym, id, tier = Tier.MEME, tags = listOf("meme")), m.copy(symbol = sym, currency = cfg.currency))
                    symbols += sym
                    ids += id
                }
            }
        }

        if (cfg.showTrending) {
            val wanted = trendingIds.filter { it !in ids }.take(15)
            if (wanted.isNotEmpty()) {
                when (val r = coinGecko.markets(cfg.currency.lowercase(), page = 1, perPage = wanted.size, ids = wanted, sparkline = true, ttlMs = 10 * 60_000L)) {
                    is SourceResult.Success -> for (m in r.value) {
                        val id = m.coingeckoId ?: continue
                        val sym = Text.normalizeSymbol(m.symbol)
                        if (id in ids || sym.isEmpty() || sym in symbols || sym in QUOTE_SYMBOLS) continue
                        val tags = if (id in memeIds) listOf("trending", "meme") else listOf("trending")
                        rows += AssetRow(Asset(sym, m.name ?: sym, id, tier = Tier.TRENDING, tags = tags), m.copy(symbol = sym, currency = cfg.currency))
                        symbols += sym
                        ids += id
                    }
                    is SourceResult.Failure -> warnings += "Trending coins: ${r.message}"
                }
            }
        }

        val dexTrendingIds = HashSet<String>()
        if (cfg.showDexRadar) {
            rows += dexRadar(cfg, symbols, ids, usdToQuote, dexTrendingIds, warnings)
        }
        return Result(rows, memeIds, dexTrendingIds, warnings, memeSymbols)
    }

    /**
     * Coinpaprika's memecoins (one tickers call for every coin, one cached tag call): the
     * [AppConfig.memecoinCount] best ranked that Bitpanda lacks, and the Bitpanda tickers that are
     * memecoins (same ticker and name). Null when Coinpaprika is unavailable too. No sparklines, so
     * these forecasts lean on the 1 h / 24 h / 7 d / 30 d changes.
     */
    private suspend fun paprikaMemecoins(cfg: AppConfig, taken: List<AssetRow>): Pair<List<PaprikaRow>, Set<String>>? {
        val api = paprika ?: return null
        val tagged = (api.tagCoins(CoinpaprikaApi.MEME_TAG) as? SourceResult.Success)?.value?.takeIf { it.isNotEmpty() } ?: return null
        val tickers = (api.tickers(cfg.currency) as? SourceResult.Success)?.value ?: return null
        val listed = taken.associate { it.asset.symbol to it.asset.name.lowercase() }
        val memes = tickers.filter { it.id in tagged && it.market.price != null }
        val bitpandaMemes = memes.filter { listed[Text.normalizeSymbol(it.symbol)] == it.name.lowercase() }.mapTo(HashSet()) { Text.normalizeSymbol(it.symbol) }
        val rows = memes.asSequence()
            .map { t -> PaprikaRow(Text.normalizeSymbol(t.symbol), t.name, t.rank ?: Int.MAX_VALUE, t.market) }
            .filter { it.symbol.isNotEmpty() && it.symbol !in listed && it.symbol !in QUOTE_SYMBOLS }
            .sortedBy { it.rank }
            .distinctBy { it.symbol }
            .take(cfg.memecoinCount.coerceIn(1, 1000))
            .map { it.copy(market = it.market.copy(symbol = it.symbol)) }
            .toList()
        logger.debug(TAG, "memecoins from Coinpaprika: ${rows.size} rows, ${bitpandaMemes.size} Bitpanda coins tagged")
        return rows to bitpandaMemes
    }

    private data class PaprikaRow(val symbol: String, val name: String, val rank: Int, val market: MarketData)

    private suspend fun memecoins(cfg: AppConfig, warnings: MutableList<String>): List<MarketData>? {
        val out = ArrayList<MarketData>()
        val pages = ((cfg.memecoinCount.coerceIn(1, 1000) + 249) / 250)
        for (page in 1..pages) {
            val perPage = minOf(250, cfg.memecoinCount - (page - 1) * 250)
            when (val r = coinGecko.markets(cfg.currency.lowercase(), page = page, perPage = perPage, ids = null, sparkline = true, ttlMs = 10 * 60_000L, category = MEME_CATEGORY)) {
                is SourceResult.Success -> out += r.value
                is SourceResult.Failure -> {
                    warnings += "Memecoins: ${r.message}"
                    return out.takeIf { it.isNotEmpty() }
                }
            }
        }
        return out
    }

    private class Candidate(val chainId: String, val address: String) {
        val foundBy = LinkedHashSet<String>()
        var info: DexInfo? = null
        var symbol: String? = null
        var name: String? = null
        var imageUrl: String? = null
        var coingeckoId: String? = null
        var graduated: Boolean? = null
    }

    private suspend fun dexRadar(
        cfg: AppConfig,
        symbols: MutableSet<String>,
        ids: MutableSet<String>,
        usdToQuote: Double?,
        dexTrendingIds: MutableSet<String>,
        warnings: MutableList<String>,
    ): List<AssetRow> {
        val candidates = LinkedHashMap<String, Candidate>()
        fun candidate(chain: String, address: String): Candidate = candidates.getOrPut(key(chain, address)) { Candidate(chain, address) }
        var reachable = 0

        val poolCalls = buildList {
            add(suspend { geckoTerminal.trendingPools(1) })
            add(suspend { geckoTerminal.trendingPools(2) })
            if (cfg.dexIncludeNewPools) add(suspend { geckoTerminal.newPools(1) })
        }
        for (call in poolCalls) {
            when (val r = call()) {
                is SourceResult.Success -> {
                    reachable++
                    for (p in r.value) {
                        val c = candidate(p.info.chainId, p.info.tokenAddress)
                        c.foundBy += p.info.foundBy
                        if (c.info == null || (p.info.liquidityUsd ?: 0.0) > (c.info?.liquidityUsd ?: 0.0)) c.info = p.info
                        c.symbol = c.symbol ?: p.baseSymbol
                        c.name = c.name ?: p.baseName
                        c.imageUrl = c.imageUrl ?: p.imageUrl
                        c.coingeckoId = c.coingeckoId ?: p.baseCoingeckoId
                    }
                }
                is SourceResult.Failure -> warnings += "DEX pools: ${r.message}"
            }
        }
        when (val r = dexScreener.promoted()) {
            is SourceResult.Success -> {
                reachable++
                r.value.forEach { candidate(it.chainId, it.tokenAddress).foundBy += it.kind }
            }
            is SourceResult.Failure -> warnings += "DexScreener: ${r.message}"
        }
        if (cfg.dexIncludePumpFun) {
            for (call in listOf(suspend { pumpFun.topByMarketCap(50) }, suspend { pumpFun.currentlyLive(30) })) {
                when (val r = call()) {
                    is SourceResult.Success -> {
                        reachable++
                        r.value.forEach { p ->
                            val c = candidate("solana", p.mint)
                            c.foundBy += "pumpfun"
                            c.graduated = p.graduated
                            c.symbol = c.symbol ?: p.symbol
                            c.name = c.name ?: p.name
                        }
                    }
                    is SourceResult.Failure -> logger.debug(TAG, "pump.fun: ${r.message}")
                }
            }
        }
        if (reachable == 0) return emptyList()

        // one uniform, deep-pool view of every candidate (DexScreener answers 30 tokens per call)
        candidates.values.filter { !isQuote(it) }.groupBy { it.chainId }.forEach { (chain, list) ->
            when (val r = dexScreener.tokens(chain, list.take(MAX_LOOKUPS_PER_CHAIN).map { it.address })) {
                is SourceResult.Success -> r.value.forEach { pair ->
                    val c = candidates[key(chain, pair.baseAddress)] ?: return@forEach
                    val previous = c.info
                    c.info = pair.info.copy(
                        foundBy = emptyList(),
                        // GeckoTerminal counts unique wallets; DexScreener only trades
                        txnsH6 = pair.info.txnsH6.copy(buyers = previous?.txnsH6?.buyers, sellers = previous?.txnsH6?.sellers),
                        txnsH24 = pair.info.txnsH24.copy(buyers = previous?.txnsH24?.buyers, sellers = previous?.txnsH24?.sellers),
                    )
                    c.symbol = pair.baseSymbol
                    c.name = pair.baseName
                    c.imageUrl = pair.imageUrl ?: c.imageUrl
                }
                is SourceResult.Failure -> warnings += "DexScreener tokens ($chain): ${r.message}"
            }
        }

        var copycats = 0
        var dust = 0
        val kept = candidates.values.mapNotNull { c ->
            val info = c.info ?: return@mapNotNull null
            if (isQuote(c)) return@mapNotNull null
            val cg = c.coingeckoId
            if (cg != null && cg in ids) { dexTrendingIds += cg; return@mapNotNull null }
            val sym = Text.normalizeSymbol(c.symbol ?: return@mapNotNull null)
            if (sym.isEmpty()) return@mapNotNull null
            if (sym in symbols) { copycats++; return@mapNotNull null }
            if ((info.liquidityUsd ?: 0.0) < cfg.dexMinLiquidityUsd || (info.volumeUsd.h24 ?: 0.0) < cfg.dexMinVolume24hUsd) { dust++; return@mapNotNull null }
            Triple(c, sym, info.copy(foundBy = c.foundBy.toList(), pumpFunGraduated = c.graduated))
        }
            // two DEX tokens with one ticker: keep the deeper pool
            .groupBy { it.second }.values.map { same -> same.maxBy { it.third.liquidityUsd ?: 0.0 } }
            .sortedByDescending { it.third.volumeUsd.h24 ?: 0.0 }
            .take(cfg.dexRadarMax.coerceIn(0, 500))
        if (copycats > 0) logger.debug(TAG, "DEX radar: left out $copycats tokens named like listed coins")
        logger.debug(TAG, "DEX radar: ${kept.size} kept, $dust below the liquidity/volume floor, ${candidates.size} candidates")

        val now = clock.nowMs()
        // checks go first to tokens that can get a forecast at all; the rest are blocked anyway
        val toCheck = kept.sortedBy { if (couldForecast(it.third, now)) 0 else 1 }.take(cfg.dexSafetyTopN).map { it.first }
        val safetyByKey = if (cfg.safetyChecks) safetyFor(toCheck, warnings) else emptyMap()
        return kept.map { (c, sym, info) ->
            symbols += sym
            val fx = usdToQuote
            val quote = if (fx == null) "USD" else cfg.currency
            val k = fx ?: 1.0
            val market = MarketData(
                symbol = sym,
                name = c.name,
                currency = quote,
                price = info.priceUsd?.times(k),
                marketCap = (info.marketCapUsd ?: info.fdvUsd)?.times(k),
                volume24h = info.volumeUsd.h24?.times(k),
                change1hPct = info.priceChangePct.h1,
                change24hPct = info.priceChangePct.h24,
                imageUrl = c.imageUrl,
                lastUpdatedMs = now,
                source = "dex",
            )
            val tags = buildList {
                if ("gt_trending" in info.foundBy) add("dex_trending")
                if ("gt_new" in info.foundBy) add("new")
                if ("ds_boost" in info.foundBy || info.boostsActive > 0) add("boosted")
                if ("pumpfun" in info.foundBy) add("pumpfun")
            }
            AssetRow(
                asset = Asset(sym, c.name ?: sym, c.coingeckoId, tier = Tier.DEX, tags = tags, chainId = c.chainId, contract = c.address),
                market = market,
                dex = info,
                safety = safetyByKey[key(c.chainId, c.address)],
            )
        }
    }

    /**
     * Contract checks, most active tokens first. Cached results (6 h) are free; at most
     * [NEW_CHECKS_PER_REFRESH] new checks per refresh keep GoPlus' anonymous quota intact, so coverage
     * grows over a few refreshes. Unchecked DEX tokens are shown without a forecast.
     */
    private suspend fun safetyFor(list: List<Candidate>, warnings: MutableList<String>): Map<String, TokenSafety> {
        val out = HashMap<String, TokenSafety>()
        var newChecks = 0
        for (c in list) {
            if (!safety.supports(c.chainId)) continue
            if (newChecks >= NEW_CHECKS_PER_REFRESH && !safety.isCached(c.chainId, c.address)) continue
            when (val r = safety.check(c.chainId, c.address)) {
                is SourceResult.Success -> {
                    if (!r.fromCache) newChecks++
                    r.value?.let { out[key(c.chainId, c.address)] = it }
                }
                is SourceResult.Failure -> {
                    if (r.status == dev.cryptopulse.core.model.SourceStatus.RATE_LIMITED) { warnings += "Safety checks: ${r.message}"; break }
                    newChecks++
                }
            }
        }
        return out
    }

    /** Not blocked by liquidity, pool age or the pump.fun bonding curve (see [RiskScorer.assess]). */
    private fun couldForecast(info: DexInfo, nowMs: Long): Boolean {
        val created = info.pairCreatedAtMs ?: return false
        return (info.liquidityUsd ?: 0.0) >= RiskScorer.MIN_LIQUIDITY_USD && nowMs - created >= 24 * 3_600_000L && info.pumpFunGraduated != false
    }

    private fun isQuote(c: Candidate): Boolean = c.symbol?.let { Text.normalizeSymbol(it) in QUOTE_SYMBOLS } == true

    companion object {
        private const val TAG = "Speculative"
        const val MEME_CATEGORY = "meme-token"
        private const val MAX_LOOKUPS_PER_CHAIN = 120
        const val NEW_CHECKS_PER_REFRESH = 12

        /** Pool quote assets and stablecoins: never radar tokens themselves. */
        val QUOTE_SYMBOLS: Set<String> = setOf(
            "SOL", "WSOL", "ETH", "WETH", "BNB", "WBNB", "USDC", "USDT", "USDCE", "DAI", "USD1", "USDE", "FDUSD", "PYUSD",
            "BTC", "WBTC", "CBBTC", "TON", "TRX", "POL", "MATIC", "AVAX", "WAVAX", "S", "WS", "EURC", "USDG", "USDS",
        )

        /** EVM addresses are case-insensitive; Solana and TON addresses are not. */
        fun key(chain: String, address: String): String = chain + ":" + if (address.startsWith("0x")) address.lowercase() else address
    }
}
