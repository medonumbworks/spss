package dev.cryptopulse.core.universe

import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.http.DiskStore
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.ListingSource
import dev.cryptopulse.core.sources.BitpandaSource
import dev.cryptopulse.core.sources.CoinGeckoSource
import dev.cryptopulse.core.sources.CoinListEntry
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.sources.bitpanda.BitpandaApi
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.Logger
import dev.cryptopulse.core.util.Text
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.Json

/**
 * Bitpanda crypto universe:
 *  1. live legacy ticker (all assets with EUR/USD/CHF/GBP prices),
 *  2. else live Public API asset list (type CRYPTO),
 *  3. else the bundled snapshot.
 * Live lists are merged with the snapshot for display names and CoinGecko hints; ids are validated
 * against CoinGecko's `/coins/list` and unknown symbols are mapped by symbol + name.
 */
class DefaultAssetUniverse(
    private val bitpanda: BitpandaSource,
    private val coinGecko: CoinGeckoSource,
    private val snapshotLoader: () -> List<Asset> = { BundledSnapshot.load() },
    private val store: DiskStore? = null,
    private val clock: Clock = Clock.SYSTEM,
    private val logger: Logger = Logger.STDERR,
) : AssetUniverse {

    private val mutex = Mutex()

    @Volatile
    private var cached: UniverseResult? = null

    override suspend fun load(config: AppConfig, force: Boolean): UniverseResult = mutex.withLock {
        val now = clock.nowMs()
        val current = cached ?: readPersisted()
        val ttl = if (current?.complete == false) DEGRADED_TTL_MS else config.universeTtlMs
        if (!force && current != null && now - current.fetchedAtMs < ttl) {
            // prices in the legacy ticker age quickly; refresh them cheaply when we can
            return@withLock refreshPricesIfPossible(current, config).also { cached = it }
        }
        val snapshot = snapshotLoader()
        val bySymbol = snapshot.associateBy { it.symbol }
        val warnings = ArrayList<String>()
        var source = ListingSource.BUNDLED_SNAPSHOT
        var prices: Map<String, Map<String, Double>> = emptyMap()
        var live: List<Asset>? = null

        when (val t = bitpanda.legacyTicker(config.bitpandaApiKey)) {
            is SourceResult.Success -> {
                prices = t.value.entries.associate { it.symbol to it.prices }
                live = t.value.entries.map { it.symbol }.filter { it !in BitpandaApi.NON_CRYPTO_SYMBOLS }.map { sym ->
                    (bySymbol[sym] ?: Asset(symbol = sym, name = sym)).copy(listingSource = ListingSource.BITPANDA_LEGACY_TICKER, verified = true)
                }
                source = ListingSource.BITPANDA_LEGACY_TICKER
            }
            is SourceResult.Failure -> {
                warnings += "Bitpanda ticker: ${t.message}"
                when (val p = bitpanda.publicAssets(config.bitpandaApiKey, "CRYPTO")) {
                    is SourceResult.Success -> {
                        live = p.value.map { pa ->
                            val base = bySymbol[pa.symbol] ?: Asset(symbol = pa.symbol, name = pa.name ?: pa.symbol)
                            base.copy(
                                name = if (bySymbol.containsKey(pa.symbol)) base.name else (pa.name ?: pa.symbol),
                                bitpandaAssetId = pa.id,
                                listingSource = ListingSource.BITPANDA_PUBLIC_API,
                                verified = true,
                            )
                        }
                        source = ListingSource.BITPANDA_PUBLIC_API
                    }
                    is SourceResult.Failure -> warnings += "Bitpanda asset list: ${p.message}; using the bundled list of ${snapshot.size} assets"
                }
            }
        }

        var assets = live ?: snapshot
        val coinList = when (val c = coinGecko.coinsList()) {
            is SourceResult.Success -> c.value
            is SourceResult.Failure -> { warnings += "CoinGecko coin list: ${c.message}"; null }
        }
        if (coinList != null) {
            assets = resolveIds(assets, coinList, marketCapRanks(assets, coinList, config))
            if (live != null) {
                // The legacy ticker may contain stocks/ETFs: keep only symbols CoinGecko knows or the snapshot confirms.
                val known = coinList.mapTo(HashSet()) { it.symbol }
                assets = assets.filter { it.symbol in bySymbol || it.symbol in known }
            }
        }
        assets = assets.filter { Text.normalizeSymbol(it.symbol).isNotEmpty() }.distinctBy { it.symbol }

        val complete = source != ListingSource.BUNDLED_SNAPSHOT && coinList != null
        if (!complete && current != null && current.complete) {
            // a short outage during a forced refresh must not replace a good list for a day
            logger.debug(TAG, "keeping the previous complete universe: ${warnings.joinToString()}")
            return@withLock refreshPricesIfPossible(current, config).also { cached = it }
        }
        val result = UniverseResult(assets, source, prices, now, warnings, complete)
        cached = result
        persist(result)
        result
    }

    /**
     * CoinGecko market-cap ranks, fetched only when some symbol has several candidate coins and no
     * valid id hint (new listings): the real coin is almost always the one with the larger market cap.
     */
    private suspend fun marketCapRanks(assets: List<Asset>, coinList: List<CoinListEntry>, config: AppConfig): Map<String, Int> {
        val ids = coinList.mapTo(HashSet()) { it.id }
        val counts = coinList.groupingBy { it.symbol }.eachCount()
        val needed = assets.any { a -> (a.coingeckoId == null || a.coingeckoId !in ids) && (counts[a.symbol] ?: 0) > 1 }
        if (!needed) return emptyMap()
        val ranks = HashMap<String, Int>()
        for (page in 1..config.coingeckoMappingPages.coerceIn(1, 8)) {
            val r = coinGecko.markets("usd", page = page, perPage = 250, ids = null, sparkline = false, ttlMs = 24 * 3_600_000L) as? SourceResult.Success ?: break
            r.value.forEachIndexed { i, m -> m.coingeckoId?.let { ranks.putIfAbsent(it, m.marketCapRank ?: ((page - 1) * 250 + i + 1)) } }
            if (r.value.size < 250) break
        }
        return ranks
    }

    private suspend fun refreshPricesIfPossible(current: UniverseResult, config: AppConfig): UniverseResult {
        if (current.source != ListingSource.BITPANDA_LEGACY_TICKER) return current
        val t = bitpanda.legacyTicker(config.bitpandaApiKey) as? SourceResult.Success ?: return current
        return current.copy(bitpandaPrices = t.value.entries.associate { it.symbol to it.prices })
    }

    private fun persist(result: UniverseResult) {
        try {
            store?.write(STORE_KEY, json.encodeToString(UniverseResult.serializer(), result))
        } catch (e: Exception) {
            logger.warn(TAG, "persist failed", e)
        }
    }

    private fun readPersisted(): UniverseResult? = try {
        store?.read(STORE_KEY)?.let { json.decodeFromString(UniverseResult.serializer(), it) }
    } catch (_: Exception) {
        null
    }

    companion object {
        private const val TAG = "AssetUniverse"
        private const val STORE_KEY = "universe_v1"
        private const val DEGRADED_TTL_MS = 15 * 60_000L
        private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

        /**
         * Validates CoinGecko id hints and maps the rest. Preference for symbol collisions:
         * exact normalized name match > name prefix/containment > shortest id (canonical coins have short ids).
         */
        fun resolveIds(assets: List<Asset>, coinList: List<CoinListEntry>, ranks: Map<String, Int> = emptyMap()): List<Asset> {
            val ids = coinList.mapTo(HashSet()) { it.id }
            val bySymbol = coinList.groupBy { it.symbol }
            return assets.map { a ->
                val hint = a.coingeckoId
                if (hint != null && hint in ids) return@map a
                val candidates = bySymbol[a.symbol].orEmpty()
                val chosen = pick(a, candidates, ranks)
                // ticker-only symbols arrive without a name; "PEOPLE" as a name would match the word "people"
                val name = if (chosen != null && a.name == a.symbol) chosen.name else a.name
                a.copy(coingeckoId = chosen?.id, name = name)
            }
        }

        private val BRIDGED = Regex("(?i)\\b(bridged?|wormhole|binance-peg|peg|old)\\b")

        private fun pick(asset: Asset, candidates: List<CoinListEntry>, ranks: Map<String, Int>): CoinListEntry? {
            if (candidates.isEmpty()) return null
            if (candidates.size == 1) return candidates.first()
            val name = Text.normalizeName(asset.name)
            val wantsWrapped = asset.name.contains("wrapped", ignoreCase = true) || asset.symbol.startsWith("W")
            val noBridged = candidates.filterNot { c ->
                BRIDGED.containsMatchIn(c.name) || c.id.contains("-wormhole") || c.id.endsWith("-bridge") ||
                    (!wantsWrapped && c.name.contains("wrapped", ignoreCase = true))
            }.ifEmpty { candidates }
            // without a real name (ticker-only symbol) the market-cap rank is the only reliable signal
            val ranked = noBridged.filter { it.id in ranks }.minByOrNull { ranks.getValue(it.id) }
            if (asset.name == asset.symbol && ranked != null) return ranked
            noBridged.firstOrNull { Text.normalizeName(it.name) == name }?.let { return it }
            noBridged.firstOrNull { val n = Text.normalizeName(it.name); name.isNotEmpty() && (n.startsWith(name) || name.startsWith(n)) }?.let { return it }
            return ranked ?: noBridged.minByOrNull { it.id.length }
        }
    }
}
