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
        if (!force && current != null && now - current.fetchedAtMs < config.universeTtlMs) {
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
            assets = resolveIds(assets, coinList)
            if (live != null) {
                // The legacy ticker may contain stocks/ETFs: keep only symbols CoinGecko knows or the snapshot confirms.
                val known = coinList.mapTo(HashSet()) { it.symbol }
                assets = assets.filter { it.symbol in bySymbol || it.symbol in known }
            }
        }
        assets = assets.filter { Text.normalizeSymbol(it.symbol).isNotEmpty() }.distinctBy { it.symbol }

        val result = UniverseResult(assets, source, prices, now, warnings)
        cached = result
        persist(result)
        result
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
        private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

        /**
         * Validates CoinGecko id hints and maps the rest. Preference for symbol collisions:
         * exact normalized name match > name prefix/containment > shortest id (canonical coins have short ids).
         */
        fun resolveIds(assets: List<Asset>, coinList: List<CoinListEntry>): List<Asset> {
            val ids = coinList.mapTo(HashSet()) { it.id }
            val bySymbol = coinList.groupBy { it.symbol }
            return assets.map { a ->
                val hint = a.coingeckoId
                if (hint != null && hint in ids) return@map a
                val candidates = bySymbol[a.symbol].orEmpty()
                val chosen = pick(a, candidates)
                a.copy(coingeckoId = chosen?.id)
            }
        }

        private fun pick(asset: Asset, candidates: List<CoinListEntry>): CoinListEntry? {
            if (candidates.isEmpty()) return null
            if (candidates.size == 1) return candidates.first()
            val name = Text.normalizeName(asset.name)
            val noBridged = candidates.filterNot { c ->
                val n = c.name.lowercase()
                n.contains("bridged") || n.contains("wormhole") || n.contains("wrapped") || n.contains("binance-peg") || n.contains("(old)") || c.id.contains("-wormhole")
            }.ifEmpty { candidates }
            noBridged.firstOrNull { Text.normalizeName(it.name) == name }?.let { return it }
            noBridged.firstOrNull { val n = Text.normalizeName(it.name); name.isNotEmpty() && (n.startsWith(name) || name.startsWith(n)) }?.let { return it }
            return noBridged.minByOrNull { it.id.length }
        }
    }
}
