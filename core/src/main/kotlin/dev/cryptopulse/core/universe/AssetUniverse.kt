package dev.cryptopulse.core.universe

import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.ListingSource
import kotlinx.serialization.Serializable

/** The set of cryptocurrencies Bitpanda lists, with CoinGecko ids resolved where possible. */
@Serializable
data class UniverseResult(
    /** Crypto only (no metals, stocks, ETFs), unique by symbol. */
    val assets: List<Asset>,
    /** The live source that produced the list, or BUNDLED_SNAPSHOT when both live sources failed. */
    val source: ListingSource,
    /** symbol -> fiat code -> price, from the legacy ticker when it was reachable. */
    val bitpandaPrices: Map<String, Map<String, Double>> = emptyMap(),
    val fetchedAtMs: Long,
    val warnings: List<String> = emptyList(),
) {
    fun bySymbol(symbol: String): Asset? = assets.firstOrNull { it.symbol == symbol }
}

/**
 * Resolves the Bitpanda crypto universe. Implemented by `DefaultAssetUniverse`, which tries, in order:
 *  1. legacy ticker `https://api.bitpanda.com/v1/ticker` (keyless, then with the optional API key),
 *  2. new Public API `https://api.public.bitpanda.com/assets?type=CRYPTO` (keyless, then with key),
 *  3. the bundled snapshot resource `bitpanda_snapshot.json`,
 * merges them with the snapshot (names, ids), maps symbols to CoinGecko ids and caches the result
 * for [AppConfig.universeTtlMs].
 */
interface AssetUniverse {
    suspend fun load(config: AppConfig, force: Boolean = false): UniverseResult
}
