package dev.cryptopulse.core.model

import kotlinx.serialization.Serializable

/** How we learned that Bitpanda lists this asset. */
@Serializable
enum class ListingSource {
    /** Live `https://api.bitpanda.com/v1/ticker` (legacy ticker, every asset in one call). */
    BITPANDA_LEGACY_TICKER,
    /** Live `https://api.public.bitpanda.com/assets?type=CRYPTO` (new Public API). */
    BITPANDA_PUBLIC_API,
    /** Bundled snapshot shipped with the app (`bitpanda_snapshot.json`). */
    BUNDLED_SNAPSHOT,
}

/** A cryptocurrency tradeable on the Bitpanda broker. */
@Serializable
data class Asset(
    /** Bitpanda symbol, normalized uppercase (e.g. "BTC", "1INCH", "ETHFI"). */
    val symbol: String,
    /** Display name as Bitpanda shows it (e.g. "Doge", "Sun (New)", "Global Dollar"). */
    val name: String,
    /** CoinGecko API id (e.g. "bitcoin"); null until resolved. */
    val coingeckoId: String? = null,
    /** UUID from the new Bitpanda Public API when known. */
    val bitpandaAssetId: String? = null,
    val listingSource: ListingSource = ListingSource.BUNDLED_SNAPSHOT,
    /** Approximate market-cap rank in the bundled snapshot (null for live sources). */
    val snapshotRank: Int? = null,
)
