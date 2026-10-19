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

/**
 * A cryptocurrency in the market list. [tier] BITPANDA = tradeable on the Bitpanda broker; the
 * other tiers (memecoins, trending coins, DEX-only tokens) are shown for information only.
 */
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
    /**
     * True when the listing is confirmed (live Bitpanda API, or seen in the Bitpanda app snapshot).
     * Unverified research candidates are hidden when their market cap proves they are not on Bitpanda.
     */
    val verified: Boolean = true,
    val tier: Tier = Tier.BITPANDA,
    /** Extra labels: "meme" (CoinGecko Meme category), "trending", "dex_trending", "boosted", "new", "pumpfun". */
    val tags: List<String> = emptyList(),
    /** DEX tokens: DexScreener chain id and token contract/mint address. */
    val chainId: String? = null,
    val contract: String? = null,
) {
    val onBitpanda: Boolean get() = tier == Tier.BITPANDA
    val isMeme: Boolean get() = tier == Tier.MEME || "meme" in tags
}
