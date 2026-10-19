package dev.cryptopulse.core.model

import kotlinx.serialization.Serializable

/** One row of the market list: the asset, its market data and (when computed) forecast and sentiment. */
@Serializable
data class AssetRow(
    val asset: Asset,
    val market: MarketData? = null,
    val forecast: Forecast? = null,
    val sentiment: SentimentSummary? = null,
    /** DEX tokens: pool, liquidity and trading flow. */
    val dex: DexInfo? = null,
    /** Contract checks (DEX tokens). */
    val safety: TokenSafety? = null,
    /** Risk for everything that is not on Bitpanda; null for Bitpanda assets. */
    val risk: RiskAssessment? = null,
) {
    /** The horizon's chance of going up, or null when there is none (risk block, too little history). */
    fun upOrNull(h: HorizonForecast?): Double? = h?.takeIf { risk?.blocked != true && it.hasCall }?.probabilityUp
}

@Serializable
data class MarketSnapshot(
    val currency: String,
    val rows: List<AssetRow>,
    val context: MarketContext? = null,
    /** Which source produced the asset universe this time. */
    val universeSource: ListingSource,
    val universeSize: Int,
    val fetchedAtMs: Long,
    val warnings: List<String> = emptyList(),
    /** USD -> [currency] used for DEX prices; kept so a restarted app converts new candles the same way. */
    val usdToQuote: Double? = null,
)

/** Everything the detail screen needs for one asset. */
@Serializable
data class AssetDetail(
    val row: AssetRow,
    /** Daily closes (oldest first), typically 90 days. */
    val daily: List<PricePoint> = emptyList(),
    /** Hourly closes for the last 7 days (oldest first). */
    val hourly: List<PricePoint> = emptyList(),
    val community: CoinCommunity? = null,
    val derivatives: DerivativesSnapshot? = null,
    /** Prices from the Bitpanda ticker in several fiat currencies, when available. */
    val bitpandaPrices: Map<String, Double> = emptyMap(),
    val fetchedAtMs: Long,
    val warnings: List<String> = emptyList(),
)

@Serializable
sealed class RefreshStatus {
    @Serializable
    data object Idle : RefreshStatus()

    @Serializable
    data class Refreshing(val step: String, val progress: Float) : RefreshStatus()

    @Serializable
    data class Error(val message: String) : RefreshStatus()
}
