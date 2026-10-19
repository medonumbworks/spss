package dev.cryptopulse.app.ui

import dev.cryptopulse.core.analysis.forecast.DefaultForecastEngine
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.Tier

/** Pure list filtering/sorting (unit-tested without Android). */
object ListOptions {
    private val NO_BOTTOM = setOf("Pegged", DefaultForecastEngine.NOT_ASSESSED, DefaultForecastEngine.NO_HISTORY)

    fun inTier(r: AssetRow, tier: ListTier): Boolean = when (tier) {
        ListTier.ALL -> true
        ListTier.BITPANDA -> r.asset.onBitpanda
        ListTier.MEME -> r.asset.isMeme
        ListTier.TRENDING -> r.asset.tier == Tier.TRENDING || "trending" in r.asset.tags
        ListTier.DEX -> r.asset.tier == Tier.DEX
    }

    /** Tier chips: All, every tier with rows, and the selected one even when it has none, so an empty list keeps its cause in sight. */
    fun visibleTiers(counts: Map<ListTier, Int>, selected: ListTier): List<ListTier> =
        ListTier.entries.filter { it == ListTier.ALL || it == selected || (counts[it] ?: 0) > 0 }

    fun apply(rows: List<AssetRow>, query: String, filter: ListFilter, sort: ListSort, tier: ListTier = ListTier.ALL): List<AssetRow> {
        val q = query.trim().lowercase()
        val filtered = rows.asSequence()
            .filter { inTier(it, tier) }
            .filter { q.isEmpty() || it.asset.symbol.lowercase().contains(q) || it.asset.name.lowercase().contains(q) }
            .filter { r ->
                val f = r.forecast
                when (filter) {
                    ListFilter.ALL -> true
                    ListFilter.UP_24H -> (r.upOrNull(f?.h24) ?: 0.0) >= 0.55
                    ListFilter.UP_7D -> (r.upOrNull(f?.d7) ?: 0.0) >= 0.55
                    ListFilter.NEAR_BOTTOM -> (f?.bottom?.score ?: 0) >= 60 && f?.bottom?.label !in NO_BOTTOM && r.risk?.blocked != true
                    ListFilter.GAINERS -> (r.market?.change24hPct ?: 0.0) > 0
                    ListFilter.LOSERS -> (r.market?.change24hPct ?: 0.0) < 0
                    ListFilter.BULLISH_SOCIAL -> (r.sentiment?.composite ?: 0.0) >= 0.08 && (r.sentiment?.totalSamples ?: 0) > 0
                }
            }.toList()
        return when (sort) {
            ListSort.MARKET_CAP -> filtered
            ListSort.CHANGE_24H -> filtered.sortedByDescending { it.market?.change24hPct ?: Double.NEGATIVE_INFINITY }
            ListSort.CHANGE_7D -> filtered.sortedByDescending { it.market?.change7dPct ?: Double.NEGATIVE_INFINITY }
            ListSort.UP_24H -> filtered.sortedByDescending { it.upOrNull(it.forecast?.h24) ?: 0.0 }
            ListSort.UP_7D -> filtered.sortedByDescending { it.upOrNull(it.forecast?.d7) ?: 0.0 }
            ListSort.BOTTOM -> filtered.sortedByDescending { r -> r.forecast?.bottom?.takeIf { it.label !in NO_BOTTOM }?.score ?: -1 }
            ListSort.SENTIMENT -> filtered.sortedByDescending { it.sentiment?.composite ?: -2.0 }
            ListSort.VOLUME -> filtered.sortedByDescending { it.dex?.volumeUsd?.h24 ?: it.market?.volume24h ?: -1.0 }
            // Bitpanda coins have no risk score; they come first (they are the regulated, tradeable ones)
            ListSort.RISK -> filtered.sortedBy { it.risk?.score ?: -1 }
            ListSort.NEWEST -> filtered.sortedByDescending { it.dex?.pairCreatedAtMs ?: Long.MIN_VALUE }
            ListSort.NAME -> filtered.sortedBy { it.asset.name.lowercase() }
        }
    }
}
