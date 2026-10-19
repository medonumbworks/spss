package dev.cryptopulse.app.ui

import dev.cryptopulse.core.model.AssetRow

/** Pure list filtering/sorting (unit-tested without Android). */
object ListOptions {
    fun apply(rows: List<AssetRow>, query: String, filter: ListFilter, sort: ListSort): List<AssetRow> {
        val q = query.trim().lowercase()
        val filtered = rows.asSequence()
            .filter { q.isEmpty() || it.asset.symbol.lowercase().contains(q) || it.asset.name.lowercase().contains(q) }
            .filter { r ->
                val f = r.forecast
                when (filter) {
                    ListFilter.ALL -> true
                    ListFilter.UP_24H -> (f?.h24?.probabilityUp ?: 0.0) >= 0.55
                    ListFilter.UP_7D -> (f?.d7?.probabilityUp ?: 0.0) >= 0.55
                    ListFilter.NEAR_BOTTOM -> (f?.bottom?.score ?: 0) >= 60 && f?.bottom?.label != "Pegged"
                    ListFilter.GAINERS -> (r.market?.change24hPct ?: 0.0) > 0
                    ListFilter.LOSERS -> (r.market?.change24hPct ?: 0.0) < 0
                    ListFilter.BULLISH_SOCIAL -> (r.sentiment?.composite ?: 0.0) >= 0.08 && (r.sentiment?.totalSamples ?: 0) > 0
                }
            }.toList()
        return when (sort) {
            ListSort.MARKET_CAP -> filtered
            ListSort.CHANGE_24H -> filtered.sortedByDescending { it.market?.change24hPct ?: Double.NEGATIVE_INFINITY }
            ListSort.CHANGE_7D -> filtered.sortedByDescending { it.market?.change7dPct ?: Double.NEGATIVE_INFINITY }
            ListSort.UP_24H -> filtered.sortedByDescending { it.forecast?.h24?.probabilityUp ?: 0.0 }
            ListSort.UP_7D -> filtered.sortedByDescending { it.forecast?.d7?.probabilityUp ?: 0.0 }
            ListSort.BOTTOM -> filtered.sortedByDescending { it.forecast?.bottom?.score ?: -1 }
            ListSort.SENTIMENT -> filtered.sortedByDescending { it.sentiment?.composite ?: -2.0 }
            ListSort.NAME -> filtered.sortedBy { it.asset.name.lowercase() }
        }
    }
}
