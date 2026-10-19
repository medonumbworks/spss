package dev.cryptopulse.app.ui

import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.MarketData
import kotlin.test.Test
import kotlin.test.assertEquals

class ListOptionsTest {
    private fun row(sym: String, name: String, change: Double, cap: Double) =
        AssetRow(Asset(sym, name), MarketData(symbol = sym, currency = "EUR", price = 1.0, change24hPct = change, marketCap = cap))

    private val rows = listOf(row("BTC", "Bitcoin", -0.8, 1e12), row("SOL", "Solana", 3.0, 5e10), row("PEPE", "Pepe", -5.0, 1e9))

    @Test
    fun searchFilterAndSort() {
        assertEquals(listOf("SOL"), ListOptions.apply(rows, "sol", ListFilter.ALL, ListSort.MARKET_CAP).map { it.asset.symbol })
        assertEquals(listOf("BTC", "PEPE"), ListOptions.apply(rows, "", ListFilter.LOSERS, ListSort.MARKET_CAP).map { it.asset.symbol })
        assertEquals(listOf("SOL", "BTC", "PEPE"), ListOptions.apply(rows, "", ListFilter.ALL, ListSort.CHANGE_24H).map { it.asset.symbol })
        assertEquals(listOf("BTC", "PEPE", "SOL"), ListOptions.apply(rows, "", ListFilter.ALL, ListSort.NAME).map { it.asset.symbol })
    }
}
