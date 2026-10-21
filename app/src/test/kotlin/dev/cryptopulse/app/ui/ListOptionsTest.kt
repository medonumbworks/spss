package dev.cryptopulse.app.ui

import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.RiskAssessment
import dev.cryptopulse.core.model.Tier
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

    @Test
    fun tiersBeyondBitpanda() {
        val meme = AssetRow(Asset("SHIB", "Shiba Inu", "shiba-inu", tier = Tier.MEME), MarketData("SHIB", currency = "EUR", price = 1e-5), risk = RiskAssessment(20, "Lower risk"))
        val dex = AssetRow(Asset("MOON", "Moon", tier = Tier.DEX), MarketData("MOON", currency = "EUR", price = 0.01), risk = RiskAssessment(90, "Too new", blocked = true))
        val pepeOnBitpanda = row("PEPE", "Pepe", -5.0, 1e9).let { it.copy(asset = it.asset.copy(tags = listOf("meme"))) }
        val all = listOf(rows[0], rows[1], pepeOnBitpanda, meme, dex)
        assertEquals(listOf("BTC", "SOL", "PEPE"), ListOptions.apply(all, "", ListFilter.ALL, ListSort.MARKET_CAP, ListTier.BITPANDA).map { it.asset.symbol })
        assertEquals(listOf("BTC", "SOL", "PEPE", "SHIB", "MOON"), ListOptions.apply(all, "", ListFilter.ALL, ListSort.RISK).map { it.asset.symbol }, "Bitpanda coins first, then lowest risk")
        assertEquals(null, dex.upOrNull(null))
    }

    @Test
    fun selectedTierChipStaysWhenItsTierEmpties() {
        val counts = mapOf(ListTier.ALL to 650, ListTier.BITPANDA to 650, ListTier.TRENDING to 0)
        assertEquals(listOf(ListTier.ALL, ListTier.BITPANDA), ListOptions.visibleTiers(counts, ListTier.ALL))
        assertEquals(listOf(ListTier.ALL, ListTier.BITPANDA, ListTier.TRENDING), ListOptions.visibleTiers(counts, ListTier.TRENDING), "trending turned off while selected")
        assertEquals(listOf(ListTier.ALL, ListTier.TRENDING), ListOptions.visibleTiers(emptyMap(), ListTier.TRENDING))
    }
}
