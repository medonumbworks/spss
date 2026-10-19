package dev.cryptopulse.core.analysis.risk

import dev.cryptopulse.core.model.DexInfo
import dev.cryptopulse.core.model.DexTxns
import dev.cryptopulse.core.model.DexWindow
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.model.TokenSafety
import dev.cryptopulse.core.sources.safety.TokenSafetyApi
import dev.cryptopulse.core.testing.Fixtures
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Fail-closed gaps and false alarms of [RiskScorer] found in review. */
class RiskScorerTest {
    private val now = 1_792_411_200_000L
    private val day = 86_400_000L

    private fun pool(chain: String, liquidity: Double = 400_000.0, cap: Double = 5_000_000.0) = DexInfo(
        chainId = chain, tokenAddress = "0xabc", pairAddress = "0xpool", liquidityUsd = liquidity, marketCapUsd = cap,
        priceChangePct = DexWindow(h1 = 1.0, h24 = 4.0), volumeUsd = DexWindow(h24 = 300_000.0),
        txnsH24 = DexTxns(900, 800), pairCreatedAtMs = now - 400 * day,
        websites = listOf("https://example.invalid"),
    )

    @Test
    fun anEvmTokenWhoseSellWasNeverTestedIsUnverified() {
        // GoPlus answered, but without a honeypot or tax result (it could not simulate a trade)
        val untested = TokenSafety("goplus", openSource = true, mintable = false, ownerRenounced = true)
        val closed = TokenSafety("goplus", openSource = false)
        for (s in listOf(untested, closed)) {
            val r = RiskScorer.assess(Tier.DEX, null, pool("ethereum"), s, now)!!
            assertEquals(RiskScorer.LABEL_UNVERIFIED, r.label, "${r.reasons}")
            assertTrue(r.blocked)
        }
        assertFalse(RiskScorer.assess(Tier.DEX, null, pool("ethereum"), untested.copy(honeypot = false), now)!!.blocked)
        assertFalse(RiskScorer.assess(Tier.DEX, null, pool("base"), untested.copy(sellTaxPct = 0.0), now)!!.blocked)
        // GoPlus never simulates Solana trades: that is not a gap
        assertFalse(RiskScorer.assess(Tier.DEX, null, pool("solana"), untested.copy(freezable = false), now)!!.blocked)
    }

    @Test
    fun smallOrUnrankedTrendingCoinsGetNoForecast() {
        fun trending(rank: Int?, cap: Double) = RiskScorer.assess(Tier.TRENDING, MarketData("NEW", currency = "EUR", marketCap = cap, marketCapRank = rank, volume24h = cap / 10), null, null, now)!!
        for (r in listOf(trending(null, 3e6), trending(null, 5e7), trending(900, 3e6), trending(900, 0.0))) {
            assertEquals(RiskScorer.LABEL_UNVERIFIED, r.label, "${r.reasons}")
            assertTrue(r.blocked)
            assertTrue(r.reasons.first().startsWith("Trending coin with"), r.reasons.first())
        }
        assertFalse(trending(300, 5e7).blocked)
        // memecoins come from CoinGecko's category, not from the trending list: unchanged
        assertFalse(RiskScorer.assess(Tier.MEME, MarketData("MEME", currency = "EUR", marketCap = 3e6, volume24h = 3e5), null, null, now)!!.blocked)
    }

    @Test
    fun aZeroMarketCapIsUnknownNotMicro() {
        val r = RiskScorer.assess(Tier.MEME, MarketData("NEW", currency = "EUR", marketCap = 0.0, volume24h = 1e6), null, null, now)!!
        assertTrue("Market cap unknown" in r.reasons, "${r.reasons}")
        assertTrue(r.reasons.none { it.startsWith("Micro cap") }, "${r.reasons}")
    }

    @Test
    fun aThinPoolDoesNotTrapHoldersOfListedOrLargeCoins() {
        val clean = TokenSafety("goplus", honeypot = false, buyTaxPct = 0.0, sellTaxPct = 0.0, openSource = true)
        fun exitReason(cap: Double, s: TokenSafety) = RiskScorer.assess(Tier.DEX, null, pool("ethereum", liquidity = 1_000_000.0, cap = cap), s, now)!!
            .reasons.any { it.contains("cannot exit") }
        assertTrue(exitReason(2e8, clean))
        assertFalse(exitReason(2e8, clean.copy(cexListed = listOf("Binance"))))
        assertFalse(exitReason(3e9, clean))
        // PEPE as a DEX row: its LP sits in its own renounced contract and it trades on Binance and Coinbase
        val pepe = TokenSafetyApi.parseGoPlusEvm(Fixtures.read("spec/goplus_eth.json"), now).values.single()
        val r = RiskScorer.assess(Tier.DEX, null, pool("ethereum", liquidity = 14_600_000.0, cap = 1.7e9), pepe, now)!!
        assertTrue(r.reasons.none { it.contains("locked or burnt") || it.contains("cannot exit") }, "${r.reasons}")
        assertEquals(RiskScorer.LABEL_LOWER, r.label)
    }
}
