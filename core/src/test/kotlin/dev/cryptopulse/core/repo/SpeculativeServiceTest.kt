package dev.cryptopulse.core.repo

import dev.cryptopulse.core.analysis.risk.RiskScorer
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.sources.coingecko.CoinGeckoApi
import dev.cryptopulse.core.sources.dex.DexScreenerApi
import dev.cryptopulse.core.sources.dex.GeckoTerminalApi
import dev.cryptopulse.core.sources.dex.PumpFunApi
import dev.cryptopulse.core.sources.paprika.CoinpaprikaApi
import dev.cryptopulse.core.sources.safety.TokenSafetyApi
import dev.cryptopulse.core.testHttp
import dev.cryptopulse.core.testing.FakeTransport
import dev.cryptopulse.core.testing.Fixtures
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.Logger
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Memecoins, trending coins and the DEX radar built from payloads captured live on 2026-10-19. */
class SpeculativeServiceTest {
    private val clock = Clock.fixed(SPEC_NOW_MS)

    private fun spec(name: String) = Fixtures.read("spec/$name")

    private fun transport() = FakeTransport()
        .onUrlContaining("category=meme-token", body = spec("coingecko_meme_markets.json"))
        .onUrlContaining("networks/trending_pools", body = spec("geckoterminal_trending.json"))
        .onUrlContaining("networks/new_pools", body = spec("geckoterminal_new_pools.json"))
        .onUrlContaining("/token-boosts/top/v1", body = spec("dexscreener_boosts_top.json"))
        .onUrlContaining("/token-boosts/latest/v1", body = spec("dexscreener_boosts_latest.json"))
        .onUrlContaining("/token-profiles/latest/v1", body = spec("dexscreener_profiles_latest.json"))
        .onUrlContaining("/community-takeovers/latest/v1", body = spec("dexscreener_takeovers.json"))
        .onUrlContaining("/tokens/v1/solana/", body = spec("dexscreener_tokens_boosted.json"))
        .onUrlContaining("pump.fun/coins/currently-live", body = spec("pumpfun_live.json"))
        .onUrlContaining("pump.fun/coins?", body = spec("pumpfun_top.json"))
        .onUrlContaining("gopluslabs.io/api/v1/solana/", body = spec("goplus_sol.json"))

    private fun service(t: FakeTransport): SpeculativeService {
        val http = testHttp(t, clock)
        return SpeculativeService(CoinGeckoApi(http), DexScreenerApi(http), GeckoTerminalApi(http), PumpFunApi(http), TokenSafetyApi(http, clock), clock, Logger.NONE, CoinpaprikaApi(http))
    }

    private val bitpanda = listOf(
        AssetRow(Asset("BTC", "Bitcoin", "bitcoin"), MarketData("BTC", "bitcoin", currency = "EUR", price = 70_000.0, marketCap = 1.4e12)),
        AssetRow(Asset("DOGE", "Doge", "dogecoin"), MarketData("DOGE", "dogecoin", currency = "EUR", price = 0.07, marketCap = 1e10)),
    )

    @Test
    fun memecoinsAndDexTokensBeyondBitpanda() = runTest {
        val r = service(transport()).load(AppConfig(), bitpanda, emptyList(), usdToQuote = 0.9, previous = emptyList())
        val byTier = r.rows.groupBy { it.asset.tier }
        assertEquals(listOf("SHIB", "PUMP"), byTier[Tier.MEME]!!.map { it.asset.symbol }, "DOGE stays the Bitpanda row")
        assertTrue("dogecoin" in r.memeIds, "Bitpanda's DOGE gets the meme tag")
        val dex = byTier[Tier.DEX]!!.associateBy { it.asset.symbol }
        assertEquals(setOf("TORCH", "EMPIREX", "SI276", "CATONICE"), dex.keys, "pools under \$10k liquidity or volume are dust")
        val torch = dex.getValue("TORCH")
        assertEquals("robinhood", torch.asset.chainId)
        assertEquals("EUR", torch.market!!.currency)
        assertEquals(0.00881099054501289 * 0.9, torch.market.price!!, 1e-12, "USD pool prices converted to the quote currency")
        assertNull(torch.safety, "no free contract check for this chain")
        val cat = dex.getValue("CATONICE")
        assertTrue("ds_boost" in cat.dex!!.foundBy)
        assertEquals(500, cat.dex.boostsActive)
        assertTrue("boosted" in cat.asset.tags)
        assertNotNull(cat.safety, "Solana tokens get a GoPlus check")
    }

    @Test
    fun riskBlocksYoungUncheckedAndThinTokens() = runTest {
        val r = service(transport()).load(AppConfig(), bitpanda, emptyList(), 0.9, emptyList())
        val dex = r.rows.filter { it.asset.tier == Tier.DEX }.associateBy { it.asset.symbol }
        fun risk(sym: String) = dex.getValue(sym).let { RiskScorer.assess(it.asset.tier, it.market, it.dex, it.safety, SPEC_NOW_MS)!! }
        assertEquals(RiskScorer.LABEL_TOO_NEW, risk("CATONICE").label, "pool created 20 h before the capture")
        assertTrue(risk("CATONICE").blocked)
        assertEquals(RiskScorer.LABEL_UNVERIFIED, risk("TORCH").label)
        assertTrue(risk("TORCH").blocked)
        val si = risk("SI276")
        assertFalse(si.label == RiskScorer.LABEL_UNVERIFIED)
    }

    @Test
    fun aDexTokenNamedLikeAListedCoinIsLeftOut() = runTest {
        val listedTorch = bitpanda + AssetRow(Asset("TORCH", "Torch (listed)", "other-torch"), MarketData("TORCH", currency = "EUR", price = 1.0))
        val r = service(transport()).load(AppConfig(), listedTorch, emptyList(), 0.9, emptyList())
        assertTrue(r.rows.none { it.asset.tier == Tier.DEX && it.asset.symbol == "TORCH" })
    }

    @Test
    fun tiersCanBeSwitchedOffAndFailuresKeepTheLastMemecoins() = runTest {
        val off = service(transport()).load(AppConfig(showMemecoins = false, showTrending = false, showDexRadar = false), bitpanda, emptyList(), 0.9, emptyList())
        assertTrue(off.rows.isEmpty())
        val previous = listOf(AssetRow(Asset("SHIB", "Shiba Inu", "shiba-inu", tier = Tier.MEME), MarketData("SHIB", "shiba-inu", currency = "EUR", price = 0.00001)))
        val down = service(FakeTransport().failUrlContaining("coingecko")).load(AppConfig(showDexRadar = false), bitpanda, emptyList(), 0.9, previous)
        assertEquals(listOf("SHIB"), down.rows.map { it.asset.symbol })
        assertTrue(down.warnings.any { it.startsWith("Memecoins") })
    }

    @Test
    fun coinpaprikaFillsTheMemeTierWhenCoinGeckoRefuses() = runTest {
        val t = FakeTransport()
            .failUrlContaining("coingecko")
            .onUrlContaining("coinpaprika.com/v1/tags/meme-coin", body = spec("coinpaprika_meme_tag.json"))
            .onUrlContaining("coinpaprika.com/v1/tickers", body = spec("coinpaprika_tickers.json"))
        val taken = listOf(AssetRow(Asset("DOGE", "Dogecoin", "dogecoin"), MarketData("DOGE", "dogecoin", currency = "EUR", price = 0.07)))
        val r = service(t).load(AppConfig(showDexRadar = false), taken, emptyList(), 0.9, emptyList())
        val memes = r.rows.filter { it.asset.tier == Tier.MEME }
        assertEquals(listOf("PEPE", "BONK", "WIF"), memes.map { it.asset.symbol }, "best ranked first; Bitpanda keeps DOGE")
        assertEquals("coinpaprika", memes.first().market!!.source)
        assertEquals("EUR", memes.first().market!!.currency)
        assertEquals(setOf("DOGE"), r.memeSymbols, "Bitpanda's DOGE gets the meme tag by ticker and name")
        assertTrue(r.warnings.any { it.startsWith("Memecoins") })
    }

    @Test
    fun safetyChecksAreBudgetedPerRefresh() = runTest {
        val t = transport()
        val svc = service(t)
        svc.load(AppConfig(dexSafetyTopN = 100), bitpanda, emptyList(), 0.9, emptyList())
        val first = t.countRequests("gopluslabs.io")
        assertTrue(first in 1..SpeculativeService.NEW_CHECKS_PER_REFRESH)
        svc.load(AppConfig(dexSafetyTopN = 100), bitpanda, emptyList(), 0.9, emptyList())
        assertEquals(first, t.countRequests("gopluslabs.io"), "cached checks are reused, not repeated")
    }

    @Test
    fun keysAreCaseSensitiveExceptOnEvm() {
        assertEquals(SpeculativeService.key("base", "0xABC"), SpeculativeService.key("base", "0xabc"))
        assertFalse(SpeculativeService.key("solana", "AbC") == SpeculativeService.key("solana", "abc"))
    }

    companion object {
        /** 2026-10-19T12:00:00Z, just after the samples were captured. */
        const val SPEC_NOW_MS = 1_792_411_200_000L
    }
}
