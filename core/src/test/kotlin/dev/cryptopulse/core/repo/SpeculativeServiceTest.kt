package dev.cryptopulse.core.repo

import dev.cryptopulse.core.analysis.risk.RiskScorer
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.http.HttpResponse
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.model.TokenSafety
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
        val usd = service(FakeTransport().failUrlContaining("coingecko")).load(AppConfig(currency = "USD", showDexRadar = false), bitpanda, emptyList(), 1.0, previous)
        assertTrue(usd.rows.isEmpty(), "EUR prices are not reused in a USD list")
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
    fun aUsedUpQuotaSkipsNewChecksButKeepsCachedOnes() = runTest {
        val (a, b, c) = listOf("a", "b", "c").map { "0x" + it.repeat(40) }
        var quotaLeft = true
        val t = FakeTransport()
            .onUrlContaining("networks/trending_pools", body = gtPools("eth", a, b, c))
            .onUrlContaining("gopluslabs.io/api/v1/token_security/1") { req ->
                HttpResponse(200, if (quotaLeft) goPlusEvm(req.url.substringAfter("contract_addresses=")) else QUOTA_REPLY, emptyMap(), req.url)
            }
        val http = testHttp(t, clock)
        val checks = TokenSafetyApi(http, clock)
        val svc = SpeculativeService(CoinGeckoApi(http), DexScreenerApi(http), GeckoTerminalApi(http), PumpFunApi(http), checks, clock, Logger.NONE)
        checks.check("ethereum", c)
        quotaLeft = false
        val limited = svc.load(DEX_ONLY, bitpanda, emptyList(), 0.9, emptyList())
        assertEquals(listOf("TK0", "TK1", "TK2"), limited.rows.map { it.asset.symbol })
        assertNull(limited.rows[0].safety)
        assertNull(limited.rows[1].safety)
        assertNotNull(limited.rows[2].safety, "a cached check after the quota reply is still read")
        assertEquals(0, t.countRequests("contract_addresses=$b"), "no new checks once GoPlus says the quota is used up")
        assertEquals(1, limited.warnings.count { it.startsWith("Safety checks") })
        quotaLeft = true
        val back = svc.load(DEX_ONLY, bitpanda, emptyList(), 0.9, emptyList())
        assertTrue(back.rows.all { it.safety != null }, "the quota reply was not cached")
    }

    @Test
    fun aHoneypotFoundWhenTheTokenWasOpenedSurvivesTheNextRefresh() = runTest {
        val a = "0x" + "a".repeat(40)
        // GoPlus misses it; honeypot.is (run when the token was opened) did not
        val t = FakeTransport()
            .onUrlContaining("networks/trending_pools", body = gtPools("eth", a))
            .onUrlContaining("gopluslabs.io/api/v1/token_security/1", body = goPlusEvm(a))
        val found = TokenSafety("goplus", honeypot = true, buyTaxPct = 0.0, sellTaxPct = 100.0, checkedAtMs = SPEC_NOW_MS - 3_600_000L)
        val opened = AssetRow(Asset("TK0", "Token 0", tier = Tier.DEX, chainId = "ethereum", contract = "0x" + "A".repeat(40)), safety = found)
        val svc = service(t)
        val row = svc.load(DEX_ONLY, bitpanda, emptyList(), 0.9, listOf(opened)).rows.single()
        assertEquals(true, row.safety!!.honeypot)
        assertEquals(100.0, row.safety.sellTaxPct)
        assertEquals(SPEC_NOW_MS - 3_600_000L, row.safety.checkedAtMs, "kept findings keep their time, so they expire")
        assertEquals(RiskScorer.LABEL_RED_FLAG, RiskScorer.assess(row.asset.tier, row.market, row.dex, row.safety, SPEC_NOW_MS)!!.label)
        val stale = opened.copy(safety = found.copy(checkedAtMs = SPEC_NOW_MS - 7 * 3_600_000L))
        assertEquals(false, svc.load(DEX_ONLY, bitpanda, emptyList(), 0.9, listOf(stale)).rows.single().safety!!.honeypot, "older than the checks' 6 h cache")
    }

    @Test
    fun tokensWithoutPoolDataAreLookedUpFirst() = runTest {
        val gt = List(130) { "GtMint$it" }
        val t = FakeTransport()
            .onUrlContaining("networks/trending_pools", body = gtPools("solana", *gt.toTypedArray()))
            .onUrlContaining("pump.fun/coins/currently-live", body = "[]")
            .onUrlContaining("pump.fun/coins?", body = spec("pumpfun_top.json"))
        service(t).load(DEX_ONLY.copy(dexIncludePumpFun = true), bitpanda, emptyList(), 0.9, emptyList())
        val looked = t.requests.filter { "/tokens/v1/solana/" in it.url }.flatMap { it.url.substringAfter("/tokens/v1/solana/").split(",") }
        val pump = PumpFunApi.parse(spec("pumpfun_top.json")).map { it.mint }
        assertEquals(pump, looked.take(pump.size), "pump.fun tokens have no pool data until DexScreener answers")
        assertEquals((pump + gt).toSet(), looked.toSet())
    }

    @Test
    fun keysAreCaseSensitiveExceptOnEvm() {
        assertEquals(SpeculativeService.key("base", "0xABC"), SpeculativeService.key("base", "0xabc"))
        assertFalse(SpeculativeService.key("solana", "AbC") == SpeculativeService.key("solana", "abc"))
    }

    companion object {
        /** 2026-10-19T12:00:00Z, just after the samples were captured. */
        const val SPEC_NOW_MS = 1_792_411_200_000L

        private val DEX_ONLY = AppConfig(showMemecoins = false, showTrending = false, dexIncludeNewPools = false, dexIncludePumpFun = false)
        private const val QUOTA_REPLY = """{"code":4029,"message":"too many requests","result":{}}"""

        /** GeckoTerminal trending pools for made-up tokens (TK0, TK1, ...): 80 days old, deep and busy, most traded first. */
        fun gtPools(network: String, vararg addresses: String): String {
            val data = addresses.mapIndexed { i, a ->
                """{"id":"${network}_pool$i","type":"pool","attributes":{"name":"TK$i / WETH","address":"pool$i","base_token_price_usd":"1.5",
                "reserve_in_usd":"500000","fdv_usd":"5000000","market_cap_usd":"5000000","pool_created_at":"2026-08-01T00:00:00Z",
                "price_change_percentage":{"h1":"1","h24":"5"},"volume_usd":{"h1":"10000","h24":"${300_000 - i}"},
                "transactions":{"h24":{"buys":900,"sells":800,"buyers":300,"sellers":250}}},
                "relationships":{"base_token":{"data":{"id":"${network}_$a","type":"token"}},"network":{"data":{"id":"$network","type":"network"}},
                "dex":{"data":{"id":"uniswap_v2","type":"dex"}}}}"""
            }
            val tokens = addresses.mapIndexed { i, a -> """{"id":"${network}_$a","type":"token","attributes":{"address":"$a","symbol":"TK$i","name":"Token $i"}}""" }
            return """{"data":[${data.joinToString(",")}],"included":[${tokens.joinToString(",")}]}"""
        }

        /** A clean, renounced GoPlus EVM answer. */
        fun goPlusEvm(address: String): String =
            """{"code":1,"message":"OK","result":{"$address":{"is_open_source":"1","is_honeypot":"0","buy_tax":"0","sell_tax":"0",
            "owner_address":"0x0000000000000000000000000000000000000000","holder_count":"5000"}}}"""
    }
}
