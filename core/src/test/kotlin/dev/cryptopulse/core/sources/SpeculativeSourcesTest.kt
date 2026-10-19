package dev.cryptopulse.core.sources

import dev.cryptopulse.core.FIXTURE_NOW_MS
import dev.cryptopulse.core.sources.cex.CexDataApi
import dev.cryptopulse.core.sources.coingecko.CoinGeckoApi
import dev.cryptopulse.core.sources.dex.DexScreenerApi
import dev.cryptopulse.core.sources.dex.GeckoTerminalApi
import dev.cryptopulse.core.sources.dex.PumpFunApi
import dev.cryptopulse.core.sources.paprika.CoinpaprikaApi
import dev.cryptopulse.core.sources.safety.TokenSafetyApi
import dev.cryptopulse.core.testHttp
import dev.cryptopulse.core.testing.FakeTransport
import dev.cryptopulse.core.testing.Fixtures
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.get
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Parsers for the memecoin / DEX sources, against payloads captured live (tools/api_samples). */
class SpeculativeSourcesTest {
    private fun spec(name: String) = Fixtures.read("spec/$name")

    @Test
    fun dexScreenerPairsKeepTheMostLiquidPool() {
        val wif = DexScreenerApi.parsePairs(JsonUtil.parse(spec("dexscreener_tokens_wif.json")).arr!!).single()
        assertEquals("WIF", wif.baseSymbol, "the \$ prefix is dropped")
        assertEquals("solana", wif.info.chainId)
        assertEquals(0.1978, wif.info.priceUsd)
        assertEquals(151012.95, wif.info.liquidityUsd)
        assertEquals(78, wif.info.txnsH1.buys)
        assertEquals(87, wif.info.txnsH1.sells)
        assertEquals(1716065147000, wif.info.pairCreatedAtMs)
        assertTrue(wif.info.socials.containsKey("twitter"))
        val boosted = DexScreenerApi.parsePairs(JsonUtil.parse(spec("dexscreener_tokens_boosted.json")).arr!!).single()
        assertEquals(500, boosted.info.boostsActive)
        val search = DexScreenerApi.parsePairs(JsonUtil.parse(spec("dexscreener_search.json"))["pairs"].arr!!)
        val deepest = DexScreenerApi.mostLiquidPerToken(search + search.map { it.copy(info = it.info.copy(liquidityUsd = 1.0)) })
        assertEquals(search.size, deepest.size)
        assertTrue(deepest.none { it.info.liquidityUsd == 1.0 })
    }

    @Test
    fun dexScreenerPromotionsAreMergedPerToken() = runTest {
        val t = FakeTransport()
            .onUrlContaining("/token-boosts/top/v1", body = spec("dexscreener_boosts_top.json"))
            .onUrlContaining("/token-boosts/latest/v1", body = spec("dexscreener_boosts_latest.json"))
            .onUrlContaining("/token-profiles/latest/v1", body = spec("dexscreener_profiles_latest.json"))
            .onUrlContaining("/community-takeovers/latest/v1", body = spec("dexscreener_takeovers.json"))
        val promos = DexScreenerApi(testHttp(t)).promoted().valueOrNull!!
        assertEquals(16, promos.size)
        assertEquals(500.0, promos.first { it.tokenAddress.startsWith("6yMZv1") }.boostAmount)
        assertEquals(setOf("ds_boost", "ds_profile", "ds_takeover"), promos.map { it.kind }.toSet())
    }

    @Test
    fun dexScreenerTokenLookupsAreBatchedByThirty() = runTest {
        val t = FakeTransport().onUrlContaining("/tokens/v1/solana/", body = spec("dexscreener_tokens_wif.json"))
        val r = DexScreenerApi(testHttp(t)).tokens("solana", List(31) { "addr$it" })
        assertIs<SourceResult.Success<*>>(r)
        assertEquals(2, t.countRequests("/tokens/v1/solana/"))
    }

    @Test
    fun geckoTerminalPoolsCarryTheirBaseToken() {
        val pools = GeckoTerminalApi.parsePools(spec("geckoterminal_trending.json"), "gt_trending")
        assertEquals(4, pools.size)
        val torch = pools.first()
        assertEquals("TORCH", torch.baseSymbol)
        assertEquals("Torch", torch.baseName)
        assertEquals("torch", torch.baseCoingeckoId)
        assertEquals("robinhood", torch.info.chainId)
        assertEquals(1096271.3927, torch.info.liquidityUsd)
        assertEquals(72.071, torch.info.priceChangePct.h24)
        assertEquals(5077, torch.info.txnsH24.buys)
        assertEquals(1178, torch.info.txnsH24.buyers)
        assertEquals(listOf("gt_trending"), torch.info.foundBy)
        assertTrue(torch.info.url!!.startsWith("https://www.geckoterminal.com/robinhood/pools/"))
        assertEquals("ethereum", GeckoTerminalApi.dexScreenerChain("eth"))
        assertEquals("eth", GeckoTerminalApi.geckoTerminalNetwork("ethereum"))
        assertEquals(4, GeckoTerminalApi.parsePools(spec("geckoterminal_new_pools.json"), "gt_new").size)
    }

    @Test
    fun geckoTerminalCandlesAreOldestFirstInMillis() {
        val candles = GeckoTerminalApi.parseCandles(spec("geckoterminal_ohlcv.json"))
        assertEquals(48, candles.size)
        assertTrue(candles.zipWithNext().all { (a, b) -> a.timeMs < b.timeMs })
        assertTrue(candles.last().timeMs == 1792407600L * 1000, "the newest candle is first in the payload")
        assertEquals(0.00066116723109505, candles.last().close)
    }

    @Test
    fun pumpFunTokensKnowWhetherTheyGraduated() {
        val coins = PumpFunApi.parse(spec("pumpfun_top.json"))
        assertEquals(listOf("UDR", "XTO", "EVERNORTHX"), coins.map { it.symbol })
        assertTrue(coins.all { it.graduated })
        assertEquals(722133809.7903008, coins.first().usdMarketCap)
        assertTrue(PumpFunApi.parse(spec("pumpfun_live.json")).isNotEmpty())
        assertTrue(PumpFunApi.parse("""[{"mint":"x","symbol":"BAD","is_banned":true},{"mint":"y","symbol":"NSFW","nsfw":true}]""").isEmpty())
    }

    @Test
    fun goPlusIgnoresOwnerSwitchesOnceOwnershipIsRenounced() {
        val pepe = TokenSafetyApi.parseGoPlusEvm(spec("goplus_eth.json"), FIXTURE_NOW_MS).values.single()
        assertEquals(false, pepe.honeypot)
        assertEquals(0.0, pepe.buyTaxPct)
        assertEquals(true, pepe.ownerRenounced)
        assertTrue(pepe.ownerPowers.isEmpty(), "PEPE's blacklist/pause switches are unusable: ${pepe.ownerPowers}")
        assertEquals(false, pepe.mintable)
        assertEquals(listOf("Binance", "Coinbase"), pepe.cexListed)
        assertTrue(pepe.holderCount!! > 500_000)
        assertTrue(pepe.top10HolderPct!! in 1.0..100.0)
        val wif = TokenSafetyApi.parseGoPlusSolana(spec("goplus_sol.json"), FIXTURE_NOW_MS).values.single()
        assertEquals(false, wif.mintable)
        assertEquals(false, wif.freezable)
        assertTrue(wif.ownerPowers.isEmpty())
        assertEquals(99.75, wif.lpLockedPct)
        assertTrue(wif.top10HolderPct!! in 13.0..100.0, "fractions become percent")
    }

    @Test
    fun goPlusFlagsOwnerPowersOfAnOwnedContract() {
        val body = """{"code":1,"result":{"0xabc":{"owner_address":"0x1111111111111111111111111111111111111111","is_honeypot":"0","buy_tax":"0.05","sell_tax":"0.25",
            "is_mintable":"1","transfer_pausable":"1","is_blacklisted":"1","owner_change_balance":"1","holders":[{"percent":"0.6"},{"percent":"0.2"}]}}}"""
        val s = TokenSafetyApi.parseGoPlusEvm(body, 0).getValue("0xabc")
        assertEquals(true, s.mintable)
        assertEquals(25.0, s.sellTaxPct!!, 1e-9)
        assertEquals(80.0, s.top10HolderPct!!, 1e-9)
        assertTrue(listOf("owner can change balances", "transfers can be paused", "owner can blacklist wallets").all { it in s.ownerPowers })
    }

    @Test
    fun honeypotIsSimulation() {
        val s = TokenSafetyApi.parseHoneypotIs(spec("honeypot_is.json"), FIXTURE_NOW_MS)
        assertEquals(false, s.honeypot)
        assertEquals(0.0, s.sellTaxPct)
        assertEquals("honeypot.is", s.source)
    }

    @Test
    fun safetyChecksOnlyForChainsTheyCover() = runTest {
        val api = TokenSafetyApi(testHttp(FakeTransport()))
        assertFalse(api.check("ton", "x").isSuccess)
        assertFalse(api.supports("ton"))
        assertTrue(api.supports("solana") && api.supports("base"))
        assertFalse(api.simulate("solana", "x").isSuccess)
    }

    @Test
    fun coinpaprikaTickersAreAFullMarketFallback() {
        val t = CoinpaprikaApi.parseTickers(spec("coinpaprika_tickers.json"), "EUR")
        val btc = t.first { it.id == "btc-bitcoin" }
        assertEquals(1, btc.rank)
        assertEquals("EUR", btc.market.currency)
        assertEquals(73436.31385838863, btc.market.price)
        assertEquals("coinpaprika", btc.market.source)
        assertEquals(2, t.count { it.symbol == "DOGE" }, "symbol collisions are kept; callers pick by rank")
    }

    @Test
    fun binanceMirrorKlinesAndOkxFunding() {
        val k = CexDataApi.parseKlines(spec("binance_vision_klines.json"))
        assertEquals(3, k.size)
        assertEquals(0.0000041, k.first().close)
        val f = CexDataApi.parseOkxFunding(spec("okx_funding.json"), "PEPE", FIXTURE_NOW_MS)!!
        assertEquals("okx", f.source)
        assertEquals(0.0001, f.fundingRate!!, 1e-12, "8 h settlement stays as is")
    }

    @Test
    fun coinGeckoMemeCategoryUsesTheMarketsParser() = runTest {
        val t = FakeTransport().onUrlContaining("category=meme-token", body = spec("coingecko_meme_markets.json"))
        val r = CoinGeckoApi(testHttp(t)).markets("eur", page = 1, perPage = 250, ids = null, sparkline = true, category = "meme-token").valueOrNull!!
        assertEquals(listOf("DOGE", "SHIB", "PUMP"), r.map { it.symbol })
        assertTrue(r.all { it.sparkline7d.size == 168 })
    }

    @Test
    fun okxUnknownInstrumentIsNotAnError() = runTest {
        val t = FakeTransport().onUrlContaining("okx.com", body = """{"code":"51001","data":[],"msg":"Instrument ID does not exist"}""")
        val r = CexDataApi(testHttp(t)).okxFunding("NOPE")
        assertNull(r.valueOrNull)
        assertTrue(r.isSuccess)
    }
}
