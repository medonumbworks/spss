package dev.cryptopulse.core.analysis.risk

import dev.cryptopulse.core.model.DexInfo
import dev.cryptopulse.core.model.DexTxns
import dev.cryptopulse.core.model.DexWindow
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.sources.dex.DexScreenerApi
import dev.cryptopulse.core.testing.Fixtures
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/** Each pump pattern of [PumpDetector] fires at its threshold and stays quiet just below it. */
class PumpDetectorTest {
    private val now = 1_792_411_200_000L
    private val hour = 3_600_000L

    /** BTC-like defaults: huge cap, modest turnover, flat price. */
    private fun market(change24h: Double? = 1.2, change1h: Double? = 0.1, cap: Double? = 1.2e12, vol: Double? = 3.5e10) =
        MarketData("BTC", currency = "EUR", price = 95_000.0, marketCap = cap, volume24h = vol, change1hPct = change1h, change24hPct = change24h)

    /** A healthy, old pool with ordinary flow. */
    private fun pool(volumeH24: Double? = 300_000.0, liquidity: Double? = 400_000.0, boosts: Int = 0, ageH: Long = 400 * 24, change24h: Double? = 4.0, change1h: Double? = 1.0) = DexInfo(
        chainId = "solana", tokenAddress = "addr", liquidityUsd = liquidity, marketCapUsd = 5_000_000.0,
        priceChangePct = DexWindow(h1 = change1h, h24 = change24h), volumeUsd = DexWindow(h24 = volumeH24),
        txnsH24 = DexTxns(900, 800), pairCreatedAtMs = now - ageH * hour, boostsActive = boosts,
    )

    @Test
    fun priceSpikesFireAtFiftyPerDayOrFifteenPerHour() {
        assertEquals(listOf("price +62% in 24 h"), PumpDetector.assess(market(change24h = 61.7), null, nowMs = now)!!.reasons)
        assertEquals(listOf("price +15% in the last hour"), PumpDetector.assess(market(change1h = 15.0), null, nowMs = now)!!.reasons)
        // 49.9 would round up to 50, but the threshold compares the raw value
        assertNull(PumpDetector.assess(market(change24h = 49.9, change1h = 14.9), null, nowMs = now))
        // a DEX-only token has no CoinGecko row: the pool's own price windows count instead
        assertEquals(listOf("price +80% in 24 h"), PumpDetector.assess(null, pool(change24h = 80.0), nowMs = now)!!.reasons)
    }

    @Test
    fun volumeFiftyTimesThePoolIsWashTrading() {
        val r = PumpDetector.assess(null, pool(volumeH24 = 20_000_000.0), nowMs = now)!!
        assertEquals(listOf("24 h volume is 50x the pool, a wash-trading pattern"), r.reasons)
        assertNull(PumpDetector.assess(null, pool(volumeH24 = 19_900_000.0), nowMs = now))
        // an empty pool divides by zero, not a pattern
        assertNull(PumpDetector.assess(null, pool(volumeH24 = 20_000_000.0, liquidity = 0.0), nowMs = now))
    }

    @Test
    fun capTradedThreeTimesOverFiresOnlyOffDex() {
        val churning = market(cap = 1e8, vol = 3.2e8)
        assertEquals(listOf("the whole market cap traded 3x over in 24 h"), PumpDetector.assess(churning, null, nowMs = now)!!.reasons)
        assertNull(PumpDetector.assess(market(cap = 1e8, vol = 2.9e8), null, nowMs = now))
        // with a pool attached, the pool's volume/liquidity check covers churn instead
        assertNull(PumpDetector.assess(churning, pool(), nowMs = now))
    }

    @Test
    fun mentionSpikesCountOnlyWithEnoughMentionsAndAPumpingPrice() {
        val pumping = market(change24h = 20.0)
        val r = PumpDetector.assess(pumping, null, mentions = 10, mentionSpike = 5.0, nowMs = now)!!
        assertEquals(listOf("Telegram mentions are 5x their 7-day norm while the price pumps"), r.reasons)
        assertNull(PumpDetector.assess(pumping, null, mentions = 10, mentionSpike = 4.9, nowMs = now))
        assertNull(PumpDetector.assess(pumping, null, mentions = 9, mentionSpike = 5.0, nowMs = now))
        assertNull(PumpDetector.assess(market(change24h = 19.9), null, mentions = 10, mentionSpike = 5.0, nowMs = now))
        // an unknown mention count never fires, however big the spike ratio
        assertNull(PumpDetector.assess(market(change24h = 25.0), null, mentionSpike = 9.0, nowMs = now))
    }

    @Test
    fun heavyBoostsOnADaysOldPoolArePaidPromotion() {
        val r = PumpDetector.assess(null, pool(boosts = 100, ageH = 71), nowMs = now)!!
        assertEquals(listOf("heavily promoted while days old"), r.reasons)
        assertNull(PumpDetector.assess(null, pool(boosts = 99, ageH = 71), nowMs = now))
        assertNull(PumpDetector.assess(null, pool(boosts = 500, ageH = 73), nowMs = now))
        // age unknown: stay quiet rather than guess
        assertNull(PumpDetector.assess(null, pool(boosts = 500).copy(pairCreatedAtMs = null), nowMs = now))
    }

    @Test
    fun reasonsAccumulateWhenSeveralPatternsFire() {
        val r = PumpDetector.assess(
            market(change24h = 55.0, cap = null, vol = null),
            pool(volumeH24 = 24_000_000.0, boosts = 150, ageH = 48),
            mentions = 40, mentionSpike = 8.3, nowMs = now,
        )!!
        assertEquals(
            listOf(
                "price +55% in 24 h",
                "24 h volume is 60x the pool, a wash-trading pattern",
                "Telegram mentions are 8x their 7-day norm while the price pumps",
                "heavily promoted while days old",
            ),
            r.reasons,
        )
    }

    @Test
    fun quietDataRaisesNoAlert() {
        assertNull(PumpDetector.assess(market(), null, nowMs = now))
        assertNull(PumpDetector.assess(market(), pool(), mentions = 3, mentionSpike = 1.1, nowMs = now))
        assertNull(PumpDetector.assess(null, null, mentions = 50, mentionSpike = 10.0, nowMs = now))
        // WIF's captured pair: +0.49% on the day, 24 h volume below its pool size, a year old
        val wif = DexScreenerApi.parsePairs(JsonUtil.parse(Fixtures.read("spec/dexscreener_tokens_wif.json")).arr!!).single().info
        assertNull(PumpDetector.assess(null, wif, nowMs = now))
    }
}
