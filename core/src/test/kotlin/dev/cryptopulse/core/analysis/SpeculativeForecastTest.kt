package dev.cryptopulse.core.analysis

import dev.cryptopulse.core.analysis.forecast.DefaultForecastEngine
import dev.cryptopulse.core.analysis.risk.RiskScorer
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.DexInfo
import dev.cryptopulse.core.model.DexTxns
import dev.cryptopulse.core.model.DexWindow
import dev.cryptopulse.core.model.Direction
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.model.TokenSafety
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SpeculativeForecastTest {
    private val now = 1_792_411_200_000L
    private val day = 86_400_000L
    private val engine = DefaultForecastEngine()

    private fun dex(ageDays: Double, liquidity: Double = 400_000.0, h24: Double = 12.0, buys: Int = 900, sells: Int = 700) = DexInfo(
        chainId = "solana", tokenAddress = "Mint111", pairAddress = "Pool111", liquidityUsd = liquidity, marketCapUsd = 8_000_000.0,
        priceChangePct = DexWindow(h1 = 2.0, h6 = 5.0, h24 = h24), volumeUsd = DexWindow(h1 = 40_000.0, h24 = 600_000.0),
        txnsH6 = DexTxns(buys, sells), txnsH24 = DexTxns(buys * 3, sells * 3), pairCreatedAtMs = now - (ageDays * day).toLong(),
        websites = listOf("https://example.invalid"), socials = mapOf("telegram" to "https://t.me/example_token"),
    )

    private val clean = TokenSafety(source = "goplus", honeypot = false, buyTaxPct = 0.0, sellTaxPct = 0.0, mintable = false, freezable = false, openSource = true)
    private val dexAsset = Asset("MOON", "Moon Token", tier = Tier.DEX, chainId = "solana", contract = "Mint111")
    private val meme = Asset("SHIB", "Shiba Inu", "shiba-inu", tier = Tier.MEME)

    private fun hourly(n: Int, start: Double = 1.0) = (0 until n).map { i -> start * exp(0.001 * i + 0.01 * sin(i / 3.0)) }
    private fun sigmoid(x: Double) = 1.0 / (1.0 + exp(-x))

    private fun input(asset: Asset, d: DexInfo?, safety: TokenSafety?, sparkline: List<Double> = emptyList()): ForecastInput {
        val market = MarketData(asset.symbol, currency = "EUR", price = sparkline.lastOrNull() ?: 0.01, marketCap = 8e6, change24hPct = d?.priceChangePct?.h24, sparkline7d = sparkline)
        return ForecastInput(asset, market, sparkline, dex = d, risk = RiskScorer.assess(asset.tier, market, d, safety, now), nowMs = now)
    }

    @Test
    fun bitpandaAssetsHaveNoRiskScore() {
        assertNull(RiskScorer.assess(Tier.BITPANDA, null, null, null, now))
    }

    @Test
    fun hardBlocks() {
        fun r(d: DexInfo?, s: TokenSafety?) = RiskScorer.assess(Tier.DEX, null, d, s, now)!!
        r(dex(40.0), clean.copy(honeypot = true)).let { assertTrue(it.blocked); assertEquals(100, it.score); assertEquals(RiskScorer.LABEL_RED_FLAG, it.label) }
        r(dex(40.0), clean.copy(sellTaxPct = 25.0)).let { assertEquals(RiskScorer.LABEL_RED_FLAG, it.label) }
        r(dex(40.0), clean.copy(freezable = true)).let { assertEquals(RiskScorer.LABEL_RED_FLAG, it.label) }
        r(dex(40.0), clean.copy(ownerPowers = listOf("someone can change balances"))).let { assertEquals(RiskScorer.LABEL_RED_FLAG, it.label) }
        r(dex(0.3), clean).let { assertEquals(RiskScorer.LABEL_TOO_NEW, it.label); assertTrue(it.blocked) }
        r(dex(40.0).copy(pumpFunGraduated = false), clean).let { assertEquals(RiskScorer.LABEL_TOO_NEW, it.label) }
        r(dex(40.0, liquidity = 6_000.0), clean).let { assertEquals(RiskScorer.LABEL_ILLIQUID, it.label) }
        r(dex(40.0), null).let { assertEquals(RiskScorer.LABEL_UNVERIFIED, it.label); assertTrue(it.blocked) }
        // a mint authority is fine on a coin that big exchanges list (USDC-type tokens)
        assertFalse(r(dex(400.0, liquidity = 5e6), clean.copy(mintable = true, cexListed = listOf("Binance"))).blocked)
        // ...but nothing excuses changing balances
        assertTrue(r(dex(400.0, liquidity = 5e6), clean.copy(cexListed = listOf("Binance"), ownerPowers = listOf("owner can change balances"))).blocked)
    }

    @Test
    fun establishedCleanTokensScoreLowerThanFreshThinOnes() {
        val old = RiskScorer.assess(Tier.DEX, null, dex(300.0, liquidity = 3_000_000.0), clean.copy(cexListed = listOf("Bybit")), now)!!
        val young = RiskScorer.assess(Tier.DEX, null, dex(2.0, liquidity = 30_000.0), clean, now)!!
        assertTrue(old.score < 25, "old ${old.score} ${old.reasons}")
        assertFalse(old.blocked)
        assertTrue(young.score > old.score + 30, "young ${young.score} ${young.reasons}")
        val bigMeme = RiskScorer.assess(Tier.MEME, MarketData("SHIB", currency = "EUR", marketCap = 6e9, volume24h = 2e8), null, null, now)!!
        assertFalse(bigMeme.blocked, "CoinGecko-listed memecoins need no contract check")
        assertEquals(RiskScorer.LABEL_LOWER, bigMeme.label)
    }

    @Test
    fun blockedTokensGetNoUpProbability() {
        val f = engine.forecast(input(dexAsset, dex(40.0), clean.copy(honeypot = true)))
        for (h in listOf(f.h24, f.d7)) {
            assertEquals(0.5, h.probabilityUp)
            assertEquals(Direction.FLAT, h.direction)
            assertEquals(0.0, h.confidence)
            assertEquals("risk_blocked", h.signals.single().name)
        }
        assertEquals(DefaultForecastEngine.NOT_ASSESSED, f.bottom.label)
        assertNull(f.bottom.reboundProbability7d)
        assertTrue(f.summary.contains("No forecast"))
        assertFalse(f.summary.contains("% up"), f.summary)
    }

    @Test
    fun youngPoolsGetNoNumberUntilTheyHaveEnoughHistory() {
        val twoDays = engine.forecast(input(dexAsset, dex(2.0), clean, hourly(48)))
        assertEquals(0.5, twoDays.h24.probabilityUp, "24 h needs 3 days of trading")
        assertEquals("history", twoDays.h24.signals.first().name)
        assertEquals(0.5, twoDays.d7.probabilityUp)
        assertTrue(twoDays.summary.startsWith("Moon Token: not enough history for 24h, not enough history for 7 days."), twoDays.summary)
        assertFalse(twoDays.summary.contains("% up"), "the 0.5 placeholder is not a forecast: ${twoDays.summary}")
        assertNull(twoDays.bottom.reboundProbability7d)
        val tenDays = engine.forecast(input(dexAsset, dex(10.0), clean, hourly(168)))
        assertTrue(tenDays.h24.signals.none { it.name == "history" })
        assertEquals("history", tenDays.d7.signals.first().name, "7 d needs 3 weeks")
        assertTrue(tenDays.summary.contains(" for 24h (${(tenDays.h24.probabilityUp * 100).toInt()}% up), not enough history for 7 days."), tenDays.summary)
        assertNull(tenDays.bottom.reboundProbability7d)
        val month = engine.forecast(input(dexAsset, dex(40.0), clean, hourly(168)))
        assertTrue(month.d7.signals.none { it.name == "history" })
        assertTrue(month.d7.signals.any { it.name == "base_rate" && it.contribution < 0 }, "DEX tokens start from a pessimistic prior")
        assertTrue(month.h24.signals.any { it.name == "dex_buy_pressure" })
        assertEquals(month.d7.probabilityUp, month.bottom.reboundProbability7d)
    }

    @Test
    fun withOnlyItsBaseRateADexTokenGetsExactlyThePrior() {
        // no candles and no trading flow: coverage is minimal, but shrinking may not soften the prior toward 50/50
        val quiet = dex(40.0).copy(priceChangePct = DexWindow(), volumeUsd = DexWindow(), txnsH6 = DexTxns(), txnsH24 = DexTxns())
        val f = engine.forecast(input(dexAsset, quiet, clean))
        assertTrue(f.h24.hasCall && f.d7.hasCall, "${f.h24.signals}")
        assertEquals(sigmoid(-0.10), f.h24.probabilityUp, 1e-4)
        assertEquals(sigmoid(-0.28), f.d7.probabilityUp, 1e-4)
    }

    @Test
    fun riskPullsTheProbabilityTowardThePriorNotACoinFlip() {
        val series = (0 until 168).map { i -> 1.0 * exp(0.004 * i) } // strong uptrend
        val lowRisk = engine.forecast(input(meme, null, null, series).copy(risk = null))
        val withRisk = engine.forecast(input(meme, null, null, series))
        assertTrue(abs(withRisk.h24.probabilityUp - 0.5) < abs(lowRisk.h24.probabilityUp - 0.5) + 1e-9, "memecoins have no 24 h base rate: the prior is 50%")
        assertTrue(withRisk.summary.endsWith("not on Bitpanda."), withRisk.summary)

        // the same 40-day DEX pool, only the risk score differs: the 7-day model is bearish on the
        // drifting series and bullish on the falling one (a rebound). Risk only discounts the upside:
        // p never rises with risk, a bullish reading converges on the prior, a bearish one stays put
        val prior = sigmoid(-0.28)
        val falling = (0 until 168).map { i -> exp(-0.003 * i + 0.01 * sin(i / 3.0)) }
        for (series in listOf(hourly(168), falling)) {
            val base = input(dexAsset, dex(40.0), clean, series)
            val p = listOf(0, 40, 60, 74).map { s -> engine.forecast(base.copy(risk = base.risk!!.copy(score = s))).d7.probabilityUp }
            assertTrue(p.zipWithNext().all { (a, b) -> b <= a + 1e-9 }, "more risk never raises the chance of going up: $p")
            if (p.first() > prior) assertEquals(prior, p.last(), 0.005) else assertTrue(p.all { abs(it - p.first()) < 1e-9 }, "bearish readings are not shrunk: $p")
        }
        val bearish = input(dexAsset, dex(40.0), clean, hourly(168))
        val safe = engine.forecast(bearish.copy(risk = bearish.risk!!.copy(score = 0))).d7
        val risky = engine.forecast(bearish.copy(risk = bearish.risk!!.copy(score = 74))).d7
        assertTrue(safe.probabilityUp < prior, "the model is bearish here: ${safe.probabilityUp}")
        assertTrue(risky.probabilityUp <= safe.probabilityUp + 1e-9, "risk never lifts a bearish token: ${risky.probabilityUp}")
        assertEquals(safe.direction, risky.direction, "a very risky DEX token reads as bearish as a safe one")
    }

    @Test
    fun hugePumpsLeanTowardGivingBack() {
        val calm = engine.forecast(input(dexAsset, dex(40.0, h24 = 5.0), clean, hourly(168)))
        val pumped = engine.forecast(input(dexAsset, dex(40.0, h24 = 400.0), clean, hourly(168)))
        val c = pumped.h24.signals.first { it.name == "dex_move_24h" }.contribution
        assertTrue(c < 0, "a 400% day weighs toward down")
        assertTrue(pumped.h24.probabilityUp <= calm.h24.probabilityUp)
    }

    @Test
    fun polymarketCrowdIsA24hSignalOnly() {
        val base = input(meme, null, null, hourly(168))
        val with = engine.forecast(base.copy(polymarketUp24 = 0.65))
        val without = engine.forecast(base)
        val sig = with.h24.signals.first { it.name == "polymarket" }
        assertEquals(0.35 * ln(0.65 / 0.35), sig.contribution, 1e-3, "a third of the crowd's own conviction")
        assertTrue(with.h24.probabilityUp > without.h24.probabilityUp, "a bullish crowd nudges the day up")
        assertTrue(with.d7.signals.none { it.name == "polymarket" }, "a daily market says nothing about 7 days")
        val capped = engine.forecast(base.copy(polymarketUp24 = 0.99)).h24.signals.first { it.name == "polymarket" }
        assertEquals(0.40, capped.contribution, 1e-9, "extreme crowd odds are capped")
    }
}
