package dev.cryptopulse.core.analysis.forecast

import dev.cryptopulse.core.analysis.ForecastInput
import dev.cryptopulse.core.model.DexInfo
import dev.cryptopulse.core.model.Horizon
import dev.cryptopulse.core.model.Tier
import kotlin.math.ln
import kotlin.math.sign
import kotlin.math.tanh

/**
 * Extra signals for assets that are not on Bitpanda.
 *
 *  - Base rate: most DEX-only tokens lose most of their value within weeks, and coins that trend
 *    today usually cool off, so unproven tokens start from a pessimistic prior instead of 50/50.
 *  - Trading flow (DEX pools): short-term momentum, give-back after very large pumps, buyer vs
 *    seller pressure and volume acceleration. Weights are small: these signals are noisy.
 */
internal object Speculative {

    fun signals(input: ForecastInput, horizon: Horizon): List<Contribution> {
        val tier = input.asset.tier
        if (tier == Tier.BITPANDA) return emptyList()
        val out = ArrayList<Contribution>()
        baseRate(tier, input.dex, input.nowMs, horizon)?.let { out += it }
        input.dex?.let { out += flow(it, horizon) }
        return out
    }

    private fun baseRate(tier: Tier, dex: DexInfo?, nowMs: Long, horizon: Horizon): Contribution? {
        val ageH = dex?.pairCreatedAtMs?.let { (nowMs - it) / 3_600_000.0 }
        val young = ageH != null && ageH < 72
        val logit = when (tier) {
            Tier.DEX -> (if (horizon == Horizon.H24) -0.10 else -0.28) + (if (young) (if (horizon == Horizon.H24) -0.08 else -0.15) else 0.0)
            Tier.TRENDING -> if (horizon == Horizon.H24) -0.03 else -0.10
            Tier.MEME -> if (horizon == Horizon.H24) 0.0 else -0.04
            Tier.BITPANDA -> 0.0
        }
        if (logit == 0.0) return null
        val text = when {
            tier == Tier.DEX && young -> "Brand-new DEX tokens usually fall: most lose most of their value within days"
            tier == Tier.DEX -> "Most DEX-only tokens lose value over time"
            tier == Tier.TRENDING -> "Coins that trend today often cool off once attention moves on"
            else -> "Memecoins outside the largest ones tend to drift down between hype waves"
        }
        return Contribution("base_rate", "Base rate for this kind of token", null, logit, text)
    }

    private fun flow(dex: DexInfo, horizon: Horizon): List<Contribution> {
        val out = ArrayList<Contribution>()
        val h24 = horizon == Horizon.H24
        dex.priceChangePct.h1?.let { m ->
            val c = tanh(m / 15.0) * (if (h24) 0.10 else 0.04)
            out += Contribution("dex_momentum_1h", "Last hour on the DEX", m, c, "Price moved ${"%.1f".format(m)}% in the last hour")
        }
        dex.priceChangePct.h24?.let { m ->
            val c = -tanh(m / 150.0) * (if (m > 0) (if (h24) 0.15 else 0.22) else (if (h24) 0.05 else 0.04))
            val text = when {
                m >= 100 -> "Up ${"%.0f".format(m)}% in 24 h: pumps this large usually give a big part back"
                m <= -50 -> "Down ${"%.0f".format(-m)}% in 24 h: some bounce is possible, many keep falling"
                else -> "24 h move ${"%.1f".format(m)}%"
            }
            out += Contribution("dex_move_24h", "24 h move on the DEX", m, c, text)
        }
        pressure(dex)?.let { (ratio, wallets) ->
            val c = tanh((ratio - 0.5) * 5) * (if (h24) 0.12 else 0.06)
            val who = if (wallets) "wallets buying vs selling" else "buy vs sell trades"
            out += Contribution("dex_buy_pressure", "Buyers vs sellers", ratio * 100, c, "${"%.0f".format(ratio * 100)}% of $who in the last hours")
        }
        val v1 = dex.volumeUsd.h1
        val v24 = dex.volumeUsd.h24
        val m1 = dex.priceChangePct.h1
        if (v1 != null && v24 != null && v24 > 0 && m1 != null && m1 != 0.0) {
            val accel = v1 / (v24 / 24.0)
            if (accel > 0) {
                val c = tanh(ln(accel)) * sign(m1) * (if (h24) 0.06 else 0.03)
                val text = if (accel >= 2) "Volume this hour is ${"%.1f".format(accel)}x the daily average" else "Hourly volume is ${"%.1f".format(accel)}x the daily average"
                out += Contribution("dex_volume_trend", "Volume acceleration", accel, c, text)
            }
        }
        return out
    }

    /** Share of buyers (unique wallets when reported, else trades) over the last 6 hours, or 24 h. */
    private fun pressure(dex: DexInfo): Pair<Double, Boolean>? {
        for (t in listOf(dex.txnsH6, dex.txnsH24)) {
            val b = t.buyers
            val s = t.sellers
            if (b != null && s != null && b + s >= 20) return b.toDouble() / (b + s) to true
            if (t.buys + t.sells >= 30) return t.buys.toDouble() / (t.buys + t.sells) to false
        }
        return null
    }
}
