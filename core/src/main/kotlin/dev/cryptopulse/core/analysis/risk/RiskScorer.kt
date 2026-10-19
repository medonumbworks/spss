package dev.cryptopulse.core.analysis.risk

import dev.cryptopulse.core.model.DexInfo
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.RiskAssessment
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.model.TokenSafety
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Explainable 0-100 risk for assets that are not on Bitpanda (higher = riskier). It is not a scam
 * detector: a low score only means none of the usual red flags were found in free data.
 *
 * Calibrated on published base rates (docs/SPECULATIVE.md): about 69% of pump.fun launches stop
 * trading on launch day, about 80% of new tokens are dead within two days, and over 80% of memecoins
 * that more than doubled showed manufactured volume. Hence:
 *  - hard blocks (no "up" probability at all): honeypot, sell/buy tax >= 20%, someone can change
 *    balances or freeze holders, live mint authority or owner backdoors (unless CEX-listed/trusted),
 *    pool liquidity < $10k, pool younger than 24 h, still on the pump.fun bonding curve, DEX
 *    tokens whose contract (or, on EVM chains, selling) could not be checked, and trending coins
 *    without a market-cap rank or under $10M, whose contracts nobody checks (fail closed);
 *  - score >= 75 also blocks the probability ("extreme risk");
 *  - below that, the forecast is shrunk toward 50% in proportion to the score ([shrink]).
 */
object RiskScorer {
    const val LABEL_LOWER = "Lower risk"
    const val LABEL_ELEVATED = "Elevated risk"
    const val LABEL_HIGH = "High risk"
    const val LABEL_EXTREME = "Extreme risk"
    const val LABEL_RED_FLAG = "Red flags"
    const val LABEL_TOO_NEW = "Too new"
    const val LABEL_ILLIQUID = "Too illiquid"
    const val LABEL_UNVERIFIED = "Unverified"

    const val MIN_LIQUIDITY_USD = 10_000.0
    const val MIN_UNCHECKED_TRENDING_CAP = 10_000_000.0
    const val PROHIBITIVE_TAX_PCT = 20.0
    const val BLOCK_SCORE = 75

    /** Multiplier for (p - 0.5): 1 at score 0, 0 at [BLOCK_SCORE]. */
    fun shrink(risk: RiskAssessment?): Double = if (risk == null) 1.0 else (1.0 - risk.score / BLOCK_SCORE.toDouble()).coerceIn(0.0, 1.0)

    fun assess(tier: Tier, market: MarketData?, dex: DexInfo?, safety: TokenSafety?, nowMs: Long): RiskAssessment? {
        if (tier == Tier.BITPANDA) return null
        val reasons = ArrayList<String>()
        val redFlags = ArrayList<String>()
        var blockLabel: String? = null
        fun block(label: String, reason: String) {
            if (blockLabel == null || label == LABEL_RED_FLAG) blockLabel = label
            redFlags += reason
        }
        var score = when (tier) {
            Tier.DEX -> 30.0.also { reasons += "Only on decentralised exchanges, not on Bitpanda" }
            Tier.TRENDING -> 22.0.also { reasons += "Trending today: attention can fade quickly" }
            else -> 18.0.also { reasons += "Memecoin: price driven by attention, not by revenue" }
        }
        val exemptByListing = safety?.trusted == true || safety?.cexListed?.isNotEmpty() == true
        // CoinGecko reports 0 when it lacks the circulating supply: unknown, not a micro cap
        val mcap = market?.marketCap?.takeIf { it > 0 } ?: dex?.marketCapUsd?.takeIf { it > 0 } ?: dex?.fdvUsd?.takeIf { it > 0 }

        // size: small caps move (and die) fastest
        when {
            mcap == null -> { score += 8; reasons += "Market cap unknown" }
            mcap >= 1e9 -> score -= 10
            mcap >= 1e8 -> score -= 4
            mcap >= 1e7 -> score += 4
            mcap >= 1e6 -> { score += 10; reasons += "Tiny market cap (${money(mcap)})" }
            else -> { score += 16; reasons += "Micro cap (${money(mcap)})" }
        }
        val move = market?.change24hPct ?: dex?.priceChangePct?.h24
        if (move != null && move >= 300) { score += 10; reasons += "Up ${pct(move)} in 24 h" }
        else if (move != null && abs(move) >= 50) { score += 5; reasons += "Moved ${pct(move)} in 24 h" }
        market?.athChangePct?.let { if (it <= -90) { score += 4; reasons += "${pct(-it)} below its all-time high" } }
        if (tier != Tier.DEX) {
            val vol = market?.volume24h
            if (mcap != null && vol != null && mcap > 0 && vol / mcap < 0.005) { score += 5; reasons += "Very little trading (${pct(vol / mcap * 100)} of market cap per day)" }
        }

        val ageH = dex?.pairCreatedAtMs?.let { (nowMs - it) / 3_600_000.0 }
        if (dex != null) {
            val liq = dex.liquidityUsd
            when {
                liq == null -> { score += 15; reasons += "Pool liquidity unknown" }
                liq < MIN_LIQUIDITY_USD -> block(LABEL_ILLIQUID, "Pool liquidity only ${money(liq)}: prices are not meaningful and selling moves them")
                liq < 25_000 -> { score += 30; reasons += "Very thin pool (${money(liq)})" }
                liq < 50_000 -> { score += 20; reasons += "Thin pool (${money(liq)})" }
                liq < 100_000 -> score += 10
                liq < 250_000 -> score += 5
            }
            when {
                ageH == null -> { score += 8; reasons += "Pool age unknown" }
                ageH < 24 -> block(LABEL_TOO_NEW, "Pool is only ${ageText(ageH)} old: most tokens this young fail")
                ageH < 72 -> { score += 25; reasons += "Pool is ${ageText(ageH)} old" }
                ageH < 24 * 7 -> { score += 15; reasons += "Pool is ${ageText(ageH)} old" }
                ageH < 24 * 30 -> score += 8
                ageH < 24 * 90 -> score += 3
            }
            if (dex.pumpFunGraduated == false) block(LABEL_TOO_NEW, "Still on the pump.fun bonding curve (about 99% never graduate)")
            val cap = dex.marketCapUsd?.takeIf { it > 0 } ?: dex.fdvUsd
            // holders of a coin that big exchanges list, or one worth $1B+, can exit elsewhere
            val otherExits = safety?.cexListed?.isNotEmpty() == true || (cap ?: 0.0) >= 1e9
            if (liq != null && cap != null && cap > 0 && liq >= MIN_LIQUIDITY_USD && !otherExits) {
                val ratio = liq / cap
                if (ratio < 0.02) { score += 15; reasons += "Pool holds only ${pct(ratio * 100)} of the market cap: large holders cannot exit" }
                else if (ratio < 0.05) score += 8
            }
            val v24 = dex.volumeUsd.h24
            if (liq != null && v24 != null && liq > 0 && v24 / liq > 50) { score += 10; reasons += "Volume is ${(v24 / liq).roundToInt()}x the pool in 24 h (often wash trading)" }
            val tx = dex.txnsH24
            val n = tx.buys + tx.sells
            if (n > 1000 && abs(tx.buys - tx.sells) / n.toDouble() < 0.05) score += 5
            if (n in 1..50) { score += 5; reasons += "Only $n trades in 24 h" }
            // boosts are paid attention, not quality: a badge, and risk only on very young tokens
            if (dex.boostsActive >= 100 && ageH != null && ageH < 72) { score += 5; reasons += "Heavily promoted while new" }
            if (dex.websites.isEmpty() && dex.socials.isEmpty()) { score += 5; reasons += "No website or social links" }
            if (ageH != null && ageH > 24 * 180 && (liq ?: 0.0) > 1_000_000) score -= 10
        }

        if (safety != null) {
            if (safety.honeypot == true) block(LABEL_RED_FLAG, "Honeypot: buying works, selling does not")
            // GoPlus leaves both out when it could not simulate a trade (closed source, for one): untested is not passed
            if (tier == Tier.DEX && dex != null && dex.chainId != "solana" && safety.honeypot == null && safety.sellTaxPct == null) {
                block(LABEL_UNVERIFIED, "Selling could not be tested (no honeypot or tax result)")
            }
            val sell = safety.sellTaxPct
            val buy = safety.buyTaxPct
            val tax = maxOf(sell ?: 0.0, buy ?: 0.0)
            when {
                tax >= PROHIBITIVE_TAX_PCT -> block(LABEL_RED_FLAG, "Trading tax ${pct(buy ?: 0.0)} buy / ${pct(sell ?: 0.0)} sell")
                tax >= 10 -> { score += 20; reasons += "Trading tax ${pct(buy ?: 0.0)} buy / ${pct(sell ?: 0.0)} sell" }
                tax >= 5 -> { score += 10; reasons += "Trading tax ${pct(buy ?: 0.0)} buy / ${pct(sell ?: 0.0)} sell" }
                tax >= 1 -> score += 3
            }
            if (safety.freezable == true && !exemptByListing) block(LABEL_RED_FLAG, "Freeze authority is active: your tokens can be frozen")
            if (safety.mintable == true && !exemptByListing) block(LABEL_RED_FLAG, "Someone can still mint new tokens")
            for (power in safety.ownerPowers) {
                when {
                    BLOCKING_POWERS.any { power.contains(it, ignoreCase = true) } && (!exemptByListing || ALWAYS_BLOCKING.any { power.contains(it, ignoreCase = true) }) ->
                        block(LABEL_RED_FLAG, "Contract: $power")
                    MAJOR_POWERS.any { power.contains(it, ignoreCase = true) } -> { score += 15; reasons += "Contract: $power" }
                    else -> { score += 5; reasons += "Contract: $power" }
                }
            }
            if (safety.openSource == false) { score += 20; reasons += "Contract source code is not published" }
            if (safety.proxy == true) { score += 10; reasons += "Upgradeable (proxy) contract" }
            (safety.transferFeePct ?: 0.0).let { if (it >= 5) { score += 20; reasons += "Transfer fee ${pct(it)}" } else if (it > 0) score += 10 }
            safety.creatorPct?.let { if (it > 10) { score += 10; reasons += "Creator holds ${pct(it)}" } else if (it > 5) score += 5 }
            (safety.lpLockedPct)?.let { if (it < 50 && dex?.pumpFunGraduated == null) { score += 10; reasons += "Only ${pct(it)} of pool liquidity is locked or burnt" } }
            if (safety.cexListed.isNotEmpty()) { score -= 15; reasons += "Also listed on ${safety.cexListed.take(3).joinToString()}" }
            if (safety.trusted == true) score -= 20
        } else if (tier == Tier.DEX) {
            block(LABEL_UNVERIFIED, "Contract not checked yet (open the token to run the free checks)")
        } else if (tier == Tier.TRENDING && (market?.marketCapRank == null || mcap == null || mcap < MIN_UNCHECKED_TRENDING_CAP)) {
            // trending lists carry days-old DEX tokens too, and nothing checks their contract or pools
            val why = when {
                market?.marketCapRank == null -> "no market-cap rank"
                mcap == null -> "an unknown market cap"
                else -> "a market cap of only ${money(mcap)}"
            }
            block(LABEL_UNVERIFIED, "Trending coin with $why: its contract and pools are not checked")
        }

        val blocked = blockLabel != null || score >= BLOCK_SCORE
        val s = when {
            blockLabel == LABEL_RED_FLAG -> 100
            blockLabel != null -> maxOf(score.roundToInt(), BLOCK_SCORE).coerceAtMost(99)
            else -> score.roundToInt().coerceIn(0, 99)
        }
        val label = blockLabel ?: when {
            s >= 75 -> LABEL_EXTREME
            s >= 50 -> LABEL_HIGH
            s >= 25 -> LABEL_ELEVATED
            else -> LABEL_LOWER
        }
        return RiskAssessment(s, label, (redFlags + reasons).distinct().take(8), blocked)
    }

    /** Always a hard block, CEX listing or not. */
    private val ALWAYS_BLOCKING = listOf("change balances", "cannot sell", "buying is blocked", "cannot be transferred", "frozen by default", "creator made honeypots")
    private val BLOCKING_POWERS = ALWAYS_BLOCKING + listOf("taken back", "hidden owner", "self-destruct")
    private val MAJOR_POWERS = listOf("change the tax", "taxes per wallet", "transfer hook", "paused")

    private fun money(v: Double): String = when {
        v >= 1e9 -> "$%.1fB".format(v / 1e9)
        v >= 1e6 -> "$%.1fM".format(v / 1e6)
        v >= 1e3 -> "$%.0fK".format(v / 1e3)
        else -> "$%.0f".format(v)
    }

    private fun pct(v: Double): String = "%.0f%%".format(v)

    private fun ageText(hours: Double): String = if (hours < 48) "${hours.roundToInt().coerceAtLeast(1)} h" else "${(hours / 24).roundToInt()} days"
}
