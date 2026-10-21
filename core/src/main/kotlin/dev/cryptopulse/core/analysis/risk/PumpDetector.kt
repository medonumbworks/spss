package dev.cryptopulse.core.analysis.risk

import dev.cryptopulse.core.model.DexInfo
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.PumpAlert
import kotlin.math.roundToInt

/**
 * Explicit pump-and-dump warning for one coin. [RiskScorer] already prices these patterns into its
 * 0-100 score; this names them, because "Elevated risk" does not tell a user *why* an +80% chart
 * is suspect.
 *
 * Conservative on purpose: every trigger sits far above normal market noise, so a warning means
 * the shape of the move matches manipulation, not merely that the coin went up. The base rates
 * justify the pessimism (docs/SPECULATIVE.md): 82-83% of memecoins that more than doubled showed
 * artificial growth such as wash trading (Mongardini & Mei, USENIX Security 2026, 34,988 memecoins
 * on ETH, BSC, Solana, Base).
 */
object PumpDetector {
    const val PRICE_SPIKE_24H_PCT = 50.0
    const val PRICE_SPIKE_1H_PCT = 15.0
    /** 24 h volume this many times the pool is churn, not demand (wash trading). */
    const val WASH_VOLUME_OVER_LIQUIDITY = 50.0
    /** Off DEX there is no pool to compare against, so compare volume to the whole float. */
    const val CHURN_VOLUME_OVER_CAP = 3.0
    const val HYPE_MENTION_SPIKE = 5.0
    const val HYPE_MIN_MENTIONS = 10
    const val HYPE_MIN_CHANGE_24H_PCT = 20.0
    const val PROMO_MIN_BOOSTS = 100
    const val PROMO_MAX_AGE_H = 72.0

    /**
     * One plain-words reason per pattern that fired; null when none did (or when both [market] and
     * [dex] are null). [mentions] is the Telegram mention count in the current window and
     * [mentionSpike] its ratio to the 7-day norm, when the sentiment feed knows them.
     */
    fun assess(market: MarketData?, dex: DexInfo?, mentions: Int? = null, mentionSpike: Double? = null, nowMs: Long): PumpAlert? {
        if (market == null && dex == null) return null
        val reasons = ArrayList<String>()
        // DEX-only tokens have no CoinGecko row, so fall back to the pool's own windows
        val change24h = market?.change24hPct ?: dex?.priceChangePct?.h24
        val change1h = market?.change1hPct ?: dex?.priceChangePct?.h1

        if (change24h != null && change24h >= PRICE_SPIKE_24H_PCT) reasons += "price +${whole(change24h)}% in 24 h"
        if (change1h != null && change1h >= PRICE_SPIKE_1H_PCT) reasons += "price +${whole(change1h)}% in the last hour"

        val liq = dex?.liquidityUsd
        val v24 = dex?.volumeUsd?.h24
        if (liq != null && v24 != null && liq > 0 && v24 > 0 && v24 / liq >= WASH_VOLUME_OVER_LIQUIDITY) {
            reasons += "24 h volume is ${whole(v24 / liq)}x the pool, a wash-trading pattern"
        }
        if (dex == null) {
            val cap = market?.marketCap
            val vol = market?.volume24h
            if (cap != null && vol != null && cap > 0 && vol / cap >= CHURN_VOLUME_OVER_CAP) {
                reasons += "the whole market cap traded ${whole(vol / cap)}x over in 24 h"
            }
        }
        // coordinated hype only counts while the price actually pumps: mentions alone can be news
        if (mentionSpike != null && mentionSpike >= HYPE_MENTION_SPIKE && (mentions ?: 0) >= HYPE_MIN_MENTIONS &&
            (change24h ?: 0.0) >= HYPE_MIN_CHANGE_24H_PCT
        ) {
            reasons += "Telegram mentions are ${whole(mentionSpike)}x their 7-day norm while the price pumps"
        }
        // paid boosts on a days-old pool are the classic promote-then-dump setup; unknown age stays quiet
        if (dex != null && dex.boostsActive >= PROMO_MIN_BOOSTS) {
            val ageH = dex.pairCreatedAtMs?.let { (nowMs - it) / 3_600_000.0 }
            if (ageH != null && ageH < PROMO_MAX_AGE_H) reasons += "heavily promoted while days old"
        }
        return if (reasons.isEmpty()) null else PumpAlert(reasons)
    }

    private fun whole(v: Double): Int = v.roundToInt()
}
