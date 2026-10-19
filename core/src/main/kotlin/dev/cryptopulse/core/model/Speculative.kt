package dev.cryptopulse.core.model

import kotlinx.serialization.Serializable

/**
 * Which part of the market list an asset belongs to. Bitpanda assets can be bought on Bitpanda;
 * everything else is shown for information only and is clearly marked as not on Bitpanda.
 */
@Serializable
enum class Tier(val label: String) {
    /** Listed on Bitpanda (live ticker, Public API or bundled snapshot). */
    BITPANDA("Bitpanda"),
    /** Popular memecoin by market cap (CoinGecko "Meme" category), not on Bitpanda. */
    MEME("Memecoin"),
    /** On CoinGecko's trending list today, not on Bitpanda. */
    TRENDING("Trending"),
    /** Found only on decentralised exchanges (trending pools, paid boosts, pump.fun): highest risk. */
    DEX("DEX radar"),
}

/** One time window of a DEX pair (DexScreener/GeckoTerminal report m5, h1, h6, h24). */
@Serializable
data class DexWindow(val m5: Double? = null, val h1: Double? = null, val h6: Double? = null, val h24: Double? = null)

@Serializable
data class DexTxns(val buys: Int = 0, val sells: Int = 0, val buyers: Int? = null, val sellers: Int? = null)

/** Where and how a DEX-only token trades: its most liquid pool and the flow in it. */
@Serializable
data class DexInfo(
    /** DexScreener chain id: "solana", "ethereum", "base", "bsc", "robinhood", ... */
    val chainId: String,
    val tokenAddress: String,
    val pairAddress: String? = null,
    val dexId: String? = null,
    /** Pool page for people who want to look (DexScreener or GeckoTerminal). */
    val url: String? = null,
    val quoteSymbol: String? = null,
    val priceUsd: Double? = null,
    val liquidityUsd: Double? = null,
    val fdvUsd: Double? = null,
    val marketCapUsd: Double? = null,
    val priceChangePct: DexWindow = DexWindow(),
    val volumeUsd: DexWindow = DexWindow(),
    val txnsH1: DexTxns = DexTxns(),
    val txnsH6: DexTxns = DexTxns(),
    val txnsH24: DexTxns = DexTxns(),
    val pairCreatedAtMs: Long? = null,
    /** DexScreener paid boosts currently active (promotion, not quality). */
    val boostsActive: Int = 0,
    val websites: List<String> = emptyList(),
    /** Social links by type ("twitter", "telegram", ...). */
    val socials: Map<String, String> = emptyMap(),
    /** Why the token is on the radar: "gt_trending", "gt_new", "ds_boost", "ds_profile", "ds_takeover", "pumpfun", "cg_trending". */
    val foundBy: List<String> = emptyList(),
    /** pump.fun: still on the bonding curve (false) or graduated to a real pool (true). */
    val pumpFunGraduated: Boolean? = null,
)

/** Contract checks from GoPlus / honeypot.is. Null = unknown (not checked or not reported). */
@Serializable
data class TokenSafety(
    val source: String,
    val honeypot: Boolean? = null,
    val buyTaxPct: Double? = null,
    val sellTaxPct: Double? = null,
    /** The supply can still be increased by someone. */
    val mintable: Boolean? = null,
    /** Solana: holders' token accounts can be frozen. */
    val freezable: Boolean? = null,
    /** Someone can change balances, take back ownership, block selling or pause transfers. */
    val ownerPowers: List<String> = emptyList(),
    /** Contract ownership renounced (EVM owner is the zero/dead address). */
    val ownerRenounced: Boolean? = null,
    val openSource: Boolean? = null,
    val proxy: Boolean? = null,
    val transferFeePct: Double? = null,
    val holderCount: Int? = null,
    /** Share of supply held by the 10 largest holders (0..100), exchanges and pools included. */
    val top10HolderPct: Double? = null,
    val creatorPct: Double? = null,
    /** Share of the liquidity that is burnt or locked (0..100) when reported. */
    val lpLockedPct: Double? = null,
    val cexListed: List<String> = emptyList(),
    /** GoPlus marks a few well-known tokens as trusted. */
    val trusted: Boolean? = null,
    val checkedAtMs: Long = 0,
) {
    /**
     * Both checks together, the worse answer winning: a honeypot or owner power either source reports
     * stays, the higher tax counts. Used to add honeypot.is to GoPlus and to keep earlier findings.
     */
    fun combinedWith(other: TokenSafety): TokenSafety = TokenSafety(
        source = listOf(source, other.source).flatMap { it.split('+') }.distinct().joinToString("+"),
        honeypot = worst(honeypot, other.honeypot),
        buyTaxPct = listOfNotNull(buyTaxPct, other.buyTaxPct).maxOrNull(),
        sellTaxPct = listOfNotNull(sellTaxPct, other.sellTaxPct).maxOrNull(),
        mintable = worst(mintable, other.mintable),
        freezable = worst(freezable, other.freezable),
        ownerPowers = (ownerPowers + other.ownerPowers).distinct(),
        ownerRenounced = if (ownerRenounced == false || other.ownerRenounced == false) false else ownerRenounced ?: other.ownerRenounced,
        openSource = if (openSource == false || other.openSource == false) false else openSource ?: other.openSource,
        proxy = worst(proxy, other.proxy),
        transferFeePct = listOfNotNull(transferFeePct, other.transferFeePct).maxOrNull(),
        holderCount = holderCount ?: other.holderCount,
        top10HolderPct = listOfNotNull(top10HolderPct, other.top10HolderPct).maxOrNull(),
        creatorPct = listOfNotNull(creatorPct, other.creatorPct).maxOrNull(),
        lpLockedPct = listOfNotNull(lpLockedPct, other.lpLockedPct).minOrNull(),
        cexListed = (cexListed + other.cexListed).distinct(),
        trusted = if (trusted == false || other.trusted == false) false else trusted ?: other.trusted,
        checkedAtMs = maxOf(checkedAtMs, other.checkedAtMs),
    )

    private fun worst(a: Boolean?, b: Boolean?): Boolean? = if (a == true || b == true) true else a ?: b
}

/** 0..100, higher = riskier, with the reasons; [blocked] = never show an "up" probability. */
@Serializable
data class RiskAssessment(
    val score: Int,
    val label: String,
    val reasons: List<String> = emptyList(),
    val blocked: Boolean = false,
)
