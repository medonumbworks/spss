package dev.cryptopulse.core.model

import kotlinx.serialization.Serializable

/** One point of a price/volume time series. */
@Serializable
data class PricePoint(val timeMs: Long, val value: Double)

/**
 * Market snapshot for one asset in one quote currency. Mostly sourced from CoinGecko
 * `/coins/markets`; `bitpandaPrice` comes from the Bitpanda ticker when available.
 */
@Serializable
data class MarketData(
    val symbol: String,
    val coingeckoId: String? = null,
    val name: String? = null,
    val currency: String,
    val price: Double? = null,
    val bitpandaPrice: Double? = null,
    val marketCap: Double? = null,
    val marketCapRank: Int? = null,
    val volume24h: Double? = null,
    val high24h: Double? = null,
    val low24h: Double? = null,
    val change1hPct: Double? = null,
    val change24hPct: Double? = null,
    val change7dPct: Double? = null,
    val change14dPct: Double? = null,
    val change30dPct: Double? = null,
    val change1yPct: Double? = null,
    val ath: Double? = null,
    val athChangePct: Double? = null,
    val athDate: String? = null,
    val atl: Double? = null,
    val atlChangePct: Double? = null,
    val atlDate: String? = null,
    val circulatingSupply: Double? = null,
    val totalSupply: Double? = null,
    /** Hourly closes for the last 7 days, oldest first (CoinGecko `sparkline_in_7d.price`). */
    val sparkline7d: List<Double> = emptyList(),
    val imageUrl: String? = null,
    val lastUpdatedMs: Long? = null,
    /** "coingecko", "bitpanda", "fixture", ... */
    val source: String = "coingecko",
)

/** Fear & Greed index from alternative.me. */
@Serializable
data class FearGreedPoint(val timeS: Long, val value: Int, val classification: String)

@Serializable
data class FearGreed(
    val value: Int,
    val classification: String,
    val timeS: Long,
    /** Oldest first. */
    val history: List<FearGreedPoint> = emptyList(),
) {
    /** Change versus 7 days ago, when history is available. */
    val change7d: Int?
        get() = history.takeIf { it.size >= 8 }?.let { value - it[it.size - 8].value }
}

/** Crowd-positioning data from perpetual futures markets (Binance/Bybit public endpoints). */
@Serializable
data class DerivativesSnapshot(
    val baseSymbol: String,
    /** Last funding rate as a fraction per 8h (0.0001 = 0.01%). */
    val fundingRate: Double? = null,
    val markPrice: Double? = null,
    /** Long/short account ratio (>1 means more long accounts). */
    val longShortRatio: Double? = null,
    val openInterestChange24hPct: Double? = null,
    val source: String = "binance",
    val fetchedAtMs: Long = 0,
)

/** Community/crowd data from CoinGecko `/coins/{id}`. */
@Serializable
data class CoinCommunity(
    val coingeckoId: String,
    val sentimentVotesUpPct: Double? = null,
    val sentimentVotesDownPct: Double? = null,
    val watchlistUsers: Long? = null,
    val redditSubscribers: Long? = null,
    val twitterFollowers: Long? = null,
    val telegramUsers: Long? = null,
    val description: String? = null,
    val homepage: String? = null,
    val subreddit: String? = null,
    val categories: List<String> = emptyList(),
)

@Serializable
data class GlobalStats(
    val totalMarketCapUsd: Double? = null,
    val totalVolume24hUsd: Double? = null,
    val marketCapChange24hPct: Double? = null,
    val btcDominancePct: Double? = null,
    val ethDominancePct: Double? = null,
)

@Serializable
data class TrendingCoin(
    val coingeckoId: String,
    val symbol: String,
    val name: String,
    val marketCapRank: Int? = null,
    val score: Int? = null,
)

/** Market-wide context shared by all per-asset forecasts. */
@Serializable
data class MarketContext(
    val fearGreed: FearGreed? = null,
    val global: GlobalStats? = null,
    val btcChange24hPct: Double? = null,
    val btcChange7dPct: Double? = null,
    /** -1..1 trend score for BTC derived from its own indicators (altcoins follow BTC). */
    val btcTrendScore: Double? = null,
    /** BTC's hourly closes (7-day sparkline), so each coin's Bitcoin signal can follow its own beta. */
    val btcHourly: List<Double> = emptyList(),
    /** When [btcHourly] ends (BTC's last price update in that response), to line it up with each coin's series. */
    val btcHourlyEndMs: Long? = null,
    /** Global crypto mood from r/CryptoCurrency, news feeds and similar broad sources. */
    val globalSentiment: SentimentSummary? = null,
    val trendingCoingeckoIds: List<String> = emptyList(),
    val fetchedAtMs: Long = 0,
)
