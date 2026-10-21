package dev.cryptopulse.core.model

import kotlinx.serialization.Serializable

@Serializable
enum class Horizon(val hours: Int, val label: String) {
    H24(24, "24h"),
    D7(168, "7d"),
}

@Serializable
enum class Direction { UP, DOWN, FLAT }

/** One explainable input to a forecast. */
@Serializable
data class Signal(
    /** Stable machine name, e.g. "rsi14", "momentum_24h", "sentiment_composite". */
    val name: String,
    /** Human label, e.g. "RSI (14, hourly)". */
    val label: String,
    /** Raw observed value (RSI level, percent change, ...). */
    val value: Double? = null,
    /** Signed contribution to the up/down logit; positive favours UP. */
    val contribution: Double,
    /** One sentence for the UI, e.g. "RSI 27 is oversold, which favours a rebound". */
    val description: String,
)

@Serializable
data class BacktestStats(
    val samples: Int,
    /** Share of walk-forward predictions whose direction was right. */
    val hitRate: Double,
    /** Mean squared error of the predicted probability (lower is better; 0.25 is coin-flip). */
    val brier: Double,
    /**
     * Hit rate a guesser with no skill would reach: one that calls "up" as often as the model did,
     * at random, on the same samples (so a model that only follows the overall trend scores no edge).
     */
    val baselineHitRate: Double,
    val note: String = "",
    /** Roughly independent samples: windows overlap, so this is samples x stride / horizon. */
    val effectiveSamples: Int = 0,
)

@Serializable
data class HorizonForecast(
    val horizon: Horizon,
    /** Calibrated probability that the price is higher at the end of the horizon. */
    val probabilityUp: Double,
    val direction: Direction,
    /** 0..1: data coverage x model agreement x backtest quality. */
    val confidence: Double,
    /** Expected price band (roughly one standard deviation) in quote currency. */
    val expectedLow: Double? = null,
    val expectedHigh: Double? = null,
    /** Signed expected move in percent (can be tiny; crypto short-term drift is small). */
    val expectedMovePct: Double? = null,
    val signals: List<Signal> = emptyList(),
    val backtest: BacktestStats? = null,
) {
    /** False when there is no call at all (safety red flags, too little history): [probabilityUp] is then a 0.5 placeholder. */
    val hasCall: Boolean get() = signals.firstOrNull()?.name !in NO_CALL_SIGNALS

    companion object {
        val NO_CALL_SIGNALS: Set<String> = setOf("risk_blocked", "history", "low_move")
    }
}

/** "Is this the lowest price?" assessment. */
@Serializable
data class BottomAssessment(
    /** 0..100, higher means closer to a local bottom with rebound conditions. */
    val score: Int,
    /** e.g. "Near local low", "Mid-range", "Near local high". */
    val label: String,
    val distanceTo7dLowPct: Double? = null,
    val distanceTo30dLowPct: Double? = null,
    val distanceTo90dLowPct: Double? = null,
    val distanceTo1yLowPct: Double? = null,
    val distanceToAtlPct: Double? = null,
    val drawdownFrom30dHighPct: Double? = null,
    val drawdownFromAthPct: Double? = null,
    val rsi14: Double? = null,
    val bollingerPercentB: Double? = null,
    /** Probability of trading higher in 7 days given the current bottom conditions. */
    val reboundProbability7d: Double? = null,
    val reasons: List<String> = emptyList(),
)

@Serializable
data class DataCoverage(
    val hourlyPoints: Int = 0,
    val dailyPoints: Int = 0,
    val hasSentiment: Boolean = false,
    val hasDerivatives: Boolean = false,
    val hasFearGreed: Boolean = false,
    val hasCommunityVotes: Boolean = false,
    val notes: List<String> = emptyList(),
)

@Serializable
data class Forecast(
    val symbol: String,
    val currency: String,
    val price: Double? = null,
    val h24: HorizonForecast,
    val d7: HorizonForecast,
    val bottom: BottomAssessment,
    val sentiment: SentimentSummary? = null,
    val coverage: DataCoverage = DataCoverage(),
    /** One or two plain sentences for the list/detail header. */
    val summary: String,
    val computedAtMs: Long,
    val modelVersion: String,
) {
    companion object {
        const val DISCLAIMER: String =
            "Forecasts are statistical estimates built from public market data and social sentiment. " +
                "They are not financial advice, nobody can know the lowest possible price in advance, " +
                "and crypto prices can move far outside the shown ranges."
    }
}
