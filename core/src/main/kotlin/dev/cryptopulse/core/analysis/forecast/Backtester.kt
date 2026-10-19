package dev.cryptopulse.core.analysis.forecast

import dev.cryptopulse.core.model.BacktestStats
import kotlin.math.exp

/**
 * Walk-forward check of the price-only part of the model: at every past step t the technical logit is
 * computed from data up to t only and compared with the realised move from t to t + horizon.
 * Samples overlap (each step shares most of its window with the next), so treat results as indicative.
 */
object Backtester {
    fun shortTerm(hourly: List<Double>, horizonSteps: Int = 24, warmup: Int = 48, stride: Int = 2): BacktestStats? =
        run(hourly, horizonSteps, warmup, stride) { prefix -> Technical.shortTerm(prefix).sumOf { it.logit } }

    fun mediumTermDaily(daily: List<Double>, horizonSteps: Int = 7, warmup: Int = 60, stride: Int = 1): BacktestStats? =
        run(daily, horizonSteps, warmup, stride) { prefix -> Technical.mediumTerm(prefix, emptyList()).first.sumOf { it.logit } }

    private inline fun run(series: List<Double>, horizon: Int, warmup: Int, stride: Int, logitAt: (List<Double>) -> Double): BacktestStats? {
        if (series.size < warmup + horizon + 10) return null
        var n = 0
        var hits = 0
        var ups = 0
        var brier = 0.0
        var t = warmup
        while (t + horizon < series.size) {
            val prefix = series.subList(0, t + 1)
            val logit = logitAt(prefix)
            val p = 1.0 / (1.0 + exp(-logit))
            val up = series[t + horizon] > series[t]
            if (up) ups++
            if ((p >= 0.5) == up) hits++
            val y = if (up) 1.0 else 0.0
            brier += (p - y) * (p - y)
            n++
            t += stride
        }
        if (n == 0) return null
        val baseUp = ups.toDouble() / n
        return BacktestStats(
            samples = n,
            hitRate = hits.toDouble() / n,
            brier = brier / n,
            baselineHitRate = maxOf(baseUp, 1 - baseUp),
            note = "Walk-forward on ${series.size} points, $n overlapping samples; baseline = always predicting the more common direction",
        )
    }
}
