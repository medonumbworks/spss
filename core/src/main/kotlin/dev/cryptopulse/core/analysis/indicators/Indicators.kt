package dev.cryptopulse.core.analysis.indicators

import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class Macd(val macd: Double, val signal: Double, val histogram: Double)

/** Classic technical indicators on close series (oldest first). All functions are pure and null-safe. */
object Indicators {

    /** Exponential moving average series, seeded with the SMA of the first [period] values. */
    fun ema(values: List<Double>, period: Int): List<Double> {
        if (values.isEmpty() || period <= 0) return emptyList()
        val k = 2.0 / (period + 1)
        val out = ArrayList<Double>(values.size)
        val seedCount = min(period, values.size)
        var prev = values.take(seedCount).average()
        for (i in values.indices) {
            prev = if (i < seedCount) values.take(i + 1).average() else values[i] * k + prev * (1 - k)
            out += prev
        }
        return out
    }

    fun lastEma(values: List<Double>, period: Int): Double? = if (values.size < period) null else ema(values, period).lastOrNull()

    fun sma(values: List<Double>, period: Int): Double? = if (values.size < period || period <= 0) null else values.takeLast(period).average()

    /** Wilder's RSI over the last value. Needs period + 1 points. Returns 50 for a flat series. */
    fun rsi(values: List<Double>, period: Int = 14): Double? {
        if (values.size < period + 1) return null
        var gain = 0.0
        var loss = 0.0
        for (i in 1..period) {
            val d = values[i] - values[i - 1]
            if (d >= 0) gain += d else loss -= d
        }
        var avgGain = gain / period
        var avgLoss = loss / period
        for (i in period + 1 until values.size) {
            val d = values[i] - values[i - 1]
            avgGain = (avgGain * (period - 1) + max(d, 0.0)) / period
            avgLoss = (avgLoss * (period - 1) + max(-d, 0.0)) / period
        }
        if (avgGain == 0.0 && avgLoss == 0.0) return 50.0
        if (avgLoss == 0.0) return 100.0
        val rs = avgGain / avgLoss
        return 100.0 - 100.0 / (1.0 + rs)
    }

    fun macd(values: List<Double>, fast: Int = 12, slow: Int = 26, signal: Int = 9): Macd? {
        if (values.size < slow + signal) return null
        val f = ema(values, fast)
        val s = ema(values, slow)
        val line = f.indices.map { f[it] - s[it] }
        val sig = ema(line.drop(slow - 1), signal)
        val m = line.last()
        val sg = sig.last()
        return Macd(m, sg, m - sg)
    }

    /** Bollinger %B: 0 = lower band, 1 = upper band (can go outside 0..1). */
    fun bollingerPercentB(values: List<Double>, period: Int = 20, k: Double = 2.0): Double? {
        if (values.size < period) return null
        val window = values.takeLast(period)
        val mean = window.average()
        val sd = stdev(window)
        if (sd == 0.0) return 0.5
        val upper = mean + k * sd
        val lower = mean - k * sd
        return (values.last() - lower) / (upper - lower)
    }

    fun logReturns(values: List<Double>): List<Double> =
        (1 until values.size).mapNotNull { i ->
            val a = values[i - 1]
            val b = values[i]
            if (a > 0 && b > 0) ln(b / a) else null
        }

    fun stdev(values: List<Double>): Double {
        if (values.size < 2) return 0.0
        val mean = values.average()
        return sqrt(values.sumOf { (it - mean) * (it - mean) } / (values.size - 1))
    }

    /** Standard deviation of log returns per step, or null with fewer than 3 points. */
    fun volatility(values: List<Double>, window: Int = values.size): Double? {
        val r = logReturns(values.takeLast(window + 1))
        return if (r.size < 2) null else stdev(r)
    }

    /** Percent change over the last [lookback] steps. */
    fun pctChange(values: List<Double>, lookback: Int): Double? {
        if (values.size <= lookback || lookback <= 0) return null
        val a = values[values.size - 1 - lookback]
        return if (a == 0.0) null else (values.last() - a) / a * 100.0
    }

    /** How far above the window minimum the last value is, in percent (0 = at the low). */
    fun distanceToMinPct(values: List<Double>, window: Int = values.size): Double? {
        val w = values.takeLast(window)
        if (w.isEmpty()) return null
        val lo = w.min()
        return if (lo <= 0) null else (w.last() - lo) / lo * 100.0
    }

    /** Drawdown from the window maximum in percent (<= 0). */
    fun drawdownFromMaxPct(values: List<Double>, window: Int = values.size): Double? {
        val w = values.takeLast(window)
        if (w.isEmpty()) return null
        val hi = w.max()
        return if (hi <= 0) null else (w.last() - hi) / hi * 100.0
    }

    /** Groups consecutive values into buckets of [size] and keeps each bucket's last value (e.g. hourly -> 4h). */
    fun resample(values: List<Double>, size: Int): List<Double> {
        if (size <= 1) return values
        val offset = values.size % size
        return (offset until values.size step size).map { start -> values[min(start + size - 1, values.size - 1)] }
            .let { if (offset > 0) listOf(values[offset - 1]) + it else it }
    }

    /** Coefficient of variation; used to detect pegged assets. */
    fun coefficientOfVariation(values: List<Double>): Double? {
        if (values.size < 3) return null
        val mean = values.average()
        return if (mean == 0.0) null else stdev(values) / mean
    }
}
