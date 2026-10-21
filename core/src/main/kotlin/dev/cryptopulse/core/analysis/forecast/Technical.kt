package dev.cryptopulse.core.analysis.forecast

import dev.cryptopulse.core.analysis.indicators.Indicators
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.math.tanh

/** One contribution to a logit before it becomes a [dev.cryptopulse.core.model.Signal]. */
internal data class Contribution(val name: String, val label: String, val value: Double?, val logit: Double, val description: String)

/**
 * Price-only signals, shared by the live forecast and the walk-forward backtest so the backtest
 * measures exactly what the app shows. Weights are deliberately small: crypto returns are noisy and
 * the model should rarely be confident.
 */
internal object Technical {

    /** 24h horizon on hourly closes. Needs >= 30 points; returns an empty list otherwise. */
    fun shortTerm(hourly: List<Double>): List<Contribution> {
        if (hourly.size < 30) return emptyList()
        val out = ArrayList<Contribution>()
        val volH = Indicators.volatility(hourly, 72)?.takeIf { it > 0 } ?: return emptyList()
        val volDay = volH * sqrt(24.0)

        Indicators.rsi(hourly, 14)?.let { rsi ->
            val c = -tanh((rsi - 50.0) / 18.0) * 0.35
            out += Contribution("rsi14_hourly", "RSI 14 (hourly)", rsi, c, rsiText(rsi, "hours"))
        }
        Indicators.bollingerPercentB(hourly, 20)?.let { b ->
            val c = -tanh((b - 0.5) * 1.6) * 0.18
            out += Contribution("bollinger_hourly", "Bollinger %B (hourly)", b, c, bollingerText(b))
        }
        Indicators.pctChange(hourly, 24)?.let { m ->
            val z = m / 100.0 / volDay
            val c = -tanh(z / 2.0) * 0.15
            out += Contribution("move_24h", "Last 24h move", m, c, if (abs(z) > 1.5) "A ${"%.1f".format(m)}% move in 24h is large for this coin; part of it often retraces" else "The last 24h move (${"%.1f".format(m)}%) is within normal range")
        }
        val emaFast = Indicators.lastEma(hourly, 12)
        val emaSlow = Indicators.lastEma(hourly, 48)
        if (emaFast != null && emaSlow != null && emaSlow > 0) {
            val gap = (emaFast - emaSlow) / emaSlow
            val c = tanh(gap / volDay) * 0.25
            out += Contribution("trend_hourly", "Trend (12h vs 48h average)", gap * 100, c, if (gap >= 0) "Short-term average is above the 2-day average (uptrend)" else "Short-term average is below the 2-day average (downtrend)")
        }
        Indicators.macd(hourly)?.let { m ->
            val price = hourly.last()
            val c = tanh(m.histogram / (price * volH)) * 0.12
            out += Contribution("macd_hourly", "MACD histogram (hourly)", m.histogram, c, if (m.histogram >= 0) "MACD momentum is turning up" else "MACD momentum is turning down")
        }
        return out
    }

    /**
     * 7-day horizon. Uses daily closes when >= 40 points are available, otherwise 4-hour bars
     * built from the hourly series (less reliable, flagged via [usedDaily]).
     */
    fun mediumTerm(daily: List<Double>, hourly: List<Double>): Pair<List<Contribution>, Boolean> {
        val usedDaily = daily.size >= 40
        val series = if (usedDaily) daily else Indicators.resample(hourly, 4)
        if (series.size < 20) return emptyList<Contribution>() to usedDaily
        val stepsPerWeek = if (usedDaily) 7 else 42
        val unit = if (usedDaily) "days" else "4-hour bars"
        val out = ArrayList<Contribution>()
        val vol = Indicators.volatility(series, if (usedDaily) 60 else 42)?.takeIf { it > 0 } ?: return emptyList<Contribution>() to usedDaily
        val volWeek = vol * sqrt(stepsPerWeek.toDouble())

        Indicators.rsi(series, 14)?.let { rsi ->
            val c = -tanh((rsi - 50.0) / 15.0) * 0.35
            out += Contribution(if (usedDaily) "rsi14_daily" else "rsi14_4h", if (usedDaily) "RSI 14 (daily)" else "RSI 14 (4-hour)", rsi, c, rsiText(rsi, unit))
        }
        Indicators.bollingerPercentB(series, 20)?.let { b ->
            val c = -tanh((b - 0.5) * 1.4) * 0.15
            out += Contribution(if (usedDaily) "bollinger_daily" else "bollinger_4h", if (usedDaily) "Bollinger %B (daily)" else "Bollinger %B (4-hour)", b, c, bollingerText(b))
        }
        if (usedDaily) {
            Indicators.pctChange(series, 30)?.let { m ->
                val z = m / 100.0 / (vol * sqrt(30.0))
                val c = tanh(z) * 0.28
                out += Contribution("momentum_30d", "30-day momentum", m, c, "Over 30 days the price moved ${"%.1f".format(m)}%; medium-term trends in crypto tend to persist a little")
            }
            val e20 = Indicators.lastEma(series, 20)
            val e50 = Indicators.lastEma(series, 50)
            if (e20 != null && e50 != null && e50 > 0) {
                val gap = (e20 - e50) / e50
                val c = tanh(gap / volWeek) * 0.22
                out += Contribution("trend_daily", "Trend (20 vs 50-day average)", gap * 100, c, if (gap >= 0) "20-day average is above the 50-day average" else "20-day average is below the 50-day average")
            }
        } else {
            val eFast = Indicators.lastEma(series, 6)
            val eSlow = Indicators.lastEma(series, 24)
            if (eFast != null && eSlow != null && eSlow > 0) {
                val gap = (eFast - eSlow) / eSlow
                val c = tanh(gap / volWeek) * 0.2
                out += Contribution("trend_4h", "Trend (1-day vs 4-day average)", gap * 100, c, if (gap >= 0) "1-day average is above the 4-day average" else "1-day average is below the 4-day average")
            }
        }
        // a 7-day hourly sparkline gives exactly 42 four-hour bars, one short of a full week back
        Indicators.pctChange(series, minOf(stepsPerWeek, series.size - 1))?.let { m ->
            val z = m / 100.0 / volWeek
            val c = -tanh(z / 2.0) * 0.1
            out += Contribution("move_7d", "Last 7 days move", m, c, if (abs(z) > 1.5) "An unusually large 7-day move (${"%.1f".format(m)}%); some mean reversion is common" else "The 7-day move (${"%.1f".format(m)}%) is within normal range")
        }
        return out to usedDaily
    }

    fun rsiText(rsi: Double, unit: String): String = when {
        rsi < 25 -> "RSI ${rsi.toInt()} is deeply oversold ($unit), which often precedes a rebound"
        rsi < 35 -> "RSI ${rsi.toInt()} is oversold ($unit), mildly favouring a bounce"
        rsi > 75 -> "RSI ${rsi.toInt()} is strongly overbought ($unit), pullbacks are common"
        rsi > 65 -> "RSI ${rsi.toInt()} is overbought ($unit), mildly favouring a pause"
        else -> "RSI ${rsi.toInt()} is neutral ($unit)"
    }

    fun bollingerText(b: Double): String = when {
        b < 0.0 -> "Price is below the lower Bollinger band (stretched to the downside)"
        b < 0.2 -> "Price is near the lower Bollinger band"
        b > 1.0 -> "Price is above the upper Bollinger band (stretched to the upside)"
        b > 0.8 -> "Price is near the upper Bollinger band"
        else -> "Price is inside its normal Bollinger range"
    }
}
