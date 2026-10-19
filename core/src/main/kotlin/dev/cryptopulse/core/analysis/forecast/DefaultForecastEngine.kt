package dev.cryptopulse.core.analysis.forecast

import dev.cryptopulse.core.analysis.ForecastInput
import dev.cryptopulse.core.analysis.Forecaster
import dev.cryptopulse.core.analysis.indicators.Indicators
import dev.cryptopulse.core.model.BacktestStats
import dev.cryptopulse.core.model.BottomAssessment
import dev.cryptopulse.core.model.DataCoverage
import dev.cryptopulse.core.model.Direction
import dev.cryptopulse.core.model.Forecast
import dev.cryptopulse.core.model.Horizon
import dev.cryptopulse.core.model.HorizonForecast
import dev.cryptopulse.core.model.Signal
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.tanh

/**
 * Explainable up/down model for 24 hours and 7 days.
 *
 * logit = technical (price only, self-calibrated by a walk-forward backtest)
 *       + market context (BTC trend, Fear & Greed as a contrarian signal, total market cap move)
 *       + crowd (social sentiment, attention spikes, futures funding and long/short positioning, community votes)
 *
 * The probability is then shrunk toward 50 % according to data coverage, and clamped to 15-85 %:
 * nobody can predict crypto prices with certainty and the UI must never suggest otherwise.
 */
class DefaultForecastEngine : Forecaster {

    override fun forecast(input: ForecastInput): Forecast {
        val market = input.market
        val price = market?.price ?: input.hourlyCloses.lastOrNull()
        val hourly = input.hourlyCloses.filter { it > 0 }
        val daily = input.dailyCloses.filter { it > 0 }
        if (isStablecoin(input.asset.symbol, input.asset.name, market?.currency, hourly, market?.change7dPct)) return stablecoinForecast(input, price)

        val coverageNotes = ArrayList<String>()
        if (hourly.size < 48) coverageNotes += "Less than 2 days of hourly prices"
        if (daily.size < 40) coverageNotes += "No long daily history loaded (open the coin for a 1-year analysis)"

        // ---- 24 hours --------------------------------------------------------------------------
        val tech24 = Technical.shortTerm(hourly)
        val bt24 = Backtester.shortTerm(hourly)
        val techScale24 = techScale(bt24)
        val ctx24 = contextSignals(input, Horizon.H24)
        val crowd24 = crowdSignals(input, Horizon.H24)
        val h24 = buildHorizon(Horizon.H24, price, hourly, daily, tech24, techScale24, ctx24 + crowd24, bt24, input)

        // ---- 7 days ----------------------------------------------------------------------------
        val (tech7, usedDaily) = Technical.mediumTerm(daily, hourly)
        val bt7 = if (usedDaily) Backtester.mediumTermDaily(daily) else null
        val techScale7 = techScale(bt7) * if (usedDaily) 1.0 else 0.8
        val ctx7 = contextSignals(input, Horizon.D7) + momentumFromMarket(input, usedDaily)
        val crowd7 = crowdSignals(input, Horizon.D7)
        val bottom = bottomAssessment(input, hourly, daily)
        val bottomSignal = Contribution("bottom_score", "Distance from recent lows", bottom.score.toDouble(), (bottom.score - 50) / 50.0 * 0.18,
            "Bottom score ${bottom.score}/100: ${bottom.label.lowercase()}")
        val d7 = buildHorizon(Horizon.D7, price, hourly, daily, tech7, techScale7, ctx7 + crowd7 + bottomSignal, bt7, input)

        val finalBottom = bottom.copy(reboundProbability7d = d7.probabilityUp)
        val coverage = DataCoverage(
            hourlyPoints = hourly.size,
            dailyPoints = daily.size,
            hasSentiment = (input.sentiment?.totalSamples ?: 0) > 0,
            hasDerivatives = input.derivatives?.fundingRate != null,
            hasFearGreed = input.context?.fearGreed != null,
            hasCommunityVotes = input.communityVotesUpPct != null,
            notes = coverageNotes,
        )
        return Forecast(
            symbol = input.asset.symbol,
            currency = market?.currency ?: "EUR",
            price = price,
            h24 = h24,
            d7 = d7,
            bottom = finalBottom,
            sentiment = input.sentiment,
            coverage = coverage,
            summary = summary(input.asset.name, h24, d7, finalBottom),
            computedAtMs = input.nowMs,
            modelVersion = MODEL_VERSION,
        )
    }

    // ------------------------------------------------------------------------------------------

    private fun buildHorizon(
        horizon: Horizon,
        price: Double?,
        hourly: List<Double>,
        daily: List<Double>,
        technical: List<Contribution>,
        techScale: Double,
        other: List<Contribution>,
        backtest: BacktestStats?,
        input: ForecastInput,
    ): HorizonForecast {
        val scaledTech = technical.map { it.copy(logit = it.logit * techScale) }
        val all = (scaledTech + other).filter { it.logit.isFinite() }
        val logit = all.sumOf { it.logit }.coerceIn(-2.5, 2.5)
        val pRaw = 1.0 / (1.0 + exp(-logit))

        val coverage = coverageScore(horizon, hourly, daily, input)
        val shrink = 0.55 + 0.35 * coverage
        val p = (0.5 + (pRaw - 0.5) * shrink).coerceIn(0.15, 0.85)
        val direction = when {
            p >= 0.55 -> Direction.UP
            p <= 0.45 -> Direction.DOWN
            else -> Direction.FLAT
        }
        val sign = if (p >= 0.5) 1.0 else -1.0
        val totalAbs = all.sumOf { abs(it.logit) }
        val agreement = if (totalAbs == 0.0) 0.5 else all.filter { it.logit * sign > 0 }.sumOf { abs(it.logit) } / totalAbs
        val btQuality = edgeZ(backtest)?.let { (0.5 + 0.12 * it).coerceIn(0.2, 0.9) } ?: 0.5
        val confidence = ((0.4 * coverage + 0.3 * agreement + 0.3 * btQuality) * 0.9).coerceIn(0.05, 0.9)

        val sigma = horizonSigma(horizon, hourly, daily)
        val (low, high, move) = if (price != null && sigma != null) {
            val drift = (2 * p - 1) * sigma * 0.5
            Triple(price * exp(drift - sigma), price * exp(drift + sigma), (exp(drift) - 1) * 100)
        } else Triple(null, null, null)

        val signals = all.sortedByDescending { abs(it.logit) }.map {
            Signal(it.name, it.label, it.value, round4(it.logit), it.description)
        }
        return HorizonForecast(horizon, round4(p), direction, round4(confidence), low, high, move?.let { round4(it) }, signals, backtest)
    }

    /**
     * 0.85 = neutral (too little evidence either way); above when the backtest beat chance clearly,
     * below when it clearly lost. "Clearly" is measured in standard errors of the hit rate.
     */
    private fun techScale(bt: BacktestStats?): Double {
        val z = edgeZ(bt) ?: return 0.85
        return when {
            z >= 2.0 -> 1.2
            z >= 1.0 -> 1.0
            z > -1.0 -> 0.85
            z > -2.0 -> 0.7
            else -> 0.45
        }
    }

    /** Backtest edge over chance in standard errors, or null with fewer than 10 independent windows. */
    private fun edgeZ(bt: BacktestStats?): Double? {
        val nEff = bt?.effectiveSamples ?: return null
        if (nEff < 10) return null
        return (bt.hitRate - bt.baselineHitRate) / sqrt(0.25 / nEff)
    }

    private fun coverageScore(horizon: Horizon, hourly: List<Double>, daily: List<Double>, input: ForecastInput): Double {
        val parts = ArrayList<Double>()
        parts += min(1.0, hourly.size / 120.0)
        if (horizon == Horizon.D7) parts += min(1.0, daily.size / 120.0)
        parts += input.sentiment?.confidence ?: 0.0
        parts += if (input.derivatives?.fundingRate != null) 1.0 else 0.4
        parts += if (input.context?.fearGreed != null) 1.0 else 0.5
        return parts.average()
    }

    private fun horizonSigma(horizon: Horizon, hourly: List<Double>, daily: List<Double>): Double? = when (horizon) {
        Horizon.H24 -> Indicators.volatility(hourly, 72)?.let { it * sqrt(24.0) }
        Horizon.D7 -> (if (daily.size >= 30) Indicators.volatility(daily, 60)?.let { it * sqrt(7.0) } else null)
            ?: Indicators.volatility(hourly, hourly.size)?.let { it * sqrt(168.0) }
    }

    // ---- context ----------------------------------------------------------------------------

    private fun contextSignals(input: ForecastInput, horizon: Horizon): List<Contribution> {
        val out = ArrayList<Contribution>()
        val ctx = input.context ?: return out
        val isBtc = input.asset.symbol == "BTC"
        if (!isBtc) ctx.btcTrendScore?.let { t ->
            val w = if (horizon == Horizon.H24) 0.22 else 0.28
            out += Contribution("btc_trend", "Bitcoin trend", t, t * w, if (t >= 0) "Bitcoin is trending up; most coins move with it" else "Bitcoin is trending down; most coins move with it")
        }
        ctx.fearGreed?.let { fg ->
            val w = if (horizon == Horizon.H24) 0.08 else 0.22
            val c = -(fg.value - 50) / 50.0 * w
            out += Contribution("fear_greed", "Fear & Greed index", fg.value.toDouble(), c, when {
                fg.value <= 25 -> "Extreme fear (${fg.value}); historically a contrarian buy zone"
                fg.value >= 75 -> "Extreme greed (${fg.value}); historically a contrarian warning"
                else -> "Market mood: ${fg.classification.lowercase()} (${fg.value})"
            })
        }
        if (horizon == Horizon.H24) ctx.global?.marketCapChange24hPct?.let { g ->
            out += Contribution("total_market", "Total crypto market (24h)", g, tanh(g / 3.0) * 0.08, "Total crypto market cap moved ${"%.1f".format(g)}% in 24h")
        }
        ctx.globalSentiment?.takeIf { it.totalSamples >= 20 }?.let { s ->
            val w = if (horizon == Horizon.H24) 0.12 else 0.1
            out += Contribution("market_mood_social", "Overall crypto chatter", s.composite, s.composite * s.confidence * w, "Overall crypto social mood is ${s.label.lowercase()}")
        }
        val id = input.asset.coingeckoId
        if (horizon == Horizon.H24 && id != null && id in ctx.trendingCoingeckoIds) {
            out += Contribution("trending", "Trending on CoinGecko", 1.0, 0.05, "Among the most searched coins on CoinGecko today")
        }
        return out
    }

    /** CoinGecko's 14d/30d changes give a medium-term trend even without loaded daily history. */
    private fun momentumFromMarket(input: ForecastInput, usedDaily: Boolean): List<Contribution> {
        if (usedDaily) return emptyList()
        val m30 = input.market?.change30dPct ?: input.market?.change14dPct ?: return emptyList()
        val label = if (input.market?.change30dPct != null) "30-day" else "14-day"
        return listOf(Contribution("momentum_market", "$label momentum", m30, tanh(m30 / 25.0) * 0.2, "Over $label the price moved ${"%.1f".format(m30)}%"))
    }

    private fun crowdSignals(input: ForecastInput, horizon: Horizon): List<Contribution> {
        val out = ArrayList<Contribution>()
        input.sentiment?.takeIf { it.totalSamples > 0 }?.let { s ->
            val w = if (horizon == Horizon.H24) 0.6 else 0.45
            val c = s.composite * s.confidence * w
            out += Contribution("sentiment", "Social sentiment", s.composite, c,
                "Sentiment from ${s.totalSamples} posts/headlines is ${s.label.lowercase()} (confidence ${(s.confidence * 100).toInt()}%)")
            val buzz = s.perSource.firstNotNullOfOrNull { it.buzzRatio }
            if (buzz != null && buzz > 0 && abs(s.composite) > 0.05) {
                val c2 = tanh(ln(buzz) / ln(2.0)) * (if (s.composite > 0) 1 else -1) * 0.08
                out += Contribution("buzz", "Attention change", buzz, c2, "Mentions changed ${"%.0f".format((buzz - 1) * 100)}% versus the previous day")
            }
        }
        input.derivatives?.fundingRate?.let { f ->
            val w = if (horizon == Horizon.H24) 0.18 else 0.22
            val c = -tanh((f - 0.0001) / 0.0004) * w
            out += Contribution("funding", "Futures funding rate", f * 100, c, when {
                f > 0.0005 -> "High positive funding (${"%.3f".format(f * 100)}%): longs are crowded, squeezes down are common"
                f < -0.0001 -> "Negative funding (${"%.3f".format(f * 100)}%): shorts are crowded, short squeezes are possible"
                else -> "Funding is normal (${"%.3f".format(f * 100)}%)"
            })
        }
        input.derivatives?.longShortRatio?.let { r ->
            val c = -tanh((r - 1.2) / 0.8) * 0.12
            out += Contribution("long_short", "Long/short accounts", r, c, if (r > 2) "Most futures accounts are long (${"%.2f".format(r)}); contrarian bearish" else if (r < 0.9) "More accounts are short (${"%.2f".format(r)}); contrarian bullish" else "Futures positioning is balanced (${"%.2f".format(r)})")
        }
        input.communityVotesUpPct?.let { v ->
            out += Contribution("community_votes", "CoinGecko community votes", v, tanh((v - 70) / 20) * 0.06, "${v.toInt()}% of CoinGecko voters feel good about it today")
        }
        return out
    }

    // ---- bottom -----------------------------------------------------------------------------

    private fun bottomAssessment(input: ForecastInput, hourly: List<Double>, daily: List<Double>): BottomAssessment {
        val market = input.market
        val reasons = ArrayList<String>()
        val parts = ArrayList<Pair<Double, Double>>() // (value 0..1, weight)
        val dist7 = Indicators.distanceToMinPct(hourly)
        val dist30 = if (daily.size >= 30) Indicators.distanceToMinPct(daily, 30) else null
        val dist90 = if (daily.size >= 90) Indicators.distanceToMinPct(daily, 90) else null
        val dist1y = if (daily.size >= 300) Indicators.distanceToMinPct(daily, 365) else null
        // without daily closes the 30-day high is unknown (the 30-day change is not a drawdown)
        val dd30 = if (daily.size >= 30) Indicators.drawdownFromMaxPct(daily, 30) else null
        val ddAth = market?.athChangePct
        val distAtl = market?.atlChangePct
        val rsi = (if (daily.size >= 20) Indicators.rsi(daily, 14) else null) ?: Indicators.rsi(Indicators.resample(hourly, 4), 14)
        val pctB = (if (daily.size >= 20) Indicators.bollingerPercentB(daily, 20) else null) ?: Indicators.bollingerPercentB(Indicators.resample(hourly, 4), 20)
        val fg = input.context?.fearGreed?.value

        dist7?.let { parts += (1 - clamp01(it / 8.0)) to 0.15; if (it < 2) reasons += "Within ${"%.1f".format(it)}% of its 7-day low" }
        dist30?.let { parts += (1 - clamp01(it / 15.0)) to 0.25; if (it < 4) reasons += "Within ${"%.1f".format(it)}% of its 30-day low" }
        dist90?.let { parts += (1 - clamp01(it / 25.0)) to 0.1 }
        dist1y?.let { parts += (1 - clamp01(it / 35.0)) to 0.1; if (it < 8) reasons += "Close to its 1-year low (+${"%.1f".format(it)}%)" }
        rsi?.let { parts += clamp01((45 - it) / 25.0) to 0.15; if (it < 30) reasons += "Oversold (RSI ${it.toInt()})" else if (it > 70) reasons += "Overbought (RSI ${it.toInt()})" }
        pctB?.let { parts += clamp01((0.3 - it) / 0.5) to 0.1 }
        dd30?.let { parts += clamp01(-it / 30.0) to 0.1; if (it < -20) reasons += "Down ${"%.0f".format(-it)}% from its 30-day high" }
        // Fear & Greed is market-wide: it may tilt a coin's score, never make one on its own
        if (parts.sumOf { it.second } < 0.25) {
            return BottomAssessment(score = 50, label = NO_HISTORY, distanceToAtlPct = distAtl, drawdownFromAthPct = ddAth,
                reasons = listOf("Not enough price history to tell where the price sits in its range"))
        }
        fg?.let { parts += clamp01((50 - it) / 40.0) to 0.05; if (it <= 25) reasons += "Market in extreme fear ($it)" }
        if (ddAth != null && ddAth < -80) reasons += "${"%.0f".format(-ddAth)}% below its all-time high"
        if (dist7 != null && dist7 > 10 && (dist30 == null || dist30 > 15)) reasons += "Well above its recent lows"

        val wsum = parts.sumOf { it.second }
        val score = if (wsum == 0.0) 50 else (parts.sumOf { it.first * it.second } / wsum * 100).toInt().coerceIn(0, 100)
        val label = when {
            score >= 70 -> "Near a local low"
            score >= 55 -> "Lower part of its range"
            score >= 40 -> "Mid-range"
            score >= 25 -> "Upper part of its range"
            else -> "Near a local high"
        }
        return BottomAssessment(
            score = score,
            label = label,
            distanceTo7dLowPct = dist7?.let { round4(it) },
            distanceTo30dLowPct = dist30?.let { round4(it) },
            distanceTo90dLowPct = dist90?.let { round4(it) },
            distanceTo1yLowPct = dist1y?.let { round4(it) },
            distanceToAtlPct = distAtl,
            drawdownFrom30dHighPct = dd30?.let { round4(it) },
            drawdownFromAthPct = ddAth,
            rsi14 = rsi?.let { round4(it) },
            bollingerPercentB = pctB?.let { round4(it) },
            reasons = reasons.take(6),
        )
    }

    // ---- misc -------------------------------------------------------------------------------

    private fun isStablecoin(symbol: String, name: String, currency: String?, hourly: List<Double>, change7d: Double?): Boolean {
        if (symbol in STABLECOINS) return true
        if (hourly.size < 48) return false
        val cv = Indicators.coefficientOfVariation(hourly) ?: return false
        if (currency == null || currency.equals("USD", ignoreCase = true)) return cv < 0.002 && abs(change7d ?: 0.0) < 0.5
        // quoted in EUR/GBP/CHF a USD peg moves with the exchange rate, so only accept a looser
        // band for coins whose symbol or name says they are pegged
        val pegHint = PEG_HINT.containsMatchIn(symbol) || PEG_NAME_HINT.containsMatchIn(name)
        return pegHint && cv < 0.01 && abs(change7d ?: 0.0) < 3.0
    }

    private fun stablecoinForecast(input: ForecastInput, price: Double?): Forecast {
        val note = "Stablecoin: built to hold its peg, so an up/down forecast is not meaningful. In ${input.market?.currency ?: "EUR"} it only moves with the exchange rate."
        fun flat(h: Horizon) = HorizonForecast(h, 0.5, Direction.FLAT, 0.9, price, price, 0.0, listOf(Signal("stablecoin", "Pegged asset", null, 0.0, note)))
        return Forecast(
            symbol = input.asset.symbol,
            currency = input.market?.currency ?: "EUR",
            price = price,
            h24 = flat(Horizon.H24),
            d7 = flat(Horizon.D7),
            bottom = BottomAssessment(50, "Pegged", reasons = listOf(note)),
            sentiment = input.sentiment,
            coverage = DataCoverage(hourlyPoints = input.hourlyCloses.size, dailyPoints = input.dailyCloses.size),
            summary = "${input.asset.name} is a stablecoin; it is designed not to go up or down.",
            computedAtMs = input.nowMs,
            modelVersion = MODEL_VERSION,
        )
    }

    private fun summary(name: String, h24: HorizonForecast, d7: HorizonForecast, bottom: BottomAssessment): String {
        fun word(h: HorizonForecast) = when {
            h.probabilityUp >= 0.62 -> "likely up"
            h.probabilityUp >= 0.55 -> "slightly bullish"
            h.probabilityUp <= 0.38 -> "likely down"
            h.probabilityUp <= 0.45 -> "slightly bearish"
            else -> "no clear direction"
        }
        val p24 = (h24.probabilityUp * 100).toInt()
        val p7 = (d7.probabilityUp * 100).toInt()
        return "$name: ${word(h24)} for 24h ($p24% up), ${word(d7)} for 7 days ($p7% up). ${bottom.label}."
    }

    companion object {
        const val MODEL_VERSION = "cryptopulse-forecast-1.0"
        val STABLECOINS: Set<String> = setOf(
            "USDT", "USDC", "EURC", "USDG", "RLUSD", "USDD", "EURCV", "DAI", "FDUSD", "TUSD", "PYUSD", "USDE", "USDS", "USD1", "EURS", "USDQ", "EURQ",
        )

        const val NO_HISTORY = "Not enough price history"
        private val PEG_HINT = Regex("USD|EUR|GBP|CHF")
        private val PEG_NAME_HINT = Regex("(?i)stable|dollar|euro")

        private fun clamp01(x: Double) = max(0.0, min(1.0, x))
        private fun round4(x: Double) = Math.round(x * 10_000.0) / 10_000.0
    }
}
