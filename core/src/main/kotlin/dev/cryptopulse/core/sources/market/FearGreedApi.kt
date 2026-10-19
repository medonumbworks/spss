package dev.cryptopulse.core.sources.market

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.model.FearGreed
import dev.cryptopulse.core.model.FearGreedPoint
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.sources.FearGreedSource
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.int
import dev.cryptopulse.core.util.lng
import dev.cryptopulse.core.util.str

/** Crypto Fear & Greed Index by alternative.me (free, attribution requested). */
class FearGreedApi(
    private val http: Http,
    private val base: String = "https://api.alternative.me",
) : FearGreedSource {
    override suspend fun fetch(days: Int): SourceResult<FearGreed> {
        val url = "$base/fng/?limit=${days.coerceIn(1, 365)}&format=json"
        return when (val r = http.getText(url, mapOf("Accept" to "application/json"), ttlMs = 30 * 60_000L, staleOkMs = 48 * 3_600_000L)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "Fear & Greed")
            is HttpResult.Ok -> try {
                val fg = parse(r.body) ?: return SourceResult.Failure(SourceStatus.EMPTY, "Fear & Greed: empty")
                SourceResult.Success(fg, r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("Fear & Greed", e)
            }
        }
    }

    companion object {
        fun parse(body: String): FearGreed? {
            val points = JsonUtil.parse(body)["data"].arr.orEmpty().mapNotNull { p ->
                val v = p["value"].int ?: return@mapNotNull null
                FearGreedPoint(p["timestamp"].lng ?: 0, v, p["value_classification"].str ?: classify(v))
            }.sortedBy { it.timeS }
            val last = points.lastOrNull() ?: return null
            return FearGreed(last.value, last.classification, last.timeS, points)
        }

        fun classify(v: Int): String = when {
            v <= 24 -> "Extreme Fear"
            v <= 44 -> "Fear"
            v <= 55 -> "Neutral"
            v <= 74 -> "Greed"
            else -> "Extreme Greed"
        }
    }
}
