package dev.cryptopulse.core.sources.polymarket

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.repo.MentionMatcher
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.util.Dates
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.bool
import dev.cryptopulse.core.util.dbl
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.str
import kotlinx.serialization.Serializable

/** One Polymarket binary market, reduced to its first outcome (the side the question asks about). */
@Serializable
data class PredictionMarket(
    val question: String,
    val slug: String,
    /** The event page on polymarket.com, where a human can read the resolution rules. */
    val url: String,
    /** Name of the first outcome ("Yes", "Up", ...). */
    val outcome: String,
    /** Last price of [outcome] in 0..1; prediction-market prices read directly as probabilities. */
    val probability: Double,
    val endDateMs: Long? = null,
    val volume24hUsd: Double? = null,
    val liquidityUsd: Double? = null,
    /** Tickers spelled as cashtags in the question ("$BTC"), uppercased. */
    val symbols: List<String> = emptyList(),
)

/**
 * Polymarket's public gamma API (keyless): prediction markets as crowd-priced probabilities.
 * `/events?tag_slug=crypto` returns the crypto events ranked by 24h volume, each carrying its
 * `markets[]`; `outcomes` and `outcomePrices` arrive as JSON arrays encoded inside strings.
 * Attribution: market odds by Polymarket (https://polymarket.com).
 */
class PolymarketApi(
    private val http: Http,
    private val base: String = "https://gamma-api.polymarket.com",
) {
    /** The most traded open crypto prediction markets, biggest first (at most ~150). */
    suspend fun cryptoMarkets(): SourceResult<List<PredictionMarket>> =
        when (val r = http.getText("$base/events?tag_slug=crypto&closed=false&order=volume24hr&ascending=false&limit=50", JSON, ttlMs = 10 * 60_000L, staleOkMs = 2 * 3_600_000L, retries = 1)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "Polymarket")
            is HttpResult.Ok -> try {
                SourceResult.Success(parseEvents(r.body), r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("Polymarket", e)
            }
        }

    companion object {
        private val JSON = mapOf("Accept" to "application/json")

        /** Markets trading less than this in 24h are noise (mostly deep out-of-the-money thresholds). */
        private const val MIN_VOLUME_24H_USD = 1_000.0
        private const val MAX_MARKETS = 150
        private const val UP_DOWN_WINDOW_MS = 48 * 3_600_000L

        /** "$BTC" but not "$150,000": a cashtag starts with a letter. */
        private val CASHTAG = Regex("\\$([A-Za-z][A-Za-z0-9]{1,9})(?![A-Za-z0-9])")
        private val UP_OR_DOWN = Regex("up[ -]or[ -]down", RegexOption.IGNORE_CASE)

        /** Hourly / 4-hour up-or-down windows carry a time of day ("1PM ET"); the daily market does not. */
        private val INTRADAY = Regex("\\d\\s*[AP]M\\b", RegexOption.IGNORE_CASE)

        /** Events stay open while individual markets inside them resolve early, so both levels are checked. */
        fun parseEvents(body: String): List<PredictionMarket> =
            JsonUtil.parse(body).arr.orEmpty().flatMap { event ->
                val eventSlug = event["slug"].str ?: return@flatMap emptyList<PredictionMarket>()
                event["markets"].arr.orEmpty().mapNotNull { m ->
                    if (m["closed"].bool == true || m["active"].bool == false) return@mapNotNull null
                    val question = m["question"].str?.trim()?.takeIf { it.isNotEmpty() } ?: return@mapNotNull null
                    val volume24h = m["volume24hr"].dbl
                    if ((volume24h ?: 0.0) < MIN_VOLUME_24H_USD) return@mapNotNull null
                    val probability = jsonStrings(m["outcomePrices"].str).firstOrNull()?.toDoubleOrNull() ?: return@mapNotNull null
                    PredictionMarket(
                        question = question,
                        slug = m["slug"].str ?: eventSlug,
                        url = "https://polymarket.com/event/$eventSlug",
                        outcome = jsonStrings(m["outcomes"].str).firstOrNull() ?: "Yes",
                        probability = probability.coerceIn(0.0, 1.0),
                        endDateMs = Dates.parseToMs(m["endDate"].str),
                        volume24hUsd = volume24h,
                        liquidityUsd = m["liquidity"].dbl,
                        symbols = CASHTAG.findAll(question).map { it.groupValues[1].uppercase() }.distinct().toList(),
                    )
                }
            }.sortedByDescending { it.volume24hUsd ?: 0.0 }.take(MAX_MARKETS)

        /**
         * The markets about one coin, most traded first (at most 5). Matching is by whole word in the
         * question ("$BTC", "BTC" or "Bitcoin", case-insensitive); a 1-2 letter ticker collides with
         * English words ("UP", "SO"), so it only counts as a cashtag or via the coin name.
         */
        fun forAsset(markets: List<PredictionMarket>, symbol: String, name: String): List<PredictionMarket> =
            markets.filter { matchesAsset(it, symbol, name) }.sortedByDescending { it.volume24hUsd ?: 0.0 }.take(5)

        /**
         * Probability (0..1) that [symbol] ends UP in its "Up or Down" market closing within 48h of
         * [nowMs]: the daily market when one is open, else the nearest intraday window. Null when the
         * coin has no such market.
         */
        fun upDownProbability(markets: List<PredictionMarket>, symbol: String, name: String, nowMs: Long): Double? {
            val pick = markets.filter { m ->
                val end = m.endDateMs ?: return@filter false
                end > nowMs && end - nowMs <= UP_DOWN_WINDOW_MS &&
                    UP_OR_DOWN.containsMatchIn(m.question) && matchesAsset(m, symbol, name)
            }.minWithOrNull(compareBy({ INTRADAY.containsMatchIn(it.question) }, { it.endDateMs ?: Long.MAX_VALUE })) ?: return null
            // the stored side is the first outcome; normalize so the answer is always P(up)
            return if (pick.outcome.equals("Down", ignoreCase = true)) 1.0 - pick.probability else pick.probability
        }

        private fun matchesAsset(m: PredictionMarket, symbol: String, name: String): Boolean {
            val sym = symbol.trim().removePrefix("$").uppercase()
            if (sym.isNotEmpty() && sym in m.symbols) return true
            if (name.isNotBlank() && wholeWord(name.trim()).containsMatchIn(m.question)) return true
            // bare tickers follow MentionMatcher's rules: case-sensitive, and never for tickers that are
            // everyday words (THE matched "Will the price of Bitcoin..." before this guard)
            return sym.length > 2 && sym !in MentionMatcher.AMBIGUOUS_SYMBOLS && sym.lowercase() !in MentionMatcher.COMMON_WORD_NAMES &&
                wholeWord(sym, ignoreCase = false).containsMatchIn(m.question)
        }

        private fun wholeWord(term: String, ignoreCase: Boolean = true) =
            Regex("(?<![\\p{L}\\p{N}])${Regex.escape(term)}(?![\\p{L}\\p{N}])", if (ignoreCase) setOf(RegexOption.IGNORE_CASE) else emptySet())

        /** `"[\"Yes\", \"No\"]"`: gamma encodes these arrays as JSON inside a JSON string. */
        private fun jsonStrings(raw: String?): List<String> =
            raw?.let { r -> runCatching { JsonUtil.parse(r).arr.orEmpty().mapNotNull { it.str } }.getOrNull() }.orEmpty()
    }
}
