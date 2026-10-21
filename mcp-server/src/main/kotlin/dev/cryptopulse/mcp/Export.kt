package dev.cryptopulse.mcp

import dev.cryptopulse.core.CryptoPulse
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.Forecast
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.sources.feed.BuzzEntry
import dev.cryptopulse.core.sources.polymarket.PolymarketApi
import dev.cryptopulse.core.sources.polymarket.PredictionMarket
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import kotlinx.coroutines.delay
import java.time.Instant

/**
 * `cryptopulse-mcp export <path>`: one full refresh rendered as the `cryptopulse.dashboard/v1`
 * JSON that the PWA renders. The schema is documented (and owned) by the comment block at the top
 * of web/app.js - field names here are load-bearing, so change that contract first. The hourly
 * dashboard workflow publishes the result as web/data/dashboard.json; `null` values are simply
 * left out because the renderer treats a missing field as null.
 */
object Export {
    const val SCHEMA = "cryptopulse.dashboard/v1"
    private const val BUZZ_LIMIT = 50
    private const val SAMPLES_PER_BUZZ = 3
    private const val POLYMARKET_LIMIT = 20
    /** CoinGecko's 429 cooldown is about a minute; one wait past it before giving up on price history. */
    private const val RATE_LIMIT_RETRY_MS = 65_000L

    suspend fun build(app: CryptoPulse, waitOutRateLimit: suspend () -> Unit = { delay(RATE_LIMIT_RETRY_MS) }): String {
        val repo = app.repository
        var snap = repo.refresh(force = true)
        // jobs that start together on shared CI addresses (a push to main starts four) can all be
        // rate-limited at once: without any price history every coin would show "not enough history"
        if (snap.rows.isNotEmpty() && snap.rows.none { it.market?.sparkline7d.orEmpty().isNotEmpty() }) {
            waitOutRateLimit()
            snap = repo.refresh(force = true)
        }
        val buzz = repo.telegramBuzz(BUZZ_LIMIT)
        val buzzBySymbol = buzz.associateBy { it.symbol }
        // The repository keeps its Polymarket list private; this direct call reuses the refresh's
        // response through the shared 10-minute HTTP cache instead of fetching again.
        val markets = if (app.config.polymarketEnabled) PolymarketApi(app.http).cryptoMarkets().valueOrNull.orEmpty() else emptyList()
        // Crossings mutate the persisted above-threshold set, so the hourly run that exports is
        // also the one that alerts: each coin fires once per crossing, not once per run.
        val alerts = if (app.config.alertsEnabled) repo.alertCrossings() else emptyList()
        val stats = repo.journalStats()

        return buildJsonObject {
            put("schema", SCHEMA)
            put("generatedAt", Instant.ofEpochMilli(snap.fetchedAtMs).toString())
            put("currency", snap.currency)
            snap.context?.fearGreed?.let { fg ->
                putJsonObject("fearGreed") {
                    put("value", fg.value)
                    put("classification", fg.classification)
                }
            }
            putJsonObject("accuracy") {
                stats.perHorizon.forEach { (label, h) ->
                    putJsonObject(label) {
                        put("pending", h.pending)
                        put("evaluated", h.evaluated)
                        put("hits", h.hits)
                        put("neutral", h.neutral)
                        h.hitRate?.let { put("hitRate", it) }
                        h.avgProbability?.let { put("avgProbability", it) }
                        h.brier?.let { put("brier", it) }
                        h.upShare?.let { put("upShare", it) }
                    }
                }
            }
            putJsonArray("rows") { snap.rows.forEach { add(row(it, buzzBySymbol[it.asset.symbol])) } }
            putJsonArray("buzz") { buzz.forEach { add(buzzJson(it)) } }
            putJsonArray("polymarket") { polymarket(markets, snap.rows).forEach { add(it) } }
            putJsonArray("alerts") {
                alerts.forEach { a ->
                    add(buildJsonObject {
                        put("symbol", a.symbol)
                        put("name", a.name)
                        put("probabilityUp", a.probabilityUp)
                        a.price?.let { put("price", it) }
                        a.currency?.let { put("currency", it) }
                    })
                }
            }
            putJsonArray("warnings") { snap.warnings.forEach { add(JsonPrimitive(it)) } }
            putJsonArray("notes") {
                add(JsonPrimitive("Recomputed hourly by the dashboard GitHub Actions workflow; prices and odds can be up to an hour old."))
                add(JsonPrimitive(Forecast.DISCLAIMER))
            }
        }.toString()
    }

    private fun row(r: AssetRow, buzz: BuzzEntry?): JsonObject = buildJsonObject {
        val m = r.market
        val f = r.forecast
        put("symbol", r.asset.symbol)
        put("name", r.asset.name)
        put("tier", r.asset.tier.name)
        putJsonArray("tags") { r.asset.tags.forEach { add(JsonPrimitive(it)) } }
        (m?.bitpandaPrice ?: m?.price)?.let { put("price", it) }
        m?.currency?.let { put("currency", it) }
        m?.change1hPct?.let { put("chg1h", it) }
        m?.change24hPct?.let { put("chg24h", it) }
        m?.change7dPct?.let { put("chg7d", it) }
        m?.marketCap?.let { put("marketCap", it) }
        val up24 = r.upOrNull(f?.h24)
        val up7 = r.upOrNull(f?.d7)
        up24?.let { put("up24", it) }
        up7?.let { put("up7", it) }
        if (up24 != null) f?.h24?.confidence?.let { put("conf24", it) }
        if (f != null && (up24 == null || up7 == null)) {
            // same decision as Format.up: blocked rows show their risk label, the rest say why the horizon has no call
            val gated = if (up24 == null) f.h24 else f.d7
            put("noCall", r.risk?.takeIf { it.blocked }?.label ?: if (gated.signals.firstOrNull()?.name == "low_move") "Moves too little" else "Not enough history")
        }
        // placeholder bottoms (pegged, no history, blocked) are not shown, like in the tables
        f?.bottom?.takeIf { it.label !in Format.NO_BOTTOM_LABELS && r.risk?.blocked != true }?.let {
            put("bottomScore", it.score)
            put("bottomLabel", it.label)
        }
        r.risk?.let { risk ->
            putJsonObject("risk") {
                put("score", risk.score)
                put("label", risk.label)
            }
        }
        r.pump?.let { pump ->
            putJsonObject("pump") { putJsonArray("reasons") { pump.reasons.forEach { add(JsonPrimitive(it)) } } }
        }
        put("mentions", buzz?.mentions ?: 0)
        buzz?.spike?.let { put("spike", it) }
        if (r.asset.tier == Tier.DEX) {
            r.asset.chainId?.let { put("chain", it) }
            r.asset.contract?.let { put("contract", it) }
        }
    }

    private fun buzzJson(b: BuzzEntry): JsonObject = buildJsonObject {
        put("symbol", b.symbol)
        put("mentions", b.mentions)
        b.mentionsPrev?.let { put("mentionsPrev", it) }
        b.baseline7d?.let { put("baseline7d", it) }
        b.spike?.let { put("spike", it) }
        put("composite", b.composite)
        putJsonArray("samples") {
            b.samples.asSequence().filter { it.url != null }.take(SAMPLES_PER_BUZZ).forEach { s ->
                add(buildJsonObject {
                    put("text", s.text)
                    put("url", s.url!!)
                    s.community?.let { put("community", it) }
                    put("score", s.score)
                })
            }
        }
    }

    /**
     * The most traded markets that are about a coin in the list. `symbols` combines the question's
     * cashtags with the listed coins the question names; markets about nothing we list are dropped.
     * Matching only runs against Bitpanda rows because short DEX tickers collide with English words.
     */
    private fun polymarket(markets: List<PredictionMarket>, rows: List<AssetRow>): List<JsonObject> {
        if (markets.isEmpty()) return emptyList()
        val matched = HashMap<PredictionMarket, LinkedHashSet<String>>()
        for (row in rows) {
            if (row.asset.tier != Tier.BITPANDA) continue
            for (m in PolymarketApi.forAsset(markets, row.asset.symbol, row.asset.name)) {
                matched.getOrPut(m) { LinkedHashSet() } += row.asset.symbol
            }
        }
        // cryptoMarkets() already sorts by 24h volume descending
        return markets.mapNotNull { m ->
            val symbols = (m.symbols + matched[m].orEmpty()).distinct()
            if (symbols.isEmpty()) return@mapNotNull null
            buildJsonObject {
                put("question", m.question)
                put("probability", m.probability)
                put("outcome", m.outcome)
                m.volume24hUsd?.let { put("volume24hUsd", it) }
                m.endDateMs?.let { put("endDateMs", it) }
                put("url", m.url)
                putJsonArray("symbols") { symbols.forEach { add(JsonPrimitive(it)) } }
            }
        }.take(POLYMARKET_LIMIT)
    }
}
