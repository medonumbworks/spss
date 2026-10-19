package dev.cryptopulse.mcp

import dev.cryptopulse.core.CryptoPulse
import dev.cryptopulse.core.model.SentimentSummary
import kotlinx.coroutines.CancellationException

/**
 * `cryptopulse-mcp check`: one real refresh against the live free APIs, reported as Markdown.
 * CI runs it on GitHub's runners (which can reach every API) so data-source regressions show up
 * with real responses, not only with test fixtures.
 */
object LiveCheck {
    suspend fun run(app: CryptoPulse, symbols: List<String> = listOf("BTC", "SOL")): String = buildString {
        val repo = app.repository
        appendLine("## CryptoPulse live data check")
        val snap = try {
            repo.refresh(force = true)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            appendLine("**Refresh failed:** ${e.message}")
            return@buildString
        }
        val rows = snap.rows
        appendLine("- Universe: ${snap.universeSource}, ${snap.universeSize} Bitpanda assets, ${rows.size} rows in total (${snap.currency})")
        appendLine("- Market data: ${rows.count { it.market?.price != null }} priced, ${rows.count { it.market?.sparkline7d?.isNotEmpty() == true }} with 7-day sparkline, ${rows.count { it.market?.bitpandaPrice != null }} with a Bitpanda price")
        appendLine("- Forecasts: ${rows.count { it.forecast != null }}, of which pegged ${rows.count { it.forecast?.bottom?.label == "Pegged" }}")
        val tiers = rows.groupingBy { it.asset.tier }.eachCount()
        appendLine("- Tiers: ${tiers.entries.joinToString { "${it.key.label} ${it.value}" }}; Bitpanda coins tagged meme ${rows.count { it.asset.onBitpanda && it.asset.isMeme }}; " +
            "memecoin prices from ${rows.filter { it.asset.tier == dev.cryptopulse.core.model.Tier.MEME }.mapNotNull { it.market?.source }.distinct().joinToString().ifEmpty { "-" }}")
        val dex = rows.filter { it.asset.tier == dev.cryptopulse.core.model.Tier.DEX }
        if (dex.isNotEmpty()) {
            appendLine("- DEX radar: ${dex.size} tokens on ${dex.mapNotNull { it.asset.chainId }.distinct().joinToString()}; contract checked ${dex.count { it.safety != null }}; " +
                "24 h forecast shown ${dex.count { it.upOrNull(it.forecast?.h24) != null }}; risk labels ${dex.groupingBy { it.risk?.label ?: "none" }.eachCount()}")
        }
        val ctx = snap.context
        appendLine("- Context: Fear & Greed ${ctx?.fearGreed?.let { "${it.value} (${it.classification})" } ?: "missing"}, global ${if (ctx?.global != null) "ok" else "missing"}, BTC trend ${ctx?.btcTrendScore?.let { "%.2f".format(it) } ?: "missing"}, trending ${ctx?.trendingCoingeckoIds?.size ?: 0}")
        val (hubStatus, hubMessage) = app.sentiment.feedState()
        appendLine("- Cloud hub feed: $hubStatus${hubMessage?.let { " ($it)" } ?: ""}; list rows with hub sentiment: ${rows.count { (it.sentiment?.totalSamples ?: 0) > 0 }}")
        if (snap.warnings.isNotEmpty()) {
            appendLine("- Warnings:")
            snap.warnings.forEach { appendLine("  - $it") }
        }
        for (symbol in symbols) {
            appendLine()
            appendLine("### $symbol detail")
            try {
                val d = repo.detail(symbol, force = true)
                val f = d.row.forecast
                appendLine("- Price ${d.row.market?.price} ${snap.currency}, Bitpanda ${d.bitpandaPrices}")
                appendLine("- History: ${d.daily.size} daily, ${d.hourly.size} hourly; community votes up ${d.community?.sentimentVotesUpPct ?: "missing"}")
                appendLine("- Derivatives: funding ${d.derivatives?.fundingRate ?: "missing"} (${d.derivatives?.source ?: "-"}), long/short ${d.derivatives?.longShortRatio ?: "missing"}")
                f?.let {
                    appendLine("- Forecast: 24h ${Format.pct(it.h24.probabilityUp)} up (confidence ${Format.pct(it.h24.confidence)}), 7d ${Format.pct(it.d7.probabilityUp)} up, bottom ${it.bottom.score}/100 ${it.bottom.label}")
                    it.d7.backtest?.let { bt -> appendLine("- 7d backtest: ${Format.pct(bt.hitRate)} vs ${Format.pct(bt.baselineHitRate)} by chance on ~${bt.effectiveSamples} independent windows") }
                }
                d.row.sentiment?.let { appendLine(sentimentLine(it)) }
                d.warnings.forEach { appendLine("  - warning: $it") }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                appendLine("- **failed:** ${e.message}")
            }
        }
        rows.firstOrNull { it.asset.tier == dev.cryptopulse.core.model.Tier.DEX }?.let { r ->
            appendLine()
            appendLine("### DEX token detail: ${r.asset.symbol} (${r.asset.chainId})")
            try {
                val d = repo.detail(r.asset.symbol, force = true)
                appendLine("- Hourly candles: ${d.row.market?.sparkline7d?.size ?: 0}; safety ${d.row.safety?.source ?: "none"}; risk ${d.row.risk?.score} ${d.row.risk?.label}")
                d.row.forecast?.let { appendLine("- 24h: ${d.row.upOrNull(it.h24)?.let { p -> Format.pct(p) } ?: "no forecast"} (${it.h24.signals.firstOrNull()?.description ?: ""})") }
                d.row.sentiment?.let { appendLine(sentimentLine(it)) }
                d.warnings.forEach { appendLine("  - warning: $it") }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                appendLine("- **failed:** ${e.message}")
            }
        }
        appendLine()
        appendLine("### Market mood")
        try {
            appendLine(sentimentLine(app.sentiment.globalMood(force = true)))
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            appendLine("- **failed:** ${e.message}")
        }
    }

    private fun sentimentLine(s: SentimentSummary): String =
        "- Sentiment ${s.label} (${"%+.2f".format(s.composite)}, confidence ${Format.pct(s.confidence)}, ${s.totalSamples} samples): " +
            s.perSource.joinToString { "${it.source} ${it.status} ${it.sampleCount}" }
}
