package dev.cryptopulse.mcp

import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.Direction
import dev.cryptopulse.core.model.Forecast
import dev.cryptopulse.core.model.HorizonForecast
import dev.cryptopulse.core.model.SentimentSummary
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.util.Text

/** Markdown helpers shared by the MCP tools and the Telegram digest. */
object Format {
    fun pct(p: Double?): String = if (p == null) "—" else "${(p * 100).toInt()}%"

    fun arrow(h: HorizonForecast?): String = when (h?.direction) {
        Direction.UP -> "▲"
        Direction.DOWN -> "▼"
        Direction.FLAT -> "■"
        null -> "·"
    }

    fun overviewTable(rows: List<AssetRow>, currency: String): String = buildString {
        appendLine("| # | Coin | Price ($currency) | 24h | 7d | Up 24h | Up 7d | Bottom | Sentiment |")
        appendLine("|---|------|------|-----|----|--------|-------|--------|-----------|")
        rows.forEachIndexed { i, r ->
            val m = r.market
            val f = r.forecast
            append("| ${i + 1} | ${r.asset.name} (${r.asset.symbol}) | ${Text.formatPrice(m?.bitpandaPrice ?: m?.price)} | ")
            append("${Text.formatPct(m?.change24hPct, 1)} | ${Text.formatPct(m?.change7dPct, 1)} | ")
            append("${arrow(f?.h24)} ${pct(f?.h24?.probabilityUp)} | ${arrow(f?.d7)} ${pct(f?.d7?.probabilityUp)} | ")
            append("${f?.bottom?.score ?: "—"} | ${r.sentiment?.let { "${it.label} (${it.totalSamples})" } ?: "—"} |")
            appendLine()
        }
    }

    fun forecast(name: String, f: Forecast): String = buildString {
        appendLine("## $name (${f.symbol}) — ${Text.formatPrice(f.price, f.currency)}")
        appendLine(f.summary)
        appendLine()
        for (h in listOf(f.h24, f.d7)) {
            appendLine("### Next ${h.horizon.label}: ${arrow(h)} ${pct(h.probabilityUp)} chance of going up (confidence ${pct(h.confidence)})")
            if (h.expectedLow != null && h.expectedHigh != null) {
                appendLine("Likely range: ${Text.formatPrice(h.expectedLow)} – ${Text.formatPrice(h.expectedHigh)} ${f.currency} (expected move ${Text.formatPct(h.expectedMovePct, 2)})")
            }
            h.backtest?.let { appendLine("Backtest: ${pct(it.hitRate)} correct direction on ${it.samples} past samples vs ${pct(it.baselineHitRate)} for the naive baseline.") }
            appendLine("Main drivers:")
            h.signals.take(7).forEach { s -> appendLine("- ${if (s.contribution >= 0) "+" else "−"} ${s.label}: ${s.description}") }
            appendLine()
        }
        val b = f.bottom
        appendLine("### Is this the bottom? Score ${b.score}/100 — ${b.label}")
        b.reasons.forEach { appendLine("- $it") }
        listOfNotNull(
            b.distanceTo7dLowPct?.let { "7-day low +${"%.1f".format(it)}%" },
            b.distanceTo30dLowPct?.let { "30-day low +${"%.1f".format(it)}%" },
            b.distanceTo1yLowPct?.let { "1-year low +${"%.1f".format(it)}%" },
            b.drawdownFromAthPct?.let { "all-time high ${"%.0f".format(it)}%" },
        ).takeIf { it.isNotEmpty() }?.let { appendLine("Distances: ${it.joinToString(" · ")}") }
        appendLine()
        if (f.coverage.notes.isNotEmpty()) appendLine("Data notes: ${f.coverage.notes.joinToString("; ")}")
        appendLine("_${Forecast.DISCLAIMER}_")
    }

    fun sentiment(s: SentimentSummary): String = buildString {
        appendLine("## Sentiment for ${s.symbol}: ${s.label} (${"%+.2f".format(s.composite)}, confidence ${pct(s.confidence)}, ${s.totalSamples} items)")
        for (src in s.perSource) {
            val status = if (src.status == SourceStatus.OK) "${src.sampleCount} items, ${"%+.2f".format(src.composite)}, bullish ${pct(src.bullishShare)} / bearish ${pct(src.bearishShare)}" else src.status.name.lowercase().replace('_', ' ')
            appendLine("- **${src.source.name.lowercase().replace('_', ' ')}**: $status${src.note?.let { " — $it" } ?: ""}")
            if (src.modelScores.isNotEmpty()) appendLine("  - models: ${src.modelScores.entries.joinToString { "${it.key} ${"%+.2f".format(it.value)}" }}")
            src.topSamples.take(3).forEach { t -> appendLine("  - (${"%+.2f".format(t.score)}) ${t.text}${t.community?.let { c -> " — $c" } ?: ""}") }
        }
    }
}
