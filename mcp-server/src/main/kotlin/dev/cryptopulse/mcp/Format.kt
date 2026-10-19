package dev.cryptopulse.mcp

import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.Direction
import dev.cryptopulse.core.model.Forecast
import dev.cryptopulse.core.model.HorizonForecast
import dev.cryptopulse.core.model.SentimentSummary
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.model.Tier
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

    /** Shown with every list of coins that Bitpanda does not list. */
    const val SPECULATIVE_NOTICE: String =
        "_Not on Bitpanda; information only. Most memecoins and DEX tokens lose most of their value: about 69% of pump.fun launches stop trading on " +
            "their first day and over 80% of new tokens are dead within two days (CoinGecko 2026). Risk scores use free contract checks; no red flag does not mean safe._"

    /** "Up" column text: a probability, or why there is none. */
    private fun up(r: AssetRow, h: HorizonForecast?): String = when {
        r.risk?.blocked == true -> "— (${r.risk!!.label.lowercase()})"
        h?.signals?.firstOrNull()?.name == "history" -> "— (too new)"
        else -> "${arrow(h)} ${pct(h?.probabilityUp)}"
    }

    private fun tierCell(r: AssetRow): String = when (r.asset.tier) {
        Tier.BITPANDA -> if (r.asset.isMeme) "Bitpanda · meme" else "Bitpanda"
        else -> r.asset.tier.label
    }

    fun overviewTable(rows: List<AssetRow>, currency: String): String = buildString {
        appendLine("| # | Coin | Where | Price ($currency) | 24h | 7d | Up 24h | Up 7d | Bottom | Risk | Sentiment |")
        appendLine("|---|------|-------|------|-----|----|--------|-------|--------|------|-----------|")
        rows.forEachIndexed { i, r ->
            val m = r.market
            val f = r.forecast
            append("| ${i + 1} | ${r.asset.name} (${r.asset.symbol}) | ${tierCell(r)} | ${Text.formatPrice(m?.bitpandaPrice ?: m?.price)} | ")
            append("${Text.formatPct(m?.change24hPct, 1)} | ${Text.formatPct(m?.change7dPct, 1)} | ")
            append("${up(r, f?.h24)} | ${up(r, f?.d7)} | ")
            append("${f?.bottom?.score?.takeIf { r.risk?.blocked != true } ?: "—"} | ${r.risk?.let { "${it.score} ${it.label}" } ?: "—"} | ")
            append("${r.sentiment?.let { "${it.label} (${it.totalSamples})" } ?: "—"} |")
            appendLine()
        }
    }

    fun radarTable(rows: List<AssetRow>, nowMs: Long): String = buildString {
        appendLine("| # | Token | Kind | Chain | Price | 1h | 24h | Liquidity | Vol 24h | Age | Buys/Sells 24h | Up 24h | Risk |")
        appendLine("|---|-------|------|-------|-------|----|-----|-----------|---------|-----|----------------|--------|------|")
        rows.forEachIndexed { i, r ->
            val m = r.market
            val d = r.dex
            append("| ${i + 1} | ${r.asset.name} (${r.asset.symbol}) | ${r.asset.tier.label}${if ("boosted" in r.asset.tags) " · boosted" else ""}${if ("pumpfun" in r.asset.tags) " · pump.fun" else ""} | ")
            append("${d?.chainId ?: "—"} | ${Text.formatPrice(m?.price)} | ${Text.formatPct(m?.change1hPct, 1)} | ${Text.formatPct(m?.change24hPct, 1)} | ")
            append("${usd(d?.liquidityUsd)} | ${usd(d?.volumeUsd?.h24 ?: m?.volume24h)} | ${age(d?.pairCreatedAtMs, nowMs)} | ")
            append("${d?.txnsH24?.let { "${it.buys}/${it.sells}" } ?: "—"} | ${up(r, r.forecast?.h24)} | ${r.risk?.let { "${it.score} ${it.label}" } ?: "—"} |")
            appendLine()
        }
    }

    fun usd(v: Double?): String = when {
        v == null -> "—"
        v >= 1e9 -> "$%.1fB".format(v / 1e9)
        v >= 1e6 -> "$%.1fM".format(v / 1e6)
        v >= 1e3 -> "$%.0fK".format(v / 1e3)
        else -> "$%.0f".format(v)
    }

    fun age(createdMs: Long?, nowMs: Long): String {
        val h = createdMs?.let { (nowMs - it) / 3_600_000.0 } ?: return "—"
        return when {
            h < 1 -> "<1 h"
            h < 48 -> "${h.toInt()} h"
            else -> "${(h / 24).toInt()} d"
        }
    }

    fun forecast(name: String, f: Forecast, row: AssetRow? = null): String = buildString {
        appendLine("## $name (${f.symbol}) — ${Text.formatPrice(f.price, f.currency)}")
        appendLine(f.summary)
        appendLine()
        row?.takeIf { !it.asset.onBitpanda }?.let { r ->
            appendLine("### Not on Bitpanda: ${r.asset.tier.label}${r.risk?.let { " — risk ${it.score}/100, ${it.label.lowercase()}" } ?: ""}")
            r.risk?.reasons?.forEach { appendLine("- $it") }
            r.dex?.let { d ->
                appendLine("Trades on ${d.dexId ?: "a DEX"} (${d.chainId})${d.quoteSymbol?.let { " against $it" } ?: ""}: liquidity ${usd(d.liquidityUsd)}, 24 h volume ${usd(d.volumeUsd.h24)}, " +
                    "pool age ${age(d.pairCreatedAtMs, System.currentTimeMillis())}, 24 h trades ${d.txnsH24.buys} buys / ${d.txnsH24.sells} sells.")
                r.asset.contract?.let { appendLine("Contract: `$it`${d.url?.let { u -> " · $u" } ?: ""}") }
            }
            r.safety?.let { s ->
                val checks = listOfNotNull(
                    s.honeypot?.let { if (it) "HONEYPOT" else "not a honeypot" },
                    if (s.buyTaxPct != null || s.sellTaxPct != null) "tax ${"%.0f".format(s.buyTaxPct ?: 0.0)}% buy / ${"%.0f".format(s.sellTaxPct ?: 0.0)}% sell" else null,
                    s.mintable?.let { if (it) "mintable" else "no minting" },
                    s.freezable?.let { if (it) "freezable" else "no freeze authority" },
                    s.holderCount?.let { "$it holders" },
                    s.top10HolderPct?.let { "top 10 hold ${"%.0f".format(it)}%" },
                    s.cexListed.takeIf { it.isNotEmpty() }?.let { "also on ${it.joinToString()}" },
                )
                appendLine("Contract checks (${s.source}): ${checks.joinToString(", ")}${if (s.ownerPowers.isNotEmpty()) "; ${s.ownerPowers.joinToString()}" else ""}.")
            }
            appendLine(SPECULATIVE_NOTICE)
            appendLine()
        }
        for (h in listOf(f.h24, f.d7)) {
            appendLine("### Next ${h.horizon.label}: ${arrow(h)} ${pct(h.probabilityUp)} chance of going up (confidence ${pct(h.confidence)})")
            if (h.expectedLow != null && h.expectedHigh != null) {
                appendLine("Likely range: ${Text.formatPrice(h.expectedLow)} – ${Text.formatPrice(h.expectedHigh)} ${f.currency} (expected move ${Text.formatPct(h.expectedMovePct, 2)})")
            }
            h.backtest?.let { appendLine("Backtest: ${pct(it.hitRate)} correct direction on ${it.samples} overlapping past samples (about ${it.effectiveSamples} independent) vs ${pct(it.baselineHitRate)} by chance.") }
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
