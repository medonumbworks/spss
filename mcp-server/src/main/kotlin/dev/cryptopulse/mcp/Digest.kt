package dev.cryptopulse.mcp

import dev.cryptopulse.core.analysis.risk.RiskScorer
import dev.cryptopulse.core.model.MarketSnapshot
import dev.cryptopulse.core.util.Text
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

/** Short market digest (English or Croatian) for the Telegram notifier and the `daily_digest` tool. */
object Digest {
    fun build(snap: MarketSnapshot, lang: String = "en", top: Int = 5): String {
        val hr = lang.lowercase().startsWith("hr")
        // the main lists are about coins one can buy on Bitpanda; speculative tiers get their own section
        val all = snap.rows.filter { it.market?.price != null && it.forecast != null && it.forecast!!.bottom.label != "Pegged" }
        val rows = all.filter { it.asset.onBitpanda }
        val spec = all.filter { !it.asset.onBitpanda }
        val time = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm 'UTC'").withZone(ZoneOffset.UTC).format(Instant.ofEpochMilli(snap.fetchedAtMs))
        val cur = snap.currency
        val ctx = snap.context
        fun t(en: String, hrText: String) = if (hr) hrText else en
        return buildString {
            appendLine("📊 CryptoPulse — $time ($cur)")
            ctx?.fearGreed?.let { appendLine(t("Fear & Greed", "Strah i pohlepa") + ": ${it.value} (${it.classification})") }
            snap.rows.firstOrNull { it.asset.symbol == "BTC" }?.market?.let { appendLine("BTC ${Text.formatPrice(it.bitpandaPrice ?: it.price, cur)} (${Text.formatPct(it.change24hPct, 1)} 24h)") }
            ctx?.global?.marketCapChange24hPct?.let { appendLine(t("Total market 24h", "Cijelo tržište 24h") + ": ${Text.formatPct(it, 1)}") }
            appendLine()
            appendLine(t("▲ Best odds to rise in 24h:", "▲ Najveća šansa rasta u 24h:"))
            rows.sortedByDescending { it.forecast!!.h24.probabilityUp * (0.5 + it.forecast!!.h24.confidence) }.take(top).forEach {
                appendLine("• ${it.asset.symbol} ${Format.pct(it.forecast!!.h24.probabilityUp)} · ${Text.formatPrice(it.market!!.bitpandaPrice ?: it.market!!.price)} ${Text.formatPct(it.market!!.change24hPct, 1)}")
            }
            appendLine(t("▲ Best odds to rise in 7 days:", "▲ Najveća šansa rasta u 7 dana:"))
            rows.sortedByDescending { it.forecast!!.d7.probabilityUp * (0.5 + it.forecast!!.d7.confidence) }.take(top).forEach {
                appendLine("• ${it.asset.symbol} ${Format.pct(it.forecast!!.d7.probabilityUp)}")
            }
            val bottoms = rows.filter { it.forecast!!.bottom.score >= 65 }.sortedByDescending { it.forecast!!.bottom.score }.take(top)
            if (bottoms.isNotEmpty()) {
                appendLine(t("⬇ Near local lows (bottom score):", "⬇ Blizu lokalnog dna (ocjena dna):"))
                bottoms.forEach { appendLine("• ${it.asset.symbol} ${it.forecast!!.bottom.score}/100, 7d ${Format.pct(it.forecast!!.d7.probabilityUp)}") }
            }
            appendLine(t("▼ Highest odds to fall in 24h:", "▼ Najveća šansa pada u 24h:"))
            rows.sortedBy { it.forecast!!.h24.probabilityUp }.take(3).forEach { appendLine("• ${it.asset.symbol} ${Format.pct(1 - it.forecast!!.h24.probabilityUp)}") }
            val movers = rows.filter { it.market!!.change24hPct != null }.sortedByDescending { kotlin.math.abs(it.market!!.change24hPct!!) }.take(3)
            if (movers.isNotEmpty()) appendLine(t("Biggest moves: ", "Najveći pomaci: ") + movers.joinToString(" · ") { "${it.asset.symbol} ${Text.formatPct(it.market!!.change24hPct, 1)}" })
            ctx?.globalSentiment?.let { appendLine(t("Social mood: ", "Raspoloženje na mrežama: ") + "${it.label} (${it.totalSamples})") }
            if (spec.isNotEmpty()) {
                appendLine()
                appendLine(t("🎲 Memecoins & DEX radar (not on Bitpanda, very high risk):", "🎲 Memecoini i DEX radar (nisu na Bitpandi, vrlo visok rizik):"))
                spec.filter { it.upOrNull(it.forecast!!.h24) != null }
                    .sortedByDescending { it.forecast!!.h24.probabilityUp * (0.5 + it.forecast!!.h24.confidence) }.take(3)
                    .forEach { appendLine("• ${it.asset.symbol} (${it.asset.tier.label}) ${Format.pct(it.forecast!!.h24.probabilityUp)} 24h · ${t("risk", "rizik")} ${it.risk?.score ?: "—"}/100") }
                spec.filter { it.dex != null }.sortedByDescending { it.dex!!.volumeUsd.h24 ?: 0.0 }.take(3).takeIf { it.isNotEmpty() }?.let { hot ->
                    appendLine(t("Most traded DEX tokens: ", "Najtrgovaniji DEX tokeni: ") + hot.joinToString(" · ") { "${it.asset.symbol} ${Text.formatPct(it.market?.change24hPct, 0)} (${it.risk?.label?.let { l -> riskLabel(l, hr) } ?: "?"})" })
                }
            }
            appendLine()
            append(t("Not financial advice. Probabilities, not certainties.", "Nije financijski savjet. Vjerojatnosti, ne sigurnosti."))
        }
    }

    private fun riskLabel(label: String, hr: Boolean): String = if (!hr) label.lowercase() else when (label) {
        RiskScorer.LABEL_LOWER -> "niži rizik"
        RiskScorer.LABEL_ELEVATED -> "povišen rizik"
        RiskScorer.LABEL_HIGH -> "visok rizik"
        RiskScorer.LABEL_EXTREME -> "ekstreman rizik"
        RiskScorer.LABEL_RED_FLAG -> "crvene zastavice"
        RiskScorer.LABEL_TOO_NEW -> "prenov"
        RiskScorer.LABEL_ILLIQUID -> "premala likvidnost"
        RiskScorer.LABEL_UNVERIFIED -> "neprovjeren"
        else -> label
    }
}
