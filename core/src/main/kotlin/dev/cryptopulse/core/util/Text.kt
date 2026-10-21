package dev.cryptopulse.core.util

import java.util.Locale

object Text {
    /** Uppercase ASCII letters and digits only: "ether.fi" -> "ETHERFI", "1inch" -> "1INCH". */
    fun normalizeSymbol(raw: String): String = raw.trim().uppercase(Locale.ROOT).filter { it.isLetterOrDigit() }

    /** Lowercase letters and digits only, used for fuzzy name matching ("The Sandbox" -> "thesandbox"). */
    fun normalizeName(raw: String): String = raw.trim().lowercase(Locale.ROOT).filter { it.isLetterOrDigit() }

    fun truncate(text: String, max: Int): String =
        if (text.length <= max) text else text.take(max - 1).trimEnd() + "…"

    fun formatPct(value: Double?, digits: Int = 2): String =
        if (value == null || value.isNaN()) "—" else String.format(Locale.ROOT, "%+.${digits}f%%", value)

    fun formatPrice(value: Double?, currency: String = ""): String {
        if (value == null || value.isNaN()) return "—"
        val abs = kotlin.math.abs(value)
        val digits = when {
            abs >= 1000 -> 2
            abs >= 1 -> 2
            abs >= 0.01 -> 4
            abs >= 0.0001 -> 6
            else -> 10
        }
        val number = String.format(Locale.ROOT, "%,.${digits}f", value)
        return if (currency.isBlank()) number else "$number $currency"
    }

    fun formatCompact(value: Double?): String {
        if (value == null || value.isNaN()) return "—"
        val abs = kotlin.math.abs(value)
        return when {
            abs >= 1e12 -> String.format(Locale.ROOT, "%.2f T", value / 1e12)
            abs >= 1e9 -> String.format(Locale.ROOT, "%.2f B", value / 1e9)
            abs >= 1e6 -> String.format(Locale.ROOT, "%.2f M", value / 1e6)
            abs >= 1e3 -> String.format(Locale.ROOT, "%.1f K", value / 1e3)
            else -> String.format(Locale.ROOT, "%.2f", value)
        }
    }
}
