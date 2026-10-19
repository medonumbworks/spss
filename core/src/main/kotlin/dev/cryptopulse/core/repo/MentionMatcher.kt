package dev.cryptopulse.core.repo

import dev.cryptopulse.core.model.Asset
import java.util.Locale

/**
 * Decides whether a text talks about an asset. Symbols that are also ordinary words or are very
 * short ("S", "M", "PI", "SUN") only count as cashtags (`$SUN`) or via the full name.
 */
class MentionMatcher(asset: Asset) {
    private val symbol = asset.symbol
    private val ambiguous = symbol.length <= 2 || symbol in AMBIGUOUS_SYMBOLS
    private val cashtag = Regex("(?i)\\$" + Regex.escape(symbol) + "\\b")
    private val bareSymbol = Regex("\\b" + Regex.escape(symbol) + "\\b")
    private val names: List<Regex> = buildNames(asset.name).map { Regex("(?i)\\b" + Regex.escape(it) + "\\b") }

    fun matches(text: String): Boolean {
        if (text.isBlank()) return false
        if (cashtag.containsMatchIn(text)) return true
        if (names.any { it.containsMatchIn(text) }) return true
        if (ambiguous) return false
        return bareSymbol.containsMatchIn(text)
    }

    /** Query for site search engines (Reddit, Google News). */
    val searchQuery: String = buildSearchQuery(asset)

    private fun buildNames(name: String): List<String> {
        val cleaned = name.replace(Regex("\\s*\\(.*?\\)\\s*"), " ").trim()
        val out = linkedSetOf<String>()
        if (cleaned.length >= 3) out += cleaned
        // "Bitpanda Ecosystem Token" -> also "Bitpanda Ecosystem"; "Pyth Network" -> "Pyth"
        val words = cleaned.split(Regex("\\s+")).filter { it.isNotBlank() }
        if (words.size >= 2) {
            val first = words.first()
            if (first.length >= 4 && first.lowercase(Locale.ROOT) !in GENERIC_WORDS) out += first
        }
        return out.toList()
    }

    private fun buildSearchQuery(asset: Asset): String {
        val cleaned = asset.name.replace(Regex("\\s*\\(.*?\\)\\s*"), " ").trim()
        val parts = linkedSetOf<String>()
        if (cleaned.isNotBlank()) parts += "\"$cleaned\""
        if (!ambiguous) parts += asset.symbol
        return parts.joinToString(" OR ")
    }

    companion object {
        /** Symbols that collide with everyday words; bare mentions are ignored for these. */
        val AMBIGUOUS_SYMBOLS: Set<String> = setOf(
            "SUN", "GRASS", "VISION", "BEAM", "SOON", "KITE", "ONE", "FLOW", "NEAR", "PI", "LIT", "DEEP",
            "B", "M", "S", "A", "CC", "GT", "ME", "W", "OP", "AR", "ATH", "ZEN", "FF", "MET", "TEL", "USELESS",
            "PUMP", "AWE", "APE", "SUPER", "CAKE", "DASH", "NEO", "SAND", "MANA", "ORCA", "COMP", "ENS",
            "MOVE", "USUAL", "GOAT", "ACT", "BAT", "TRUMP", "IP", "RAY", "JUP", "GRT", "ALGO", "LINK", "RENDER",
            "ASTER", "PLASMA", "MONAD", "SONIC", "SOON", "MORPHO", "GRASS", "STRK", "ZK", "BONK", "WIF",
        )
        private val GENERIC_WORDS = setOf("the", "network", "protocol", "token", "coin", "finance", "chain", "global", "official")
    }
}
