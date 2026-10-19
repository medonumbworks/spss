package dev.cryptopulse.core.repo

import dev.cryptopulse.core.model.Asset
import java.util.Locale

/**
 * Decides whether a text talks about an asset.
 *  - `$SYM` cashtags always count.
 *  - The bare symbol counts unless it is ambiguous (<= 2 letters or an everyday word such as SUN, NEAR, LINK).
 *  - The name counts, except names that are everyday words ("Vision", "Sonic", "Stellar"): those only count
 *    when followed by a crypto context word ("Stellar network", "Vision token").
 *  - A name equal to an ambiguous symbol ("Sun" for SUN, "Pi" for PI) never counts on its own.
 * tools/sentiment_hub/assets.py mirrors these rules; keep both in sync.
 */
class MentionMatcher(asset: Asset) {
    private val symbol = asset.symbol
    private val ambiguous = symbol.length <= 2 || symbol in AMBIGUOUS_SYMBOLS
    private val cashtag = Regex("(?i)\\$" + Regex.escape(symbol) + "\\b")
    private val bareSymbol = Regex("\\b" + Regex.escape(symbol) + "\\b")
    /** An unresolved listing whose "name" is just its ticker ("PEOPLE"); the name would match the word. */
    private val placeholder = asset.coingeckoId == null && asset.name == asset.symbol
    private val names: List<Regex> = (buildNames(asset.name) + EXTRA_NAMES[symbol].orEmpty()).map { name ->
        if (name.lowercase(Locale.ROOT) in COMMON_WORD_NAMES) {
            Regex("(?i)\\b" + Regex.escape(name) + "\\s+(" + CONTEXT_WORDS + ")\\b")
        } else {
            Regex("(?i)\\b" + Regex.escape(name) + "\\b")
        }
    }

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
        if (cleaned.length >= 3 && !placeholder && !(ambiguous && cleaned.equals(symbol, ignoreCase = true))) out += cleaned
        // "Bitpanda Ecosystem Token" -> also "Bitpanda"; "Pyth Network" -> "Pyth"
        val words = cleaned.split(Regex("\\s+")).filter { it.isNotBlank() }
        if (words.size >= 2) {
            val first = words.first()
            if (first.length >= 4 && first.lowercase(Locale.ROOT) !in GENERIC_WORDS && !(ambiguous && first.equals(symbol, ignoreCase = true))) out += first
        }
        return out.toList()
    }

    private fun buildSearchQuery(asset: Asset): String {
        val cleaned = asset.name.replace(Regex("\\s*\\(.*?\\)\\s*"), " ").trim()
        val parts = linkedSetOf<String>()
        if (cleaned.isNotBlank()) parts += if (cleaned.lowercase(Locale.ROOT) in COMMON_WORD_NAMES) "\"$cleaned crypto\"" else "\"$cleaned\""
        if (!ambiguous) parts += asset.symbol
        return parts.joinToString(" OR ")
    }

    companion object {
        /** Symbols that collide with everyday words; bare mentions are ignored for these. */
        val AMBIGUOUS_SYMBOLS: Set<String> = setOf(
            "SUN", "GRASS", "VISION", "BEAM", "SOON", "KITE", "ONE", "FLOW", "NEAR", "PI", "LIT", "DEEP",
            "B", "M", "S", "A", "CC", "GT", "ME", "W", "OP", "AR", "ATH", "ZEN", "FF", "MET", "TEL", "USELESS",
            "PUMP", "AWE", "APE", "SUPER", "CAKE", "DASH", "NEO", "SAND", "MANA", "ORCA", "COMP", "ENS",
            "MOVE", "USUAL", "SPX", "GOAT", "ACT", "BAT", "TRUMP", "IP", "RAY", "JUP", "GRT", "ALGO", "LINK", "RENDER",
            "ASTER", "PLASMA", "MONAD", "SONIC", "MORPHO", "STRK", "ZK", "BONK", "WIF", "HOME", "ID", "NOT", "MEME",
            "BLAST", "CORE", "TREE", "ACX", "BAND", "MEW", "MOG", "BIO", "AVA", "WAL", "MON", "KAS", "ONDO", "CELO",
        )

        /** Asset names that are ordinary English words; they need a crypto context word to count. */
        val COMMON_WORD_NAMES: Set<String> = setOf(
            "vision", "soon", "kite", "plasma", "beam", "grass", "sonic", "lighter", "canton", "sun", "flare", "orca", "dash",
            "stellar", "cosmos", "polygon", "optimism", "render", "compound", "theta", "tron", "quant", "mantle", "aster",
            "stacks", "jupiter", "monad", "golem", "maple", "falcon",
            "magic", "core", "blast", "home", "tree", "band", "flow", "near", "pi",
            "useless", "useless coin", "osmosis", "harmony", "ontology", "waves",
            "curve", "stargate", "gate", "immutable", "movement", "story", "space", "book", "metal", "nano",
            "memecoin", "the graph",
        )
        private const val CONTEXT_WORDS = "token|tokens|coin|coins|network|protocol|price|crypto|chain|blockchain|ecosystem|foundation|labs|mainnet|staking|airdrop|etf|finance|dao|dex|swap"
        /** Never used as a first-word alias ("Reserve Rights" is not "the Federal Reserve"). */
        private val GENERIC_WORDS = setOf(
            "the", "network", "protocol", "token", "coin", "finance", "chain", "global", "official", "ethereum", "bitcoin", "standard",
            "internet", "basic", "trust", "eur", "usd", "reserve", "world", "yield", "across", "ripple", "defi",
        )
        /** Common names that differ from the listing name. */
        private val EXTRA_NAMES: Map<String, List<String>> = mapOf("DOGE" to listOf("Dogecoin"))
    }
}
