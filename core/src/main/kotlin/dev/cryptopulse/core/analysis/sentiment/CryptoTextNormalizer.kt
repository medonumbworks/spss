package dev.cryptopulse.core.analysis.sentiment

/**
 * Rewrites social/crypto text so VADER sees crypto slang as single lexicon tokens. A Python twin in the
 * cloud sentiment hub applies the same rules byte for byte, so they are part of the contract:
 *
 * 1. **URLs**: every match of `(https?://\S+|www\.\S+)` (case-sensitive) is replaced by one space.
 *    `\S` means "not Python whitespace" (`str.isspace()`), which this port spells out explicitly.
 * 2. **Cashtags/hashtags**: a `$` or `#` immediately followed by an ASCII letter is removed
 *    (`[$#](?=[A-Za-z])`): "$BTC" -> "BTC", "#Bitcoin" -> "Bitcoin"; "$100" and "#1" are kept.
 * 3. **Phrases** (multi-word lexicon terms), longest first (length in code points; ties keep the given
 *    order), one pass per phrase over the current text: every case-insensitive whole-word occurrence is
 *    replaced by the phrase lowercased with spaces turned into underscores ("Rug Pull" -> "rug_pull").
 *    Whole word means the character before and after is not `[A-Za-z0-9]` (or is the text boundary);
 *    the boundary test itself is case-sensitive ASCII. Python:
 *    `re.sub(r"(?<![A-Za-z0-9])(?i:" + re.escape(p) + r")(?![A-Za-z0-9])", lambda m: key, text)`.
 *    Because `_` is not in the boundary class, a later phrase can still match next to an earlier key.
 * 4. **Emoji**: each mapped emoji, longest key first (code points; ties keep map order), is replaced
 *    everywhere by `" " + token + " "` (plain substring replacement, e.g. "🚀" -> " emoji_rocket ").
 * 5. **Whitespace**: runs of Python whitespace collapse to one space and the result is trimmed
 *    (`" ".join(text.split())`).
 *
 * Case of every other word is preserved because VADER uses ALL CAPS emphasis. The normalizer does not
 * know the lexicon: callers pass the phrases (space form) and the emoji -> token map, normally from
 * [CryptoLexicon]. [Vader.cryptoDefault] applies it automatically; [Vader.default] never does.
 */
object CryptoTextNormalizer {
    private val URL = Regex("(https?://[^${PyStr.WHITESPACE_CLASS}]+|www\\.[^${PyStr.WHITESPACE_CLASS}]+)")
    private val TAG = Regex("[\$#](?=[A-Za-z])")

    /** One-shot normalization; phrase regexes are only built for phrases that can occur in [text]. */
    fun normalize(text: String, phrases: List<String>, emojiTokens: Map<String, String>): String =
        Compiled(phrases, emojiTokens, precompile = false).normalize(text)

    /** Pre-sorts and pre-compiles the rules for repeated use (what [Vader.cryptoDefault] holds). */
    fun compile(phrases: List<String>, emojiTokens: Map<String, String>): Compiled =
        Compiled(phrases, emojiTokens, precompile = true)

    /** Immutable, thread-safe compiled rule set. */
    class Compiled internal constructor(
        phrases: List<String>,
        emojiTokens: Map<String, String>,
        precompile: Boolean,
    ) {
        private class PhraseRule(val phrase: String, val key: String, val regex: Regex?)

        private val phraseRules: List<PhraseRule> = phrases
            .sortedByDescending { PyStr.len(it) }
            .map { p -> PhraseRule(p, CryptoLexicon.phraseKey(p), if (precompile) phraseRegex(p) else null) }

        private val emojiRules: List<Pair<String, String>> = emojiTokens.entries
            .sortedByDescending { PyStr.len(it.key) }
            .filter { it.key.isNotEmpty() }
            .map { it.key to " ${it.value} " }

        fun normalize(text: String): String {
            // a) URLs -> single space
            var t = if (text.contains("://") || text.contains("www.")) URL.replace(text, " ") else text
            // b) "$BTC" -> "BTC", "#Bitcoin" -> "Bitcoin"
            if (t.indexOf('$') >= 0 || t.indexOf('#') >= 0) t = TAG.replace(t, "")
            // c) phrases, longest first. The ignore-case substring test is a strict superset of what the
            //    regex can match (same per-character case folding), so skipping on a miss is exact.
            for (rule in phraseRules) {
                if (!t.contains(rule.phrase, ignoreCase = true)) continue
                val regex = rule.regex ?: phraseRegex(rule.phrase)
                t = regex.replace(t) { rule.key }
            }
            // d) emoji -> " emoji_<name> "
            for ((emoji, replacement) in emojiRules) t = t.replace(emoji, replacement)
            // e) collapse whitespace and trim
            return PyStr.collapseWhitespace(t)
        }
    }

    private fun phraseRegex(phrase: String): Regex =
        Regex("(?<![A-Za-z0-9])(?iu:${Regex.escape(phrase)})(?![A-Za-z0-9])")
}
