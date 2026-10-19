package dev.cryptopulse.core.analysis.sentiment

import java.util.Collections

/**
 * The crypto slang overlay parsed from `crypto_lexicon.tsv` (format documented in the file header and
 * in `docs/SENTIMENT.md`).
 *
 * @property terms lexicon key -> valence, ready to overlay on the VADER lexicon: single tokens as-is,
 *   phrases under their underscore key ("rug pull" -> "rug_pull") and emoji under their token
 *   ("emoji_rocket"). Later lines win for duplicate keys, like the Python reader.
 * @property phrases multi-word terms in their space form, in file order (duplicates removed).
 * @property emojiTokens emoji -> token ("🚀" -> "emoji_rocket"), in file order.
 * @property entries every data line, for validation and tooling.
 */
data class CryptoLexicon(
    val terms: Map<String, Double>,
    val phrases: List<String>,
    val emojiTokens: Map<String, String>,
    val entries: List<Entry>,
) {
    /** One data line. [token] is set for emoji lines; [comment] for term lines that carry one. */
    data class Entry(
        val line: Int,
        val term: String,
        val valence: Double,
        val token: String? = null,
        val comment: String? = null,
    ) {
        val isEmoji: Boolean get() = token != null
        val isPhrase: Boolean get() = token == null && term.contains(' ')

        /** The key this entry occupies in the merged lexicon. */
        val lexiconKey: String get() = token ?: phraseKey(term)
    }

    companion object {
        const val MIN_VALENCE: Double = -4.0
        const val MAX_VALENCE: Double = 4.0
        const val EMOJI_TOKEN_PREFIX: String = "emoji_"

        private val EMOJI_TOKEN = Regex("emoji_[a-z0-9_]+")

        private val BUNDLED: CryptoLexicon by lazy { parse(SentimentResources.read(SentimentResources.CRYPTO_LEXICON)) }

        /** The bundled `sentiment/crypto_lexicon.tsv` (parsed once). */
        fun bundled(): CryptoLexicon = BUNDLED

        /** Lexicon key of a term: lowercase, spaces turned into underscores. */
        fun phraseKey(term: String): String = PyStr.lower(term).replace(' ', '_')

        /**
         * Parses the TSV format. Accepts exactly what the reference Python reader accepts and additionally
         * rejects malformed entries (bad valence, non-lowercase terms, irregular spacing, bad emoji token)
         * with the offending line number, so a broken edit fails fast in tests.
         */
        fun parse(content: String): CryptoLexicon {
            val terms = LinkedHashMap<String, Double>()
            val phrases = ArrayList<String>()
            val seenPhrases = HashSet<String>()
            val emoji = LinkedHashMap<String, String>()
            val entries = ArrayList<Entry>()

            content.split('\n').forEachIndexed { index, rawLine ->
                val lineNo = index + 1
                val line = rawLine.trimEnd('\r', '\n')
                if (PyStr.strip(line).isEmpty() || line.startsWith("#")) return@forEachIndexed
                val cols = line.split('\t')
                require(cols.size in 2..3) { "crypto lexicon line $lineNo: expected term<TAB>valence[<TAB>comment]" }
                val term = cols[0]
                val valence = cols[1].toDoubleOrNull()
                    ?: throw IllegalArgumentException("crypto lexicon line $lineNo: valence '${cols[1]}' is not a number")
                require(valence in MIN_VALENCE..MAX_VALENCE) {
                    "crypto lexicon line $lineNo: valence $valence outside [$MIN_VALENCE, $MAX_VALENCE]"
                }
                require(term.isNotEmpty()) { "crypto lexicon line $lineNo: empty term" }
                val extra = cols.getOrNull(2) ?: ""

                if (extra.startsWith(EMOJI_TOKEN_PREFIX)) {
                    require(EMOJI_TOKEN.matches(extra)) {
                        "crypto lexicon line $lineNo: emoji token '$extra' must match ${EMOJI_TOKEN.pattern}"
                    }
                    require(term.none { PyStr.isSpace(it) }) { "crypto lexicon line $lineNo: emoji contains whitespace" }
                    emoji[term] = extra
                    terms[extra] = valence
                    entries += Entry(lineNo, term, valence, token = extra)
                } else {
                    require(term == PyStr.lower(term)) { "crypto lexicon line $lineNo: term '$term' must be lowercase" }
                    require(term == PyStr.strip(term) && !term.contains("  ") && term.none { PyStr.isSpace(it) && it != ' ' }) {
                        "crypto lexicon line $lineNo: term '$term' must use single spaces between words"
                    }
                    if (term.contains(' ') && seenPhrases.add(term)) phrases += term
                    terms[phraseKey(term)] = valence
                    entries += Entry(lineNo, term, valence, comment = extra.ifEmpty { null })
                }
            }
            return CryptoLexicon(
                terms = Collections.unmodifiableMap(terms),
                phrases = Collections.unmodifiableList(phrases),
                emojiTokens = Collections.unmodifiableMap(emoji),
                entries = Collections.unmodifiableList(entries),
            )
        }
    }
}
