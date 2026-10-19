package dev.cryptopulse.core.analysis.sentiment

import java.util.Collections
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * VADER output, identical to the dict returned by `SentimentIntensityAnalyzer.polarity_scores`:
 * [neg], [neu] and [pos] are proportions rounded to 3 decimals and [compound] is the normalized
 * (-1..1) score rounded to 4 decimals, using Python's round-half-even on the exact binary value.
 */
data class VaderScores(
    val neg: Double,
    val neu: Double,
    val pos: Double,
    val compound: Double,
)

/**
 * One scored text with the details callers need to explain a score.
 *
 * @property text the text VADER actually scored, i.e. the input after this instance's preprocessor
 *   (the crypto normalizer for [Vader.cryptoDefault], unchanged for [Vader.default]).
 * @property tokens VADER's `words_and_emoticons` (after emoji -> description replacement).
 * @property lexiconValences for each token, the raw lexicon valence when VADER looked the token up in
 *   the lexicon and found it; `null` for tokens that are not in the lexicon and for tokens VADER skips
 *   as modifiers (booster words and "kind" in "kind of").
 */
data class VaderAnalysis(
    val text: String,
    val tokens: List<String>,
    val lexiconValences: List<Double?>,
    val scores: VaderScores,
)

/**
 * Faithful Kotlin port of `vaderSentiment.SentimentIntensityAnalyzer` 3.3.2 (C.J. Hutto, MIT licence,
 * see `sentiment/VADER_LICENSE.txt`). Every rule of the Python implementation is mirrored, including
 * its quirks, and `VaderParityTest` checks the output against the Python package:
 *
 * - emoji -> description replacement per code point (only single-code-point emoji keys can match, as
 *   in Python) with a space inserted before a description that follows a non-space character;
 * - `SentiText` tokenization: Python `str.split()` and `_strip_punc_if_word` (strip
 *   `string.punctuation`, keep the original token when 2 or fewer code points remain);
 * - `allcap_differential` and `C_INCR` (0.733) ALL CAPS emphasis for lexicon and booster words;
 * - `B_INCR`/`B_DECR` (+-0.293) boosters up to three words back, damped by 0.95 and 0.9;
 * - `N_SCALAR` (-0.74) negation via the `NEGATE` list and any word containing "n't", the
 *   "never so/this" (x1.25) and "without doubt" exceptions, "no" as negator vs. lexicon word, "least";
 * - "kind of" and booster words are skipped as modifiers; "but" halves earlier and multiplies later
 *   sentiments by 1.5 (with Python's `list.index` first-equal-value quirk);
 * - `SPECIAL_CASES` idioms and booster bi-/tri-grams; `SENTIMENT_LADEN_IDIOMS` exist but, as in 3.3.2,
 *   are never consulted ([sentimentLadenIdiomsCheck] is the unused port of that "future work");
 * - punctuation emphasis (`!` x0.292 up to 4, `?` 2-3 x0.18 or 0.96 for more), `_sift_sentiment_scores`,
 *   `normalize(alpha = 15)` and Python rounding.
 *
 * Instances are immutable and thread-safe: the lexicons are copied into private maps that are never
 * mutated after construction.
 *
 * **Which entry points normalize text?** [polarityScores] and [analyze] first apply this instance's
 * [preprocessor]: none for the constructor default and [default] (pure VADER), the
 * [CryptoTextNormalizer] for [cryptoDefault]. [rawPolarityScores] and [analyzeRaw] never preprocess.
 *
 * @param lexicon word -> valence, like `vader_lexicon.txt` (looked up with the token lowercased; keys
 *   containing uppercase letters can therefore never match, exactly as in Python).
 * @param emojiLexicon emoji -> textual description, like `emoji_utf8_lexicon.txt`.
 * @param preprocessor optional text rewrite applied by [polarityScores]/[analyze] before VADER runs.
 */
class Vader @JvmOverloads constructor(
    lexicon: Map<String, Double>,
    emojiLexicon: Map<String, String>,
    private val preprocessor: ((String) -> String)? = null,
) {
    private val lexicon: Map<String, Double> = Collections.unmodifiableMap(HashMap(lexicon))
    private val emojiByCodePoint: Map<Int, String>

    /** Lowest single-code-point emoji key; a cheap pre-check before the map lookup. */
    private val minEmojiCodePoint: Int

    init {
        // Python iterates the text one code point at a time and asks `chr in self.emojis`, so only
        // single-code-point keys can ever match; multi-code-point keys (ZWJ sequences, flags, keycaps,
        // emoji + U+FE0F) are unreachable in VADER 3.3.2 and are dropped here.
        val byCodePoint = HashMap<Int, String>()
        for ((emoji, description) in emojiLexicon) {
            if (emoji.isNotEmpty() && PyStr.len(emoji) == 1) byCodePoint[emoji.codePointAt(0)] = description
        }
        emojiByCodePoint = Collections.unmodifiableMap(byCodePoint)
        minEmojiCodePoint = byCodePoint.keys.minOrNull() ?: Int.MAX_VALUE
    }

    /** Number of entries in the (merged) lexicon. */
    val lexiconSize: Int get() = lexicon.size

    /** True when [polarityScores] rewrites the text before scoring (e.g. [cryptoDefault]). */
    val preprocesses: Boolean get() = preprocessor != null

    /** Raw lexicon valence for [token] as VADER looks it up (lowercased), or `null`. */
    fun valence(token: String): Double? = lexicon[PyStr.lower(token)]

    /** Applies this instance's preprocessor; identity when there is none. */
    fun preprocess(text: String): String = preprocessor?.invoke(text) ?: text

    /** VADER scores of [text] after this instance's preprocessor (see class docs). */
    fun polarityScores(text: String): VaderScores = analyzeRaw(preprocess(text)).scores

    /** VADER 3.3.2 `polarity_scores(text)` exactly, never preprocessing. */
    fun rawPolarityScores(text: String): VaderScores = analyzeRaw(text).scores

    /** Like [polarityScores] but also returns tokens and the lexicon hits. */
    fun analyze(text: String): VaderAnalysis = analyzeRaw(preprocess(text))

    /** Like [rawPolarityScores] but also returns tokens and the lexicon hits. */
    fun analyzeRaw(text: String): VaderAnalysis {
        // convert emojis to their textual descriptions
        val noEmoji = replaceEmojis(text)
        val words = wordsAndEmoticons(noEmoji)
        val n = words.size
        val lower = Array(n) { PyStr.lower(words[it]) }
        val isCapDiff = allCapDifferential(words)
        val sentiments = DoubleArray(n)
        val raw = arrayOfNulls<Double>(n)

        for (i in 0 until n) {
            val item = lower[i]
            // check for vader_lexicon words that may be used as modifiers or negations
            if (BOOSTER_DICT.containsKey(item)) continue
            if (i < n - 1 && item == "kind" && lower[i + 1] == "of") continue
            val lexValence = lexicon[item] ?: continue
            raw[i] = lexValence
            sentiments[i] = sentimentValence(lexValence, words, lower, isCapDiff, i)
        }

        butCheck(lower, sentiments)
        return VaderAnalysis(
            text = text,
            tokens = Collections.unmodifiableList(words),
            lexiconValences = Collections.unmodifiableList(raw.asList()),
            scores = scoreValence(sentiments, noEmoji),
        )
    }

    private fun replaceEmojis(text: String): String {
        var first = -1
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            if (cp >= minEmojiCodePoint && emojiByCodePoint.containsKey(cp)) {
                first = i
                break
            }
            i += Character.charCount(cp)
        }
        if (first < 0) return PyStr.strip(text)

        val sb = StringBuilder(text.length + 32)
        sb.append(text, 0, first)
        var prevSpace = first == 0 || text[first - 1] == ' '
        i = first
        while (i < text.length) {
            val cp = text.codePointAt(i)
            val size = Character.charCount(cp)
            val description = emojiByCodePoint[cp]
            if (description != null) {
                if (!prevSpace) sb.append(' ')
                sb.append(description)
                prevSpace = false
            } else {
                sb.append(text, i, i + size)
                prevSpace = cp == ' '.code
            }
            i += size
        }
        return PyStr.strip(sb.toString())
    }

    private fun sentimentValence(
        lexValence: Double,
        words: List<String>,
        lower: Array<String>,
        isCapDiff: Boolean,
        i: Int,
    ): Double {
        val n = words.size
        val itemLower = lower[i]
        var valence = lexValence

        // "no" as negation for an adjacent lexicon item vs "no" as its own stand-alone lexicon item
        if (itemLower == "no" && i != n - 1 && lexicon.containsKey(lower[i + 1])) valence = 0.0
        if ((i > 0 && lower[i - 1] == "no") ||
            (i > 1 && lower[i - 2] == "no") ||
            (i > 2 && lower[i - 3] == "no" && (lower[i - 1] == "or" || lower[i - 1] == "nor"))
        ) {
            valence = lexValence * N_SCALAR
        }

        // sentiment-laden word in ALL CAPS (while others aren't)
        if (PyStr.isUpper(words[i]) && isCapDiff) {
            if (valence > 0) valence += C_INCR else valence -= C_INCR
        }

        for (startI in 0 until 3) {
            // dampen the scalar modifier of preceding words based on their distance from the item
            val back = i - (startI + 1)
            if (i > startI && !lexicon.containsKey(lower[back])) {
                var s = scalarIncDec(words[back], lower[back], valence, isCapDiff)
                if (startI == 1 && s != 0.0) s *= 0.95
                if (startI == 2 && s != 0.0) s *= 0.9
                valence += s
                valence = negationCheck(valence, lower, startI, i)
                if (startI == 2) valence = specialIdiomsCheck(valence, lower, i)
            }
        }

        return leastCheck(valence, lower, i)
    }

    private fun leastCheck(valence: Double, lower: Array<String>, i: Int): Double {
        // negation case using "least"
        if (i > 1 && !lexicon.containsKey(lower[i - 1]) && lower[i - 1] == "least") {
            if (lower[i - 2] != "at" && lower[i - 2] != "very") return valence * N_SCALAR
        } else if (i > 0 && !lexicon.containsKey(lower[i - 1]) && lower[i - 1] == "least") {
            return valence * N_SCALAR
        }
        return valence
    }

    companion object {
        const val B_INCR: Double = 0.293
        const val B_DECR: Double = -0.293
        const val C_INCR: Double = 0.733
        const val N_SCALAR: Double = -0.74

        val NEGATE: Set<String> = Collections.unmodifiableSet(hashSetOf(
            "aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
            "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
            "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
            "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
            "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
            "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
            "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
            "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite",
        ))

        val BOOSTER_DICT: Map<String, Double> = Collections.unmodifiableMap(buildMap<String, Double> {
            listOf(
                "absolutely", "amazingly", "awfully", "completely", "considerable", "considerably",
                "decidedly", "deeply", "effing", "enormous", "enormously", "entirely", "especially",
                "exceptional", "exceptionally", "extreme", "extremely", "fabulously", "flipping", "flippin",
                "frackin", "fracking", "fricking", "frickin", "frigging", "friggin", "fully", "fuckin",
                "fucking", "fuggin", "fugging", "greatly", "hella", "highly", "hugely", "incredible",
                "incredibly", "intensely", "major", "majorly", "more", "most", "particularly", "purely",
                "quite", "really", "remarkably", "so", "substantially", "thoroughly", "total", "totally",
                "tremendous", "tremendously", "uber", "unbelievably", "unusually", "utter", "utterly", "very",
            ).forEach { put(it, B_INCR) }
            listOf(
                "almost", "barely", "hardly", "just enough", "kind of", "kinda", "kindof", "kind-of",
                "less", "little", "marginal", "marginally", "occasional", "occasionally", "partly",
                "scarce", "scarcely", "slight", "slightly", "somewhat", "sort of", "sorta", "sortof", "sort-of",
            ).forEach { put(it, B_DECR) }
        })

        /** Defined in 3.3.2 but never consulted by `polarity_scores` ("future work"). */
        val SENTIMENT_LADEN_IDIOMS: Map<String, Double> = Collections.unmodifiableMap(linkedMapOf(
            "cut the mustard" to 2.0, "hand to mouth" to -2.0, "back handed" to -2.0, "blow smoke" to -2.0,
            "blowing smoke" to -2.0, "upper hand" to 1.0, "break a leg" to 2.0, "cooking with gas" to 2.0,
            "in the black" to 2.0, "in the red" to -2.0, "on the ball" to 2.0, "under the weather" to -2.0,
        ))

        val SPECIAL_CASES: Map<String, Double> = Collections.unmodifiableMap(linkedMapOf(
            "the shit" to 3.0, "the bomb" to 3.0, "bad ass" to 1.5, "badass" to 1.5, "bus stop" to 0.0,
            "yeah right" to -2.0, "kiss of death" to -1.5, "to die for" to 3.0, "beating heart" to 3.5,
        ))

        private val DEFAULT: Vader by lazy {
            Vader(SentimentResources.vaderLexicon, SentimentResources.vaderEmojiLexicon)
        }

        private val CRYPTO_DEFAULT: Vader by lazy { crypto(CryptoLexicon.bundled()) }

        /**
         * Plain VADER 3.3.2 with the bundled `vader_lexicon.txt` and `emoji_utf8_lexicon.txt`; no text
         * normalization. Loaded once (lazily) and shared.
         */
        fun default(): Vader = DEFAULT

        /**
         * VADER with `crypto_lexicon.tsv` overlaid on the VADER lexicon (crypto values win; phrases under
         * their underscore keys, emoji under their `emoji_<name>` tokens). Its [polarityScores] and
         * [analyze] first run [CryptoTextNormalizer] with the crypto phrases and emoji. Loaded once and shared.
         */
        fun cryptoDefault(): Vader = CRYPTO_DEFAULT

        /** Builds a crypto-flavoured VADER from any [CryptoLexicon] (see [cryptoDefault]). */
        fun crypto(cryptoLexicon: CryptoLexicon): Vader {
            val merged = LinkedHashMap<String, Double>(SentimentResources.vaderLexicon)
            merged.putAll(cryptoLexicon.terms)
            val normalizer = CryptoTextNormalizer.compile(cryptoLexicon.phrases, cryptoLexicon.emojiTokens)
            return Vader(merged, SentimentResources.vaderEmojiLexicon, normalizer::normalize)
        }

        /** `make_lex_dict`: parses `vader_lexicon.txt` content (later duplicates win, like a Python dict). */
        fun parseLexicon(content: String): Map<String, Double> {
            val out = LinkedHashMap<String, Double>(8192)
            content.trimEnd('\n').split('\n').forEachIndexed { index, line ->
                if (line.isEmpty()) return@forEachIndexed
                val cols = PyStr.strip(line).split('\t')
                require(cols.size >= 2) { "vader lexicon line ${index + 1}: expected word<TAB>measure" }
                out[cols[0]] = PyStr.strip(cols[1]).toDouble()
            }
            return out
        }

        /** `make_emoji_dict`: parses `emoji_utf8_lexicon.txt` content. */
        fun parseEmojiLexicon(content: String): Map<String, String> {
            val out = LinkedHashMap<String, String>(4096)
            content.trimEnd('\n').split('\n').forEachIndexed { index, line ->
                val cols = PyStr.strip(line).split('\t')
                require(cols.size >= 2) { "emoji lexicon line ${index + 1}: expected emoji<TAB>description" }
                out[cols[0]] = cols[1]
            }
            return out
        }

        /** `normalize(score, alpha)`: maps an unbounded sum into -1..1. */
        fun normalize(score: Double, alpha: Double = 15.0): Double {
            val normScore = score / sqrt(score * score + alpha)
            return when {
                normScore < -1.0 -> -1.0
                normScore > 1.0 -> 1.0
                else -> normScore
            }
        }

        /** `negated([word])` for a single (already lowercased) word, with `include_nt=True`. */
        internal fun negated(wordLower: String): Boolean =
            NEGATE.contains(PyStr.lower(wordLower)) || wordLower.contains("n't")

        /** `SentiText._words_and_emoticons`. */
        internal fun wordsAndEmoticons(text: String): List<String> =
            PyStr.split(text).map(::stripPuncIfWord)

        /** `SentiText._strip_punc_if_word`. */
        internal fun stripPuncIfWord(token: String): String {
            var start = 0
            var end = token.length
            while (start < end && PyStr.isPunctuation(token[start])) start++
            while (end > start && PyStr.isPunctuation(token[end - 1])) end--
            if (start == 0 && end == token.length) return token
            val stripped = token.substring(start, end)
            return if (PyStr.len(stripped) <= 2) token else stripped
        }

        /** `allcap_differential`: true when some but not all tokens are ALL CAPS. */
        internal fun allCapDifferential(words: List<String>): Boolean {
            var allCapWords = 0
            for (w in words) if (PyStr.isUpper(w)) allCapWords++
            val capDifferential = words.size - allCapWords
            return capDifferential > 0 && capDifferential < words.size
        }

        /** `scalar_inc_dec`. */
        internal fun scalarIncDec(word: String, wordLower: String, valence: Double, isCapDiff: Boolean): Double {
            var scalar = 0.0
            val booster = BOOSTER_DICT[wordLower]
            if (booster != null) {
                scalar = booster
                if (valence < 0) scalar *= -1
                // booster/dampener word in ALL CAPS (while others aren't)
                if (PyStr.isUpper(word) && isCapDiff) {
                    if (valence > 0) scalar += C_INCR else scalar -= C_INCR
                }
            }
            return scalar
        }

        /** `_negation_check`. */
        internal fun negationCheck(valence: Double, lower: Array<String>, startI: Int, i: Int): Double {
            var v = valence
            when (startI) {
                0 -> if (negated(lower[i - 1])) v *= N_SCALAR // 1 word preceding the lexicon word
                1 -> when {
                    lower[i - 2] == "never" && (lower[i - 1] == "so" || lower[i - 1] == "this") -> v *= 1.25
                    lower[i - 2] == "without" && lower[i - 1] == "doubt" -> Unit
                    negated(lower[i - 2]) -> v *= N_SCALAR // 2 words preceding
                }
                2 -> when {
                    // Python precedence: (never and (so or this)) or (so or this) one word back.
                    (lower[i - 3] == "never" && (lower[i - 2] == "so" || lower[i - 2] == "this")) ||
                        (lower[i - 1] == "so" || lower[i - 1] == "this") -> v *= 1.25
                    lower[i - 3] == "without" && (lower[i - 2] == "doubt" || lower[i - 1] == "doubt") -> Unit
                    negated(lower[i - 3]) -> v *= N_SCALAR // 3 words preceding
                }
            }
            return v
        }

        /** `_special_idioms_check` (only reached with i >= 3). */
        internal fun specialIdiomsCheck(valence: Double, lower: Array<String>, i: Int): Double {
            var v = valence
            val n = lower.size
            val onezero = "${lower[i - 1]} ${lower[i]}"
            val twoonezero = "${lower[i - 2]} ${lower[i - 1]} ${lower[i]}"
            val twoone = "${lower[i - 2]} ${lower[i - 1]}"
            val threetwoone = "${lower[i - 3]} ${lower[i - 2]} ${lower[i - 1]}"
            val threetwo = "${lower[i - 3]} ${lower[i - 2]}"

            for (seq in arrayOf(onezero, twoonezero, twoone, threetwoone, threetwo)) {
                val special = SPECIAL_CASES[seq]
                if (special != null) {
                    v = special
                    break
                }
            }
            if (n - 1 > i) {
                SPECIAL_CASES["${lower[i]} ${lower[i + 1]}"]?.let { v = it }
            }
            if (n - 1 > i + 1) {
                SPECIAL_CASES["${lower[i]} ${lower[i + 1]} ${lower[i + 2]}"]?.let { v = it }
            }
            // booster/dampener bi-grams such as 'sort of' or 'kind of'
            for (nGram in arrayOf(threetwoone, threetwo, twoone)) {
                BOOSTER_DICT[nGram]?.let { v += it }
            }
            return v
        }

        /**
         * `_sentiment_laden_idioms_check`: ported for completeness only. vaderSentiment 3.3.2 never calls
         * it, so neither does [Vader]; calling it would break parity with the Python package.
         */
        internal fun sentimentLadenIdiomsCheck(valence: Double, sentiTextLower: String): Double {
            val idiomValences = SENTIMENT_LADEN_IDIOMS.filterKeys { sentiTextLower.contains(it) }.values
            return if (idiomValences.isEmpty()) valence else idiomValences.sum() / idiomValences.size
        }

        /**
         * `_but_check`, including Python's `sentiments.index(sentiment)` behaviour: each step rescales the
         * FIRST element equal to the current value, which is not always the current position.
         */
        internal fun butCheck(lower: Array<String>, sentiments: DoubleArray) {
            val bi = lower.indexOf("but")
            if (bi < 0) return
            for (k in sentiments.indices) {
                val sentiment = sentiments[k]
                var si = 0
                while (si < k && sentiments[si] != sentiment) si++
                if (si < bi) {
                    sentiments[si] = sentiment * 0.5
                } else if (si > bi) {
                    sentiments[si] = sentiment * 1.5
                }
            }
        }

        /** `_punctuation_emphasis` = `_amplify_ep` + `_amplify_qm`. */
        internal fun punctuationEmphasis(text: String): Double {
            var ep = 0
            var qm = 0
            for (c in text) {
                if (c == '!') ep++ else if (c == '?') qm++
            }
            if (ep > 4) ep = 4
            val epAmplifier = ep * 0.292
            val qmAmplifier = if (qm > 1) (if (qm <= 3) qm * 0.18 else 0.96) else 0.0
            return epAmplifier + qmAmplifier
        }

        /** `score_valence` including `_sift_sentiment_scores` and the final rounding. */
        internal fun scoreValence(sentiments: DoubleArray, text: String): VaderScores {
            if (sentiments.isEmpty()) return VaderScores(neg = 0.0, neu = 0.0, pos = 0.0, compound = 0.0)

            var sumS = PyStr.sum(sentiments)
            // compute and add emphasis from punctuation in text
            val punctEmphAmplifier = punctuationEmphasis(text)
            if (sumS > 0) sumS += punctEmphAmplifier else if (sumS < 0) sumS -= punctEmphAmplifier
            val compound = normalize(sumS)

            // discriminate between positive, negative and neutral sentiment scores
            var posSum = 0.0
            var negSum = 0.0
            var neuCount = 0
            for (s in sentiments) {
                if (s > 0) posSum += s + 1 // compensates for neutral words that are counted as 1
                if (s < 0) negSum += s - 1 // when used with abs(), compensates for neutrals
                if (s == 0.0) neuCount += 1
            }
            if (posSum > abs(negSum)) posSum += punctEmphAmplifier else if (posSum < abs(negSum)) negSum -= punctEmphAmplifier

            val total = posSum + abs(negSum) + neuCount
            val pos = abs(posSum / total)
            val neg = abs(negSum / total)
            val neu = abs(neuCount / total)
            return VaderScores(
                neg = PyStr.round(neg, 3),
                neu = PyStr.round(neu, 3),
                pos = PyStr.round(pos, 3),
                compound = PyStr.round(compound, 4),
            )
        }
    }
}
