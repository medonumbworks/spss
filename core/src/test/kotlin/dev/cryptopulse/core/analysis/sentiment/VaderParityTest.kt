package dev.cryptopulse.core.analysis.sentiment

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * Parity with the Python package. `vader_reference.tsv` was produced by vaderSentiment 3.3.2
 * (`SentimentIntensityAnalyzer().polarity_scores`, CPython 3.13); `crypto_reference.tsv` by the Python
 * twin (same lexicon overlay + the documented normalizer rules). Every row must match.
 */
class VaderParityTest {
    private val tolerance = 1e-3

    private fun mismatch(expected: VaderScores, actual: VaderScores): Boolean =
        abs(expected.neg - actual.neg) > tolerance ||
            abs(expected.neu - actual.neu) > tolerance ||
            abs(expected.pos - actual.pos) > tolerance ||
            abs(expected.compound - actual.compound) > tolerance

    @Test
    fun `every reference sentence matches python vaderSentiment 3_3_2`() {
        val rows = ReferenceFixtures.vaderRows()
        assertTrue(rows.size >= 100, "need at least 100 reference rows, have ${rows.size}")
        val vader = Vader.default()
        val failures = rows.mapNotNull { row ->
            val actual = vader.polarityScores(row.sentence)
            if (mismatch(row.expected, actual)) "line ${row.line} '${row.sentence}': python=${row.expected} kotlin=$actual" else null
        }
        if (failures.isNotEmpty()) fail("${failures.size}/${rows.size} rows differ:\n" + failures.joinToString("\n"))
    }

    @Test
    fun `reference scores are reproduced exactly after python rounding`() {
        // Same algorithm, same IEEE arithmetic and the same rounding: not just within tolerance.
        val vader = Vader.default()
        val inexact = ReferenceFixtures.vaderRows().filter { vader.polarityScores(it.sentence) != it.expected }
        assertEquals(emptyList(), inexact.map { it.sentence })
    }

    @Test
    fun `reference set covers the required categories`() {
        val sentences = ReferenceFixtures.vaderRows().map { it.sentence }
        assertTrue("" in sentences, "empty string")
        assertTrue(sentences.any { it.contains('\t') } && sentences.any { it.contains('\n') }, "escaped tabs/newlines")
        assertTrue(sentences.any { it.contains("http") }, "URL")
        assertTrue(sentences.any { it.contains(" but ") }, "but")
        assertTrue(sentences.any { it.contains("kind of") }, "kind of")
        assertTrue(sentences.any { it.contains("!!!") } && sentences.any { it.contains("??") }, "punctuation")
        assertTrue(sentences.any { it.contains("😀") } && sentences.any { it.contains(":)") }, "emoji and emoticons")
        assertTrue(sentences.contains("VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!"), "README examples")
    }

    @Test
    fun `crypto default matches the python twin including the normalized text`() {
        val rows = ReferenceFixtures.cryptoRows()
        assertTrue(rows.size >= 50, "crypto rows: ${rows.size}")
        val vader = Vader.cryptoDefault()
        val failures = rows.mapNotNull { row ->
            val normalized = vader.preprocess(row.sentence)
            val actual = vader.polarityScores(row.sentence)
            when {
                normalized != row.normalized -> "line ${row.line} '${row.sentence}': normalized python='${row.normalized}' kotlin='$normalized'"
                mismatch(row.expected, actual) -> "line ${row.line} '${row.sentence}': python=${row.expected} kotlin=$actual"
                else -> null
            }
        }
        if (failures.isNotEmpty()) fail("${failures.size}/${rows.size} rows differ:\n" + failures.joinToString("\n"))
    }

    @Test
    fun `one-shot normalizer equals the compiled normalizer on every crypto fixture`() {
        val lexicon = CryptoLexicon.bundled()
        val compiled = CryptoTextNormalizer.compile(lexicon.phrases, lexicon.emojiTokens)
        for (row in ReferenceFixtures.cryptoRows()) {
            assertEquals(row.normalized, CryptoTextNormalizer.normalize(row.sentence, lexicon.phrases, lexicon.emojiTokens))
            assertEquals(row.normalized, compiled.normalize(row.sentence))
        }
    }
}

/** Reads the Python-generated parity fixtures under `src/test/resources/sentiment/`. */
private object ReferenceFixtures {
    data class VaderRow(val line: Int, val sentence: String, val expected: VaderScores)
    data class CryptoRow(val line: Int, val sentence: String, val normalized: String, val expected: VaderScores)

    fun vaderRows(): List<VaderRow> = rows("vader_reference.tsv", columns = 5).map { (line, c) ->
        VaderRow(line, unescape(c[0]), scores(c, 1))
    }

    fun cryptoRows(): List<CryptoRow> = rows("crypto_reference.tsv", columns = 6).map { (line, c) ->
        CryptoRow(line, unescape(c[0]), unescape(c[1]), scores(c, 2))
    }

    private fun scores(c: List<String>, from: Int) =
        VaderScores(neg = c[from].toDouble(), neu = c[from + 1].toDouble(), pos = c[from + 2].toDouble(), compound = c[from + 3].toDouble())

    private fun rows(name: String, columns: Int): List<Pair<Int, List<String>>> {
        val text = ReferenceFixtures::class.java.getResourceAsStream("/sentiment/$name")
            ?.use { it.readBytes().toString(Charsets.UTF_8) }
            ?: error("missing test resource /sentiment/$name")
        val lines = text.split('\n')
        require(lines.first().startsWith("sentence\t")) { "$name: missing header" }
        return lines.drop(1).mapIndexedNotNull { index, line ->
            if (line.isEmpty()) return@mapIndexedNotNull null
            val cols = line.split('\t')
            require(cols.size == columns) { "$name line ${index + 2}: expected $columns columns, got ${cols.size}" }
            (index + 2) to cols
        }
    }

    /** Reverses the generator's escaping: `\\` -> backslash, `\t`, `\n`, `\r`. */
    fun unescape(s: String): String {
        if (!s.contains('\\')) return s
        val sb = StringBuilder(s.length)
        var i = 0
        while (i < s.length) {
            val c = s[i]
            if (c == '\\' && i + 1 < s.length) {
                when (s[i + 1]) {
                    '\\' -> sb.append('\\')
                    't' -> sb.append('\t')
                    'n' -> sb.append('\n')
                    'r' -> sb.append('\r')
                    else -> error("bad escape \\${s[i + 1]} in '$s'")
                }
                i += 2
            } else {
                sb.append(c)
                i++
            }
        }
        return sb.toString()
    }
}
