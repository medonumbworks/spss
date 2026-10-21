package dev.cryptopulse.core.analysis.sentiment

import java.security.MessageDigest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

/** Rule-level checks of the port; expected values were taken from vaderSentiment 3.3.2 in CPython. */
class VaderRulesTest {

    private fun sha256(resource: String): String {
        val bytes = VaderRulesTest::class.java.getResourceAsStream("/sentiment/$resource")!!.use { it.readBytes() }
        return MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    }

    @Test
    fun `bundled lexicons are verbatim copies of the vaderSentiment 3_3_2 package files`() {
        assertEquals("1ec9c6e9ee19aade328f8beb393a6afa71a5bb3acf7d3cc22d4ef568df374bf5", sha256("vader_lexicon.txt"))
        assertEquals("b8d54223ae1ce22a3e12c1f745316b71678c328eb5f5d3063a842f37cfbe2823", sha256("emoji_utf8_lexicon.txt"))
        val license = SentimentResources.read("VADER_LICENSE.txt")
        assertTrue(license.contains("MIT License") && license.contains("C.J. Hutto"))
    }

    @Test
    fun `lexicon parsing mirrors make_lex_dict and make_emoji_dict`() {
        // Python: len(SentimentIntensityAnalyzer().lexicon) == 7506 (7520 lines, 14 duplicate keys), emojis == 3570
        assertEquals(7506, SentimentResources.vaderLexicon.size)
        assertEquals(3570, SentimentResources.vaderEmojiLexicon.size)
        assertEquals(-2.7, SentimentResources.vaderLexicon["scam"])
        assertEquals("rocket", SentimentResources.vaderEmojiLexicon["🚀"])

        val parsed = Vader.parseLexicon("good\t1.9\t0.9\t[1]\r\n  spaced \t 2.0\t0\r\ngood\t2.5\t0\t[]\r\n\n")
        // Python: "  spaced \t 2.0\t0".strip().split("\t") == ["spaced ", " 2.0", "0"]: only the line is stripped.
        assertEquals(mapOf("good" to 2.5, "spaced " to 2.0), parsed, "CRLF, strip() and later-duplicate-wins")
        assertFailsWith<IllegalArgumentException> { Vader.parseLexicon("nocolumns\n") }
        assertEquals(mapOf("😀" to "grinning face"), Vader.parseEmojiLexicon("😀\tgrinning face\r\n"))
    }

    @Test
    fun `uppercase lexicon keys and multi code point emoji keys are unreachable like in python`() {
        val vader = Vader(mapOf(":D" to 2.0, "red" to -1.0, "heart" to 3.0), mapOf("❤️" to "red heart"))
        assertEquals(0.0, vader.rawPolarityScores(":D").compound, "lookup lowercases the token, key ':D' never matches")
        // U+2764 U+FE0F is two code points; Python checks one code point at a time, so it never matches.
        assertEquals(listOf("I", "❤️"), vader.analyzeRaw("I ❤️").tokens)
    }

    @Test
    fun `emoji replacement inserts a space only after non-space characters`() {
        val vader = Vader.default()
        assertEquals(listOf("great", "grinning", "face"), vader.analyzeRaw("great😀").tokens)
        assertEquals(listOf("grinning", "facegreat"), vader.analyzeRaw("😀great").tokens)
        assertEquals(listOf("grinning", "face", "grinning", "face"), vader.analyzeRaw("😀😀").tokens)
    }

    @Test
    fun `tokenization follows str split and strip_punc_if_word`() {
        assertEquals(":)", Vader.stripPuncIfWord(":)"))
        assertEquals("ok!", Vader.stripPuncIfWord("ok!"), "2 or fewer code points left: keep the original token")
        assertEquals("good", Vader.stripPuncIfWord("good!!"))
        assertEquals("hello", Vader.stripPuncIfWord("'hello'"))
        assertEquals("don't", Vader.stripPuncIfWord("don't."))
        assertEquals("rug_pull", Vader.stripPuncIfWord("rug_pull,"))
        assertEquals(listOf("good", "job", "x​y"), Vader.wordsAndEmoticons(" good job　x​y\u001F "))
    }

    @Test
    fun `python isupper semantics`() {
        assertTrue(PyStr.isUpper("VADER!"))
        assertTrue(PyStr.isUpper(":D"))
        assertTrue(PyStr.isUpper("𝐆𝐑𝐄𝐀𝐓"), "mathematical bold GREAT")
        assertFalse(PyStr.isUpper("123"))
        assertFalse(PyStr.isUpper("I'll"))
        assertFalse(PyStr.isUpper("ǅ"), "titlecase letter")
        assertTrue(Vader.allCapDifferential(listOf("GREAT", "day")))
        assertFalse(Vader.allCapDifferential(listOf("GREAT", "DAY")))
        assertFalse(Vader.allCapDifferential(listOf("great", "day")))
    }

    @Test
    fun `but check reproduces the list index first-equal-value quirk`() {
        // Python: SentimentIntensityAnalyzer._but_check(["a","b","but","c"], [1.9, 0.95, 0, 2.0]) == [0.475, 0.95, 0, 3.0]
        val first = doubleArrayOf(1.9, 0.95, 0.0, 2.0)
        Vader.butCheck(arrayOf("a", "b", "but", "c"), first)
        assertContentEquals(doubleArrayOf(0.475, 0.95, 0.0, 3.0), first)
        // _but_check(["x","but","y","z","w"], [0, 0, 1.5, 2.25, 1.5]) == [0.0, 0, 3.375, 2.25, 2.25]
        val second = doubleArrayOf(0.0, 0.0, 1.5, 2.25, 1.5)
        Vader.butCheck(arrayOf("x", "but", "y", "z", "w"), second)
        assertContentEquals(doubleArrayOf(0.0, 0.0, 3.375, 2.25, 2.25), second)
    }

    @Test
    fun `punctuation emphasis and normalize`() {
        assertEquals(0.0, Vader.punctuationEmphasis("good?"))
        assertEquals(4 * 0.292, Vader.punctuationEmphasis("good!!!!!!!"))
        assertEquals(2 * 0.18, Vader.punctuationEmphasis("good??"))
        assertEquals(3 * 0.18 + 0.292, Vader.punctuationEmphasis("good?!??"))
        assertEquals(0.96, Vader.punctuationEmphasis("good????"))
        assertEquals(0.0, Vader.normalize(0.0))
        assertEquals(4.0 / Math.sqrt(31.0), Vader.normalize(4.0))
        assertTrue(Vader.normalize(-1e9) >= -1.0 && Vader.normalize(1e9) <= 1.0)
    }

    @Test
    fun `negation boosters caps and idioms behave like the reference`() {
        val v = Vader.default()
        // Values from vaderSentiment 3.3.2.
        assertEquals(0.4404, v.polarityScores("The book was good.").compound)
        assertEquals(-0.7424, v.polarityScores("VADER is not smart, handsome, nor funny.").compound)
        assertEquals(0.431, v.polarityScores("At least it isn't a horrible book.").compound)
        assertEquals(0.3832, v.polarityScores("The book was only kind of good.").compound)
        assertEquals(-0.7042, v.polarityScores("The plot was good, but the characters are uncompelling and the dialog is not great.").compound)
        assertEquals(0.9469, v.polarityScores("VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!").compound)
        assertEquals(0.5672, v.polarityScores("Sentiment analysis has never been this good!").compound)
        assertEquals(0.7013, v.polarityScores("Without a doubt, excellent idea.").compound)
        assertEquals(-0.1695, v.polarityScores("Roger Dodger is one of the least compelling variations on this theme.").compound)
    }

    @Test
    fun `sentiment laden idioms are ported but not consulted, as in 3_3_2`() {
        assertEquals(-2.0, Vader.sentimentLadenIdiomsCheck(0.0, "i feel under the weather"))
        assertEquals(1.5, Vader.sentimentLadenIdiomsCheck(1.5, "nothing idiomatic"))
        assertEquals(0.0, Vader.default().polarityScores("I feel under the weather today").compound)
    }

    @Test
    fun `python rounding and compensated sum`() {
        assertEquals(0.062, PyStr.round(0.0625, 3), "exact tie rounds half to even")
        assertEquals(0.064, PyStr.round(0.0635, 3), "0.0635 is slightly above the tie in binary")
        assertEquals(-0.0, PyStr.round(-1e-17, 4))
        assertEquals(0.6, PyStr.sum(doubleArrayOf(0.1, 0.2, 0.3)), "Neumaier: CPython 3.12+ sum() returns 0.6")
        assertNotEquals(0.6, 0.1 + 0.2 + 0.3)
    }

    @Test
    fun `empty and whitespace-only text score zero everywhere`() {
        val zero = VaderScores(0.0, 0.0, 0.0, 0.0)
        assertEquals(zero, Vader.default().polarityScores(""))
        assertEquals(zero, Vader.default().polarityScores(" \t\n "))
        assertEquals(zero, Vader.cryptoDefault().polarityScores(""))
        assertEquals(VaderScores(0.0, 1.0, 0.0, 0.0), Vader.default().polarityScores("12345"))
    }

    @Test
    fun `defaults are cached and only the crypto default normalizes`() {
        assertSame(Vader.default(), Vader.default())
        assertSame(Vader.cryptoDefault(), Vader.cryptoDefault())
        assertFalse(Vader.default().preprocesses)
        assertTrue(Vader.cryptoDefault().preprocesses)
        assertEquals("Rug Pull 🚀", Vader.default().preprocess("Rug Pull 🚀"))
        assertEquals("rug_pull emoji_rocket", Vader.cryptoDefault().preprocess("Rug Pull 🚀"))
        // rawPolarityScores never normalizes, even on the crypto instance.
        val crypto = Vader.cryptoDefault()
        assertEquals(crypto.polarityScores("rug_pull emoji_rocket"), crypto.polarityScores("Rug Pull 🚀"))
        assertNotEquals(crypto.rawPolarityScores("Rug Pull"), crypto.polarityScores("Rug Pull"))
    }

    @Test
    fun `crypto overlay wins over vader values and adds phrase and emoji keys`() {
        val plain = Vader.default()
        val crypto = Vader.cryptoDefault()
        assertEquals(-2.7, plain.valence("scam"))
        assertEquals(-3.0, crypto.valence("scam"))
        assertEquals(-1.9, plain.valence("block"))
        assertEquals(0.0, crypto.valence("block"))
        assertNull(plain.valence("rug_pull"))
        assertEquals(-3.2, crypto.valence("rug_pull"))
        assertEquals(2.8, crypto.valence("emoji_rocket"))
        assertEquals(2.3, crypto.valence("BULLISH"), "lookup lowercases")
        assertEquals(plain.valence("good"), crypto.valence("good"))
        assertTrue(crypto.lexiconSize > plain.lexiconSize)
    }

    @Test
    fun `lexicon valences in an analysis skip modifiers`() {
        val a = Vader.default().analyzeRaw("it is kind of very good")
        assertEquals(listOf("it", "is", "kind", "of", "very", "good"), a.tokens)
        assertEquals(listOf(null, null, null, null, null, 1.9), a.lexiconValences, "'kind' before 'of' and boosters are modifiers")
        assertEquals(2.4, Vader.default().analyzeRaw("she is kind").lexiconValences[2])
    }
}
