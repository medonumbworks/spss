package dev.cryptopulse.core.analysis.sentiment

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CryptoLexiconTest {
    private val lexicon = CryptoLexicon.bundled()
    private val raw = SentimentResources.read(SentimentResources.CRYPTO_LEXICON)

    @Test
    fun `file documents its format in leading comment lines`() {
        val header = raw.lineSequence().takeWhile { it.startsWith("#") || it.isEmpty() }.joinToString("\n")
        assertTrue(raw.startsWith("# "))
        for (needle in listOf("term<TAB>valence", "EMOJI<TAB>valence<TAB>emoji_<name>", "[-4, 4]", "rug_pull", "def load_crypto_lexicon")) {
            assertTrue(header.contains(needle), "header should mention '$needle'")
        }
        assertFalse(raw.contains('\r'), "LF line endings only, so every reader splits lines identically")
    }

    @Test
    fun `has at least 250 well formed entries`() {
        assertTrue(lexicon.entries.size >= 250, "entries: ${lexicon.entries.size}")
        for (e in lexicon.entries) {
            assertTrue(e.valence in -4.0..4.0, "line ${e.line}")
            if (!e.isEmoji) {
                assertEquals(e.term.lowercase(), e.term, "line ${e.line}")
                assertFalse(e.term.contains("  ") || e.term != e.term.trim(), "line ${e.line}")
            }
        }
        val dataLines = raw.lines().filter { it.isNotBlank() && !it.startsWith("#") }
        assertEquals(lexicon.entries.size, dataLines.size)
        assertTrue(dataLines.all { it.split('\t').size in 2..3 }, "2 or 3 tab separated columns")
    }

    @Test
    fun `no duplicate terms and no collisions with vader modifiers`() {
        val terms = lexicon.entries.map { it.term }
        assertEquals(terms.size, terms.toSet().size, "duplicates: ${terms.groupBy { it }.filterValues { it.size > 1 }.keys}")
        val keys = lexicon.entries.filter { !it.isEmoji }.map { it.lexiconKey }
        assertEquals(keys.size, keys.toSet().size)
        for (key in lexicon.terms.keys) {
            assertFalse(key in Vader.BOOSTER_DICT, "'$key' is a booster word; VADER would skip it")
            assertFalse(key in Vader.NEGATE, "'$key' is a negation word")
        }
    }

    @Test
    fun `covers the required crypto vocabulary`() {
        val required = listOf(
            "bullish", "bearish", "moon", "mooning", "to the moon", "pump", "dump", "dumping", "hodl", "fud", "fomo",
            "rekt", "rug", "rug pull", "rugged", "scam", "ponzi", "exploit", "hack", "hacked", "drained", "depeg",
            "liquidated", "liquidation", "capitulation", "breakout", "breakdown", "rally", "surge", "soar", "plunge",
            "crash", "dip", "buy the dip", "accumulate", "accumulation", "oversold", "overbought", "all time high",
            "ath", "new ath", "short squeeze", "dead cat bounce", "bagholder", "wagmi", "ngmi", "lfg", "sell-off",
            "selloff", "delisted", "delisting", "listing", "listed", "partnership", "mainnet", "upgrade", "airdrop",
            "token unlock", "etf approval", "etf approved", "etf rejected", "sec lawsuit", "insolvency", "bankrupt",
            "halving", "whale accumulation", "whales dumping", "bear market", "bull market", "bull run",
            "golden cross", "death cross", "higher highs", "lower lows", "undervalued", "overvalued", "adoption",
            "inflows", "outflows",
        )
        val terms = lexicon.entries.filter { !it.isEmoji }.associate { it.term to it.valence }
        val missing = required.filter { it !in terms }
        assertEquals(emptyList(), missing)
        val bullish = listOf("bullish", "to the moon", "hodl", "buy the dip", "new ath", "wagmi", "golden cross", "inflows", "etf approved")
        val bearish = listOf("bearish", "rug pull", "scam", "rekt", "dead cat bounce", "ngmi", "death cross", "outflows", "etf rejected")
        bullish.forEach { assertTrue(terms.getValue(it) > 0, it) }
        bearish.forEach { assertTrue(terms.getValue(it) < 0, it) }
    }

    @Test
    fun `covers the required emoji with emoji_ tokens`() {
        val required = listOf(
            "🚀", "🌙", "📈", "📉", "💎", "🙌", "🔥",
            "🐂", "🐻", "💀", "🩸", "⚠️", "🟢", "🔴",
            "💩", "🤡",
        )
        assertEquals(emptyList(), required.filter { it !in lexicon.emojiTokens })
        assertEquals("emoji_rocket", lexicon.emojiTokens["🚀"])
        assertEquals(2.8, lexicon.terms["emoji_rocket"])
        assertEquals(lexicon.emojiTokens["⚠️"], lexicon.emojiTokens["⚠"], "with and without U+FE0F")
        assertTrue(lexicon.emojiTokens.values.all { it.matches(Regex("emoji_[a-z0-9_]+")) })
        assertTrue(lexicon.terms.getValue("emoji_bear") < 0 && lexicon.terms.getValue("emoji_bull") > 0)
    }

    @Test
    fun `phrases are stored under underscore keys and listed in space form`() {
        assertEquals(-3.2, lexicon.terms["rug_pull"])
        assertTrue("rug pull" in lexicon.phrases)
        assertFalse("rug pull" in lexicon.terms)
        assertTrue(lexicon.phrases.all { it.contains(' ') })
        assertEquals(lexicon.entries.count { it.isPhrase }, lexicon.phrases.size)
        assertEquals(lexicon.entries.size - lexicon.emojiTokens.size + lexicon.emojiTokens.values.toSet().size, lexicon.terms.size)
    }

    @Test
    fun `overlapping phrases chain into lexicon entries unless listed as unscored`() {
        // '_' is not a word character for the phrase matcher (rule c), so phrases that share words chain
        // into ONE token: "strong buy" + "buy signal" -> "strong_buy_signal". That token scores only if it
        // is an entry itself. The chains below do not occur in real posts and are knowingly left unscored;
        // a new overlap fails this test until its combined phrase is added or listed here.
        val unscored = listOf(
            "bottom is in the green", "bottom is in the red", "de-peg restored", "frozen withdrawals halted",
            "frozen withdrawals paused", "funds stolen funds", "halted withdrawals halted", "halted withdrawals paused",
            "in the green candle", "in the green candles", "in the red candle", "in the red candles",
            "local bottom is in the green", "local bottom is in the red", "local top is in the green",
            "local top is in the red", "lost its peg restored", "number go up only", "open interest rate",
            "open interest rate cut", "open interest rate cuts", "open interest rate hike", "open interest rate hikes",
            "open interest rates", "panic sell signal", "paused withdrawals halted", "paused withdrawals paused",
            "stolen funds stolen", "strong buy the dip", "top is in the green", "top is in the red",
            "withdrawals halted withdrawals", "withdrawals paused withdrawals",
        )
        val hyphenated = lexicon.terms.keys.filter { '-' in it && '_' !in it && !it.startsWith(CryptoLexicon.EMOJI_TOKEN_PREFIX) }
        val chains = sortedSetOf<String>()
        for (a in lexicon.phrases) {
            val aw = a.split(' ')
            for (b in lexicon.phrases) {
                val bw = b.split(' ')
                for (k in 1 until minOf(aw.size, bw.size)) {
                    if (a != b && aw.takeLast(k) == bw.take(k)) chains += (aw + bw.drop(k)).joinToString(" ")
                }
            }
            // a phrase next to a hyphenated term that starts or ends with one of its words: "panic sell-off"
            for (t in hyphenated) {
                if (t.startsWith(aw.last() + "-")) chains += (aw.dropLast(1) + t).joinToString(" ")
                if (t.endsWith("-" + aw.first())) chains += (listOf(t) + aw.drop(1)).joinToString(" ")
            }
        }
        val vader = Vader.cryptoDefault()
        val lost = chains.filter { chain -> vader.analyze(chain).lexiconValences.any { it == null } }
        assertEquals(unscored, lost)
        for (scored in listOf("to the moon soon", "strong buy signal", "strong sell signal", "interest rate cut", "panic sell-off")) {
            assertTrue(scored in chains && scored in lexicon.phrases, scored)
        }
    }

    @Test
    fun `parsed collections are read-only so the shared instance cannot be altered`() {
        @Suppress("UNCHECKED_CAST")
        val terms = lexicon.terms as MutableMap<String, Double>
        assertFailsWith<UnsupportedOperationException> { terms["moon"] = -4.0 }
        assertFailsWith<UnsupportedOperationException> { (lexicon.phrases as MutableList<String>).clear() }
        assertFailsWith<UnsupportedOperationException> { (Vader.NEGATE as MutableSet<String>).clear() }
    }

    @Test
    fun `parser mirrors the python reader and rejects malformed lines`() {
        val parsed = CryptoLexicon.parse(
            "# comment\n\nmoon\t2.4\nrug pull\t-3.2\tcomment text\n🚀\t2.8\temoji_rocket\r\nmoon\t2.5\n",
        )
        assertEquals(mapOf("moon" to 2.5, "rug_pull" to -3.2, "emoji_rocket" to 2.8), parsed.terms)
        assertEquals(listOf("rug pull"), parsed.phrases)
        assertEquals(mapOf("🚀" to "emoji_rocket"), parsed.emojiTokens)
        assertEquals("comment text", parsed.entries[1].comment)

        assertFailsWith<IllegalArgumentException> { CryptoLexicon.parse("moon\n") }
        assertFailsWith<IllegalArgumentException> { CryptoLexicon.parse("moon\tbig\n") }
        assertFailsWith<IllegalArgumentException> { CryptoLexicon.parse("moon\t4.5\n") }
        assertFailsWith<IllegalArgumentException> { CryptoLexicon.parse("Moon\t2.0\n") }
        assertFailsWith<IllegalArgumentException> { CryptoLexicon.parse("rug  pull\t-3\n") }
        assertFailsWith<IllegalArgumentException> { CryptoLexicon.parse("🚀\t2.8\temoji_Rocket\n") }
        val error = assertFailsWith<IllegalArgumentException> { CryptoLexicon.parse("# c\nok\t1\nbad\n") }
        assertTrue(error.message!!.contains("line 3"))
    }
}
