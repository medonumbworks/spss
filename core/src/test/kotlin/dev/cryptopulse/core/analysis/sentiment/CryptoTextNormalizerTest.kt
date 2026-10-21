package dev.cryptopulse.core.analysis.sentiment

import kotlin.test.Test
import kotlin.test.assertEquals

class CryptoTextNormalizerTest {
    private val phrases = listOf("rug pull", "to the moon", "buy the dip", "the dip", "new ath", "all-time high")
    private val emoji = linkedMapOf("🚀" to "emoji_rocket", "⚠️" to "emoji_warning", "⚠" to "emoji_warning")

    private fun norm(text: String) = CryptoTextNormalizer.normalize(text, phrases, emoji)

    @Test
    fun `a - urls are replaced by a single space`() {
        assertEquals("check and now", norm("check https://t.co/AbC123?x=1 and www.scam.io/path now"))
        assertEquals("see", norm("see http://example.com"))
        assertEquals("HTTPS://SHOUT.COM stays", norm("HTTPS://SHOUT.COM stays"), "the regex is case-sensitive")
        assertEquals("a b", norm("a https://x.y b"), "\\S stops at Python whitespace such as NBSP")
        assertEquals("example.com stays", norm("example.com stays"))
    }

    @Test
    fun `b - cashtags and hashtags lose their sigil only before a letter`() {
        assertEquals("BTC and Bitcoin", norm("\$BTC and #Bitcoin"))
        assertEquals("\$100 #1 \$ # \$\$ETH", norm("\$100 #1 \$ # \$\$\$ETH"))
        assertEquals("price \$5", norm("price \$5"))
    }

    @Test
    fun `c - phrases are matched case-insensitively on whole words and lowercased with underscores`() {
        assertEquals("this is a rug_pull!", norm("this is a Rug Pull!"))
        assertEquals("BTC to_the_moon", norm("BTC TO THE MOON"))
        assertEquals("rugpull rug pulled xrug pull", norm("rugpull rug pulled xrug pull"), "no partial-word matches")
        assertEquals("(rug_pull)", norm("(rug pull)"))
        assertEquals("rug pull2", norm("rug pull2"), "digits count as word characters")
        assertEquals("rug pull", norm("rug  pull"), "phrases need single spaces; whitespace is collapsed afterwards")
        assertEquals("all-time_high", norm("ALL-TIME HIGH"))
    }

    @Test
    fun `c - longest phrase wins and the underscore counts as a boundary`() {
        assertEquals("buy_the_dip", norm("Buy the dip"), "'buy the dip' before 'the dip'")
        assertEquals("another the_dip", norm("another the dip"))
        // 'the dip' is applied after 'buy the dip'; '_' is not [A-Za-z0-9], so cascades are possible:
        assertEquals("x_y", CryptoTextNormalizer.normalize("x y", listOf("x y"), emptyMap()))
        assertEquals(
            "in_the_red_candles",
            CryptoTextNormalizer.normalize("in the red candles", listOf("in the red", "red candles"), emptyMap()),
        )
    }

    @Test
    fun `c - phrase text is escaped and replacements are literal`() {
        val special = listOf("s&p 500", "c++ dev", "a.b c")
        assertEquals("s&p_500 up, c++_dev hired, axb c", CryptoTextNormalizer.normalize("S&P 500 up, C++ dev hired, axb c", special, emptyMap()))
        assertEquals("a.b_c", CryptoTextNormalizer.normalize("a.b c", special, emptyMap()))
    }

    @Test
    fun `d - mapped emoji become spaced tokens, longest key first`() {
        assertEquals("BTC emoji_rocket emoji_rocket", norm("BTC🚀🚀"))
        assertEquals("hacked emoji_warning", norm("hacked ⚠️"), "U+FE0F variant does not leave a stray selector")
        assertEquals("hacked emoji_warning", norm("hacked⚠"))
        assertEquals("😀 stays", norm("😀 stays"), "unmapped emoji are left for VADER's emoji lexicon")
    }

    @Test
    fun `e - whitespace collapses and case of other words is preserved`() {
        assertEquals("HODL the BTC now", norm("  HODL\t\tthe\n BTC now \r\n"))
        assertEquals("", norm(" \t "))
        assertEquals("", norm(""))
    }

    @Test
    fun `rules apply in order a to e`() {
        // URL removal happens first: the URL swallows "#rug", the remaining "pull" stays a plain word.
        assertEquals("pull rug_pull", norm("https://x.com/#rug pull rug pull"))
        // Tag removal happens before phrase matching: "#Rug Pull" -> "Rug Pull" -> "rug_pull".
        assertEquals("rug_pull emoji_rocket", norm("#Rug Pull🚀"))
    }

    @Test
    fun `bundled phrases and emoji normalize the spec examples`() {
        val lexicon = CryptoLexicon.bundled()
        val n = { t: String -> CryptoTextNormalizer.normalize(t, lexicon.phrases, lexicon.emojiTokens) }
        assertEquals("BTC to_the_moon emoji_rocket emoji_rocket", n("BTC to the moon 🚀🚀"))
        assertEquals("This is a rug_pull, total scam", n("This is a rug pull, total scam"))
        assertEquals("whales_dumping, death_cross forming", n("Whales dumping, death cross forming"))
        assertEquals("PEPE pumping emoji_fire", n("\$PEPE pumping 🔥"))
        assertEquals("HODL and buy_the_dip", n("HODL and buy the dip"))
    }
}
