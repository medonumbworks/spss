package dev.cryptopulse.core.analysis.sentiment

import dev.cryptopulse.core.analysis.TextSentiment
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LexiconSentimentScorerTest {
    private val scorer = LexiconSentimentScorer()

    private fun s(text: String): Double = scorer.score(text).score

    private fun assertAbove(threshold: Double, text: String) {
        val score = s(text)
        assertTrue(score > threshold, "'$text' scored $score, expected > $threshold")
    }

    private fun assertBelow(threshold: Double, text: String) {
        val score = s(text)
        assertTrue(score < threshold, "'$text' scored $score, expected < $threshold")
    }

    // --- crypto behaviour -------------------------------------------------------------------------

    @Test fun `moon with rockets is strongly positive`() = assertAbove(0.5, "BTC to the moon 🚀🚀")

    @Test fun `rug pull and scam is strongly negative`() = assertBelow(-0.5, "This is a rug pull, total scam")

    @Test fun `negated bullish is negative`() = assertBelow(0.0, "Not bullish on ETH at all")

    @Test fun `hodl and buy the dip is positive`() = assertAbove(0.0, "HODL and buy the dip")

    @Test fun `dumping and liquidated is negative`() = assertBelow(-0.4, "SOL dumping hard, got liquidated")

    @Test fun `listing with partnership is positive`() = assertAbove(0.0, "Bitpanda lists new token, partnership announced")

    @Test fun `whales dumping with death cross is negative`() = assertBelow(0.0, "Whales dumping, death cross forming")

    @Test fun `cashtag pumping with fire is positive`() = assertAbove(0.3, "\$PEPE pumping 🔥")

    @Test
    fun `neutral factual text stays near zero`() {
        listOf(
            "Bitcoin price is 65000 EUR today on Bitpanda.",
            "ETH trades at 3200 USD on Bitpanda",
            "Solana network processed 40 million transactions yesterday",
            "The block height reached 850000.",
            "The next Bitcoin block is expected in ten minutes",
            "BTC 24h volume was 32 billion dollars",
        ).forEach { assertTrue(abs(s(it)) < 0.2, "'$it' scored ${s(it)}") }
    }

    @Test
    fun `urls are ignored`() {
        assertEquals(scorer.score("Great news"), scorer.score("Great news https://example.com/scam-crash-rekt-dump"))
        assertEquals(0.0, s("www.rugpull-scam.io"))
    }

    @Test
    fun `phrases override their parts`() {
        assertBelow(0.0, "dead cat bounce")
        assertAbove(0.0, "nice bounce")
        assertBelow(0.0, "classic pump and dump")
        assertAbove(0.0, "pumping")
        assertAbove(0.0, "buy the dip")
        assertBelow(0.0, "another dip")
    }

    @Test fun `short squeeze is positive`() = assertAbove(0.3, "Short squeeze incoming 📈")

    @Test
    fun `new all time highs are positive`() {
        assertAbove(0.5, "new ATH!")
        assertAbove(0.5, "Bitcoin breaking out, new ATH")
        assertAbove(0.3, "all-time high again")
    }

    @Test
    fun `etf decisions`() {
        assertAbove(0.4, "ETF approved")
        assertAbove(0.4, "Spot ETF approval expected")
        assertBelow(-0.4, "ETF rejected")
    }

    @Test fun `sec lawsuit is negative`() = assertBelow(-0.3, "SEC lawsuit against the exchange")

    @Test fun `insolvency and halted withdrawals are strongly negative`() = assertBelow(-0.5, "Exchange is insolvent, withdrawals halted")

    @Test
    fun `bull and bear markets`() {
        assertAbove(0.3, "bull run starting")
        assertAbove(0.3, "bull market vibes")
        assertBelow(-0.3, "bear market blues")
        assertAbove(0.3, "🐂📈")
        assertBelow(-0.3, "🐻📉")
    }

    @Test
    fun `technical signals`() {
        assertAbove(0.0, "Golden cross on the daily")
        assertBelow(0.0, "Death cross on the daily")
        assertAbove(0.0, "higher highs and higher lows")
        assertBelow(0.0, "lower lows everywhere")
        assertAbove(0.0, "RSI oversold")
        assertBelow(0.0, "RSI overbought")
    }

    @Test
    fun `valuation and flows`() {
        assertAbove(0.0, "ETH looks undervalued")
        assertBelow(0.0, "DOGE is overvalued")
        assertAbove(0.0, "Record ETF inflows today")
        assertBelow(0.0, "Heavy outflows from exchanges")
    }

    @Test
    fun `security incidents are negative`() {
        assertBelow(-0.5, "Protocol hacked and drained ⚠️")
        assertBelow(-0.3, "USDC depeg")
        assertBelow(-0.3, "Bridge exploit, funds stolen")
        assertBelow(-0.3, "exit scam confirmed")
    }

    @Test
    fun `project events`() {
        assertAbove(0.0, "Airdrop confirmed")
        assertAbove(0.0, "Mainnet upgrade went live")
        assertBelow(0.0, "Token unlock next week")
        assertAbove(0.0, "Halving is coming")
        assertAbove(0.3, "Whale accumulation continues 💎🙌")
        assertBelow(0.0, "Token delisted from the exchange")
    }

    @Test
    fun `slang`() {
        assertAbove(0.3, "WAGMI")
        assertBelow(-0.3, "NGMI")
        assertAbove(0.5, "LFG 🚀")
        assertBelow(-0.3, "got rekt")
        assertBelow(0.0, "bagholder again 🤡")
        assertBelow(-0.3, "Capitulation everywhere 💀")
        assertBelow(-0.3, "Massive sell-off this morning")
        assertBelow(-0.3, "selloff deepens")
    }

    @Test
    fun `vader rules still apply to crypto terms`() {
        assertTrue(s("BULLISH on BTC") > s("bullish on btc"), "ALL CAPS emphasis")
        assertTrue(s("bullish!!!") > s("bullish"), "exclamation emphasis")
        assertTrue(s("very bullish") > s("bullish"), "booster")
        assertBelow(0.0, "Was bullish but now bearish")
        assertAbove(0.0, "No FUD, just HODL")
        assertAbove(0.3, "Never been this bullish")
    }

    @Test
    fun `case of phrases and tags does not matter`() {
        assertEquals(s("rug pull"), s("Rug Pull"))
        assertEquals(s("Bitcoin to the moon"), s("#Bitcoin to the moon"))
        assertEquals(s("BTC breaking out"), s("\$BTC breaking out"))
    }

    @Test
    fun `emoji signals`() {
        listOf("💀", "🩸", "🤡", "💩", "🔴", "⚠️", "⚠", "📉", "🐻")
            .forEach { assertBelow(0.0, it) }
        listOf("🚀", "🌙", "📈", "💎🙌", "🔥", "🐂", "🟢")
            .forEach { assertAbove(0.0, it) }
        assertEquals(s("⚠"), s("⚠️"))
    }

    // --- TextSentiment details -------------------------------------------------------------------

    @Test
    fun `matched tokens and hits`() {
        val result = scorer.score("BTC to the moon 🚀🚀, but FUD and scam everywhere")
        assertEquals(listOf("to_the_moon", "emoji_rocket"), result.matchedPositive)
        assertEquals(listOf("fud", "scam"), result.matchedNegative)
        assertEquals(5, result.hits)
        assertEquals(Vader.cryptoDefault().polarityScores("BTC to the moon 🚀🚀, but FUD and scam everywhere").compound, result.score)
    }

    @Test
    fun `matched lists use raw lexicon polarity and stay distinct`() {
        val negated = scorer.score("Not bullish")
        assertEquals(listOf("bullish"), negated.matchedPositive)
        assertTrue(negated.score < 0)
        val repeated = scorer.score("moon moon MOON")
        assertEquals(listOf("moon"), repeated.matchedPositive)
        assertEquals(3, repeated.hits)
    }

    @Test
    fun `matched lists keep at most eight tokens in order of appearance`() {
        val words = listOf("amazing", "great", "awesome", "excellent", "bullish", "moon", "rally", "surge", "soar", "wagmi", "lfg", "breakout")
        val result = scorer.score(words.joinToString(" ") + " scam rekt")
        assertEquals(words.take(8), result.matchedPositive)
        assertEquals(listOf("scam", "rekt"), result.matchedNegative)
        assertEquals(14, result.hits)
    }

    @Test
    fun `modifiers and neutral words are not hits`() {
        assertEquals(TextSentiment(score = 0.0, hits = 0), scorer.score(""))
        assertEquals(TextSentiment(score = 0.0, hits = 0), scorer.score("Bitcoin block 850000 mined"))
        val result = scorer.score("very kind of bullish")
        assertEquals(listOf("bullish"), result.matchedPositive)
        assertEquals(1, result.hits)
    }

    @Test
    fun `plain vader scorer does not normalize`() {
        val plain = LexiconSentimentScorer(Vader.default())
        assertEquals(Vader.default().polarityScores("rug pull 🚀").compound, plain.score("rug pull 🚀").score)
        assertTrue(plain.score("rug pull").matchedNegative.isEmpty())
        assertEquals(listOf("rug_pull"), scorer.score("rug pull").matchedNegative)
    }

    // --- concurrency and speed -------------------------------------------------------------------

    private fun corpus(n: Int): List<String> {
        val templates = listOf(
            "BTC to the moon 🚀 #%d", "This is a rug pull, total scam %d", "Not bullish on ETH at all, day %d",
            "HODL and buy the dip \$SOL %d", "Whales dumping, death cross forming on %d", "Bitcoin price is %d EUR today",
            "LFG!!! new ATH %d 🔥🔥", "Exchange hacked https://t.co/%d funds drained ⚠️",
            "kind of bearish but whales accumulating %d", "Partnership announced with %d banks, very bullish",
        )
        return List(n) { templates[it % templates.size].format(it) }
    }

    @Test
    fun `scoring is thread safe`() {
        val texts = corpus(2_000)
        val expected = texts.map { LexiconSentimentScorer().score(it) }
        val shared = LexiconSentimentScorer()
        val pool = Executors.newFixedThreadPool(8)
        try {
            val futures = (0 until 8).map { worker ->
                pool.submit(Callable { texts.indices.map { i -> (i + worker * 97) % texts.size }.associateWith { shared.score(texts[it]) } })
            }
            for (future in futures) {
                for ((i, result) in future.get(60, TimeUnit.SECONDS)) assertEquals(expected[i], result, texts[i])
            }
        } finally {
            pool.shutdownNow()
        }
    }

    @Test
    fun `scores ten thousand short texts in under three seconds`() {
        val texts = corpus(10_000)
        scorer.score("warm up the lazily loaded lexicons 🚀")
        val start = System.nanoTime()
        var checksum = 0.0
        for (t in texts) checksum += scorer.score(t).score
        val elapsedMs = (System.nanoTime() - start) / 1_000_000
        assertTrue(elapsedMs < 3_000, "10,000 texts took $elapsedMs ms")
        assertTrue(checksum.isFinite())
    }
}
