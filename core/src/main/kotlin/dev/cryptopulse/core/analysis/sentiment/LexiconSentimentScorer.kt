package dev.cryptopulse.core.analysis.sentiment

import dev.cryptopulse.core.analysis.SentimentScorer
import dev.cryptopulse.core.analysis.TextSentiment

/**
 * On-device [SentimentScorer] backed by the VADER port.
 *
 * - [TextSentiment.score] is VADER's compound score (-1..1) of the text after `vader`'s preprocessor.
 *   With the default [Vader.cryptoDefault] that is the [CryptoTextNormalizer] output scored with the
 *   VADER + crypto slang lexicon; pass [Vader.default] for plain, un-normalized VADER.
 * - [TextSentiment.matchedPositive] / [TextSentiment.matchedNegative]: distinct lexicon tokens
 *   (lowercase, e.g. "bullish", "rug_pull", "emoji_rocket") whose raw lexicon valence is > 0 / < 0,
 *   in order of first appearance, at most [MAX_MATCHES] each. Raw valence means context rules
 *   (negation, "but", ...) do not move a word between the lists: "not bullish" still lists "bullish".
 * - [TextSentiment.hits]: number of sentiment-bearing tokens, counting repeats: tokens VADER looked up
 *   in the lexicon with a non-zero valence (booster words and "kind" in "kind of" are modifiers, not hits).
 *
 * Stateless apart from the immutable [Vader], so one instance can be shared across threads.
 */
class LexiconSentimentScorer(private val vader: Vader = Vader.cryptoDefault()) : SentimentScorer {

    override fun score(text: String): TextSentiment {
        val analysis = vader.analyze(text)
        val positive = LinkedHashSet<String>()
        val negative = LinkedHashSet<String>()
        var hits = 0
        for (i in analysis.tokens.indices) {
            val valence = analysis.lexiconValences[i] ?: continue
            if (valence == 0.0) continue
            hits++
            val token = PyStr.lower(analysis.tokens[i])
            if (valence > 0) {
                if (positive.size < MAX_MATCHES) positive.add(token)
            } else if (negative.size < MAX_MATCHES) {
                negative.add(token)
            }
        }
        return TextSentiment(
            score = analysis.scores.compound,
            matchedPositive = positive.toList(),
            matchedNegative = negative.toList(),
            hits = hits,
        )
    }

    companion object {
        const val MAX_MATCHES: Int = 8
    }
}
