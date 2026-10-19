package dev.cryptopulse.core.analysis.sentiment

/** Classpath access to the bundled lexicons under `/sentiment/` (parsed lazily, once). */
internal object SentimentResources {
    const val VADER_LEXICON = "vader_lexicon.txt"
    const val VADER_EMOJI_LEXICON = "emoji_utf8_lexicon.txt"
    const val CRYPTO_LEXICON = "crypto_lexicon.tsv"

    val vaderLexicon: Map<String, Double> by lazy { Vader.parseLexicon(read(VADER_LEXICON)) }
    val vaderEmojiLexicon: Map<String, String> by lazy { Vader.parseEmojiLexicon(read(VADER_EMOJI_LEXICON)) }

    /** Reads `/sentiment/[name]` as strict UTF-8 (like Python's `codecs.open(..., encoding='utf-8')`). */
    fun read(name: String): String {
        val stream = SentimentResources::class.java.getResourceAsStream("/sentiment/$name")
            ?: error("missing classpath resource /sentiment/$name")
        val bytes = stream.use { it.readBytes() }
        val decoder = Charsets.UTF_8.newDecoder()
            .onMalformedInput(java.nio.charset.CodingErrorAction.REPORT)
            .onUnmappableCharacter(java.nio.charset.CodingErrorAction.REPORT)
        return decoder.decode(java.nio.ByteBuffer.wrap(bytes)).toString()
    }
}
