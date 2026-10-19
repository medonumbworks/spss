package dev.cryptopulse.core.repo

import dev.cryptopulse.core.model.Asset
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MentionMatcherTest {
    @Test
    fun cashtagsNamesAndAmbiguousSymbols() {
        val sol = MentionMatcher(Asset("SOL", "Solana"))
        assertTrue(sol.matches("SOL is pumping"))
        assertTrue(sol.matches("I like solana"))
        assertFalse(sol.matches("console log"))
        val sun = MentionMatcher(Asset("SUN", "Sun (New)"))
        assertFalse(sun.matches("The sun is shining"), "ambiguous bare symbol")
        assertTrue(sun.matches("Bought more \$SUN today"))
        val s = MentionMatcher(Asset("S", "Sonic"))
        assertFalse(s.matches("It's a big S"))
        assertTrue(s.matches("Sonic network upgrade"))
        assertTrue(MentionMatcher(Asset("BEST", "Bitpanda Ecosystem Token")).matches("Bitpanda Ecosystem Token burn"))
        val xlm = MentionMatcher(Asset("XLM", "Stellar"))
        assertFalse(xlm.matches("A stellar quarter for tech stocks"))
        assertTrue(xlm.matches("Stellar network upgrade goes live"))
        assertTrue(xlm.matches("XLM breaks out"))
        val vsn = MentionMatcher(Asset("VSN", "Vision"))
        assertFalse(vsn.matches("Our vision for 2027"))
        assertTrue(vsn.matches("\$VSN and the Vision token merge"))
        assertFalse(MentionMatcher(Asset("PI", "Pi")).matches("pi is 3.14"))
    }
}
