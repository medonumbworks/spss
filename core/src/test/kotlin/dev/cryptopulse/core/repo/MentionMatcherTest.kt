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
        assertFalse(MentionMatcher(Asset("RSR", "Reserve Rights")).matches("Federal Reserve holds rates"))
        assertFalse(MentionMatcher(Asset("GRT", "The Graph")).matches("Look at the graph"))
        assertTrue(MentionMatcher(Asset("CRV", "Curve DAO Token")).matches("Curve Finance exploit drains pools"))
        assertTrue(MentionMatcher(Asset("PENDLE", "Pendle")).matches("Pendle TVL climbs to 5B"))
        assertTrue(MentionMatcher(Asset("DOGE", "Doge", "dogecoin")).matches("Dogecoin jumps 12%"))
        assertTrue(MentionMatcher(Asset("FLOKI", "FLOKI", "floki")).matches("Floki Inu launches a game"), "a real coin named like its ticker")
        val people = MentionMatcher(Asset("PEOPLE", "PEOPLE"))
        assertFalse(people.matches("So many people feel safe"), "an unnamed listing must not match the word")
        assertTrue(people.matches("PEOPLE rallies 30%"))
    }
}
