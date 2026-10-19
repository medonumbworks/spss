package dev.cryptopulse.core.universe

import dev.cryptopulse.core.FIXTURE_NOW_MS
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.ListingSource
import dev.cryptopulse.core.sources.CoinListEntry
import dev.cryptopulse.core.sources.bitpanda.BitpandaApi
import dev.cryptopulse.core.sources.coingecko.CoinGeckoApi
import dev.cryptopulse.core.testHttp
import dev.cryptopulse.core.testing.FakeTransport
import dev.cryptopulse.core.testing.Fixtures
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.Logger
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class UniverseTest {
    private val snapshot = listOf(
        Asset("BTC", "Bitcoin", "bitcoin", snapshotRank = 1),
        Asset("BEST", "Bitpanda Ecosystem Token", "bitpanda-ecosystem-token", snapshotRank = 88),
        Asset("PEPE", "Pepe", "pepe", snapshotRank = 37),
        Asset("GALA", "Gala", "gala", verified = false),
    )

    private fun universe(t: FakeTransport): DefaultAssetUniverse {
        val http = testHttp(t)
        return DefaultAssetUniverse(BitpandaApi(http), CoinGeckoApi(http), { snapshot }, null, Clock.fixed(FIXTURE_NOW_MS), Logger.NONE)
    }

    @Test
    fun legacyTickerIsPreferredAndFiltered() = runTest {
        val t = FakeTransport()
            .onUrlContaining("api.bitpanda.com/v1/ticker", body = Fixtures.read("bitpanda_ticker.json"))
            .onUrlContaining("/coins/list", body = Fixtures.read("coingecko_coins_list.json"))
        val r = universe(t).load(AppConfig())
        assertEquals(ListingSource.BITPANDA_LEGACY_TICKER, r.source)
        val symbols = r.assets.map { it.symbol }.toSet()
        assertTrue("BTC" in symbols && "SOL" in symbols && "BEST" in symbols)
        assertFalse("XAU" in symbols, "metals excluded")
        assertFalse("AAPL" in symbols, "unknown non-crypto symbols excluded")
        assertEquals("pepe", r.bySymbol("PEPE")!!.coingeckoId)
        assertEquals("solana", r.bySymbol("SOL")!!.coingeckoId, "new symbol mapped through /coins/list")
        assertEquals(72529.49, r.bitpandaPrices.getValue("BTC").getValue("EUR"))
    }

    @Test
    fun publicApiFallbackFollowsCursor() = runTest {
        val t = FakeTransport()
            .onUrlContaining("api.bitpanda.com/v1/ticker", code = 401, body = "{}")
            .onUrlContaining("cursor=abc", body = Fixtures.read("bitpanda_public_assets_page2.json"))
            .onUrlContaining("api.public.bitpanda.com/v1/assets", body = Fixtures.read("bitpanda_public_assets_page1.json"))
            .onUrlContaining("/coins/list", body = Fixtures.read("coingecko_coins_list.json"))
        val r = universe(t).load(AppConfig())
        assertEquals(ListingSource.BITPANDA_PUBLIC_API, r.source)
        assertEquals(listOf("BTC", "ETH", "NEWCOIN"), r.assets.map { it.symbol })
        assertEquals("new-coin", r.bySymbol("NEWCOIN")!!.coingeckoId)
        assertTrue(r.assets.all { it.bitpandaAssetId != null })
    }

    @Test
    fun bundledSnapshotWhenBitpandaIsUnreachable() = runTest {
        val t = FakeTransport().failUrlContaining("bitpanda.com")
        val r = universe(t).load(AppConfig())
        assertEquals(ListingSource.BUNDLED_SNAPSHOT, r.source)
        assertEquals(4, r.assets.size)
        assertTrue(r.warnings.isNotEmpty())
    }

    @Test
    fun resolveIdsSkipsBridgedTokens() {
        val list = listOf(CoinListEntry("pepe-bridged-wormhole", "PEPE", "Pepe (Wormhole bridged)"), CoinListEntry("pepe", "PEPE", "Pepe"))
        val out = DefaultAssetUniverse.resolveIds(listOf(Asset("PEPE", "Pepe", "wrong-id")), list)
        assertEquals("pepe", out.single().coingeckoId)
    }

    @Test
    fun bundledResourceIsValid() {
        val assets = BundledSnapshot.load()
        assertTrue(assets.size >= 150)
        assertEquals("BTC", assets.first().symbol)
        assertTrue(assets.count { it.verified } >= 150)
        assertEquals(assets.size, assets.map { it.symbol }.toSet().size)
    }
}
