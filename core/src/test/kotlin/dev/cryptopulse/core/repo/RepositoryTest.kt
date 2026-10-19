package dev.cryptopulse.core.repo

import dev.cryptopulse.core.CryptoPulse
import dev.cryptopulse.core.FIXTURE_NOW_MS
import dev.cryptopulse.core.analysis.SentimentScorer
import dev.cryptopulse.core.analysis.TextSentiment
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.fastLimiter
import dev.cryptopulse.core.http.InMemoryDiskStore
import dev.cryptopulse.core.model.Direction
import dev.cryptopulse.core.model.ListingSource
import dev.cryptopulse.core.model.PricePoint
import dev.cryptopulse.core.model.SentimentSource
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.testing.FakeTransport
import dev.cryptopulse.core.testing.Fixtures
import dev.cryptopulse.core.universe.BundledSnapshot
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.Logger
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class RepositoryTest {
    private val clock = Clock.fixed(FIXTURE_NOW_MS)
    private val scorer = SentimentScorer { t ->
        val s = t.lowercase()
        TextSentiment(if ("surge" in s || "bullish" in s || "rall" in s) 0.6 else if ("hack" in s || "outage" in s || "dump" in s) -0.6 else 0.0)
    }

    private fun fullTransport(feed: String? = Fixtures.read("sentiment_feed.json")) = FakeTransport()
        .onUrlContaining("api.bitpanda.com/v1/ticker", body = Fixtures.read("bitpanda_ticker.json"))
        .onUrlContaining("/coins/list", body = Fixtures.read("coingecko_coins_list.json"))
        .onUrlContaining("/coins/markets", body = Fixtures.read("coingecko_markets.json"))
        .onUrlContaining("/market_chart", body = Fixtures.read("coingecko_market_chart_btc.json"))
        .onUrlContaining("/global", body = Fixtures.read("coingecko_global.json"))
        .onUrlContaining("/search/trending", body = Fixtures.read("coingecko_trending.json"))
        .onUrlContaining("api.coingecko.com/api/v3/coins/", body = Fixtures.read("coingecko_coin_btc.json"))
        .onUrlContaining("alternative.me", body = Fixtures.read("fear_greed.json"))
        .onUrlContaining("premiumIndex", body = Fixtures.read("binance_premium.json"))
        .let { t -> if (feed == null) t.failUrlContaining("sentiment-feed/feed.json") else t.onUrlContaining("sentiment-feed/feed.json", body = feed) }
        .onUrlContaining("reddit.com", body = Fixtures.read("reddit_search.json"))
        .onUrlContaining("stocktwits.com", body = Fixtures.read("stocktwits_sol.json"))
        .onUrlContaining("t.me/s/", body = Fixtures.read("telegram_preview.html"))
        .onUrlContaining("news.google.com", body = Fixtures.read("rss_cointelegraph.xml"))
        .onUrlContaining("rss", body = Fixtures.read("rss_cointelegraph.xml"))
        .onUrlContaining("feed", body = Fixtures.read("atom_feed.xml"))

    private fun app(t: FakeTransport, store: InMemoryDiskStore? = null) =
        CryptoPulse.create(AppConfig(), t, store, scorer, clock = clock, logger = Logger.NONE, rateLimiter = fastLimiter(clock))

    @Test
    fun refreshBuildsTheMarketList() = runTest {
        val cp = app(fullTransport())
        val snap = cp.repository.refresh()
        assertEquals(ListingSource.BITPANDA_LEGACY_TICKER, snap.universeSource)
        assertEquals("BTC", snap.rows.first().asset.symbol, "sorted by market cap")
        val btc = snap.rows.first()
        assertEquals(72529.49, btc.market!!.bitpandaPrice)
        assertNotNull(btc.forecast)
        assertEquals(22, snap.context!!.fearGreed!!.value)
        val sol = cp.repository.row("SOL")!!
        assertEquals(SentimentSource.SENTIMENT_HUB, sol.sentiment!!.perSource.single { it.sampleCount > 0 }.source, "list uses the cloud hub")
        // share of all mentions: (120 / 420) / (60 / 370), not the raw 120 / 60
        assertEquals(1.762, sol.sentiment!!.perSource.first().buzzRatio!!, 1e-3)
        assertTrue(sol.forecast!!.h24.signals.any { it.name == "sentiment" })
        assertEquals(Direction.FLAT, cp.repository.row("USDT")?.forecast?.h24?.direction ?: Direction.FLAT)
        val pepe = cp.repository.row("PEPE")!!
        assertEquals(null, pepe.market?.coingeckoId?.takeIf { it != "pepe" }, "PEPE has no CoinGecko row in the fixture but keeps the Bitpanda price")
        assertEquals(0.0000035064, pepe.market!!.price)
    }

    @Test
    fun detailLoadsHistorySentimentAndDerivatives() = runTest {
        val cp = app(fullTransport())
        cp.repository.refresh()
        val d = cp.repository.detail("SOL")
        assertEquals(366, d.daily.size)
        assertEquals(168, d.hourly.size)
        assertEquals(78.5, d.community!!.sentimentVotesUpPct)
        assertEquals(-0.0002, d.derivatives!!.fundingRate)
        val s = d.row.sentiment!!
        val status = s.perSource.associate { it.source to it.status }
        assertEquals(SourceStatus.OK, status[SentimentSource.REDDIT])
        assertEquals(SourceStatus.OK, status[SentimentSource.STOCKTWITS])
        assertEquals(null, status[SentimentSource.TELEGRAM], "the fresh hub already scanned the public channels")
        assertEquals(SourceStatus.OK, status[SentimentSource.NEWS])
        assertEquals(SourceStatus.OK, status[SentimentSource.SENTIMENT_HUB])
        assertTrue(s.totalSamples > 120)
        assertTrue(d.row.forecast!!.d7.signals.any { it.name == "rsi14_daily" }, "daily history used for 7d")
        assertNotNull(d.row.forecast!!.d7.backtest)
        assertEquals(mapOf("EUR" to 92.74, "USD" to 104.2), d.bitpandaPrices)
    }

    @Test
    fun phoneScansTelegramItselfWithoutAHubFeed() = runTest {
        val cp = app(fullTransport(feed = null))
        cp.repository.refresh()
        val s = cp.repository.detail("SOL").row.sentiment!!
        assertEquals(SourceStatus.OK, s.perSource.single { it.source == SentimentSource.TELEGRAM }.status)
    }

    @Test
    fun oldHubFeedsFadeAndThenStopCounting() = runTest {
        val feed = Fixtures.read("sentiment_feed.json")
        fun withFeed(generatedAt: String) = fullTransport(feed.replace("2025-10-17T14:30:00Z", generatedAt))
        val tenHours = app(withFeed("2025-10-17T05:00:00Z")).repository.refresh().rows.single { it.asset.symbol == "SOL" }
        val hub = tenHours.sentiment!!.perSource.single { it.source == SentimentSource.SENTIMENT_HUB }
        assertTrue(hub.sampleCount in 1 until 120, "faded: ${hub.sampleCount}")
        assertEquals(null, hub.buzzRatio, "no day-over-day change from an old document")
        val threeDays = app(withFeed("2025-10-14T15:00:00Z")).repository.refresh().rows.single { it.asset.symbol == "SOL" }
        assertTrue(threeDays.sentiment?.perSource.orEmpty().none { it.source == SentimentSource.SENTIMENT_HUB })
    }

    @Test
    fun detailAfterACurrencySwitchNeverMixesCurrencies() = runTest {
        val t = fullTransport()
        val cp = app(t)
        cp.repository.refresh()
        cp.updateConfig(AppConfig(currency = "USD"))
        val d = cp.repository.detail("SOL")
        assertEquals("USD", d.row.market!!.currency)
        assertTrue(t.requests.any { "market_chart" in it.url && "vs_currency=usd" in it.url })
        assertFalse(t.requests.any { "market_chart" in it.url && "vs_currency=eur" in it.url })
    }

    @Test
    fun turningTheHubOffStopsUsingItsLastFeed() = runTest {
        val cp = app(fullTransport())
        cp.repository.refresh()
        cp.updateConfig(AppConfig(sentimentFeedUrl = ""))
        val s = cp.sentiment.forAsset(cp.repository.row("SOL")!!.asset, force = true)
        assertTrue(s.perSource.none { it.source == SentimentSource.SENTIMENT_HUB && it.sampleCount > 0 })
        assertEquals(SourceStatus.OK, s.perSource.single { it.source == SentimentSource.TELEGRAM }.status, "the phone scans Telegram itself again")
    }

    @Test
    fun dailyClosesKeepYesterdayNextToTheLiveSample() {
        val day = 86_400_000L
        val d0 = 20_000 * day
        val daily = MarketRepository.toDaily(
            listOf(PricePoint(d0, 100.0), PricePoint(d0 + day, 101.0), PricePoint(d0 + 2 * day, 102.0), PricePoint(d0 + 2 * day + 15 * 3_600_000L, 110.0)),
        )
        assertEquals(listOf(100.0, 101.0, 102.0, 110.0), daily.map { it.value })
    }

    @Test
    fun snapshotIsPersistedForColdStarts() = runTest {
        val store = InMemoryDiskStore()
        app(fullTransport(), store).repository.refresh()
        val cold = app(FakeTransport(), store)
        assertNotNull(cold.repository.snapshot.value)
        assertEquals("BTC", cold.repository.snapshot.value!!.rows.first().asset.symbol)
    }

    @Test
    fun offlineRefreshKeepsPreviousDataAndReportsError() = runTest {
        val store = InMemoryDiskStore()
        val cp = app(fullTransport(), store)
        val first = cp.repository.refresh()
        val offline = app(FakeTransport().failUrlContaining("http"), store)
        val again = offline.repository.refresh(force = true)
        assertEquals(first.rows.size, again.rows.size)
    }

    @Test
    fun unverifiedCoinsWithLargeCapsAreHiddenInSnapshotMode() = runTest {
        val verified = BundledSnapshot.load().filter { it.verified && it.coingeckoId != null }.take(25)
        val rows = verified.mapIndexed { i, a ->
            """{"id":"${a.coingeckoId}","symbol":"${a.symbol.lowercase()}","name":"${a.name}","current_price":1.0,"market_cap":${1e12 / (i + 1) / (i + 1)},"sparkline_in_7d":{"price":[1.0,1.01,1.02]}}"""
        } + """{"id":"gala","symbol":"gala","name":"GALA","current_price":0.02,"market_cap":5.0E10}""" +
            """{"id":"0x","symbol":"zrx","name":"0x Protocol","current_price":0.2,"market_cap":1.0E7}"""
        val t = FakeTransport()
            .failUrlContaining("bitpanda.com")
            .failUrlContaining("/coins/list")
            .onUrlContaining("/coins/markets", body = "[" + rows.joinToString(",") + "]")
        val snap = app(t).repository.refresh()
        assertEquals(ListingSource.BUNDLED_SNAPSHOT, snap.universeSource)
        val symbols = snap.rows.map { it.asset.symbol }.toSet()
        assertFalse("GALA" in symbols, "unverified with a big cap is not on Bitpanda")
        assertTrue("ZRX" in symbols, "unverified small cap stays")
        assertTrue(snap.warnings.any { it.contains("unconfirmed") })
    }
}
