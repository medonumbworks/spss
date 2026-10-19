package dev.cryptopulse.core.sources

import dev.cryptopulse.core.model.SentimentSource
import dev.cryptopulse.core.sources.bitpanda.BitpandaApi
import dev.cryptopulse.core.sources.coingecko.CoinGeckoApi
import dev.cryptopulse.core.sources.feed.SentimentFeedApi
import dev.cryptopulse.core.sources.market.DerivativesApi
import dev.cryptopulse.core.sources.market.FearGreedApi
import dev.cryptopulse.core.sources.social.NewsFeeds
import dev.cryptopulse.core.sources.social.RedditApi
import dev.cryptopulse.core.sources.social.StockTwitsApi
import dev.cryptopulse.core.sources.social.TelegramWeb
import dev.cryptopulse.core.testing.Fixtures
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SourcesParsingTest {

    @Test
    fun bitpandaLegacyTickerParsesStringPrices() {
        val entries = BitpandaApi.parseLegacyTicker(Fixtures.read("bitpanda_ticker.json")).associateBy { it.symbol }
        assertEquals(72529.49, entries.getValue("BTC").prices.getValue("EUR"), 1e-9)
        assertEquals(0.0000035064, entries.getValue("PEPE").prices.getValue("EUR"), 1e-15)
        assertTrue("XAU" in entries && "XAU" in BitpandaApi.NON_CRYPTO_SYMBOLS)
    }

    @Test
    fun bitpandaPublicAssetsParsesCursor() {
        val (assets, cursor) = BitpandaApi.parsePublicAssets(Fixtures.read("bitpanda_public_assets_page1.json"))
        assertEquals(listOf("BTC", "ETH"), assets.map { it.symbol })
        assertEquals("abc", cursor)
        assertEquals("CRYPTO", assets.first().type)
        val (page2, end) = BitpandaApi.parsePublicAssets(Fixtures.read("bitpanda_public_assets_page2.json"))
        assertEquals("NEWCOIN", page2.single().symbol)
        assertNull(end)
    }

    @Test
    fun coingeckoMarketsParseEveryField() {
        val rows = CoinGeckoApi.parseMarkets(Fixtures.read("coingecko_markets.json"), "EUR").associateBy { it.coingeckoId }
        val btc = rows.getValue("bitcoin")
        assertEquals("BTC", btc.symbol)
        assertEquals(72529.49, btc.price)
        assertEquals(168, btc.sparkline7d.size)
        assertEquals(-12.0, btc.change30dPct)
        assertEquals(-38.5, btc.athChangePct)
        assertNotNull(btc.lastUpdatedMs)
        assertEquals(-1.69, rows.getValue("ethereum").change24hPct)
    }

    @Test
    fun coingeckoChartCoinGlobalTrending() {
        val chart = CoinGeckoApi.parseMarketChart(Fixtures.read("coingecko_market_chart_btc.json"))
        assertEquals(366, chart.prices.size)
        assertTrue(chart.prices.zipWithNext().all { (a, b) -> a.timeMs < b.timeMs })
        val coin = CoinGeckoApi.parseCoin("bitcoin", Fixtures.read("coingecko_coin_btc.json"))
        assertEquals(78.5, coin.sentimentVotesUpPct)
        assertEquals("http://www.bitcoin.org", coin.homepage)
        assertFalse(coin.description!!.contains("<a"))
    }

    @Test
    fun fearGreedSortsOldestFirst() {
        val fg = FearGreedApi.parse(Fixtures.read("fear_greed.json"))!!
        assertEquals(22, fg.value)
        assertEquals("Extreme Fear", fg.classification)
        assertEquals(8, fg.history.size)
        assertEquals(-33, fg.change7d)
    }

    @Test
    fun redditSkipsStickiedPosts() {
        val posts = RedditApi.parseListing(Fixtures.read("reddit_search.json"))
        assertEquals(listOf("abc1", "abc2"), posts.map { it.id })
        assertEquals("r/CryptoCurrency", posts.first().community)
        assertEquals(1_760_698_800_000L, posts.first().createdMs)
        assertTrue(posts.all { it.source == SentimentSource.REDDIT })
    }

    @Test
    fun stocktwitsKeepsExplicitLabels() {
        val posts = StockTwitsApi.parseStream(Fixtures.read("stocktwits_sol.json"), "SOL.X")
        assertEquals(listOf("Bullish", "Bearish", null), posts.map { it.explicitSentiment })
    }

    @Test
    fun telegramPreviewParsing() {
        val posts = TelegramWeb.parsePreview(Fixtures.read("telegram_preview.html"), "cointelegraph")
        assertEquals(listOf("cointelegraph/51001", "cointelegraph/51003"), posts.map { it.id }, "photo-only message is skipped")
        val first = posts.first()
        assertTrue(first.title.contains("surge Analysts") || first.title.contains("surge  Analysts") || first.title.contains("surge Analysts"), first.title)
        assertEquals(12_400, first.score)
        assertEquals("https://t.me/cointelegraph/51001", first.url)
        assertEquals("@cointelegraph", first.community)
        assertEquals(987, posts[1].score)
        assertEquals(3_100_000, TelegramWeb.parseViews("3.1M"))
    }

    @Test
    fun rssAndAtom() {
        val rss = NewsFeeds.parseFeed(Fixtures.read("rss_cointelegraph.xml"), "CoinTelegraph")
        assertEquals(2, rss.size)
        assertEquals("Bitcoin price rallies as whales accumulate", rss[0].title)
        assertEquals("BTC surged after inflows.", rss[0].summary)
        assertNotNull(rss[0].publishedMs)
        val atom = NewsFeeds.parseFeed(Fixtures.read("atom_feed.xml"), "Example")
        assertEquals("https://example.invalid/eth", atom.single().url)
    }

    @Test
    fun derivativesNormalizeContracts() {
        val map = DerivativesApi.parseBinancePremium(Fixtures.read("binance_premium.json"), 0)
        assertEquals(setOf("BTC", "PEPE", "SOL"), map.keys)
        assertEquals(0.0008, map.getValue("PEPE").fundingRate)
        assertEquals("BTC", DerivativesApi.baseOf("BTCUSDT"))
        assertEquals("PEPE", DerivativesApi.baseOf("1000PEPEUSDT"))
        assertNull(DerivativesApi.baseOf("ETHUSDC"))
        val bybit = DerivativesApi.parseBybitTickers(Fixtures.read("bybit_tickers.json"), 0)
        assertEquals(0.00012, bybit.getValue("BTC").fundingRate)
    }

    @Test
    fun sentimentFeedParses() {
        val doc = SentimentFeedApi.parse(Fixtures.read("sentiment_feed.json"))
        assertEquals(listOf("vader", "finbert", "cryptobert"), doc.models)
        assertTrue(doc.telegramAccountConnected)
        val sol = doc.assets.getValue("SOL")
        assertEquals(120, sol.mentions)
        assertEquals(0.45, sol.scores["cryptobert"])
        assertEquals(40 to 0.45, sol.bySource["telegram_account"])
        assertEquals(1, sol.samples.size)
        assertNotNull(sol.samples.first().tsMs)
        assertEquals(1650, doc.market!!.mentions)
    }
}
