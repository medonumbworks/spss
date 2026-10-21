package dev.cryptopulse.core.repo

import dev.cryptopulse.core.CryptoPulse
import dev.cryptopulse.core.FIXTURE_NOW_MS
import dev.cryptopulse.core.analysis.SentimentScorer
import dev.cryptopulse.core.analysis.TextSentiment
import dev.cryptopulse.core.analysis.risk.RiskScorer
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.fastLimiter
import dev.cryptopulse.core.http.HttpRequest
import dev.cryptopulse.core.http.HttpResponse
import dev.cryptopulse.core.http.HttpTransport
import dev.cryptopulse.core.http.InMemoryDiskStore
import dev.cryptopulse.core.model.Asset
import dev.cryptopulse.core.model.AssetRow
import dev.cryptopulse.core.model.DexInfo
import dev.cryptopulse.core.model.DexWindow
import dev.cryptopulse.core.model.ListingSource
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.MarketSnapshot
import dev.cryptopulse.core.model.Tier
import dev.cryptopulse.core.sources.dex.Candle
import dev.cryptopulse.core.testing.FakeTransport
import dev.cryptopulse.core.testing.Fixtures
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.Logger
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** The market list and the detail view with memecoin and DEX rows: contract checks, charts, identity, currencies. */
class SpeculativeRepositoryTest {
    private var nowMs = FIXTURE_NOW_MS
    private val clock = Clock { nowMs }
    private val scorer = SentimentScorer { TextSentiment(0.0) }

    /** USD -> EUR from the Bitpanda ticker fixture (BTC in both currencies). */
    private val fx = 72529.49 / 81500.12

    private val tokenA = "0x" + "a".repeat(40)
    private val tokenB = "0x" + "b".repeat(40)
    private val poolA = "0x" + "1".repeat(40)
    private val poolB = "0x" + "2".repeat(40)

    private val cleanGoPlus = Fixtures.read("spec/goplus_eth.json")
    private val honeypotGoPlus = cleanGoPlus.replace("\"is_honeypot\":\"0\"", "\"is_honeypot\":\"1\"").replace("\"sell_tax\":\"0\"", "\"sell_tax\":\"0.35\"")
    private val cleanSimulation = Fixtures.read("spec/honeypot_is.json")

    /** Mutable answers, read when a request arrives. */
    private var trending = pools(tokenA, poolA)
    private val goplus = HashMap<String, String>()
    private val simulation = HashMap<String, String>()
    private val candles = HashMap<String, String>()
    private var memecoinsUp = true

    /** FakeTransport plus work (a refresh, say) that runs while one request is in flight. */
    private class HookTransport(val fake: FakeTransport) : HttpTransport {
        var before: (suspend (HttpRequest) -> Unit)? = null
        override suspend fun execute(request: HttpRequest): HttpResponse {
            before?.invoke(request)
            return fake.execute(request)
        }
    }

    private fun answer(req: HttpRequest, byKey: Map<String, String>) =
        byKey.entries.firstOrNull { it.key in req.url }?.let { HttpResponse(200, it.value, emptyMap(), req.url) } ?: HttpResponse(404, "", emptyMap(), req.url)

    private fun transport(bitpanda: Boolean = true, marketChart: Boolean = true, klines: String? = null) = FakeTransport()
        .onUrlContaining("networks/trending_pools") { HttpResponse(200, trending, emptyMap(), it.url) }
        .onUrlContaining("/ohlcv/hour") { answer(it, candles) }
        .onUrlContaining("gopluslabs.io") { answer(it, goplus) }
        .onUrlContaining("honeypot.is") { answer(it, simulation) }
        .onUrlContaining("category=meme-token") { req ->
            if (memecoinsUp) HttpResponse(200, Fixtures.read("spec/coingecko_meme_markets.json"), emptyMap(), req.url) else HttpResponse(429, "", emptyMap(), req.url)
        }
        .onUrlContaining("premiumIndex", body = Fixtures.read("binance_premium.json").replace("]", """,{"symbol":"FROGUSDT","markPrice":"2.0","lastFundingRate":"0.0009"}]"""))
        .onUrlContaining("globalLongShortAccountRatio", body = """[{"longShortRatio":"2.5"}]""")
        .let { t -> if (klines != null) t.onUrlContaining("data-api.binance.vision/api/v3/klines", body = klines) else t }
        .let { t -> if (bitpanda) t.onUrlContaining("api.bitpanda.com/v1/ticker", body = Fixtures.read("bitpanda_ticker.json")) else t.failUrlContaining("bitpanda.com") }
        .let { t -> if (bitpanda) t.onUrlContaining("/coins/list", body = Fixtures.read("coingecko_coins_list.json")) else t.failUrlContaining("/coins/list") }
        .onUrlContaining("/coins/markets", body = Fixtures.read("coingecko_markets.json"))
        .let { t -> if (marketChart) t.onUrlContaining("/market_chart", body = Fixtures.read("coingecko_market_chart_btc.json")) else t.failUrlContaining("/market_chart") }
        .onUrlContaining("/global", body = Fixtures.read("coingecko_global.json"))
        .onUrlContaining("/search/trending", body = Fixtures.read("coingecko_trending.json"))
        .onUrlContaining("api.coingecko.com/api/v3/coins/", body = Fixtures.read("coingecko_coin_btc.json"))
        .onUrlContaining("alternative.me", body = Fixtures.read("fear_greed.json"))
        .onUrlContaining("sentiment-feed/feed.json", body = Fixtures.read("sentiment_feed.json"))
        .onUrlContaining("t.me/s/", body = Fixtures.read("telegram_preview.html"))

    private fun app(t: HttpTransport, config: AppConfig = AppConfig(showMemecoins = false, showTrending = false, showDexRadar = true), store: InMemoryDiskStore? = null) =
        CryptoPulse.create(config, t, store, scorer, clock = clock, logger = Logger.NONE, rateLimiter = fastLimiter(clock))

    @Test
    fun aCleanSimulationNeverClearsWhatGoPlusFound() = runTest {
        goplus[tokenA] = honeypotGoPlus
        simulation[tokenA] = cleanSimulation
        val cp = app(transport())
        cp.repository.refresh()
        val d = cp.repository.detail("FROG")
        val s = d.row.safety!!
        assertEquals(true, s.honeypot, "honeypot.is said no, GoPlus said yes")
        assertEquals(35.0, s.sellTaxPct!!, 1e-9, "the higher tax counts")
        assertEquals("goplus+honeypot.is", s.source)
        assertEquals(RiskScorer.LABEL_RED_FLAG, d.row.risk!!.label)
        assertNull(d.row.upOrNull(d.row.forecast?.h24))
        assertEquals(true, cp.repository.row("FROG")!!.safety!!.honeypot)
    }

    @Test
    fun aSimulationThatDidNotRunIsNotACheck() = runTest {
        goplus[tokenA] = """{"code":1,"message":"OK","result":{}}"""
        simulation[tokenA] = """{"token":{"name":"Frog","symbol":"FROG","totalHolders":12},"summary":{"risk":"unknown","flags":[]},"simulationSuccess":false,
            "simulationError":"no pair found","contractCode":{"openSource":true,"isProxy":false}}"""
        val cp = app(transport())
        cp.repository.refresh()
        val d = cp.repository.detail("FROG")
        assertNull(d.row.safety, "neither source checked the contract")
        assertEquals(RiskScorer.LABEL_UNVERIFIED, d.row.risk!!.label)
        assertTrue(d.warnings.any { it.startsWith("No free contract check") })
    }

    @Test
    fun theDexChartUsesTheExchangeRateAndTheNewerCandlePrice() = runTest {
        goplus[tokenA] = cleanGoPlus
        simulation[tokenA] = cleanSimulation
        val cp = app(transport())
        cp.repository.refresh()
        assertEquals(2.0 * fx, cp.repository.row("FROG")!!.market!!.price!!, 1e-9)
        nowMs += 5 * 60_000L
        // the token rose from 1 to 3 USD; the list still has the 2 USD of the last refresh
        val closes = (0 until 48).map { 1.0 + 2.0 * it / 47 }
        candles[poolA] = ohlcv(closes)
        val d = cp.repository.detail("FROG")
        val m = d.row.market!!
        assertEquals(closes.map { it * fx }, m.sparkline7d, "candles converted at the exchange rate, not scaled to the old price")
        assertEquals(3.0 * fx, m.price!!, 1e-9, "the candles are newer than the list price")
        assertEquals(nowMs, d.hourly.last().timeMs)
        assertEquals(3.0 * fx, d.hourly.last().value, 1e-9)
    }

    @Test
    fun quietHoursKeepTheLastClose() {
        val h = 3_600_000L
        fun c(hour: Long, close: Double) = Candle(hour * h, close, close, close, close, 1.0)
        val at = 1000 * h + 20 * 60_000L
        assertEquals(
            listOf(1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0),
            MarketRepository.hourlyGrid(listOf(c(1000, 3.0), c(990, 1.0), c(993, 2.0)), at),
            "a young pool starts at its first candle; hours without trades keep the last close",
        )
        // 2 candles a week apart: GeckoTerminal's last 168 candles of a thin pool can cover more than a week
        val week = MarketRepository.hourlyGrid(listOf(c(800, 5.0), c(900, 6.0)), at)
        assertEquals(168, week.size)
        assertEquals(5.0, week.first(), "the close before the window carries into it")
        assertEquals(5.0, week[900 - 833 - 1])
        assertEquals(6.0, week[900 - 833])
        assertEquals(6.0, week.last())
        assertEquals(listOf(4.0), MarketRepository.hourlyGrid(listOf(c(1001, 4.0)), at), "a candle ahead of the clock still counts")
    }

    @Test
    fun aTickerThatMovedToAnotherTokenGetsItsOwnDetail() = runTest {
        goplus[tokenA] = cleanGoPlus
        goplus[tokenB] = honeypotGoPlus
        candles[poolA] = ohlcv(List(48) { 2.0 })
        candles[poolB] = ohlcv(List(48) { 0.5 })
        val t = transport()
        val cp = app(t)
        cp.repository.refresh()
        assertEquals(tokenA, cp.repository.detail("FROG").row.asset.contract)
        nowMs += 11 * 60_000L
        trending = pools(tokenB, poolB, liquidityUsd = 900_000.0)
        cp.repository.refresh(force = true)
        val d = cp.repository.detail("FROG")
        assertEquals(tokenB, d.row.asset.contract, "not the cached detail of the token that had the ticker before")
        assertEquals(poolB, d.row.dex!!.pairAddress)
        assertEquals(true, d.row.safety!!.honeypot)
        assertEquals(1, t.countRequests("$poolB/ohlcv"))
    }

    @Test
    fun aRefreshDuringTheDetailNeverMixesTwoTokens() = runTest {
        goplus[tokenA] = cleanGoPlus
        goplus[tokenB] = honeypotGoPlus
        candles[poolA] = ohlcv(List(48) { 2.0 })
        candles[poolB] = ohlcv(List(48) { 0.5 })
        val t = HookTransport(transport())
        val cp = app(t)
        cp.repository.refresh()
        t.before = { req ->
            if ("$poolA/ohlcv" in req.url) {
                t.before = null
                // while the detail of token A loads, a refresh gives the ticker to token B
                nowMs += 11 * 60_000L
                trending = pools(tokenB, poolB, liquidityUsd = 900_000.0)
                cp.repository.refresh(force = true)
            }
        }
        val d = cp.repository.detail("FROG")
        assertEquals(tokenA, d.row.asset.contract, "the detail is still about the token that was opened")
        assertEquals(false, d.row.safety!!.honeypot)
        val listed = cp.repository.row("FROG")!!
        assertEquals(tokenB, listed.asset.contract)
        assertEquals(poolB, listed.dex!!.pairAddress, "token A's pool is not written into token B's row")
        assertEquals(true, listed.safety!!.honeypot, "token A's clean checks are not written into token B's row")
        assertEquals(RiskScorer.LABEL_RED_FLAG, listed.risk!!.label)
        assertEquals(tokenB, cp.repository.detail("FROG").row.asset.contract, "token A's detail was not cached for token B")
    }

    @Test
    fun aRefreshDuringTheDetailKeepsTheNewerListData() = runTest {
        goplus[tokenA] = cleanGoPlus
        simulation[tokenA] = cleanSimulation
        candles[poolA] = ohlcv(List(48) { 2.0 })
        val t = HookTransport(transport())
        val cp = app(t)
        cp.repository.refresh()
        t.before = { req ->
            if ("$poolA/ohlcv" in req.url) {
                t.before = null
                nowMs += 11 * 60_000L
                trending = pools(tokenA, poolA, priceUsd = 2.5, liquidityUsd = 600_000.0)
                cp.repository.refresh(force = true)
            }
        }
        val d = cp.repository.detail("FROG")
        val listed = cp.repository.row("FROG")!!
        assertEquals(2.5 * fx, listed.market!!.price!!, 1e-9, "the refresh's price stays")
        assertEquals(600_000.0, listed.dex!!.liquidityUsd, "the refresh's pool data stays")
        assertEquals(48, listed.market.sparkline7d.size, "the detail adds the chart")
        assertEquals("goplus+honeypot.is", listed.safety!!.source, "and the contract checks")
        assertEquals(listed, d.row)
    }

    @Test
    fun dexTokensGetNoFuturesDataOfASameTickerPerpetual() = runTest {
        goplus[tokenA] = cleanGoPlus
        simulation[tokenA] = cleanSimulation
        candles[poolA] = ohlcv(List(48) { 2.0 })
        val t = transport()
        val cp = app(t)
        cp.repository.refresh()
        val frog = cp.repository.row("FROG")!!
        assertFalse(frog.risk!!.blocked, "precondition: the token gets a forecast")
        assertFalse(frog.forecast!!.coverage.hasDerivatives, "Binance's FROGUSDT perpetual is another coin")
        assertTrue(cp.repository.row("BTC")!!.forecast!!.coverage.hasDerivatives, "listed coins keep their funding")
        val d = cp.repository.detail("FROG")
        assertNull(d.derivatives)
        assertTrue(d.row.forecast!!.h24.signals.none { it.name == "funding" || it.name == "long_short" })
        assertEquals(0, t.countRequests("globalLongShortAccountRatio?symbol=FROGUSDT"))
        cp.repository.enrichSentiment(n = 50)
        assertFalse(cp.repository.row("FROG")!!.forecast!!.coverage.hasDerivatives)
    }

    @Test
    fun dexPricesUseAStablecoinRateWithoutBitpandaPricesOrFutures() = runTest {
        goplus[tokenA] = cleanGoPlus
        val cp = app(transport(bitpanda = false), AppConfig(showMemecoins = false, showTrending = false, showDexRadar = true, derivativesEnabled = false))
        val snap = cp.repository.refresh()
        assertEquals(ListingSource.BUNDLED_SNAPSHOT, snap.universeSource)
        val m = cp.repository.row("FROG")!!.market!!
        assertEquals("EUR", m.currency, "USDT's EUR price is the exchange rate")
        assertEquals(2.0 * 0.889, m.price!!, 1e-9)
    }

    @Test
    fun memecoinsFromAnotherCurrencyAreNotReused() = runTest {
        val cp = app(transport(), AppConfig(showMemecoins = true, showTrending = false, showDexRadar = false))
        assertTrue(cp.repository.refresh().rows.any { it.asset.tier == Tier.MEME && it.market?.currency == "EUR" })
        // CoinGecko and Coinpaprika both refuse the memecoins after a switch to USD
        memecoinsUp = false
        cp.updateConfig(AppConfig(currency = "USD", showMemecoins = true, showTrending = false, showDexRadar = false))
        val usd = cp.repository.refresh()
        assertEquals("USD", usd.currency)
        assertTrue(usd.rows.all { it.market == null || it.market.currency == "USD" }, usd.rows.filter { it.market?.currency != "USD" }.map { it.asset.symbol }.toString())
    }

    @Test
    fun binanceHistoryOfAnotherCoinIsRejectedInEveryCurrency() = runTest {
        // SOL costs 92.74 EUR (104.2 USD); a Binance pair 1000x as expensive is a different coin
        val wrong = app(transport(marketChart = false, klines = klines(List(60) { 104_200.0 })))
        wrong.repository.refresh()
        val d = wrong.repository.detail("SOL")
        assertTrue(d.daily.isEmpty(), "EUR rows are checked too")
        assertFalse(d.warnings.any { it.startsWith("Daily history from Binance") })

        val right = app(transport(marketChart = false, klines = klines(List(60) { 104.2 })))
        right.repository.refresh()
        val ok = right.repository.detail("SOL")
        assertEquals(60, ok.daily.size, "the same coin differs only by the USD rate")
        assertEquals(92.74, ok.daily.last().value, 1e-9)
    }

    @Test
    fun backgroundSentimentReadsADexTokensOwnChannel() = runTest {
        val store = InMemoryDiskStore()
        val frog = AssetRow(
            Asset("FROG", "Frog", tier = Tier.DEX, chainId = "ethereum", contract = tokenA),
            MarketData("FROG", currency = "EUR", price = 1.0, marketCap = 1e7, lastUpdatedMs = nowMs, source = "dex"),
            dex = DexInfo("ethereum", tokenA, poolA, socials = mapOf("telegram" to "https://t.me/frogchannel")),
        )
        val snap = MarketSnapshot("EUR", listOf(frog), universeSource = ListingSource.BITPANDA_LEGACY_TICKER, universeSize = 1, fetchedAtMs = nowMs)
        store.write(MarketRepository.SNAPSHOT_KEY, Json.encodeToString(MarketSnapshot.serializer(), snap))
        val t = transport()
        val cp = app(t, store = store)
        cp.repository.enrichSentiment(n = 5)
        assertEquals(1, t.countRequests("t.me/s/frogchannel"))
    }

    @Test
    fun tiersSwitchedOffStayOutOfAPersistedSnapshot() = runTest {
        val store = InMemoryDiskStore()
        fun row(sym: String, tier: Tier) = AssetRow(Asset(sym, sym, tier = tier), MarketData(sym, currency = "EUR", price = 1.0, lastUpdatedMs = nowMs))
        val old = listOf(row("BTC", Tier.BITPANDA), row("PEPE2", Tier.MEME), row("FROG", Tier.DEX), row("HOT", Tier.TRENDING))
        store.write(MarketRepository.SNAPSHOT_KEY, Json.encodeToString(MarketSnapshot.serializer(),
            MarketSnapshot("EUR", old, universeSource = ListingSource.BITPANDA_LEGACY_TICKER, universeSize = 1, fetchedAtMs = nowMs)))
        // the defaults: memecoins and the DEX radar off, trending on
        val cp = app(transport(), config = AppConfig(), store = store)
        assertEquals(listOf("BTC", "HOT"), cp.repository.snapshot.value?.rows?.map { it.asset.symbol })
    }

    /** One GeckoTerminal trending pool on Ethereum for a token with the ticker FROG. */
    private fun pools(token: String, pool: String, priceUsd: Double = 2.0, liquidityUsd: Double = 500_000.0): String = """
        {"data":[{"id":"eth_$pool","type":"pool","attributes":{"address":"$pool","name":"FROG / WETH","base_token_price_usd":"$priceUsd",
          "reserve_in_usd":"$liquidityUsd","fdv_usd":"20000000","market_cap_usd":"20000000","pool_created_at":"2025-09-01T00:00:00Z",
          "price_change_percentage":{"h1":"0.5","h24":"4.0"},"volume_usd":{"h24":"300000"},
          "transactions":{"h1":{"buys":20,"sells":18,"buyers":15,"sellers":12},"h6":{"buys":120,"sells":100,"buyers":80,"sellers":70},
          "h24":{"buys":500,"sells":450,"buyers":300,"sellers":280}}},
          "relationships":{"base_token":{"data":{"id":"eth_$token","type":"token"}},"dex":{"data":{"id":"uniswap_v2","type":"dex"}},
          "network":{"data":{"id":"eth","type":"network"}}}}],
         "included":[{"id":"eth_$token","type":"token","attributes":{"address":"$token","symbol":"FROG","name":"Frog"}}]}
    """.trimIndent()

    /** GeckoTerminal hourly candles (newest first, seconds) whose last close is in the current hour. */
    private fun ohlcv(closes: List<Double>): String {
        val hour = 3_600_000L
        val end = nowMs.floorDiv(hour) * hour
        val rows = closes.mapIndexed { i, c -> "[${(end - (closes.size - 1 - i) * hour) / 1000},$c,$c,$c,$c,1000.0]" }.reversed()
        return """{"data":{"attributes":{"ohlcv_list":[${rows.joinToString(",")}]}}}"""
    }

    /** Binance daily klines (oldest first, milliseconds) ending today. */
    private fun klines(closes: List<Double>): String {
        val day = 86_400_000L
        val end = nowMs.floorDiv(day) * day
        val rows = closes.mapIndexed { i, c -> """[${end - (closes.size - 1 - i) * day},"$c","$c","$c","$c","10.0"]""" }
        return "[" + rows.joinToString(",") + "]"
    }
}
