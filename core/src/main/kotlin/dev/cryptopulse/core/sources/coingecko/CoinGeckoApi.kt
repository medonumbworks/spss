package dev.cryptopulse.core.sources.coingecko

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.model.CoinCommunity
import dev.cryptopulse.core.model.GlobalStats
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.model.PricePoint
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.model.TrendingCoin
import dev.cryptopulse.core.sources.CoinGeckoSource
import dev.cryptopulse.core.sources.CoinListEntry
import dev.cryptopulse.core.sources.MarketChart
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.util.Dates
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.dbl
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.int
import dev.cryptopulse.core.util.lng
import dev.cryptopulse.core.util.obj
import dev.cryptopulse.core.util.str
import kotlinx.serialization.json.JsonElement
import java.net.URLEncoder

/**
 * CoinGecko public API (free, keyless; an optional free Demo key raises the limit to ~30 req/min).
 * Docs: https://docs.coingecko.com/v3.0.1/reference/introduction
 */
class CoinGeckoApi(
    private val http: Http,
    private val demoKeyProvider: () -> String? = { null },
    private val base: String = "https://api.coingecko.com/api/v3",
    private val marketsTtlMs: Long = 3 * 60_000L,
) : CoinGeckoSource {

    private fun headers(): Map<String, String> {
        val key = demoKeyProvider()
        return if (key.isNullOrBlank()) mapOf("Accept" to "application/json") else mapOf("Accept" to "application/json", "x-cg-demo-api-key" to key)
    }

    override suspend fun markets(vsCurrency: String, page: Int, perPage: Int, ids: List<String>?, sparkline: Boolean, ttlMs: Long?, category: String?): SourceResult<List<MarketData>> {
        val vs = vsCurrency.lowercase()
        val params = buildList {
            add("vs_currency=$vs")
            add("order=market_cap_desc")
            add("per_page=${perPage.coerceIn(1, 250)}")
            add("page=${page.coerceAtLeast(1)}")
            add("sparkline=$sparkline")
            add("price_change_percentage=1h,24h,7d,14d,30d,1y")
            add("locale=en")
            if (!ids.isNullOrEmpty()) add("ids=" + enc(ids.joinToString(","))) else if (!category.isNullOrBlank()) add("category=" + enc(category))
        }.joinToString("&")
        val url = "$base/coins/markets?$params"
        return when (val r = http.getText(url, headers(), ttlMs = ttlMs ?: marketsTtlMs, staleOkMs = 6 * 3_600_000L)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "CoinGecko markets")
            is HttpResult.Ok -> try {
                SourceResult.Success(parseMarkets(r.body, vsCurrency.uppercase()), r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("CoinGecko markets", e)
            }
        }
    }

    override suspend fun marketChart(id: String, vsCurrency: String, days: Int): SourceResult<MarketChart> {
        val url = "$base/coins/${enc(id)}/market_chart?vs_currency=${vsCurrency.lowercase()}&days=$days"
        return when (val r = http.getText(url, headers(), ttlMs = 30 * 60_000L, staleOkMs = 24 * 3_600_000L)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "CoinGecko history")
            is HttpResult.Ok -> try {
                SourceResult.Success(parseMarketChart(r.body), r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("CoinGecko history", e)
            }
        }
    }

    override suspend fun coinCommunity(id: String): SourceResult<CoinCommunity> {
        val url = "$base/coins/${enc(id)}?localization=false&tickers=false&market_data=false&community_data=true&developer_data=false&sparkline=false"
        return when (val r = http.getText(url, headers(), ttlMs = 6 * 3_600_000L, staleOkMs = 7 * 24 * 3_600_000L)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "CoinGecko coin")
            is HttpResult.Ok -> try {
                SourceResult.Success(parseCoin(id, r.body), r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("CoinGecko coin", e)
            }
        }
    }

    override suspend fun trending(): SourceResult<List<TrendingCoin>> =
        when (val r = http.getText("$base/search/trending", headers(), ttlMs = 15 * 60_000L, staleOkMs = 6 * 3_600_000L)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "CoinGecko trending")
            is HttpResult.Ok -> try {
                val coins = JsonUtil.parse(r.body)["coins"].arr.orEmpty().mapNotNull { el ->
                    val item = el["item"].obj ?: return@mapNotNull null
                    TrendingCoin(
                        coingeckoId = item["id"].str ?: return@mapNotNull null,
                        symbol = (item["symbol"].str ?: "").uppercase(),
                        name = item["name"].str ?: "",
                        marketCapRank = item["market_cap_rank"].int,
                        score = item["score"].int,
                    )
                }
                SourceResult.Success(coins, r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("CoinGecko trending", e)
            }
        }

    override suspend fun global(): SourceResult<GlobalStats> =
        when (val r = http.getText("$base/global", headers(), ttlMs = 10 * 60_000L, staleOkMs = 6 * 3_600_000L)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "CoinGecko global")
            is HttpResult.Ok -> try {
                val d = JsonUtil.parse(r.body)["data"]
                SourceResult.Success(
                    GlobalStats(
                        totalMarketCapUsd = d["total_market_cap"]["usd"].dbl,
                        totalVolume24hUsd = d["total_volume"]["usd"].dbl,
                        marketCapChange24hPct = d["market_cap_change_percentage_24h_usd"].dbl,
                        btcDominancePct = d["market_cap_percentage"]["btc"].dbl,
                        ethDominancePct = d["market_cap_percentage"]["eth"].dbl,
                    ),
                    r.fromCache, r.ageMs,
                )
            } catch (e: Exception) {
                SourceResult.parseError("CoinGecko global", e)
            }
        }

    override suspend fun coinsList(): SourceResult<List<CoinListEntry>> =
        when (val r = http.getText("$base/coins/list", headers(), ttlMs = 3 * 24 * 3_600_000L, staleOkMs = 60 * 24 * 3_600_000L, timeoutMs = 45_000)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "CoinGecko coin list")
            is HttpResult.Ok -> try {
                val list = JsonUtil.parse(r.body).arr.orEmpty().mapNotNull { el ->
                    val id = el["id"].str ?: return@mapNotNull null
                    CoinListEntry(id, (el["symbol"].str ?: "").uppercase(), el["name"].str ?: id)
                }
                if (list.isEmpty()) SourceResult.Failure(SourceStatus.EMPTY, "CoinGecko coin list empty") else SourceResult.Success(list, r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("CoinGecko coin list", e)
            }
        }

    companion object {
        private fun enc(s: String) = URLEncoder.encode(s, "UTF-8")

        fun parseMarkets(body: String, currency: String): List<MarketData> =
            JsonUtil.parse(body).arr.orEmpty().mapNotNull { parseMarketRow(it, currency) }

        private fun parseMarketRow(el: JsonElement, currency: String): MarketData? {
            val o = el.obj ?: return null
            val id = o["id"].str ?: return null
            return MarketData(
                symbol = (o["symbol"].str ?: "").uppercase(),
                coingeckoId = id,
                name = o["name"].str,
                currency = currency,
                price = o["current_price"].dbl,
                marketCap = o["market_cap"].dbl,
                marketCapRank = o["market_cap_rank"].int,
                volume24h = o["total_volume"].dbl,
                high24h = o["high_24h"].dbl,
                low24h = o["low_24h"].dbl,
                change1hPct = o["price_change_percentage_1h_in_currency"].dbl,
                change24hPct = o["price_change_percentage_24h_in_currency"].dbl ?: o["price_change_percentage_24h"].dbl,
                change7dPct = o["price_change_percentage_7d_in_currency"].dbl,
                change14dPct = o["price_change_percentage_14d_in_currency"].dbl,
                change30dPct = o["price_change_percentage_30d_in_currency"].dbl,
                change1yPct = o["price_change_percentage_1y_in_currency"].dbl,
                ath = o["ath"].dbl,
                athChangePct = o["ath_change_percentage"].dbl,
                athDate = o["ath_date"].str,
                atl = o["atl"].dbl,
                atlChangePct = o["atl_change_percentage"].dbl,
                atlDate = o["atl_date"].str,
                circulatingSupply = o["circulating_supply"].dbl,
                totalSupply = o["total_supply"].dbl,
                sparkline7d = o["sparkline_in_7d"]["price"].arr.orEmpty().mapNotNull { it.dbl },
                imageUrl = o["image"].str,
                lastUpdatedMs = Dates.parseToMs(o["last_updated"].str),
                source = "coingecko",
            )
        }

        fun parseMarketChart(body: String): MarketChart {
            val root = JsonUtil.parse(body)
            fun series(key: String) = root[key].arr.orEmpty().mapNotNull { p ->
                val pair = p.arr ?: return@mapNotNull null
                val t = pair.getOrNull(0).lng ?: return@mapNotNull null
                val v = pair.getOrNull(1).dbl ?: return@mapNotNull null
                PricePoint(t, v)
            }.sortedBy { it.timeMs }
            return MarketChart(series("prices"), series("total_volumes"), series("market_caps"))
        }

        fun parseCoin(id: String, body: String): CoinCommunity {
            val o = JsonUtil.parse(body)
            val community = o["community_data"]
            val desc = o["description"]["en"].str?.replace(Regex("<[^>]+>"), "")?.trim()?.take(1200)
            return CoinCommunity(
                coingeckoId = id,
                sentimentVotesUpPct = o["sentiment_votes_up_percentage"].dbl,
                sentimentVotesDownPct = o["sentiment_votes_down_percentage"].dbl,
                watchlistUsers = o["watchlist_portfolio_users"].lng,
                redditSubscribers = community["reddit_subscribers"].lng,
                twitterFollowers = community["twitter_followers"].lng,
                telegramUsers = community["telegram_channel_user_count"].lng,
                description = desc?.takeIf { it.isNotBlank() },
                homepage = o["links"]["homepage"].arr?.firstOrNull { !it.str.isNullOrBlank() }.str,
                subreddit = o["links"]["subreddit_url"].str?.takeIf { it.isNotBlank() && it != "https://www.reddit.com" },
                categories = o["categories"].arr.orEmpty().mapNotNull { it.str }.filter { it.isNotBlank() },
            )
        }
    }
}
