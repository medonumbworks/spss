package dev.cryptopulse.core.sources.paprika

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.model.MarketData
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.util.Dates
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.dbl
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.int
import dev.cryptopulse.core.util.str
import kotlinx.serialization.Serializable

/** One Coinpaprika ticker, already in the requested quote currency. */
@Serializable
data class PaprikaTicker(val id: String, val symbol: String, val name: String, val rank: Int?, val market: MarketData)

/**
 * Coinpaprika free API (keyless). `/v1/tickers` returns the top ~2000 coins with prices in one
 * request, which makes it the fallback when CoinGecko's free tier answers 429. The free plan has
 * a monthly call budget, so this is cached for 10 minutes and only used when needed.
 * Attribution: data by Coinpaprika (https://coinpaprika.com).
 */
class CoinpaprikaApi(
    private val http: Http,
    private val base: String = "https://api.coinpaprika.com/v1",
) {
    suspend fun tickers(currency: String): SourceResult<List<PaprikaTicker>> {
        val quote = currency.uppercase()
        return when (val r = http.getText("$base/tickers?quotes=$quote", JSON, ttlMs = 10 * 60_000L, staleOkMs = 6 * 3_600_000L, retries = 1, timeoutMs = 45_000)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "Coinpaprika")
            is HttpResult.Ok -> try {
                SourceResult.Success(parseTickers(r.body, quote), r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("Coinpaprika", e)
            }
        }
    }

    companion object {
        private val JSON = mapOf("Accept" to "application/json")

        fun parseTickers(body: String, quote: String): List<PaprikaTicker> =
            JsonUtil.parse(body).arr.orEmpty().mapNotNull { t ->
                val id = t["id"].str ?: return@mapNotNull null
                val symbol = t["symbol"].str?.uppercase() ?: return@mapNotNull null
                val q = t["quotes"][quote] ?: return@mapNotNull null
                val price = q["price"].dbl
                PaprikaTicker(
                    id = id,
                    symbol = symbol,
                    name = t["name"].str ?: symbol,
                    rank = t["rank"].int?.takeIf { it > 0 },
                    market = MarketData(
                        symbol = symbol,
                        name = t["name"].str,
                        currency = quote,
                        price = price,
                        marketCap = q["market_cap"].dbl?.takeIf { it > 0 },
                        marketCapRank = t["rank"].int?.takeIf { it > 0 },
                        volume24h = q["volume_24h"].dbl,
                        change1hPct = q["percent_change_1h"].dbl,
                        change24hPct = q["percent_change_24h"].dbl,
                        change7dPct = q["percent_change_7d"].dbl,
                        change30dPct = q["percent_change_30d"].dbl,
                        change1yPct = q["percent_change_1y"].dbl,
                        ath = q["ath_price"].dbl,
                        athChangePct = q["percent_from_price_ath"].dbl,
                        athDate = q["ath_date"].str,
                        circulatingSupply = t["circulating_supply"].dbl,
                        totalSupply = t["total_supply"].dbl,
                        lastUpdatedMs = Dates.parseToMs(t["last_updated"].str),
                        source = "coinpaprika",
                    ),
                )
            }
    }
}
