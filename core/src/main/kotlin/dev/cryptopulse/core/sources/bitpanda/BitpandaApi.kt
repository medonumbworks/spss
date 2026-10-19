package dev.cryptopulse.core.sources.bitpanda

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.sources.BitpandaLegacyTicker
import dev.cryptopulse.core.sources.BitpandaPublicAsset
import dev.cryptopulse.core.sources.BitpandaPublicTicker
import dev.cryptopulse.core.sources.BitpandaSource
import dev.cryptopulse.core.sources.BitpandaTickerEntry
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.dbl
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.obj
import dev.cryptopulse.core.util.str
import java.net.URLEncoder

/**
 * Bitpanda price/asset endpoints.
 *  - Legacy ticker `https://api.bitpanda.com/v1/ticker`: one call, every asset, prices in EUR/USD/CHF/GBP/...
 *    (announced to be retired on 2027-01-31).
 *  - New Public API `https://api.public.bitpanda.com/v1/assets` + `/v1/tickers/{id}`; community reports say
 *    these answer without a key, the docs say a key is required. We try keyless first, then the optional key.
 */
class BitpandaApi(
    private val http: Http,
    private val clock: Clock = Clock.SYSTEM,
    private val legacyBase: String = "https://api.bitpanda.com/v1",
    private val publicBase: String = "https://api.public.bitpanda.com",
) : BitpandaSource {

    override suspend fun legacyTicker(apiKey: String?): SourceResult<BitpandaLegacyTicker> {
        val url = "$legacyBase/ticker"
        var result = http.getText(url, headers = JSON, ttlMs = 60_000, cacheKey = "bitpanda:legacy-ticker")
        if (result is HttpResult.Failed && result.isUnauthorized && !apiKey.isNullOrBlank()) {
            result = http.getText(url, headers = JSON + ("X-Api-Key" to apiKey), ttlMs = 60_000, cacheKey = "bitpanda:legacy-ticker:key")
        }
        return when (result) {
            is HttpResult.Failed -> SourceResult.failureOf(result, "Bitpanda ticker")
            is HttpResult.Ok -> try {
                val entries = parseLegacyTicker(result.body)
                if (entries.isEmpty()) SourceResult.Failure(SourceStatus.EMPTY, "Bitpanda ticker returned no assets")
                else SourceResult.Success(BitpandaLegacyTicker(entries, clock.nowMs() - result.ageMs), result.fromCache, result.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError("Bitpanda ticker", e)
            }
        }
    }

    override suspend fun publicAssets(apiKey: String?, type: String?): SourceResult<List<BitpandaPublicAsset>> {
        val all = ArrayList<BitpandaPublicAsset>()
        var cursor: String? = null
        var pages = 0
        var useKey = false
        var pathPrefix = "/v1"
        while (pages < MAX_PAGES) {
            val params = buildList {
                add("pageSize=100")
                if (!type.isNullOrBlank()) add("type=" + enc(type))
                if (cursor != null) add("cursor=" + enc(cursor!!))
            }.joinToString("&")
            val url = "$publicBase$pathPrefix/assets?$params"
            val headers = if (useKey && !apiKey.isNullOrBlank()) JSON + ("x-api-key" to apiKey) else JSON
            val result = http.getText(url, headers = headers, ttlMs = 6 * 3_600_000L, cacheKey = "bitpanda:public-assets:$pathPrefix:$type:$cursor:$useKey")
            if (result is HttpResult.Failed) {
                if (pages == 0 && result.code == 404 && pathPrefix == "/v1") { pathPrefix = ""; continue }
                if (pages == 0 && result.isUnauthorized && !useKey && !apiKey.isNullOrBlank()) { useKey = true; continue }
                return if (all.isNotEmpty()) SourceResult.Success(all) else SourceResult.failureOf(result, "Bitpanda assets")
            }
            result as HttpResult.Ok
            val (page, next) = try {
                parsePublicAssets(result.body)
            } catch (e: Exception) {
                return if (all.isNotEmpty()) SourceResult.Success(all) else SourceResult.parseError("Bitpanda assets", e)
            }
            all += page
            pages++
            if (next.isNullOrBlank() || page.isEmpty() || next == cursor) break
            cursor = next
        }
        val filtered = if (type.isNullOrBlank()) all else all.filter { a -> a.type == null || a.type.contains("crypto", ignoreCase = true) || a.type.equals(type, true) }
        return if (filtered.isEmpty()) SourceResult.Failure(SourceStatus.EMPTY, "Bitpanda assets: empty list") else SourceResult.Success(filtered.distinctBy { it.symbol })
    }

    override suspend fun publicTicker(assetId: String, apiKey: String?): SourceResult<BitpandaPublicTicker> {
        val candidates = listOf("$publicBase/v1/tickers/${enc(assetId)}", "$publicBase/assets/${enc(assetId)}/ticker", "$publicBase/v1/assets/${enc(assetId)}/ticker")
        var last: HttpResult.Failed? = null
        for (url in candidates) {
            for (withKey in listOf(false, true)) {
                if (withKey && apiKey.isNullOrBlank()) continue
                val headers = if (withKey) JSON + ("x-api-key" to apiKey!!) else JSON
                when (val r = http.getText(url, headers = headers, ttlMs = 30_000)) {
                    is HttpResult.Ok -> return try {
                        val root = JsonUtil.parse(r.body)
                        val o = root["data"].obj ?: root.obj
                        val price = o["price"].dbl ?: throw IllegalStateException("no price")
                        SourceResult.Success(BitpandaPublicTicker(o["assetId"].str ?: assetId, price, o["currency"].str ?: "EUR", o["timestamp"].str))
                    } catch (e: Exception) {
                        SourceResult.parseError("Bitpanda ticker", e)
                    }
                    is HttpResult.Failed -> {
                        last = r
                        if (!r.isUnauthorized) break
                    }
                }
            }
        }
        return last?.let { SourceResult.failureOf(it, "Bitpanda ticker") } ?: SourceResult.Failure(SourceStatus.UNAVAILABLE, "Bitpanda ticker unavailable")
    }

    companion object {
        private const val MAX_PAGES = 30
        private val JSON = mapOf("Accept" to "application/json")
        private fun enc(s: String) = URLEncoder.encode(s, "UTF-8")

        /** Metals and index products that the legacy ticker mixes in with cryptocurrencies. */
        val NON_CRYPTO_SYMBOLS: Set<String> = setOf("XAU", "XAG", "XPT", "XPD", "BCI5", "BCI10", "BCI25", "BCIDL", "BCIIL", "BCIMEME", "BEST-EUR")

        /** `{"BTC":{"EUR":"62000.1","USD":"..."}, ...}` -> entries (values may be strings or numbers). */
        fun parseLegacyTicker(body: String): List<BitpandaTickerEntry> {
            val root = JsonUtil.parse(body).obj ?: return emptyList()
            return root.mapNotNull { (symbol, value) ->
                val prices = value.obj?.mapNotNull { (cur, v) -> v.dbl?.let { cur.uppercase() to it } }?.toMap().orEmpty()
                if (prices.isEmpty()) null else BitpandaTickerEntry(symbol.uppercase(), prices)
            }
        }

        /** `{"data":[{"id":..,"symbol":..,"type":..}], "cursor":".."}` (also accepts `next_cursor` / `meta.next_cursor`). */
        fun parsePublicAssets(body: String): Pair<List<BitpandaPublicAsset>, String?> {
            val root = JsonUtil.parse(body)
            val data = root["data"].arr ?: root.arr ?: return emptyList<BitpandaPublicAsset>() to null
            val assets = data.mapNotNull { el ->
                val o = el.obj ?: return@mapNotNull null
                val attrs = o["attributes"].obj ?: o
                val id = o["id"].str ?: attrs["id"].str ?: return@mapNotNull null
                val symbol = attrs["symbol"].str ?: return@mapNotNull null
                BitpandaPublicAsset(
                    id = id,
                    symbol = symbol.uppercase(),
                    name = attrs["name"].str,
                    type = attrs["type"].str ?: attrs["asset_type"].str,
                    group = attrs["group"].str ?: attrs["asset_group"].str,
                    isin = attrs["isin"].str,
                )
            }
            val cursor = root["cursor"].str ?: root["next_cursor"].str ?: root["meta"]["next_cursor"].str ?: root["links"]["next_cursor"].str
            return assets to cursor
        }
    }
}
