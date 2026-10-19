package dev.cryptopulse.core.http

import dev.cryptopulse.core.util.Clock
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.net.URI

/** Allowed request rate for one host. */
data class Rate(val requestsPerMinute: Double, val burst: Int = 2)

/**
 * Per-host token bucket. Keeps us inside the free tiers of CoinGecko, Reddit, StockTwits etc.
 * `acquire` suspends until a token is available; it never throws.
 */
class RateLimiter(
    private val rules: Map<String, Rate> = DEFAULT_RULES,
    private val defaultRate: Rate = Rate(requestsPerMinute = 30.0, burst = 3),
    private val clock: Clock = Clock.SYSTEM,
    private val sleeper: suspend (Long) -> Unit = { delay(it) },
) {
    private class Bucket(var tokens: Double, var lastRefillMs: Long)

    private val mutex = Mutex()
    private val buckets = HashMap<String, Bucket>()
    private val blockedUntil = java.util.concurrent.ConcurrentHashMap<String, Long>()

    /** After an HTTP 429: callers should not contact [host] again before [untilMs]. */
    fun penalize(host: String, untilMs: Long) {
        blockedUntil.merge(host.lowercase(), untilMs) { a, b -> maxOf(a, b) }
    }

    /** Remaining cool-down for [host] in ms, or 0 when it may be contacted. */
    fun cooldownMs(host: String): Long {
        val until = blockedUntil[host.lowercase()] ?: return 0L
        return (until - clock.nowMs()).coerceAtLeast(0L)
    }

    fun rateFor(host: String): Rate {
        val h = host.lowercase()
        rules[h]?.let { return it }
        // suffix match: "oauth.reddit.com" -> "reddit.com"
        return rules.entries.firstOrNull { h == it.key || h.endsWith("." + it.key) }?.value ?: defaultRate
    }

    suspend fun acquire(host: String) {
        val rate = rateFor(host)
        val perMs = rate.requestsPerMinute / 60_000.0
        while (true) {
            val waitMs = mutex.withLock {
                val now = clock.nowMs()
                val bucket = buckets.getOrPut(host.lowercase()) { Bucket(rate.burst.toDouble(), now) }
                val refill = (now - bucket.lastRefillMs).coerceAtLeast(0) * perMs
                bucket.tokens = (bucket.tokens + refill).coerceAtMost(rate.burst.toDouble())
                bucket.lastRefillMs = now
                if (bucket.tokens >= 1.0) {
                    bucket.tokens -= 1.0
                    0L
                } else {
                    ((1.0 - bucket.tokens) / perMs).toLong().coerceAtLeast(10L)
                }
            }
            if (waitMs == 0L) return
            sleeper(waitMs)
        }
    }

    suspend fun acquireForUrl(url: String) = acquire(hostOf(url))

    companion object {
        fun hostOf(url: String): String = try {
            URI(url).host ?: url
        } catch (_: Exception) {
            url
        }

        /** Conservative defaults for the free tiers we rely on. */
        val DEFAULT_RULES: Map<String, Rate> = mapOf(
            "api.coingecko.com" to Rate(requestsPerMinute = 10.0, burst = 3),
            "api.bitpanda.com" to Rate(requestsPerMinute = 20.0, burst = 3),
            "api.public.bitpanda.com" to Rate(requestsPerMinute = 30.0, burst = 5),
            "reddit.com" to Rate(requestsPerMinute = 8.0, burst = 2),
            "api.stocktwits.com" to Rate(requestsPerMinute = 60.0, burst = 5),
            "t.me" to Rate(requestsPerMinute = 20.0, burst = 3),
            "telegram.me" to Rate(requestsPerMinute = 20.0, burst = 3),
            "news.google.com" to Rate(requestsPerMinute = 20.0, burst = 3),
            "api.alternative.me" to Rate(requestsPerMinute = 10.0, burst = 2),
            "fapi.binance.com" to Rate(requestsPerMinute = 30.0, burst = 5),
            "api.binance.com" to Rate(requestsPerMinute = 30.0, burst = 5),
            "api.bybit.com" to Rate(requestsPerMinute = 30.0, burst = 5),
            // memecoin / DEX sources (documented free limits, with headroom)
            "api.dexscreener.com" to Rate(requestsPerMinute = 50.0, burst = 6),
            "api.geckoterminal.com" to Rate(requestsPerMinute = 20.0, burst = 3),
            "frontend-api-v3.pump.fun" to Rate(requestsPerMinute = 20.0, burst = 2),
            "api.gopluslabs.io" to Rate(requestsPerMinute = 20.0, burst = 3),
            "api.honeypot.is" to Rate(requestsPerMinute = 20.0, burst = 2),
            "api.coinpaprika.com" to Rate(requestsPerMinute = 6.0, burst = 2),
            "data-api.binance.vision" to Rate(requestsPerMinute = 60.0, burst = 5),
            "www.okx.com" to Rate(requestsPerMinute = 20.0, burst = 3),
            // prediction markets
            "gamma-api.polymarket.com" to Rate(requestsPerMinute = 30.0, burst = 5),
        )
    }
}
