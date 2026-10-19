package dev.cryptopulse.core.http

import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.Logger
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap

sealed class HttpResult {
    data class Ok(val body: String, val fromCache: Boolean, val ageMs: Long, val code: Int = 200) : HttpResult()
    data class Failed(val reason: String, val code: Int? = null, val staleBody: String? = null) : HttpResult() {
        val isRateLimited: Boolean get() = code == 429
        val isUnauthorized: Boolean get() = code == 401 || code == 403
        val isGeoBlocked: Boolean get() = code == 451
    }
}

/**
 * Facade every data source uses: cache -> rate limiter -> transport -> retry/backoff -> stale fallback.
 *
 * Caching semantics of [getText]:
 *  - fresh cache entry (age < ttlMs): returned immediately, no network;
 *  - otherwise one network attempt (plus retries on 429/5xx/IOException);
 *  - on failure an entry younger than staleOkMs is returned as `Ok(fromCache = true)`;
 *  - identical concurrent calls are serialized per cache key so only one hits the network.
 */
class Http(
    private val transport: HttpTransport,
    private val rateLimiter: RateLimiter = RateLimiter(),
    private val cache: HttpCache = HttpCache(),
    private val clock: Clock = Clock.SYSTEM,
    private val logger: Logger = Logger.STDERR,
    private val sleeper: suspend (Long) -> Unit = { delay(it) },
) {
    private val keyLocks = ConcurrentHashMap<String, Mutex>()

    suspend fun getText(
        url: String,
        headers: Map<String, String> = emptyMap(),
        ttlMs: Long = 0L,
        staleOkMs: Long = ttlMs * 12,
        cacheKey: String = url,
        retries: Int = 2,
        timeoutMs: Long = 20_000,
    ): HttpResult {
        val now = clock.nowMs()
        if (ttlMs > 0) {
            cache.get(cacheKey)?.let { entry ->
                val age = entry.ageMs(now)
                if (age < ttlMs) return HttpResult.Ok(entry.body, fromCache = true, ageMs = age)
            }
        }
        val lock = keyLocks.getOrPut(cacheKey) { Mutex() }
        return lock.withLock {
            // another coroutine may have refreshed it while we waited
            if (ttlMs > 0) {
                cache.get(cacheKey)?.let { entry ->
                    val age = entry.ageMs(clock.nowMs())
                    if (age < ttlMs) return@withLock HttpResult.Ok(entry.body, fromCache = true, ageMs = age)
                }
            }
            val result = fetchWithRetry(HttpRequest(url, headers, timeoutMs = timeoutMs), retries)
            when (result) {
                is HttpResult.Ok -> {
                    if (ttlMs > 0 || staleOkMs > 0) cache.put(cacheKey, result.body)
                    result
                }
                is HttpResult.Failed -> {
                    val stale = cache.get(cacheKey)
                    if (stale != null && stale.ageMs(clock.nowMs()) < staleOkMs) {
                        logger.warn(TAG, "Using stale cache for $url after failure: ${result.reason}")
                        HttpResult.Ok(stale.body, fromCache = true, ageMs = stale.ageMs(clock.nowMs()))
                    } else {
                        result.copy(staleBody = stale?.body)
                    }
                }
            }
        }
    }

    /** POST/other methods are rare (Bitpanda quotes etc.); no caching. */
    suspend fun request(request: HttpRequest, retries: Int = 1): HttpResult = fetchWithRetry(request, retries)

    private suspend fun fetchWithRetry(request: HttpRequest, retries: Int): HttpResult {
        var attempt = 0
        var lastFailure: HttpResult.Failed? = null
        val host = RateLimiter.hostOf(request.url)
        while (attempt <= retries) {
            rateLimiter.acquire(host)
            val response = try {
                transport.execute(request)
            } catch (e: IOException) {
                lastFailure = HttpResult.Failed("network: ${e.message ?: e::class.simpleName}")
                logger.debug(TAG, "attempt ${attempt + 1} failed for ${request.url}: ${e.message}")
                null
            } catch (e: IllegalArgumentException) {
                return HttpResult.Failed("bad request: ${e.message}")
            }
            if (response != null) {
                if (response.isSuccess) return HttpResult.Ok(response.body, fromCache = false, ageMs = 0, code = response.code)
                lastFailure = HttpResult.Failed("HTTP ${response.code}", response.code)
                val retryable = response.code == 429 || response.code == 408 || response.code in 500..599
                if (!retryable) return lastFailure
                val retryAfter = response.retryAfterMs
                if (attempt < retries) sleeper(retryAfter?.coerceIn(1_000, 30_000) ?: backoffMs(attempt))
            } else if (attempt < retries) {
                sleeper(backoffMs(attempt))
            }
            attempt++
        }
        return lastFailure ?: HttpResult.Failed("unknown failure")
    }

    private fun backoffMs(attempt: Int): Long = (1_500L shl attempt).coerceAtMost(12_000L)

    private companion object {
        const val TAG = "Http"
    }
}
