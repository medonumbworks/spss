package dev.cryptopulse.core.sources

import dev.cryptopulse.core.FIXTURE_NOW_MS
import dev.cryptopulse.core.fastLimiter
import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpCache
import dev.cryptopulse.core.http.HttpResponse
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.http.Rate
import dev.cryptopulse.core.http.RateLimiter
import dev.cryptopulse.core.testing.FakeTransport
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.Logger
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class HttpTest {
    private var now = FIXTURE_NOW_MS
    private val clock = Clock { now }

    @Test
    fun freshCacheAvoidsNetwork() = runTest {
        val t = FakeTransport().onUrlContaining("x", body = "one")
        val http = Http(t, fastLimiter(clock), HttpCache(null, clock), clock, Logger.NONE, sleeper = {})
        assertEquals("one", (http.getText("https://a.test/x", ttlMs = 60_000) as HttpResult.Ok).body)
        now += 30_000
        val second = http.getText("https://a.test/x", ttlMs = 60_000) as HttpResult.Ok
        assertTrue(second.fromCache)
        assertEquals(1, t.countRequests("/x"))
    }

    @Test
    fun staleCacheIsServedWhenNetworkFails() = runTest {
        var fail = false
        val t = FakeTransport().onUrlContaining("x") { req -> if (fail) HttpResponse(503, "", emptyMap(), req.url) else HttpResponse(200, "cached", emptyMap(), req.url) }
        val http = Http(t, fastLimiter(clock), HttpCache(null, clock), clock, Logger.NONE, sleeper = {})
        http.getText("https://a.test/x", ttlMs = 60_000, staleOkMs = 3_600_000)
        fail = true
        now += 120_000
        val r = http.getText("https://a.test/x", ttlMs = 60_000, staleOkMs = 3_600_000, retries = 1)
        assertIs<HttpResult.Ok>(r)
        assertEquals("cached", r.body)
        assertTrue(r.fromCache)
        assertEquals(3, t.countRequests("/x"), "1 initial + 2 attempts")
    }

    @Test
    fun retriesOn429ThenSucceeds() = runTest {
        var calls = 0
        val sleeps = ArrayList<Long>()
        val t = FakeTransport().onUrlContaining("y") { req -> calls++; if (calls == 1) HttpResponse(429, "", mapOf("Retry-After" to "2"), req.url) else HttpResponse(200, "ok", emptyMap(), req.url) }
        val http = Http(t, fastLimiter(clock), HttpCache(null, clock), clock, Logger.NONE, sleeper = { sleeps += it })
        val r = http.getText("https://a.test/y")
        assertIs<HttpResult.Ok>(r)
        assertEquals(listOf(2_000L), sleeps)
    }

    @Test
    fun longRetryAfterCoolsTheHostDownInsteadOfWaiting() = runTest {
        val sleeps = ArrayList<Long>()
        val t = FakeTransport().onUrlContaining("z") { req -> HttpResponse(429, "", mapOf("Retry-After" to "60"), req.url) }
        val http = Http(t, fastLimiter(clock), HttpCache(null, clock), clock, Logger.NONE, sleeper = { sleeps += it })
        val first = http.getText("https://a.test/z")
        assertIs<HttpResult.Failed>(first)
        assertEquals(429, first.code)
        assertEquals(emptyList(), sleeps, "no 60 s stall inside a refresh")
        val second = http.getText("https://a.test/z2")
        assertIs<HttpResult.Failed>(second)
        assertEquals(1, t.requests.size, "the host is not contacted again during its cool-down")
        now += 61_000
        http.getText("https://a.test/z3", retries = 0)
        assertEquals(2, t.requests.size)
    }

    @Test
    fun notFoundIsNotRetried() = runTest {
        val t = FakeTransport()
        val http = Http(t, fastLimiter(clock), HttpCache(null, clock), clock, Logger.NONE, sleeper = {})
        val r = http.getText("https://a.test/missing", retries = 3)
        assertIs<HttpResult.Failed>(r)
        assertEquals(404, r.code)
        assertEquals(1, t.requests.size)
    }

    @Test
    fun rateLimiterWaitsForTokens() = runTest {
        var fakeNow = 0L
        val waits = ArrayList<Long>()
        val limiter = RateLimiter(mapOf("api.test" to Rate(60.0, 1)), clock = { fakeNow }, sleeper = { waits += it; fakeNow += it })
        limiter.acquire("api.test")
        limiter.acquire("api.test")
        assertEquals(1, waits.size)
        assertTrue(waits.first() in 900L..1_100L, "about one second per request at 60/min, got ${waits.first()}")
        assertEquals(Rate(10.0, 3), RateLimiter().rateFor("api.coingecko.com"))
        assertEquals(RateLimiter.DEFAULT_RULES.getValue("reddit.com"), RateLimiter().rateFor("www.reddit.com"))
    }
}
