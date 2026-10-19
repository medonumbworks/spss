package dev.cryptopulse.core

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpCache
import dev.cryptopulse.core.http.Rate
import dev.cryptopulse.core.http.RateLimiter
import dev.cryptopulse.core.testing.FakeTransport
import dev.cryptopulse.core.util.Clock
import dev.cryptopulse.core.util.Logger

/** 2026-10-17T15:00:00Z, the moment the fixtures were "recorded". */
const val FIXTURE_NOW_MS: Long = 1_760_713_200_000L

fun fastLimiter(clock: Clock = Clock.fixed(FIXTURE_NOW_MS)) =
    RateLimiter(rules = emptyMap(), defaultRate = Rate(requestsPerMinute = 1_000_000.0, burst = 1_000_000), clock = clock, sleeper = {})

fun testHttp(transport: FakeTransport, clock: Clock = Clock.fixed(FIXTURE_NOW_MS)) =
    Http(transport, fastLimiter(clock), HttpCache(null, clock), clock, Logger.NONE, sleeper = {})
