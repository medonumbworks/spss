package dev.cryptopulse.core.http

import java.io.IOException

data class HttpRequest(
    val url: String,
    val headers: Map<String, String> = emptyMap(),
    val method: String = "GET",
    val body: String? = null,
    val contentType: String? = null,
    val timeoutMs: Long = 20_000,
)

data class HttpResponse(
    val code: Int,
    val body: String,
    val headers: Map<String, String> = emptyMap(),
    val finalUrl: String = "",
) {
    val isSuccess: Boolean get() = code in 200..299

    fun header(name: String): String? = headers.entries.firstOrNull { it.key.equals(name, ignoreCase = true) }?.value

    /** Retry-After header in milliseconds when present and numeric. */
    val retryAfterMs: Long?
        get() = header("Retry-After")?.trim()?.toLongOrNull()?.let { it * 1000 }
}

/**
 * The only thing that talks to the network. Implementations: [OkHttpTransport] for production,
 * `FakeTransport`/`FixtureTransport` for tests and offline demo mode.
 *
 * Implementations must return non-2xx responses as [HttpResponse] (not throw) and throw
 * [IOException] only for transport failures (DNS, timeout, TLS, connection reset).
 */
interface HttpTransport {
    @Throws(IOException::class)
    suspend fun execute(request: HttpRequest): HttpResponse
}
