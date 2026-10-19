package dev.cryptopulse.core.testing

import dev.cryptopulse.core.http.HttpRequest
import dev.cryptopulse.core.http.HttpResponse
import dev.cryptopulse.core.http.HttpTransport
import java.io.IOException

/**
 * Transport for tests and the offline demo mode: routes are matched in order; the first matcher
 * whose predicate accepts the URL wins. Unmatched URLs return 404 (or throw when [strict]).
 */
class FakeTransport(private val strict: Boolean = false) : HttpTransport {
    class Route(val matches: (HttpRequest) -> Boolean, val respond: (HttpRequest) -> HttpResponse)

    private val routes = ArrayList<Route>()
    val requests = ArrayList<HttpRequest>()

    fun on(predicate: (HttpRequest) -> Boolean, respond: (HttpRequest) -> HttpResponse): FakeTransport {
        routes += Route(predicate, respond)
        return this
    }

    /** Matches when the URL contains [fragment]. */
    fun onUrlContaining(fragment: String, code: Int = 200, body: String, headers: Map<String, String> = emptyMap()): FakeTransport =
        on({ it.url.contains(fragment) }) { HttpResponse(code, body, headers, it.url) }

    /** Matches when the URL contains [fragment]; body is produced lazily from the request. */
    fun onUrlContaining(fragment: String, respond: (HttpRequest) -> HttpResponse): FakeTransport =
        on({ it.url.contains(fragment) }, respond)

    /** Simulates a transport failure (DNS/timeout). */
    fun failUrlContaining(fragment: String, message: String = "simulated network failure"): FakeTransport =
        on({ it.url.contains(fragment) }) { throw IOException(message) }

    override suspend fun execute(request: HttpRequest): HttpResponse {
        requests += request
        val route = routes.firstOrNull { it.matches(request) }
        if (route == null) {
            if (strict) throw IllegalStateException("No fake route for ${request.url}")
            return HttpResponse(404, "", emptyMap(), request.url)
        }
        return route.respond(request)
    }

    fun countRequests(fragment: String): Int = requests.count { it.url.contains(fragment) }
}

/** Loads fixture files from the classpath (`/fixtures/<name>`), shared by tests and offline mode. */
object Fixtures {
    fun read(name: String): String =
        Fixtures::class.java.getResourceAsStream("/fixtures/$name")?.use { it.readBytes().toString(Charsets.UTF_8) }
            ?: throw IllegalArgumentException("Fixture not found on classpath: /fixtures/$name")

    fun exists(name: String): Boolean = Fixtures::class.java.getResource("/fixtures/$name") != null
}
