package dev.cryptopulse.core.http

import kotlinx.coroutines.suspendCancellableCoroutine
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/** OkHttp-backed transport usable on Android and the JVM. */
class OkHttpTransport(
    private val userAgent: String = DEFAULT_USER_AGENT,
    client: OkHttpClient? = null,
) : HttpTransport {

    private val client: OkHttpClient = client ?: OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(25, TimeUnit.SECONDS)
        .callTimeout(40, TimeUnit.SECONDS)
        .followRedirects(true)
        .followSslRedirects(true)
        .retryOnConnectionFailure(true)
        .build()

    override suspend fun execute(request: HttpRequest): HttpResponse {
        val builder = Request.Builder().url(request.url)
        builder.header("User-Agent", userAgent)
        builder.header("Accept", "application/json, text/html, application/xml;q=0.9, */*;q=0.8")
        request.headers.forEach { (k, v) -> builder.header(k, v) }
        when (request.method.uppercase()) {
            "GET" -> builder.get()
            "POST" -> builder.post((request.body ?: "").toRequestBody((request.contentType ?: "application/json").toMediaTypeOrNull()))
            else -> builder.method(request.method, request.body?.toRequestBody((request.contentType ?: "application/json").toMediaTypeOrNull()))
        }
        val call = client.newBuilder()
            .callTimeout(request.timeoutMs, TimeUnit.MILLISECONDS)
            .build()
            .newCall(builder.build())
        return call.await()
    }

    private suspend fun Call.await(): HttpResponse = suspendCancellableCoroutine { cont ->
        enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (!cont.isCancelled) cont.resumeWithException(e)
            }

            override fun onResponse(call: Call, response: Response) {
                response.use { r ->
                    val body = try {
                        r.body.string()
                    } catch (e: IOException) {
                        if (!cont.isCancelled) cont.resumeWithException(e)
                        return
                    }
                    val headers = LinkedHashMap<String, String>()
                    for (i in 0 until r.headers.size) headers[r.headers.name(i)] = r.headers.value(i)
                    cont.resume(HttpResponse(r.code, body, headers, r.request.url.toString()))
                }
            }
        })
        cont.invokeOnCancellation { cancel() }
    }

    companion object {
        const val DEFAULT_USER_AGENT: String =
            "CryptoPulse/0.1 (+https://github.com/medonumbworks/spss; crypto overview app; contact via GitHub issues)"
    }
}
