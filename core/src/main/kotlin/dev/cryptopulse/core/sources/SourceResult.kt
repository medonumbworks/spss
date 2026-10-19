package dev.cryptopulse.core.sources

import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.model.SourceStatus

/** Result of one data-source call. Sources never throw for expected failures; they return [Failure]. */
sealed class SourceResult<out T> {
    data class Success<T>(val value: T, val fromCache: Boolean = false, val ageMs: Long = 0) : SourceResult<T>()
    data class Failure(val status: SourceStatus, val message: String, val httpCode: Int? = null) : SourceResult<Nothing>()

    val valueOrNull: T? get() = (this as? Success)?.value
    val isSuccess: Boolean get() = this is Success

    inline fun <R> map(transform: (T) -> R): SourceResult<R> = when (this) {
        is Success -> Success(transform(value), fromCache, ageMs)
        is Failure -> this
    }

    companion object {
        /** Maps an HTTP failure to a typed source failure. */
        fun failureOf(failed: HttpResult.Failed, what: String): Failure {
            val status = when {
                failed.isRateLimited -> SourceStatus.RATE_LIMITED
                failed.isUnauthorized -> SourceStatus.UNAUTHORIZED
                failed.isGeoBlocked -> SourceStatus.GEO_BLOCKED
                else -> SourceStatus.UNAVAILABLE
            }
            return Failure(status, "$what: ${failed.reason}", failed.code)
        }

        fun parseError(what: String, error: Throwable): Failure =
            Failure(SourceStatus.PARSE_ERROR, "$what: ${error.message ?: error::class.simpleName}")

        fun disabled(what: String): Failure = Failure(SourceStatus.DISABLED, "$what disabled in settings")
    }
}
