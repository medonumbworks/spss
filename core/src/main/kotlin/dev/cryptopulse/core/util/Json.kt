package dev.cryptopulse.core.util

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

/** Lenient JSON helpers: third-party APIs change shapes and send numbers as strings. */
object JsonUtil {
    val lenient: Json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        coerceInputValues = true
        explicitNulls = false
    }

    fun parse(text: String): JsonElement = lenient.parseToJsonElement(text)
}

val JsonElement?.obj: JsonObject? get() = this as? JsonObject
val JsonElement?.arr: JsonArray? get() = this as? JsonArray

operator fun JsonElement?.get(key: String): JsonElement? = (this as? JsonObject)?.get(key)

val JsonElement?.str: String?
    get() = when (this) {
        null, JsonNull -> null
        is JsonPrimitive -> content.takeIf { it != "null" }
        else -> null
    }

val JsonElement?.dbl: Double?
    get() = when (this) {
        null, JsonNull -> null
        is JsonPrimitive -> content.trim().toDoubleOrNull()?.takeIf { !it.isNaN() && !it.isInfinite() }
        else -> null
    }

val JsonElement?.lng: Long? get() = this.dbl?.toLong()
val JsonElement?.int: Int? get() = this.dbl?.toInt()

val JsonElement?.bool: Boolean?
    get() = (this as? JsonPrimitive)?.content?.lowercase()?.let { if (it == "true") true else if (it == "false") false else null }

object Dates {
    private val RSS_PATTERNS: List<DateTimeFormatter> = listOf(
        "d MMM yyyy HH:mm:ss Z", "d MMM yyyy HH:mm Z", "d MMM yyyy HH:mm:ss z", "d MMM yyyy HH:mm z",
    ).map { DateTimeFormatter.ofPattern(it, Locale.ENGLISH) }

    /**
     * Parses ISO-8601 (with or without offset) and RFC-822/1123 dates as used by RSS; returns epoch millis or null.
     * The weekday of RSS dates is ignored because many feeds get it wrong.
     */
    fun parseToMs(raw: String?): Long? {
        val s = raw?.trim()?.takeIf { it.isNotEmpty() } ?: return null
        runCatching { return Instant.parse(s).toEpochMilli() }
        runCatching { return OffsetDateTime.parse(s).toInstant().toEpochMilli() }
        runCatching { return OffsetDateTime.parse(s, DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ssZ")).toInstant().toEpochMilli() }
        val noWeekday = s.substringAfter(',', s).trim().replace(Regex("\\s+"), " ")
        for (f in RSS_PATTERNS) {
            runCatching { return ZonedDateTime.parse(noWeekday, f).toInstant().toEpochMilli() }
        }
        return null
    }
}
