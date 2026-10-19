package dev.cryptopulse.core.http

import dev.cryptopulse.core.util.Clock
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap

/** Key/value persistence for cached responses and snapshots. Android uses the cache dir, tests use memory. */
interface DiskStore {
    fun read(key: String): String?
    fun write(key: String, value: String)
    fun remove(key: String)
}

class InMemoryDiskStore : DiskStore {
    private val map = ConcurrentHashMap<String, String>()
    override fun read(key: String): String? = map[key]
    override fun write(key: String, value: String) { map[key] = value }
    override fun remove(key: String) { map.remove(key) }
    val size: Int get() = map.size
}

/** Two-level (memory + optional disk) cache with explicit timestamps; TTL decisions are made by the caller. */
class HttpCache(
    private val store: DiskStore? = null,
    private val clock: Clock = Clock.SYSTEM,
    private val maxMemoryEntries: Int = 512,
) {
    @Serializable
    data class Entry(val body: String, val storedAtMs: Long) {
        fun ageMs(nowMs: Long): Long = (nowMs - storedAtMs).coerceAtLeast(0)
    }

    private val memory = object : LinkedHashMap<String, Entry>(64, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, Entry>?): Boolean = size > maxMemoryEntries
    }

    @Synchronized
    fun get(key: String): Entry? {
        memory[key]?.let { return it }
        val raw = store?.read(diskKey(key)) ?: return null
        return try {
            json.decodeFromString(Entry.serializer(), raw).also { memory[key] = it }
        } catch (_: Exception) {
            store.remove(diskKey(key))
            null
        }
    }

    @Synchronized
    fun put(key: String, body: String): Entry {
        val entry = Entry(body, clock.nowMs())
        memory[key] = entry
        try {
            store?.write(diskKey(key), json.encodeToString(Entry.serializer(), entry))
        } catch (_: Exception) {
            // disk is best effort
        }
        return entry
    }

    @Synchronized
    fun invalidate(key: String) {
        memory.remove(key)
        store?.remove(diskKey(key))
    }

    private fun diskKey(key: String): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(key.toByteArray())
        return "http_" + digest.joinToString("") { "%02x".format(it) }
    }

    private companion object {
        val json = Json { ignoreUnknownKeys = true }
    }
}
