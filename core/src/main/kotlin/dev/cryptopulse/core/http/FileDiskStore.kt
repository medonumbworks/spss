package dev.cryptopulse.core.http

import java.io.File

/** [DiskStore] backed by one file per key in [dir] (Android cache dir, ~/.cache/cryptopulse for the MCP server). */
class FileDiskStore(private val dir: File) : DiskStore {
    init {
        dir.mkdirs()
    }

    private fun file(key: String) = File(dir, key.replace(Regex("[^A-Za-z0-9._-]"), "_") + ".json")

    override fun read(key: String): String? = try {
        file(key).takeIf { it.isFile }?.readText()
    } catch (_: Exception) {
        null
    }

    override fun write(key: String, value: String) {
        val target = file(key)
        val tmp = File(dir, target.name + ".tmp")
        tmp.writeText(value)
        if (!tmp.renameTo(target)) {
            target.delete()
            tmp.renameTo(target)
        }
    }

    override fun remove(key: String) {
        file(key).delete()
    }
}
