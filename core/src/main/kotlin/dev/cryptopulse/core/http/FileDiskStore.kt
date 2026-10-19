package dev.cryptopulse.core.http

import java.io.File
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption

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
        // unique temp file per write: concurrent writers of one key must not share (and lose) it
        val tmp = File.createTempFile(target.name, ".tmp", dir)
        try {
            tmp.writeText(value)
            try {
                Files.move(tmp.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(tmp.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }
        } finally {
            tmp.delete()
        }
    }

    override fun remove(key: String) {
        file(key).delete()
    }
}
