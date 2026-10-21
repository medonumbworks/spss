package dev.cryptopulse.core.util

enum class LogLevel { DEBUG, INFO, WARN, ERROR }

/**
 * Tiny logging facade. The MCP server must keep stdout clean (it is the stdio transport),
 * so the default implementation writes to stderr. Android wraps android.util.Log.
 */
fun interface Logger {
    fun log(level: LogLevel, tag: String, message: String, error: Throwable?)

    fun debug(tag: String, message: String) = log(LogLevel.DEBUG, tag, message, null)
    fun info(tag: String, message: String) = log(LogLevel.INFO, tag, message, null)
    fun warn(tag: String, message: String, error: Throwable? = null) = log(LogLevel.WARN, tag, message, error)
    fun error(tag: String, message: String, error: Throwable? = null) = log(LogLevel.ERROR, tag, message, error)

    companion object {
        val NONE: Logger = Logger { _, _, _, _ -> }
        val STDERR: Logger = Logger { level, tag, message, error ->
            System.err.println("[$level] $tag: $message" + (error?.let { " (${it::class.simpleName}: ${it.message})" } ?: ""))
        }
    }
}
