package dev.cryptopulse.mcp

import dev.cryptopulse.core.CryptoPulse
import dev.cryptopulse.core.config.AppConfig
import dev.cryptopulse.core.http.FileDiskStore
import io.ktor.server.cio.CIO
import io.ktor.server.engine.embeddedServer
import io.ktor.utils.io.streams.asInput
import io.modelcontextprotocol.kotlin.sdk.server.StdioServerTransport
import io.modelcontextprotocol.kotlin.sdk.server.mcpStreamableHttp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.runBlocking
import kotlinx.io.asSink
import kotlinx.io.buffered
import java.io.File
import kotlin.system.exitProcess

private const val USAGE = """CryptoPulse MCP server - Bitpanda crypto prices, 24h/7d forecasts and sentiment (free sources only)

Usage:
  cryptopulse-mcp                       MCP over stdio (Claude Desktop, Claude Code, Cursor, ...)
  cryptopulse-mcp --http [port]         MCP over Streamable HTTP on http://127.0.0.1:<port>/mcp (default 3333)
  cryptopulse-mcp digest [en|hr]        Print a market digest (used by the Telegram notifier)
  cryptopulse-mcp check                 Refresh once from the live APIs and print a health report
  cryptopulse-mcp export <path>         Refresh once and write the PWA dashboard JSON (cryptopulse.dashboard/v1)

Environment (all optional):
  CRYPTOPULSE_CURRENCY      EUR (default), USD, CHF, GBP
  COINGECKO_DEMO_KEY        free CoinGecko demo key for higher rate limits
  BITPANDA_API_KEY          only needed if Bitpanda stops serving its price list without a key
  CRYPTOPULSE_FEED_URL      cloud sentiment feed URL (default: this repo's sentiment-feed branch; empty disables)
  CRYPTOPULSE_CACHE_DIR     cache directory (default ~/.cache/cryptopulse)
"""

fun configFromEnv(): AppConfig {
    val env = System.getenv()
    val base = AppConfig()
    return base.copy(
        currency = env["CRYPTOPULSE_CURRENCY"]?.uppercase()?.takeIf { it in AppConfig.SUPPORTED_CURRENCIES } ?: base.currency,
        coingeckoDemoKey = env["COINGECKO_DEMO_KEY"]?.takeIf { it.isNotBlank() },
        bitpandaApiKey = env["BITPANDA_API_KEY"]?.takeIf { it.isNotBlank() },
        sentimentFeedUrl = env["CRYPTOPULSE_FEED_URL"] ?: base.sentimentFeedUrl,
        backgroundSentimentTopN = 20,
    )
}

fun createApp(): CryptoPulse {
    val cacheDir = System.getenv("CRYPTOPULSE_CACHE_DIR")?.let { File(it) } ?: File(System.getProperty("user.home"), ".cache/cryptopulse")
    return CryptoPulse.create(config = configFromEnv(), store = FileDiskStore(cacheDir))
}

fun main(args: Array<String>) {
    // stdout is the MCP stdio channel: nothing else may print there.
    System.setProperty("kotlin-logging.logStartupMessage", "false")
    System.setProperty("org.slf4j.simpleLogger.logFile", System.getProperty("org.slf4j.simpleLogger.logFile") ?: "System.err")
    when (args.firstOrNull()) {
        "-h", "--help", "help" -> { println(USAGE); return }
        "digest" -> {
            val app = createApp()
            val snap = runBlocking { app.repository.refresh(force = true).also { app.repository.enrichSentiment(10) } }
            // alertCrossings() consumes the persisted crossing state of this cache dir: fine for the
            // telegram-digest workflow (its own cache), but do not share CRYPTOPULSE_CACHE_DIR with
            // another alert consumer such as the dashboard workflow
            val extras = Digest.Extras(
                buzz = runBlocking { app.repository.telegramBuzz(5) },
                journal = app.repository.journalStats(),
                alerts = if (app.config.alertsEnabled) app.repository.alertCrossings() else emptyList(),
                alertThresholdPct = app.config.alertThresholdPct,
            )
            println(Digest.build(app.repository.snapshot.value ?: snap, args.getOrNull(1) ?: "en", extras = extras))
            exitProcess(0)
        }
        "check" -> {
            val app = createApp()
            println(runBlocking { LiveCheck.run(app) })
            exitProcess(0)
        }
        "export" -> {
            val out = args.getOrNull(1) ?: run { System.err.println(USAGE); exitProcess(2) }
            val app = createApp()
            val body = runBlocking { Export.build(app) }
            val file = File(out).absoluteFile
            file.parentFile?.mkdirs()
            file.writeText(body)
            val snap = app.repository.snapshot.value
            println("Dashboard: ${snap?.rows?.size ?: 0} rows, ${snap?.warnings?.size ?: 0} warnings, ${body.length} chars -> $file")
            exitProcess(0)
        }
        "--http" -> runHttp(args.getOrNull(1)?.toIntOrNull() ?: 3333)
        null, "--stdio" -> runStdio()
        else -> { System.err.println(USAGE); exitProcess(2) }
    }
}

private fun runStdio() {
    val protocolOut = System.out
    System.setOut(System.err) // any stray println goes to stderr, never into the JSON-RPC stream
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    val tools = CryptoPulseTools(createApp(), scope)
    val server = tools.createServer()
    // MCP stdio clients shut a server down by closing its stdin; exit shortly after EOF so nothing lingers.
    val input = ExitOnEofInputStream(System.`in`) { protocolOut.flush() }
    val transport = StdioServerTransport(input.asInput(), protocolOut.asSink().buffered())
    runBlocking {
        val session = server.createSession(transport)
        val done = Job()
        session.onClose { done.complete() }
        done.join()
    }
    exitProcess(0)
}

/** Calls [beforeExit] and exits the JVM ~1.5 s after the wrapped stream reports end of input. */
private class ExitOnEofInputStream(input: java.io.InputStream, private val beforeExit: () -> Unit) : java.io.FilterInputStream(input) {
    @Volatile
    private var scheduled = false

    override fun read(): Int = super.read().also { if (it < 0) scheduleExit() }

    override fun read(b: ByteArray, off: Int, len: Int): Int = super.read(b, off, len).also { if (it < 0) scheduleExit() }

    private fun scheduleExit() {
        if (scheduled) return
        scheduled = true
        Thread {
            Thread.sleep(1_500)
            runCatching { beforeExit() }
            exitProcess(0)
        }.apply { isDaemon = true; name = "stdin-eof-exit" }.start()
    }
}

private fun runHttp(port: Int) {
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    val tools = CryptoPulseTools(createApp(), scope)
    System.err.println("CryptoPulse MCP listening on http://127.0.0.1:$port/mcp")
    embeddedServer(CIO, host = "127.0.0.1", port = port) {
        mcpStreamableHttp { tools.createServer() }
    }.start(wait = true)
}
