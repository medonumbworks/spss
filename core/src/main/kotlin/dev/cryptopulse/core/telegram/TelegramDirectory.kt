package dev.cryptopulse.core.telegram

import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.dbl
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.str
import kotlinx.serialization.Serializable

@Serializable
data class TelegramEntry(
    val handle: String,
    val title: String,
    /** "channel" (readable without login via t.me/s) or "group" (needs the user's account). */
    val kind: String,
    val lang: String = "en",
    val category: String = "news",
    val symbols: List<String> = emptyList(),
    val confidence: Double = 0.7,
) {
    val isChannel: Boolean get() = kind.equals("channel", ignoreCase = true)
}

/**
 * Curated public crypto Telegram channels and groups (`telegram/channels.json`), shared with the
 * cloud hub. Channels are scanned on the phone; groups are read by the hub through the user's account.
 */
class TelegramDirectory(val entries: List<TelegramEntry>) {

    /** Channels readable without login, best first (news/analytics/whales before signals, English first). */
    fun publicChannels(max: Int = Int.MAX_VALUE): List<TelegramEntry> =
        entries.filter { it.isChannel && it.confidence >= 0.5 }
            .sortedWith(compareByDescending<TelegramEntry> { CATEGORY_PRIORITY[it.category] ?: 0 }.thenByDescending { if (it.lang == "en") 1 else 0 }.thenByDescending { it.confidence })
            .take(max)

    /** Official/project channels that talk about one asset. */
    fun channelsFor(symbol: String): List<TelegramEntry> =
        entries.filter { it.isChannel && symbol.uppercase() in it.symbols }

    val groups: List<TelegramEntry> get() = entries.filter { !it.isChannel }

    companion object {
        private val CATEGORY_PRIORITY = mapOf("news" to 6, "analytics" to 5, "whales" to 5, "exchange" to 4, "bitpanda" to 4, "project" to 3, "community" to 2, "signals" to 1)

        /** Small built-in fallback used only when the bundled resource is missing. */
        val FALLBACK: List<TelegramEntry> = listOf(
            TelegramEntry("cointelegraph", "Cointelegraph", "channel", "en", "news"),
            TelegramEntry("CoinDeskGlobal", "CoinDesk", "channel", "en", "news"),
            TelegramEntry("WatcherGuru", "Watcher Guru", "channel", "en", "news"),
            TelegramEntry("whale_alert_io", "Whale Alert", "channel", "en", "whales"),
            TelegramEntry("binance_announcements", "Binance Announcements", "channel", "en", "exchange"),
        )

        fun parse(json: String): TelegramDirectory {
            val root = JsonUtil.parse(json)
            val list = root["entries"].arr.orEmpty().mapNotNull { e ->
                val handle = e["handle"].str?.trim()?.removePrefix("@")?.takeIf { it.isNotBlank() } ?: return@mapNotNull null
                TelegramEntry(
                    handle = handle,
                    title = e["title"].str ?: handle,
                    kind = e["kind"].str ?: "channel",
                    lang = e["lang"].str ?: "en",
                    category = e["category"].str ?: "news",
                    symbols = e["symbols"].arr.orEmpty().mapNotNull { it.str?.uppercase() },
                    confidence = e["confidence"].dbl ?: 0.7,
                )
            }.distinctBy { it.handle.lowercase() }
            return TelegramDirectory(list)
        }

        fun loadBundled(): TelegramDirectory {
            val text = TelegramDirectory::class.java.getResourceAsStream("/telegram/channels.json")?.use { it.readBytes().toString(Charsets.UTF_8) }
            return text?.let { runCatching { parse(it) }.getOrNull() }?.takeIf { it.entries.isNotEmpty() } ?: TelegramDirectory(FALLBACK)
        }
    }
}
