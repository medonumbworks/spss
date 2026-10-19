package dev.cryptopulse.core.sources.social

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.model.SentimentSource
import dev.cryptopulse.core.model.SourceStatus
import dev.cryptopulse.core.sources.SocialPost
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.sources.TelegramSource
import dev.cryptopulse.core.util.Dates
import org.jsoup.Jsoup

/**
 * Reads public Telegram channels without an account through the official web preview
 * `https://t.me/s/<channel>` (the same page telegram.org shows to browsers). Only broadcast
 * channels have a preview; groups need the user's account (see tools/sentiment_hub). When t.me
 * cannot be reached (the domain was suspended for a day in July 2026) the same page is read from
 * telegram.me.
 */
class TelegramWeb(
    private val http: Http,
    private val base: String = DEFAULT_BASE,
    private val ttlMs: Long = 20 * 60_000L,
) : TelegramSource {

    override suspend fun channelMessages(channel: String): SourceResult<List<SocialPost>> {
        val handle = channel.trim().removePrefix("@").removePrefix("https://t.me/").removePrefix("s/").trim('/')
        if (!HANDLE.matches(handle)) return SourceResult.Failure(SourceStatus.PARSE_ERROR, "Invalid Telegram handle: $channel")
        val headers = mapOf("Accept" to "text/html", "Accept-Language" to "en")
        var r = http.getText("$base/s/$handle", headers, ttlMs = ttlMs, retries = 1)
        if (r is HttpResult.Failed && r.code == null && base == DEFAULT_BASE) {
            r = http.getText("$MIRROR_BASE/s/$handle", headers, ttlMs = ttlMs, retries = 0)
        }
        return when (r) {
            is HttpResult.Failed -> SourceResult.failureOf(r, "Telegram @$handle")
            is HttpResult.Ok -> try {
                val posts = parsePreview(r.body, handle)
                if (posts.isEmpty() && !r.body.contains("tgme_widget_message")) {
                    SourceResult.Failure(SourceStatus.EMPTY, "Telegram @$handle has no public preview (group or private)")
                } else {
                    SourceResult.Success(posts, r.fromCache, r.ageMs)
                }
            } catch (e: Exception) {
                SourceResult.parseError("Telegram @$handle", e)
            }
        }
    }

    companion object {
        private val HANDLE = Regex("^[A-Za-z][A-Za-z0-9_]{3,31}$")
        const val DEFAULT_BASE = "https://t.me"
        const val MIRROR_BASE = "https://telegram.me"

        /** Parses the t.me/s HTML preview. Messages without text (photo-only) are skipped. */
        fun parsePreview(html: String, handle: String): List<SocialPost> {
            val doc = Jsoup.parse(html, "https://t.me/")
            return doc.select("div.tgme_widget_message[data-post]").mapNotNull { msg ->
                val post = msg.attr("data-post").takeIf { it.isNotBlank() } ?: return@mapNotNull null
                val textEl = msg.selectFirst("div.tgme_widget_message_text") ?: return@mapNotNull null
                textEl.select("br").forEach { it.after(" ") }
                val text = textEl.text().trim()
                if (text.isEmpty()) return@mapNotNull null
                val time = msg.selectFirst("a.tgme_widget_message_date time[datetime]")?.attr("datetime")
                    ?: msg.selectFirst("time[datetime]")?.attr("datetime")
                val created = Dates.parseToMs(time) ?: return@mapNotNull null
                val views = parseViews(msg.selectFirst("span.tgme_widget_message_views")?.text())
                SocialPost(
                    source = SentimentSource.TELEGRAM,
                    id = post,
                    title = text.take(1500),
                    url = "https://t.me/$post",
                    author = msg.selectFirst("a.tgme_widget_message_owner_name")?.text(),
                    createdMs = created,
                    score = views,
                    community = "@" + post.substringBefore('/').ifBlank { handle },
                )
            }
        }

        /** "1.2K" -> 1200, "3.4M" -> 3400000, "987" -> 987. */
        fun parseViews(raw: String?): Int {
            val s = raw?.trim()?.uppercase()?.replace(",", "") ?: return 0
            val mult = when {
                s.endsWith("K") -> 1_000.0
                s.endsWith("M") -> 1_000_000.0
                else -> 1.0
            }
            val num = s.trimEnd('K', 'M').toDoubleOrNull() ?: return 0
            return (num * mult).toLong().coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
        }
    }
}
