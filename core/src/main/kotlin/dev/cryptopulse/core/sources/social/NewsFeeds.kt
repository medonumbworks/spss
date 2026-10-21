package dev.cryptopulse.core.sources.social

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.sources.NewsItem
import dev.cryptopulse.core.sources.NewsSource
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.util.Dates
import org.jsoup.Jsoup
import org.jsoup.parser.Parser
import java.net.URLEncoder

/** Free news: Google News RSS search plus any crypto RSS/Atom feed. */
class NewsFeeds(
    private val http: Http,
    private val ttlMs: Long = 20 * 60_000L,
) : NewsSource {

    override suspend fun googleNews(query: String, lang: String, country: String): SourceResult<List<NewsItem>> {
        val q = URLEncoder.encode("$query when:7d", "UTF-8")
        val hl = lang
        val gl = country
        val ceid = "$country:${lang.substringBefore('-')}"
        val url = "https://news.google.com/rss/search?q=$q&hl=$hl&gl=$gl&ceid=${URLEncoder.encode(ceid, "UTF-8")}"
        return fetch(url, "Google News")
    }

    override suspend fun rssFeed(feedUrl: String, sourceName: String): SourceResult<List<NewsItem>> = fetch(feedUrl, sourceName)

    private suspend fun fetch(url: String, name: String): SourceResult<List<NewsItem>> =
        when (val r = http.getText(url, mapOf("Accept" to "application/rss+xml, application/atom+xml, application/xml, text/xml"), ttlMs = ttlMs, retries = 1)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, name)
            is HttpResult.Ok -> try {
                SourceResult.Success(parseFeed(r.body, name), r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError(name, e)
            }
        }

    companion object {
        /** Parses RSS 2.0 (`item`) and Atom (`entry`). HTML in descriptions is stripped. */
        fun parseFeed(xml: String, defaultSource: String): List<NewsItem> {
            val doc = Jsoup.parse(xml, "", Parser.xmlParser())
            val items = doc.select("item").map { item ->
                NewsItem(
                    title = clean(item.selectFirst("title")?.text()),
                    summary = item.selectFirst("description")?.text()?.let { clean(it) }?.takeIf { it.isNotBlank() }?.take(600),
                    url = item.selectFirst("link")?.text()?.trim()?.takeIf { it.isNotBlank() },
                    sourceName = item.selectFirst("source")?.text()?.trim()?.takeIf { it.isNotBlank() } ?: defaultSource,
                    publishedMs = Dates.parseToMs(item.selectFirst("pubDate")?.text() ?: item.selectFirst("dc|date")?.text()),
                )
            }
            val entries = doc.select("entry").map { e ->
                NewsItem(
                    title = clean(e.selectFirst("title")?.text()),
                    summary = (e.selectFirst("summary") ?: e.selectFirst("content"))?.text()?.let { clean(it) }?.takeIf { it.isNotBlank() }?.take(600),
                    url = e.select("link").firstOrNull { it.attr("rel").isBlank() || it.attr("rel") == "alternate" }?.attr("href")?.takeIf { it.isNotBlank() },
                    sourceName = defaultSource,
                    publishedMs = Dates.parseToMs(e.selectFirst("published")?.text() ?: e.selectFirst("updated")?.text()),
                )
            }
            return (items + entries).filter { it.title.isNotBlank() }
        }

        private fun clean(raw: String?): String {
            if (raw.isNullOrBlank()) return ""
            val noTags = if (raw.contains('<')) Jsoup.parse(raw).text() else raw
            return noTags.replace(Regex("\\s+"), " ").trim()
        }
    }
}
