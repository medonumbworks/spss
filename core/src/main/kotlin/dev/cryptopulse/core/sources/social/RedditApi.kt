package dev.cryptopulse.core.sources.social

import dev.cryptopulse.core.http.Http
import dev.cryptopulse.core.http.HttpResult
import dev.cryptopulse.core.model.SentimentSource
import dev.cryptopulse.core.sources.RedditSource
import dev.cryptopulse.core.sources.SocialPost
import dev.cryptopulse.core.sources.SourceResult
import dev.cryptopulse.core.util.JsonUtil
import dev.cryptopulse.core.util.arr
import dev.cryptopulse.core.util.dbl
import dev.cryptopulse.core.util.get
import dev.cryptopulse.core.util.int
import dev.cryptopulse.core.util.obj
import dev.cryptopulse.core.util.str
import java.net.URLEncoder

/**
 * Reddit's public JSON listings (no login). Reddit asks for a descriptive User-Agent and allows
 * roughly 10 unauthenticated requests per minute; cloud IPs are sometimes refused with 403/429.
 */
class RedditApi(
    private val http: Http,
    private val base: String = "https://www.reddit.com",
    private val ttlMs: Long = 20 * 60_000L,
) : RedditSource {

    override suspend fun search(query: String, subreddit: String?, sort: String, timeRange: String, limit: Int): SourceResult<List<SocialPost>> {
        val q = URLEncoder.encode(query, "UTF-8")
        val path = if (subreddit.isNullOrBlank()) "/search.json" else "/r/${URLEncoder.encode(subreddit, "UTF-8")}/search.json"
        val restrict = if (subreddit.isNullOrBlank()) "" else "&restrict_sr=1"
        val url = "$base$path?q=$q&sort=$sort&t=$timeRange&limit=${limit.coerceIn(1, 100)}&raw_json=1&type=link$restrict"
        return fetch(url, "Reddit search")
    }

    override suspend fun hot(subreddit: String, limit: Int): SourceResult<List<SocialPost>> =
        fetch("$base/r/${URLEncoder.encode(subreddit, "UTF-8")}/hot.json?limit=${limit.coerceIn(1, 100)}&raw_json=1", "Reddit r/$subreddit")

    private suspend fun fetch(url: String, what: String): SourceResult<List<SocialPost>> =
        when (val r = http.getText(url, HEADERS, ttlMs = ttlMs, retries = 1)) {
            is HttpResult.Failed -> SourceResult.failureOf(r, what)
            is HttpResult.Ok -> try {
                SourceResult.Success(parseListing(r.body), r.fromCache, r.ageMs)
            } catch (e: Exception) {
                SourceResult.parseError(what, e)
            }
        }

    companion object {
        private val HEADERS = mapOf("Accept" to "application/json")

        fun parseListing(body: String): List<SocialPost> =
            JsonUtil.parse(body)["data"]["children"].arr.orEmpty().mapNotNull { child ->
                val d = child["data"].obj ?: return@mapNotNull null
                if (d["stickied"].str == "true" || d["over_18"].str == "true") return@mapNotNull null
                val id = d["id"].str ?: return@mapNotNull null
                val created = d["created_utc"].dbl ?: return@mapNotNull null
                SocialPost(
                    source = SentimentSource.REDDIT,
                    id = id,
                    title = d["title"].str ?: "",
                    body = (d["selftext"].str ?: "").take(2000),
                    url = d["permalink"].str?.let { "https://www.reddit.com$it" },
                    author = d["author"].str,
                    createdMs = (created * 1000).toLong(),
                    score = d["score"].int ?: 0,
                    comments = d["num_comments"].int ?: 0,
                    community = d["subreddit"].str?.let { "r/$it" },
                    upvoteRatio = d["upvote_ratio"].dbl,
                )
            }
    }
}
