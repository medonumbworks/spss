"""Newest posts of crypto subreddits via Reddit's public ``.json`` listings (no login).

Reddit often answers 403 or 429 to cloud IP ranges such as GitHub Actions; the status is
recorded and the run continues.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from sentiment_hub.common import Message, SourceResult, clean_text, combine_statuses, describe_counts, status_for_http
from sentiment_hub.net import NetError

log = logging.getLogger("sentiment_hub.reddit")

NAME = "reddit"
SUBREDDITS = ("CryptoCurrency", "CryptoMarkets", "altcoin", "SatoshiStreetBets")
USER_AGENT = "linux:dev.cryptopulse.sentimenthub:1.0 (free hourly crypto sentiment research; +https://github.com/medonumbworks/spss)"
SELFTEXT_CHARS = 1500


def listing_url(subreddit: str) -> str:
    return f"https://www.reddit.com/r/{subreddit}/new.json?limit=100&raw_json=1"


def parse_listing(payload: Any, now: datetime, lookback_hours: float) -> list[Message]:
    cutoff = now - timedelta(hours=lookback_hours)
    out: list[Message] = []
    children = (payload or {}).get("data", {}).get("children", []) if isinstance(payload, dict) else []
    for child in children:
        data = child.get("data") if isinstance(child, dict) else None
        if not isinstance(data, dict) or data.get("stickied"):
            continue
        try:
            ts = datetime.fromtimestamp(float(data.get("created_utc")), tz=timezone.utc)
        except (TypeError, ValueError):
            continue
        if ts < cutoff or ts > now + timedelta(minutes=10):
            continue
        title = clean_text(data.get("title"))
        body = data.get("selftext") or ""
        if body.strip() in ("[removed]", "[deleted]"):
            body = ""
        text = clean_text(f"{title}\n{body[:SELFTEXT_CHARS]}")
        if not text:
            continue
        post_id = str(data.get("id") or data.get("name") or "")
        permalink = data.get("permalink") or ""
        url = f"https://www.reddit.com{permalink}" if permalink.startswith("/") else (data.get("url") or None)
        try:
            score = max(0.0, float(data.get("score") or 0))
        except (TypeError, ValueError):
            score = 0.0
        out.append(
            Message(
                source=NAME,
                key=f"reddit:{post_id or url or title.lower()}",
                text=text,
                ts=ts,
                url=url,
                community="r/" + str(data.get("subreddit") or ""),
                public=True,
                engagement=score,
            )
        )
    return out


def collect(
    http,
    now: datetime,
    lookback_hours: float = 48,
    subreddits: Sequence[str] = SUBREDDITS,
    pause_s: float = 2.0,
) -> SourceResult:
    items: list[Message] = []
    seen: set[str] = set()
    statuses: list[str] = []
    problems: Counter = Counter()
    ok_subs = 0
    for i, sub in enumerate(subreddits):
        if i and pause_s:
            time.sleep(pause_s)
        url = listing_url(sub)
        try:
            resp = http.get(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        except NetError as exc:
            log.info("reddit r/%s: %s", sub, exc)
            statuses.append("unavailable")
            problems["unavailable"] += 1
            continue
        if resp.status != 200:
            mapped = status_for_http(resp.status)
            log.info("reddit r/%s: HTTP %d", sub, resp.status)
            statuses.append(mapped)
            problems[f"HTTP {resp.status}"] += 1
            continue
        try:
            posts = parse_listing(resp.json(), now, lookback_hours)
        except ValueError as exc:  # HTML block page instead of JSON
            log.info("reddit r/%s: not JSON (%s)", sub, exc)
            statuses.append("error")
            problems["not JSON"] += 1
            continue
        ok_subs += 1
        fresh = [p for p in posts if p.key not in seen]
        seen.update(p.key for p in fresh)
        items.extend(fresh)
        statuses.append("ok" if fresh else "empty")
    detail = f"{ok_subs}/{len(subreddits)} subreddits"
    if problems:
        detail += f" ({describe_counts(problems)})"
    status = combine_statuses(statuses, has_items=bool(items))
    log.info("reddit: %s, %d posts, %s", status, len(items), detail)
    return SourceResult(NAME, status, items, detail)
