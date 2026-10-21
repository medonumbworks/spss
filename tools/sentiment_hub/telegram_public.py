"""Public Telegram channels via the login-free web preview ``https://t.me/s/<handle>``.

Only broadcast channels have a preview; groups and users redirect to ``https://t.me/<handle>``
and are skipped (the account source can read public groups). Requests are polite: one per
second for the whole host, at most four channels in flight, a browser-like User-Agent.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from sentiment_hub.assets import ChannelEntry, normalize_handle
from sentiment_hub.common import Message, SourceResult, clean_text, combine_statuses, describe_counts, parse_iso, status_for_http
from sentiment_hub.net import NetError, retry_after_seconds

log = logging.getLogger("sentiment_hub.telegram_public")

NAME = "telegram_public"
BASE = "https://t.me"
# Same preview pages; used when t.me cannot be reached (the domain was suspended for a day in July 2026).
MIRROR_BASE = "https://telegram.me"
DEFAULT_MAX_PAGES = 5
DEFAULT_CONCURRENCY = 4
REQUEST_INTERVAL_S = 1.0

_VIEWS_RE = re.compile(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*([KMB])?\s*$", re.IGNORECASE)
_MULTIPLIERS = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


@dataclass
class PreviewPost:
    post_id: int
    data_post: str
    text: str
    ts: datetime | None
    views: int


@dataclass
class PreviewPage:
    is_preview: bool
    posts: list[PreviewPost] = field(default_factory=list)
    #: ``data-before`` of the "load older messages" link, when present.
    before: int | None = None


@dataclass
class ChannelFetch:
    handle: str
    #: ok | empty | no_preview | unavailable | rate_limited | unauthorized | error | skipped
    status: str
    messages: list[Message] = field(default_factory=list)
    pages: int = 0
    note: str = ""


def parse_views(raw: str | None) -> int:
    """``"523"`` -> 523, ``"1.2K"`` -> 1200, ``"3.4M"`` -> 3400000, ``"1,234"`` -> 1234."""
    if not raw:
        return 0
    text = raw.strip().replace(" ", "")
    if re.fullmatch(r"[0-9]{1,3}(,[0-9]{3})+", text):
        return int(text.replace(",", ""))
    match = _VIEWS_RE.match(text)
    if not match:
        return 0
    number = float(match.group(1).replace(",", "."))
    suffix = (match.group(2) or "").upper()
    return int(round(number * _MULTIPLIERS.get(suffix, 1)))


def _message_text(node) -> str:
    candidates = node.select("div.tgme_widget_message_text")
    chosen = None
    for cand in candidates:
        if cand.find_parent(class_="tgme_widget_message_reply") is not None:
            continue  # quoted message of a reply
        if "js-message_text" in (cand.get("class") or []):
            chosen = cand
            break
        if chosen is None:
            chosen = cand
    if chosen is None:
        return ""
    return clean_text(chosen.get_text(" ", strip=True))


def parse_preview(html: str) -> PreviewPage:
    soup = BeautifulSoup(html or "", "html.parser")
    nodes = soup.select("div.tgme_widget_message[data-post]")
    is_preview = bool(nodes) or soup.select_one(".tgme_channel_history") is not None
    posts: list[PreviewPost] = []
    for node in nodes:
        data_post = str(node.get("data-post") or "").strip()
        _, _, raw_id = data_post.rpartition("/")
        if not raw_id.isdigit():
            continue
        time_node = node.select_one("a.tgme_widget_message_date time[datetime]")
        views_node = node.select_one("span.tgme_widget_message_views")
        posts.append(
            PreviewPost(
                post_id=int(raw_id),
                data_post=data_post,
                text=_message_text(node),
                ts=parse_iso(time_node.get("datetime")) if time_node is not None else None,
                views=parse_views(views_node.get_text(strip=True) if views_node is not None else None),
            )
        )
    before = None
    more = soup.select_one("a.tme_messages_more[data-before]")
    if more is not None and str(more.get("data-before")).isdigit():
        before = int(more.get("data-before"))
    return PreviewPage(is_preview=is_preview, posts=posts, before=before)


def _is_preview_url(url: str) -> bool:
    return urlparse(url).path.lower().startswith("/s/")


def _get(http, url: str):
    """GET with one polite retry after a 429."""
    resp = http.get(url)
    if resp.status == 429:
        wait = retry_after_seconds(resp)
        limiter = http.limiter_for(url) if hasattr(http, "limiter_for") else None
        if limiter is not None:
            limiter.penalize(wait)
        else:
            time.sleep(wait)
        resp = http.get(url)
    return resp


def fetch_channel(
    http,
    handle: str,
    now: datetime,
    lookback_hours: float = 48,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> ChannelFetch:
    cutoff = now - timedelta(hours=lookback_hours)
    base_url = f"{BASE}/s/{handle}"
    posts: dict[int, PreviewPost] = {}
    before: int | None = None
    pages = 0
    status = "ok"
    note = ""
    while pages < max_pages:
        url = base_url if before is None else f"{base_url}?before={before}"
        try:
            resp = _get(http, url)
        except NetError as exc:
            if pages == 0 and base_url.startswith(BASE + "/"):
                base_url = f"{MIRROR_BASE}/s/{handle}"
                continue
            if pages == 0:
                return ChannelFetch(handle, "unavailable", note=str(exc))
            note = f"page {pages + 1}: {exc}"
            break
        if resp.status != 200:
            mapped = status_for_http(resp.status)
            if pages == 0:
                return ChannelFetch(handle, mapped, note=f"HTTP {resp.status}")
            note = f"page {pages + 1}: HTTP {resp.status}"
            break
        if not _is_preview_url(resp.url):
            return ChannelFetch(handle, "no_preview", note="redirected to " + urlparse(resp.url).path)
        try:
            page = parse_preview(resp.text)
        except Exception as exc:  # pragma: no cover - bs4's html.parser is very lenient
            if pages == 0:
                return ChannelFetch(handle, "error", note=f"parse: {exc}")
            break
        pages += 1
        if not page.is_preview:
            if pages == 1:
                return ChannelFetch(handle, "no_preview", note="not a channel preview")
            break
        fresh = [p for p in page.posts if p.post_id not in posts]
        if not fresh:
            break
        for p in fresh:
            posts[p.post_id] = p
        min_id = min(p.post_id for p in page.posts)
        dated = [p.ts for p in page.posts if p.ts is not None]
        if (dated and min(dated) < cutoff) or min_id <= 1:
            break
        if before is not None and min_id >= before:
            break  # no progress
        before = min_id

    messages: list[Message] = []
    for p in sorted(posts.values(), key=lambda x: x.post_id):
        if not p.text or p.ts is None or p.ts < cutoff or p.ts > now + timedelta(minutes=10):
            continue
        messages.append(
            Message(
                source=NAME,
                key=f"tg:{p.data_post.lower()}",
                text=p.text,
                ts=p.ts,
                url=f"{BASE}/{p.data_post}",
                community=handle,
                public=True,
                engagement=float(p.views),
            )
        )
    if not messages:
        status = "empty"
    return ChannelFetch(handle, status, messages, pages, note)


def collect(
    entries: Sequence[ChannelEntry],
    http,
    now: datetime,
    lookback_hours: float = 48,
    max_pages: int = DEFAULT_MAX_PAGES,
    concurrency: int = DEFAULT_CONCURRENCY,
    budget_s: float | None = None,
) -> SourceResult:
    handles: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if getattr(entry, "kind", "channel") == "group":
            continue  # groups have no web preview; the account source reads them
        handle = normalize_handle(entry.handle)
        if handle and handle.lower() not in seen:
            seen.add(handle.lower())
            handles.append(handle)
    if not handles:
        return SourceResult(NAME, "empty", detail="no channels configured")

    deadline = time.monotonic() + budget_s if budget_s else None
    progress_lock = threading.Lock()
    done = [0]

    def work(handle: str) -> ChannelFetch:
        if deadline is not None and time.monotonic() > deadline:
            return ChannelFetch(handle, "skipped", note="time budget exhausted")
        try:
            result = fetch_channel(http, handle, now, lookback_hours, max_pages)
        except Exception as exc:  # never let one channel fail the run
            log.warning("t.me/s/%s: unexpected %s: %s", handle, type(exc).__name__, exc)
            result = ChannelFetch(handle, "error", note=type(exc).__name__)
        with progress_lock:
            done[0] += 1
            log.debug("t.me/s/%s: %s, %d messages, %d pages %s", handle, result.status, len(result.messages), result.pages, result.note)
            if done[0] % 25 == 0:
                log.info("telegram_public: %d/%d channels done", done[0], len(handles))
        return result

    with ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="tme") as pool:
        results = list(pool.map(work, handles))

    items = [m for r in results for m in r.messages]
    readable = [r for r in results if r.status in ("ok", "empty")]
    counts = Counter(r.status for r in results if r.status not in ("ok", "empty"))
    detail = f"{len(readable)}/{len(handles)} channels"
    if counts:
        detail += f" ({describe_counts(counts)})"
    statuses = ["unavailable" if r.status in ("no_preview", "skipped") else r.status for r in results]
    status = combine_statuses(statuses, has_items=bool(items))
    log.info("telegram_public: %s, %d messages, %s", status, len(items), detail)
    return SourceResult(
        NAME,
        status,
        items,
        detail,
        meta={
            "covered_handles": {r.handle.lower() for r in readable},
            "per_channel": {r.handle: r.status for r in results},
        },
    )
