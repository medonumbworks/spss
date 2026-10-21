"""Crypto news headlines from free RSS 2.0 / Atom / RSS 1.0 feeds, parsed with ``xml.etree``."""

from __future__ import annotations

import html
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from html.entities import name2codepoint
from typing import Sequence
from urllib.parse import urljoin, urlsplit

from sentiment_hub.common import Message, SourceResult, as_utc, clean_text, combine_statuses, describe_counts, is_http_url, parse_iso, status_for_http
from sentiment_hub.net import NetError

log = logging.getLogger("sentiment_hub.rss")

NAME = "rss"
FEEDS: tuple[tuple[str, str], ...] = (
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Decrypt", "https://decrypt.co/feed"),
    ("Bitcoin Magazine", "https://bitcoinmagazine.com/.rss/full/"),
    ("CryptoSlate", "https://cryptoslate.com/feed/"),
    ("The Block", "https://www.theblock.co/rss.xml"),
)
SUMMARY_CHARS = 600
_XML_ENTITIES = {"amp", "lt", "gt", "quot", "apos"}
_ENTITY_RE = re.compile(r"&(#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);?")
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class FeedItem:
    title: str
    link: str | None
    summary: str
    ts: datetime | None
    guid: str | None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def _fix_entities(text: str) -> str:
    """HTML entities that XML does not define (``&nbsp;``) and bare ``&`` break ElementTree."""

    def repl(match: re.Match) -> str:
        body = match.group(1)
        if body.startswith("#"):
            return match.group(0) if match.group(0).endswith(";") else match.group(0) + ";"
        if body in _XML_ENTITIES and match.group(0).endswith(";"):
            return match.group(0)
        if body in name2codepoint and match.group(0).endswith(";"):
            return f"&#{name2codepoint[body]};"
        return "&amp;" + match.group(0)[1:]

    fixed = _ENTITY_RE.sub(repl, text)
    return re.sub(r"&(?![A-Za-z#][A-Za-z0-9]*;)", "&amp;", fixed)


def _html_to_text(raw: str | None) -> str:
    if not raw:
        return ""
    text = raw
    if "<" in text:
        try:
            from bs4 import BeautifulSoup

            text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
        except Exception:
            text = _TAG_RE.sub(" ", text)
    return clean_text(html.unescape(text))


def _parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    raw = raw.strip()
    try:
        return as_utc(parsedate_to_datetime(raw))
    except (TypeError, ValueError, IndexError):
        pass
    return parse_iso(raw)


def _child_text(elem: ET.Element, *names: str) -> str | None:
    for name in names:
        for child in elem:
            if _local(child.tag) == name:
                text = "".join(child.itertext()).strip()
                if text:
                    return text
    return None


def _atom_link(entry: ET.Element) -> str | None:
    best = None
    for child in entry:
        if _local(child.tag) != "link":
            continue
        href = (child.get("href") or "").strip()
        rel = (child.get("rel") or "alternate").lower()
        if href and rel == "alternate":
            return href
        if href and best is None:
            best = href
        if not href and (child.text or "").strip():
            return child.text.strip()  # RSS 2.0 style <link>url</link>
    return best


def parse_feed(content: bytes | str) -> list[FeedItem]:
    """Parses RSS 2.0, RSS 1.0 (RDF) and Atom. Raises ``ET.ParseError`` on hopeless input."""
    data = content.encode("utf-8") if isinstance(content, str) else content
    data = data.lstrip(b"\xef\xbb\xbf \t\r\n")
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        root = ET.fromstring(_fix_entities(data.decode("utf-8", errors="replace")).encode("utf-8"))
    items: list[FeedItem] = []
    for elem in root.iter():
        kind = _local(elem.tag)
        if kind not in ("item", "entry"):
            continue
        title = _html_to_text(_child_text(elem, "title"))
        summary = _html_to_text(_child_text(elem, "description", "summary", "encoded", "content"))
        if kind == "entry":
            link = _atom_link(elem)
            ts = _parse_date(_child_text(elem, "published", "updated", "date"))
            guid = _child_text(elem, "id")
        else:
            link = _child_text(elem, "link") or _atom_link(elem) or elem.get(
                "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
            )
            ts = _parse_date(_child_text(elem, "pubdate", "date", "published", "updated"))
            guid = _child_text(elem, "guid")
        if not title and not summary:
            continue
        items.append(FeedItem(title=title, link=link, summary=summary[:SUMMARY_CHARS], ts=ts, guid=guid))
    return items


def absolute_link(link: str | None, feed_url: str) -> str | None:
    """Item links resolved against the feed URL; only absolute ``http(s)`` links are kept (a
    relative ``/news/x`` or a ``tag:``/``urn:`` id would otherwise end up as a sample URL and make
    the whole feed fail validation)."""
    if not link or not link.strip():
        return None
    try:
        parts = urlsplit(urljoin(feed_url, link.strip()))
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc or any(c.isspace() for c in parts.geturl()):
        return None
    url = parts._replace(scheme=parts.scheme.lower()).geturl()
    return url if is_http_url(url) else None


def item_text(item: FeedItem) -> str:
    """Headline plus summary; this is what gets scored and mention-matched."""
    if item.summary and item.summary != item.title:
        title = item.title.rstrip()
        sep = " " if not title or title[-1] in ".!?:" else ". "
        return clean_text(title + sep + item.summary)
    return clean_text(item.title)


def collect(
    http,
    now: datetime,
    lookback_hours: float = 48,
    feeds: Sequence[tuple[str, str]] = FEEDS,
    pause_s: float = 0.0,
) -> SourceResult:
    cutoff = now - timedelta(hours=lookback_hours)
    items: list[Message] = []
    seen: set[str] = set()
    statuses: list[str] = []
    problems: Counter = Counter()
    undated = 0
    ok_feeds = 0
    for i, (name, url) in enumerate(feeds):
        if i and pause_s:
            time.sleep(pause_s)
        try:
            resp = http.get(url, headers={"Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.5"})
        except NetError as exc:
            log.info("rss %s: %s", name, exc)
            statuses.append("unavailable")
            problems["unavailable"] += 1
            continue
        if resp.status != 200:
            mapped = status_for_http(resp.status)
            log.info("rss %s: HTTP %d", name, resp.status)
            statuses.append(mapped)
            problems[mapped] += 1
            continue
        try:
            parsed = parse_feed(resp.content)
        except Exception as exc:
            log.info("rss %s: unparseable (%s)", name, exc)
            statuses.append("error")
            problems["error"] += 1
            continue
        ok_feeds += 1
        fresh = 0
        for item in parsed:
            if item.ts is None:
                undated += 1
                continue
            if item.ts < cutoff or item.ts > now + timedelta(minutes=10):
                continue
            link = absolute_link(item.link, url)
            key = "rss:" + (link or item.guid or item.title).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            fresh += 1
            items.append(
                Message(
                    source=NAME,
                    key=key,
                    text=item_text(item),
                    ts=item.ts,
                    url=link,
                    community=name,
                    public=True,
                    engagement=0.0,
                )
            )
        statuses.append("ok" if fresh else "empty")
        log.debug("rss %s: %d items, %d in window", name, len(parsed), fresh)
    detail = f"{ok_feeds}/{len(feeds)} feeds"
    if problems:
        detail += f" ({describe_counts(problems)})"
    if undated:
        detail += f"; {undated} undated items skipped"
    status = combine_statuses(statuses, has_items=bool(items))
    log.info("rss: %s, %d items, %s", status, len(items), detail)
    return SourceResult(NAME, status, items, detail)
