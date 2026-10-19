"""Shared types and small helpers used by every source and by the aggregation."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

#: Source names in the order they appear in the feed's ``sources`` array.
SOURCE_NAMES = ("telegram_public", "telegram_account", "rss", "reddit")

#: Every status a source may report.
STATUSES = ("ok", "empty", "rate_limited", "unauthorized", "unavailable", "disabled", "error")

#: Failure statuses, most informative first (used to break ties when combining).
_FAILURE_PRIORITY = ("rate_limited", "unauthorized", "unavailable", "error")

#: tools/sentiment_hub/common.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]

_WS_RE = re.compile(r"\s+")


@dataclass
class Message:
    """One text collected from a source (Telegram post, news item, Reddit post)."""

    source: str
    #: Dedupe key, unique across sources (a public Telegram post read by both Telegram sources
    #: gets the same key so it is only counted once).
    key: str
    text: str
    #: Timezone-aware UTC timestamp.
    ts: datetime
    url: str | None = None
    #: Channel handle, feed name or subreddit. ``None`` for private chats.
    community: str | None = None
    #: False for texts from private Telegram chats: they only count in aggregate numbers and may
    #: never be shown (no samples, no titles, no logging).
    public: bool = True
    #: Views (Telegram) or score/upvotes (Reddit); 0 when unknown.
    engagement: float = 0.0


@dataclass
class SourceResult:
    name: str
    status: str
    items: list[Message] = field(default_factory=list)
    detail: str = ""
    #: Extra data passed between stages (never written to the feed).
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: datetime) -> datetime:
    """Naive datetimes are treated as UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_z(dt: datetime) -> str:
    """``2026-10-19T10:00:00Z`` (second precision, UTC)."""
    return as_utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        return as_utc(datetime.fromisoformat(text))
    except ValueError:
        return None


def clean_text(text: str | None) -> str:
    """Collapse all whitespace runs to one space and trim."""
    if not text:
        return ""
    return _WS_RE.sub(" ", text).strip()


def utf16_len(text: str) -> int:
    """Length as Kotlin/Java ``String.length`` counts it (emoji outside the BMP count twice)."""
    return len(text.encode("utf-16-le")) // 2


def truncate_display(text: str, limit: int = 280) -> str:
    """Whitespace collapsed and cut to ``limit`` characters, counted both as Python code points
    and as UTF-16 units, with a trailing ellipsis when cut."""
    s = clean_text(text)
    if utf16_len(s) <= limit:
        return s
    out: list[str] = []
    used = 0
    for ch in s:
        size = 2 if ord(ch) > 0xFFFF else 1
        if used + size > limit - 1:
            break
        out.append(ch)
        used += size
    return "".join(out).rstrip() + "…"


def status_for_http(code: int) -> str:
    if code == 429:
        return "rate_limited"
    if code in (401, 403):
        return "unauthorized"
    if code in (404, 410) or 500 <= code <= 599:
        return "unavailable"
    if 200 <= code <= 299:
        return "ok"
    return "error"


def combine_statuses(statuses: Iterable[str], has_items: bool) -> str:
    """Overall status of a source made of several fetches (channels, feeds, subreddits)."""
    if has_items:
        return "ok"
    values = [s for s in statuses if s]
    if not values:
        return "empty"
    if any(s in ("ok", "empty") for s in values):
        return "empty"
    failures = [s for s in values if s in _FAILURE_PRIORITY]
    if not failures:
        return "disabled" if all(s == "disabled" for s in values) else "error"
    counts = Counter(failures)
    return max(_FAILURE_PRIORITY, key=lambda s: (counts.get(s, 0), -_FAILURE_PRIORITY.index(s)))


def describe_counts(counts: Counter) -> str:
    """``Counter({'unavailable': 2, 'rate_limited': 1})`` -> ``"2 unavailable, 1 rate_limited"``."""
    return ", ".join(f"{n} {k}" for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
