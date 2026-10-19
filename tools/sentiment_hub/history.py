"""Rolling per-asset mention counts, one entry per hub run (``cryptopulse.mention-history/v1``).

The feed's ``baseline7d``/``spike`` come from here: every run appends its per-asset mention
counts, and the mean over the entries older than 24 h is the baseline the current count is
compared against. Each count is already a rolling 24 h window, so the entries are comparable,
and entries younger than 24 h are excluded because they overlap the current window. The file is
published next to ``feed.json`` on the ``sentiment-feed`` branch and fetched back before the
next run.

    {"schema": "cryptopulse.mention-history/v1",
     "entries": [{"ts": "2026-10-19T10:00:00Z", "counts": {"BTC": 251, ...}}, ...]}
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

from sentiment_hub.common import as_utc, iso_z, parse_iso

SCHEMA = "cryptopulse.mention-history/v1"
MAX_AGE = timedelta(days=7)
MAX_ENTRIES = 200  # a 7-day window of hourly runs plus slack; keeps the file small
BASELINE_MIN_AGE = timedelta(hours=24)  # younger counts overlap the current 24 h window
MIN_BASELINE_ENTRIES = 6  # fewer old runs than this is too little history for a mean


def empty_history() -> dict:
    return {"schema": SCHEMA, "entries": []}


def _clean_entries(raw: object) -> list[dict]:
    """Entries with a parseable ``ts`` and sane ``counts``; everything else is dropped so one
    corrupted entry never loses the whole history."""
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        ts = parse_iso(entry.get("ts")) if isinstance(entry.get("ts"), str) else None
        counts = entry.get("counts")
        if ts is None or not isinstance(counts, dict):
            continue
        clean = {
            sym: c
            for sym, c in counts.items()
            if isinstance(sym, str) and isinstance(c, int) and not isinstance(c, bool) and c >= 0
        }
        out.append({"ts": iso_z(ts), "counts": clean})
    out.sort(key=lambda e: e["ts"])  # iso_z is fixed-width, so string order is time order
    return out


def load_history(path: str | Path | None) -> dict:
    """The stored history, or an empty one when ``path`` is None or the file is missing,
    unreadable or corrupt - losing one baseline is better than failing the whole run."""
    if not path:
        return empty_history()
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty_history()
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        return empty_history()
    return {"schema": SCHEMA, "entries": _clean_entries(raw.get("entries"))}


def append_run(history: dict, generated_at: datetime, counts: Mapping[str, int]) -> dict:
    """Record one run's counts in place: drop entries older than ``MAX_AGE`` (relative to
    ``generated_at``) and keep at most the newest ``MAX_ENTRIES``."""
    entries = list(history.get("entries") or [])
    entries.append({"ts": iso_z(generated_at), "counts": {str(s): int(c) for s, c in counts.items()}})
    cutoff = iso_z(as_utc(generated_at) - MAX_AGE)
    entries = sorted((e for e in entries if e["ts"] >= cutoff), key=lambda e: e["ts"])
    history["schema"] = SCHEMA
    history["entries"] = entries[-MAX_ENTRIES:]
    return history


def baselines(history: dict, now: datetime) -> dict[str, float]:
    """Per-symbol mean mention count over the entries older than 24 h; a symbol absent from an
    old entry counts as 0 there. Empty when fewer than ``MIN_BASELINE_ENTRIES`` entries qualify:
    too little history to call anything a spike."""
    cutoff = iso_z(as_utc(now) - BASELINE_MIN_AGE)
    old = [e["counts"] for e in history.get("entries") or [] if e["ts"] < cutoff]
    if len(old) < MIN_BASELINE_ENTRIES:
        return {}
    symbols = set().union(*old)
    return {sym: sum(c.get(sym, 0) for c in old) / len(old) for sym in symbols}


def dumps_history(history: dict) -> str:
    # compact on purpose: at 200 entries the indent would multiply the file size
    return json.dumps(history, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def save_history(history: dict, path: str | Path) -> Path:
    """Atomic write (temp file + rename), like ``feed.write_feed``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".history-", suffix=".json", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(dumps_history(history))
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target
