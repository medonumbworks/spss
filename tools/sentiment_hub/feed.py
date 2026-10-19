"""The published JSON feed (``cryptopulse.sentiment-feed/v1``): builder, validator and writer.

The Kotlin client depends on this exact shape; ``validate_feed`` is the executable contract.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from sentiment_hub.common import SOURCE_NAMES, STATUSES, SourceResult, is_http_url, iso_z, utf16_len
from sentiment_hub.nlp import ALL_MODELS

SCHEMA = "cryptopulse.sentiment-feed/v1"
PUBLIC_URL = "https://raw.githubusercontent.com/medonumbworks/spss/sentiment-feed/feed.json"
DECIMALS = 4
MAX_SAMPLES = 5
MAX_SAMPLE_CHARS = 280

TOP_KEYS = ("schema", "generatedAt", "windowHours", "models", "sources", "market", "assets")
SOURCE_KEYS = ("name", "status", "items", "detail")
MARKET_KEYS = ("mentions", "composite", "bullish", "bearish", "neutral")
ASSET_KEYS = ("symbol", "mentions", "mentionsPrev", "composite", "scores", "bullish", "bearish", "neutral", "bySource", "samples")
BY_SOURCE_KEYS = ("mentions", "composite")
SAMPLE_KEYS = ("text", "score", "source", "community", "url", "ts")
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SYMBOL_RE = re.compile(r"[A-Z0-9]+")  # asset symbols: ASCII uppercase letters and digits (fullmatch)


def is_feed_symbol(symbol: Any) -> bool:
    """True when ``symbol`` may appear in ``assets[].symbol`` ("币安人生" or "ZERØ" may not)."""
    return isinstance(symbol, str) and SYMBOL_RE.fullmatch(symbol) is not None


def rnd(value: float) -> float:
    """Rounded to 4 decimals; never ``-0.0``, never NaN."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return 0.0
    out = round(float(value), DECIMALS)
    return 0.0 if out == 0 else out


def _source_entry(result: SourceResult) -> dict:
    return {"name": result.name, "status": result.status, "items": len(result.items), "detail": result.detail or ""}


def build_feed(
    generated_at: datetime,
    window_hours: int,
    models: Sequence[str],
    sources: Sequence[SourceResult],
    market: dict,
    assets: Sequence[dict],
) -> dict:
    by_name = {s.name: s for s in sources}
    ordered_sources = [
        _source_entry(by_name[name]) if name in by_name else {"name": name, "status": "disabled", "items": 0, "detail": "not run"}
        for name in SOURCE_NAMES
    ]
    out_assets = []
    for a in assets:
        if a["mentions"] <= 0:
            continue
        out_assets.append(
            {
                "symbol": a["symbol"],
                "mentions": int(a["mentions"]),
                "mentionsPrev": int(a["mentionsPrev"]),
                "composite": rnd(a["composite"]),
                "scores": {m: rnd(a["scores"][m]) for m in ALL_MODELS if m in a["scores"]},
                "bullish": rnd(a["bullish"]),
                "bearish": rnd(a["bearish"]),
                "neutral": rnd(a["neutral"]),
                "bySource": {
                    src: {"mentions": int(v["mentions"]), "composite": rnd(v["composite"])}
                    for src, v in sorted(a["bySource"].items(), key=lambda kv: SOURCE_NAMES.index(kv[0]) if kv[0] in SOURCE_NAMES else 99)
                },
                "samples": [
                    {
                        "text": s["text"],
                        "score": rnd(s["score"]),
                        "source": s["source"],
                        "community": s["community"],
                        "url": s["url"],
                        "ts": s["ts"],
                    }
                    for s in a["samples"][:MAX_SAMPLES]
                ],
            }
        )
    out_assets.sort(key=lambda x: -x["mentions"])  # stable: keeps the aggregator's tie order
    return {
        "schema": SCHEMA,
        "generatedAt": iso_z(generated_at),
        "windowHours": int(window_hours),
        "models": [m for m in ALL_MODELS if m in models],
        "sources": ordered_sources,
        "market": {
            "mentions": int(market["mentions"]),
            "composite": rnd(market["composite"]),
            "bullish": rnd(market["bullish"]),
            "bearish": rnd(market["bearish"]),
            "neutral": rnd(market["neutral"]),
        },
        "assets": out_assets,
    }


# -- validation -------------------------------------------------------------------------------


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)) and math.isfinite(v)


def _decimals_ok(v: float) -> bool:
    return abs(round(v, DECIMALS) - v) < 1e-12


def _check_keys(obj: Any, keys: Sequence[str], where: str, errors: list[str]) -> bool:
    if not isinstance(obj, dict):
        errors.append(f"{where}: expected object")
        return False
    missing = [k for k in keys if k not in obj]
    extra = [k for k in obj if k not in keys]
    if missing:
        errors.append(f"{where}: missing {missing}")
    if extra:
        errors.append(f"{where}: unexpected {extra}")
    if not missing and not extra and list(obj) != list(keys):
        errors.append(f"{where}: keys out of order, expected {list(keys)}")
    return not missing


def _check_score(v: Any, where: str, errors: list[str], lo: float = -1.0, hi: float = 1.0) -> None:
    if not _is_num(v):
        errors.append(f"{where}: expected number, got {v!r}")
    elif not (lo <= v <= hi):
        errors.append(f"{where}: {v} outside [{lo}, {hi}]")
    elif not _decimals_ok(float(v)):
        errors.append(f"{where}: {v} has more than {DECIMALS} decimals")


def _check_shares(obj: dict, where: str, errors: list[str], has_mentions: bool) -> None:
    for k in ("bullish", "bearish", "neutral"):
        _check_score(obj.get(k), f"{where}.{k}", errors, 0.0, 1.0)
    try:
        total = obj["bullish"] + obj["bearish"] + obj["neutral"]
    except (KeyError, TypeError):
        return
    if has_mentions and abs(total - 1.0) > 0.001:
        errors.append(f"{where}: shares sum to {total}, expected 1")
    if not has_mentions and total != 0:
        errors.append(f"{where}: shares must be 0 without mentions")


def validate_feed(feed: Any) -> list[str]:
    """Returns a list of problems; empty means the feed matches the v1 contract."""
    errors: list[str] = []
    if not _check_keys(feed, TOP_KEYS, "feed", errors):
        return errors
    if feed["schema"] != SCHEMA:
        errors.append(f"schema: expected {SCHEMA!r}")
    if not isinstance(feed["generatedAt"], str) or not _TS_RE.match(feed["generatedAt"]):
        errors.append("generatedAt: expected YYYY-MM-DDTHH:MM:SSZ")
    if not _is_int(feed["windowHours"]) or feed["windowHours"] <= 0:
        errors.append("windowHours: expected positive integer")

    models = feed["models"]
    if not isinstance(models, list) or any(m not in ALL_MODELS for m in models) or len(set(models)) != len(models):
        errors.append(f"models: expected unique subset of {list(ALL_MODELS)}")
        models = [m for m in models if m in ALL_MODELS] if isinstance(models, list) else []

    sources = feed["sources"]
    if not isinstance(sources, list):
        errors.append("sources: expected array")
        sources = []
    names = []
    for i, s in enumerate(sources):
        where = f"sources[{i}]"
        if not _check_keys(s, SOURCE_KEYS, where, errors):
            continue
        names.append(s["name"])
        if s["name"] not in SOURCE_NAMES:
            errors.append(f"{where}.name: unknown {s['name']!r}")
        if s["status"] not in STATUSES:
            errors.append(f"{where}.status: unknown {s['status']!r}")
        if not _is_int(s["items"]) or s["items"] < 0:
            errors.append(f"{where}.items: expected non-negative integer")
        if not isinstance(s["detail"], str):
            errors.append(f"{where}.detail: expected string")
    if names != list(SOURCE_NAMES):
        errors.append(f"sources: expected names {list(SOURCE_NAMES)} in order, got {names}")

    market = feed["market"]
    if _check_keys(market, MARKET_KEYS, "market", errors):
        if not _is_int(market["mentions"]) or market["mentions"] < 0:
            errors.append("market.mentions: expected non-negative integer")
        _check_score(market["composite"], "market.composite", errors)
        _check_shares(market, "market", errors, has_mentions=_is_int(market["mentions"]) and market["mentions"] > 0)

    assets = feed["assets"]
    if not isinstance(assets, list):
        errors.append("assets: expected array")
        return errors
    previous = None
    seen_symbols: set[str] = set()
    for i, a in enumerate(assets):
        where = f"assets[{i}]"
        if not _check_keys(a, ASSET_KEYS, where, errors):
            continue
        sym = a["symbol"]
        if not is_feed_symbol(sym):
            errors.append(f"{where}.symbol: expected uppercase letters/digits")
        elif sym in seen_symbols:
            errors.append(f"{where}.symbol: duplicate {sym}")
        else:
            seen_symbols.add(sym)
        if not _is_int(a["mentions"]) or a["mentions"] <= 0:
            errors.append(f"{where}.mentions: expected positive integer")
            continue
        if previous is not None and a["mentions"] > previous:
            errors.append(f"{where}: assets not sorted by mentions desc")
        previous = a["mentions"]
        if _is_int(market.get("mentions") if isinstance(market, dict) else None) and a["mentions"] > market["mentions"]:
            errors.append(f"{where}.mentions: more than market.mentions")
        if not _is_int(a["mentionsPrev"]) or a["mentionsPrev"] < 0:
            errors.append(f"{where}.mentionsPrev: expected non-negative integer")
        _check_score(a["composite"], f"{where}.composite", errors)
        _check_shares(a, where, errors, has_mentions=True)
        scores = a["scores"]
        if not isinstance(scores, dict) or any(k not in models for k in scores):
            errors.append(f"{where}.scores: keys must be a subset of models {models}")
        else:
            for k, v in scores.items():
                _check_score(v, f"{where}.scores.{k}", errors)
        by_source = a["bySource"]
        if not isinstance(by_source, dict) or not by_source:
            errors.append(f"{where}.bySource: expected non-empty object")
        else:
            total = 0
            for src, v in by_source.items():
                w = f"{where}.bySource.{src}"
                if src not in SOURCE_NAMES:
                    errors.append(f"{w}: unknown source")
                if not _check_keys(v, BY_SOURCE_KEYS, w, errors):
                    continue
                if not _is_int(v["mentions"]) or v["mentions"] <= 0:
                    errors.append(f"{w}.mentions: expected positive integer")
                else:
                    total += v["mentions"]
                _check_score(v["composite"], f"{w}.composite", errors)
            if total != a["mentions"]:
                errors.append(f"{where}.bySource: mentions sum {total} != {a['mentions']}")
        samples = a["samples"]
        if not isinstance(samples, list) or len(samples) > MAX_SAMPLES:
            errors.append(f"{where}.samples: expected at most {MAX_SAMPLES}")
            continue
        for j, s in enumerate(samples):
            w = f"{where}.samples[{j}]"
            if not _check_keys(s, SAMPLE_KEYS, w, errors):
                continue
            if not isinstance(s["text"], str) or not s["text"] or utf16_len(s["text"]) > MAX_SAMPLE_CHARS:
                errors.append(f"{w}.text: expected 1..{MAX_SAMPLE_CHARS} chars")
            _check_score(s["score"], f"{w}.score", errors)
            if s["source"] not in SOURCE_NAMES:
                errors.append(f"{w}.source: unknown {s['source']!r}")
            for k in ("community", "url"):
                if not isinstance(s[k], str) or not s[k]:
                    errors.append(f"{w}.{k}: expected non-empty string")
            if isinstance(s["url"], str) and s["url"] and not is_http_url(s["url"]):
                errors.append(f"{w}.url: expected http(s) URL")
            if not isinstance(s["ts"], str) or not _TS_RE.match(s["ts"]):
                errors.append(f"{w}.ts: expected YYYY-MM-DDTHH:MM:SSZ")
    return errors


def dumps(feed: dict) -> str:
    return json.dumps(feed, ensure_ascii=False, indent=1, allow_nan=False) + "\n"


def write_feed(feed: dict, path: str | Path) -> Path:
    """Atomic write (temp file + rename) so a crash never leaves half a feed behind."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".feed-", suffix=".json", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(dumps(feed))
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target
