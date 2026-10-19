#!/usr/bin/env python3
"""CryptoPulse sentiment hub: collect -> score -> aggregate -> publish one JSON feed.

    python tools/sentiment_hub/hub.py --out feed.json [--models vader,finbert,cryptobert]
        [--hours 24] [--assets PATH] [--channels PATH] [--lexicon PATH] [--max-channels N]
        [--no-reddit] [--no-rss] [--no-telegram] [--no-memecoins]

Logs go to stderr and the run ends with a one-line summary. Every source is optional: when all
of them fail the hub still writes a valid feed with the statuses and no assets.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script: import as the `sentiment_hub` package and keep
    _here = Path(__file__).resolve().parent  # generic module names (common, net, feed) off sys.path
    sys.path[:] = [x for x in sys.path if Path(x or ".").resolve() != _here]
    sys.path.insert(0, str(_here.parent))

import argparse
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping

from sentiment_hub import __version__
from sentiment_hub import memecoins, reddit, rss, telegram_account, telegram_public
from sentiment_hub.aggregate import aggregate, message_weight
from sentiment_hub.assets import DEFAULT_LEXICON_PATH, Asset, load_assets, load_channels
from sentiment_hub.common import SOURCE_NAMES, Message, SourceResult, parse_iso, utcnow
from sentiment_hub.feed import build_feed, validate_feed, write_feed
from sentiment_hub.lexicon import load_lexicon
from sentiment_hub.mentions import MentionMatcher, build_matchers
from sentiment_hub.net import BROWSER_USER_AGENT, HttpClient
from sentiment_hub.nlp import ALL_MODELS, parse_models, score_messages

log = logging.getLogger("sentiment_hub")

NEWS_USER_AGENT = "Mozilla/5.0 (compatible; CryptoPulseSentimentHub/1.0; +https://github.com/medonumbworks/spss)"
DEFAULT_TELEGRAM_BUDGET_S = 12 * 60
DEFAULT_ACCOUNT_BUDGET_S = 13 * 60


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hub.py",
        description="Collect free crypto chatter, score it and write the CryptoPulse sentiment feed.",
    )
    p.add_argument("--out", required=True, help="where to write feed.json")
    p.add_argument("--models", default=",".join(ALL_MODELS), help="comma list of vader,finbert,cryptobert (vader is always used)")
    p.add_argument("--hours", type=int, default=24, help="current window in hours (default 24); the previous window has the same length")
    p.add_argument("--assets", default=None, help="Bitpanda asset snapshot (default core/src/main/resources/bitpanda/bitpanda_snapshot.json)")
    p.add_argument("--channels", default=None, help="Telegram channel catalogue (default core/src/main/resources/telegram/channels.json)")
    p.add_argument("--lexicon", default=None, help="crypto VADER overlay (default core/src/main/resources/sentiment/crypto_lexicon.tsv)")
    p.add_argument("--max-channels", type=int, default=None, metavar="N", help="read only the N highest-confidence catalogue entries")
    p.add_argument("--no-reddit", action="store_true", help="skip Reddit")
    p.add_argument("--no-rss", action="store_true", help="skip the news feeds")
    p.add_argument("--no-telegram", action="store_true", help="skip both Telegram sources")
    p.add_argument("--no-memecoins", action="store_true", help="count Bitpanda assets only (no CoinGecko memecoins, no discovered cashtags)")
    p.add_argument("--telegram-budget", type=float, default=DEFAULT_TELEGRAM_BUDGET_S, metavar="SECONDS", help=argparse.SUPPRESS)
    p.add_argument("--account-budget", type=float, default=DEFAULT_ACCOUNT_BUDGET_S, metavar="SECONDS", help=argparse.SUPPRESS)
    p.add_argument("--now", default=None, help=argparse.SUPPRESS)  # ISO timestamp, for reproducible runs
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def setup_logging(verbose: bool) -> None:
    root = logging.getLogger()
    if not any(getattr(h, "_sentiment_hub", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        handler._sentiment_hub = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for noisy in ("urllib3", "telethon", "transformers", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _guard(name: str, fn: Callable[[], SourceResult]) -> SourceResult:
    started = time.monotonic()
    try:
        result = fn()
    except Exception as exc:  # one failing source never fails the run
        log.exception("%s crashed", name)
        result = SourceResult(name, "error", detail=f"{type(exc).__name__}")
    log.debug("%s finished in %.1fs", name, time.monotonic() - started)
    return result


def collect_sources(
    args: argparse.Namespace,
    entries: list,
    now: datetime,
    lookback_hours: float,
    tg_http: Any = None,
    news_http: Any = None,
    account_env: Mapping[str, str] | None = None,
    account_client_factory: Callable | None = None,
) -> dict[str, SourceResult]:
    tg_http = tg_http or HttpClient(user_agent=BROWSER_USER_AGENT, timeout=20, host_intervals={"t.me": telegram_public.REQUEST_INTERVAL_S})
    news_http = news_http or HttpClient(user_agent=NEWS_USER_AGENT, timeout=20)
    results: dict[str, SourceResult] = {}
    futures: dict[str, Future] = {}
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="source") as pool:
        if args.no_telegram:
            results["telegram_public"] = SourceResult("telegram_public", "disabled", detail="disabled by --no-telegram")
            results["telegram_account"] = SourceResult("telegram_account", "disabled", detail="disabled by --no-telegram")
        else:
            futures["telegram_public"] = pool.submit(
                _guard,
                "telegram_public",
                lambda: telegram_public.collect(entries, tg_http, now, lookback_hours, budget_s=args.telegram_budget),
            )

            def covered() -> set[str]:
                return set(futures["telegram_public"].result().meta.get("covered_handles", set()))

            futures["telegram_account"] = pool.submit(
                _guard,
                "telegram_account",
                lambda: telegram_account.collect(
                    entries,
                    now,
                    lookback_hours,
                    env=account_env,
                    client_factory=account_client_factory,
                    skip_handles=covered,
                    budget_s=args.account_budget,
                ),
            )
        if args.no_rss:
            results["rss"] = SourceResult("rss", "disabled", detail="disabled by --no-rss")
        else:
            futures["rss"] = pool.submit(_guard, "rss", lambda: rss.collect(news_http, now, lookback_hours))
        if args.no_reddit:
            results["reddit"] = SourceResult("reddit", "disabled", detail="disabled by --no-reddit")
        else:
            futures["reddit"] = pool.submit(_guard, "reddit", lambda: reddit.collect(news_http, now, lookback_hours))
        for name, fut in futures.items():
            results[name] = fut.result()
    return results


def merge_messages(results: Mapping[str, SourceResult], now: datetime, lookback_hours: float) -> tuple[list[Message], int]:
    """Messages in the lookback window, deduplicated by key (a public post read by both Telegram
    sources counts once) and by identical text within the same community (spam/bot repeats)."""
    cutoff = now - timedelta(hours=lookback_hours)
    future_limit = now + timedelta(minutes=10)
    out: list[Message] = []
    keys: set[str] = set()
    texts: set[tuple[str, str]] = set()
    dropped = 0
    for name in SOURCE_NAMES:
        result = results.get(name)
        if result is None:
            continue
        for m in result.items:
            if not m.text or not m.text.strip() or m.ts < cutoff or m.ts > future_limit:
                continue
            community = m.community.lower() if m.community else "private:" + m.key.split("/")[0]
            text_key = (community, m.text.strip().lower()[:200])
            if m.key in keys or text_key in texts:
                dropped += 1
                continue
            keys.add(m.key)
            texts.add(text_key)
            out.append(m)
    return out, dropped


def split_windows(messages: list[Message], now: datetime, hours: float) -> tuple[list[Message], list[Message]]:
    start_current = now - timedelta(hours=hours)
    start_previous = now - timedelta(hours=2 * hours)
    current = [m for m in messages if m.ts > start_current]
    previous = [m for m in messages if start_previous < m.ts <= start_current]
    return current, previous


def summary_line(out: str, feed: dict, results: Mapping[str, SourceResult], elapsed: float, notes: list[str]) -> str:
    parts = [
        f"sentiment-hub {__version__}: wrote {out} in {elapsed:.0f}s",
        f"models {','.join(feed['models']) or 'none'}",
        f"{feed['market']['mentions']} messages scored, {len(feed['assets'])} assets",
    ]
    for s in feed["sources"]:
        parts.append(f"{s['name']} {s['status']} {s['items']}" + (f" ({s['detail']})" if s["detail"] else ""))
    if notes:
        parts.append("notes: " + "; ".join(notes))
    return " | ".join(parts)


def _step_summary(feed: dict, notes: list[str]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    lines = [
        "### Sentiment hub",
        "",
        f"Generated {feed['generatedAt']}, models: {', '.join(feed['models']) or 'none'}, "
        f"{feed['market']['mentions']} messages, {len(feed['assets'])} assets, market composite {feed['market']['composite']}",
        "",
        "| source | status | items | detail |",
        "|---|---|---|---|",
    ]
    lines += [f"| {s['name']} | {s['status']} | {s['items']} | {s['detail'].replace('|', '/')} |" for s in feed["sources"]]
    if feed["assets"]:
        lines += ["", "| asset | mentions | prev | composite |", "|---|---|---|---|"]
        lines += [f"| {a['symbol']} | {a['mentions']} | {a['mentionsPrev']} | {a['composite']} |" for a in feed["assets"][:15]]
    if notes:
        lines += ["", "Notes: " + "; ".join(notes)]
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError:
        pass


def run(
    args: argparse.Namespace,
    now: datetime | None = None,
    tg_http: Any = None,
    news_http: Any = None,
    account_env: Mapping[str, str] | None = None,
    account_client_factory: Callable | None = None,
    pipeline_factory: Callable | None = None,
) -> tuple[dict, dict[str, SourceResult], list[str]]:
    now = now or (parse_iso(args.now) if args.now else None) or utcnow()
    now = now.replace(microsecond=0)
    hours = args.hours
    lookback = 2 * hours
    models_requested = parse_models(args.models)

    assets, assets_note = load_assets(args.assets)
    news_http = news_http or HttpClient(user_agent=NEWS_USER_AGENT, timeout=20)
    meme_note = "off"
    if not getattr(args, "no_memecoins", False):
        memes, meme_note = memecoins.fetch_memecoins(news_http)
        before = len(assets)
        assets = memecoins.merge_assets(assets, memes)
        meme_note = f"{meme_note}, {len(assets) - before} not on Bitpanda"
    entries, channels_note = load_channels(args.channels)
    if args.max_channels is not None:
        entries = entries[: max(0, args.max_channels)]
    lexicon, lexicon_note = load_lexicon(args.lexicon or DEFAULT_LEXICON_PATH)
    log.info(
        "start: %d assets (%s; memecoins: %s), %d Telegram entries (%s), lexicon %s, models %s, window %dh",
        len(assets), assets_note, meme_note, len(entries), channels_note, lexicon_note, ",".join(models_requested), hours,
    )

    results = collect_sources(args, entries, now, lookback, tg_http, news_http, account_env, account_client_factory)
    messages, dropped = merge_messages(results, now, lookback)
    current, previous = split_windows(messages, now, hours)
    log.info("collected %d unique messages (%d duplicates dropped): %d current, %d previous window", len(messages), dropped, len(current), len(previous))

    # Most important first: the transformer models have a per-run text and time limit.
    current.sort(key=lambda m: -message_weight(m.ts, m.engagement, now))
    report = score_messages(current, models_requested, lexicon, pipeline_factory=pipeline_factory)
    notes = list(report.notes)
    if not report.models and current:
        raise RuntimeError("no scorer could run (is vaderSentiment installed?)")
    if assets_note == "fallback":
        notes.append("asset snapshot missing, built-in assets used")
    if channels_note == "fallback":
        notes.append("channel catalogue missing, built-in channels used")
    if lexicon_note != "loaded":
        notes.append("crypto lexicon missing, plain VADER")

    matchers = build_matchers(assets)
    if not getattr(args, "no_memecoins", False):
        # brand-new tickers that several communities talk about: tracked by "$TICKER" only
        fresh = memecoins.discover_cashtags(current, {a.symbol for a in assets})
        matchers += [MentionMatcher(Asset(symbol=s, name=s), cashtag_only=True) for s in fresh]
        if fresh:
            log.info("tracking %d new cashtags: %s", len(fresh), ", ".join(fresh[:15]))
    agg = aggregate(report.scored, previous, matchers, now, report.models)
    feed = build_feed(now, hours, report.models, [results[n] for n in SOURCE_NAMES if n in results], agg.market, agg.assets)
    return feed, results, notes


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        parse_models(args.models)
    except ValueError as exc:
        parser.error(str(exc))
    if args.hours <= 0:
        parser.error("--hours must be positive")
    setup_logging(args.verbose)
    started = time.monotonic()
    try:
        feed, results, notes = run(args)
    except Exception:
        log.exception("sentiment hub failed")
        return 1
    errors = validate_feed(feed)
    if errors:
        for e in errors[:50]:
            log.error("feed invalid: %s", e)
        log.error("not writing an invalid feed (%d problems)", len(errors))
        return 1
    write_feed(feed, args.out)
    _step_summary(feed, notes)
    print(summary_line(args.out, feed, results, time.monotonic() - started, notes), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
