"""Scoring a past run against the draw that actually happened.

This is the only genuinely out-of-sample evidence the project can accumulate: a
combinations file written *before* a draw, scored against the numbers drawn afterwards.
Each record names the results directory it scored, so the prediction cannot be edited
after the fact without the hash changing.

One draw proves nothing on its own — a single ticket's hits have a standard deviation of
1.2 around 5.5 — but the records add up, and they are kept whatever they say.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Sequence

from methods.common import Pool

from .portfolio import hits_against, single_ticket_reference


def parse_numbers(text: str) -> list[int]:
    out = sorted(int(x) for x in text.replace(";", ",").replace(" ", ",").split(",") if x.strip())
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate numbers in {text!r}")
    return out


def score_predictions(combinations: dict, game_key: str, pool: Pool, drawn: Sequence[int],
                      paying: Sequence[int], source: str) -> dict:
    """Score every ticket a run produced for one game against the drawn numbers."""
    drawn = sorted(drawn)
    if len(drawn) != pool.pick or not all(1 <= n <= pool.size for n in drawn):
        raise ValueError(f"expected {pool.pick} numbers in 1..{pool.size}, got {drawn}")
    paying = set(paying)
    rows = []
    for reading in combinations.get("readings", [{"reading": 1, "seed": None,
                                                  "games": combinations["games"]}]):
        block = reading["games"][game_key]
        for kind, entries in (("method", block["methods"]), ("ensemble", block["ensembles"])):
            for name, entry in entries.items():
                picks = entry["picks"][pool.name]
                h = hits_against([tuple(picks)], drawn)[0]
                rows.append({"reading": reading["reading"], "kind": kind, "name": name,
                             "picks": picks, "hits": h, "in_paying_tier": h in paying})
    primary = [r for r in rows if r["kind"] == "ensemble" and r["name"] == "ensemble_equal"]
    ref = single_ticket_reference(pool, paying)
    hits = [r["hits"] for r in rows]
    return {
        "draw_date": combinations["draw"]["local"][:10],
        "drawn": drawn,
        "predictions_from": source,
        "game": game_key, "pool": pool.name,
        "expected_hits_per_ticket": pool.pick * pool.pick / pool.size,
        "tickets_scored": len(rows),
        "mean_hits": sum(hits) / len(hits),
        "max_hits": max(hits), "min_hits": min(hits),
        "tickets_in_paying_tier": sum(1 for r in rows if r["in_paying_tier"]),
        "p_single_ticket_paying": ref["p_paying"],
        "primary_tickets": [{"reading": r["reading"], "picks": r["picks"], "hits": r["hits"],
                             "in_paying_tier": r["in_paying_tier"]} for r in primary],
        "primary_mean_hits": (sum(r["hits"] for r in primary) / len(primary)) if primary else None,
        "all_tickets": sorted(rows, key=lambda r: (-r["hits"], r["kind"], r["name"], r["reading"])),
    }


def load_combinations(results_dir: Path) -> tuple[dict, str]:
    path = Path(results_dir) / "final" / "combinations.json"
    if not path.exists():
        raise SystemExit(f"--score-from: {path} does not exist")
    return json.loads(path.read_text(encoding="utf-8")), str(path)


def append_log(log_path: Path, record: dict) -> list[dict]:
    """Keep one record per (draw date, game); never silently drop an older one."""
    log_path = Path(log_path)
    existing = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    key = (record["draw_date"], record["game"])
    kept = [r for r in existing if (r["draw_date"], r["game"]) != key]
    kept.append(record)
    return sorted(kept, key=lambda r: (r["draw_date"], r["game"]))


def summarize_log(records: list[dict], pool: Pool) -> dict:
    """Aggregate every scored draw: this is the running out-of-sample record."""
    primary = [t for r in records for t in r["primary_tickets"]]
    hits = [t["hits"] for t in primary]
    expected = pool.pick * pool.pick / pool.size
    if not hits:
        return {"draws": len(records), "primary_tickets": 0}
    return {
        "draws": len(records), "primary_tickets": len(hits),
        "mean_hits": sum(hits) / len(hits), "expected_mean_hits": expected,
        "difference": sum(hits) / len(hits) - expected,
        "tickets_in_paying_tier": sum(1 for t in primary if t["in_paying_tier"]),
        "note": "Too few draws for any inference; kept as a running record.",
    }


def today_or(d: str | None) -> date:
    return date.fromisoformat(d) if d else date.today()
