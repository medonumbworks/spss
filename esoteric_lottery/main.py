#!/usr/bin/env python3
"""Esoteric lottery research pipeline.

    python main.py                      # full run for the draw in config.yaml
    python main.py --skip-backtest      # final combinations only
    python main.py --draw-date 2026-10-24 --readings 4 --backtest-from results \
                   --results results/draw_2026-10-24

Steps: (1) every method for the target draw and all three games, once per *reading*
(reading k re-seeds the simulated casts and the SHA-256 tie-break with ``seed + k - 1``;
reading 1 is the standard run; the raw numbers of the astronomical and numerological
methods do not depend on the seed, only the hash fill of numbers they left open), with
intermediate results in results/final/intermediate/; (2) chronological backtest on every
supplied history file, or reuse of a previous run's backtest, which does not depend on
the target draw; (3) ensembles per reading; (4) results/final/combinations.json,
results/manifest.json and results/REPORT.md. Everything except the timestamp in
manifest.json is reproducible byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation import report  # noqa: E402
from evaluation.backtest import run_backtest  # noqa: E402
from evaluation.data import crosscheck_xlsx, load_draws, membership_matrix, split_index  # noqa: E402
from evaluation.quality import data_quality  # noqa: E402
from evaluation.stats import hit_pmf  # noqa: E402
from methods import build_methods  # noqa: E402
from methods.common import (Component, Event, Game, MethodOutput, Place, Subject, games_from_config,  # noqa: E402
                            local_datetime, to_jsonable, write_json)
from methods.ensemble import aggregate, equal_weights  # noqa: E402

PACKAGES = ["pyswisseph", "numpy", "scipy", "pandas", "scikit-learn", "PyYAML", "openpyxl"]


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def place_of(spec: dict) -> Place:
    return Place(spec["name"], float(spec["lat"]), float(spec["lon"]), spec["tz"])


def subject_of(cfg: dict) -> Subject:
    s = cfg["subject"]
    place = place_of(s["birth_place"])
    return Subject(local_datetime(s["birth_date"], s["birth_time"], place.tz), place, s.get("name"))


def event_of(cfg: dict, draw_date: str | None = None, draw_time: str | None = None) -> Event:
    d = cfg["draw"]
    place = place_of(d["place"])
    return Event(local_datetime(draw_date or d["date"], draw_time or d["time"], place.tz), place)


def theoretical(games: dict) -> dict:
    out = {}
    for key, game in games.items():
        pools = {}
        top = 1.0
        for p in game.pools:
            pmf = hit_pmf(p)
            pools[p.name] = {"pmf": {str(k): float(v) for k, v in enumerate(pmf)},
                             "expected_hits": float(sum(k * v for k, v in enumerate(pmf)))}
            top *= pmf[-1]
        out[key] = {"pools": pools, "p_all_numbers_right": float(top), "one_in": round(1 / top)}
    sin = games.get("sve_ili_nista")
    if sin:
        out["sve_ili_nista"]["p_zero_hits"] = float(hit_pmf(sin.pools[0])[0])
        out["sve_ili_nista"]["combinations"] = math.comb(sin.pools[0].size, sin.pools[0].pick)
    return out


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_digest() -> str:
    h = hashlib.sha256()
    for p in sorted([ROOT / "main.py", *ROOT.glob("methods/*.py"), *ROOT.glob("evaluation/*.py")]):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def history_hashes(cfg: dict) -> dict[str, str]:
    out = {}
    for hcfg in (cfg.get("history") or {}).values():
        if not hcfg:
            continue
        out[hcfg["csv"]] = file_sha256(ROOT / hcfg["csv"])
        x = hcfg.get("crosscheck_xlsx")
        if x:
            out[x["path"]] = file_sha256(ROOT / x["path"])
    return out


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def run_final(methods, subject, event, games, out_dir: Path) -> dict[str, MethodOutput]:
    outputs = {}
    for m in methods:
        try:
            o = m.run(subject, event, games.values())
        except Exception as exc:
            o = MethodOutput(m.key, m.label, m.kind, "failed", [Component("method", "failed", repr(exc))],
                             list(m.sources), {}, [], [], error=repr(exc))
        outputs[m.key] = o
        write_json(out_dir / f"{m.key}.json", o.to_json())
    return outputs


def reuse_backtest(from_dir: Path, game_key: str, config_sha: str, data_hashes: dict) -> dict | None:
    """A previous run's backtest summary, after checking that config and data are unchanged."""
    summary_path = from_dir / "backtest" / game_key / "summary.json"
    if not summary_path.exists():
        return None
    manifest_path = from_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"--backtest-from: {manifest_path} is missing, the previous run cannot be verified")
    prev = json.loads(manifest_path.read_text(encoding="utf-8"))
    if prev.get("config_sha256") != config_sha:
        raise SystemExit(f"--backtest-from: config.yaml changed since {from_dir} was produced; rerun the backtest")
    if prev.get("history_files_sha256") not in (None, data_hashes):
        raise SystemExit(f"--backtest-from: history files changed since {from_dir} was produced; rerun the backtest")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "reused"
    summary["reused_from"] = {"summary": str(summary_path), "generated_at": prev.get("generated_at"),
                              "code_sha256": prev.get("code_sha256")}
    return summary


def game_block(game_key: str, game: Game, outputs: dict[str, MethodOutput], ensembles: dict) -> dict:
    g = {"label": game.label, "methods": {}, "ensembles": {}}
    for k, o in outputs.items():
        if o.status != "failed":
            g["methods"][k] = {"label": o.label, "status": o.status, "picks": o.picks[game_key],
                               "derived_from_method": o.derived[game_key]}
    for k, e in ensembles.items():
        if game_key in e["picks"]:
            g["ensembles"][k] = {"picks": e["picks"][game_key]}
    return g


def coverage(tickets: list[dict], game: Game) -> dict:
    """How the readings' tickets relate: distinct tickets, numbers covered, pairwise overlap."""
    out = {}
    for pool in game.pools:
        if pool.kind != "set":
            continue
        sets = [set(t["picks"][pool.name]) for t in tickets]
        union = set().union(*sets)
        out[pool.name] = {
            "distinct_tickets": len({tuple(sorted(s)) for s in sets}),
            "numbers_covered": len(union),
            "numbers_never_played": sorted(set(range(1, pool.size + 1)) - union),
            "pairwise_common": [[len(a & b) for b in sets] for a in sets],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--draw-date", help="override draw.date from config.yaml (YYYY-MM-DD)")
    ap.add_argument("--draw-time", help="override draw.time from config.yaml (HH:MM, local civil time)")
    ap.add_argument("--readings", type=int, default=1,
                    help="number of readings; reading k re-seeds the simulated casts and the tie-break with seed + k - 1")
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--backtest-from", help="results directory of a previous run whose backtest is reused")
    args = ap.parse_args(argv)
    if args.readings < 1:
        ap.error("--readings must be at least 1")

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    out = Path(args.results)
    subject, games = subject_of(cfg), games_from_config(cfg)
    event = event_of(cfg, args.draw_date, args.draw_time)
    seed = int(cfg["seed"])
    config_sha, data_hashes = file_sha256(cfg_path), history_hashes(cfg)

    print(f"[1/4] methods for the draw {event.local.isoformat()} ({event.place.name}), {args.readings} reading(s)")
    readings = []
    for k in range(1, args.readings + 1):
        seed_k = seed + (k - 1)
        idir = out / "final" / "intermediate" / ("" if k == 1 else f"readings/{k}")
        outputs_k = run_final(build_methods(cfg, seed=seed_k), subject, event, games, idir)
        readings.append({"reading": k, "seed": seed_k, "outputs": outputs_k, "dir": idir})
        if k == 1:
            for o in outputs_k.values():
                print(f"      {o.key:<18} {o.status}")
        else:
            print(f"      reading {k} (seed {seed_k}): " + ", ".join(f"{o.key}={o.status}" for o in outputs_k.values()
                                                                   if o.status != "ran") or "all ran")
    outputs = readings[0]["outputs"]
    methods = build_methods(cfg)

    print("[2/4] backtests")
    backtests: dict = {}
    for game_key, game in games.items():
        hcfg = (cfg.get("history") or {}).get(game_key)
        if args.skip_backtest:
            backtests[game_key] = {"status": "not_run", "reason": "--skip-backtest"}
            continue
        if not hcfg:
            backtests[game_key] = {"status": "not_run", "reason": "no historical draw file supplied"}
            print(f"      {game_key}: not run (no historical draw file supplied)")
            continue
        reused = reuse_backtest(Path(args.backtest_from), game_key, config_sha, data_hashes) if args.backtest_from else None
        if reused:
            backtests[game_key] = reused
            print(f"      {game_key}: reused {reused['reused_from']['summary']}")
            continue
        df, validation = load_draws(ROOT / hcfg["csv"], hcfg, game)
        if hcfg.get("crosscheck_xlsx"):
            x = hcfg["crosscheck_xlsx"]
            validation["xlsx_crosscheck"] = crosscheck_xlsx(df, ROOT / x["path"], x, game.pools[0])
        split = split_index(df, date.fromisoformat(cfg["evaluation"]["evaluation_start"]))
        print(f"      {game_key}: {len(df)} draws, {split} training / {len(df) - split} evaluation")
        bt = run_backtest(df, game, subject, methods, event.place, hcfg["draw_time"], split, cfg)
        quality = data_quality(membership_matrix(df, game.pools[0]), game.pools[0],
                               int(cfg["evaluation"]["quality_replicates"]), seed)
        bdir = out / "backtest" / game_key
        for s, rows in bt["predictions"].items():
            write_json(bdir / "predictions" / f"{s}.json", rows, one_row_per_line=True)
        summary = {k: v for k, v in bt.items() if k != "predictions"}
        summary.update({"status": "ran", "validation": validation, "data_quality": quality,
                        "protocol": cfg["evaluation"]})
        write_json(bdir / "summary.json", summary)
        backtests[game_key] = summary

    print("[3/4] ensembles")
    ens_key = f"ensemble|{event.local.isoformat()}"
    for r in readings:
        suffix = "" if r["reading"] == 1 else f"|reading{r['reading']}"
        outs = r["outputs"].values()
        ensembles = {"ensemble_equal": aggregate(outs, equal_weights(outs), games.values(), f"{ens_key}|equal{suffix}")}
        for game_key, bt in backtests.items():
            if bt.get("status") in ("ran", "reused"):
                w = {k: v["weight"] for k, v in bt["train_weights"].items()}
                ensembles[f"ensemble_train_weighted_{game_key}"] = aggregate(
                    outs, w, [games[game_key]], f"{ens_key}|train_weighted{suffix}")
        r["ensembles"] = ensembles
        r["games"] = {gk: game_block(gk, g, r["outputs"], ensembles) for gk, g in games.items()}
        write_json(r["dir"] / "ensemble.json", ensembles)

    combinations = {
        "draw": {"local": event.local.isoformat(), "utc": event.utc.isoformat(), "place": event.place.name},
        "n_readings": args.readings,
        "games": readings[0]["games"],
        "readings": [{"reading": r["reading"], "seed": r["seed"], "games": r["games"]} for r in readings],
        "tickets": {}, "coverage": {},
    }
    for game_key, game in games.items():
        tickets = [{"reading": r["reading"], "seed": r["seed"], "strategy": "ensemble_equal",
                    "picks": r["ensembles"]["ensemble_equal"]["picks"][game_key]} for r in readings]
        combinations["tickets"][game_key] = tickets
        combinations["coverage"][game_key] = coverage(tickets, game)
    write_json(out / "final" / "combinations.json", combinations)

    print("[4/4] manifest and report")
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "packages": versions,
        "config_sha256": config_sha, "code_sha256": code_digest(), "history_files_sha256": data_hashes,
        "git_commit_at_run": git_commit(),
        "command_line": {"draw_date": args.draw_date, "draw_time": args.draw_time, "readings": args.readings,
                         "skip_backtest": args.skip_backtest, "backtest_from": args.backtest_from},
        "inputs": {"birth_local": subject.birth_local.isoformat(), "birth_utc": subject.birth_utc.isoformat(),
                   "birth_place": subject.place.name, "draw_local": event.local.isoformat(),
                   "draw_utc": event.utc.isoformat(), "draw_place": event.place.name},
        "methods": {k: {"status": o.status, "kind": o.kind, "error": o.error,
                        "components": [to_jsonable(c) for c in o.components]} for k, o in outputs.items()},
        "readings": [{"reading": r["reading"], "seed": r["seed"],
                      "statuses": {k: o.status for k, o in r["outputs"].items()}} for r in readings],
        "backtests": {k: {"status": v.get("status"), "reason": v.get("reason"),
                          "failed_methods": list(v.get("failed_methods", {})), "reused_from": v.get("reused_from")}
                      for k, v in backtests.items()},
    }
    write_json(out / "manifest.json", manifest)
    theory = theoretical(games)
    write_json(out / "theoretical_probabilities.json", theory)
    (out / "REPORT.md").write_text(report.render(cfg, manifest, combinations, backtests, theory, outputs),
                                   encoding="utf-8")
    print(f"      wrote {out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
