#!/usr/bin/env python3
"""Esoteric lottery research pipeline.

    python main.py                      # full run: final combinations, backtest, report
    python main.py --skip-backtest      # final combinations only
    python main.py --results /tmp/out   # write somewhere else

Steps: (1) every method for the target draw and all three games, intermediate results
to results/final/intermediate/*.json; (2) chronological backtest on every supplied
history file; (3) ensembles; (4) results/final/combinations.json, results/manifest.json
and results/REPORT.md. Everything except the timestamp in manifest.json is reproducible
byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
from methods.common import (Component, Event, MethodOutput, Place, Subject, games_from_config,  # noqa: E402
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


def event_of(cfg: dict) -> Event:
    d = cfg["draw"]
    place = place_of(d["place"])
    return Event(local_datetime(d["date"], d["time"], place.tz), place)


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


def code_digest() -> str:
    h = hashlib.sha256()
    for p in sorted([ROOT / "main.py", *ROOT.glob("methods/*.py"), *ROOT.glob("evaluation/*.py")]):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


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
        write_json(out_dir / "final" / "intermediate" / f"{m.key}.json", o.to_json())
    return outputs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--skip-backtest", action="store_true")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    out = Path(args.results)
    subject, event, games = subject_of(cfg), event_of(cfg), games_from_config(cfg)
    methods = build_methods(cfg)
    seed = int(cfg["seed"])

    print(f"[1/4] methods for the draw {event.local.isoformat()} ({event.place.name})")
    outputs = run_final(methods, subject, event, games, out)
    for o in outputs.values():
        print(f"      {o.key:<18} {o.status}")

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
    ensembles = {"ensemble_equal": aggregate(outputs.values(), equal_weights(outputs.values()), games.values(),
                                             f"{ens_key}|equal")}
    for game_key, bt in backtests.items():
        if bt.get("status") == "ran":
            w = {k: v["weight"] for k, v in bt["train_weights"].items()}
            ensembles[f"ensemble_train_weighted_{game_key}"] = aggregate(
                outputs.values(), w, [games[game_key]], f"{ens_key}|train_weighted")
    write_json(out / "final" / "intermediate" / "ensemble.json", ensembles)

    combinations = {"draw": {"local": event.local.isoformat(), "utc": event.utc.isoformat(), "place": event.place.name},
                    "games": {}}
    for game_key, game in games.items():
        g = {"label": game.label, "methods": {}, "ensembles": {}}
        for k, o in outputs.items():
            if o.status != "failed":
                g["methods"][k] = {"label": o.label, "status": o.status, "picks": o.picks[game_key],
                                   "derived_from_method": o.derived[game_key]}
        for k, e in ensembles.items():
            if game_key in e["picks"]:
                g["ensembles"][k] = {"picks": e["picks"][game_key]}
        combinations["games"][game_key] = g
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
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(), "code_sha256": code_digest(),
        "git_commit_at_run": git_commit(),
        "inputs": {"birth_local": subject.birth_local.isoformat(), "birth_utc": subject.birth_utc.isoformat(),
                   "birth_place": subject.place.name, "draw_local": event.local.isoformat(),
                   "draw_utc": event.utc.isoformat(), "draw_place": event.place.name},
        "methods": {k: {"status": o.status, "kind": o.kind, "error": o.error,
                        "components": [to_jsonable(c) for c in o.components]} for k, o in outputs.items()},
        "backtests": {k: {"status": v.get("status"), "reason": v.get("reason"),
                          "failed_methods": list(v.get("failed_methods", {}))} for k, v in backtests.items()},
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
