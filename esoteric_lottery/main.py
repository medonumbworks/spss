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

from evaluation import outcome, portfolio, report  # noqa: E402
from evaluation.backtest import run_backtest  # noqa: E402
from evaluation.data import crosscheck_xlsx, load_draws, membership_matrix, split_index  # noqa: E402
from evaluation.quality import data_quality  # noqa: E402
from evaluation.stats import hit_pmf  # noqa: E402
from methods import build_methods  # noqa: E402
from methods.common import (Component, Event, Game, MethodOutput, Place, Pool, Subject,  # noqa: E402
                            games_from_config, local_datetime, to_jsonable, write_json)
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


def _digest(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(set(paths)):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def code_digest() -> str:
    """Every .py file of the pipeline."""
    return _digest([ROOT / "main.py", *ROOT.glob("methods/*.py"), *ROOT.glob("evaluation/*.py")])


def backtest_code_digest() -> str:
    """The code a backtest result depends on: everything except the report renderer,
    which only formats already-computed numbers. Reuse of a backtest requires this to
    be unchanged."""
    return _digest([ROOT / "main.py", *ROOT.glob("methods/*.py"),
                    *(p for p in ROOT.glob("evaluation/*.py") if p.name != "report.py")])


def history_hashes(cfg: dict, required: bool = True) -> dict[str, str | None]:
    """SHA-256 of every supplied history file. With ``required=False`` a missing file is
    recorded as null instead of raising (``--skip-backtest`` never reads them)."""
    out: dict[str, str | None] = {}
    for hcfg in (cfg.get("history") or {}).values():
        if not hcfg:
            continue
        paths = [hcfg["csv"]] + ([hcfg["crosscheck_xlsx"]["path"]] if hcfg.get("crosscheck_xlsx") else [])
        for rel in paths:
            if required or (ROOT / rel).exists():
                out[rel] = file_sha256(ROOT / rel)
            else:
                out[rel] = None
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


def reuse_backtest(from_dir: Path, game_key: str, config_sha: str, data_hashes: dict,
                   code_sha: str) -> dict | None:
    """A previous run's backtest summary, or None if that run has no backtest for this game.

    A backtest result depends on the configuration, the history files and the code that
    computes it — not on the target draw. All three are therefore verified against the
    source run's manifest, and anything missing or different stops the pipeline rather
    than silently recomputing (a mistyped directory must not look like a fresh run).
    """
    manifest_path = from_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"--backtest-from: {manifest_path} does not exist, so nothing can be reused from there "
                         "(check the path; it is relative to the current directory)")
    prev = json.loads(manifest_path.read_text(encoding="utf-8"))
    if prev.get("config_sha256") != config_sha:
        raise SystemExit(f"--backtest-from: config.yaml changed since {from_dir} was produced; rerun the backtest")
    prev_data = prev.get("history_files_sha256")
    if prev_data is None:
        raise SystemExit(f"--backtest-from: {manifest_path} has no history_files_sha256, so the data behind its "
                         "backtest cannot be verified; rerun the backtest")
    if prev_data != data_hashes:
        raise SystemExit(f"--backtest-from: history files changed since {from_dir} was produced; rerun the backtest")
    prev_code = prev.get("backtest_code_sha256")
    if prev_code is None:
        raise SystemExit(f"--backtest-from: {manifest_path} has no backtest_code_sha256 (it predates this check); "
                         "rerun the backtest")
    if prev_code != code_sha:
        raise SystemExit(f"--backtest-from: the code that computes the backtest changed since {from_dir} was "
                         "produced; rerun the backtest")
    summary_path = from_dir / "backtest" / game_key / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "reused"
    # The path is stored relative to the pipeline directory so the report does not depend
    # on the current working directory; no timestamp, so reruns stay byte-identical.
    try:
        rel = summary_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        rel = str(summary_path.resolve())
    summary["reused_from"] = {"summary": rel, "generated_at": prev.get("generated_at"),
                              "config_sha256": config_sha, "backtest_code_sha256": prev_code,
                              "code_sha256": prev.get("code_sha256")}
    return summary


def fixed_tickets(results_dir: str | None, game_key: str, pool: Pool) -> list[dict]:
    """Every distinct ticket a previous run proposed for this game, with where it came
    from: the ones already in the player's hands, which new tickets must be packed
    against rather than repeat — and whose companion numbers they must not reuse."""
    if not results_dir:
        return []
    path = Path(results_dir) / "final" / "combinations.json"
    if not path.exists():
        raise SystemExit(f"--portfolio-fixed-from: {path} does not exist")
    data = json.loads(path.read_text(encoding="utf-8"))
    block = (data.get("portfolio") or {}).get(game_key)
    if not block or block.get("status") != "ran":
        raise SystemExit(f"--portfolio-fixed-from: {path} holds no portfolio for {game_key}")
    seen: dict[tuple[int, ...], dict] = {}
    for vname, v in block.get("variants", {}).items():
        for t, src in zip(v.get("tickets", []), v.get("selection", {}).get("sources", [])):
            seen.setdefault(tuple(t), {"ticket": list(t), "sources": src, "variant": vname})
    return [seen[t] for t in sorted(seen)]


def companion_pool(cfg: dict, subject: Subject, event: Event, games: dict, digit_game: str,
                   readings: list[dict], needed: int, max_extra: int = 60) -> dict[tuple[str, int], list[int]]:
    """Companion numbers (Jokers) to pair with the selected tickets, one each.

    The readings already computed may not hold ``needed`` distinct values: the methods
    that do not depend on the seed repeat theirs across readings. Further readings are
    computed for the companion number alone — the same methods on the same moment, only
    re-seeded — until there are enough distinct values or ``max_extra`` is reached.
    """
    pool = games[digit_game].pools[0]
    out: dict[tuple[str, int], list[int]] = {}
    for r in readings:
        for k, o in r["outputs"].items():
            if o.status != "failed":
                out[(k, r["reading"])] = o.picks[digit_game][pool.name]
        for k, ens in r.get("ensembles", {}).items():
            if digit_game in ens["picks"]:
                out[(k, r["reading"])] = ens["picks"][digit_game][pool.name]
    seed, extra = int(cfg["seed"]), 0
    while len({tuple(v) for v in out.values()}) < needed and extra < max_extra:
        extra += 1
        k_reading = len(readings) + extra
        for m in build_methods(cfg, seed=seed + k_reading - 1):
            try:
                o = m.run(subject, event, [games[digit_game]])
            except Exception:
                continue
            out[(m.key, k_reading)] = o.picks[digit_game][pool.name]
    return out


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
    ap.add_argument("--score-drawn", metavar="NUMBERS",
                    help="score an earlier run against the numbers actually drawn, e.g. "
                         "--score-drawn '2,3,6,7,8,9,11,13,20,21,22' --score-from results/draw_2026-10-24; "
                         "writes results/out_of_sample/ and exits")
    ap.add_argument("--score-from", help="results directory holding the predictions to score")
    ap.add_argument("--score-game", default="sve_ili_nista")
    ap.add_argument("--portfolio-add", type=int, metavar="N",
                    help="select N further tickets for the game(s) with a system size, packed against "
                         "the ones already committed (see --portfolio-fixed-from)")
    ap.add_argument("--portfolio-fixed-from", metavar="DIR",
                    help="results directory whose tickets are treated as already played, so the new "
                         "ones are chosen to complement them")
    args = ap.parse_args(argv)
    if args.readings < 1:
        ap.error("--readings must be at least 1")
    if args.score_drawn and not args.score_from:
        ap.error("--score-drawn needs --score-from (the run that predicted that draw)")

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    out = Path(args.results)
    subject, games = subject_of(cfg), games_from_config(cfg)
    event = event_of(cfg, args.draw_date, args.draw_time)
    seed = int(cfg["seed"])
    config_sha = file_sha256(cfg_path)
    data_hashes = history_hashes(cfg, required=not args.skip_backtest)
    bt_code_sha = backtest_code_digest()
    if args.backtest_from and Path(args.backtest_from).resolve() == out.resolve():
        ap.error("--backtest-from must not be the same directory as --results (the provenance would point at itself)")

    if args.score_drawn:
        combos, source = outcome.load_combinations(Path(args.score_from))
        pool = games[args.score_game].pools[0]
        paying = (cfg.get("portfolio") or {}).get("paying_hits", {}).get(args.score_game, [])
        record = outcome.score_predictions(combos, args.score_game, pool,
                                           outcome.parse_numbers(args.score_drawn), paying, source)
        log_path = out / "out_of_sample" / "log.json"
        records = outcome.append_log(log_path, record)
        write_json(log_path, records)
        write_json(out / "out_of_sample" / "summary.json", outcome.summarize_log(records, pool))
        print(f"      scored {record['tickets_scored']} tickets of {source} against {record['drawn']}")
        print(f"      mean hits {record['mean_hits']:.3f} (expected {record['expected_hits_per_ticket']:.3f}), "
              f"best {record['max_hits']}, in a paying tier: {record['tickets_in_paying_tier']}")
        print(f"      wrote {log_path}")
        return 0

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
            odd = ", ".join(f"{o.key}={o.status}" for o in outputs_k.values() if o.status != "ran")
            print(f"      reading {k} (seed {seed_k}): " + (odd or "all ran"))
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
        reused = (reuse_backtest(Path(args.backtest_from), game_key, config_sha, data_hashes, bt_code_sha)
                  if args.backtest_from else None)
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

    print("[3/4] ensembles and ticket portfolio")
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

    pcfg = cfg.get("portfolio") or {}
    portfolios: dict = {}
    if pcfg.get("enabled"):
        for game_key, paying in (pcfg.get("paying_hits") or {}).items():
            pool = games[game_key].pools[0]
            m = int((pcfg.get("tickets_per_game") or {}).get(game_key, pcfg["tickets"]))
            size = (pcfg.get("system_size") or {}).get(game_key)
            top = (pcfg.get("top_hits") or {}).get(game_key)
            ranges = pcfg.get("overlap_range")
            orange = tuple(ranges.get(game_key, ranges.get("default", [0, 99])) if isinstance(ranges, dict)
                           else ranges)
            # Every ticket set the user may want compared: all sources together, and —
            # for a system game — the ensembles on their own against the methods alone.
            groups = {"all": ("method", "ensemble")}
            if size:
                groups |= {"ensemble_only": ("ensemble",), "methods_only": ("method",)}
            variants, notes, extension = {}, {}, None
            for name, kinds in groups.items():
                cands = portfolio.candidate_tickets(readings, game_key, pool, size, kinds)
                if len(cands) < m:
                    notes[name] = f"only {len(cands)} distinct candidates, need {m}"
                    continue
                # "all" keeps the original tie-break key, so adding variants never
                # silently changes a set that was already published.
                suffix = "" if name == "all" else f"|{name}"
                v = portfolio.select_portfolio(
                    cands, m, pool, paying, orange, int(pcfg["exact_evaluations"]),
                    f"portfolio|{event.local.isoformat()}|{game_key}{suffix}", top)
                v["candidates"] = cands
                variants[name] = v
            if size and args.portfolio_add:
                committed = fixed_tickets(args.portfolio_fixed_from, game_key, pool)
                fixed = [tuple(c["ticket"]) for c in committed]
                cands = portfolio.candidate_tickets(readings, game_key, pool, size, ("method", "ensemble"),
                                                    int((pcfg.get("system_slices") or {}).get(game_key, 1)))
                packing = portfolio.select_portfolio_greedy(
                    cands, int(args.portfolio_add), pool,
                    f"portfolio|{event.local.isoformat()}|{game_key}|add",
                    int((pcfg.get("max_overlap_for_packing") or {}).get(game_key, 2)),
                    fixed, min(paying))
                added = [tuple(t) for t in packing["tickets"]]
                jg = (cfg.get("portfolio") or {}).get("companion_game", "joker")
                digit_pool = (companion_pool(cfg, subject, event, games, jg, readings,
                                             len(committed) + len(added)) if jg in games else {})
                if digit_pool:
                    # Assigned across the WHOLE final set, the committed tickets first, so
                    # no companion number is played twice across old and new together.
                    all_sources = [c["sources"] for c in committed] + packing["sources"]
                    assigned = portfolio.assign_companion_digits(
                        all_sources, digit_pool, f"companion|{event.local.isoformat()}|{game_key}")
                    packing["companion_numbers"] = assigned[len(committed):]
                    packing["companion_numbers_committed"] = assigned[: len(committed)]
                    packing["full_set"] = [
                        {"ticket": t, "companion": a["digits"], "group": g}
                        for t, a, g in zip([c["ticket"] for c in committed] + packing["tickets"],
                                           assigned,
                                           ["committed"] * len(committed) + ["added"] * len(added))]
                packing.update({
                    "evaluation_added_only": portfolio.evaluate_auto(added, pool, paying, top),
                    "evaluation_with_fixed": portfolio.evaluate_auto(list(fixed) + added, pool, paying, top),
                    "evaluation_fixed_only": (portfolio.evaluate_auto(list(fixed), pool, paying, top)
                                              if fixed else None),
                })
                extension = packing     # a different size, so never compared as a variant
                print(f"      portfolio {game_key}: added {len(added)} systems, "
                      f"P(at least one paying) {packing['evaluation_added_only']['p_at_least_one_paying']:.4f} "
                      f"alone, {packing['evaluation_with_fixed']['p_at_least_one_paying']:.4f} "
                      f"with the {len(fixed)} already committed")
            if not variants:
                portfolios[game_key] = {"status": "not_run", "reason": "; ".join(notes.values())}
                continue
            best_name = max(variants, key=lambda n: (variants[n]["p_at_least_one_paying"],
                                                     variants[n]["expected_best_ticket_hits"]))
            chosen = dict(variants[best_name])
            chosen.update({"status": "ran", "paying_hits": list(paying), "system_size": size,
                           "tickets_in_set": m, "best_variant": best_name, "variants": variants,
                           "variants_not_run": notes, "extension": extension})
            chosen["single_ticket"] = portfolio.single_ticket_reference(pool, paying, size, top)
            naive = [tuple(t["picks"][pool.name]) for t in combinations["tickets"][game_key]][:m]
            alternatives = {"same_ticket_m_times": [tuple(chosen["tickets"][0])] * m}
            if not size and len(naive) == m:
                alternatives["readings_ensembles"] = naive
            for name, v in variants.items():
                alternatives[f"variant_{name}"] = [tuple(t) for t in v["tickets"]]
            chosen["comparison"] = portfolio.compare_structures(
                [tuple(t) for t in chosen["tickets"]], alternatives, pool, paying, top)
            portfolios[game_key] = chosen
            print(f"      portfolio {game_key}: best '{best_name}', P(at least one paying) "
                  f"{chosen['p_at_least_one_paying']:.4f}"
                  + (f", systems of {size}" if size else ""))
    combinations["portfolio"] = portfolios
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
        "config_sha256": config_sha, "code_sha256": code_digest(), "backtest_code_sha256": bt_code_sha,
        "history_files_sha256": data_hashes, "git_commit_at_run": git_commit(),
        "command_line": {"config": args.config, "results": args.results, "draw_date": args.draw_date,
                         "draw_time": args.draw_time, "readings": args.readings,
                         "skip_backtest": args.skip_backtest, "backtest_from": args.backtest_from},
        "inputs": {"birth_local": subject.birth_local.isoformat(), "birth_utc": subject.birth_utc.isoformat(),
                   "birth_place": subject.place.name, "draw_local": event.local.isoformat(),
                   "draw_utc": event.utc.isoformat(), "draw_place": event.place.name},
        "methods": {k: {"status": o.status, "kind": o.kind, "error": o.error,
                        "components": [to_jsonable(c) for c in o.components]} for k, o in outputs.items()},
        "readings": [{"reading": r["reading"], "seed": r["seed"],
                      "statuses": {k: o.status for k, o in r["outputs"].items()},
                      "not_ran": {k: {"status": o.status, "error": o.error,
                                      "components": [to_jsonable(c) for c in o.components if c.status != "ran"]}
                                  for k, o in r["outputs"].items() if o.status != "ran"}} for r in readings],
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
