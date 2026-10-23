"""Chronological (walk-forward) backtest.

Leakage rules enforced here and checked by tests/test_backtest_leakage.py:

* A prediction for draw t is computed from the draw's own date and time, the birth data
  and, for the data-driven controls, only the rows ``matrix[:t]`` of earlier draws.
  Esoteric methods never see any draw result.
* The training/evaluation split is a fixed date in config.yaml. The only fitted
  quantities (train-weighted ensemble weights, logistic-regression coefficients) are
  computed from training draws and frozen before the first evaluation draw.
* All tests, confidence intervals and corrections are computed on the evaluation draws;
  training-period figures are descriptive.
"""
from __future__ import annotations

import traceback
from dataclasses import replace

import numpy as np
import pandas as pd

from methods.common import Event, Game, MethodOutput, Place, Subject, local_datetime
from methods.ensemble import aggregate, equal_weights, train_weights

from . import stats
from .controls import HotFrequencyControl, LogisticControl
from .data import membership_matrix


def _hits(ranking: list[int], row: np.ndarray, pick: int) -> int:
    return int(sum(row[n - 1] for n in ranking[:pick]))


def _borda_scores(ranking: list[int], size: int) -> np.ndarray:
    s = np.zeros(size)
    for pos, n in enumerate(ranking):
        s[n - 1] = size - pos
    return s


def run_backtest(df: pd.DataFrame, game: Game, subject: Subject, methods: list, place: Place, draw_time: str,
                 split: int, cfg: dict) -> dict:
    ev_cfg = cfg["evaluation"]
    seed = int(cfg["seed"])
    pools = [p for p in game.pools if p.kind == "set"]
    matrices = {p.name: membership_matrix(df, p) for p in pools}
    T = len(df)

    rankings: dict[str, dict[str, list]] = {}
    sequences: dict[str, list] = {m.key: [None] * T for m in methods}
    errors: dict[str, list] = {m.key: [] for m in methods}
    stored: list[list[MethodOutput]] = []

    def put(strategy: str, pool: str, t: int, ranking: list[int] | None) -> None:
        rankings.setdefault(strategy, {}).setdefault(pool, [None] * T)[t] = ranking

    for t in range(T):
        event = Event(local_datetime(df["date"].iloc[t], draw_time, place.tz), place)
        outputs = []
        for m in methods:
            try:
                out = m.run(subject, event, [game])
            except Exception as exc:  # recorded, never silently ignored
                errors[m.key].append({"date": df["date"].iloc[t].isoformat(), "error": repr(exc),
                                      "trace": traceback.format_exc(limit=3)})
                continue
            sequences[m.key][t] = out.sequence
            light = replace(out, intermediate={}, components=out.components)
            outputs.append(light)
            for p in pools:
                put(m.key, p.name, t, out.ranking[game.key][p.name])
        stored.append(outputs)
        ens = aggregate(outputs, equal_weights(outputs), [game], f"ensemble_equal|{event.local.isoformat()}")
        for p in pools:
            put("ensemble_equal", p.name, t, ens["ranking"][game.key][p.name])

    hot = HotFrequencyControl(int(ev_cfg["controls"]["hot_window"]))
    for p in pools:
        for t in range(T):
            put(hot.key, p.name, t, hot.ranking(matrices[p.name], t, p, f"{game.key}|{p.name}|{t}"))

    failed = {k: v for k, v in errors.items() if v}
    method_keys = [m.key for m in methods if m.key not in failed]

    def hits_of(strategy: str, p, idx) -> np.ndarray:
        return np.array([_hits(rankings[strategy][p.name][t], matrices[p.name][t], p.pick) for t in idx])

    train_idx, eval_idx = range(0, split), range(split, T)
    primary = pools[0]
    k, pmf0 = np.arange(primary.pick + 1), stats.hit_pmf(primary)
    mu0 = float(np.dot(k, pmf0))
    sd0 = float(np.sqrt(np.dot((k - mu0) ** 2, pmf0)))
    train_means = {k: float(hits_of(k, primary, train_idx).mean()) for k in method_keys}
    tw = train_weights(train_means, len(train_idx), mu0, sd0, float(ev_cfg["ensemble_beta"]))
    weights = {k: v["weight"] for k, v in tw.items()}
    for t in eval_idx:
        event_iso = local_datetime(df["date"].iloc[t], draw_time, place.tz).isoformat()
        ens = aggregate(stored[t], weights, [game], f"ensemble_train_weighted|{event_iso}")
        for p in pools:
            put("ensemble_train_weighted", p.name, t, ens["ranking"][game.key][p.name])

    logit = LogisticControl(int(ev_cfg["controls"]["logistic_warmup"]), float(ev_cfg["controls"]["logistic_C"]),
                            int(ev_cfg["controls"]["cv_splits"]))
    logit_reports = {}
    for p in pools:
        logit.fit(matrices[p.name], split)
        logit_reports[p.name] = logit.cv_report
        for t in eval_idx:
            put(logit.key, p.name, t, logit.ranking(matrices[p.name], t, p, f"{game.key}|{p.name}|{t}"))

    strategies = method_keys + ["ensemble_equal", "ensemble_train_weighted", hot.key, logit.key]
    reps_boot = int(ev_cfg["bootstrap_replicates"])
    results: dict = {"pools": {}}
    for p in pools:
        pmf = stats.hit_pmf(p)
        tail = tuple(ev_cfg["extreme_tail"][game.key]) if game.key in ev_cfg.get("extreme_tail", {}) else None
        null = stats.random_ticket_null(matrices[p.name][split:], p, int(ev_cfg["mc_replicates"]), seed)
        per = {}
        for s in strategies:
            entry = {}
            for period, idx in (("training", train_idx), ("evaluation", eval_idx)):
                if rankings[s][p.name][idx[0]] is None:
                    entry[period] = None
                    continue
                h = hits_of(s, p, idx)
                entry[period] = stats.summarize_hits(h, pmf, reps_boot, seed, tail)
            ev = entry["evaluation"]
            ev["mc_p_two_sided"] = stats.mc_p_value(ev["mean_hits"], null, ev["expected_mean_hits"])
            scores = np.array([_borda_scores(rankings[s][p.name][t], p.size) for t in eval_idx])
            ev["auc"] = stats.auc_with_ci(scores, matrices[p.name][split:], int(ev_cfg["auc_bootstrap"]), seed)
            per[s] = entry
        pvals = {s: per[s]["evaluation"]["exact_test"]["p_two_sided"] for s in strategies}
        holm, bh = stats.holm(pvals), stats.benjamini_hochberg(pvals)
        alpha = float(ev_cfg["alpha"])
        family = {s: {"p_exact_two_sided": pvals[s], "p_holm": holm[s], "p_bh": bh[s],
                      "significant_uncorrected": pvals[s] < alpha, "significant_holm": holm[s] < alpha,
                      "significant_bh": bh[s] < alpha} for s in strategies}
        results["pools"][p.name] = {
            "strategies": per, "family_tests": family, "family_size": len(strategies), "alpha": alpha,
            "random_ticket_null": {"replicates": len(null), "mean": float(null.mean()), "sd": float(null.std(ddof=1)),
                                   "quantiles": {q: float(np.quantile(null, float(q))) for q in
                                                 ("0.025", "0.5", "0.975")}},
            "minimum_detectable_effect": stats.minimum_detectable_effect(len(eval_idx), sd0, alpha, len(strategies)),
            "sd_hits_null": sd0,
        }

    predictions = {}
    for s in strategies:
        rows = []
        for t in range(T):
            r = {"date": df["date"].iloc[t].isoformat(), "label": df["label"].iloc[t],
                 "period": "training" if t < split else "evaluation", "picks": {}, "hits": {}}
            for p in pools:
                rk = rankings[s][p.name][t]
                if rk is None:
                    continue
                r["picks"][p.name] = sorted(rk[: p.pick])
                r["hits"][p.name] = _hits(rk, matrices[p.name][t], p.pick)
                r["drawn"] = {p.name: list(df[p.name].iloc[t])}
            if s in sequences:
                r["raw_sequence"] = sequences[s][t]
            if r["picks"]:
                rows.append(r)
        predictions[s] = rows

    return {
        "results": results,
        "predictions": predictions,
        "train_weights": tw,
        "train_mean_hits": train_means,
        "logistic_control": logit_reports,
        "failed_methods": failed,
        "split": {"n_training": len(train_idx), "n_evaluation": len(eval_idx),
                  "training_last_date": df["date"].iloc[split - 1].isoformat(),
                  "evaluation_first_date": df["date"].iloc[split].isoformat(),
                  "evaluation_last_date": df["date"].iloc[T - 1].isoformat()},
        "strategies": strategies,
    }
