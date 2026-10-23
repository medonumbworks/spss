"""Ensemble aggregation of the method outputs.

* Number pools: weighted Borda count (J.-C. de Borda, *Mémoire sur les élections au
  scrutin*, 1781). In each method's full ranking of the pool, the first number gets N
  points, the last 1; points are multiplied by the method's weight and summed. Ties are
  broken by the same SHA-256 hash as everywhere else.
* Digit pools (Joker): for every position, a weighted plurality vote over the digits the
  methods chose; ties by hash.

Weights:

* ``equal`` (pre-registered primary ensemble): every method that ran gets weight 1.
* ``train_weighted`` (experimental): w_m ∝ exp(beta · z_m), where z_m is method m's
  z-score of mean hits against the uniform expectation on the *training* draws only;
  beta is fixed in config.yaml and never tuned. Weights are frozen before the
  evaluation period starts.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np

from .common import Game, MethodOutput, Pool, rank_pool


def borda_points(ranking: list[int], pool: Pool) -> np.ndarray:
    pts = np.zeros(pool.size)
    for pos, n in enumerate(ranking):
        pts[n - 1] = pool.size - pos
    return pts


def aggregate(outputs: Iterable[MethodOutput], weights: Mapping[str, float], games: Iterable[Game],
              key: str) -> dict:
    """Returns {"picks", "ranking", "points"} per game and pool."""
    outputs = [o for o in outputs if o.status in ("ran", "partial") and weights.get(o.key, 0) > 0]
    result = {"picks": {}, "ranking": {}, "points": {}, "weights": {o.key: float(weights[o.key]) for o in outputs}}
    for game in games:
        result["picks"][game.key], result["ranking"][game.key], result["points"][game.key] = {}, {}, {}
        for pool in game.pools:
            pkey = f"{key}|{game.key}|{pool.name}"
            if pool.kind == "digits":
                digits, tallies = [], []
                for pos in range(pool.pick):
                    votes = np.zeros(pool.size)
                    for o in outputs:
                        votes[o.picks[game.key][pool.name][pos]] += weights[o.key]
                    ranking = rank_pool(votes, Pool(pool.name, pool.size, 1, "set"), f"{pkey}|{pos}")
                    digits.append(ranking[0] - 1)
                    tallies.append(votes.tolist())
                result["picks"][game.key][pool.name] = digits
                result["ranking"][game.key][pool.name] = digits
                result["points"][game.key][pool.name] = tallies
            else:
                pts = np.zeros(pool.size)
                for o in outputs:
                    pts += weights[o.key] * borda_points(o.ranking[game.key][pool.name], pool)
                ranking = rank_pool(pts, pool, pkey)
                result["picks"][game.key][pool.name] = sorted(ranking[: pool.pick])
                result["ranking"][game.key][pool.name] = ranking
                result["points"][game.key][pool.name] = pts.tolist()
    return result


def equal_weights(outputs: Iterable[MethodOutput]) -> dict[str, float]:
    return {o.key: 1.0 for o in outputs if o.status in ("ran", "partial")}


def train_weights(train_mean_hits: Mapping[str, float], n_train: int, expected: float, sd: float,
                  beta: float) -> dict[str, dict]:
    """Softmax of z-scores computed on the training period only."""
    z = {k: (m - expected) / (sd / math.sqrt(n_train)) for k, m in train_mean_hits.items()}
    zmax = max(z.values())
    raw = {k: math.exp(beta * (v - zmax)) for k, v in z.items()}
    total = sum(raw.values())
    return {k: {"z_train": z[k], "weight": raw[k] / total * len(raw)} for k in sorted(raw)}

