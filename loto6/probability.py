"""Egzaktne vjerojatnosti enumeracijom svih C(45,6) = 8.145.060 izvlacenja u numpyju (bez simulacije)."""
from __future__ import annotations
import itertools
import time
from math import comb

import numpy as np

N_DRAWS = comb(45, 6)   # 8145060
_CACHE: dict[str, np.ndarray] = {}


def all_draws() -> np.ndarray:
    """uint8 (8145060, 6), leksikografski; kesirano u modulu."""
    if "draws" in _CACHE:
        return _CACHE["draws"]
    t0 = time.time()
    comb4 = np.array(list(itertools.combinations(range(1, 46), 4)), dtype=np.uint8)   # C(45,4) = 148995
    first = comb4[:, 0]
    blocks = []
    for a in range(1, 41):
        for b in range(a + 1, 42):
            idx = int(np.searchsorted(first, b + 1))
            tail = comb4[idx:]
            if len(tail) == 0:
                continue
            head = np.empty((len(tail), 2), dtype=np.uint8)
            head[:, 0] = a
            head[:, 1] = b
            blocks.append(np.hstack([head, tail]))
    draws = np.vstack(blocks)
    assert draws.shape == (N_DRAWS, 6), draws.shape
    _CACHE["draws"] = draws
    print(f"  [probability] izgradjeno {N_DRAWS} izvlacenja u {time.time() - t0:.1f} s")
    return draws


def hits_for(draws: np.ndarray, combo: list[int]) -> np.ndarray:
    mask = np.zeros(46, dtype=bool)
    mask[list(combo)] = True
    return mask[draws].sum(axis=1).astype(np.uint8)


def hits_matrix(draws: np.ndarray, combos: list[list[int]]) -> np.ndarray:
    out = np.empty((len(draws), len(combos)), dtype=np.uint8)
    for j, c in enumerate(combos):
        out[:, j] = hits_for(draws, c)
    return out


def hypergeom_counts() -> dict[int, int]:
    return {k: comb(6, k) * comb(39, 6 - k) for k in range(7)}


def set_probabilities(combos: list[list[int]]) -> dict:
    draws = all_draws()
    best = np.zeros(len(draws), dtype=np.uint8)
    per = []
    total_hits = 0
    for c in combos:
        assert len(c) == 6 and len(set(c)) == 6 and all(1 <= n <= 45 for n in c), c
        h = hits_for(draws, c)
        np.maximum(best, h, out=best)
        cnt = np.bincount(h, minlength=7)
        total_hits += int((h.astype(np.int64)).sum())
        per.append({"combo": sorted(c), "p3": cnt[3] / N_DRAWS, "p4": cnt[4] / N_DRAWS, "p5": cnt[5] / N_DRAWS,
                    "p6": cnt[6] / N_DRAWS, "counts": cnt.tolist()})
    bc = np.bincount(best, minlength=7)
    ge = lambda k: int(bc[k:].sum()) / N_DRAWS
    return {"n_draws": N_DRAWS, "n_combos": len(combos), "p_any_ge3": ge(3), "p_any_ge4": ge(4), "p_any_ge5": ge(5),
            "p_any_6": ge(6), "best_counts": bc.tolist(), "expected_hits_per_combo": total_hits / (N_DRAWS * len(combos)),
            "per_combo": per}


def selftest() -> None:
    d = all_draws()
    assert d.shape == (8145060, 6) and d.dtype == np.uint8
    assert tuple(d[0]) == (1, 2, 3, 4, 5, 6) and tuple(d[-1]) == (40, 41, 42, 43, 44, 45)
    assert np.all(d[:, 1:] > d[:, :-1])
    hc = hypergeom_counts()
    assert hc[3] == 182780 and hc[4] == 11115 and hc[5] == 234 and hc[6] == 1 and sum(hc.values()) == N_DRAWS
    r = set_probabilities([[1, 2, 3, 4, 5, 6]])
    pc = r["per_combo"][0]["counts"]
    assert pc == [hc[k] for k in range(7)], pc
    assert abs(r["expected_hits_per_combo"] - 0.8) < 1e-12
    assert r["p_any_6"] == 1 / N_DRAWS and abs(r["p_any_ge5"] - 235 / N_DRAWS) < 1e-15
    assert abs(r["p_any_ge3"] - (182780 + 11115 + 234 + 1) / N_DRAWS) < 1e-15
    r2 = set_probabilities([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    assert r2["p_any_6"] == 2 / N_DRAWS and abs(r2["expected_hits_per_combo"] - 0.8) < 1e-12
    print(f"  jedna kombinacija: p3={pc[3]/N_DRAWS:.6f} p4={pc[4]/N_DRAWS:.8f} p5={pc[5]/N_DRAWS:.9f} p6={pc[6]/N_DRAWS:.10f}")
    print("probability selftest OK")


if __name__ == "__main__":
    selftest()
