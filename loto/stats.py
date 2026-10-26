"""Statistika 882 izvlačenja Loto 6/45 (2018-01-07 … 2026-08-09) za Blok 2.

Pokretanje: python3 stats.py  → out/stats.json + ispis sažetka.
"""
from __future__ import annotations

import itertools
import json
import math
from collections import Counter

import numpy as np
from scipy import stats as sps

import common as C

draws = C.load_draws()
ND = len(draws)
nums = [d["nums"] for d in draws]
EXP = ND * 6 / 45


def freq(window=None):
    sel = nums if window is None else nums[-window:]
    return Counter(x for c in sel for x in c)


def gaps():
    """Broj izvlačenja od zadnjeg pojavljivanja svakog broja (0 = u zadnjem kolu)."""
    g = {}
    for x in range(1, 46):
        k = 0
        for c in reversed(nums):
            if x in c:
                break
            k += 1
        g[x] = k
    return g


def chi2(counter, n_draws):
    obs = np.array([counter.get(x, 0) for x in range(1, 46)], dtype=float)
    exp = n_draws * 6 / 45
    stat = ((obs - exp) ** 2 / exp).sum()
    return float(stat), float(1 - sps.chi2.cdf(stat, 44))


def backtest(window=60, k=12, start=100):
    """Prediktivna moć: za svako kolo t≥start, koliko izvučenih padne u top-k 'vrućih' / dno-k 'hladnih' /
    k 'najkašnjenijih' iz prethodnih `window` kola. Očekivano pod slučajnošću 6k/45."""
    hot_hits, cold_hits, late_hits, pareto_hits = [], [], [], []
    for t in range(start, ND):
        prev = nums[t - window:t]
        f = Counter(x for c in prev for x in c)
        order = sorted(range(1, 46), key=lambda x: (-f.get(x, 0), x))
        hot = set(order[:k]); cold = set(order[-k:])
        g = {}
        for x in range(1, 46):
            kk = 0
            for c in reversed(nums[:t]):
                if x in c:
                    break
                kk += 1
            g[x] = kk
        late = set(sorted(range(1, 46), key=lambda x: -g[x])[:k])
        fa = Counter(x for c in nums[:t] for x in c)
        pareto = set(sorted(range(1, 46), key=lambda x: (-fa[x], x))[:9])
        s = set(nums[t])
        hot_hits.append(len(s & hot)); cold_hits.append(len(s & cold))
        late_hits.append(len(s & late)); pareto_hits.append(len(s & pareto))

    def summ(h, kk):
        h = np.array(h, dtype=float)
        p = 6 * kk / 45
        # hipergeometrijska varijanca za broj pogodaka u skupu veličine kk
        var = 6 * (kk / 45) * (1 - kk / 45) * (39 / 44)
        z = (h.mean() - p) / math.sqrt(var / len(h))
        return dict(mean=float(h.mean()), expected=p, n=len(h), z=float(z), p_value=float(2 * (1 - sps.norm.cdf(abs(z)))))

    return dict(hot=summ(hot_hits, k), cold=summ(cold_hits, k), late=summ(late_hits, k), pareto9=summ(pareto_hits, 9))


def main():
    out = {}
    f_all = freq(); f60 = freq(60); f100 = freq(100); f200 = freq(200)
    out["n_draws"] = ND
    out["first"] = draws[0]["date"]; out["last"] = draws[-1]["date"]
    out["expected_per_number"] = EXP
    out["freq_all"] = {x: f_all.get(x, 0) for x in range(1, 46)}
    out["freq_60"] = {x: f60.get(x, 0) for x in range(1, 46)}
    out["freq_100"] = {x: f100.get(x, 0) for x in range(1, 46)}
    out["freq_200"] = {x: f200.get(x, 0) for x in range(1, 46)}
    out["bonus_freq"] = dict(Counter(d["bonus"] for d in draws))
    out["gaps"] = gaps()
    out["chi2_all"] = chi2(f_all, ND)
    out["chi2_60"] = chi2(f60, 60)
    out["chi2_100"] = chi2(f100, 100)
    out["chi2_200"] = chi2(f200, 200)
    rank_all = sorted(range(1, 46), key=lambda x: (-f_all[x], x))
    out["rank_all"] = rank_all
    out["pareto9"] = rank_all[:9]
    out["cold9"] = rank_all[-9:]
    rank60 = sorted(range(1, 46), key=lambda x: (-f60.get(x, 0), x))
    out["hot12_60"] = rank60[:12]
    out["cold12_60"] = rank60[-12:]
    g = out["gaps"]
    out["late12"] = sorted(range(1, 46), key=lambda x: -g[x])[:12]
    # parovi
    pc = Counter(p for c in nums for p in itertools.combinations(c, 2))
    out["pair_expected"] = ND * 15 / 990
    out["top_pairs"] = [(list(p), n) for p, n in pc.most_common(25)]
    out["pair_count_dist"] = dict(Counter(pc.values()))
    # simulacija maksimuma para pod uniformnošću
    rng = np.random.default_rng(20261029)
    maxes = []
    for _ in range(300):
        cc = Counter()
        for _ in range(ND):
            c = rng.choice(45, 6, replace=False) + 1
            cc.update(itertools.combinations(sorted(c.tolist()), 2))
        maxes.append(max(cc.values()))
    out["pair_max_observed"] = pc.most_common(1)[0][1]
    out["pair_max_sim_mean"] = float(np.mean(maxes)); out["pair_max_sim_p95"] = float(np.percentile(maxes, 95))
    # profil
    sums = np.array([sum(c) for c in nums]); odds = np.array([sum(x % 2 for x in c) for c in nums])
    le22 = np.array([sum(x <= 22 for x in c) for c in nums]); le31 = np.array([sum(x <= 31 for x in c) for c in nums])
    cons = np.array([sum(1 for i in range(5) if c[i + 1] - c[i] == 1) for c in nums])
    spread = np.array([c[-1] - c[0] for c in nums])
    dec = np.array([len(set((x - 1) // 10 for x in c)) for c in nums])
    out["sum"] = dict(mean=float(sums.mean()), std=float(sums.std()), pct={p: float(np.percentile(sums, p)) for p in (5, 10, 25, 50, 75, 90, 95)})
    out["odd"] = dict(mean=float(odds.mean()), dist=dict(Counter(odds.tolist())))
    out["le22"] = dict(mean=float(le22.mean()), dist=dict(Counter(le22.tolist())))
    out["le31"] = dict(mean=float(le31.mean()), dist=dict(Counter(le31.tolist())))
    out["consec"] = dict(mean=float(cons.mean()), p_any=float((cons > 0).mean()),
                         p_any_theory=1 - math.comb(40, 6) / math.comb(45, 6), dist=dict(Counter(cons.tolist())))
    out["spread"] = dict(mean=float(spread.mean()), pct={p: float(np.percentile(spread, p)) for p in (5, 50, 95)})
    out["decades"] = dict(mean=float(dec.mean()), dist=dict(Counter(dec.tolist())))
    # ponavljanja iz prethodnog kola
    rep = np.array([len(set(nums[i]) & set(nums[i - 1])) for i in range(1, ND)])
    out["repeat_prev"] = dict(mean=float(rep.mean()), expected=0.8, dist=dict(Counter(rep.tolist())))
    # backtest
    out["backtest_60_12"] = backtest(60, 12)
    out["backtest_100_12"] = backtest(100, 12)
    # zadnjih 10 izvlačenja
    out["last10"] = [(d["date"], list(d["nums"]), d["bonus"]) for d in draws[-10:]]
    (C.HERE / "out").mkdir(exist_ok=True)
    json.dump(out, open(C.HERE / "out" / "stats.json", "w"), indent=1, ensure_ascii=False, default=str)

    print(f"izvlačenja: {ND} ({out['first']} … {out['last']}), očekivano po broju {EXP:.1f}")
    print("chi2 sve:", out["chi2_all"], " chi2 60:", out["chi2_60"], " chi2 200:", out["chi2_200"])
    print("Pareto 9:", out["pareto9"], [f_all[x] for x in out["pareto9"]])
    print("hladnih 9:", out["cold9"], [f_all[x] for x in out["cold9"]])
    print("vrućih 12 (60 kola):", out["hot12_60"], [f60.get(x, 0) for x in out["hot12_60"]])
    print("hladnih 12 (60 kola):", out["cold12_60"], [f60.get(x, 0) for x in out["cold12_60"]])
    print("kašnjenje top12:", [(x, g[x]) for x in out["late12"]])
    print("parovi top10:", out["top_pairs"][:10], "očekivano", round(out["pair_expected"], 1),
          "max sim", out["pair_max_sim_mean"], out["pair_max_sim_p95"])
    print("zbroj:", out["sum"]); print("neparni:", out["odd"]); print("≤31:", out["le31"]); print("uzastopni:", out["consec"])
    print("ponavljanje iz prethodnog:", out["repeat_prev"])
    print("backtest 60/12:", json.dumps(out["backtest_60_12"], indent=0))
    print("backtest 100/12:", json.dumps(out["backtest_100_12"], indent=0))


if __name__ == "__main__":
    main()
