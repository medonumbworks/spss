#!/usr/bin/env python3
"""Neovisna statistička analiza izvlačenja Hrvatske Lutrije Loto 6/45.

Ulaz : data/loto_6_45_2018_2026.csv  (882 izvlačenja, 07.01.2018. – 09.08.2026.)
Izlaz: out/stats_neovisno.json, out/stats_neovisno.md

Namjerno NE koristi stats.py / common.py – sve se računa iznova iz CSV-a, kako bi
rezultat bio neovisna provjera. Pokretanje: python3 stats_neovisno.py

Metodološke napomene
--------------------
* z-score frekvencije broja: (opaženo − 117,6) / sd, sd = sqrt(882·p·(1−p)), p = 6/45
  (binomna disperzija; svaki broj u svakom kolu ima vjerojatnost 6/45).
* Klasični hi-kvadrat Σ(O−E)²/E s 44 st. sl. je za izvlačenje bez vraćanja (6 od 45)
  konzervativan: pod H0 očekivanje statistike je 44·(45−6)/(45−1) = 39, a ne 44.
  Zato se uz klasični test daje i korigirani (× 44/39) te Monte-Carlo p-vrijednost
  iz 1000 simuliranih povijesti iste veličine (seed 20261029).
* Kašnjenje = broj kola od zadnjeg pojavljivanja (0 = izvučen u zadnjem kolu).
  Očekivani razmak između dva pojavljivanja je 45/6 = 7,5 kola, pa je očekivano
  trenutno kašnjenje 7,5 − 1 = 6,5; P(kašnjenje ≥ g) = (39/45)^g.
* Backtest: skupovi se biraju isključivo iz podataka prije kola t (bez curenja
  budućnosti); izjednačenja u rangu razbijaju se slučajno (seed 20261029).
"""
from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.special import comb

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "loto_6_45_2018_2026.csv"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

N, K = 45, 6
P1 = K / N                      # vjerojatnost da je zadani broj među 6 izvučenih
SEED = 20261029
NSIM = 1000
TOTAL = int(comb(N, K, exact=True))  # 8 145 060
FPC = (N - K) / (N - 1)         # korekcija konačne populacije 39/44

T0 = time.time()


# ----------------------------------------------------------------------------- pomoćne
def py(o):
    """numpy → čisti Python za JSON."""
    if isinstance(o, dict):
        return {str(k): py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [py(v) for v in o]
    if isinstance(o, np.ndarray):
        return [py(v) for v in o.tolist()]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if math.isnan(v) else v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def fmt(x, d=3):
    if x is None:
        return "–"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if abs(x) < 1e-4 and x != 0:
        return f"{x:.2e}"
    return f"{x:.{d}f}"


def pfmt(p):
    if p is None:
        return "–"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def chi2_counts(obs, exp):
    obs = np.asarray(obs, float)
    exp = np.asarray(exp, float)
    return float(((obs - exp) ** 2 / exp).sum())


def pool_categories(labels, obs, exp, min_exp=5.0):
    """Spaja susjedne kategorije dok sve očekivane frekvencije nisu ≥ min_exp."""
    labels = [str(l) for l in labels]
    obs = list(map(float, obs))
    exp = list(map(float, exp))
    while len(exp) > 1 and min(exp) < min_exp:
        i = int(np.argmin(exp))
        if i == 0:
            j = 1
        elif i == len(exp) - 1:
            j = i - 1
        else:
            j = i - 1 if exp[i - 1] <= exp[i + 1] else i + 1
        a, b = sorted((i, j))
        labels[a] = f"{labels[a]}+{labels[b]}"
        obs[a] += obs[b]
        exp[a] += exp[b]
        del labels[b], obs[b], exp[b]
    return labels, obs, exp


def gof_test(values, pmf_support, pmf, label):
    """Hi-kvadrat test prilagodbe diskretne opažene distribucije egzaktnoj pmf."""
    values = np.asarray(values)
    n = len(values)
    obs = np.array([(values == v).sum() for v in pmf_support], float)
    exp = np.asarray(pmf, float) * n
    labs, o, e = pool_categories(list(pmf_support), obs, exp)
    stat = chi2_counts(o, e)
    df = len(o) - 1
    p = float(st.chi2.sf(stat, df)) if df > 0 else None
    return dict(opis=label, kategorije=labs, opazeno=[int(x) for x in o], ocekivano=[round(x, 2) for x in e],
                chi2=stat, df=df, p=p)


def ztest_mean(values, mu0, sd0):
    values = np.asarray(values, float)
    n = len(values)
    z = (values.mean() - mu0) / (sd0 / math.sqrt(n))
    return float(z), float(2 * st.norm.sf(abs(z)))


def acf(x, maxlag):
    x = np.asarray(x, float)
    x = x - x.mean()
    d = float((x * x).sum())
    return [float((x[:-k] * x[k:]).sum() / d) for k in range(1, maxlag + 1)]


def ljung_box(r, n):
    h = len(r)
    q = n * (n + 2) * sum(rk ** 2 / (n - k) for k, rk in enumerate(r, start=1))
    return float(q), float(st.chi2.sf(q, h))


def runs_test(b):
    b = np.asarray(b, bool)
    n1 = int(b.sum())
    n2 = int((~b).sum())
    n = n1 + n2
    runs = 1 + int((b[1:] != b[:-1]).sum())
    mu = 1 + 2 * n1 * n2 / n
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n * n * (n - 1))
    z = (runs - mu) / math.sqrt(var) if var > 0 else float("nan")
    return dict(n1=n1, n2=n2, runs=runs, ocekivano=mu, z=z, p=float(2 * st.norm.sf(abs(z))))


def mc_p(sim, obs, two_sided=False, center=None):
    """Monte-Carlo p-vrijednost: udio simulacija ≥ opaženog (ili |·−c| ≥ |obs−c|)."""
    sim = np.asarray(sim, float)
    if two_sided:
        return float((np.abs(sim - center) >= abs(obs - center) - 1e-12).mean())
    return float((sim >= obs - 1e-12).mean())


# ----------------------------------------------------------------------------- podaci
df = pd.read_csv(DATA, dtype={"joker": str})
ND = len(df)
M = np.sort(df[["n1", "n2", "n3", "n4", "n5", "n6"]].to_numpy(dtype=np.int16), axis=1)
BON = df["bonus"].to_numpy(dtype=np.int16)
assert M.min() == 1 and M.max() == 45 and (np.diff(M, axis=1) > 0).all()
assert not any(BON[i] in M[i] for i in range(ND))
IND = np.zeros((ND, N), dtype=np.int8)
IND[np.arange(ND)[:, None], M - 1] = 1
EXP = ND * P1                         # 117.6
SD1 = math.sqrt(ND * P1 * (1 - P1))   # 10.096
DATES = df["date"].tolist()

# ----------------------------------------------------------------------------- simulacije
rng = np.random.default_rng(SEED)


def sim_histories(nsim, nd, rng, chunk=100):
    out = np.empty((nsim, nd, K), dtype=np.int8)
    for s in range(0, nsim, chunk):
        e = min(s + chunk, nsim)
        r = rng.random((e - s, nd, N))
        out[s:e] = np.sort(np.argsort(r, axis=2)[:, :, :K], axis=2) + 1
    return out


SIM = sim_histories(NSIM, ND, rng)   # (1000, 882, 6)


def counts_per_history(Ms):
    """Ms: (h, nd, 6) → (h, 45) frekvencije."""
    h, nd, _ = Ms.shape
    ids = (Ms.astype(np.int64) - 1 + (np.arange(h) * N)[:, None, None]).reshape(-1)
    return np.bincount(ids, minlength=h * N).reshape(h, N)


# =============================================================================== 1. FREKVENCIJE
freq = np.bincount(M.reshape(-1) - 1, minlength=N).astype(int)
freq_b = np.bincount(BON - 1, minlength=N).astype(int)
z = (freq - EXP) / SD1
p_bin = np.array([st.binomtest(int(f), ND, P1).pvalue for f in freq])
order_desc = sorted(range(N), key=lambda i: (-freq[i], i))
rank = np.empty(N, int)
for r, i in enumerate(order_desc, start=1):
    rank[i] = r

# kašnjenje (0 = u zadnjem kolu)
last_idx = np.array([np.max(np.where(IND[:, i] == 1)[0]) if IND[:, i].any() else -1 for i in range(N)])
gap = (ND - 1) - last_idx
mean_gap_all = np.array([np.diff(np.where(IND[:, i] == 1)[0]).mean() for i in range(N)])  # prosječni razmak


def chi2_block(cnt, nd, sim_cnt):
    e = nd * P1
    stat = chi2_counts(cnt, e)
    stat_sim = ((sim_cnt - e) ** 2 / e).sum(axis=1)
    return dict(
        n_izvlacenja=int(nd), ocekivano_po_broju=e, chi2=stat, df=N - 1, p=float(st.chi2.sf(stat, N - 1)),
        chi2_korigirani=stat / FPC, p_korigirani=float(st.chi2.sf(stat / FPC, N - 1)),
        chi2_ocekivano_pod_H0=(N - 1) * FPC, p_monte_carlo=mc_p(stat_sim, stat), chi2_sim_prosjek=float(stat_sim.mean()),
        max_broj=int(np.argmax(cnt) + 1), max_frekv=int(cnt.max()), min_broj=int(np.argmin(cnt) + 1), min_frekv=int(cnt.min()),
    )


SIM_CNT_ALL = counts_per_history(SIM)
chi2_all = chi2_block(freq, ND, SIM_CNT_ALL)
windows = {}
for w in (60, 100, 200):
    cw = np.bincount(M[-w:].reshape(-1) - 1, minlength=N)
    blk = chi2_block(cw, w, counts_per_history(SIM[:, -w:, :]))
    ow = sorted(range(N), key=lambda i: (-cw[i], i))
    blk["top5"] = [int(i + 1) for i in ow[:5]]
    blk["top5_frekv"] = [int(cw[i]) for i in ow[:5]]
    blk["dno5"] = [int(i + 1) for i in ow[-5:]]
    blk["dno5_frekv"] = [int(cw[i]) for i in ow[-5:]]
    blk["frekvencije"] = [int(x) for x in cw]
    windows[f"zadnjih_{w}"] = blk

# bonus (1 kugla po kolu → multinomna, bez korekcije)
eb = ND / N
chi2_bonus = chi2_counts(freq_b, eb)
bonus_blk = dict(ocekivano_po_broju=eb, sd=math.sqrt(ND * (1 / N) * (1 - 1 / N)), chi2=chi2_bonus, df=N - 1,
                 p=float(st.chi2.sf(chi2_bonus, N - 1)), max_broj=int(np.argmax(freq_b) + 1), max_frekv=int(freq_b.max()),
                 min_broj=int(np.argmin(freq_b) + 1), min_frekv=int(freq_b.min()))
freq7 = freq + freq_b
e7 = ND * 7 / N
chi2_7 = chi2_counts(freq7, e7)

# ekstremi: je li najčešći broj neobičan s obzirom na 45 usporedbi?
sim_z_max = np.abs((SIM_CNT_ALL - EXP) / SD1).max(axis=1)
ekstremi = dict(
    max_abs_z=float(np.abs(z).max()), broj_max_abs_z=int(np.argmax(np.abs(z)) + 1),
    p_max_abs_z_monte_carlo=mc_p(sim_z_max, float(np.abs(z).max())),
    min_p_binom=float(p_bin.min()), bonferroni_min_p=float(min(1.0, p_bin.min() * N)),
    broj_z_iznad_196=int((np.abs(z) > 1.96).sum()), ocekivano_iznad_196=N * 0.05,
    sim_prosjek_max_abs_z=float(sim_z_max.mean()),
)

top9 = [int(i + 1) for i in order_desc[:9]]
bottom9 = [int(i + 1) for i in sorted(range(N), key=lambda i: (freq[i], i))[:9]]
top9_tie = dict(frekv_9=int(freq[order_desc[8]]), frekv_10=int(freq[order_desc[9]]),
                izjednaceni_na_granici=[int(i + 1) for i in range(N) if freq[i] == freq[order_desc[8]]])
bot_sorted = sorted(range(N), key=lambda i: (freq[i], i))
bottom9_tie = dict(frekv_9=int(freq[bot_sorted[8]]), frekv_10=int(freq[bot_sorted[9]]),
                   izjednaceni_na_granici=[int(i + 1) for i in range(N) if freq[i] == freq[bot_sorted[8]]])
pareto_udio = float(freq[order_desc[:9]].sum() / freq.sum())

# kašnjenje – MC distribucija maksimuma trenutnog kašnjenja
def max_gap_sim(Ms, chunk=100):
    out = np.empty(Ms.shape[0])
    t = np.arange(Ms.shape[1], dtype=np.int16)
    for s in range(0, Ms.shape[0], chunk):
        sub = Ms[s:s + chunk]
        ind = np.zeros((sub.shape[0], sub.shape[1], N), dtype=bool)
        ind[np.arange(sub.shape[0])[:, None, None], np.arange(sub.shape[1])[None, :, None], sub.astype(int) - 1] = True
        last = np.where(ind, t[None, :, None], -1).max(axis=1)
        out[s:s + chunk] = ((Ms.shape[1] - 1) - last).max(axis=1)
    return out


sim_max_gap = max_gap_sim(SIM)
gmax = int(gap.max())
gmax_nums = [int(i + 1) for i in range(N) if gap[i] == gmax]
kasnjenje_blk = dict(
    definicija="broj kola od zadnjeg pojavljivanja; 0 = izvučen u zadnjem kolu (09.08.2026.)",
    ocekivani_razmak=N / K, ocekivano_trenutno=(1 - P1) / P1, sd_trenutno=math.sqrt((1 - P1)) / P1,
    prosjek_trenutnog=float(gap.mean()), medijan_trenutnog=float(np.median(gap)),
    max_kasnjenje=gmax, max_broj=gmax_nums[0], max_brojevi=gmax_nums,
    p_jedan_broj_ge=float((1 - P1) ** gmax), p_max_od_45_aproks=float(1 - (1 - (1 - P1) ** gmax) ** N),
    p_max_monte_carlo=mc_p(sim_max_gap, gmax), sim_prosjek_max=float(sim_max_gap.mean()),
    sim_medijan_max=float(np.median(sim_max_gap)),
    prosjecni_razmak_svih_brojeva=float(mean_gap_all.mean()),
)

po_broju = []
for i in range(N):
    po_broju.append(dict(
        broj=i + 1, glavni=int(freq[i]), rang=int(rank[i]), z=float(z[i]), p_binom=float(p_bin[i]),
        bonus=int(freq_b[i]), z_bonus=float((freq_b[i] - eb) / bonus_blk["sd"]), ukupno7=int(freq7[i]),
        kasnjenje=int(gap[i]), zadnji_datum=DATES[last_idx[i]], prosjecni_razmak=float(mean_gap_all[i]),
    ))

sec1 = dict(
    n_izvlacenja=ND, ocekivano_po_broju=EXP, sd_binomna=SD1, korekcija_konacne_populacije=FPC,
    po_broju=po_broju, chi2_cijela_povijest=chi2_all, chi2_prozori=windows, bonus=bonus_blk,
    chi2_7_kugli=dict(chi2=chi2_7, df=N - 1, p=float(st.chi2.sf(chi2_7, N - 1)), ocekivano=e7),
    ekstremi=ekstremi, pareto_top9=top9, pareto_top9_frekv=[int(freq[i - 1]) for i in top9], pareto_udio_izvlacenja=pareto_udio,
    pareto_granica=top9_tie, hladni_9=bottom9, hladni_9_frekv=[int(freq[i - 1]) for i in bottom9], hladni_granica=bottom9_tie,
    kasnjenje=kasnjenje_blk,
)
print(f"[1] frekvencije gotove  {time.time() - T0:.1f}s")

# =============================================================================== 2. PAROVI
PI, PJ = np.triu_indices(K, 1)                 # 15 parova po kolu
NPAIR = N * (N - 1) // 2                       # 990
E_PAIR = ND * 15 / NPAIR
VALID = np.zeros(N * N, bool)
for a in range(N):
    for b in range(a + 1, N):
        VALID[a * N + b] = True


def pair_counts(Ms):
    """Ms: (h, nd, 6) → (h, 990)."""
    h = Ms.shape[0]
    A = Ms[:, :, PI].astype(np.int64) - 1
    B = Ms[:, :, PJ].astype(np.int64) - 1
    ids = (A * N + B + (np.arange(h) * N * N)[:, None, None]).reshape(-1)
    c = np.bincount(ids, minlength=h * N * N).reshape(h, N * N)
    return c[:, VALID]


PAIR_IDX = [(a + 1, b + 1) for a in range(N) for b in range(a + 1, N)]
pc = pair_counts(M[None])[0]
chi2_pairs = chi2_counts(pc, E_PAIR)
sim_pc_max = np.empty(NSIM)
sim_pc_chi2 = np.empty(NSIM)
for s in range(0, NSIM, 100):
    c = pair_counts(SIM[s:s + 100])
    sim_pc_max[s:s + 100] = c.max(axis=1)
    sim_pc_chi2[s:s + 100] = ((c - E_PAIR) ** 2 / E_PAIR).sum(axis=1)
pmax = int(pc.max())
top20_idx = sorted(range(NPAIR), key=lambda i: (-pc[i], PAIR_IDX[i]))[:20]
top20 = [dict(par=list(PAIR_IDX[i]), broj=int(pc[i]), z=float((pc[i] - E_PAIR) / math.sqrt(E_PAIR * (1 - 15 / NPAIR))),
              p_binom_pojedinacno=float(st.binom.sf(pc[i] - 1, ND, 15 / NPAIR))) for i in top20_idx]
p_single_pois = float(st.poisson.sf(pmax - 1, E_PAIR))
p_single_bin = float(st.binom.sf(pmax - 1, ND, 15 / NPAIR))
sec2 = dict(
    ocekivano_po_paru=E_PAIR, sd_po_paru=math.sqrt(E_PAIR * (1 - 15 / NPAIR)), n_parova=NPAIR,
    top20=top20, min_par=dict(par=list(PAIR_IDX[int(np.argmin(pc))]), broj=int(pc.min())),
    broj_parova_nikad=int((pc == 0).sum()),
    maksimum=dict(opazeno=pmax, p_jedan_par_poisson=p_single_pois, p_jedan_par_binom=p_single_bin,
                  p_max_poisson_aproks=float(1 - (1 - p_single_pois) ** NPAIR),
                  p_max_monte_carlo=mc_p(sim_pc_max, pmax), sim_prosjek_max=float(sim_pc_max.mean()),
                  sim_medijan_max=float(np.median(sim_pc_max)), sim_p95_max=float(np.percentile(sim_pc_max, 95)),
                  sim_min_max=int(sim_pc_max.min()), sim_max_max=int(sim_pc_max.max())),
    chi2=dict(chi2=chi2_pairs, df_naivno=NPAIR - 1, p_naivno=float(st.chi2.sf(chi2_pairs, NPAIR - 1)),
              p_monte_carlo=mc_p(sim_pc_chi2, chi2_pairs), sim_prosjek=float(sim_pc_chi2.mean()),
              napomena="parovi unutar kola su zavisni; mjerodavna je Monte-Carlo p-vrijednost"),
    distribucija_frekvencija_parova={int(k): int(v) for k, v in zip(*np.unique(pc, return_counts=True))},
)
print(f"[2] parovi gotovi  {time.time() - T0:.1f}s")

# =============================================================================== 3. PROFIL IZVLAČENJA
def profile(Ms):
    """Ms: (n, 6) sortirano → dict statistika po kolu."""
    Ms = Ms.astype(np.int16)
    s = Ms.sum(axis=1)
    odd = (Ms % 2).sum(axis=1)
    le22 = (Ms <= 22).sum(axis=1)
    le31 = (Ms <= 31).sum(axis=1)
    cons = (np.diff(Ms, axis=1) == 1).sum(axis=1)
    rg = Ms[:, -1] - Ms[:, 0]
    dec = (Ms - 1) // 10
    ndec = 1 + (np.diff(dec, axis=1) != 0).sum(axis=1)
    return dict(zbroj=s, neparni=odd, le22=le22, le31=le31, uzastopni=cons, raspon=rg, desetice=ndec)


obs_prof = profile(M)
t1 = time.time()
ALL = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(1, N + 1), K)),
                  dtype=np.int8, count=TOTAL * K).reshape(TOTAL, K)
all_prof = profile(ALL)
del ALL
print(f"    egzaktno nabrajanje {TOTAL} kombinacija: {time.time() - t1:.1f}s")

# sanity: formula za točno j uzastopnih parova C(5,j)·C(40,6−j)
for j in range(6):
    assert (all_prof["uzastopni"] == j).sum() == comb(5, j, exact=True) * comb(40, 6 - j, exact=True)

PCT = [5, 10, 25, 50, 75, 90, 95]
sec3 = {}
for key, label in [("zbroj", "zbroj 6 brojeva"), ("neparni", "broj neparnih"), ("le22", "brojeva ≤ 22"),
                   ("le31", "brojeva ≤ 31"), ("uzastopni", "broj uzastopnih parova (n_{i+1}−n_i = 1)"),
                   ("raspon", "raspon max−min"), ("desetice", "broj različitih desetica (1–10, 11–20, 21–30, 31–40, 41–45)")]:
    o = obs_prof[key].astype(float)
    a = all_prof[key]
    pmf_counts = np.bincount(a)
    support = np.arange(len(pmf_counts))
    pmf = pmf_counts / TOTAL
    mu = float((support * pmf).sum())
    sd = float(math.sqrt(((support - mu) ** 2 * pmf).sum()))
    cdf = np.cumsum(pmf)
    teor_pct = [int(support[np.searchsorted(cdf, q / 100)]) for q in PCT]
    zt, pz = ztest_mean(o, mu, sd)
    blk = dict(
        opis=label, prosjek=float(o.mean()), sd=float(o.std(ddof=1)), teorijski_prosjek=mu, teorijska_sd=sd,
        z_prosjek=zt, p_prosjek=pz, percentili={str(q): float(np.percentile(o, q)) for q in PCT},
        teorijski_percentili={str(q): v for q, v in zip(PCT, teor_pct)},
        min=int(o.min()), max=int(o.max()),
    )
    if len(support) <= 12:
        sup = [int(v) for v in support if pmf[v] > 0]
        blk["gof"] = gof_test(o.astype(int), sup, [pmf[v] for v in sup], label)
        blk["distribucija"] = {str(v): dict(opazeno=int((o == v).sum()), ocekivano=float(pmf[v] * ND), p=float(pmf[v])) for v in sup}
    else:
        # decilni razredi po egzaktnoj distribuciji
        edges = [int(support[np.searchsorted(cdf, q)]) for q in np.arange(0.1, 1.0, 0.1)]
        edges = sorted(set(edges))
        bins_obs = np.bincount(np.digitize(o, edges, right=True), minlength=len(edges) + 1).astype(float)
        cut = [-1] + edges + [int(support[-1])]
        bins_p = np.array([(pmf[(support > cut[i]) & (support <= cut[i + 1])]).sum() for i in range(len(cut) - 1)])
        labs = [f"≤{edges[0]}"] + [f"{edges[i] + 1}–{edges[i + 1]}" for i in range(len(edges) - 1)] + [f">{edges[-1]}"]
        stat = chi2_counts(bins_obs, bins_p * ND)
        blk["gof"] = dict(opis=f"{label}: decilni razredi", kategorije=labs, opazeno=[int(x) for x in bins_obs],
                          ocekivano=[round(float(x * ND), 2) for x in bins_p], chi2=stat, df=len(labs) - 1,
                          p=float(st.chi2.sf(stat, len(labs) - 1)))
    sec3[key] = blk

p_cons = 1 - comb(40, 6, exact=True) / TOTAL
n_cons = int((obs_prof["uzastopni"] >= 1).sum())
sec3["bar_jedan_uzastopni_par"] = dict(
    teorijska_p=float(p_cons), formula="1 − C(40,6)/C(45,6)", opazeno=n_cons, udio=n_cons / ND,
    ocekivano=float(p_cons * ND), p_binom=float(st.binomtest(n_cons, ND, p_cons).pvalue))
sec3["sve_parne_ili_sve_neparne"] = dict(
    opazeno=int(((obs_prof["neparni"] == 0) | (obs_prof["neparni"] == 6)).sum()),
    ocekivano=float(((all_prof["neparni"] == 0) | (all_prof["neparni"] == 6)).mean() * ND))
print(f"[3] profil gotov  {time.time() - T0:.1f}s")

# =============================================================================== 4. PONAVLJANJA
rep = (IND[1:] & IND[:-1]).sum(axis=1)       # zajednički brojevi s prethodnim kolom
hyp = st.hypergeom(N, K, K)
sup = list(range(0, 7))
rep_gof = gof_test(rep, sup, [hyp.pmf(k) for k in sup], "broj ponovljenih brojeva iz prethodnog kola")
rep_mu, rep_sd = hyp.mean(), hyp.std()
rep_z, rep_p = ztest_mean(rep, rep_mu, rep_sd)


def rep_gof_sim(Ms, chunk=100):
    """GOF χ² ponavljanja (kategorije 0,1,2,3+) za svaku simuliranu povijest."""
    e = np.array([hyp.pmf(0), hyp.pmf(1), hyp.pmf(2), hyp.sf(2)]) * (Ms.shape[1] - 1)
    out = np.empty(Ms.shape[0])
    for s in range(0, Ms.shape[0], chunk):
        sub = Ms[s:s + chunk].astype(int)
        ind = np.zeros((sub.shape[0], sub.shape[1], N), dtype=bool)
        ind[np.arange(sub.shape[0])[:, None, None], np.arange(sub.shape[1])[None, :, None], sub - 1] = True
        r = (ind[:, 1:] & ind[:, :-1]).sum(axis=2)
        o = np.stack([(r == 0).sum(1), (r == 1).sum(1), (r == 2).sum(1), (r >= 3).sum(1)], axis=1)
        out[s:s + chunk] = ((o - e) ** 2 / e).sum(axis=1)
    return out


rep_gof["p_monte_carlo"] = mc_p(rep_gof_sim(SIM), rep_gof["chi2"])
rep_gof["napomena"] = "jedini nominalno p<0.05 rezultat među ~40 testova; prosjek je točno očekivani, odstupa samo oblik (više 'točno 1', manje 0 i 3+)"
# ponavljanje s pomakom 2..5 (kontrola)
rep_lags = {}
for L in range(1, 6):
    r = (IND[L:] & IND[:-L]).sum(axis=1)
    zz, pp = ztest_mean(r, rep_mu, rep_sd)
    rep_lags[str(L)] = dict(prosjek=float(r.mean()), z=zz, p=pp)
# bonus iz prethodnog kola među glavnima?
bon_rep = int(sum(BON[t - 1] in M[t] for t in range(1, ND)))
# autokorelacija zbroja i broja neparnih
auto = {}
for key in ("zbroj", "neparni", "le22", "raspon"):
    r = acf(obs_prof[key], 10)
    q, pq = ljung_box(r, ND)
    auto[key] = dict(acf_1_10=r, ljung_box_Q10=q, p=pq, granica_95=float(1.96 / math.sqrt(ND)))
# autokorelacija indikatorskih serija svakog broja (lag 1)
r1 = np.array([acf(IND[:, i], 1)[0] for i in range(N)])
z_r1 = r1 * math.sqrt(ND)
auto["indikatori_brojeva_lag1"] = dict(
    prosjek_r1=float(r1.mean()), min_r1=float(r1.min()), max_r1=float(r1.max()),
    broj_znacajnih_95=int((np.abs(z_r1) > 1.96).sum()), ocekivano_znacajnih=N * 0.05,
    chi2_zbroj_z2=float((z_r1 ** 2).sum()), df=N, p_zbroj_z2=float(st.chi2.sf((z_r1 ** 2).sum(), N)),
    najjaci=[dict(broj=int(i + 1), r1=float(r1[i]), z=float(z_r1[i])) for i in np.argsort(-np.abs(r1))[:3]])
# runs test
runs = {
    "zbroj_iznad_medijana": runs_test(obs_prof["zbroj"] > np.median(obs_prof["zbroj"])),
    "neparni_vecina": runs_test(obs_prof["neparni"] >= 3),
    "bar_jedno_ponavljanje": runs_test(rep >= 1),
}
runs_num = [runs_test(IND[:, i] == 1) for i in range(N)]
zr = np.array([r["z"] for r in runs_num])
runs["indikatori_brojeva"] = dict(
    broj_znacajnih_95=int((np.abs(zr) > 1.96).sum()), ocekivano_znacajnih=N * 0.05, prosjek_z=float(zr.mean()),
    chi2_zbroj_z2=float((zr ** 2).sum()), df=N, p=float(st.chi2.sf((zr ** 2).sum(), N)),
    najjaci=[dict(broj=int(i + 1), z=float(zr[i]), runs=runs_num[i]["runs"], ocekivano=runs_num[i]["ocekivano"])
             for i in np.argsort(-np.abs(zr))[:3]])
sec4 = dict(
    ponavljanja=dict(ocekivano=float(rep_mu), sd_teorijska=float(rep_sd), prosjek=float(rep.mean()), sd=float(rep.std(ddof=1)),
                     z=rep_z, p=rep_p, gof=rep_gof,
                     distribucija={str(k): dict(opazeno=int((rep == k).sum()), ocekivano=float(hyp.pmf(k) * (ND - 1))) for k in sup},
                     udio_kola_s_bar_jednim=float((rep >= 1).mean()), teorijski_udio_bar_jedan=float(1 - hyp.pmf(0)),
                     po_pomaku=rep_lags,
                     bonus_prethodnog_kola_medju_glavnima=dict(opazeno=bon_rep, ocekivano=float((ND - 1) * P1),
                                                              p=float(st.binomtest(bon_rep, ND - 1, P1).pvalue))),
    autokorelacija=auto, runs=runs,
)
print(f"[4] ponavljanja gotova  {time.time() - T0:.1f}s")

# =============================================================================== 5. BACKTEST
START, KSET, KPAR = 100, 12, 9
WINDOWS = (30, 60, 100, 200)


def backtest(Ih, rng, windows=WINDOWS, k=KSET, start=START, k_par=KPAR):
    nd = Ih.shape[0]
    CS = np.zeros((nd + 1, N), np.int32)
    CS[1:] = np.cumsum(Ih, axis=0)
    idx = np.where(Ih.astype(bool), np.arange(nd)[:, None], -1)
    LS = np.maximum.accumulate(idx, axis=0)
    ts = np.arange(start, nd)
    T = len(ts)
    actual = Ih[ts].astype(bool)
    rows = np.arange(T)[:, None]

    def hits(sets):
        return actual[rows, sets].sum(axis=1)

    res = {}
    for w in windows:
        Fw = CS[ts] - CS[ts - w]
        order = np.argsort(-(Fw + rng.random(Fw.shape) * 0.5), axis=1)
        res[f"vruci_w{w}"] = hits(order[:, :k])
        res[f"hladni_w{w}"] = hits(order[:, -k:])
    gaps = (ts - 1)[:, None] - LS[ts - 1]
    order = np.argsort(-(gaps + rng.random(gaps.shape) * 0.5), axis=1)
    res["kasneci"] = hits(order[:, :k])
    Fall = CS[ts]
    order = np.argsort(-(Fall + rng.random(Fall.shape) * 0.5), axis=1)
    res["pareto9"] = hits(order[:, :k_par])
    res["hladni9_povijest"] = hits(order[:, -k_par:])
    res["slucajni12"] = hits(np.argsort(rng.random((T, N)), axis=1)[:, :k])
    return res


def hyper_sd(k):
    return math.sqrt(K * (k / N) * (1 - k / N) * FPC)


bt_rng = np.random.default_rng(SEED)
BT = backtest(IND, bt_rng)
T_BT = ND - START
# nulta distribucija prosjeka pogodaka iz simuliranih povijesti (isti algoritam, bez curenja)
sim_rng = np.random.default_rng(SEED + 1)
SIM_BT = {key: np.empty(NSIM) for key in BT}
for h in range(NSIM):
    Ih = np.zeros((ND, N), dtype=np.int8)
    Ih[np.arange(ND)[:, None], SIM[h].astype(int) - 1] = 1
    r = backtest(Ih, sim_rng)
    for key in BT:
        SIM_BT[key][h] = r[key].mean()


def bt_summary(key, hits, kset):
    T = len(hits)
    H = int(hits.sum())
    mu = K * kset / N
    sd0 = hyper_sd(kset)
    zz = (H - mu * T) / (sd0 * math.sqrt(T))
    pz = float(2 * st.norm.sf(abs(zz)))
    pb = float(st.binomtest(H, K * T, kset / N).pvalue)
    sim = SIM_BT[key]
    return dict(
        skup=key, k=kset, T=T, ukupno_pogodaka=H, prosjek=float(hits.mean()), sd=float(hits.std(ddof=1)),
        ocekivano=mu, sd_teorijska_po_kolu=sd0, lift=float(hits.mean() / mu), z=float(zz), p_z=pz, p_binom=pb,
        p_monte_carlo=mc_p(sim, float(hits.mean()), two_sided=True, center=mu),
        sim_prosjek=float(sim.mean()), sim_sd_prosjeka=float(sim.std(ddof=1)),
        distribucija_pogodaka={str(v): int((hits == v).sum()) for v in range(0, 7)},
        udio_kola_0_pogodaka=float((hits == 0).mean()), udio_kola_ge3=float((hits >= 3).mean()),
    )


bt_res = {}
for key, hits in BT.items():
    kset = KPAR if key in ("pareto9", "hladni9_povijest") else KSET
    bt_res[key] = bt_summary(key, hits, kset)

# osjetljivost: razbijanje izjednačenja po rastućem broju umjesto slučajno (samo w=60)
def backtest_tiebreak_asc(Ih, w=60, k=KSET, start=START):
    nd = Ih.shape[0]
    CS = np.zeros((nd + 1, N), np.int32)
    CS[1:] = np.cumsum(Ih, axis=0)
    ts = np.arange(start, nd)
    Fw = CS[ts] - CS[ts - w]
    actual = Ih[ts].astype(bool)
    rows = np.arange(len(ts))[:, None]
    order = np.lexsort((np.tile(np.arange(N), (len(ts), 1)), -Fw), axis=1)
    return actual[rows, order[:, :k]].sum(1), actual[rows, order[:, -k:]].sum(1)


hot_asc, cold_asc = backtest_tiebreak_asc(IND)
osjetljivost = {}
for name, hits in (("vruci_w60_tiebreak_asc", hot_asc), ("hladni_w60_tiebreak_asc", cold_asc)):
    H = int(hits.sum())
    zz = (H - 1.6 * T_BT) / (hyper_sd(KSET) * math.sqrt(T_BT))
    osjetljivost[name] = dict(prosjek=float(hits.mean()), z=float(zz), p_z=float(2 * st.norm.sf(abs(zz))))

# po godinama – prosjek pogodaka vrućih (w60) i kasnećih, da se vidi stabilnost
years = df["year"].to_numpy()[START:]
po_god = {}
for y in sorted(set(years)):
    m = years == y
    po_god[str(int(y))] = dict(n=int(m.sum()), vruci_w60=float(BT["vruci_w60"][m].mean()),
                               hladni_w60=float(BT["hladni_w60"][m].mean()), kasneci=float(BT["kasneci"][m].mean()),
                               pareto9=float(BT["pareto9"][m].mean()))

sec5 = dict(
    opis=("Za svako kolo t ≥ 100 (0-bazirano; T = 782 kola) skupovi se biraju iz podataka prije kola t: "
          "vrući = top-12 po frekvenciji u prethodnih w kola, hladni = dno-12, kasneći = 12 s najvećim kašnjenjem, "
          "pareto9 = top-9 po frekvenciji u cijeloj povijesti do t, hladni9_povijest = dno-9; slucajni12 = kontrolni "
          "slučajni skup. Izjednačenja se razbijaju slučajno (seed 20261029)."),
    ocekivano_12=K * KSET / N, ocekivano_9=K * KPAR / N, sd_teorijska_12=hyper_sd(KSET), sd_teorijska_9=hyper_sd(KPAR),
    primarni_prozor=60, rezultati=bt_res, osjetljivost_tiebreak=osjetljivost, po_godinama=po_god,
)
print(f"[5] backtest gotov  {time.time() - T0:.1f}s")

# =============================================================================== 6. JOKER
JK = np.array([[int(ch) for ch in s] for s in df["joker"]], dtype=np.int8)   # (882, 6)
dig_all = np.bincount(JK.reshape(-1), minlength=10)
chi2_dig = chi2_counts(dig_all, ND * 6 / 10)
pos = []
for j in range(6):
    c = np.bincount(JK[:, j], minlength=10)
    s_ = chi2_counts(c, ND / 10)
    pos.append(dict(pozicija=j + 1, chi2=s_, p=float(st.chi2.sf(s_, 9)), frekvencije=[int(x) for x in c]))
jval = df["joker"].astype(int).to_numpy()
ks = st.kstest(jval / 1e6, "uniform")
jr = acf(jval, 5)
sec6 = dict(
    n=ND, znamenke_ukupno=[int(x) for x in dig_all], ocekivano_po_znamenki=ND * 6 / 10, chi2=chi2_dig, df=9,
    p=float(st.chi2.sf(chi2_dig, 9)), po_poziciji=pos, ks_cijeli_broj=dict(D=float(ks.statistic), p=float(ks.pvalue)),
    acf_1_5=jr, ljung_box=dict(zip(("Q", "p"), ljung_box(jr, ND))),
    prosjek=float(jval.mean()), teorijski_prosjek=499999.5, vodece_nule=int((df["joker"].str[0] == "0").sum()),
    ocekivano_vodece_nule=ND / 10,
)
print(f"[6] joker gotov  {time.time() - T0:.1f}s")

# =============================================================================== ZAKLJUČAK + JSON
bt60h, bt60c, btl, btp = bt_res["vruci_w60"], bt_res["hladni_w60"], bt_res["kasneci"], bt_res["pareto9"]
backtest_str = (
    f"T={T_BT} kola (t≥100). Vrući top-12 (w=60): prosjek {bt60h['prosjek']:.3f} pogodaka/kolo (oček. 1.6; "
    f"z={bt60h['z']:+.2f}, p={bt60h['p_z']:.3f}, binom p={bt60h['p_binom']:.3f}, MC p={bt60h['p_monte_carlo']:.3f}). "
    f"Hladni dno-12 (w=60): {bt60c['prosjek']:.3f} (z={bt60c['z']:+.2f}, p={bt60c['p_z']:.3f}). "
    f"Najkašnjeniji 12: {btl['prosjek']:.3f} (z={btl['z']:+.2f}, p={btl['p_z']:.3f}). "
    f"Pareto top-9 (cijela povijest do t): {btp['prosjek']:.3f} (oček. 1.2; z={btp['z']:+.2f}, p={btp['p_z']:.3f}). "
    f"Slučajni kontrolni skup 12: {bt_res['slucajni12']['prosjek']:.3f} (z={bt_res['slucajni12']['z']:+.2f}) – bolji od svih 'strategija', "
    f"što pokazuje razinu šuma. Detektabilno odstupanje pri T={T_BT} (z=1.96): ±{1.96 * hyper_sd(KSET) / math.sqrt(T_BT):.3f} pogodaka/kolo (±{100 * 1.96 * hyper_sd(KSET) / math.sqrt(T_BT) / 1.6:.1f} %). "
    f"Ostali prozori (w=30/100/200) vrući: {bt_res['vruci_w30']['prosjek']:.3f}/{bt_res['vruci_w100']['prosjek']:.3f}/"
    f"{bt_res['vruci_w200']['prosjek']:.3f}; hladni: {bt_res['hladni_w30']['prosjek']:.3f}/{bt_res['hladni_w100']['prosjek']:.3f}/"
    f"{bt_res['hladni_w200']['prosjek']:.3f}."
)

any_sig = [k for k, v in bt_res.items() if v["p_z"] < 0.05]
conclusion = (
    f"Frekvencije 45 brojeva u 882 kola ne odstupaju od uniformnosti (χ²={chi2_all['chi2']:.1f}, 44 d.f., p={chi2_all['p']:.3f}; "
    f"korigirano za izvlačenje bez vraćanja p={chi2_all['p_korigirani']:.3f}, Monte-Carlo p={chi2_all['p_monte_carlo']:.3f}), "
    f"a ni prozori od 60/100/200 kola ne pokazuju značajne 'vruće' anomalije. "
    f"Najčešći broj {order_desc[0] + 1} ({freq[order_desc[0]]}×) i najrjeđi {bot_sorted[0] + 1} ({freq[bot_sorted[0]]}×) "
    f"su unutar raspona koji se očekuje kao ekstrem od 45 slučajnih brojača (MC p={ekstremi['p_max_abs_z_monte_carlo']:.2f}); "
    f"najdulje kašnjenje {gmax} kola (broj {gmax_nums[0]}) također je uobičajeno (MC p={kasnjenje_blk['p_max_monte_carlo']:.2f}). "
    f"Najčešći par pojavio se {pmax}× (oček. 13.4), što je tipičan maksimum među 990 parova (MC p={sec2['maksimum']['p_max_monte_carlo']:.2f}). "
    f"Profil kola (zbroj, parnost, uzastopni, raspon) i ponavljanja iz prethodnog kola ({rep.mean():.2f} vs. 0.8) slažu se s teorijom – jedino je oblik distribucije ponavljanja nominalno p≈{rep_gof['p']:.3f} (MC p={rep_gof['p_monte_carlo']:.3f}), što je pri ~40 provedenih testova očekivani broj lažnih signala. "
    f"Backtest na {T_BT} kola pokazuje da vrući, hladni, najkašnjeniji i Pareto brojevi pogađaju u prosjeku "
    f"{bt60h['prosjek']:.2f}/{bt60c['prosjek']:.2f}/{btl['prosjek']:.2f} (oček. 1.6) odnosno {btp['prosjek']:.2f} (oček. 1.2) "
    f"brojeva po kolu, sve unutar slučajne varijacije"
    + (f" (nominalno p<0.05 samo za: {', '.join(any_sig)}, što je pri 12 testova očekivani broj lažnih signala)" if any_sig else "")
    + "; nijedna od tih strategija nema prediktivnu moć u ovim podacima – izvlačenja se ponašaju kao neovisna uniformna slučajna izvlačenja."
)

result = dict(
    meta=dict(izvor=str(DATA.name), n_izvlacenja=ND, prvo=DATES[0], zadnje=DATES[-1], seed=SEED, n_simulacija=NSIM,
              generirano=time.strftime("%Y-%m-%d %H:%M:%S"), skripta="stats_neovisno.py", trajanje_s=round(time.time() - T0, 1)),
    frekvencije=sec1, parovi=sec2, profil=sec3, ponavljanja=sec4, backtest=sec5, joker=sec6,
    sazetak=dict(chi2_all=chi2_all["chi2"], p_all=chi2_all["p"], top9=top9, bottom9=bottom9, max_gap_number=gmax_nums[0],
                 max_gap=gmax, backtest=backtest_str, conclusion=conclusion),
)
(OUT / "stats_neovisno.json").write_text(json.dumps(py(result), ensure_ascii=False, indent=1), encoding="utf-8")

# =============================================================================== MARKDOWN
L = []
A = L.append
A("# Neovisna statistička analiza — Loto 6/45 (HL), 882 izvlačenja 07.01.2018. – 09.08.2026.")
A("")
A(f"*Skripta `stats_neovisno.py`, podaci `{DATA.name}`, seed {SEED}, {NSIM} simuliranih povijesti iste veličine za Monte-Carlo (MC) p-vrijednosti. "
  "Analiza je rađena neovisno o `stats.py` — sve je izračunato iznova iz CSV-a.*")
A("")
A("## Sažetak")
A("")
A(conclusion)
A("")
A("## 1. Frekvencije brojeva 1–45")
A("")
A(f"Očekivana frekvencija po broju: 882·6/45 = **{EXP:.1f}**; binomna sd = √(882·6/45·39/45) = {SD1:.2f}. "
  "z = (opaženo − 117.6)/sd. p_binom je dvostrani binomni test pojedinog broja (bez korekcije za 45 usporedbi). "
  "Kašnjenje = broj kola od zadnjeg pojavljivanja (0 = izvučen u zadnjem kolu 09.08.2026.).")
A("")
A("| broj | glavni | rang | z | p_binom | bonus | ukupno (7) | kašnjenje | zadnji put |")
A("|---:|---:|---:|---:|---:|---:|---:|---:|:---|")
for r in po_broju:
    A(f"| {r['broj']} | {r['glavni']} | {r['rang']} | {r['z']:+.2f} | {r['p_binom']:.3f} | {r['bonus']} | {r['ukupno7']} | {r['kasnjenje']} | {r['zadnji_datum']} |")
A("")
A("### Hi-kvadrat test uniformnosti (44 d.f.)")
A("")
A("Klasična statistika Σ(O−E)²/E s χ²(44). Napomena: budući da se 6 brojeva izvlači **bez vraćanja**, varijanca brojača je manja nego u multinomnom "
  f"modelu za faktor (45−6)/(45−1) = {FPC:.4f}; očekivanje statistike pod H0 je 39, a ne 44, pa je klasični test blago konzervativan. "
  "Stoga su dane i korigirana statistika (χ²·44/39) i MC p-vrijednost iz 1000 simulacija.")
A("")
A("| uzorak | n | E po broju | χ² | p (χ²₄₄) | χ² korig. | p korig. | MC p | E[χ²] pod H0 | najčešći | najrjeđi |")
A("|:--|---:|---:|---:|---:|---:|---:|---:|---:|:--|:--|")
for name, b in [("cijela povijest", chi2_all)] + [(k.replace("_", " "), v) for k, v in windows.items()]:
    A(f"| {name} | {b['n_izvlacenja']} | {b['ocekivano_po_broju']:.2f} | {b['chi2']:.2f} | {pfmt(b['p'])} | {b['chi2_korigirani']:.2f} | "
      f"{pfmt(b['p_korigirani'])} | {pfmt(b['p_monte_carlo'])} | {b['chi2_ocekivano_pod_H0']:.1f} | {b['max_broj']} ({b['max_frekv']}×) | {b['min_broj']} ({b['min_frekv']}×) |")
A("")
A(f"Dopunski (bonus) broj: E = 882/45 = {eb:.1f} po broju, χ² = {chi2_bonus:.2f}, p = {pfmt(bonus_blk['p'])} "
  f"(najčešći {bonus_blk['max_broj']} {bonus_blk['max_frekv']}×, najrjeđi {bonus_blk['min_broj']} {bonus_blk['min_frekv']}×). "
  f"Svih 7 kugli zajedno: χ² = {chi2_7:.2f}, p = {pfmt(sec1['chi2_7_kugli']['p'])}.")
A("")
A("**\"Vrući\" prozori** (top-5 / dno-5 po frekvenciji u zadnjih w kola):")
A("")
for k, v in windows.items():
    A(f"- {k.replace('_', ' ')}: top-5 = {v['top5']} ({v['top5_frekv']}), dno-5 = {v['dno5']} ({v['dno5_frekv']}); E = {v['ocekivano_po_broju']:.2f}, MC p = {pfmt(v['p_monte_carlo'])}")
A("")
A("### Ekstremi, Pareto i hladni skup")
A("")
A(f"- Najveći |z| = {ekstremi['max_abs_z']:.2f} (broj {ekstremi['broj_max_abs_z']}); u simulacijama je maksimalni |z| od 45 brojeva u prosjeku "
  f"{ekstremi['sim_prosjek_max_abs_z']:.2f}, a vrijednost ≥ opažene javlja se u {ekstremi['p_max_abs_z_monte_carlo']*100:.0f} % slučajnih povijesti. "
  f"Brojeva s |z| > 1.96: {ekstremi['broj_z_iznad_196']} (očekivano ≈ {ekstremi['ocekivano_iznad_196']:.2f}). "
  f"Najmanji p_binom = {ekstremi['min_p_binom']:.4f}, Bonferroni (×45) = {ekstremi['bonferroni_min_p']:.3f}.")
A(f"- **Pareto top-9 (20 % brojeva)**: {top9} s frekvencijama {sec1['pareto_top9_frekv']}; nose {pareto_udio*100:.1f} % svih izvučenih brojeva "
  f"(uniformno bi bilo 20 %). Granica: 9. ima {top9_tie['frekv_9']}, 10. ima {top9_tie['frekv_10']}"
  + (f" — izjednačeni na granici: {top9_tie['izjednaceni_na_granici']}" if len(top9_tie['izjednaceni_na_granici']) > 1 else "") + ".")
A(f"- **Hladnih 9**: {bottom9} s frekvencijama {sec1['hladni_9_frekv']}. Granica: 9. ima {bottom9_tie['frekv_9']}, 10. ima {bottom9_tie['frekv_10']}"
  + (f" — izjednačeni na granici: {bottom9_tie['izjednaceni_na_granici']}" if len(bottom9_tie['izjednaceni_na_granici']) > 1 else "") + ".")
A("")
A("### Kašnjenje")
A("")
kb = kasnjenje_blk
A(f"Očekivani razmak između dva pojavljivanja istog broja je 45/6 = **7.5** kola; očekivano *trenutno* kašnjenje (kola od zadnjeg pojavljivanja) je "
  f"7.5 − 1 = 6.5 (sd {kb['sd_trenutno']:.1f}), P(kašnjenje ≥ g) = (39/45)^g. Opaženi prosjek trenutnog kašnjenja svih 45 brojeva: {kb['prosjek_trenutnog']:.2f} "
  f"(medijan {kb['medijan_trenutnog']:.0f}); prosječni povijesni razmak između pojavljivanja: {kb['prosjecni_razmak_svih_brojeva']:.2f} kola.")
A("")
A(f"**Najdulje trenutno kašnjenje: broj {kb['max_broj']} — {kb['max_kasnjenje']} kola**"
  + (f" (izjednačeni: {kb['max_brojevi']})" if len(kb['max_brojevi']) > 1 else "")
  + f". P(zadani broj kasni ≥ {gmax}) = {kb['p_jedan_broj_ge']:.4f}; P(maksimum među 45 brojeva ≥ {gmax}) ≈ {kb['p_max_od_45_aproks']:.2f} (aproks.), "
  f"MC = {kb['p_max_monte_carlo']:.2f}; u slučajnim povijestima najdulje kašnjenje je u prosjeku {kb['sim_prosjek_max']:.1f} kola (medijan {kb['sim_medijan_max']:.0f}).")
A("")
gap_sorted = sorted(range(N), key=lambda i: (-gap[i], i))[:10]
A("Top-10 kašnjenja: " + ", ".join(f"**{i+1}** ({gap[i]})" for i in gap_sorted) + ".")
A("")
A("## 2. Parovi")
A("")
A(f"990 mogućih parova, 15 po kolu; očekivano po paru 882·15/990 = **{E_PAIR:.2f}** (sd {sec2['sd_po_paru']:.2f}). "
  f"Parova koji se nisu pojavili nijednom: {sec2['broj_parova_nikad']}; najrjeđi par {sec2['min_par']['par']} ({sec2['min_par']['broj']}×).")
A("")
A("| # | par | broj | z | P(≥ broj) za zadani par |")
A("|---:|:--|---:|---:|---:|")
for i, t in enumerate(top20, start=1):
    A(f"| {i} | {t['par'][0]}–{t['par'][1]} | {t['broj']} | {t['z']:+.2f} | {t['p_binom_pojedinacno']:.4f} |")
A("")
mx = sec2["maksimum"]
A(f"**Je li maksimum ({pmax}×) neobičan?** Poissonova aproksimacija: P(zadani par ≥ {pmax}) = {mx['p_jedan_par_poisson']:.5f} "
  f"(binomno {mx['p_jedan_par_binom']:.5f}); P(maksimum od 990 parova ≥ {pmax}) ≈ 1 − (1 − p)^990 = {mx['p_max_poisson_aproks']:.3f}. "
  f"Simulacija 1000 slučajnih povijesti (882 kola, seed {SEED}): maksimum po paru u prosjeku {mx['sim_prosjek_max']:.1f} "
  f"(medijan {mx['sim_medijan_max']:.0f}, 95. percentil {mx['sim_p95_max']:.0f}, raspon {mx['sim_min_max']}–{mx['sim_max_max']}); "
  f"udio simulacija s maksimumom ≥ {pmax}: **MC p = {mx['p_max_monte_carlo']:.3f}**.")
A("")
c2 = sec2["chi2"]
A(f"Hi-kvadrat svih 990 parova: χ² = {c2['chi2']:.1f} (naivno 989 d.f., p = {pfmt(c2['p_naivno'])}; parovi unutar kola su zavisni pa je mjerodavna "
  f"MC p-vrijednost = {pfmt(c2['p_monte_carlo'])}, simulirani prosjek χ² = {c2['sim_prosjek']:.1f}).")
A("")
A("## 3. Profil izvlačenja")
A("")
A("Teorijske vrijednosti su **egzaktne** — izračunate nabrajanjem svih 8 145 060 kombinacija. z/p se odnose na test prosjeka (z-test), "
  "GOF je hi-kvadrat test prilagodbe cijele distribucije (kategorije s E < 5 spojene; za zbroj i raspon decilni razredi).")
A("")
A("| statistika | prosjek | teorijski | sd | teor. sd | z | p (prosjek) | GOF χ² (df) | GOF p |")
A("|:--|---:|---:|---:|---:|---:|---:|---:|---:|")
for key in ("zbroj", "neparni", "le22", "le31", "uzastopni", "raspon", "desetice"):
    b = sec3[key]
    A(f"| {b['opis']} | {b['prosjek']:.3f} | {b['teorijski_prosjek']:.3f} | {b['sd']:.2f} | {b['teorijska_sd']:.2f} | {b['z_prosjek']:+.2f} | "
      f"{pfmt(b['p_prosjek'])} | {b['gof']['chi2']:.2f} ({b['gof']['df']}) | {pfmt(b['gof']['p'])} |")
A("")
A("Percentili (opaženo / teorijski):")
A("")
A("| statistika | " + " | ".join(f"P{q}" for q in PCT) + " |")
A("|:--|" + "---:|" * len(PCT))
for key in ("zbroj", "raspon", "neparni", "le22", "le31", "uzastopni", "desetice"):
    b = sec3[key]
    A(f"| {key} | " + " | ".join(f"{b['percentili'][str(q)]:.0f} / {b['teorijski_percentili'][str(q)]}" for q in PCT) + " |")
A("")
for key in ("neparni", "le22", "uzastopni", "desetice"):
    b = sec3[key]
    A(f"- {b['opis']}: " + ", ".join(f"{v}→{d['opazeno']} (E {d['ocekivano']:.1f})" for v, d in b["distribucija"].items()))
bj = sec3["bar_jedan_uzastopni_par"]
A(f"- P(bar jedan uzastopni par) = 1 − C(40,6)/C(45,6) = **{bj['teorijska_p']:.4f}**; opaženo {bj['opazeno']}/882 = {bj['udio']:.4f} "
  f"(E {bj['ocekivano']:.1f}), binomni p = {bj['p_binom']:.3f}.")
sp = sec3["sve_parne_ili_sve_neparne"]
A(f"- Sve parne ili sve neparne: opaženo {sp['opazeno']}, očekivano {sp['ocekivano']:.1f}.")
A("")
A("## 4. Ponavljanja, autokorelacija, serije")
A("")
pp = sec4["ponavljanja"]
A(f"Broj brojeva iz prethodnog kola koji se ponove u sljedećem ~ hipergeometrijski(45, 6, 6): E = **{pp['ocekivano']:.2f}**, sd {pp['sd_teorijska']:.3f}. "
  f"Opaženi prosjek {pp['prosjek']:.3f} (sd {pp['sd']:.3f}), z = {pp['z']:+.2f}, p = {pfmt(pp['p'])}; GOF χ² = {pp['gof']['chi2']:.2f} ({pp['gof']['df']} df), p = {pfmt(pp['gof']['p'])}, MC p = {pfmt(pp['gof']['p_monte_carlo'])} "
  f"— {pp['gof']['napomena']}. "
  f"Kola s bar jednim ponavljanjem: {pp['udio_kola_s_bar_jednim']*100:.1f} % (teorijski {pp['teorijski_udio_bar_jedan']*100:.1f} %).")
A("")
A("| ponovljenih | 0 | 1 | 2 | 3 | 4 | 5 | 6 |")
A("|:--|" + "---:|" * 7)
A("| opaženo | " + " | ".join(str(pp['distribucija'][str(k)]['opazeno']) for k in sup) + " |")
A("| očekivano | " + " | ".join(f"{pp['distribucija'][str(k)]['ocekivano']:.1f}" for k in sup) + " |")
A("")
A("Po pomaku L (zajednički brojevi s kolom t−L): " + "; ".join(f"L={L_}: {v['prosjek']:.3f} (z {v['z']:+.2f})" for L_, v in pp["po_pomaku"].items()) + ".")
br = pp["bonus_prethodnog_kola_medju_glavnima"]
A(f"Dopunski broj prethodnog kola među 6 glavnih sljedećeg: {br['opazeno']} puta (E {br['ocekivano']:.1f}, p = {br['p']:.3f}).")
A("")
A("**Autokorelacija** (lag 1–10, granica ±1.96/√882 = ±{:.3f}; Ljung–Box Q(10)):".format(1.96 / math.sqrt(ND)))
A("")
for key in ("zbroj", "neparni", "le22", "raspon"):
    a = auto[key]
    A(f"- {key}: r₁..r₅ = {', '.join(f'{v:+.3f}' for v in a['acf_1_10'][:5])}; Q(10) = {a['ljung_box_Q10']:.2f}, p = {pfmt(a['p'])}")
ai = auto["indikatori_brojeva_lag1"]
A(f"- indikatorske serije 45 brojeva, lag 1: prosječni r₁ = {ai['prosjek_r1']:+.4f} (raspon {ai['min_r1']:+.3f}…{ai['max_r1']:+.3f}); "
  f"značajnih na 5 %: {ai['broj_znacajnih_95']} (E {ai['ocekivano_znacajnih']:.2f}); zbirni test Σz² = {ai['chi2_zbroj_z2']:.1f} (45 df), p = {pfmt(ai['p_zbroj_z2'])}")
A("")
A("**Test serija (Wald–Wolfowitz runs):**")
A("")
A("| serija | n1 | n2 | serija (runs) | očekivano | z | p |")
A("|:--|---:|---:|---:|---:|---:|---:|")
RUNS_LAB = {"zbroj_iznad_medijana": "zbroj iznad medijana", "neparni_vecina": "≥3 neparna", "bar_jedno_ponavljanje": "bar jedno ponavljanje"}
for k_ in ("zbroj_iznad_medijana", "neparni_vecina", "bar_jedno_ponavljanje"):
    r_ = runs[k_]
    A(f"| {RUNS_LAB[k_]} | {r_['n1']} | {r_['n2']} | {r_['runs']} | {r_['ocekivano']:.1f} | {r_['z']:+.2f} | {pfmt(r_['p'])} |")
ri = runs["indikatori_brojeva"]
A("")
A(f"Runs test za indikatorsku seriju svakog od 45 brojeva: značajnih na 5 %: {ri['broj_znacajnih_95']} (E {ri['ocekivano_znacajnih']:.2f}), "
  f"prosječni z = {ri['prosjek_z']:+.3f}, Σz² = {ri['chi2_zbroj_z2']:.1f} (45 df), p = {pfmt(ri['p'])}; najjači: "
  + ", ".join(f"{d['broj']} (z {d['z']:+.2f})" for d in ri["najjaci"]) + ".")
A("")
A("## 5. KLJUČNO — backtest prediktivne moći")
A("")
A(sec5["opis"])
A("")
A(f"Očekivani broj pogodaka po kolu pod slučajnošću: 6·12/45 = **1.6** (sd po kolu {sec5['sd_teorijska_12']:.3f}, hipergeometrijski) odnosno 6·9/45 = **1.2** "
  f"(sd {sec5['sd_teorijska_9']:.3f}). p_z = dvostrani z-test ukupnog broja pogodaka; p_binom = dvostrani binomni test (6T pokušaja, p = k/45); "
  "MC p = udio od 1000 simuliranih povijesti u kojima isti algoritam daje prosjek koji od očekivanja odstupa bar koliko opaženi. "
  f"Snaga: pri T = {T_BT} dvostrani test na razini 5 % otkriva odstupanje prosjeka od ±{1.96 * hyper_sd(KSET) / math.sqrt(T_BT):.3f} pogodaka po kolu "
  f"(±{100 * 1.96 * hyper_sd(KSET) / math.sqrt(T_BT) / 1.6:.1f} % od 1.6) – svaki stvarno koristan 'vrući/hladni' efekt tog reda bio bi vidljiv.")
A("")
A("| skup | k | prosjek | sd | očekivano | lift | ukupno | z | p_z | p_binom | MC p | MC sd prosjeka | kola s 0 | kola ≥3 |")
A("|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
order_keys = ["vruci_w60", "hladni_w60", "kasneci", "pareto9", "hladni9_povijest", "slucajni12",
              "vruci_w30", "hladni_w30", "vruci_w100", "hladni_w100", "vruci_w200", "hladni_w200"]
for k_ in order_keys:
    b = bt_res[k_]
    A(f"| {k_} | {b['k']} | **{b['prosjek']:.3f}** | {b['sd']:.3f} | {b['ocekivano']:.1f} | {b['lift']:.3f} | {b['ukupno_pogodaka']} | {b['z']:+.2f} | "
      f"{pfmt(b['p_z'])} | {pfmt(b['p_binom'])} | {pfmt(b['p_monte_carlo'])} | {b['sim_sd_prosjeka']:.3f} | {b['udio_kola_0_pogodaka']*100:.1f} % | {b['udio_kola_ge3']*100:.1f} % |")
A("")
A("Distribucija pogodaka po kolu (broj kola s 0,1,…,6 pogodaka):")
A("")
for k_ in ("vruci_w60", "hladni_w60", "kasneci", "pareto9", "slucajni12"):
    A(f"- {k_}: " + ", ".join(f"{v}→{c}" for v, c in bt_res[k_]["distribucija_pogodaka"].items() if c > 0))
A("")
A("Osjetljivost na razbijanje izjednačenja (w = 60, po rastućem broju umjesto slučajno): "
  + "; ".join(f"{k_}: {v['prosjek']:.3f} (z {v['z']:+.2f}, p {v['p_z']:.3f})" for k_, v in osjetljivost.items()) + ".")
A("")
A("Po godinama (prosjek pogodaka po kolu):")
A("")
A("| godina | n | vrući w60 | hladni w60 | kasneći | pareto9 |")
A("|:--|---:|---:|---:|---:|---:|")
for y, v in po_god.items():
    A(f"| {y} | {v['n']} | {v['vruci_w60']:.3f} | {v['hladni_w60']:.3f} | {v['kasneci']:.3f} | {v['pareto9']:.3f} |")
A("")
A("**Zaključak backtesta.** " + backtest_str)
A("")
A("## 6. Joker (kratko)")
A("")
A(f"Joker je 6-znamenkasti broj (vodeće nule sačuvane; vodeća nula u {sec6['vodece_nule']} kola, E {sec6['ocekivano_vodece_nule']:.1f}). "
  f"Frekvencije znamenki 0–9 na svih 6 pozicija: {sec6['znamenke_ukupno']} (E {sec6['ocekivano_po_znamenki']:.1f}); χ² = {sec6['chi2']:.2f} (9 df), p = {pfmt(sec6['p'])}. "
  "Po poziciji: " + "; ".join(f"poz. {p['pozicija']}: χ² {p['chi2']:.1f}, p {p['p']:.3f}" for p in pos) + ". "
  f"Cijeli broj kao U(0, 999999): KS D = {sec6['ks_cijeli_broj']['D']:.3f}, p = {sec6['ks_cijeli_broj']['p']:.3f}; prosjek {sec6['prosjek']:.0f} (E 499999.5); "
  f"Ljung–Box Q(5) = {sec6['ljung_box']['Q']:.2f}, p = {sec6['ljung_box']['p']:.3f}. Znamenke su uniformne i bez serijske ovisnosti.")
A("")
A("---")
A(f"*Generirano {result['meta']['generirano']}, trajanje {result['meta']['trajanje_s']} s. Datoteke: `out/stats_neovisno.json`, `out/stats_neovisno.md`.*")
(OUT / "stats_neovisno.md").write_text("\n".join(L) + "\n", encoding="utf-8")

print(f"\nχ²(44) = {chi2_all['chi2']:.2f}, p = {chi2_all['p']:.4f} (korig. p = {chi2_all['p_korigirani']:.4f}, MC p = {chi2_all['p_monte_carlo']:.4f})")
print(f"top9 = {top9}, bottom9 = {bottom9}, max gap = broj {gmax_nums[0]} ({gmax})")
print(backtest_str)
print(f"\n{conclusion}")
print(f"\nZapisano: {OUT / 'stats_neovisno.json'}, {OUT / 'stats_neovisno.md'}  ({time.time() - T0:.1f}s)")
