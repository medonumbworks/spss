"""Sveobuhvatna analiza povijesti igre Sve ili Ništa (11/22). Ispis JSON + čitljiv sažetak."""
import os
import sys, json, math, itertools
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sin_lib as L

DATA = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else 'analysis.json'
draws = L.load_draws(DATA)
T = len(draws)
SD_HITS = math.sqrt(11 * (11 / 22) * (11 / 22) * (11 / 21))  # 1.2002
M = np.zeros((T, 23), dtype=np.int8)
for i, d in enumerate(draws):
    M[i, d['nums']] = 1
M = M[:, 1:]  # stupci 0..21 ~ brojevi 1..22
res = {'n_draws': T, 'first': str(draws[0]['datum']), 'last': str(draws[-1]['datum']), 'last_kolo': f"{draws[-1]['kolo']}/{draws[-1]['godina']}"}

# 1. frekvencija + chi2
freq = M.sum(0)
exp = T * 11 / 22
sd = math.sqrt(T * 0.5 * 0.5 * (11 / 22) * (11 / 22) / (11 / 22) ) if False else math.sqrt(T * 0.25)  # Bernoulli(0.5) po kolu
z = (freq - exp) / sd
chi2 = float(((freq - exp) ** 2 / exp).sum())
# chi2 za 11-od-22 bez vraćanja: varijanca po ćeliji je T*p(1-p)*(22-11)/(22-1) → korekcija faktorom 21/11
chi2_adj = chi2 * 21 / 11
res['frequency'] = {int(n + 1): int(freq[n]) for n in range(22)}
res['z'] = {int(n + 1): round(float(z[n]), 2) for n in range(22)}
res['chi2'] = dict(expected_per_number=exp, chi2_naive=round(chi2, 2), p_naive=round(L.chi2_sf(chi2, 21), 4),
                  chi2_finite_population_adjusted=round(chi2_adj, 2), p_adjusted=round(L.chi2_sf(chi2_adj, 21), 4),
                  note='naive assumes independent Bernoulli(0.5) cells; adjusted divides by the finite-population factor (22-11)/(22-1)')
order = np.argsort(-freq)
res['hot_all'] = [int(n + 1) for n in order[:11]]
res['cold_all'] = [int(n + 1) for n in order[-11:][::-1]]
res['max_abs_z'] = round(float(np.abs(z).max()), 2)
# Monte Carlo za max|z| pod nultom hipotezom (uniformno 11/22)
rng = np.random.default_rng(12345)
def sim_hist(T):
    out = np.zeros((T, 22), dtype=np.int8)
    for i in range(T):
        out[i, rng.choice(22, 11, replace=False)] = 1
    return out
mc_maxz = []
for _ in range(1000):
    f = sim_hist(T).sum(0)
    mc_maxz.append(np.abs((f - exp) / sd).max())
mc_maxz = np.array(mc_maxz)
res['mc_p_max_abs_z'] = round(float((mc_maxz >= np.abs(z).max()).mean()), 3)

# 2. prozori
for w in (30, 60, 90, 180):
    fw = M[-w:].sum(0)
    o = np.argsort(-fw)
    res[f'window_{w}'] = dict(freq={int(n + 1): int(fw[n]) for n in range(22)}, hot=[int(n + 1) for n in o[:11]], cold=[int(n + 1) for n in o[-11:][::-1]])

# 3. hot-hand persistence: top-11 by freq in [t-w, t) vs draw t
def persistence(w, mode='hot'):
    hits = []
    for t in range(w, T):
        fw = M[t - w:t].sum(0)
        key = -fw if mode == 'hot' else fw
        sel = np.lexsort((np.arange(22), key))[:11]
        hits.append(int(M[t, sel].sum()))
    hits = np.array(hits)
    se = SD_HITS / math.sqrt(len(hits))  # sd hipergeometrijske(22,11,11) = 1.2002
    zz = (hits.mean() - 5.5) / se
    return dict(n=len(hits), mean_hits=round(float(hits.mean()), 3), z=round(float(zz), 2), p_two_sided=round(2 * (1 - 0.5 * (1 + math.erf(abs(zz) / math.sqrt(2)))), 3),
                extreme_frac=round(float(((hits <= 3) | (hits >= 8)).mean()), 4), jackpots=int(((hits == 0) | (hits == 11)).sum()))
res['persistence'] = {f'{mode}_{w}': persistence(w, mode) for w in (30, 60, 90) for mode in ('hot', 'cold')}

# 4. razmaci (gaps) + uvjetna vjerojatnost pojave s obzirom na trenutni razmak
gaps_now = L.gaps(draws)
res['gaps_now'] = gaps_now
res['overdue'] = [n for n, g in sorted(gaps_now.items(), key=lambda kv: (-kv[1], kv[0]))][:11]
cond = defaultdict(lambda: [0, 0])  # gap -> [pojavio, ukupno]
for n in range(22):
    last = None
    for t in range(T):
        if last is not None:
            g = t - 1 - last
            gb = min(g, 4)
            cond[gb][1] += 1
            if M[t, n]: cond[gb][0] += 1
        if M[t, n]: last = t
res['cond_by_gap'] = {('%d' % g if g < 4 else '4+'): dict(appear=c[0], total=c[1], rate=round(c[0] / c[1], 4)) for g, c in sorted(cond.items())}
# chi2 2x5
tab = np.array([[c[0], c[1] - c[0]] for g, c in sorted(cond.items())], dtype=float)
rowsum = tab.sum(1, keepdims=True); colsum = tab.sum(0, keepdims=True); tot = tab.sum()
E = rowsum @ colsum / tot
chi_g = float(((tab - E) ** 2 / E).sum())
res['cond_by_gap_chi2'] = dict(chi2=round(chi_g, 2), df=4, p=round(L.chi2_sf(chi_g, 4), 4))

# 5. preklapanje uzastopnih kola vs hipergeometrijska
def hyper_pmf(k, N=22, K=11, n=11):
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)
ov = (M[1:] * M[:-1]).sum(1)
obs = Counter(int(x) for x in ov)
res['overlap_lag1'] = dict(mean=round(float(ov.mean()), 3), expected=5.5, se=round(SD_HITS / math.sqrt(len(ov)), 3),
                          dist={k: dict(obs=obs.get(k, 0), exp=round(hyper_pmf(k) * len(ov), 1)) for k in range(12)})
# lag-1 Markov pooled
a = int((M[1:] * M[:-1]).sum()); b = int((M[:-1] * (1 - M[1:])).sum()); c = int(((1 - M[:-1]) * M[1:]).sum()); d = int(((1 - M[:-1]) * (1 - M[1:])).sum())
p_in_given_in = a / (a + b); p_in_given_out = c / (c + d)
res['markov_lag1'] = dict(p_in_given_in=round(p_in_given_in, 4), p_in_given_out=round(p_in_given_out, 4), table=[[a, b], [c, d]])
# autokorelacija po broju, lag 1..7, prosjek
ac = {}
for lag in range(1, 8):
    cors = []
    for n in range(22):
        x = M[:-lag, n].astype(float); y = M[lag:, n].astype(float)
        if x.std() > 0 and y.std() > 0:
            cors.append(float(np.corrcoef(x, y)[0, 1]))
    cors = np.array(cors)
    ac[lag] = dict(mean_r=round(float(cors.mean()), 4), n_sig=int((np.abs(cors) > 1.96 / math.sqrt(T - lag)).sum()), expected_sig=round(22 * 0.05, 1))
res['autocorr'] = ac

# 6. struktura: egzaktna distribucija svih C(22,11) kombinacija
allc = np.array(list(itertools.combinations(range(1, 23), 11)), dtype=np.int8)
sums = allc.sum(1); evens = (allc % 2 == 0).sum(1); lows = (allc <= 11).sum(1)
consec = (np.diff(allc, axis=1) == 1).sum(1)
obs_sum = np.array([sum(d['nums']) for d in draws]); obs_even = np.array([sum(1 for n in d['nums'] if n % 2 == 0) for d in draws])
obs_low = np.array([sum(1 for n in d['nums'] if n <= 11) for d in draws]); obs_consec = np.array([sum(1 for a_, b_ in zip(d['nums'], d['nums'][1:]) if b_ == a_ + 1) for d in draws])
def compare(theo, obs, name, bins=None):
    cnt_t = Counter(theo.tolist()); cnt_o = Counter(obs.tolist())
    total = len(theo)
    out = dict(theo_mean=round(float(theo.mean()), 3), theo_sd=round(float(theo.std()), 3), obs_mean=round(float(obs.mean()), 3), obs_sd=round(float(obs.std()), 3))
    out['z_mean'] = round(float((obs.mean() - theo.mean()) / (theo.std() / math.sqrt(len(obs)))), 2)
    keys = sorted(cnt_t)
    if bins is None: bins = [[k] for k in keys]
    chi = 0.0; df = -1; rows = []
    for bgrp in bins:
        e = sum(cnt_t[k] for k in bgrp) / total * len(obs); o = sum(cnt_o.get(k, 0) for k in bgrp)
        if e < 1: continue
        chi += (o - e) ** 2 / e; df += 1; rows.append((bgrp[0], bgrp[-1], o, round(e, 1)))
    out['chi2'] = round(chi, 2); out['df'] = df; out['p'] = round(L.chi2_sf(chi, df), 4) if df > 0 else None
    out['bins'] = rows
    return out
sum_bins = [list(range(a_, a_ + 10)) for a_ in range(66, 190, 10)]
res['structure'] = dict(
    sum=compare(sums, obs_sum, 'sum', sum_bins),
    even=compare(evens, obs_even, 'even', [[k] for k in range(0, 12)]),
    low=compare(lows, obs_low, 'low', [[k] for k in range(0, 12)]),
    consecutive_pairs=compare(consec, obs_consec, 'consec', [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]),
)
# najčešći raspon zbroja egzaktno (središnjih 50%)
q = np.percentile(sums, [25, 50, 75]); res['structure']['sum_quartiles_exact'] = [float(x) for x in q]
res['structure']['even_dist_exact'] = {int(k): round(v / len(evens), 4) for k, v in sorted(Counter(evens.tolist()).items())}
# positional
pos_theo = allc.mean(0); pos_obs = np.array([d['nums'] for d in draws]).mean(0)
res['positional'] = dict(theo=[round(float(x), 2) for x in pos_theo], obs=[round(float(x), 2) for x in pos_obs])

# 7. parovi
pc = L.pair_counts(draws)
exp_pair = T * 55 / 231
sd_pair = math.sqrt(T * (55 / 231) * (1 - 55 / 231))
pz = {k: (v - exp_pair) / sd_pair for k, v in pc.items()}
top = sorted(pz.items(), key=lambda kv: -kv[1])[:5]; bot = sorted(pz.items(), key=lambda kv: kv[1])[:5]
res['pairs'] = dict(expected=round(exp_pair, 2), top=[(list(k), pc[k], round(v, 2)) for k, v in top], bottom=[(list(k), pc[k], round(v, 2)) for k, v in bot])
# MC za max |z| parova
mc_pair = []
for _ in range(300):
    S = sim_hist(T).astype(np.int32)
    C = S.T @ S
    iu = np.triu_indices(22, 1)
    mc_pair.append(np.abs((C[iu] - exp_pair) / sd_pair).max())
mc_pair = np.array(mc_pair)
obs_max_pair = max(abs(v) for v in pz.values())
res['pairs']['max_abs_z'] = round(obs_max_pair, 2); res['pairs']['mc_p_max_abs_z'] = round(float((mc_pair >= obs_max_pair).mean()), 3); res['pairs']['mc_mean_max_abs_z'] = round(float(mc_pair.mean()), 2)

# 8. dan u tjednu
days = ['Pon', 'Uto', 'Sri', 'Čet', 'Pet', 'Sub', 'Ned']
tab = np.zeros((22, 7))
for i, d in enumerate(draws):
    j = days.index(d['dan']); tab[:, j] += M[i]
rowsum = tab.sum(1, keepdims=True); colsum = tab.sum(0, keepdims=True); tot = tab.sum()
E = rowsum @ colsum / tot
chi_d = float(((tab - E) ** 2 / E).sum())
res['weekday'] = dict(chi2=round(chi_d, 2), df=126, p=round(L.chi2_sf(chi_d, 126), 4), chi2_adj=round(chi_d * 21 / 11, 2), p_adj=round(L.chi2_sf(chi_d * 21 / 11, 126), 4), draws_per_day={days[j]: int(colsum[0, j] / 11) for j in range(7)})
# nedjelja posebno (ciljno kolo 291 je u nedjelju)
sun = tab[:, 6]; nsun = int(colsum[0, 6] / 11)
res['sunday'] = dict(n=nsun, freq={int(n + 1): int(sun[n]) for n in range(22)}, hot=[int(n + 1) for n in np.argsort(-sun)[:11]])
rest = tab.sum(1) - sun
t2 = np.stack([sun, rest], 1); E2 = t2.sum(1, keepdims=True) @ t2.sum(0, keepdims=True) / t2.sum()
chi_s = float(((t2 - E2) ** 2 / E2).sum()) * 21 / 11
res['sunday']['chi2_adj_vs_rest'] = round(chi_s, 2); res['sunday']['p_adj_vs_rest'] = round(L.chi2_sf(chi_s, 21), 4)

# 9. ponavljanja
seen = Counter(tuple(d['nums']) for d in draws)
res['exact_repeats'] = int(sum(1 for v in seen.values() if v > 1))
res['expected_exact_repeats'] = round(math.comb(T, 2) / 705432, 3)
Mi = M.astype(np.int32); G = Mi @ Mi.T; np.fill_diagonal(G, -1)
res['max_overlap_any_pair_of_draws'] = int(G.max()); res['pairs_of_draws_with_10_overlap'] = int((G == 10).sum() // 2)
res['expected_pairs_with_10_overlap'] = round(math.comb(T, 2) * hyper_pmf(10), 2)

# 10. matematika igre
PRICE = 1.30
PRIZES = {'11_ili_0': (80000.0, (0, 11)), '10_ili_1': (260.0, (1, 10)), '9_ili_2': (26.0, (2, 9))}  # sekundarni dobici prema sekundarnim izvorima (geek.hr, hrk.hr) – provjeriti u pravilima HL
tiers = {}
for name, (amount, ks) in PRIZES.items():
    pk = sum(hyper_pmf(k) for k in ks)
    tiers[name] = dict(iznos=amount, pogodaka=list(ks), kombinacija=sum(math.comb(11, k) * math.comb(11, 11 - k) for k in ks), p=pk, odds=f'1 : {1/pk:,.0f}', ev=round(pk * amount, 4))
ev_total = sum(t['ev'] for t in tiers.values())
res['game_math'] = dict(total_combos=705432, p_jackpot_per_ticket=2 / 705432, odds='1 : 352716', mean_hits=5.5, sd_hits=round(SD_HITS, 4),
                        p_hits={k: round(hyper_pmf(k), 6) for k in range(12)}, p_extreme_le3_or_ge8=round(sum(hyper_pmf(k) for k in (0, 1, 2, 3, 8, 9, 10, 11)), 5),
                        price=PRICE, tiers=tiers, p_any_prize=sum(t['p'] for t in tiers.values()), ev_total=round(ev_total, 4), return_pct=round(100 * ev_total / PRICE, 1),
                        ev_jackpot_only=round(tiers['11_ili_0']['ev'], 4), return_pct_jackpot_only=round(100 * tiers['11_ili_0']['ev'] / PRICE, 1))

json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: res[k] for k in ['n_draws', 'first', 'last', 'last_kolo', 'chi2', 'max_abs_z', 'mc_p_max_abs_z', 'hot_all', 'cold_all', 'overdue', 'persistence', 'cond_by_gap', 'cond_by_gap_chi2', 'markov_lag1', 'autocorr', 'weekday', 'exact_repeats', 'expected_exact_repeats', 'max_overlap_any_pair_of_draws', 'pairs_of_draws_with_10_overlap', 'expected_pairs_with_10_overlap']}, ensure_ascii=False, indent=1))
print('overlap', res['overlap_lag1']['mean'], res['overlap_lag1']['se'])
print('pairs', res['pairs'])
print('structure sum', {k: v for k, v in res['structure']['sum'].items() if k != 'bins'}, res['structure']['sum_quartiles_exact'])
print('structure even', {k: v for k, v in res['structure']['even'].items() if k != 'bins'}, res['structure']['even_dist_exact'])
print('structure low', {k: v for k, v in res['structure']['low'].items() if k != 'bins'})
print('structure consec', {k: v for k, v in res['structure']['consecutive_pairs'].items() if k != 'bins'})
print('positional', res['positional'])
print('windows hot30', res['window_30']['hot'], 'cold30', res['window_30']['cold'])
print('sunday hot', res['sunday']['hot'], res['sunday']['n'])
print('game', res['game_math'])
