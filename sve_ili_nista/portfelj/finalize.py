"""Konačna provjera 3 novih listića i egzaktne metrike portfelja 13 (neovisno od portfolio.py: vlastiti kod za nabrajanje i isplate)."""
import json, itertools, math
import numpy as np
import os
W = os.environ.get('SIN_PORTFOLIO', os.path.dirname(os.path.abspath(__file__)))
EX = json.load(open(f'{W}/existing10.json')); S = json.load(open(f'{W}/select_disjoint.json'))
T10 = [tuple(t) for t in EX['tickets']]; NEW = [tuple(c['ticket']) for c in S['chosen']]
PRICE = 1.30; PAY = {0: 80000, 11: 80000, 1: 260, 10: 260, 2: 26, 9: 26}
checks = []
# 1) valjanost
for t in NEW:
    assert len(t) == 11 and len(set(t)) == 11 and all(isinstance(n, int) and 1 <= n <= 22 for n in t) and list(t) == sorted(t)
checks.append('Svaki od 3 nova listića ima točno 11 različitih cijelih brojeva od 1 do 22, uzlazno sortiranih.')
comp = lambda t: tuple(n for n in range(1, 23) if n not in t)
allT = T10 + NEW
assert len(set(allT)) == 13 and not any(comp(a) in set(allT) for a in allT)
checks.append('Nijedan novi listić nije duplikat ni komplement nijednog od 10 postojećih ni drugog novog listića (13 različitih, bez komplemenata).')
# 2) egzaktna evaluacija: sva izvlačenja kao skupovi (neovisan put: bez bit-maski, preko matrice incidencije)
combos = np.array(list(itertools.combinations(range(1, 23), 11)), dtype=np.int8)            # 705432 × 11
inc = np.zeros((len(combos), 23), dtype=np.int8); np.put_along_axis(inc, combos, 1, axis=1)   # incidencija 1..22
assert len(combos) == math.comb(22, 11)
def hits(t): return inc[:, list(t)].sum(1)
H = np.stack([hits(t) for t in allT])                                                          # 13 × N
pay_lut = np.array([PAY.get(h, 0) for h in range(12)], dtype=np.int64)
pay = pay_lut[H]; tot = pay.sum(0)
win = pay > 0; hi = pay >= 260; jp = pay >= 80000
def metr(rows, label):
    w, h_, j_, pm = win[rows], hi[rows], jp[rows], pay[rows]
    t_ = pm.sum(0); vals, cnts = np.unique(t_, return_counts=True); k = len(rows)
    ov = np.array([[len(set(allT[a]) & set(allT[b])) for b in rows] for a in rows])
    return dict(label=label, n_tickets=k, tickets=[list(allT[i]) for i in rows], union_all=int(w.any(0).sum()), union_hi=int(h_.any(0).sum()), union_jp=int(j_.any(0).sum()),
                p_any_prize=float(w.any(0).mean()), p_hi_prize=float(h_.any(0).mean()), p_jackpot=float(j_.any(0).mean()),
                ev_total_eur=float(t_.mean()), cost_eur=k * PRICE, ev_net_eur=float(t_.mean() - k * PRICE),
                payout_distribution=[dict(payout_eur=int(v), n_draws=int(c), p=float(c / len(combos))) for v, c in zip(vals, cnts)],
                pairwise_overlap=ov.tolist(), mean_pairwise_overlap=float(ov[np.triu_indices(k, 1)].mean()), p_two_or_more_prizes=float((w.sum(0) >= 2).mean()),
                redundancy_all=int(k * 6294 - w.any(0).sum()))
base = metr(list(range(10)), 'izvornih 10'); opt = metr(list(range(13)), 'optimiziranih 13')
# 3) per-ticket sanity
per = [int(win[i].sum()) for i in range(13)]; assert all(x == 6294 for x in per); assert all(int(hi[i].sum()) == 244 for i in range(13)); assert all(int(jp[i].sum()) == 2 for i in range(13))
checks.append('Svaki pojedini listić dobiva u točno 6 294 izvlačenja (2 glavna + 242 po 260 € + 6 050 po 26 €; ishoda ≥ 260 €: 244), kako teorija nalaže.')
assert opt['union_all'] == base['union_all'] + 3 * 6294 and opt["union_hi"] == base["union_hi"] + 3 * 244 and opt['union_jp'] == base['union_jp'] + 6
checks.append(f"Unija dobitnih ishoda 13 listića je {opt['union_all']} = {base['union_all']} + 3 × 6 294: svaki novi listić dodaje sve svoje dobitne ishode bez preklapanja (gornja međa postignuta).")
assert abs(opt['ev_total_eur'] - 13 * 380220 / 705432) < 1e-9 and abs(base['ev_total_eur'] - 10 * 380220 / 705432) < 1e-9
checks.append('Očekivana isplata iznosi točno 0,538989 € po listiću (10 listića 5,39 €, 13 listića 7,01 €) i ne ovisi o izboru brojeva.')
ovs = [[len(set(n) & set(t)) for t in T10] for n in NEW]; assert all(k in (5, 6) for r in ovs for k in r)
assert all(len(set(NEW[i]) & set(NEW[j])) in (5, 6) for i in range(3) for j in range(i + 1, 3))
checks.append('Svaki novi listić dijeli točno 5 ili 6 brojeva sa svakim od 10 postojećih i s druga dva nova, što je nužan i dovoljan uvjet dobitne disjunktnosti.')
assert not (win[10:].sum(0) >= 1).__and__(win[:10].any(0)).any() or True
both = int(((win[10:].any(0)) & (win[:10].any(0))).sum()); assert both == 0
checks.append('Ne postoji nijedno izvlačenje u kojem istodobno dobiva neki novi i neki postojeći listić (egzaktno: 0 od 705 432).')
# 4) usporedba brojki s portfolio.py (bit-maske) ako postoji
try:
    R = json.load(open(f'{W}/portfolio_result.json'))
    same = (R['base']['union_all'], R['base']['union_hi'], R['base']['union_jp']) == (base['union_all'], base['union_hi'], base['union_jp']) and R['optimized']['union_all'] == opt['union_all']
    checks.append(f"Neovisni izračun preko matrice incidencije poklapa se s izračunom preko bit-maski (portfolio.py): {same}.")
except FileNotFoundError:
    pass
json.dump(dict(new_tickets=[list(t) for t in NEW], overlaps=ovs, base=base, optimized=opt, checks=checks, rows_in_coverage_list=[c['row'] for c in S['chosen']]),
          open(f'{W}/final13.json', 'w'), indent=1)
print('\n'.join(checks)); print('base', base['union_all'], base['union_hi'], base['union_jp'], round(base['ev_total_eur'], 4)); print('opt', opt['union_all'], opt['union_hi'], opt['union_jp'], round(opt['ev_total_eur'], 4), 'p_any', round(opt['p_any_prize'], 5))
print('dist opt', [(d['payout_eur'], d['n_draws']) for d in opt['payout_distribution']])
print('dist base', [(d['payout_eur'], d['n_draws']) for d in base['payout_distribution']])
