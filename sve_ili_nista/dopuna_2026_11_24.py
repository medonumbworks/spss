# -*- coding: utf-8 -*-
"""Deterministička dopuna za 24.11.2026. (bez slučajnosti, kombinacije se NE mijenjaju):
redoslijed igranja u CSV (#;tip;sustav;redoslijed;B1..B11), granice P(isplata ≥ ulog) po k, udaljenost od promjene nakšatre/tithija/yoge.
PYTHONPATH=<pyswisseph> python3 dopuna_2026_11_24.py <sin csv> <rezultati json> <ephe dir>"""
import sys, json
from math import comb, ceil
from fractions import Fraction
import numpy as np, pandas as pd
import swisseph as swe
SIN, RJ, EPHE = sys.argv[1:4]
B = [f'B{i}' for i in range(1, 12)]
s = pd.read_csv(SIN, sep=';')
R = json.load(open(RJ, encoding='utf-8')); O = R['L_opt']
ids = s['#'].tolist(); combos = [list(map(int, r)) for r in s[B].values]
assert combos[6:] == O['L'], 'CSV i rezultati se ne slažu'
order = O['order']; rank = {c: i + 1 for i, c in enumerate(order)}
TOTAL = comb(22, 11)
ALL = np.arange(1 << 22, dtype=np.uint32); ALL = ALL[np.bitwise_count(ALL) == 11]
def bm(c): return sum(1 << (v - 1) for v in c)
K = {i: np.bitwise_count(ALL & np.uint32(bm(c))) for i, c in zip(ids, combos)}
WL = np.zeros(12, bool); WL[[0, 1, 2, 3, 8, 9, 10, 11]] = True
seq = []
for k in range(1, 10):
    sub = order[:k]
    anyw = np.zeros(TOTAL, bool); nJ = np.zeros(TOTAL, np.int16); n83 = np.zeros(TOTAL, np.int16); unk = np.zeros(TOTAL, bool)
    for c in sub:
        kk = K[c]; anyw |= WL[kk]; nJ += (kk == 0) | (kk == 11); n83 += (kk == 3) | (kk == 8); unk |= (kk == 1) | (kk == 2) | (kk == 9) | (kk == 10)
    stake = Fraction(13, 10) * k; need = ceil(stake / Fraction(26, 10))
    ge = (nJ > 0) | (n83 >= need)
    row = dict(k=k, kombinacije=sub, ulog=float(stake), need83=need, p_any=int(anyw.sum()), p_ge_low=int(ge.sum()), p_ge_up=int((ge | unk).sum()))
    old = O['seq'][k - 1]; assert old['p_any'] == row['p_any'] and old['p_ge'] == row['p_ge_low']
    row['best_subset'] = old['best_subset']; row['best_p'] = old['best_p']; row['greedy_optimal'] = old['greedy_optimal']
    seq.append(row)
# udaljenost od promjene (linearno prema trenutnoj brzini)
swe.set_ephe_path(EPHE); swe.set_sid_mode(swe.SIDM_LAHIRI)
jd = R['E']['JD_UT']; FL = swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED
(m, _, _, ms, _, _), fm = swe.calc_ut(jd, swe.MOON, FL); (su, _, _, ss, _, _), fs = swe.calc_ut(jd, swe.SUN, FL)
assert fm & swe.FLG_SWIEPH and fs & swe.FLG_SWIEPH
NAK = 360 / 27
def dist(x, span, speed):
    x %= 360; st = (x // span) * span
    return dict(od_pocetka_min=(x - st) / speed * 1440, do_kraja_min=(st + span - x) / speed * 1440)
gr = dict(naksatra=dist(m, NAK, ms), tithi=dist(m - su, 12, ms - ss), yoga=dist(m + su, NAK, ms + ss))
R['redoslijed'] = dict(order=order, rank=rank, seq=seq); R['granice'] = gr
json.dump(R, open(RJ, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
s.insert(3, 'redoslijed', [rank[i] for i in ids])
s.to_csv(SIN, sep=';', index=False, encoding='utf-8')
print(open(SIN, encoding='utf-8').read())
for r in seq: print(r)
print(json.dumps(gr))
