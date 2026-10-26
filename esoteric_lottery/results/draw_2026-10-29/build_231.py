# Zapis kako je nastao loto_231_kombinacija.csv. Ulazi: hist.json (882 izvlacenja iz
# loto_6_45_2018_2026.csv), ez29_2000.json (10 metoda, 60 citanja, 2026-10-29 20:00 CET),
# banned.json (597 vec odigranih/objavljenih kombinacija). Pokrenuto sa sharpness = 6.0.
"""231 kombinacija za Loto 6/45, izvlacenje 2026-10-29 20:00 CET.
Blokovi se grade od najuzeg ogranicenja prema najsirem; ispisuju se redom 1-4."""
import sys, json, itertools; sys.path.insert(0,'/home/user/spss/esoteric_lottery')
import numpy as np
from evaluation.themed import (TripleLedger, coverage_block, weighted_block,
                               empirical_ok, unpopular_ok, not_a_pattern, frequency_table)
from methods.common import seeded_rng
D='/tmp/claude-0/-home-user-spss/06ab165b-5154-5417-a4c3-6b8725791e2d/scratchpad'
SHARP=float(sys.argv[1]) if len(sys.argv)>1 else 3.0
hist=json.load(open(f'{D}/hist.json')); ez=json.load(open(f'{D}/ez29_2000.json'))
banned={tuple(c) for c in json.load(open(f'{D}/banned.json'))}
ledger=TripleLedger()
N=lambda d:{int(k):v for k,v in d.items()}
freq,gap,recent=N(hist['freq']),N(hist['gap']),N(hist['recent'])
def borda(score):
    return {n:45.0-i for i,n in enumerate(sorted(range(1,46), key=lambda n:(-score[n],n)))}
b_freq,b_hot,b_gap=borda(freq),borda(recent),borda(gap)
w2=np.array([0.35*b_freq[n]+0.40*b_hot[n]+0.25*b_gap[n] for n in range(1,46)])
pareto=sorted(range(1,46), key=lambda n:-w2[n-1])[:9]
for n in pareto: w2[n-1]*=1.6

TRIES=20000
# 4. kontrarni (najuze)
rng4=seeded_rng("l6|29.10|kontra")
w4=np.array([1.0+2.0*(n>31)+0.6*(13<=n<=31) for n in range(1,46)])
g4=weighted_block(31, rng4, w4, lambda t: unpopular_ok(t) and not_a_pattern(t),
                  ledger, banned); print(f"4. blok: {len(g4)}")
# 3. ezoterija, 5 po metodi
rng3=seeded_rng("l6|29.10|ezo"); g3=[]; g3_src=[]
for key in sorted(ez['per_method']):
    v=np.array(ez['per_method'][key],dtype=float)
    w=(v-v.min())/(v.max()-v.min())*9.0+1.0
    g3+=weighted_block(5, rng3, w, not_a_pattern, ledger, banned); g3_src+=[key]*5
print(f"3. blok: {len(g3)}")
# 2. statistika
rng2=seeded_rng("l6|29.10|stat")
g2=weighted_block(50, rng2, w2, lambda t: empirical_ok(t) and not_a_pattern(t),
                  ledger, banned); print(f"2. blok: {len(g2)}")
# 1. pokrivenost (najsire)
rng1=seeded_rng("l6|29.10|pokrivenost")
g1=coverage_block(100, rng1, ledger, banned, sharpness=SHARP); print(f"1. blok: {len(g1)}")

all_t=g1+g2+g3+g4
json.dump({'g1':[list(t) for t in g1],'g2':[list(t) for t in g2],'g3':[list(t) for t in g3],
           'g3_src':g3_src,'g4':[list(t) for t in g4],'pareto':sorted(pareto),
           'sharpness':SHARP}, open(f'{D}/t231.json','w'))
mx=max(len(set(a)&set(b)) for a,b in itertools.combinations(all_t,2))
print(f"\nukupno {len(all_t)}, razlicitih {len(set(all_t))}, najveci presjek para = {mx}")
f1=frequency_table(g1); fa=frequency_table(all_t)
print(f"1. blok frekvencije: {min(f1.values())}-{max(f1.values())} (cilj 13-14)")
print(f"svih 231: {min(fa.values())}-{max(fa.values())} (prosjek {231*6/45:.1f})")
