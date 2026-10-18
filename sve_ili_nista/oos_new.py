"""Provjera izvan uzorka: tvrdnje i strategije izvedene iz 635 kola (do 28.09.2026.) testiraju se na 18 novih kola 272–289."""
import os
import sys, json, math, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sin_lib as L
d = L.load_draws(sys.argv[1]); old, new = d[:635], d[635:]
assert len(old) == 635 and old[-1]['kolo'] == 271 and len(new) == 18
SD = math.sqrt(11 * 0.5 * 0.5 * 11 / 21)
M = np.zeros((len(old), 22), int)
for i, x in enumerate(old): M[i, np.array(x['nums']) - 1] = 1
N = np.zeros((len(new), 22), int)
for i, x in enumerate(new): N[i, np.array(x['nums']) - 1] = 1
def top(freq, k=11, rev=False):
    order = sorted(range(22), key=lambda n: ((freq[n] if rev else -freq[n]), n))
    return sorted(int(n + 1) for n in order[:k])
fa = M.sum(0); f30 = M[-30:].sum(0); f90 = M[-90:].sum(0)
gaps = L.gaps(old); od = sorted(gaps, key=lambda n: (-gaps[n], n))[:11]
tickets = {
  'vrući-30': top(f30), 'vrući-90': top(f90), 'vrući-ukupno': top(fa), 'hladni-ukupno': top(fa, rev=True),
  'zakašnjeli-11': sorted(od), 'ponovi-zadnje (28.09.)': old[-1]['nums'], 'komplement-zadnjeg': L.complement(old[-1]['nums']),
  'tilted-model najbolji': [1, 3, 5, 6, 7, 8, 15, 16, 17, 19, 22],
}
out = {'tickets': {}, 'claims': {}}
for name, t in tickets.items():
    idx = np.array(t) - 1; hits = N[:, idx].sum(1)
    z = (hits.mean() - 5.5) / (SD / math.sqrt(len(hits)))
    out['tickets'][name] = dict(ticket=t, hits=hits.tolist(), mean=round(float(hits.mean()), 3), z=round(float(z), 2),
                                jackpots=int(((hits == 0) | (hits == 11)).sum()), extreme=int(((hits <= 3) | (hits >= 8)).sum()))
# tvrdnje
cnt = N.sum(0)
out['claims']['broj 13 "hladan" (izvučen x od 18; očekivano 9)'] = int(cnt[12])
out['claims']['broj 7 "vruć" (x od 18; očekivano 9)'] = int(cnt[6])
pair = lambda a, b: int((N[:, a - 1] & N[:, b - 1]).sum())
e_pair = 18 * 55 / 231; e_tri = 18 * math.comb(19, 8) / math.comb(22, 11)
out['claims']['par 7-15 "sretan" (x od 18; očekivano %.2f)' % e_pair] = pair(7, 15)
out['claims']['par 13-20 "nesretan" (x od 18; očekivano %.2f)' % e_pair] = pair(13, 20)
out['claims']['trojka 7-15-16 "favorizirana" (x od 18; očekivano %.2f)' % e_tri] = int((N[:, 6] & N[:, 14] & N[:, 15]).sum())
out['claims']['trojka 11-13-20 "potisnuta" (x od 18; očekivano %.2f)' % e_tri] = int((N[:, 10] & N[:, 12] & N[:, 19]).sum())
out['claims']['parnih brojeva u prosjeku (očekivano 5,5)'] = round(float(np.mean([sum(1 for n in x['nums'] if n % 2 == 0) for x in new])), 2)
ov = [len(set(new[i]['nums']) & set((old + new)[634 + i]['nums'])) for i in range(18)]
out['claims']['preklapanje s prethodnim kolom (očekivano 5,5)'] = round(float(np.mean(ov)), 2)
out['new_freq'] = {int(n + 1): int(cnt[n]) for n in range(22)}
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
for k, v in out['tickets'].items(): print(f"{k:24} {v['ticket']}  prosj. {v['mean']}  z={v['z']}  ekstremi {v['extreme']}/18  dobitci {v['jackpots']}")
for k, v in out['claims'].items(): print(f"  {k}: {v}")
