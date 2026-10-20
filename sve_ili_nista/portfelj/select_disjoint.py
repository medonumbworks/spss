"""Odabir 3 dodatna listića među svim kandidatima koji su 'dobitno disjunktni' (k ∈ {5,6} prema svim listićima),
uz presudu izjednačenja redoslijedom dnevnog CSPRNG popisa pokrića (kolo 293). Egzaktna provjera svake tvrdnje."""
import sys, json, csv, itertools, time
import numpy as np
import os
W = os.environ.get('SIN_WORK', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOTAL = 705432
PAY = np.array([80000, 260, 26, 0, 0, 0, 0, 0, 0, 26, 260, 80000], dtype=np.int64); LUT_ALL = PAY > 0; LUT_HI = PAY >= 260
pc = np.bitwise_count
mask = lambda t: np.uint32(sum(1 << (n - 1) for n in t)); unmask = lambda m: [n for n in range(1, 23) if (int(m) >> (n - 1)) & 1]
idx = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(22), 11)), dtype=np.int8, count=TOTAL * 11).reshape(TOTAL, 11)
DRAWS = np.left_shift(np.uint32(1), idx.astype(np.uint32)).sum(1).astype(np.uint32)
CANDS = DRAWS[(DRAWS & 1) == 1]
EX = json.load(open(f'{W}/portfolio/existing10.json')); T10 = [mask(t) for t in EX['tickets']]
# 1) teorem: dva listića s k zajedničkih brojeva nikad ne dobivaju u istom kolu ako k ∈ {5,6} -> izravna provjera na svim parovima klasa k
K = np.stack([pc(CANDS & t) for t in T10])                      # 10 × 352716
disjoint_all = np.all((K == 5) | (K == 6), axis=0)
print('candidates (complement classes) prize-disjoint from all 10:', int(disjoint_all.sum()), 'of', len(CANDS))
# egzaktna potvrda na slučajnom uzorku od 300 takvih kandidata: unija mora biti točno 61050 + 6294
H10 = np.stack([pc(DRAWS & t) for t in T10]); M10 = LUT_ALL[H10].any(0); base_union = int(M10.sum())
rng = np.random.default_rng(7); sample = rng.choice(np.where(disjoint_all)[0], 300, replace=False)
ok = all(int((M10 | LUT_ALL[pc(DRAWS & CANDS[i])]).sum()) == base_union + 6294 for i in sample)
print('exact check on 300 sampled disjoint candidates: each adds exactly 6294 new winning draws ->', ok)
# i obrnuto: kandidat s nekim k izvan {5,6} dodaje manje od 6294
sample2 = rng.choice(np.where(~disjoint_all)[0], 300, replace=False)
ok2 = all(int((M10 | LUT_ALL[pc(DRAWS & CANDS[i])]).sum()) < base_union + 6294 for i in sample2)
print('exact check on 300 non-disjoint candidates: each adds fewer than 6294 ->', ok2)
# 2) presuda izjednačenja: prvi redovi dnevnog popisa pokrića koji su disjunktni prema 10 i međusobno
rows = []
with open(f'{W}/kolo293/kombinacije_kolo293_pokrice.csv', encoding='utf-8-sig') as f:
    r = csv.reader(f, delimiter=';'); next(r)
    for x in r: rows.append((int(x[0]), [int(v) for v in x[1:12]]))
chosen = []; cur = list(T10); skipped = []
for rb, t in rows[10:]:
    m = mask(t); ks = [int(pc(m & c)) for c in cur]
    if all(k in (5, 6) for k in ks):
        chosen.append((rb, t, ks)); cur.append(m)
        if len(chosen) == 3: break
    else:
        skipped.append((rb, [k for k in ks if k not in (5, 6)]))
print('chosen rows:', [(rb, t) for rb, t, _ in chosen]); print('rows skipped before completing:', len(skipped), skipped[:8])
T13 = T10 + [mask(t) for _, t, _ in chosen]
H = np.stack([pc(DRAWS & t) for t in T13]); tot = PAY[H].sum(0)
res = dict(union_all=int(LUT_ALL[H].any(0).sum()), union_hi=int(LUT_HI[H].any(0).sum()), union_jp=int((PAY[H] >= 80000).any(0).sum()),
           ev=float(tot.sum() / TOTAL), p_two_prizes=float((LUT_ALL[H].sum(0) >= 2).sum() / TOTAL), max_possible=13 * 6294)
print(res)
json.dump(dict(n_disjoint_candidates=int(disjoint_all.sum()), chosen=[dict(row=rb, ticket=t, overlaps=ks) for rb, t, ks in chosen], skipped=skipped, result=res),
          open(f'{W}/portfolio/select_disjoint.json', 'w'), indent=1)
