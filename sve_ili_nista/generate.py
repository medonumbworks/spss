"""Generator kombinacija za Sve ili Ništa (11/22) temeljen na CSPRNG-u, s provjerljivim zapisom entropije.

Entropija: 32 bajta iz os.urandom (secrets.token_bytes). Svaki broj se izvodi determinističkim
SHA-256 DRBG-om nad tom entropijom (rejection sampling), pa se cijeli ispis može reproducirati iz
zapisane heksadecimalne entropije: python3 generate.py <csv> --entropy <hex>.
"""
import os
import sys, json, math, hashlib, secrets, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sin_lib as L

ap = argparse.ArgumentParser()
ap.add_argument('csv'); ap.add_argument('--entropy', default=None); ap.add_argument('--extra', default=None, help='JSON s dodatnim kolima [{date,kolo,numbers}]')
ap.add_argument('--out', default='combos.json'); ap.add_argument('--n-uniform', type=int, default=6)
a = ap.parse_args()

draws = L.load_draws(a.csv)
extra = []
if a.extra:
    for e in json.load(open(a.extra)):
        extra.append(dict(datum=e['date'], kolo=e['kolo'], nums=sorted(e['numbers'])))
all_draws = draws + extra
T = len(all_draws)
M = np.zeros((T, 22), dtype=np.int32)
for i, d in enumerate(all_draws):
    for n in d['nums']: M[i, n - 1] = 1

entropy = bytes.fromhex(a.entropy) if a.entropy else secrets.token_bytes(32)

class DRBG:
    """SHA-256 counter DRBG nad 32-bajtnom entropijom; 'uniform(n)' bez pristranosti (rejection sampling)."""
    def __init__(self, seed: bytes, label: bytes):
        self.seed = seed; self.label = label; self.ctr = 0; self.buf = b''
    def _byte(self):
        if not self.buf:
            self.buf = hashlib.sha256(self.seed + self.label + self.ctr.to_bytes(8, 'big')).digest(); self.ctr += 1
        b, self.buf = self.buf[0], self.buf[1:]
        return b
    def uniform(self, n):
        """Cijeli broj u [0, n) bez pristranosti: koristi dovoljno bajtova za n, rejection sampling."""
        nbytes = max(1, (int(n).bit_length() + 7) // 8)
        span = 256 ** nbytes; lim = span - (span % n)
        while True:
            v = 0
            for _ in range(nbytes): v = (v << 8) | self._byte()
            if v < lim: return v % n
    def sample(self, k, pool):
        pool = list(pool); out = []
        for _ in range(k):
            j = self.uniform(len(pool)); out.append(pool.pop(j))
        return sorted(out)
    def weighted(self, k, weights):
        """Uzorkovanje bez vraćanja ∝ težinama; težine cjelobrojne (ili zaokružene na 1/1000)."""
        pool = {n: max(1, int(round(w * 1000))) for n, w in weights.items()}
        out = []
        for _ in range(k):
            tot = sum(pool.values()); r = self.uniform(tot); acc = 0
            for n, w in pool.items():
                acc += w
                if r < acc: out.append(n); del pool[n]; break
        return sorted(out)

freq_all = M.sum(0); freq60 = M[-60:].sum(0); freq30 = M[-30:].sum(0)
gaps = {}
for n in range(22):
    idx = np.where(M[:, n] == 1)[0]; gaps[n + 1] = T - 1 - int(idx[-1]) if len(idx) else T

combos = []
_seen = set()
def add(name, kind, combo, why):
    """Dodaj kombinaciju; odbij duplikat ili komplement već dodane (identični ishodi u svim kategorijama)."""
    key = tuple(combo); comp = tuple(L.complement(combo))
    if key in _seen or comp in _seen:
        return False
    _seen.add(key)
    desc = L.describe(combo)
    combos.append(dict(name=name, strategy=kind, numbers=combo, complement=list(comp), **desc, why=why))
    return True

# A) čisti CSPRNG — uniformno nad svih 705 432 kombinacija
for i in range(a.n_uniform):
    g = DRBG(entropy, f'uniform-{i}'.encode())
    while not add(f'U{i+1}', 'csprng-uniform', g.sample(11, range(1, 23)), 'uniformno iz svih 705 432 kombinacija'):
        pass
# B) CSPRNG ponderiran "vrućim" brojevima zadnjih 60 kola
g = DRBG(entropy, b'hot60'); w = {n + 1: 1 + freq60[n] for n in range(22)}
while not add('H1', 'csprng-weighted-hot60', g.weighted(11, w), 'težina = 1 + broj pojavljivanja u zadnjih 60 kola'): pass
g = DRBG(entropy, b'hot30'); w = {n + 1: 1 + freq30[n] for n in range(22)}
while not add('H2', 'csprng-weighted-hot30', g.weighted(11, w), 'težina = 1 + broj pojavljivanja u zadnjih 30 kola'): pass
# C) CSPRNG ponderiran "zakašnjelim" brojevima (veći razmak = veća težina)
g = DRBG(entropy, b'overdue'); w = {n: 1 + gaps[n] for n in gaps}
while not add('Z1', 'csprng-weighted-overdue', g.weighted(11, w), 'težina = 1 + broj kola od zadnjeg pojavljivanja'): pass
# D) CSPRNG ponderiran hladnim brojevima (ukupno najrjeđi)
g = DRBG(entropy, b'cold'); mx = int(freq_all.max()); w = {n + 1: 1 + (mx - freq_all[n]) for n in range(22)}
while not add('C1', 'csprng-weighted-cold', g.weighted(11, w), 'težina = 1 + (max. frekvencija − frekvencija broja)'): pass
# E) "tipična struktura": uniformni CSPRNG uz filtar zbroj u središnjih 50 % (116–137) i 5–6 parnih
g = DRBG(entropy, b'typical'); tries = 0
while True:
    c = g.sample(11, range(1, 23)); tries += 1; d = L.describe(c)
    if 116 <= d['zbroj'] <= 137 and d['parni'] in (5, 6) and add('S1', 'csprng-typical-structure', c, f'uniformni CSPRNG + filtar zbroj 116–137 i 5–6 parnih (odbačeno {tries-1} uzoraka)'): break

out = dict(entropy_hex=entropy.hex(), n_draws_used=T, last_draw=str(all_draws[-1]['datum']), combos=combos,
           note='Svaka kombinacija pokriva 2 od 705 432 ishoda (11/11 ili 0/11): p = 1/352 716 po listiću, neovisno o strategiji.')
json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1, default=str)
print('entropy', entropy.hex())
for c in combos:
    print(f"{c['name']:3} {c['strategy']:26} {' '.join(f'{n:2d}' for n in c['numbers'])}  | zbroj {c['zbroj']:3d} parni {c['parni']} niski {c['niski']} uzast {c['uzastopni']}")
