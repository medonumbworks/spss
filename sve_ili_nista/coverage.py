"""Potpuni popis pokrića za Sve ili Ništa (11/22), CSPRNG redoslijed, provjerljivo iz zapisane entropije.

Svaka od 705 432 mogućih izvlačenja pogađa se kao 11/11 jednim listićem i kao 0/11 njegovim komplementom.
Parovi {kombinacija, komplement} ima 352 716; popis sadrži točno jednog predstavnika svakog para
(CSPRNG bira kojeg) u CSPRNG-permutiranom redoslijedu. Zato prvih N redaka pokriva 2N različitih ishoda:
P(glavni dobitak | prvih N listića) = N / 352 716 točno, a svih 352 716 jamči glavni dobitak.

Upotreba: python3 coverage.py <out.csv> [--entropy HEX] [--price 1.30] [--prize 80000]
"""
import sys, csv, hashlib, secrets, argparse, itertools, time
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--entropy'); ap.add_argument('--price', type=float, default=1.30); ap.add_argument('--prize', type=float, default=80000.0)
a = ap.parse_args()
entropy = bytes.fromhex(a.entropy) if a.entropy else secrets.token_bytes(32)
TOTAL = 352716

class DRBG:
    def __init__(self, seed, label): self.seed, self.label, self.ctr, self.buf, self.pos = seed, label, 0, b'', 0
    def _refill(self):
        # 64 blokova SHA-256 odjednom radi brzine
        self.buf = b''.join(hashlib.sha256(self.seed + self.label + (self.ctr + i).to_bytes(8, 'big')).digest() for i in range(64))
        self.ctr += 64; self.pos = 0
    def _byte(self):
        if self.pos >= len(self.buf): self._refill()
        b = self.buf[self.pos]; self.pos += 1; return b
    def uniform(self, n):
        nb = max(1, (int(n).bit_length() + 7) // 8); span = 256 ** nb; lim = span - span % n
        while True:
            v = 0
            for _ in range(nb): v = (v << 8) | self._byte()
            if v < lim: return v % n

t0 = time.time()
# predstavnici parova: sve kombinacije koje sadrže broj 1 (komplement nikad ne sadrži 1) -> točno 352 716, bez parova komplemenata
base = np.zeros((TOTAL, 22), dtype=np.bool_)
for i, rest in enumerate(itertools.combinations(range(1, 22), 10)):
    base[i, 0] = True; base[i, list(rest)] = True
assert i + 1 == TOTAL
# CSPRNG: za svaki par nasumično odaberi kombinaciju ili njezin komplement
g_flip = DRBG(entropy, b'flip')
flip_bytes = np.frombuffer(b''.join(bytes([g_flip._byte()]) for _ in range((TOTAL + 7) // 8)), dtype=np.uint8)
flip = np.unpackbits(flip_bytes)[:TOTAL].astype(bool)
base[flip] = ~base[flip]
# CSPRNG Fisher–Yates permutacija
g_perm = DRBG(entropy, b'perm')
perm = np.arange(TOTAL)
for j in range(TOTAL - 1, 0, -1):
    k = g_perm.uniform(j + 1); perm[j], perm[k] = perm[k], perm[j]
base = base[perm]
nums = np.nonzero(base)[1].reshape(TOTAL, 11) + 1
# provjere ispravnosti
assert (base.sum(1) == 11).all()
keys = (base.astype(np.int64) * (1 << np.arange(22, dtype=np.int64))).sum(1)
ckeys = ((~base).astype(np.int64) * (1 << np.arange(22, dtype=np.int64))).sum(1)
allk = np.concatenate([keys, ckeys])
assert len(np.unique(allk)) == 2 * TOTAL, 'pokriće nije potpuno'
with open(a.out, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f, delimiter=';')
    w.writerow(['Rb'] + [f'B{i}' for i in range(1, 12)] + ['Zbroj', 'Parni', 'Kumulativna_vjerojatnost_%', 'Kumulativni_ulog_EUR'])
    for r in range(TOTAL):
        row = nums[r].tolist(); n = r + 1
        w.writerow([n] + row + [sum(row), sum(1 for x in row if x % 2 == 0), f'{100 * n / TOTAL:.5f}'.replace('.', ','), f'{n * a.price:.2f}'.replace('.', ',')])
print(f'entropy {entropy.hex()}')
print(f'rows {TOTAL}, all {2*TOTAL} outcomes covered exactly once; {time.time()-t0:.1f}s')
print('first 5:', nums[:5].tolist())
