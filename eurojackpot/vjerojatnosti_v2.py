"""Točne vjerojatnosti seta: svih 2.118.760 glavnih × 66 euro ishoda (numpy bitmaske)."""
import itertools
from math import comb
import numpy as np

_C = np.array(list(itertools.combinations(range(50), 5)), dtype=np.uint64)
DM = np.zeros(len(_C), np.uint64)
for _j in range(5):
    DM |= np.left_shift(np.uint64(1), _C[:, _j])
del _C
EP = list(itertools.combinations(range(12), 2))
NG, NE = len(DM), len(EP)
TOT = NG * NE
assert NG == 2118760 and NE == 66
RAZ = [(5, 2), (5, 1), (5, 0), (4, 2), (4, 1), (4, 0), (3, 2), (2, 2), (3, 1), (3, 0), (1, 2), (2, 1)]
pmh = {k: comb(5, k) * comb(45, 5 - k) / comb(50, 5) for k in range(6)}
peh = {k: comb(2, k) * comb(10, 2 - k) / comb(12, 2) for k in range(3)}
P1 = sum(pmh[a] * peh[b] for a, b in RAZ)
_CODES = {}

def codes(n):
    if n not in _CODES:
        _CODES[n] = np.array(list(itertools.product(range(3), repeat=n)))[:, ::-1]  # znamenka c = code_c
    return _CODES[n]

def maska(g):
    return np.uint64(sum(1 << (x - 1) for x in g))

def glavni_cache(G):
    """G: lista glavnih kombinacija (1-based). Vraća sve što ovisi samo o glavnim brojevima."""
    n = len(G)
    m = np.stack([np.bitwise_count(DM & maska(g)) for g in G]).astype(np.int8)
    A = (m >= 3).any(0)
    w = (3 ** np.arange(n)).astype(np.int64)[:, None]
    idx = (np.minimum(m, 2).astype(np.int64) * w).sum(0)
    hist = np.bincount(idx[~A], minlength=3 ** n)
    sel = np.nonzero((m >= 4).any(0))[0]
    return dict(n=n, A=int(A.sum()), hist=hist, m4=m[:, sel])

def euro_matrica(Eu):
    return np.array([[len(set(x - 1 for x in e) & set(p)) for e in Eu] for p in EP], dtype=np.int8)  # 66 x n

def vjerojatnosti(S, cache=None):
    G = [g for g, _ in S]; Eu = [e for _, e in S]
    if cache is None:
        cache = glavni_cache(G)
    n = cache['n']; E = euro_matrica(Eu)
    cd = codes(n)
    tab = ((cd[:, None, :] + E[None, :, :]) >= 3).any(2).sum(1)
    any_cnt = cache['A'] * NE + int(cache['hist'] @ tab)
    m4 = cache['m4']  # n x k
    top = ((m4.T[:, None, :] == 5) | ((m4.T[:, None, :] == 4) & (E[None, :, :] == 2))).any(2).sum()
    p52 = ((m4.T[:, None, :] == 5) & (E[None, :, :] == 2)).any(2).sum()
    return dict(P_5p2=p52 / TOT, P_top=top / TOT, P_bilo_koji=any_cnt / TOT, ocek_dobitnih=n * P1)

if __name__ == '__main__':
    import json, os, time
    OUT = os.path.dirname(os.path.abspath(__file__))
    r = json.load(open(os.path.join(OUT, 'rezultat.json')))
    S = [(g, e) for g, e in r['set']]
    t = time.time(); v = vjerojatnosti(S); print(v, time.time() - t)
    print('v1:', r['vjerojatnosti'])
