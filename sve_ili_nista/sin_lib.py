"""Zajednička knjižnica za analizu igre Sve ili Ništa (HL): 11 od 22, dnevna izvlačenja.
Učitavanje podataka, osnovne statistike i CSPRNG generator.
"""
import csv, io, math, secrets, itertools
from collections import Counter, defaultdict
from datetime import datetime

N_MAX = 22
K = 11
TOTAL_COMBOS = math.comb(N_MAX, K)  # 705432

def load_draws(path):
    """Vrati listu dictova: {rb, godina, kolo, datum(date), dan, nums(list[int] sorted)} samo za kola s brojevima."""
    txt = open(path, 'rb').read().decode('utf-8-sig')
    rows = list(csv.reader(io.StringIO(txt), delimiter=';'))
    hdr = rows[0]
    out = []
    for r in rows[1:]:
        if len(r) < 16:
            continue
        nums = [int(x) for x in r[5:16] if x.strip()]
        if len(nums) != K:
            continue
        d = datetime.strptime(r[3].strip(), '%d.%m.%Y').date()
        out.append(dict(rb=int(r[0]), godina=int(r[1]), kolo=int(r[2]), datum=d, dan=r[4], nums=sorted(nums)))
    return out

def frequency(draws):
    c = Counter()
    for d in draws:
        c.update(d['nums'])
    return {n: c.get(n, 0) for n in range(1, N_MAX + 1)}

def gaps(draws):
    """Broj kola od zadnjeg pojavljivanja (0 = izvučen u zadnjem kolu)."""
    last = {n: None for n in range(1, N_MAX + 1)}
    for i, d in enumerate(draws):
        for n in d['nums']:
            last[n] = i
    L = len(draws)
    return {n: (L - 1 - last[n]) if last[n] is not None else L for n in range(1, N_MAX + 1)}

def pair_counts(draws):
    c = Counter()
    for d in draws:
        for a, b in itertools.combinations(d['nums'], 2):
            c[(a, b)] += 1
    return c

def chi_square_uniform(freq, n_draws):
    exp = n_draws * K / N_MAX
    chi2 = sum((f - exp) ** 2 / exp for f in freq.values())
    return chi2, exp

def chi2_sf(x, df):
    """Survival function hi-kvadrat distribucije (regularizirana gornja gama), bez scipy."""
    return _gammaincc(df / 2.0, x / 2.0)

def _gammaincc(a, x):
    if x < a + 1:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)

def _gser(a, x, itmax=500, eps=3e-12):
    gln = math.lgamma(a)
    ap = a; s = 1.0 / a; delta = s
    for _ in range(itmax):
        ap += 1; delta *= x / ap; s += delta
        if abs(delta) < abs(s) * eps:
            break
    return s * math.exp(-x + a * math.log(x) - gln)

def _gcf(a, x, itmax=500, eps=3e-12):
    gln = math.lgamma(a)
    b = x + 1 - a; c = 1 / 1e-300; d = 1 / b; h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a); b += 2
        d = an * d + b
        if abs(d) < 1e-300: d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300: c = 1e-300
        d = 1 / d; delta = d * c; h *= delta
        if abs(delta - 1) < eps:
            break
    return math.exp(-x + a * math.log(x) - gln) * h

# ---------- CSPRNG ----------
_rng = secrets.SystemRandom()

def csprng_uniform_combo():
    """Uniformno slučajna 11-člana kombinacija iz 1..22 (os.urandom preko secrets)."""
    return sorted(_rng.sample(range(1, N_MAX + 1), K))

def csprng_weighted_combo(weights):
    """Uzorkovanje bez vraćanja proporcionalno težinama (CSPRNG). weights: dict n->w>0."""
    pool = dict(weights)
    chosen = []
    for _ in range(K):
        tot = sum(pool.values())
        r = _rng.random() * tot
        acc = 0.0
        for n, w in pool.items():
            acc += w
            if r <= acc:
                chosen.append(n); del pool[n]; break
        else:
            n = next(iter(pool)); chosen.append(n); del pool[n]
    return sorted(chosen)

def complement(combo):
    s = set(combo)
    return [n for n in range(1, N_MAX + 1) if n not in s]

def describe(combo):
    return dict(zbroj=sum(combo), parni=sum(1 for n in combo if n % 2 == 0),
                niski=sum(1 for n in combo if n <= 11),
                uzastopni=sum(1 for a, b in zip(combo, combo[1:]) if b == a + 1))
