# loto_ezoterija.py – Meihua + Lenormand + RW + Golden Dawn, hibridna optimizacija
# Pokretanje: python loto_ezoterija.py  (prije toga popuni ULAZ niže)
# Potrebno: Python 3.8+ i numpy. Ne mijenjati ništa osim bloka ULAZ.
import secrets, datetime, itertools, json, sys
from collections import Counter
import numpy as np

# ===================== ULAZ =====================
IGRA        = "SIN"        # "SIN" (Sve ili ništa) ili "L6" (Loto 6 jednostavne)
DATUM       = "2026-10-05" # datum izvlačenja
SIMULIRANO  = "12:15"      # simulirano vrijeme bacanja (CEST), sredina Venerina sata
GODINA_BROJ = 7            # 丙午 → 午 = 7 (vrijedi do kineske Nove godine 6.2.2027.)
LUNARNI_MJ  = 8            # lunarni mjesec
LUNARNI_DAN = 25           # lunarni dan
SAT_BROJ    = 7            # grana sata po pravom sunčevom vremenu (子1 … 亥12)
FIKSNI_PAR  = [2,3,4,7,9,11,14,16,17,20,22]   # samo za SIN; komplement se računa sam
MIJESANJA   = 300          # broj miješanja po kartaškom listiću u hibridu
MC_UZORAKA  = 400000       # Monte Carlo izvlačenja za L6
# ================================================

now = datetime.datetime.now(datetime.timezone.utc).astimezone()
print("Kod je izvršen. Stvarno vrijeme izvršavanja:", now.isoformat())
print(f"Igra {IGRA} | izvlačenje {DATUM} | simulirano {SIMULIRANO} CEST | godina {GODINA_BROJ}, lunarni {LUNARNI_MJ}/{LUNARNI_DAN}, sat {SAT_BROJ}")
assert IGRA in ("SIN", "L6")
N, K = (22, 11) if IGRA == "SIN" else (45, 6)

# ---------- Meihua ----------
TL = {1:(1,1,1),2:(1,1,0),3:(1,0,1),4:(1,0,0),5:(0,1,1),6:(0,1,0),7:(0,0,1),8:(0,0,0)}  # linije odozdo prema gore
LT = {v:k for k,v in TL.items()}
TN = {1:"乾 Qian",2:"兌 Dui",3:"離 Li",4:"震 Zhen",5:"巽 Xun",6:"坎 Kan",7:"艮 Gen",8:"坤 Kun"}
KW = {1:[1,10,13,25,44,6,33,12],2:[43,58,49,17,28,47,31,45],3:[14,38,30,21,50,64,56,35],4:[34,54,55,51,32,40,62,16],
      5:[9,61,37,42,57,59,53,20],6:[5,60,63,3,48,29,39,8],7:[26,41,22,27,18,4,52,23],8:[11,19,36,24,46,7,15,2]}  # KW[gornji][donji-1]
kw = lambda u,l: KW[u][l-1]
def hexa(lines, mv):
    lo, up = LT[tuple(lines[:3])], LT[tuple(lines[3:])]
    nl, nu = LT[tuple(lines[1:4])], LT[tuple(lines[2:5])]
    ch = list(lines); ch[mv-1] ^= 1
    cl, cu = LT[tuple(ch[:3])], LT[tuple(ch[3:])]
    ti, yo = (up, lo) if mv <= 3 else (lo, up)
    return dict(ben=(up,lo,kw(up,lo)), hu=(nu,nl,kw(nu,nl)), bian=(cu,cl,kw(cu,cl)), mv=mv, ti=ti, yong=yo, lines=lines)
Z = GODINA_BROJ + LUNARNI_MJ + LUNARNI_DAN
up = Z % 8 or 8; low = (Z + SAT_BROJ) % 8 or 8; mv = (Z + SAT_BROJ) % 6 or 6
print(f"\nMEIHUA: Z = {GODINA_BROJ}+{LUNARNI_MJ}+{LUNARNI_DAN} = {Z}; gornji = {Z} mod 8 = {up} {TN[up]}; "
      f"donji = ({Z}+{SAT_BROJ}) mod 8 = {low} {TN[low]}; pomična = ({Z}+{SAT_BROJ}) mod 6 = {mv}  (ostatak 0 → 8 odnosno 6)")
base = list(TL[low]) + list(TL[up])
H = {"本":hexa(base,mv), "錯":hexa([1-x for x in base],mv), "綜":hexa(base[::-1],7-mv), "錯綜":hexa([1-x for x in base[::-1]],7-mv)}
S = {}
for k,h in H.items():
    S[k] = [h["ben"][0],h["ben"][1],h["ben"][2],h["hu"][0],h["hu"][1],h["hu"][2],h["bian"][0],h["bian"][1],h["bian"][2],h["mv"]]
    print(f"  {k}: linije {h['lines']} | 本 {TN[h['ben'][0]]}/{TN[h['ben'][1]]} = {h['ben'][2]} | 互 {TN[h['hu'][0]]}/{TN[h['hu'][1]]} = {h['hu'][2]} | "
          f"變 {TN[h['bian'][0]]}/{TN[h['bian'][1]]} = {h['bian'][2]} | pomična {h['mv']} | 體 {TN[h['ti']]} 用 {TN[h['yong']]} | S = {S[k]}")

U = set(range(1, N+1))
def stream(Sq):
    c = 0; i = 0
    while True:
        s = Sq[i % len(Sq)]; c += s; i += 1
        yield (i, s, c, (c-1) % N + 1)
def meihua(Sq, ok):
    out = []; log = []; g = stream(Sq)
    def nxt():
        i,s,c,cand = next(g); x = cand
        while x in out: x = x % N + 1
        out.append(x); log.append(f"korak {i}: +{s} → zbroj {c} → (({c}−1) mod {N})+1 = {cand}" + ("" if x == cand else f" zauzet → {x}"))
    while len(out) < K: nxt()
    n = 0
    while not ok(out):
        log.append(f"ODBAČEN zadnji broj {out.pop()} (provjera nije prošla), nastavljam niz"); nxt(); n += 1
        if n > 500: sys.exit("Meihua: nema valjanog rješenja")
    return out, log

# ---------- provjere ----------
FIX = [set(FIKSNI_PAR), U - set(FIKSNI_PAR)] if IGRA == "SIN" else []
def popular_ok(t):
    s = sorted(t)
    if sum(x <= 31 for x in s) > 3: return False
    if any(s[i]+1 == s[i+1] == s[i+2]-1 for i in range(len(s)-2)): return False
    if max(Counter(x % 10 for x in s).values()) > 2: return False
    if max(Counter((x-1)//10 for x in s).values()) > 3: return False
    if sum(x % 2 for x in s) in (0, len(s)): return False
    d = [s[i+1]-s[i] for i in range(len(s)-1)]
    if any(d[i] == d[i+1] == d[i+2] for i in range(len(d)-2)): return False
    return True
def valid(t, others):
    A = set(t)
    if len(A) != K or not A <= U: return False
    if IGRA == "SIN": return all(A != set(B) and A != U - set(B) for B in list(others) + FIX)
    return all(len(A & set(B)) <= 2 for B in others)

# ---------- vektori dobitka ----------
if IGRA == "SIN":
    D = np.array([sum(1 << x for x in c) for c in itertools.combinations(range(22), 11)], dtype=np.int64)
    LUT = np.array([bin(i).count("1") for i in range(1 << 11)], dtype=np.int16)
    PR = np.zeros(12); PR[[0,11]] = 80000; PR[[1,10]] = 260; PR[[2,9]] = 26; PR[[3,8]] = 2.6
    def hits(t):
        a = D & np.int64(sum(1 << (x-1) for x in t)); return LUT[a & 2047] + LUT[a >> 11]
    def wv(t): return PR[hits(t)] > 0
    CIJENA = 1.30
    print(f"\nTočan izračun preko svih {len(D)} mogućih izvlačenja.")
else:
    g = np.random.default_rng(int.from_bytes(secrets.token_bytes(8), "big"))
    parts = []
    for _ in range(MC_UZORAKA // 50000):
        idx = np.argsort(g.random((50000, 45)), axis=1)[:, :6].astype(np.int64)
        parts.append(np.bitwise_or.reduce(np.left_shift(np.int64(1), idx), axis=1))
    D = np.concatenate(parts)
    LUT = np.array([bin(i).count("1") for i in range(1 << 15)], dtype=np.int16)
    def hits(t):
        a = D & np.int64(sum(1 << (x-1) for x in t)); return LUT[a & 32767] + LUT[(a >> 15) & 32767] + LUT[a >> 30]
    def wv(t): return hits(t) >= 3
    CIJENA = 0.25
    print(f"\nMonte Carlo: {len(D)} slučajnih izvlačenja (6 od 45).")

# ---------- Meihua listići ----------
T = {}; LOG = {}
nazivi = [("M_本_SVE","本"),("M_錯_NIŠTA","錯"),("M_綜_SVE","綜"),("M_錯綜_NIŠTA","錯綜")] if IGRA == "SIN" else [("M_本","本"),("M_錯","錯"),("M_綜","綜"),("M_錯綜","錯綜")]
for nm, k in nazivi:
    o, l = meihua(S[k], lambda a: valid(a, list(T.values())))
    T[nm] = o; LOG[nm] = l
U_any = np.zeros(len(D), bool)
for t in T.values(): U_any |= wv(t)
print("P(barem 1) nakon 4 Meihua:", round(U_any.mean(), 4))

# ---------- karte ----------
LN = dict(enumerate("Jahač Djetelina Brod Kuća Stablo Oblaci Zmija Lijes Buket Kosa Bič Ptice Dijete Lisica Medvjed Zvijezde Roda Pas Toranj Vrt Planina Raskrižje Miševi Srce Prsten Knjiga Pismo Gospodin Gospođa Ljiljan Sunce Mjesec Ključ Ribe Sidro Križ".split(), 1))
maj = ["Mag","Visoka svećenica","Carica","Car","Jerofant","Ljubavnici","Kola","Snaga","Pustinjak","Kolo sreće","Pravda","Obješeni","Smrt","Umjerenost","Đavao","Kula","Zvijezda","Mjesec","Sunce","Sud","Svijet"]
RW = {i+1: maj[i] for i in range(21)}; RW[22] = "Luda"
rk = ["As","2","3","4","5","6","7","8","9","10","Paž","Vitez","Kraljica","Kralj"]
for si, s in enumerate(["štapova","pehara","mačeva","novčića"]):
    for r, x in enumerate(rk): RW[23 + si*14 + r] = f"{x} {s}"
gdm = ["Luda","Mag","Visoka svećenica","Carica","Car","Jerofant","Ljubavnici","Kola","Snaga","Pustinjak","Kolo sreće","Pravda","Obješeni","Smrt","Umjerenost","Đavao","Kula","Zvijezda","Mjesec","Sunce","Sud","Svemir"]
GD = {i+1: gdm[i] for i in range(22)}; n = 23
for s, rs in [("štapova",[5,6,7]),("novčića",[8,9,10]),("mačeva",[2,3,4]),("pehara",[5,6,7]),("štapova",[8,9,10]),("novčića",[2,3,4]),
              ("mačeva",[5,6,7]),("pehara",[8,9,10]),("štapova",[2,3,4]),("novčića",[5,6,7]),("mačeva",[8,9,10]),("pehara",[2,3,4])]:
    for r in rs: GD[n] = f"{r} {s}"; n += 1
for s in ["štapova","pehara","mačeva","novčića"]: GD[n] = f"As {s}"; n += 1
for s in ["štapova","pehara","mačeva","novčića"]:
    for c in ["Vitez","Kraljica","Princ","Princeza"]: GD[n] = f"{c} {s}"; n += 1

rng = secrets.SystemRandom()
def izvuci(deck, d):
    """Pravilo pretvorbe karata u brojeve."""
    if IGRA == "SIN":
        t = [c for c in d if c <= 22][:11]; return t, None
    if deck == "LEN":   # Lenormand: zbrojevi susjednih karata, > 45 → −45, duplikat se preskače
        out = []; p = []
        for i in range(35):
            s = d[i] + d[i+1]; v = s - 45 if s > 45 else s
            ok = v not in out
            p.append(f"{d[i]} {LN[d[i]]} + {d[i+1]} {LN[d[i+1]]} = {s}" + (f" − 45 = {v}" if s > 45 else "") + ("" if ok else " (već uzet, preskočeno)"))
            if ok: out.append(v)
            if len(out) == 6: break
        return out, p
    return [c for c in d if c <= 45][:6], None

karte = ([("L_SVE_1","LEN",36),("L_NIŠTA_1","LEN",36),("L_SVE_2","LEN",36),("L_NIŠTA_2","LEN",36),("RW_SVE","RW",78),("GD_NIŠTA","GD",78)]
         if IGRA == "SIN" else [("L_1","LEN",36),("L_2","LEN",36),("L_3","LEN",36),("L_4","LEN",36),("RW","RW",78),("GD","GD",78)])
DECK = {}
for nm, deck, n in karte:
    best = None; nval = 0
    for i in range(1, MIJESANJA + 1):
        d = list(range(1, n+1)); rng.shuffle(d)
        t, p = izvuci(deck, d)
        if not valid(t, list(T.values())): continue
        if IGRA == "L6" and not popular_ok(t): continue
        nval += 1; v = wv(t); gg = (U_any | v).mean()
        if best is None or gg > best[0]: best = (gg, t, v, d, i, p)
    if best is None: sys.exit(f"{nm}: nijedno od {MIJESANJA} miješanja nije prošlo provjere")
    gg, t, v, d, i, p = best
    T[nm] = t; U_any |= v
    NM = LN if deck == "LEN" else (RW if deck == "RW" else GD)
    if IGRA == "SIN" or deck != "LEN":
        lim = 22 if IGRA == "SIN" else 45
        p = [f"{c} {NM[c]}" for c in [c for c in d if c <= lim][:K]]
    DECK[nm] = dict(izabrano=i, valjanih=nval, P=round(gg, 4), redoslijed=d, postupak=p)

# ---------- Joker ----------
jd = list(range(1, 37)); rng.shuffle(jd)
b = H["本"]
JOKER = {"Meihua": "".join(map(str, [b["ben"][0], b["ben"][1], b["hu"][0], b["hu"][1], b["bian"][0], b["bian"][1]])),
         "Lenormand": "".join(str(c % 10) for c in jd[:7])[:6], "Lenormand_karte": [f"{c} {LN[c]}" for c in jd[:6]]}

# ---------- ispis ----------
print("\n================ LISTIĆI ================")
for k, t in T.items(): print(f"{k:14s} {' '.join(f'{x:2d}' for x in sorted(t))}   (redom izvučeno: {t})")
print("\nJOKER:", JOKER)
print("\n================ POSTUPAK ================")
for k, l in LOG.items():
    print(f"\n{k}:"); [print("   ", r) for r in l]
for k, v in DECK.items():
    print(f"\n{k}: izabrano miješanje #{v['izabrano']} od {MIJESANJA} (valjanih {v['valjanih']}), P(barem 1) nakon dodavanja {v['P']}")
    for r in v["postupak"]: print("   ", r)
    print("    puni redoslijed:", v["redoslijed"])

print("\n================ PROVJERE ================")
L = list(T.values())
print("Bez ponavljanja i u rasponu:", all(len(set(t)) == K and set(t) <= U for t in L))
if IGRA == "SIN":
    W = set()
    for t in L: W.add(frozenset(t)); W.add(frozenset(U - set(t)))
    print("Različitih ishoda za glavni dobitak (mora biti 20):", len(W))
    print("Nijedan = ili komplement fiksnog para:", all(set(t) not in FIX for t in L))
    print("Nijedan par listića jednak ili komplementaran:", all(set(a) != set(c) and set(a) != U - set(c) for i, a in enumerate(L) for c in L[i+1:]))
else:
    print("Najveće preklapanje dviju kombinacija:", max(len(set(a) & set(c)) for i, a in enumerate(L) for c in L[i+1:]))
    for k, t in T.items(): print(f"  filtar popularnosti {k}: {'prolazi' if popular_ok(t) else 'ne prolazi'}")

print("\n================ VJEROJATNOSTI ================")
def rand_set():
    u = np.zeros(len(D), bool)
    for _ in range(10): u |= wv(sorted(rng.sample(range(1, N+1), K)))
    return u.mean()
rs = [rand_set() for _ in range(20 if IGRA == "SIN" else 10)]
if IGRA == "SIN":
    pay = np.zeros(len(D))
    for t in L: pay += PR[hits(t)]
    print(f"P(barem 1 dobitan listić): {U_any.mean():.4f} | 10 nasumičnih: prosjek {np.mean(rs):.4f} (raspon {min(rs):.4f}–{max(rs):.4f})")
    print(f"P(dobitak ≥ ulog {CIJENA*len(L):.2f} €): {(pay >= CIJENA*len(L)-1e-9).mean():.4f}")
    print(f"Očekivani povrat: {pay.mean():.2f} € na ulog {CIJENA*len(L):.2f} € → očekivani gubitak {CIJENA*len(L)-pay.mean():.2f} €")
    print("Kvote po listiću: 11 ili 0 = 1:352.716 | 10 ili 1 = 1:2.915 | 9 ili 2 = 1:117 | 8 ili 3 = 1:13 | ukupno 1:11,61")
else:
    print(f"P(barem jedna kombinacija s 3+): {U_any.mean():.4f} | 10 nasumičnih: prosjek {np.mean(rs):.4f} (raspon {min(rs):.4f}–{max(rs):.4f})")
    print(f"Ulog {CIJENA*len(L):.2f} € ({len(L)} kombinacija × {CIJENA} €). Po kombinaciji: 6 = 1:8.145.060 | 5 = 1:34.808 | 4 = 1:733 | 3 = 1:45 | 3+ = 2,38 %")
json.dump({"T": T, "JOKER": JOKER, "DECK": DECK, "LOG": LOG}, open("rezultat.json", "w"), ensure_ascii=False)
print("\nSpremljeno u rezultat.json")
