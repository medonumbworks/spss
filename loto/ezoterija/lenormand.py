"""Lenormand (Grand Tableau) za Loto 6/45 — izvlačenje četvrtak 29.10.2026. u 20:00 CET, Zagreb.

ŠTO JE METODA
-------------
Petit Lenormand je špil od 36 karata (po Marie Anne Lenormand, 1772–1843; špil potječe iz
igre "Das Spiel der Hoffnung" J. K. Hechtela, Nürnberg 1799/1800). Svaka karta ima redni
broj 1–36 i sliku:
  1 Jahač, 2 Djetelina, 3 Brod, 4 Kuća, 5 Stablo, 6 Oblaci, 7 Zmija, 8 Lijes, 9 Buket,
  10 Kosa, 11 Bič, 12 Ptice, 13 Dijete, 14 Lisica, 15 Medvjed, 16 Zvijezde, 17 Roda,
  18 Pas, 19 Toranj, 20 Vrt, 21 Planina, 22 Raskrižje, 23 Miševi, 24 Srce, 25 Prsten,
  26 Knjiga, 27 Pismo, 28 Gospodin, 29 Gospođa, 30 Ljiljan, 31 Sunce, 32 Mjesec,
  33 Ključ, 34 Ribe, 35 Sidro, 36 Križ.
U "velikoj slici" (Grand Tableau) polaže se svih 36 karata u 4 reda po 9. Položaj 1–36
(čita se red po red, slijeva nadesno) je "kuća" karte koja tamo stoji u izvornom
redoslijedu (kuća 1 = Jahač, …, kuća 34 = Ribe, kuća 36 = Križ). Tri klasične tehnike:
susjedstvo (karte koje se dodiruju u nizu), kuće (koja je karta pala u čiju kuću) i
9-karatni kvadrat (3×3) oko značajne karte. Za novac je značajna karta 34 RIBE, za sreću
2 DJETELINA, za uspjeh 31 SUNCE, za nadu/vodstvo 16 ZVIJEZDE, za rješenje 33 KLJUČ, za
"vezanje" dobitka 25 PRSTEN, za postojanost 35 SIDRO.

IZVORI / PRAVILA
----------------
(a) Referentno pravilo iz korisnikove skripte (data/loto_ezoterija_original.py, izvuci()
    za deck "LEN"): špil se promiješa, zbrajaju se SUSJEDNE karte d[i]+d[i+1] (i = 0…34),
    zbroj > 45 → −45, duplikat se preskače, uzima se prvih 6 brojeva.
(b) KUĆE: karta koja je pala u kuću Ribe (34), Djeteline (2), Sunca (31), Zvijezda (16),
    Ključa (33), Prstena (25) i Sidra (35) — tim redom (novac, sreća, uspjeh, nada,
    rješenje, ugovor, stabilnost). Iz svake kuće h s kartom c nastaju dva broja: broj
    karte c i zbroj c+h (> 45 → −45). Tri izvedbe: (b1) samo brojevi karata u kućama,
    (b2) samo zbrojevi karta+kuća, (b3) naizmjence karta/zbroj po kućama Ribe → Djetelina
    → Sunce → … dok nema 6 različitih.
(c) KVADRAT 3×3 oko RIBA i oko DJETELINE: najprije središnja karta, pa 8 susjeda u
    redoslijedu čitanja (gore-lijevo → dolje-desno), uzimaju se brojevi karata; kad je
    značajna karta na rubu ili u kutu i kvadrat ima < 6 karata, dopunjuje se zbrojevima
    središte+susjed (> 45 → −45) istim redom.
Miješanje je jedini izvor slučajnosti i vezano je uz datum: čitanje i koristi
random.Random(20261029 + i), i = 0 … 99 (sto velikih slika), nikad SystemRandom.

RANGIRANJE
----------
Ocjena čitanja = Σ težina(karta) · centralnost(karta) za ključne karte Djetelina (1,0),
Ribe (1,0), Sunce (0,8), Zvijezde (0,7), Ključ (0,7); centralnost = 1 − d/d_max, gdje je d
euklidska udaljenost položaja karte od središta slike (red 1,5, stupac 4 u indeksima od
0; d_max = √(1,5² + 4²) ≈ 4,27). Bonusi: ključna karta u jednoj od sedam značajnih kuća
+0,15; dvije ključne karte koje se dodiruju (i dijagonalno) +0,10 po paru. Kandidat =
ocjena čitanja × težina izvedbe: (a) susjedni zbrojevi 1,00; (b3) 0,92; (b1) 0,88;
(b2) 0,84; (c) kvadrat Ribe 0,80; (c) kvadrat Djeteline 0,76 — jače izvedbe iz
istog čitanja idu prije slabijih, a čitanja s ključnim kartama bliže središtu prije
ostalih. Ne prima se kandidat koji s već prihvaćenim dijeli 5 ili 6 brojeva
(raznolikost), inače nema nikakva statističkog filtriranja.

POSTUPAK ZA 29.10.2026. 20:00 (sažeto; cijeli izvod s prvih 10 velikih slika,
svim zbrojevima i kućama generira postupak_tekst() i zapisuje se u out/ezo_lenormand.json)
1. seed_i = 20261029 + i (datum 29.10.2026. kao 20261029), i = 0 … 99.
2. Za svaki i: d = [1..36], random.Random(seed_i).shuffle(d); d je velika slika
   (d[0..8] red 1 = kuće 1–9, d[9..17] red 2 = kuće 10–18, d[18..26] red 3 = kuće 19–27,
   d[27..35] red 4 = kuće 28–36).
3. Izvedbe (a), (b1), (b2), (b3), (c-Ribe), (c-Djetelina) → do 6 kandidata po čitanju.
4. Ocjena čitanja po položaju ključnih karata; kandidati se sortiraju po ocjena ×
   težina izvedbe (nerješeno: manji i, pa redoslijed izvedbi), uz pravilo raznolikosti.
5. Ključni brojevi = brojevi sedam značajnih karata (34, 2, 31, 16, 33, 25, 35) + brojevi
   koji se najčešće pojavljuju među prvih 30 rangiranih kandidata.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

SLUG = "lenormand"
METODA = "Lenormand"
DATUM_TEKST = "četvrtak 29.10.2026. u 20:00 CET (sat Jupitera), Zagreb"
SEED0 = 20261029
N_CITANJA = 100
OUT_DIR = Path(__file__).resolve().parent.parent / "out"

# --------------------------------------------------------------------------- karte
KARTE = dict(enumerate(
    "Jahač Djetelina Brod Kuća Stablo Oblaci Zmija Lijes Buket Kosa Bič Ptice Dijete Lisica "
    "Medvjed Zvijezde Roda Pas Toranj Vrt Planina Raskrižje Miševi Srce Prsten Knjiga Pismo "
    "Gospodin Gospođa Ljiljan Sunce Mjesec Ključ Ribe Sidro Križ".split(), 1))
assert len(KARTE) == 36

# značajne kuće — redoslijed = prioritet čitanja (novac, sreća, uspjeh, nada, rješenje, ugovor, stabilnost)
ZNACAJNE_KUCE: list[tuple[int, str]] = [
    (34, "novac"), (2, "sreća"), (31, "uspjeh"), (16, "nada"), (33, "rješenje"),
    (25, "ugovor"), (35, "stabilnost"),
]
# ključne karte za ocjenu čitanja i njihove težine
KLJUCNE_KARTE: dict[int, float] = {2: 1.0, 34: 1.0, 31: 0.8, 16: 0.7, 33: 0.7}
BONUS_KUCA = 0.15      # ključna karta pala u značajnu kuću
BONUS_SUSJED = 0.10    # dvije ključne karte dodiruju se (8-susjedstvo)

# težine izvedbi (jače prije slabijih)
TEZINA_IZVEDBE: dict[str, float] = {
    "a": 1.00, "b3": 0.92, "b1": 0.88, "b2": 0.84, "c34": 0.80, "c2": 0.76,
}
NAZIV_IZVEDBE: dict[str, str] = {
    "a": "susjedni zbrojevi", "b1": "karte u značajnim kućama", "b2": "karta+kuća",
    "b3": "naizmjence karta/zbroj (Ribe→Djetelina→Sunce…)", "c34": "kvadrat 3×3 oko Riba",
    "c2": "kvadrat 3×3 oko Djeteline",
}
REDOVA, STUPACA = 4, 9
SREDISTE = ((REDOVA - 1) / 2, (STUPACA - 1) / 2)          # (1.5, 4.0)
D_MAX = math.hypot(SREDISTE[0], SREDISTE[1])             # ≈ 4.272


def ime(c: int) -> str:
    return f"{c} {KARTE[c]}"


def mod45(s: int) -> int:
    """Pravilo iz korisnikove skripte: zbroj > 45 → −45 (najveći mogući zbroj je 36+36 = 72)."""
    return s - 45 if s > 45 else s


def polozaj(p: int) -> tuple[int, int]:
    """Indeks u špilu 0..35 → (red, stupac) velike slike 4×9."""
    return divmod(p, STUPACA)


def centralnost(p: int) -> float:
    r, c = polozaj(p)
    return 1.0 - math.hypot(r - SREDISTE[0], c - SREDISTE[1]) / D_MAX


# --------------------------------------------------------------------------- izvedbe
def _prvih6(niz: list[int]) -> list[int]:
    out: list[int] = []
    for v in niz:
        if 1 <= v <= 45 and v not in out:
            out.append(v)
        if len(out) == 6:
            break
    return out


def izvedba_a(d: list[int]) -> tuple[list[int], list[str]]:
    """(a) Referentno pravilo: zbrojevi susjednih karata, > 45 → −45, duplikat se preskače, prvih 6."""
    out: list[int] = []
    log: list[str] = []
    for i in range(35):
        s = d[i] + d[i + 1]
        v = mod45(s)
        ok = v not in out
        log.append(f"{ime(d[i])} + {ime(d[i + 1])} = {s}" + (f" − 45 = {v}" if s > 45 else "")
                   + ("" if ok else " (već uzet, preskočeno)"))
        if ok:
            out.append(v)
        if len(out) == 6:
            break
    return out, log


def kuce(d: list[int]) -> list[dict]:
    """(b) Za svaku značajnu kuću: koja karta je pala u nju, broj karte i zbroj karta+kuća."""
    out = []
    for h, znac in ZNACAJNE_KUCE:
        c = d[h - 1]
        out.append(dict(kuca=h, znacenje=znac, karta=c, zbroj=c + h, broj=mod45(c + h)))
    return out


def izvedba_b(d: list[int]) -> dict[str, tuple[list[int], list[str]]]:
    K = kuce(d)
    log = [f"kuća {ime(k['kuca'])} ({k['znacenje']}): karta {ime(k['karta'])} → broj {k['karta']}; "
           f"{k['karta']}+{k['kuca']} = {k['zbroj']}" + (f" − 45 = {k['broj']}" if k['zbroj'] > 45 else "")
           for k in K]
    b1 = _prvih6([k["karta"] for k in K])
    b2 = _prvih6([k["broj"] for k in K])
    mix: list[int] = []
    for k in K:
        mix += [k["karta"], k["broj"]]
    b3 = _prvih6(mix)
    return {"b1": (b1, log), "b2": (b2, log), "b3": (b3, log)}


def kvadrat(d: list[int], karta: int) -> tuple[list[int], list[str], list[int]]:
    """(c) 3×3 kvadrat oko zadane karte: središte, pa susjedi redoslijedom čitanja; dopuna zbrojevima."""
    p = d.index(karta)
    r0, c0 = polozaj(p)
    susjedi: list[int] = []
    for r in (r0 - 1, r0, r0 + 1):
        for c in (c0 - 1, c0, c0 + 1):
            if (r, c) != (r0, c0) and 0 <= r < REDOVA and 0 <= c < STUPACA:
                susjedi.append(d[r * STUPACA + c])
    niz = [karta] + susjedi
    log = [f"{ime(karta)} je na položaju {p + 1} (red {r0 + 1}, stupac {c0 + 1}, kuća {ime(p + 1)}); "
           f"kvadrat: " + ", ".join(ime(x) for x in susjedi)]
    brojevi = _prvih6(niz)
    if len(brojevi) < 6:
        dopuna = [mod45(karta + s) for s in susjedi]
        log.append("rubni položaj (< 6 karata) → dopuna zbrojevima središte+susjed: "
                   + ", ".join(f"{karta}+{s}={karta + s}" + (f"−45={mod45(karta + s)}" if karta + s > 45 else "")
                               for s in susjedi))
        brojevi = _prvih6(niz + dopuna)
    return brojevi, log, niz


# --------------------------------------------------------------------------- ocjena čitanja
def ocjena_citanja(d: list[int]) -> tuple[float, list[str]]:
    poz = {c: d.index(c) for c in KLJUCNE_KARTE}
    znac_kuce = {h for h, _ in ZNACAJNE_KUCE}
    log: list[str] = []
    s = 0.0
    for c, w in KLJUCNE_KARTE.items():
        p = poz[c]
        r, col = polozaj(p)
        cen = centralnost(p)
        s += w * cen
        txt = f"{ime(c)}: položaj {p + 1} (red {r + 1}, st. {col + 1}), centralnost {cen:.2f} × {w:.1f} = {w * cen:.2f}"
        if (p + 1) in znac_kuce:
            s += BONUS_KUCA
            txt += f"; u značajnoj kući {ime(p + 1)} +{BONUS_KUCA:.2f}"
        log.append(txt)
    karte = sorted(KLJUCNE_KARTE)
    for i, a in enumerate(karte):
        for b in karte[i + 1:]:
            ra, ca = polozaj(poz[a])
            rb, cb = polozaj(poz[b])
            if max(abs(ra - rb), abs(ca - cb)) == 1:
                s += BONUS_SUSJED
                log.append(f"{KARTE[a]} i {KARTE[b]} se dodiruju +{BONUS_SUSJED:.2f}")
    return round(s, 4), log


# --------------------------------------------------------------------------- čitanje
def citanje(i: int) -> dict:
    """Jedna velika slika: seed 20261029 + i, sve izvedbe, ocjena."""
    seed = SEED0 + i
    rng = random.Random(seed)
    d = list(range(1, 37))
    rng.shuffle(d)
    a, log_a = izvedba_a(d)
    B = izvedba_b(d)
    c34, log_c34, kv34 = kvadrat(d, 34)
    c2, log_c2, kv2 = kvadrat(d, 2)
    ocj, log_o = ocjena_citanja(d)
    izvedbe = [
        dict(oznaka="a", brojevi=a, log=log_a),
        dict(oznaka="b3", brojevi=B["b3"][0], log=B["b3"][1]),
        dict(oznaka="b1", brojevi=B["b1"][0], log=B["b1"][1]),
        dict(oznaka="b2", brojevi=B["b2"][0], log=B["b2"][1]),
        dict(oznaka="c34", brojevi=c34, log=log_c34, kvadrat=kv34),
        dict(oznaka="c2", brojevi=c2, log=log_c2, kvadrat=kv2),
    ]
    for iz in izvedbe:
        iz["tezina"] = TEZINA_IZVEDBE[iz["oznaka"]]
        iz["ocjena"] = round(ocj * iz["tezina"], 4)
    return dict(i=i, seed=seed, d=d, kuce=kuce(d), ocjena=ocj, ocjena_log=log_o, izvedbe=izvedbe)


def _obrazlozenje(ct: dict, iz: dict) -> str:
    d = ct["d"]
    o = iz["oznaka"]
    if o == "a":
        parovi = []
        out: list[int] = []
        for k in range(35):
            v = mod45(d[k] + d[k + 1])
            if v not in out:
                out.append(v)
                parovi.append(f"{d[k]}+{d[k + 1]}={v}")
            if len(out) == 6:
                break
        detalj = "susjedni zbrojevi " + ", ".join(parovi)
    elif o in ("b1", "b2", "b3"):
        # ponovi izvedbu istim redom kao _prvih6 i zabilježi koja kuća je dala koji broj
        out: list[int] = []
        dijelovi = []
        for k in ct["kuce"]:
            zb = f"{k['karta']}+{k['kuca']}={k['broj']}"
            stavke = {"b1": [("k", k["karta"])], "b2": [("z", k["broj"])],
                      "b3": [("k", k["karta"]), ("z", k["broj"])]}[o]
            novi = []
            for vrsta, v in stavke:
                if len(out) < 6 and v not in out:
                    out.append(v)
                    novi.append(vrsta)
            if novi:
                if o == "b1":
                    dijelovi.append(f"{KARTE[k['kuca']]}←{k['karta']}")
                elif o == "b2":
                    dijelovi.append(f"{KARTE[k['kuca']]}:{zb}")
                else:
                    dijelovi.append(f"{KARTE[k['kuca']]}←{k['karta']}" + (f" ({zb})" if "z" in novi else ""))
            if len(out) == 6:
                break
        assert sorted(out) == sorted(iz["brojevi"]), (o, out, iz["brojevi"])
        detalj = NAZIV_IZVEDBE[o] + ": " + ", ".join(dijelovi)
    else:
        kv = iz["kvadrat"]
        detalj = NAZIV_IZVEDBE[o] + f" (položaj {d.index(kv[0]) + 1}): " + ", ".join(ime(x) for x in kv[:6])
    return f"Velika slika #{ct['i']} (seed {ct['seed']}, ocjena {ct['ocjena']:.2f}×{iz['tezina']:.2f}): {detalj}"


# --------------------------------------------------------------------------- kandidati
_CACHE: dict[int, dict] = {}


def sva_citanja() -> list[dict]:
    if not _CACHE:
        for i in range(N_CITANJA):
            _CACHE[i] = citanje(i)
    return [_CACHE[i] for i in range(N_CITANJA)]


def kandidati(n: int = 80, max_preklop: int = 4) -> list[dict]:
    """Rangirani kandidati: ocjena čitanja × težina izvedbe; raznolikost: ≤ max_preklop zajedničkih brojeva."""
    bazen = []
    for ct in sva_citanja():
        for k, iz in enumerate(ct["izvedbe"]):
            if len(iz["brojevi"]) == 6:
                bazen.append((-iz["ocjena"], ct["i"], k, ct, iz))
    bazen.sort(key=lambda t: (t[0], t[1], t[2]))
    out: list[dict] = []
    prihvaceni: list[set[int]] = []
    for _, _, _, ct, iz in bazen:
        t = sorted(iz["brojevi"])
        assert len(t) == 6 and len(set(t)) == 6 and all(1 <= x <= 45 for x in t), t
        st = set(t)
        if any(len(st & p) > max_preklop for p in prihvaceni):
            continue
        prihvaceni.append(st)
        out.append({"brojevi": t, "rang": len(out) + 1, "obrazlozenje": _obrazlozenje(ct, iz)})
        if len(out) >= n:
            break
    return out


def kljucni_brojevi(k_dodatnih: int = 5) -> list[int]:
    """Sedam značajnih karata + brojevi najčešći među prvih 30 rangiranih kandidata."""
    kljuc = [h for h, _ in ZNACAJNE_KUCE]
    cnt: Counter = Counter()
    for kd in kandidati(80)[:30]:
        cnt.update(kd["brojevi"])
    for x, _ in sorted(cnt.items(), key=lambda t: (-t[1], t[0])):
        if x not in kljuc:
            kljuc.append(x)
        if len(kljuc) >= len(ZNACAJNE_KUCE) + k_dodatnih:
            break
    return kljuc


# --------------------------------------------------------------------------- postupak (tekst)
def _tablica_slike(d: list[int]) -> list[str]:
    L = ["| red \\ kuća | " + " | ".join(f"k{c + 1}" for c in range(STUPACA)) + " |",
         "|---|" + "---|" * STUPACA]
    for r in range(REDOVA):
        celije = []
        for c in range(STUPACA):
            p = r * STUPACA + c
            x = d[p]
            tag = f"**{ime(x)}**" if x in KLJUCNE_KARTE else ime(x)
            celije.append(f"{tag} (kuća {p + 1})")
        L.append(f"| red {r + 1} | " + " | ".join(celije) + " |")
    return L


def postupak_tekst(n_detaljno: int = 10) -> str:
    L: list[str] = []
    L.append(f"# Lenormand — Grand Tableau za Loto 6/45, {DATUM_TEKST}\n")
    L.append("## 1. Metoda\n")
    L.append("Petit Lenormand, 36 karata: " + ", ".join(ime(c) for c in range(1, 37)) + ".\n")
    L.append("Velika slika 4×9: položaj p (1–36, red po red) je kuća karte p (kuća 1 = Jahač … kuća 36 = Križ). "
             "Značajne kuće (prioritet): " + ", ".join(f"{ime(h)} = {z}" for h, z in ZNACAJNE_KUCE) + ".\n")
    L.append("## 2. Pravila izvedbe\n")
    L.append("- **(a) susjedni zbrojevi** (referentno pravilo iz korisnikove skripte, deck LEN): d[i]+d[i+1], "
             "i = 0…34; zbroj > 45 → −45; duplikat se preskače; prvih 6.")
    L.append("- **(b) kuće**: karta c u značajnoj kući h daje broj c i zbroj c+h (> 45 → −45). "
             "(b1) samo brojevi karata, (b2) samo zbrojevi, (b3) naizmjence karta/zbroj redom Ribe → Djetelina → Sunce → "
             "Zvijezde → Ključ → Prsten → Sidro, dok nema 6 različitih.")
    L.append("- **(c) kvadrat 3×3** oko Riba (34) i oko Djeteline (2): središte, pa susjedi redoslijedom čitanja "
             "(gore-lijevo → dolje-desno), brojevi karata; na rubu/kutu (< 6 karata) dopuna zbrojevima središte+susjed.")
    L.append(f"- **Miješanje**: čitanje i → random.Random({SEED0} + i).shuffle([1..36]), i = 0…{N_CITANJA - 1}. "
             "Nikad SystemRandom; isti datum → iste slike.\n")
    L.append("## 3. Rangiranje\n")
    L.append("Ocjena čitanja = Σ težina·centralnost za " + ", ".join(f"{KARTE[c]} ({w:.1f})" for c, w in KLJUCNE_KARTE.items())
             + f"; centralnost = 1 − d/d_max, d = euklidska udaljenost od središta (red {SREDISTE[0] + 1}, stupac {SREDISTE[1] + 1}), "
             f"d_max = {D_MAX:.3f}. Bonus +{BONUS_KUCA} ključna karta u značajnoj kući, +{BONUS_SUSJED} po paru ključnih karata koje se dodiruju.")
    L.append("Kandidat = ocjena čitanja × težina izvedbe: " + ", ".join(f"{NAZIV_IZVEDBE[o]} {w:.2f}" for o, w in TEZINA_IZVEDBE.items())
             + ". Nerješeno → manji i, pa redoslijed izvedbi. Raznolikost: kandidat koji s već prihvaćenim dijeli 5+ brojeva se preskače.\n")

    L.append(f"## 4. Prvih {n_detaljno} velikih slika (i = 0 … {n_detaljno - 1}) — potpuni izvod\n")
    for ct in sva_citanja()[:n_detaljno]:
        d = ct["d"]
        L.append(f"### Velika slika #{ct['i']} (seed {ct['seed']}) — ocjena {ct['ocjena']:.3f}\n")
        L.append("Redoslijed špila: " + " ".join(map(str, d)) + "\n")
        L += _tablica_slike(d)
        L.append("")
        L.append("Ocjena čitanja:")
        L += [f"- {x}" for x in ct["ocjena_log"]]
        L.append("")
        for iz in ct["izvedbe"]:
            o = iz["oznaka"]
            L.append(f"**({o}) {NAZIV_IZVEDBE[o]}** → {' · '.join(f'{x:02d}' for x in sorted(iz['brojevi']))} "
                     f"(redom izvedeno {iz['brojevi']}; ocjena {ct['ocjena']:.3f} × {iz['tezina']:.2f} = {iz['ocjena']:.3f})")
            if o in ("a", "b3", "c34", "c2"):          # b1/b2 dijele isti zapis kuća kao b3
                L += [f"- {x}" for x in iz["log"]]
            L.append("")

    L.append(f"## 5. Sva čitanja — ocjene i položaji ključnih karata (položaj 1–36 = kuća)\n")
    L.append("| i | seed | ocjena | Djetelina | Ribe | Sunce | Zvijezde | Ključ | (a) susjedni zbrojevi |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for ct in sorted(sva_citanja(), key=lambda c: (-c["ocjena"], c["i"])):
        d = ct["d"]
        poz = " | ".join(str(d.index(c) + 1) for c in (2, 34, 31, 16, 33))
        a = ct["izvedbe"][0]["brojevi"]
        L.append(f"| {ct['i']} | {ct['seed']} | {ct['ocjena']:.3f} | {poz} | {' '.join(f'{x:02d}' for x in sorted(a))} |")
    L.append("")
    L.append("## 6. Ključni brojevi\n")
    L.append("Sedam značajnih karata (" + ", ".join(ime(h) for h, _ in ZNACAJNE_KUCE) +
             ") + najčešći brojevi među prvih 30 rangiranih kandidata: " + ", ".join(map(str, kljucni_brojevi())) + "\n")
    return "\n".join(L)


# --------------------------------------------------------------------------- provjere
def _provjere() -> None:
    kand = kandidati(80)
    assert len(kand) >= 60, len(kand)
    skupovi = [tuple(k["brojevi"]) for k in kand]
    assert len(set(skupovi)) == len(skupovi), "duplikati"
    for k in kand:
        b = k["brojevi"]
        assert b == sorted(b) and len(set(b)) == 6 and all(isinstance(x, int) and 1 <= x <= 45 for x in b), b
    razl = {x for k in kand[:60] for x in k["brojevi"]}
    assert len(razl) >= 30, f"premalo različitih brojeva u prvih 60: {len(razl)}"
    assert [k["rang"] for k in kand] == list(range(1, len(kand) + 1))
    # determinizam
    assert kandidati(80) == kand
    # referentno pravilo (a) na poznatom špilu
    a, _ = izvedba_a(list(range(1, 37)))
    assert a == [3, 5, 7, 9, 11, 13], a
    # kvadrat u kutu dopunjuje se zbrojevima
    d = list(range(1, 37))
    d[0], d[33] = 34, 1
    b, _, kv = kvadrat(d, 34)
    assert len(kv) == 4 and len(b) == 6, (kv, b)


if __name__ == "__main__":
    _provjere()
    tekst = postupak_tekst()
    print(tekst)
    kand = kandidati(80)
    print("\n## Prvih 10 kandidata\n")
    for k in kand[:10]:
        print(f"{k['rang']:3d}. {' · '.join(f'{x:02d}' for x in k['brojevi'])}  — {k['obrazlozenje']}")
    kljuc = kljucni_brojevi()
    print("\nKljučni brojevi:", kljuc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"ezo_{SLUG}.json"
    out.write_text(json.dumps({"metoda": METODA, "slug": SLUG, "postupak": tekst,
                               "kljucni_brojevi": kljuc, "kandidati": kand},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nZapisano: {out}  ({len(kand)} kandidata)")
