"""I Ching (Yijing, klasični King Wen poredak) za Loto 6/45 — izvlačenje četvrtak 29.10.2026. u 20:00 CET, Zagreb.

ŠTO JE METODA
-------------
I Ching (易經, "Knjiga promjena") sustav je 64 heksagrama; svaki heksagram čini šest linija
(yang ⚊ = 1, yin ⚋ = 0) koje se broje ODOZDO prema gore (1. = donja, 6. = gornja). Linije
nastaju bacanjem: metoda tri novčića (Tang/Song dinastija, 火珠林 "Huo Zhu Lin") daje po liniji
vrijednost 6 (stara yin, pomična), 7 (mlada yang), 8 (mlada yin) ili 9 (stara yang, pomična) s
vjerojatnostima 1/8, 3/8, 3/8, 1/8; metoda stolisnika (蓍 — 50 stabljika, Zhu Xi "Yixue Qimeng",
1186.) daje iste vrijednosti s vjerojatnostima 1/16, 5/16, 7/16, 3/16. Pomične linije (6, 9) se
preokreću i daju "relacijski" (promijenjeni) heksagram. Klasične transformacije heksagrama:
  - nuklearni (互卦 hu gua): linije 2-3-4 čine donji, 3-4-5 gornji trigram;
  - suprotni (錯卦 cuo gua): sve linije preokrenute (komplement);
  - obrnuti (綜卦 zong gua): heksagram okrenut naglavce (linije u obrnutom redu).
Ovo je KLASIČNI Yijing (King Wen brojevi 1..64), NE Meihua Yishu (koji je zasebna metoda).

IZVORI / PRAVILA
----------------
  - King Wen poredak i imena: Wilhelm/Baynes "The I Ching or Book of Changes" (1950), Legge (1882).
  - Trigrami (linije odozdo; Fuxi/"prijenebeski" broj 1..8):
      乾 Qian 111 = 1   兌 Dui 110 = 2   離 Li 101 = 3   震 Zhen 100 = 4
      巽 Xun 011 = 5    坎 Kan 010 = 6   艮 Gen 001 = 7  坤 Kun 000 = 8
    (Fuxi broj = 8 − (4·l1 + 2·l2 + l3)).
  - Binarna (Shao Yong / Leibniz 1703 "Explication de l'arithmétique binaire") vrijednost heksagrama:
      vrijednost = Σ l_i · 2^(6−i)  (DONJA linija je najznačajniji bit, GORNJA najmanje značajan),
      pa je 坤 = 0, 剝 = 1, 比 = 2, 觀 = 3, … 乾 = 63; "Shao Yongov redni broj" = vrijednost + 1 (1..64).
  - Smjer linija: tablica King Wen ↔ linije ovdje je zapisana ODOZDO PREMA GORE, npr.
      1 乾 = 111111, 2 坤 = 000000, 11 泰 = 111000 (Qian dolje, Kun gore), 12 否 = 000111,
      63 既濟 = 101010 (odozdo; zapisano odozgo je 010101), 64 未濟 = 010101 (odozdo).
    Tablica se u bloku __main__ provjerava svojstvom King Wen poretka: svaki par (2k−1, 2k) čine
    međusobno obrnuti (zong) heksagrami, a ako je heksagram sam sebi obrnut, par čine suprotni (cuo).
  - Imena linija (klasično 初九, 九二, 六三, …): pozicija p (1..6) i vrijednost v (9 = yang, 6 = yin)
    daju broj 10·p + v, npr. 九二 → 29, 六五 → 56.
  - Broj trigrama heksagrama: 10·(Fuxi gornjeg) + Fuxi donjeg (tradicionalno se gornji trigram
    imenuje prvi: 水天需 = Kan nad Qian → 61).
  - Svođenje u 1..45 (pravilo "svedi"): x → ((x − 1) mod 45) + 1, tj. 46..64 → 1..19, 65..90 → 20..45.
  - Slučajnost samo kod bacanja: random.Random(20261029 + i), i = 0, 1, 2, …; parni i = tri novčića
    (svaki novčić rng.randint(2, 3): 2 = yin/strana s natpisom, 3 = yang), neparni i = stolisnik
    (rng.randrange(16): 0 → 6, 1..5 → 7, 6..12 → 8, 13..15 → 9). Nikad SystemRandom.

POSTUPAK ZA 29.10.2026. 20:00 CET (četvrtak, 87. kolo) — sažetak; cijeli izvod daje
postupak_tekst() i zapisuje se u out/ezo_iching.json.

(a) DATUMSKI HEKSAGRAMI  (linije odozdo; hu = nuklearni, cuo = suprotni, zong = obrnuti,
    bin = Shao Yongov redni broj (binarna vrijednost + 1), trig = 10·gornji + donji Fuxi, → = svedi)
   dan 29      → 29 坎 Kan   Ponor        010010  Kan(6)/Kan(6)   hu 27 頤   cuo 30 離   zong 29 (sam)  bin 18+1=19   trig 66→21
   mjesec 10   → 10 履 Lü    Stupanje     110111  Dui(2)/Qian(1)  hu 37 家人 cuo 15 謙   zong 9 小畜    bin 55+1=56→11 trig 12
   sat 20      → 20 觀 Guan  Promatranje  000011  Kun(8)/Xun(5)   hu 23 剝   cuo 34 大壯 zong 19 臨     bin 3+1=4     trig 58→13
   godina '26  → 26 大畜 Da Chu Ukroćena snaga velikog 111001 Qian(1)/Gen(7) hu 54 歸妹→9 cuo 45 萃 zong 25 無妄 bin 57+1=58→13 trig 71→26
   2026 mod 64 → 42 益 Yi    Povećanje    100011  Zhen(4)/Xun(5)  hu 23 剝   cuo 32 恆   zong 41 損     bin 35+1=36   trig 54→9
   2+9 = 11    → 11 泰 Tai   Mir          111000  Qian(1)/Kun(8)  hu 54 歸妹→9 cuo 12 否 zong 12 否    bin 56+1=57→12 trig 81→36
   fadik 22    → 22 賁 Bi    Ljupkost     101001  Li(3)/Gen(7)    hu 40 解   cuo 47 困→2 zong 21 噬嗑   bin 41+1=42   trig 73→28
   kolo 87→23  → 23 剝 Bo    Raspadanje   000001  Kun(8)/Gen(7)   hu 2 坤    cuo 43 夬   zong 24 復     bin 1+1=2     trig 78→33
   sporedni:  1 乾 (1+0 mjeseca) 111111; 2 坤 (prva znamenka) 000000; 4 蒙 Meng (četvrtak = 4. dan) 010001
   Imena linija (10p+v): 29 坎 → 16 29 36 46→1 59→14 66→21;  10 履 → 19 29 36 49→4 59→14 69→24;
   20 觀 → 16 26 36 1 14 24;  26 大畜 → 19 29 39 1 11 24;  42 益 → 19 26 36 1 14 24;  11 泰 → 19 29 39 1 11 21;
   22 賁 → 19 26 39 1 11 24;  23 剝 → 16 26 36 1 11 24.

(b) KANDIDATI IZ DATUMSKIH HEKSAGRAMA (rang 1 = najjače)
   1. King Wen brojevi glavnih datumskih heksagrama: 10 11 20 26 29 42
   2. King Wen brojevi s drugim slojem datuma (fadik 22, kolo 23): 10 20 22 23 26 29
   3. nuklearni (hu) svih osam: 27 37 23 9 40 2 → 2 9 23 27 37 40
   4. suprotni (cuo): 30 15 34 45 32 12 → 12 15 30 32 34 45
   5. obrnuti (zong): 29 9 19 25 41 12 → 9 12 19 25 29 41
   6. Shao Yongovi binarni brojevi (prvih 6 različitih): 19 11 4 13 36 12 → 4 11 12 13 19 36
   7. brojevi trigrama: 21 12 13 26 9 36
   8.–18. "obitelj" svakog heksagrama (KW, hu, cuo, zong, bin, trig, imena linija — prvih 6 različitih),
   19.–29. imena linija svakog heksagrama (točno 6 brojeva po heksagramu).

(c) BACANJA: i = 0, 1, 2, … (seed 20261029 + i; parni novčići, neparni stolisnik); svako bacanje:
   primarni heksagram, pomične linije, relacijski heksagram → brojevi redom: KW primarnog, KW
   relacijskog, imena pomičnih linija (10p+v), hu primarnog, bin primarnog, zbroj vrijednosti linija
   (36..54), cuo, zong, bin relacijskog, trig — prvih 6 različitih nakon svođenja. Duplikati se
   preskaču i bacanje se nastavlja dok se ne skupi n kandidata.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

SLUG = "iching"
METODA = "I Ching"
DATUM_TEKST = "četvrtak 29.10.2026. u 20:00 CET (87. kolo), Zagreb"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"
SEED0 = 20261029

# --------------------------------------------------------------------------- trigrami
# ime: (linije odozdo, Fuxi broj, znak, hrvatski)
TRIGRAMI = {
    "Qian": ((1, 1, 1), 1, "乾", "Nebo"),
    "Dui":  ((1, 1, 0), 2, "兌", "Jezero"),
    "Li":   ((1, 0, 1), 3, "離", "Vatra"),
    "Zhen": ((1, 0, 0), 4, "震", "Grom"),
    "Xun":  ((0, 1, 1), 5, "巽", "Vjetar"),
    "Kan":  ((0, 1, 0), 6, "坎", "Voda"),
    "Gen":  ((0, 0, 1), 7, "艮", "Planina"),
    "Kun":  ((0, 0, 0), 8, "坤", "Zemlja"),
}
TRIGRAM_OD_LINIJA = {v[0]: k for k, v in TRIGRAMI.items()}

# --------------------------------------------------------------------------- King Wen tablica
# (King Wen broj, donji trigram, gornji trigram, znak, pinyin, hrvatski naziv)
KING_WEN = [
    (1, "Qian", "Qian", "乾", "Qian", "Stvaralačko"),
    (2, "Kun", "Kun", "坤", "Kun", "Primajuće"),
    (3, "Zhen", "Kan", "屯", "Zhun", "Početna teškoća"),
    (4, "Kan", "Gen", "蒙", "Meng", "Mladenačka ludost"),
    (5, "Qian", "Kan", "需", "Xu", "Čekanje"),
    (6, "Kan", "Qian", "訟", "Song", "Sukob"),
    (7, "Kan", "Kun", "師", "Shi", "Vojska"),
    (8, "Kun", "Kan", "比", "Bi", "Zajedništvo"),
    (9, "Qian", "Xun", "小畜", "Xiao Chu", "Ukroćena snaga malog"),
    (10, "Dui", "Qian", "履", "Lü", "Stupanje"),
    (11, "Qian", "Kun", "泰", "Tai", "Mir"),
    (12, "Kun", "Qian", "否", "Pi", "Zastoj"),
    (13, "Li", "Qian", "同人", "Tong Ren", "Zajednica s ljudima"),
    (14, "Qian", "Li", "大有", "Da You", "Posjedovanje u velikom"),
    (15, "Gen", "Kun", "謙", "Qian", "Skromnost"),
    (16, "Kun", "Zhen", "豫", "Yu", "Oduševljenje"),
    (17, "Zhen", "Dui", "隨", "Sui", "Slijeđenje"),
    (18, "Xun", "Gen", "蠱", "Gu", "Rad na pokvarenom"),
    (19, "Dui", "Kun", "臨", "Lin", "Približavanje"),
    (20, "Kun", "Xun", "觀", "Guan", "Promatranje"),
    (21, "Zhen", "Li", "噬嗑", "Shi He", "Zagriz"),
    (22, "Li", "Gen", "賁", "Bi", "Ljupkost"),
    (23, "Kun", "Gen", "剝", "Bo", "Raspadanje"),
    (24, "Zhen", "Kun", "復", "Fu", "Povratak"),
    (25, "Zhen", "Qian", "無妄", "Wu Wang", "Nedužnost"),
    (26, "Qian", "Gen", "大畜", "Da Chu", "Ukroćena snaga velikog"),
    (27, "Zhen", "Gen", "頤", "Yi", "Hranjenje"),
    (28, "Xun", "Dui", "大過", "Da Guo", "Prevaga velikog"),
    (29, "Kan", "Kan", "坎", "Kan", "Ponor"),
    (30, "Li", "Li", "離", "Li", "Prianjanje (Vatra)"),
    (31, "Gen", "Dui", "咸", "Xian", "Utjecaj"),
    (32, "Xun", "Zhen", "恆", "Heng", "Trajanje"),
    (33, "Gen", "Qian", "遯", "Dun", "Povlačenje"),
    (34, "Qian", "Zhen", "大壯", "Da Zhuang", "Snaga velikog"),
    (35, "Kun", "Li", "晉", "Jin", "Napredak"),
    (36, "Li", "Kun", "明夷", "Ming Yi", "Pomračenje svjetla"),
    (37, "Li", "Xun", "家人", "Jia Ren", "Obitelj"),
    (38, "Dui", "Li", "睽", "Kui", "Suprotnost"),
    (39, "Gen", "Kan", "蹇", "Jian", "Zapreka"),
    (40, "Kan", "Zhen", "解", "Xie", "Oslobođenje"),
    (41, "Dui", "Gen", "損", "Sun", "Smanjenje"),
    (42, "Zhen", "Xun", "益", "Yi", "Povećanje"),
    (43, "Qian", "Dui", "夬", "Guai", "Proboj"),
    (44, "Xun", "Qian", "姤", "Gou", "Susret"),
    (45, "Kun", "Dui", "萃", "Cui", "Okupljanje"),
    (46, "Xun", "Kun", "升", "Sheng", "Uzdizanje"),
    (47, "Kan", "Dui", "困", "Kun", "Iscrpljenost"),
    (48, "Xun", "Kan", "井", "Jing", "Bunar"),
    (49, "Li", "Dui", "革", "Ge", "Preokret"),
    (50, "Xun", "Li", "鼎", "Ding", "Kotao"),
    (51, "Zhen", "Zhen", "震", "Zhen", "Grom"),
    (52, "Gen", "Gen", "艮", "Gen", "Mirovanje"),
    (53, "Gen", "Xun", "漸", "Jian", "Postupni razvoj"),
    (54, "Dui", "Zhen", "歸妹", "Gui Mei", "Djevojka koja se udaje"),
    (55, "Li", "Zhen", "豐", "Feng", "Obilje"),
    (56, "Gen", "Li", "旅", "Lü", "Putnik"),
    (57, "Xun", "Xun", "巽", "Xun", "Blago prodiranje"),
    (58, "Dui", "Dui", "兌", "Dui", "Radost"),
    (59, "Kan", "Xun", "渙", "Huan", "Rasap"),
    (60, "Dui", "Kan", "節", "Jie", "Ograničenje"),
    (61, "Dui", "Xun", "中孚", "Zhong Fu", "Unutarnja istina"),
    (62, "Gen", "Zhen", "小過", "Xiao Guo", "Prevaga malog"),
    (63, "Li", "Kan", "既濟", "Ji Ji", "Nakon dovršenja"),
    (64, "Kan", "Li", "未濟", "Wei Ji", "Prije dovršenja"),
]
HEKS = {}          # KW broj -> dict(linije, donji, gornji, znak, pinyin, hr)
KW_OD_LINIJA = {}  # linije (odozdo) -> KW broj
for _n, _d, _g, _znak, _py, _hr in KING_WEN:
    _lin = TRIGRAMI[_d][0] + TRIGRAMI[_g][0]
    HEKS[_n] = dict(kw=_n, linije=_lin, donji=_d, gornji=_g, znak=_znak, pinyin=_py, hr=_hr)
    KW_OD_LINIJA[_lin] = _n
assert len(KW_OD_LINIJA) == 64, "King Wen tablica mora dati 64 različita heksagrama"


# --------------------------------------------------------------------------- osnovne operacije
def svedi(x: int) -> int:
    """Svođenje bilo kojeg pozitivnog cijelog broja u 1..45: ((x−1) mod 45) + 1."""
    return (x - 1) % 45 + 1


def kw(linije) -> int:
    return KW_OD_LINIJA[tuple(linije)]


def linije(n: int) -> tuple:
    return HEKS[n]["linije"]


def binarna(linije) -> int:
    """Shao Yong/Leibniz vrijednost 0..63: donja linija = 32, gornja = 1."""
    return sum(l << (5 - i) for i, l in enumerate(linije))


def nuklearni(linije) -> tuple:
    l = tuple(linije)
    return (l[1], l[2], l[3], l[2], l[3], l[4])


def cuo(linije) -> tuple:
    return tuple(1 - l for l in linije)


def zong(linije) -> tuple:
    return tuple(reversed(tuple(linije)))


def trigrami(linije) -> tuple[str, str]:
    l = tuple(linije)
    return TRIGRAM_OD_LINIJA[l[:3]], TRIGRAM_OD_LINIJA[l[3:]]


def fuxi(ime: str) -> int:
    return TRIGRAMI[ime][1]


def broj_trigrama(linije) -> int:
    d, g = trigrami(linije)
    return 10 * fuxi(g) + fuxi(d)


def imena_linija(linije) -> list[int]:
    """初九/六二…: 10·p + v, v = 9 (yang) ili 6 (yin); NEsvedeno (19..69)."""
    return [10 * (p + 1) + (9 if l else 6) for p, l in enumerate(linije)]


def crtez(linije) -> str:
    """Zapis odozgo prema dolje kao u knjigama (gornja linija prva)."""
    return " ".join("⚊" if l else "⚋" for l in reversed(tuple(linije)))


def bits(linije) -> str:
    return "".join(str(l) for l in linije)


def opis(n: int) -> str:
    h = HEKS[n]
    return f"{n} {h['znak']} {h['pinyin']} ({h['hr']})"


def analiza(n: int) -> dict:
    """Sve klasične derivacije jednog heksagrama (King Wen broj n)."""
    l = linije(n)
    d, g = trigrami(l)
    hu, su, ob = kw(nuklearni(l)), kw(cuo(l)), kw(zong(l))
    b = binarna(l)
    return dict(
        kw=n, linije=l, bits=bits(l), donji=d, gornji=g, fuxi_donji=fuxi(d), fuxi_gornji=fuxi(g),
        hu=hu, cuo=su, zong=ob, bin=b, shao=b + 1, trig=broj_trigrama(l),
        imena=imena_linija(l), yang=[i + 1 for i, x in enumerate(l) if x],
        yin=[i + 1 for i, x in enumerate(l) if not x],
    )


# --------------------------------------------------------------------------- datumski heksagrami
# (KW broj, izvor)
DATUMSKI = [
    (29, "dan 29"),
    (10, "mjesec 10"),
    (20, "sat 20"),
    (26, "godina '26"),
    (42, "2026 mod 64 = 42"),
    (11, "2 + 9 = 11"),
    (22, "fadik datuma 2+9+1+0+2+0+2+6 = 22"),
    (23, "87. kolo mod 64 = 23"),
]
SPOREDNI = [
    (1, "mjesec 1+0 = 1"),
    (2, "prva znamenka dana / 2. dan tjedna od Mjeseca"),
    (4, "četvrtak = 4. dan tjedna"),
]


def _prvih6(brojevi) -> list[int] | None:
    out: list[int] = []
    for x in brojevi:
        y = svedi(x)
        if y not in out:
            out.append(y)
        if len(out) == 6:
            return sorted(out)
    return None


def obitelj(n: int) -> list[int]:
    a = analiza(n)
    red = [a["kw"], a["hu"], a["cuo"], a["zong"], a["shao"], a["trig"], *a["imena"],
           sum(a["yang"]) or 45, sum(a["yin"]) or 45]
    return _prvih6(red)


def kanonski() -> list[tuple[list[int], str]]:
    """Kandidati izravno iz datumskih heksagrama, od najjače derivacije prema slabijoj."""
    A = {n: analiza(n) for n, _ in DATUMSKI + SPOREDNI}
    glavni = [n for n, _ in DATUMSKI]
    out: list[tuple[list[int], str]] = []
    out.append((sorted(glavni[:6]), "King Wen brojevi glavnih datumskih heksagrama (dan 29, mjesec 10, sat 20, '26, 2026→42, 2+9=11)"))
    out.append((sorted([22, 23, 29, 10, 20, 26]), "King Wen brojevi: fadik 22 賁, kolo 87→23 剝 + dan, mjesec, sat, godina"))
    out.append((_prvih6(A[n]["hu"] for n in glavni), "nuklearni (互卦 hu gua) heksagrami osam datumskih heksagrama"))
    out.append((_prvih6(A[n]["cuo"] for n in glavni), "suprotni (錯卦 cuo gua) heksagrami — komplement linija"))
    out.append((_prvih6(A[n]["zong"] for n in glavni), "obrnuti (綜卦 zong gua) heksagrami — okrenuti naglavce"))
    out.append((_prvih6(A[n]["shao"] for n in glavni), "Shao Yongovi binarni redni brojevi (vrijednost+1) datumskih heksagrama"))
    out.append((_prvih6(A[n]["trig"] for n in glavni), "brojevi trigrama 10·gornji+donji (Fuxi) datumskih heksagrama"))
    for n, izvor in DATUMSKI:
        out.append((obitelj(n), f"obitelj {opis(n)} [{izvor}]: KW, hu {A[n]['hu']}, cuo {A[n]['cuo']}, zong {A[n]['zong']}, bin {A[n]['shao']}, trig {A[n]['trig']}, linije"))
    for n, izvor in DATUMSKI:
        out.append((sorted(svedi(x) for x in A[n]["imena"]), f"imena šest linija {opis(n)} [{izvor}]: " + " ".join(f"{x}→{svedi(x)}" if x > 45 else str(x) for x in A[n]["imena"])))
    for n, izvor in SPOREDNI:
        out.append((obitelj(n), f"obitelj sporednog {opis(n)} [{izvor}]: hu {A[n]['hu']}, cuo {A[n]['cuo']}, zong {A[n]['zong']}, bin {A[n]['shao']}, trig {A[n]['trig']}"))
    for n, izvor in SPOREDNI:
        out.append((sorted(svedi(x) for x in A[n]["imena"]), f"imena šest linija sporednog {opis(n)} [{izvor}]"))
    return out


# --------------------------------------------------------------------------- bacanja
def baci_novcice(rng: random.Random) -> list[int]:
    """Tri novčića po liniji: 2 = yin (strana s natpisom), 3 = yang; zbroj 6/7/8/9 (1/8, 3/8, 3/8, 1/8)."""
    return [sum(rng.randint(2, 3) for _ in range(3)) for _ in range(6)]


def baci_stolisnik(rng: random.Random) -> list[int]:
    """Stolisnik (49 stabljika, tri odvajanja): vjerojatnosti 6: 1/16, 7: 5/16, 8: 7/16, 9: 3/16."""
    out = []
    for _ in range(6):
        r = rng.randrange(16)
        out.append(6 if r == 0 else 7 if r <= 5 else 8 if r <= 12 else 9)
    return out


def bacanje(i: int) -> dict:
    seed = SEED0 + i
    rng = random.Random(seed)
    if i % 2 == 0:
        metoda, vals = "tri novčića", baci_novcice(rng)
    else:
        metoda, vals = "stolisnik", baci_stolisnik(rng)
    prim = tuple(1 if v in (7, 9) else 0 for v in vals)
    pomicne = [p + 1 for p, v in enumerate(vals) if v in (6, 9)]
    rel = tuple((1 - l) if (p + 1) in pomicne else l for p, l in enumerate(prim))
    a, b = analiza(kw(prim)), analiza(kw(rel))
    red = [a["kw"]]
    if b["kw"] != a["kw"]:
        red.append(b["kw"])
    red += [10 * p + vals[p - 1] for p in pomicne]
    red += [a["hu"], a["shao"], sum(vals), a["cuo"], a["zong"], b["shao"], a["trig"], b["hu"], b["trig"], *a["imena"]]
    brojevi = _prvih6(red)
    koraci = []
    for x in red:
        y = svedi(x)
        koraci.append(f"{x}→{y}" if x != y else str(x))
    return dict(i=i, seed=seed, metoda=metoda, vrijednosti=vals, primarni=a["kw"], relacijski=b["kw"],
                pomicne=pomicne, brojevi=brojevi, red=koraci[:10], zbroj=sum(vals))


def bacanje_obrazlozenje(bc: dict) -> str:
    pom = ",".join(f"{p}{'九' if bc['vrijednosti'][p-1] == 9 else '六'}" for p in bc["pomicne"]) or "nema"
    rel = f" → {opis(bc['relacijski'])}" if bc["relacijski"] != bc["primarni"] else " (bez pomičnih)"
    return (f"{bc['metoda']} #{bc['i']} (seed {bc['seed']}): linije {''.join(map(str, bc['vrijednosti']))} = "
            f"{opis(bc['primarni'])}{rel}; pomične {pom}; brojevi {' '.join(bc['red'][:8])}")


# --------------------------------------------------------------------------- kandidati
def kandidati(n: int = 80) -> list[dict]:
    out: list[dict] = []
    videni: set[tuple] = set()

    def dodaj(brojevi, obr):
        t = tuple(sorted(brojevi))
        assert len(t) == 6 and len(set(t)) == 6 and all(1 <= x <= 45 for x in t), t
        if t in videni:
            return False
        videni.add(t)
        out.append({"brojevi": list(t), "rang": len(out) + 1, "obrazlozenje": obr})
        return True

    for brojevi, obr in kanonski():
        if brojevi is not None:
            dodaj(brojevi, "Datum: " + obr)
    i = 0
    while len(out) < n and i < 2000:
        bc = bacanje(i)
        if bc["brojevi"] is not None:
            dodaj(bc["brojevi"], "Bacanje: " + bacanje_obrazlozenje(bc))
        i += 1
    return out[:n]


def kljucni_brojevi(k: int = 12) -> list[int]:
    """Težinski najčešći brojevi iz derivacija datumskih heksagrama (KW 3, hu 2, cuo/zong 1.5, bin/trig 1, linije 0.5)."""
    w: Counter = Counter()
    for n, _ in DATUMSKI:
        a = analiza(n)
        w[svedi(a["kw"])] += 3
        w[svedi(a["hu"])] += 2
        w[svedi(a["cuo"])] += 1.5
        w[svedi(a["zong"])] += 1.5
        w[svedi(a["shao"])] += 1
        w[svedi(a["trig"])] += 1
        for x in a["imena"]:
            w[svedi(x)] += 0.5
    for n, _ in SPOREDNI:
        a = analiza(n)
        w[svedi(a["kw"])] += 1
        w[svedi(a["hu"])] += 0.5
    return [x for x, _ in sorted(w.items(), key=lambda t: (-t[1], t[0]))[:k]]


# --------------------------------------------------------------------------- postupak (tekst)
def postupak_tekst(n_bacanja_prikaz: int = 60) -> str:
    L = [f"# {METODA} (klasični Yijing, King Wen poredak) — {DATUM_TEKST}\n"]
    L.append("## 1. Konvencije\n")
    L.append("- Linije se broje **odozdo** (1 = donja … 6 = gornja); yang ⚊ = 1, yin ⚋ = 0. Zapis `bits` je odozdo prema gore; crtež je odozgo kao u knjigama.")
    L.append("- Trigrami (odozdo) i Fuxi brojevi: " + ", ".join(f"{v[2]} {k} {bits(v[0])} = {v[1]} ({v[3]})" for k, v in TRIGRAMI.items()))
    L.append("- Binarna (Shao Yong/Leibniz) vrijednost: Σ l_i·2^(6−i), donja linija = 32, gornja = 1; 坤 = 0, 剝 = 1, 比 = 2, 觀 = 3 … 乾 = 63. Shao Yongov redni broj = vrijednost + 1.")
    L.append("- Nuklearni 互卦: linije 2-3-4 dolje, 3-4-5 gore. Suprotni 錯卦: komplement. Obrnuti 綜卦: naglavce.")
    L.append("- Imena linija: 10·p + v (v = 9 yang, 6 yin): 初九 19, 九二 29, 六三 36, 九四 49→4, 六五 56→11, 上九 69→24 …")
    L.append("- Broj trigrama: 10·Fuxi(gornji) + Fuxi(donji). Svođenje u 1..45: ((x−1) mod 45)+1.")
    L.append("- Provjera tablice: 1 乾 = 111111, 2 坤 = 000000, 11 泰 = 111000, 63 既濟 = 101010 (odozdo), 64 未濟 = 010101; "
             "svaki King Wen par (2k−1, 2k) = zong, a za simetrične = cuo (provjereno programski za sva 32 para).\n")
    L.append("## 2. Datumski heksagrami i njihove derivacije\n")
    L.append("| izvor | heksagram | linije (odozdo) | crtež (odozgo) | trigrami (donji/gornji) | hu | cuo | zong | bin (0..63) → Shao | trig | yang linije | imena linija → svedeno |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, izvor in DATUMSKI + SPOREDNI:
        a = analiza(n)
        L.append(f"| {izvor} | {opis(n)} | {a['bits']} | {crtez(a['linije'])} | {a['donji']}({a['fuxi_donji']})/{a['gornji']}({a['fuxi_gornji']}) "
                 f"| {opis(a['hu'])} | {opis(a['cuo'])} | {opis(a['zong'])} | {a['bin']} → {a['shao']}→{svedi(a['shao'])} "
                 f"| {a['trig']}→{svedi(a['trig'])} | {a['yang']} | {' '.join(f'{x}→{svedi(x)}' if x > 45 else str(x) for x in a['imena'])} |")
    L.append("")
    L.append("## 3. Kandidati iz datumskih heksagrama (rang 1 = najjača derivacija)\n")
    for i, (brojevi, obr) in enumerate(kanonski(), 1):
        L.append(f"{i}. `{' · '.join(f'{x:02d}' for x in brojevi)}` — {obr}")
    L.append("")
    L.append("## 4. Bacanja (deterministički: random.Random(20261029 + i); parni i = tri novčića, neparni i = stolisnik)\n")
    L.append("- Tri novčića: svaki novčić 2 (yin) ili 3 (yang), zbroj 6 = stara yin (pomična), 7 = mlada yang, 8 = mlada yin, 9 = stara yang (pomična); P = 1/8, 3/8, 3/8, 1/8.")
    L.append("- Stolisnik: P(6,7,8,9) = 1/16, 5/16, 7/16, 3/16 (rng.randrange(16): 0→6, 1..5→7, 6..12→8, 13..15→9).")
    L.append("- Linije se bacaju odozdo. Brojevi redom: KW primarnog, KW relacijskog, imena pomičnih linija (10p+v), hu, Shao, zbroj vrijednosti linija, cuo, zong, Shao relacijskog, trig … prvih 6 različitih nakon svođenja.\n")
    L.append("| i | metoda | seed | linije (odozdo) | primarni | pomične | relacijski | zbroj | niz brojeva (→ svedeno) | kandidat |")
    L.append("|---:|---|---:|---|---|---|---|---:|---|---|")
    for i in range(n_bacanja_prikaz):
        bc = bacanje(i)
        L.append(f"| {i} | {bc['metoda']} | {bc['seed']} | {''.join(map(str, bc['vrijednosti']))} | {opis(bc['primarni'])} | {bc['pomicne'] or '—'} "
                 f"| {opis(bc['relacijski']) if bc['relacijski'] != bc['primarni'] else '—'} | {bc['zbroj']} | {' '.join(bc['red'])} "
                 f"| `{' · '.join(f'{x:02d}' for x in bc['brojevi'])}` |")
    L.append("")
    L.append("## 5. Ključni brojevi\n")
    L.append("Težinski zbroj po svim derivacijama datumskih heksagrama (KW 3, hu 2, cuo/zong 1.5, Shao/trig 1, imena linija 0.5): " + ", ".join(map(str, kljucni_brojevi())))
    return "\n".join(L)


# --------------------------------------------------------------------------- provjere
def _provjere():
    # poznati heksagrami (linije odozdo)
    assert bits(linije(1)) == "111111" and bits(linije(2)) == "000000"
    assert bits(linije(63)) == "101010" and bits(linije(64)) == "010101"
    assert bits(linije(11)) == "111000" and bits(linije(12)) == "000111"
    assert bits(linije(29)) == "010010" and bits(linije(30)) == "101101"
    assert bits(linije(23)) == "000001" and bits(linije(24)) == "100000"
    assert bits(linije(42)) == "100011" and bits(linije(43)) == "111110" and bits(linije(44)) == "011111"
    # King Wen parovi: zong, a za simetrične cuo
    for k in range(1, 64, 2):
        a, b = linije(k), linije(k + 1)
        assert b == (cuo(a) if zong(a) == a else zong(a)), (k, a, b)
    # Shao Yong niz: Kun 0, Bo 1, Bi 2, Guan 3, Yu 4, Jin 5, Cui 6, Pi 7, Qian(謙) 8, Qian(乾) 63
    for n, v in [(2, 0), (23, 1), (8, 2), (20, 3), (16, 4), (35, 5), (45, 6), (12, 7), (15, 8), (1, 63)]:
        assert binarna(linije(n)) == v, (n, v)
    # derivacije iz docstringa
    A = {n: analiza(n) for n, _ in DATUMSKI}
    assert (A[29]["hu"], A[29]["cuo"], A[29]["zong"], A[29]["shao"], A[29]["trig"]) == (27, 30, 29, 19, 66)
    assert (A[10]["hu"], A[10]["cuo"], A[10]["zong"], A[10]["shao"], A[10]["trig"]) == (37, 15, 9, 56, 12)
    assert (A[20]["hu"], A[20]["cuo"], A[20]["zong"], A[20]["shao"], A[20]["trig"]) == (23, 34, 19, 4, 58)
    assert (A[26]["hu"], A[26]["cuo"], A[26]["zong"], A[26]["shao"], A[26]["trig"]) == (54, 45, 25, 58, 71)
    assert (A[42]["hu"], A[42]["cuo"], A[42]["zong"], A[42]["shao"], A[42]["trig"]) == (23, 32, 41, 36, 54)
    assert (A[11]["hu"], A[11]["cuo"], A[11]["zong"], A[11]["shao"], A[11]["trig"]) == (54, 12, 12, 57, 81)
    assert (A[22]["hu"], A[22]["cuo"], A[22]["zong"], A[22]["shao"], A[22]["trig"]) == (40, 47, 21, 42, 73)
    assert (A[23]["hu"], A[23]["cuo"], A[23]["zong"], A[23]["shao"], A[23]["trig"]) == (2, 43, 24, 2, 78)
    assert [svedi(x) for x in A[29]["imena"]] == [16, 29, 36, 1, 14, 21]
    assert [svedi(x) for x in A[10]["imena"]] == [19, 29, 36, 4, 14, 24]
    assert svedi(46) == 1 and svedi(64) == 19 and svedi(45) == 45 and svedi(90) == 45 and svedi(66) == 21
    # vjerojatnosti bacanja (empirijski, deterministično)
    rng = random.Random(1)
    c = Counter(v for _ in range(4000) for v in baci_stolisnik(rng))
    tot = sum(c.values())
    assert abs(c[8] / tot - 7 / 16) < 0.02 and abs(c[6] / tot - 1 / 16) < 0.01
    c = Counter(v for _ in range(4000) for v in baci_novcice(rng))
    tot = sum(c.values())
    assert abs(c[7] / tot - 3 / 8) < 0.02 and abs(c[9] / tot - 1 / 8) < 0.01


# --------------------------------------------------------------------------- main
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
