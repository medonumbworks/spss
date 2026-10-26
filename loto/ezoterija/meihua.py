"""Meihua Yishu (梅花易數, "Numerologija cvijeta šljive") za Loto 6/45 — izvlačenje četvrtak
29.10.2026. u 20:00 CET (87. kolo), Zagreb.

ŠTO JE METODA
-------------
Meihua Yishu je proročka numerologija pripisana Shao Yongu (邵雍, 1011.–1077., dinastija Song).
Za razliku od klasičnog Yijinga (novčići/stolisnik), heksagram se ne baca nego se IZRAČUNA iz
brojeva trenutka — ovdje "vremenskom metodom" (年月日時起卦, postavljanje po godini, mjesecu, danu
i satu):
  - godina = broj zemaljske grane (子1 丑2 寅3 卯4 辰5 巳6 午7 未8 申9 酉10 戌11 亥12),
  - mjesec i dan = KINESKI LUNARNI mjesec i dan,
  - sat = grana dvosatnog razdoblja (子 23–01 h … 戌 19–21 h … 亥 21–23 h) po pravom sunčevom vremenu.
  Z = godina + mjesec + dan.  Gornji trigram = Z mod 8, donji trigram = (Z + sat) mod 8,
  pomična linija = (Z + sat) mod 6; ostatak 0 znači 8 odnosno 6.
  Trigrami nose prijenebeske (先天, Fuxi) brojeve: 乾1 兌2 離3 震4 巽5 坎6 艮7 坤8.
Čitanje: 本卦 (temeljni heksagram = situacija), 互卦 (nuklearni: linije 2-3-4 čine donji, 3-4-5 gornji
trigram = proces), 變卦 (promijenjeni: pomična linija preokrenuta = ishod); 體 (tijelo, subjekt) je
trigram BEZ pomične linije, 用 (upotreba, objekt) trigram S pomičnom linijom; odnos 體/用 čita se
preko pet faza (乾兌 metal, 震巽 drvo, 坎 voda, 離 vatra, 艮坤 zemlja).
Korisnikova skripta dodaje još tri klasične transformacije: 錯卦 (komplement svih linija), 綜卦
(heksagram naglavce, pomična linija 7 − m) i 錯綜 (oboje), pa od svake od četiri varijante gradi
"niz S" od 10 brojeva i iz njega "tok" kumulativnih zbrojeva sveden u 1..45.

IZVORI / PRAVILA (reproducirano iz korisnikove skripte data/loto_ezoterija_original.py — blokovi
"Meihua", "hexa", "stream", "meihua"; tablice TL/LT/KW su preuzete doslovno)
-----------------------------------------------------------------------------
  - TL: trigram → linije ODOZDO (1 = yang, 0 = yin): 乾 111, 兌 110, 離 101, 震 100, 巽 011, 坎 010,
    艮 001, 坤 000.  KW[gornji][donji] = King Wen broj heksagrama (provjereno: svih 64 točno jednom;
    乾/乾 = 1, 坤/坤 = 2, 坎/震 = 3 屯, 艮/坎 = 4 蒙, 坤/乾 = 11 泰, 坎/離 = 63 既濟, 離/坎 = 64 未濟).
  - hexa(linije, m): 本 = (gornji, donji, KW), 互 = (3-4-5, 2-3-4, KW), 變 = linije s preokrenutom
    m-tom linijom, 體/用: za m ≤ 3 體 = gornji, 用 = donji; za m ≥ 4 體 = donji, 用 = gornji.
  - niz S = [本 gornji, 本 donji, KW 本, 互 gornji, 互 donji, KW 互, 變 gornji, 變 donji, KW 變, m].
  - tok: c = kumulativni zbroj niza S (ciklički), svaki korak daje ((c − 1) mod 45) + 1; broj već
    zauzet u listiću pomiče se pravilom x → (x mod 45) + 1 dok ne bude slobodan.
  - listić = prvih 6 brojeva toka; provjera valid(): 6 različitih u 1..45 i najviše 2 zajednička broja
    s već prihvaćenim listićima (samo vlastiti kandidati; popular_ok filtar skripte se NE primjenjuje).
    Ako provjera ne prođe, skripta odbacuje ZADNJI broj i uzima sljedeći korak toka.
  - DOPUNA (dokumentirana): kad se prvih 5 brojeva već preklapa s nekim listićem u ≥ 3 broja, pravilo
    "odbaci zadnji" ne može nikad uspjeti (izvorna skripta bi stala porukom "nema valjanog rješenja");
    tada se odbacuje cijeli listić i uzima sljedećih 6 koraka toka ("preskok prozora"). Za ovaj datum
    to je nužno za 綜 i 錯綜, jer su 小過 (62) i 中孚 (61) simetrični heksagrami (綜 = 本, 錯綜 = 錯):
    njihovi nizovi S razlikuju se od 本/錯 tek na mjestima 7–10 (變 i m), pa su prva 4 koraka toka
    (i prvih 6 brojeva) identična.
  - Nastavak toka: iza brojeva koji su ušli u osnovni listić svakih sljedećih 6 koraka istog toka
    daje novi kandidat (bez pakiranja, samo pravilo "zauzet" unutar prozora).
  - 互/變 kao korijen: 互卦 i 變卦 temeljnog heksagrama uzimaju se kao vlastiti 本卦 s istom pomičnom
    linijom m i od njih se gradi niz S i tok (za 變 kao korijen ista pomična linija vraća u 本卦 —
    ciklus 本 → 變 → 本).  後天 varijanta: isti niz S za 本, ali s brojevima trigrama iz Luoshua
    (後天數: 坎1 坤2 震3 巽4 乾6 兌7 艮8 離9) umjesto prijenebeskih.
  - Izravni listić: prvih 6 različitih vrijednosti niza S svedenih u 1..45 (x → ((x−1) mod 45)+1).
  - Sekundarna varijanta: sat-grana 10 (酉). NAPOMENA: po standardnim granicama (戌 = 19–21 h) i
    zonsko vrijeme 20:00 CET i pravo sunčevo 20:20 padaju u 戌 = 11, pa je 11 primarna vrijednost;
    grana 10 uključena je po uputi kao test osjetljivosti (npr. ako se trenutak pitanja uzme prije
    19 h, ili konvencija 子 = 00–02 h primijenjena na 19:00 UTC).
  - Bez slučajnosti: metoda je potpuno deterministična (ništa se ne baca ni miješa).

POSTUPAK ZA 29.10.2026. 20:00 CET (sažetak; cijeli izvod s logom svih koraka daje postupak_tekst()
i zapisuje se u out/ezo_meihua.json)
----------------------------------------------------------------------------------------------
  Ulaz: godina 丙午 → 午 = 7; lunarni datum 9. mjesec 20. dan (lunardate potvrđuje: 2026-10-29 =
  LunarDate(2026, 9, 20)); sat: sunčevo podne 11:39:48 CET → 20:00 CET = 20:20 pravog sunčevog
  vremena → 戌 = 11.
  Z = 7 + 9 + 20 = 36; gornji = 36 mod 8 = 4 震 Zhen (grom); donji = (36 + 11) mod 8 = 47 mod 8 = 7
  艮 Gen (planina); pomična = 47 mod 6 = 5.
  本卦: linije odozdo 001 100 = 艮 dolje / 震 gore = 雷山小過 Xiao Guo, KW 62 (Prevaga malog).
    互卦: 2-3-4 = 011 巽 (5), 3-4-5 = 110 兌 (2) → 澤風大過 Da Guo, KW 28.
    變卦: 5. linija preokrenuta → 001 110 = 艮 / 兌 = 澤山咸 Xian, KW 31.  體 = 艮 (zemlja), 用 = 震 (drvo).
    S(本) = [4, 7, 62, 2, 5, 28, 2, 7, 31, 5]  (zbroj 153).
  錯卦: 110 011 = 兌 / 巽 = 風澤中孚 Zhong Fu, KW 61; 互 = 震/艮 = 山雷頤 27; 變 (5. linija) = 兌/艮 =
    山澤損 41; 體 = 兌, 用 = 巽.  S(錯) = [5, 2, 61, 7, 4, 27, 7, 2, 41, 5]  (zbroj 161).
  綜卦: 小過 naglavce = 小過 (simetričan), pomična 7 − 5 = 2; 變 = 巽/震 = 雷風恆 32; 體 = 震, 用 = 艮.
    S(綜) = [4, 7, 62, 2, 5, 28, 4, 5, 32, 2]  (zbroj 151).
  錯綜: 中孚 naglavce = 中孚, pomična 2; 變 = 震/巽 = 風雷益 42; 體 = 巽, 用 = 兌.
    S(錯綜) = [5, 2, 61, 7, 4, 27, 5, 4, 42, 2]  (zbroj 159).
  Tok 本: +4 → 4 → 4; +7 → 11 → 11; +62 → 73 → 28; +2 → 75 → 30; +5 → 80 → 35; +28 → 108 → 18
    → listić 本 = 4 11 18 28 30 35.
  Tok 錯: 5, 7, 23, 30, 34, 16 → listić 錯 = 5 7 16 23 30 34 (1 zajednički s 本 → prolazi).
  Tok 綜: 4 11 28 30 35 18 = identično 本 → odbačen zadnji (18), prvih 5 se preklapa u 5 brojeva →
    preskok prozora → koraci 7–12: 22, 27, 14, 16, 20, 27 zauzet → 28 → listić 綜 = 14 16 20 22 27 28.
  Tok 錯綜: 5 7 23 30 34 16 = identično 錯 → preskok → koraci 7–12: 21, 25, 22, 24, 29, 31 →
    listić 錯綜 = 21 22 24 25 29 31.
  互 kao korijen (大過, m = 5): S = [2, 5, 28, 1, 1, 1, 4, 5, 32, 5] → tok 2 7 35 36 37 38.
  變 kao korijen (咸, m = 5): S = [2, 7, 31, 1, 5, 44, 4, 7, 62, 5] → tok 2 9 40 41 1 45.
  後天 varijanta 本: S = [3, 8, 62, 7, 4, 28, 7, 8, 31, 5] → tok 3 11 28 35 39 22 (3 zajednička s 本) →
    preskok → koraci 7–12: 29 37 23 28 31 39 → listić 23 28 29 31 37 39.
  Izravni 本: 4 7 17 2 5 28;  brojevi trenutka: Z 36, Z+sat 47→2, KW 62→17, 28, 31, m 5.
  Sekundarna varijanta (sat 10 酉): donji = 46 mod 8 = 6 坎, pomična = 46 mod 6 = 4 → 本 = 雷水解 Xie 40,
    互 = 水火既濟 63, 變 = 地水師 7; S = [4, 6, 40, 6, 3, 63, 8, 6, 7, 4] → tok 4 10 5 11 14 32; itd.
  Rang: 本 (tok, izravni, trenutak, 後天) → 互/變 kao korijen → 錯/綜/錯綜 → nastavak tokova →
    varijanta sat 10.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

SLUG = "meihua"
METODA = "Meihua Yishu"
DATUM_TEKST = "četvrtak 29.10.2026. u 20:00 CET (87. kolo), Zagreb"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"

N, K = 45, 6
U = set(range(1, N + 1))

# ------------------------------------------------------------------ ULAZ (kao u skripti, IGRA = "L6")
GODINA_BROJ = 7      # 丙午 → 午 = 7
LUNARNI_MJ = 9       # 9. lunarni mjesec
LUNARNI_DAN = 20     # 20. lunarni dan
SAT_BROJ = 11        # 戌 (19–21 h); pravo sunčevo vrijeme 20:20 CET
SAT_ALT = 10         # 酉 — sekundarna varijanta (test osjetljivosti)

# ------------------------------------------------------------------ tablice (doslovno iz skripte)
TL = {1: (1, 1, 1), 2: (1, 1, 0), 3: (1, 0, 1), 4: (1, 0, 0),
      5: (0, 1, 1), 6: (0, 1, 0), 7: (0, 0, 1), 8: (0, 0, 0)}   # linije odozdo prema gore
LT = {v: k for k, v in TL.items()}
TN = {1: "乾 Qian", 2: "兌 Dui", 3: "離 Li", 4: "震 Zhen", 5: "巽 Xun", 6: "坎 Kan", 7: "艮 Gen", 8: "坤 Kun"}
TN_HR = {1: "nebo", 2: "jezero", 3: "vatra", 4: "grom", 5: "vjetar", 6: "voda", 7: "planina", 8: "zemlja"}
WX = {1: "metal", 2: "metal", 3: "vatra", 4: "drvo", 5: "drvo", 6: "voda", 7: "zemlja", 8: "zemlja"}
HT = {1: 6, 2: 7, 3: 9, 4: 3, 5: 4, 6: 1, 7: 8, 8: 2}           # 後天 (Luoshu) brojevi trigrama
KW = {1: [1, 10, 13, 25, 44, 6, 33, 12], 2: [43, 58, 49, 17, 28, 47, 31, 45],
      3: [14, 38, 30, 21, 50, 64, 56, 35], 4: [34, 54, 55, 51, 32, 40, 62, 16],
      5: [9, 61, 37, 42, 57, 59, 53, 20], 6: [5, 60, 63, 3, 48, 29, 39, 8],
      7: [26, 41, 22, 27, 18, 4, 52, 23], 8: [11, 19, 36, 24, 46, 7, 15, 2]}   # KW[gornji][donji-1]
KW_IME = {
    1: ("乾", "Qian", "Stvaralačko"), 2: ("坤", "Kun", "Primajuće"), 3: ("屯", "Zhun", "Početna teškoća"),
    4: ("蒙", "Meng", "Mladenačka ludost"), 5: ("需", "Xu", "Čekanje"), 6: ("訟", "Song", "Sukob"),
    7: ("師", "Shi", "Vojska"), 8: ("比", "Bi", "Zajedništvo"), 9: ("小畜", "Xiao Chu", "Ukroćena snaga malog"),
    10: ("履", "Lü", "Stupanje"), 11: ("泰", "Tai", "Mir"), 12: ("否", "Pi", "Zastoj"),
    13: ("同人", "Tong Ren", "Zajednica s ljudima"), 14: ("大有", "Da You", "Posjedovanje u velikom"),
    15: ("謙", "Qian", "Skromnost"), 16: ("豫", "Yu", "Oduševljenje"), 17: ("隨", "Sui", "Slijeđenje"),
    18: ("蠱", "Gu", "Rad na pokvarenom"), 19: ("臨", "Lin", "Približavanje"), 20: ("觀", "Guan", "Promatranje"),
    21: ("噬嗑", "Shi He", "Progrizanje"), 22: ("賁", "Bi", "Ljupkost"), 23: ("剝", "Bo", "Raspadanje"),
    24: ("復", "Fu", "Povratak"), 25: ("無妄", "Wu Wang", "Nevinost"), 26: ("大畜", "Da Chu", "Ukroćena snaga velikog"),
    27: ("頤", "Yi", "Hranjenje"), 28: ("大過", "Da Guo", "Prevaga velikog"), 29: ("坎", "Kan", "Ponor"),
    30: ("離", "Li", "Prianjanje"), 31: ("咸", "Xian", "Utjecaj"), 32: ("恆", "Heng", "Trajanje"),
    33: ("遯", "Dun", "Povlačenje"), 34: ("大壯", "Da Zhuang", "Snaga velikog"), 35: ("晉", "Jin", "Napredak"),
    36: ("明夷", "Ming Yi", "Pomračenje svjetla"), 37: ("家人", "Jia Ren", "Obitelj"), 38: ("睽", "Kui", "Suprotnost"),
    39: ("蹇", "Jian", "Zapreka"), 40: ("解", "Xie", "Oslobođenje"), 41: ("損", "Sun", "Smanjenje"),
    42: ("益", "Yi", "Povećanje"), 43: ("夬", "Guai", "Proboj"), 44: ("姤", "Gou", "Susret"),
    45: ("萃", "Cui", "Okupljanje"), 46: ("升", "Sheng", "Uspon"), 47: ("困", "Kun", "Tjeskoba"),
    48: ("井", "Jing", "Bunar"), 49: ("革", "Ge", "Preokret"), 50: ("鼎", "Ding", "Kotao"),
    51: ("震", "Zhen", "Grom"), 52: ("艮", "Gen", "Mirovanje"), 53: ("漸", "Jian", "Razvoj"),
    54: ("歸妹", "Gui Mei", "Djevojka koja se udaje"), 55: ("豐", "Feng", "Obilje"), 56: ("旅", "Lü", "Putnik"),
    57: ("巽", "Xun", "Blago"), 58: ("兌", "Dui", "Radost"), 59: ("渙", "Huan", "Raspršivanje"),
    60: ("節", "Jie", "Ograničenje"), 61: ("中孚", "Zhong Fu", "Unutarnja istina"), 62: ("小過", "Xiao Guo", "Prevaga malog"),
    63: ("既濟", "Ji Ji", "Nakon dovršetka"), 64: ("未濟", "Wei Ji", "Prije dovršetka"),
}
GRANE = {1: "子", 2: "丑", 3: "寅", 4: "卯", 5: "辰", 6: "巳", 7: "午", 8: "未", 9: "申", 10: "酉", 11: "戌", 12: "亥"}

OSNOVNI = ["本", "錯", "綜", "錯綜"]            # redoslijed pakiranja kao u skripti
IZVEDENI = ["互k", "變k", "本後天"]               # dodatni tokovi (互/變 kao korijen, 後天 brojevi)
OPIS_TOKA = {"本": "本卦 (temeljni)", "錯": "錯卦 (komplement)", "綜": "綜卦 (naglavce)", "錯綜": "錯綜 (oboje)",
             "互k": "互卦 kao korijen", "變k": "變卦 kao korijen", "本後天": "本卦 s 後天 brojevima trigrama"}


# ------------------------------------------------------------------ osnovne funkcije (iz skripte)
def kw(u: int, l: int) -> int:
    return KW[u][l - 1]


def svedi(x: int) -> int:
    return (x - 1) % N + 1


def ime(k: int) -> str:
    z, p, h = KW_IME[k]
    return f"{z} {p} ({k})"


def hexa(lines: list[int], mv: int) -> dict:
    """Točno kao u skripti: 本/互/變 trigrami i King Wen brojevi, 體/用 prema pomičnoj liniji."""
    lo, up = LT[tuple(lines[:3])], LT[tuple(lines[3:])]
    nl, nu = LT[tuple(lines[1:4])], LT[tuple(lines[2:5])]
    ch = list(lines)
    ch[mv - 1] ^= 1
    cl, cu = LT[tuple(ch[:3])], LT[tuple(ch[3:])]
    ti, yo = (up, lo) if mv <= 3 else (lo, up)
    return dict(ben=(up, lo, kw(up, lo)), hu=(nu, nl, kw(nu, nl)), bian=(cu, cl, kw(cu, cl)),
                mv=mv, ti=ti, yong=yo, lines=list(lines), changed=ch)


def niz_S(h: dict) -> list[int]:
    return [h["ben"][0], h["ben"][1], h["ben"][2], h["hu"][0], h["hu"][1], h["hu"][2],
            h["bian"][0], h["bian"][1], h["bian"][2], h["mv"]]


def linije(gornji: int, donji: int) -> list[int]:
    return list(TL[donji]) + list(TL[gornji])


def valid(t: list[int], others: list[list[int]]) -> bool:
    """Kao u skripti za L6: 6 različitih u 1..45 i ≤ 2 zajednička s već prihvaćenim listićima."""
    A = set(t)
    if len(A) != K or not A <= U:
        return False
    return all(len(A & set(B)) <= 2 for B in others)


def _moze_proci(partial: list[int], others: list[list[int]]) -> bool:
    """Nužan uvjet: prvih K−1 brojeva smije se s bilo kojim listićem preklapati u najviše 2 broja."""
    A = set(partial)
    return all(len(A & set(B)) <= 2 for B in others)


class Tok:
    """Trajni tok kumulativnih zbrojeva niza S (generator 'stream' iz skripte, koji se ne resetira)."""

    def __init__(self, S: list[int]):
        self.S = list(S)
        self.i = 0
        self.c = 0

    def korak(self) -> tuple[int, int, int, int]:
        s = self.S[self.i % len(self.S)]
        self.c += s
        self.i += 1
        return self.i, s, self.c, (self.c - 1) % N + 1


def _nxt(tok: Tok, out: list[int], log: list[str]) -> None:
    i, s, c, cand = tok.korak()
    x = cand
    while x in out:
        x = x % N + 1
    out.append(x)
    log.append(f"korak {i}: +{s} → zbroj {c} → (({c}−1) mod {N})+1 = {cand}" + ("" if x == cand else f" zauzet → {x}"))


def listic(tok: Tok, others: list[list[int]], log: list[str], max_prozora: int = 30) -> list[int] | None:
    """Funkcija meihua() iz skripte (bez popular_ok) + dokumentirani preskok prozora kad pravilo
    'odbaci zadnji broj' ne može uspjeti. Vraća listić u redoslijedu izvlačenja ili None."""
    out: list[int] = []
    while len(out) < K:
        _nxt(tok, out, log)
    n = 0
    prozor = 1
    while not valid(out, others):
        zadnji = out.pop()
        log.append(f"ODBAČEN zadnji broj {zadnji} (provjera nije prošla), nastavljam niz")
        if not _moze_proci(out, others):
            sudar = next(B for B in others if len(set(out) & set(B)) > 2)
            log.append(f"prvih {K-1} brojeva {out} preklapa se s listićem {sorted(sudar)} u "
                       f"{len(set(out) & set(sudar))} broja → pravilo 'odbaci zadnji' ne može uspjeti "
                       f"(izvorna skripta: 'nema valjanog rješenja') → PRESKOK PROZORA: odbacujem cijeli listić, "
                       f"nastavljam niz od koraka {tok.i + 1}")
            out.clear()
            prozor += 1
            if prozor > max_prozora:
                log.append("nema valjanog rješenja ni nakon preskoka prozora")
                return None
            while len(out) < K:
                _nxt(tok, out, log)
            continue
        _nxt(tok, out, log)
        n += 1
        if n > 500:
            log.append("nema valjanog rješenja (500 pokušaja)")
            return None
    log.append(f"PRIHVAĆEN listić {out} → sortirano {sorted(out)} (zadnji potrošeni korak {tok.i})")
    return out


def prozor(tok: Tok) -> dict:
    """Nastavak toka: sljedećih 6 koraka kao novi kandidat (pravilo 'zauzet' samo unutar prozora)."""
    out: list[int] = []
    log: list[str] = []
    od = tok.i + 1
    while len(out) < K:
        _nxt(tok, out, log)
    return dict(brojevi=sorted(out), redom=out, od=od, do=tok.i, log=log)


def izravni(S: list[int]) -> list[int] | None:
    out: list[int] = []
    for x in S:
        y = svedi(x)
        if y not in out:
            out.append(y)
        if len(out) == K:
            break
    return sorted(out) if len(out) == K else None


# ------------------------------------------------------------------ cijeli izračun za zadani sat
def izracun(sat: int, W: int = 6) -> dict:
    Z = GODINA_BROJ + LUNARNI_MJ + LUNARNI_DAN
    up = Z % 8 or 8
    low = (Z + sat) % 8 or 8
    mv = (Z + sat) % 6 or 6
    base = linije(up, low)
    H = {"本": hexa(base, mv), "錯": hexa([1 - x for x in base], mv),
         "綜": hexa(base[::-1], 7 - mv), "錯綜": hexa([1 - x for x in base[::-1]], 7 - mv)}
    b = H["本"]
    H["互k"] = hexa(linije(b["hu"][0], b["hu"][1]), mv)
    H["變k"] = hexa(b["changed"], mv)
    S = {k: niz_S(h) for k, h in H.items()}
    S["本後天"] = [HT[x] if i in (0, 1, 3, 4, 6, 7) else x for i, x in enumerate(S["本"])]
    H["本後天"] = H["本"]          # isti heksagram, samo 後天 brojevi trigrama u nizu S
    tokovi = {k: Tok(S[k]) for k in S}
    listici: dict[str, list[int] | None] = {}
    logovi: dict[str, list[str]] = {}
    prihvaceni: list[list[int]] = []
    for k in OSNOVNI + IZVEDENI:
        log: list[str] = []
        t = listic(tokovi[k], prihvaceni, log)
        listici[k] = t
        logovi[k] = log
        if t is not None:
            prihvaceni.append(t)
    prozori = {k: [prozor(tokovi[k]) for _ in range(W)] for k in S}
    izr = {k: izravni(S[k]) for k in S}
    trenutak = sorted({svedi(Z), svedi(Z + sat), svedi(b["ben"][2]), svedi(b["hu"][2]), svedi(b["bian"][2]), mv})
    return dict(sat=sat, Z=Z, up=up, low=low, mv=mv, base=base, H=H, S=S, listici=listici, logovi=logovi,
                prozori=prozori, izravni=izr, trenutak=trenutak if len(trenutak) == K else None)


def _opis_h(h: dict) -> str:
    return (f"linije {''.join(map(str, h['lines']))} | 本 {TN[h['ben'][0]]}/{TN[h['ben'][1]]} = {ime(h['ben'][2])} | "
            f"互 {TN[h['hu'][0]]}/{TN[h['hu'][1]]} = {ime(h['hu'][2])} | 變 {TN[h['bian'][0]]}/{TN[h['bian'][1]]} = "
            f"{ime(h['bian'][2])} | pomična {h['mv']} | 體 {TN[h['ti']]} ({WX[h['ti']]}) 用 {TN[h['yong']]} ({WX[h['yong']]})")


def _fmt(t: list[int]) -> str:
    return " · ".join(f"{x:02d}" for x in sorted(t))


# ------------------------------------------------------------------ kandidati
def kandidati(n: int = 80) -> list[dict]:
    """Najmanje 60 (ili n) različitih kandidata, rangiranih po unutarnjoj logici metode."""
    n = max(n, 60)
    W, Wa = 6, 6
    while True:
        out = _slozi(izracun(SAT_BROJ, W), izracun(SAT_ALT, Wa))
        if len(out) >= n or W >= 60:
            break
        W += 4
        Wa += 4
    return out[:n]


def _slozi(G: dict, A: dict) -> list[dict]:
    out: list[dict] = []
    videni: set[tuple[int, ...]] = set()

    def dodaj(brojevi, obr: str) -> bool:
        if brojevi is None:
            return False
        t = tuple(sorted(brojevi))
        assert len(t) == K and len(set(t)) == K and all(1 <= x <= N for x in t), t
        if t in videni:
            return False
        videni.add(t)
        out.append({"brojevi": list(t), "rang": len(out) + 1, "obrazlozenje": obr})
        return True

    def s_opis(G, k):
        return f"S={G['S'][k]}"

    def koraci(G, k):
        log = G["logovi"][k]
        prihv = [l for l in log if l.startswith("PRIHVAĆEN")]
        zadnji = prihv[-1].rsplit("korak ", 1)[-1].rstrip(")") if prihv else "?"
        return f"zadnji korak {zadnji}"

    b = G["H"]["本"]
    # 1) 本
    dodaj(G["listici"]["本"], f"本卦 {ime(b['ben'][2])}: tok niza {s_opis(G, '本')}, koraci 1–6")
    dodaj(G["izravni"]["本"], f"本卦 {ime(b['ben'][2])}: izravni brojevi niza S (svedeno u 1..45)")
    dodaj(G["trenutak"], f"本卦 brojevi trenutka: Z={G['Z']}, Z+sat={G['Z'] + G['sat']}→{svedi(G['Z'] + G['sat'])}, "
                         f"KW 本 {b['ben'][2]}→{svedi(b['ben'][2])}, 互 {b['hu'][2]}, 變 {b['bian'][2]}, pomična {G['mv']}")
    dodaj(G["listici"]["本後天"], f"本卦 s 後天 (Luoshu) brojevima trigrama: tok {s_opis(G, '本後天')}, {koraci(G, '本後天')}")
    # 2) 互 / 變 kao korijen
    for k, naziv in (("互k", f"互卦 {ime(b['hu'][2])} kao korijen"), ("變k", f"變卦 {ime(b['bian'][2])} kao korijen")):
        dodaj(G["listici"][k], f"{naziv}, pomična {G['mv']}: tok {s_opis(G, k)}, {koraci(G, k)}")
    for k, naziv in (("互k", f"互卦 {ime(b['hu'][2])} kao korijen"), ("變k", f"變卦 {ime(b['bian'][2])} kao korijen")):
        dodaj(G["izravni"][k], f"{naziv}: izravni brojevi niza S")
    # 3) 錯 / 綜 / 錯綜
    for k in ("錯", "綜", "錯綜"):
        h = G["H"][k]
        dodaj(G["listici"][k], f"{OPIS_TOKA[k]} {ime(h['ben'][2])}, pomična {h['mv']}: tok {s_opis(G, k)}, {koraci(G, k)}")
    for k in ("錯", "綜", "錯綜"):
        h = G["H"][k]
        dodaj(G["izravni"][k], f"{OPIS_TOKA[k]} {ime(h['ben'][2])}: izravni brojevi niza S")
    # 4) nastavak tokova (naizmjence po tokovima, prozor po prozor)
    W = len(G["prozori"]["本"])
    for w in range(W):
        for k in OSNOVNI + IZVEDENI:
            p = G["prozori"][k][w]
            dodaj(p["brojevi"], f"nastavak toka {OPIS_TOKA[k]} {ime(G['H'][k]['ben'][2])}: koraci {p['od']}–{p['do']} → {p['redom']}")
    # 5) sekundarna varijanta: sat-grana 10 (酉)
    ba = A["H"]["本"]
    pref = f"sekundarna varijanta sat-grana {A['sat']} {GRANE[A['sat']]}"
    for k in OSNOVNI:
        h = A["H"][k]
        dodaj(A["listici"][k], f"{pref}: {OPIS_TOKA[k]} {ime(h['ben'][2])}, pomična {h['mv']}: tok {s_opis(A, k)}, {koraci(A, k)}")
    dodaj(A["izravni"]["本"], f"{pref}: 本卦 {ime(ba['ben'][2])} izravni brojevi niza S")
    dodaj(A["trenutak"], f"{pref}: brojevi trenutka Z={A['Z']}, Z+sat={A['Z'] + A['sat']}, KW {ba['ben'][2]}, {ba['hu'][2]}, {ba['bian'][2]}, pomična {A['mv']}")
    for k in ("互k", "變k"):
        dodaj(A["listici"][k], f"{pref}: {OPIS_TOKA[k]} {ime(A['H'][k]['ben'][2])}: tok {s_opis(A, k)}, {koraci(A, k)}")
    for k in ("錯", "綜", "錯綜"):
        dodaj(A["izravni"][k], f"{pref}: {OPIS_TOKA[k]} {ime(A['H'][k]['ben'][2])} izravni brojevi niza S")
    Wa = len(A["prozori"]["本"])
    for w in range(Wa):
        for k in OSNOVNI + IZVEDENI:
            p = A["prozori"][k][w]
            dodaj(p["brojevi"], f"{pref}: nastavak toka {OPIS_TOKA[k]} {ime(A['H'][k]['ben'][2])}: koraci {p['od']}–{p['do']} → {p['redom']}")
    return out


def kljucni_brojevi(k: int = 12) -> list[int]:
    """Težinski najjači brojevi čitanja: KW 本 3, 互/變 2.5, trigrami i pomična 2, Z i Z+sat 1.5,
    listić 本 1, KW i 變 ostalih transformacija 1, korijeni 互/變 0.75."""
    G = izracun(SAT_BROJ, 1)
    b = G["H"]["本"]
    w: Counter = Counter()
    w[svedi(b["ben"][2])] += 3
    w[svedi(b["hu"][2])] += 2.5
    w[svedi(b["bian"][2])] += 2.5
    for x in (G["up"], G["low"], G["mv"]):
        w[x] += 2
    w[svedi(G["Z"])] += 1.5
    w[svedi(G["Z"] + G["sat"])] += 1.5
    w[G["sat"]] += 1
    for x in G["listici"]["本"] or []:
        w[x] += 1
    for kk in ("錯", "綜", "錯綜"):
        h = G["H"][kk]
        w[svedi(h["ben"][2])] += 1
        w[svedi(h["bian"][2])] += 1
    for kk in ("互k", "變k"):
        h = G["H"][kk]
        w[svedi(h["hu"][2])] += 0.75
        w[svedi(h["bian"][2])] += 0.75
    return [x for x, _ in sorted(w.items(), key=lambda t: (-t[1], t[0]))[:k]]


# ------------------------------------------------------------------ postupak (tekst)
def _log_blok(L: list[str], naslov: str, log: list[str]) -> None:
    L.append(f"**{naslov}**\n")
    L.append("```")
    L.extend(log)
    L.append("```\n")


def _sekcija_izracuna(L: list[str], G: dict, naslov: str) -> None:
    L.append(f"## {naslov}\n")
    L.append(f"- Z = {GODINA_BROJ} + {LUNARNI_MJ} + {LUNARNI_DAN} = {G['Z']}; gornji = {G['Z']} mod 8 = {G['up']} {TN[G['up']]} "
             f"({TN_HR[G['up']]}); donji = ({G['Z']} + {G['sat']}) mod 8 = {G['Z'] + G['sat']} mod 8 = {G['low']} {TN[G['low']]} "
             f"({TN_HR[G['low']]}); pomična = {G['Z'] + G['sat']} mod 6 = {G['mv']} (ostatak 0 → 8 odnosno 6).")
    L.append(f"- Temeljne linije (odozdo): {''.join(map(str, G['base']))} = {TN[G['low']]} dolje / {TN[G['up']]} gore.\n")
    L.append("| varijanta | opis |")
    L.append("|---|---|")
    for k in OSNOVNI + ["互k", "變k"]:
        L.append(f"| {OPIS_TOKA[k]} | {_opis_h(G['H'][k])} |")
    L.append("")
    L.append("Nizovi S = [本 gornji, 本 donji, KW 本, 互 gornji, 互 donji, KW 互, 變 gornji, 變 donji, KW 變, pomična]:\n")
    for k in OSNOVNI + IZVEDENI:
        S = G["S"][k]
        L.append(f"- S({k}) = {S}  (zbroj {sum(S)}, ≡ {sum(S) % N} mod 45; svedeno: {[svedi(x) for x in S]})")
    L.append("")
    b = G["H"]["本"]
    L.append(f"體/用 (informativno): 體 {TN[b['ti']]} = {WX[b['ti']]}, 用 {TN[b['yong']]} = {WX[b['yong']]}; "
             f"互 {TN[b['hu'][0]]}/{TN[b['hu'][1]]} ({WX[b['hu'][0]]}/{WX[b['hu'][1]]}), "
             f"變 {TN[b['bian'][0]]}/{TN[b['bian'][1]]} ({WX[b['bian'][0]]}/{WX[b['bian'][1]]}).\n")
    L.append("### Osnovni listići — cijeli log koraka (pakiranje ≤ 2 zajednička broja samo među vlastitim listićima, redoslijed 本, 錯, 綜, 錯綜, zatim 互k, 變k, 本後天)\n")
    for k in OSNOVNI + IZVEDENI:
        t = G["listici"][k]
        _log_blok(L, f"{OPIS_TOKA[k]} {ime(G['H'][k]['ben'][2])} → listić {_fmt(t) if t else 'NEMA'}", G["logovi"][k])
    L.append("### Izravni listići (prvih 6 različitih vrijednosti niza S svedenih u 1..45)\n")
    for k in OSNOVNI + IZVEDENI:
        t = G["izravni"][k]
        L.append(f"- {OPIS_TOKA[k]}: {_fmt(t) if t else 'manje od 6 različitih'}")
    if G["trenutak"]:
        L.append(f"- brojevi trenutka (Z, Z+sat, KW 本/互/變, pomična): {_fmt(G['trenutak'])}")
    L.append("")
    L.append("### Nastavak tokova (prozori po 6 koraka iza osnovnog listića)\n")
    L.append("| tok | prozor | koraci | redom izvučeno | kandidat |")
    L.append("|---|---:|---|---|---|")
    for k in OSNOVNI + IZVEDENI:
        for j, p in enumerate(G["prozori"][k], 1):
            L.append(f"| {OPIS_TOKA[k]} | {j} | {p['od']}–{p['do']} | {p['redom']} | `{_fmt(p['brojevi'])}` |")
    L.append("")


def postupak_tekst(W: int = 6) -> str:
    G = izracun(SAT_BROJ, W)
    A = izracun(SAT_ALT, W)
    L = [f"# {METODA} (梅花易數, vremenska metoda 年月日時起卦) — {DATUM_TEKST}\n"]
    L.append("## 1. Ulaz i pravila\n")
    L.append(f"- Godina 丙午 (Vatreni konj) → grana 午 = {GODINA_BROJ}. Lunarni datum: {LUNARNI_MJ}. mjesec, {LUNARNI_DAN}. dan "
             "(lunardate: 2026-10-29 → LunarDate(2026, 9, 20), nije prijestupni).")
    L.append(f"- Sat: sunčevo podne u Zagrebu 29.10.2026. je 11:39:48 CET → 20:00 CET = 20:20 pravog sunčevog vremena → dvosatno "
             f"razdoblje 戌 (19–21 h) = grana {SAT_BROJ}. (I zonsko vrijeme 20:00 pada u 戌; grana {SAT_ALT} 酉 je sekundarna varijanta po uputi, vidi §3.)")
    L.append("- Trigrami i prijenebeski (先天) brojevi: " + ", ".join(f"{TN[i]} {''.join(map(str, TL[i]))} = {i} ({TN_HR[i]}, {WX[i]})" for i in range(1, 9)) + ".")
    L.append("- 後天 (Luoshu) brojevi trigrama: " + ", ".join(f"{TN[i]} = {HT[i]}" for i in range(1, 9)) + ".")
    L.append("- Gornji = Z mod 8, donji = (Z + sat) mod 8, pomična = (Z + sat) mod 6 (0 → 8 / 6). 互 = linije 2-3-4 dolje, 3-4-5 gore. "
             "變 = pomična linija preokrenuta. 體 = trigram bez pomične linije (m ≤ 3 → gornji, m ≥ 4 → donji), 用 = drugi.")
    L.append("- 錯 = komplement linija (ista pomična), 綜 = naglavce (pomična 7 − m), 錯綜 = oboje. 互k/變k = 互卦/變卦 uzeti kao korijen s istom pomičnom linijom.")
    L.append("- Tok: kumulativni zbroj c niza S (ciklički), korak daje ((c − 1) mod 45) + 1; zauzet broj → (x mod 45) + 1. Listić = 6 koraka; "
             "provjera: ≤ 2 zajednička broja s prihvaćenim listićima; ne prođe li, odbacuje se zadnji broj i uzima sljedeći korak (skripta). "
             "Ako se već prvih 5 preklapa u ≥ 3 broja, pravilo ne može uspjeti → preskok cijelog prozora (dopuna, dokumentirana u logu).")
    L.append("- Bez slučajnosti; popular_ok filtar skripte se ne primjenjuje.\n")
    _sekcija_izracuna(L, G, f"2. Primarni izračun — sat-grana {SAT_BROJ} {GRANE[SAT_BROJ]}")
    L.append(f"Napomena: 小過 (62) i 中孚 (61) simetrični su heksagrami (naglavce daju sebe), pa je 綜 = 本 i 錯綜 = 錯 osim pomične linije (2 umjesto 5), "
             f"變卦 i zamjene 體/用; nizovi S razlikuju se tek na mjestima 7–10, a prvih 6 koraka toka je identično — zato log pokazuje preskok prozora.\n")
    _sekcija_izracuna(L, A, f"3. Sekundarna varijanta — sat-grana {SAT_ALT} {GRANE[SAT_ALT]} (test osjetljivosti)")
    L.append("## 4. Rang kandidata\n")
    L.append("1. 本卦: tok (koraci 1–6), izravni brojevi niza S, brojevi trenutka, 後天 varijanta toka;")
    L.append("2. 互卦 i 變卦 kao korijen (tok, zatim izravni);")
    L.append("3. 錯 / 綜 / 錯綜 (tok, zatim izravni; duplikati izravnih zbog simetrije otpadaju);")
    L.append("4. nastavak svih sedam tokova, prozor po prozor naizmjence;")
    L.append(f"5. sekundarna varijanta sat-grana {SAT_ALT} (isti redoslijed).\n")
    L.append("## 5. Ključni brojevi\n")
    L.append("Težinski zbroj (KW 本 3; 互/變 2.5; gornji, donji, pomična 2; Z i Z+sat 1.5; sat 1; listić 本 1; KW i 變 od 錯/綜/錯綜 1; korijeni 0.75): "
             + ", ".join(map(str, kljucni_brojevi())))
    b = G["H"]["本"]
    L.append(f"\nJoker iz skripte (informativno, nije kandidat za Loto): "
             + "".join(map(str, [b["ben"][0], b["ben"][1], b["hu"][0], b["hu"][1], b["bian"][0], b["bian"][1]])))
    return "\n".join(L)


# ------------------------------------------------------------------ provjere
def _provjere() -> None:
    assert sorted(x for r in KW.values() for x in r) == list(range(1, 65))
    for (u, l), k in {(1, 1): 1, (8, 8): 2, (6, 4): 3, (7, 6): 4, (8, 1): 11, (1, 8): 12, (6, 3): 63, (3, 6): 64,
                      (4, 7): 62, (5, 2): 61, (2, 5): 28, (7, 4): 27, (2, 7): 31, (4, 5): 32, (5, 4): 42, (7, 2): 41}.items():
        assert kw(u, l) == k, (u, l, k)
    G = izracun(SAT_BROJ, 2)
    assert (G["Z"], G["up"], G["low"], G["mv"]) == (36, 4, 7, 5)
    b = G["H"]["本"]
    assert b["ben"] == (4, 7, 62) and b["hu"] == (2, 5, 28) and b["bian"] == (2, 7, 31) and (b["ti"], b["yong"]) == (7, 4)
    assert G["S"]["本"] == [4, 7, 62, 2, 5, 28, 2, 7, 31, 5]
    assert G["S"]["錯"] == [5, 2, 61, 7, 4, 27, 7, 2, 41, 5]
    assert G["S"]["綜"] == [4, 7, 62, 2, 5, 28, 4, 5, 32, 2]
    assert G["S"]["錯綜"] == [5, 2, 61, 7, 4, 27, 5, 4, 42, 2]
    assert sorted(G["listici"]["本"]) == [4, 11, 18, 28, 30, 35]
    assert sorted(G["listici"]["錯"]) == [5, 7, 16, 23, 30, 34]
    assert sorted(G["listici"]["綜"]) == [14, 16, 20, 22, 27, 28]
    assert sorted(G["listici"]["錯綜"]) == [21, 22, 24, 25, 29, 31]
    assert G["izravni"]["本"] == [2, 4, 5, 7, 17, 28]
    prih = [t for t in G["listici"].values() if t]
    assert all(len(set(a) & set(c)) <= 2 for i, a in enumerate(prih) for c in prih[i + 1:])
    A = izracun(SAT_ALT, 1)
    assert (A["low"], A["mv"]) == (6, 4) and A["H"]["本"]["ben"] == (4, 6, 40)
    assert svedi(45) == 45 and svedi(46) == 1 and svedi(62) == 17 and svedi(90) == 45
    # neobvezne vanjske provjere datuma (fallback ako biblioteka nedostaje)
    try:
        import lunardate  # type: ignore
        ld = lunardate.LunarDate.from_solar_date(2026, 10, 29) if hasattr(lunardate.LunarDate, "from_solar_date") \
            else lunardate.LunarDate.fromSolarDate(2026, 10, 29)
        assert (ld.month, ld.day) == (LUNARNI_MJ, LUNARNI_DAN), ld
    except ImportError:
        pass
    try:
        import ephem  # type: ignore
        o = ephem.Observer()
        o.lat, o.lon, o.elevation = "45.815", "15.98", 120
        o.date = "2026/10/29 06:00:00"
        podne_utc = o.next_transit(ephem.Sun()).datetime()        # sunčevo podne (UTC) ≈ 10:39:48
        # pravo sunčevo vrijeme za 20:00 CET = 19:00 UTC: 12:00 + (19:00 − podne_utc)
        minuta = 12 * 60 + (19 * 60 - (podne_utc.hour * 60 + podne_utc.minute + podne_utc.second / 60))
        assert 20 * 60 + 15 <= minuta <= 20 * 60 + 25, minuta      # ≈ 20:20
        assert 19 * 60 <= minuta < 21 * 60                         # → 戌 = 11
    except ImportError:
        pass


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    _provjere()
    tekst = postupak_tekst()
    print(tekst)
    kand = kandidati(80)
    print("\n## Prvih 10 kandidata\n")
    for c in kand[:10]:
        print(f"{c['rang']:3d}. {_fmt(c['brojevi'])}  — {c['obrazlozenje']}")
    kljuc = kljucni_brojevi()
    print("\nKljučni brojevi:", kljuc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"ezo_{SLUG}.json"
    out.write_text(json.dumps({"metoda": METODA, "slug": SLUG, "postupak": tekst,
                               "kljucni_brojevi": kljuc, "kandidati": kand},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nZapisano: {out}  ({len(kand)} kandidata)")
