"""I Ching: King Wen tablica, stabljike (6:7:8:9 = 1:5:7:3), nuklearni i promijenjeni heksagram, Fu Xi kod, Meihua linije."""
from __future__ import annotations
import numpy as np

# Fu Xi brojevi trigrama; bitovi = linije odozdo prema gore (1 = yang)
TRIGRAM_FUXI = {"Qian": 1, "Dui": 2, "Li": 3, "Zhen": 4, "Xun": 5, "Kan": 6, "Gen": 7, "Kun": 8}
TRIGRAM_LINES = {"Qian": (1, 1, 1), "Dui": (1, 1, 0), "Li": (1, 0, 1), "Zhen": (1, 0, 0), "Xun": (0, 1, 1),
                 "Kan": (0, 1, 0), "Gen": (0, 0, 1), "Kun": (0, 0, 0)}
_LINES_TO_NAME = {v: k for k, v in TRIGRAM_LINES.items()}
FUXI_NAME = {v: k for k, v in TRIGRAM_FUXI.items()}

# King Wen: (gornji, donji) -> broj
_KW_BY_NAME = {
    "Qian": {"Qian": 1, "Zhen": 25, "Kan": 6, "Gen": 33, "Kun": 12, "Xun": 44, "Li": 13, "Dui": 10},
    "Zhen": {"Qian": 34, "Zhen": 51, "Kan": 40, "Gen": 62, "Kun": 16, "Xun": 32, "Li": 55, "Dui": 54},
    "Kan": {"Qian": 5, "Zhen": 3, "Kan": 29, "Gen": 39, "Kun": 8, "Xun": 48, "Li": 63, "Dui": 60},
    "Gen": {"Qian": 26, "Zhen": 27, "Kan": 4, "Gen": 52, "Kun": 23, "Xun": 18, "Li": 22, "Dui": 41},
    "Kun": {"Qian": 11, "Zhen": 24, "Kan": 7, "Gen": 15, "Kun": 2, "Xun": 46, "Li": 36, "Dui": 19},
    "Xun": {"Qian": 9, "Zhen": 42, "Kan": 59, "Gen": 53, "Kun": 20, "Xun": 57, "Li": 37, "Dui": 61},
    "Li": {"Qian": 14, "Zhen": 21, "Kan": 64, "Gen": 56, "Kun": 35, "Xun": 50, "Li": 30, "Dui": 38},
    "Dui": {"Qian": 43, "Zhen": 17, "Kan": 47, "Gen": 31, "Kun": 45, "Xun": 28, "Li": 49, "Dui": 58},
}
KING_WEN: dict[tuple[int, int], int] = {(TRIGRAM_FUXI[u], TRIGRAM_FUXI[l]): n for u, row in _KW_BY_NAME.items() for l, n in row.items()}

HEXAGRAM_NAMES = {
    1: "Qian / Stvaralacko", 2: "Kun / Primajuce", 3: "Zhun / Pocetna teskoca", 4: "Meng / Mladenacka ludost",
    5: "Xu / Cekanje", 6: "Song / Sukob", 7: "Shi / Vojska", 8: "Bi / Zajednistvo", 9: "Xiao Xu / Mala pripitomljujuca snaga",
    10: "Lu / Gazenje", 11: "Tai / Mir", 12: "Pi / Zastoj", 13: "Tong Ren / Zajednistvo s ljudima", 14: "Da You / Veliki posjed",
    15: "Qian / Skromnost", 16: "Yu / Odusevljenje", 17: "Sui / Slijedjenje", 18: "Gu / Rad na pokvarenom",
    19: "Lin / Priblizavanje", 20: "Guan / Promatranje", 21: "Shi He / Progrizanje", 22: "Bi / Ljupkost",
    23: "Bo / Raspadanje", 24: "Fu / Povratak", 25: "Wu Wang / Nevinost", 26: "Da Xu / Velika pripitomljujuca snaga",
    27: "Yi / Hranjenje", 28: "Da Guo / Prevaga velikog", 29: "Kan / Bezdan", 30: "Li / Prianjanje",
    31: "Xian / Utjecaj", 32: "Heng / Trajanje", 33: "Dun / Povlacenje", 34: "Da Zhuang / Snaga velikog",
    35: "Jin / Napredak", 36: "Ming Yi / Pomracenje svjetla", 37: "Jia Ren / Obitelj", 38: "Kui / Suprotnost",
    39: "Jian / Zapreka", 40: "Xie / Oslobodjenje", 41: "Sun / Smanjenje", 42: "Yi / Povecanje",
    43: "Guai / Proboj", 44: "Gou / Susret", 45: "Cui / Okupljanje", 46: "Sheng / Uzdizanje",
    47: "Kun / Tlacenje", 48: "Jing / Bunar", 49: "Ge / Preokret", 50: "Ding / Kotao",
    51: "Zhen / Pobudjujuce", 52: "Gen / Mirovanje", 53: "Jian / Razvoj", 54: "Gui Mei / Djevojka koja se udaje",
    55: "Feng / Obilje", 56: "Lu / Putnik", 57: "Xun / Blago", 58: "Dui / Radosno",
    59: "Huan / Rasprsenje", 60: "Jie / Ogranicenje", 61: "Zhong Fu / Unutarnja istina", 62: "Xiao Guo / Prevaga malog",
    63: "Ji Ji / Nakon dovrsenja", 64: "Wei Ji / Prije dovrsenja",
}

YARROW_P = {6: 1 / 16, 7: 5 / 16, 8: 7 / 16, 9: 3 / 16}


def cast_yarrow(rng: np.random.Generator) -> list[int]:
    vals = [6, 7, 8, 9]
    p = [YARROW_P[v] for v in vals]
    return [int(rng.choice(vals, p=p)) for _ in range(6)]


def _is_yang(v: int) -> int:
    return 1 if v in (7, 9) else 0


def _trigram(three: list[int]) -> int:
    return TRIGRAM_FUXI[_LINES_TO_NAME[tuple(_is_yang(v) for v in three)]]


def lines_to_trigrams(lines: list[int]) -> tuple[int, int]:
    assert len(lines) == 6
    return _trigram(lines[3:6]), _trigram(lines[0:3])


def king_wen(lines: list[int]) -> int:
    return KING_WEN[lines_to_trigrams(lines)]


def nuclear(lines: list[int]) -> list[int]:
    lower = [7 if _is_yang(v) else 8 for v in lines[1:4]]
    upper = [7 if _is_yang(v) else 8 for v in lines[2:5]]
    return lower + upper


def changed(lines: list[int]) -> list[int]:
    return [8 if v == 9 else 7 if v == 6 else v for v in lines]


def moving_positions(lines: list[int]) -> list[int]:
    return [i + 1 for i, v in enumerate(lines) if v in (6, 9)]


def fuxi_code(lines: list[int]) -> int:
    u, l = lines_to_trigrams(lines)
    return u * 8 + l


def hexagram_name(n: int) -> str:
    return HEXAGRAM_NAMES[n]


def lines_text(lines: list[int]) -> str:
    sym = {6: "-- x --  (6, stari yin, pomicna)", 7: "-------  (7, mladi yang)", 8: "--   --  (8, mladi yin)",
           9: "---o---  (9, stari yang, pomicna)"}
    out = []
    for i in range(6, 0, -1):
        out.append(f"{i}: {sym[lines[i-1]]}")
    u, l = lines_to_trigrams(lines)
    out.append(f"gornji {FUXI_NAME[u]} ({u}), donji {FUXI_NAME[l]} ({l})")
    return "\n".join(out)


def opposite(lines: list[int]) -> list[int]:
    """Suprotni heksagram (cuo gua): sve linije obrnute, staticne."""
    return [8 if _is_yang(v) else 7 for v in lines]


def reversed_hex(lines: list[int]) -> list[int]:
    """Obrnuti heksagram (zong gua): naopako, staticne linije."""
    return [7 if _is_yang(v) else 8 for v in reversed(lines)]


def _reserve(lines: list[int]) -> list[tuple[str, int]]:
    """Rezervna kaskada I Chinga: KW suprotnog, KW obrnutog, KW nuklearnog promijenjenog, Fu Xi kod nuklearnog i
    promijenjenog, Fu Xi gornjeg i donjeg trigrama, vrijednosti linija 1-6, polozaji nepomicnih linija."""
    u, l = lines_to_trigrams(lines)
    res = [(f"KW suprotnog {HEXAGRAM_NAMES[king_wen(opposite(lines))]}", king_wen(opposite(lines))),
           (f"KW obrnutog {HEXAGRAM_NAMES[king_wen(reversed_hex(lines))]}", king_wen(reversed_hex(lines))),
           ("KW nuklearnog promijenjenog", king_wen(nuclear(changed(lines)))),
           ("FuXi kod nuklearnog", fuxi_code(nuclear(lines))), ("FuXi kod promijenjenog", fuxi_code(changed(lines))),
           (f"FuXi gornji {FUXI_NAME[u]}", u), (f"FuXi donji {FUXI_NAME[l]}", l)]
    res += [(f"linija{i+1}", v) for i, v in enumerate(lines)]
    res += [(f"nepomicna linija", i + 1) for i, v in enumerate(lines) if v in (7, 8)]
    return res


def iching_cascade(lines: list[int]) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """(kaskada, rezerva): KW primarnog, nuklearnog, promijenjenog, zbroj linija, polozaji pomicnih, Fu Xi kod."""
    kw = king_wen(lines)
    nu = king_wen(nuclear(lines))
    ch = king_wen(changed(lines))
    casc = [(f"KW primarni {HEXAGRAM_NAMES[kw]}", kw), (f"KW nuklearni {HEXAGRAM_NAMES[nu]}", nu),
            (f"KW promijenjeni {HEXAGRAM_NAMES[ch]}" + (" (bez pomicnih = primarni)" if ch == kw else ""), ch),
            ("zbroj linija", sum(lines))]
    casc += [(f"pomicna linija", p) for p in moving_positions(lines)]
    casc.append((f"FuXi gornji*8+donji", fuxi_code(lines)))
    return casc, _reserve(lines)


def meihua_lines(upper_fuxi: int, lower_fuxi: int, moving: int) -> list[int]:
    lo = TRIGRAM_LINES[FUXI_NAME[lower_fuxi]]
    up = TRIGRAM_LINES[FUXI_NAME[upper_fuxi]]
    lines = [7 if b else 8 for b in lo] + [7 if b else 8 for b in up]
    assert 1 <= moving <= 6
    lines[moving - 1] = 9 if lines[moving - 1] == 7 else 6
    return lines


def meihua_cascade(upper: int, lower: int, moving: int, sum1: int, sum2: int) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    lines = meihua_lines(upper, lower, moving)
    casc, res = iching_cascade(lines)
    casc += [("zbroj godina+mjesec+dan", sum1), ("zbroj godina+mjesec+dan+sat", sum2)]
    return casc, res


def selftest() -> None:
    F = TRIGRAM_FUXI
    assert KING_WEN[(F["Kun"], F["Kun"])] == 2
    assert KING_WEN[(F["Xun"], F["Kun"])] == 20
    assert KING_WEN[(F["Gen"], F["Kun"])] == 23
    assert KING_WEN[(F["Dui"], F["Xun"])] == 28
    assert KING_WEN[(F["Qian"], F["Qian"])] == 1 and KING_WEN[(F["Kan"], F["Kan"])] == 29 and KING_WEN[(F["Li"], F["Li"])] == 30
    assert KING_WEN[(F["Zhen"], F["Zhen"])] == 51 and KING_WEN[(F["Gen"], F["Gen"])] == 52 and KING_WEN[(F["Xun"], F["Xun"])] == 57
    assert KING_WEN[(F["Dui"], F["Dui"])] == 58 and KING_WEN[(F["Kun"], F["Qian"])] == 11 and KING_WEN[(F["Qian"], F["Kun"])] == 12
    assert KING_WEN[(F["Kan"], F["Li"])] == 63 and KING_WEN[(F["Li"], F["Kan"])] == 64
    assert KING_WEN[(F["Kun"], F["Zhen"])] == 24 and KING_WEN[(F["Qian"], F["Xun"])] == 44 and KING_WEN[(F["Dui"], F["Qian"])] == 43
    assert sorted(KING_WEN.values()) == list(range(1, 65))
    assert len(HEXAGRAM_NAMES) == 64
    assert king_wen([7] * 6) == 1 and king_wen([8] * 6) == 2
    assert king_wen(nuclear([7] * 6)) == 1 and king_wen(nuclear([8] * 6)) == 2
    jiji = [7, 8, 7, 8, 7, 8]           # Kan nad Li = 63
    assert king_wen(jiji) == 63 and king_wen(nuclear(jiji)) == 64
    assert changed([9, 6, 7, 8, 9, 6]) == [8, 7, 7, 8, 8, 7] and moving_positions([9, 6, 7, 8, 9, 6]) == [1, 2, 5, 6]
    assert fuxi_code([8] * 6) == 72 and fuxi_code([7] * 6) == 9
    # Xun nad Kun = 20: donji Kun (8,8,8), gornji Xun (0,1,1) -> linije 8,7,7
    assert king_wen([8, 8, 8, 8, 7, 7]) == 20
    assert meihua_lines(5, 8, 1) == [6, 8, 8, 8, 7, 7]
    assert king_wen(opposite([8] * 6)) == 1 and king_wen(reversed_hex([7, 8, 8, 8, 8, 8])) == 23  # Fu naopako = Bo
    assert len(_reserve([8] * 6)) == 7 + 6 + 6
    rng = np.random.default_rng(0)
    cnt = {6: 0, 7: 0, 8: 0, 9: 0}
    n = 100_000 // 6
    for _ in range(n):
        for v in cast_yarrow(rng):
            cnt[v] += 1
    tot = sum(cnt.values())
    for v, p in YARROW_P.items():
        assert abs(cnt[v] / tot - p) < 0.01, (v, cnt[v] / tot, p)
    print("empirijske frekvencije stabljika:", {v: round(cnt[v] / tot, 4) for v in cnt})
    print(lines_text([9, 6, 7, 8, 9, 6]))
    print("iching selftest OK")


if __name__ == "__main__":
    selftest()
