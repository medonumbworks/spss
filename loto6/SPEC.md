# Interna specifikacija (za implementaciju modula) — Loto 6/45 registar

Jezik koda: Python 3.13, venv `/home/user/spss/.venv` (pyswisseph 2.10.3.2, lunardate 0.3.0,
pyluach 2.3.0, convertdate 2.5.1, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1).
Svi moduli zive u `/home/user/spss/loto6/` i uvoze se kao obicni moduli (pokretanje iz tog direktorija).
Nema mrežnog pristupa. Nema predikcije: svaka kombinacija ima 1 : 8.145.060 za sesticu.

## Zajednicke konvencije (core.py, config.py — vec napisani, NE mijenjati potpise)

- `m45(n) = ((n-1) % 45) + 1` (Python modulo; vrijedi i za n <= 0).
- Kaskada = uredjen popis kandidata `[(oznaka:str, vrijednost:int), ...]` (sirove vrijednosti PRIJE m45).
- Kombinacija = prvih 6 razlicitih m45-vrijednosti kaskade (redom). Ako ih je < 6, dopuna iz
  rezervne kaskade sustava (isti format), oznaka `dopuna=True`. Ako ni to nije dovoljno, krajnja
  rezerva: najmanji broj 1..45 koji jos nije u kombinaciji, oznaka `dopuna_krajnja=True`.
- Joker = zadnje znamenke prvih 6 sirovih vrijednosti kaskade (prije m45), ukljucujuci duplikate,
  kao tekst s vodecim nulama; `abs()` vrijednosti; ako kaskada ima < 6 vrijednosti nastavlja se rezervnom.
- Slucajnost: `rng_for(draw_ordinal:int, system_id:int) -> numpy.random.Generator` =
  `default_rng(SeedSequence(SEED, spawn_key=(draw_ordinal, system_id)))`. SEED se cita iz `seed.txt`
  (jedan za cijeli posao, zapisuje se u svaki izlaz). draw_ordinal = 1-based redni broj kola u razdoblju
  (fiksni/godisnji izracuni koriste draw_ordinal = 0). system_id = broj sustava (1..15).
- Datum/vrijeme: `datetime` s `ZoneInfo("Europe/Zagreb")` (aware). Kolo = lokalni datum + DRAW_TIME_LOCAL.
- Imena dana (bez dijakritika, velika slova): PONEDJELJAK UTORAK SRIJEDA CETVRTAK PETAK SUBOTA NEDJELJA.
- Planet dana (weekday Mon=0..Sun=6): Mjesec, Mars, Merkur, Jupiter, Venera, Saturn, Sunce.
- Brojevi planeta (kaldejska/numeroloska tablica iz zadatka): Sunce 1, Mjesec 2, Jupiter 3, Merkur 5,
  Venera 6, Saturn 8, Mars 9 (Rahu 4, Ketu 7 samo u Jyotisu).

## Moduli koje treba napisati (svaki s `if __name__ == "__main__": selftest()` koji podize AssertionError)

### astro.py (pyswisseph, Moshier efemerida: flag `swe.FLG_MOSEPH`; sideralno `swe.SIDM_LAHIRI`)
- `jd_ut(dt: datetime) -> float` (aware datetime -> JD UT)
- `tropical_lon(jd: float, body: int) -> float` (0..360)
- `sidereal_lon(jd: float, body: int) -> float` (Lahiri, 0..360)
- `ascendant(jd: float, lat: float, lon: float, sidereal: bool=False) -> float` (Placidus, 'P')
- `sunrise_sunset(day: date, lat: float, lon: float) -> tuple[datetime, datetime]` (lokalni aware datetimei;
  izlazak i zalazak ISTOG lokalnog dana; standardni gornji rub + refrakcija = swe.rise_trans flag 0)
- `is_day(dt: datetime, lat: float, lon: float) -> bool` (Sunce iznad horizonta: izmedju izlaska i zalaska tog dana)
- `planetary_hour(dt: datetime, lat: float, lon: float) -> dict` s kljucevima:
  `planet` (hr ime), `hour_index` (1..24 od izlaska planetarnog dana), `day_ruler`, `is_day_hour`,
  `start`, `end` (aware datetimei), `sunrise`, `sunset`, `next_sunrise`, `planetary_day_date` (date).
  Planetarni dan traje od izlaska do sljedeceg izlaska; ako je dt prije izlaska tog civilnog dana, pripada
  prethodnom planetarnom danu. 12 dnevnih sati (izlazak->zalazak)/12 i 12 nocnih (zalazak->sljedeci izlazak)/12.
  Prvi sat = vladar dana; dalje kaldejski red Saturn, Jupiter, Mars, Sunce, Venera, Merkur, Mjesec (ciklicki).
- `nakshatra(lon_sid: float) -> tuple[int, int, str]` -> (1..27, pada 1..4, gospodar hr: Ketu, Venera, Sunce,
  Mjesec, Mars, Rahu, Jupiter, Saturn, Merkur ciklicki od Asvini)
- `rashi(lon: float) -> int` (1..12)
- `vimshottari(natal_moon_sid: float, birth: datetime, at: datetime) -> dict` s
  `mahadasha`, `antardasha` (hr imena gospodara), `md_start`, `md_end`, `ad_start`, `ad_end`, `balance_years`.
  Godina = 365.25 dana. Redoslijed i trajanja: Ketu 7, Venera 20, Sunce 6, Mjesec 10, Mars 7, Rahu 18,
  Jupiter 16, Saturn 19, Merkur 17. Antardase proporcionalno (MD_trajanje * AD_gospodar/120), pocevsi od MD gospodara.
- `pars_fortunae(asc: float, sun: float, moon: float, is_day: bool) -> float` (dan: asc+moon-sun; noc: asc+sun-moon; mod 360)
- `equation_of_time_minutes(jd: float) -> float` (swe.time_equ vraca dane -> * 1440)
- `true_solar_time(dt: datetime, lon: float) -> datetime` (naive, "sat pravog suncevog vremena": UT + lon/15 h + EoT)
- `deg_in_sign(lon: float) -> int` (1..30 = floor(lon % 30) + 1), `sign_no(lon) -> int` (1..12)
- `PLANET_HR`: dict swe konstanta -> hr ime; `DAY_RULER_BODY`: weekday -> swe konstanta.
Selftest: natal (1991-05-31 18:20 Europe/Zagreb, 44.1167N 15.2333E): tropsko Sunce u 9..10 Blizanaca
(69.5..70.0), Mjesec sideralno u Purva Asadhi (nakshatra 20), ASC tropski ~14.9 Skorpiona; Vimsottari
u studenom 2026 = Rahu/Jupiter (provjeri rucno iz ostatka Venere ~8.74 g); planetary_hour: sat 1 pocinje
na izlazak i vladar = vladar dana; 24 sata pokrivaju tocno izlazak->sljedeci izlazak.

### calendars.py
- `hebrew_date(dt: datetime, lat: float, lon: float) -> dict`: `year, month_no, month_name, day, after_sunset,
  sunset` (pyluach `HebrewDate.from_pydate`; ako je dt >= zalazak tog dana u Zagrebu -> pydate + 1 dan;
  unakrsna provjera s convertdate.hebrew mora dati isti (y, m, d) — assert; napomena: pyluach i convertdate
  numeriraju mjesece od Nisana=1, Tisri=7, provjeri i dokumentiraj u docstringu).
- `chinese_lunar(dt: datetime, lon: float) -> dict`: `lunar_year, lunar_month, lunar_day, is_leap_month,
  year_branch (1..12, ((lunar_year-4) % 12)+1), year_ganzhi (str), hour_branch (1..12), tst (true solar time),
  branch_name`. Dvosatni ogranak po PRAVOM suncevom vremenu (astro.true_solar_time): Zi=1 23:00-01:00,
  Chou=2 01-03, Yin=3 03-05, Mao=4 05-07, Chen=5 07-09, Si=6 09-11, Wu=7 11-13, Wei=8 13-15, Shen=9 15-17,
  You=10 17-19, Xu=11 19-21, Hai=12 21-23. Ako je TST >= 23:00 lunarni dan se uzima SLJEDECI (rani Zi =
  novi dan), dokumentirati. lunardate: `LunarDate.from_solar_date`.
Selftest: kineska nova godina 2026 = 2026-02-17 (丙午), 2027 = 2027-02-06 (丁未); year_branch 2026 = 7, 2027 = 8;
Hebrew 2026-11-01 prije zalaska = 21 Hesvan 5787; Ros Hasana 5787 = 2026-09-12 (1 Tisri).

### iching.py
- Trigrami Fu Xi: Qian 1 (111), Dui 2 (110), Li 3 (101), Zhen 4 (100), Xun 5 (011), Kan 6 (010), Gen 7 (001), Kun 8 (000);
  bitovi su linije ODOZDO prema gore (prva znamenka = donja linija; 1 = yang).
- `KING_WEN: dict[tuple[int,int], int]` kljuc (gornji_fuxi, donji_fuxi) -> broj 1..64; cijela tablica upisana rucno.
- `cast_yarrow(rng) -> list[int]` 6 linija odozdo prema gore, vrijednosti 6/7/8/9 s vjerojatnostima 1/16, 5/16, 7/16, 3/16
  (koristiti `rng.choice([6,7,8,9], p=[1/16,5/16,7/16,3/16])` po liniji).
- `lines_to_trigrams(lines) -> (upper_fuxi, lower_fuxi)`; yang = 7 ili 9, yin = 6 ili 8.
- `king_wen(lines) -> int`; `nuclear(lines) -> list[int]` (donji = linije 2-4, gornji = linije 3-5, kao staticne 7/8);
  `changed(lines) -> list[int]` (9->8, 6->7, ostale ostaju; ako nema pomicnih, vraca iste linije);
  `moving_positions(lines) -> list[int]` (1..6); `fuxi_code(lines) -> int` (gornji*8 + donji);
  `hexagram_name(n) -> str` (pinyin + hr/eng ime, npr. "Qian / Stvaralacko") za svih 64.
- `iching_cascade(lines) -> list[(oznaka, vrijednost)]` redom: KW primarnog, KW nuklearnog, KW promijenjenog,
  zbroj linija, polozaji pomicnih linija (svaki zasebno), Fu Xi kod. Rezerva: vrijednosti linija (6 kom), Fu Xi
  gornjeg i donjeg trigrama, KW nuklearnog promijenjenog.
- `meihua_lines(upper_fuxi, lower_fuxi, moving) -> list[int]` (staticne 7/8, pomicna linija 9 ako yang, 6 ako yin).
Selftest asserti: KW(Kun,Kun)=2, KW(Xun nad Kun)=20, KW(Gen nad Kun)=23, KW(Dui nad Xun)=28, KW(Qian,Qian)=1,
KW(Kan,Kan)=29, KW(Li,Li)=30, KW(Zhen,Zhen)=51, KW(Gen,Gen)=52, KW(Xun,Xun)=57, KW(Dui,Dui)=58, KW(Kun nad Qian)=11,
KW(Qian nad Kun)=12, KW(Kan nad Li)=63, KW(Li nad Kan)=64; svih 64 vrijednosti razlicito; nuklearni od 1 = 1,
nuklearni od 2 = 2, nuklearni od 63 (Kan nad Li) = 64; empirijska provjera vjerojatnosti stabljika (1e5 bacanja, tol 0.01).

### kamea.py
- `KAMEA: dict[str, list[list[int]]]` za Saturn 3x3, Jupiter 4x4, Mars 5x5, Sunce 6x6, Venera 7x7, Merkur 8x8, Mjesec 9x9
  (Agrippine tablice, De occulta philosophia II.22; koristi standardne Agrippine rasporede).
- `magic_constant(n) = n*(n*n+1)//2`; `check(planet)` assertira sve retke, stupce i obje dijagonale = konstanta
  i da su brojevi tocno 1..n^2.
- `kamea_cascade(planet) -> list[(oznaka, vrijednost)]`: sredisnji red (za parni n dva sredisnja reda: n/2-1 pa n/2,
  0-indeksirano), zatim konstanta, zatim ukupni zbroj n^2(n^2+1)/2. Rezerva: ostali redovi odozgo prema dolje, pa stupci slijeva.
Selftest: check za svih 7; Saturn sredisnji red = [3,5,7]; Sunce konstanta 111, zbroj 666; Mjesec konstanta 369, zbroj 3321.

### cards.py
- `LENORMAND: list[str]` 36 imena (1 Jahac ... 36 Kriz, standardni redoslijed Petit Lenormand).
- `lenormand_cascade(rng) -> (tableau: list[list[int]] 4 reda po 9, cascade)`: `rng.permutation(36)+1`; cascade =
  zbrojevi susjednih karata u 1. redu (8 zbrojeva, oznaka "r1:k1+k2"), zatim preostalih 27 karata redom (redovi 2-4).
- `TAROT: list[str]` 78 imena; indeksi: velike arkane 1-21 (Mag=1 ... Svijet=21), Luda=22, stapovi 23-36 (As..Kralj),
  pehari 37-50, macevi 51-64, diskovi 65-78.
- `tarot_cascade(rng) -> (order: list[int], cascade)`: `rng.permutation(78)+1`; cascade = karte u redoslijedu mijesanja s
  indeksom <= 45 (svih 45 takvih, oznaka = ime karte).

### numerology.py
- `CHALDEAN` iz zadatka (A1 B2 C3 D4 E5 F8 G3 H5 I1 J1 K2 L3 M4 N5 O7 P8 Q1 R2 S3 T4 U6 V6 W6 X5 Y1 Z7);
  `chaldean_word(s) -> int` (zbroj slova, razmaci se ignoriraju); `chaldean_letters(s) -> list[(slovo, vr)]`.
- `GEMATRIA_WORDS = [("מזל",77),("ברכה",227),("שפע",450),("זהב",14),("חי",18),("הצלחה",138),("כסף",160),("עושר",576)]`
  + `HEBREW_VALUES` standardne (bez sofita: ך=20, ם=40, ן=50, ף=80, ץ=90) i assert da izracun daje navedene vrijednosti;
  `hebrew_letters(word) -> list[(slovo, vr)]`.
- `PYTHAGOREAN` (A1 B2 C3 D4 E5 F6 G7 H8 I9 J1 K2 L3 M4 N5 O6 P7 Q8 R9 S1 T2 U3 V4 W5 X6 Y7 Z8); assert IVAN = 19.
- `digit_sum(n)`, `reduce_le45(n) -> (vrijednost, koraci)` (ponavljaj zbroj znamenki dok > 45), `reduce_single(n)`
  (do jedne znamenke, bez cuvanja majstorskih brojeva), `date_digit_sum(d: date)` (zbroj svih znamenki dd mm yyyy).

### history.py
- `load_history(path) -> dict(df, rejected: list[(line_no, raw, reason)], n_valid, first_date, last_date,
  expected_missing: int)`; CSV `datum;kolo;n1;...;n6`, datum u dd.mm.yyyy. ili yyyy-mm-dd; provjera 6 razlicitih
  brojeva 1..45, duplikati (isti datum ili isti kolo), odbaceni retci s razlogom; `expected_missing` = broj dana
  kola (config.DRAW_WEEKDAYS) izmedju zadnjeg datuma datoteke (iskljucivo) i danasnjeg datuma (ukljucivo).
- `frequencies(df, last_n=None) -> dict[int,int]`; `pareto(df) -> (top6, rank7_12)` (nerjeseno: manji broj);
  `vruci(df) -> (top6, rank7_12, window_used)` (zadnjih 100 kola; nerjeseno zadnjih 250, pa manji broj).
- Ako datoteka ne postoji: vratiti None i jasno ispisati da sustavi 1-4 nisu dostupni; nista se ne izmislja.
Selftest sa sintetickom datotekom u tmp (ukljucujuci neispravne retke).

### probability.py (egzaktno, enumeracija C(45,6)=8.145.060 u numpyju; bez simulacije)
- `all_draws() -> np.ndarray` uint8 oblika (8145060, 6) (izgradnja vektorizirano ili np.fromiter; kesirati u modulu).
- `hits_matrix(draws, combos) -> np.ndarray` uint8 (n_draws, n_combos).
- `set_probabilities(combos: list[list[int]]) -> dict`: `p_any_ge3, p_any_ge4, p_any_ge5, p_any_6` (udio izvlacenja u
  kojima bar jedna kombinacija ima >=k), `expected_hits_per_combo` (prosjek po kombinaciji = tocno 0.8),
  `per_combo` (lista dict s p3,p4,p5,p6 za svaku), `n_draws`=8145060.
- Selftest: za jednu kombinaciju p6 = 1/8145060, p5 = 234/8145060, p4 = 11115/8145060, p3 = 182780/8145060
  (hipergeometrijski); za 2 disjunktne kombinacije p_any_6 = 2/8145060.

### score.py (CLI: `python score.py rezultati.csv [loto6_godina.csv]`)
- Ulaz rezultati.csv `datum;n1;...;n6`; spoj s loto6_godina.csv po datumu; pogoci po sustavu i kolu (tablica),
  prosjek po sustavu, ocekivanje 6*6/45 = 0.80 po kombinaciji, slucajna referenca: 1000 setova iste strukture
  (isti broj kola i kombinacija po kolu, slucajne kombinacije, rng iz seed.txt) -> raspodjela prosjeka, percentil
  opazenog; chi-kvadrat raspodjele pogodaka (0..6, spojiti >=4) prema hipergeometrijskim ocekivanjima, p-vrijednost
  (scipy.stats). Unaprijed dogovoren prag: samo prosjek >= 1.00 pogodaka po kombinaciji na >= 50 kola smatra se
  odstupanjem; ispis mora to eksplicitno reci i nikad ne tvrditi prediktivnost.

## Format izlaznih CSV-ova (run_year.py; pise ih integrator)
UTF-8 s BOM, separator `;`, datumi dd.mm.yyyy. `loto6_godina.csv` i `loto6_fiksni_set.csv`:
`datum;dan;vrijeme;sustav;n1;n2;n3;n4;n5;n6;joker;izvod;zamjena;seed` (n1..n6 uzlazno sortirani).
`loto6_konvergencija.csv`: `datum;broj;broj_izvora;izvori`.
