# Registar Loto 6/45 - 15 sustava, fiksni set, konvergencija, egzaktne vjerojatnosti

Sve se racuna lokalno skriptama u ovom direktoriju (bez interneta). Seed slucajnog dijela (karte, stabljike):
`secrets.randbits(64)` = **15973270160073881434**, jedan za cijeli posao, zapisan u `seed.txt` i u stupcu `seed`
svakog izlaznog retka. Svaka kombinacija 6/45 ima istu sansu 1 : 8.145.060 za sesticu; sustavi odredjuju KOJE se od
jednako vjerojatnih kombinacija igraju.

## Pokretanje

```
cd /home/user/spss && python3 -m venv .venv && .venv/bin/pip install -r loto6/requirements.txt
cd loto6
../.venv/bin/python selftest_all.py      # selftestovi svih modula (asserti)
../.venv/bin/python run_year.py          # puni izracun -> CSV-ovi, izvod_prvo_kolo.txt, vjerojatnosti.txt
../.venv/bin/python pregled.py           # kombinacije_pregled.md (citljiva tablica)
../.venv/bin/python score.py rezultati.csv   # ocjena nakon izvlacenja (datum;n1;...;n6)
```

## Ulazi i usvojene postavke

| stavka | vrijednost | status |
|---|---|---|
| natal | 31.05.1991. 18:20 CEST (UTC+2), Zadar 44 07 N 15 14 E, ime IVAN | zadano |
| mjesto kola | Zagreb 45 49 N 15 59 E | zadano |
| vrijeme kola | 20:00 lokalno (Europe/Zagreb; u razdoblju CET) | PRETPOSTAVKA (nije upisano) |
| dani kola | srijeda i nedjelja | PRETPOSTAVKA (nije upisano) |
| razdoblje | 01.11.2026. - 31.12.2026. (18 kola) | korisnik: "do kraja godine" |
| cijena kombinacije | 1,00 EUR | PRETPOSTAVKA (nije upisano) |
| dobitne vrste | 6, 5, 4, 3 pogotka | PRETPOSTAVKA (nije upisano) |
| povijest.csv | **ne postoji** | sustavi 1-4 (Pareto 1/2, Vruci 1/2) i izvor Statistika NISU izracunati; nista nije izmisljeno |
| odigrano.csv | ne postoji | nema kombinacija za izbjegavanje |

Prilozena datoteka `Eurojackpot_500_analiza_4.xlsx` je analiza Eurojackpota (5/50 + 2/12) i nije povijest Lota 6/45,
pa nije koristena. Cim se doda `povijest.csv` (datum;kolo;n1;...;n6, dd.mm.yyyy. ili yyyy-mm-dd), `run_year.py`
automatski ukljucuje Pareto 1/2, Vruci 1/2, izvor Statistika u Plecku, tie-break "cesci u povijesti" i puni fiksni set
od 10 kombinacija. Provjera povijesti: struktura, duplikati (datum i broj kola), raspon 1-45, 6 razlicitih brojeva,
ispis odbacenih redaka s razlogom, procjena broja kola koja nedostaju do danas.

Sve postavke su u `config.py` (`DRAW_TIME_LOCAL`, `DRAW_WEEKDAYS`, `PERIOD_START/END`, `PRICE_EUR`, `WIN_CLASSES`).

## Kontrola efemerida (prolazi)

Swiss Ephemeris (pyswisseph 2.10.3.2, Moshierova efemerida bez datoteka), Placidus, sideralno Lahiri.
Natalno Sunce 9,72 Blizanaca (trazeno 9-10); natalni Mjesec sideralno 20,84 Strijelca = nakshatra 20 Purva Asadha, pada 3
(trazeno). Ostalo: natalni ASC 14,89 Skorpiona, MC 26,67 Lava, ayanamsa 23,737, rodjenje danju (zalazak u Zadru 20:34 CEST),
natalni Pars Fortunae 19,75 Blizanaca. Vimsottari: ostatak Venere 8,744 g; na 01.11.2026. MD Rahu (27.02.2023.-26.02.2041.),
AD Jupiter (09.11.2025.-03.04.2028.). Kompletan ispis: `izvod_prvo_kolo.txt`.

## Pravila pretvorbe

* `m45(n) = ((n-1) mod 45) + 1` (i za n <= 0, Python modulo).
* Kaskada = uredjen popis kandidata (oznaka, sirova vrijednost). Kombinacija = prvih 6 razlicitih m45-vrijednosti redom.
  Ako ih je manje od 6: dopuna iz rezervne kaskade istog sustava (oznaka `dopuna iz rezervne kaskade` u stupcu `zamjena`);
  ako ni to ne dostaje: najmanji slobodan broj (oznaka `krajnja rezerva`; u ovom razdoblju se nije dogodilo).
* Joker = zadnje znamenke prvih 6 sirovih vrijednosti kaskade (prije m45, ukljucujuci duplikate), tekst s vodecim nulama.
* Brojevi planeta: Sunce 1, Mjesec 2, Jupiter 3, Rahu 4, Merkur 5, Venera 6, Ketu 7, Saturn 8, Mars 9.
* Slucajnost: `numpy.random.default_rng(SeedSequence(SEED, spawn_key=(redni_broj_kola, broj_sustava)))` - svako kolo i
  sustav ima vlastiti podniz, pa rezultat jednog kola ne ovisi o drugima.
* n1..n6 u CSV-u su sortirani uzlazno; redoslijed kaskade je u stupcu `izvod` i u izvodu prvog kola.

## Sustavi (kaskada -> rezerva)

1-4. **Pareto 1/2, Vruci 1/2** (`history.py`): 6 najcescih u cijeloj povijesti / rang 7-12; 6 najcescih u zadnjih 100 kola
   (nerjeseno: zadnjih 250, pa manji broj) / rang 7-12. Racunaju se jednom na stanju datoteke. *Nedostupno bez povijest.csv.*
5. **Kaldejska**: LOTO (21), HRVATSKA LUTRIJA (42), ime dana (NEDJELJA 25, SRIJEDA 17), ZAGREB (20), IVAN (13), dan u mjesecu,
   zbroj znamenki datuma (ddmmyyyy), broj planeta dana. Rezerva: vrijednosti pojedinih slova rijeci redom, pa kumulativni zbrojevi.
6. **Gematrija**: hebrejski dan u mjesecu u trenutku kola (pyluach, unakrsno convertdate; poslije zalaska u Zagrebu = sljedeci dan),
   zatim מזל 77, ברכה 227, שפע 450, זהב 14, חי 18, הצלחה 138, כסף 160, עושר 576 (provjereno zbrajanjem slova, bez sofita).
   Rezerva: vrijednosti pojedinih slova.
7. **Hermetika**: planetarni sat u trenutku kola (izlazak/zalazak za Zagreb, swe.rise_trans; 12 dnevnih i 12 nocnih sati
   nejednake duljine; planetarni dan od izlaska do izlaska; prvi sat = vladar dana, kaldejski red Saturn, Jupiter, Mars, Sunce,
   Venera, Merkur, Mjesec) -> Agrippina kamea tog planeta (3x3 ... 9x9, asertima provjereni redovi, stupci, dijagonale i skup 1..n^2).
   Kaskada: sredisnji red (dva sredisnja za parni red), konstanta, ukupni zbroj. Rezerva: ostali redovi, pa stupci.
8. **I Ching**: 6 linija stabljikama (6:7:8:9 = 1:5:7:3 od 16). Kaskada: King Wen primarnog, nuklearnog (linije 2-4 dolje,
   3-5 gore), promijenjenog (bez pomicnih = primarni), zbroj vrijednosti linija, polozaji pomicnih linija, Fu Xi gornjeg x8 + donjeg.
   Cijela King Wen tablica je u `iching.py` s asertima (Kun/Kun 2, Xun nad Kun 20, Gen nad Kun 23, Dui nad Xun 28, ...).
   Rezerva: KW suprotnog (sve linije obrnute), KW obrnutog (naopako), KW nuklearnog promijenjenog, Fu Xi kod nuklearnog i
   promijenjenog, Fu Xi gornjeg i donjeg trigrama, vrijednosti linija, polozaji nepomicnih linija.
9. **Lenormand**: 36 karata promijesano seedom, Grand Tableau 4 x 9; kaskada = 8 zbrojeva susjednih karata 1. reda, zatim
   preostalih 27 karata redom. Rezerva: karte 1. reda.
10. **Meihua Yishu**: kineski lunarni datum (lunardate), godina = redni broj zemaljskog ogranka (丙午 2026 = 7), dvosatni ogranak po
    pravom suncevom vremenu Zagreba (UT + 15 59 E / 15 + jednadzba vremena; Zi = 1 od 23:00, TST >= 23:00 racuna se kao sljedeci dan);
    gornji = (g+m+d) mod 8, donji = (+sat) mod 8, pomicna = (+sat) mod 6 (0 -> 8 odn. 6); staticne linije 7/8, pomicna 9/6.
    Kaskada kao I Ching + oba zbroja. Rezerva kao I Ching.
11. **Tarot**: 78 karata (velike arkane 1-21, Luda 22, stapovi 23-36, pehari 37-50, macevi 51-64, diskovi 65-78), promijesano seedom;
    kaskada = karte u redoslijedu mijesanja s indeksom <= 45 (uvijek 45 kandidata).
12. **Pitagorejska**: IVAN 19, zivotni put 29 pa 11, rodjendanski 31, mjesec kola, zbroj znamenki datuma kola, osobna godina
    (mjesec rodjenja 5 + dan rodjenja 31 + zbroj znamenki godine kola 10 = 46 -> 10), osobni mjesec (osobna godina + mjesec),
    osobni dan (osobni mjesec + dan), univerzalni dan (univerzalna godina + mjesec + dan). Vrijednost > 45 reducira se zbrojem
    znamenki dok nije <= 45; <= 45 ostaje nereducirana. Rezerva: potpuno reducirane znamenke, pa +9, +18, +27, +36.
13. **Jyotis** (sideralno Lahiri): redni broj nakshatre Mjeseca, broj njezina gospodara i obitelj (+9, +18, +27, +36), rasi Mjeseca,
    rasi lagne, gospodar tekuce Vimsottari mahadase i antardase iz natalnog Mjeseca (godina 365,25 d), vladar dana (vara po civilnom danu;
    kolo u 20:00 je poslije izlaska Sunca). Rezerva: nakshatra Sunca, rasi Sunca, rasi Marsa, Merkura, Jupitera, Venere, Saturna, pada.
14. **Zapadna** (tropski): stupanj u znaku (floor(lon mod 30)+1) Sunca, Mjeseca, Ascendenta, vladara dana, Pars Fortunae
    (danju ASC+Mjesec-Sunce, nocu ASC+Sunce-Mjesec; dan/noc = Sunce iznad/ispod horizonta u trenutku kola) i natalnog PoF; zatim
    brojevi znakova tih tocaka; zatim stupanj+30 za tocke s duljinom >= 180. Rezerva: stupnjevi Merkura, Venere, Marsa, Jupitera, Saturna, MC.
15. **Plecko** (filtar konvergencije): izvori = Statistika (sustavi 1-4 kao jedan izvor; ovdje nedostupan) + 10 ezoterijskih
    (kombinacije sustava 5-14 prije pravila seta). Za svaki broj broj razlicitih izvora; signal >= 2, sum = 1. Kaskada = signal
    brojevi po (vise izvora, cesci u povijesti ako postoji, manji broj); rezerva = sum istim redom (oznaka dopune). Svi brojevi
    sa svojim izvorima su u `loto6_konvergencija.csv`.

**Pravilo seta** (po kolu, kombinacije po rangu 1-15): nijedan par ne dijeli vise od 2 broja. Prekrsaj: u kombinaciji nizeg
ranga zamjenjuje se zadnji (po redoslijedu kaskade) od zajednickih brojeva sljedecim kandidatom iz njezine kaskade koji ne stvara
novi prekrsaj; ako par dijeli 4 ili vise brojeva, zamjene se ponavljaju jedan broj po koraku. Svaka zamjena je zapisana u stupcu
`zamjena` (stari->novi, s kojim sustavom, koji kandidat). Kombinacije iz `odigrano.csv` tretiraju se kao prekrsaj (ne ponavljaju se).
Plecko je po konstrukciji najcesce zamjenjivan (nominira upravo brojeve drugih sustava).

**Fiksni set** (nepromijenjen cijelo razdoblje): Pareto 1, Pareto 2, Vruci 1, Vruci 2 (nedostupni bez povijesti), Pitagorejska
godisnja (osobna i univerzalna godina umjesto dnevnih brojeva), Jyotis natalna i Zapadna natalna (samo natalne tocke), Kaldejska
godisnja (bez imena dana i datuma, s godinom 2026 i zbrojem njezinih znamenki), Plecko godisnja (6 brojeva koji su u najvise kola
razdoblja bili signal) i Plecko godisnja 2 (sljedecih 6). Isto pravilo preklapanja <= 2. Rezultat: 6 od 10 kombinacija.

## Rezultat (01.11.-31.12.2026., 18 kola)

Sve kombinacije: `loto6_godina.csv` (198 redaka = 18 kola x 11 sustava) i citljivo `kombinacije_pregled.md`.
Konvergencija po kolu: `loto6_konvergencija.csv` (622 redaka). Kompletan izvod prvog kola: `izvod_prvo_kolo.txt`.

Fiksni set:

| sustav | kombinacija | joker |
|---|---|---|
| Pitagorejska godisnja | 1 10 11 19 29 31 | 991100 |
| Jyotis natalna | 6 15 20 24 33 42 | 065432 |
| Zapadna natalna | 3 8 10 15 20 25 | 055503 |
| Kaldejska godisnja | 1 10 13 20 21 42 | 120360 |
| Plecko godisnja | 2 13 14 19 20 29 | 092491 |
| Plecko godisnja 2 | 1 4 13 16 18 32 | 321481 |

Brojevi koji su najcesce bili signal u razdoblju: 20 (18 kola), 29 (16), 2, 14, 19 (15), 11, 13, 32 (11). Napomena: 20 (ZAGREB i
stupanj natalnog PoF), 13 (IVAN), 21 (LOTO) i 42 (HRVATSKA LUTRIJA) su stalni nominatori Kaldejske/Zapadne, pa su po konstrukciji
cesto signal.

Statistika izracuna: 56 zamjena po pravilu seta (sve zapisane), 10 dopuna iz rezervne kaskade (I Ching 9, Meihua 1; heksagrami bez
pomicnih linija daju manje od 6 razlicitih vrijednosti), 0 krajnjih rezervi.

## Vjerojatnosti (egzaktna enumeracija svih 8.145.060 izvlacenja, `probability.py`)

| set | P(bar jedna >= 3) | P(bar jedna >= 4) | P(bar jedna >= 5) | P(bar jedna 6) | ulog po kolu | ulog za 18 kola |
|---|---|---|---|---|---|---|
| set jednog kola (11 kombinacija, 01.11.2026.) | 0,235560 (1 : 4,2) | 0,01528423 (1 : 65,4) | 0,000317370 (1 : 3151) | 0,0000013505 (1 : 740.460) | 11,00 EUR | 198,00 EUR |
| fiksni set (6 kombinacija) | 0,129593 (1 : 7,7) | 0,00833008 (1 : 120,0) | 0,000173111 (1 : 5777) | 0,0000007366 (1 : 1.357.510) | 6,00 EUR | 108,00 EUR |

Ocekivani pogoci po kombinaciji = 36/45 = 0,80 tocno (provjereno enumeracijom). Vrijednosti za ostala kola su istog reda velicine
(ovise samo o preklapanjima unutar seta, koja su ogranicena na <= 2).

## score.py

`python score.py rezultati.csv [loto6_godina.csv]` -> pogoci po sustavu i kolu, prosjek po sustavu, ocekivanje 0,80, slucajna
referenca (1.000 setova iste strukture, rng iz seeda), chi-kvadrat raspodjele pogodaka prema hipergeometrijskoj (kategorije 0, 1, 2, >= 3)
i p-vrijednost; zapisuje `score_izvjestaj.csv`. Unaprijed dogovoren prag: samo prosjek >= 1,00 pogodaka po kombinaciji na >= 50 kola
smatra se odstupanjem.

## Datoteke

| datoteka | sadrzaj |
|---|---|
| `config.py` | postavke (natal, Zagreb, vrijeme/dani kola, razdoblje, cijena) |
| `core.py` | m45, kaskada -> kombinacija, Joker, rng, pravilo seta |
| `astro.py` | swisseph: polozaji, ASC/MC, izlazak/zalazak, planetarni sati, nakshatra, Vimsottari, PoF, jednadzba vremena |
| `calendars.py` | hebrejski datum (pyluach + convertdate), kineski lunarni datum i dvosatni ogranak po pravom suncevom vremenu |
| `iching.py`, `kamea.py`, `cards.py`, `numerology.py` | I Ching/Meihua, Agrippine kameje, Lenormand/Tarot, kaldejska/gematrija/pitagorejska |
| `history.py` | povijest.csv: provjera, Pareto, Vruci |
| `systems.py` | 15 sustava, konteksti kola, Plecko, godisnje varijante |
| `run_year.py`, `izvod.py`, `pregled.py` | glavni izracun, izvod kola, pregled |
| `probability.py`, `score.py` | egzaktne vjerojatnosti, ocjenjivanje |
| `selftest_all.py` | pokrece sve selftestove |
| izlazi | `loto6_godina.csv`, `loto6_fiksni_set.csv`, `loto6_konvergencija.csv`, `izvod_prvo_kolo.txt`, `vjerojatnosti.txt`, `izracun_log.txt`, `kombinacije_pregled.md` |

CSV: UTF-8 s BOM, separator `;`, stupci `datum;dan;vrijeme;sustav;n1;n2;n3;n4;n5;n6;joker;izvod;zamjena;seed`
(fiksni set: `datum` = razdoblje, `dan` = "fiksni set"). Unutar polja `izvod`/`zamjena` nema znaka `;`.

## Paketi i verzije

Python 3.13.17; pyswisseph 2.10.3.2 (Swiss Ephemeris 2.10.03, Moshier), lunardate 0.3.0, pyluach 2.3.0, convertdate 2.5.1,
numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, openpyxl 3.1.5 (samo za pregled prilozene xlsx datoteke). Popis: `requirements.txt`.

## Ogranicenja i sto nije provjereno

* Vrijeme i dani kola, cijena i dobitne vrste su pretpostavke (korisnik ih nije upisao); promjena u `config.py` i ponovno pokretanje
  mijenja sve izvedene vrijednosti (planetarni sat, hebrejski dan, ogranak, osobni dan...).
* Bez `povijest.csv` nema sustava 1-4, izvora Statistika ni tie-breaka po ucestalosti; fiksni set ima 6 umjesto 10 kombinacija.
* Efemeride su Moshierove (tocnost reda 0,1" za planete, do ~1" za Mjesec), bez datoteka Swiss Ephemerisa; izlazak/zalazak su za
  razinu mora, standardnu refrakciju i gornji rub Sunca; nije uracunata nadmorska visina Zagreba.
* Interpretacijske odluke (navedene gore): univerzalni dan kao univerzalna godina + mjesec + dan; osobna godina iz nereduciranog dana 31;
  Meihua statične linije 7/8 i pomicna 9/6 za zbroj linija; rani Zi (TST >= 23:00) kao sljedeci dan; rezervne kaskade su definirane
  ovdje jer ih zadatak ne propisuje; pravilo seta prosireno na visestruke zamjene kad par dijeli >= 4 broja.
* Vjerojatnosti su egzaktne za navedene setove; `score.py` je testiran samo na sintetickim rezultatima (nema stvarnih izvlacenja).
* Nije provjereno prema sluzbenim podacima HL-a (bez interneta): raspored izvlacenja, cijena, dobitne vrste, format Jokera.
