# Ezoterično generiranje lutrijskih brojeva – reproducibilni istraživački pipeline

Devet ezoteričnih metoda (Meihua Yishu, I Ching, zapadna i vedska astrologija, pitagorejska i
kaldejska numerologija, gematrija, tarot, Lenormand) i njihov ensemble generiraju kombinacije za
**Sve ili Ništa 11/22**, **Eurojackpot 5/50 + 2/12** i **Joker (6 znamenki)**. Svaka metoda je zatim
podvrgnuta kronološkom backtestu na priloženim povijesnim izvlačenjima i uspoređena s uniformnom
slučajnom osnovicom, uz korekciju za višestruko testiranje.

> **Važno.** Izvlačenja su slučajna i neovisna. Nijedna metoda ovdje ne može promijeniti vjerojatnost
> dobitka, i backtest to potvrđuje (vidi [Rezultati](#rezultati)). Pipeline postoji da bi se ta tvrdnja
> *mjerila* poštenim protokolom, a ne da bi se nešto obećavalo. Nijedan broj u ovom repozitoriju nije
> izmišljen ni procijenjen ručno: sve je izračunato kodom niže i može se ponovno izračunati.

## Brzi početak

```bash
cd esoteric_lottery
python3 -m venv .venv
.venv/bin/pip install -U pip setuptools wheel   # pyswisseph se kompajlira; stari Debianov setuptools pada ("install_layout")
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                      # 107 unit testova
.venv/bin/python main.py                        # cijeli pipeline, ~40 s
```

Rezultati idu u `results/`:

| Datoteka | Sadržaj |
|---|---|
| `results/REPORT.md` | čitljiv izvještaj (hrvatski): što je izvršeno, kombinacije, sve tablice backtesta |
| `results/final/combinations.json` | konačne kombinacije svake metode i ensemblea za sve tri igre |
| `results/final/intermediate/<metoda>.json` | svi međurezultati za ciljano izvlačenje (karte, heksagrami, lunarni datum, panchanga, …) |
| `results/final/intermediate/ensemble.json` | Borda bodovi, težine, glasovi po poziciji (Joker) |
| `results/backtest/sve_ili_nista/summary.json` | validacija podataka, split, statistike, testovi, korekcije, kvaliteta izvlačenja |
| `results/backtest/sve_ili_nista/predictions/<strategija>.json` | za svako kolo: listić, izvučeni brojevi, pogoci, sirovi brojevi metode |
| `results/theoretical_probabilities.json` | egzaktne distribucije pogodaka i vjerojatnosti glavnog dobitka |
| `results/manifest.json` | status svake metode i komponente, verzije paketa, SHA-256 konfiguracije, koda i podataka, argumenti naredbenog retka, vrijeme izvođenja |
| `results/draw_<datum>/` | ista struktura (bez `backtest/`) za kasnija izvlačenja pokrenuta s `--draw-date`; očitanja ≥ 2 u `final/intermediate/readings/<k>/` |

`python main.py --skip-backtest` računa samo kombinacije; `--config` i `--results` mijenjaju putanje.

### Drugo izvlačenje i više listića

```bash
.venv/bin/python main.py --draw-date 2026-10-24 --readings 4 --backtest-from results \
                         --results results/draw_2026-10-24
```

* `--draw-date` / `--draw-time` mijenjaju samo trenutak izvlačenja (ostalo ostaje iz `config.yaml`).
* `--readings N` daje N **očitanja**: očitanje *k* ponovno baca simulirane orakule (I Ching, tarot,
  Lenormand) i SHA-256 tie-break sa seedom `seed + (k − 1)`. Očitanje 1 je identično običnom
  izvođenju. Sirovi brojevi astronomskih i numeroloških metoda ne ovise o seedu; kod njih se može
  promijeniti samo popuna brojeva koje metoda nije sama odredila. Primarni listić svakog očitanja je
  ensemble s jednakim težinama; `combinations.json` dobiva `readings`, `tickets` i `coverage`
  (različiti listići, pokriveni brojevi, zajednički brojevi po parovima), a izvještaj odjeljak
  „Listići po očitanjima”.
* `--backtest-from DIR` ponovno koristi backtest ranijeg izvođenja umjesto da ga računa: backtest ovisi
  samo o konfiguraciji i povijesnim podacima, ne o datumu izvlačenja. Prije ponovne upotrebe
  uspoređuju se SHA-256 `config.yaml` i datoteka s povijesti iz `DIR/manifest.json`; ako se razlikuju,
  pipeline staje. Status u izvještaju i manifestu je „ponovno korišten” s putanjom i vremenom izvora.

Očitanja ne mijenjaju vjerojatnost dobitka; četiri listića su četiri jednako (ne)vjerojatne
kombinacije, i to međusobno korelirane (ensemble ih gradi iz istih determinističkih metoda, pa
dijele 8–9 od 11 brojeva).

## Struktura

```
esoteric_lottery/
├── main.py                  orkestracija (kombinacije → backtest → ensemble → manifest → izvještaj)
├── config.yaml              svi ulazi i parametri, fiksirani prije prvog backtesta
├── requirements.txt
├── data/                    priložena izvlačenja Sve ili Ništa (CSV + XLSX, nepromijenjeni)
├── methods/
│   ├── common.py            tipovi, mapiranje u raspon, SHA-256 tie-break, determinističan JSON
│   ├── ephemeris.py         pyswisseph omotač (Moshier efemeride, kuće, sunčev izlazak, mladi mjeseci)
│   ├── chinese_calendar.py  kineski lunisolarni kalendar izračunat iz efemerida
│   ├── meihua.py            1. Meihua Yishu
│   ├── iching.py            2. I Ching (stabljike stolisnika)
│   ├── western_astrology.py 3. zapadna astrologija
│   ├── vedic_astrology.py   4. vedska astrologija
│   ├── pythagorean.py       5. pitagorejska numerologija
│   ├── chaldean.py          6. kaldejska numerologija
│   ├── gematria.py          7. gematrija
│   ├── tarot.py             8. tarot
│   ├── lenormand.py         9. Lenormand
│   └── ensemble.py          10. agregacija (Borda / glasanje po poziciji)
├── evaluation/
│   ├── data.py              učitavanje i validacija izvlačenja, usporedba CSV ↔ XLSX
│   ├── backtest.py          kronološki backtest bez curenja podataka
│   ├── stats.py             egzaktni testovi, CI, χ², MC osnovica, AUC, Holm/BH, snaga
│   ├── controls.py          statističke kontrole (sklearn)
│   ├── quality.py           slučajnost samih izvlačenja
│   └── report.py            REPORT.md
├── tests/                   unit testovi (pytest)
└── results/                 generirani izlazi
```

Pipeline je u vlastitom direktoriju jer korijen repozitorija već ima `README.md` i `requirements.txt`
drugog projekta (CryptoPulse); ništa izvan `esoteric_lottery/` nije mijenjano.

## Ulazi i pretpostavke

| Ulaz | Vrijednost | Napomena |
|---|---|---|
| Rođenje | 1991-05-31 18:20, Zadar (44.1194° N, 15.2314° E) | `zoneinfo` Europe/Zagreb → CEST (UTC+2) → 16:20 UTC |
| Izvlačenje | 2026-10-23 19:00, Zagreb (45.8150° N, 15.9819° E) | CEST (UTC+2, ljetno vrijeme završava 25. 10.) → 17:00 UTC |
| Ime | nije zadano | sve što traži ime je **preskočeno** i tako označeno (`subject.name` u config.yaml) |
| Vrijeme povijesnih izvlačenja | 19:00 lokalno | **pretpostavka**: datoteke nemaju vrijeme izvlačenja |
| Koordinate | središta gradova (GeoNames) | utječu samo na kuće/ascendent i sunčevo vrijeme |

Efemeride: Swiss Ephemeris (pyswisseph 2.10) u **Moshier** načinu (`FLG_MOSEPH`), koji ne treba vanjske
datoteke pa daje identične brojeve na svakom računalu; njegova je pogreška daleko ispod najfinije
podjele korištene ovdje (7.2° luka).

## Protokol: sprječavanje curenja podataka i prenaučenosti

Sve ispod je zapisano u `config.yaml` (blok `evaluation`) **prije prvog pokretanja backtesta** i nije
mijenjano nakon što su rezultati bili poznati.

* **Kronološki split.** Treniranje: sva kola prije 2026-01-01 (365 kola, 2025.). Evaluacija:
  2026-01-01 – 2026-09-28 (270 kola). Sve inferencijske tvrdnje odnose se samo na evaluaciju;
  brojke iz treniranja su opisne.
* **Nema pogleda u budućnost.** Predviđanje za kolo *t* računa se samo iz datuma/vremena tog kola,
  podataka o rođenju i (samo za statističke kontrole) redaka `M[:t]` ranijih kola. Ezoterične metode
  uopće ne vide rezultate izvlačenja. Testovi `tests/test_data_and_backtest.py` to provjeravaju:
  prepisivanje budućih kola ne mijenja nijedno ranije predviđanje, a zamjena svih brojeva drugim
  brojevima ne mijenja nijedan ezoterični listić.
* **Nema optimizacije na evaluaciji.** Jedine naučene veličine (težine *train-weighted* ensemblea,
  koeficijenti logističke regresije) uče se samo na treniranju i zamrzavaju se prije prvog
  evaluacijskog kola. Hiperparametri (orbovi, broj bacanja, β = 0.5, C = 1.0, prozor 30) su fiksirani
  unaprijed i nisu ugađani. Algoritmi su napisani prije prvog backtesta; nakon njega mijenjani su
  samo formatiranje izvještaja i oblik JSON-a, a ponovljeno izvođenje dalo je bajt-identičan
  `summary.json`.
* **Primarna mjera:** prosječan broj pogodaka po kolu (glavni bazen). **Primarni test:** egzaktni
  dvostrani test – pod H0 (listić fiksiran prije izvlačenja, izvlačenje uniformno) pogoci svakog kola
  su hipergeometrijski Hyp(22, 11, 11) *bez obzira na listić*, pa je zbroj preko *n* kola egzaktna
  *n*-struka konvolucija. Dvostrano zato što je u Sve ili Ništa i sustavno *manje* pogodaka
  informacija (komplement listića bi tada imao više).
* **Obitelj testova:** 13 strategija (9 metoda, 2 ensemblea, 2 kontrole), α = 0.05, **Holm**
  (primarno, kontrola FWER) i Benjamini-Hochberg (FDR).
* **Uniformna slučajna osnovica:** 10 000 strategija koje za svako evaluacijsko kolo uzmu svježi
  uniformno slučajan listić, bodovanih na *istim* stvarnim kolima → empirijska distribucija prosjeka
  i Monte Carlo p-vrijednost (neovisna provjera egzaktnog testa).
* **Intervali pouzdanosti:** t-interval i bootstrap (10 000 ponavljanja) za prosjek; Wilson za udjele;
  bootstrap po kolima za AUC.
* **Sekundarno (eksploratorno, nekorigirano):** χ² usporedba cijele distribucije pogodaka s H0
  (spajanje razreda dok očekivano ≥ 5), udio ekstremnih ishoda (≤ 3 ili ≥ 8 pogodaka, binomni test),
  AUC rangova metode za pojavljivanje broja.
* **Snaga:** s 270 kola i 13 testova najmanji pomak koji se otkriva s 80 % snage je ±0.27 pogodaka
  po kolu (≈ 5 % od 5.5). Manji stvarni učinci ne bi bili vidljivi – to je granica ovih podataka, ne
  dokaz da učinka „ima malo”.
* **Kvaliteta podataka:** provjera broja/raspona/duplikata, stupaca `Zbroj` i `Parni`, kronologije,
  usporedba CSV ↔ XLSX (identično), kolo 178/2026 bez brojeva je izostavljeno; opisni testovi
  slučajnosti samih izvlačenja (učestalost, preklapanje uzastopnih kola, najdulji niz).

## Metode

Zajedničko za sve: **mapiranje u lutrijski raspon je eksperimentalno** – nijedna tradicija ne
definira kako se njeni brojevi pretvaraju u lutrijski listić. Svaka metoda proizvodi uređeni niz
„sirovih” brojeva; broj *n* dobiva bod `1/(i+1)` za svaku poziciju *i* čija vrijednost modularno
pada na *n* (`((v − 1) mod N) + 1`, dakle 0 → N), a uzima se *k* brojeva s najviše bodova.
Neriješeno se razrješava SHA-256 hashom (metoda, igra, bazen, trenutak izvlačenja), što je
reproducibilno i bez sklonosti malim brojevima. Ako metoda ne proizvede dovoljno različitih brojeva,
ostatak popunjava hash, a stupac **„iz metode”** u izvještaju kaže koliko je brojeva metoda stvarno
odredila. Joker: posljednja znamenka prvih šest sirovih vrijednosti metode.

| # | Metoda | Tradicionalni dio | Eksperimentalni dio | Glavni izvori |
|---|---|---|---|---|
| 1 | **Meihua Yishu** | 年月日時起卦: gornji trigram = (godina + mjesec + dan) mod 8, donji = (… + sat) mod 8, pomični crt = (… + sat) mod 6; brojevi iz kineskog lunarnog kalendara (zemaljske grane godine i sata), trigrami po redu ranijeg neba; izvorni, međusobni (互) i promijenjeni (變) heksagram, tijelo/upotreba (體/用) i pet faza | sat po pravom sunčevom vremenu (zadano u configu), bacanje i za izvlačenje i za rođenje, mapiranje | 《梅花易數》 (pripisano Shao Yongu), primjer 觀梅占 je unit test; GB/T 33661-2017 za pravila kalendara |
| 2 | **I Ching** | postupak sa 49 stabljika stolisnika (tri promjene po crti → 6/7/8/9), heksagram odozdo prema gore, promjenjive crte, King Wen redoslijed | podjela hrpa iz PCG64 generatora sjemenog SHA-256-om (rođenje, izvlačenje, seed) – reproducibilna zamjena za ruke; 12 bacanja; Joker = vrijednosti crta prvog bacanja | Xici zhuan I.9; Wilhelm/Baynes 1950 |
| 3 | **Zapadna astrologija** | tropski geocentrični položaji Sunca … Plutona, Placidus kuće, ASC/MC (natal Zadar, tranzit Zagreb); Ptolemejevi aspekti tranzit→natal | orbovi (8/8/6/6/4°) i „zodijačka gustoća”: zodijak podijeljen na N lukova (12 lukova = znakovi za Euro brojeve), von Mises jezgra (scipy) oko svake točke težine 1 + jakost aspekata | Swiss Ephemeris; Ptolemej, *Tetrabiblos* I.13 |
| 4 | **Vedska astrologija** | siderički zodijak (Lahiri), panchanga (tithi, nakshatra i pada, yoga, karana, vara od izlaska Sunca), lagna po cijelim znakovima, Vimshottari maha/antar/pratyantar daša iz natalnog Mjeseca, brojevi planeta (Sunce 1 … Mars 9) | redoslijed i mapiranje sirovog niza; godina daše = 365.25 dana | *Brihat Parashara Hora Shastra* pogl. 46; standardne definicije panchange; SIDM_LAHIRI |
| 5 | **Pitagorejska numerologija** | slova 1–9, životni put (Goodwin: zbroj reduciranih mjeseca, dana, godine; master brojevi), rođendan, stav, vrhunci, izazovi, osobna godina/mjesec/dan, univerzalni dan | brojevi imena mjesta (Zadar, Zagreb), mapiranje; **brojevi imena preskočeni** (ime nije zadano) | J. Jordan 1965; M. O. Goodwin 1981 |
| 6 | **Kaldejska numerologija** | Cheirova tablica 1–8 (bez 9), složeni brojevi, broj rođenja, broj sudbine, planet dana u tjednu | vrijednosti imena mjesta, mapiranje; **broj imena preskočen** | Cheiro, *Book of Numbers*, 1926 |
| 7 | **Gematrija** | hebrejska mispar hechrachi (שלום = 376), grčka izopsefija (Ἰησοῦς = 888), latinska tablica po Agrippi (A1 … Z500), engleska ordinalna i obrnuta | izbor tekstova (hrvatski dan i mjesec izvlačenja i rođenja, mjesta), mapiranje; **hebrejski/grčki i ime preskočeni** (nije zadan takav tekst) | Agrippa, *De occulta philosophia* II.20 (1533) – tablicu valja provjeriti u primarnom izdanju; Scholem, *Kabbalah* (1974) |
| 8 | **Tarot** | Rider-Waite-Smith numeracija (Snaga 8, Pravda 11; Luda = 22 na kraju niza), Greerine rođendanske karte (5 + 31 + 1991 = 2027 → 11 Pravda → 2 Velika svećenica) i karta godine (2026 → 10 Kolo sreće → 1 Mag) | indeks 78 karata (Velika arkana 1–22 izravno na brojeve 1–22), sjemeno „miješanje” i niz od 22 karte | Waite 1910; M. K. Greer 1984/2002 |
| 9 | **Lenormand** | numeracija 36 karata (1 Jahač … 36 Križ) | karta dana rođenja (31 = Sunce) i dana izvlačenja, sjemeni Grand Tableau 4 × 9 čitan po redovima | Hechtel, *Das Spiel der Hoffnung*, oko 1799.; Katz & Goodwin 2013 |
| 10 | **Ensemble** | Borda brojanje (1781): u rangu svake metode prvi broj dobiva N bodova, zadnji 1; Joker: glasanje po poziciji | *equal* (primarni, prijavljen unaprijed) i *train-weighted* (softmax z-vrijednosti iz treniranja, β = 0.5, zamrznuto) | J.-C. de Borda, 1781 |

Statističke kontrole (nisu ezoterične; isti protokol): „vrući brojevi” (najčešći u zadnjih 30 kola) i
logistička regresija (scikit-learn) na značajkama povijesti (učestalost u 10/30/100 kola, razmak od
zadnjeg pojavljivanja, prošlo kolo), naučena samo na treniranju; `TimeSeriesSplit` unutar treniranja
prikazuje se samo informativno.

## Što se izvršilo, a što nije

**Izvršeno:** svih 9 metoda i oba ensemblea za sve tri igre; kineski kalendar, natalna i tranzitna
karta, aspekti, siderički položaji, panchanga, Vimshottari, numerologija datuma, latinske šifre
gematrije, tarot i Lenormand; backtest Sve ili Ništa (635 kola) s 13 strategija; 107 unit testova.

**Djelomično / preskočeno (označeno u `manifest.json` i izvještaju):**

* pitagorejski brojevi imena (izraz, duša, osobnost), kaldejski broj imena, gematrija imena – **ime
  nije zadano**; upišite `subject.name` u `config.yaml`;
* hebrejska i grčka gematrija – nema hebrejskog/grčkog teksta (`gematria.extra_texts`); funkcije su
  implementirane i testirane;
* Lenormandova karta upitatelja (Gospodin/Gospođa) – spol nije ulaz, ne koristi se.

**Nije moguće s priloženim podacima:**

* **backtest Eurojackpota i Jokera** – nije priložena nijedna povijesna datoteka. Kombinacije za te
  igre zato *nisu testirane ni na čemu*. Kod je generički: dodajte CSV i blok u `history:` u
  `config.yaml` (stupci po bazenu) i backtest će se pokrenuti istim protokolom;
* stvarna vremena povijesnih izvlačenja (pretpostavljeno 19:00);
* nagradni razredi Sve ili Ništa nisu modelirani (izvještaj daje pogotke, ekstreme i 0/11).

## Rezultati

Izvođenje na priloženim podacima (detalji i sve tablice: [`results/REPORT.md`](results/REPORT.md)).
Evaluacija: 270 kola, 2026-01-01 – 2026-09-28; očekivanje pod H0 = 5.500 pogodaka, SD 1.200.

| Strategija | Prosjek pogodaka | 95 % CI | p (egzaktni) | p (Holm) |
|---|---|---|---|---|
| Meihua Yishu | 5.493 | [5.346, 5.639] | 0.939 | 1.000 |
| I Ching | 5.489 | [5.348, 5.630] | 0.899 | 1.000 |
| Zapadna astrologija | 5.593 | [5.444, 5.741] | 0.214 | 1.000 |
| Vedska astrologija | 5.541 | [5.403, 5.678] | 0.594 | 1.000 |
| Pitagorejska numerologija | 5.570 | [5.417, 5.724] | 0.348 | 1.000 |
| Kaldejska numerologija | 5.593 | [5.447, 5.738] | 0.214 | 1.000 |
| Gematrija | 5.415 | [5.269, 5.561] | 0.254 | 1.000 |
| Tarot | 5.296 | [5.158, 5.435] | 0.0057 | 0.074 |
| Lenormand | 5.526 | [5.379, 5.673] | 0.742 | 1.000 |
| Ensemble (jednake težine) | 5.559 | [5.404, 5.715] | 0.432 | 1.000 |
| Ensemble (težine iz treniranja) | 5.400 | [5.248, 5.552] | 0.179 | 1.000 |
| Kontrola: vrući brojevi | 5.356 | [5.214, 5.497] | 0.051 | 0.611 |
| Kontrola: logistička regresija | 5.541 | [5.402, 5.679] | 0.594 | 1.000 |

* **Nijedna strategija ne preživljava Holmovu (ni BH) korekciju.** Jedini nekorigirani p < 0.05 je
  tarot, i to u smjeru *manje* pogodaka (5.296), dok je u treniranju imao više (5.611); tarot je uz to
  sjemeno pseudo-slučajno miješanje. To je upravo vrsta lažno pozitivnog nalaza koju korekcija hvata.
* 95 % raspon prosjeka 10 000 nasumičnih strategija je [5.355, 5.648]; 12 od 13 strategija je unutar
  njega.
* 95 % CI AUC-a (sekundarna mjera) sadrži 0.5 za sve strategije osim tarota (0.481 [0.467, 0.496],
  opet u smjeru manje pogodaka). Logistička kontrola u vremenskoj unakrsnoj validaciji ima log-loss
  lošiji od konstantnog p = 0.5 u 4 od 5 foldova, a u jednom bolji za 0.0001.
* Sama izvlačenja izgledaju uniformno i neovisno: učestalost brojeva (simulirani p = 0.33),
  preklapanje uzastopnih kola (egzaktni p = 0.16), najdulji niz pojavljivanja (simulirani p = 0.33).

### Konačne kombinacije za 2026-10-23 19:00 (Zagreb)

Primarni (unaprijed prijavljeni) rezultat je **ensemble s jednakim težinama**; kombinacije svake
metode su u `results/final/combinations.json` i u izvještaju.

| Igra | Ensemble (jednake težine) |
|---|---|
| Sve ili Ništa 11/22 | 01 02 04 05 06 07 09 10 11 13 15 |
| Eurojackpot 5/50 + 2/12 | 01 02 07 09 23 + 02 11 |
| Joker | 670764 |

Ove kombinacije imaju točno istu vjerojatnost kao bilo koje druge: Sve ili Ništa 11/11 (ili 0/11)
1 : 705 432, Eurojackpot 5 + 2 1 : 139 838 160, Joker svih 6 znamenki 1 : 1 000 000. Eurojackpot i
Joker kombinacije nisu ni backtestirane (nema podataka).

### Četiri očitanja za 2026-10-24 19:00 (Zagreb)

Izvođenje `--draw-date 2026-10-24 --readings 4 --backtest-from results`, rezultati u
`results/draw_2026-10-24/` (izvještaj: [`results/draw_2026-10-24/REPORT.md`](results/draw_2026-10-24/REPORT.md)).
Sve ili Ništa, ensemble s jednakim težinama po očitanju:

| Očitanje | Seed | Sve ili Ništa 11/22 |
|---|---|---|
| 1 | 20261023 | 01 02 04 05 06 07 08 09 10 11 18 |
| 2 | 20261024 | 01 02 03 04 07 08 09 10 11 14 18 |
| 3 | 20261025 | 01 02 03 04 05 06 07 08 09 10 15 |
| 4 | 20261026 | 01 02 04 05 07 08 09 10 11 12 14 |

Četiri listića pokrivaju 15 od 22 broja (13, 16, 17, 19–22 nisu ni na jednom) i dijele 8–9 brojeva
po paru. Ta **sklonost malim brojevima** je strukturno svojstvo ensemblea: numerološke metode daju
pretežno jednoznamenkaste sirove brojeve, Meihua brojeve trigrama 1–8, pa Borda zbroj sustavno
favorizira 1–11. Na očekivani broj pogodaka to ne utječe (5.5 za svaki listić), ali listići su
međusobno korelirani. Pojedinačne metode (tablice u izvještaju) su raznolikije.

## Reproducibilnost

* Jedan globalni seed (`seed: 20261023`); svako „slučajno” bacanje ili miješanje je PCG64 generator
  sjemen SHA-256-om ulaza, pa isti ulaz uvijek daje isti listić.
* JSON se piše sa sortiranim ključevima i zaokruženim decimalama; dva uzastopna izvođenja daju
  bajt-identične datoteke osim `generated_at` u `manifest.json`.
* `manifest.json` bilježi SHA-256 konfiguracije i svih `.py` datoteka, verzije paketa i Pythona.
