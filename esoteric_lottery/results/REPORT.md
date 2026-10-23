# Izvještaj: ezoterično generiranje lutrijskih brojeva

> Lutrijska izvlačenja su slučajna. Nijedna metoda u ovom izvještaju ne mijenja vjerojatnost dobitka; ovo je istraživanje koje to *mjeri*, a ne preporuka za igru. Svi brojevi niže izračunati su ovim pipelineom; ništa nije procijenjeno ručno.

## Ulazni podaci

* Rođenje: 1991-05-31T18:20:00+02:00 (Zadar) = 1991-05-31T16:20:00+00:00 UTC
* Izvlačenje: 2026-10-23T19:00:00+02:00 (Zagreb) = 2026-10-23T17:00:00+00:00 UTC
* Ime: nije zadano (izračuni koji traže ime su preskočeni)
* Seed: 20261023; konfiguracija SHA-256 `0d284f49b32f45c6…`, kod SHA-256 `351613b3defd0ff3…`

## Što se stvarno izvršilo

| Metoda | Status | Komponenta | Status komponente | Napomena |
|---|---|---|---|---|
| Meihua Yishu (梅花易數, vremensko bacanje) | izvršeno | chinese_lunar_calendar | izvršeno | computed from Swiss Ephemeris new moons |
|  |  | draw_time_cast | izvršeno |  |
|  |  | birth_time_cast | izvršeno |  |
| I Ching (simulirano bacanje stabljika stolisnika) | izvršeno | yarrow_stalk_cast | izvršeno | 12 hexagrams cast |
|  |  | king_wen_lookup | izvršeno |  |
| Zapadna (tropska) astrologija | izvršeno | natal_chart | izvršeno | Placidus (P), Moshier ephemeris |
|  |  | transit_chart | izvršeno |  |
|  |  | transit_to_natal_aspects | izvršeno | 38 aspects within orb |
| Vedska astrologija (Jyotish, Panchanga, Vimshottari) | izvršeno | sidereal_positions_lahiri | izvršeno |  |
|  |  | panchanga | izvršeno |  |
|  |  | whole_sign_lagna | izvršeno |  |
|  |  | vimshottari_dasha | izvršeno | maha/antar/pratyantar |
| Pitagorejska numerologija | djelomično | date_numbers | izvršeno | life path, birthday, attitude, pinnacles, challenges |
|  |  | personal_cycles | izvršeno | personal year/month/day, universal day |
|  |  | name_numbers | preskočeno | no name in config.yaml (subject.name is null) |
|  |  | place_name_numbers | izvršeno | experimental |
| Kaldejska numerologija (Cheiro) | djelomično | draw_date_numbers | izvršeno |  |
|  |  | birth_date_numbers | izvršeno |  |
|  |  | name_number | preskočeno | no name in config.yaml (subject.name is null) |
|  |  | place_name_numbers | izvršeno | experimental |
| Gematrija (hebrejska, grčka, latinska/Agrippa, engleska ordinalna) | djelomično | latin_ciphers | izvršeno | ordinal, reverse ordinal, Agrippa |
|  |  | name_cipher | preskočeno | no name in config.yaml (subject.name is null) |
|  |  | hebrew_mispar_hechrachi | preskočeno | no Hebrew text supplied (gematria.extra_texts) |
|  |  | greek_isopsephy | preskočeno | no Greek text supplied (gematria.extra_texts) |
| Tarot (Rider-Waite-Smith, Greerine rođendanske karte) | izvršeno | greer_birth_cards | izvršeno |  |
|  |  | greer_year_cards | izvršeno |  |
|  |  | seeded_spread | izvršeno | 22 cards from a 78-card deck |
| Lenormand (36 karata, Grand Tableau) | izvršeno | day_of_month_cards | izvršeno | experimental |
|  |  | seeded_grand_tableau | izvršeno | 4 x 9 |

* Backtest sve_ili_nista: **izvršeno**
* Backtest eurojackpot: **nije pokrenuto** – nije priložena datoteka s povijesnim izvlačenjima
* Backtest joker: **nije pokrenuto** – nije priložena datoteka s povijesnim izvlačenjima

## Konačne kombinacije

Stupac *iz metode* kaže koliko je brojeva metoda sama odredila; ostatak popunjava reproducibilni SHA-256 tie-break (vidi README, „Mapiranje u raspon”).

### Sve ili Ništa 11/22

| Metoda | Brojevi (11/22) | Iz metode |
|---|---|---|
| Meihua Yishu (梅花易數, vremensko bacanje) | 01 02 04 06 08 09 10 15 18 19 22 | 5/11 |
| I Ching (simulirano bacanje stabljika stolisnika) | 02 03 05 06 07 11 14 15 17 19 20 | 11/11 |
| Zapadna (tropska) astrologija | 01 04 05 08 09 10 13 14 15 18 19 | 11/11 |
| Vedska astrologija (Jyotish, Panchanga, Vimshottari) | 01 02 03 04 05 06 07 09 12 13 20 | 11/11 |
| Pitagorejska numerologija | 01 02 03 04 05 06 07 09 10 11 21 | 10/11 |
| Kaldejska numerologija (Cheiro) | 01 02 04 05 06 07 09 14 15 16 20 | 10/11 |
| Gematrija (hebrejska, grčka, latinska/Agrippa, engleska ordinalna) | 01 02 04 08 09 10 11 15 16 21 22 | 11/11 |
| Tarot (Rider-Waite-Smith, Greerine rođendanske karte) | 01 02 03 07 09 10 11 13 16 18 22 | 11/11 |
| Lenormand (36 karata, Grand Tableau) | 01 04 06 07 08 09 10 11 13 16 19 | 11/11 |
| **Ensemble (Borda, jednake težine) – primarni** | **01 02 04 05 06 07 09 10 11 13 15** | – |
| **Ensemble (težine iz razdoblja treniranja)** | **01 02 05 06 07 09 10 11 14 15 16** | – |

### Eurojackpot 5/50 + 2/12

| Metoda | Glavni brojevi (5/50) | Euro brojevi (2/12) | Iz metode |
|---|---|---|---|
| Meihua Yishu (梅花易數, vremensko bacanje) | 04 06 08 23 45 | 08 09 | 5/5, 2/2 |
| I Ching (simulirano bacanje stabljika stolisnika) | 05 14 15 28 46 | 02 10 | 5/5, 2/2 |
| Zapadna (tropska) astrologija | 08 17 18 32 43 | 05 08 | 5/5, 2/2 |
| Vedska astrologija (Jyotish, Panchanga, Vimshottari) | 04 07 13 25 26 | 01 02 | 5/5, 2/2 |
| Pitagorejska numerologija | 01 02 07 09 11 | 07 11 | 5/5, 2/2 |
| Kaldejska numerologija (Cheiro) | 05 06 07 16 23 | 05 11 | 5/5, 2/2 |
| Gematrija (hebrejska, grčka, latinska/Agrippa, engleska ordinalna) | 03 20 26 32 46 | 05 10 | 5/5, 2/2 |
| Tarot (Rider-Waite-Smith, Greerine rođendanske karte) | 01 02 09 10 11 | 02 11 | 5/5, 2/2 |
| Lenormand (36 karata, Grand Tableau) | 01 09 16 23 31 | 07 11 | 5/5, 2/2 |
| **Ensemble (Borda, jednake težine) – primarni** | **01 02 07 09 23** | **02 11** | – |

### Joker (6 znamenki)

| Metoda | Joker | Iz metode |
|---|---|---|
| Meihua Yishu (梅花易數, vremensko bacanje) | 684835 | 6/6 |
| I Ching (simulirano bacanje stabljika stolisnika) | 977989 | 6/6 |
| Zapadna (tropska) astrologija | 080143 | 6/6 |
| Vedska astrologija (Jyotish, Panchanga, Vimshottari) | 363561 | 6/6 |
| Pitagorejska numerologija | 112774 | 6/6 |
| Kaldejska numerologija (Cheiro) | 425761 | 6/6 |
| Gematrija (hebrejska, grčka, latinska/Agrippa, engleska ordinalna) | 326604 | 6/6 |
| Tarot (Rider-Waite-Smith, Greerine rođendanske karte) | 979024 | 6/6 |
| Lenormand (36 karata, Grand Tableau) | 691639 | 6/6 |
| **Ensemble (Borda, jednake težine) – primarni** | **670764** | – |

## Teorijske vjerojatnosti (matematika, ne procjena)

* Sve ili Ništa: 705.432 kombinacija; P(11 pogodaka) = P(0 pogodaka) = 1/705.432; očekivano 5.5 pogodaka po listiću bez obzira na metodu.
* Eurojackpot: P(5+2) = 1/139.838.160.
* Joker: P(svih 6 znamenki na svojim mjestima) = 1/1.000.000.

## Backtest: sve_ili_nista

### Podaci

* Datoteka `sve_ili_nista_sva_kola.csv`: 636 redaka, 635 potpunih izvlačenja (2025-01-01 – 2026-09-28).
* Bez brojeva (izostavljeno): 2026/178 (2026-06-27).
* Kalendarske praznine: 2026-06-27.
* Provjere (sve prošle): broj i raspon brojeva, bez duplikata u kolu, kronološki redoslijed, jedinstveni datumi, stupac Zbroj odgovara, stupac Parni odgovara.
* Usporedba s `Sve_ili_Nista_sva_kola.xlsx` (list „Izvlačenja”): identično.
* Treniranje: 365 kola (do 2025-12-31); evaluacija: 270 kola (2026-01-01 – 2026-09-28).
* Pretpostavljeno vrijeme izvlačenja u povijesti: 19:00 (datoteke ne sadrže vrijeme).

### Evaluacijsko razdoblje (bazen `main`)

Nulta hipoteza: listić je fiksiran prije izvlačenja, a izvlačenje je uniformno slučajno → pogoci su hipergeometrijski, očekivano 5.500 po kolu (SD 1.200). Obitelj testova: 13 strategija, α = 0.05, Holm (primarno) i Benjamini-Hochberg.

| Strategija | Prosjek pogodaka | 95% CI (t) | 95% CI (bootstrap) | Razlika | p (egzaktni) | p (MC) | p Holm | p BH | AUC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| Meihua Yishu (梅花易數, vremensko bacanje) | 5.493 | [5.346, 5.639] | [5.344, 5.637] | -0.007 | 0.9394 | 0.9359 | 1.0000 | 0.9394 | 0.502 [0.486, 0.519] |
| I Ching (simulirano bacanje stabljika stolisnika) | 5.489 | [5.348, 5.630] | [5.348, 5.630] | -0.011 | 0.8991 | 0.8976 | 1.0000 | 0.9394 | 0.502 [0.487, 0.515] |
| Zapadna (tropska) astrologija | 5.593 | [5.444, 5.741] | [5.448, 5.741] | +0.093 | 0.2141 | 0.2146 | 1.0000 | 0.5501 | 0.511 [0.495, 0.526] |
| Vedska astrologija (Jyotish, Panchanga, Vimshottari) | 5.541 | [5.403, 5.678] | [5.404, 5.674] | +0.041 | 0.5944 | 0.5866 | 1.0000 | 0.7727 | 0.505 [0.489, 0.521] |
| Pitagorejska numerologija | 5.570 | [5.417, 5.724] | [5.419, 5.722] | +0.070 | 0.3482 | 0.3461 | 1.0000 | 0.6466 | 0.516 [0.499, 0.531] |
| Kaldejska numerologija (Cheiro) | 5.593 | [5.447, 5.738] | [5.444, 5.741] | +0.093 | 0.2141 | 0.2146 | 1.0000 | 0.5501 | 0.514 [0.500, 0.530] |
| Gematrija (hebrejska, grčka, latinska/Agrippa, engleska ordinalna) | 5.415 | [5.269, 5.561] | [5.270, 5.556] | -0.085 | 0.2539 | 0.2535 | 1.0000 | 0.5501 | 0.494 [0.480, 0.510] |
| Tarot (Rider-Waite-Smith, Greerine rođendanske karte) | 5.296 | [5.158, 5.435] | [5.159, 5.433] | -0.204 | 0.0057 | 0.0063 | 0.0742 | 0.0742 | 0.481 [0.467, 0.496] |
| Lenormand (36 karata, Grand Tableau) | 5.526 | [5.379, 5.673] | [5.378, 5.674] | +0.026 | 0.7417 | 0.7350 | 1.0000 | 0.8765 | 0.500 [0.486, 0.515] |
| Ensemble (Borda, jednake težine) – primarni | 5.559 | [5.404, 5.715] | [5.404, 5.715] | +0.059 | 0.4319 | 0.4305 | 1.0000 | 0.7018 | 0.507 [0.490, 0.523] |
| Ensemble (težine iz razdoblja treniranja) | 5.400 | [5.248, 5.552] | [5.248, 5.552] | -0.100 | 0.1790 | 0.1787 | 1.0000 | 0.5501 | 0.496 [0.479, 0.513] |
| Kontrola: najčešći u zadnjih 30 kola | 5.356 | [5.214, 5.497] | [5.211, 5.493] | -0.144 | 0.0509 | 0.0577 | 0.6106 | 0.3308 | 0.485 [0.470, 0.500] |
| Kontrola: logistička regresija (sklearn) | 5.541 | [5.402, 5.679] | [5.404, 5.681] | +0.041 | 0.5944 | 0.5866 | 1.0000 | 0.7727 | 0.499 [0.484, 0.515] |

Uniformna slučajna osnovica (10.000 nasumičnih strategija na istim kolima): prosjek 5.501, 95% raspon [5.355, 5.648]. *p (MC)* je udio tih strategija koje od 5.5 odstupaju barem koliko i promatrana.
Najmanji učinak koji ovaj test s 80% snage može otkriti (Bonferroni, 13 testova): ±0.273 pogodaka po kolu.

**Višestruko testiranje.** Bez korekcije p < α: Tarot (Rider-Waite-Smith, Greerine rođendanske karte). Nakon Holmove korekcije nijedna strategija nije značajna. Ni BH (FDR) ne ostavlja nijednu.

#### Distribucija broja pogodaka (evaluacija)

χ² (usporedba cijele distribucije s H0) i stopa ekstrema su *sekundarne, eksploratorne* mjere bez korekcije: na 26 takvih testova slučajno se očekuje oko 1.3 p-vrijednosti < 0.05.

| Strategija | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | χ² p | ekstremi (≤3 ili ≥8) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *očekivano (H0)* | 0.0 | 0.0 | 1.2 | 10.4 | 41.7 | 81.7 | 81.7 | 41.7 | 10.4 | 1.2 | 0.0 | 0.0 | – | 23.2 |
| Meihua Yishu (梅花易數, vremensko bacanje) | 0 | 0 | 1 | 10 | 47 | 79 | 75 | 48 | 8 | 2 | 0 | 0 | 0.7712 | 21 (0.078, CI [0.051, 0.116], p=0.7444) |
| I Ching (simulirano bacanje stabljika stolisnika) | 0 | 0 | 0 | 11 | 43 | 82 | 83 | 40 | 10 | 1 | 0 | 0 | 0.9991 | 22 (0.081, CI [0.054, 0.120], p=0.9135) |
| Zapadna (tropska) astrologija | 0 | 0 | 0 | 14 | 36 | 71 | 91 | 44 | 11 | 3 | 0 | 0 | 0.5024 | 28 (0.104, CI [0.073, 0.146], p=0.2793) |
| Vedska astrologija (Jyotish, Panchanga, Vimshottari) | 0 | 0 | 2 | 8 | 37 | 76 | 100 | 37 | 9 | 1 | 0 | 0 | 0.3058 | 20 (0.074, CI [0.048, 0.112], p=0.5867) |
| Pitagorejska numerologija | 0 | 0 | 2 | 10 | 40 | 81 | 72 | 50 | 13 | 1 | 1 | 0 | 0.5673 | 27 (0.100, CI [0.070, 0.142], p=0.3863) |
| Kaldejska numerologija (Cheiro) | 0 | 0 | 3 | 5 | 45 | 69 | 84 | 52 | 12 | 0 | 0 | 0 | 0.3063 | 20 (0.074, CI [0.048, 0.112], p=0.5867) |
| Gematrija (hebrejska, grčka, latinska/Agrippa, engleska ordinalna) | 0 | 0 | 1 | 10 | 55 | 75 | 81 | 37 | 9 | 2 | 0 | 0 | 0.3687 | 22 (0.081, CI [0.054, 0.120], p=0.9135) |
| Tarot (Rider-Waite-Smith, Greerine rođendanske karte) | 0 | 0 | 1 | 12 | 49 | 102 | 66 | 31 | 8 | 1 | 0 | 0 | 0.0249 | 22 (0.081, CI [0.054, 0.120], p=0.9135) |
| Lenormand (36 karata, Grand Tableau) | 0 | 0 | 2 | 10 | 41 | 79 | 79 | 49 | 8 | 2 | 0 | 0 | 0.8872 | 22 (0.081, CI [0.054, 0.120], p=0.9135) |
| Ensemble (Borda, jednake težine) – primarni | 0 | 0 | 3 | 11 | 36 | 83 | 78 | 40 | 16 | 3 | 0 | 0 | 0.2877 | 33 (0.122, CI [0.088, 0.167], p=0.0390) |
| Ensemble (težine iz razdoblja treniranja) | 0 | 0 | 5 | 9 | 53 | 73 | 76 | 45 | 8 | 1 | 0 | 0 | 0.3325 | 23 (0.085, CI [0.057, 0.125], p=1.0000) |
| Kontrola: najčešći u zadnjih 30 kola | 0 | 0 | 3 | 10 | 49 | 87 | 75 | 39 | 7 | 0 | 0 | 0 | 0.4997 | 20 (0.074, CI [0.048, 0.112], p=0.5867) |
| Kontrola: logistička regresija (sklearn) | 0 | 0 | 1 | 6 | 42 | 87 | 74 | 53 | 5 | 2 | 0 | 0 | 0.1661 | 14 (0.052, CI [0.031, 0.085], p=0.0497) |

#### Razdoblje treniranja (samo opisno)

| Strategija | n | Prosjek | 95% CI (t) | p (egzaktni, nekorigirano) |
|---|---|---|---|---|
| Meihua Yishu (梅花易數, vremensko bacanje) | 365 | 5.482 | [5.356, 5.608] | 0.7936 |
| I Ching (simulirano bacanje stabljika stolisnika) | 365 | 5.595 | [5.467, 5.722] | 0.1381 |
| Zapadna (tropska) astrologija | 365 | 5.532 | [5.410, 5.653] | 0.6314 |
| Vedska astrologija (Jyotish, Panchanga, Vimshottari) | 365 | 5.425 | [5.295, 5.554] | 0.2390 |
| Pitagorejska numerologija | 365 | 5.386 | [5.259, 5.513] | 0.0737 |
| Kaldejska numerologija (Cheiro) | 365 | 5.521 | [5.404, 5.637] | 0.7601 |
| Gematrija (hebrejska, grčka, latinska/Agrippa, engleska ordinalna) | 365 | 5.532 | [5.403, 5.660] | 0.6314 |
| Tarot (Rider-Waite-Smith, Greerine rođendanske karte) | 365 | 5.611 | [5.480, 5.742] | 0.0811 |
| Lenormand (36 karata, Grand Tableau) | 365 | 5.416 | [5.290, 5.543] | 0.1907 |
| Ensemble (Borda, jednake težine) – primarni | 365 | 5.460 | [5.337, 5.584] | 0.5415 |
| Ensemble (težine iz razdoblja treniranja) | – | – | – | – (nema van-uzoračkih predviđanja) |
| Kontrola: najčešći u zadnjih 30 kola | 365 | 5.477 | [5.349, 5.605] | 0.7272 |
| Kontrola: logistička regresija (sklearn) | – | – | – | – (nema van-uzoračkih predviđanja) |

#### Težine ensemblea iz razdoblja treniranja (zamrznute prije evaluacije)

| Metoda | z (treniranje) | težina |
|---|---|---|
| Kaldejska numerologija (Cheiro) | +0.327 | 0.998 |
| Gematrija (hebrejska, grčka, latinska/Agrippa, engleska ordinalna) | +0.502 | 1.089 |
| I Ching (simulirano bacanje stabljika stolisnika) | +1.505 | 1.798 |
| Lenormand (36 karata, Grand Tableau) | -1.330 | 0.436 |
| Meihua Yishu (梅花易數, vremensko bacanje) | -0.283 | 0.735 |
| Pitagorejska numerologija | -1.810 | 0.343 |
| Tarot (Rider-Waite-Smith, Greerine rođendanske karte) | +1.766 | 2.049 |
| Vedska astrologija (Jyotish, Panchanga, Vimshottari) | -1.199 | 0.465 |
| Zapadna (tropska) astrologija | +0.502 | 1.089 |

#### Logistička kontrola: vremenska unakrsna validacija unutar treniranja

| Fold | kola (train/test) | log-loss | log-loss uz p=0.5 | AUC |
|---|---|---|---|---|
| 1 | 45/44 | 0.6936 | 0.6931 | 0.494 |
| 2 | 89/44 | 0.6943 | 0.6931 | 0.494 |
| 3 | 133/44 | 0.6942 | 0.6931 | 0.490 |
| 4 | 177/44 | 0.6931 | 0.6931 | 0.505 |
| 5 | 221/44 | 0.6935 | 0.6931 | 0.492 |

#### Kvaliteta i slučajnost samih izvlačenja (opisno)

* Učestalost brojeva: statistika 12.25, simulirani p = 0.3328 (2000 simulacija); raspon pojavljivanja 287–340, očekivano 317.5.
* Preklapanje uzastopnih kola: prosjek 5.568 (očekivano 5.500), egzaktni p = 0.1596.
* Najdulji niz uzastopnih pojavljivanja: 14 (broj 5); simulirani p (maksimum preko svih brojeva) = 0.3343, medijan u simulaciji 13.

## Backtest: eurojackpot

Nije pokrenut: nije priložena datoteka s povijesnim izvlačenjima. Bez povijesnih izvlačenja nema evaluacije; kombinacije za ovu igru nisu testirane ni na čemu.

## Backtest: joker

Nije pokrenut: nije priložena datoteka s povijesnim izvlačenjima. Bez povijesnih izvlačenja nema evaluacije; kombinacije za ovu igru nisu testirane ni na čemu.

## Zaključak

* sve_ili_nista/main: nijedna metoda, ensemble ni statistička kontrola ne odstupa od uniformne slučajne osnovice nakon korekcije za višestruko testiranje. Opaženo je kompatibilno s time da su sve kombinacije jednako (ne)vjerojatne.
* eurojackpot: backtest nije bio moguć (nije priložena datoteka s povijesnim izvlačenjima); o uspješnosti se ne može reći ništa.
* joker: backtest nije bio moguć (nije priložena datoteka s povijesnim izvlačenjima); o uspješnosti se ne može reći ništa.

Detalji, izvori i ograničenja: `README.md`. Svi međurezultati: `results/final/intermediate/*.json`, `results/backtest/*/summary.json`, `results/backtest/*/predictions/*.json`.
