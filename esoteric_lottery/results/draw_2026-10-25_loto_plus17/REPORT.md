# Izvještaj: ezoterično generiranje lutrijskih brojeva

> Lutrijska izvlačenja su slučajna. Nijedna metoda u ovom izvještaju ne mijenja vjerojatnost dobitka; ovo je istraživanje koje to *mjeri*, a ne preporuka za igru. Svi brojevi niže izračunati su ovim pipelineom; ništa nije procijenjeno ručno.

## Ulazni podaci

* Rođenje: 1991-05-31T18:20:00+02:00 (Zadar) = 1991-05-31T16:20:00+00:00 UTC
* Izvlačenje: 2026-10-25T19:00:00+01:00 (Zagreb) = 2026-10-25T18:00:00+00:00 UTC
* Ime: nije zadano (izračuni koji traže ime su preskočeni)
* Seed: 20261023; konfiguracija SHA-256 `56c1a7c5ad032d0f…`, kod SHA-256 `e8b661731d34fd76…`
* Očitanja: 5 (seedovi 20261023, 20261024, 20261025, 20261026, 20261027). Očitanje k koristi seed 20261023 + (k − 1): njime se ponovno bacaju simulirani dijelovi orakula (I Ching, miješanje tarota, Lenormandov Grand Tableau) i ponovno izvlači SHA-256 tie-break svake metode, a tie-break ensemblea dodatno se soli rednim brojem očitanja. Sirovi brojevi Meihue, zapadne i vedske astrologije, pitagorejske i kaldejske numerologije te gematrije ne ovise o seedu; kod njih se mijenja samo popuna brojeva koje metoda nije sama odredila. Očitanje 1 je standardno pokretanje.

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
|  |  | transit_to_natal_aspects | izvršeno | 34 aspects within orb |
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

* Backtest sve_ili_nista: **ponovno korišten** – iz `results/backtest/sve_ili_nista/summary.json`; backtest ne ovisi o datumu ciljanog izvlačenja, a konfiguracija, povijesni podaci i kod koji ga računa provjereno su nepromijenjeni (SHA-256)
* Backtest eurojackpot: **nije pokrenut** – nije priložena datoteka s povijesnim izvlačenjima
* Backtest loto: **nije pokrenut** – nije priložena datoteka s povijesnim izvlačenjima
* Backtest joker: **nije pokrenut** – nije priložena datoteka s povijesnim izvlačenjima

## Listići po očitanjima

Primarni listić svakog očitanja je ensemble s jednakim težinama (Borda). Svi listići imaju jednaku vjerojatnost dobitka; razlike među njima dolaze samo iz ponovno bačenih simuliranih dijelova orakula i ponovno izvučenog tie-breaka. Više listića znači više odigranih kombinacija, a ne bolje kombinacije.

### Sve ili Ništa 11/22

| Očitanje | Seed | Ensemble (jednake težine) | Ensemble (težine iz razdoblja treniranja) |
|---|---|---|---|
| 1 | 20261023 | **01 02 03 04 05 06 07 08 09 11 15** | 01 02 04 05 06 07 08 09 10 11 15 |
| 2 | 20261024 | **01 02 03 04 05 06 07 08 09 10 15** | 01 02 04 05 06 08 09 10 11 15 19 |
| 3 | 20261025 | **01 02 03 04 05 06 07 08 09 10 15** | 01 02 03 04 06 07 08 09 10 15 16 |
| 4 | 20261026 | **01 02 03 04 05 06 07 08 09 10 15** | 01 02 03 07 08 09 10 13 15 16 18 |
| 5 | 20261027 | **01 02 03 04 05 07 08 09 10 11 15** | 01 02 03 04 07 09 10 11 15 17 18 |

Pokrivenost (`main`): različitih listića 3 od 5; pokriveni brojevi 12/22; ni na jednom listiću: 12, 13, 14, 16, 17, 18, 19, 20, 21, 22. Zajednički brojevi po parovima očitanja: 1–2: 10, 1–3: 10, 1–4: 10, 1–5: 10, 2–3: 11, 2–4: 11, 2–5: 10, 3–4: 11, 3–5: 10, 4–5: 10. (Pokrivenost je opis odnosa među listićima, ne mjera kvalitete: očekivani broj pogodaka svakog pojedinog listića je isti.)

### Eurojackpot 5/50 + 2/12

| Očitanje | Seed | Ensemble (jednake težine) | Ensemble (težine iz razdoblja treniranja) |
|---|---|---|---|
| 1 | 20261023 | **01 02 05 11 20 + 01 07** | – |
| 2 | 20261024 | **01 02 03 04 05 + 01 07** | – |
| 3 | 20261025 | **01 02 09 10 32 + 01 07** | – |
| 4 | 20261026 | **01 02 09 11 15 + 01 07** | – |
| 5 | 20261027 | **01 02 09 11 32 + 01 07** | – |

Za ovu igru nema ensemblea «Ensemble (težine iz razdoblja treniranja)» (stupac je prazan) jer backtest nije pokrenut: nije priložena datoteka s povijesnim izvlačenjima. Nijedan listić za ovu igru nije ni na čemu testiran.

Pokrivenost (`main`): različitih listića 5 od 5; pokriveni brojevi 11/50; ni na jednom listiću: 6, 7, 8, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50. Zajednički brojevi po parovima očitanja: 1–2: 3, 1–3: 2, 1–4: 3, 1–5: 3, 2–3: 2, 2–4: 2, 2–5: 2, 3–4: 3, 3–5: 4, 4–5: 4. (Pokrivenost je opis odnosa među listićima, ne mjera kvalitete: očekivani broj pogodaka svakog pojedinog listića je isti.)

Pokrivenost (`euro`): različitih listića 1 od 5; pokriveni brojevi 2/12; ni na jednom listiću: 2, 3, 4, 5, 6, 8, 9, 10, 11, 12. Zajednički brojevi po parovima očitanja: 1–2: 2, 1–3: 2, 1–4: 2, 1–5: 2, 2–3: 2, 2–4: 2, 2–5: 2, 3–4: 2, 3–5: 2, 4–5: 2. (Pokrivenost je opis odnosa među listićima, ne mjera kvalitete: očekivani broj pogodaka svakog pojedinog listića je isti.)

### Loto 6/45

| Očitanje | Seed | Ensemble (jednake težine) | Ensemble (težine iz razdoblja treniranja) |
|---|---|---|---|
| 1 | 20261023 | **01 02 06 07 16 30** | – |
| 2 | 20261024 | **01 02 07 09 15 20** | – |
| 3 | 20261025 | **01 02 06 07 10 31** | – |
| 4 | 20261026 | **01 02 06 09 15 16** | – |
| 5 | 20261027 | **01 02 07 10 11 15** | – |

Za ovu igru nema ensemblea «Ensemble (težine iz razdoblja treniranja)» (stupac je prazan) jer backtest nije pokrenut: nije priložena datoteka s povijesnim izvlačenjima. Nijedan listić za ovu igru nije ni na čemu testiran.

Pokrivenost (`main`): različitih listića 5 od 5; pokriveni brojevi 12/45; ni na jednom listiću: 3, 4, 5, 8, 12, 13, 14, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45. Zajednički brojevi po parovima očitanja: 1–2: 3, 1–3: 4, 1–4: 4, 1–5: 3, 2–3: 3, 2–4: 4, 2–5: 4, 3–4: 3, 3–5: 4, 4–5: 3. (Pokrivenost je opis odnosa među listićima, ne mjera kvalitete: očekivani broj pogodaka svakog pojedinog listića je isti.)

### Joker (6 znamenki)

| Očitanje | Seed | Ensemble (jednake težine) | Ensemble (težine iz razdoblja treniranja) |
|---|---|---|---|
| 1 | 20261023 | **117614** | – |
| 2 | 20261024 | **570901** | – |
| 3 | 20261025 | **162911** | – |
| 4 | 20261026 | **115914** | – |
| 5 | 20261027 | **965901** | – |

Za ovu igru nema ensemblea «Ensemble (težine iz razdoblja treniranja)» (stupac je prazan) jer backtest nije pokrenut: nije priložena datoteka s povijesnim izvlačenjima. Nijedan listić za ovu igru nije ni na čemu testiran.

## Izbor skupa listića (jedino što se stvarno može optimizirati)

Pojedinačni listić se ne može poboljšati: pod nultom hipotezom svaki listić od 11 brojeva ima istu hipergeometrijsku razdiobu pogodaka, pa i istu vjerojatnost dobitka. Backtest u `results/` ne nalazi ništa što to pobija.

Ono što se kod **više listića zajedno** mijenja jest njihova *zajednička* razdioba. Očekivani zbroj pogodaka je nepromjenjiv, ali vjerojatnost da **barem jedan** listić padne u isplativi razred ovisi o tome koliko se listići međusobno preklapaju. Dvije krajnosti su loše: gotovo isti listići dobivaju i gube zajedno, a **komplementarni** par (preklapanje 0) je zamka jer je tada `pogoci(A) + pogoci(Ā) = 11`, pa oba listića ulaze u isplativi razred u isti mah. Optimum je u sredini.

Cilj ovisi **isključivo o preklapanjima listića**, nikad o tome koji su brojevi na njima: skup i njegova kopija s preimenovanim brojevima imaju identičnu ocjenu (`tests/test_portfolio.py`). Zato taj izbor ne sadrži nikakvo predviđanje — to je kombinatorika, ne prognoza. Sami brojevi i dalje dolaze isključivo iz ezoteričnih metoda; bira se samo koja će se četiri njihova listića igrati zajedno.

### Sve ili Ništa 11/22

Isplativi broj pogodaka: 0, 1, 2, 3, 8, 9, 10, 11 (razredi su simetrični: 11 ili 0 je glavni zgoditak, zatim 10/1, 9/2, 8/3; 4–7 ne donosi ništa). Iznosi dobitaka nigdje se ne računaju jer službena tablica nije provjerena.

| # | Listić | Izvor (metoda#očitanje) |
|---|---|---|
| 1 | **01 02 03 04 05 06 07 09 10 11 14** | pythagorean#3 |
| 2 | **01 02 03 07 08 09 10 13 15 16 18** | ensemble_train_weighted_sve_ili_nista#4 |
| 3 | **01 02 05 09 10 11 13 15 19 20 21** | tarot#2 |
| 4 | **01 02 06 07 08 11 14 15 16 20 21** | iching#1 |

Preklapanja parova: [6, 6, 6, 6, 6, 6] (neutralno bi bilo 5.5). Izabrano egzaktnim izračunom zajedničke razdiobe: od 39 različitih listića koje su metode dale, 45912 kombinacija po četiri prošlo je filtar preklapanja [3, 8], a najboljih 200 ocijenjeno je egzaktno.

| Skup listića | Preklapanja | P(barem jedan dobitan) | E[zbroj pogodaka] | E[najbolji listić] | P(barem jedan 11 ili 0) |
|---|---|---|---|---|---|
| **izabrani skup** | [6, 6, 6, 6, 6, 6] | **0.3147** (1 u 3) | 22.000 | 6.654 | 1.134e-05 |
| isti listić 4 puta | [11, 11, 11, 11, 11, 11] | **0.0861** (1 u 12) | 22.000 | 5.500 | 2.835e-06 |
| 4 ensemble listića po očitanjima | [10, 10, 10, 11, 11, 11] | **0.1269** (1 u 8) | 22.000 | 5.762 | 5.670e-06 |
| variant_all | [6, 6, 6, 6, 6, 6] | **0.3147** (1 u 3) | 22.000 | 6.654 | 1.134e-05 |
| jedan jedini listić | – | 0.0861 (1 u 12) | 5.500 | 5.500 | 2.835e-06 |

**Što se promijenilo:** vjerojatnost da barem jedan od četiri listića nešto osvoji raste s 0.0861 (isti listić četiri puta) na 0.3147. **Što se nije promijenilo:** očekivani zbroj pogodaka je 22.0 u svakom slučaju, a vjerojatnost glavnog zgoditka po listiću ostaje 1 prema 705.432. Ovo je preraspodjela ishoda, ne prednost.

### Loto 6/45

Isplativi broj pogodaka: 3, 4, 5, 6 (razredi su simetrični: 11 ili 0 je glavni zgoditak, zatim 10/1, 9/2, 8/3; 4–7 ne donosi ništa). Iznosi dobitaka nigdje se ne računaju jer službena tablica nije provjerena.

| # | Listić | Izvor (metoda#očitanje) |
|---|---|---|
| 1 | **01 02 04 06 09 11 31** | pythagorean#1, pythagorean#2, pythagorean#3, pythagorean#4, pythagorean#5 |
| 2 | **01 03 08 15 32 36 40** | meihua#1, meihua#2, meihua#3, meihua#4, meihua#5 |
| 3 | **02 05 13 15 16 18 25** | iching#4 |
| 4 | **06 07 10 14 20 30 43** | iching#1 |
| 5 | **12 19 24 25 31 32 35** | lenormand#3 |

Preklapanja parova: [1, 1, 1, 1, 1, 0, 1, 0, 1, 0] (neutralno bi bilo 1.0888888888888888). Izabrano egzaktnim izračunom zajedničke razdiobe: od 26 različitih listića koje su metode dale, 2261 kombinacija po četiri prošlo je filtar preklapanja [0, 2], a najboljih 200 ocijenjeno je egzaktno.

| Skup listića | Preklapanja | P(barem jedan dobitan) | E[zbroj pogodaka] | E[najbolji listić] | P(barem jedan 11 ili 0) |
|---|---|---|---|---|---|
| **izabrani skup** | [1, 1, 1, 1, 1, 0, 1, 0, 1, 0] | **0.1894** (1 u 5) | 4.667 | 1.970 | 4.297e-06 |
| isti listić 4 puta | [7, 7, 7, 7, 7, 7, 7, 7, 7, 7] | **0.0394** (1 u 25) | 4.667 | 0.933 | 8.594e-07 |
| variant_all | [1, 1, 1, 1, 1, 0, 1, 0, 1, 0] | **0.1894** (1 u 5) | 4.667 | 1.970 | 4.297e-06 |
| variant_ensemble_only | [4, 3, 5, 4, 5, 5, 5, 4, 4, 4] | **0.1213** (1 u 8) | 4.667 | 1.534 | 4.297e-06 |
| variant_methods_only | [1, 1, 1, 1, 1, 0, 1, 0, 1, 0] | **0.1894** (1 u 5) | 4.667 | 1.970 | 4.297e-06 |
| jedan jedini listić | – | 0.0394 (1 u 25) | 0.933 | 0.933 | 8.594e-07 |

**Što se promijenilo:** vjerojatnost da barem jedan od četiri listića nešto osvoji raste s 0.0394 (isti listić četiri puta) na 0.1894. **Što se nije promijenilo:** očekivani zbroj pogodaka je 4.7 u svakom slučaju, a vjerojatnost glavnog zgoditka po listiću ostaje 1 prema 705.432. Ovo je preraspodjela ishoda, ne prednost.

## Konačne kombinacije (očitanje 1)

Stupac *iz metode* kaže koliko je brojeva metoda sama odredila; ostatak popunjava reproducibilni SHA-256 tie-break (vidi README, „Mapiranje u raspon”).

### Sve ili Ništa 11/22

| Metoda | Brojevi (11/22) | Iz metode |
|---|---|---|
| Meihua Yishu (梅花易數, vremensko bacanje) | 01 03 04 06 08 10 14 15 16 18 21 | 10/11 |
| I Ching (simulirano bacanje stabljika stolisnika) | 01 02 06 07 08 11 14 15 16 20 21 | 11/11 |
| Zapadna (tropska) astrologija | 01 02 04 05 08 09 14 15 18 19 20 | 11/11 |
| Vedska astrologija (Jyotish, Panchanga, Vimshottari) | 01 02 03 04 05 07 08 09 12 15 20 | 11/11 |
| Pitagorejska numerologija | 01 02 03 04 05 06 07 09 10 11 13 | 10/11 |
| Kaldejska numerologija (Cheiro) | 01 02 03 04 05 06 07 09 15 18 20 | 10/11 |
| Gematrija (hebrejska, grčka, latinska/Agrippa, engleska ordinalna) | 01 04 05 08 09 10 11 15 16 17 22 | 11/11 |
| Tarot (Rider-Waite-Smith, Greerine rođendanske karte) | 01 02 04 06 07 09 10 11 13 16 19 | 11/11 |
| Lenormand (36 karata, Grand Tableau) | 01 02 03 05 06 08 09 10 12 13 14 | 11/11 |
| **Ensemble (Borda, jednake težine) – primarni** | **01 02 03 04 05 06 07 08 09 11 15** | – |
| **Ensemble (težine iz razdoblja treniranja)** | **01 02 04 05 06 07 08 09 10 11 15** | – |

### Eurojackpot 5/50 + 2/12

| Metoda | Glavni brojevi (5/50) | Euro brojevi (2/12) | Iz metode |
|---|---|---|---|
| Meihua Yishu (梅花易數, vremensko bacanje) | 03 08 15 36 40 | 03 12 | 5/5, 2/2 |
| I Ching (simulirano bacanje stabljika stolisnika) | 05 07 20 30 43 | 06 07 | 5/5, 2/2 |
| Zapadna (tropska) astrologija | 10 18 21 32 43 | 05 08 | 5/5, 2/2 |
| Vedska astrologija (Jyotish, Panchanga, Vimshottari) | 01 04 07 15 30 | 01 03 | 5/5, 2/2 |
| Pitagorejska numerologija | 01 02 06 09 11 | 09 11 | 5/5, 2/2 |
| Kaldejska numerologija (Cheiro) | 01 07 09 18 25 | 01 07 | 5/5, 2/2 |
| Gematrija (hebrejska, grčka, latinska/Agrippa, engleska ordinalna) | 03 05 11 43 46 | 01 11 | 5/5, 2/2 |
| Tarot (Rider-Waite-Smith, Greerine rođendanske karte) | 01 02 10 11 22 | 10 11 | 5/5, 2/2 |
| Lenormand (36 karata, Grand Tableau) | 03 08 25 31 34 | 01 07 | 5/5, 2/2 |
| **Ensemble (Borda, jednake težine) – primarni** | **01 02 05 11 20** | **01 07** | – |

### Loto 6/45

| Metoda | Brojevi (11/22) | Iz metode |
|---|---|---|
| Meihua Yishu (梅花易數, vremensko bacanje) | 01 03 08 15 36 40 | 6/11 |
| I Ching (simulirano bacanje stabljika stolisnika) | 06 07 10 14 30 43 | 6/11 |
| Zapadna (tropska) astrologija | 09 16 17 19 29 39 | 6/11 |
| Vedska astrologija (Jyotish, Panchanga, Vimshottari) | 01 02 04 07 15 30 | 6/11 |
| Pitagorejska numerologija | 01 02 06 09 11 31 | 6/11 |
| Kaldejska numerologija (Cheiro) | 01 06 07 09 18 25 | 6/11 |
| Gematrija (hebrejska, grčka, latinska/Agrippa, engleska ordinalna) | 03 06 16 19 20 30 | 6/11 |
| Tarot (Rider-Waite-Smith, Greerine rođendanske karte) | 01 02 10 11 25 27 | 6/11 |
| Lenormand (36 karata, Grand Tableau) | 03 08 25 31 32 34 | 6/11 |
| **Ensemble (Borda, jednake težine) – primarni** | **01 02 06 07 16 30** | – |

### Joker (6 znamenki)

| Metoda | Joker | Iz metode |
|---|---|---|
| Meihua Yishu (梅花易數, vremensko bacanje) | 831605 | 6/6 |
| I Ching (simulirano bacanje stabljika stolisnika) | 787789 | 6/6 |
| Zapadna (tropska) astrologija | 260953 | 6/6 |
| Vedska astrologija (Jyotish, Panchanga, Vimshottari) | 515011 | 6/6 |
| Pitagorejska numerologija | 112994 | 6/6 |
| Kaldejska numerologija (Cheiro) | 427911 | 6/6 |
| Gematrija (hebrejska, grčka, latinska/Agrippa, engleska ordinalna) | 153604 | 6/6 |
| Tarot (Rider-Waite-Smith, Greerine rođendanske karte) | 681640 | 6/6 |
| Lenormand (36 karata, Grand Tableau) | 143828 | 6/6 |
| **Ensemble (Borda, jednake težine) – primarni** | **117614** | – |

## Teorijske vjerojatnosti (matematika, ne procjena)

* Sve ili Ništa: 705.432 kombinacija; P(11 pogodaka) = P(0 pogodaka) = 1/705.432; očekivano 5.5 pogodaka po listiću bez obzira na metodu.
* Eurojackpot: P(5+2) = 1/139.838.160.
* Joker: P(svih 6 znamenki na svojim mjestima) = 1/1.000.000.

## Backtest: sve_ili_nista

Ponovno korišten iz `results/backtest/sve_ili_nista/summary.json`, bez ponovnog izračuna. Backtest ne ovisi o datumu ciljanog izvlačenja, nego o konfiguraciji, povijesnim podacima i kodu koji ga računa; sve troje je provjereno nepromijenjeno (SHA-256 konfiguracije `56c1a7c5ad032d0f…`, koda `503ebf46cd036767…`, i povijesnih datoteka). Pokretanje koje ga je izračunalo navedeno je u `manifest.json` (`backtests.*.reused_from`).

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

## Backtest: loto

Nije pokrenut: nije priložena datoteka s povijesnim izvlačenjima. Bez povijesnih izvlačenja nema evaluacije; kombinacije za ovu igru nisu testirane ni na čemu.

## Backtest: joker

Nije pokrenut: nije priložena datoteka s povijesnim izvlačenjima. Bez povijesnih izvlačenja nema evaluacije; kombinacije za ovu igru nisu testirane ni na čemu.

## Zaključak

* sve_ili_nista/main: nijedna metoda, ensemble ni statistička kontrola ne odstupa od uniformne slučajne osnovice nakon korekcije za višestruko testiranje. Opaženo je kompatibilno s time da su sve kombinacije jednako (ne)vjerojatne.
* eurojackpot: backtest nije bio moguć (nije priložena datoteka s povijesnim izvlačenjima); o uspješnosti se ne može reći ništa.
* loto: backtest nije bio moguć (nije priložena datoteka s povijesnim izvlačenjima); o uspješnosti se ne može reći ništa.
* joker: backtest nije bio moguć (nije priložena datoteka s povijesnim izvlačenjima); o uspješnosti se ne može reći ništa.

Detalji, izvori i ograničenja: `README.md`. Svi međurezultati su u direktoriju s rezultatima ovog pokretanja (`results/draw_2026-10-25_loto_plus17`): `final/combinations.json`, `final/intermediate/*.json`; backtest ovog pokretanja je ponovno korišten iz `results/backtest/sve_ili_nista/summary.json`.
