# Neovisna statistička analiza — Loto 6/45 (HL), 882 izvlačenja 07.01.2018. – 09.08.2026.

*Skripta `stats_neovisno.py`, podaci `loto_6_45_2018_2026.csv`, seed 20261029, 1000 simuliranih povijesti iste veličine za Monte-Carlo (MC) p-vrijednosti. Analiza je rađena neovisno o `stats.py` — sve je izračunato iznova iz CSV-a.*

## Sažetak

Frekvencije 45 brojeva u 882 kola ne odstupaju od uniformnosti (χ²=28.3, 44 d.f., p=0.969; korigirano za izvlačenje bez vraćanja p=0.913, Monte-Carlo p=0.906), a ni prozori od 60/100/200 kola ne pokazuju značajne 'vruće' anomalije. Najčešći broj 35 (137×) i najrjeđi 7 (100×) su unutar raspona koji se očekuje kao ekstrem od 45 slučajnih brojača (MC p=0.93); najdulje kašnjenje 38 kola (broj 36) također je uobičajeno (MC p=0.17). Najčešći par pojavio se 25× (oček. 13.4), što je tipičan maksimum među 990 parova (MC p=0.92). Profil kola (zbroj, parnost, uzastopni, raspon) i ponavljanja iz prethodnog kola (0.79 vs. 0.8) slažu se s teorijom – jedino je oblik distribucije ponavljanja nominalno p≈0.048 (MC p=0.043), što je pri ~40 provedenih testova očekivani broj lažnih signala. Backtest na 782 kola pokazuje da vrući, hladni, najkašnjeniji i Pareto brojevi pogađaju u prosjeku 1.60/1.61/1.55 (oček. 1.6) odnosno 1.19 (oček. 1.2) brojeva po kolu, sve unutar slučajne varijacije; nijedna od tih strategija nema prediktivnu moć u ovim podacima – izvlačenja se ponašaju kao neovisna uniformna slučajna izvlačenja.

## 1. Frekvencije brojeva 1–45

Očekivana frekvencija po broju: 882·6/45 = **117.6**; binomna sd = √(882·6/45·39/45) = 10.10. z = (opaženo − 117.6)/sd. p_binom je dvostrani binomni test pojedinog broja (bez korekcije za 45 usporedbi). Kašnjenje = broj kola od zadnjeg pojavljivanja (0 = izvučen u zadnjem kolu 09.08.2026.).

| broj | glavni | rang | z | p_binom | bonus | ukupno (7) | kašnjenje | zadnji put |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 115 | 26 | -0.26 | 0.843 | 23 | 138 | 4 | 2026-07-26 |
| 2 | 115 | 27 | -0.26 | 0.843 | 21 | 136 | 3 | 2026-07-30 |
| 3 | 126 | 9 | +0.83 | 0.400 | 13 | 139 | 13 | 2026-06-25 |
| 4 | 125 | 10 | +0.73 | 0.458 | 22 | 147 | 1 | 2026-08-06 |
| 5 | 112 | 30 | -0.55 | 0.620 | 26 | 138 | 5 | 2026-07-23 |
| 6 | 119 | 19 | +0.14 | 0.882 | 19 | 138 | 9 | 2026-07-09 |
| 7 | 100 | 45 | -1.74 | 0.083 | 22 | 122 | 0 | 2026-08-09 |
| 8 | 120 | 18 | +0.24 | 0.804 | 17 | 137 | 3 | 2026-07-30 |
| 9 | 110 | 36 | -0.75 | 0.488 | 16 | 126 | 4 | 2026-07-26 |
| 10 | 108 | 40 | -0.95 | 0.373 | 17 | 125 | 1 | 2026-08-06 |
| 11 | 124 | 12 | +0.63 | 0.520 | 18 | 142 | 1 | 2026-08-06 |
| 12 | 125 | 11 | +0.73 | 0.458 | 25 | 150 | 5 | 2026-07-23 |
| 13 | 128 | 6 | +1.03 | 0.298 | 19 | 147 | 3 | 2026-07-30 |
| 14 | 109 | 39 | -0.85 | 0.428 | 20 | 129 | 4 | 2026-07-26 |
| 15 | 132 | 2 | +1.43 | 0.151 | 23 | 155 | 6 | 2026-07-19 |
| 16 | 115 | 28 | -0.26 | 0.843 | 22 | 137 | 5 | 2026-07-23 |
| 17 | 114 | 29 | -0.36 | 0.766 | 17 | 131 | 0 | 2026-08-09 |
| 18 | 117 | 22 | -0.06 | 1.000 | 21 | 138 | 2 | 2026-08-02 |
| 19 | 127 | 7 | +0.93 | 0.347 | 20 | 147 | 2 | 2026-08-02 |
| 20 | 117 | 23 | -0.06 | 1.000 | 27 | 144 | 0 | 2026-08-09 |
| 21 | 123 | 14 | +0.53 | 0.586 | 16 | 139 | 11 | 2026-07-02 |
| 22 | 107 | 41 | -1.05 | 0.322 | 20 | 127 | 6 | 2026-07-19 |
| 23 | 121 | 17 | +0.34 | 0.729 | 23 | 144 | 12 | 2026-06-28 |
| 24 | 131 | 3 | +1.33 | 0.181 | 22 | 153 | 1 | 2026-08-06 |
| 25 | 116 | 24 | -0.16 | 0.921 | 19 | 135 | 13 | 2026-06-25 |
| 26 | 123 | 15 | +0.53 | 0.586 | 14 | 137 | 2 | 2026-08-02 |
| 27 | 106 | 42 | -1.15 | 0.276 | 20 | 126 | 0 | 2026-08-09 |
| 28 | 116 | 25 | -0.16 | 0.921 | 14 | 130 | 3 | 2026-07-30 |
| 29 | 110 | 37 | -0.75 | 0.488 | 19 | 129 | 1 | 2026-08-06 |
| 30 | 111 | 31 | -0.65 | 0.552 | 16 | 127 | 7 | 2026-07-16 |
| 31 | 123 | 16 | +0.53 | 0.586 | 18 | 141 | 4 | 2026-07-26 |
| 32 | 124 | 13 | +0.63 | 0.520 | 19 | 143 | 7 | 2026-07-16 |
| 33 | 111 | 32 | -0.65 | 0.552 | 17 | 128 | 9 | 2026-07-09 |
| 34 | 127 | 8 | +0.93 | 0.347 | 25 | 152 | 4 | 2026-07-26 |
| 35 | 137 | 1 | +1.92 | 0.060 | 21 | 158 | 4 | 2026-07-26 |
| 36 | 111 | 33 | -0.65 | 0.552 | 22 | 133 | 38 | 2026-03-29 |
| 37 | 130 | 4 | +1.23 | 0.216 | 26 | 156 | 2 | 2026-08-02 |
| 38 | 101 | 44 | -1.64 | 0.102 | 25 | 126 | 19 | 2026-06-04 |
| 39 | 130 | 5 | +1.23 | 0.216 | 12 | 142 | 9 | 2026-07-09 |
| 40 | 111 | 34 | -0.65 | 0.552 | 12 | 123 | 0 | 2026-08-09 |
| 41 | 119 | 20 | +0.14 | 0.882 | 22 | 141 | 0 | 2026-08-09 |
| 42 | 111 | 35 | -0.65 | 0.552 | 21 | 132 | 1 | 2026-08-06 |
| 43 | 106 | 43 | -1.15 | 0.276 | 18 | 124 | 3 | 2026-07-30 |
| 44 | 119 | 21 | +0.14 | 0.882 | 16 | 135 | 15 | 2026-06-18 |
| 45 | 110 | 38 | -0.75 | 0.488 | 17 | 127 | 35 | 2026-04-09 |

### Hi-kvadrat test uniformnosti (44 d.f.)

Klasična statistika Σ(O−E)²/E s χ²(44). Napomena: budući da se 6 brojeva izvlači **bez vraćanja**, varijanca brojača je manja nego u multinomnom modelu za faktor (45−6)/(45−1) = 0.8864; očekivanje statistike pod H0 je 39, a ne 44, pa je klasični test blago konzervativan. Stoga su dane i korigirana statistika (χ²·44/39) i MC p-vrijednost iz 1000 simulacija.

| uzorak | n | E po broju | χ² | p (χ²₄₄) | χ² korig. | p korig. | MC p | E[χ²] pod H0 | najčešći | najrjeđi |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|:--|:--|
| cijela povijest | 882 | 117.60 | 28.26 | 0.9686 | 31.88 | 0.9134 | 0.9060 | 39.0 | 35 (137×) | 7 (100×) |
| zadnjih 60 | 60 | 8.00 | 30.75 | 0.9348 | 34.69 | 0.8413 | 0.8290 | 39.0 | 4 (16×) | 36 (3×) |
| zadnjih 100 | 100 | 13.33 | 38.25 | 0.7156 | 43.15 | 0.5078 | 0.5130 | 39.0 | 32 (20×) | 21 (6×) |
| zadnjih 200 | 200 | 26.67 | 35.33 | 0.8215 | 39.85 | 0.6499 | 0.6590 | 39.0 | 16 (40×) | 7 (17×) |

Dopunski (bonus) broj: E = 882/45 = 19.6 po broju, χ² = 31.57, p = 0.9196 (najčešći 20 27×, najrjeđi 39 12×). Svih 7 kugli zajedno: χ² = 29.47, p = 0.9544.

**"Vrući" prozori** (top-5 / dno-5 po frekvenciji u zadnjih w kola):

- zadnjih 60: top-5 = [4, 8, 13, 23, 34] ([16, 12, 11, 11, 11]), dno-5 = [43, 10, 29, 36, 45] ([6, 5, 4, 3, 3]); E = 8.00, MC p = 0.8290
- zadnjih 100: top-5 = [32, 31, 4, 5, 16] ([20, 19, 18, 18, 18]), dno-5 = [29, 45, 21, 36, 38] ([8, 8, 6, 6, 6]); E = 13.33, MC p = 0.5130
- zadnjih 200: top-5 = [16, 31, 34, 35, 44] ([40, 35, 34, 34, 34]), dno-5 = [40, 10, 21, 7, 38] ([22, 21, 19, 17, 17]); E = 26.67, MC p = 0.6590

### Ekstremi, Pareto i hladni skup

- Najveći |z| = 1.92 (broj 35); u simulacijama je maksimalni |z| od 45 brojeva u prosjeku 2.49, a vrijednost ≥ opažene javlja se u 93 % slučajnih povijesti. Brojeva s |z| > 1.96: 0 (očekivano ≈ 2.25). Najmanji p_binom = 0.0596, Bonferroni (×45) = 1.000.
- **Pareto top-9 (20 % brojeva)**: [35, 15, 24, 37, 39, 13, 19, 34, 3] s frekvencijama [137, 132, 131, 130, 130, 128, 127, 127, 126]; nose 22.1 % svih izvučenih brojeva (uniformno bi bilo 20 %). Granica: 9. ima 126, 10. ima 125.
- **Hladnih 9**: [7, 38, 27, 43, 22, 10, 14, 9, 29] s frekvencijama [100, 101, 106, 106, 107, 108, 109, 110, 110]. Granica: 9. ima 110, 10. ima 110 — izjednačeni na granici: [9, 29, 45].

### Kašnjenje

Očekivani razmak između dva pojavljivanja istog broja je 45/6 = **7.5** kola; očekivano *trenutno* kašnjenje (kola od zadnjeg pojavljivanja) je 7.5 − 1 = 6.5 (sd 7.0), P(kašnjenje ≥ g) = (39/45)^g. Opaženi prosjek trenutnog kašnjenja svih 45 brojeva: 6.16 (medijan 4); prosječni povijesni razmak između pojavljivanja: 7.48 kola.

**Najdulje trenutno kašnjenje: broj 36 — 38 kola**. P(zadani broj kasni ≥ 38) = 0.0043; P(maksimum među 45 brojeva ≥ 38) ≈ 0.18 (aproks.), MC = 0.17; u slučajnim povijestima najdulje kašnjenje je u prosjeku 30.5 kola (medijan 29).

Top-10 kašnjenja: **36** (38), **45** (35), **38** (19), **44** (15), **3** (13), **25** (13), **23** (12), **21** (11), **6** (9), **33** (9).

## 2. Parovi

990 mogućih parova, 15 po kolu; očekivano po paru 882·15/990 = **13.36** (sd 3.63). Parova koji se nisu pojavili nijednom: 0; najrjeđi par [1, 25] (4×).

| # | par | broj | z | P(≥ broj) za zadani par |
|---:|:--|---:|---:|---:|
| 1 | 13–21 | 25 | +3.21 | 0.0026 |
| 2 | 35–43 | 25 | +3.21 | 0.0026 |
| 3 | 4–21 | 24 | +2.93 | 0.0052 |
| 4 | 4–37 | 24 | +2.93 | 0.0052 |
| 5 | 34–35 | 24 | +2.93 | 0.0052 |
| 6 | 34–37 | 24 | +2.93 | 0.0052 |
| 7 | 1–34 | 23 | +2.66 | 0.0098 |
| 8 | 3–15 | 23 | +2.66 | 0.0098 |
| 9 | 3–37 | 23 | +2.66 | 0.0098 |
| 10 | 4–35 | 23 | +2.66 | 0.0098 |
| 11 | 3–26 | 22 | +2.38 | 0.0177 |
| 12 | 11–31 | 22 | +2.38 | 0.0177 |
| 13 | 13–39 | 22 | +2.38 | 0.0177 |
| 14 | 15–39 | 22 | +2.38 | 0.0177 |
| 15 | 35–37 | 22 | +2.38 | 0.0177 |
| 16 | 4–6 | 21 | +2.10 | 0.0310 |
| 17 | 6–23 | 21 | +2.10 | 0.0310 |
| 18 | 6–44 | 21 | +2.10 | 0.0310 |
| 19 | 8–12 | 21 | +2.10 | 0.0310 |
| 20 | 11–37 | 21 | +2.10 | 0.0310 |

**Je li maksimum (25×) neobičan?** Poissonova aproksimacija: P(zadani par ≥ 25) = 0.00283 (binomno 0.00262); P(maksimum od 990 parova ≥ 25) ≈ 1 − (1 − p)^990 = 0.940. Simulacija 1000 slučajnih povijesti (882 kola, seed 20261029): maksimum po paru u prosjeku 26.5 (medijan 26, 95. percentil 29, raspon 23–33); udio simulacija s maksimumom ≥ 25: **MC p = 0.917**.

Hi-kvadrat svih 990 parova: χ² = 871.3 (naivno 989 d.f., p = 0.9970; parovi unutar kola su zavisni pa je mjerodavna MC p-vrijednost = 0.9690, simulirani prosjek χ² = 974.6).

## 3. Profil izvlačenja

Teorijske vrijednosti su **egzaktne** — izračunate nabrajanjem svih 8 145 060 kombinacija. z/p se odnose na test prosjeka (z-test), GOF je hi-kvadrat test prilagodbe cijele distribucije (kategorije s E < 5 spojene; za zbroj i raspon decilni razredi).

| statistika | prosjek | teorijski | sd | teor. sd | z | p (prosjek) | GOF χ² (df) | GOF p |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|
| zbroj 6 brojeva | 137.739 | 138.000 | 30.24 | 29.95 | -0.26 | 0.7960 | 6.60 (9) | 0.6791 |
| broj neparnih | 3.095 | 3.067 | 1.17 | 1.15 | +0.74 | 0.4617 | 3.00 (6) | 0.8092 |
| brojeva ≤ 22 | 2.934 | 2.933 | 1.18 | 1.15 | +0.02 | 0.9814 | 6.22 (6) | 0.3991 |
| brojeva ≤ 31 | 4.133 | 4.133 | 1.06 | 1.07 | -0.02 | 0.9849 | 9.04 (5) | 0.1074 |
| broj uzastopnih parova (n_{i+1}−n_i = 1) | 0.678 | 0.667 | 0.74 | 0.72 | +0.46 | 0.6422 | 1.32 (3) | 0.7254 |
| raspon max−min | 32.661 | 32.857 | 6.75 | 6.77 | -0.86 | 0.3892 | 6.03 (9) | 0.7373 |
| broj različitih desetica (1–10, 11–20, 21–30, 31–40, 41–45) | 3.702 | 3.732 | 0.71 | 0.71 | -1.25 | 0.2123 | 3.85 (3) | 0.2785 |

Percentili (opaženo / teorijski):

| statistika | P5 | P10 | P25 | P50 | P75 | P90 | P95 |
|:--|---:|---:|---:|---:|---:|---:|---:|
| zbroj | 90 / 89 | 98 / 99 | 116 / 117 | 137 / 138 | 159 / 159 | 176 / 177 | 189 / 187 |
| raspon | 20 / 20 | 23 / 23 | 28 / 28 | 33 / 34 | 38 / 38 | 41 / 41 | 42 / 42 |
| neparni | 1 / 1 | 2 / 2 | 2 / 2 | 3 / 3 | 4 / 4 | 5 / 5 | 5 / 5 |
| le22 | 1 / 1 | 1 / 1 | 2 / 2 | 3 / 3 | 4 / 4 | 4 / 4 | 5 / 5 |
| le31 | 2 / 2 | 3 / 3 | 3 / 3 | 4 / 4 | 5 / 5 | 5 / 5 | 6 / 6 |
| uzastopni | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 | 2 / 2 |
| desetice | 3 / 3 | 3 / 3 | 3 / 3 | 4 / 4 | 4 / 4 | 5 / 5 | 5 / 5 |

- broj neparnih: 0→6 (E 8.1), 1→67 (E 65.6), 2→204 (E 200.4), 3→280 (E 295.3), 4→222 (E 221.5), 5→91 (E 80.2), 6→12 (E 10.9)
- brojeva ≤ 22: 0→16 (E 10.9), 1→81 (E 80.2), 2→219 (E 221.5), 3→274 (E 295.3), 4→219 (E 200.4), 5→67 (E 65.6), 6→6 (E 8.1)
- broj uzastopnih parova (n_{i+1}−n_i = 1): 0→416 (E 415.6), 1→346 (E 356.3), 2→109 (E 99.0), 3→10 (E 10.7), 4→1 (E 0.4), 5→0 (E 0.0)
- broj različitih desetica (1–10, 11–20, 21–30, 31–40, 41–45): 1→0 (E 0.1), 2→34 (E 27.0), 3→289 (E 290.7), 4→465 (E 455.9), 5→94 (E 108.3)
- P(bar jedan uzastopni par) = 1 − C(40,6)/C(45,6) = **0.5287**; opaženo 466/882 = 0.5283 (E 466.4), binomni p = 1.000.
- Sve parne ili sve neparne: opaženo 18, očekivano 19.0.

## 4. Ponavljanja, autokorelacija, serije

Broj brojeva iz prethodnog kola koji se ponove u sljedećem ~ hipergeometrijski(45, 6, 6): E = **0.80**, sd 0.784. Opaženi prosjek 0.795 (sd 0.734), z = -0.21, p = 0.8366; GOF χ² = 7.93 (3 df), p = 0.0475, MC p = 0.0430 — jedini nominalno p<0.05 rezultat među ~40 testova; prosjek je točno očekivani, odstupa samo oblik (više 'točno 1', manje 0 i 3+). Kola s bar jednim ponavljanjem: 62.2 % (teorijski 59.9 %).

| ponovljenih | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|:--|---:|---:|---:|---:|---:|---:|---:|
| opaženo | 333 | 409 | 126 | 13 | 0 | 0 | 0 |
| očekivano | 352.9 | 373.7 | 133.4 | 19.8 | 1.2 | 0.0 | 0.0 |

Po pomaku L (zajednički brojevi s kolom t−L): L=1: 0.795 (z -0.21); L=2: 0.833 (z +1.25); L=3: 0.821 (z +0.81); L=4: 0.777 (z -0.88); L=5: 0.840 (z +1.52).
Dopunski broj prethodnog kola među 6 glavnih sljedećeg: 131 puta (E 117.5, p = 0.181).

**Autokorelacija** (lag 1–10, granica ±1.96/√882 = ±0.066; Ljung–Box Q(10)):

- zbroj: r₁..r₅ = -0.024, +0.006, +0.022, -0.026, +0.066; Q(10) = 9.26, p = 0.5077
- neparni: r₁..r₅ = -0.039, -0.016, +0.032, -0.071, +0.046; Q(10) = 15.54, p = 0.1136
- le22: r₁..r₅ = -0.018, -0.010, +0.020, -0.023, -0.010; Q(10) = 7.11, p = 0.7152
- raspon: r₁..r₅ = +0.030, -0.060, +0.001, +0.014, -0.009; Q(10) = 11.33, p = 0.3320
- indikatorske serije 45 brojeva, lag 1: prosječni r₁ = -0.0017 (raspon -0.093…+0.067); značajnih na 5 %: 2 (E 2.25); zbirni test Σz² = 42.7 (45 df), p = 0.5697

**Test serija (Wald–Wolfowitz runs):**

| serija | n1 | n2 | serija (runs) | očekivano | z | p |
|:--|---:|---:|---:|---:|---:|---:|
| zbroj iznad medijana | 437 | 445 | 447 | 442.0 | +0.34 | 0.7343 |
| ≥3 neparna | 605 | 277 | 399 | 381.0 | +1.41 | 0.1595 |
| bar jedno ponavljanje | 548 | 333 | 410 | 415.3 | -0.38 | 0.7058 |

Runs test za indikatorsku seriju svakog od 45 brojeva: značajnih na 5 %: 2 (E 2.25), prosječni z = +0.014, Σz² = 42.9 (45 df), p = 0.5621; najjači: 20 (z +2.64), 44 (z -2.01), 12 (z +1.58).

## 5. KLJUČNO — backtest prediktivne moći

Za svako kolo t ≥ 100 (0-bazirano; T = 782 kola) skupovi se biraju iz podataka prije kola t: vrući = top-12 po frekvenciji u prethodnih w kola, hladni = dno-12, kasneći = 12 s najvećim kašnjenjem, pareto9 = top-9 po frekvenciji u cijeloj povijesti do t, hladni9_povijest = dno-9; slucajni12 = kontrolni slučajni skup. Izjednačenja se razbijaju slučajno (seed 20261029).

Očekivani broj pogodaka po kolu pod slučajnošću: 6·12/45 = **1.6** (sd po kolu 1.020, hipergeometrijski) odnosno 6·9/45 = **1.2** (sd 0.922). p_z = dvostrani z-test ukupnog broja pogodaka; p_binom = dvostrani binomni test (6T pokušaja, p = k/45); MC p = udio od 1000 simuliranih povijesti u kojima isti algoritam daje prosjek koji od očekivanja odstupa bar koliko opaženi. Snaga: pri T = 782 dvostrani test na razini 5 % otkriva odstupanje prosjeka od ±0.071 pogodaka po kolu (±4.5 % od 1.6) – svaki stvarno koristan 'vrući/hladni' efekt tog reda bio bi vidljiv.

| skup | k | prosjek | sd | očekivano | lift | ukupno | z | p_z | p_binom | MC p | MC sd prosjeka | kola s 0 | kola ≥3 |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vruci_w60 | 12 | **1.601** | 1.018 | 1.6 | 1.001 | 1252 | +0.03 | 0.9776 | 0.9737 | 0.9840 | 0.038 | 12.9 % | 17.0 % |
| hladni_w60 | 12 | **1.611** | 1.036 | 1.6 | 1.007 | 1260 | +0.31 | 0.7576 | 0.7664 | 0.7650 | 0.037 | 12.8 % | 18.4 % |
| kasneci | 12 | **1.550** | 0.978 | 1.6 | 0.969 | 1212 | -1.37 | 0.1693 | 0.1979 | 0.1590 | 0.036 | 14.3 % | 15.6 % |
| pareto9 | 9 | **1.193** | 0.967 | 1.2 | 0.994 | 933 | -0.21 | 0.8342 | 0.8552 | 0.8640 | 0.033 | 26.0 % | 9.3 % |
| hladni9_povijest | 9 | **1.243** | 0.949 | 1.2 | 1.036 | 972 | +1.30 | 0.1927 | 0.2215 | 0.2000 | 0.034 | 22.9 % | 9.6 % |
| slucajni12 | 12 | **1.652** | 1.037 | 1.6 | 1.033 | 1292 | +1.43 | 0.1525 | 0.1812 | 0.1340 | 0.035 | 13.0 % | 20.2 % |
| vruci_w30 | 12 | **1.632** | 1.015 | 1.6 | 1.020 | 1276 | +0.87 | 0.3845 | 0.4186 | 0.4280 | 0.038 | 12.3 % | 18.5 % |
| hladni_w30 | 12 | **1.610** | 0.969 | 1.6 | 1.006 | 1259 | +0.27 | 0.7845 | 0.7917 | 0.7940 | 0.038 | 11.9 % | 17.4 % |
| vruci_w100 | 12 | **1.625** | 1.024 | 1.6 | 1.016 | 1271 | +0.69 | 0.4875 | 0.5197 | 0.4980 | 0.036 | 13.3 % | 17.9 % |
| hladni_w100 | 12 | **1.575** | 1.025 | 1.6 | 0.985 | 1232 | -0.67 | 0.5008 | 0.5414 | 0.5100 | 0.036 | 14.1 % | 18.2 % |
| vruci_w200 | 12 | **1.595** | 1.077 | 1.6 | 0.997 | 1247 | -0.15 | 0.8829 | 0.9080 | 0.8990 | 0.037 | 15.0 % | 20.1 % |
| hladni_w200 | 12 | **1.664** | 1.055 | 1.6 | 1.040 | 1301 | +1.75 | 0.0808 | 0.1022 | 0.1520 | 0.036 | 14.5 % | 21.2 % |

Distribucija pogodaka po kolu (broj kola s 0,1,…,6 pogodaka):

- vruci_w60: 0→101, 1→279, 2→269, 3→101, 4→29, 5→2, 6→1
- hladni_w60: 0→100, 1→287, 2→251, 3→109, 4→31, 5→4
- kasneci: 0→112, 1→271, 2→277, 3→102, 4→19, 5→1
- pareto9: 0→203, 1→311, 2→195, 3→62, 4→9, 5→2
- slucajni12: 0→102, 1→258, 2→264, 3→133, 4→18, 5→7

Osjetljivost na razbijanje izjednačenja (w = 60, po rastućem broju umjesto slučajno): vruci_w60_tiebreak_asc: 1.611 (z +0.31, p 0.758); hladni_w60_tiebreak_asc: 1.592 (z -0.22, p 0.828).

Po godinama (prosjek pogodaka po kolu):

| godina | n | vrući w60 | hladni w60 | kasneći | pareto9 |
|:--|---:|---:|---:|---:|---:|
| 2019 | 95 | 1.758 | 1.463 | 1.411 | 1.347 |
| 2020 | 105 | 1.590 | 1.600 | 1.590 | 1.171 |
| 2021 | 104 | 1.538 | 1.644 | 1.606 | 1.221 |
| 2022 | 104 | 1.635 | 1.577 | 1.606 | 1.154 |
| 2023 | 103 | 1.534 | 1.777 | 1.524 | 1.000 |
| 2024 | 103 | 1.631 | 1.689 | 1.485 | 1.320 |
| 2025 | 104 | 1.702 | 1.558 | 1.596 | 1.125 |
| 2026 | 64 | 1.328 | 1.547 | 1.578 | 1.234 |

**Zaključak backtesta.** T=782 kola (t≥100). Vrući top-12 (w=60): prosjek 1.601 pogodaka/kolo (oček. 1.6; z=+0.03, p=0.978, binom p=0.974, MC p=0.984). Hladni dno-12 (w=60): 1.611 (z=+0.31, p=0.758). Najkašnjeniji 12: 1.550 (z=-1.37, p=0.169). Pareto top-9 (cijela povijest do t): 1.193 (oček. 1.2; z=-0.21, p=0.834). Slučajni kontrolni skup 12: 1.652 (z=+1.43) – bolji od svih 'strategija', što pokazuje razinu šuma. Detektabilno odstupanje pri T=782 (z=1.96): ±0.071 pogodaka/kolo (±4.5 %). Ostali prozori (w=30/100/200) vrući: 1.632/1.625/1.595; hladni: 1.610/1.575/1.664.

## 6. Joker (kratko)

Joker je 6-znamenkasti broj (vodeće nule sačuvane; vodeća nula u 95 kola, E 88.2). Frekvencije znamenki 0–9 na svih 6 pozicija: [540, 541, 523, 577, 512, 578, 509, 500, 503, 509] (E 529.2); χ² = 14.38 (9 df), p = 0.1093. Po poziciji: poz. 1: χ² 7.6, p 0.573; poz. 2: χ² 4.2, p 0.895; poz. 3: χ² 10.9, p 0.282; poz. 4: χ² 9.2, p 0.415; poz. 5: χ² 1.8, p 0.994; poz. 6: χ² 7.1, p 0.623. Cijeli broj kao U(0, 999999): KS D = 0.028, p = 0.463; prosjek 493540 (E 499999.5); Ljung–Box Q(5) = 1.60, p = 0.902. Znamenke su uniformne i bez serijske ovisnosti.

---
*Generirano 2026-10-26 13:53:23, trajanje 15.1 s. Datoteke: `out/stats_neovisno.json`, `out/stats_neovisno.md`.*
