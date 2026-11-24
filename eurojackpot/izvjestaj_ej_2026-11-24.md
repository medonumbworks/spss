# Eurojackpot (5/50 + 2/12) – 6 kombinacija za uto 24.11.2026.

## 1. Kombinacije

| # | Tip | Sustav/metoda | 5 glavnih | 2 euro | zbroj | neparni | redoslijed |
|---|---|---|---|---|---|---|---|
| E1 | ezoterijska | Plečkov filtar (11 sustava) | 1 4 11 16 30 | 6 11 | 62 | 2 | 1 |
| E2 | ezoterijska | Istok/Zapad naizmjence: Meihua, zapadna, I Ching, kamea Marsa, Jyotiš, kabala | 6 21 27 30 44 | 3 8 | 128 | 2 | 4 |
| S1 | statistička | Pareto 20/20 (statistički tipična) | 4 17 21 34 37 | 5 7 | 113 | 3 | 5 |
| S2 | statistička | vrući (zadnjih 100) (statistički tipična) | 5 17 30 35 42 | 1 4 | 129 | 3 | 6 |
| L1 | logička | najveća točna P(barem jedan dobitak) | 12 19 28 33 46 | 2 9 | 138 | 2 | 2 |
| L2 | logička | najveća točna P(barem jedan dobitak) | 8 13 31 38 48 | 10 12 | 138 | 2 | 3 |

## 2. Provjere i vjerojatnosti

- Susjedni glavni brojevi: nema.
- Zajednički glavni brojevi: najviše 1 po paru (E1-E2 1, E1-S1 1, E1-S2 1, E2-S1 1, E2-S2 1, S1-S2 1; ostali parovi 0).
- Euro parovi pokrivaju 1–12 točno jednom: da. Svih 6 listića je različito, pa je 6 različitih ishoda 5+2. Raspon glavnih brojeva 1–48, ukupno 25 različitih.
- Točno, preko svih 2.118.760 × 66 = 139.838.160 ishoda (numpy, bitmaske):
  - P(5+2) = 6/139.838.160 = 1 : 23.306.360
  - **P(barem jedan dobitak) = 25.458.267/139.838.160 = 18,206 %**
  - P(barem 3 glavna pogotka u nekoj kombinaciji) = 60540/2.118.760 = 2,857 %
  - očekivani broj dobitnih kombinacija = 399381/2118760 = 0,1885 (svaka kombinacija: P(dobitak) = 133127/4237520 = 3,142 % ≈ 1 : 31,8)
  - samo E1, E2, S1, S2: 12,078 %
- 1000 slučajnih setova od 6 listića (bez ograničenja, bez duplikata), također točno: P(barem jedan dobitak) prosjek 17,427 %, sd 0,373 pp, raspon 15,594 %–18,241 % → ovaj set **99,7. percentil**; P(≥3 glavna) prosjek 2,834 % → 81,8. percentil.
- L1–L2: uz zadanu podjelu euro brojeva svaki par L-listića bez zajedničkih glavnih brojeva s ostalima daje isti maksimum (25.458.267), zbog simetrije. Isti maksimum dalo je svih 8 greedy pokretanja s lokalnom zamjenom. Konkretni brojevi L1–L2 zato su jedan od jednako dobrih izbora, dobiven slučajnim kandidatima iz `secrets`.

**Redoslijed igranja** (pohlepno prema doprinosu P(barem jedan dobitak); cijena kombinacije c nije potvrđena):

| k | kombinacije | ulog | P(barem jedan dobitak) | P(≥3 glavna) | najbolji podskup od k |
|---|---|---|---|---|---|
| 1 | E1 | 1 × c | 3,142 % | 0,478 % | isti |
| 2 | E1, L1 | 2 × c | 6,266 % | 0,956 % | isti |
| 3 | E1, L1, L2 | 3 × c | 9,373 % | 1,434 % | isti |
| 4 | E1, L1, L2, E2 | 4 × c | 12,397 % | 1,910 % | isti |
| 5 | E1, L1, L2, E2, S1 | 5 × c | 15,341 % | 2,385 % | isti |
| 6 | E1, L1, L2, E2, S1, S2 | 6 × c | 18,206 % | 2,857 % | isti |

- k = 1: svih 6 listića ima istu P (3,142 %) i isti najbolji par, pa odlučuje redoslijed u tablici (E1).
- Za svaki k prvih k je ujedno točno najbolji podskup (provjereni su svi podskupovi).

## 3. Kandidati po sustavu

| Sustav | prvih 10 glavnih | euro (prva 4) | ključni podaci |
|---|---|---|---|
| Sinkronicitet | 15 35 7 44 5 24 11 1 28 31 | 5 11 3 7 | glavni ≥2×: [15, 35, 7, 44, 5]; euro ≥2×: [5, 11]; redni dan 328→28; starost 35; zadnje izvlačenje [15, 28, 29, 34, 35] |
| Arhetipovi | 4 12 1 16 7 10 28 22 40 36 | 4 12 1 7 | zadani popis |
| I Ching | 21 39 3 4 | 9 3 4 | bacanja [7, 8, 8, 7, 8, 7] (odozdo); heksagram 21 (Li/Zhen) → 21; nuklearni 39; promjenjive crte nema |
| Meihua | 44 6 1 3 5 | 8 6 1 3 | Y=7 (午), M=10, D=16, H=12 (亥); gornji 1 Qian, donji 5 Xun, crta 3; heksagram 44 → 6, nuklearni 1 |
| Jyotiš | 20 4 16 9 23 | 8 4 9 11 | Mjesec 40.68° sid. → nakšatra 4 Rohini (65 min od ulaska, 1210 min do izlaska); tithi 16 Krishna Pratipada (246/983 min); yoga 20 Shiva (503/691 min); Mars=9; srednji Rahu 300.58° → nakšatra 23 Dhanishta; ayanamsa 24.2328° |
| Zapadna astrologija | 30 27 26 14 3 5 10 15 | 6 3 2 5 | Mars Lav 29.52° → 30; Jupiter Lav 26.49° → 27; Venera Vaga 25.04° → 26; Merkur Škorpion 13.54° → 14; Sunce Strijelac 2.51° → 3; Mjesec Blizanci 4.91° → 5 |
| Kamea Marsa | 11 12 13 14 15 3 8 18 23 5 | 11 12 1 2 | provjereno u kodu: 5 redova, 5 stupaca, 2 dijagonale = 65, brojevi 1–25 jednom; glavna [11, 12, 13, 14, 15], sporedna [3, 8, 13, 18, 23] |
| Kabala | 45 5 17 27 1 26 18 10 | 11 5 7 3 | Maadim 95→45 (euro 11), Gevura 5, IVAN 67→17, mazal 77→27, LOTO 51→1, 26, 18, 10 |
| Tarot | 30 49 1 4 25 5 22 29 31 20 | 6 1 3 4 | Sunce Strijelac 2.51°, 1. dekan → Štapovi 8 (30); izvučeno: Pehari Kraljica (49), Mačevi As (51), VA 4 (4), Štapovi 3 (25), VA 5 (5), Pentakli Paž (75), VA 22 (22), Štapovi 7 (29), Štapovi 9 (31), VA 20 (20) |
| Numerologija | 11 29 19 13 31 9 | 11 5 7 1 | osobni dan 9 (znamenke 31.05. + 24.11.2026.), univerzalni dan 9 |
| Planetarni sat | 1 16 4 9 8 40 15 32 | 1 4 9 8 | Helsinki: izlazak 08:40:27, zalazak 15:32:45, sutrašnji izlazak 08:42:45 (EET); 21:00 = 16. sat (4. noćni, 19:50:15–21:16:05), vladar Sunce (=1); 69.7 min od početka, 16.1 min do kraja; 1. dnevni sat = Mars |
| **Plečkov filtar** (11 sustava) | 4 1 16 3 5 11 30 12 44 15 | 11 6 4 5 | glavni broj/glasovi/zbroj pozicija: 4/5/14, 1/5/15, 16/3/9, 3/3/12, 5/3/12, 11/2/2, 30/2/2, 12/2/4, 44/2/5, 15/2/6, 27/2/6, 13/2/7, 7/2/8, 9/2/8, 14/2/8, 20/1/1; euro: 11/4/5, 6/3/4, 4/3/5, 5/3/5, 8/2/2, 1/2/3, 3/2/4, 12/2/4, 9/1/1 |

- **Signal-brojevi (≥ 2 glasa):** glavni 4, 1, 16, 3, 5, 11, 30, 12, 44, 15, 27, 13, 7, 9, 14; euro 11, 6, 4, 5, 8, 1, 3, 12.
- **Slaganje:**
  - E1 preskače 3 i 5 (susjedni broju 4).
  - E2 preskače 11, 20, 45: 11 bi dao 2 zajednička broja s E1, 20 je susjedni broju 21, a 45 broju 44.
  - Euro za E2 su glasovi istočnih i zapadnih sustava; 11 i 6 su preskočeni jer su već u E1.

**Pravilnosti u zadnjih 5 izvlačenja:**
- 06.11.2026: 7 30 32 35 44 + 2 6
- 10.11.2026: 7 15 17 40 44 + 5 11
- 13.11.2026: 5 12 27 35 44 + 4 12
- 17.11.2026: 1 5 7 15 19 + 1 11
- 20.11.2026: 15 28 29 34 35 + 5 9
- Glavni brojevi 3 puta: 15, 35, 7, 44; 2 puta: 5. Euro 2 puta: 5 i 11.
- Neparnih: [2, 3, 3, 5, 3]; zbrojevi: [148, 123, 123, 47, 141].
- Susjedni parovi samo 20.11. (28–29 i 34–35). Ponovljeno iz prethodnog izvlačenja: [1, 2, 1, 1, 1].
- 17.11. je imalo samo dvije desetice (zbroj 47). Ni u jednom od 5 izvlačenja nije bilo 34 od 50 brojeva.

**Lunarni datum** (lokalni datum 24.11.): lunar_python 丙午, mj. 10, dan 16, sat 亥; sxtwl mj. 10, dan 16; HKO mj. 10, dan 16. Sva tri se slažu. U Pekingu je tada 25.11.2026 03:00 = mj. 10, dan 17, sat 寅, što nije korišteno.

**Osjetljivost na vrijeme izvlačenja:**
- Mjesec prelazi u 6° Blizanaca 8,4 min nakon 21:00 EET. On je 6. u zapadnom popisu, pa to ne mijenja E1 ni E2.
- Planetarni sat (Sunce) traje do 21:16:05 EET. Ako izvlačenje kasni više od 16 min, vladar sata postaje Venera i mijenjaju se glasovi.

## 4. Statistika i backtest

Walk-forward od 100. izvlačenja (n = 388; svaka metoda koristi samo izvlačenja prije t; listić = prvih 5 glavnih + prva 2 euro po metodi):

| Metoda | dobitni događaji (oček. 12,2) | razredi | z | p (jednostr.) | p Bonf. (×6) | prosj. glavnih (0,5) | prosj. euro (0,333) | p Bonf. glavni / euro |
|---|---|---|---|---|---|---|---|---|
| vrući (zadnjih 50) | 11 | 2+2: 1, 3+1: 1, 3+0: 1, 1+2: 4, 2+1: 4 | -0,35 | 0,676 | 1,000 | 0,497 | 0,410 | 1,000 / 0,016 |
| vrući (zadnjih 100) | 14 | 3+0: 3, 1+2: 3, 2+1: 8 | 0,53 | 0,337 | 1,000 | 0,515 | 0,332 | 1,000 / 1,000 |
| hladni (cijela povijest) | 10 | 3+0: 2, 1+2: 4, 2+1: 4 | -0,64 | 0,778 | 1,000 | 0,477 | 0,361 | 1,000 / 1,000 |
| prezreli | 12 | 2+2: 1, 3+0: 2, 1+2: 3, 2+1: 6 | -0,06 | 0,562 | 1,000 | 0,482 | 0,348 | 1,000 / 1,000 |
| Pareto 20/20 | 16 | 1+2: 5, 2+1: 11 | 1,11 | 0,166 | 0,997 | 0,505 | 0,353 | 1,000 / 1,000 |
| slučajna (secrets) – kontrola | 10 | 2+2: 1, 3+1: 1, 3+0: 1, 1+2: 1, 2+1: 6 | -0,64 | 0,778 | 1,000 | 0,500 | 0,317 | 1,000 / 1,000 |

- **Nijedna metoda nema više dobitnih događaja od slučaja** (sve p nakon Bonferronija ≥ 0,997).
- Jedini nominalno značajan rezultat su vrući euro brojevi (zadnjih 50): prosjek 0,41 euro pogotka prema 0,333, z = 3,0, p ×6 = 0,016. To je sporedna mjera. Uz korekciju za sve testove (6 metoda × 3 mjere) p ≈ 0,048, dakle na samoj granici. Dobitnih događaja ta metoda nema više (11 prema 12,2 očekivano), a χ² ni Spearman za euro brojeve ne pokazuju odstupanje. Ne smatram to dokazom.
- S1–S2 su zato **„statistički tipične”**: dvije najbolje metode po z (Pareto 20/20 i vrući 100) primijenjene na svih 487 izvlačenja. Od 20 najbolje rangiranih brojeva metode uzeta je najbolje rangirana petorka koja zadovoljava tipičnost (zbroj 112,5–142,5, 2–3 neparna, barem 4 desetice) i §2.
  - S1 (Pareto 20/20): sirovih 5 = [17, 21, 23, 35, 37] → nisu tipični (5 neparnih); euro po rangu metode.
  - S2 (vrući (zadnjih 100)): sirovih 5 = [5, 17, 23, 35, 37] → nisu tipični (5 neparnih); euro po rangu metode.

Testovi slučajnosti (487 izvlačenja):
- χ² glavnih brojeva (ispravljen za izvlačenje bez vraćanja, df 49): 47,50, p = 0,534; frekvencije 34–63, očekivano 48,7.
- χ² euro brojeva (df 11): 14,88, p = 0,188; frekvencije 68–103, očekivano 81,2.
- Stabilnost vrućih brojeva (1. vs 2. polovica, Spearman): glavni ρ = -0,065, p = 0,655; euro ρ = 0,155, p = 0,631.

**Backtest** (6 listića na 487 izvlačenja; kombinacije nisu mijenjane prema njemu):

| # | 4+0 | 3+2 | 2+2 | 3+1 | 3+0 | 1+2 | 2+1 | ukupno |
|---|---|---|---|---|---|---|---|---|
| E1 | 1 | 0 | 0 | 1 | 3 | 3 | 8 | 16 |
| E2 | 0 | 0 | 0 | 2 | 1 | 1 | 12 | 16 |
| S1 | 0 | 1 | 1 | 0 | 3 | 1 | 14 | 20 |
| S2 | 0 | 0 | 0 | 1 | 3 | 2 | 19 | 25 |
| L1 | 0 | 0 | 0 | 0 | 1 | 2 | 4 | 7 |
| L2 | 0 | 0 | 0 | 2 | 1 | 5 | 7 | 15 |

- Viši razredi (5+2, 5+1, 5+0, 4+2, 4+1) nisu se dogodili ni jednom.
- Ukupno 99 dobitnih događaja (očekivano 91,8). Slučajni setovi: prosjek 91,8, sd 9,7 → **77,0. percentil**.
- Izvlačenja s barem jednim dobitkom: 96/487.
- S1–S2 su odabrani na istim podacima, pa njihov backtest nije neovisan.

## 5. Izvori, verzije, RNG, pretpostavke, što nije izračunato

**Izvori**

- **Povijest izvlačenja:** 487 izvlačenja od 25.03.2022. do 20.11.2026. (kola 523–1009), u formatu 5/50 + 2/12.
  - [euro-jackpot.net](https://www.euro-jackpot.net/results-archive-2022), godišnje arhive za 2022.–2026.
  - [LotteryExtreme](https://www.lotteryextreme.com/eurojackpot/results), mjesečne arhive.
  - Svih 487 izvlačenja identično je u oba izvora. Brojevi su u rasponima 1–50 i 1–12, bez duplikata datuma i izvlačenja.
  - Zadnjih 5 izvlačenja iz upita podudara se s oba izvora.
- **Vrijeme izvlačenja:** korišteno je 21:00 EET = 19:00 UTC, kako je zadano. Izvori se ne slažu:
  - [euro-jackpot.net](https://www.euro-jackpot.net/) (EN i HR) navodi „uvijek u 20:00 CET u Helsinkiju”, što je isto što i 21:00 EET.
  - [njemački FAQ iste stranice](https://www.euro-jackpot.net/faq) navodi „gegen 21:00 Uhr MEZ”.
  - [Službena stranica](https://www.eurojackpot.com/faq) tekst učitava JavaScriptom, pa ga nisam mogao pročitati.
- **Lunarni datum:** [HKO tablica 2026](https://www.hko.gov.hk/en/gts/time/calendar/text/files/T2026e.txt): 10. lunarni mjesec počinje 9.11., pa je 24.11. 16. dan. lunar_python i sxtwl daju isto.
- **Efemeride:** Swiss Ephemeris 2.10.03 s datotekama sepl_18.se1 i semo_18.se1 iz [aloistr/swisseph](https://github.com/aloistr/swisseph). Za svaki položaj provjereno je da su datoteke korištene (FLG_SWIEPH). Ayanamsa Lahiri 24,2328°.
- **Kamea Marsa:** kvadrat iz upita; u kodu su provjereni zbroj 65 i brojevi 1–25. Agrippin izvorni tekst nije dohvaćen.
- **Mreža:** pristup stranicama s rezultatima, HL-u, službenoj stranici i HKO-u omogućen je u novom okruženju „spss” (uz tvoje odobrenje).

**Verzije:** Python 3.13.18, pyswisseph 2.10.3.2 (izgrađen iz izvornog koda), lunar_python 1.4.9, sxtwl 2.0.7, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1.

**RNG:** `secrets.SystemRandom` koristi se za:
- bacanja novčića za I Ching i 10 karata tarota (spremljeno u `slucajno_ej_2026-11-24.json`, ne izvlači se ponovno);
- slučajnu kontrolnu metodu;
- kandidate za L;
- 1000 slučajnih setova.

**Pretpostavke**

- Svođenje: glavni broj > 50 umanjuje se za 50, euro broj > 12 za 12; godina 2026 → 1. Redni dan 328 → 28.
- Euro popis svakog sustava je isti niz izvornih vrijednosti sveden na 1–12 (npr. Maadim 95 → glavni 45, euro 11). Iznimka je sinkronicitet, čiji su euro brojevi ponovljeni euro brojevi 5 i 11.
- I Ching nije imao promjenjivih crta, pa je promijenjeni heksagram isti kao osnovni.
- Brojevi planeta: Sunce 1, Mjesec 2, Jupiter 3, Merkur 5, Venera 6, Saturn 8, Mars 9.
- Planetarni sat, popis: vladar sata, redni sat, noćni sat, vladar dana, izlazak (h, min), zalazak (h, min), po helsinškom vremenu. Nadmorska visina 15 m.
- E2 uzima brojeve redom: Meihua, zapadna, I Ching, kamea, Jyotiš, kabala.
- Desetice su 1–10, 11–20, …, 41–50.
- Statističke metode:
  - vrući: pri izjednačenju manji broj;
  - hladni: najrjeđi u cijeloj povijesti;
  - Pareto: 20 % najčešćih (10 glavnih, 2 euro) u zadnjih 20 % izvlačenja, poredano po ukupnoj frekvenciji.
- Bonferroni ×6.
- Slučajni setovi: 6 neograničenih listića.
- Redoslijed: pri jednakosti odlučuje redoslijed u tablici.

**Nije izračunato**

- **Cijena kombinacije i rok uplate kod HL:** lutrija.hr automatskom pristupu vraća 403. Zato je ulog naveden kao k × c. Neslužbeno, njemački FAQ euro-jackpot.net navodi zatvaranje uplate za Hrvatsku u 18:30 CET; ta stranica navodi porez u kunama, pa je podatak možda zastario.
- **Točno službeno vrijeme izvlačenja:** službenu stranicu nije bilo moguće pročitati, a ostali izvori proturječe.
- **Najmanje 500 izvlačenja:** u formatu 5/50 + 2/12 do 20.11.2026. postoji samo 487. Starija izvlačenja imaju euro 2/10, pa ih nisam miješao.
- **Isplate po razredima:** nisu tražene i nisu korištene. Očekivani povrat ne može se povećati izborom brojeva.

## 6. Napomena

**Nijedna metoda ne mijenja vjerojatnost pojedine kombinacije ni očekivani povrat; šansa za 5+2 je 1 : 139.838.160 po kombinaciji.** Mijenja se samo raspored rizika. Ovaj set ima 6 različitih jackpot-ishoda i P(barem jedan dobitak) = 18,21 %, prema prosjeku od 17,43 % za slučajne setove.
