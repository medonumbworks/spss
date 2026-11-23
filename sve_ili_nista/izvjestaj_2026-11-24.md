# Sve ili ništa (11 od 22) – 9 kombinacija za uto 24.11.2026., kolo 328, 19:00

## 1. Kombinacije

| # | Tip | Sustav/metoda | 11 brojeva | parni:neparni | zbroj | redoslijed igranja |
|---|---|---|---|---|---|---|
| E1 | ezoterijska | Plečkov filtar (svi sustavi) | 1 3 4 5 7 9 10 11 12 14 16 | 5:6 | 92 | 1 |
| E2 | ezoterijska | Istok: Meihua + I Ching + Jyotiš | 1 2 3 4 7 9 10 12 16 19 20 | 6:5 | 103 | 9 |
| E3 | ezoterijska | Zapad: zapadna astr. + kamea Marsa + kabala + tarot + numerologija | 1 4 5 7 8 11 12 13 14 16 19 | 5:6 | 110 | 8 |
| S1 | statistička | vrući/hladni (cijela povijest) (statistički tipična) | 1 3 5 6 7 8 15 16 18 19 22 | 5:6 | 120 | 2 |
| S2 | statistička | vrući (zadnjih 50) (statistički tipična) | 3 4 5 7 8 12 14 15 17 18 20 | 6:5 | 123 | 6 |
| S3 | statistička | vrući (zadnjih 100) (statistički tipična) | 1 2 4 7 8 9 14 17 18 20 21 | 6:5 | 121 | 7 |
| L1 | logička | maks. pokrivenost (greedy + lokalna zamjena, više pokretanja) | 1 3 5 6 10 13 14 18 19 20 21 | 5:6 | 130 | 3 |
| L2 | logička | maks. pokrivenost (greedy + lokalna zamjena, više pokretanja) | 2 3 4 6 7 8 10 11 14 17 19 | 6:5 | 101 | 5 |
| L3 | logička | maks. pokrivenost (greedy + lokalna zamjena, više pokretanja) | 2 3 5 6 8 9 11 13 16 18 20 | 6:5 | 111 | 4 |

## 2. Redoslijed igranja

| k | kombinacije | ulog | P(barem jedan dobitak) | P(isplata ≥ ulog): donja – gornja granica |
|---|---|---|---|---|
| 1 | E1 | 1,30 € | 8,611 % | 7,7190 % – 8,611 % |
| 2 | E1, S1 | 2,60 € | 16,711 % | 14,9843 % – 16,711 % |
| 3 | E1, S1, L1 | 3,90 € | 24,324 % | 1,3226 % – 3,996 % |
| 4 | E1, S1, L1, L3 | 5,20 € | 31,469 % | 2,5698 % – 6,127 % |
| 5 | E1, S1, L1, L3, L2 | 6,50 € | 38,166 % | 0,1817 % – 4,641 % |
| 6 | E1, S1, L1, L3, L2, S2 | 7,80 € | 44,413 % | 0,3456 % – 5,696 % |
| 7 | E1, S1, L1, L3, L2, S2, S3 | 9,10 € | 49,646 % | 0,0309 % – 6,254 % |
| 8 | E1, S1, L1, L3, L2, S2, S3, E3 | 10,40 € | 54,317 % | 0,0768 % – 7,146 % |
| 9 | E1, S1, L1, L3, L2, S2, S3, E3, E2 | 11,70 € | 58,740 % | 0,0082 % – 7,923 % |

- Donja granica uzima samo potvrđene isplate (jackpot ili barem ⌈k/2⌉ dobitaka 8/3, jer 2,60 € × ⌈k/2⌉ ≥ 1,30 € × k). Gornja granica vrijedi ako bi bilo koji dobitak 10/1 ili 9/2 sam pokrio ulog; iznosi tih razreda nisu poznati.
- k = 1: svih 9 kombinacija ima istu P = 8,611 % i isti najbolji par (16,711 %), pa prednost ima redoslijed u tablici (E1).
- Prvih k po ovom redoslijedu točno je optimalno za k = 1, 2, 3 i 9. Za k = 4–8 najbolji podskupovi nisu ugniježđeni (provjereno svih C(9,k)):
  - k = 4: E1, L1, L2, L3 → 31,470 % (redoslijed daje 31,469 %)
  - k = 5: E2, S1, L1, L2, L3 → 38,169 % (redoslijed daje 38,166 %)
  - k = 6: E2, E3, S1, L1, L2, L3 → 44,415 % (redoslijed daje 44,413 %)
  - k = 7: E2, E3, S1, S2, L1, L2, L3 → 50,241 % (redoslijed daje 49,646 %)
  - k = 8: E2, E3, S1, S2, S3, L1, L2, L3 → 55,125 % (redoslijed daje 54,317 %)
- Očekivana potvrđena isplata za bilo kojih k kombinacija = k × 0,4275 € (donja granica).

## 3. Provjere

- Duplikati: nema. Parovi kombinacija–komplement: nema, dakle 18 različitih jackpot-ishoda.
- Zajednički brojevi među 36 parova: najmanje 4, najviše 8 (dopušteno 3–8).
- Meko pravilo (5:6/6:5, niz ≤ 4) zadovoljeno je u svih 9 kombinacija. Jedina ispravka: E2 5→12, jer bi inače nastao niz 1–5.
- Točne vjerojatnosti (sva 705.432 izvlačenja, numpy bitmaske):
  - P(jackpot) = 18/705.432 = 0,0026 % (1 : 39.191)
  - **P(barem jedan dobitak) = 414370/705.432 = 58,740 %**
  - po razredima (barem jedna kombinacija): 10/1 0,309 % · 9/2 7,617 % · 8/3 53,450 %
  - očekivani broj dobitnih kombinacija = 22779/29393 = 0,7750
  - očekivana isplata iz potvrđenih iznosa = 452355/117572 € = **3,8475 €** uz ulog 11,70 € (32,9 %, donja granica)
  - P(isplata ≥ ulog): od 0,0082 % (jackpot ili ≥ 5× 8/3) do 7,923 %
  - broj dobitaka 8/3: 0 → 46,550 % · 1 → 39,497 % · 2 → 12,047 % · 3 ili više → 1,906 %
  - samo E1–E3 i S1–S3: 41,907 %; L1–L3 dodaju 16,833 %
- Usporedba s 1000 slučajnih setova (bez komplemenata i duplikata):
  - P(barem jedan dobitak): prosjek 55,521 %, sd 1,229 pp, raspon 51,234 %–58,632 %. Ovaj set je **bolji od svih 1000** (100,0. percentil).
  - P(potvrđena isplata ≥ ulog): prosjek 0,0293 %, raspon 0,0054 %–0,1780 %. Ovaj set je na 2,2. percentilu: najveća pokrivenost znači najmanje istodobnih dobitaka 8/3.
- Pretraga L1–L3: greedy 414114, lokalna zamjena 414300, zatim 16 ponovnih pokretanja u 130 s → 414370.

## 4. Kandidati po sustavu

| Sustav | prvih 11 | ključni podaci |
|---|---|---|
| Sinkronicitet | 4 12 16 18 22 2 11 1 20 13 9 | ≥4× u zadnjih 5 (kola 323–327): [4, 12, 16, 18, 22]; 24, 11, 2026→1, kolo 328→20, redni dan 328→20, starost 35→13, 31→9, 5, 18, 20, kolo 327 |
| Arhetipovi | 4 12 1 16 7 10 6 22 18 14 | zadani popis |
| I Ching | 19 7 20 3 4 5 6 | bacanja [7, 8, 9, 6, 9, 8] (odozdo); heksagram 63 (Kan/Li) → 51; nuklearni 64; promjenjive crte [3, 4, 5] |
| Meihua | 3 10 9 1 4 2 7 16 11 | Y=7 (午), M=10, D=16, H=11 (戌); gornji 1 Qian, donji 4 Zhen, crta 2; heksagram 25 → 10, nuklearni 53 |
| Jyotiš | 20 4 16 9 1 | Mjesec 40.05° sid., nakšatra 4 Rohini; tithi 16; vladar dana Mars=9; yoga 20; srednji Rahu 300.58° → nakšatra 23 Dhanishta; ayanamsa 24.2328°; nakšatra: 4,7 min od ulaska, 21 h 10 min do izlaska; tithi 16 = Krishna Pratipada: 3 h 6 min od početka, 17 h 24 min do kraja; yoga 20 = Shiva: 7 h 23 min od početka, 12 h 31 min do kraja (linearno prema trenutnoj brzini) |
| Zapadna astrologija | 8 5 4 14 3 10 15 | Mars 149.50° (Lav 29.50°)→30; Jupiter 146.49° (Lav 26.49°)→27; Venera 205.02° (Vaga 25.02°)→26; Merkur 223.48° (Škorpion 13.48°)→14; Sunce 242.47° (Strijelac 2.47°)→3; Mjesec 64.28° (Blizanci 4.28°)→5 |
| Kamea Marsa | 11 12 13 14 15 3 8 18 1 5 21 | kvadrat iz upita (Agrippa II, 22); u kodu provjereno: 5 redova, 5 stupaca, 2 dijagonale = 65, brojevi 1–25 jednom; glavna dijagonala [11, 12, 13, 14, 15], sporedna [3, 8, 13, 18, 23]; 65→21 |
| Kabala | 7 5 1 11 4 18 10 | Maadim 95→7, Gevura 5, IVAN 67→1, mazal 77→11, LOTO 51→7 (dupl.), 26→4, 18, 10 |
| Tarot | 8 14 16 3 15 12 22 1 2 17 | Sunce Strijelac 2.47°, 1. dekan → Štapovi 8 (30); izvučeno: Mačevi 8(58), Velika arkana 16(16), Novčići 5(69), Velika arkana 8(8), Mačevi 9(59), Novčići Kralj(78), Pehari 8(44), Pehari 9(45), Pehari 10(46), Pehari 3(39) |
| Numerologija | 11 7 19 13 9 | osobna godina 1, osobni mjesec 3, osobni dan 9 (rođen 31.05.; 24.11.2026.); univerzalni dan 9 |
| Lenormand | 3 14 2 10 5 6 13 18 17 16 4 | karte: 25 Prsten, 36 Križ, 24 Srce, 2 Djetelina, 10 Kosa, 5 Stablo, 6 Oblaci, 35 Sidro, 18 Pas |
| Planetarni sat | 9 15 3 7 16 17 11 12 13 14 8 | izlazak 07:07:41, zalazak 16:17:23, sutrašnji izlazak 07:08:59 (CET); 19:00 = 15. sat (3. noćni, 18:45:59–20:00:17), vladar Mars (=9); od početka sata 14.0 min, do kraja 60.3 min; 1. dnevni sat = Mars; planet sata Mars → dodane dijagonale kamee Marsa |
| **Plečkov filtar** (12 sustava) | 4 3 7 16 12 14 9 1 5 10 11 | broj/glasovi/zbroj pozicija: 4/7/22, 3/7/24, 7/5/14, 16/5/18, 12/4/12, 14/4/12, 9/4/13, 1/4/15, 5/4/15, 10/4/18, 11/3/6, 15/3/12, 2/3/15, 8/2/2, 19/2/4, 20/2/4, 13/2/7, 18/2/10, 22/1/5, 6/1/6, 17/1/6, 21/0/0 |

Pravilnosti u zadnjih 5 kola (323, 324, 325, 326, 327): 19.11.2026: 3 4 7 9 11 12 13 14 15 21 22; 20.11.2026: 1 3 9 11 13 14 16 18 19 20 22; 21.11.2026: 4 5 6 7 11 12 16 17 18 20 22; 22.11.2026: 3 4 5 9 12 13 14 15 16 18 22; 23.11.2026: 1 4 5 6 7 8 12 16 17 18 19
- Nijedan broj nije u svih 5 kola. U 4 od 5: [4, 12, 16, 18, 22]. Ni u jednom: [2, 10].
- Ponovljeno iz prethodnog kola: [7, 6, 5, 6, 5]; parni: [4, 5, 7, 6, 6]; zbrojevi: [131, 146, 138, 131, 113]; najdulji niz: [5, 3, 4, 5, 5] (tri kola imaju niz od 5).
- Lunarni datum (lokalni 24.11.): lunar_python 丙午 god., mj. 10 (十), dan 16 (十六), sat 戌; sxtwl mj. 10 (prijestupni=False), dan 16 – slažu se. U Pekingu je tada 25.11.2026 02:00 (Peking) = lunarni mj. 10, dan 17, sat 丑; nije korišteno.

## 5. Statistika i backtest

Walk-forward od 100. poznatog kola (n = 557; praznine preskočene):

| Metoda | dobitni događaji (oček. 48,0) | 10/1 | 9/2 | 8/3 | z | p (jednostr.) | p Bonf. (×7) | prosj. pogodaka | p Bonf. (prosjek) | χ² raspodjele; p Bonf. |
|---|---|---|---|---|---|---|---|---|---|---|
| vrući/hladni (cijela povijest) | 58 | 1 | 2 | 55 | 1,52 | 0,078 | 0,544 | 5,474 | 1,00 | 6,32; 1,00 |
| vrući (zadnjih 50) | 52 | 0 | 3 | 49 | 0,61 | 0,291 | 1,000 | 5,504 | 1,00 | 6,16; 1,00 |
| vrući (zadnjih 100) | 50 | 0 | 3 | 47 | 0,31 | 0,401 | 1,000 | 5,465 | 1,00 | 0,79; 1,00 |
| Pareto 20/20 | 48 | 0 | 3 | 45 | 0,01 | 0,520 | 1,000 | 5,429 | 1,00 | 6,34; 1,00 |
| prezreli (najdulje neizvučeni) | 44 | 0 | 1 | 43 | -0,60 | 0,746 | 1,000 | 5,474 | 1,00 | 4,58; 1,00 |
| zadnje kolo / komplement | 44 | 0 | 1 | 43 | -0,60 | 0,746 | 1,000 | 5,526 | 1,00 | 4,58; 1,00 |
| slučajna (secrets) – kontrola | 47 | 0 | 11 | 36 | -0,15 | 0,580 | 1,000 | 5,557 | 1,00 | 6,00; 1,00 |

- Ishod 11/0 nije se dogodio ni u jednoj metodi.
- Hladni brojevi su komplement vrućih, a „komplement zadnjeg kola” je komplement „zadnjeg kola”. Parovi imaju iste dobitne događaje, pa je svaki vođen kao jedna metoda (7 metoda).
- **Nijedna metoda nakon Bonferronija nije bolja od slučaja** (najmanji p = 0,544). S1–S3 su zato samo „statistički tipične” kombinacije (zbroj 126,5 ± 10, parnost 5:6/6:5) iz tri različite metode: tri najbolje po z, primijenjene na sva 656 kola.
- Prilagodbe S (sljedeći brojevi po rangu iste metode): S1 17→22 (prije: zbroj 115, parnih 4); S2 bez zamjene (prije: zbroj 123, parnih 6); S3 10→21 (prije: zbroj 110, parnih 7).

Testovi slučajnosti (656 kola):
- χ² frekvencija, ispravljen za izvlačenje bez vraćanja (df 21): 24,29, p = 0,279. Naivni χ²: 12,73, p = 0,918. Frekvencije 296–352, očekivano 328.
- Stabilnost vrućih brojeva (1. vs 2. polovica, Spearman): ρ = 0,044, p = 0,847.
- Parnost: opaženo [36, 101, 218, 178, 91, 32], očekivano [28.2, 101.3, 198.5, 198.5, 101.3, 28.2] (≤3, 4, 5, 6, 7, ≥8); χ² = 7,70, df 5, p = 0,173.
- Zbroj: prosjek 126,62 (teorija 126,5), sd 15,61 (teorija 15,23), z = 0,21; χ² po decilima = 2,09, df 9, p = 0,990.

Backtest 9 kombinacija (656 kola):

| # | pogoci 1·2·3·4·5·6·7·8·9·10 | 11/0 | 10/1 | 9/2 | 8/3 | ukupno | potvrđena isplata |
|---|---|---|---|---|---|---|---|
| E1 | 0·5·32·105·182·194·108·27·3·0 | 0 | 0 | 8 | 59 | 67 | 153,40 € |
| E2 | 0·3·21·107·199·193·100·30·3·0 | 0 | 0 | 6 | 51 | 57 | 132,60 € |
| E3 | 0·4·25·118·177·195·112·20·5·0 | 0 | 0 | 9 | 45 | 54 | 117,00 € |
| S1 | 0·2·20·84·183·222·106·33·5·1 | 0 | 1 | 7 | 53 | 61 | 137,80 € |
| S2 | 0·3·30·94·184·209·94·40·2·0 | 0 | 0 | 5 | 70 | 75 | 182,00 € |
| S3 | 1·3·26·95·196·201·106·24·4·0 | 0 | 1 | 7 | 50 | 58 | 130,00 € |
| L1 | 1·2·27·116·200·189·91·27·3·0 | 0 | 1 | 5 | 54 | 60 | 140,40 € |
| L2 | 0·3·27·105·178·215·99·25·4·0 | 0 | 0 | 7 | 52 | 59 | 135,20 € |
| L3 | 0·1·24·106·200·204·96·21·3·1 | 0 | 1 | 4 | 45 | 50 | 117,00 € |

- Dobitni događaji: 541 (očekivano 508,4). Slučajni setovi: prosjek 509,1, sd 22,6 → 91,2. percentil.
- Potvrđena isplata 1245,40 € uz ulog 7675,20 € (16,2 %). To je donja granica: 4× 10/1 i 58× 9/2 nemaju poznat iznos. Slučajni setovi: prosjek 2307,01 € (uključuje njihove jackpotove) → 84,5. percentil.
- Kola s barem jednim dobitkom: 418/656 (63,7 %).
- S1–S3 su odabrani na istim podacima, pa njihov backtest nije neovisan. Kombinacije nisu mijenjane prema backtestu.

## 6. Povratna informacija, izvori, pretpostavke

**Kolo 327 (23.11.2026.: 1 4 5 6 7 8 12 16 17 18 19)**

| # | tvoji pogoci | izračunato | razred |
|---|---|---|---|
| E1 | 5 | 5 | - |
| E2 | 5 | 5 | - |
| E3 | 6 | 6 | - |
| S1 | 8 | 8 | 8/3 |
| S2 | 3 | 3 | 8/3 |
| S3 | 6 | 6 | - |
| L1 | 5 | 5 | - |
| L2 | 4 | 4 | - |
| L3 | 4 | 4 | - |

- Svih 9 vrijednosti podudara se s tvojim listićem.
- Dobitak: S1 (8) i S2 (3) = 2 × 2,60 € = 5,20 € uz ulog 11,70 €.
- Samo zabilježeno; metode nisu mijenjane prema jednom kolu.

**Izvori podataka.** Datoteka `kola_do_2026-11-23.csv` ima ukupno 656 kola:
- 635 kola iz priloga (01.01.2025.–28.09.2026.; CSV i XLSX su identični);
- 21 kolo 307–327/2026 iz upita (LotteryExtreme i listić).

Provjereno je da svako kolo ima 11 različitih brojeva 1–22, uzlazno, da je kolo jednako rednom danu i da nema duplikata datuma.

**Trenutak.** 24.11.2026. u 19:00 Europe/Zagreb (CET) = 18:00 UT, JD 2461369,25.

**Isplate.** 11/0 = 80.000 € i 8/3 = 2,60 € (iz upita); ulog je 1,30 € po kombinaciji.

**Verzije.** Python 3.13.18; Swiss Ephemeris 2.10.03 (pyswisseph 2.10.3.2 izgrađen iz izvornog koda) s datotekama sepl_18.se1 i semo_18.se1 iz aloistr/swisseph. Za svaki položaj provjereno je da je korišten FLG_SWIEPH. Ayanamsa Lahiri 24,2328°. Ostalo: lunar_python 1.4.9, sxtwl 2.0.7, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, openpyxl 3.1.5.

**RNG.** Svi slučajni elementi dolaze iz `secrets.SystemRandom`.
- I Ching, tarot i Lenormand izvučeni su jednom i spremljeni u `slucajno_2026-11-24.json`; ponovno pokretanje ih ne izvlači ponovno.
- Isti generator koriste slučajna kontrolna metoda, 1000 slučajnih setova i kandidati za L.
- Ova dopuna je deterministička: kombinacije nisu ponovno računate, dodani su samo redoslijed, granice i udaljenosti od promjene.

**Pretpostavke i konvencije:**
- Svođenje: od brojeva većih od 22 oduzima se 22; godina se svodi zbrojem znamenki (2026 → 1).
- Brojevi planeta: Sunce 1, Mjesec 2, Jupiter 3, Merkur 5, Venera 6, Saturn 8, Mars 9 (u skladu s „Mars = 9”).
- Planetarni sat, popis: vladar sata, redni sat, noćni sat, vladar dana, sat i minuta izlaska, sat i minuta zalaska. Planet sata je Mars, pa su dodane dijagonale kamee Marsa.
- Kamea Marsa: korišten je kvadrat iz upita; provjereni su magični zbroj 65 i brojevi 1–25.
- Meihua: korišten je lokalni lunarni datum, H = 11.
- Tarot indeksi i dekani po Golden Dawnu.
- Meko pravilo: do 2 zamjene sljedećim brojevima s Plečkova popisa.
- Spojene metode: vrući/hladni i zadnje kolo/komplement; Bonferroni ×7.
- Redoslijed igranja: pohlepno prema doprinosu P(barem jedan dobitak); pri jednakosti odlučuje redoslijed u tablici.
- Osobni podaci korišteni su kako su zadani; izračunati su samo osobni dan 9 i univerzalni dan 9.

**Nije izračunato i zašto:**
- **Isplate 10/1 i 9/2:** iznos nije poznat. Zato su očekivana isplata, povrat i P(isplata ≥ ulog) samo donje granice; za P(isplata ≥ ulog) navedena je i gornja granica.
- **Neovisna provjera kola 307–327:** vanjske stranice su u ovoj sesiji blokirane.
- **Kola 29.09.–02.11.2026. (35) i kolo 178/2026:** nema ih ni u jednom izvoru koji imam. U walk-forwardu su preskočena.
- **Tekst Agrippe za kamea Marsa:** nije dohvatljiv. Kvadrat je uzet iz upita, a njegova svojstva provjerena su u kodu.
- **Kamea Sunca i Jupitera:** nije bila potrebna, jer je planet sata Mars.

**Ispravak izvještaja od 23.11.** U Plečkovu filtru glasalo je 11 sustava, a ne 12 kako je pisalo. Kombinacije time nisu promijenjene.

**Datoteke** (u `sve_ili_nista/`):
- `sin_2026-11-24.csv` (#;tip;sustav;redoslijed;B1;…;B11)
- `izvjestaj_2026-11-24.md`
- `rezultati_2026-11-24.json`
- `kola_do_2026-11-23.csv`
- skripte: `prosiri_podatke.py`, `sin_2026_11_24.py`, `l_opt_2026_11_24.py`, `dopuna_2026_11_24.py`, `izvjestaj_gen_2026_11_24.py`

## 7. Napomena

**Nijedna metoda ne mijenja vjerojatnost pojedine kombinacije ni očekivani povrat; mijenja se samo raspored rizika.**
- Svaka kombinacija ima P(11/0) = 2/705.432 i P(dobitak) = 8,611 %.
- Očekivana potvrđena isplata po kombinaciji je 0,4275 € i ista je za sve kombinacije.
- Ovaj set ima najveću P(barem jedan dobitak) (58,74 %), ali zauzvrat manju vjerojatnost više istodobnih dobitaka 8/3 od tipičnog slučajnog seta.
