# Sve ili ništa (11 od 22) – 9 kombinacija za uto 24.11.2026., kolo 328, 19:00

## 1. Kombinacije

| # | Tip | Sustav/metoda | 11 brojeva (uzlazno) | parni:neparni | zbroj |
|---|---|---|---|---|---|
| E1 | ezoterijska | Plečkov filtar (svi sustavi) | 1 3 4 5 7 9 10 11 12 14 16 | 5:6 | 92 |
| E2 | ezoterijska | Istok: Meihua + I Ching + Jyotiš | 1 2 3 4 7 9 10 12 16 19 20 | 6:5 | 103 |
| E3 | ezoterijska | Zapad: zapadna astr. + kamea Marsa + kabala + tarot + numerologija | 1 4 5 7 8 11 12 13 14 16 19 | 5:6 | 110 |
| S1 | statistička | vrući/hladni (cijela povijest) (statistički tipična) | 1 3 5 6 7 8 15 16 18 19 22 | 5:6 | 120 |
| S2 | statistička | vrući (zadnjih 50) (statistički tipična) | 3 4 5 7 8 12 14 15 17 18 20 | 6:5 | 123 |
| S3 | statistička | vrući (zadnjih 100) (statistički tipična) | 1 2 4 7 8 9 14 17 18 20 21 | 6:5 | 121 |
| L1 | logička | maks. pokrivenost (greedy + lokalna zamjena, više pokretanja) | 1 3 5 6 10 13 14 18 19 20 21 | 5:6 | 130 |
| L2 | logička | maks. pokrivenost (greedy + lokalna zamjena, više pokretanja) | 2 3 4 6 7 8 10 11 14 17 19 | 6:5 | 101 |
| L3 | logička | maks. pokrivenost (greedy + lokalna zamjena, više pokretanja) | 2 3 5 6 8 9 11 13 16 18 20 | 6:5 | 111 |

## 2. Provjere i točne vjerojatnosti

- Duplikati: nema; parovi kombinacija–komplement: nema → 18 različitih jackpot-ishoda.
- Zajednički brojevi među 36 parova: najmanje 4, najviše 8 (dopušteno 3–8).
- Meko pravilo (5:6/6:5, niz ≤ 4): zadovoljeno u svih 9. Ispravke E: E1: bez zamjene; E2: 5→12; E3: bez zamjene (E2: niz 1–5 bio bi dulji od 4).
- Točno (sva 705.432 izvlačenja, numpy bitmaske):
  - P(jackpot, 11 ili 0 na barem jednoj) = 18/705.432 = 0,0026 % (1 : 39.191)
  - **P(barem jedan dobitak)** = 414370/705.432 = **58,740 %**
  - po razredima (barem jedna kombinacija): 11/0 0,0026 % · 10/1 0,309 % · 9/2 7,617 % · 8/3 53,450 %
  - očekivani broj dobitnih kombinacija = 22779/29393 = 0,7750 (isti za svaki set od 9)
  - E1–E3 + S1–S3 sami: 41,907 %; L1–L3 dodaju 16,833 %
- Isplate, samo potvrđene (11/0 = 80.000 €, 8/3 = 2,60 €; 10/1 i 9/2 = nepoznato, računato kao 0 → donja granica):
  - očekivana isplata = 452355/117572 € = **3,8475 €** po kolu uz ulog 11,70 € (32,9 % uloga, donja granica)
  - P(isplata ≥ ulog) ≥ 58/705.432 = 0,0082 % (potrebno: jackpot ili ≥ 5 dobitaka 8/3); gornja granica uz bilo koje iznose 10/1 i 9/2: 7,923 %
  - broj dobitaka 8/3 u setu: 0 → 46,550 % · 1 → 39,497 % · 2 → 12,047 % · 3+ → 1,906 %
- 1000 slučajnih setova (bez komplemenata/duplikata):
  - P(barem jedan dobitak): prosjek 55,521 %, sd 1,229 pp, raspon 51,234 %–58,632 % → ovaj set **100,0. percentil** (bolji od svih 1000).
  - P(potvrđena isplata ≥ ulog): prosjek 0,0293 %, raspon 0,0054 %–0,1780 % → ovaj set 2,2. percentil. Najveća pokrivenost znači najmanje istodobnih dobitaka 8/3 (raspored rizika, ne očekivana vrijednost).
- Pretraga L: greedy 414114 → lokalna zamjena 414300; zatim 16 ponovnih pokretanja (greedy + lokalna zamjena, 130 s) → 414370.

## 3. Redoslijed igranja (pohlepno po doprinosu P(barem jedan dobitak))

| k | dodaje | prvih k | P(barem jedan dobitak) | ulog | P(potvrđena isplata ≥ ulog) | najbolji podskup od k (točno, svih C(9,k)) | njegova P |
|---|---|---|---|---|---|---|---|
| 1 | E1 | E1 | 8,611 % | 1,30 € | 7,7190 % (≥1× 8/3) | isti | 8,611 % |
| 2 | S1 | E1, S1 | 16,711 % | 2,60 € | 14,9843 % (≥1× 8/3) | isti | 16,711 % |
| 3 | L1 | E1, S1, L1 | 24,324 % | 3,90 € | 1,3226 % (≥2× 8/3) | isti | 24,324 % |
| 4 | L3 | E1, S1, L1, L3 | 31,469 % | 5,20 € | 2,5698 % (≥2× 8/3) | E1, L1, L2, L3 | 31,470 % |
| 5 | L2 | E1, S1, L1, L3, L2 | 38,166 % | 6,50 € | 0,1817 % (≥3× 8/3) | E2, S1, L1, L2, L3 | 38,169 % |
| 6 | S2 | E1, S1, L1, L3, L2, S2 | 44,413 % | 7,80 € | 0,3456 % (≥3× 8/3) | E2, E3, S1, L1, L2, L3 | 44,415 % |
| 7 | S3 | E1, S1, L1, L3, L2, S2, S3 | 49,646 % | 9,10 € | 0,0309 % (≥4× 8/3) | E2, E3, S1, S2, L1, L2, L3 | 50,241 % |
| 8 | E3 | E1, S1, L1, L3, L2, S2, S3, E3 | 54,317 % | 10,40 € | 0,0768 % (≥4× 8/3) | E2, E3, S1, S2, S3, L1, L2, L3 | 55,125 % |
| 9 | E2 | E1, S1, L1, L3, L2, S2, S3, E3, E2 | 58,740 % | 11,70 € | 0,0082 % (≥5× 8/3) | isti | 58,740 % |

- k = 1: svih 9 kombinacija ima istu P = 8,611 % i isti najbolji par (16,711 %); prednost prema redoslijedu u tablici (E1).
- Prvih k iz pohlepnog redoslijeda optimalno je za k = 1, 2, 3 i 9. Za k = 4–8 najbolji podskupovi nisu ugniježđeni (vidi stupac), pa nijedan redoslijed nije najbolji za svaki k. Ako igraš točno k kombinacija, uzmi podskup iz pretposljednjeg stupca.
- P(potvrđena isplata ≥ ulog) za k kombinacija: jackpot ili barem ⌈k/2⌉ dobitaka 8/3 (2,60 € × ⌈k/2⌉ ≥ 1,30 € × k).
- Očekivana potvrđena isplata za bilo kojih k kombinacija = k × 0,4275 €.

## 4. Kandidati po sustavu

| Sustav | prvih 11 (svedeno 1–22) | ključni podaci |
|---|---|---|
| Sinkronicitet | 4 12 16 18 22 2 11 1 20 13 9 | ≥4× u zadnjih 5 (kola 323–327): [4, 12, 16, 18, 22]; 24, 11, 2026→1, kolo 328→20, redni dan 328→20, starost 35→13, 31→9, 5, 18, 20, kolo 327 |
| Arhetipovi | 4 12 1 16 7 10 6 22 18 14 | zadani popis |
| I Ching | 19 7 20 3 4 5 6 | bacanja [7, 8, 9, 6, 9, 8] (odozdo); heksagram 63 (Kan/Li) → 51; nuklearni 64; promjenjive crte [3, 4, 5] |
| Meihua | 3 10 9 1 4 2 7 16 11 | Y=7 (午), M=10, D=16, H=11 (戌); gornji 1 Qian, donji 4 Zhen, crta 2; heksagram 25 → 10, nuklearni 53 |
| Jyotiš | 20 4 16 9 1 | Mjesec 40.05° sid., nakšatra 4 Rohini; tithi 16; vladar dana Mars=9; yoga 20; srednji Rahu 300.58° → nakšatra 23 Dhanishta; ayanamsa 24.2328° |
| Zapadna astrologija | 8 5 4 14 3 10 15 | Mars 149.50° (Lav 29.50°)→30; Jupiter 146.49° (Lav 26.49°)→27; Venera 205.02° (Vaga 25.02°)→26; Merkur 223.48° (Škorpion 13.48°)→14; Sunce 242.47° (Strijelac 2.47°)→3; Mjesec 64.28° (Blizanci 4.28°)→5 |
| Kamea Marsa | 11 12 13 14 15 3 8 18 1 5 21 | kvadrat iz upita (Agrippa II, 22); u kodu provjereno: 5 redova, 5 stupaca, 2 dijagonale = 65, brojevi 1–25 jednom; glavna dijagonala [11, 12, 13, 14, 15], sporedna [3, 8, 13, 18, 23]; 65→21 |
| Kabala | 7 5 1 11 4 18 10 | Maadim 95→7, Gevura 5, IVAN 67→1, mazal 77→11, LOTO 51→7 (dupl.), 26→4, 18, 10 |
| Tarot | 8 14 16 3 15 12 22 1 2 17 | Sunce Strijelac 2.47°, 1. dekan → Štapovi 8 (30); izvučeno: Mačevi 8(58), Velika arkana 16(16), Novčići 5(69), Velika arkana 8(8), Mačevi 9(59), Novčići Kralj(78), Pehari 8(44), Pehari 9(45), Pehari 10(46), Pehari 3(39) |
| Numerologija | 11 7 19 13 9 | osobna godina 1, osobni mjesec 3, osobni dan 9 (rođen 31.05.; 24.11.2026.); univerzalni dan 9 |
| Lenormand | 3 14 2 10 5 6 13 18 17 16 4 | karte: 25 Prsten, 36 Križ, 24 Srce, 2 Djetelina, 10 Kosa, 5 Stablo, 6 Oblaci, 35 Sidro, 18 Pas |
| Planetarni sat | 9 15 3 7 16 17 11 12 13 14 8 | izlazak 07:07:41, zalazak 16:17:23, sutrašnji izlazak 07:08:59 (CET); 19:00 = 15. sat (3. noćni, 18:45:59–20:00:17), vladar Mars (=9); od početka sata 14.0 min, do kraja 60.3 min; 1. dnevni sat = Mars; planet sata Mars → dodane dijagonale kamee Marsa |
| **Plečkov filtar** (12 sustava) | 4 3 7 16 12 14 9 1 5 10 11 | broj/glasovi/zbroj pozicija: 4/7/22, 3/7/24, 7/5/14, 16/5/18, 12/4/12, 14/4/12, 9/4/13, 1/4/15, 5/4/15, 10/4/18, 11/3/6, 15/3/12, 2/3/15, 8/2/2, 19/2/4, 20/2/4, 13/2/7, 18/2/10, 22/1/5, 6/1/6, 17/1/6, 21/0/0 |

Pravilnosti u zadnjih 5 kola (323, 324, 325, 326, 327): 19.11.2026: 3 4 7 9 11 12 13 14 15 21 22; 20.11.2026: 1 3 9 11 13 14 16 18 19 20 22; 21.11.2026: 4 5 6 7 11 12 16 17 18 20 22; 22.11.2026: 3 4 5 9 12 13 14 15 16 18 22; 23.11.2026: 1 4 5 6 7 8 12 16 17 18 19
- nijedan broj u svih 5; u 4 od 5: [4, 12, 16, 18, 22]; nijednom: [2, 10]; ponovljeno iz prethodnog kola: [7, 6, 5, 6, 5]; parni: [4, 5, 7, 6, 6]; zbrojevi: [131, 146, 138, 131, 113]; najdulji niz: [5, 3, 4, 5, 5] (tri kola imaju niz od 5).
- Lunarni datum (lokalni 24.11.): lunar_python 丙午 god., mj. 10 (十), dan 16 (十六), sat 戌; sxtwl mj. 10 (prijestupni=False), dan 16 – slažu se. U Pekingu: 25.11.2026 02:00 (Peking) = lunarni mj. 10, dan 17, sat 丑.
- Jyotiš: Mjesec je 0,05° iza granice Krittika/Rohini (granični slučaj).

## 5. Statistika (walk-forward od 100. poznatog kola, n = 557; praznine preskočene)

| Metoda | dobitni događaji (oček. 48,0) | 10/1 | 9/2 | 8/3 | z | p (jednostr.) | p Bonf. (×7) | prosj. pogodaka | p Bonf. (prosjek) | χ² raspodjele; p Bonf. |
|---|---|---|---|---|---|---|---|---|---|---|
| vrući/hladni (cijela povijest) | 58 | 1 | 2 | 55 | 1,52 | 0,078 | 0,544 | 5,474 | 1,00 | 6,32; 1,00 |
| vrući (zadnjih 50) | 52 | 0 | 3 | 49 | 0,61 | 0,291 | 1,000 | 5,504 | 1,00 | 6,16; 1,00 |
| vrući (zadnjih 100) | 50 | 0 | 3 | 47 | 0,31 | 0,401 | 1,000 | 5,465 | 1,00 | 0,79; 1,00 |
| Pareto 20/20 | 48 | 0 | 3 | 45 | 0,01 | 0,520 | 1,000 | 5,429 | 1,00 | 6,34; 1,00 |
| prezreli (najdulje neizvučeni) | 44 | 0 | 1 | 43 | -0,60 | 0,746 | 1,000 | 5,474 | 1,00 | 4,58; 1,00 |
| zadnje kolo / komplement | 44 | 0 | 1 | 43 | -0,60 | 0,746 | 1,000 | 5,526 | 1,00 | 4,58; 1,00 |
| slučajna (secrets) – kontrola | 47 | 0 | 11 | 36 | -0,15 | 0,580 | 1,000 | 5,557 | 1,00 | 6,00; 1,00 |

Niti jedan ishod 11/0 ni u jednoj metodi. Hladni su komplement vrućih, a „komplement zadnjeg kola” je komplement „zadnjeg kola” (isti dobitni događaji) → svaki par vođen je kao jedna metoda (7 metoda, Bonferroni ×7).

**Nijedna metoda nakon Bonferronija nije bolja od slučaja** (najmanji p = 0,544). S1–S3 su zato samo „statistički tipične” kombinacije (zbroj 126,5 ± 10, parnost 5:6/6:5) iz tri različite metode – tri najbolje rangirane po z, primijenjene na sva 656 kola i prilagođene sljedećim brojevima po rangu iste metode.
- Prilagodbe S: S1: 17→22 (sirovo zbroj 115, parni 4); S2: bez zamjene (sirovo zbroj 123, parni 6); S3: 10→21 (sirovo zbroj 110, parni 7).

Testovi slučajnosti (656 kola):
- χ² frekvencija (ispravljen za izvlačenje bez vraćanja, df 21): 24,29, p = 0,279 (naivni: 12,73, p = 0,918); frekvencije 296–352, očekivano 328.
- Stabilnost vrućih (1. vs 2. polovica, Spearman): ρ = 0,044, p = 0,847.
- Parnost: opaženo [36, 101, 218, 178, 91, 32] vs očekivano [28.2, 101.3, 198.5, 198.5, 101.3, 28.2] (≤3, 4, 5, 6, 7, ≥8), χ² = 7,70, df 5, p = 0,173.
- Zbroj: prosjek 126,62 (teorija 126,5), sd 15,61 (teorija 15,23), z = 0,21; χ² po decilima = 2,09, df 9, p = 0,990.

## 6. Backtest 9 kombinacija (656 kola)

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

- Dobitnih događaja 541 (očekivano 508,4); 1000 slučajnih setova prosjek 509,1, sd 22,6 → 91,2. percentil.
- Potvrđena isplata 1245,40 € prema ulogu 7675,20 € (16,2 %, donja granica: 4× 10/1 i 58× 9/2 bez poznatog iznosa); slučajni setovi prosjek 2307,01 € (sd 9402,15 €, uključuje njihove jackpotove) → 84,5. percentil.
- Kola s barem jednim dobitkom: 418/656 (63,7 %).
- S1–S3 su odabrani na istim podacima (in-sample) – backtest nije neovisan. Kombinacije nisu mijenjane prema backtestu.

## 7. Povratna informacija – kolo 327 (23.11.2026.: 1 4 5 6 7 8 12 16 17 18 19)

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

- Svih 9 podudara se s tvojim listićem. Dobitak S1 (8) i S2 (3) = 2 × 2,60 € = 5,20 € uz ulog 11,70 €. Samo zabilježeno; metode nisu mijenjane prema jednom kolu.

## 8. Izvori, verzije, RNG, nije izračunato, pretpostavke

**Podaci** (`kola_do_2026-11-23.csv`, ukupno 656 kola):
- 635 kola iz priloga (01.01.2025.–28.09.2026.; CSV = XLSX).
- 21 kolo 307–327/2026 iz upita (LotteryExtreme i listić). Ta kola nisu neovisno provjerena jer su vanjski izvori u ovoj sesiji blokirani.
- Sve provjere su prošle: 11 različitih brojeva 1–22, uzlazno, kolo = redni dan, nema duplikata datuma.
- Nedostaju kolo 178/2026 i 35 kola od 29.09. do 02.11.2026. U walk-forwardu su preskočena, a niz nakon praznine tretiran je normalno.

**Trenutak:** 24.11.2026. u 19:00 Europe/Zagreb (CET) = 18:00 UT, JD 2461369,25.

**Isplate:**
- 11/0 = 80.000 € i 8/3 = 2,60 € (iz upita); ulog 1,30 € po kombinaciji.
- **10/1 i 9/2: nije izračunato** (iznos nepoznat). Svi povrati i P(isplata ≥ ulog) zato su donje granice; za P(isplata ≥ ulog) navedena je i gornja granica.

**Osobni podaci** korišteni su kako su zadani, bez preračunavanja. Iz 31.05. i 24.11.2026. izračunati su samo osobni dan (9) i univerzalni dan (9); 11 i 22 se ne svode.

**Izračun položaja:** Swiss Ephemeris 2.10.03 (pyswisseph 2.10.3.2 izgrađen iz izvornog koda), datoteke sepl_18.se1 i semo_18.se1 iz aloistr/swisseph. Za svaki položaj provjereno je da je korišten FLG_SWIEPH. Ayanamsa Lahiri 24,2328°. Izlazak i zalazak Sunca računaju se za 120 m nadmorske visine, gornji rub diska uz refrakciju.

**Verzije:** Python 3.13.18, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, lunar_python 1.4.9, sxtwl 2.0.7, openpyxl 3.1.5.

**RNG:** `secrets.SystemRandom`. Izvlačenja za I Ching, tarot i Lenormand izvučena su jednom i spremljena u `slucajno_2026-11-24.json`, pa ponovno pokretanje skripte ne izvlači ništa ponovno. Isti generator daje i slučajnu kontrolnu metodu, 1000 slučajnih setova i kandidate za L.

**Konvencije:**
- Svođenje: od brojeva > 22 oduzima se 22; godina se svodi zbrojem znamenki (2026 → 1).
- Brojevi planeta: Sunce 1, Mjesec 2, Jupiter 3, Merkur 5, Venera 6, Saturn 8, Mars 9. To je u skladu s „Mars = 9” iz upita.
- Planetarni sat, popis: vladar sata, redni sat (1–24), noćni sat, vladar dana, sat i minuta izlaska, sat i minuta zalaska. Planet sata je Mars, pa su dodane dijagonale kamee Marsa.
- 19:00 je 14,0 min nakon početka i 60,3 min prije kraja 3. noćnog sata (18:45:59–20:00:17).
- Kamea Marsa je uzeta iz upita. Agrippin tekst nije bilo moguće dohvatiti, ali su u kodu provjereni magični zbroj 65 i brojevi 1–25.
- Meihua: korišten je lokalni lunarni datum (10. mjesec, 16. dan), H = 11. Datum u Pekingu je naveden, ali nije korišten.
- Tarot indeksi i dekani kao jučer (Golden Dawn).
- Meko pravilo: do 2 zamjene sljedećim brojevima s Plečkova popisa, počevši od zadnjeg dodanog.
- Spojene metode: vrući/hladni i zadnje kolo/komplement; ukupno 7 metoda, Bonferroni ×7.
- Redoslijed igranja: pri jednakosti prednost ima redoslijed u tablici.

**Ispravak jučerašnjeg izvještaja (23.11.):** u Plečkovu filtru glasalo je 11 sustava, a ne 12 kako je pisalo. To nije utjecalo na kombinacije.

**Nije izračunato:**
- isplate 10/1 i 9/2;
- neovisna provjera kola 307–327;
- kola od 29.09. do 02.11.2026.;
- kamea Sunca i Jupitera (nisu bile potrebne jer je planet sata Mars).

**Napomena o upitu:** §7 je prekinut nakon „… | parni”. Izlaz slijedi jučerašnju strukturu, uz dodani redoslijed igranja i povratnu informaciju za kolo 327.

## 9. Napomena

**Nijedna metoda ne mijenja vjerojatnost pojedine kombinacije ni očekivani povrat.**
- Svaka kombinacija ima P(11/0) = 2/705.432 i P(dobitak) = 8,611 %.
- Očekivana potvrđena isplata po kombinaciji je 0,4275 € (donja granica) i ista je za sve kombinacije.
- Mijenja se samo raspored rizika. Ovaj set ima najveću P(barem jedan dobitak) = 58,74 %, ali zauzvrat ima manju vjerojatnost više istodobnih dobitaka 8/3 od tipičnog slučajnog seta.
