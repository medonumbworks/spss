# Sve ili ništa (11 od 22) – 9 kombinacija za pon 23.11.2026.

## 1. Kombinacije

| # | Tip | Sustav/metoda | 11 brojeva (uzlazno) | parni:neparni | zbroj |
|---|---|---|---|---|---|
| E1 | ezoterijska | Plečkov filtar (svi sustavi) | 1 2 4 7 8 9 14 15 19 20 21 | 5:6 | 120 |
| E2 | ezoterijska | Istok: Meihua + I Ching + Jyotiš | 1 2 3 7 8 10 14 15 18 19 20 | 6:5 | 117 |
| E3 | ezoterijska | Zapad: zapadna astr. + kabala + tarot + numerologija (kamea nije izračunata) | 1 3 4 5 7 8 9 11 19 20 21 | 3:8 ⚠ meko pravilo | 108 |
| S1 | statistička | vrući (cijela povijest) (statistički tipična) | 1 3 5 6 7 8 15 16 18 19 22 | 5:6 | 120 |
| S2 | statistička | hladni (cijela povijest) (statistički tipična) | 2 4 5 9 10 11 13 14 19 20 22 | 6:5 | 129 |
| S3 | statistička | vrući (zadnjih 50) (statistički tipična) | 1 3 5 7 8 10 12 15 17 20 22 | 5:6 | 120 |
| L1 | logička | maks. pokrivenost (greedy + lokalna zamjena, više pokretanja) | 1 2 5 6 7 10 11 15 16 20 21 | 5:6 | 114 |
| L2 | logička | maks. pokrivenost (greedy + lokalna zamjena, više pokretanja) | 2 3 7 9 10 12 13 16 19 21 22 | 5:6 | 134 |
| L3 | logička | maks. pokrivenost (greedy + lokalna zamjena, više pokretanja) | 1 3 6 11 12 14 15 19 20 21 22 | 5:6 | 144 |

## 2. Provjere

- Duplikati: nema; parovi kombinacija–komplement: nema.
- Zajednički brojevi među 36 parova: najmanje 3, najviše 8 (dopušteno 3–8).
- Meko pravilo (5:6/6:5, niz ≤ 4): zadovoljeno osim E3 (3:8) – jedna zamjena s Plečkova popisa ne može ispraviti parnost za 2; tvrda pravila vrijede.
- Točno (enumeracija svih 705.432 izvlačenja, numpy bitmaske):
  - P(Sve ili Ništa, 11 ili 0 na barem jednoj) = 18/705432 = 0,0026 % (1 : 39.191)
  - P(barem jedan dobitak, razredi 11/0, 10/1, 9/2, 8/3) = 406832/705432 = 57,671 %
  - po razredima (barem jedna kombinacija u razredu): 11/0 0,0026 %, 10/1 0,309 %, 9/2 7,557 %, 8/3 52,801 %
  - očekivani broj dobitnih kombinacija = 9 × 60.744/705.432 = 0,7750 (isto za svaki set od 9)
  - samo E1–E3 + S1–S3: 40,539 %; L1–L3 dodaju 17,133 %
- 1000 slučajnih setova (bez komplemenata/duplikata): P(barem jedan) prosjek 55,525 %, sd 1,260 pp, raspon 51,143 %–58,689 %; ovaj set je na 96,6. percentilu.
- Pretraga L: prvi prolaz greedy 406642 → lokalna zamjena 406672; zatim 19 novih greedy pokretanja (800 kandidata/poziciji) + lokalna zamjena u 150 s, najbolje 406832.

## 3. Kandidati po sustavu

| Sustav | prvih 11 (svedeno 1–22) | ključni podaci |
|---|---|---|
| Sinkronicitet | 1 12 14 17 22 2 4 5 19 7 9 | ≥2× u zadnjih 5: [1, 12, 14, 17, 22, 2, 4, 5, 19, 7, 9, 11, 15, 8, 10, 3, 18]; kolo 327→19, starost 35→13; 2026→1 (kako je zadano) |
| Arhetipovi | 4 12 1 16 7 10 6 22 18 14 | zadani popis |
| I Ching | 2 18 10 6 8 5 | bacanja [8, 7, 7, 8, 8, 6] (odozdo); heksagram 46 (Kun/Xun) → 18; nuklearni 54; promjenjive crte [6] |
| Meihua | 14 15 18 8 3 1 7 10 11 | Y=7 (午), M=10, D=15, H=11 (戌); gornji 8 Kun, donji 3 Li, crta 1; heksagram 36 → 15, nuklearni 40 |
| Jyotiš | 20 2 15 19 1 | Mjesec 25.72° sid., nakšatra 2 Bharani; tithi 15; vladar dana Mjesec=2; yoga 19; srednji Rahu 300.63° → nakšatra 23 Dhanishta; ayanamsa 24.2328° |
| Zapadna astrologija | 20 5 3 13 2 10 15 | Mjesec 49.95° (Bik 19.95°)→20; Jupiter 146.43° (Lav 26.43°)→27; Venera 204.67° (Vaga 24.67°)→25; Merkur 222.30° (Škorpion 12.30°)→13; Sunce 241.50° (Strijelac 1.50°)→2 |
| Kabala | 21 9 1 11 7 4 18 10 | Levana 87→21, Jesod 9, IVAN 67→1, mazal 77→11, LOTO 51→7, 26→4, 18, 10 |
| Tarot | 8 21 4 5 20 6 18 19 1 7 | Sunce Strijelac 1.50°, 1. dekan → Štapovi 8 (30); izvučeno: Velika arkana 21(21), Pehari Vitez(48), Novčići 7(71), Mačevi Kralj(64), Pehari 6(42), Pehari Kralj(50), Mačevi Vitez(62), Pehari 5(41), Pehari 9(45), Štapovi 7(29) |
| Numerologija | 11 7 19 13 9 8 | univerzalni dan 23.11.2026 = 8; osobni dan NIJE IZRAČUNAT (nema datuma rođenja) |
| Lenormand | 19 2 9 21 4 14 10 8 6 11 3 | karte: 19 Toranj, 2 Djetelina, 9 Buket, 21 Planina, 26 Knjiga, 14 Lisica, 32 Mjesec, 30 Ljiljani, 6 Oblaci |
| Planetarni sat | 2 15 3 7 6 16 18 | izlazak 07:06, zalazak 16:18, sutrašnji izlazak 07:07 (CET); 20:00 = 15. sat dana (3. noćni), vladar Mjesec (=2); 1. dnevni sat = Mjesec |
| Kamea Mjeseca | – | NIJE IZRAČUNATO – izvor nije bilo moguće dohvatiti i potvrditi |
| **Plečkov filtar** | 2 1 4 7 8 15 20 21 19 9 14 | puni poredak (broj/glasovi/zbroj pozicija): 2/6/17, 1/5/18, 4/4/15, 7/4/16, 8/4/16, 15/3/7, 20/3/7, 21/3/7, 19/3/8, 9/3/10, 14/3/10, 3/3/11, 5/3/12, 6/3/15, 10/3/15, 12/2/4, 11/2/5, 18/2/5, 13/2/8, 16/2/10, 17/1/4, 22/1/5 |

Pravilnosti u zadnjih 5 kola (24.–28.09.2026.): 24.09.2026: 1 3 4 5 7 12 14 17 18 19 22; 25.09.2026: 1 2 6 7 10 14 15 16 17 21 22; 26.09.2026: 1 2 3 4 7 8 9 12 13 18 19; 27.09.2026: 1 5 8 10 11 12 14 17 19 20 22; 28.09.2026: 1 2 4 5 9 11 12 14 15 17 22
- broj 1 u svih 5; 12, 14, 17, 22 u 4 od 5; nema broja koji se nije pojavio; ponovljeno iz prethodnog kola: [6, 5, 3, 4, 7]; parni: [5, 6, 5, 6, 5]; zbrojevi: [122, 131, 96, 139, 112]; najdulji niz: [3, 4, 4, 3, 2].
- Lunarni datum: lunar_python 丙午 god., mj. 10 (十), dan 15 (十五), sat 戌; sxtwl mj. 10 (prijestupni=False), dan 15 – slažu se.

## 4. Statistika (walk-forward, kola 100–635, n = 536)

Očekivano (n·P(k)): 1:0,1, 2:2,3, 3:20,7, 4:82,7, 5:162,2, 6:162,2, 7:82,7, 8:20,7, 9:2,3, 10:0,1

| Metoda | dobitni događaji (oček. 46,2) | 11/0 | 10/1 | 9/2 | 8/3 | z | p (jednostrano) | p Bonf. (×9) | prosj. pogodaka | p Bonf. (prosjek) | χ² raspodjele, p Bonf. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vrući (cijela povijest) | 55 | 0 | 1 | 1 | 53 | 1,36 | 0,102 | 0,914 | 5,481 | 1,00 | 5,85; 1,00 |
| vrući (zadnjih 50) | 52 | 0 | 0 | 3 | 49 | 0,90 | 0,203 | 1,000 | 5,506 | 1,00 | 6,76; 1,00 |
| vrući (zadnjih 100) | 47 | 0 | 0 | 3 | 44 | 0,13 | 0,470 | 1,000 | 5,478 | 1,00 | 0,62; 1,00 |
| hladni (cijela povijest) | 55 | 0 | 1 | 1 | 53 | 1,36 | 0,102 | 0,914 | 5,519 | 1,00 | 5,85; 1,00 |
| Pareto 20/20 | 47 | 0 | 0 | 3 | 44 | 0,13 | 0,470 | 1,000 | 5,431 | 1,00 | 6,39; 1,00 |
| prezreli (najdulje neizvučeni) | 43 | 0 | 0 | 1 | 42 | -0,49 | 0,708 | 1,000 | 5,466 | 1,00 | 4,01; 1,00 |
| zadnje kolo ponovljeno | 43 | 0 | 0 | 1 | 42 | -0,49 | 0,708 | 1,000 | 5,534 | 1,00 | 4,01; 1,00 |
| komplement zadnjeg kola | 43 | 0 | 0 | 1 | 42 | -0,49 | 0,708 | 1,000 | 5,466 | 1,00 | 4,01; 1,00 |
| slučajna (secrets) – kontrola | 52 | 0 | 0 | 5 | 47 | 0,90 | 0,203 | 1,000 | 5,580 | 1,00 | 3,76; 1,00 |

**Nijedna metoda nije značajno bolja od slučaja** (sve p nakon Bonferronija ≥ 0,91). S1–S3 su zato samo „statistički tipične” kombinacije: tri najbolje rangirane metode (po z dobitnih događaja; vrući/cijela povijest i hladni/cijela povijest su komplementarne pa imaju identičan rezultat) primijenjene na svih 635 kola, uz prilagodbu na zbroj 116,5–136,5, parnost 5:6/6:5 i pravila §2.

Testovi slučajnosti (635 kola):
- χ² frekvencija 22 broja (ispravljen za izvlačenje bez vraćanja, df 21): 23,39, p = 0,324 (naivni χ²: 12,25, p = 0,933). Raspon frekvencija 287–340, očekivano 317,5.
- Stabilnost vrućih brojeva (frekvencije 1. vs 2. polovice, Spearman): ρ = 0,038, p = 0,867 – nema stabilnosti.
- Parnost (broj parnih, hipergeometrijska): opaženo [35, 98, 212, 172, 86, 32] vs očekivano [27.3, 98.0, 192.1, 192.1, 98.0, 27.3] (≤3, 4, 5, 6, 7, ≥8), χ² = 8,58, df 5, p = 0,127.
- Zbroj: prosjek 126,59 (teorija 126,5), sd 15,63 (teorija 15,23), z = 0,15; χ² po decilima točne raspodjele = 2,46, df 9, p = 0,982.

## 5. Backtest 9 kombinacija (635 kola)

| # | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 11/0 | 10/1 | 9/2 | 8/3 | ukupno |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 0 | 0 | 3 | 27 | 86 | 169 | 231 | 92 | 23 | 3 | 1 | 0 | 0 | 1 | 6 | 50 | 57 |
| E2 | 0 | 0 | 6 | 15 | 91 | 194 | 197 | 106 | 21 | 5 | 0 | 0 | 0 | 0 | 11 | 36 | 47 |
| E3 | 0 | 0 | 1 | 22 | 108 | 167 | 187 | 125 | 24 | 1 | 0 | 0 | 0 | 0 | 2 | 46 | 48 |
| S1 | 0 | 0 | 2 | 19 | 80 | 178 | 212 | 106 | 32 | 5 | 1 | 0 | 0 | 1 | 7 | 51 | 59 |
| S2 | 0 | 0 | 1 | 29 | 115 | 201 | 178 | 95 | 14 | 2 | 0 | 0 | 0 | 0 | 3 | 43 | 46 |
| S3 | 0 | 0 | 2 | 26 | 98 | 184 | 182 | 110 | 30 | 2 | 1 | 0 | 0 | 1 | 4 | 56 | 61 |
| L1 | 0 | 0 | 5 | 20 | 102 | 168 | 208 | 106 | 22 | 4 | 0 | 0 | 0 | 0 | 9 | 42 | 51 |
| L2 | 0 | 0 | 0 | 25 | 110 | 184 | 190 | 105 | 18 | 3 | 0 | 0 | 0 | 0 | 3 | 43 | 46 |
| L3 | 0 | 0 | 2 | 16 | 92 | 209 | 203 | 86 | 22 | 5 | 0 | 0 | 0 | 0 | 7 | 38 | 45 |

- Ukupno dobitnih događaja: 460 (očekivano 492,1); 1000 slučajnih setova: prosjek 492,1, sd 22,1 → 7,8. percentil.
- Kola s barem jednim dobitkom: 340/635 (53,5 %); 200 slučajnih setova prosjek 352,4 → 20,2. percentil.
- Povrat prema ulogu: **nije izračunato** (isplate nisu potvrđene). Ulog za usporedbu: 635 kola × 11,70 € = 7.429,50 €.
- S1–S3 su odabrani i prilagođeni na istim podacima (in-sample), pa njihov backtest nije neovisan. Kombinacije nisu mijenjane prema backtestu.

## 6. Izvori, verzije, RNG, nije izračunato, pretpostavke

**Podaci:** priloženi `sve_ili_nista_sva_kola.csv` (separator `;`) i `Sve_ili_Nista_sva_kola.xlsx` (list „Izvlačenja”, pročitan i list „Napomene”). CSV i XLSX su identični. Ima 636 redaka, od kojih 635 s brojevima, od 01.01.2025. do 28.09.2026. Sve provjere su prošle: 11 različitih brojeva 1–22 po kolu, Zbroj i Parni odgovaraju brojevima, nema duplikata datuma, a broj kola je redni dan u godini. Kolo 178/2026 (27.06.2026.) je izostavljeno.

**Nova kola 29.09.–22.11.2026.: nisu dohvaćena.** lotteryextreme.com, sveilinista.hr i lutrija.hr blokira mrežna politika ove sesije (proxy 403 / DNS). Rađeno je samo s datotekom, pa nedostaje 55 kola. Zato je „zadnjih 5 kola” za sinkronicitet = 24.–28.09.2026.

**Ostali izvori i verzije:**
- Dobitni razredi i isplate nisu potvrđeni (lutrija.hr nedostupna). **Pretpostavka:** razredi 11/0, 10/1, 9/2, 8/3. Iznosi isplata nisu korišteni. Ulog je 1,30 € × 9 = 11,70 €.
- Vrijeme izvlačenja nije pronađeno. **Pretpostavka:** 20:00 Europe/Zagreb (CET) = 19:00 UT, JD 2461368,29167.
- Swiss Ephemeris 2.10.03: pyswisseph 2.10.3.2 izgrađen iz izvornog koda jer pip build nije uspio. Datoteke sepl_18.se1 i semo_18.se1 preuzete su s github.com/aloistr/swisseph (`ephe/`). Za svaki položaj provjereno je da je korišten FLG_SWIEPH (bez prelaska na Moshier). Ayanamsa Lahiri iznosi 24,2328°.
- Izlazak i zalazak Sunca: `swe.rise_trans`, 45,815° N, 15,982° E, 120 m, gornji rub uz refrakciju.
- Verzije: Python 3.13.18, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, lunar_python 1.4.9, sxtwl 2.0.7, openpyxl 3.1.5.

**RNG:** `secrets.SystemRandom` (CSPRNG operacijskog sustava) koristi se za novčiće I Chinga, tarot, Lenormand, slučajnu kontrolnu metodu, 1000 slučajnih setova i kandidate u pretrazi L. Rezultat nije ponovljiv, pa su izvučene vrijednosti zapisane u tablici 3. Skripta nije ponovno pokretana radi „boljih” izvlačenja.

**Nije izračunato:**
- Kamea Mjeseca (Agrippa, *De occulta philosophia* II, 22): izvor se nije mogao dohvatiti ni potvrditi (mreža blokirana). Po pravilu je sustav izostavljen iz Plečkova filtra (glasa 12 sustava) i iz E3.
- Osobni dan: nema datuma rođenja.
- Povrat prema ulogu: isplate nisu potvrđene.
- Kola nakon 28.09.2026.

**Konvencije i pretpostavke:**
- Svođenje: od brojeva > 22 oduzima se 22. Pravilo 0 → 22 bilo je u kodu, ali nije zatrebalo.
- „Godina 2026 → 1” uzeta je doslovno kako je zadano (pravilom −22 bilo bi 2).
- Lunarni datum i Meihua računaju se po lokalnom datumu 23.11.2026.; u Pekingu je tada već 24.11. u 03:00. Dobiveno je: 10. mjesec, 15. dan, H = 戌 = 11. Trigrami su po Xiantian redu (Qian 1 … Kun 8), heksagrami po King Wen redu (tablica provjerena u kodu: 64 jedinstvena).
- I Ching popis: osnovni, promijenjeni i nuklearni heksagram, promjenjive crte, gornji i donji trigram (Xiantian broj). Meihua popis: osnovni, promijenjeni i nuklearni heksagram, gornji trigram, donji trigram, crta, Y, M, D, H.
- Tarot indeksi: 1–21 velika arkana, 22 Luda, 23–36 Štapovi, 37–50 Pehari, 51–64 Mačevi, 65–78 Novčići. Dekanske karte su po Golden Dawn: Sunce je u 1. dekanu Strijelca, što daje 8 štapova.
- Planetarni sat: Sunce 1, Mjesec 2, Mars 3, Merkur 4, Jupiter 5, Venera 6, Saturn 7. Popis: vladar sata, redni sat (1–24), noćni sat, vladar dana, sat i minuta izlaska, sat i minuta zalaska. 20:00 je oko 30 s prije kraja 3. noćnog sata (granični slučaj).
- Plečko: brojevi bez glasova bili bi na kraju, uzlazno; takvih nije bilo.
- Slaganje E: ako zamjena zadnjeg dodanog broja ne pomaže, zamjenjuje se prethodni. E2: 6 → 7, jer je parnost bila 7:4. E3 ostaje 3:8 (meko pravilo).
- Metode S:
  - vrući = 11 najčešćih (pri izjednačenju manji broj);
  - hladni = 11 najrjeđih, što je komplement vrućih iz iste povijesti, pa je rezultat identičan;
  - Pareto = 4 najčešća u zadnjih 20 % kola + popuna vrućima iz cijele povijesti;
  - prezreli = najdulji razmak od zadnjeg pojavljivanja.
  - Kontrolna slučajna metoda isključena je iz odabira. Bonferroni m = 9.
- Prilagodbe S (sljedeći brojevi po rangu iste metode, radi tipičnosti i §2):
  - S1: 17 → 22 (zbroj 115 → 120, parnost 4:7 → 5:6);
  - S2: 12 → 5 i 21 → 19 (zbroj 138 → 129, parnost 7:4 → 6:5);
  - S3: 4 → 22 (zbroj 102 → 120).
- L1–L3 također poštuju meko pravilo.
- χ² frekvencija: statistika Σ(O−E)²/(n·¼·22/21), df 21, zbog negativne korelacije brojeva unutar kola.

## 7. Napomena

**Nijedna metoda ne mijenja vjerojatnost pojedine kombinacije ni očekivani povrat.** Svaka kombinacija ima P(11/0) = 2/705.432, a P(bilo koji dobitak) = 60.744/705.432 = 8,61 %. Za svaki set od 9 kombinacija očekivani broj dobitnih kombinacija je 0,775. Odabir mijenja samo raspored rizika, primjerice koliko se dobitni događaji preklapaju unutar seta.
