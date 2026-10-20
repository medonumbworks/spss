# Sve ili Ništa – kombinacije za utorak 20.10.2026. (kolo 293/2026)

> **Kratki odgovor.** Ne postoji kombinacija s 30 %, 50 % ili 99 % šanse. Svaki listić u ovoj igri ima točno istu vjerojatnost glavnog dobitka, **2 od 705 432 = 1 : 352 716 (0,00028 %)**, bez obzira na to kako su brojevi odabrani. Analizirana su 656 izvlačenja: preko 40 statističkih testova, šest skeptičnih provjera, dva neovisna backtesta i provjera izvan uzorka na 21 novih kola. Nijedan obrazac ne predviđa iduće izvlačenje. Šansa raste **samo s brojem različitih listića**, linearno: N listića = N / 352 716. Zato su kombinacije ispod poredane kao popis pokrića: igrate li prvih N redaka, vaša je šansa točno ona u stupcu „Kumulativna šansa”.

Izvlačenje: utorak 20.10.2026. u 19:00, uplata do 18:55. Cijena kombinacije: 1,30 €. Glavni dobitak: 80 000 € (11/11 ili 0/11).

## 0. Kako je prošlo kolo 292

Izvučeno je ` 3  4  5  7  8  9 10 12 14 15 21`. Pogoci prvih 20 redaka jučerašnjeg popisa:

| Red | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pogodaka | 5 | 5 | 6 | 7 | 5 | 6 | 6 | 5 | 4 | 4 | 7 | 6 | 5 | 7 | 7 | 6 | 6 | 6 | 5 | 6 |

- Prvih 20 redaka: bez dobitka.
- Prvih 100 redaka: red 56 (9 pogodaka, 26 €*).
- Prvih 1000 redaka: red 56 (9 pogodaka, 26 €*), red 219 (2 pogotka, 26 €*), red 308 (9 pogodaka, 26 €*), red 401 (9 pogodaka, 26 €*), red 427 (2 pogotka, 26 €*), red 551 (9 pogodaka, 26 €*) ….
- Glavni dobitak (0 ili 11 pogodaka) bio je u retku 165 643 popisa, uz kumulativni ulog 215 335,90 €.
- \* Iznosi 260 € i 26 € navedeni su prema neslužbenim izvorima.

## 1. Kombinacije za kolo 293

Generirane kriptografski sigurnim generatorom (CSPRNG): 32 bajta entropije iz `os.urandom` (Python `secrets`), proširena SHA-256 DRBG-om bez pristranosti. Zapisana entropija omogućuje svakome da reproducira i provjeri cijeli popis:

```
python3 coverage.py kombinacije.csv --entropy f828f1825383d149f9955298d7082eed09c9e0190aeaa06c36fedf9b598ebac9
```

Popis sadrži svih 352 716 kombinacija tako da je svaki od 705 432 mogućih ishoda pokriven točno jednom (kombinacija ili njezin komplement, nikad oboje), u nasumičnom redoslijedu. Prvih 20 redaka:

| Red | Kombinacija | Zbroj | Parni | Kumulativna šansa | Kumulativni ulog |
| --- | --- | --- | --- | --- | --- |
| 1 | ` 4  5  6  7  9 12 14 15 16 18 19` | 125 | 6 | 0,00028 % | 1,30 € |
| 2 | ` 1  4  5  6 10 11 12 13 18 21 22` | 123 | 6 | 0,00057 % | 2,60 € |
| 3 | ` 2  4  5  9 10 11 16 17 20 21 22` | 137 | 6 | 0,00085 % | 3,90 € |
| 4 | ` 2  5  7  9 11 12 13 15 18 19 20` | 131 | 4 | 0,00113 % | 5,20 € |
| 5 | ` 1  2  3  5  6  7  8 18 19 20 22` | 111 | 6 | 0,00142 % | 6,50 € |
| 6 | ` 4  5  6  7 12 14 15 17 19 21 22` | 142 | 5 | 0,00170 % | 7,80 € |
| 7 | ` 2  3  4  6  8 10 12 15 18 19 21` | 118 | 7 | 0,00198 % | 9,10 € |
| 8 | ` 1  2  4  7  9 13 15 16 19 20 22` | 128 | 5 | 0,00227 % | 10,40 € |
| 9 | ` 3  6  7  8 11 12 14 15 16 19 22` | 133 | 6 | 0,00255 % | 11,70 € |
| 10 | ` 1  3  4  5  9 10 13 14 17 18 22` | 116 | 5 | 0,00284 % | 13,00 € |
| 11 | ` 6  8  9 10 13 14 16 18 19 20 21` | 154 | 7 | 0,00312 % | 14,30 € |
| 12 | ` 3  4  5  6 11 13 16 17 18 19 20` | 132 | 5 | 0,00340 % | 15,60 € |
| 13 | ` 2  4  5  6  7  8 10 13 16 20 22` | 113 | 8 | 0,00369 % | 16,90 € |
| 14 | ` 2  6  8  9 13 14 16 18 20 21 22` | 149 | 8 | 0,00397 % | 18,20 € |
| 15 | ` 2  4  5  7  9 11 12 13 14 20 21` | 118 | 5 | 0,00425 % | 19,50 € |
| 16 | ` 3  4  5  7  8  9 10 16 18 20 22` | 122 | 7 | 0,00454 % | 20,80 € |
| 17 | ` 1  3  4  5  6 12 13 14 17 18 21` | 114 | 5 | 0,00482 % | 22,10 € |
| 18 | ` 1  4  6  7  8 11 12 13 18 19 21` | 120 | 5 | 0,00510 % | 23,40 € |
| 19 | ` 1  2  3  4  8 10 11 12 13 15 19` | 98 | 5 | 0,00539 % | 24,70 € |
| 20 | ` 3  7  9 11 12 14 17 18 20 21 22` | 154 | 5 | 0,00567 % | 26,00 € |

Cijeli popis je u prilogu (CSV, 352 716 redaka, stupci kumulativne šanse i uloga).

### 1.1 Koliko listića za koju šansu

| Listića (prvih N redaka) | Šansa glavnog dobitka | Ulog | Očekivani povrat, samo glavni dobitak | Očekivani povrat, sve kategorije* | Očekivani gubitak* |
| --- | --- | --- | --- | --- | --- |
| 1 | 0,0003 % | 1,30 € | 0,23 € | 0,54 € | 0,76 € |
| 5 | 0,0014 % | 6,50 € | 1,13 € | 2,69 € | 3,81 € |
| 11 | 0,0031 % | 14,30 € | 2,49 € | 5,93 € | 8,37 € |
| 20 | 0,0057 % | 26,00 € | 4,54 € | 10,78 € | 15,22 € |
| 100 | 0,0284 % | 130,00 € | 22,68 € | 53,90 € | 76,10 € |
| 1 000 | 0,2835 % | 1 300,00 € | 226,81 € | 538,99 € | 761,01 € |
| 10 000 | 2,84 % | 13 000,00 € | 2 268,11 € | 5 389,89 € | 7 610,11 € |
| 35 272 | 10,00 % | 45 853,60 € | 8 000,09 € | 19 011,22 € | 26 842,38 € |
| 105 815 | 30,00 % | 137 559,50 € | 24 000,05 € | 57 033,11 € | 80 526,39 € |
| 176 358 | 50,00 % | 229 265,40 € | 40 000,00 € | 95 055,00 € | 134 210,40 € |
| 246 902 | 70,00 % | 320 972,60 € | 56 000,18 € | 133 077,43 € | 187 895,17 € |
| 317 445 | 90,00 % | 412 678,50 € | 72 000,14 € | 171 099,32 € | 241 579,18 € |
| 349 189 | 99,00 % | 453 945,70 € | 79 200,04 € | 188 208,99 € | 265 736,71 € |
| 352 716 | 100,00 % | 458 530,80 € | 80 000,00 € | 190 110,00 € | 268 420,80 € |

\* Kategorije 260 € (10 ili 1 pogodak) i 26 € (9 ili 2 pogotka) navode samo sekundarni izvori. Nisu potvrđene u pravilima Hrvatske Lutrije. Na svakoj razini očekivani povrat je 17,4 % uloga samo s glavnim dobitkom, odnosno 41,5 % sa svim kategorijama. Više listića povećava šansu, ali i očekivani gubitak u istom omjeru.

### 1.2 Varijante na temelju povijesti (za usporedbu)

Ove su kombinacije također izvučene CSPRNG-om, ali s težinama prema povijesti. Imaju **istu** šansu 1 : 352 716 kao bilo koji redak gore. Navedene su samo zato što ste tražili odabir „baziran na podacima”.

| Oznaka | Težina | Kombinacija | Zbroj | Parni |
| --- | --- | --- | --- | --- |
| H1 | vrući brojevi, zadnjih 60 kola | ` 1  2  5  6  8 11 14 15 16 21 22` | 121 | 6 |
| H2 | vrući brojevi, zadnjih 30 kola | ` 2  7  8 10 11 14 15 17 19 21 22` | 146 | 5 |
| Z1 | „zakašnjeli” brojevi | ` 1  2  3  7 11 13 15 16 18 19 20` | 125 | 4 |
| C1 | ukupno najrjeđi brojevi | ` 1  2  4  6  9 10 11 12 17 18 21` | 111 | 6 |
| S1 | zbroj 116–137 i 5–6 parnih (središnjih 50 %) | ` 1  4  5  8 11 12 15 16 18 21 22` | 133 | 6 |

**Pravila koja stvarno štede novac:**
- Nikad ne igrajte kombinaciju i njezin komplement (brojeve koji u njoj nisu). Imaju identične ishode u svim kategorijama, pa plaćate dvaput za istu šansu.
- Nikad ne igrajte istu kombinaciju dvaput. Popis u prilogu jamči oboje.
- „Vrući”, „hladni” i „zakašnjeli” brojevi ne mijenjaju šansu (odjeljci 4 i 5).

## 2. Matematika igre

| Kategorija | Ishoda od 705 432 | Šansa po listiću | Iznos | Očekivana vrijednost |
| --- | --- | --- | --- | --- |
| 11 ili 0 pogodaka | 2 | 0,00028 % (1 : 352 716) | 80 000 € | 0,2268 € |
| 10 ili 1 pogodak* | 242 | 0,03431 % (1 : 2 915) | 260 € | 0,0892 € |
| 9 ili 2 pogotka* | 6 050 | 0,85763 % (1 : 117) | 26 € | 0,2230 € |

Broj pogodaka po listiću prati hipergeometrijsku razdiobu: očekivano 5,5, standardna devijacija 1,2002. Šansa za ≤3 ili ≥8 pogodaka je 8,61 %.

## 3. Podaci

- **Priloženi podaci:** 635 kola s brojevima, od 01.01.2025. do 28.09.2026. (kolo 271). Kolo 178/2026 (27.06.2026.) nema brojeva. CSV i XLSX su identični, a stupci Zbroj i Parni slažu se s brojevima.
- **Dopuna 272–292:** kola su pronađena pretragom javnih izvora i provjerena drugim, neovisnim pretraživanjem. Nijedan izvor nije službena stranica Hrvatske Lutrije, pa su dopunjena kola privremena.
- **Analiza ukupno:** 656 kola, zadnje 19.10.2026. (kolo 292/2026).

| Kolo | Datum | Brojevi | Status | Izvori |
| --- | --- | --- | --- | --- |
| 272 | 29.09.2026 | ` 3  5  8 12 14 15 16 17 20 21 22` | brojevi iz 1 izvora | lotteryextreme.com, sveilinista.hr |
| 273 | 30.09.2026 | ` 1  4  5  9 12 14 17 18 19 21 22` | brojevi iz 2+ izvora | lotteryextreme.com, sveilinista.hr |
| 274 | 01.10.2026 | ` 1  5 10 11 12 15 16 17 20 21 22` | brojevi iz 2+ izvora | sveilinista.hr, lotteryextreme.com |
| 275 | 02.10.2026 | ` 3  5  8 10 13 14 16 18 19 20 22` | brojevi iz 1 izvora | lotteryextreme.com, sveilinista.hr |
| 276 | 03.10.2026 | ` 4  7  9 10 11 12 13 14 17 19 21` | brojevi iz 2+ izvora | sveilinista.hr, lotteryextreme.com |
| 277 | 04.10.2026 | ` 1  8  9 10 14 15 16 18 20 21 22` | brojevi iz 2+ izvora | lotteryextreme.com, sveilinista.hr |
| 278 | 05.10.2026 | ` 1  2  4  7  8 11 12 14 15 19 20` | brojevi iz 1 izvora | lotteryextreme.com, sveilinista.hr |
| 279 | 06.10.2026 | ` 1  2  5  8  9 13 14 15 16 18 20` ⚠ | brojevi iz 2 izvora (dodatna provjera) | lotteryextreme.com, sveilinista.hr |
| 280 | 07.10.2026 | ` 2  6  9 10 13 14 15 17 19 20 21` | brojevi iz 2+ izvora | lotteryextreme.com, sveilinista.hr |
| 281 | 08.10.2026 | ` 1  3  7  9 10 11 12 13 15 16 17` | brojevi iz 2+ izvora | zadarskilist.novilist.hr, lotteryextreme.com, zagorje-international.hr |
| 282 | 09.10.2026 | ` 1  3  5  7  8  9 11 12 15 19 20` | brojevi iz 2+ izvora | sveilinista.hr, lotteryextreme.com |
| 283 | 10.10.2026 | ` 5  7  8 11 14 15 16 17 18 19 20` | brojevi iz 2+ izvora | sveilinista.hr, lotteryextreme.com |
| 284 | 11.10.2026 | ` 3  5 11 12 13 15 17 18 19 20 21` | brojevi iz 2+ izvora | sveilinista.hr, lotteryextreme.com |
| 285 | 12.10.2026 | ` 2  3  4  9 11 13 15 16 17 18 21` | brojevi iz 2+ izvora | sveilinista.hr, lotteryextreme.com |
| 286 | 13.10.2026 | ` 1  3  7  8 10 11 12 13 14 18 22` | brojevi iz 2+ izvora | glasistre.hr, rijekadanas.com, dalmacijanews.hr |
| 287 | 14.10.2026 | ` 1  2  5  9 10 12 13 16 20 21 22` | brojevi iz 1 izvora | lotteryextreme.com |
| 288 | 15.10.2026 | ` 1  2  4  5  6 10 12 13 18 20 22` | brojevi iz 2+ izvora | net.hr, emedjimurje.net.hr, 023.hr |
| 289 | 16.10.2026 | ` 3  5  7  8  9 10 11 14 17 19 21` | brojevi iz 1 izvora | lotteryextreme.com |
| 290 | 17.10.2026 | ` 4  5  7  9 10 11 12 13 19 21 22` | brojevi iz 1 izvora | lotteryextreme.com |
| 291 | 18.10.2026 | ` 1  2  6  7 11 12 13 16 17 19 22` | snimka zaslona korisnika | snimka zaslona korisnika |
| 292 | 19.10.2026 | ` 3  4  5  7  8  9 10 12 14 15 21` | snimka zaslona korisnika | snimka zaslona korisnika |

⚠ Kolo 279 (06.10.) identično je izvlačenju od 23.06.2026. Slučajno ponavljanje jednog od 21 novih kola ima vjerojatnost oko 1,9 %. Brojeve daje lotteryextreme.com. Arhiva sveilinista.hr dala je iste brojeve u dodatnoj pretrazi bez brojeva u upitu. Oba su agregatori, a službeni izvor nije dostupan, pa je moguća greška u izvoru. Na zaključke to ne utječe.

## 4. Što pokazuju 656 izvlačenja

**Frekvencija brojeva.** Očekivano je 328,0 izvlačenja po broju. Hi-kvadrat s korekcijom za izvlačenje bez vraćanja iznosi 21,70 (df 21, p = 0,417). Najveće odstupanje (|z| = 2,26) uobičajeno je za 22 broja: Monte Carlo p = 0,439.

| Broj | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Puta | 334 | 320 | 338 | 316 | 332 | 322 | 350 | 333 | 327 | 306 | 324 | 314 | 299 | 315 | 350 | 343 | 338 | 328 | 345 | 324 | 329 | 329 |
| z | 0,47 | -0,62 | 0,78 | -0,94 | 0,31 | -0,47 | 1,72 | 0,39 | -0,08 | -1,72 | -0,31 | -1,09 | -2,26 | -1,02 | 1,72 | 1,17 | 0,78 | 0,00 | 1,33 | -0,31 | 0,08 | 0,08 |
| Razmak | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 4 | 1 | 4 | 0 | 1 |

Razmak je broj kola od zadnjeg pojavljivanja, stanje nakon 19.10.2026.

- Najčešći ukupno: 7, 15, 19, 16, 17, 3, 1, 8, 5, 22, 21
- Najrjeđi ukupno: 13, 10, 12, 14, 4, 2, 6, 20, 11, 9, 18
- Vrući u zadnjih 30 kola: 1, 5, 9, 10, 11, 12, 14, 15, 17, 19, 22
- „Zakašnjeli”: 18, 20, 1, 2, 6, 11, 13, 16, 17, 19, 22
- Brojevi s jednakim brojem izvlačenja ili jednakim razmakom razvrstani su proizvoljno.

**Pogađaju li vrući ili hladni brojevi iduće kolo?** Za svako kolo uzeto je 11 najčešćih ili najrjeđih brojeva iz prethodnih w kola. Očekivano je 5,5 pogodaka.

| Strategija | Kola | Prosjek pogodaka | z | p | Udio ekstrema (≤3 ili ≥8) |
| --- | --- | --- | --- | --- | --- |
| vrući, zadnjih 30 | 626 | 5,414 | -1,80 | 0,072 | 0,0703 |
| hladni, zadnjih 30 | 626 | 5,559 | 1,23 | 0,218 | 0,0815 |
| vrući, zadnjih 60 | 596 | 5,497 | -0,07 | 0,946 | 0,0906 |
| hladni, zadnjih 60 | 596 | 5,529 | 0,58 | 0,562 | 0,0856 |
| vrući, zadnjih 90 | 566 | 5,477 | -0,46 | 0,649 | 0,0795 |
| hladni, zadnjih 90 | 566 | 5,500 | 0,00 | 1,000 | 0,0972 |

Nijedna varijanta nije značajna. Teorijski udio ekstrema je 0,0861, a samo ekstremi vode do dobitka 0/11 ili 11/11.

**„Zakašnjeli” brojevi.** Šansa pojave je oko 0,50 bez obzira na to koliko dugo broj nije izvučen (hi-kvadrat 4,56, df 4, p = 0,335).

| Razmak prije kola | Izvučen | Prilika | Udio |
| --- | --- | --- | --- |
| 0 | 3645 | 7205 | 0,5059 |
| 1 | 1733 | 3551 | 0,4880 |
| 2 | 926 | 1818 | 0,5094 |
| 3 | 432 | 892 | 0,4843 |
| 4+ | 458 | 912 | 0,5022 |

**Ovisnost među kolima.** Prosječno preklapanje uzastopnih kola je 5,565, uz očekivano 5,5 i standardnu pogrešku 0,047. P(izvučen | izvučen jučer) = 0,5059, a P(izvučen | nije jučer) = 0,4941.

**Dan u tjednu.** Korigirani hi-kvadrat za 22 broja × 7 dana iznosi 139,66 (df 126, p = 0,191). Utorak u odnosu na ostale dane: 21,02 (df 21, p = 0,458). Uz korekciju za 7 dana to je p ≈ 1,00, dakle bez efekta tog dana.

**Struktura izvlačenja u odnosu na točnu teoriju** (nabrojanih svih 705 432 kombinacija):

| Statistika | Teorija: prosjek | Teorija: sd | Opaženo: prosjek | Opaženo: sd | p |
| --- | --- | --- | --- | --- | --- |
| Zbroj | 126,50 | 15,23 | 126,77 | 15,65 | 0,091 |
| Parnih | 5,50 | 1,20 | 5,41 | 1,24 | 0,095 |
| Niskih (≤11) | 5,50 | 1,20 | 5,49 | 1,26 | 0,751 |
| Uzastopnih parova | 5,00 | 1,20 | 4,99 | 1,21 | 0,896 |

**Parovi.** Najčešći par je 7-15 (191 puta), najrjeđi 13-20 (123 puta). Očekivano je 156,2. Najveći |z| među 231 parom iznosi 3,19, što je uobičajeno pod slučajnošću (Monte Carlo p = 0,28).

**Ponavljanja.** Identičnih izvlačenja: 2 (06.12.2025. = 09.08.2026.; 23.06.2026. = 06.10.2026.), uz očekivano 0,305 među 656 kola. Vjerojatnost da se slučajno dogodi barem 2 ponavljanja je oko 3,8 %. Ponavljanje od 06.10.2026. počiva na neslužbenim izvorima (odjeljak 3).

## 5. Neovisne provjere

**Dvije neovisne analize** (frekvencije/parovi/razmaci te ovisnost/vrijeme/struktura) provele su preko 40 testova na 635 kola, uz Monte Carlo kalibraciju. Nijedan test ne ostaje značajan nakon korekcije za višestruka testiranja. Druga analiza imala je 0 od 30 p-vrijednosti ispod 0,05.

**Šest skeptika** dobilo je zadatak pobiti najjače uočene „pravilnosti”. Za svaku su neovisno ponovili izračun, ispravili za broj usporedbi i provjerili izvan uzorka (procjena na izvlačenjima 1–423, od 01.01.2025. do 27.02.2026., test na izvlačenjima 424–635, od 28.02. do 28.09.2026.):

| Tvrdnja | Presuda | Pouzdanost | Korigirani p | Ključni razlog |
| --- | --- | --- | --- | --- |
| Vrući brojevi zadnjih 30 kola nastavljaju dolaziti | pobijeno | visoka | 0,237 | Efekt ide u krivom smjeru (vrući pogađaju manje) i ovisi o tome kako se lome izjednačenja: obrnuto pravilo daje p = 0,29. Najbolja varijanta s treninga obrnula se na testu. |
| Trojka 11-13-20 potisnuta, 7-15-16 favorizirana | pobijeno | srednja | 0,057 | Favorizirani smjer nikad se ne ponavlja. Slaba postojanost potisnutih trojki potječe od broja 13, a 13 je u 21 novih kola izvučen 12 puta uz očekivano 10,5. |
| Broj 8 rjeđe dolazi 5 kola nakon svoje pojave | pobijeno | visoka | 0,150 | Na testu p = 0,15. Korelacija trening/test preko 154 ćelije iznosi 0,07. Listić po ovom pravilu imao je prosječno 5,51 pogodaka. |
| Par 7-15 „sretan”, par 13-20 „nesretan” | pobijeno | visoka | 0,231 | Uz frekvencije pojedinih brojeva par 7-15 ima samo z = 1,99 (5. od 231). Par 13-20 se u 21 novih kola pojavio 6 puta, uz očekivano 5,0. |
| Kola razmaknuta 28 dana dijele manje brojeva | pobijeno | visoka | 0,215 | Ne ponavlja se izvan uzorka: na točnom kalendarskom razmaku smjer je obrnut (z = +0,09). |
| Ponovljeno izvlačenje znači da stroj favorizira kombinacije | pobijeno | visoka | 1,000 | Jedno ponavljanje među 635 kola ima vjerojatnost 25 %. Igranje svih ranijih kombinacija dalo je prosječno 5,50 pogodaka. |

**Dva neovisna backtesta** (535 kola, svaki listić sastavljen samo iz prethodnih kola):

| Strategija | Prosjek pogodaka (1) | Prosjek pogodaka (2) | p (2) | Udio ekstrema (2) |
| --- | --- | --- | --- | --- |
| slučajno (300 listića/kolo) | 5,501 | 5,497 | 0,366 | 0,0856 |
| vrući, 30 kola | 5,370 | 5,370 | 0,012 | 0,0692 |
| vrući, 90 kola | 5,493 | 5,493 | 0,900 | 0,0841 |
| vrući, ukupno | 5,482 | 5,482 | 0,732 | 0,1028 |
| hladni, ukupno | 5,523 | 5,523 | 0,653 | 0,0972 |
| „zakašnjeli” | 5,465 | 5,465 | 0,505 | 0,0804 |
| ponovi zadnje kolo | 5,535 | 5,535 | 0,505 | 0,0804 |
| komplement zadnjeg | 5,465 | 5,465 | 0,505 | 0,0804 |
| ponderirano vrući 60 (300/kolo) | 5,496 | 5,501 | 0,856 | 0,0864 |
| ponderirano zakašnjeli (300/kolo) | 5,499 | 5,502 | 0,938 | 0,0859 |
| tipična struktura (300/kolo) | 5,501 | 5,494 | 0,061 | 0,0855 |

Jedini nominalni rezultat (vrući, 30 kola, p = 0,012) ne preživljava korekciju za 10 strategija (Holm p = 0,12). Ide prema *manje* pogodaka i smanjuje udio ekstrema, pa za ovu igru ne pomaže.

**Provjera na 21 novih kola** (272–292), koje nijedna od ranijih analiza (na 635 kola) nije vidjela. Listići su sastavljeni samo iz podataka do 28.09.:

| Listić iz starih podataka | Kombinacija | Prosjek pogodaka | z | Ekstremi |
| --- | --- | --- | --- | --- |
| vrući, zadnjih 30 | ` 1  3  4  5  7  8 12 14 17 20 22` | 5,667 | 0,64 | 3/21 |
| vrući, zadnjih 90 | ` 1  2  4  5  7  8 10 17 18 20 21` | 5,429 | -0,27 | 0/21 |
| vrući, ukupno | ` 1  3  5  6  7  8 15 16 17 18 19` | 5,190 | -1,18 | 4/21 |
| hladni, ukupno | ` 2  4  5  9 10 11 12 13 14 20 21` | 5,952 | 1,73 | 3/21 |
| „zakašnjeli” (= komplement zadnjeg kola) | ` 3  6  7  8 10 13 16 18 19 20 21` | 5,238 | -1,00 | 2/21 |
| ponovi zadnje kolo (28.09.) | ` 1  2  4  5  9 11 12 14 15 17 22` | 5,762 | 1,00 | 2/21 |
| najbolji po modelu frekvencija | ` 1  3  5  6  7  8 15 16 17 19 22` | 5,238 | -1,00 | 1/21 |

| Tvrdnja iz starih podataka | Na 21 novih kola | Očekivano slučajno |
| --- | --- | --- |
| Broj 13 je „hladan” | izvučen 12 puta | 10,5 |
| Broj 7 je „vruć” | izvučen 10 puta | 10,5 |
| Par 7-15 je „sretan” | zajedno 5 puta | 5,00 |
| Par 13-20 je „nesretan” | zajedno 6 puta | 5,00 |
| Trojka 7-15-16 je favorizirana | zajedno 2 puta | 2,25 |
| Trojka 11-13-20 je potisnuta | zajedno 1 put | 2,25 |
| Premalo parnih brojeva | prosječno 5,05 | 5,50 |
| Uzastopna kola dijele više brojeva | prosječno 5,48 | 5,50 |

Nijedna „pravilnost” se nije potvrdila na novim podacima.

## 6. Zaključak

- Izvlačenja su statistički nerazlučiva od savršeno slučajnog izvlačenja 11 od 22. To je očekivano za nadzirano lutrijsko izvlačenje.
- „Predviđanje” koje povećava šansu nije moguće. Ne mogu ga dati ni veći model ni više računanja, jer ga u podacima nema.
- Ako igrate, dovoljno je odabrati kombinacije koje se međusobno ne ponavljaju i nisu komplementi. CSPRNG popis u prilogu to jamči i poredan je po kumulativnoj šansi, ali nije bolji od bilo kojeg drugog takvog odabira.
- Za jedan listić uzmite red 1. Za N listića uzmite redove 1 do N.
- Očekivani povrat je negativan na svakoj razini uloga. Igrajte samo iznos koji ste spremni izgubiti. Ako igranje postane problem, obratite se liječniku obiteljske medicine ili Hrvatskom zavodu za javno zdravstvo.

## Prilozi

- `kombinacije_kolo293_pokrice.zip`: svih 352 716 kombinacija, CSPRNG redoslijed, s kumulativnom šansom i ulogom.
- `kombinacije_kolo293_prvih100.csv`: prvih 100 redaka istog popisa.
- `sve_ili_nista_prosireno.csv`: priloženi podaci plus dopunjena kola 272–292.
- Skripte: `coverage.py` (popis pokrića), `generate.py` (varijante), `analyze.py` (analiza), `oos_new.py` (provjera izvan uzorka).