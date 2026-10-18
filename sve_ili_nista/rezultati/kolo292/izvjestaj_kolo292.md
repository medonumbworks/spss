# Sve ili Ništa – kombinacije za ponedjeljak 19.10.2026. (kolo 292/2026)

> **Kratki odgovor.** Ne postoji kombinacija s 30 %, 50 % ili 99 % šanse. Svaki listić u ovoj igri ima točno istu vjerojatnost glavnog dobitka, **2 od 705 432 = 1 : 352 716 (0,00028 %)**, bez obzira na to kako su brojevi odabrani. Analizirana su 655 izvlačenja: preko 40 statističkih testova, šest skeptičnih provjera, dva neovisna backtesta i provjera izvan uzorka na 20 novih kola. Nijedan obrazac ne predviđa iduće izvlačenje. Šansa raste **samo s brojem različitih listića**, linearno: N listića = N / 352 716. Zato su kombinacije ispod poredane kao popis pokrića: igrate li prvih N redaka, vaša je šansa točno ona u stupcu „Kumulativna šansa”.

Izvlačenje: ponedjeljak 19.10.2026. u 19:00, uplata do 18:55. Cijena kombinacije: 1,30 €. Glavni dobitak: 80 000 € (11/11 ili 0/11).

## 0. Kako je prošlo kolo 291

Izvučeno je ` 1  2  6  7 11 12 13 16 17 19 22`. Pogoci prvih 20 redaka jučerašnjeg popisa:

| Red | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pogodaka | 4 | 6 | 4 | 3 | 7 | 7 | 6 | 7 | 5 | 8 | 6 | 4 | 7 | 5 | 4 | 6 | 2 | 5 | 5 | 8 |

- Prvih 20 redaka: red 17 (2 pogotka, 26 €*).
- Prvih 100 redaka: red 17 (2 pogotka, 26 €*).
- Prvih 1000 redaka: red 17 (2 pogotka, 26 €*), red 111 (2 pogotka, 26 €*), red 302 (9 pogodaka, 26 €*), red 374 (2 pogotka, 26 €*), red 556 (9 pogodaka, 26 €*), red 646 (2 pogotka, 26 €*) ….
- Glavni dobitak (0 ili 11 pogodaka) bio je u retku 207 480 popisa, uz kumulativni ulog 269 724,00 €.
- \* Iznosi 260 € i 26 € navedeni su prema neslužbenim izvorima.

## 1. Kombinacije za kolo 292

Generirane kriptografski sigurnim generatorom (CSPRNG): 32 bajta entropije iz `os.urandom` (Python `secrets`), proširena SHA-256 DRBG-om bez pristranosti. Zapisana entropija omogućuje svakome da reproducira i provjeri cijeli popis:

```
python3 coverage.py kombinacije.csv --entropy 39fdcf8a96a146a2f3c94ed2e4423439cbe125e8b1d97272b39a51ab190fcb9d
```

Popis sadrži svih 352 716 kombinacija tako da je svaki od 705 432 mogućih ishoda pokriven točno jednom (kombinacija ili njezin komplement, nikad oboje), u nasumičnom redoslijedu. Prvih 20 redaka:

| Red | Kombinacija | Zbroj | Parni | Kumulativna šansa | Kumulativni ulog |
| --- | --- | --- | --- | --- | --- |
| 1 | ` 2  4  6  8 12 14 16 18 19 20 21` | 140 | 9 | 0,00028 % | 1,30 € |
| 2 | ` 1  2  3  5  8 10 17 19 20 21 22` | 128 | 5 | 0,00057 % | 2,60 € |
| 3 | ` 3  4  6  9 10 11 13 14 16 18 21` | 125 | 6 | 0,00085 % | 3,90 € |
| 4 | ` 4  5  6  7  8  9 11 14 17 20 21` | 122 | 5 | 0,00113 % | 5,20 € |
| 5 | ` 4  6  7 11 12 15 17 19 20 21 22` | 154 | 5 | 0,00142 % | 6,50 € |
| 6 | ` 3  7 12 13 14 15 16 18 19 20 21` | 158 | 5 | 0,00170 % | 7,80 € |
| 7 | ` 3  6  7  8 10 11 13 14 15 18 20` | 125 | 6 | 0,00198 % | 9,10 € |
| 8 | ` 2  5  6  7 11 12 14 15 18 19 22` | 131 | 6 | 0,00227 % | 10,40 € |
| 9 | ` 2  6  7  8 10 11 14 16 18 20 22` | 134 | 9 | 0,00255 % | 11,70 € |
| 10 | ` 2  3  5  6  7 11 13 14 16 17 19` | 113 | 4 | 0,00284 % | 13,00 € |
| 11 | ` 3  4  7  8 11 12 15 16 17 19 21` | 133 | 4 | 0,00312 % | 14,30 € |
| 12 | ` 1  3  4  6  9 10 12 13 14 16 20` | 108 | 7 | 0,00340 % | 15,60 € |
| 13 | ` 1  2  4  7  9 13 14 15 16 18 20` | 119 | 6 | 0,00369 % | 16,90 € |
| 14 | ` 1  2  4  5  9 10 12 14 16 21 22` | 116 | 7 | 0,00397 % | 18,20 € |
| 15 | ` 4  5  7  8  9 10 11 15 16 20 22` | 127 | 6 | 0,00425 % | 19,50 € |
| 16 | ` 3  6  7  9 10 12 16 19 20 21 22` | 145 | 6 | 0,00454 % | 20,80 € |
| 17 | ` 2  5  6  8 11 12 14 15 17 18 21` | 129 | 6 | 0,00482 % | 22,10 € |
| 18 | ` 1  4  7  8 11 12 13 15 16 20 21` | 128 | 5 | 0,00510 % | 23,40 € |
| 19 | ` 5  6  8  9 10 12 13 16 17 19 22` | 137 | 6 | 0,00539 % | 24,70 € |
| 20 | ` 1  2  4  6  8  9 10 11 15 17 21` | 104 | 5 | 0,00567 % | 26,00 € |

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
| H1 | vrući brojevi, zadnjih 60 kola | ` 2  3  5  6  7  8 10 14 18 20 21` | 114 | 7 |
| H2 | vrući brojevi, zadnjih 30 kola | ` 1  2  5  7  8 12 13 16 17 18 19` | 118 | 5 |
| Z1 | „zakašnjeli” brojevi | ` 3  4  8  9 10 11 12 15 18 20 22` | 132 | 7 |
| C1 | ukupno najrjeđi brojevi | ` 2  8  9 10 11 12 13 18 20 21 22` | 146 | 7 |
| S1 | zbroj 116–137 i 5–6 parnih (središnjih 50 %) | ` 2  5  6  7  8  9 10 15 16 17 22` | 117 | 6 |

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
- **Dopuna 272–291:** kola su pronađena pretragom javnih izvora i provjerena drugim, neovisnim pretraživanjem. Nijedan izvor nije službena stranica Hrvatske Lutrije, pa su dopunjena kola privremena.
- **Analiza ukupno:** 655 kola, zadnje 18.10.2026. (kolo 291/2026).

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

⚠ Kolo 279 (06.10.) identično je izvlačenju od 23.06.2026. Slučajno ponavljanje jednog od 20 novih kola ima vjerojatnost oko 1,8 %. Brojeve daje lotteryextreme.com. Arhiva sveilinista.hr dala je iste brojeve u dodatnoj pretrazi bez brojeva u upitu. Oba su agregatori, a službeni izvor nije dostupan, pa je moguća greška u izvoru. Na zaključke to ne utječe.

## 4. Što pokazuju 655 izvlačenja

**Frekvencija brojeva.** Očekivano je 327,5 izvlačenja po broju. Hi-kvadrat s korekcijom za izvlačenje bez vraćanja iznosi 21,74 (df 21, p = 0,415). Najveće odstupanje (|z| = 2,23) uobičajeno je za 22 broja: Monte Carlo p = 0,485.

| Broj | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Puta | 334 | 320 | 337 | 315 | 331 | 322 | 349 | 332 | 326 | 305 | 324 | 313 | 299 | 314 | 349 | 343 | 338 | 328 | 345 | 324 | 328 | 329 |
| z | 0,51 | -0,59 | 0,74 | -0,98 | 0,27 | -0,43 | 1,68 | 0,35 | -0,12 | -1,76 | -0,27 | -1,13 | -2,23 | -1,05 | 1,68 | 1,21 | 0,82 | 0,04 | 1,37 | -0,27 | 0,04 | 0,12 |
| Razmak | 0 | 0 | 2 | 1 | 1 | 0 | 0 | 2 | 1 | 1 | 0 | 0 | 0 | 2 | 6 | 0 | 0 | 3 | 0 | 3 | 1 | 0 |

Razmak je broj kola od zadnjeg pojavljivanja, stanje nakon 18.10.2026.

- Najčešći ukupno: 7, 15, 19, 16, 17, 3, 1, 8, 5, 22, 18
- Najrjeđi ukupno: 13, 10, 12, 14, 4, 2, 6, 20, 11, 9, 21
- Vrući u zadnjih 30 kola: 1, 5, 10, 11, 12, 14, 15, 16, 17, 19, 22
- „Zakašnjeli”: 15, 18, 20, 3, 8, 14, 4, 5, 9, 10, 21
- Brojevi s jednakim brojem izvlačenja ili jednakim razmakom razvrstani su proizvoljno.

**Pogađaju li vrući ili hladni brojevi iduće kolo?** Za svako kolo uzeto je 11 najčešćih ili najrjeđih brojeva iz prethodnih w kola. Očekivano je 5,5 pogodaka.

| Strategija | Kola | Prosjek pogodaka | z | p | Udio ekstrema (≤3 ili ≥8) |
| --- | --- | --- | --- | --- | --- |
| vrući, zadnjih 30 | 625 | 5,413 | -1,82 | 0,069 | 0,0704 |
| hladni, zadnjih 30 | 625 | 5,558 | 1,22 | 0,224 | 0,0816 |
| vrući, zadnjih 60 | 595 | 5,494 | -0,12 | 0,905 | 0,0908 |
| hladni, zadnjih 60 | 595 | 5,533 | 0,67 | 0,505 | 0,0840 |
| vrući, zadnjih 90 | 565 | 5,478 | -0,44 | 0,661 | 0,0796 |
| hladni, zadnjih 90 | 565 | 5,501 | 0,02 | 0,986 | 0,0973 |

Nijedna varijanta nije značajna. Teorijski udio ekstrema je 0,0861, a samo ekstremi vode do dobitka 0/11 ili 11/11.

**„Zakašnjeli” brojevi.** Šansa pojave je oko 0,50 bez obzira na to koliko dugo broj nije izvučen (hi-kvadrat 4,75, df 4, p = 0,314).

| Razmak prije kola | Izvučen | Prilika | Udio |
| --- | --- | --- | --- |
| 0 | 3643 | 7194 | 0,5064 |
| 1 | 1728 | 3546 | 0,4873 |
| 2 | 923 | 1815 | 0,5085 |
| 3 | 432 | 890 | 0,4854 |
| 4+ | 457 | 911 | 0,5016 |

**Ovisnost među kolima.** Prosječno preklapanje uzastopnih kola je 5,570, uz očekivano 5,5 i standardnu pogrešku 0,047. P(izvučen | izvučen jučer) = 0,5064, a P(izvučen | nije jučer) = 0,4936.

**Dan u tjednu.** Korigirani hi-kvadrat za 22 broja × 7 dana iznosi 139,73 (df 126, p = 0,190). Ponedjeljak u odnosu na ostale dane: 10,70 (df 21, p = 0,968). Uz korekciju za 7 dana to je p ≈ 1,00, dakle bez efekta tog dana.

**Struktura izvlačenja u odnosu na točnu teoriju** (nabrojanih svih 705 432 kombinacija):

| Statistika | Teorija: prosjek | Teorija: sd | Opaženo: prosjek | Opaženo: sd | p |
| --- | --- | --- | --- | --- | --- |
| Zbroj | 126,50 | 15,23 | 126,80 | 15,64 | 0,100 |
| Parnih | 5,50 | 1,20 | 5,41 | 1,24 | 0,101 |
| Niskih (≤11) | 5,50 | 1,20 | 5,49 | 1,26 | 0,774 |
| Uzastopnih parova | 5,00 | 1,20 | 4,99 | 1,21 | 0,877 |

**Parovi.** Najčešći par je 7-15 (190 puta), najrjeđi 13-20 (123 puta). Očekivano je 155,9. Najveći |z| među 231 parom iznosi 3,12, što je uobičajeno pod slučajnošću (Monte Carlo p = 0,30).

**Ponavljanja.** Identičnih izvlačenja: 2 (06.12.2025. = 09.08.2026.; 23.06.2026. = 06.10.2026.), uz očekivano 0,304 među 655 kola. Vjerojatnost da se slučajno dogodi barem 2 ponavljanja je oko 3,8 %. Ponavljanje od 06.10.2026. počiva na neslužbenim izvorima (odjeljak 3).

## 5. Neovisne provjere

**Dvije neovisne analize** (frekvencije/parovi/razmaci te ovisnost/vrijeme/struktura) provele su preko 40 testova na 635 kola, uz Monte Carlo kalibraciju. Nijedan test ne ostaje značajan nakon korekcije za višestruka testiranja. Druga analiza imala je 0 od 30 p-vrijednosti ispod 0,05.

**Šest skeptika** dobilo je zadatak pobiti najjače uočene „pravilnosti”. Za svaku su neovisno ponovili izračun, ispravili za broj usporedbi i provjerili izvan uzorka (procjena na izvlačenjima 1–423, od 01.01.2025. do 27.02.2026., test na izvlačenjima 424–635, od 28.02. do 28.09.2026.):

| Tvrdnja | Presuda | Pouzdanost | Korigirani p | Ključni razlog |
| --- | --- | --- | --- | --- |
| Vrući brojevi zadnjih 30 kola nastavljaju dolaziti | pobijeno | visoka | 0,237 | Efekt ide u krivom smjeru (vrući pogađaju manje) i ovisi o tome kako se lome izjednačenja: obrnuto pravilo daje p = 0,29. Najbolja varijanta s treninga obrnula se na testu. |
| Trojka 11-13-20 potisnuta, 7-15-16 favorizirana | pobijeno | srednja | 0,057 | Favorizirani smjer nikad se ne ponavlja. Slaba postojanost potisnutih trojki potječe od broja 13, a 13 je u 20 novih kola izvučen 12 puta uz očekivano 10,0. |
| Broj 8 rjeđe dolazi 5 kola nakon svoje pojave | pobijeno | visoka | 0,150 | Na testu p = 0,15. Korelacija trening/test preko 154 ćelije iznosi 0,07. Listić po ovom pravilu imao je prosječno 5,51 pogodaka. |
| Par 7-15 „sretan”, par 13-20 „nesretan” | pobijeno | visoka | 0,231 | Uz frekvencije pojedinih brojeva par 7-15 ima samo z = 1,99 (5. od 231). Par 13-20 se u 20 novih kola pojavio 6 puta, uz očekivano 4,8. |
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

**Provjera na 20 novih kola** (272–291), koje nijedna od ranijih analiza (na 635 kola) nije vidjela. Listići su sastavljeni samo iz podataka do 28.09.:

| Listić iz starih podataka | Kombinacija | Prosjek pogodaka | z | Ekstremi |
| --- | --- | --- | --- | --- |
| vrući, zadnjih 30 | ` 1  3  4  5  7  8 12 14 17 20 22` | 5,600 | 0,37 | 3/20 |
| vrući, zadnjih 90 | ` 1  2  4  5  7  8 10 17 18 20 21` | 5,400 | -0,37 | 0/20 |
| vrući, ukupno | ` 1  3  5  6  7  8 15 16 17 18 19` | 5,200 | -1,12 | 4/20 |
| hladni, ukupno | ` 2  4  5  9 10 11 12 13 14 20 21` | 5,900 | 1,49 | 3/20 |
| „zakašnjeli” (= komplement zadnjeg kola) | ` 3  6  7  8 10 13 16 18 19 20 21` | 5,250 | -0,93 | 2/20 |
| ponovi zadnje kolo (28.09.) | ` 1  2  4  5  9 11 12 14 15 17 22` | 5,750 | 0,93 | 2/20 |
| najbolji po modelu frekvencija | ` 1  3  5  6  7  8 15 16 17 19 22` | 5,250 | -0,93 | 1/20 |

| Tvrdnja iz starih podataka | Na 20 novih kola | Očekivano slučajno |
| --- | --- | --- |
| Broj 13 je „hladan” | izvučen 12 puta | 10,0 |
| Broj 7 je „vruć” | izvučen 9 puta | 10,0 |
| Par 7-15 je „sretan” | zajedno 4 puta | 4,76 |
| Par 13-20 je „nesretan” | zajedno 6 puta | 4,76 |
| Trojka 7-15-16 je favorizirana | zajedno 2 puta | 2,14 |
| Trojka 11-13-20 je potisnuta | zajedno 1 put | 2,14 |
| Premalo parnih brojeva | prosječno 5,05 | 5,50 |
| Uzastopna kola dijele više brojeva | prosječno 5,65 | 5,50 |

Nijedna „pravilnost” se nije potvrdila na novim podacima.

## 6. Zaključak

- Izvlačenja su statistički nerazlučiva od savršeno slučajnog izvlačenja 11 od 22. To je očekivano za nadzirano lutrijsko izvlačenje.
- „Predviđanje” koje povećava šansu nije moguće. Ne mogu ga dati ni veći model ni više računanja, jer ga u podacima nema.
- Ako igrate, dovoljno je odabrati kombinacije koje se međusobno ne ponavljaju i nisu komplementi. CSPRNG popis u prilogu to jamči i poredan je po kumulativnoj šansi, ali nije bolji od bilo kojeg drugog takvog odabira.
- Za jedan listić uzmite red 1. Za N listića uzmite redove 1 do N.
- Očekivani povrat je negativan na svakoj razini uloga. Igrajte samo iznos koji ste spremni izgubiti. Ako igranje postane problem, obratite se liječniku obiteljske medicine ili Hrvatskom zavodu za javno zdravstvo.

## Prilozi

- `kombinacije_kolo292_pokrice.zip`: svih 352 716 kombinacija, CSPRNG redoslijed, s kumulativnom šansom i ulogom.
- `kombinacije_kolo292_prvih100.csv`: prvih 100 redaka istog popisa.
- `sve_ili_nista_prosireno.csv`: priloženi podaci plus dopunjena kola 272–291.
- Skripte: `coverage.py` (popis pokrića), `generate.py` (varijante), `analyze.py` (analiza), `oos_new.py` (provjera izvan uzorka).