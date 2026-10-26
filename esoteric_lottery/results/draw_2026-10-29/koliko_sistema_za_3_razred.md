# Koliko skraćenih sistema za 3. razred (ili bolji)?

Pitanje: koliko skraćenih sistema 30 treba odigrati da se završi barem u 3. razredu
(dakle 1., 2. ili 3. — što znači barem **5 pogodaka** na jednoj kombinaciji).

## Po jednoj kombinaciji

Od 8.145.060 mogućih izvlačenja, fiksna kombinacija od 6 brojeva pogodi:

| razred | uvjet | broj izvlačenja |
|---|---|---|
| 1. | 6 pogodaka | 1 |
| 2. | 5 + dodatni | 6 = 234/39 |
| 3. | 5 pogodaka | 228 = 234 × 38/39 |
| **1.–3.** | **≥ 5 pogodaka** | **235** |

p = 235/8.145.060 = 2.885184e-05 = **1 u 34.660** po kombinaciji.

## Zašto se vjerojatnosti ovdje gotovo **zbrajaju**

Dvije odigrane kombinacije mogu obje imati ≥5 pogodaka u istom izvlačenju samo ako dijele
barem 4 broja (iz |T₁∩D| + |T₂∩D| = |D∩(T₁∪T₂)| + |D∩T₁∩T₂| ≤ 6 + |T₁∩T₂|). Dakle sve
kombinacije koje dijele 3 ili manje brojeva imaju **međusobno isključive** događaje, pa je
P(barem jedna) = M × p **točno**, a ne 1−(1−p)^M. Uobičajeni „nezavisni“ račun ovdje
podcjenjuje odgovor.

## Po jednom skraćenom sistemu 30 (323 kombinacije)

| | |
|---|---|
| E[dobitnih kombinacija u 1.–3. razredu] | 0.0093191 = 1 u 107.3 — **egzaktno, ne ovisi o dizajnu** |
| P(barem jedna) — gornja granica | 0.009319 |
| P(barem jedna) — izmjereno, dobro raspršen dizajn | 0,008516 |
| P(barem jedna) — izmjereno, slučajan dizajn | 0,008027 |
| **radna vrijednost** | **0,0085 ≈ 1 u 118** |

(Pakirni limit D(20,6,4) = 276 < 323: unutar jednog sistema parovi koji dijele 4 broja su
neizbježni, pa je točan P uvijek ispod gornje granice. Izmjereno iscrpnim nabrajanjem
dobitnih izvlačenja za sintetičke dizajne — službeni HL dizajn nije dostupan.)

## Odgovor

| cilj | skraćenih sistema | kombinacija | trošak |
|---|---|---|---|
| 10 % šanse | **12** | 3.876 | 969,00 € |
| 25 % šanse | **30** | 9.690 | 2.422,50 € |
| 50 % šanse | **59** | 19.057 | 4.764,25 € |
| 75 % šanse ⚠ | **89** | 28.747 | 7.186,75 € |
| 90 % šanse ⚠ | **106** | 34.238 | 8.559,50 € |
| E[jedan dobitak] = 1 | 107.3 | 34.660 | 8.664,96 € |
| **zajamčeno, u svakom izvlačenju** | **≥ 646** | ≥ 208.628 | **≥ 52.164,50 €** |

⚠ Redci iznad ~60 sistema su optimistični: toliko dvadesetki ne može se više birati s malim
presjecima (59 × 20 = 1.180 mjesta na 45 brojeva), pa sudari rastu i stvarni P pada ispod
linearne procjene.

Zajamčeni redak dolazi iz Schönheimove granice za pokrivanje: da bi se *sigurno* imalo 5
pogodaka, odigrane kombinacije moraju sadržavati svaku od C(45,5) = 1.221.759 petorki,
a za to treba barem 208.628 kombinacija. Sa skraćenim sistemima je još gore, jer listić od
20 brojeva može pokriti samo petorke unutar tih 20.

## Što to vrijedi

Na 59 sistema (19.057 kombinacija, 4.764,25 €), uz fondove iz prikazanog izvlačenja

| razred | fond | tuđih dobitnika | moji E[dobici] | povrat |
|---|---|---|---|---|
| 5. (3 pogotka) | 7.765,40 € | 4.735 | 427,7 | 643,25 € |
| 4. (4 pogotka) | 3.335,08 € | 277 | 26,0 | 286,24 € |
| 3. (5 pogodaka) | 5.559,45 € | 3 | 0,53 | 741,42 € |
| 2. i 1. | 0 dobitnika — fond se prenosi | — | — | nije procjenjivo |
| **ukupno** | | | | **1.670,91 € na ulog 4.764,25 € = 35 %** |

Povrat je računat s razrjeđenjem: vlastiti dobitnici smanjuju iznos po dobitniku
(4.735 + 428 dobitnika petice dijeli isti fond). Vrijednost koja nedostaje do ~50 % leži u
1. i 2. razredu, koji su se u tom izvlačenju prenijeli.

**Zaključak: 3. razred je platio 1.853,15 €. Za 50 % šanse da ga dohvatiš treba
4.764,25 € — 2,6 puta više od dobitka. Za sigurnost treba
52.164,50 € — 28 puta više.** Ne postoji broj sistema pri kojem se to isplati.
