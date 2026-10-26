# Loto 6/45 — izvlačenje četvrtak 29.10.2026.

## Tri skraćena sistema 30 (po 20 brojeva)

Tri **disjunktna** skraćena sistema 30 ne postoje: 3 × 20 = 60 brojeva, a Loto ima 45.
Najmanji mogući ukupni presjek triju dvadesetki je 15 (dokaz: zbroj presjeka je
sum_n C(m_n, 2) uz sum_n m_n = 60 preko 45 brojeva, a to je konveksno, pa je najmanje
kad 15 brojeva igra dvaput i 30 jednom), i jedini raspored tih 15 u kojem nijedan
pojedinačni presjek ne prijeđe 5 je 5 + 5 + 5. To je struktura ispod: svih 45 brojeva
je pokriveno, a nijedna od 3 × 323 = 969 kombinacija ne može se ponoviti — za to bi
dva listića morala dijeliti 6 brojeva.

| # | 20 brojeva | Joker |
|---|---|---|
| 1 | `02 03 04 05 06 07 11 15 17 19 20 26 29 31 32 36 37 39 40 41` | 439431 |
| 2 | `01 02 03 04 05 08 09 10 18 21 22 23 24 27 34 35 41 42 44 45` | 162431 |
| 3 | `01 07 08 09 10 12 13 14 16 19 25 28 30 31 32 33 36 38 43 45` | 039434 |

Presjeci: 1–2 = 5, 1–3 = 5, 2–3 = 5. Unija = 45/45. Trošak 3 × 80,75 = 242,25 EUR za 969 kombinacija.

Brojevi koji igraju na dva listića (jedina ezoterična sloboda u ovoj strukturi): `01 02 03 04 05 07 08 09 10 19 31 32 36 41 45`

## Točni brojevi (egzaktna hipergeometrijska razdioba, bez simulacije)

| veličina | vrijednost |
|---|---|
| P(nijedan listić ne drži 3 od 6 izvučenih) | 0,011188 = 1 u 89 |
| P(barem jedan listić drži 3 od 6 izvučenih) | 0,988812 |
| P(barem jedan listić drži 4 od 6 izvučenih) | 0,630567 |
| P(barem jedan listić drži 5 od 6 izvučenih) | 0,156608 = 1 u 6,4 |
| P(barem jedan listić drži 6 od 6 izvučenih) | 0,014276 = 1 u 70,0 |
| E[zbroj pogodaka triju listića] | 8,0000 — invarijantno, 3 × 6 × 20/45 |
| E[broj dobitnih kombinacija s 6 pogotka] | 0,00012 |
| E[broj dobitnih kombinacija s 5 pogotka] | 0,02784 |
| E[broj dobitnih kombinacija s 4 pogotka] | 1,32233 |
| E[broj dobitnih kombinacija s 3 pogotka] | 21,74494 |
| P(6/6 među odigranim kombinacijama) | 969/8.145.060 = 1 u 8.406 |
| P(6/6) na jednom listiću | 323/8.145.060 = 1 u 25.217 |

## Što je ovdje izbor, a što nije

Sve vjerojatnosti u tablici zavise **samo** o strukturi presjeka, ne o tome koji su
brojevi odabrani: svaka dvadesetka od 45 brojeva ima istu razdiobu pogodaka. Struktura
5/5/5 je odabrana jer među svim rasporedima koji pokrivaju svih 45 brojeva maksimizira
gornje razrede — P(≥4) = 0,6306 i P(≥5) = 0,1566 su najviše moguće — i jedina isključuje
ponovljenu kombinaciju. Raspored 0/7/8 (dva listića disjunktna, treći iz ostatka) diže
samo najniži razred, P(≥3) na 0,9937, a gornje snižava.

Jedini slobodan izbor je **koji** 15 brojeva igra dvaput. Tu odlučuje ansambl deset
metoda za trenutak izvlačenja 2026-10-29 19:00 CET (40 čitanja, Bordin zbir), uz uvjet
da svaki listić drži 8 brojeva iz 1–15, 6 iz 16–31 i 6 iz 32–45. Listić 1 je
nepromijenjen onaj koji je već dan za četvrtak.

Ograda: P(listić drži k od 6 izvučenih) je nužan, ne dovoljan uvjet za dobitak: skraćeni sistem igra 323 od C(20,6)=38760 kombinacija, a službeni HL dizajn tih 323 kombinacija nije bio dostupan za provjeru (lutrija.hr nedostupan iz kontejnera). Očekivani broj dobitnih kombinacija po razredu je točan i ne zavisi od dizajna.
