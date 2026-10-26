# Literatura — što je provjereno, što je ispravljeno

Provjeravao sam citate koje sam ranije prenio iz priloženog PDF-a, jer ih nisam bio sam
verificirao. Dvoje je trebalo ispraviti, a dvije nove reference mijenjaju zaključke.

## Ispravci

**1. Cook & Clotfelter (1993) nije studija o tome koje brojeve ljudi izbjegavaju.**
*„The Peculiar Scale Economies of Lotto”, American Economic Review 83(3), 634–643.* Rad je o
ekonomiji razmjera u prodaji lota — zašto prodaja po glavi raste s veličinom populacije. Kasnija
literatura njemu pripisuje **naziv** „conscious selection”, ali empirijski dokaz da se brojevi
iznad 31 rjeđe igraju nije u tom radu. Ranije sam ga naveo kao izvor tog nalaza; to je bilo
netočno. Pravi izvori su Farrell et al. (2000), Baker & McHale (2009) i Ziemba et al. (1986).

**2. Chernoff (1981) nije u JASA-i i nije uspio.**
*„How to beat the Massachusetts numbers game: An application of some basic ideas in probability
and statistics”, The Mathematical Intelligencer 3, 166–172.* Dvije stvari koje PDF ne spominje:
njegov prvi sustav, građen na 851 izvlačenju, „proved disappointing”; a o objavljivanju sustava
piše da bi *„the resulting popularity of the numbers constituting the system would destroy their
value”*. To je isti argument koji PDF daje pod „Bolje od kopiranja” — samo stariji 45 godina.
Chernoff je radio na igri s državnim rezom od 40 %. **Loto 6/45 reže 50 %**, dakle teže.

**3. Moj χ² test bio je kriv.**
Genest, Lockhart & Stephens (2002) upozoravaju da Pearsonova statistika kod izvlačenja **bez
ponavljanja** ne slijedi χ² razdiobu ni u velikim uzorcima. Ranije sam napisao „χ² = 28,3 pri 44
stupnja slobode, kritična vrijednost 60,5”. Točno je: E[X] = 45(1 − 6/45) = **39,0**, a 95.
percentil je **53,7** (simulacija 40.000 nizova po 882 izvlačenja; srednja vrijednost simulacije
38,98 potvrđuje analitičku 39,00). Zaključak se ne mijenja — 28,3 je i dalje duboko unutar
normalnog — ali brojke kojima sam ga potkrijepio bile su pogrešne.

## Novo — što tempera blok 4

**Papachristou & Karamanis (1998)**, *„Investigating efficiency in betting markets: Evidence from
the Greek 6/49 Lotto”, Journal of Banking & Finance 22(12), 1597–1615.* Na grčkom 6/49 — najbliži
europski analogon hrvatskom 6/45 — nalaze da nepopularni brojevi **postoje**, ali **nisu dovoljno
nepopularni** da proizvedu pozitivno očekivanje. To je izravan protuteg optimizmu oko bloka 4.

**Moffitt & Ziemba** o kanadskom 6/49 daju veličinu efekta: popularna kombinacija dijeli jackpot
s udjelom ≈ 17 %, nepopularna ≈ 36 % — efekt je **velik uvjetno na pogodak**. Ali za pozitivno
očekivanje sindikatu treba prijenos preko **≈ 37 milijuna dolara**. Hrvatski Loto nije blizu toga.

**Ziemba et al. (1986)** na kanadskom 6/49 nabrajaju 19 statistički nepopularnih brojeva: 1, 10,
20, 28, 29, 30, 32, 34 i sve osim dva iznad 38. Primijeti **1, 10 i 20** — moje pravilo „najviše
jedan broj ≤ 12” je zato malo pregrubo. Nisam prerađivao blok 4 zbog toga: nalaz je vezan za
Kanadu osamdesetih, a isti autori bilježe da su se nepopularnosti s vremenom **vratile prema
sredini** (efekt učenja). Robustan dio nalaza, ponovljen u svim radovima, jest zona iznad 31.

**Baker & McHale (2009)**, *JRSS A 172(4), 813–834.* Svjesni odabir proizvodi **predisperziju** u
broju dobitnika i **korelaciju između razreda**. Praktična posljedica: moja procjena razrjeđenja
(5,18 mojih trojki na 4.735 tuđih) koristi brojeve dobitnika iz jednog kola kao da su stabilni.
Nisu — raspršenje je veće nego što Poissonov model sugerira, pa je i povrat neizvjesniji.

**Haigh (1997)**, *JRSS A 160(2), 187–206.* Na prvih 96 kola UK lota: frekvencije i vremena čekanja
su u skladu sa slučajnošću, ali **kombinacije koje igrači biraju daleko su od slučajnih**. Točno
ta dva nalaza zajedno opravdavaju ovu strategiju: brojeve ne možeš predvidjeti, gužvu možeš.

## Pakiranje protiv pokrivanja — zašto presjek ≤ 2, a ne jamstvo trojke

Literatura o *lotto dizajnima* (Li & van Rees, *Lotto design tables*, J. Combin. Designs 10 (2002),
335–359) traži **LD(n,k,p,t)**: najmanji broj listića takav da **svako** izvlačenje dijeli barem t
brojeva s nekim listićem. Za naš slučaj to bi bio L(45,6,6,3) — jamstvo trojke. Objavljene tablice
ne dosežu n = 45 (Li-jeva tablica ide do n ≤ 20; Bate & van Rees riješili su L(n,6,6,2) za n ≤ 54),
pa vrijednost nisam našao. Ali odluka ne ovisi o njoj:

| | moj set (pakiranje) | pokrivni dizajn |
|---|---|---|
| P(1.–3. razred) | 0,0066648 — maksimum | niže |
| P(ništa) | 0,000713 | 0 |

Jamstvo trojke vrijedi 0,000713 × 1,64 € = **0,00117 €**. Svaki par listića koji dijeli 4 broja
gubi 4 izvlačenja iz unije 1.–3. razreda, što košta 4/8.145.060 × 1.853,15 € = **0,00091 €**.
Dakle **već 1,3 takva para pojedu cijelu vrijednost jamstva**, a pokrivni dizajn ih treba stotine.
Pakiranje je ispravan izbor za cilj „1.–3. razred”, i to bez imalo dvojbe.

Koliko znam, ta usporedba nije u literaturi — lotto dizajni su cijeli o pokrivanju. Pravilo
„dva listića mogu oba imati 5 pogodaka samo ako dijele 2·5 − 6 = 4 broja” je elementarno i
dokazano je u `evaluation/tiers.py`; ne pripisujem ga nikome.

---

### Puni popis

- Baker, R. D. & McHale, I. G. (2009). Modelling the probability distribution of prize winnings in
  the UK National Lottery: consequences of conscious selection. *JRSS A* 172(4), 813–834.
  doi:10.1111/j.1467-985X.2009.00599.x
- Bate, J. A. & van Rees, G. H. J. The size of the smallest strong lottery design. [L(n,6,6,2), n ≤ 54]
- Chernoff, H. (1981). How to beat the Massachusetts numbers game. *The Mathematical
  Intelligencer* 3, 166–172. doi:10.1007/BF03022976
- Cook, P. J. & Clotfelter, C. T. (1993). The peculiar scale economies of lotto. *AER* 83(3), 634–643.
- Farrell, L., Hartley, R., Lanot, G. & Walker, I. (2000). The demand for lotto: the role of
  conscious selection. *JBES* 18(2), 228–241. doi:10.1080/07350015.2000.10524865
- Genest, C., Lockhart, R. A. & Stephens, M. A. (2002). χ² and the lottery. *JRSS D* 51(2).
- Haigh, J. (1997). The statistics of the National Lottery. *JRSS A* 160(2), 187–206.
- Henze, N. & Riedwyl, H. (1998). *How to Win More*. A K Peters.
- Li, P. C. & van Rees, G. H. J. (2002). Lotto design tables. *J. Combin. Designs* 10, 335–359.
- Moffitt, M. & Ziemba, W. T. Does it pay to buy the pot in the Canadian 6/49 lotto?
- Papachristou, G. & Karamanis, D. (1998). Investigating efficiency in betting markets: evidence
  from the Greek 6/49 lotto. *Journal of Banking & Finance* 22(12), 1597–1615.
- Ziemba, W. T., Brumelle, S. L., Gautier, A. & Schwartz, S. L. (1986). *Dr. Z's 6/49 Lotto Guidebook.*

**Ograda o dohvatu:** arxiv.org i sciencedirect.com nisu dostupni iz ovog kontejnera, pa su
Papachristou & Karamanis, Moffitt & Ziemba i Genest et al. provjereni preko bibliografskih zapisa
i sažetaka, ne punog teksta. Bibliografski podaci (časopis, godište, stranice, DOI) jesu
provjereni. Citate iz sažetaka i sekundarnih izvora označio sam kao takve u tekstu gore.
