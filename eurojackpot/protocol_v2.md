# Protokol v2 – 8 kombinacija, Eurojackpot 27.11.2026. (HL kolo 95)
Zapisano PRIJE ijednog novog izračuna (git commit ovog dokumenta je vremenski dokaz). Definicije sustava, preslikavanja i svođenja ostaju kao u `protocol.md` i `sustavi.py` (v1). Slučajni elementi (I Ching, Tarot, Lenormand, numpy sjeme) uzimaju se iz `slucajni_elementi.json` i NE izvlače se ponovno.

Ulazi koji nisu zadani: „već odigrano" nije upisano → **ništa**. Cijena 2 € → ulog 8 × 2 € = 16 €.

## A. Cilj (leksikografski, sve točno, svih 2.118.760 × 66 ishoda)
1. P(5+2) = 8 / 139.838.160 (isto za bilo kojih 8 različitih kombinacija).
2. P(5+2, 5+1, 5+0, 4+2) najveća: postiže se kad dvije kombinacije dijele ≤ 2 glavna broja (tada su događaji „4+ pogodaka" disjunktni). §3 (≤ 1 zajednički) to jamči.
3. P(bilo koji dobitak) ≥ P_ref − 0,3 postotnog boda.
   P_ref = max(najbolji §3-raspored, najbolji pronađeni neograničeni raspored) za 40 različitih glavnih brojeva:
   - §3-raspored euro parova (8 različitih parova, 4 broja dvaput, 8 jednom, svih 12 prisutno, bez susjednih parova) je graf čiji su svi dijelovi putovi (krajevi stupnja 1) i eventualno ciklus od stupnjeva 2. Postoji točno 7 oblika: unutarnji vrhovi po 4 puta (4,0,0,0), (3,1,0,0), (2,2,0,0), (2,1,1,0), (1,1,1,1); ciklus 3 + (1,0,0,0); ciklus 4 + (0,0,0,0). Svih 7 računa se točno (susjednost parova ne utječe na vjerojatnost, ali se pri izboru brojeva poštuje).
   - Neograničeno (bilo kojih 8 euro parova): lokalna pretraga (zamjena jednog euro broja), 20 početaka, sjeme numpy_seed + 20. Služi samo kao referenca.
4. Unutar toga najveći ukupni ezoterijski glas. Očekivani povrat po euru je isti za svaki set i ne može se povećati.

## B. Struktura (tvrda pravila, §3)
8 kombinacija; 5 različitih glavnih 1–50 uzlazno, bez susjednih; 2 različita euro 1–12, par nije susjedan; dvije kombinacije dijele ≤ 1 glavni broj; 8 različitih euro parova; 4 euro broja dvaput, 8 jednom, svih 12 u setu.

## C. Prošireni empirijski test (§4)
- Podaci: glavni brojevi svih kola od 23.03.2012. (arhivi euro-jackpot.net 2012–2026; uzorak provjeren s lotteryextreme.com); euro brojevi samo kola od 25.03.2022. (2/12). Provjera: 5 različitih 1–50; euro 2 različita 1–12 (od 2022.); bez duplikata datuma.
- Isti 9 determinističkih sustava, isto preslikavanje, isti listić (prvih 5 glavnih + 2 euro, dopuna najmanjim nekorištenim), izračun za datum kola u 19:00 UTC, kao u `test_ezoterije.py`.
- Glavni dio: kola od indeksa 5 cijelog niza 2012–2026 (Sinkronicitet koristi 5 prethodnih kola). Euro dio: kola 2/12 od indeksa 5 serije od 25.03.2022. (isti 483 kola kao v1); za starija kola euro pogoci se ne broje.
- Statistika: T = Σ glavnih pogodaka + Σ euro pogodaka (gdje postoje); E = 0,5·N_g + N_e/3; Var = N_g·0,41327 + N_e·0,25253; z = (T − E)/√Var. Posebno se navode z_glavni i z_euro.
- Podjela: kola glavnog dijela po vremenu; starija polovica (⌊N/2⌋) = trening, novija = test (euro pogoci padaju samo u test – to se navodi).
- Pravilo odluke: težina 2 samo ako p_trening (dvostrani) < 0,05/9 I p_test (jednostrani, isti smjer) < 0,05; inače 1. Bez iznimaka (ni za planetarni sat).
- Kontrola: istih 1000 fiksnih slučajnih listića kao v1 (numpy default_rng(numpy_seed), isti redoslijed generiranja); najbolji na treningu → test; udio koji prelazi Bonferronijev prag.
- Dobitni događaji (razredi) samo za kola 2/12. Sustavi se ne biraju ni rangiraju prema backtestu.

## D. Glasovi (§5)
- Vremenski prozor: 19:00, 19:15, 19:30, 19:45, 20:00 UTC (20:00–21:00 CET). Vremenski ovisni: Meihua (H iz lokalnog vremena Helsinkija: dvosat 21–23 = 亥 = 12), Jyotiš, Zapadna, Planetarni sat; kontrolno i karta dekana (Tarot).
- Za svaki sustav: skup glasova = unija prvih 6 glavnih (prvih 2 euro) u svih 5 trenutaka. Broj prisutan u svih 5 trenutaka: stabilnost 1; inače 0,5. Nevremenski sustavi: stabilnost 1.
- Redoslijed brojeva sustava: popis u 19:00 UTC, zatim brojevi koji se pojavljuju tek kasnije, redom vremena i pozicije.
- Glas = 1 × težina × stabilnost. Plečkov poredak: više glasova; pri jednakom manji zbroj pozicija (pozicija u popisu sustava, 1…); zatim manji broj. Signal = ≥ 2 glasa; šum = > 0 i < 2.
- Euro poredak isto; „4 broja s najviše euro glasova" = prva 4 u euro poretku.

## E. Slaganje (pravilo unaprijed)
- K1 = Plečko: 5 najjačih signalnih brojeva redom poretka, preskačući one koji krše §3. Euro izvor K1 = euro poredak.
- K2–K8 = po jedan sustav (od 12: 9 determinističkih + I Ching, Tarot, Lenormand), redom: (1) broj brojeva iz njegova skupa glasova koji su u signalu, više prvo; (2) zbroj Plečkovih glasova tih brojeva, više prvo; (3) abecedno.
- Svaka kombinacija uzima brojeve svog sustava redom (cijeli popis sustava, ne samo prvih 6) i preskače broj koji krši §3 (susjedni u kombinaciji; > 1 zajednički s bilo kojom ranijom kombinacijom).
- Punilo (*) samo kad sustavu ponestane brojeva: dopušten broj s najviše glasova; pri jednakom najudaljeniji od ostalih u kombinaciji (najveća najmanja udaljenost); zatim manji.
- Euro: kapaciteti 2 za prva 4 euro broja, 1 za ostalih 8. Kombinacije redom K1…K8 biraju par među dopuštenima (nesusjedan, nekorišten, ostatak se može dovršiti) po ključu: (broj članova para u prva 2 euro sustava, više) → (zbroj pozicija u euro popisu sustava, manje; odsutan = 99) → (zbroj euro glasova, više) → (manji brojevi).
- Ako P(bilo koji) < P_ref − 0,3 pp: ponavljaj (a) među glavnim brojevima koji su zajednički s drugom kombinacijom uzmi onaj s najmanje glasova (pri jednakom kasnija kombinacija, veći broj) i zamijeni ga brojem koji nije u setu, poštuje §3 i ima najviše glasova (pri jednakom najudaljeniji, manji); (b) kad zajedničkih više nema, zamijeni euro brojeve između dviju kombinacija (zamjena koja najviše podiže P, poštujući §3). Svaka zamjena se navodi.

## F. Vjerojatnosti, usporedbe, backtest (§6)
- Točno: P(5+2), P(5+2/5+1/5+0/4+2), P(bilo koji), očekivani broj dobitnih kombinacija; novi set i zajedno s već odigranima (ništa).
- Usporedba s P_ref i s 1000 slučajnih setova od 8 kombinacija (svaka nezavisno uniformna; sjeme numpy_seed + 12) → percentil (jednaki se broje pola).
- Redoslijed igranja: pohlepno po doprinosu P(bilo koji); pri jednakom manji indeks.
- Backtest: (1) razredi dobitaka na 488 kola 2/12; (2) glavni pogoci (3, 4, 5) na svim kolima od 2012.; percentil prema istih 1000 slučajnih setova (primarno: broj dobitnih listića 2022–2026; sekundarno: broj listića s ≥ 3 glavna 2012–2026).
- Isplate: ako se s euro-jackpot.net mogu dohvatiti iznosi po razredima za kola u kojima set ima dobitak, povrat = Σ isplata / (16 € × 488); inače „nije izračunato". Set se ne prilagođava backtestu.

## G. Rokovi
14:30 protokol · 15:30 privremeni set (težine v1 = sve 1) · 17:00 konačni set (težine iz C) · 17:15 tvrdi kraj (ako konačni nije gotov, vrijedi privremeni).
