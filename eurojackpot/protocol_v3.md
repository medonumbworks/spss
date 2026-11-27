# Protokol v3 – Eurojackpot 27.11.2026. (HL kolo 95), 13 kombinacija
T0 = 2026-11-27 15:06:46 (Europe/Zagreb). Zapisano i commitano PRIJE novih izračuna.
Definicije sustava, svođenja, vremenska stabilnost (19:00–20:00 UTC, korak 15 min) i glasovi ostaju kao u v2 (`slaganje_v2.py`, `set_v2_konacni.json`); težine sve 1 (test v2). Slučajni elementi iz `slucajni_elementi.json`, bez novog izvlačenja. Novi testovi sustava se ne rade.

## 0. Ulazi koji nedostaju
- „3 dodatne": u sesiji nakon v2 nisu dane, u `evidencija_kombinacija.csv` ih nema → **nisu poznate; B se slaže uz 8 kombinacija iz v2** (status „uplaćeno"). B ima 10 poznatih kombinacija (ulog 20 €); prag isplate ostaje 26 € kako je zadano.
- Set iz v1 (6 kombinacija) nije dio „mojih 11" i ne ulazi u ograničenja.

## 1. Bazen i glasovi
- Bazen glavnih = unija punih popisa svih 12 sustava u svih 5 trenutaka (19:00, 19:15, 19:30, 19:45, 20:00 UTC). Euro bazen = unija punih euro popisa. Broj izvan bazena ne smije u set.
- Glas = 1 × težina (1) × stabilnost, za prvih 6 glavnih i prva 2 euro po sustavu (kao v2). Ostali brojevi bazena: glas 0. Plečkov poredak i signal (≥ 2 glasa) kao v2.

## 2. Slaganje (ezoterijski raspored E0)
- A: K1 = Plečko (5 najjačih signalnih brojeva koji poštuju §4), zatim po jedna kombinacija za svaki od 12 sustava, redom pravila iz v2 (broj glasačkih brojeva sustava u signalu ↓, zbroj njihovih glasova ↓, abecedno).
- B: 8 uplaćenih iz v2 (ne diraju se) + 2 nova sustava od {Sinkronicitet, Lenormand, Tarot, Kamea (Venera), Numerologija}: ona dva s najviše glasačkih (prvih 6) brojeva u signalu (pri jednakom zbroj glasova ↓, abecedno).
- Kombinacija uzima redom: (1) brojeve vlastitog sustava iz punog popisa; (2) kad ponestane: Plečkov signal, jači prvi (†); (3) ostale brojeve bazena po Plečkovom poretku (†). Broj se preskače ako krši §4.
- §4: 5 različitih glavnih, bez susjednih; dvije kombinacije (nove međusobno i nove prema uplaćenima) dijele ≤ 2 glavna broja.
- Euro: kandidati = euro popis vlastitog sustava, zatim Plečkov euro poredak (K1: samo poredak); bira se prvi par (i < j po redu kandidata) koji je nesusjedan i još nekorišten (13 različitih parova; u B i prema uplaćenima).

## 3. Isplate i vjerojatnosti
- Isplata razreda = medijan „Prize Per Winner" zadnjih 100 kola 2/12 (do 24.11.2026.) s euro-jackpot.net; samo kola u kojima razred ima ≥ 1 dobitnika. Scenariji: 25. i 75. percentil istim pravilom.
- Isplata ishoda = zbroj isplata svih dobitnih kombinacija. P(isplata ≥ 26 €), P(bilo koji), P(5+2), P(5+2/5+1/5+0/4+2), očekivana isplata.
- Konačne brojke točno preko svih 2.118.760 × 66 ishoda. Pretraga i pohlepni redoslijed na uzorku od 300.000 ishoda (uniformno s ponavljanjem, numpy sjeme = numpy_seed + 30), konačne vrijednosti točno.
- Očekivani povrat po euru = očekivana isplata / ulog (isti za svaki set) uz medijan isplata.

## 4. Referentni i krajnji rasporedi (prije izbora)
- R3 / R4: najbolja P(≥ 26 €) / P(bilo koji) BEZ ograničenja bazena (brojevi 1–50, 1–12; §4 vrijedi). Lokalna pretraga (promjena jednog glavnog ili euro broja, prihvaća se poboljšanje na uzorku) iz E0 i 3 slučajna početka (sjeme numpy_seed + 31…33), vremensko ograničenje po pretrazi. Za B mijenjaju se samo 2 nove.
- X3 / X4: isto, ali samo brojevi iz bazena (krajnji rasporedi unutar bazena). Prikazuju se P(≥ 26 €) i P(bilo koji) oba.

## 5. Izbor (leksikografski: 3, zatim 4, zatim glas)
- Korak 3: ako P3(E0) < R3 − 0,3 pp: pohlepne zamjene. Dopuštene pozicije = brojevi (glavni ili euro) najniže razine glasa na kojoj postoji poboljšanje; zamjena samo brojem iz bazena uz §4; bira se najveće poboljšanje P3 (na uzorku), pri jednakom više glasova pa manji broj. Do cilja ili najviše 40 zamjena. Ako cilj nije dostignut → vraća se E0 (ezoterija ima prednost) i navodi se razlika.
- Korak 4: isto za P4, uz uvjet da P3 ne padne ispod R3 − 0,3 pp (ako je korak 3 uspio) odnosno ispod P3 polaznog rasporeda (ako nije). Ako cilj R4 − 0,3 pp nije dostignut → vraćaju se zamjene koraka 4 i navodi razlika.
- Pri jednakom ishodu prednost ima manje preklapanja, zatim veći glas.

## 6. Ostalo
- Redoslijed igranja A: pohlepno po doprinosu P(bilo koji) na uzorku; za k = 1…13 točno: ulog, P(bilo koji), P(isplata ≥ ulog).
- Usporedbe: A prema 8 iz v2; prema 1000 slučajnih setova od 13 kombinacija (sjeme numpy_seed + 40; P na uzorku). B prema 1000 setova „8 uplaćenih + 2 slučajne" (sjeme + 41).
- Backtest A i B na 488 kola 2/12 sa stvarnim isplatama s euro-jackpot.net: razredi, isplate/ulog, kola s isplatom ≥ ulog, percentil (ukupna isplata) prema istim slučajnim setovima. Set se ne prilagođava backtestu.
- Rokovi: T0+15 protokol, T0+40 privremene tablice (E0 s točnim vjerojatnostima), T0+70 konačne, tvrdi kraj T0+80 (16:26); ako konačne nisu gotove, vrijede privremene.
