# Protokol testa ezoterije – Eurojackpot 27.11.2026. (zapisano PRIJE računanja testa)

Podaci: `povijest_eurojackpot_2022-03-25_2026-11-24.csv` (488 kola, euro-jackpot.net, zadnjih 20 kola provjereno na lotteryextreme.com).
Kola s indeksom 0–4 služe samo kao povijest za Sinkronicitet; testira se 483 kola (indeks 5…487).
Trening = starija polovica (prvih ⌊483/2⌋ = 241 kolo), test = novija polovica (242 kola).

## Zajednička pravila
- Vrijeme izračuna za svako kolo: datum kola, 19:00 UTC; mjesto Helsinki 60,17° N, 24,94° E.
- Svođenje: glavni > 50 → −50 dok ne bude ≤ 50; vrijednosti ≤ 0 se odbacuju. Euro: isti sirovi popis, > 12 → −12. Bez ponavljanja, zadrži prvo pojavljivanje.
- Listić sustava = prvih 5 glavnih + prva 2 euro. Ako popis ima manje od 5 (2) brojeva, dopunjava se najmanjim nekorištenim brojevima 1, 2, 3 … (deterministički, neovisno o izvlačenju).
- Euro popis = (izričiti euro popis sustava, ako postoji) pa sirovi popis sveden na 1–12.

## Deterministički sustavi (k = 9)
1. Sinkronicitet: glavni brojevi koji se ponavljaju u 5 kola PRIJE tog kola (broj pojavljivanja ≥ 2; češći prvi, pri jednakom noviji zadnji nastup prvi, zatim manji), zatim dan, mjesec, godina svedena na 1 znamenku, redni broj kola u kalendarskoj godini (HL kolo), redni dan u godini, dob osobe (rođ. 31.05.1991.), 31, 5, 18, 20, zatim brojevi prethodnog kola uzlazno. Euro: euro brojevi koji se ponavljaju u 5 prethodnih kola, istim redom.
2. Arhetipovi: 4, 12, 1, 16, 7, 10, 28, 22, 40, 36 (konstanta).
3. Meihua: Y = redni broj zemaljske grane lunarne godine (子 = 1 … 午 = 7 … 亥 = 12), M i D = lunarni mjesec (prijestupni → apsolutna vrijednost) i dan (lunar_python), H = 12 (19:00 UTC = 21:00/22:00 u Helsinkiju, dvosat 21–23). Gornji = (Y+M+D) mod 8, donji = (Y+M+D+H) mod 8 (0 → 8), crta = (Y+M+D+H) mod 6 (0 → 6); trigrami prema ranom nebu (1 ☰ … 8 ☷). Popis: osnovni, promijenjeni, nuklearni (King Wen), crta, gornji, donji trigram.
4. Jyotiš (Lahiri, sideralno, Moshier): 20, nakšatra Mjeseca (1–27), tithi (1–30), vladar dana (Ned 1, Pon 2, Uto 3, Sri 4, Čet 5, Pet 6, Sub 7), yoga (1–27), nakšatra srednjeg Rahua (1–27).
5. Zapadna (tropski): stupanj u znaku + 1 redom za Veneru, Jupiter, Merkur, Sunce, Mjesec, Mars; zatim 10, 15.
6. Kamea dana u tjednu (Agrippa): petak = Venera 7×7 (dijagonala, sporedna dijagonala, 7, 175); utorak = Mars 5×5 (dijagonala, sporedna dijagonala, 5, 65).
7. Kabala: petak = Nogah 58, Necah 7; utorak = Ma'adim 95, Gevura 5; zatim IVAN 67, mazal 77, LOTO 51, 26, 18, 10.
8. Numerologija: 11, 29, 19, 13, 31, osobni dan (znamenke 31.05. + znamenke datuma kola), univerzalni dan (znamenke datuma kola); svođenje na 1 znamenku, 11 i 22 se ne svode.
9. Planetarni sat u 19:00 UTC u Helsinkiju (izlazak/zalazak Sunca iz Swiss Ephemeris, kaldejski red, prvi sat dana = vladar dana). Popis: broj planeta (Sunce 1, Mjesec 2, Mars 3, Merkur 4, Jupiter 5, Venera 6, Saturn 7), red kamee planeta (Saturn 3, Jupiter 4, Mars 5, Sunce 6, Venera 7, Merkur 8, Mjesec 9), konstanta kamee (15, 34, 65, 111, 175, 260, 369), redni broj sata od izlaska Sunca (1–24), redni broj sata unutar noći/dana (1–12).

Slučajni sustavi (I Ching, Tarot, Lenormand): ne mogu se testirati (slučajni po konstrukciji), težina 1.

## Mjere
- Po kolu: glavni pogoci (H0: hipergeometrijska, očekivanje 0,5, var 0,41327), euro pogoci (očekivanje 1/3, var 0,25253), dobitni događaj (12 razreda, točna vjerojatnost po listiću).
- Primarna statistika: T = glavni + euro pogoci po kolu; z = (ΣT − N·5/6) / √(N·0,66580).
- Trening: dvostrani p; prag p < 0,05/9 = 0,005556. Test: jednostrani p u smjeru z s treninga, prag p < 0,05.
- Težina 2 samo ako su ispunjena oba uvjeta; inače 1.
- Kontrola: 1000 fiksnih slučajnih listića (numpy Generator sa sjemenom iz `secrets`, spremljenim u JSON); najbolji na treningu (najveći z) → njegov z i p na testu; plus udio slučajnih listića koji prelaze Bonferronijev prag na treningu.
- Definicije sustava se nakon ovog zapisa ne mijenjaju.
