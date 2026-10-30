# Zadatak korisnika (izvorni tekst, za provjeru uskladjenosti)

ULOGA I OKVIR
Radiš kao statističar i registrar za Loto 6/45 (Hrvatska Lutrija). Sve računaj lokalno skriptama u ovom direktoriju; ne pretražuj internet i ne dohvaćaj ništa s weba. Ništa od ovoga nema prediktivnu moć: svaka kombinacija ima 1 : 8.145.060 za šesticu, metode samo biraju KOJE od jednako vjerojatnih kombinacija igram — ne tvrdi suprotno nigdje u izlazu. Piši hrvatski; nazive sustava, datoteka i stupaca bez dijakritika.

ULAZI
- povijest.csv: izvlačenja Lota 6/45 (datum;kolo;n1;...;n6). Provjeri strukturu, duplikate, raspon 1-45 i 6 različitih brojeva; odbačene retke ispiši. Ako datoteka ne završava na danas, navedi koliko kola nedostaje; ne izmišljaj ih.
- odigrano.csv (opcionalno): kombinacije koje već igram; ne ponavljaj ih.
- Natalni podaci: 31.05.1991., 18:20 CEST (UTC+2), Zadar 44°07'N 15°14'E. Ime za numerologiju: IVAN.
- Mjesto kola: Zagreb 45°49'N 15°59'E. Vrijeme kola: [UPIŠI] lokalno. Dani kola: [UPIŠI]. Razdoblje: [UPIŠI]. Cijena kombinacije: [UPIŠI] €. Dobitne vrste: [UPIŠI].
  (Korisnik nije upisao; usvojene pretpostavke: 20:00, srijeda i nedjelja, 01.11.2026.-31.12.2026. — korisnik naknadno trazio "do kraja godine", 1,00 EUR, 6/5/4/3.)
- Slučajni dio (karte, stabljike): seed = secrets.randbits(64), jedan za cijeli posao, zapisan u svaki izlaz.

PRAVILA PRETVORBE
- m45(n) = ((n-1) mod 45) + 1. Kombinacija = prvih 6 različitih vrijednosti iz kaskade kandidata sustava; ako ih je manje od 6, dopuni rezervnim pravilom tog sustava i označi.
- Joker: zadnje znamenke prvih 6 vrijednosti iste kaskade (prije m45), tekst s vodećim nulama.

SUSTAVI (15 po kolu)
1-2. Pareto 1 = 6 najčešćih brojeva u cijeloj povijesti (nerješeno: manji broj); Pareto 2 = rang 7-12.
3-4. Vruci 1 = 6 najčešćih u zadnjih 100 kola datoteke (nerješeno: zadnjih 250, pa manji broj); Vruci 2 = rang 7-12. Ovo četvero računa se jednom, na stanju datoteke.
5. Kaldejska: A1 B2 C3 D4 E5 F8 G3 H5 I1 J1 K2 L3 M4 N5 O7 P8 Q1 R2 S3 T4 U6 V6 W6 X5 Y1 Z7. Kaskada: LOTO, HRVATSKA LUTRIJA, ime dana bez dijakritika, ZAGREB, IVAN, dan u mjesecu, zbroj znamenki datuma, broj planeta dana (Sunce 1, Mjesec 2, Jupiter 3, Merkur 5, Venera 6, Saturn 8, Mars 9).
6. Gematrija: hebrejski dan u mjesecu u trenutku kola (nakon zalaska u Zagrebu = sljedeći dan), zatim מזל 77, ברכה 227, שפע 450, זהב 14, חי 18, הצלחה 138, כסף 160, עושר 576 (standardne vrijednosti, bez sofita).
7. Hermetika: planetarni sat u trenutku kola (izlazak/zalazak za Zagreb, dnevni i noćni sati nejednake duljine, kaldejski red od vladara dana) → Agrippina kamea tog planeta (Saturn 3x3, Jupiter 4x4, Mars 5x5, Sunce 6x6, Venera 7x7, Merkur 8x8, Mjesec 9x9); kaskada = središnji red (dva središnja za parne), pa konstanta i ukupni zbroj. Asertom provjeri redove, stupce i dijagonale.
8. I Ching: 6 linija stabljikama (6:7:8:9 = 1:5:7:3 od 16). Kaskada: King Wen broj primarnog, nuklearnog (linije 2-4 dolje, 3-5 gore) i promijenjenog heksagrama, zbroj vrijednosti linija, položaji pomičnih linija, Fu Xi gornjeg x8 + donjeg (Qian 1, Dui 2, Li 3, Zhen 4, Xun 5, Kan 6, Gen 7, Kun 8). Upiši cijelu King Wen tablicu i testiraj asertima (Kun nad Kun = 2, Xun nad Kun = 20, Gen nad Kun = 23, Dui nad Xun = 28).
9. Lenormand: promiješaj 36 karata, Grand Tableau; kaskada = zbrojevi susjednih karata u prvom redu od 9, zatim preostale karte redom.
10. Meihua Yishu: kineski lunarni datum (lunardate) i dvosatni ogranak po pravom sunčevom vremenu Zagreba; gornji trigram = (godina+mjesec+dan) mod 8, donji = (+sat) mod 8, pomična = (+sat) mod 6 (0 → 8 odn. 6); godina = redni broj zemaljskog ogranka (丙午 2026 = 7, 丁未 2027 = 8). Kaskada kao I Ching + oba zbroja.
11. Tarot: 78 karata; velike arkane 1-21, Luda 22, štapovi 23-36, pehari 37-50, mačevi 51-64, diskovi 65-78; kaskada = karte u redoslijedu miješanja s indeksom ≤ 45.
12. Pitagorejska: IVAN 19, životni put 29/11, rođendanski 31, mjesec kola, zbroj znamenki datuma kola, osobna godina (mjesec + dan rođenja + godina kola), osobni mjesec, osobni dan, univerzalni dan; nereducirano kad je ≤ 45.
13. Jyotiš: sideralno, Lahiri. Kaskada: redni broj nakšatre Mjeseca (1-27), broj njezina gospodara (Sunce 1, Mjesec 2, Jupiter 3, Rahu 4, Merkur 5, Venera 6, Ketu 7, Saturn 8, Mars 9) i njegova obitelj (+9, +18, +27, +36), raši Mjeseca i lagne, gospodari tekuće Vimšotari mahadaše i antardaše iz natalnog Mjeseca, vladar dana.
14. Zapadna: tropski. Kaskada: stupanj u znaku (1-30) Sunca, Mjeseca, Ascendenta, vladara dana, Pars Fortunae (dnevna/noćna formula) i natalnog PoF; zatim brojevi znakova tih točaka; zatim stupanj+30 za točke u 2. polovici zodijaka.
15. Plecko (filtar konvergencije): izvori = Statistika (sustavi 1-4 kao JEDAN izvor) + 10 ezoterijskih. Za svaki broj izbroji različite izvore koji ga nominiraju u tom kolu; signal = ≥ 2 izvora, šum = 1. Kombinacija = 6 brojeva s najviše izvora (nerješeno: češći u povijesti, pa manji); ako signala ima manje od 6, dopuni iz šuma istim redom i označi.

PRAVILO SETA (po kolu)
Nijedan par od 15 kombinacija ne dijeli više od 2 broja. Prekršaj: zamijeni zadnji broj kombinacije nižeg ranga sljedećim iz njezine kaskade i zabilježi zamjenu.

GODIŠNJI FIKSNI SET (10 kombinacija, nepromijenjene cijelu godinu)
Pareto 1, Pareto 2, Vruci 1, Vruci 2; Pitagorejska godišnja (osobna godina umjesto dnevnih brojeva); Jyotis natalna i Zapadna natalna (samo natalne točke); Kaldejska godišnja (bez imena dana i datuma, s godinom); Plecko godišnja = 6 brojeva koji su u najviše kola razdoblja bili signal; 10. = sljedećih 6 po istom kriteriju. Isto pravilo preklapanja ≤ 2.

VJEROJATNOSTI
Egzaktno, enumeracijom svih C(45,6) = 8.145.060 izvlačenja u numpyju, bez simulacije: za set jednog kola i za fiksni set P(bar jedna kombinacija s ≥3, ≥4, ≥5, 6 pogodaka), očekivani pogoci po kombinaciji, ulog po kolu i godišnje.

IZLAZ
- loto6_godina.csv (UTF-8 BOM, separator ;): datum;dan;vrijeme;sustav;n1;...;n6;joker;izvod;zamjena;seed — redak po kombinaciji za svako kolo u razdoblju.
- loto6_fiksni_set.csv, isti format.
- loto6_konvergencija.csv: datum;broj;broj_izvora;izvori.
- score.py: ulaz rezultati.csv (datum;n1;...;n6) → pogoci po sustavu i kolu, prosjek po sustavu, očekivanje 6·6/45 = 0,80 po kombinaciji, slučajna referenca (1.000 setova iste strukture), χ² i p. Prag: samo prosjek ≥ 1,00 pogodaka po kombinaciji na ≥ 50 kola smatra se odstupanjem.
- README.md: sva pravila, paketi i verzije, seed, ograničenja, što nije provjereno.
Kontrola efemerida: natalno Sunce ≈ 9-10° Blizanaca, natalni Mjesec sideralno u Purva Ašadhi.
