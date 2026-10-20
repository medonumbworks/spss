# Sve ili Ništa – 3 dodatna listića uz postojećih 10 (kolo 293/2026, utorak 20.10.2026.)

## 1. Rezultat

Tri nova listića, egzaktno optimalna za portfelj od 13:

11. ` 3  4  5  6 11 13 16 17 18 19 20`  (zajedničkih brojeva sa svakim od 10 postojećih: [6, 6, 6, 6, 6, 5, 5, 5, 5, 6])
12. ` 1  2  4  5  6  8  9 11 13 14 19`  (zajedničkih brojeva sa svakim od 10 postojećih: [6, 6, 5, 6, 6, 5, 5, 6, 5, 6])
13. ` 2  4  7  8 11 13 14 16 18 21 22`  (zajedničkih brojeva sa svakim od 10 postojećih: [5, 6, 6, 5, 5, 5, 5, 6, 6, 5])

Kriterij: maksimizirati vjerojatnost barem jednog dobitka portfelja (primarno), zatim barem jednog dobitka ≥ 260 € (sekundarno). Postignuta je **teorijska gornja međa**: svaki novi listić dodaje svih svojih 6 294 dobitnih ishoda bez ijednog preklapanja s ostalih 12. Bolje od toga matematički nije moguće.

| Portfelj | Listića | P(≥1 dobitak bilo koje kategorije) | P(≥1 dobitak ≥ 260 €) | P(≥1 glavni dobitak) | Očekivana isplata | Ulog | Očekivani neto |
| --- | --- | --- | --- | --- | --- | --- | --- |
| izvornih 10 | 10 | 8,6543 % (61050 od 705 432) | 0,3459 % (2440) | 0,00284 % (20) | 5,39 € | 13,00 € | -7,61 € |
| optimiziranih 13 | 13 | 11,3309 % (79932 od 705 432) | 0,4497 % (3172) | 0,00369 % (26) | 7,01 € | 16,90 € | -9,89 € |

Doprinos triju novih listića: +18882 dobitnih ishoda (+2,6767 % apsolutno, +30,9 % relativno), +732 ishoda ≥ 260 €, +6 ishoda glavnog dobitka. Očekivana isplata raste točno linearno (0,5390 € po listiću) i ne ovisi o izboru listića: nijedan izbor ne mijenja očekivanu vrijednost.

## 2. Zašto je ovo optimum: dobitna disjunktnost

Egzaktnim nabrajanjem svih 705 432 izvlačenja izračunato je u koliko izvlačenja **oba** listića dobivaju, ovisno o broju zajedničkih brojeva k:

| k zajedničkih | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oba dobivaju (ishoda) | 6294 | 2244 | 948 | 324 | 72 | 0 | 0 | 72 | 324 | 948 | 2244 | 6294 |
| oba ≥ 260 € (ishoda) | 244 | 44 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 44 | 244 |

Dva listića koja dijele **točno 5 ili 6 brojeva nikad ne dobivaju u istom kolu**, pa im se dobitni skupovi ne preklapaju. Dokaz: s k = 5 (ili 6) zajedničkih brojeva, da bi oba imala ≤ 2 pogotka izvan tih 11 brojeva mora pasti ≥ 7 izvučenih, a izvan ima ih samo 5 (6); da bi jedan imao ≥ 9, a drugi ≤ 2, zajednički dio morao bi istodobno sadržavati ≥ 3 i ≤ 2 pogotka. Komplement (k = 0) i duplikat (k = 11) preklapaju se potpuno, a „skoro-komplementi” (k = 1 ili 10) u 2 244 od 6 294 ishoda.

Među 352 716 klasa kandidata (listić i komplement imaju isti dobitni skup) točno **1816** ih ima k ∈ {5, 6} prema svih 10 postojećih. Svaki od njih dodaje točno 6 294 nova ishoda (egzaktno provjereno na uzorku od 300), a svaki drugi kandidat dodaje manje (provjereno na 300). Tri odabrana listića uzajamno također dijele 5 ili 6 brojeva, pa je svih 18 882 novih ishoda različito.

## 3. Postupak pretrage

1. **Egzaktan prolaz** kroz svih 352 716 klasa kandidata: za svaku je izračunano preklapanje njezina dobitnog skupa s unijom dobitnih skupova 10 postojećih listića (61 050 ishoda).
2. **Pohlepni odabir** s egzaktnim odbacivanjem (dobitak kandidata ≤ 6 294 − preklapanje s unijom 10), zatim **koordinatni uspon** sa zamjenama; 6 ponovnih pokretanja iz različitih prvih izbora. Svih 6 završilo je na istoj vrijednosti 79 932, što je gornja međa 61 050 + 3 × 6 294.
3. **Cjelobrojno programiranje i Monte Carlo** nisu bili potrebni: problem se egzaktno svodi na uvjet k ∈ {5, 6} prema svim listićima, a taj je uvjet provjeren iscrpno.
4. **Presuda izjednačenja:** 1816 klasa daje identičnu (maksimalnu) vrijednost svih triju kriterija. Odabrana su prva tri retka dnevnog CSPRNG popisa pokrića za kolo 293 koja zadovoljavaju uvjet prema svih 10 i međusobno (redci 12, 1510, 2115; preskočeno 2102 redaka). Time se izbjegavaju „lijepi” obrasci poput 1 2 3 4 5 6 7 … koje bi dao leksikografski odabir, a izbor ostaje provjerljiv iz zapisane entropije.

## 4. Usporedba s drugim načinima dodavanja 3 listića

| Strategija | Novi listići | P(≥1 dobitak) | Ishoda | P(≥1 ≥ 260 €) | Glavni dobitak (ishoda) | Prosj. k prema ostalima |
| --- | --- | --- | --- | --- | --- | --- |
| Samo izvornih 10 | — | 8,6543 % | 61050 | 0,3459 % | 20 | 5,31 |
| **10 + 3 optimizirana (ovaj odabir)** |  3  4  5  6 11 13 16 17 18 19 20<br> 1  2  4  5  6  8  9 11 13 14 19<br> 2  4  7  8 11 13 14 16 18 21 22 | 11,3309 % | 79932 | 0,4497 % | 26 | 5,41 |
| 10 + 3 slučajna (prosjek 300 trojki) | CSPRNG/PRNG slučajno | 11,0767 % | 78 138 | 0,4493 % | 26 | — |
| 10 + 3 slučajna (najbolja od 300) | — | 11,2504 % | 79364 | — | 26 | — |
| 10 + 3 frekvencijska (najčešći ukupno, najčešći zadnjih 30, najdulje „zakašnjeli”) |  1  3  5  7  8 15 16 17 19 21 22<br> 1  5  9 10 11 12 14 15 17 19 22<br> 1  2  6 11 13 16 17 18 19 20 22 | 11,1892 % | 78932 | 0,4497 % | 26 | 5,40 |
| 10 + 3 najveće raznolikosti (najveća najmanja udaljenost do listića ili komplementa) |  1  2  3  4  5  6  7  9 10 11 15<br> 1  2  3  4  5  6 12 13 14 16 20<br> 1  2  3  4  7 11 12 16 17 18 19 | 11,3309 % | 79932 | 0,4497 % | 26 | 5,46 |
| 10 + 3 naivne raznolikosti (najveća Hammingova udaljenost, bez točnih komplemenata) |  1  2  3  8 11 12 13 14 17 20 21<br> 1  3  6  7  8  9 10 11 13 16 17<br> 1  8  9 10 14 15 16 17 18 20 21 | 11,0568 % | 77998 | 0,4485 % | 26 | 5,08 |

Optimizirani odabir bolji je od 100,0 % slučajnih trojki i od najbolje slučajne trojke. Frekvencijski listići (vrući/zakašnjeli brojevi) nemaju prediktivnu vrijednost (odjeljak 6) i uz to se dobitno preklapaju s postojećima. „Naivna raznolikost” bira skoro-komplemente, koji su najgori mogući izbor: dijele 2 244 od 6 294 dobitnih ishoda s listićem kojemu su skoro-komplement. Ispravno definirana raznolikost (udaljenost do listića *ili njegova komplementa*) poklapa se s optimumom.

## 5. Raspodjela ukupne isplate portfelja (egzaktno, po izvlačenju)

| Ukupna isplata | Izvornih 10: ishoda | Izvornih 10: vjerojatnost | Optimiziranih 13: ishoda | Optimiziranih 13: vjerojatnost |
| --- | --- | --- | --- | --- |
| 0 € | 644382 | 91,34573 % | 625500 | 88,66907 % |
| 26 € | 56846 | 8,05832 % | 74996 | 10,63122 % |
| 52 € | 1746 | 0,24751 % | 1746 | 0,24751 % |
| 78 € | 18 | 0,00255 % | 18 | 0,00255 % |
| 260 € | 2312 | 0,32774 % | 3038 | 0,43066 % |
| 286 € | 108 | 0,01531 % | 108 | 0,01531 % |
| 80 000 € | 20 | 0,00284 % | 26 | 0,00369 % |

Vjerojatnost dvaju ili više dobitaka u istom kolu: izvornih 10 0,2654 %, optimiziranih 13 0,2654 % (nove listiće nikad ne prati dobitak zajedno s nekim drugim listićem).

## 6. Ponovna procjena prediktivne moći

**Bayes.** 656 draws, 7216 ball slots. Per-number counts range 299 (no. 13) to 350 (nos. 7 and 15); null mean 328. Beta(1,1) posterior means 0.456-0.533; 95% intervals: only no. 13 excludes 0.5 ([0.418,0.494]), nos. 7/15 are [0.495,0.571]; one of 22 intervals missing 0.5 is exactly what 22 independent-looking tests produce at 5%. P(p_n>0.5): min 0.012 (no. 13), max 0.957 (nos. 7, 15). Beta(50,50) shrinks to 0.462-0.529 and P(p>0.5) 0.017-0.945 (CI for no. 13 then [0.426,0.497]). Dispersion test with the 11-of-22 covariance (Var=N/4, Cov=-N/84, so sum (c-N/2)^2/((N/4)(22/21)) ~ chi2_21): Q=21.70, p=0.417 analytic, 0.452 by 500-history Monte Carlo - the spread of counts is exactly what uniform draws give. Hierarchical method of moments: sample variance of rates 4.125e-4 vs null expectation 3.992e-4 (MC 4.03e-4) -> tau^2=1.33e-5, tau=0.0036, i.e. true inclusion rates differ by at most ~0.4 percentage points around 0.5 and even that is within the sampling noise of tau itself. Bayes factors for 'all p_n=0.5' vs free rates: product-binomial Beta(1,1) marginal likelihood (variance-correct) 8.1e23 (log10 23.9); Beta(50,50) 2.5e5; Dirichlet-multinomial on ball slots alpha=1: 5.3e24, alpha=0.5: 3.7e28, alpha=50: 1.2e7; BIC approximation with n=656 draws 4.3e24 (with n=7216 slots 1.1e38). Every variant is decisive for the uniform model. Implication for picking numbers: the posterior ranking is noise. Even taking the Beta(1,1) posterior means literally and playing the 11 highest (7,15,19,16,3,17,1,8,5,21,22) would multiply the jackpot probability by only 1.22 (1/352716 -> about 1/288000) under a conditional-Bernoulli model, and the walk-forward 'long-run frequency' ticket shows that ranking does not even persist out of sample (5.504 hits/draw vs 5.5). No number deserves preference; the complement symmetry means the 'cold' ticket is equally (un)attractive.

**Entropija i slučajnost.** Shannon entropy of the empirical number distribution 4.45829 bits vs log2(22)=4.45943; the 0.0011-bit deficit equals the finite-sample deficit of uniform histories (500-history MC mean 4.45832, sd 0.00034, z=-0.08). Compression of the 656x22 incidence matrix packed to 1804 bytes: zlib(9) ratio 1.00610 observed, simulated 1.00610 +- 0.0 (z=0; degenerate because zlib/lzma emit incompressible input as stored blocks with fixed overhead, lzma 1.0333 likewise identical for all histories); the informative variants are bz2 1.2373 vs 1.2192 +- 0.0087 (z=+2.08, observed LESS compressible than 99% of uniform histories) and zlib on the sorted numbers-as-bytes encoding (7216 bytes) 0.4020 vs 0.3988 +- 0.0017 (z=+1.86, also less compressible). No compressor finds structure. Serial test: overlap of consecutive draws vs Hypergeometric(22,11,11) in bins <=3,4,5,6,7,>=8: chi2=4.54 on 5 df, p=0.474 (MC p=0.456); mean lag-1 overlap 5.565 vs 5.5 (z=+1.38); lag 1..10 overlap z-scores +1.38,+0.46,+1.03,-0.36,-0.28,-1.27,-0.51,+1.54,+1.88,+0.46 (none significant after 10 lags); lag-1 conditional entropy H(X_t|X_t-1)=0.99990 bits, mutual information 0.0001 bits, P(1|1)=0.506, P(1|0)=0.494. Other diagnostics: Wald-Wolfowitz runs test aggregated over the 22 number series z=-1.42 (p=0.16), 0/22 numbers individually significant; sum of drawn numbers mean 127.1 vs 126.5 (z=+0.45); even-count mean 5.41 vs 5.5 (z=-1.89, p=0.06); weekday x number association MC p=0.20; pooled gap distribution vs Geometric(1/2) chi2=4.81 on 5 df, p=0.44. All axes are consistent with i.i.d. uniform 11-of-22 draws.

**Strojno učenje (kronološka validacija).** Per-number LR is WORSE than the coin-flip baseline out of sample: log-loss 0.7371 vs ln2=0.6931 (paired per-draw t=+12.1 in the wrong direction; MC p that a null draw sequence would give an equal or lower log-loss = 0.22), Brier 0.2680 vs 0.25, AUC 0.5065 (MC p=0.12). It is badly overconfident: predicted deciles 0.26 -> observed 0.49, 0.75 -> observed 0.51; predicted probabilities span 0.015-0.988 with no corresponding signal. Its top-11 ticket scored 5.564 hits/draw vs 5.5 (z=+1.13, p=0.26 normal, 0.24 MC) and produced FEWER extreme outcomes (<=3 or >=8 hits) than chance: 0.0768 vs 0.0861 (35 of 456 draws; MC p=0.79), hits histogram 2:2,3:15,4:64,5:143,6:126,7:88,8:17,9:1, never 0/1/10/11. Stronger ridge (L2=10): log-loss 0.7145, hits 5.546 (z=0.82), extreme 0.0724. Pooled LR (shared coefficients): log-loss 0.6944, AUC 0.491, hits 5.496, extreme 0.1075 (MC p=0.07). Markov lag-1 per number: log-loss 0.6957, AUC 0.499, hits 5.478. Pooled gap-hazard ('due' numbers): log-loss 0.6938, hits 5.546 (tie-break average 5.489). Hot-30 rule: hits 5.393 (z=-1.91, p=0.056), its exact complement cold-30: 5.607 (same |z|, extreme 0.0680) - the smallest p among all tests. Hot-90: 5.456; long-run frequency: 5.504. Random-ticket baseline (2000 random tickets per draw vs real outcomes): 5.5017 hits, extreme 0.0861; the SD of mean hits across 4000 one-random-ticket-per-draw strategies is 0.0557 vs theory 1.2002/sqrt(456)=0.0562, validating the z formula. Holm-Bonferroni over 45 model x metric tests: min raw p=0.056, min adjusted p=1.0. If the per-number LR probabilities were literally true the top ticket's jackpot odds would be 50x better (median 23x) - but those probabilities are refuted by the out-of-sample log-loss; under the pooled LR the implied multiplier is 1.19 and under the static posterior-mean ticket 1.22. Realised predictive power: none.

| Model | Metrika | Vrijednost | Slučajna osnova | p |
| --- | --- | --- | --- | --- |
| logreg_pooled_L2=1 (shared coefficients, same features) | mean log-loss (MC one-sided p) | 0,6944 | 0,6931 | 0,967 |
| logreg_pooled_L2=1 | top-11 mean hits (two-sided normal p) | 5,4956 | 5,5000 | 0,938 |
| logreg_pooled_L2=1 | top-11 extreme fraction (MC one-sided p) | 0,1075 | 0,0861 | 0,070 |
| logreg_per_number_L2=10 | mean log-loss (MC one-sided p) | 0,7145 | 0,6931 | 0,141 |
| logreg_per_number_L2=10 | top-11 mean hits (two-sided normal p) | 5,5461 | 5,5000 | 0,413 |
| markov_lag1_per_number (Laplace-smoothed transition rates) | mean log-loss (MC one-sided p) | 0,6957 | 0,6931 | 0,569 |
| markov_lag1_per_number | AUC (MC one-sided p) | 0,4994 | 0,5000 | 0,548 |
| markov_lag1_per_number | top-11 mean hits (two-sided normal p) | 5,4781 | 5,5000 | 0,696 |
| gap_hazard_pooled ('due' numbers, hazard by current gap) | mean log-loss (MC one-sided p) | 0,6938 | 0,6931 | 0,678 |
| gap_hazard_pooled | top-11 mean hits (two-sided normal p; tie-break average 5.489) | 5,5461 | 5,5000 | 0,413 |
| rolling_freq_30_hot (11 hottest numbers of last 30 draws) | top-11 mean hits (two-sided normal p; tie-break average 5.391) | 5,3925 | 5,5000 | 0,056 |
| rolling_freq_30_hot | mean log-loss of shrunk window frequency (MC one-sided p) | 0,7041 | 0,6931 | 0,808 |
| cold_30 (exact complement of hot-30 ticket) | top-11 mean hits (two-sided normal p; mirror of hot-30) | 5,6075 | 5,5000 | 0,056 |
| cold_30 (exact complement of hot-30 ticket) | top-11 extreme fraction (MC one-sided p) | 0,0680 | 0,0861 | 0,931 |
| rolling_freq_90_hot | top-11 mean hits (two-sided normal p) | 5,4561 | 5,5000 | 0,435 |
| longrun_freq_hot (11 most frequent numbers over all prior draws) | top-11 mean hits (two-sided normal p) | 5,5044 | 5,5000 | 0,938 |
| longrun_freq_hot | top-11 extreme fraction (MC one-sided p) | 0,1031 | 0,0861 | 0,110 |
| random_ticket_baseline (2000 random tickets per draw vs real outcomes) | mean hits | 5,5017 | 5,5000 | — |
| random_ticket_baseline | extreme fraction | 0,0861 | 0,0861 | — |
| Holm-Bonferroni over all 45 model x metric tests | min adjusted p (min raw p = 0.0559) | 1,0000 | 0,0500 | 1,000 |

Zaključak neovisne dijagnostike: prediktivna moć nije pronađena. The 656-draw history (01.01.2025-19.10.2026, one empty row skipped, Dan column matches the dates, one 2-day calendar gap) is statistically indistinguishable from i.i.d. uniform 11-of-22 draws on every axis examined: marginal frequencies (chi2 dispersion p=0.42, hierarchical between-number SD 0.0036, Bayes factors of 1e5-1e24 for 'all rates = 0.5' depending on prior), Shannon entropy (z=-0.08 vs uniform histories), compressibility (no compressor beats the uniform-history reference; bz2 and byte encodings find the real data slightly LESS compressible), serial dependence (lag-1 overlap chi2 p=0.47, conditional entropy 0.99990 bits, lags 1-10 all |z|<1.9), runs, gaps, sums, parity and weekday effects. In strictly chronological walk-forward validation over 456 test draws, no predictor beats the random baseline: the per-number logistic regression is significantly WORSE than a coin flip (log-loss 0.737 vs 0.693, Brier 0.268 vs 0.25) because it fits noise, its AUC is 0.507, and its top-11 ticket scores 5.56 hits/draw (z=+1.1) with fewer extreme outcomes than chance (0.077 vs 0.086). Pooled LR, Markov lag-1, gap-hazard, hot/cold 30/90-draw and long-run frequency rules all land within noise (|z|<=1.9); the smallest raw p of 45 tests is 0.056 and the Holm-adjusted minimum is 1.0. Even the most charitable reading - taking the Beta(1,1) posterior means as true and playing the 11 highest - would raise the jackpot probability by a factor of 1.22 (1/352716 -> ~1/288000), moving the expected return of a 1.30 EUR ticket from 0.539 EUR (41.5%) to 0.590 EUR (45.4%) under the stated prize table, still a loss of more than half the stake; the realised out-of-sample edge is zero. predictive_power_found=false. Outputs: /tmp/claude-0/-home-user-spss/133e74c5-7dba-5a24-94e5-4c9205b6e08b/scratchpad/work/portfolio/agents/diag/diag_analysis.py (script), .../diag/results.json (all numbers incl. per-number posteriors, MC null distributions, calibration, economics), .../diag/run_log.txt (console log).

Ranije analize (dvije neovisne, šest skeptičnih provjera, dva backtesta, provjera izvan uzorka na 21 novom kolu) ostaju na snazi: frekvencijske anomalije, parovi, trojke, vremenski obrasci i „vrući”/„hladni” brojevi ne preživljavaju korekciju za višestruka testiranja i ne ponavljaju se izvan uzorka. Izvlačenja su u skladu s neovisnim uniformnim izvlačenjem 11 od 22. Zato optimizacija portfelja ne pokušava pogoditi brojeve nego samo ukloniti preklapanje dobitnih ishoda među listićima, što je jedino što se matematički može poboljšati.

## 7. Neovisna provjera (drugi, neovisno napisani program)

- Izvornih 10: unija dobitnih ishoda 61050 (ovdje 61050), ≥ 260 €: 2440 (ovdje 2440), glavni: 20 (ovdje 20), očekivana isplata 5,3899 € (ovdje 5,3899 €).
- Najbolja 3 dodatna prema neovisnoj pretrazi (Exact. All 352716 candidate 11-subsets containing number 1 (the other 352716 are complements with identical prize sets) were evaluated exactly at every step: for the union U of the current portfolio, ): unija 13 = 79932 (ovdje 79932); listići [[1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14], [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17], [1, 3, 4, 5, 8, 9, 11, 12, 15, 17, 19]].
- Tablica f(k) poklapa se: True.
- Slučajne trojke (neovisno): prosjek 78 139, maksimum 79 288.
- Napomena provjeravatelja: All numbers are exact counts over the 705432 enumerated draws (22-bit integers, numpy bitwise_count); runtime ~20 s. Results JSON: /tmp/claude-0/-home-user-spss/133e74c5-7dba-5a24-94e5-4c9205b6e08b/scratchpad/work/portfolio/agents/portfolio/results.json. Sanity checks passed: single ticket wins in 6294 draws (2 jackpot + 242 at 260 + 6050 at 26), payout sum 380220, EV 0.5389889 EUR. PORTFOLIO 10 (unchanged): union_all 61050 (max possible 62940, so 1890 draws lost to overlap), union_hi 2440 (max 2440, no overlap in the high tiers), union_jp 20 (max 20); P(any prize)=0.086543, P(>=260)=0.003459,

## 8. Konačna provjera

- Svaki od 3 nova listića ima točno 11 različitih cijelih brojeva od 1 do 22, uzlazno sortiranih.
- Nijedan novi listić nije duplikat ni komplement nijednog od 10 postojećih ni drugog novog listića (13 različitih, bez komplemenata).
- Svaki pojedini listić dobiva u točno 6 294 izvlačenja (2 glavna + 242 po 260 € + 6 050 po 26 €; ishoda ≥ 260 €: 244), kako teorija nalaže.
- Unija dobitnih ishoda 13 listića je 79932 = 61050 + 3 × 6 294: svaki novi listić dodaje sve svoje dobitne ishode bez preklapanja (gornja međa postignuta).
- Očekivana isplata iznosi točno 0,538989 € po listiću (10 listića 5,39 €, 13 listića 7,01 €) i ne ovisi o izboru brojeva.
- Svaki novi listić dijeli točno 5 ili 6 brojeva sa svakim od 10 postojećih i s druga dva nova, što je nužan i dovoljan uvjet dobitne disjunktnosti.
- Ne postoji nijedno izvlačenje u kojem istodobno dobiva neki novi i neki postojeći listić (egzaktno: 0 od 705 432).
- Neovisni izračun preko matrice incidencije poklapa se s izračunom preko bit-maski (portfolio.py): True.

## 9. Ograničenja

- Iznosi 260 € i 26 € potječu iz sekundarnih izvora i nisu potvrđeni u službenim pravilima; glavni dobitak od 80 000 € potvrđen je u medijima. Ako niže kategorije ne postoje, optimalnost ostaje (disjunktnost vrijedi i za skup {0, 11}), ali vjerojatnosti „bilo kojeg dobitka” treba čitati kao vjerojatnosti glavnog dobitka.
- Portfelj od 13 listića ima 11,33 % šanse za bilo kakav dobitak, ali očekivani neto rezultat je gubitak od oko 9,89 € na ulog od 16,90 €. Optimizacija ne mijenja očekivani povrat, samo uklanja uzaludno dupliranje dobitnih ishoda.