"""Human-readable report (Croatian) built only from the computed results."""
from __future__ import annotations

from methods.common import MethodOutput

STATUS_HR = {"ran": "izvršeno", "partial": "djelomično", "failed": "neuspjelo", "skipped": "preskočeno",
             "not_run": "nije pokrenuto"}
# Separate map: these agree with the masculine "backtest", not with neuter "izvršeno".
BACKTEST_STATUS_HR = {"ran": "izvršen", "reused": "ponovno korišten", "not_run": "nije pokrenut"}
STRATEGY_HR = {
    "ensemble_equal": "Ensemble (Borda, jednake težine) – primarni",
    "ensemble_train_weighted": "Ensemble (težine iz razdoblja treniranja)",
    "control_hot_frequency": "Kontrola: najčešći u zadnjih 30 kola",
    "control_logistic": "Kontrola: logistička regresija (sklearn)",
}
CHECKS_HR = {"count and range per pool": "broj i raspon brojeva", "no duplicates within a draw": "bez duplikata u kolu",
             "chronological order": "kronološki redoslijed", "unique dates": "jedinstveni datumi",
             "sum_column matches": "stupac Zbroj odgovara", "even_column matches": "stupac Parni odgovara"}
REASON_HR = {"no historical draw file supplied": "nije priložena datoteka s povijesnim izvlačenjima",
             "--skip-backtest": "pokrenuto s --skip-backtest"}


def _fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _p(x):
    if x is None:
        return "–"
    return "<0.0001" if x < 1e-4 else f"{x:.4f}"


def _nums(xs):
    return " ".join(f"{n:02d}" for n in xs)


def _ticket(game_key: str, picks: dict) -> str:
    if game_key == "eurojackpot":
        return f"{_nums(picks['main'])} + {_nums(picks['euro'])}"
    if game_key == "joker":
        return "".join(str(d) for d in picks["digits"])
    return _nums(picks["main"])


def _label(key: str, outputs: dict[str, MethodOutput]) -> str:
    if key in outputs:
        return outputs[key].label
    return STRATEGY_HR.get(key, key)


def render(cfg: dict, manifest: dict, combinations: dict, backtests: dict, theory: dict,
           outputs: dict[str, MethodOutput]) -> str:
    L: list[str] = []
    inp = manifest["inputs"]
    L += ["# Izvještaj: ezoterično generiranje lutrijskih brojeva", "",
          "> Lutrijska izvlačenja su slučajna. Nijedna metoda u ovom izvještaju ne mijenja vjerojatnost dobitka; "
          "ovo je istraživanje koje to *mjeri*, a ne preporuka za igru. Svi brojevi niže izračunati su ovim "
          "pipelineom; ništa nije procijenjeno ručno.", "",
          "## Ulazni podaci", "",
          f"* Rođenje: {inp['birth_local']} ({inp['birth_place']}) = {inp['birth_utc']} UTC",
          f"* Izvlačenje: {inp['draw_local']} ({inp['draw_place']}) = {inp['draw_utc']} UTC",
          f"* Ime: {'zadano' if cfg['subject'].get('name') else 'nije zadano (izračuni koji traže ime su preskočeni)'}",
          f"* Seed: {cfg['seed']}; konfiguracija SHA-256 `{manifest['config_sha256'][:16]}…`, "
          f"kod SHA-256 `{manifest['code_sha256'][:16]}…`"]
    n_readings = combinations.get("n_readings", 1)
    if n_readings > 1:
        seeds = ", ".join(str(r["seed"]) for r in combinations["readings"])
        L.append(f"* Očitanja: {n_readings} (seedovi {seeds}). Očitanje k koristi seed {cfg['seed']} + (k − 1): njime "
                 "se ponovno bacaju simulirani dijelovi orakula (I Ching, miješanje tarota, Lenormandov Grand "
                 "Tableau) i ponovno izvlači SHA-256 tie-break svake metode, a tie-break ensemblea dodatno se soli "
                 "rednim brojem očitanja. Sirovi brojevi Meihue, zapadne i vedske astrologije, pitagorejske i "
                 "kaldejske numerologije te gematrije ne ovise o seedu; kod njih se mijenja samo popuna brojeva koje "
                 "metoda nije sama odredila. Očitanje 1 je standardno pokretanje.")
    L.append("")

    L += ["## Što se stvarno izvršilo", "", "| Metoda | Status | Komponenta | Status komponente | Napomena |",
          "|---|---|---|---|---|"]
    for key, m in manifest["methods"].items():
        first = True
        for c in m["components"]:
            L.append(f"| {_label(key, outputs) if first else ''} | {STATUS_HR.get(m['status'], m['status']) if first else ''} "
                     f"| {c['name']} | {STATUS_HR.get(c['status'], c['status'])} | {c['detail']} |")
            first = False
    L.append("")
    for g, b in manifest["backtests"].items():
        reason = f" – {REASON_HR.get(b['reason'], b['reason'])}" if b.get("reason") else ""
        if b.get("reused_from"):
            reason = (f" – iz `{b['reused_from']['summary']}`; backtest ne ovisi o datumu ciljanog izvlačenja, a "
                      "konfiguracija, povijesni podaci i kod koji ga računa provjereno su nepromijenjeni (SHA-256)")
        L.append(f"* Backtest {g}: **{BACKTEST_STATUS_HR.get(b['status'], b['status'])}**{reason}")
    L.append("")

    if n_readings > 1:
        tw_label = STRATEGY_HR["ensemble_train_weighted"]
        L += ["## Listići po očitanjima", "",
              "Primarni listić svakog očitanja je ensemble s jednakim težinama (Borda). Svi listići imaju jednaku "
              "vjerojatnost dobitka; razlike među njima dolaze samo iz ponovno bačenih simuliranih dijelova orakula "
              "i ponovno izvučenog tie-breaka. Više listića znači više odigranih kombinacija, a ne bolje "
              "kombinacije.", ""]
        for game_key, tickets in combinations["tickets"].items():
            L += [f"### {combinations['games'][game_key]['label']}", "",
                  f"| Očitanje | Seed | Ensemble (jednake težine) | {tw_label} |", "|---|---|---|---|"]
            for t, r in zip(tickets, combinations["readings"]):
                ens = r["games"][game_key]["ensembles"]
                tw = next((e["picks"] for k, e in ens.items() if k.startswith("ensemble_train")), None)
                L.append(f"| {t['reading']} | {t['seed']} | **{_ticket(game_key, t['picks'])}** | "
                         f"{_ticket(game_key, tw) if tw else '–'} |")
            if backtests.get(game_key, {}).get("status") not in ("ran", "reused"):
                L += ["", f"Za ovu igru nema ensemblea «{tw_label}» (stupac je prazan) jer backtest nije pokrenut: "
                      f"{REASON_HR.get(backtests.get(game_key, {}).get('reason'), backtests.get(game_key, {}).get('reason'))}. "
                      "Nijedan listić za ovu igru nije ni na čemu testiran."]
            for pool_name, c in combinations["coverage"].get(game_key, {}).items():
                size = next(p["size"] for p in cfg["games"][game_key]["pools"] if p["name"] == pool_name)
                pairs = ", ".join(f"{i + 1}–{j + 1}: {c['pairwise_common'][i][j]}"
                                  for i in range(len(tickets)) for j in range(i + 1, len(tickets)))
                L += ["", f"Pokrivenost (`{pool_name}`): različitih listića {c['distinct_tickets']} od {len(tickets)}; "
                      f"pokriveni brojevi {c['numbers_covered']}/{size}; ni na jednom listiću: "
                      f"{', '.join(map(str, c['numbers_never_played'])) or 'nema'}. Zajednički brojevi po parovima "
                      f"očitanja: {pairs}. (Pokrivenost je opis odnosa među listićima, ne mjera kvalitete: "
                      "očekivani broj pogodaka svakog pojedinog listića je isti.)"]
            L.append("")
        bad = [(r["reading"], k, v) for r in manifest["readings"] for k, v in (r.get("not_ran") or {}).items()
               if v["status"] == "failed"]
        if bad:
            L += ["", "**Metode koje su pale u pojedinim očitanjima** (njihovi brojevi ne ulaze u listić tog "
                  "očitanja):", ""]
            for reading, key, v in bad:
                L.append(f"* očitanje {reading}, `{key}`: {v['error']}")
            L.append("")

    L += [f"## Konačne kombinacije{' (očitanje 1)' if n_readings > 1 else ''}", "",
          "Stupac *iz metode* kaže koliko je brojeva metoda sama odredila; ostatak popunjava reproducibilni "
          "SHA-256 tie-break (vidi README, „Mapiranje u raspon”).", ""]
    for game_key, g in combinations["games"].items():
        L += [f"### {g['label']}", ""]
        if game_key == "eurojackpot":
            L += ["| Metoda | Glavni brojevi (5/50) | Euro brojevi (2/12) | Iz metode |", "|---|---|---|---|"]
            for k, m in g["methods"].items():
                d = m["derived_from_method"]
                L.append(f"| {m['label']} | {_nums(m['picks']['main'])} | {_nums(m['picks']['euro'])} "
                         f"| {d['main']}/5, {d['euro']}/2 |")
            for k, e in g["ensembles"].items():
                L.append(f"| **{STRATEGY_HR.get(k, k)}** | **{_nums(e['picks']['main'])}** | **{_nums(e['picks']['euro'])}** | – |")
        elif game_key == "joker":
            L += ["| Metoda | Joker | Iz metode |", "|---|---|---|"]
            for k, m in g["methods"].items():
                L.append(f"| {m['label']} | {''.join(str(d) for d in m['picks']['digits'])} "
                         f"| {m['derived_from_method']['digits']}/6 |")
            for k, e in g["ensembles"].items():
                L.append(f"| **{STRATEGY_HR.get(k, k)}** | **{''.join(str(d) for d in e['picks']['digits'])}** | – |")
        else:
            L += ["| Metoda | Brojevi (11/22) | Iz metode |", "|---|---|---|"]
            for k, m in g["methods"].items():
                L.append(f"| {m['label']} | {_nums(m['picks']['main'])} | {m['derived_from_method']['main']}/11 |")
            for k, e in g["ensembles"].items():
                name = STRATEGY_HR.get("ensemble_train_weighted" if k.startswith("ensemble_train") else k, k)
                L.append(f"| **{name}** | **{_nums(e['picks']['main'])}** | – |")
        L.append("")

    L += ["## Teorijske vjerojatnosti (matematika, ne procjena)", ""]
    sin = theory.get("sve_ili_nista")
    if sin:
        L.append(f"* Sve ili Ništa: {sin['combinations']:,} kombinacija; P(11 pogodaka) = P(0 pogodaka) = "
                 f"1/{sin['one_in']:,}; očekivano 5.5 pogodaka po listiću bez obzira na metodu.".replace(",", "."))
    ej = theory.get("eurojackpot")
    if ej:
        L.append(f"* Eurojackpot: P(5+2) = 1/{ej['one_in']:,}.".replace(",", "."))
    jk = theory.get("joker")
    if jk:
        L.append(f"* Joker: P(svih 6 znamenki na svojim mjestima) = 1/{jk['one_in']:,}.".replace(",", "."))
    L.append("")

    for game_key, bt in backtests.items():
        L += [f"## Backtest: {game_key}", ""]
        if bt.get("status") not in ("ran", "reused"):
            L += [f"Nije pokrenut: {REASON_HR.get(bt.get('reason'), bt.get('reason'))}. Bez povijesnih izvlačenja nema evaluacije; "
                  "kombinacije za ovu igru nisu testirane ni na čemu.", ""]
            continue
        if bt.get("status") == "reused":
            rf = bt["reused_from"]
            L += [f"Ponovno korišten iz `{rf['summary']}`, bez ponovnog izračuna. Backtest ne ovisi o datumu ciljanog "
                  "izvlačenja, nego o konfiguraciji, povijesnim podacima i kodu koji ga računa; sve troje je "
                  f"provjereno nepromijenjeno (SHA-256 konfiguracije `{rf['config_sha256'][:16]}…`, koda "
                  f"`{rf['backtest_code_sha256'][:16]}…`, i povijesnih datoteka). Pokretanje koje ga je izračunalo "
                  "navedeno je u `manifest.json` (`backtests.*.reused_from`).", ""]
        v, sp = bt["validation"], bt["split"]
        L += ["### Podaci", "",
              f"* Datoteka `{v['file']}`: {v['rows_in_file']} redaka, {v['complete_draws']} potpunih izvlačenja "
              f"({v['first_date']} – {v['last_date']}).",
              f"* Bez brojeva (izostavljeno): {', '.join(r['label'] + ' (' + r['date'] + ')' for r in v['rows_without_numbers']) or 'nema'}.",
              f"* Kalendarske praznine: {', '.join(v['calendar_gaps']) or 'nema'}.",
              f"* Provjere (sve prošle): {', '.join(CHECKS_HR.get(c, c) for c in v['checks_passed'])}."]
        if "xlsx_crosscheck" in v:
            x = v["xlsx_crosscheck"]
            L.append(f"* Usporedba s `{x['file']}` (list „{x['sheet']}”): "
                     f"{'identično' if x['identical'] else 'RAZLIKE: ' + ', '.join(x['mismatched_dates'] + x['only_in_xlsx'])}.")
        L += [f"* Treniranje: {sp['n_training']} kola (do {sp['training_last_date']}); evaluacija: "
              f"{sp['n_evaluation']} kola ({sp['evaluation_first_date']} – {sp['evaluation_last_date']}).",
              f"* Pretpostavljeno vrijeme izvlačenja u povijesti: {cfg['history'][game_key]['draw_time']} "
              "(datoteke ne sadrže vrijeme).", ""]
        if bt.get("failed_methods"):
            L += [f"* Metode koje su pale u backtestu (isključene): {', '.join(bt['failed_methods'])}", ""]

        for pool_name, pr in bt["results"]["pools"].items():
            fam = pr["family_tests"]
            # Methods first, then ensembles and controls. A reused summary comes back from
            # JSON with alphabetically sorted keys, so the order is taken from the list.
            order = [s for s in bt.get("strategies", []) if s in pr["strategies"]] or list(pr["strategies"])
            rows = [(s, pr["strategies"][s]) for s in order]
            L += [f"### Evaluacijsko razdoblje (bazen `{pool_name}`)", "",
                  f"Nulta hipoteza: listić je fiksiran prije izvlačenja, a izvlačenje je uniformno slučajno → "
                  f"pogoci su hipergeometrijski, očekivano {pr['strategies'][order[0]]['evaluation']['expected_mean_hits']:.3f} "
                  f"po kolu (SD {pr['sd_hits_null']:.3f}). Obitelj testova: {pr['family_size']} strategija, "
                  f"α = {pr['alpha']}, Holm (primarno) i Benjamini-Hochberg.", "",
                  "| Strategija | Prosjek pogodaka | 95% CI (t) | 95% CI (bootstrap) | Razlika | p (egzaktni) | p (MC) | p Holm | p BH | AUC [95% CI] |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for s, e in rows:
                ev = e["evaluation"]
                t_lo, t_hi = ev["ci95_t"]
                b_lo, b_hi = ev["ci95_bootstrap"]
                auc = ev["auc"]
                L.append(f"| {_label(s, outputs)} | {ev['mean_hits']:.3f} | [{t_lo:.3f}, {t_hi:.3f}] | "
                         f"[{b_lo:.3f}, {b_hi:.3f}] | {ev['difference']:+.3f} | {_p(fam[s]['p_exact_two_sided'])} | "
                         f"{_p(ev['mc_p_two_sided'])} | {_p(fam[s]['p_holm'])} | {_p(fam[s]['p_bh'])} | "
                         f"{auc['auc']:.3f} [{auc['ci95'][0]:.3f}, {auc['ci95'][1]:.3f}] |")
            null = pr["random_ticket_null"]
            reps = f"{null['replicates']:,}".replace(",", ".")
            L += ["", f"Uniformna slučajna osnovica ({reps} nasumičnih strategija na istim kolima): "
                  f"prosjek {null['mean']:.3f}, 95% raspon [{null['quantiles']['0.025']:.3f}, "
                  f"{null['quantiles']['0.975']:.3f}]. *p (MC)* je udio tih strategija koje od 5.5 odstupaju "
                  "barem koliko i promatrana.",
                  f"Najmanji učinak koji ovaj test s 80% snage može otkriti (Bonferroni, {pr['family_size']} testova): "
                  f"±{pr['minimum_detectable_effect']:.3f} pogodaka po kolu.", ""]

            any_uncorr = [s for s in fam if fam[s]["significant_uncorrected"]]
            any_holm = [s for s in fam if fam[s]["significant_holm"]]
            any_bh = [s for s in fam if fam[s]["significant_bh"]]
            L += ["**Višestruko testiranje.** "
                  + (f"Bez korekcije p < α: {', '.join(_label(s, outputs) for s in any_uncorr)}. " if any_uncorr
                     else "Nijedna strategija nema p < α ni bez korekcije. ")
                  + (f"Nakon Holmove korekcije značajno: {', '.join(_label(s, outputs) for s in any_holm)}."
                     if any_holm else "Nakon Holmove korekcije nijedna strategija nije značajna.")
                  + (f" BH (FDR): {', '.join(_label(s, outputs) for s in any_bh)}." if any_bh
                     else " Ni BH (FDR) ne ostavlja nijednu."), ""]

            first = rows[0][1]["evaluation"]
            # Numeric, not lexicographic: a reused summary's keys come back as sorted strings.
            ks = sorted(first["hit_distribution"], key=int)
            n_sec = 2 * pr["family_size"]
            L += ["#### Distribucija broja pogodaka (evaluacija)", "",
                  f"χ² (usporedba cijele distribucije s H0) i stopa ekstrema su *sekundarne, eksploratorne* mjere bez "
                  f"korekcije: na {n_sec} takvih testova slučajno se očekuje oko {0.05 * n_sec:.1f} p-vrijednosti < 0.05.",
                  "",
                  "| Strategija | " + " | ".join(ks) + " | χ² p | ekstremi (≤3 ili ≥8) |",
                  "|---|" + "---|" * len(ks) + "---|---|",
                  "| *očekivano (H0)* | " + " | ".join(f"{first['expected_distribution'][k]:.1f}" for k in ks)
                  + f" | – | {first['extreme_rate']['expected_rate'] * first['n_draws']:.1f} |"
                  if "extreme_rate" in first else ""]
            for s, e in rows:
                ev = e["evaluation"]
                ext = ev.get("extreme_rate")
                ext_s = (f"{ext['count']} ({ext['rate']:.3f}, CI [{ext['ci95_wilson'][0]:.3f}, "
                         f"{ext['ci95_wilson'][1]:.3f}], p={_p(ext['binomial_p_two_sided'])})") if ext else "–"
                L.append(f"| {_label(s, outputs)} | " + " | ".join(str(ev["hit_distribution"][k]) for k in ks)
                         + f" | {_p(ev['chi2_gof']['p_value'])} | {ext_s} |")
            L.append("")

            L += ["#### Razdoblje treniranja (samo opisno)", "", "| Strategija | n | Prosjek | 95% CI (t) | p (egzaktni, nekorigirano) |",
                  "|---|---|---|---|---|"]
            for s, e in rows:
                tr = e["training"]
                if tr is None:
                    L.append(f"| {_label(s, outputs)} | – | – | – | – (nema van-uzoračkih predviđanja) |")
                    continue
                L.append(f"| {_label(s, outputs)} | {tr['n_draws']} | {tr['mean_hits']:.3f} | "
                         f"[{tr['ci95_t'][0]:.3f}, {tr['ci95_t'][1]:.3f}] | {_p(tr['exact_test']['p_two_sided'])} |")
            L.append("")

        L += ["#### Težine ensemblea iz razdoblja treniranja (zamrznute prije evaluacije)", "",
              "| Metoda | z (treniranje) | težina |", "|---|---|---|"]
        for k, w in bt["train_weights"].items():
            L.append(f"| {_label(k, outputs)} | {w['z_train']:+.3f} | {w['weight']:.3f} |")
        L.append("")
        for pool_name, lr in bt["logistic_control"].items():
            L += ["#### Logistička kontrola: vremenska unakrsna validacija unutar treniranja", "",
                  "| Fold | kola (train/test) | log-loss | log-loss uz p=0.5 | AUC |", "|---|---|---|---|---|"]
            for i, f in enumerate(lr["time_series_cv"], 1):
                L.append(f"| {i} | {f['train_draws']}/{f['test_draws']} | {f['log_loss']:.4f} | "
                         f"{f['log_loss_p_0_5']:.4f} | {f['auc']:.3f} |")
            L.append("")

        q = bt["data_quality"]
        f, o, r = q["frequency"], q["consecutive_overlap"], q["longest_run"]
        L += ["#### Kvaliteta i slučajnost samih izvlačenja (opisno)", "",
              f"* Učestalost brojeva: statistika {f['statistic']:.2f}, simulirani p = {_p(f['p_value_simulated'])} "
              f"({f['simulations']} simulacija); raspon pojavljivanja {min(f['counts'].values())}–{max(f['counts'].values())}, "
              f"očekivano {f['expected_each']:.1f}.",
              f"* Preklapanje uzastopnih kola: prosjek {o['mean']:.3f} (očekivano {o['expected']:.3f}), "
              f"egzaktni p = {_p(o['exact_test']['p_two_sided'])}.",
              f"* Najdulji niz uzastopnih pojavljivanja: {r['max_observed']} (broj {', '.join(map(str, r['number_with_max']))}); "
              f"simulirani p (maksimum preko svih brojeva) = {_p(r['p_value_simulated_max_over_numbers'])}, "
              f"medijan u simulaciji {r['simulated_max_median']:.0f}.", ""]

    L += ["## Zaključak", ""]
    concl = []
    for game_key, bt in backtests.items():
        if bt.get("status") not in ("ran", "reused"):
            concl.append(f"* {game_key}: backtest nije bio moguć ({REASON_HR.get(bt.get('reason'), bt.get('reason'))}); o uspješnosti se ne može reći ništa.")
            continue
        for pool_name, pr in bt["results"]["pools"].items():
            holm_sig = [s for s, t in pr["family_tests"].items() if t["significant_holm"]]
            if holm_sig:
                concl.append(f"* {game_key}/{pool_name}: nakon Holmove korekcije značajno odstupanje od slučajnog imaju "
                             f"{', '.join(_label(s, outputs) for s in holm_sig)}. Smjer i veličina su u tablici; "
                             "prije bilo kakvog zaključka treba replikacija na novim kolima.")
            else:
                concl.append(f"* {game_key}/{pool_name}: nijedna metoda, ensemble ni statistička kontrola ne odstupa od "
                             "uniformne slučajne osnovice nakon korekcije za višestruko testiranje. Opaženo je "
                             "kompatibilno s time da su sve kombinacije jednako (ne)vjerojatne.")
    reused = {bt["reused_from"]["summary"] for bt in backtests.values() if bt.get("reused_from")}
    where_bt = ("; backtest ovog pokretanja je ponovno korišten iz " + ", ".join(f"`{p}`" for p in sorted(reused))
                if reused else ", `backtest/*/summary.json`, `backtest/*/predictions/*.json`")
    L += concl + ["", "Detalji, izvori i ograničenja: `README.md`. Svi međurezultati su u direktoriju s rezultatima "
                  f"ovog pokretanja (`{manifest['command_line'].get('results', 'results')}`): "
                  f"`final/combinations.json`, `final/intermediate/*.json`{where_bt}.", ""]
    return "\n".join(line for line in L if line is not None)
