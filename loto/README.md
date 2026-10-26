# Loto 6/45 — 231 kombinacija za 29.10.2026.

Sve je reproducibilno iz ovog direktorija (Python 3.13, numpy, scipy, ephem, lunardate).

| datoteka | što radi |
|---|---|
| `common.py` | učitavanje izvlačenja i registra odigranih, pakiranje (nijedna trojka dvaput), egzaktno nabrajanje svih 8.145.060 izvlačenja |
| `stats.py` | statistika 882 izvlačenja → `out/stats.json` (frekvencije, vrući/hladni/kašnjenje, parovi, profil, backtest prediktivne moći) |
| `stats_neovisno.py` | neovisna statistička analiza (drugi autor/agent) → `out/stats_neovisno.md` |
| `ezoterija/*.py` | 10 ezoteričnih metoda, svaka samostalan modul s cijelim izvodom za 29.10.2026. 20:00 CET → `out/ezo_<metoda>.json` |
| `generate.py` | gradi 231 kombinaciju (blok 3 → 2 → 4 → 1) → `out/set231.json` |
| `verify.py` | neovisna provjera skupa: presjeci, odigrane, frekvencije, egzaktne vjerojatnosti → `out/verify.json` |
| `make_report.py` | sastavlja završni dokument `loto_231_29102026.md` |
| `data/` | priloženi ulazi: izvlačenja 2018–2026 (CSV), registar odigranih 16.08.2026., sistemi, prethodni set od 231, originalna ezoterijska skripta |

Redoslijed: `python3 stats.py && for m in ezoterija/*.py; do python3 $m; done && python3 generate.py && python3 verify.py && python3 make_report.py`
