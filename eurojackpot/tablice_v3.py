"""v3 – konačne tablice A i B (markdown) iz JSON rezultata."""
import json, os
OUT = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(OUT, 'v3_privremeni.json')))
OA = json.load(open(os.path.join(OUT, 'v3_optimizacija_A.json'))); OB = json.load(open(os.path.join(OUT, 'v3_optimizacija_B.json')))
IZ = json.load(open(os.path.join(OUT, 'v3_izvjestaj.json')))
v2src = [l.split(';')[4] for l in open(os.path.join(OUT, 'eurojackpot_2026-11-27_kolo95_v2.csv')).read().splitlines()[1:]]
pp = lambda x: f"{x * 100:.3f} %".replace('.', ',')
def tablica(naslov, S, izvori, ozn, upl, toc, ulog):
    L = [f"### {naslov}", "| # | Status | Sustav | 5 glavnih († = broj iz drugog sustava) | 2 euro |", "|---|---|---|---|---|"]
    i = 0
    for c, (g, e) in enumerate(upl):
        i += 1; L.append(f"| {i} | uplaćeno | {v2src[c]} (v2-{c + 1:02d}) | {' '.join(map(str, g))} | {e[0]} {e[1]} |")
    for (g, e), src, (tg, te) in zip(S, izvori, ozn):
        i += 1
        L.append(f"| {i} | NOVO | {src} | {' '.join(str(x) + ('†' if x in tg else '') for x in g)} | {' '.join(str(x) + ('†' if x in te else '') for x in e)} |")
    L += ["", f"- Ukupni ulog: {ulog} €" + (" (8 uplaćenih = 16 € + 2 nove = 4 €)" if upl else ""),
          f"- P(5+2, 5+1, 5+0 ili 4+2) = {toc['P_top']:.4e}".replace('.', ',') + f" (≈ 1 : {1 / toc['P_top']:,.0f})".replace(',', '.'),
          f"- P(bilo koji dobitak) = {pp(toc['P_bilo_koji'])}",
          f"- P(isplata ≥ 26 €) = {pp(toc['P_ge_26_med'])} (medijan isplata; 25. perc. {pp(toc['P_ge_26_p25'])}, 75. perc. {pp(toc['P_ge_26_p75'])})", ""]
    return L
L = tablica("TABLICA A – 13 kombinacija ispočetka", OA['konacni'], [o['sustav'] for o in P['oznake_A']], IZ['oznake_A'], [], OA['tocno']['konacni'], 26)
L += tablica("TABLICA B – 8 uplaćenih (v2) + 2 NOVE (3 dodatne nisu poznate)", OB['konacni'], P['izvori_B'], IZ['oznake_B'], P['uplaceni'], OB['tocno']['konacni'], 20)
open(os.path.join(OUT, 'v3_konacne_tablice.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
