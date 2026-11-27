"""§6 v2 – backtest konačnog seta (glavni od 2012., euro od 2022.), isplate (euro-jackpot.net), CSV evidencija."""
import json, os, re, sys, subprocess, datetime as dt
from collections import defaultdict, Counter
from zoneinfo import ZoneInfo
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
FAZA = sys.argv[1] if len(sys.argv) > 1 else 'konacni'
r = json.load(open(os.path.join(OUT, f'set_v2_{FAZA}.json')))
R = json.load(open(os.path.join(OUT, 'slucajni_elementi.json')))
SEED = R['numpy_seed']; N_KOMB = 8
SET = [(g, e) for g, e in r['set']]
RAZ = [(5, 2), (5, 1), (5, 0), (4, 2), (4, 1), (3, 2), (4, 0), (2, 2), (3, 1), (3, 0), (1, 2), (2, 1)]  # službeni redoslijed od 2022.
hist = [l.strip().split(';') for l in open(os.path.join(OUT, 'povijest_eurojackpot_2012-03-23_2026-11-24.csv')).readlines()[1:]]
hist = [(r_[0], set(map(int, r_[1:6])), set(map(int, r_[6:8])), r_[8]) for r_ in hist]

def bt(S):
    raz = Counter(); glavni = Counter(); dob_kola = []
    for d, G, E, f in hist:
        for c, (g, e) in enumerate(S):
            m = len(G & set(g)); glavni[m] += 1
            if f == '2/12':
                k = (m, len(E & set(e)))
                if k in RAZ: raz[k] += 1; dob_kola.append((d, c, k))
    return raz, glavni, dob_kola

raz, glavni, dob_kola = bt(SET)
rngs = np.random.default_rng(SEED + 12)
rand_sets = [[(sorted((rngs.choice(50, 5, replace=False) + 1).tolist()), sorted((rngs.choice(12, 2, replace=False) + 1).tolist()))
              for _ in range(N_KOMB)] for _ in range(1000)]
rr = [bt(s) for s in rand_sets]
r_dob = np.array([sum(x[0].values()) for x in rr]); r_g3 = np.array([sum(v for m, v in x[1].items() if m >= 3) for x in rr])
moj_dob = sum(raz.values()); moj_g3 = sum(v for m, v in glavni.items() if m >= 3)
pct = lambda a, x: float((a < x).mean() * 100 + (a == x).mean() * 50)

# ---------- isplate (euro-jackpot.net) ----------
PAY = '/tmp/dl/pay'; os.makedirs(PAY, exist_ok=True)
datumi = sorted({d for d, _, _ in dob_kola})
need = [d for d in datumi if not os.path.exists(f'{PAY}/{d}.html') or os.path.getsize(f'{PAY}/{d}.html') < 50000]
if need:
    url = lambda d: f"https://www.euro-jackpot.net/results/{d[8:10]}-{d[5:7]}-{d[:4]}"
    for i in range(0, len(need), 8):
        ps = [subprocess.Popen(['curl', '-s', '-m', '40', '-A', 'Mozilla/5.0', '-o', f'{PAY}/{d}.html', url(d)]) for d in need[i:i + 8]]
        [p.wait() for p in ps]
LAB = {('5', '2'): (5, 2), ('5', '1'): (5, 1), ('5', None): (5, 0), ('4', '2'): (4, 2), ('4', '1'): (4, 1), ('3', '2'): (3, 2),
       ('4', None): (4, 0), ('2', '2'): (2, 2), ('3', '1'): (3, 1), ('3', None): (3, 0), ('1', '2'): (1, 2), ('2', '1'): (2, 1)}
def isplate(d):
    try:
        h = open(f'{PAY}/{d}.html', errors='ignore').read()
    except FileNotFoundError:
        return None
    i = h.find('Prize Breakdown'); j = h.find('European Prize Breakdowns', i)
    t = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' | ', h[i:j if j > 0 else i + 8000]))
    out = {}
    for m in re.finditer(r'Match (\d)(?: and (\d) Euro Numbers?)? [\s|]*&euro;([\d,\.]+) [\s|]*([\d,]+)', t):
        out[LAB[(m.group(1), m.group(2))]] = float(m.group(3).replace(',', ''))
    return out if len(out) == 12 else None
pay = {d: isplate(d) for d in datumi}
nedostaje = [d for d, v in pay.items() if v is None]
povrat = None
if not nedostaje:
    ukupno = sum(pay[d][k] for d, _, k in dob_kola)
    ulog = 16.0 * sum(1 for x in hist if x[3] == '2/12')
    povrat = dict(isplate_eur=round(ukupno, 2), ulog_eur=ulog, povrat_po_euru=round(ukupno / ulog, 4),
                  po_razredu={f"{a}+{b}": round(sum(pay[d][k] for d, _, k in dob_kola if k == (a, b)), 2) for a, b in RAZ if raz[(a, b)]})

# ---------- CSV ----------
sada = dt.datetime.now(ZoneInfo('Europe/Zagreb')).strftime('%Y-%m-%d %H:%M')
vg = {int(k): v for k, v in r['glasovi_g'].items()}; ve = {int(k): v for k, v in r['glasovi_e'].items()}
hdr = 'ID;izvor;datum_i_sat_izrade;kolo;sustav;G1;G2;G3;G4;G5;E1;E2;punila;stabilnost;glasovi;seed;pogoci_glavni;pogoci_euro;razred;isplata'
rows = []
for c, ((g, e), src) in enumerate(zip(SET, r['izvori'])):
    s = r['sustavi'].get(src)
    def stab(x):
        if x in r['punila'][c]: return '*'
        if x in r['zamjenski'][c]: return '†'
        if src == 'Plečko': return '1'
        v = s['stab_g'].get(str(x))
        if v is None:  # broj sustava izvan njegovih prvih 6: stabilan ako je u popisu u svih 5 trenutaka
            pr = r['promjene_u_prozoru'].get(src)
            v = 1.0 if pr is None or all(x in p[0] for p in pr.values()) else 0.5
        return str(v).replace('.', ',')
    pun = ','.join([f"{x}*" for x in r['punila'][c]] + [f"{x}†" for x in r['zamjenski'][c]])
    gl = 'G:' + '/'.join(str(vg.get(x, 0)).replace('.', ',') for x in g) + ' E:' + '/'.join(str(ve.get(x, 0)).replace('.', ',') for x in e)
    rows.append(';'.join([f"EJ-2026-11-27-K95-v2-{c + 1:02d}", 'Claude Code', sada, '95', src] + list(map(str, g)) + list(map(str, e)) +
                         [pun, '/'.join(stab(x) for x in g), gl, str(SEED), '', '', '', '']))
if FAZA == 'konacni':
    open(os.path.join(OUT, 'eurojackpot_2026-11-27_kolo95_v2.csv'), 'w').write(hdr + '\n' + '\n'.join(rows) + '\n')
    ev = os.path.join(OUT, 'evidencija_kombinacija.csv')
    stari = open(ev).read().splitlines()
    if stari[0] != hdr:  # proširi zaglavlje: novi stupci stabilnost;glasovi (stari redovi dobivaju prazna polja)
        novi = [hdr]
        for l in stari[1:]:
            p = l.split(';'); novi.append(';'.join(p[:13] + ['', ''] + p[13:]))
        stari = novi
    ids = {l.split(';')[0] for l in stari}
    stari += [l for l in rows if l.split(';')[0] not in ids]
    open(ev, 'w').write('\n'.join(stari) + '\n')

res = dict(razredi={f"{a}+{b}": raz[(a, b)] for a, b in RAZ if raz[(a, b)]}, dobitnih_listica=moj_dob, kola_s_dobitkom=len(datumi),
           ocekivano=sum(1 for x in hist if x[3] == '2/12') * N_KOMB * 0.031416252902641,
           glavni_pogoci_2012={str(m): glavni[m] for m in sorted(glavni)}, listica_3plus_glavna=moj_g3,
           percentil_dobitni=pct(r_dob, moj_dob), percentil_3plus=pct(r_g3, moj_g3),
           slucajni_dobitni=[int(r_dob.min()), float(r_dob.mean()), int(r_dob.max())], slucajni_3plus=[int(r_g3.min()), float(r_g3.mean()), int(r_g3.max())],
           isplate_nedostaju=nedostaje, povrat=povrat, csv_redovi=rows)
json.dump(res, open(os.path.join(OUT, f'backtest_v2_{FAZA}.json'), 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k != 'csv_redovi'}, ensure_ascii=False, indent=1, default=str))
print('\n'.join(rows))
