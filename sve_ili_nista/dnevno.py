"""Dnevni postupak za Sve ili Ništa: jedna naredba od podataka do izvještaja i kombinacija za zadani dan.

Upotreba:
  python3 dnevno.py --datum 19.10.2026 --osnovni podaci/sve_ili_nista.csv --dopuna podaci/extra_draws.json \
                    --izlaz rezultati/ [--nedostaje 290:17.10.2026 ...] [--entropy HEX]

Koraci: spoji dopunjena kola -> analiza (ciljni dan u tjednu) -> provjera izvan uzorka -> svježa CSPRNG entropija
-> varijante -> potpuni popis pokrića -> prvih 100, zip, sha256 -> izvještaj.
"""
import argparse, json, os, secrets, subprocess, sys, zipfile, hashlib
from datetime import datetime
HERE = os.path.dirname(os.path.abspath(__file__))
NOM = ['ponedjeljak', 'utorak', 'srijeda', 'četvrtak', 'petak', 'subota', 'nedjelja']
ACC = ['ponedjeljak', 'utorak', 'srijedu', 'četvrtak', 'petak', 'subotu', 'nedjelju']
ap = argparse.ArgumentParser()
ap.add_argument('--datum', required=True); ap.add_argument('--osnovni', required=True); ap.add_argument('--dopuna', required=True)
ap.add_argument('--izlaz', required=True); ap.add_argument('--nedostaje', nargs='*', default=[]); ap.add_argument('--entropy')
ap.add_argument('--workflow', default=os.path.join(HERE, 'workflow.json'))
ap.add_argument('--prethodni-popis', help='CSV popisa pokrića za prethodno kolo (za pregled pogodaka)')
a = ap.parse_args()
d = datetime.strptime(a.datum, '%d.%m.%Y').date(); kolo = d.timetuple().tm_yday; dow = d.weekday()
out = os.path.abspath(a.izlaz); os.makedirs(out, exist_ok=True)
def run(*args, env=None):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env)
    if r.returncode: sys.exit(f'GREŠKA u {args[0]}:\n{r.stdout}\n{r.stderr}')
    return r.stdout
ext = os.path.join(out, 'sve_ili_nista_prosireno.csv')
print(run(os.path.join(HERE, 'extend_data.py'), a.osnovni, a.dopuna, ext).strip())
env = dict(os.environ, TARGET_DOW=str(dow))
an = os.path.join(out, f'analysis_{kolo}.json'); run(os.path.join(HERE, 'analyze.py'), ext, an, env=env)
oos = os.path.join(out, f'oos_{kolo}.json'); run(os.path.join(HERE, 'oos_new.py'), ext, oos)
ent = a.entropy or secrets.token_bytes(32).hex()
entf = os.path.join(out, f'entropy_{kolo}.txt'); open(entf, 'w').write(ent + '\n')
cmb = os.path.join(out, f'combos_{kolo}.json'); run(os.path.join(HERE, 'generate.py'), ext, '--entropy', ent, '--n-uniform', '0', '--out', cmb)
cov = os.path.join(out, f'kombinacije_kolo{kolo}_pokrice.csv'); run(os.path.join(HERE, 'coverage.py'), cov, '--entropy', ent)
with open(cov, encoding='utf-8-sig') as f, open(os.path.join(out, f'kombinacije_kolo{kolo}_prvih100.csv'), 'w', encoding='utf-8-sig') as g:
    for i, line in enumerate(f):
        if i > 100: break
        g.write(line)
with zipfile.ZipFile(cov[:-4] + '.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z: z.write(cov, os.path.basename(cov))
open(cov[:-4] + '.sha256', 'w').write(hashlib.sha256(open(cov, 'rb').read()).hexdigest() + '\n')
miss = [dict(kolo=int(m.split(':')[0]), date=m.split(':')[1]) for m in a.nedostaje]
cfg = dict(work=HERE, data=ext, kolo=kolo, target_date=a.datum, target_day_acc=ACC[dow], target_day_nom=NOM[dow], analysis=an, combos=cmb, oos=oos,
           extra=a.dopuna, entropy=entf, coverage_csv=cov, missing=miss)
if a.prethodni_popis: cfg['prev'] = dict(kolo=kolo - 1, coverage_csv=os.path.abspath(a.prethodni_popis))
cfgf = os.path.join(out, f'cfg_{kolo}.json'); json.dump(cfg, open(cfgf, 'w'), ensure_ascii=False, indent=1)
rep = os.path.join(out, f'izvjestaj_kolo{kolo}.md'); run(os.path.join(HERE, 'render_daily.py'), cfgf, rep)
print(f'kolo {kolo} ({NOM[dow]} {a.datum}) gotovo; entropija {ent}\n  {rep}\n  {cov[:-4]}.zip')
