# -*- coding: utf-8 -*-
"""Eurojackpot povijest 5/50 + 2/12 (od 25.03.2022.): euro-jackpot.net (godišnje arhive) + LotteryExtreme (mjesečne arhive), usporedba.
python3 -I ej_podaci.py <dir s ejn_YYYY.html> <dir s le_YYYY-MM.html> <izlazni csv>"""
import re, sys, glob, json
from datetime import date
EJN, LE, OUT = sys.argv[1:4]
START = date(2022, 3, 25)
a = {}
for f in sorted(glob.glob(f'{EJN}/ejn_*.html')):
    s = open(f, encoding='utf-8').read()
    for d, body in re.findall(r'href="/results/(\d\d-\d\d-\d{4})".*?<ul class="balls small">(.*?)</ul>', s, re.S):
        dd, mm, yy = map(int, d.split('-')); dt = date(yy, mm, dd)
        mains = list(map(int, re.findall(r'class="ball"><span>(\d+)</span>', body)))
        euros = list(map(int, re.findall(r'class="euro"><span>(\d+)</span>', body)))
        if dt >= START: a[dt] = (mains, euros)
b, bn = {}, {}
for f in sorted(glob.glob(f'{LE}/le_*.html')):
    s = open(f, encoding='utf-8', errors='replace').read()
    t = re.sub(r'<script.*?</script>', '', s, flags=re.S); t = re.sub(r'<[^>]+>', ' ', t); t = re.sub(r'\s+', ' ', t)
    for dd, mm, yy, no, nums in re.findall(r'\((\d\d)/(\d\d)/(\d{4})\) \((\d+)\)\s*(?:&nbsp;)?\s*Video and Winners ((?:\d+ ){7})', t):
        dt = date(int(yy), int(mm), int(dd)); n = list(map(int, nums.split()))
        if dt >= START: b[dt] = (n[:5], n[5:]); bn[dt] = int(no)
prob = []
for dt in sorted(set(a) | set(b)):
    if dt not in a or dt not in b: prob.append((str(dt), 'samo u jednom izvoru')); continue
    if sorted(a[dt][0]) != sorted(b[dt][0]) or sorted(a[dt][1]) != sorted(b[dt][1]): prob.append((str(dt), a[dt], b[dt]))
rows = []
for dt in sorted(b):
    m, e = sorted(b[dt][0]), sorted(b[dt][1])
    ok = len(set(m)) == 5 and len(set(e)) == 2 and all(1 <= x <= 50 for x in m) and all(1 <= x <= 12 for x in e)
    if not ok: prob.append((str(dt), 'raspon/duplikat', m, e))
    rows.append((bn[dt], dt.strftime('%d.%m.%Y'), dt.strftime('%a'), *m, *e))
nos = [r[0] for r in rows]
gaps = [(x, y) for x, y in zip(nos, nos[1:]) if y != x + 1]
dup_dates = len(rows) - len({r[1] for r in rows})
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('Kolo_LE;Datum;Dan;M1;M2;M3;M4;M5;E1;E2\n')
    for r in rows: fh.write(';'.join(map(str, r)) + '\n')
print(json.dumps(dict(euro_jackpot_net=len(a), lotteryextreme=len(b), razlike=prob, praznine_u_numeraciji=gaps, dupli_datumi=dup_dates,
                      prvo=rows[0][:2], zadnje=rows[-1][:2], dani=sorted({r[2] for r in rows})), ensure_ascii=False, default=str))
