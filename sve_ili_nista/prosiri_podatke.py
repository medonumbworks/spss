# -*- coding: utf-8 -*-
"""Dodaje kola 307–327/2026 (od korisnika: LotteryExtreme + listić) postojećim podacima i provjerava ih.
python3 -I prosiri_podatke.py <izvorni csv> <izlazni csv>"""
import sys
from datetime import date
import pandas as pd
SRC, OUT = sys.argv[1:3]
NEW = """307;03.11.2026;6 7 9 11 12 15 16 17 18 19 20
308;04.11.2026;2 3 4 5 10 12 13 14 15 16 17
309;05.11.2026;2 3 4 5 6 9 11 14 16 21 22
310;06.11.2026;2 3 4 6 9 11 12 13 14 18 20
311;07.11.2026;2 3 5 6 7 8 12 14 20 21 22
312;08.11.2026;7 8 9 10 11 12 14 15 17 18 22
313;09.11.2026;1 3 4 7 8 9 10 13 16 20 21
314;10.11.2026;2 3 4 6 7 14 15 17 18 20 21
315;11.11.2026;1 4 5 8 9 11 14 16 18 19 21
316;12.11.2026;1 7 9 11 14 15 16 17 18 19 21
317;13.11.2026;2 6 7 8 9 10 11 12 16 20 21
318;14.11.2026;1 2 3 4 5 6 10 15 17 19 20
319;15.11.2026;2 5 7 13 15 17 18 19 20 21 22
320;16.11.2026;2 3 4 6 9 11 15 16 17 18 21
321;17.11.2026;1 2 4 9 10 12 13 14 15 18 22
322;18.11.2026;4 5 7 8 11 13 14 15 16 19 21
323;19.11.2026;3 4 7 9 11 12 13 14 15 21 22
324;20.11.2026;1 3 9 11 13 14 16 18 19 20 22
325;21.11.2026;4 5 6 7 11 12 16 17 18 20 22
326;22.11.2026;3 4 5 9 12 13 14 15 16 18 22
327;23.11.2026;1 4 5 6 7 8 12 16 17 18 19"""
DAN = ['Pon', 'Uto', 'Sri', 'Čet', 'Pet', 'Sub', 'Ned']
df = pd.read_csv(SRC, sep=';', encoding='utf-8-sig')
B = [f'B{i}' for i in range(1, 12)]
rows, problems = [], []
for line in NEW.splitlines():
    k, dt, ns = line.split(';'); k = int(k); n = list(map(int, ns.split()))
    d, m, y = map(int, dt.split('.')); dd = date(y, m, d)
    if len(n) != 11 or len(set(n)) != 11 or not all(1 <= v <= 22 for v in n): problems.append((k, 'brojevi'))
    if n != sorted(n): problems.append((k, 'nije uzlazno'))
    if dd.timetuple().tm_yday != k: problems.append((k, 'kolo != redni dan'))
    rows.append(dict(Rb=None, Godina=y, Kolo=k, Datum=dt, Dan=DAN[dd.weekday()], **{b: v for b, v in zip(B, n)},
                     Zbroj=sum(n), Parni=sum(1 for v in n if v % 2 == 0)))
if df['Kolo'].iloc[-1] >= 307 or set(df['Datum']) & {r['Datum'] for r in rows}: problems.append(('preklapanje', ''))
out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
out['Rb'] = range(1, len(out) + 1)
out.to_csv(OUT, sep=';', index=False, encoding='utf-8')
print('nova kola:', len(rows), 'problemi:', problems or 'nema', '| ukupno redaka:', len(out), '| s brojevima:', int(out[B].notna().all(axis=1).sum()))
