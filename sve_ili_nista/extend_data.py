"""Dodaje provjerena kola (JSON lista {date,kolo,numbers}) na kraj izvornog CSV-a i piše prošireni CSV."""
import sys, json, csv, io
from datetime import datetime
src, extra_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(src, 'rb').read().decode('utf-8-sig')
rows = list(csv.reader(io.StringIO(txt), delimiter=';'))
hdr, body = rows[0], rows[1:]
last_rb = int(body[-1][0])
days = ['Pon', 'Uto', 'Sri', 'Čet', 'Pet', 'Sub', 'Ned']
extra = sorted(json.load(open(extra_json)), key=lambda e: e['kolo'])
seen = {(r[1], r[2]) for r in body}
added = 0
for e in extra:
    nums = sorted(int(x) for x in e['numbers'])
    assert len(nums) == 11 and len(set(nums)) == 11 and all(1 <= n <= 22 for n in nums), e
    d = datetime.strptime(e['date'], '%d.%m.%Y').date()
    assert d.timetuple().tm_yday == int(e['kolo']), ('kolo/date mismatch', e)
    if (str(d.year), str(e['kolo'])) in seen: continue
    last_rb += 1; added += 1
    body.append([str(last_rb), str(d.year), str(e['kolo']), d.strftime('%d.%m.%Y'), days[d.weekday()]] + [str(n) for n in nums] + [str(sum(nums)), str(sum(1 for n in nums if n % 2 == 0))])
with open(out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f, delimiter=';'); w.writerow(hdr); w.writerows(body)
print(f'added {added} draws -> {out} ({len(body)} rows)')
