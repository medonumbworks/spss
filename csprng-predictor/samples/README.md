# Sample files

These files can be imported with **Import CSV / XML** in the app (copy them to the phone,
e.g. into *Downloads*, then pick them in the file chooser). `draws.csv` holds 40 consecutive
daily draws from 2026-09-07 to 2026-10-16 in the app's own format — a `date,n1,...,n11` header
followed by one line per draw with an ISO date and 11 distinct numbers from 1..22 (the same
format *Export draws (CSV)* writes); any delimiter out of `,` `;` tab `|` or spaces works, and
dates may also be written `16.10.2026`, `2026/10/16`, `10/16/2026`, `20261016` and so on.
`draws-nodates.csv` holds 10 draws as bare semicolon-separated rows without dates, oldest
first: on import choose *Rows are oldest first* and the date of the last row, and the app
assigns one day per row going backwards. `draws.xml` holds the last 6 draws of `draws.csv`
(2026-10-11 … 2026-10-16) written in the three supported XML forms — numbers as `<n>` child
elements, a `numbers="…"` attribute, and `<date>`/`<numbers>` child elements — with ISO,
`dd.MM.yyyy` and `yyyy/MM/dd` dates; importing it after `draws.csv` shows the duplicate
detection (*Imported 0 new draws (6 duplicates skipped)*). The numbers were generated
deterministically with a seeded pseudo-random generator, so they contain no pattern the model
could exploit: expect backtest results close to the random baseline of 5.5 hits out of 11.
