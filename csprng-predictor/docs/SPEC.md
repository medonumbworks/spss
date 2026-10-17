# CSPRNG Predictor — Build Specification

An Android app (brand new, unrelated to the Python code elsewhere in this repo) for a
daily "pick 11 numbers out of 1..22" draw. The user feeds it past draws (CSV / XML import,
or manual entry), it predicts the 11 numbers for the next draw, and when the real draw is
entered it scores itself and adapts its prediction model for the following day.

Honesty note that must appear in the app (Stats screen + About text) and the README:
a true CSPRNG has no exploitable pattern, so over many draws the model's average hit
count is expected to stay near the random baseline of 5.5 / 11. The app measures and
displays this so the user can see for themselves whether the source behaves randomly.

---------------------------------------------------------------------------------------

## 0. Hard constraints (the build toolchain enforces these)

* **Language: Java only.** Source level is compiled with `javac --release 8`.
* **NO lambdas, NO method references, NO default/static interface methods, NO
  `java.util.stream`.** The dexer is the legacy `dx` (no `invokedynamic` support at
  minSdk 24). Use anonymous inner classes and plain loops. A single lambda breaks the build.
* **No third-party or AndroidX libraries.** Only `android.*`, `java.*`, `javax.xml.*`,
  `org.w3c.dom.*`, `org.xml.sax.*`. **Do not use `org.json`** (the core must also run on a
  plain JVM in unit tests). Persist data as plain text (CSV / `key=value`).
* `minSdkVersion 24`, `targetSdkVersion 34`, compiled against an API 34 framework jar.
  Do not call APIs newer than 24 without a `Build.VERSION.SDK_INT` guard.
* `java.time` is **not** available (API 26+). Use `java.util.Calendar` / `GregorianCalendar`
  / `SimpleDateFormat` or manual arithmetic. All dates in the app are ISO strings
  `yyyy-MM-dd`.
* Package: `com.medonumbworks.csprngpredictor`. Core logic in sub-package `.core`
  (pure Java, **no `android.*` imports anywhere in `core`**), UI in `.ui`, app glue in the
  root package.
* Everything must compile without warnings-as-errors concerns; avoid raw types where easy.
* Keep each source file self-contained and readable; comments where logic is non-obvious.

Project layout (relative to repo root `csprng-predictor/`):

```
csprng-predictor/
  app/src/main/AndroidManifest.xml
  app/src/main/java/com/medonumbworks/csprngpredictor/        (App glue: Repository, Prefs)
  app/src/main/java/com/medonumbworks/csprngpredictor/core/   (pure Java engine)
  app/src/main/java/com/medonumbworks/csprngpredictor/ui/     (Activities, views)
  app/src/main/res/...                                         (layouts, values, mipmap)
  app/src/test/java/com/medonumbworks/csprngpredictor/core/   (JUnit 4 tests for core)
  app/build.gradle, build.gradle, settings.gradle, gradle.properties (Android Studio path)
  scripts/build-apk.sh      (SDK-less build used in this environment -> dist/*.apk)
  scripts/run-tests.sh      (compiles core + tests with JUnit 4 and runs them on the JVM)
  keystore/debug.keystore   (alias androiddebugkey, store/key password "android")
  samples/draws.csv, samples/draws.xml, samples/draws-nodates.csv
  dist/                     (built APK is committed here)
  docs/SPEC.md (this file), README.md
```

---------------------------------------------------------------------------------------

## 1. Core engine — `com.medonumbworks.csprngpredictor.core`

All classes `public final` unless stated. Exact public signatures below are the contract
between the core, the UI and the tests. You may add private helpers freely and additional
public methods if clearly useful, but never remove/rename the ones listed.

### 1.1 `Game`
```java
public final class Game {
  public static final int MAX_NUMBER = 22;      // numbers are 1..22
  public static final int PICK = 11;            // 11 numbers per draw
  public static final double EXPECTED_HITS = 5.5; // 11 * 11 / 22 for a random draw
  public static boolean isValidNumber(int n);  // 1..22
  public static boolean isValidDraw(int[] nums); // length 11, all valid, distinct
}
```

### 1.2 `Draw`
```java
public final class Draw implements Comparable<Draw> {
  public final String date;     // ISO yyyy-MM-dd, never null
  public final int[] numbers;   // defensive copy, sorted ascending, distinct, 1..22, length 11
  public Draw(String isoDate, int[] numbers);          // throws IllegalArgumentException if invalid
  public boolean contains(int n);
  public String numbersString(String separator);        // "1 4 7 ..." sorted
  public int hitsAgainst(int[] predicted);              // |numbers ∩ predicted|
  @Override equals/hashCode on (date, numbers); compareTo: by date, then numbers lexicographically.
  @Override toString -> date + ": " + numbersString(" ")
}
```

### 1.3 `DateUtil`  (static only)
```java
public static String today();                              // local date, ISO
public static boolean isIso(String s);                    // strict yyyy-MM-dd with a real calendar date
public static String addDays(String iso, int days);       // ISO arithmetic via GregorianCalendar
public static int daysBetween(String isoA, String isoB);  // b - a in days
public static int compare(String isoA, String isoB);      // lexicographic is fine for ISO
public static String normalize(String token, boolean dayFirst); // see rules; returns ISO or null
public static String findDate(String text, boolean dayFirst);   // first date-like substring in text, normalized; null if none
public static String display(String iso);                 // "dd.MM.yyyy" for the UI
```
`normalize` accepts (trimmed, optional surrounding quotes): `yyyy-MM-dd`, `yyyy/MM/dd`,
`yyyy.MM.dd`, `yyyyMMdd` (8 digits), `dd.MM.yyyy`, `dd-MM-yyyy`, `dd/MM/yyyy`,
`MM/dd/yyyy`, `MM-dd-yyyy`, 2-digit years in the day-first/month-first forms (`dd.MM.yy` ->
20yy), and ISO date-times (`2026-10-16T20:00:00`, `2026-10-16 20:00`) by taking the date part.
Ambiguity rule for `a/b/yyyy`, `a-b-yyyy`, `a.b.yyyy`: if a > 12 -> day-first; else if b > 12
-> month-first; else use the `dayFirst` flag. Month/day must form a real calendar date,
otherwise null. Years must be 1900..2100.

### 1.4 Parsing — `ParseResult`, `DrawParser`
```java
public final class ParseResult {
  public final List<Draw> dated = new ArrayList<Draw>();      // rows with a recognised date
  public final List<int[]> undated = new ArrayList<int[]>();  // rows with 11 numbers but no date (file order)
  public final List<String> warnings = new ArrayList<String>(); // human readable, max ~20 kept + "... and N more"
  public int skippedLines;                                     // lines/elements that had no usable draw
  public int totalLines;
  public boolean isEmpty();                                     // no dated and no undated
}

public final class DrawParser {
  public static ParseResult parse(String content, String fileNameHint, boolean dayFirst);
  public static ParseResult parseCsv(String content, boolean dayFirst);   // CSV / TSV / TXT
  public static ParseResult parseXml(String content, boolean dayFirst);
  public static int[] extractNumbers(String text);   // all ints found in text that are 1..22, in order, duplicates kept
  public static int[] asDraw(int[] candidates);      // returns sorted distinct copy if exactly 11 distinct valid numbers, else null
  public static List<Draw> assignDates(List<int[]> undated, String endDateIso, boolean oldestFirst);
}
```
Rules:
* `parse` picks XML when the file name ends with `.xml` (case-insensitive) **or** the trimmed
  content starts with `<`. Otherwise CSV.
* **CSV/TXT**: split into lines (`\r?\n`). Delimiter per line: try `,` `;` `\t` `|`; if none,
  whitespace. Header lines (no 11 valid numbers) are skipped silently (counted in
  `skippedLines`). Per line:
  1. Find the date: the first cell whose `DateUtil.normalize` succeeds; if none, try
     `DateUtil.findDate` on the whole line (handles `"Draw 2026-10-16: 1 4 7 ..."`).
  2. Numbers: from all cells **except the date cell**, run `extractNumbers`. A cell may
     itself hold several numbers separated by spaces, `-`, `/`, `+` (e.g. `"1 4 7 9 11 ..."`).
     Pure-digit cells outside 1..22 (ids, years) are ignored. If a cell is a date-like token
     its digits are not numbers (never treat `16` from `2026-10-16` as a draw number — the
     date cell is removed before number extraction; if the date came from `findDate`,
     remove that substring first).
  3. If the candidate list has exactly 11 distinct valid numbers -> draw. If it has more
     than 11 candidates, search for a contiguous run of exactly 11 cells that each parse as
     a single valid number (prefer the run immediately after the date cell, else the first
     such run); if found use it, else skip the line with a warning
     (`"line 12: 13 numbers found, expected 11"`). Fewer than 11 -> skip with warning.
  4. Dated -> `dated`; no date -> `undated` (keep file order).
* **XML**: parse with `javax.xml.parsers.DocumentBuilderFactory` (namespace-aware off,
  external entities / DTD loading disabled where the feature exists; wrap in try/catch).
  Walk all elements depth-first. An element is a *draw element* if
  `asDraw(extractNumbers(ownText(e)))` succeeds, where `ownText(e)` = its attribute values
  (excluding attributes that look like dates) + direct text nodes + the text of child
  elements that have no element children — **and** none of its descendant elements is
  itself a draw element (take the deepest). Date for a draw element: first match of
  (a) an attribute named like `date|datum|day|drawdate|draw_date|time|datetime` on it,
  (b) any attribute value on it that normalizes as a date, (c) a child element named like
  those names, (d) `DateUtil.findDate` over its own text, (e) the nearest ancestor's
  date-like attribute or child. None -> undated.
  Supported canonical forms (also used by the sample file):
  ```xml
  <draws>
    <draw date="2026-10-16"><n>1</n><n>4</n>...</draw>
    <draw date="2026-10-15" numbers="2 5 6 8 9 12 13 17 19 20 22"/>
    <draw><date>15.10.2026</date><numbers>2,5,6,8,9,12,13,17,19,20,22</numbers></draw>
  </draws>
  ```
* `assignDates`: undated rows are consecutive daily draws; `oldestFirst=true` means the
  first row is the oldest; the last chronological row gets `endDateIso`, earlier ones
  `endDate - k` days.
* Content may be up to ~20 MB; parsing must be O(n) and never throw on garbage input
  (return warnings instead).

### 1.5 Storage formats — `DrawCsv`, `DrawSet`
```java
public final class DrawCsv {
  public static final String HEADER = "date,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11";
  public static String write(List<Draw> draws);           // header + one line per draw, "\n" endings
  public static List<Draw> read(String csv);               // strict reader for the app's own file; tolerant of header/blank lines
}

public final class DrawSet {                                // sorted (date asc, then insertion), deduplicated
  public DrawSet(); public DrawSet(Collection<Draw> initial);
  public boolean add(Draw d);                              // false if an equal draw exists
  public int addAll(Collection<Draw> ds);                  // number actually added
  public boolean remove(Draw d);
  public void clear();
  public int size(); public boolean isEmpty();
  public List<Draw> all();                                 // unmodifiable, chronological (oldest first)
  public List<Draw> newestFirst();
  public Draw latest();                                    // null if empty
  public List<Draw> onDate(String iso);
  public Draw firstAfter(String iso);                      // earliest draw with date > iso, or null
}
```
Chronological order within the same date = insertion order (stable).

### 1.6 Strategies ("experts")
```java
public interface Strategy {
  String id();          // stable, used in persisted weights: "uniform", "freq", "recent10", "recent30", "decay", "gap", "transition", "repeat", "pairs"
  String name();        // UI label
  String description(); // one sentence for the UI
  double[] scores(History h); // length 22, index i -> number i+1; non-negative, not all zero
}

public final class History {                // immutable read-only view of draws[0..t)
  public History(List<Draw> drawsChronological);  // may be empty
  public int size(); public Draw get(int i); public Draw last();   // last() null when empty
  public int[] counts();                       // all-time counts per number (length 22)
  public int[] countsInLast(int window);
  public double[] decayedCounts(double halfLifeDraws); // sum of 0.5^(age/halfLife), age 0 = last draw
  public int[] drawsSinceSeen();               // for each number: draws since it last appeared (0 = in last draw); size() if never seen
  public int[][] transitionCounts();           // T[i][j] = times number j+1 appeared in the draw following a draw containing i+1
  public int[] containsCounts();               // C[i] = times number i+1 appeared in any draw that has a successor
  public double repeatRate();                  // mean over t>=1 of |draw[t] ∩ draw[t-1]| / 11 ; 0.5 when fewer than 2 draws
  public int[][] pairCounts();                 // P[i][j] co-occurrence counts (symmetric)
}
```
All History statistics are computed lazily and cached; `History` objects are created
per replay step, so make the expensive ones (transition, pair) incremental: implement a
package-private mutable `RunningStats` that `Replay` updates draw by draw and that
`History` wraps, so that a full replay over N draws is O(N * 22 * 11) — not O(N^2).

Default strategies (`Strategies.defaults(ModelParams p)` returns them in this order):
1. `uniform`   — all 1.0. (Control expert; if it ends up with the highest weight the data is random.)
2. `freq`      — `(count[i] + 1)` Laplace-smoothed all-time frequency.
3. `recent10`  — `countsInLast(p.shortWindow) + 0.5`.
4. `recent30`  — `countsInLast(p.longWindow) + 0.5`.
5. `decay`     — `decayedCounts(p.halfLife) + 0.5`.
6. `gap`       — overdue numbers: `drawsSinceSeen[i] + 1` (never seen -> size()+1).
7. `transition`— Markov from the last draw: for each j, average over i in last draw of
                 `(T[i][j] + 1) / (C[i] + 2)`; uniform 1.0 when no last draw.
8. `repeat`    — numbers in the last draw get `repeatRate()`, the others get `1 - repeatRate()`;
                 uniform when no last draw.
9. `pairs`     — co-occurrence affinity with the last draw: for each j, sum over i in last
                 draw of `(P[i][j] + 1) / (counts[i] + 2)`; uniform when no last draw.

### 1.7 Model parameters — `ModelParams`
```java
public final class ModelParams {
  public double eta = 0.5;        // Hedge learning rate (0..5)
  public double share = 0.02;     // fixed-share mixing toward uniform weights (0..0.5)
  public double biasRate = 0.1;   // per-number correction rate (0..1)
  public double biasDecay = 0.05; // per-number correction decay (0..1)
  public int shortWindow = 10;    // recent10 window (2..200)
  public int longWindow = 30;     // recent30 window (2..1000)
  public double halfLife = 15;    // decay half-life in draws (1..500)
  public boolean dayFirst = true; // ambiguous date parsing
  public ModelParams copy();
  public String serialize();                 // "key=value" lines, Locale.US number format
  public static ModelParams parse(String s); // missing/invalid keys keep defaults; never throws
  public void clamp();                       // enforce the ranges above
  @Override equals/hashCode/toString.
}
```

### 1.8 Prediction objects
```java
public final class Prediction {
  public final String basedOnDate;   // date of the latest draw used ("" if none)
  public final int basedOnCount;     // number of draws used
  public final String targetDate;    // basedOnDate + 1 day, or today if no draws
  public final int[] numbers;        // the 11 picks, sorted ascending
  public final double[] scores;      // length 22, final combined score per number, normalised to sum = 11 (i.e. an expected count; 0.5 each when uniform)
  public final String[] expertIds;   // in strategy order
  public final double[] expertWeights; // normalised (sum 1), same order
  public final int[][] expertPicks;  // each expert's own top-11 (for the UI and for scoring experts)
  public Prediction(...all fields...);
  public double scoreOf(int number);
  public int rankOf(int number);       // 1 = highest score
  public String numbersString(String sep);
}

public final class PredictionRecord {   // one replay step
  public final Prediction prediction;
  public final Draw actual;             // never null in replay records
  public final int hits;                // actual.hitsAgainst(prediction.numbers)
  public final int[] expertHits;        // per expert
}
```

### 1.9 The learner — `Model`
```java
public final class Model {
  public Model(ModelParams params, List<Strategy> strategies);
  public Prediction predict(History h);                  // pure, does not mutate
  public void observe(Prediction p, Draw actual);        // Hedge + bias update, bookkeeping
  public double[] weights();  public double[] biases();  // copies
  public int observed();                                 // number of observe() calls
  public int[] expertHitsTotal();                        // cumulative hits per expert
  public double[] expertMeanHits();                      // per expert, 0 if observed()==0
  public List<Strategy> strategies();
  public ModelParams params();
}
```
`predict`:
1. For each strategy k compute `s_k = scores(h)`, clamp negatives to 0, add 1e-9, and
   normalise to `p_k` with sum = 11 (`p_k[i] = 11 * s_k[i] / sum`).
2. Combined `S[i] = sum_k w_k * p_k[i]` with weights normalised to sum 1.
3. Apply bias: `S'[i] = S[i] * exp(b[i])`, then renormalise to sum 11.
4. Pick the top 11 by `S'` descending; ties broken by the lower number first. Store
   `scores = S'`, the per-expert top-11 (`expertPicks`, same tie rule).
`observe(p, actual)`:
1. For each expert, `hits_k = actual.hitsAgainst(p.expertPicks[k])`; cumulative totals.
2. Loss `l_k = 1 - hits_k / 11`. Hedge: `w_k *= exp(-eta * l_k)`; normalise; fixed share:
   `w_k = (1 - share) * w_k + share / K`.
3. Per-number bias: for each i, `target = actual.contains(i+1) ? 1 : 0`,
   `expected = p.scores[i] / 11` (a probability), `b[i] = (1 - biasDecay) * b[i] + biasRate * (target - expected)`,
   clamp to [-1, 1].
4. `observed++`.
Weights start at `1/K` and biases at 0. All arithmetic must be deterministic (no
randomness anywhere in the core).

### 1.10 Walk-forward replay — `Replay`, `ReplayResult`, `Stats`
```java
public final class ReplayResult {
  public final Model model;                      // state after observing all draws
  public final List<PredictionRecord> records;   // one per draw index t >= 1 (prediction made from draws[0..t), scored on draws[t])
  public final Prediction next;                  // prediction for the next (unseen) draw
  public final Stats stats;                      // over `records`
  public final ModelParams params;
}

public final class Replay {
  public static ReplayResult run(List<Draw> chronological, ModelParams params);
  public static ReplayResult run(List<Draw> chronological, ModelParams params, int evaluateLastN); // stats only over the last N records
}

public final class Stats {
  public final int n;                // evaluated predictions
  public final double meanHits;      // 0 when n == 0
  public final double sd;            // sample SD of hits (0 when n < 2)
  public final double edge;          // meanHits - 5.5
  public final double z;             // (meanHits - 5.5) / (1.2 / sqrt(n)); 0 when n == 0   (1.2 ≈ sqrt(hypergeometric variance 1.4405))
  public final int[] histogram;      // length 12, index = hits
  public final int best, worst;      // -1 when n == 0
  public final double lastTenMean;   // mean over the last min(10,n) records
  public static Stats of(List<PredictionRecord> records);
  public static double hypergeometricPmf(int k);     // P(hits == k) for a random 11-of-22 pick, k in 0..11
  public static double hypergeometricTail(int k);    // P(hits >= k)
  public static double expectedHits();               // 5.5
  public static double variance();                   // 11 * 0.5 * 0.5 * 11/21
  public String summaryText();                        // multi-line human readable summary used by the UI
}
```
Replay: `RunningStats` starts empty; for `t = 0..N-1`: if `t >= 1` (i.e. there is history)
`p = model.predict(historyView)`, `rec = (p, draws[t])`, `model.observe(p, draws[t])`;
then push `draws[t]` into the running stats. After the loop, `next = model.predict(full history)`.
`next.targetDate` = latest date + 1 day (or `DateUtil.today()` when no draws).

### 1.11 `AutoTune`
```java
public final class AutoTune {
  public static final class Candidate { public final ModelParams params; public final Stats stats; }
  public static Candidate tune(List<Draw> chronological, ModelParams base, int evaluateLastN);
  // grid: eta in {0.1, 0.3, 1.0} x halfLife in {5, 15, 40} x biasRate in {0, 0.05, 0.2}
  // (keep other fields from base). Pick the highest meanHits over the last evaluateLastN
  // records (ties -> fewer changes from base, then first in grid order). If fewer than 15
  // draws, return base unchanged with its stats.
}
```

### 1.12 Issued predictions — `IssuedPrediction`, `IssuedLog`
The replay gives *backtest* accuracy. Separately the app logs the predictions it actually
showed to the user so they can be checked against the real draw later.
```java
public final class IssuedPrediction {
  public final String issuedAt;      // ISO date the prediction was shown
  public final String basedOnDate;   // latest draw date it was based on ("" if none)
  public final int basedOnCount;
  public final String targetDate;
  public final int[] numbers;        // 11
  public String actualDate;          // null while pending
  public int[] actual;               // null while pending
  public int hits;                   // -1 while pending
  public boolean isPending();
}

public final class IssuedLog {
  public List<IssuedPrediction> all();                 // chronological by issue order
  public IssuedPrediction pendingFor(String basedOnDate, int basedOnCount); // existing pending entry for the same basis, or null
  public IssuedPrediction add(IssuedPrediction p);     // appends (no dedupe)
  public int settle(DrawSet draws);                    // for each pending entry: actual = draws.firstAfter(basedOnDate) (or, when basedOnDate is "", the earliest draw); if found fill actualDate/actual/hits. Returns number settled.
  public IssuedPrediction lastSettled();               // most recently settled entry or null
  public String toCsv();                                // header: issuedAt,basedOnDate,basedOnCount,targetDate,numbers,actualDate,actual,hits  (numbers space-separated inside one cell; empty cells when pending)
  public static IssuedLog fromCsv(String csv);          // tolerant; never throws
  public Stats stats();                                 // over settled entries (build PredictionRecord-like hit list; implement a small overload Stats.ofHits(int[] hits))
  public void clear();
}
```
Also add `public static Stats ofHits(List<Integer> hits)` to `Stats`.

---------------------------------------------------------------------------------------

## 2. Android app — glue (`com.medonumbworks.csprngpredictor`)

### 2.1 `Prefs`  (SharedPreferences wrapper, file "settings")
* `ModelParams loadParams()`, `void saveParams(ModelParams)` (stored as the serialized string
  under key `params`), `boolean dayFirst()` mirrors `params.dayFirst`.
* `int evaluateLastN()` default 200 (used for stats/auto-tune), setter.

### 2.2 `Repository`  (singleton `Repository.get(Context)`; uses `context.getApplicationContext()`)
Files in `context.getFilesDir()`: `draws.csv` (DrawCsv), `issued.csv` (IssuedLog), params
via Prefs. All methods synchronous; UI calls them from a background `Thread` when they may
be slow (import, replay with many draws, auto-tune) and posts results with
`new Handler(Looper.getMainLooper())` / `runOnUiThread`.
```java
DrawSet draws();                       // live set (reload from disk on first access)
IssuedLog issued();
ModelParams params();  void setParams(ModelParams p);   // invalidates the cached replay, persists
ReplayResult replay();                 // cached; recomputed lazily after any draw/params change (Replay.run(draws.all(), params, prefs.evaluateLastN()))
int addDraws(Collection<Draw> ds);     // add + persist + invalidate + settleIssued(); returns added count
boolean addDraw(Draw d);  boolean removeDraw(Draw d);  void clearDraws();
Prediction currentPrediction();        // replay().next; ALSO logs it in IssuedLog if there is no pending entry with the same basedOnDate/basedOnCount; persists
int settleIssued();                    // issued().settle(draws()) + persist; returns settled
void clearIssued();
String exportDrawsCsv(); String exportIssuedCsv();
void save();                            // persists everything (atomic write: write tmp file then rename)
```
Loading must be tolerant: a corrupt file yields an empty set plus a logged warning
(`android.util.Log.w`), never a crash.

---------------------------------------------------------------------------------------

## 3. Android UI (`.ui`) — framework widgets only, XML layouts

Theme: `res/values/styles.xml` -> `AppTheme` parent `@android:style/Theme.DeviceDefault.Light.DarkActionBar`
with `android:colorPrimary` `#1E5EFF`, `android:colorPrimaryDark` `#1240B8`, `android:colorAccent` `#FF6D00`;
`res/values-v29/styles.xml` same name with parent `@android:style/Theme.DeviceDefault.DayNight`
and the same colours. Keep the action bar (it provides a title and overflow menus). Every
Activity extends `android.app.Activity` and is declared in the manifest with
`android:exported="false"` except the launcher (`true`). Support rotation (no state loss:
keep selections in `onSaveInstanceState` for the manual entry screen; others reload from
the Repository).

### 3.1 Custom view `NumberGridView extends ViewGroup`
Shows the 22 numbers as circles in a wrapping grid (circle size derived from width: 6
columns on phones; computed in `onMeasure` from the available width, min 40dp, max 72dp,
8dp gaps). Each cell is a `TextView` (child view) with the number; the view supports three
modes via public API:
```java
public void setMode(int mode);  // MODE_DISPLAY (highlight selected numbers), MODE_SELECT (tap toggles, max 11, callback), MODE_HEAT (fill intensity from scores)
public void setSelected(Set<Integer> numbers);  public Set<Integer> getSelectedNumbers();
public void setHighlighted(Set<Integer> numbers);  // secondary highlight ring (e.g. actual draw hits) drawn in a different colour
public void setScores(double[] scores22);            // heat mode: intensity proportional to score / max score; also shows "x.x" small text under the number? (no — keep the number only; show percentages in a separate list)
public void setOnSelectionChangedListener(OnSelectionChangedListener l); // interface with void onSelectionChanged(Set<Integer> selected)
public void setMaxSelection(int max);
```
Colours: selected = colorPrimary with white text; hit (selected AND highlighted) = green
`#2E7D32`; highlighted-only = outline in accent; unselected = `#E3E7EE` with dark text (use
`#2A2F3A` / light text variants in dark mode via `values-night` colours).
Implement backgrounds with `GradientDrawable` (oval) created in code — no PNGs required.

### 3.2 `MainActivity` (launcher) — layout `activity_main.xml` in a `ScrollView`
* Title: app name. Action bar overflow menu: Import file, Add draw, History, Stats &
  accuracy, Settings, Export draws (CSV), Export predictions (CSV), About.
* Card "Next prediction": line `For <display(targetDate)> · based on N draws (latest <date>)`;
  `NumberGridView` in MODE_DISPLAY with the 11 picks selected; below, a one-line text of
  the 11 numbers (`"1 · 4 · 7 ..."`) in large font for easy copying; a "Copy" button that
  copies the numbers to the clipboard (`ClipboardManager`) with a Toast.
* Card "Confidence by number": `NumberGridView` in MODE_HEAT with `next.scores`, plus a text
  listing the top 11 with their expected-count scores as percentages
  (`score/11*100`, e.g. `"7: 61%"`).
* Card "Last check": if `issued().lastSettled()` exists: `Prediction for <date>: X / 11 hits`
  with a MODE_DISPLAY grid where predicted numbers are selected and actual numbers are
  highlighted (hits turn green); otherwise text "No checked prediction yet — add the real
  draw when it is published and the app will score itself."
* Card "Model": backtest line `Backtest: mean X.XX hits / 11 over N draws (random = 5.50)`
  and the three heaviest experts with weights `"transition 23% · decay 19% · freq 15%"`.
* Big buttons: "Add today's draw" -> `AddDrawActivity`; "Import CSV / XML" -> starts
  `Intent.ACTION_OPEN_DOCUMENT` (category OPENABLE, type `*/*`,
  `EXTRA_MIME_TYPES = {"text/csv","text/comma-separated-values","text/plain","text/xml","application/xml","application/octet-stream"}`)
  and on result starts `ImportActivity` with the Uri (`setData`, grant read permission flag).
* Empty state when there are no draws: hide prediction cards except a hint card explaining
  the three ways to add data and the expected file format (`date,n1,...,n11` per line).
* Everything is recomputed in `onResume` (`Repository.currentPrediction()` on a background
  thread; show a small `ProgressBar` while computing; guard against the activity being
  finished before the result arrives).

### 3.3 `AddDrawActivity` — manual entry
* Date row: a Button showing `display(date)` that opens `DatePickerDialog`; default =
  latest draw date + 1 if that is <= today, else today. Buttons "Today" and "Yesterday".
* `EditText` "Quick entry" (hint `"e.g. 1 4 7 9 11 13 15 17 19 21 22"`): on each text change
  parse with `DrawParser.extractNumbers` and sync the grid selection (distinct, max 11);
  typing more than 11 shows an inline error.
* `NumberGridView` in MODE_SELECT; counter text `"7 / 11 selected"`.
* Save button (enabled only with 11 selected): builds the `Draw`; if an equal draw exists
  -> Toast "This draw is already saved"; if another draw exists on that date -> AlertDialog
  "A draw for <date> already exists: <numbers>. Replace it, add as another draw on the same
  day, or cancel?" (Replace removes the old one). After save: `Repository.addDraw`, then
  `settleIssued()`; show a result dialog: if an issued prediction was just settled, show
  `"Scored your prediction for <date>: X / 11 hits"` with the hit numbers; then finish
  (Main refreshes and shows the new next-day prediction).
* Clear button.

### 3.4 `ImportActivity` — preview + confirm
* Reads the Uri on a background thread (`ContentResolver.openInputStream`, UTF-8, cap at
  25 MB with an error beyond). Determines a file name via `OpenableColumns.DISPLAY_NAME`.
* Runs `DrawParser.parse(content, name, prefs.dayFirst())`.
* Shows: file name, `"Found D dated draws, U undated rows, S skipped lines"`, the first
  warnings (scrollable TextView), and a preview `ListView` of up to 50 parsed draws
  (`date — numbers`).
* If `undated > 0`: an options block: radio "Rows are oldest first" / "newest first";
  a date button "Last row's date" (default today; `DatePickerDialog`); these are applied
  through `DrawParser.assignDates` on import.
* If the file has only ambiguous dates the user can toggle "Day-first dates" (re-parses).
* Import button -> `Repository.addDraws(all)`, then `settleIssued()`; result dialog
  `"Imported A new draws (B duplicates skipped)."` + settled info; OK finishes.
* Handles `SecurityException`/IO errors with an error dialog, never a crash.

### 3.5 `HistoryActivity`
* `ListView` newest first via a custom `BaseAdapter` (row: date bold, numbers below).
  Header text `"N draws · from <first> to <last>"`.
* Tap -> dialog with the draw and buttons "Delete" / "Close". Long-press also deletes
  (confirm). Overflow menu: "Add draw", "Delete all draws" (confirm), "Export CSV".
* Export = `Intent.ACTION_CREATE_DOCUMENT` (type `text/csv`, title `draws-<today>.csv`),
  write `Repository.exportDrawsCsv()` to the returned Uri.

### 3.6 `StatsActivity`
Scrollable; sections:
1. "How well does it predict?" — `Stats.summaryText()` from `replay().stats` plus a plain
   text histogram (`"5 hits ████████ 31%"` lines for 0..11 using the ratio to the max) and
   the honesty note (see top of this document) including the sentence:
   `"A random guess scores 5.5 / 11 on average; P(≥8 hits) ≈ " + percent(hypergeometricTail(8))`.
2. "Expert weights" — table-like rows: name, weight %, mean hits (from `model.expertMeanHits()`),
   description, sorted by weight desc.
3. "Your checked predictions" — `issued().stats()` summary + `ListView`-like rows (use a
   `LinearLayout` filled in code to avoid nested scrolling) for the last 30 issued
   predictions: target date, numbers, actual, hits (pending shown as "pending").
4. Buttons: "Auto-tune model" (background thread; shows a ProgressDialog-like state using a
   `ProgressBar`; on completion a dialog with old vs new params and mean hits; "Apply" saves
   via `Repository.setParams` and refreshes), "Reset learning to defaults" (sets default
   `ModelParams` keeping `dayFirst`), "Clear checked predictions" (confirm),
   "Export predictions CSV" (ACTION_CREATE_DOCUMENT `predictions-<today>.csv`).

### 3.7 `SettingsActivity`
Plain form (`EditText`s with `inputType=numberDecimal` / `number`, a `Switch` for
day-first dates, an `EditText` for "Evaluate last N draws"). Fields map 1:1 to
`ModelParams` (+ evaluateLastN). Save validates ranges via `ModelParams.clamp()` and shows the
clamped values. "Restore defaults" button. Help text under each field (one line).

### 3.8 About (dialog from Main's menu)
App name + version, what it does, the honesty note, file format help, and the note that the
model re-learns from the full history each time data changes (so corrections and
backfills are always consistent).

### 3.9 Resources
* `res/values/strings.xml` — all user-visible strings (English). `app_name` = `CSPRNG Predictor`.
* `res/values/colors.xml`, `res/values-night/colors.xml` (grid colours, card background).
* `res/values/styles.xml`, `res/values-v29/styles.xml`.
* `res/drawable/card_bg.xml` (rounded rectangle shape), `res/drawable/button_primary.xml`.
* Launcher icon: `res/mipmap-anydpi-v26/ic_launcher.xml` adaptive icon with
  `res/drawable/ic_launcher_background.xml` (solid colour `#1E5EFF`) and
  `res/drawable/ic_launcher_foreground.xml` (vector: a white circle ring with a 3x3 dot
  grid or simply the digits are NOT needed — keep it as simple shapes), plus legacy PNGs
  `res/mipmap-mdpi|hdpi|xhdpi|xxhdpi|xxxhdpi/ic_launcher.png` (48/72/96/144/192 px)
  generated by `scripts/gen_icons.py` (Pillow): blue rounded square with a white 11-dot
  pattern. Manifest uses `android:icon="@mipmap/ic_launcher"`.
* Menus: `res/menu/main_menu.xml`, `history_menu.xml`.

### 3.10 Manifest
`package="com.medonumbworks.csprngpredictor"`, `versionCode 1`, `versionName "1.0"`,
`<uses-sdk minSdkVersion 24 targetSdkVersion 34/>` (the SDK-less build passes these to
aapt2 too), no permissions (SAF only), `android:allowBackup="true"`, label/icon/theme as above,
`MainActivity` exported with MAIN/LAUNCHER, other activities `exported="false"`,
`android:windowSoftInputMode="adjustResize"` on AddDraw and Settings,
`android:configChanges` NOT used (handle rotation properly instead).

---------------------------------------------------------------------------------------

## 4. Build & test scripts

### 4.1 `scripts/build-apk.sh` (bash, `set -euo pipefail`, run from any cwd)
Environment used here: Ubuntu packages `aapt` (gives `aapt`+`aapt2`), `apksigner`,
`zipalign`, `dalvik-exchange` (`dx`), JDK 21. Framework jar: Robolectric's
`org.robolectric:android-all:14-robolectric-10818077` from Maven Central (it contains the
API-34 classes and `resources.arsc`, so it works for both `javac -cp` and `aapt2 link -I`).
Steps (print each step; fail loudly):
1. Resolve `ROOT` = directory containing `scripts/`. `TOOLS=$ROOT/tools` (git-ignored).
2. `ANDROID_JAR`: `$ANDROID_JAR` env if set; else `$ANDROID_HOME/platforms/android-34/android.jar`
   if it exists; else download the Robolectric jar into `$TOOLS/android-all-14.jar` with
   `curl -fL --retry 3` (skip if present).
3. Tool discovery: `aapt2`, `zipalign`, `apksigner` on PATH or under
   `$ANDROID_HOME/build-tools/*/`; dexer: prefer `d8` if found on PATH or in build-tools,
   else `dx` / `dalvik-exchange`. Print which one is used.
4. Clean `$ROOT/build`, then: `aapt2 compile --dir app/src/main/res -o build/res.zip`;
   `aapt2 link -o build/unaligned.apk -I "$ANDROID_JAR" --manifest app/src/main/AndroidManifest.xml --java build/gen --min-sdk-version 24 --target-sdk-version 34 --version-code $VERSION_CODE --version-name $VERSION_NAME --auto-add-overlay build/res.zip`
   (read VERSION_CODE/VERSION_NAME from `version.properties` at the project root:
   `versionCode=1`, `versionName=1.0`).
5. `javac --release 8 -Xlint:-options -encoding UTF-8 -cp "$ANDROID_JAR" -d build/classes $(find app/src/main/java build/gen -name '*.java')`.
   Note that `app/src/test` is NOT compiled here.
6. Dex: d8 -> `d8 --release --min-api 24 --lib "$ANDROID_JAR" --output build/dex $(find build/classes -name '*.class')`;
   dx -> `dx --dex --min-sdk-version=24 --output=build/dex/classes.dex build/classes`.
7. `cp build/unaligned.apk build/withdex.apk; (cd build/dex && zip -q ../withdex.apk classes.dex)`.
8. `zipalign -f -p 4 build/withdex.apk build/aligned.apk`.
9. `apksigner sign --ks keystore/debug.keystore --ks-pass pass:android --ks-key-alias androiddebugkey --key-pass pass:android --v1-signing-enabled true --v2-signing-enabled true --v3-signing-enabled true --out dist/csprng-predictor-v$VERSION_NAME.apk build/aligned.apk`
   (`KEYSTORE`, `KS_PASS`, `KS_ALIAS` env overrides for a real release key).
10. `apksigner verify --verbose dist/...apk` and, if `aapt` exists, `aapt dump badging` first
    lines. Print the final path and size.

### 4.2 `scripts/run-tests.sh`
Downloads `junit-4.13.2.jar` and `hamcrest-core-1.3.jar` from Maven Central into `tools/`
(skip if present), compiles `app/src/main/java/**/core/*.java` and
`app/src/test/java/**/*.java` with `javac --release 8` into `build/test-classes`, runs
`java -cp ... org.junit.runner.JUnitCore <all test classes>` (discover test classes by
listing `*Test.class` files), exits non-zero on failure. Also run a quick "no lambda"
guard: `grep -rnE '\) *->|::' app/src/main/java` must find nothing (print offending lines).

### 4.3 `scripts/gen_icons.py`
Pillow script that writes the legacy launcher PNGs (sizes above) and a 512px
`docs/icon-512.png`.

### 4.4 Gradle project (for Android Studio users; cannot be verified in this environment)
`settings.gradle` (`pluginManagement` with google()/mavenCentral()/gradlePluginPortal(),
`rootProject.name = "csprng-predictor"`, `include ':app'`), root `build.gradle`
(`plugins { id 'com.android.application' version '8.7.3' apply false }`), `gradle.properties`
(`android.useAndroidX=false`, `org.gradle.jvmargs=-Xmx2g`), `app/build.gradle`:
```groovy
plugins { id 'com.android.application' }
android {
  namespace 'com.medonumbworks.csprngpredictor'
  compileSdk 34
  defaultConfig { applicationId 'com.medonumbworks.csprngpredictor'; minSdk 24; targetSdk 34; versionCode 1; versionName '1.0' }
  signingConfigs { debug { storeFile file('../keystore/debug.keystore'); storePassword 'android'; keyAlias 'androiddebugkey'; keyPassword 'android' } }
  buildTypes { release { minifyEnabled false; signingConfig signingConfigs.debug } }
  compileOptions { sourceCompatibility JavaVersion.VERSION_1_8; targetCompatibility JavaVersion.VERSION_1_8 }
}
dependencies { testImplementation 'junit:junit:4.13.2' }
```
Note: with AGP the manifest must not contain `package=` (namespace is in Gradle) — but the
SDK-less aapt2 build needs it. Keep `package=` in the manifest; AGP 8 only warns and uses
the namespace. Add the Gradle wrapper via `gradle wrapper --gradle-version 8.14.3`
(offline-capable) so `./gradlew assembleDebug` works for Android Studio users.

---------------------------------------------------------------------------------------

## 5. Tests (`app/src/test/java/com/medonumbworks/csprngpredictor/core/`), JUnit 4

Minimum coverage (each a separate test class, descriptive method names):
* `DateUtilTest`: every format listed in 1.3, ambiguity rule, invalid dates (31.02.2026),
  `addDays` across month/year boundaries, `findDate` inside text, 2-digit years.
* `DrawParserCsvTest`: comma/semicolon/tab/pipe/space delimiters, header lines, numbers in
  one cell, date in the middle, extra id columns (run-of-11 rule), date digits never leak
  into numbers, undated rows, warnings for 10/12-number lines, CRLF, BOM, empty input,
  garbage input.
* `DrawParserXmlTest`: the three canonical forms, nested wrappers, attribute-dates with
  different formats, date on an ancestor, malformed XML (returns empty result with a warning,
  no exception), deepest-element rule.
* `DrawSetCsvTest`: round trip write/read, dedupe, ordering, `firstAfter`, `onDate`.
* `StrategyTest`: each default strategy returns length-22 non-negative arrays on empty /
  one / many draws; transition and gap behave as specified on a hand-built history.
* `ModelTest`: predict is deterministic and returns 11 distinct sorted numbers; weights sum
  to 1 after observe; an expert that always predicts perfectly gains weight over one that
  is always wrong (construct two fake strategies); bias moves toward observed numbers.
* `ReplayTest`: record count = N-1; `next.targetDate` = latest + 1; `Stats` numbers
  (histogram sums to n, hypergeometric pmf sums to 1 over 0..11, tail(0)=1); on 300
  deterministic pseudo-random draws (seeded `java.util.Random` **in the test only**) the mean
  hits is between 4.0 and 7.0 (sanity, not a claim of skill); a planted pattern (e.g. draw t
  always equals draw t-1 rotated by one number) yields mean hits > 8.
* `AutoTuneTest`: returns base for < 15 draws; otherwise returns a grid member with stats.
* `IssuedLogTest`: CSV round trip incl. pending entries, `settle` picks the first draw after
  basedOnDate, `stats()`.
* `ModelParamsTest`: serialize/parse round trip, clamp, defaults on garbage.

---------------------------------------------------------------------------------------

## 6. Samples
* `samples/draws.csv` — 40 dated draws, `date,n1..n11` with ISO dates (generated
  deterministically; include a header line).
* `samples/draws-nodates.csv` — 10 rows of 11 numbers, semicolon-separated, no dates.
* `samples/draws.xml` — mixes the three canonical XML forms, 6 draws.
* `samples/README.md` — one paragraph on formats.

## 7. README.md (project root `csprng-predictor/README.md`)
What the app does, honesty note, screenshots section placeholder, how to install the APK
(`dist/`), file formats with examples, how the model works (experts + Hedge + bias +
replay), how it "corrects itself", build instructions for both paths (Android Studio /
`scripts/build-apk.sh` with the Ubuntu packages and the Maven Central framework jar),
how to run tests, and the debug-key warning.
