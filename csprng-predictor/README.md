<p align="center"><img src="docs/icon-512.png" width="112" alt="CSPRNG Predictor icon"></p>

# CSPRNG Predictor

An Android app for a daily **"pick 11 numbers out of 1..22"** draw. You feed it the past
draws (CSV / XML import or manual entry), it predicts the 11 numbers for the next draw, and
when you enter the real draw it scores its own prediction and adapts its model for the
following day.

> **Honesty note.** A true CSPRNG (cryptographically secure pseudo-random number generator)
> has no exploitable pattern. Over many draws the model's average hit count is therefore
> expected to stay near the **random baseline of 5.5 / 11**. The app measures and displays
> this (backtest and real, checked predictions) so you can see for yourself whether the source
> behaves randomly. A result clearly and persistently above 5.5 would mean the source is *not*
> a good random generator, not that the app has found a trick that works on fair draws.

- Android 7.0+ (API 24), targets Android 14 (API 34)
- No permissions, no network access, no ads, no tracking; your data stays on the phone
- Pure Java, framework widgets only (no AndroidX, no third-party libraries)

---------------------------------------------------------------------------------------

## Contents

1. [Screenshots](#screenshots)
2. [Install](#install)
3. [Using the app](#using-the-app)
4. [File formats](#file-formats)
5. [How the model works](#how-the-model-works)
6. [How it corrects itself](#how-it-corrects-itself)
7. [What to expect: the random baseline](#what-to-expect-the-random-baseline)
8. [Building](#building)
9. [Running the tests](#running-the-tests)
10. [Project layout](#project-layout)
11. [Debug key warning](#debug-key-warning)

---------------------------------------------------------------------------------------

## Screenshots

*Screenshots will be added here.*

| Next prediction | Add draw | Stats & accuracy |
|:---:|:---:|:---:|
| *(placeholder)* | *(placeholder)* | *(placeholder)* |

---------------------------------------------------------------------------------------

## Install

The ready-made APK is in [`dist/`](dist/): **`dist/csprng-predictor-v1.0.apk`**.

**On the phone**

1. Copy the APK to the phone (USB, cloud drive, e-mail, ...) and tap it in a file manager.
2. Android asks you to allow installing apps from that source ("Install unknown apps");
   allow it for the file manager / browser you used, then tap **Install**.
3. Google Play Protect may warn that the app is from an unknown developer (it is signed with
   a public debug key, see [Debug key warning](#debug-key-warning)); choose
   *Install anyway*.

**With adb** (USB debugging enabled):

```bash
adb install -r dist/csprng-predictor-v1.0.apk
```

Updates built by `scripts/build-apk.sh` are signed with the same key, so they install over the
previous version and keep your data. If you switch to a different signing key you must
uninstall first, which deletes the app's data: export your draws (CSV) before doing that.

---------------------------------------------------------------------------------------

## Using the app

**Daily routine**

1. **Load the history once.** Use *Import CSV / XML* (see [File formats](#file-formats)) or
   add draws by hand. The more draws, the more meaningful the statistics.
2. **Read the prediction.** The main screen shows the 11 predicted numbers for the next draw
   date (latest draw + 1 day) and a *Copy* button. The prediction you were shown is logged.
3. **Enter the real draw** when it is published: *Add today's draw*, tap the 11 numbers (or
   type them into *Quick entry*, e.g. `1 4 7 9 11 13 15 17 19 21 22`) and *Save*. The app
   immediately scores the prediction it showed you (*"Scored your prediction for ...: X / 11
   hits"*), re-learns from the full history and shows the prediction for the next day.

**Screens**

| Screen | What it does |
|---|---|
| **Main** | Next prediction (grid + large copyable number line), *Confidence by number* heat map with the top picks as percentages, *Last check* (predicted vs. actual, hits in green), *Model* card (backtest mean and the three heaviest experts). Overflow menu: Import file, Add draw, History, Stats & accuracy, Settings, Export draws (CSV), Export predictions (CSV), About. |
| **Add draw** | Date picker (+ *Today* / *Yesterday*), quick-entry text field, tap grid, *Save* (enabled at exactly 11 numbers). Warns about duplicates and offers *Replace / add as another draw / cancel* when the date already has a draw. |
| **Import** | Preview before anything is saved: dated draws, undated rows, skipped lines, warnings and the first 50 parsed draws. For undated rows choose the row order and the date of the last row. A *Day-first dates* switch re-parses ambiguous dates. |
| **History** | All draws, newest first. Tap a draw to view or delete it, long-press to delete. Menu: Add draw, Delete all draws, Export CSV. |
| **Stats & accuracy** | Backtest summary and hit histogram, the honesty note, expert weights with their mean hits, your checked (real) predictions, *Auto-tune model*, *Reset learning to defaults*, *Clear checked predictions*, *Export predictions CSV*. |
| **Settings** | All model parameters (see [below](#parameters)), day-first dates, *Evaluate last N draws*. Values are clamped to their valid ranges on save. |

Exports use Android's file picker (*Storage Access Framework*), so the app never needs a
storage permission.

---------------------------------------------------------------------------------------

## File formats

Ready-to-import examples are in [`samples/`](samples/) (see [samples/README.md](samples/README.md)).
Files of up to 25 MB are accepted. Parsing never crashes on bad input: lines that do not
contain a usable draw are skipped and reported as warnings in the import preview.

### CSV / TSV / TXT

The app's own format (also what *Export draws (CSV)* writes):

```csv
date,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11
2026-10-15,2,3,6,9,12,15,16,17,19,21,22
2026-10-16,1,3,7,8,9,13,14,16,17,18,20
```

The importer is deliberately tolerant:

- **Delimiters:** `,` `;` tab `|` or plain whitespace (detected per line).
- **Header lines** and any other line without 11 valid numbers are skipped silently.
- **The date may be in any column**, or embedded in text: `Draw 2026-10-16: 1 4 7 ...`.
- **Several numbers in one cell** are fine: `2026-10-16;"1 4 7 9 11 13 15 17 19 21 22"` or
  `1-4-7-9-...`, `1/4/7/...`, `1+4+7+...`.
- **Extra columns** such as a draw id or a year are ignored when they are outside 1..22. If a
  line has more than 11 candidates, the app looks for a run of exactly 11 single-number
  cells (preferring the run right after the date); otherwise the line is skipped with a
  warning such as `line 12: 13 numbers found, expected 11`.
- **Date digits never leak into the numbers** (the `16` of `2026-10-16` is not a draw number).
- **UTF-8 with or without BOM, LF or CRLF line endings.**

**Accepted date formats**

| Format | Example | Notes |
|---|---|---|
| `yyyy-MM-dd`, `yyyy/MM/dd`, `yyyy.MM.dd` | `2026-10-16` | ISO style, preferred |
| `yyyyMMdd` | `20261016` | 8 digits |
| `dd.MM.yyyy`, `dd-MM-yyyy`, `dd/MM/yyyy` | `16.10.2026` | day first |
| `MM/dd/yyyy`, `MM-dd-yyyy` | `10/16/2026` | month first |
| two-digit years | `16.10.26` | read as 20yy |
| date-times | `2026-10-16T20:00:00`, `2026-10-16 20:00` | the time is ignored |

Ambiguity rule for `a/b/yyyy`, `a-b-yyyy`, `a.b.yyyy`: if `a > 12` the date is day-first,
else if `b > 12` it is month-first, else the *Day-first dates* setting decides (default: on).
Dates must be real calendar dates (`31.02.2026` is rejected) between 1900 and 2100.

**Rows without dates**

```csv
2;3;5;6;7;9;12;13;14;19;22
1;2;3;6;10;12;13;14;16;17;22
```

Undated rows are treated as consecutive daily draws. In the import preview you choose whether
the rows are *oldest first* or *newest first* and the date of the last (most recent) row;
the app assigns one day per row going backwards from that date.

### XML

Any reasonably structured XML works. The app looks for the deepest elements whose own text
(attribute values that are not dates, direct text and simple child elements) contains exactly
11 distinct numbers from 1..22. Three canonical forms, which may be mixed in one file:

```xml
<draws>
  <draw date="2026-10-16"><n>1</n><n>3</n><n>7</n><n>8</n><n>9</n><n>13</n><n>14</n><n>16</n><n>17</n><n>18</n><n>20</n></draw>
  <draw date="2026-10-15" numbers="2 3 6 9 12 15 16 17 19 21 22"/>
  <draw><date>14.10.2026</date><numbers>1,2,4,5,8,9,10,13,17,19,20</numbers></draw>
</draws>
```

The date of a draw is taken from (in this order) a `date`/`datum`/`day`/`drawdate`/
`draw_date`/`time`/`datetime` attribute, any attribute value that parses as a date, a child
element with one of those names, a date inside the element's text, or the nearest ancestor's
date attribute/child. Draws without any date are handled like undated CSV rows. DTDs and
external entities are not loaded.

### Exported predictions

*Export predictions (CSV)* writes the log of predictions the app actually showed you:

```csv
issuedAt,basedOnDate,basedOnCount,targetDate,numbers,actualDate,actual,hits
2026-10-16,2026-10-16,40,2026-10-17,1 3 4 7 8 9 13 14 16 17 20,2026-10-17,2 3 4 6 9 11 13 15 16 19 21,5
2026-10-17,2026-10-17,41,2026-10-18,1 2 3 4 6 9 13 14 16 17 20,,,
```

Empty `actualDate,actual,hits` cells mean the prediction is still pending (its draw has not
been entered yet).

---------------------------------------------------------------------------------------

## How the model works

The predictor is an **ensemble of simple "experts"** combined with an online learning
algorithm (**Hedge** with fixed share) plus a small **per-number bias correction**. Everything
is deterministic: the same history and settings always give the same prediction.

### The experts

Each expert looks at the draws so far and gives every number 1..22 a score.

| id | Idea | Score for number *j* |
|---|---|---|
| `uniform` | Control expert: "it is random" | 1 for every number |
| `freq` | All-time hot numbers | count(*j*) + 1 (Laplace smoothed) |
| `recent10` | Hot over a short window | count in the last *shortWindow* draws + 0.5 |
| `recent30` | Hot over a long window | count in the last *longWindow* draws + 0.5 |
| `decay` | Hot, with exponentially fading memory | Σ 0.5^(age / *halfLife*) + 0.5 |
| `gap` | Overdue numbers | draws since *j* last appeared + 1 |
| `transition` | Markov chain from the last draw | mean over *i* in the last draw of (T[i][j] + 1) / (C[i] + 2), where T counts "*j* followed a draw containing *i*" |
| `repeat` | Repeats from the last draw | observed repeat rate for numbers in the last draw, 1 − rate for the others |
| `pairs` | Co-occurrence with the last draw | Σ over *i* in the last draw of (P[i][j] + 1) / (count(*i*) + 2) |

If the `uniform` expert ends up with the largest weight, the data looks random to the model;
that is the expected outcome for a real CSPRNG.

### Combining them

1. Every expert's scores are normalised to **expected counts that sum to 11** (so a score of
   0.5 means "50 % chance this number is drawn", the uniform value).
2. The combined score is the weighted average of the experts, using the current weights
   (they start equal and always sum to 1).
3. A **per-number bias** *b* multiplies each score by e^b; the result is renormalised to 11.
4. The **11 numbers with the highest scores** are the prediction (ties go to the lower number).
   The final scores are what the *Confidence by number* card shows (score / 11 × 100 %).

### Learning after each draw

When the actual draw for a prediction is known:

- **Hedge:** each expert's own top-11 is scored; its loss is `1 − hits/11` and its weight is
  multiplied by `e^(−eta · loss)`, then all weights are renormalised.
- **Fixed share:** a fraction *share* of the total weight is spread evenly over all experts,
  so an expert that was bad for a while can recover quickly if it starts working.
- **Bias:** for every number, `b ← (1 − biasDecay) · b + biasRate · (drawn − predicted probability)`,
  clamped to [−1, 1]. Numbers that keep appearing more often than predicted are nudged up,
  the others down; the decay makes old corrections fade.

### Walk-forward replay (backtest)

Whenever the draws or the settings change, the model is rebuilt **from scratch**: starting
with no knowledge it walks through the history in date order, predicts draw *t* from draws
*0 .. t−1* only, scores that prediction, learns from it, and moves on. The final state
predicts the next, unseen draw. Because every backtest prediction uses only earlier data,
the backtest accuracy is an honest estimate (no look-ahead). Internal statistics are updated
incrementally, so a full replay is linear in the number of draws.

The Stats screen reports, over the last *N* backtest predictions (*Evaluate last N draws*,
default 200): mean hits, standard deviation, edge (= mean − 5.5), a z-score
`(mean − 5.5) / (1.2 / √n)`, best/worst, the mean of the last ten and a hit histogram.

**Auto-tune** replays the history for a small grid of settings (`eta` ∈ {0.1, 0.3, 1.0} ×
`halfLife` ∈ {5, 15, 40} × `biasRate` ∈ {0, 0.05, 0.2}) and offers the one with the best mean
hits over the evaluation window (needs at least 15 draws). Treat a small improvement with
suspicion: picking the best of 27 settings on the same data is itself a form of overfitting.

### Parameters

| Setting | Default | Range | Meaning |
|---|---|---|---|
| `eta` | 0.5 | 0 – 5 | Hedge learning rate: how fast weights move towards the better experts |
| `share` | 0.02 | 0 – 0.5 | Fixed-share mixing towards equal weights |
| `biasRate` | 0.1 | 0 – 1 | Strength of the per-number correction |
| `biasDecay` | 0.05 | 0 – 1 | How quickly per-number corrections fade |
| `shortWindow` | 10 | 2 – 200 | Window of the `recent10` expert (draws) |
| `longWindow` | 30 | 2 – 1000 | Window of the `recent30` expert (draws) |
| `halfLife` | 15 | 1 – 500 | Half-life of the `decay` expert (draws) |
| Day-first dates | on | on/off | How ambiguous dates like `03/04/2026` are read |
| Evaluate last N draws | 200 | | Window for the backtest statistics and auto-tune |

---------------------------------------------------------------------------------------

## How it corrects itself

1. **Every prediction shown is logged** ("issued"), together with the draw date and the number
   of draws it was based on.
2. **When you add the real draw** (manually or by import), each pending prediction is settled
   against the first draw after the date it was based on, and the app tells you how many of
   the 11 numbers it hit. These real results are tracked separately from the backtest under
   *Your checked predictions*.
3. **The model re-learns from the full history** after every change: the walk-forward replay
   updates the expert weights (experts that would have done better gain weight) and the
   per-number biases, then produces the next day's prediction. Because the replay always starts
   from zero, corrections, deletions and back-filled older draws are always reflected
   consistently; there is no hidden state that can drift.
4. **Auto-tune** (optional) re-selects the learning parameters on the accumulated data.

Self-correction cannot create information that is not there. With fair random draws the
weights wander, `uniform` tends to stay competitive, and the mean stays near 5.5.

---------------------------------------------------------------------------------------

## What to expect: the random baseline

For any fixed pick of 11 numbers and a fair draw of 11 out of 22, the number of hits follows
a hypergeometric distribution with mean **5.5** and variance 1.44 (standard deviation ≈ 1.2):

| hits | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|
| P(exactly) | 0.43 % | 3.86 % | 15.4 % | 30.3 % | 30.3 % | 15.4 % | 3.86 % | 0.43 % | 0.017 % | 1 in 705 432 |
| P(at least) | 99.98 % | 99.6 % | 95.7 % | 80.3 % | 50.0 % | 19.7 % | 4.3 % | 0.45 % | 0.017 % | 1 in 705 432 |

So 7 or more hits happen on about one day in five by pure chance, and 8+ on about one day in
23. Judge the model by its **mean over many draws**: after *n* draws the mean of a random
guesser has a standard error of about 1.2 / √n (0.12 after 100 draws, 0.085 after 200). A
z-score between −2 and +2 is entirely consistent with a random source.

---------------------------------------------------------------------------------------

## Building

There are two ways to build. The SDK-less script is the one that was used and verified to
produce `dist/csprng-predictor-v1.0.apk`; the Gradle project is provided for Android Studio
users but **could not be tested in the build environment** (no Android SDK and no access to
Google's Maven repository there).

### Path A: `scripts/build-apk.sh` (no Android SDK needed; verified)

Tested on Ubuntu 24.04 with OpenJDK 21. Install the tools:

```bash
sudo apt install openjdk-21-jdk-headless zip curl \
                 aapt apksigner zipalign dalvik-exchange
```

| Ubuntu package | Provides |
|---|---|
| `aapt` | `aapt2` (resource compiler/linker) and `aapt` (used for `aapt dump badging`) |
| `apksigner` | APK signing and verification (v1 + v2 + v3) |
| `zipalign` | 4-byte alignment of the APK |
| `dalvik-exchange` | the legacy `dx` dexer (Java bytecode → `classes.dex`) |
| `openjdk-21-jdk-headless` | `javac` (any JDK ≥ 9 works; sources are compiled with `--release 8`) |

The four Android tools come from Ubuntu's *universe* component (versions verified on 24.04
"noble": `aapt` 1:14~beta1, `apksigner` 31.0.2, `zipalign` and `dalvik-exchange` 10.0.0+r36);
enable it with `sudo add-apt-repository universe` if `apt` cannot find them.

**Framework jar.** Instead of the SDK's `android.jar` the script uses Robolectric's
`android-all` jar from Maven Central, which contains the API-34 classes *and*
`resources.arsc`, so it works both for `javac -cp` and for `aapt2 link -I`:

- Maven coordinates: **`org.robolectric:android-all:14-robolectric-10818077`**
- URL: <https://repo1.maven.org/maven2/org/robolectric/android-all/14-robolectric-10818077/android-all-14-robolectric-10818077.jar>
  (≈ 131 MB, SHA-1 `94b1490a891e9be559aa35c87cd8a0c163f32d83`)

The script downloads it once into `tools/android-all-14.jar` (git-ignored, checksum
verified) if that file is missing. To use something else, set `ANDROID_JAR=/path/to/android.jar`,
or set `ANDROID_HOME` to an SDK that has `platforms/android-34/android.jar`.

**Build:**

```bash
./scripts/build-apk.sh          # works from any directory
```

Steps (each is printed; the script stops at the first error):

1. Read `versionCode` / `versionName` from [`version.properties`](version.properties).
2. Locate the framework jar (see above).
3. Find `aapt2`, `zipalign`, `apksigner` and a dexer on `PATH` or in
   `$ANDROID_HOME/build-tools/*/`; **`d8` is preferred** when present, otherwise `dx` /
   `dalvik-exchange`.
4. `aapt2 compile` the resources and `aapt2 link` them with the manifest
   (min SDK 24, target SDK 34, version from `version.properties`), generating `R.java`.
5. `javac --release 8` all of `app/src/main/java` (tests are not included), then guard the
   bytecode: no `invokedynamic` (lambdas / method references), no `java.time`,
   `java.util.stream`, `java.util.Base64`, `java.nio.file`, `org.json`; the `core` package
   must not import `android.*`.
6. Dex (`d8 --min-api 24` or `dx --min-sdk-version=24`), add `classes.dex` to the APK,
   `zipalign -p 4`.
7. Sign with `apksigner` (v1 + v2 + v3), verify, copy to
   `dist/csprng-predictor-v<versionName>.apk`, print `aapt dump badging`, the size and SHA-256.

Intermediate files go to `build/` (git-ignored).

**Signing with your own key:**

```bash
keytool -genkeypair -v -keystore ~/keys/release.keystore -alias release \
        -keyalg RSA -keysize 4096 -validity 10000
KEYSTORE=~/keys/release.keystore KS_ALIAS=release KS_PASS='...' KEY_PASS='...' \
    ./scripts/build-apk.sh
```

(`KEY_PASS` defaults to `KS_PASS`. Passwords are handed to `apksigner` through the
environment, not the command line.)

**Bumping the version:** edit `version.properties` (used by the script) *and* the
`versionCode` / `versionName` in `app/build.gradle` (used by Gradle).

### Path B: Android Studio / Gradle (untested here)

Requirements: Android Studio (Ladybug or newer) or a command-line Android SDK with platform 34,
JDK 17+, and internet access to Google's Maven repository.

- Open the `csprng-predictor/` folder in Android Studio, or use the Gradle wrapper
  (Gradle 8.14.3, Android Gradle Plugin 8.7.3):

  ```bash
  ./gradlew assembleDebug        # app/build/outputs/apk/debug/
  ./gradlew assembleRelease      # signed with keystore/debug.keystore (see app/build.gradle)
  ./gradlew test                 # JUnit tests in app/src/test
  ```

- The manifest keeps `package="com.medonumbworks.csprngpredictor"` because the SDK-less
  `aapt2` build needs it. AGP 8 takes the package from `namespace` in `app/build.gradle` and
  only warns about the attribute. If a future AGP version turns that warning into an error,
  remove the attribute for the Gradle build (the script needs it back).
- The Gradle wrapper files (`gradlew`, `gradlew.bat`, `gradle/wrapper/`) were generated with
  `gradle wrapper --gradle-version 8.14.3`; the wrapper downloads Gradle itself on first use.

### Launcher icons

The legacy PNG icons (`res/mipmap-*dpi/ic_launcher.png`, 48–192 px) and `docs/icon-512.png`
are generated by a Pillow script; Android 8+ uses the vector adaptive icon in
`res/mipmap-anydpi-v26/ic_launcher.xml` instead.

```bash
sudo apt install python3-pil   # or: pip install Pillow
python3 scripts/gen_icons.py
```

---------------------------------------------------------------------------------------

## Running the tests

The prediction engine (`com.medonumbworks.csprngpredictor.core`) is plain Java with no Android
dependencies, so its JUnit 4 tests run on a normal JVM:

```bash
./scripts/run-tests.sh                        # all tests
./scripts/run-tests.sh DateUtilTest ModelTest # selected test classes
```

The script

1. checks `app/src/main/java` for lambda / method-reference syntax (`grep -rnE '\) *->|::'`),
   banned imports (`java.time`, `java.util.stream`, `java.util.Base64`, `java.nio.file`,
   `org.json`, `androidx`) and `android.*` imports in `core`, printing any offending lines;
2. downloads `junit-4.13.2.jar` and `hamcrest-core-1.3.jar` from Maven Central into `tools/`
   (once);
3. compiles `app/src/main/java/**/core/*.java` with `javac --release 8` and *no* Android jar
   (proving the core is Android-free) and checks its bytecode for `invokedynamic`;
4. compiles `app/src/test/java/**/*.java` and runs `org.junit.runner.JUnitCore` over every
   `*Test` class into `build/test-classes/`;
5. exits non-zero if anything fails (with no test sources it prints a notice and exits 0).

With Gradle: `./gradlew test`.

---------------------------------------------------------------------------------------

## Project layout

```
csprng-predictor/
  app/src/main/AndroidManifest.xml
  app/src/main/java/com/medonumbworks/csprngpredictor/        app glue: Repository, Prefs
  app/src/main/java/com/medonumbworks/csprngpredictor/core/   pure-Java engine (parsing, experts, model, replay, stats)
  app/src/main/java/com/medonumbworks/csprngpredictor/ui/     activities and the NumberGridView
  app/src/main/res/                                           layouts, strings, styles, icons
  app/src/test/java/com/medonumbworks/csprngpredictor/core/   JUnit 4 tests for the engine
  app/build.gradle, build.gradle, settings.gradle, gradle.properties, gradlew   Gradle (Android Studio)
  scripts/build-apk.sh      SDK-less build -> dist/*.apk
  scripts/run-tests.sh      JVM unit tests + source guards
  scripts/gen_icons.py      launcher PNGs
  keystore/debug.keystore   public debug key (alias androiddebugkey, password "android")
  samples/                  example CSV / XML files
  dist/                     the built APK
  docs/SPEC.md              full specification
  version.properties        versionCode / versionName for the script build
```

**Code constraints** (enforced by the build): Java only, compiled with `--release 8` and dexed by
the legacy `dx` at minSdk 24, so no lambdas, method references, default/static interface
methods or `java.util.stream`; no `java.time`, `java.util.Base64`, `java.nio.file`, `org.json`,
AndroidX or third-party libraries; APIs newer than 24 only behind `Build.VERSION.SDK_INT`
checks. See [docs/SPEC.md](docs/SPEC.md) for the full contract.

App data lives in the app's private storage (`draws.csv`, `issued.csv` and the settings).
`android:allowBackup="true"` lets Android's own backup include it.

---------------------------------------------------------------------------------------

## Debug key warning

The APK in `dist/` (and every build that does not override `KEYSTORE`) is signed with
**`keystore/debug.keystore`, a debug key whose passwords (`android`) are published in this
repository**. That is fine for installing the app on your own devices, but:

- anyone can sign an APK with the same key, so Android cannot tell a genuine update from a
  modified one signed by someone else;
- Google Play does not accept debug-signed apps, and Play Protect may warn about them.

For anything beyond personal use, create your own release key (see
[Signing with your own key](#path-a-scriptsbuild-apksh-no-android-sdk-needed-verified)) and keep
it private. Switching keys requires uninstalling the debug-signed app first (export your data
before that).
