# On-device sentiment: VADER + crypto slang

CryptoPulse scores social posts (Reddit, StockTwits, Telegram) and news headlines on the phone with
`LexiconSentimentScorer` (package `dev.cryptopulse.core.analysis.sentiment`). It is a faithful Kotlin port
of **VADER** (vaderSentiment 3.3.2, C.J. Hutto, MIT licence in `core/src/main/resources/sentiment/VADER_LICENSE.txt`)
plus a crypto slang overlay. It needs no network access, no API key and no model download.

## How it works

1. **Normalize** (`CryptoTextNormalizer`, crypto entry points only), in this order:
   a) URLs matching `(https?://\S+|www\.\S+)` become one space; b) `$` or `#` directly before an ASCII
   letter is dropped (`$BTC` -> `BTC`); c) multi-word lexicon phrases, longest first, are matched
   case-insensitively on whole words (neighbours not `[A-Za-z0-9]`) and replaced by their lowercase
   underscore key (`Rug Pull` -> `rug_pull`); d) mapped emoji become ` emoji_<name> `; e) whitespace runs
   collapse to one space and the text is trimmed. Other words keep their case, because VADER boosts
   ALL CAPS words.
2. **Score with VADER**: the bundled `vader_lexicon.txt` (7,506 words, emoticons, slang) overlaid with
   `crypto_lexicon.tsv` (447 entries; crypto values win). All VADER rules apply unchanged: negation
   ("not bullish"), boosters ("very", "kinda"), ALL CAPS, `!`/`?` emphasis, "but" shifts, idioms.
3. **Report** `TextSentiment(score, matchedPositive, matchedNegative, hits)`: `score` is VADER's compound
   (-1..1); the lists hold up to 8 distinct lexicon tokens with positive/negative *raw* valence, in order
   of appearance; `hits` counts every sentiment-bearing token.

| Entry point | Normalizes? |
|---|---|
| `Vader.default().polarityScores(text)` | No: plain VADER 3.3.2, bit-for-bit |
| `Vader.cryptoDefault().polarityScores(text)` / `.analyze(text)` | Yes |
| `vader.rawPolarityScores(text)` / `.analyzeRaw(text)` on any instance | Never |
| `LexiconSentimentScorer()` (uses `Vader.cryptoDefault()`) | Yes |
| `LexiconSentimentScorer(Vader.default())` | No |

`crypto_lexicon.tsv` is UTF-8 with `#` comments: `term<TAB>valence[<TAB>comment]` with valence in [-4, 4],
or `EMOJI<TAB>valence<TAB>emoji_<name>` for emoji. The file header contains the 18-line Python reader that
the cloud twin uses, and the normalizer KDoc gives the equivalent Python `re` calls. Parity is tested: `vader_reference.tsv`
(194 sentences scored by the Python package) and `crypto_reference.tsv` (83 sentences scored by the Python
twin, including the exact normalized text) must match within 1e-3 (the VADER rows even match exactly, digit for digit).

## Why VADER fits a phone

- **Rules and a lexicon, no model**: about 570 KB of text resources inside the APK, parsed once on
  first use; nothing to download and no GPU or NNAPI needed.
- **Offline and private**: posts never leave the device to be scored.
- **Fast and battery friendly**: roughly 25-50 microseconds per short post on a desktop JVM (the test
  suite scores 10,000 posts in 0.25-0.5 s against a 3 s budget). Instances are immutable, so one is shared
  across threads.
- **Built for short social text**: VADER was designed and validated on tweets, so it understands
  emoticons, emoji, slang, capitals and `!!!`, which is what crypto chatter looks like.
- **Explainable**: the matched tokens show why a post scored the way it did.

## Limits

- **English only.** Other languages are scored as if they were English. Most words are unknown and
  therefore neutral, but false friends distort the score: German "Die Preise steigen heute stark"
  ("prices rise strongly today") scores -0.60 because "die" is -2.9 in VADER.
- **Sarcasm and irony** ("great, another rug pull") are scored literally.
- **Slang drifts.** New memes and tickers need lexicon updates; a term absent from the lexicon is neutral.
- **Context-free valences.** "short" or "long" positions, "support" levels and coin names that are
  English words can be misread. Some technical words VADER rates (e.g. "block", "difficulty") are set
  to 0 in the crypto lexicon.
- **Inherited VADER 3.3.2 quirks are kept on purpose** for parity: "kind of" before a negative word makes
  it *more* negative, the "but" rule rescales the first equal value, and a zero-valence word in ALL CAPS
  gets -0.733. The normalizer only matches phrases written with single spaces ("rug  pull" is missed).
- **Overlapping phrases chain into one token.** `_` is not a word character for the phrase matcher, so
  "strong buy" + "buy signal" turn "strong buy signal" into the single token `strong_buy_signal`, which
  scores 0 unless it is a lexicon entry itself. The combinations that occur in real posts are entries
  ("to the moon soon", "strong sell signal", "interest rate cut", "panic sell-off", ...);
  `CryptoLexiconTest` lists the remaining chains, which are left unscored on purpose.

## How FinBERT and CryptoBERT complement it

Transformer models read whole sentences and handle context, negation scope and some sarcasm much
better than a lexicon. **FinBERT** (financial news and filings) and **CryptoBERT** (crypto social posts)
weigh roughly 110-500 MB each and need hundreds of megabytes of RAM and noticeable CPU per post. That
is too heavy for battery-friendly on-device use, so they run in the free cloud sentiment hub on GitHub
Actions on a schedule, over shared public feeds, and publish their scores. The phone keeps scoring every
post with VADER, so it still has a usable signal offline or before the hub has published anything; when
model scores exist they add context-aware second and third opinions on the same posts. The hub's Python
twin reads the same `crypto_lexicon.tsv` and applies the same normalization rules, so the lexicon
baseline is identical on the phone and in the cloud.
