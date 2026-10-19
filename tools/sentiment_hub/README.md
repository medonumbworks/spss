# CryptoPulse sentiment hub

A free cloud job that turns crypto chatter into one small JSON feed for the CryptoPulse Android
app and MCP server. It runs every hour on GitHub Actions (free for the public repository
`medonumbworks/spss`) and publishes

    https://raw.githubusercontent.com/medonumbworks/spss/sentiment-feed/feed.json

```
 channels.json ──► telegram_public  (t.me/s/<handle> web previews, no login)  ─┐
 your account  ──► telegram_account (Telethon, optional secrets)               ├─► dedupe ─► score ─► aggregate ─► feed.json
 6 news feeds  ──► rss              (RSS 2.0 / Atom / RSS 1.0)                 │    VADER-crypto        per Bitpanda asset
 4 subreddits  ──► reddit           (public .json listings)                    ─┘    FinBERT, CryptoBERT
```

Everything is free: no paid API, no API key except your own Telegram login (optional).

## Run it locally

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r tools/sentiment_hub/requirements.txt -r tools/sentiment_hub/requirements-dev.txt
# optional, ~200 MB CPU torch + transformers for FinBERT and CryptoBERT (models download on first use, ~1 GB):
pip install -r tools/sentiment_hub/requirements-nlp.txt --extra-index-url https://download.pytorch.org/whl/cpu

python tools/sentiment_hub/hub.py --out feed.json
python -m pytest -q tools/sentiment_hub/tests
```

```
hub.py --out PATH [--models vader,finbert,cryptobert] [--hours 24] [--assets PATH] [--channels PATH]
       [--lexicon PATH] [--max-channels N] [--no-reddit] [--no-rss] [--no-telegram] [-v]
```

| option | default |
|---|---|
| `--models` | `vader,finbert,cryptobert`; VADER always runs. Missing transformers/torch or a failed model download is logged and the remaining scorers continue |
| `--hours` | `24`: the current window. Messages `hours..2*hours` old only count as `mentionsPrev`; Telegram/RSS/Reddit are read for `2*hours` (48 h) |
| `--assets` | `core/src/main/resources/bitpanda/bitpanda_snapshot.json` (built-in top 25 when missing) |
| `--channels` | `core/src/main/resources/telegram/channels.json` (5 built-in channels when missing) |
| `--lexicon` | `core/src/main/resources/sentiment/crypto_lexicon.tsv` (plain VADER when missing) |
| `--max-channels` | all entries; `N` keeps the N highest-confidence ones |

Logs go to stderr; the last line is a one-line summary. Exit code 0 whenever a valid feed was
written (also when every source failed), 1 on an internal error (nothing is written then), 2 on
bad arguments.

## Sources

Each source returns its items plus one status: `ok`, `empty`, `rate_limited`, `unauthorized`,
`unavailable`, `disabled` or `error`. One failing source never fails the run.

| source | what | notes |
|---|---|---|
| `telegram_public` | every entry of `channels.json`, `https://t.me/s/<handle>` | BeautifulSoup parses `div.tgme_widget_message[data-post]`; paginates with `?before=<smallest id>` until 48 h are covered or 5 pages; 1 request/s for the host, 4 channels in flight, browser User-Agent; 404s and redirects to `t.me/<handle>` (groups have no preview) are skipped; 12 min budget. `detail` = `readable/attempted channels (...)` |
| `telegram_account` | your groups/supergroups/channels + public catalogue handles | see below; `disabled` without secrets |
| `rss` | CoinTelegraph, CoinDesk, Decrypt, Bitcoin Magazine, CryptoSlate, The Block | `xml.etree` only; HTML entities such as `&nbsp;` are repaired; undated items are skipped |
| `reddit` | r/CryptoCurrency, r/CryptoMarkets, r/altcoin, r/SatoshiStreetBets `new.json?limit=100` | descriptive User-Agent; Reddit often answers 403/429 to cloud IPs, which is recorded and skipped |

A post that both Telegram sources read is counted once (same key `tg:<handle>/<id>`); identical
texts within one community (bot spam) are counted once.

### Your own Telegram account (optional)

Reading with your account reaches public **groups** (no web preview) and the groups/channels you
are a member of.

1. Get `api_id` and `api_hash` at <https://my.telegram.org> -> *API development tools*.
2. Run once on your computer: `pip install telethon && python tools/sentiment_hub/login.py`
   (phone number, login code, 2FA password if enabled). It prints a `StringSession` and the exact
   steps; `--out session.txt` writes it to a mode-600 file instead.
3. Add the repository secrets `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_SESSION` at
   <https://github.com/medonumbworks/spss/settings/secrets/actions>, or
   `gh secret set TELEGRAM_SESSION --repo medonumbworks/spss < session.txt`.
4. Optional repository *variable* `TELEGRAM_AUTO_JOIN=1`: join up to 5 public `kind=group`
   catalogue entries per run (highest confidence first). Without it nothing is ever joined.

**The session string is a full login to your account. Never commit or share it.** Revoke it in
Telegram: *Settings -> Devices -> CryptoPulse Sentiment Hub -> Terminate*. `login.py --check`
verifies the three variables.

What the account source reads:

* every group, supergroup and channel dialog (1:1 chats, bots and Saved Messages are never
  opened), up to 300 dialogs, messages of the last 48 h, at most 400 per chat;
* every public handle of `channels.json` that is not a dialog, by username and **without
  joining**. Handles the public web preview already read in the same run are skipped (the posts
  would be duplicates, and every username lookup counts against Telegram's daily resolve limit);
  at most 60 lookups per run;
* `FloodWaitError`: waits when <= 60 s and retries once, otherwise skips the chat (a long flood
  wait on a username lookup stops further lookups for this run); 13 min budget.

**Privacy.** A chat without a public username is private. Its messages only feed the aggregate
numbers (`mentions`, `composite`, shares, `bySource.telegram_account`). They are never used as
samples, and the titles, ids and texts of private chats are never logged or written anywhere.
Workflow logs of a public repository are public: the hub logs private chats only as counts.

### Sending yourself a report

`python tools/sentiment_hub/notify.py report.md [--title "..."] [--dry-run]` sends a text file to
your *Saved Messages* (with the three `TELEGRAM_*` variables) or through a bot
(`TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`). Longer texts are split into numbered parts of at most
4000 characters. Without configuration it says so and exits 0.

## Scoring

| scorer | score in -1..1 | notes |
|---|---|---|
| `vader` | VADER compound | `vaderSentiment==3.3.2`, lexicon updated with the crypto overlay, text normalized first (below) |
| `finbert` | `p(positive) - p(negative)` | `ProsusAI/finbert` |
| `cryptobert` | `p(Bullish) - p(Bearish)` | `ElKulako/cryptobert` |

Transformers run on CPU with batch size 32 and truncation to 128 tokens; their input is the text
with URLs removed. Per message, `composite` = weighted mean with cryptobert 0.45, vader 0.30,
finbert 0.25, renormalized over the scores that exist. Label: `composite >= 0.10` bullish,
`<= -0.10` bearish, else neutral. Duplicate texts are scored once. To keep the hourly job far
below its 40 minute limit, each transformer model scores at most 15000 distinct texts and stops
after 7 minutes, heaviest-weighted messages first; VADER always scores every text and the
composite of the remaining messages is renormalized over the scores they have.

### Crypto overlay and normalizer (must match the Kotlin implementation)

`crypto_lexicon.tsv`: lines starting with `#` are comments; `term<TAB>valence<TAB>comment`;
terms are lowercased; a term with spaces is a phrase stored under its underscore key
(`to the moon` -> `to_the_moon`); an emoji line is `EMOJI<TAB>valence<TAB>emoji_<name>` and stores
the valence under `emoji_<name>`. Before VADER, in this order:

1. URLs `https?://\S+|www\.\S+` -> one space
2. `[$#]` immediately followed by an ASCII letter is removed (`$BTC` -> `BTC`, `$100` stays)
3. phrases, longest first (ties alphabetical), each applied to the whole text before the next:
   case-insensitive, whole word where a boundary is a character outside `[A-Za-z0-9]` or the
   text edge, replaced by the underscore key. Words must be separated by exactly one space
   here, because whitespace is only collapsed in step 5
4. mapped emoji (longest sequence first) -> ` emoji_<name> `
5. whitespace runs -> one space, trim (Unicode whitespace, as Python's `\s`)

`tests/test_lexicon.py` holds the reference cases.

## Aggregation

* current = younger than `windowHours`; `mentionsPrev` = mentions `windowHours..2*windowHours` old
* weight = `0.5^(age_h / 12) * (1 + log10(1 + views_or_score))`; RSS items have engagement 0
* per asset: `mentions` = number of matching messages; `composite`, `scores.*`, `bullish`,
  `bearish`, `neutral` are weighted; `bySource` the same per source
* mention matching is a port of `core/.../repo/MentionMatcher.kt`: `$SYM` always, the full name
  (and the first word of multi-word names) case-insensitively, the bare symbol case-sensitively
  only when it is not ambiguous (length <= 2 or in `AMBIGUOUS_SYMBOLS`); `test_mentions.py`
  compares the symbol lists with the Kotlin source
* `market` = all scored messages of the current window, whether or not they name an asset
* up to 5 samples per asset, public sources only, strongest `weight * (0.25 + |composite|)` first

## Feed `cryptopulse.sentiment-feed/v1`

```json
{
 "schema": "cryptopulse.sentiment-feed/v1",
 "generatedAt": "2026-10-19T10:00:00Z",
 "windowHours": 24,
 "models": ["vader", "finbert", "cryptobert"],
 "sources": [
  {"name": "telegram_public", "status": "ok", "items": 1530, "detail": "80/87 channels (5 no_preview, 2 unavailable)"},
  {"name": "telegram_account", "status": "disabled", "items": 0, "detail": "no TELEGRAM_SESSION secret"},
  {"name": "rss", "status": "ok", "items": 120, "detail": "6/6 feeds"},
  {"name": "reddit", "status": "unauthorized", "items": 0, "detail": "0/4 subreddits (4 HTTP 403)"}
 ],
 "market": {"mentions": 1650, "composite": 0.12, "bullish": 0.41, "bearish": 0.22, "neutral": 0.37},
 "assets": [
  {"symbol": "BTC", "mentions": 120, "mentionsPrev": 95, "composite": 0.21,
   "scores": {"vader": 0.18, "finbert": 0.25, "cryptobert": 0.30},
   "bullish": 0.45, "bearish": 0.15, "neutral": 0.40,
   "bySource": {"telegram_public": {"mentions": 80, "composite": 0.2}},
   "samples": [{"text": "...", "score": 0.5, "source": "telegram_public", "community": "cointelegraph",
                "url": "https://t.me/cointelegraph/123", "ts": "2026-10-19T09:12:00Z"}]}
 ]
}
```

Guarantees (checked by `feed.validate_feed`, which runs before every write):

* keys exactly as above and in this order; timestamps `YYYY-MM-DDTHH:MM:SSZ` (UTC)
* `sources` always has the four entries in this order; `items` = messages collected in the 48 h lookback
* `models` lists only scorers that actually produced scores, in the order vader, finbert,
  cryptobert; it is `[]` when there was nothing to score
* every float is rounded to 4 decimals; composites and scores in -1..1, shares in 0..1 and
  summing to 1 (+-0.001) when there are mentions, all 0 otherwise
* `assets`: only `mentions > 0`, sorted by `mentions` desc (ties: total weight); `scores` holds
  only models that scored this asset; `bySource` mentions add up to `mentions`
* `samples`: at most 5, never from a private chat, `text` 1..280 characters (counted as UTF-16
  units like Kotlin's `String.length`, cut texts end with `…`), `community` and `url` non-empty
* when every source fails the feed is still written: statuses filled in, `market` all zero,
  `assets: []`

## GitHub Actions

`.github/workflows/sentiment-hub.yml` runs at minute 17 of every hour, on manual dispatch and on
pushes that touch `tools/sentiment_hub/**` or the workflow (any branch except `sentiment-feed`).
It installs the requirements (CPU torch from `download.pytorch.org/whl/cpu`; if that install fails
the run continues with VADER only), caches pip and `~/.cache/huggingface`, runs the tests, runs the
hub with all models and the optional Telegram secrets, then force-pushes `feed.json` and a short
README as a single orphan commit to the branch `sentiment-feed` with the built-in `GITHUB_TOKEN`
(`permissions: contents: write`), so the repository does not grow. A failing run publishes
nothing and the previous feed stays online.

Note: GitHub pauses scheduled workflows of public repositories after 60 days without repository
activity; re-enable it in the Actions tab if that happens. `raw.githubusercontent.com` caches
files for about 5 minutes.

## Files

| file | role |
|---|---|
| `hub.py` | CLI: load data files, collect (sources in parallel), dedupe, score, aggregate, validate, write |
| `assets.py` | loaders for `bitpanda_snapshot.json` and `channels.json` (with built-in fallbacks) |
| `mentions.py` | port of `MentionMatcher.kt` |
| `lexicon.py` | crypto overlay loader and normalizer |
| `nlp.py` | VADER, FinBERT, CryptoBERT scorers, composite and labels |
| `telegram_public.py`, `telegram_account.py`, `rss.py`, `reddit.py` | sources |
| `aggregate.py` | weights, per-asset and market numbers, samples |
| `feed.py` | feed builder, validator, atomic writer |
| `net.py`, `common.py` | HTTP client with per-host rate limits, shared types |
| `notify.py`, `login.py` | Telegram report sender, one-time Telegram login |
| `tests/` | pytest suite with hand-written fixtures; no network, transformers always mocked |
