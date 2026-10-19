# CryptoPulse

Android app and MCP server that list **every cryptocurrency tradeable on Bitpanda** with live prices,
the **probability of going up in the next 24 hours and 7 days**, a **"near the bottom?" score**, and
**social sentiment** from Reddit, StockTwits, Telegram and news. Everything uses **free** sources and
free compute; no account or API key is required.

> Probabilities, not certainties. Not financial advice. Nobody can know the lowest possible price in advance.

## Get the app

1. Open the latest build on your phone:
   **https://github.com/medonumbworks/spss/releases/tag/apk-latest** and download `CryptoPulse.apk`.
2. Allow your browser to install apps when Android asks, then open CryptoPulse.
3. Newer builds install over the old one (same signing key).

Every push to the repository builds and tests everything with GitHub Actions
(`.github/workflows/build.yml`) and replaces that release.

## Dashboard (PWA)

A lightweight web dashboard with the same data, no install needed: every coin with its 24h/7d
up-probability and pump warnings, the Telegram buzz ranking, Polymarket odds and the realised
accuracy of past forecasts. The `dashboard` workflow (`.github/workflows/dashboard.yml`)
recomputes everything hourly (`cryptopulse-mcp export`, schema `cryptopulse.dashboard/v1`
documented in `web/app.js`) and publishes the static site from `web/`:

* **https://medonumbworks.github.io/spss/** — once GitHub Pages is on (the workflow tries to
  enable it itself; otherwise flip Settings → Pages → Source to "GitHub Actions" once).
* The [`pwa` branch](https://github.com/medonumbworks/spss/tree/pwa) always holds the latest
  published site as plain files, whatever the Pages setting.

Install it on Android like an app: open the URL in Chrome → menu (⋮) → **Add to Home screen** →
Install. It keeps working offline with the last downloaded data. Like the sentiment hub, GitHub
runs the hourly schedule only on the default branch; until then run it from the Actions tab.

When a Bitpanda coin's 24 h chance of rising crosses the 50 % threshold, the hourly run can also
send a short Telegram message. It reuses the `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` secrets
of the `telegram-digest` workflow; without them that step just logs that it skipped. As with
everything here: information only, probabilities, not financial advice.

## What you see

* **Market list**: all Bitpanda coins sorted by market cap (like the Bitpanda app), with price,
  24h change, a 7-day sparkline, `24h ▲ 58%` / `7d ▼ 44%` chips, a bottom meter and a sentiment dot.
  Search, filters (likely up 24h/7d, near a low, gainers, losers, bullish chatter) and sorting.
* **Coin detail**: 7D/90D/1Y chart with the likely 24h range, both forecasts with confidence, likely
  price range, the drivers behind them and a backtest, the bottom assessment, sentiment per source
  with example posts (tap to open), futures funding, long/short ratio, community votes.
* **Beyond Bitpanda (information only)**: the 250 largest memecoins, today's trending coins and a
  **DEX radar** of speculative tokens (GeckoTerminal trending/new pools, DexScreener boosts and
  takeovers, pump.fun), marked `MEME` / `TRENDING` / `DEX`, each with a 0-100 risk score, free
  contract checks (GoPlus, honeypot.is), pool liquidity, age and buyers vs sellers. Tokens with red
  flags, pools younger than a day or unchecked contracts get no "up" probability. Tier chips switch
  between All, Bitpanda, Memecoins, Trending and DEX radar. Details: [docs/SPECULATIVE.md](docs/SPECULATIVE.md).
* **Settings**: currency (EUR, USD, CHF, GBP), each source on/off, the speculative tiers on/off
  (and the minimum pool liquidity), how many Telegram channels the phone reads, the cloud hub,
  optional keys.

## Where the data comes from (all free)

| What | Source | Key needed |
|------|--------|-----------|
| List of Bitpanda cryptos and Bitpanda prices | `api.bitpanda.com/v1/ticker`, then `api.public.bitpanda.com/v1/assets`, then the bundled list of 268 coins (158 confirmed from the Bitpanda app) | no |
| Prices, market cap, history, community votes, trending | CoinGecko public API | no (optional free demo key) |
| Market mood | alternative.me Crypto Fear & Greed | no |
| Futures positioning | Binance / Bybit public endpoints | no |
| Social | Reddit JSON, StockTwits streams, Telegram public channels (`t.me/s/...`) | no |
| News | Google News RSS, CoinTelegraph, CoinDesk, Decrypt, Bitcoin Magazine | no |
| Heavy NLP + Telegram account | the **cloud sentiment hub** on GitHub Actions (free for public repos) | no |

Quora has no public API and forbids scraping, so it is not used.

## Sentiment: VADER on the phone, FinBERT and CryptoBERT in the cloud

* **VADER** (rule/lexicon based) is ported to Kotlin and runs on the phone, extended with a crypto
  slang lexicon (moon, rekt, rug pull, HODL, buy the dip, 🚀, ...). It needs no model download and
  works offline. See [docs/SENTIMENT.md](docs/SENTIMENT.md).
* **FinBERT** (financial news) and **CryptoBERT** (crypto social posts) are transformer models of
  110-500 MB. They run in the free **cloud sentiment hub** (`tools/sentiment_hub`, GitHub Actions,
  hourly), which reads every channel in the Telegram directory, the news feeds and Reddit, scores
  each text with all three models and publishes one small JSON file that the app downloads:
  https://raw.githubusercontent.com/medonumbworks/spss/sentiment-feed/feed.json
* While that feed is fresh the phone does not rescan the same Telegram channels and RSS feeds (the
  posts would count twice). An older feed counts less (half weight every 6 h after the first 3 h)
  and after 48 h it is ignored, so the app never runs on stale chatter.
* GitHub runs the hourly schedule only for workflows on the repository's default branch; until this
  branch is merged, the hub runs on pushes to `tools/sentiment_hub/**` or from the Actions tab
  ("Run workflow").

## Telegram

1. **Public channels, no login**: the phone reads the most important public crypto channels through
   Telegram's web preview; the cloud hub reads every channel in `core/src/main/resources/telegram/channels.json`.
2. **Your own Telegram account** (optional, still free): add three secrets to this GitHub repository
   and the hub also reads every group and channel you are a member of, plus public groups by name.
   Only per-coin totals are published, never texts or names of private chats.
   Step by step: [tools/sentiment_hub/README.md](tools/sentiment_hub/README.md).
3. **Digest to your Telegram**: the `telegram-digest` workflow can send a short briefing to your
   Saved Messages (or a bot chat) a few times a day.

## MCP server

The same engine is available to Claude Desktop, Claude Code, Cursor and other MCP clients.

Download `cryptopulse-mcp.zip` from the release (needs Java 17+), unzip, and add it to your client:

```json
{
  "mcpServers": {
    "cryptopulse": {
      "command": "/path/to/cryptopulse-mcp/bin/cryptopulse-mcp"
    }
  }
}
```

Or run it over HTTP: `cryptopulse-mcp --http 3333` and connect to `http://127.0.0.1:3333/mcp`.
`cryptopulse-mcp check` refreshes once from the live APIs and prints a health report (CI runs it on
every push and shows it in the job summary); `cryptopulse-mcp digest hr` prints the Croatian digest.

Tools: `market_overview` (with `tier`: all, bitpanda, meme, trending, dex, speculative),
`speculative_radar` (memecoins and DEX tokens with liquidity, pool age, contract checks and risk),
`get_forecast`, `get_sentiment`, `find_opportunities`, `get_prices`,
`list_bitpanda_assets`, `market_mood`, `telegram_sources`, `daily_digest` (English or Croatian), and
the `crypto_briefing` prompt. Optional environment variables: `CRYPTOPULSE_CURRENCY`,
`COINGECKO_DEMO_KEY`, `BITPANDA_API_KEY`, `CRYPTOPULSE_FEED_URL`, `CRYPTOPULSE_CACHE_DIR`.

## Forecast model

Explained in [docs/FORECAST.md](docs/FORECAST.md): technical signals, market context and crowd signals
combined into a logit, self-checked with a walk-forward backtest, shrunk toward 50 % when data is thin.

## Build it yourself

```bash
./gradlew :app:assembleDebug                                   # Android APK (needs the Android SDK)
./gradlew :core:test :mcp-server:test -Pcryptopulse.skipAndroid=true
./gradlew :mcp-server:installDist -Pcryptopulse.skipAndroid=true  # MCP server in mcp-server/build/install
```

Project layout: `core` (shared Kotlin: sources, sentiment, forecast), `app` (Android, Jetpack Compose),
`mcp-server` (Kotlin MCP SDK), `tools/sentiment_hub` (Python, GitHub Actions), `docs`.

The older Streamlit prototype (`spss8.py`) is kept as it was.

---

## Ukratko na hrvatskom

* **Aplikacija**: na mobitelu otvori https://github.com/medonumbworks/spss/releases/tag/apk-latest,
  preuzmi `CryptoPulse.apk` i instaliraj (dopusti instalaciju iz preglednika).
* **Nadzorna ploča (PWA)**: https://medonumbworks.github.io/spss/ (rezerva: grana `pwa`) — satno
  osvježen pregled svih coinova, Telegram buzz, Polymarket i točnost prošlih prognoza u
  pregledniku. Na Androidu: Chrome → izbornik (⋮) → "Dodaj na početni zaslon". Samo informativno.
* Prikazuje sve kriptovalute s Bitpande, cijene, **vjerojatnost rasta u 24 h i 7 dana**, ocjenu
  "blizu dna" i sentiment s Reddita, StockTwitsa, Telegrama i vijesti. Sve je besplatno, bez ključeva.
* **VADER** radi na mobitelu (s rječnikom kripto slenga), **FinBERT i CryptoBERT** rade besplatno
  u oblaku (GitHub Actions) i šalju rezultat aplikaciji.
* **Telegram**: javni kanali rade odmah. Za tvoj Telegram račun (sve grupe u kojima jesi) dodaj tri
  tajne u GitHub repozitorij prema uputama u [docs/TELEGRAM_HR.md](docs/TELEGRAM_HR.md). Sažetak
  tržišta može stizati u tvoje "Saved Messages" na hrvatskom.
* **Satni raspored** oblaka GitHub pokreće samo s glavne (default) grane. Dok ova grana nije spojena,
  oblak se pokreće ručno: Actions → `sentiment-hub` → "Run workflow". Ako oblak zastane, aplikacija
  stare podatke postupno zanemaruje (nakon 48 h potpuno) i sama čita Telegram.
* **Memecoini i špekulativni tokeni**: uz Bitpandu aplikacija prikazuje 250 najvećih memecoina,
  coinove koji su danas u trendu i **DEX radar** (novi i popularni tokeni s decentraliziranih burzi,
  pump.fun). Označeni su s `MEME` / `TRENDING` / `DEX`, nisu na Bitpandi i služe samo za
  informaciju. Svaki ima ocjenu rizika 0-100 i besplatnu provjeru ugovora (honeypot, porezi,
  zamrzavanje, mintanje). Bez prognoze ostaju tokeni s crvenim zastavicama, mlađi od 24 h ili
  neprovjereni. Većina takvih tokena izgubi većinu vrijednosti.
* Ovo su vjerojatnosti, ne sigurnosti, i nije financijski savjet.
