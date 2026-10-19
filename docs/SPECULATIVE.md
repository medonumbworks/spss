# Memecoins, trending coins and the DEX radar

Besides every coin Bitpanda lists, CryptoPulse shows three extra tiers. They are **information
only**: none of these coins can be bought on Bitpanda, and the app never links to a place to buy.

| Tier | What | Source (free, keyless) | Forecast |
|---|---|---|---|
| `MEME` | the 250 largest memecoins by market cap that Bitpanda does not list | CoinGecko `/coins/markets?category=meme-token` (with 7-day sparklines) | same model as Bitpanda coins, with a risk shrink |
| `TRENDING` | CoinGecko's trending coins that Bitpanda does not list | CoinGecko `/search/trending` + `/coins/markets?ids=` | same, with a risk shrink and a small "cools off" prior |
| `DEX` | tokens that trade only on decentralised exchanges | GeckoTerminal trending and new pools, DexScreener boosts / profiles / community takeovers, pump.fun top and live coins; one DexScreener lookup (30 tokens per call) gives every token the same fields | trading-flow model, gated by pool age and contract checks |

Bitpanda coins that are memecoins (DOGE, PEPE, WIF, ...) stay Bitpanda rows and get a `meme` tag, so
the "Memecoins" filter shows them too.

## How the radar is built

1. Candidates: 2 pages of GeckoTerminal trending pools, 1 page of new pools (optional),
   DexScreener promotions, pump.fun (optional).
2. Each candidate is looked up on DexScreener (deepest pool wins; GeckoTerminal's unique buyer and
   seller counts are kept).
3. Left out: pool quote assets and stablecoins; tokens CoinGecko already maps to a listed coin (that
   row gets a `dex_trending` tag instead); **tokens with the same ticker as a listed coin** (almost
   always copycats); pools under $10k liquidity or $10k 24 h volume (both adjustable).
4. At most 150 tokens, most traded first. Prices are converted from USD with the Bitpanda ticker's
   BTC EUR/USD ratio.
5. Contract checks: GoPlus (EVM chains and Solana) for the most active tokens, one address per call
   (GoPlus silently answers only the first address of a list), at most 12 new checks per refresh;
   checks are cached for 6 hours, so coverage grows over a few refreshes. Opening a token re-checks it
   and adds the honeypot.is buy/sell simulation (Ethereum, BSC, Base) and a week of hourly candles from
   GeckoTerminal.

## Base rates (why the model is pessimistic)

* About 69% of pump.fun tokens made their last trade on launch day, more than 80% were dead within
  two days, and about 4.6% still traded after 90 days (CoinGecko, 18.7M pump.fun tokens, 2024-2026).
* About 1% of pump.fun launches ever graduate to a real pool (Dune dashboards, 2024-2026).
* 82-83% of memecoins that more than doubled showed artificial growth such as wash trading
  (Mongardini & Mei, USENIX Security 2026, 34,988 memecoins on ETH, BSC, Solana, Base).
* 53% of the ~20M tokens listed on GeckoTerminal since 2021 are dead (CoinGecko, end of 2025).

## Risk score (0-100, higher = riskier)

`core/src/main/kotlin/dev/cryptopulse/core/analysis/risk/RiskScorer.kt`, shown with its reasons.

**No "up" probability at all** (label in brackets):

* honeypot, sell or buy tax of 20% or more, someone can change balances, block selling, freeze holder
  accounts, a live mint authority or owner backdoors (hidden owner, take back ownership,
  self-destruct) unless the token is listed on a big exchange or marked trusted (**Red flags**, score 100);
* pool younger than 24 hours, or still on the pump.fun bonding curve (**Too new**);
* pool liquidity under $10k (**Too illiquid**);
* a DEX token whose contract has not been checked (**Unverified**; fail closed);
* any score of 75 or more (**Extreme risk**).

Points (selection): liquidity under $25k / $50k / $100k / $250k (+30/+20/+10/+5), pool age under
3 days / 7 days / 30 days / 90 days (+25/+15/+8/+3), pool under 2% / 5% of the market cap (+15/+8),
24 h volume over 50x the pool (+10, often wash trading), +300% in a day (+10), tax 10-20% / 5-10%
(+20/+10), unpublished source code (+20), upgradeable proxy (+10), owner switches on an owned contract
(+5 to +15), creator holding over 10% (+10), little locked liquidity (+10), no website or socials
(+5). Credits: listed on a big exchange (-15), GoPlus trusted (-20), older than 180 days with over
$1M liquidity (-10). Paid DexScreener boosts are shown as a badge and only count (+5) on very
promoted tokens younger than 3 days. Bands: under 25 lower, 25-49 elevated, 50-74 high, 75+ extreme.

## Forecast changes for these tiers

* The model's probability is pulled toward 50% in proportion to the risk:
  `p = 0.5 + (p_model - 0.5) * (1 - score / 75)`.
* DEX tokens start from a pessimistic prior (24 h: -0.10 logit, 7 d: -0.28, more for pools younger
  than 3 days), are clamped to 20-80% before the risk shrink, and need **3 days of trading for a 24 h
  call and 3 weeks for a 7-day call**; younger pools show "not enough history".
* DEX trading flow: last-hour momentum (small), give-back after very large 24 h pumps, unique buyers
  vs sellers over 6 h (GeckoTerminal) or trade counts, and volume acceleration in the direction of the
  last hour. All weights are small; these signals are noisy and often manipulated.
* A token's own Telegram channel (from its DexScreener links) is read for sentiment when the token is
  opened; DEX tickers only count as mentions with a `$` cashtag or the full name.

## Cloud hub

The hourly sentiment hub (`tools/sentiment_hub`) also counts mentions of the 250 largest memecoins
(one CoinGecko call per run) and of new `$TICKER` cashtags that at least two communities use three
times or more (tracked by cashtag only), so DEX tokens can get hub sentiment too.

## Attribution and terms

Data from CoinGecko, GeckoTerminal, DexScreener, pump.fun, GoPlus Security, honeypot.is, Coinpaprika
and Binance's public market data. All are used through their public keyless endpoints, within the
documented rate limits (see `RateLimiter.DEFAULT_RULES`), for display only (no resale). The endpoints
and their answers are captured from GitHub's network by `.github/workflows/api-samples.yml`
(`tools/api_samples/urls.txt`) so the test fixtures follow real payloads.
