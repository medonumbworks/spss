# How the forecast works

CryptoPulse gives two numbers per coin: the probability that the price is **higher in 24 hours** and
**higher in 7 days**, plus a 0-100 **bottom score** ("how close is it to a local low, and are rebound
conditions present?"). Everything is computed on the device (or in the MCP server) from free data.

## Inputs

| Group | Signals | Source |
|-------|---------|--------|
| Price (24h horizon) | RSI 14, Bollinger %B, last 24h move vs. volatility, 12h vs 48h trend, MACD histogram | CoinGecko 7-day hourly sparkline |
| Price (7d horizon) | daily RSI 14, %B, 30-day momentum, 20 vs 50-day trend, last 7d move | CoinGecko 1-year daily history (detail screen) or 4-hour bars |
| Market context | Bitcoin trend (weighted by how closely the coin follows BTC), Fear & Greed (contrarian), total market cap move, overall social mood, CoinGecko trending | CoinGecko, alternative.me, Reddit/news/Telegram |
| Crowd | coin sentiment x confidence, attention change, futures funding (contrarian), long/short accounts (contrarian), CoinGecko votes | Reddit, StockTwits, Telegram, news, cloud hub, Binance/Bybit, CoinGecko |

Each signal adds a small signed amount to a logit. Weights are deliberately small because short-term
crypto returns are mostly noise.

## Self-check (walk-forward backtest)

The price-only part is replayed over the available history: at every past step the model sees only
the data up to that step and its call is compared with what happened next. The baseline is chance:
a guesser with no skill that says "up" as often as the model did (so merely following the overall
trend earns no credit). Past windows overlap, so only roughly independent windows count
(samples x step / horizon); with fewer than 10 the weight stays neutral. The weight is cut when the
model lost to chance by more than one standard error, cut more beyond two, and raised a little only
when it won by one or two standard errors. The detail screen shows the hit rate, the independent
window count and the chance level.

## Humility rules

* Probabilities are shrunk toward 50 % when data coverage is thin and clamped to 15-85 %.
* "Likely up/down" needs at least 55 % / at most 45 %; everything in between is "no clear direction".
* Stablecoins (USDT, USDC, EURC, ...) are reported as pegged, without a forecast. Unknown ones are
  recognised by a flat price; in EUR/GBP/CHF a USD peg moves with the exchange rate, so there the
  name or symbol must also say it is pegged ("USD", "Dollar", "Stable", ...).
* Futures funding is converted to an 8-hour rate first (many contracts settle every 4 h or 1 h).
* The likely range is about one standard deviation of recent volatility around a small drift.
* **Too small to trade**: when a coin's typical move over the horizon (the standard deviation of the
  last week's hourly returns, scaled to 24 h or 7 days) is under **1 %**, about the cost of buying
  and selling once, there is no call ("Moves too little"); the drivers stay visible with zero weight.
  Example: TRON moves about 0.8-0.9 % a day in EUR, so it gets no 24 h call while its 7-day call
  stays; Bitcoin runs about 1.6 % and XRP about 3 % on the same measure.
* **Bitcoin's trend counts by how much a coin follows it**: the Bitcoin-trend weight is multiplied by
  0.3 + 0.7 x beta, where beta compares the coin's hourly returns with Bitcoin's over the last week
  (clamped to 0-3). A coin with beta 0.1 (TRON) gets about a third of the weight, one with beta 2.9
  (NEAR) about 2.3 times. Without Bitcoin's hourly series the plain weight applies.

## Realized accuracy (forecast journal)

Every call shown is logged with its price and likely range and scored once its horizon passes. A
move smaller than half the typical move (a quarter of the likely range, which spans one typical move
either way) is **neutral**: noise, neither a hit nor a miss, and counted
separately. Horizons without a call (too new, red flags, moves too little) are not logged; a 50 % call
is priced but never scored. The hit rate is compared with 50 % and with the share of coins that
simply rose.

## Bottom score

Weighted mix of: distance to the 7-day, 30-day, 90-day and 1-year lows, RSI, Bollinger %B, drawdown
from the 30-day high (only with daily history) and market-wide fear. A coin without enough price
history of its own gets a neutral 50 ("Not enough price history"): market-wide fear may tilt a
score, never make one. 70+ = near a local low, 55-70 lower part of the range,
40-55 mid-range, 25-40 upper part, below 25 near a local high. Nobody knows the absolute bottom; the
score only says whether the price is near recent lows *and* typical rebound conditions are present.
The 7-day probability next to it is the actual estimate of whether it goes up from here.

Model version: `cryptopulse-forecast-1.2` (see `core/src/main/kotlin/dev/cryptopulse/core/analysis/forecast`).
