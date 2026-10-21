# CryptoPulse dashboard (PWA shell)

Static, build-free dashboard: `index.html` + `app.js` + `style.css` + `sw.js`, installable on
Android via Chrome's "Add to Home screen". It renders `data/dashboard.json`
(schema `cryptopulse.dashboard/v1`; the comment block at the top of `app.js` is the contract and
the single source of truth — the exporter in `mcp-server/.../Export.kt` must match it).

## How it is published

The `dashboard` workflow (`.github/workflows/dashboard.yml`) runs hourly:

1. builds the MCP server and runs `cryptopulse-mcp export` (one full refresh: prices, forecasts,
   Telegram buzz, Polymarket, pump warnings, journal accuracy, alert crossings),
2. copies this directory over it, with the fresh `data/dashboard.json` winning over the committed
   sample,
3. force-pushes the result as a single commit to the `pwa` branch and, when GitHub Pages is
   enabled, deploys the same files to https://medonumbworks.github.io/spss/,
4. optionally sends the fresh alert crossings to Telegram (`TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`).

The committed `data/dashboard.json` is a hand-written sample so the shell can be developed and
validated offline; the published site always carries a real export.

## Develop and validate locally

```bash
python3 -m http.server -d web 8080   # then open http://localhost:8080
node web/tools/validate.js           # headless end-to-end check (needs playwright + Chromium)
```

When the loaded data is older than 2 h the Telegram tab also refreshes live from the public
sentiment feed; older than 3 h turns the "updated" chip amber. Information only, not financial advice.
