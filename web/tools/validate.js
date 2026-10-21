/* End-to-end check of the PWA shell against the shipped sample data.
 *
 * Serves web/ itself on an ephemeral port and drives headless Chromium through all
 * four tabs at a 412x915 phone viewport plus one wide pass. Fails on any console
 * error, page error, failed request, horizontal scroll or missing content, and also
 * exercises the three degraded paths: stale data + live feed merge, a minimal payload
 * with missing optional fields, and a dead data endpoint.
 *
 * Run from the repo root:  node web/tools/validate.js
 * Needs the playwright package and a Chromium (PLAYWRIGHT_BROWSERS_PATH as usual).
 * Set SHOTS_DIR=/some/dir to also save screenshots of every pass.
 */
'use strict';
const fs = require('fs');
const http = require('http');
const path = require('path');
const { chromium } = require('playwright');

const WEB_DIR = path.resolve(__dirname, '..');
const DASH_JSON = path.join(WEB_DIR, 'data', 'dashboard.json');
// a real captured feed payload; the Kotlin client tests pin the same file
const FEED_JSON = path.resolve(WEB_DIR, '..', 'core', 'src', 'test', 'resources', 'fixtures', 'sentiment_feed.json');
const FEED_URL = 'https://raw.githubusercontent.com/medonumbworks/spss/sentiment-feed/feed.json';
const VIEWPORT = { width: 412, height: 915 };
const SHOTS = process.env.SHOTS_DIR || '';

const MIME = {
  '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.json': 'application/json',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.webmanifest': 'application/manifest+json',
};

function serveWebDir() {
  const server = http.createServer((req, res) => {
    const rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    const file = path.join(WEB_DIR, rel === '/' ? 'index.html' : rel.replace(/^\/+/, ''));
    if (!file.startsWith(WEB_DIR) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404); res.end('not found'); return;
    }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    // the committed sample ages past the 2 h live-refresh trigger within hours of being written; the
    // main pass must stay hermetic, so the server rejuvenates it (the stale passes route their own copy)
    if (file === DASH_JSON) {
      const fresh = JSON.parse(fs.readFileSync(file, 'utf8'));
      fresh.generatedAt = new Date().toISOString().replace(/\.\d+Z$/, 'Z');
      res.end(JSON.stringify(fresh));
      return;
    }
    res.end(fs.readFileSync(file));
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server)));
}

const problems = [];

function watch(page, label) {
  page.on('console', (msg) => {
    if (msg.type() === 'error' || msg.type() === 'warning') {
      if (msg.text().includes('blocked by Playwright')) return; // we block SW in interception passes
      problems.push(`[${label}] console.${msg.type()}: ${msg.text()}`);
    }
  });
  page.on('pageerror', (err) => problems.push(`[${label}] pageerror: ${err.message}`));
  page.on('requestfailed', (req) => problems.push(`[${label}] requestfailed: ${req.url()} ${req.failure() && req.failure().errorText}`));
}

async function assertNoHScroll(page, label) {
  const over = await page.evaluate(() => {
    const d = document.documentElement;
    return Math.max(d.scrollWidth - d.clientWidth, document.body.scrollWidth - d.clientWidth);
  });
  if (over > 0) problems.push(`[${label}] horizontal scroll: ${over}px overflow`);
}

async function shot(page, name) {
  if (SHOTS) await page.screenshot({ path: path.join(SHOTS, name + '.png') });
}

(async () => {
  if (SHOTS) fs.mkdirSync(SHOTS, { recursive: true });
  const server = await serveWebDir();
  const BASE = 'http://127.0.0.1:' + server.address().port;
  const browser = await chromium.launch();

  // ---- main pass: the shipped sample, service worker allowed ----
  const ctx = await browser.newContext({ viewport: VIEWPORT });
  let page = await ctx.newPage();
  watch(page, 'main');
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForSelector('#rows tr');

  if ((await page.locator('#rows tr').count()) !== 10) problems.push('[main] expected 10 coin rows');
  const updated = await page.locator('#updated').textContent();
  if (!/updated .*ago|updated just now/.test(updated || '')) problems.push(`[main] updated chip wrong: ${updated}`);
  const fg = await page.locator('#fg').textContent();
  if (!(fg || '').includes('59') || !(fg || '').includes('Greed')) problems.push(`[main] fear&greed chip wrong: ${fg}`);

  // default sort is market cap desc
  const firstCoin = await page.locator('#rows tr .coin-name').first().textContent();
  if (firstCoin !== 'Bitcoin') problems.push(`[main] default sort should put Bitcoin first, got ${firstCoin}`);

  // null forecast cells carry the em dash and the noCall reason
  const moonText = await page.locator('#rows tr', { hasText: 'MOONCAT' }).textContent();
  if (!moonText.includes('—') || !moonText.includes('Unverified')) problems.push('[main] MOONCAT null cells wrong');

  // pump badge opens with its reasons
  const pump = page.locator('#rows .pump summary').first();
  if (await pump.count() === 0) problems.push('[main] no pump badge rendered');
  else {
    await pump.click();
    if ((await page.locator('#rows .pump[open] li').count()) < 2) problems.push('[main] pump reasons not shown');
  }

  // numeric sort both ways, nulls last either way
  await page.locator('#coins thead button[data-sort="chg24h"]').click();
  if ((await page.locator('#rows tr .coin-sym').first().textContent()) !== 'MOONCAT') problems.push('[main] chg24h desc wrong');
  await page.locator('#coins thead button[data-sort="chg24h"]').click();
  if ((await page.locator('#rows tr .coin-sym').first().textContent()) !== 'PEPE') problems.push('[main] chg24h asc wrong');
  await page.locator('#coins thead button[data-sort="chg7d"]').click();
  if ((await page.locator('#rows tr .coin-sym').last().textContent()) !== 'MOONCAT') problems.push('[main] null chg7d not last (desc)');
  await page.locator('#coins thead button[data-sort="chg7d"]').click();
  if ((await page.locator('#rows tr .coin-sym').last().textContent()) !== 'MOONCAT') problems.push('[main] null chg7d not last (asc)');
  // string sort starts ascending
  await page.locator('#coins thead button[data-sort="name"]').click();
  if ((await page.locator('#rows tr .coin-name').first().textContent()) !== 'Bitcoin') problems.push('[main] name sort wrong');

  // only All / Bitpanda / Trending: the memecoin and DEX tabs are gone
  const tierChips = await page.locator('#tiers button').evaluateAll((bs) => bs.map((b) => b.dataset.tier));
  if (JSON.stringify(tierChips) !== '["ALL","BITPANDA","TRENDING"]') problems.push(`[main] tier chips wrong: ${JSON.stringify(tierChips)}`);
  await page.locator('#tiers button[data-tier="BITPANDA"]').click();
  if ((await page.locator('#rows tr').count()) !== 6) problems.push('[main] Bitpanda filter wrong');
  await page.locator('#tiers button[data-tier="TRENDING"]').click();
  const trendSyms = await page.locator('#rows tr .coin-sym').allTextContents();
  if (trendSyms.length !== 4 || !trendSyms.includes('WIF') || !trendSyms.includes('MOONCAT')) {
    problems.push(`[main] Trending filter wrong: ${JSON.stringify(trendSyms)}`);
  }
  await page.locator('#tiers button[data-tier="ALL"]').click();

  await page.fill('#search', 'dogwif');
  if ((await page.locator('#rows tr').count()) !== 1) problems.push('[main] search wrong');
  await page.fill('#search', 'zzz');
  if (!(await page.locator('#rows-empty').isVisible())) problems.push('[main] search empty-state missing');
  await page.fill('#search', '');

  if ((await page.locator('#polymarket > li').count()) !== 3) problems.push('[main] expected 3 polymarket cards');
  if (!/noopener/.test(await page.locator('#polymarket a').first().getAttribute('rel') || '')) problems.push('[main] polymarket link missing noopener');
  const pmText = await page.locator('#polymarket > li').first().textContent();
  if (!pmText.includes('44% Up') || !pmText.includes('$1.2M')) problems.push(`[main] polymarket card wrong: ${pmText}`);

  await page.evaluate(() => window.scrollTo(0, 0));
  await assertNoHScroll(page, 'overview');
  await shot(page, 'tab-overview');

  await page.locator('.tabs button[data-tab="telegram"]').click();
  if ((await page.locator('#buzz > li').count()) !== 6) problems.push('[main] expected 6 buzz entries');
  const buzzTop = await page.locator('#buzz > li').first().textContent();
  if (!buzzTop.includes('WIF') || !buzzTop.includes('×5.2 vs 7-day norm')) problems.push('[main] top buzz should be WIF with spike');
  const buzzBtc = await page.locator('#buzz > li', { hasText: 'BinanceLiquidations' }).textContent();
  if (!buzzBtc.includes('prev: 69')) problems.push('[main] BTC buzz should fall back to "prev: 69"');
  const sampleLink = page.locator('#buzz .samples a').first();
  if ((await sampleLink.getAttribute('target')) !== '_blank') problems.push('[main] sample link not _blank');
  if (!/noopener/.test(await sampleLink.getAttribute('rel') || '')) problems.push('[main] sample link missing noopener');
  await assertNoHScroll(page, 'telegram');
  await shot(page, 'tab-telegram');

  await page.locator('.tabs button[data-tab="accuracy"]').click();
  if ((await page.locator('#accuracy .acc-card').count()) !== 2) problems.push('[main] expected 2 accuracy cards');
  const accText = await page.locator('#tab-accuracy').textContent();
  for (const needle of ['55%', '0.248', 'evaluated', 'pending', 'Brier', 'logged and scored against the realized price']) {
    if (!accText.includes(needle)) problems.push(`[main] accuracy missing "${needle}"`);
  }
  await assertNoHScroll(page, 'accuracy');
  await shot(page, 'tab-accuracy');

  await page.locator('.tabs button[data-tab="about"]').click();
  const about = await page.locator('#tab-about').textContent();
  for (const needle of ['not financial advice', 'CoinGecko', 'every hour', 'lose most of their value', 'rate limits']) {
    if (!about.includes(needle)) problems.push(`[main] about missing "${needle}"`);
  }
  await assertNoHScroll(page, 'about');
  await shot(page, 'tab-about');

  const swState = await page.evaluate(() => Promise.race([
    navigator.serviceWorker.ready.then((reg) => (reg && reg.active ? reg.active.state : 'none')),
    new Promise((resolve) => setTimeout(() => resolve('timeout'), 5000)),
  ]));
  if (swState !== 'activated' && swState !== 'activating') problems.push(`[main] service worker state: ${swState}`);
  await page.close();

  // ---- wide pass: the real table layout ----
  page = await ctx.newPage();
  await page.setViewportSize({ width: 920, height: 800 });
  watch(page, 'wide');
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForSelector('#rows tr');
  await assertNoHScroll(page, 'wide');
  await shot(page, 'wide-overview');
  await page.close();
  await ctx.close();

  // ---- interception passes: block the SW so page.route sees every request ----
  const ctx2 = await browser.newContext({ viewport: VIEWPORT, serviceWorkers: 'block' });

  // stale dashboard: amber chip, and the Telegram tab refreshes live from the feed
  page = await ctx2.newPage();
  watch(page, 'stale');
  const raw = JSON.parse(fs.readFileSync(DASH_JSON, 'utf8'));
  raw.generatedAt = new Date(Date.now() - 5 * 3600e3).toISOString().replace(/\.\d+Z$/, 'Z');
  const staleBody = JSON.stringify(raw);
  await page.route('**/data/dashboard.json', (route) => route.fulfill({ contentType: 'application/json', body: staleBody }));
  const feed = fs.readFileSync(FEED_JSON, 'utf8');
  await page.route(FEED_URL, (route) => route.fulfill({ contentType: 'application/json', body: feed }));
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForSelector('#rows tr');
  if (!/warn/.test(await page.locator('#updated').getAttribute('class') || '')) problems.push('[stale] updated chip should be amber past 3 h');
  await page.locator('.tabs button[data-tab="telegram"]').click();
  await page.waitForFunction(() => document.getElementById('buzz-meta').textContent.includes('refreshed live'));
  // the fixture carries BTC 300 / SOL 120, which must replace the sample's 251 / 96
  // (match on the symbol heading; sample text can mention other coins)
  const btcBuzz = await page.locator('#buzz > li:has(.buzz-sym:text-is("BTC"))').textContent();
  if (!btcBuzz.includes('300 mentions')) problems.push(`[stale] BTC buzz should show live 300 mentions, got: ${btcBuzz.slice(0, 80)}`);
  const solBuzz = await page.locator('#buzz > li:has(.buzz-sym:text-is("SOL"))').textContent();
  if (!solBuzz.includes('120 mentions')) problems.push(`[stale] SOL buzz should show live 120 mentions, got: ${solBuzz.slice(0, 80)}`);
  await shot(page, 'stale-telegram');
  await page.close();

  // stale dashboard with the feed unreachable: a quiet note, no crash
  page = await ctx2.newPage();
  page.on('pageerror', (err) => problems.push(`[stalefail] pageerror: ${err.message}`));
  await page.route('**/data/dashboard.json', (route) => route.fulfill({ contentType: 'application/json', body: staleBody }));
  await page.route(FEED_URL, (route) => route.abort());
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForSelector('#note:not([hidden])');
  if (!(await page.locator('#note').textContent()).includes('live Telegram refresh failed')) problems.push('[stalefail] note wrong');
  await page.close();

  // minimal payload: missing optional fields and unknown extras must not crash
  page = await ctx2.newPage();
  watch(page, 'minimal');
  const minimal = JSON.stringify({
    schema: 'cryptopulse.dashboard/v1',
    generatedAt: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
    futureField: { nested: [1, 2, 3] },
    rows: [
      { symbol: 'XYZ', name: 'Mystery', tier: 'WEIRD_NEW_TIER', unknownField: true },
      { symbol: 'ABC' },
      // an export from before the memecoin and DEX tiers were removed: those rows stay hidden
      { symbol: 'OLDMEME', tier: 'MEME' },
      { symbol: 'OLDDEX', tier: 'DEX', currency: 'USD' },
    ],
  });
  await page.route('**/data/dashboard.json', (route) => route.fulfill({ contentType: 'application/json', body: minimal }));
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForSelector('#rows tr');
  if ((await page.locator('#rows tr').count()) !== 2) problems.push('[minimal] minimal rows not rendered');
  await page.locator('.tabs button[data-tab="telegram"]').click();
  if (!(await page.locator('#buzz-empty').isVisible())) problems.push('[minimal] buzz empty state missing');
  await page.locator('.tabs button[data-tab="accuracy"]').click();
  if (!(await page.locator('#accuracy-empty').isVisible())) problems.push('[minimal] accuracy empty state missing');
  await page.close();

  // dead data endpoint: the visible, quiet error with a retry button
  page = await ctx2.newPage();
  page.on('pageerror', (err) => problems.push(`[broken] pageerror: ${err.message}`));
  await page.route('**/data/dashboard.json', (route) => route.fulfill({ status: 500, body: 'boom' }));
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForSelector('#load-error:not([hidden])');
  await page.close();
  await ctx2.close();

  await browser.close();
  server.close();

  if (problems.length) {
    console.log('PROBLEMS (' + problems.length + '):');
    for (const p of problems) console.log(' - ' + p);
    process.exit(1);
  }
  console.log('ALL CHECKS PASSED');
})().catch((err) => { console.error('FATAL: ' + (err && err.stack || err)); process.exit(2); });
