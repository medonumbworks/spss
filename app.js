/* CryptoPulse dashboard renderer. Plain JS, no build step.
 *
 * DATA CONTRACT — ./data/dashboard.json ("cryptopulse.dashboard/v1")
 * The hourly CI export must produce exactly this shape (field names are load-bearing).
 * The renderer ignores unknown/extra fields and tolerates missing optional ones, so the
 * exporter may add fields without breaking deployed shells.
 *
 * {
 *   "schema": "cryptopulse.dashboard/v1",
 *   "generatedAt": "2026-10-19T15:00:00Z",      // UTC, drives the "updated N ago" header
 *   "currency": "EUR",                           // default display currency (ISO 4217)
 *   "fearGreed": {"value": 59, "classification": "Greed"} | null,
 *   "accuracy": {                                // forecast journal, per horizon; either horizon may be missing
 *     "24h": {"pending": 12, "evaluated": 40, "hits": 22, "neutral": 5, "hitRate": 0.55,
 *             "avgProbability": 0.52, "brier": 0.248, "upShare": 0.5},
 *     "7d":  {...same keys...}
 *   },
 *   "rows": [{                                   // one entry per coin, any order (we sort client-side)
 *     "symbol": "BTC", "name": "Bitcoin",
 *     "tier": "BITPANDA" | "TRENDING",           // older exports may still carry MEME / DEX rows: hidden
 *     "tags": ["meme", ...],                     // "meme" only adds a small tag next to the name
 *     "price": 73045.34, "currency": "EUR",      // per-row currency (DEX rows may stay in USD)
 *     "chg1h": 0.1, "chg24h": -1.2, "chg7d": 3.4,  // percent, null when unknown (e.g. token too new)
 *     "marketCap": 1.4e12,                       // in `currency`; default sort key, null sorts last
 *     "up24": 0.49, "up7": 0.51,                 // P(price higher in 24h / 7d); null = no call
 *     "conf24": 0.54,                            // confidence of the 24h call, null when no call
 *     "noCall": "Unverified",                    // optional short reason when up24/up7 are null (blocked / too new / moves too little)
 *     "bottomScore": 25,                         // 0..100, 100 = at its lows; null when unknown
 *     "bottomLabel": "Upper part of its range",  // human label for bottomScore, null with it
 *     "risk": {"score": 62, "label": "High"} | null,       // rendered as "62 High"
 *     "pump": {"reasons": ["...", ...]} | null,  // non-null => show the "pump?" warning badge
 *     "mentions": 251, "spike": 5.2 | null,      // Telegram mentions and multiple vs 7-day norm
 *     "chain": "solana", "contract": "..."       // DEX rows only
 *   }, ...],
 *   "buzz": [{                                   // Telegram tab, ranked by mentions desc
 *     "symbol": "BTC", "mentions": 251, "mentionsPrev": 69,
 *     "baseline7d": 120.5,                       // 7-day hourly-window norm, null when unknown
 *     "spike": 2.08 | null,                      // mentions / baseline7d when notable, else null
 *     "composite": 0.02,                         // sentiment composite in [-1, 1]
 *     "samples": [{"text": "...", "url": "https://t.me/...", "community": "...", "score": -0.6}]
 *   }, ...],
 *   "polymarket": [{
 *     "question": "Bitcoin Up or Down on October 19?",
 *     "probability": 0.44, "outcome": "Up",      // probability of `outcome`
 *     "volume24hUsd": 1200000, "endDateMs": 1760915999000,
 *     "url": "https://polymarket.com/event/...", "symbols": ["BTC"]
 *   }, ...],
 *   "alerts": [{"symbol": "BTC", "name": "Bitcoin",// Bitpanda coins whose 24 h up-chance crossed the
 *     "probabilityUp": 0.57, "price": 73045.34,  // threshold this run; the CI Telegram step reads it,
 *     "currency": "EUR"}, ...],                  // this page ignores it (public data either way)
 *   "warnings": ["..."],                         // shown as amber banners under the tabs
 *   "notes": ["..."]                             // footer lines (cadence, disclaimer)
 * }
 *
 * Freshness fallback: when dashboard.json is older than 2 h, the Telegram tab is also
 * refreshed live from the public sentiment feed (cryptopulse.sentiment-feed/v1), merged
 * into `buzz` by symbol. All fetches are guarded; failures degrade to quiet notes.
 */
'use strict';

const DATA_URL = './data/dashboard.json';
const FEED_URL = 'https://raw.githubusercontent.com/medonumbworks/spss/sentiment-feed/feed.json';
const STALE_WARN_MS = 3 * 3600e3;   // header turns amber
const FEED_AFTER_MS = 2 * 3600e3;   // live Telegram refresh kicks in
const SAMPLE_MAX_CHARS = 110;

const state = {
  data: null,
  sortKey: 'marketCap',
  sortDir: -1,      // -1 desc, 1 asc
  tier: 'ALL',
  query: '',
};

const $ = (id) => document.getElementById(id);
const isNum = (v) => typeof v === 'number' && isFinite(v);
const arr = (v) => (Array.isArray(v) ? v : []);
const str = (v) => (typeof v === 'string' ? v : '');

/* createElement helper: el('span', {class: 'x', text: unsafeData}, children...) */
function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null) continue;
    if (k === 'text') node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

/* -- formatting ------------------------------------------------------------------ */

function fmtPrice(value, currency) {
  if (!isNum(value)) return '—';
  const cur = str(currency) || str(state.data && state.data.currency) || 'EUR';
  const opts = Math.abs(value) < 1
    ? { style: 'currency', currency: cur, maximumSignificantDigits: 4 }
    : { style: 'currency', currency: cur, minimumFractionDigits: 2, maximumFractionDigits: 2 };
  try {
    return new Intl.NumberFormat(undefined, opts).format(value);
  } catch (e) {
    return value + ' ' + cur; // unknown currency code: show it raw rather than crash
  }
}

function fmtPct(value) {
  if (!isNum(value)) return '—';
  return (value > 0 ? '+' : '') + value.toFixed(1) + '%';
}

function fmtProb(value) {
  return isNum(value) ? Math.round(value * 100) + '%' : '—';
}

function fmtCompact(value) {
  if (!isNum(value)) return '—';
  try {
    return new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value);
  } catch (e) {
    return String(Math.round(value));
  }
}

function fmtAge(ms) {
  const min = Math.max(0, Math.round(ms / 60000));
  if (min < 1) return 'updated just now';
  if (min < 60) return 'updated ' + min + ' min ago';
  const h = Math.floor(min / 60);
  if (h < 48) return 'updated ' + h + ' h ' + (min % 60) + ' min ago';
  return 'updated ' + Math.floor(h / 24) + ' d ago';
}

function truncate(text, max) {
  const s = str(text);
  return s.length > max ? s.slice(0, max - 1).trimEnd() + '…' : s;
}

function compColor(score) {
  if (!isNum(score)) return 'null-cell';
  if (score > 0.05) return 'pos';
  if (score < -0.05) return 'neg';
  return 'null-cell';
}

/* -- header ---------------------------------------------------------------------- */

function renderHeader() {
  const d = state.data;
  const chip = $('updated');
  const ts = Date.parse(str(d.generatedAt));
  if (isFinite(ts)) {
    const age = Date.now() - ts;
    chip.textContent = fmtAge(age);
    chip.classList.toggle('warn', age > STALE_WARN_MS);
    chip.hidden = false;
  } else {
    chip.hidden = true;
  }
  const fg = $('fg');
  const f = d.fearGreed;
  if (f && isNum(f.value)) {
    fg.textContent = 'F&G ' + f.value + (str(f.classification) ? ' · ' + f.classification : '');
    fg.hidden = false;
  } else {
    fg.hidden = true;
  }
}

function renderWarnings() {
  const list = $('warnings');
  list.textContent = '';
  const warnings = arr(state.data.warnings).filter((w) => str(w));
  for (const w of warnings) list.append(el('li', { text: w }));
  list.hidden = warnings.length === 0;
}

function renderNotes() {
  $('notes').textContent = arr(state.data.notes).filter((n) => str(n)).join(' ');
}

/* -- overview table -------------------------------------------------------------- */

/* sort accessors; null means "sorts last in either direction" */
const SORT_KEYS = {
  name: (r) => str(r.name) || str(r.symbol) || null,
  price: (r) => (isNum(r.price) ? r.price : null),
  chg24h: (r) => (isNum(r.chg24h) ? r.chg24h : null),
  chg7d: (r) => (isNum(r.chg7d) ? r.chg7d : null),
  up24: (r) => (isNum(r.up24) ? r.up24 : null),
  up7: (r) => (isNum(r.up7) ? r.up7 : null),
  bottomScore: (r) => (isNum(r.bottomScore) ? r.bottomScore : null),
  risk: (r) => (r.risk && isNum(r.risk.score) ? r.risk.score : null),
  marketCap: (r) => (isNum(r.marketCap) ? r.marketCap : null),
};

// the memecoin and DEX-radar tiers were removed: an older export's rows of theirs stay hidden
const REMOVED_TIERS = ['MEME', 'DEX'];

function matchesTier(row, tier) {
  if (REMOVED_TIERS.indexOf(row.tier) >= 0) return false;
  if (tier === 'ALL') return true;
  return row.tier === tier;
}

function visibleRows() {
  const q = state.query.trim().toLowerCase();
  const get = SORT_KEYS[state.sortKey] || SORT_KEYS.marketCap;
  return arr(state.data.rows)
    .filter((r) => r && typeof r === 'object')
    .filter((r) => matchesTier(r, state.tier))
    .filter((r) => !q || str(r.name).toLowerCase().includes(q) || str(r.symbol).toLowerCase().includes(q))
    .sort((a, b) => {
      const va = get(a);
      const vb = get(b);
      if (va === null && vb === null) return 0;
      if (va === null) return 1; // nulls last regardless of direction
      if (vb === null) return -1;
      const cmp = typeof va === 'string' || typeof vb === 'string'
        ? String(va).localeCompare(String(vb))
        : va - vb;
      return cmp * state.sortDir;
    });
}

const TIER_LABEL = { BITPANDA: 'Bitpanda', MEME: 'Meme', TRENDING: 'Trending', DEX: 'DEX' };

function numCell(label, textValue, cls) {
  return el('td', { class: 'num' + (cls ? ' ' + cls : ''), 'data-l': label, text: textValue });
}

function pctCell(label, value) {
  return numCell(label, fmtPct(value), isNum(value) ? (value > 0 ? 'pos' : value < 0 ? 'neg' : '') : 'null-cell');
}

/* "Up 24h"/"Up 7d": em dash plus the row's noCall reason (blocked / too new / moves too little) */
function probCell(label, value, noCall) {
  if (isNum(value)) return numCell(label, fmtProb(value));
  const td = el('td', { class: 'num null-cell', 'data-l': label, text: '—' });
  if (str(noCall)) td.append(el('span', { class: 'null-why', text: noCall }));
  return td;
}

function coinCell(row) {
  const td = el('td', { class: 'coin' });
  const line = el('div', {});
  line.append(el('span', { class: 'coin-name', text: str(row.name) || str(row.symbol) || '?' }));
  if (str(row.symbol)) line.append(el('span', { class: 'coin-sym', text: row.symbol }));
  const tier = TIER_LABEL[row.tier];
  if (tier) line.append(el('span', { class: 'tag', text: tier }));
  if (row.tier !== 'MEME' && arr(row.tags).indexOf('meme') >= 0) line.append(el('span', { class: 'tag', text: 'meme' }));
  const reasons = row.pump ? arr(row.pump.reasons).filter((x) => str(x)) : [];
  if (row.pump && reasons.length === 0) reasons.push('pump-like move detected');
  if (reasons.length) {
    const ul = el('ul', {});
    for (const why of reasons) ul.append(el('li', { text: why }));
    line.append(el('details', { class: 'pump' }, el('summary', { text: '⚠ pump?' }), ul));
  }
  td.append(line);
  if (str(row.chain)) {
    td.append(el('span', { class: 'sub', text: row.chain + (str(row.contract) ? ' · ' + truncate(row.contract, 14) : '') }));
  }
  return td;
}

function renderRows() {
  const tbody = $('rows');
  tbody.textContent = '';
  const rows = visibleRows();
  for (const r of rows) {
    const tr = el('tr', {});
    tr.append(coinCell(r));
    tr.append(numCell('Price', fmtPrice(r.price, r.currency), isNum(r.price) ? '' : 'null-cell'));
    tr.append(pctCell('24h', r.chg24h));
    tr.append(pctCell('7d', r.chg7d));
    tr.append(probCell('Up 24h', r.up24, r.noCall));
    tr.append(probCell('Up 7d', r.up7, r.noCall));
    const bottom = el('td', { class: 'num' + (isNum(r.bottomScore) ? '' : ' null-cell'), 'data-l': 'Bottom' });
    bottom.textContent = isNum(r.bottomScore) ? String(r.bottomScore) : '—';
    if (str(r.bottomLabel)) {
      bottom.title = r.bottomLabel;
      bottom.append(el('span', { class: 'sub', text: r.bottomLabel }));
    }
    tr.append(bottom);
    const risk = r.risk && isNum(r.risk.score)
      ? numCell('Risk', r.risk.score + ' ' + str(r.risk.label), 'risk-' + (str(r.risk.label).split(' ')[0] || 'x'))
      : numCell('Risk', '—', 'null-cell');
    tr.append(risk);
    tbody.append(tr);
  }
  $('rows-empty').hidden = rows.length > 0;
}

function renderSortButtons() {
  for (const btn of document.querySelectorAll('#coins thead button')) {
    if (btn.dataset.sort === state.sortKey) btn.dataset.dir = state.sortDir === 1 ? 'asc' : 'desc';
    else delete btn.dataset.dir;
    const th = btn.closest('th');
    if (th) th.setAttribute('aria-sort', btn.dataset.dir === 'asc' ? 'ascending' : btn.dataset.dir === 'desc' ? 'descending' : 'none');
  }
}

/* -- polymarket ------------------------------------------------------------------ */

function renderPolymarket() {
  const list = $('polymarket');
  list.textContent = '';
  const markets = arr(state.data.polymarket).filter((m) => m && str(m.question));
  for (const m of markets) {
    const li = el('li', {});
    const q = str(m.url)
      ? el('a', { class: 'pm-q', href: m.url, target: '_blank', rel: 'noopener noreferrer', text: m.question })
      : el('span', { class: 'pm-q', text: m.question });
    li.append(q);
    const pct = isNum(m.probability) ? Math.max(0, Math.min(100, Math.round(m.probability * 100))) : null;
    if (pct !== null) {
      const fill = el('div', { class: 'pm-fill' });
      fill.style.width = pct + '%';
      li.append(el('div', { class: 'pm-bar' }, fill));
    }
    const meta = el('div', { class: 'pm-meta' });
    if (pct !== null) meta.append(el('span', { class: 'pm-prob', text: pct + '% ' + (str(m.outcome) || 'Yes') }));
    if (isNum(m.volume24hUsd)) meta.append(el('span', { text: '$' + fmtCompact(m.volume24hUsd) + ' 24h vol' }));
    if (isNum(m.endDateMs)) meta.append(el('span', { text: 'ends ' + new Date(m.endDateMs).toLocaleDateString() }));
    li.append(meta);
    list.append(li);
  }
  $('polymarket-empty').hidden = markets.length > 0;
}

/* -- telegram buzz --------------------------------------------------------------- */

function renderBuzz() {
  const list = $('buzz');
  list.textContent = '';
  const entries = arr(state.data.buzz)
    .filter((b) => b && str(b.symbol))
    .slice()
    .sort((a, b) => (isNum(b.mentions) ? b.mentions : 0) - (isNum(a.mentions) ? a.mentions : 0));
  for (const b of entries) {
    const li = el('li', {});
    const head = el('div', { class: 'buzz-head' });
    head.append(el('span', { class: 'buzz-sym', text: b.symbol }));
    head.append(el('span', { class: 'buzz-mentions', text: (isNum(b.mentions) ? b.mentions : '?') + ' mentions' }));
    if (isNum(b.spike)) {
      head.append(el('span', { class: 'buzz-spike', text: '×' + b.spike.toFixed(1) + ' vs 7-day norm' }));
    } else if (isNum(b.mentionsPrev)) {
      head.append(el('span', { class: 'buzz-mentions', text: 'prev: ' + b.mentionsPrev }));
    }
    if (isNum(b.composite)) {
      head.append(el('span', {
        class: 'buzz-comp ' + compColor(b.composite),
        text: (b.composite > 0 ? '+' : '') + b.composite.toFixed(2),
      }));
    }
    li.append(head);
    const samples = arr(b.samples).filter((s) => s && str(s.text));
    if (samples.length) {
      const ul = el('ul', { class: 'samples' });
      for (const s of samples) {
        const item = el('li', {});
        const dot = el('span', { class: 's-dot' });
        dot.style.background = isNum(s.score) && s.score > 0.05 ? 'var(--pos)' : isNum(s.score) && s.score < -0.05 ? 'var(--neg)' : 'var(--muted)';
        const body = el('span', {}, dot, el('span', { class: 's-text', text: truncate(s.text, SAMPLE_MAX_CHARS) }));
        if (str(s.community)) body.append(el('span', { class: 's-comm', text: ' — ' + s.community }));
        // samples open the original Telegram post in a new tab
        item.append(str(s.url)
          ? el('a', { href: s.url, target: '_blank', rel: 'noopener noreferrer' }, body)
          : body);
        ul.append(item);
      }
      li.append(ul);
    }
    list.append(li);
  }
  $('buzz-empty').hidden = entries.length > 0;
}

/* -- accuracy -------------------------------------------------------------------- */

const HORIZON_TITLE = { '24h': '24-hour calls', '7d': '7-day calls' };

function renderAccuracy() {
  const box = $('accuracy');
  box.textContent = '';
  const acc = (state.data.accuracy && typeof state.data.accuracy === 'object') ? state.data.accuracy : {};
  let any = false;
  for (const horizon of ['24h', '7d']) {
    const a = acc[horizon];
    if (!a || typeof a !== 'object') continue;
    any = true;
    const card = el('div', { class: 'acc-card' });
    card.append(el('h3', { text: HORIZON_TITLE[horizon] || horizon }));
    const grid = el('div', { class: 'acc-grid' });
    const stat = (value, label) => grid.append(el('div', {}, el('span', { class: 'v', text: value }), el('span', { class: 'k', text: label })));
    stat(isNum(a.evaluated) ? String(a.evaluated) : '—', 'evaluated');
    if (isNum(a.hitRate)) {
      const pts = Math.round((a.hitRate - 0.5) * 100);
      stat(fmtProb(a.hitRate), 'hit rate (' + (pts >= 0 ? '+' : '') + pts + ' pts vs 50% chance)');
    } else {
      stat('—', 'hit rate');
    }
    stat(fmtProb(a.avgProbability), 'avg forecast');
    stat(isNum(a.brier) ? a.brier.toFixed(3) : '—', 'Brier score (lower is better)');
    stat(isNum(a.pending) ? String(a.pending) : '—', 'pending');
    if (isNum(a.neutral) && a.neutral > 0) stat(String(a.neutral), 'neutral (moved less than half its typical move)');
    if (isNum(a.upShare)) stat(fmtProb(a.upShare), 'share that actually rose (baseline)');
    card.append(grid);
    if (isNum(a.hits) && isNum(a.evaluated)) {
      card.append(el('p', { class: 'acc-note', text: a.hits + ' of ' + a.evaluated + ' evaluated forecasts were on the right side.' }));
    }
    box.append(card);
  }
  $('accuracy-empty').hidden = any;
}

/* -- live feed fallback ----------------------------------------------------------- */

function note(text) {
  const p = $('note');
  p.textContent = text || '';
  p.hidden = !text;
}

/* Merge a cryptopulse.sentiment-feed/v1 payload into state.data.buzz by symbol. */
function mergeFeed(feed) {
  if (!feed || typeof feed !== 'object') return false;
  // only coins this dashboard lists: the hub also counts memecoins and new cashtags
  const listed = new Set(arr(state.data.rows).map((r) => r && r.symbol).filter(Boolean));
  const assets = arr(feed.assets).filter((a) => a && str(a.symbol) && isNum(a.mentions) && a.mentions > 0 &&
    (!listed.size || listed.has(a.symbol)));
  if (!assets.length) return false;
  const bySymbol = new Map();
  const merged = [];
  for (const b of arr(state.data.buzz)) {
    if (b && str(b.symbol)) {
      bySymbol.set(b.symbol, b);
      merged.push(b);
    }
  }
  for (const a of assets) {
    let entry = bySymbol.get(a.symbol);
    if (!entry) {
      entry = { symbol: a.symbol, baseline7d: null, spike: null };
      merged.push(entry);
    }
    entry.mentions = isNum(a.mentions) ? a.mentions : entry.mentions;
    entry.mentionsPrev = isNum(a.mentionsPrev) ? a.mentionsPrev : entry.mentionsPrev;
    entry.composite = isNum(a.composite) ? a.composite : entry.composite;
    entry.samples = arr(a.samples)
      .filter((s) => s && str(s.text))
      .map((s) => ({ text: s.text, url: s.url, community: s.community, score: s.score }));
    // the feed publishes its own 7-day baseline and spike; fall back to recomputing from a stale one
    if (isNum(a.baseline7d)) entry.baseline7d = a.baseline7d;
    if (isNum(a.spike)) {
      entry.spike = a.spike;
    } else if (isNum(entry.baseline7d) && entry.baseline7d > 0 && isNum(entry.mentions)) {
      const ratio = entry.mentions / entry.baseline7d;
      entry.spike = ratio >= 1.5 ? Math.round(ratio * 100) / 100 : null;
    }
  }
  state.data.buzz = merged;
  return true;
}

async function maybeRefreshFromFeed() {
  const ts = Date.parse(str(state.data.generatedAt));
  if (!isFinite(ts) || Date.now() - ts <= FEED_AFTER_MS) return;
  try {
    const res = await fetch(FEED_URL, { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const feed = await res.json();
    if (mergeFeed(feed)) {
      renderBuzz();
      const meta = $('buzz-meta');
      meta.textContent = 'Dashboard data is stale; mentions below were refreshed live from the sentiment feed'
        + (str(feed.generatedAt) ? ' (' + feed.generatedAt + ')' : '') + '.';
    }
  } catch (e) {
    note('Dashboard data is more than 2 h old and the live Telegram refresh failed; showing the last export.');
  }
}

/* -- tabs + controls --------------------------------------------------------------- */

function selectTab(name) {
  for (const btn of document.querySelectorAll('.tabs button')) {
    const on = btn.dataset.tab === name;
    btn.setAttribute('aria-selected', on ? 'true' : 'false');
    const panel = $('tab-' + btn.dataset.tab);
    if (panel) panel.hidden = !on;
  }
  if (location.hash !== '#' + name) {
    try { history.replaceState(null, '', '#' + name); } catch (e) { /* sandboxed pages */ }
  }
}

function wireControls() {
  document.querySelector('.tabs').addEventListener('click', (ev) => {
    const btn = ev.target.closest('button[data-tab]');
    if (btn) selectTab(btn.dataset.tab);
  });

  $('search').addEventListener('input', (ev) => {
    state.query = ev.target.value || '';
    renderRows();
  });

  $('tiers').addEventListener('click', (ev) => {
    const btn = ev.target.closest('button[data-tier]');
    if (!btn) return;
    state.tier = btn.dataset.tier;
    for (const b of document.querySelectorAll('#tiers button')) {
      b.setAttribute('aria-pressed', b === btn ? 'true' : 'false');
    }
    renderRows();
  });

  document.querySelector('#coins thead').addEventListener('click', (ev) => {
    const btn = ev.target.closest('button[data-sort]');
    if (!btn) return;
    const key = btn.dataset.sort;
    if (state.sortKey === key) {
      state.sortDir = -state.sortDir;
    } else {
      state.sortKey = key;
      // strings read naturally ascending, numbers biggest-first
      state.sortDir = btn.dataset.kind === 'str' ? 1 : -1;
    }
    renderSortButtons();
    renderRows();
  });

  $('retry').addEventListener('click', () => load());

  const initial = location.hash.replace('#', '');
  if (['overview', 'telegram', 'accuracy', 'about'].indexOf(initial) >= 0) selectTab(initial);
}

/* -- load ------------------------------------------------------------------------- */

function renderAll() {
  renderHeader();
  renderWarnings();
  renderNotes();
  renderSortButtons();
  renderRows();
  renderPolymarket();
  renderBuzz();
  renderAccuracy();
}

async function load() {
  $('load-error').hidden = true;
  try {
    // network-first: the service worker falls back to its cached copy when offline
    const res = await fetch(DATA_URL, { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    if (!data || typeof data !== 'object' || !Array.isArray(data.rows)) throw new Error('unexpected payload');
    state.data = data;
    renderAll();
    maybeRefreshFromFeed();
  } catch (e) {
    if (state.data) {
      note('Refresh failed; showing the data already loaded.');
    } else {
      $('load-error').hidden = false;
    }
  }
}

wireControls();
load();
// keep the "updated N min ago" chip honest while the page sits open
setInterval(() => { if (state.data) renderHeader(); }, 60000);

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch(() => { /* http or blocked: app still works */ });
  });
}
