// Dashboard v2 — multimodal live BTCUSDT predictor.
//
// Layout responsibilities:
//   1. Boot: pull /api/health, /api/candles, /api/regimes; wire the main
//      lightweight-charts candle series.
//   2. SSE: subscribe to /api/live/stream and .update() the last bar on
//      every kline tick so the chart animates in real time.
//   3. Auto-select the current bar for predictions and top-K similar
//      matches, refreshing every 5s so the panel keeps up with new bars.
//   4. Render top-5 similar patterns as inline SVG candle mini-charts.
//   5. Poll /api/news for the latest scored articles (every 30s).
//   6. Poll /api/paper/state every 2s: balance, position, verdict.
//   7. Poll /api/paper/predictions periodically to draw the model-
//      accuracy card + rolling accuracy sparkline.
//   8. Optionally auto-start the paper trader.

// ---------------------------------------------------------------------
// Tiny helpers
// ---------------------------------------------------------------------
const API = (path, params = {}) => {
  const url = new URL(path, window.location.origin);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v);
  }
  return fetch(url).then(r => {
    if (!r.ok) return r.json().then(j => { throw new Error(j.detail || r.statusText); });
    return r.json();
  });
};

async function postJson(path, params = {}) {
  const url = new URL(path, window.location.origin);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined) url.searchParams.set(k, v);
  }
  const r = await fetch(url, { method: 'POST' });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || r.statusText);
  }
  return r.json();
}

const REGIME_CLASS = { 0: 'bear', 1: 'sideways', 2: 'bull' };
const REGIME_NAME = { 0: 'bear', 1: 'sideways', 2: 'bull' };

function isNum(x) { return typeof x === 'number' && Number.isFinite(x); }
function fmtUsd(n)  { return isNum(n) ? `${n < 0 ? '-' : ''}$${Math.abs(n).toFixed(2)}` : '—'; }
function fmtPct(x, places = 1) { return isNum(x) ? `${(x * 100).toFixed(places)}%` : '—'; }
function fmtNum(x, places = 3) { return isNum(x) ? x.toFixed(places) : '—'; }
function isoToSec(iso) { return Math.floor(Date.parse(iso) / 1000); }
function secToIso(sec) { return new Date(sec * 1000).toISOString(); }
function fmtDate(iso, len = 16) {
  return (iso || '').replace('+00:00', 'Z').replace('T', ' ').slice(0, len);
}

// ---------------------------------------------------------------------
// State + DOM handles
// ---------------------------------------------------------------------
const state = {
  chart: null,
  series: null,
  candles: [],
  regimes: [],
  currentBarTime: null,   // ISO — the most-recent labelled bar we can predict on
  lastPrediction: null,
  autoStartAttempted: false,
  markMode: false,          // waiting for two clicks
  markStartSec: null,       // first click (UNIX seconds)
  markEndSec: null,         // second click
  markPriceLines: [],       // Lightweight-charts price lines used for the band
};

const dom = {
  meta: document.getElementById('meta'),
  chart: document.getElementById('chart'),
  regimeStrip: document.getElementById('regime-strip'),
  limit: document.getElementById('limit-select'),
  reload: document.getElementById('reload-btn'),
  liveBadge: document.getElementById('live-badge'),

  similarGrid: document.getElementById('similar-grid'),
  regimeToggle: document.getElementById('regime-toggle'),

  prediction: document.getElementById('prediction-body'),
  predictionFreshness: document.getElementById('prediction-freshness'),

  paperStart: document.getElementById('paper-start-btn'),
  paperStop:  document.getElementById('paper-stop-btn'),
  paperReset: document.getElementById('paper-reset-btn'),
  paperStatus: document.getElementById('paper-status'),
  paperSummary: document.getElementById('paper-summary'),
  equitySpark: document.getElementById('equity-spark'),
  paperTrades: document.getElementById('paper-trades'),
  autostartToggle: document.getElementById('autostart-toggle'),

  accuracySummary: document.getElementById('accuracy-summary'),
  accuracySpark: document.getElementById('accuracy-spark'),
  accuracyCount: document.getElementById('accuracy-count'),

  newsBody: document.getElementById('news-body'),
  newsCount: document.getElementById('news-count'),

  gradcam: document.getElementById('gradcam-body'),

  retrainStatus: document.getElementById('retrain-status'),
  retrainSummary: document.getElementById('retrain-summary'),
  retrainBtn: document.getElementById('retrain-btn'),
};

// ---------------------------------------------------------------------
// Main chart bootstrap
// ---------------------------------------------------------------------
function initChart() {
  const chart = LightweightCharts.createChart(dom.chart, {
    layout: { background: { color: '#0b1015' }, textColor: '#d6dee8' },
    grid: {
      vertLines: { color: '#1b232d' },
      horzLines: { color: '#1b232d' },
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#233040' },
    timeScale: { borderColor: '#233040', timeVisible: true, secondsVisible: false },
    autoSize: true,
  });
  const series = chart.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350',
    borderUpColor: '#26a69a', borderDownColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
  });
  chart.subscribeClick(param => {
    if (!param || !param.time) return;
    if (state.markMode) {
      handleMarkClick(Number(param.time));
      return;
    }
    const iso = secToIso(Number(param.time));
    state.currentBarTime = iso;
    refreshPredictionAndSimilar(iso);
  });
  state.chart = chart;
  state.series = series;
}

// ---------------------------------------------------------------------
// Mark-range drawing tool — click A then B on the chart to define
// a shaded band, then auto-run range analysis + similar-pattern
// suggestions for the marked window's END bar.
// ---------------------------------------------------------------------
function setMarkHint(text) {
  let el = document.getElementById('mark-hint');
  if (!text) {
    if (el) el.remove();
    return;
  }
  if (!el) {
    el = document.createElement('div');
    el.id = 'mark-hint';
    document.getElementById('chart').appendChild(el);
  }
  el.textContent = text;
}

function toggleMarkMode() {
  state.markMode = !state.markMode;
  const btn = document.getElementById('mark-btn');
  btn.classList.toggle('active', state.markMode);
  btn.textContent = state.markMode ? '✎ Marking… click A' : '✎ Mark range';
  document.getElementById('clear-mark-btn').hidden = !hasMark();
  setMarkHint(state.markMode ? 'Click the first candle to start the range' : null);
  if (!state.markMode && !hasMark()) {
    clearMarkOverlay();
  }
}

function hasMark() {
  return state.markStartSec != null && state.markEndSec != null;
}

async function handleMarkClick(timeSec) {
  if (state.markStartSec == null) {
    state.markStartSec = timeSec;
    setMarkHint(`A locked at ${fmtDate(secToIso(timeSec))}Z — click the second candle`);
    return;
  }
  state.markEndSec = timeSec;
  if (state.markEndSec < state.markStartSec) {
    [state.markStartSec, state.markEndSec] = [state.markEndSec, state.markStartSec];
  }
  toggleMarkMode();                         // exit mark mode
  document.getElementById('clear-mark-btn').hidden = false;
  drawMarkOverlay();
  await runMarkAnalysis();
}

function drawMarkOverlay() {
  clearMarkOverlay();
  if (!hasMark()) return;
  const s = state.markStartSec, e = state.markEndSec;
  // Find close prices at the two endpoints to place horizontal band lines.
  const inRange = state.candles.filter(c =>
    isoToSec(c.open_time) >= s && isoToSec(c.open_time) <= e);
  if (inRange.length === 0) return;
  const hi = Math.max(...inRange.map(c => c.high));
  const lo = Math.min(...inRange.map(c => c.low));
  const first = inRange[0], last = inRange[inRange.length - 1];
  // Two horizontal price lines at the range's high and low form a band.
  const styleTop = state.series.createPriceLine({
    price: hi, color: '#4dabf7', lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    axisLabelVisible: true,
    title: `mark hi ${hi.toFixed(0)}`,
  });
  const styleBot = state.series.createPriceLine({
    price: lo, color: '#4dabf7', lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    axisLabelVisible: true,
    title: `mark lo ${lo.toFixed(0)}`,
  });
  state.markPriceLines = [styleTop, styleBot];
  const bars = inRange.length;
  const spanHrs = ((e - s) / 3600).toFixed(1);
  const chg = ((last.close - first.close) / first.close * 100).toFixed(2);
  setMarkHint(`Marked ${bars} bars · ${spanHrs}h · Δ ${chg}%`);
}

function clearMarkOverlay() {
  for (const pl of state.markPriceLines) {
    try { state.series.removePriceLine(pl); } catch {}
  }
  state.markPriceLines = [];
}

function clearMark() {
  state.markStartSec = null;
  state.markEndSec = null;
  clearMarkOverlay();
  setMarkHint(null);
  document.getElementById('clear-mark-btn').hidden = true;
}

async function runMarkAnalysis() {
  const panel = document.getElementById('analyze-panel');
  const body  = document.getElementById('analyze-body');
  const label = document.getElementById('analyze-range');
  panel.hidden = false;
  body.innerHTML = '<div class="empty">analyzing marked range…</div>';

  const startIso = secToIso(state.markStartSec);
  const endIso   = secToIso(state.markEndSec);

  try {
    const a = await API('/api/analyze/range', { start: startIso, end: endIso });
    label.textContent = `${fmtDate(a.start)}Z → ${fmtDate(a.end)}Z · ${a.bars} bars · MARKED`;
    renderAnalysis(a);
  } catch (e) {
    body.innerHTML = `<div class="empty">analysis failed: ${e.message}</div>`;
  }

  // Also update the "similar patterns" strip using the end bar of the
  // mark — so the top-K reflects what the user selected, not the live
  // bar. Backend does its own fallback if the end bar has no embedding.
  try {
    await refreshSimilarWithFallback(endIso);
  } catch { /* already surfaces its own error */ }
}

document.getElementById('mark-btn').addEventListener('click', toggleMarkMode);
document.getElementById('clear-mark-btn').addEventListener('click', () => {
  clearMark();
});

async function loadHealth() {
  const h = await API('/api/health');
  const range = `${h.time_range_start.slice(0, 10)} → ${h.time_range_end.slice(0, 10)}`;
  dom.meta.textContent =
    `${h.pair} · ${h.interval} · ${h.total_bars.toLocaleString()} bars · ` +
    `${h.n_features} feats · ${range} · anchor cut @ row ${h.anchor_cut_row.toLocaleString()}`;
}

async function loadCandlesAndRegimes() {
  const limit = Number(dom.limit.value) || 500;
  const [candles, regimes] = await Promise.all([
    API('/api/candles', { limit }),
    API('/api/regimes', { limit }),
  ]);
  state.candles = candles;
  state.regimes = regimes;

  const series = candles.map(c => ({
    time: isoToSec(c.open_time),
    open: c.open, high: c.high, low: c.low, close: c.close,
  }));
  state.series.setData(series);
  state.chart.timeScale().fitContent();

  renderRegimeStrip(regimes);

  // The most-recent labelled bar (i.e., not the very last one) becomes
  // the "current query bar" for auto-refresh.
  if (candles.length >= 2) {
    state.currentBarTime = candles[candles.length - 2].open_time;
    refreshPredictionAndSimilar(state.currentBarTime);
  }
}

function renderRegimeStrip(regimes) {
  dom.regimeStrip.innerHTML = '';
  for (const r of regimes) {
    const div = document.createElement('div');
    div.className = `cell ${REGIME_CLASS[r.regime] || 'sideways'}`;
    div.title = `${r.regime_name} · ${r.open_time}`;
    dom.regimeStrip.appendChild(div);
  }
}

// ---------------------------------------------------------------------
// Prediction + top-K similar auto-refresh
// ---------------------------------------------------------------------
async function refreshPredictionAndSimilar(iso) {
  // Try `iso` first; if the API 404s (bar not labelled), walk backward.
  const candidates = pickCandidateBars(iso);
  for (const ts of candidates) {
    try {
      const p = await API('/api/predict', { open_time: ts });
      state.lastPrediction = p;
      state.currentBarTime = ts;
      renderPrediction(p);
      // Similar + Grad-CAM depend on the CNN embedding parquet, which
      // may not include the very-newest bars if the CNN wasn't
      // re-extracted. Handle each independently with fallback.
      refreshSimilarWithFallback(ts);
      refreshGradcamWithFallback(ts);
      dom.predictionFreshness.textContent = `${fmtDate(ts)}Z · refreshed ${new Date().toLocaleTimeString()}`;
      return;
    } catch (e) {
      /* try older bar */
    }
  }
  dom.prediction.innerHTML = `<div class="empty">no labelled bar available yet</div>`;
}

function pickCandidateBars(preferredIso) {
  const list = state.candles.slice(-8).map(c => c.open_time).reverse();
  if (preferredIso) return [preferredIso, ...list.filter(x => x !== preferredIso)];
  return list;
}

// One request, backend auto-falls-back to the newest embedded bar if
// the live bar has no embedding yet. No walk-backward on the client.
async function refreshSimilarWithFallback(preferredIso) {
  try {
    const s = await API('/api/similar/detailed', {
      open_time: preferredIso, k: 5,
      regime_filter: dom.regimeToggle.checked,
      fallback: true,
    });
    renderSimilarStrip(s.matches);
    if (s.fallback_used) {
      dom.similarGrid.insertAdjacentHTML('afterbegin',
        `<div style="grid-column:1/-1;font-size:11px;color:var(--fg-muted);font-style:italic;margin-bottom:4px;">`
        + `showing patterns for ${fmtDate(s.used_open_time)}Z — the CNN embedding parquet has no entry yet for ${fmtDate(preferredIso)}Z. `
        + `Run <code>python scripts/refresh_all.py</code> (no <code>--skip-cnn</code>) to bring the top-K up to today.`
        + `</div>`);
    }
  } catch (e) {
    dom.similarGrid.classList.add('empty');
    dom.similarGrid.innerHTML =
      'no similar-pattern lookup available: ' + e.message;
  }
}

// Grad-CAM only tries the requested bar. If it 404s we cache a "skip"
// flag until the next fresh currentBarTime, so we don't hammer the
// backend on the 15 s auto-refresh.
let gradcamSkipUntil = null;
async function refreshGradcamWithFallback(preferredIso) {
  if (gradcamSkipUntil === preferredIso) return;
  try {
    await loadGradcam(preferredIso);
    gradcamSkipUntil = null;
  } catch (e) {
    gradcamSkipUntil = preferredIso;
    dom.gradcam.innerHTML =
      `<div class="empty">grad-cam unavailable for this bar: ${e.message}</div>`;
  }
}

function renderPrediction(p) {
  const dirClass = p.label === 1 ? 'up' : 'down';
  const cpDown = p.conformal_set.down
    ? '<span class="pill down">↓</span>'
    : '<span style="opacity:0.3">—</span>';
  const cpUp = p.conformal_set.up
    ? '<span class="pill up">↑</span>'
    : '<span style="opacity:0.3">—</span>';
  const regimePill = p.regime !== null
    ? `<span class="pill ${REGIME_CLASS[p.regime] || 'sideways'}">${p.regime_name}</span>`
    : `<span class="empty">—</span>`;
  const inDemo = p.is_in_demo_window
    ? `<span class="pill up">held-out</span>`
    : `<span class="pill warn">in fit window</span>`;

  dom.prediction.innerHTML = `
    <div class="kv"><span class="k">Bar</span><span class="v">${fmtDate(p.open_time)}Z</span></div>
    <div class="kv"><span class="k">P(up)</span><span class="v">${fmtNum(p.p_up, 4)}</span></div>
    <div class="kv"><span class="k">Direction</span><span class="v"><span class="pill ${dirClass}">${p.label === 1 ? 'UP' : 'DOWN'}</span></span></div>
    <div class="kv"><span class="k">Conformal 90% set</span><span class="v">${cpDown} ${cpUp}</span></div>
    <div class="kv"><span class="k">Regime</span><span class="v">${regimePill}</span></div>
    <div class="kv"><span class="k">Window</span><span class="v">${inDemo}</span></div>
  `;
}

// ---------------------------------------------------------------------
// Top-5 similar patterns as SVG candle mini-charts
// ---------------------------------------------------------------------
async function loadSimilarDetailed(iso) {
  const s = await API('/api/similar/detailed', {
    open_time: iso, k: 5, regime_filter: dom.regimeToggle.checked,
  });
  renderSimilarStrip(s.matches);
}

function renderSimilarStrip(matches) {
  if (!matches || matches.length === 0) {
    dom.similarGrid.classList.add('empty');
    dom.similarGrid.textContent = 'no matches (try unchecking the regime filter)';
    return;
  }
  dom.similarGrid.classList.remove('empty');
  dom.similarGrid.innerHTML = matches.map(m => renderMiniCard(m)).join('');
  // Wire clicks
  dom.similarGrid.querySelectorAll('.mini').forEach(el => {
    el.addEventListener('click', () => {
      const iso = el.getAttribute('data-ts');
      state.currentBarTime = iso;
      refreshPredictionAndSimilar(iso);
    });
  });
}

function renderMiniCard(m) {
  const svg = miniCandleSvg(m.window);
  const regimeName = m.regime_name || REGIME_NAME[m.regime] || 'unknown';
  const regimeClass = REGIME_CLASS[m.regime] || 'sideways';
  const nextCls = m.next_direction === 1 ? 'up' : (m.next_direction === 0 ? 'down' : '');
  const nextGlyph = m.next_direction === 1 ? '↑ next hour' : (m.next_direction === 0 ? '↓ next hour' : '—');
  return `
    <div class="mini" data-ts="${m.open_time}" title="jump to ${m.open_time}">
      <div class="mini-header">
        <span>${fmtDate(m.open_time, 16)}Z</span>
        <span class="pill ${regimeClass}" style="font-size:9px;padding:0 6px;">${regimeName}</span>
      </div>
      ${svg}
      <div class="mini-footer">
        <span class="sim">sim ${fmtNum(m.similarity, 4)}</span>
        <span class="next ${nextCls}">${nextGlyph}</span>
      </div>
    </div>
  `;
}

function miniCandleSvg(bars) {
  if (!Array.isArray(bars) || bars.length === 0) return '';
  const W = 200, H = 90, PAD = 3;
  const highs = bars.map(b => b.h), lows = bars.map(b => b.l);
  const maxP = Math.max(...highs);
  const minP = Math.min(...lows);
  const range = (maxP - minP) || 1;
  const barW = Math.max(1, (W - 2 * PAD) / bars.length);

  const y = p => H - PAD - ((p - minP) / range) * (H - 2 * PAD);

  const parts = bars.map((b, i) => {
    const cx = PAD + i * barW + barW / 2;
    const yh = y(b.h), yl = y(b.l);
    const yo = y(b.o), yc = y(b.c);
    const color = b.c >= b.o ? '#26a69a' : '#ef5350';
    const bodyTop = Math.min(yo, yc);
    const bodyH = Math.max(1, Math.abs(yc - yo));
    return `
      <line x1="${cx.toFixed(1)}" x2="${cx.toFixed(1)}" y1="${yh.toFixed(1)}" y2="${yl.toFixed(1)}" stroke="${color}" stroke-width="0.6"/>
      <rect x="${(cx - barW * 0.35).toFixed(1)}" y="${bodyTop.toFixed(1)}" width="${(barW * 0.7).toFixed(1)}" height="${bodyH.toFixed(1)}" fill="${color}"/>
    `;
  }).join('');
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${parts}</svg>`;
}

// ---------------------------------------------------------------------
// Grad-CAM (kept simple, only refreshes on new query bar)
// ---------------------------------------------------------------------
async function loadGradcam(iso) {
  const r = await fetch(`/api/gradcam?open_time=${encodeURIComponent(iso)}`);
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || r.statusText);
  }
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  dom.gradcam.innerHTML = `<img src="${url}" alt="Grad-CAM" />`;
}

// ---------------------------------------------------------------------
// News feed
// ---------------------------------------------------------------------
async function refreshNews() {
  try {
    const n = await API('/api/news', { limit: 30, ticker: 'BTC' });
    dom.newsCount.textContent = `${n.count} recent BTC articles`;
    if (!n.items || n.items.length === 0) {
      dom.newsBody.className = 'empty';
      dom.newsBody.textContent = 'no news yet — run scripts/fetch_news.py + score_news.py';
      return;
    }
    dom.newsBody.className = '';
    dom.newsBody.innerHTML = n.items.map(item => {
      const s = item.sent_score;
      let sentCls = 'none', sentText = '—';
      if (typeof s === 'number' && Number.isFinite(s)) {
        sentCls = s > 0.15 ? 'pos' : (s < -0.15 ? 'neg' : 'neu');
        sentText = (s >= 0 ? '+' : '') + s.toFixed(2);
      }
      return `
        <div class="news-row">
          <time>${fmtDate(item.published_at, 16)}</time>
          <div class="title">
            <span class="src">${item.source}</span>
            <a href="${item.url}" target="_blank" rel="noopener">${escapeHtml(item.title)}</a>
          </div>
          <span class="sent ${sentCls}">${sentText}</span>
        </div>
      `;
    }).join('');
  } catch (e) {
    dom.newsBody.className = 'empty';
    dom.newsBody.textContent = `news unavailable: ${e.message}`;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

// ---------------------------------------------------------------------
// Paper trader panel
// ---------------------------------------------------------------------
async function refreshPaper() {
  try {
    const s = await API('/api/paper/state');
    renderPaperPanel(s);
    renderAccuracy(s.accuracy || {}, s.predictions_labelled || 0);
  } catch (e) {
    dom.paperStatus.textContent = `paper state unavailable: ${e.message}`;
  }
}

function renderPaperPanel(s) {
  if (s.is_running) {
    dom.paperStatus.classList.add('running');
    const streamOk = s.stream && s.stream.connected ? '🟢' : '🟡';
    dom.paperStatus.textContent =
      `${streamOk} running · ${s.bars_seen} bars seen · ${s.bars_traded} trades · ${s.bars_abstained} abstained`;
  } else {
    dom.paperStatus.classList.remove('running');
    dom.paperStatus.textContent = s.bars_seen === 0
      ? 'not started · click Start (or enable auto-start above)'
      : `stopped after ${s.bars_seen} bars`;
  }

  const balance = isNum(s.balance) ? s.balance : 0;
  const initial = isNum(s.initial_balance) ? s.initial_balance : 100;
  const pnl = balance - initial;
  const pnlClass = pnl > 0 ? 'pos' : (pnl < 0 ? 'neg' : '');
  const stats = s.stats || {};
  const winRate = fmtPct(stats.win_rate);
  const sharpe = fmtNum(stats.sharpe_per_trade);
  const verdict = stats.verdict || '—';
  const verdictClass = verdict === 'model is working' ? 'pos'
    : verdict === 'model is losing' ? 'neg' : '';

  let openLine = '';
  if (s.open_position) {
    const op = s.open_position;
    const dir = op.direction || '?';
    const dollars = isNum(op.dollars) ? op.dollars.toFixed(2) : '—';
    const entry = isNum(op.entry_price) ? op.entry_price.toFixed(2) : '—';
    openLine = `
      <div class="k">Open position</div>
      <div class="v ${dir === 'long' ? 'pos' : 'neg'}">
        ${dir.toUpperCase()} $${dollars} @${entry}
      </div>`;
  }

  dom.paperSummary.innerHTML = `
    <div class="k">Balance</div><div class="v">${fmtUsd(balance)}</div>
    <div class="k">P&amp;L</div><div class="v ${pnlClass}">${fmtUsd(pnl)}</div>
    <div class="k">Win rate (last 50)</div><div class="v">${winRate}</div>
    <div class="k">Sharpe/trade</div><div class="v">${sharpe}</div>
    <div class="k">Trading verdict</div><div class="v ${verdictClass}">${verdict}</div>
    ${openLine}
  `;
  drawEquitySpark(s.equity_curve || [], initial);

  const trades = s.recent_trades || [];
  if (trades.length === 0) {
    dom.paperTrades.innerHTML = '<div class="empty">no closed trades yet</div>';
    return;
  }
  dom.paperTrades.innerHTML = trades.slice().reverse().slice(0, 8).map(t => {
    const pnl = isNum(t.pnl_dollars) ? t.pnl_dollars : 0;
    const cls = pnl > 0 ? 'pos' : 'neg';
    const dir = t.direction || '?';
    return `
      <div class="trade-row">
        <span class="dir ${dir === 'long' ? 'pos' : 'neg'}">${dir}</span>
        <time>${fmtDate(t.exit_time)}</time>
        <span class="pnl ${cls}">${fmtUsd(t.pnl_dollars)} (${fmtPct(t.pnl_pct, 2)})</span>
      </div>
    `;
  }).join('');
}

function drawEquitySpark(curve, initialBalance) {
  const svg = dom.equitySpark;
  svg.innerHTML = '';
  if (!Array.isArray(curve) || curve.length === 0) return;
  const W = 240, H = 60, PAD = 4;
  const balances = curve.map(p => p.balance).filter(isNum);
  if (balances.length === 0) return;
  const baseB = isNum(initialBalance) ? initialBalance : 100;
  const minB = Math.min(...balances, baseB);
  const maxB = Math.max(...balances, baseB);
  const range = (maxB - minB) || 1;
  const xs = i => PAD + (i / (curve.length - 1 || 1)) * (W - 2 * PAD);
  const ys = b => H - PAD - ((b - minB) / range) * (H - 2 * PAD);
  const bl = ys(baseB);

  const ns = 'http://www.w3.org/2000/svg';
  const line = document.createElementNS(ns, 'line');
  line.setAttribute('class', 'baseline');
  line.setAttribute('x1', PAD); line.setAttribute('x2', W - PAD);
  line.setAttribute('y1', bl); line.setAttribute('y2', bl);
  svg.appendChild(line);

  const pts = curve.map((p, i) => `${xs(i).toFixed(1)},${ys(p.balance).toFixed(1)}`).join(' ');
  const poly = document.createElementNS(ns, 'polyline');
  poly.setAttribute('class', 'line');
  poly.setAttribute('points', pts);
  svg.appendChild(poly);
}

// ---------------------------------------------------------------------
// Model-accuracy card
// ---------------------------------------------------------------------
function renderAccuracy(acc, _labelledFallback) {
  const n = acc.n_labelled || 0;
  dom.accuracyCount.textContent = `${n} labelled`;
  if (n === 0) {
    dom.accuracySummary.className = 'empty';
    dom.accuracySummary.textContent =
      'no closed bars yet — start the paper trader and wait for the next UTC hour';
    dom.accuracySpark.innerHTML = '';
    return;
  }
  dom.accuracySummary.className = '';

  const overall = acc.overall_accuracy;
  const conf = acc.confident_accuracy;
  const unc = acc.uncertain_accuracy;
  const verdict = acc.verdict || '—';
  const clsFor = (x, threshold = 0.50) => (!isNum(x) ? 'mut'
    : x > threshold + 0.02 ? 'pos'
    : x < threshold - 0.02 ? 'neg' : 'mut');

  dom.accuracySummary.innerHTML = `
    <div class="stat ${clsFor(overall)}">
      <div class="lbl">All predictions</div>
      <div class="val">${fmtPct(overall, 1)}</div>
      <div class="sub">${n} labelled</div>
    </div>
    <div class="stat ${clsFor(conf)}">
      <div class="lbl">Confident (singleton)</div>
      <div class="val">${fmtPct(conf, 1)}</div>
      <div class="sub">${acc.confident_count} bars</div>
    </div>
    <div class="stat ${clsFor(unc, 0.50)}">
      <div class="lbl">Uncertain (doubleton)</div>
      <div class="val">${fmtPct(unc, 1)}</div>
      <div class="sub">${acc.uncertain_count} bars</div>
    </div>
    <div class="stat ${verdict === 'model beats a coin' ? 'pos' : verdict === 'model below coin' ? 'neg' : 'mut'}">
      <div class="lbl">Verdict</div>
      <div class="val" style="font-size:14px; line-height:1.6; padding-top:2px;">${verdict}</div>
      <div class="sub">rolling</div>
    </div>
  `;

  // Rolling accuracy sparkline — needs the full predictions list.
  API('/api/paper/predictions', { limit: 500 })
    .then(p => drawAccuracySpark(p.predictions || []))
    .catch(() => { /* ignore, keep last render */ });
}

function drawAccuracySpark(preds) {
  const svg = dom.accuracySpark;
  svg.innerHTML = '';
  const labelled = preds.filter(p => p.correct !== null && p.correct !== undefined);
  if (labelled.length < 2) return;

  // Compute rolling accuracy across all labelled predictions.
  let correct = 0;
  const pts = [];
  labelled.forEach((p, i) => {
    if (p.correct) correct += 1;
    pts.push(correct / (i + 1));
  });

  const W = 800, H = 60, PAD = 4;
  const xs = i => PAD + (i / (pts.length - 1 || 1)) * (W - 2 * PAD);
  const ys = a => H - PAD - a * (H - 2 * PAD);   // 0 → bottom, 1 → top
  const baseline = ys(0.5);

  const ns = 'http://www.w3.org/2000/svg';
  const bl = document.createElementNS(ns, 'line');
  bl.setAttribute('class', 'baseline');
  bl.setAttribute('x1', PAD); bl.setAttribute('x2', W - PAD);
  bl.setAttribute('y1', baseline); bl.setAttribute('y2', baseline);
  svg.appendChild(bl);

  const poly = document.createElementNS(ns, 'polyline');
  poly.setAttribute('class', 'accline');
  poly.setAttribute('points',
    pts.map((v, i) => `${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(' '));
  svg.appendChild(poly);
}

// ---------------------------------------------------------------------
// SSE live-feed subscription
// ---------------------------------------------------------------------
const SSE_URL = '/api/live/stream';
let liveSse = null;
let liveReconnectDelay = 1000;

function setLiveBadge(connected) {
  if (!dom.liveBadge) return;
  dom.liveBadge.classList.toggle('online', !!connected);
  dom.liveBadge.classList.toggle('offline', !connected);
  dom.liveBadge.textContent = connected ? '● LIVE' : '◌ offline';
  dom.liveBadge.title = connected ? 'connected via same-origin SSE relay' : 'connecting…';
}

function startLiveChartFeed() {
  if (liveSse !== null) return;
  try {
    liveSse = new EventSource(SSE_URL);
  } catch {
    scheduleLiveReconnect();
    return;
  }
  liveSse.addEventListener('hello', () => { liveReconnectDelay = 1000; setLiveBadge(true); });
  liveSse.addEventListener('open',  () => setLiveBadge(true));
  liveSse.addEventListener('message', ev => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    const t = Math.floor(Date.parse(msg.open_time) / 1000);
    const bar = { time: t, open: +msg.open, high: +msg.high, low: +msg.low, close: +msg.close };
    if (!state.series) return;
    try { state.series.update(bar); } catch { /* time-ordering issue, next tick recovers */ }
  });
  liveSse.addEventListener('error', () => {
    setLiveBadge(false);
    if (liveSse && liveSse.readyState === EventSource.CLOSED) {
      liveSse = null; scheduleLiveReconnect();
    }
  });
}
function scheduleLiveReconnect() {
  const wait = liveReconnectDelay;
  liveReconnectDelay = Math.min(liveReconnectDelay * 2, 60_000);
  setTimeout(startLiveChartFeed, wait);
}

// ---------------------------------------------------------------------
// Paper-trader controls
// ---------------------------------------------------------------------
function currentBackfill() {
  const el = document.getElementById('backfill-input');
  const v = el ? parseInt(el.value, 10) : 0;
  return Number.isFinite(v) && v >= 0 ? v : 0;
}

dom.paperStart.addEventListener('click', async () => {
  const n = currentBackfill();
  dom.paperStart.disabled = true;
  dom.paperStatus.textContent = n > 0
    ? `backfilling ${n} historical bars…`
    : `starting…`;
  try {
    await postJson('/api/paper/start', { backfill_bars: n });
    refreshPaper();
  } catch (e) {
    dom.paperStatus.textContent = `start failed: ${e.message}`;
  } finally {
    dom.paperStart.disabled = false;
  }
});
dom.paperStop.addEventListener('click', async () => {
  try { await postJson('/api/paper/stop'); refreshPaper(); }
  catch (e) { dom.paperStatus.textContent = `stop failed: ${e.message}`; }
});
dom.paperReset.addEventListener('click', async () => {
  try { await postJson('/api/paper/reset', { initial_balance: 100 }); refreshPaper(); }
  catch (e) { dom.paperStatus.textContent = `reset failed: ${e.message}`; }
});
dom.reload.addEventListener('click', () => loadCandlesAndRegimes());

// ---------------------------------------------------------------------
// "Analyze visible range" — takes the current chart view, hits
// /api/analyze/range, renders a rich per-window breakdown.
// ---------------------------------------------------------------------
document.getElementById('analyze-btn').addEventListener('click', analyzeVisible);

async function analyzeVisible() {
  const btn = document.getElementById('analyze-btn');
  btn.disabled = true; btn.textContent = 'Analyzing…';
  const panel = document.getElementById('analyze-panel');
  const body  = document.getElementById('analyze-body');
  const label = document.getElementById('analyze-range');
  panel.hidden = false;
  body.innerHTML = '<div class="empty">computing…</div>';

  try {
    const range = state.chart.timeScale().getVisibleRange();
    if (!range || !range.from || !range.to) throw new Error('no visible range');
    const start = secToIso(range.from);
    const end   = secToIso(range.to);
    const a = await API('/api/analyze/range', { start, end });
    label.textContent = `${fmtDate(a.start)}Z → ${fmtDate(a.end)}Z · ${a.bars} bars`;
    renderAnalysis(a);
  } catch (e) {
    body.innerHTML = `<div class="empty">analysis failed: ${e.message}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = '🔬 Analyze visible range';
  }
}

function renderAnalysis(a) {
  const body = document.getElementById('analyze-body');
  const rh = a.regime_histogram || {};
  const total = (rh.bear + rh.sideways + rh.bull + rh.unknown) || 1;
  const seg = (k) => (100 * (rh[k] || 0) / total).toFixed(1);
  const ret = a.ohlc.return_pct;
  const retCls = ret > 0 ? 'pos' : (ret < 0 ? 'neg' : 'mut');
  const news = a.news || { count: 0 };
  const preds = a.predictions || {};
  const acc = preds.overall_accuracy;
  const accCls = acc == null ? 'mut' : (acc > 0.52 ? 'pos' : (acc < 0.48 ? 'neg' : 'mut'));

  const hist = preds.p_up_histogram || [];
  const maxH = Math.max(1, ...hist);
  const bins = hist.length || 1;
  const histBars = hist.map((v, i) => {
    const x = (100 / bins) * i, w = 100 / bins - 1;
    const h = (v / maxH) * 60;
    return `<rect class="hist-bar" x="${x}" y="${70 - h}" width="${w}" height="${h}"/>`;
  }).join('');
  const half = 50;
  const halfLine = `<line x1="${half}" y1="0" x2="${half}" y2="70" stroke="var(--fg-muted)" stroke-width="0.5" stroke-dasharray="2 2"/>`;

  const priceLine = `${fmtUsd(a.ohlc.first_close)} → ${fmtUsd(a.ohlc.last_close)}`;

  body.innerHTML = `
    <div class="stat ${retCls}">
      <div class="lbl">Return</div>
      <div class="val">${(ret * 100).toFixed(2)}%</div>
      <div class="sub">${priceLine}</div>
    </div>

    <div class="stat">
      <div class="lbl">Regime mix</div>
      <div class="regime-bar" title="bear ${seg('bear')}% · sideways ${seg('sideways')}% · bull ${seg('bull')}% · unknown ${seg('unknown')}%">
        <div class="bear"      style="width:${seg('bear')}%"></div>
        <div class="sideways"  style="width:${seg('sideways')}%"></div>
        <div class="bull"      style="width:${seg('bull')}%"></div>
        <div class="unknown"   style="width:${seg('unknown')}%"></div>
      </div>
      <div class="sub">bear ${seg('bear')}% · side ${seg('sideways')}% · bull ${seg('bull')}%</div>
    </div>

    <div class="stat ${accCls}">
      <div class="lbl">Accuracy in range</div>
      <div class="val">${fmtPct(acc, 1)}</div>
      <div class="sub">${preds.n || 0} preds · ${preds.n_traded || 0} traded · ${preds.n_abstained || 0} abstained</div>
    </div>

    <div class="stat">
      <div class="lbl">P(up) distribution</div>
      <svg viewBox="0 0 100 70" preserveAspectRatio="none">${halfLine}${histBars}</svg>
      <div class="sub">bin width ${(1/bins).toFixed(3)}</div>
    </div>

    <div class="stat" style="grid-column: span 4;">
      <div class="lbl">Top news in range · ${news.count} articles · mean sent ${news.mean == null ? '—' : news.mean.toFixed(2)}</div>
      <div class="news-mini">
        ${(news.top || []).map(n => `
          <div class="row">
            <time>${fmtDate(n.published_at)}Z</time>
            <div>
              <span class="src" style="color:var(--fg-muted);margin-right:4px;">${n.source}</span>
              <a href="${n.url}" target="_blank" rel="noopener" style="color:var(--fg);text-decoration:none;">${escapeHtml(n.title)}</a>
              <span class="sent ${n.sent_score > 0.15 ? 'pos' : (n.sent_score < -0.15 ? 'neg' : 'neu')}" style="margin-left:6px;font-size:10px;">${n.sent_score == null ? '—' : (n.sent_score >= 0 ? '+' : '') + n.sent_score.toFixed(2)}</span>
            </div>
          </div>`).join('') || '<div class="empty">no news in range</div>'}
      </div>
    </div>
  `;
}
dom.regimeToggle.addEventListener('change', () => {
  if (state.currentBarTime) loadSimilarDetailed(state.currentBarTime).catch(() => {});
});

async function maybeAutoStart() {
  if (!dom.autostartToggle.checked || state.autoStartAttempted) return;
  state.autoStartAttempted = true;
  try {
    const s = await API('/api/paper/state');
    if (!s.is_running) {
      // Auto-start with the backfill amount the user has set (default 500).
      await postJson('/api/paper/start', { backfill_bars: currentBackfill() });
    }
  } catch (e) {
    // best-effort; user can still click Start manually
    console.warn('auto-start skipped:', e && e.message);
  }
}

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------
async function boot() {
  initChart();
  await loadHealth();
  await loadCandlesAndRegimes();
}

// ---------------------------------------------------------------------
// Auto-retrain widget
// ---------------------------------------------------------------------
function fmtDuration(sec) {
  if (!Number.isFinite(sec) || sec == null) return '—';
  if (sec < 60) return `${Math.round(sec)}s`;
  if (sec < 3600) return `${Math.round(sec / 60)}m`;
  return `${(sec / 3600).toFixed(1)}h`;
}

async function refreshRetrain() {
  try {
    const r = await API('/api/retrain/status');
    renderRetrain(r);
  } catch (e) {
    dom.retrainStatus.textContent = `unavailable`;
  }
}

function renderRetrain(r) {
  const label = r.is_running_refit
    ? 'refit running…'
    : (r.last_run_at ? `last refit ${fmtDate(r.last_run_at)}Z` : 'not yet run');
  dom.retrainStatus.textContent = label;

  const last = r.last_result || {};
  const timings = last.timings_seconds || {};
  const okCls = last.ok === false ? 'neg' : (last.ok === true ? 'pos' : '');
  const rows = [
    ['Auto interval', fmtDuration(r.interval_seconds)],
    ['Next auto-refit in', fmtDuration(r.seconds_to_next_auto_refit)],
    ['Last HMM refit', timings.hmm_s != null ? `${timings.hmm_s}s` : '—'],
    ['Last FAISS rebuild', timings.faiss_s != null ? `${timings.faiss_s}s` : '—'],
    ['Last fusion refit', timings.reload_and_fusion_s != null ? `${timings.reload_and_fusion_s}s` : '—'],
    ['Bars on disk', last.total_bars_now ? last.total_bars_now.toLocaleString() : '—'],
  ];
  dom.retrainSummary.classList.remove('empty');
  dom.retrainSummary.innerHTML = rows.map(([k, v]) =>
    `<div class="k">${k}</div><div class="v ${okCls}">${v}</div>`
  ).join('');
}

dom.retrainBtn.addEventListener('click', async () => {
  dom.retrainBtn.disabled = true;
  dom.retrainBtn.textContent = 'Refitting…';
  try {
    const r = await postJson('/api/retrain/trigger');
    if (!r.ok) {
      dom.retrainStatus.textContent = `errors: ${(r.errors || []).join('; ') || 'see logs'}`;
    }
    await refreshRetrain();
  } catch (e) {
    dom.retrainStatus.textContent = `refit failed: ${e.message}`;
  } finally {
    dom.retrainBtn.disabled = false;
    dom.retrainBtn.textContent = 'Refit now';
  }
});

// Periodic refreshes
setInterval(refreshPaper, 2000);
setInterval(refreshNews, 30_000);
setInterval(refreshRetrain, 15_000);
setInterval(() => {
  if (state.currentBarTime) refreshPredictionAndSimilar(state.currentBarTime);
}, 15_000);

boot()
  .then(() => {
    refreshPaper();
    refreshNews();
    refreshRetrain();
    startLiveChartFeed();
    maybeAutoStart();
  })
  .catch(e => {
    dom.meta.textContent = `boot failed: ${e.message}`;
    console.error(e);
  });
