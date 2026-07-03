// Phase 7 dashboard — single-page UI.
//
// Loads candles + regimes, renders a TradingView Lightweight Charts
// candle series, and lets the user click any bar to fetch the fused
// prediction, the K most-similar historical bars, and the Grad-CAM PNG
// for that bar's chart window.

const API = (path, params = {}) => {
  const url = new URL(path, window.location.origin);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v);
  }
  return fetch(url).then(r => {
    if (!r.ok) {
      return r.json().then(j => { throw new Error(j.detail || r.statusText); });
    }
    return r.json();
  });
};

const REGIME_CLASS = { 0: 'bear', 1: 'sideways', 2: 'bull' };

// ---------------------------------------------------------------------
// State + DOM handles
// ---------------------------------------------------------------------
const state = {
  candles: [],
  regimes: [],
  chart: null,
  series: null,
  selected: null,
};

const dom = {
  meta: document.getElementById('meta'),
  chart: document.getElementById('chart'),
  regimeStrip: document.getElementById('regime-strip'),
  limit: document.getElementById('limit-select'),
  end: document.getElementById('end-input'),
  reload: document.getElementById('reload-btn'),
  prediction: document.getElementById('prediction-body'),
  similar: document.getElementById('similar-body'),
  regimeToggle: document.getElementById('regime-toggle'),
  gradcam: document.getElementById('gradcam-body'),
  paperStart: document.getElementById('paper-start-btn'),
  paperStop: document.getElementById('paper-stop-btn'),
  paperReset: document.getElementById('paper-reset-btn'),
  paperStatus: document.getElementById('paper-status'),
  paperSummary: document.getElementById('paper-summary'),
  equitySpark: document.getElementById('equity-spark'),
  paperTrades: document.getElementById('paper-trades'),
  liveBadge: document.getElementById('live-badge'),
};

// ---------------------------------------------------------------------
// Chart setup
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
    // ``param.time`` is the UNIX seconds we assigned per candle.
    const ts = Number(param.time);
    onCandleClick(ts);
  });
  state.chart = chart;
  state.series = series;
}

// Convert an ISO open_time → UNIX seconds (Lightweight Charts time format).
function isoToSec(iso) { return Math.floor(Date.parse(iso) / 1000); }
function secToIso(sec) { return new Date(sec * 1000).toISOString(); }

// ---------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------
async function loadHealth() {
  const h = await API('/api/health');
  const range = `${h.time_range_start.slice(0,10)} → ${h.time_range_end.slice(0,10)}`;
  dom.meta.textContent =
    `${h.pair} · ${h.interval} · ${h.total_bars.toLocaleString()} bars · ${h.n_features} feats · ${range} · anchor cut @ row ${h.anchor_cut_row.toLocaleString()}`;
}

async function loadCandlesAndRegimes() {
  const limit = Number(dom.limit.value) || 500;
  const endIso = dom.end.value.trim() || undefined;

  const [candles, regimes] = await Promise.all([
    API('/api/candles', { end: endIso, limit }),
    API('/api/regimes', { end: endIso, limit }),
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
}

function renderRegimeStrip(regimes) {
  dom.regimeStrip.innerHTML = '';
  for (const r of regimes) {
    const div = document.createElement('div');
    div.className = `cell ${REGIME_CLASS[r.regime] || 'sideways'}`;
    div.title = `${r.regime_name}  ${r.open_time}`;
    dom.regimeStrip.appendChild(div);
  }
}

// ---------------------------------------------------------------------
// Click handler
// ---------------------------------------------------------------------
async function onCandleClick(timeSec) {
  const iso = secToIso(timeSec);
  state.selected = iso;
  dom.prediction.innerHTML = `<div class="kv"><span class="k">querying...</span></div>`;
  dom.similar.innerHTML = '<div class="empty">querying…</div>';
  dom.gradcam.innerHTML = '<div class="empty">rendering…</div>';

  await Promise.all([
    runPredict(iso),
    runSimilar(iso),
    runGradcam(iso),
  ]);
}

async function runPredict(iso) {
  try {
    const p = await API('/api/predict', { open_time: iso });
    dom.prediction.innerHTML = renderPrediction(p);
  } catch (e) {
    dom.prediction.innerHTML = `<div class="empty">predict failed: ${e.message}</div>`;
  }
}

function renderPrediction(p) {
  const dirClass = p.label === 1 ? 'up' : 'down';
  const cpDown = p.conformal_set.down ? '<span class="pill down">↓</span>' : '<span style="opacity:0.3">—</span>';
  const cpUp = p.conformal_set.up ? '<span class="pill up">↑</span>' : '<span style="opacity:0.3">—</span>';
  const regimePill = p.regime !== null
    ? `<span class="pill ${REGIME_CLASS[p.regime] || 'sideways'}">${p.regime_name}</span>`
    : `<span class="empty">—</span>`;

  const inDemo = p.is_in_demo_window
    ? `<span class="pill" style="background:rgba(38,166,154,0.18);color:var(--bull)">held-out</span>`
    : `<span class="pill warn">in fit window</span>`;

  return `
    <div class="kv"><span class="k">Bar</span><span class="v">${p.open_time.replace('+00:00','Z')}</span></div>
    <div class="kv"><span class="k">P(up)</span><span class="v">${p.p_up.toFixed(4)}</span></div>
    <div class="kv"><span class="k">Direction</span><span class="v"><span class="pill ${dirClass}">${p.label === 1 ? 'UP' : 'DOWN'}</span></span></div>
    <div class="kv"><span class="k">Conformal 90% set</span><span class="v">${cpDown} ${cpUp}</span></div>
    <div class="kv"><span class="k">Regime</span><span class="v">${regimePill}</span></div>
    <div class="kv"><span class="k">Window</span><span class="v">${inDemo}</span></div>
  `;
}

async function runSimilar(iso) {
  const regimeFilter = dom.regimeToggle.checked;
  try {
    const s = await API('/api/similar', {
      open_time: iso, k: 10, regime_filter: regimeFilter,
    });
    dom.similar.innerHTML = renderSimilar(s);
    attachMatchClicks();
  } catch (e) {
    dom.similar.innerHTML = `<div class="empty">similar failed: ${e.message}</div>`;
  }
}

function renderSimilar(s) {
  if (s.matches.length === 0) {
    return `<div class="empty">no matches (try unchecking the regime filter)</div>`;
  }
  return s.matches.map((m, i) => {
    const cls = REGIME_CLASS[m.regime] || 'sideways';
    return `
      <div class="match-row" data-ts="${m.open_time}">
        <time>${m.open_time.replace('+00:00','Z').replace('T',' ').slice(0,16)}</time>
        <span class="sim">${m.similarity.toFixed(4)}</span>
        <span class="pill ${cls}">${m.regime_name}</span>
      </div>
    `;
  }).join('');
}

function attachMatchClicks() {
  for (const el of dom.similar.querySelectorAll('.match-row')) {
    el.addEventListener('click', () => {
      const ts = el.getAttribute('data-ts');
      const sec = isoToSec(ts);
      // Try to centre the chart on the matched bar (best-effort: only
      // works if the bar is already loaded into the candle series).
      state.chart.timeScale().scrollToPosition(0, false);
      onCandleClick(sec);
    });
  }
}

async function runGradcam(iso) {
  try {
    const r = await fetch(`/api/gradcam?open_time=${encodeURIComponent(iso)}`);
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail || r.statusText);
    }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    dom.gradcam.innerHTML = `<img src="${url}" alt="Grad-CAM" />`;
  } catch (e) {
    dom.gradcam.innerHTML = `<div class="empty">grad-cam unavailable: ${e.message}</div>`;
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

dom.reload.addEventListener('click', () => loadCandlesAndRegimes());
dom.regimeToggle.addEventListener('change', () => {
  if (state.selected) runSimilar(state.selected);
});

// -------------------------------------------------------------------
// Phase 8 — live paper-trading panel
// -------------------------------------------------------------------
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

dom.paperStart.addEventListener('click', async () => {
  try { await postJson('/api/paper/start'); refreshPaper(); }
  catch (e) { dom.paperStatus.textContent = `start failed: ${e.message}`; }
});
dom.paperStop.addEventListener('click', async () => {
  try { await postJson('/api/paper/stop'); refreshPaper(); }
  catch (e) { dom.paperStatus.textContent = `stop failed: ${e.message}`; }
});
dom.paperReset.addEventListener('click', async () => {
  try { await postJson('/api/paper/reset', { initial_balance: 100 }); refreshPaper(); }
  catch (e) { dom.paperStatus.textContent = `reset failed: ${e.message}`; }
});

async function refreshPaper() {
  try {
    const s = await API('/api/paper/state');
    renderPaperPanel(s);
  } catch (e) {
    dom.paperStatus.textContent = `paper state unavailable: ${e.message}`;
  }
}

// Stricter than the global `isFinite`, which treats null as 0.
function isNum(x) { return typeof x === 'number' && Number.isFinite(x); }

function fmtUsd(n) {
  if (!isNum(n)) return '—';
  return `${n < 0 ? '-' : ''}$${Math.abs(n).toFixed(2)}`;
}
function fmtPct(x, places = 2) {
  if (!isNum(x)) return '—';
  return `${(x * 100).toFixed(places)}%`;
}
function fmtNum(x, places = 3) {
  if (!isNum(x)) return '—';
  return x.toFixed(places);
}

function renderPaperPanel(s) {
  // Status line
  if (s.is_running) {
    dom.paperStatus.classList.add('running');
    const streamOk = s.stream && s.stream.connected ? '🟢' : '🟡';
    dom.paperStatus.textContent =
      `${streamOk} running · ${s.bars_seen} bars seen · ${s.bars_traded} trades · ${s.bars_abstained} abstained`;
  } else {
    dom.paperStatus.classList.remove('running');
    dom.paperStatus.textContent = s.bars_seen === 0
      ? 'not started · click Start to subscribe to Binance live feed'
      : `stopped after ${s.bars_seen} bars`;
  }

  // Summary grid — every value is `null`-safe so a pre-start snapshot
  // (no trades, no signal, NaN stats) doesn't crash the formatter.
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
  let signalLine = '';
  if (s.last_signal) {
    const ls = s.last_signal;
    const dec = (ls.decision || '?').toUpperCase();
    const pUp = fmtNum(ls.p_up);
    signalLine = `
      <div class="k">Last signal</div>
      <div class="v">${dec} · P(up)=${pUp}</div>`;
  }

  dom.paperSummary.innerHTML = `
    <div class="k">Balance</div><div class="v">${fmtUsd(balance)}</div>
    <div class="k">P&amp;L</div><div class="v ${pnlClass}">${fmtUsd(pnl)}</div>
    <div class="k">Win rate (last 50)</div><div class="v">${winRate}</div>
    <div class="k">Sharpe/trade</div><div class="v">${sharpe}</div>
    <div class="k">Verdict</div><div class="v ${verdictClass}">${verdict}</div>
    ${openLine}
    ${signalLine}
  `;

  // Sparkline
  drawEquitySpark(s.equity_curve, s.initial_balance);

  // Recent trades
  const trades = s.recent_trades || [];
  if (trades.length === 0) {
    dom.paperTrades.innerHTML = '<div class="empty">no closed trades yet</div>';
    return;
  }
  dom.paperTrades.innerHTML = trades.slice().reverse().slice(0, 10).map(t => {
    const pnl = isNum(t.pnl_dollars) ? t.pnl_dollars : 0;
    const cls = pnl > 0 ? 'pos' : 'neg';
    const dir = t.direction || '?';
    const ex = (t.exit_time || '')
      .replace('+00:00', 'Z').replace('T', ' ').slice(0, 16);
    return `
      <div class="trade-row">
        <span class="dir ${dir === 'long' ? 'pos' : 'neg'}">${dir}</span>
        <time>${ex}</time>
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
  const xs = (i) => PAD + (i / (curve.length - 1 || 1)) * (W - 2*PAD);
  const ys = (b) => H - PAD - ((b - minB) / range) * (H - 2*PAD);

  const baseline = ys(baseB);
  const ns = 'http://www.w3.org/2000/svg';

  const bl = document.createElementNS(ns, 'line');
  bl.setAttribute('class', 'baseline');
  bl.setAttribute('x1', PAD); bl.setAttribute('x2', W - PAD);
  bl.setAttribute('y1', baseline); bl.setAttribute('y2', baseline);
  svg.appendChild(bl);

  const pts = curve.map((p, i) => `${xs(i).toFixed(1)},${ys(p.balance).toFixed(1)}`).join(' ');
  const poly = document.createElementNS(ns, 'polyline');
  poly.setAttribute('class', 'line');
  poly.setAttribute('points', pts);
  svg.appendChild(poly);
}

// Poll the paper-trader state every 2 s.
setInterval(refreshPaper, 2000);

// -------------------------------------------------------------------
// Live chart feed via Server-Sent Events (same origin).
// -------------------------------------------------------------------
// The FastAPI backend holds a single always-on WebSocket to Binance
// and fans every kline tick out to any /api/live/stream client. The
// browser subscribes with EventSource — a native, same-origin,
// firewall-friendly protocol — so no ad blocker / geo block / corp
// proxy can prevent live updates as long as the browser can reach
// localhost:8000, which it obviously can.
// -------------------------------------------------------------------
const SSE_URL = '/api/live/stream';
let liveSse = null;
let liveReconnectDelay = 1000;   // ms

function setLiveBadge(connected) {
  if (!dom.liveBadge) return;
  dom.liveBadge.classList.toggle('online', !!connected);
  dom.liveBadge.classList.toggle('offline', !connected);
  dom.liveBadge.textContent = connected ? '● LIVE' : '◌ offline';
  dom.liveBadge.title = connected
    ? 'connected via same-origin SSE relay'
    : 'connecting…';
}

function startLiveChartFeed() {
  if (liveSse !== null) return;
  try {
    liveSse = new EventSource(SSE_URL);
  } catch (e) {
    console.error('SSE construction failed', e);
    scheduleLiveReconnect();
    return;
  }

  liveSse.addEventListener('hello', () => {
    liveReconnectDelay = 1000;
    setLiveBadge(true);
  });

  liveSse.addEventListener('open', () => {
    // The 'hello' event above is the more informative signal; but if
    // proxies strip named events, at least fall back to the raw open.
    setLiveBadge(true);
  });

  liveSse.addEventListener('message', ev => {
    let msg;
    try { msg = JSON.parse(ev.data); }
    catch { return; }

    const t = Math.floor(Date.parse(msg.open_time) / 1000);
    const bar = {
      time:  t,
      open:  Number(msg.open),
      high:  Number(msg.high),
      low:   Number(msg.low),
      close: Number(msg.close),
    };
    if (!state.series) return;
    try {
      state.series.update(bar);
    } catch (e) {
      console.debug('live update skipped', e && e.message);
    }
  });

  liveSse.addEventListener('error', () => {
    setLiveBadge(false);
    // EventSource auto-reconnects, but if the server was restarted we
    // want to close and reopen for a clean state.
    if (liveSse && liveSse.readyState === EventSource.CLOSED) {
      liveSse = null;
      scheduleLiveReconnect();
    }
  });
}

function scheduleLiveReconnect() {
  const wait = liveReconnectDelay;
  liveReconnectDelay = Math.min(liveReconnectDelay * 2, 60_000);
  setTimeout(startLiveChartFeed, wait);
}

boot()
  .then(() => {
    refreshPaper();
    startLiveChartFeed();
  })
  .catch(e => {
    dom.meta.textContent = `boot failed: ${e.message}`;
    console.error(e);
  });
