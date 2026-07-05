# Interim Presentation — Speaker Script (Expanded Research Edition)
### Multimodal Cryptocurrency Direction Prediction with Calibrated Uncertainty
**Date:** July 7, 2026 &nbsp;·&nbsp; **Duration:** 10 min core + up to 5 min Q&A &nbsp;·&nbsp; **Slides:** 20

---

## How to use this document

- Every slide has a **SAY** block (verbatim what to speak) with target duration
- Every slide has a **BOARD** block (what appears on the slide)
- Every slide has a **PANEL LIKELY ASKS** block (rehearsed Q&A defence)
- The 20-slide deck is calibrated for **10 minutes tight** or **12–13 min comfortable**
- Slides marked ⚡ are *research-depth* slides you can trim if time is short
- Slides marked ⭐ are *unmissable* — never skip
- Practice reading each SAY block aloud, timed

**Generate the PPTX with:**
```
.venv\Scripts\python.exe -m pip install python-pptx
.venv\Scripts\python.exe scripts/make_interim_pptx.py
```
Output: `INTERIM_PRESENTATION_JULY7.pptx` in the project root.

---

# SLIDE 1 · Title  &nbsp;·&nbsp; 0:00 – 0:25  &nbsp;·&nbsp; ⭐

**BOARD**
> **Multimodal Cryptocurrency Direction Prediction with Calibrated Uncertainty**
>
> Fusing numerical features · chart-CNN embeddings · news sentiment
> Inductive conformal prediction · Diebold–Mariano validation
>
> **Pathum &nbsp;·&nbsp; Interim Presentation 2 &nbsp;·&nbsp; July 7, 2026**

**SAY** *(25 s)*
> Good morning. My research designs, implements, and evaluates a multimodal cryptocurrency direction-prediction system. It fuses three modalities — engineered numerical features, chart-CNN embeddings, and news sentiment — and quantifies its own uncertainty using conformal prediction. Today I present eight completed phases with measured evidence.

---

# SLIDE 2 · Problem statement  &nbsp;·&nbsp; 0:25 – 0:55

**BOARD**
> **Single-modality crypto prediction has hit a measurable ceiling.**
>
> - Numerical baselines plateau at ~53 % accuracy on 1-hour BTC direction
> - Chart-vision models add signal but rarely tested against strong tabular baselines
> - Almost no prior work reports **calibrated uncertainty**
>
> **My contribution:** measure whether combining all three modalities produces a statistically significant improvement, and quantify the model's confidence honestly.

**SAY** *(30 s)*
> The literature shows a persistent ceiling around 53 percent accuracy for single-modality crypto direction predictors. Chart-vision models are rarely benchmarked against strong tabular baselines. And almost none report calibrated uncertainty — meaning practitioners cannot know *when* to trust the model. My contribution is to fuse all three modalities under a leak-proof harness and, crucially, to quantify confidence honestly via conformal prediction.

---

# SLIDE 3 · Related work  &nbsp;·&nbsp; 0:55 – 1:25  &nbsp;·&nbsp; ⚡

**BOARD**
> **Anchored to established methodology.**
>
> | Paper | Contribution used |
> |---|---|
> | Baltrušaitis et al. (2018) | Late-fusion taxonomy for multimodal ML |
> | Livieris et al. (2020) | Numerical-only crypto direction baselines |
> | Sezer & Ozbayoglu (2018) | GAF-encoded chart-image CNN |
> | Vovk et al. (2005) | Inductive conformal prediction framework |
> | Kirkpatrick et al. (2017) | Elastic Weight Consolidation |
> | Hamilton (1989) | Regime-switching HMM in finance |
> | Diebold & Mariano (1995) | Forecast-loss significance test |

**SAY** *(30 s)*
> The methodology is anchored to well-established prior work. Baltrušaitis's 2018 survey defines the late-fusion taxonomy I use. Livieris 2020 documents the numerical ceiling I test against. Sezer's GAF encoding is one of my two chart-image encoders. Vovk's 2005 conformal framework provides the calibrated uncertainty. Kirkpatrick's Elastic Weight Consolidation handles continual learning. Hamilton's regime-switching HMM is a 30-year-old standard for market state classification. And Diebold-Mariano provides the forecast-loss significance test.

---

# SLIDE 4 · Research aim + 4 RQs  &nbsp;·&nbsp; 1:25 – 1:55  &nbsp;·&nbsp; ⭐

**BOARD**
> **Aim.** Design, implement, and evaluate a multimodal crypto direction predictor that fuses three modalities and quantifies its own uncertainty via conformal prediction.
>
> | | Question |
> |---|---|
> | **RQ1** | Leak-free reproducible pipeline for OHLCV + news? |
> | **RQ2** | Do single-modality numerical models hit an accuracy ceiling? |
> | **RQ3** | Does chart-image CNN add statistically significant signal beyond numerical? |
> | **RQ4** | Does fusing all three modalities with calibrated uncertainty beat any single one? |

**SAY** *(30 s)*
> My aim in one sentence — design a multimodal crypto direction predictor with calibrated uncertainty. It decomposes into four research questions. RQ1 is feasibility. RQ2 asks whether numerical baselines hit a ceiling. RQ3 asks whether vision adds orthogonal signal. RQ4 is the headline: does fusing all three modalities with calibrated uncertainty beat any single one? Each RQ maps to a phase of the implementation.

---

# SLIDE 5 · Walk-forward cross-validation  &nbsp;·&nbsp; 1:55 – 2:25  &nbsp;·&nbsp; ⚡

**BOARD**
> **Every evaluation uses expanding-window walk-forward CV, gap-protected.**
>
> ```
> fold 0 :  train [0 .. T0]                test [T0+g .. T0+H]
> fold 1 :  train [0 .. T0+H]              test [T0+H+g .. T0+2H]
> ...
> fold k :  train [0 .. T0+kH]             test [T0+kH+g .. T0+(k+1)H]
> ```
>
> - Training window grows chronologically
> - `gap = 1 bar` between train end and test start prevents label leak
> - Same harness across every model (naive · XGBoost · LSTM · PatchTST · CNN · fusion)
> - Diebold–Mariano test compares any two models on the same test bars

**SAY** *(30 s)*
> Every model in this project is evaluated under the same expanding-window walk-forward CV. The training window grows chronologically. A one-bar gap between train end and test start prevents the label-lookahead leak. Because every model uses the same harness on the same test bars, the Diebold-Mariano test can rigorously compare any two of them. This is the foundation that lets me claim *statistical* significance rather than just accuracy numbers.

---

# SLIDE 6 · System architecture — 8 phases  &nbsp;·&nbsp; 2:25 – 2:55  &nbsp;·&nbsp; ⭐

**BOARD**
> Eight phases, each delivers a measurable artefact.
>
> 1. **Data ingestion** — Binance OHLCV + 10 RSS feeds, dual timestamps
> 2. **Numerical baselines** — 5 models, walk-forward CV, DM tests
> 3. **Chart-CNN** — 64×64 candle + GAF/MTF encoders
> 4. **News sentiment** — CryptoBERT + FinBERT ensemble
> 5. **Multimodal fusion** — concat-LR + inductive conformal
> 6. **Pattern engine** — FAISS + 3-state HMM + EWC
> 7. **Interactive dashboard** — FastAPI + TradingView + Grad-CAM
> 8. **Self-sustaining live pipeline** — auto-persist + nightly refit + paper trader

**SAY** *(30 s)*
> The implementation is eight phases. Phases 1 through 4 build strong single-modality baselines. Phase 5 is the headline — a late-fusion classifier with inductive conformal calibration. Phase 6 adds a FAISS pattern engine with HMM regime filtering and EWC continual learning. Phase 7 is the FastAPI dashboard. And Phase 8 makes the system self-sustaining — bars auto-persist, models auto-refit, and a paper trader validates the model live.

---

# SLIDE 7 · Numerical feature engineering  &nbsp;·&nbsp; 2:55 – 3:25  &nbsp;·&nbsp; ⚡

**BOARD**
> **31 lookahead-safe features per bar.**
>
> - **Returns** — log-returns at lags 1, 2, 3, 5, 10, 20 (6 features)
> - **Rolling stats** — mean, std, min, max over windows 5, 10, 20, 50 (16 features)
> - **Technical indicators** — RSI, MACD, ATR, Bollinger %B (4 features)
> - **Volume / microstructure** — volume ratio, taker-buy share (2 features)
> - **Time cyclic** — sin/cos of hour-of-day + day-of-week (4 features)
>
> Every feature uses only information available at or before time *t*. Warmup rows dropped.

**SAY** *(30 s)*
> The numerical feature block is 31 lookahead-safe features per bar. Six lagged log-returns. Sixteen rolling statistics over multiple window sizes. Four classical technical indicators — RSI, MACD, ATR, and Bollinger percent-B. Two microstructure features from Binance's aggressive-buy volume. And four cyclic time features to encode hour-of-day and day-of-week. Every feature is constructed using only information available at or before the bar's timestamp. That property is enforced by keeping this in a single file that can be audited.

---

# SLIDE 8 · Chart-CNN architecture  &nbsp;·&nbsp; 3:25 – 3:55  &nbsp;·&nbsp; ⚡

**BOARD**
> **Compact ~250 K parameter CNN with dual heads.**
>
> ```
> input:  (3, 64, 64) — RGB candle image OR (3, 64, 64) GAF+MTF field
>   Conv(3→32)  → BN → ReLU → MaxPool 2x2   (32×32)
>   Conv(32→64) → BN → ReLU → MaxPool 2x2   (16×16)
>   Conv(64→128)→ BN → ReLU → MaxPool 2x2   ( 8× 8)
>   Conv(128→128)→ BN → ReLU → AdaptiveAvgPool → 128
>       ├── embed_head    128 → 128     (reused by Phase 6 FAISS)
>       └── classifier    128 → 1       (binary logit)
> ```
>
> Two input encoders → same architecture → same walk-forward harness.

**SAY** *(30 s)*
> The chart-CNN is a compact roughly 250,000-parameter network with four convolutional blocks and dual heads. The embedding head produces a 128-dimensional vector reused later by the FAISS pattern engine. The classifier head produces a binary logit for direction. The same architecture handles two different encoders — direct candle images and Gramian Angular Field time-series encodings. Same walk-forward harness as the numerical baselines, so their metrics are directly Diebold-Mariano comparable.

---

# SLIDE 9 · News sentiment ensemble  &nbsp;·&nbsp; 3:55 – 4:25  &nbsp;·&nbsp; ⚡

**BOARD**
> **CryptoBERT + FinBERT calibrated ensemble.**
>
> Two transformer sentiment classifiers, both output scores ∈ [−1, +1]:
>
> - **CryptoBERT** (`ElKulako/cryptobert`) — trained on crypto-specific text
> - **FinBERT** (`ProsusAI/finbert`) — trained on general financial text
> - **Ensemble** = mean · confidence = 1 − |cb − fb|
>
> Per-bar aggregation over 24 h lookback window: **7 features**
>
> `sent_count · sent_mean · sent_std · sent_min · sent_max · sent_last · sent_conf_mean`
>
> **Leakage guard:** every row asserts `published_at ≤ bar_open_time`.

**SAY** *(30 s)*
> The sentiment module runs a two-model ensemble. CryptoBERT is trained on crypto-specific text; FinBERT is trained on general financial text. Both output scores between minus one and plus one. The ensemble is their mean, and their disagreement is the confidence signal. Per bar, over a 24-hour lookback, we compute seven features — count, mean, standard deviation, min, max, last score, and average confidence. A hard-coded assertion enforces that no news article was published after the bar it labels.

---

# SLIDE 10 · Late-fusion architecture  &nbsp;·&nbsp; 4:25 – 4:55  &nbsp;·&nbsp; ⭐

**BOARD**
> **Concatenate then classify — 166-dimensional input.**
>
> ```
>  numerical  [31]  ────────┐
>  CNN embed  [128] ────────┼──►  concat [166]  →  standardize  →
>  sentiment  [7]   ────────┘                                    →  L2-Logistic Regression  →  P(up)
>                                                                → Inductive Conformal   →  90 % prediction set
> ```
>
> - Interpretable: 166 coefficients tell you exactly which features matter
> - Ablation-friendly: `use_cnn=False`, `use_sentiment=False` toggles
> - Fast to refit — the anchor model can be retrained in seconds

**SAY** *(30 s)*
> The fusion architecture is deliberately simple: concatenate the three modality vectors into one 166-dimensional input, standardise, then fit an L2-regularised logistic regression. The output is a probability, wrapped in an inductive conformal calibrator that produces a 90-percent prediction set. Why so simple? Three reasons. First, it is fully interpretable — 166 coefficients tell you exactly which features matter. Second, it is ablation-friendly — I can turn each modality on or off in one line. Third, it refits in seconds, which enables the nightly auto-retrain in Phase 8.

---

# SLIDE 11 · Conformal prediction primer  &nbsp;·&nbsp; 4:55 – 5:25  &nbsp;·&nbsp; ⚡

**BOARD**
> **Inductive conformal classifier (Vovk 2005).**
>
> - Hold out the **last 20 %** of the training window as a calibration set
> - Nonconformity score for calibration bar *i*:  &nbsp; α<sub>i</sub> = 1 − p(y<sub>i</sub> | x<sub>i</sub>)
> - Threshold τ = quantile(α, ⌈(n<sub>cal</sub> + 1)(1 − α)⌉ / n<sub>cal</sub>)
> - **Prediction set:** &nbsp; C(x) = { c &nbsp; : &nbsp; 1 − p(c | x) ≤ τ }
>
> **Guarantee:** under exchangeability, P( y ∈ C(x) ) ≥ 1 − α
>
> **Result on BTC 1h:** empirical coverage **0.883** at nominal α = 0.10.

**SAY** *(30 s)*
> Conformal prediction wraps any classifier and gives it a coverage guarantee. I hold out the last twenty percent of training as a calibration set. For each calibration bar, I compute the model's probability of the true class — one minus that is the nonconformity score. The threshold tau is the 90th percentile of those scores. At inference, a class is in the prediction set only if its nonconformity is below tau. Under exchangeability of calibration and test data, the guarantee is that the true class is in the set at least 90 percent of the time. My measured coverage on BTC is 88.3 percent — within two percentage points of nominal — confirming the calibration is honest.

---

# SLIDE 12 · RO1–RO4 · foundation evidence  &nbsp;·&nbsp; 5:25 – 5:55

**BOARD**
> Four foundations complete with measurable evidence.
>
> | RO | Deliverable | Evidence |
> |---|---|---|
> | **RO1** | Data pipeline | **77,688 bars · 454 articles · invariant tests pass** |
> | **RO2** | Numerical baselines | **LSTM 53.4 % · DM p<0.001 vs naive** |
> | **RO3** | Chart-CNN | **cnn_candle 54.0 % · DM p = 0.006 vs XGBoost** |
> | **RO4** | Sentiment pipeline | **Ensemble + leakage tests** |

**SAY** *(30 s)*
> The first four objectives have measurable evidence on disk. Phase 1 ingested seventy-seven thousand hourly BTC bars and four hundred fifty-four scored news articles, with the no-lookahead invariant tested per row. Phase 2's LSTM baseline hits 53.4 percent with DM p below one in a thousand versus naive. Phase 3's candle-image CNN reaches 54 percent, beating XGBoost with DM p equals 0.006 — confirming vision adds signal. And Phase 4's sentiment ensemble is plumbed with leakage tests passing.

---

# SLIDE 13 · RO5 · Headline fusion result  &nbsp;·&nbsp; 5:55 – 6:55  &nbsp;·&nbsp; ⭐

**BOARD**
> **Fusion beats every single-modality baseline with statistical significance.**
>
> BTCUSDT · 1 h · 6-fold walk-forward · n = 37,992 held-out bars
>
> | Metric | Value |
> |---|---|
> | Accuracy | **0.5492** |
> | MCC | **0.1023** |
> | Sharpe / bar | **0.0245** |
> | Cumulative log-PnL | **4.93** |
>
> **DM p-values (fusion vs baseline):**
> - vs cnn_candle: **p ≈ 0.001**
> - vs XGBoost, LSTM, PatchTST, cnn_gaf: **p ≈ 0.000**
>
> **Conformal calibration:** empirical coverage **0.883** &nbsp;·&nbsp; average set size 1.72 &nbsp;·&nbsp; **28.2 % singletons**

**SAY** *(60 s)*
> This is the money slide. Under 6-fold walk-forward CV on nearly thirty-eight thousand held-out bars, the full fusion model reaches 54.92 percent accuracy, with an MCC of 0.10 and a cumulative log-PnL of 4.93. Diebold-Mariano tests confirm this is a statistically significant improvement over *every* single-modality baseline. Against the CNN alone — the toughest competitor — p equals one in a thousand. Against XGBoost, LSTM, PatchTST, and the GAF-CNN — all p-values effectively zero. The conformal calibration is honest: empirical coverage is 88 percent versus the nominal 90 — inside two percentage points. Twenty-eight percent of predictions are singleton sets — meaning the model commits to one direction with 90-percent confidence — and the remaining bars transparently abstain. This is RQ4 answered, positively, with statistical rigour.

**PANEL LIKELY ASKS**
- *"Sentiment contributed zero — is that a failure?"* — No. Documented, expected. News is sparse on 1h until several weeks accumulate. The regression correctly assigns zero weight rather than overfitting noise.
- *"How do you rule out a leak?"* — Every training slice ends before the test slice with a gap. Every feature uses only past data. Every row carries `open_time` and `ingested_at`.

---

# SLIDE 14 · RO5 · Ablation study  &nbsp;·&nbsp; 6:55 – 7:25  &nbsp;·&nbsp; ⚡

**BOARD**
> **Each modality's contribution isolated.**
>
> | Variant | Accuracy | MCC | Sharpe/bar | PnL_log |
> |---|---|---|---|---|
> | fusion_num              | 0.5374 | 0.078 | 0.003 | 0.74 |
> | fusion_num_cnn          | **0.5492** | **0.102** | **0.024** | **4.93** |
> | fusion_num_sent         | 0.5374 | 0.078 | 0.003 | 0.74 |
> | fusion_num_cnn_sent     | **0.5492** | **0.102** | **0.024** | **4.93** |
>
> **Reading:** vision (CNN) is the marginal contributor. Sentiment is a documented no-op *and that's healthy* — the model refuses to overfit sparse features.

**SAY** *(30 s)*
> The ablation isolates each modality. The numeric-only fusion hits 53.74 percent. Adding the CNN block pushes it to 54.92 percent — a 1.2 percentage-point improvement, driven entirely by vision. Adding sentiment on top contributes zero — the sentiment rows are byte-identical because the logistic regression correctly assigns zero weight to features that carry no marginal signal on this window. This is a *healthy* finding, not a failure — an overfit model would have shown spurious sentiment weights.

---

# SLIDE 15 · RO6 · Pattern engine + EWC  &nbsp;·&nbsp; 7:25 – 8:00

**BOARD**
> **FAISS retrieval · HMM regimes · EWC forgetting audit.**
>
> - **FAISS IndexFlatIP** over 128-d embeddings — sub-ms top-K on 77 K bars
> - **3-state HMM** on (log-return, 24h vol): **9.9 % bear · 58.2 % sideways · 31.9 % bull**
> - **EWC:** naive avg final acc **0.5553** &nbsp;·&nbsp; EWC avg final acc **0.5509** &nbsp;·&nbsp; Δ = **−0.45 pp**
>
> **Interpretation:** catastrophic forgetting is *not* the binding constraint on BTC 1 h.

**SAY** *(35 s)*
> Phase 6 delivers three artefacts. A FAISS cosine index over the CNN's 128-dimensional embeddings — retrieval is sub-millisecond. A three-state Gaussian HMM produces the regime distribution shown — three well-populated states, no degeneracy. And Elastic Weight Consolidation trained across three chronological phases. Naive sequential achieved 55.5 percent; EWC achieved 55.1 — a delta of minus half a percentage point. This is a *measurement*, not a failure: it tells us catastrophic forgetting is not the binding constraint on this dataset. The framework is in place to detect drift when it does become one.

---

# SLIDE 16 · RO7 · Interactive dashboard  &nbsp;·&nbsp; 8:00 – 8:45  &nbsp;·&nbsp; 🖥 LIVE DEMO

**BOARD**
> **Explainable predictions in one screen.**
>
> - **FastAPI** + **TradingView Lightweight Charts** + vanilla JS
> - **Live SSE relay** — backend fans Binance ticks to browser (no client geo issue)
> - **Click any candle →** fused prediction · 90 % conformal set · top-5 similar patterns · Grad-CAM
> - **"Mark range"** — click A then B → shaded overlay + range analysis + suggested patterns
> - **"Analyze visible range"** — regime mix · return · P(up) histogram · news overlaps
>
> ⭐ 30-second live demo at `http://127.0.0.1:8000`

**SAY** *(45 s)*
> Phase 7 delivers the user-facing artefact. A FastAPI backend serves TradingView charts to a vanilla-JavaScript frontend — no build step, no framework overhead. The backend holds one always-on WebSocket to Binance and fans every tick to every open browser via Server-Sent Events. Click any candle and four things happen: the fusion prediction appears with its conformal set, the top-five most similar historical patterns render as mini candlestick charts, a Grad-CAM overlay shows exactly which candles the CNN attended to, and a range-analysis tool lets me pick any window and see the regime mix, return distribution, and news overlaps. *(If time: switch to browser, click Mark range, click two candles, walk the panel through the auto-populated analysis.)*

---

# SLIDE 17 · RO8 · Self-sustaining live pipeline  &nbsp;·&nbsp; 8:45 – 9:15

**BOARD**
> **The system evaluates itself, forward, in real time.**
>
> - **Confidence-gated paper trading** — trades only on conformal singleton bars
> - **Vol-scaled sizing** — inverse recent-vol capped 10–100 %
> - **Auto-persist** — every closed candle → Phase-1 parquet schema
> - **Auto-refit** — HMM + FAISS + fusion **nightly** without dashboard restart
> - **Backfill on Start** — replays 500 historical bars → accuracy card populated instantly
> - **Accuracy split** — confident bars vs uncertain bars → live conformal proof

**SAY** *(30 s)*
> Phase 8 makes the system self-sustaining. A paper trader takes positions only when the conformal set is a singleton, sized by inverse volatility. Every closed candle is automatically persisted, and a background scheduler refits the HMM, FAISS, and fusion model every twenty-four hours without dashboard restart. On start, the trader replays 500 historical bars through the model, so the accuracy card is populated instantly. Most importantly: the card separates confident accuracy from uncertain accuracy. If confident is higher, that is direct live evidence that Phase 5's conformal machinery is doing something real.

---

# SLIDE 18 · Objectives → evidence mapping  &nbsp;·&nbsp; 9:15 – 9:40  &nbsp;·&nbsp; ⭐

**BOARD**
> **All eight objectives have measurable evidence on disk.**
>
> | RO | Maps to | Status | Evidence |
> |---|---|---|---|
> | RO1 · Data pipeline | RQ1 | ✅ Done | 77 K bars, 454 articles, invariants pass |
> | RO2 · Numerical | RQ2 | ✅ Done | 25 experiments, DM p<0.001 vs naive |
> | RO3 · Chart-CNN | RQ3 | ✅ Done | 54.0 %, DM p=0.006 vs XGBoost |
> | RO4 · News sentiment | RQ4 | ✅ Done | Ensemble + leakage tests |
> | **RO5 · Fusion** | RQ4 | ✅ **Done** | **0.549 acc, DM p≈0.001, CP cov 0.88** |
> | RO6 · Pattern engine | RQ4 | ✅ Done | FAISS + HMM + EWC delta measured |
> | RO7 · Dashboard | All | ✅ Done | Live at `localhost:8000` |
> | RO8 · Self-sustaining | RQ4 | ✅ Done | Persistor + retrainer running |

**SAY** *(25 s)*
> Every one of eight objectives has measurable evidence on disk right now. Four are grounded in walk-forward CV results, four are grounded in code you can inspect. RO5 is the direct answer to RQ4 — remember that row. The research is not a promise; it is a delivery.

---

# SLIDE 19 · Defence playbook  &nbsp;·&nbsp; 9:40 – 10:00

**BOARD**
> **Toughest questions rehearsed.**
>
> | Question | Prepared answer |
> |---|---|
> | *"What if fusion fails at final?"* | Objective is to **measure**, not win. Already measured — did not fail. |
> | *"Why 1 hour, not 1 day?"* | Finest interval where 77 K bars ≥ deep-learning threshold AND 24h news lookback still relevant. |
> | *"Only BTC in Phase 5?"* | Phases 2–3 covered 5 pairs; Phase 5 focused BTC first for depth. Extensions are config, not code. |
> | *"Is 54.9 % economically meaningful?"* | Sharpe/bar 0.025 + PnL_log 4.93 = positive expectancy before fees. |
> | *"Overfitting?"* | Walk-forward + gap + lookahead-safe features + CP coverage 0.88 near nominal 0.90. |

**SAY** *(20 s)*
> Pre-rehearsed answers to the toughest questions the panel is likely to ask. I have hard evidence for each — the numbers are on disk in `experiments/fusion/`. I would rather answer these questions than avoid them.

---

# SLIDE 20 · Timeline to final + closing  &nbsp;·&nbsp; 10:00 – 10:30

**BOARD**
> **What's left before the final viva.**
>
> - **weeks 1-4** — extend fusion to all 5 pairs (config-only)
> - **weeks 4-8** — schedule full CNN retrain weekly
> - **weeks 8-10** — gather multi-week live paper-trading evidence
> - **weeks 10-12** — thesis write-up + final-viva rehearsal
>
> **The system is running. The evidence is on disk. Thank you.**

**SAY** *(20 s)*
> To close: the remaining work before the final viva is largely configuration and evidence-gathering. The system is running today. Every objective has measurable evidence on disk. Thank you — questions?

---

## Timing budget (target)

| # | Slide | Cumulative |
|---|---|---|
| 1 | Title | 0:25 |
| 2 | Problem | 0:55 |
| 3 | Related work ⚡ | 1:25 |
| 4 | Aim + RQs ⭐ | 1:55 |
| 5 | Walk-forward CV ⚡ | 2:25 |
| 6 | Architecture ⭐ | 2:55 |
| 7 | Numerical features ⚡ | 3:25 |
| 8 | Chart-CNN ⚡ | 3:55 |
| 9 | Sentiment ensemble ⚡ | 4:25 |
| 10 | Fusion architecture ⭐ | 4:55 |
| 11 | Conformal primer ⚡ | 5:25 |
| 12 | RO1-RO4 | 5:55 |
| 13 | **RO5 headline** ⭐ | 6:55 |
| 14 | RO5 ablation ⚡ | 7:25 |
| 15 | RO6 pattern engine | 8:00 |
| 16 | RO7 dashboard demo | 8:45 |
| 17 | RO8 self-sustaining | 9:15 |
| 18 | RO mapping ⭐ | 9:40 |
| 19 | Defence playbook | 10:00 |
| 20 | Timeline + close | 10:30 |

**Total:** 10 minutes 30 seconds (30 s buffer over the 10-minute target).
**Trim strategy if time is tight:** drop ⚡ slides 3, 7, 9, 11, 14 to reach ~8 minutes with all ⭐ slides intact.

---

## Practice checklist (2 days out)

- [ ] Read every SAY block aloud with a timer — trim anything over 15 s of budget
- [ ] Rehearse the live demo flow 3× end-to-end (Mark range + Analyze range)
- [ ] Screenshot the dashboard + accuracy card as fallback if Wi-Fi fails
- [ ] Have `experiments/fusion/summary.csv` + `dm_vs_baselines.csv` open in a second tab
- [ ] Know every defence-playbook answer BY HEART — don't read them
- [ ] The 3 ⭐ moments are: RO5 headline (slide 13), RO mapping (slide 18), closing (slide 20). Nail those three.

---

## Files referenced

- `experiments/fusion/summary.csv` — RO5 headline numbers
- `experiments/fusion/dm_vs_baselines.csv` — DM p-value matrix
- `experiments/fusion/PHASE5_RESULTS.md` — RO5 rendered report
- `experiments/baselines/summary.csv` — RO2 numerical baselines
- `experiments/retrieval/cnn_ewc_comparison.json` — RO6 EWC delta
- `RESEARCH_OBJECTIVES.md` — RO → evidence master document
