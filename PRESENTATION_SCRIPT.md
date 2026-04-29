# Presentation Script — Multimodal Crypto Direction Prediction

**Total length:** ~14–15 minutes of speaking + ~5 minutes Q&A.
**Pace:** Slow down. Use the natural pauses marked with `[pause]`.
**Practice tip:** Read it aloud once tonight, then twice tomorrow morning. Don't memorise word-for-word — memorise the *flow* of each slide.

---

## Slide 1 — Title (~30 seconds)

> Good morning everyone. Thank you for being here.
>
> My research project is called **Multimodal Crypto Direction Prediction**.
> The full system combines three things — a chart-based CNN, news sentiment from large language models, and an interactive pattern-matching tool. It's evaluated on real Binance market data, and every prediction it makes comes with a calibrated confidence score.
>
> [pause]
>
> My name is Pathum, and over the next 15 minutes I'll walk you through the problem, the solution, and the work I've already completed.

---

## Slide 2 — Agenda (~30 seconds)

> Here's what I'll cover today.
>
> First the introduction and the problem we're solving. Then my objectives. After that, a short literature review and the research gap I'm filling. Then I'll explain the proposed solution and the methodology I'll use. Finally I'll show the timeline and the progress I've already made — three of the seven phases are already done — and then we'll have time for questions.

---

## Slide 3 — Introduction & Background (~1 minute)

> Cryptocurrency markets are a unique domain.
>
> They trade twenty-four hours a day, seven days a week. They are extremely volatile. And they are influenced by a mix of three very different signals — technical patterns from the price charts, on-chain blockchain events, and news that can spread across the world in minutes.
>
> [pause]
>
> Predicting whether the next price candle will go up or down is a research problem that draws on three traditions in machine learning.
>
> First, **numerical time-series models** like ARIMA, XGBoost, and Transformers, which look at the numbers directly.
>
> Second, **computer vision** — using convolutional neural networks on chart images, the same way a human trader scans a chart visually.
>
> And third, **natural language processing** — models like FinBERT and CryptoBERT that read news headlines and score them as positive or negative.
>
> Each of these traditions works on its own, but they have rarely been combined. That's where my project sits.

---

## Slide 4 — Why This Topic Is Important Now (~1 minute)

> Why does this matter today?
>
> Four reasons.
>
> **One — retail exposure.** Hundreds of millions of ordinary people now hold cryptocurrency, and most of them don't have access to the analytical tools that big trading desks use.
>
> **Two — information asymmetry.** Useful signals exist — patterns, sentiment, on-chain data — but they live in separate silos. A unified model could level the playing field.
>
> **Three — methodology.** Crypto is a stress test for machine learning. It's noisy, it's non-stationary, and the regime changes constantly. Whatever we learn here transfers to other markets like stocks and commodities.
>
> **And four — academic and regulatory interest.** Both researchers and regulators are now demanding more transparent and calibrated forecasting models, not just black-box predictions.

---

## Slide 5 — Problem Statement (~1 minute)

> Now the problem.
>
> Existing crypto-prediction models suffer from three weaknesses.
>
> **The first** is the single-modality bottleneck. Most models use only one type of input — either numbers, or charts, or news. They never combine them.
>
> **The second** is the lack of uncertainty quantification. Most models output a label like "up" or "down", but they never tell you *how confident* they are. For a trader managing risk, that's almost useless.
>
> **The third** is static pattern recall. Pattern-matching tools assume that if today's chart looks like a chart from two years ago, the outcome will be similar. But the market regime changes — a bull market pattern doesn't behave the same way in a bear market.
>
> [pause]
>
> So in one sentence: **existing models exploit only one modality, give no calibrated confidence, and ignore market regime when comparing patterns.** That's what my research solves.

---

## Slide 6 — Evidence (~1 minute)

> I want to show you concrete evidence that this problem is real.
>
> This chart shows direction-prediction accuracy on Bitcoin one-hour candles, evaluated using six folds of walk-forward cross-validation.
>
> The two on the left in light blue are naive baselines — random guessing essentially.
>
> The three in dark blue are the strongest numerical models I tested — XGBoost, LSTM, and PatchTST.
>
> And the two in orange on the right are my chart-based CNN models.
>
> [pause]
>
> Look at the dark blue bars. XGBoost gets 53.3% accuracy. LSTM gets 53.4%. PatchTST gets 52.9%. Three completely different model families — gradient boosting, recurrent networks, and transformers — and they all converge to roughly the same fifty-three percent ceiling.
>
> This is strong evidence that the bottleneck is *information*, not *model capacity*. Adding a fancier numerical model won't help. We need a new modality. That's the core motivation for this research.

---

## Slide 7 — Aim and Objectives (~1 minute)

> The overall aim is shown at the top.
>
> *To design, implement, and evaluate a multimodal cryptocurrency direction-prediction system that combines chart imagery, numerical features, and news sentiment, and that quantifies its own uncertainty using conformal prediction.*
>
> I have five specific research objectives.
>
> **RO1** — Build a leak-free data pipeline for five major cryptocurrency pairs.
>
> **RO2** — Establish strong numerical baselines.
>
> **RO3** — Train chart-based CNN models and compare them statistically using the Diebold-Mariano test.
>
> **RO4** — Fuse all three modalities with a late-fusion classifier and add calibrated confidence intervals.
>
> **RO5** — Deliver an interactive dashboard where the user selects a chart region and the system returns historical matches with explainable predictions.
>
> Each objective directly addresses one part of the problem statement.

---

## Slide 8 — Literature Review (~1 minute)

> A short review of the most relevant existing work.
>
> Sezer and Ozbayoglu in 2018 used a CNN on GAF-encoded financial time series. Their contribution was to show that visual encodings can match LSTMs, but they used a single modality and gave no uncertainty estimates.
>
> Livieris and colleagues in 2020 built a CNN-LSTM hybrid for crypto. Strong benchmark numbers, but again, point predictions only — no calibrated confidence.
>
> Wang and Oates in 2015 originated the GAF and MTF imaging techniques that Sezer's work later applied to finance.
>
> Araci's FinBERT model in 2019 established the financial-NLP baseline that I'm building on.
>
> And Nie and colleagues in 2023 introduced PatchTST — a state-of-the-art patch-based transformer — but it works only on numerical channels.
>
> All five sources are peer-reviewed, all from within the last eight years.

---

## Slide 9 — Identified Gaps (~1 minute)

> From this review I identified five clear gaps.
>
> On the left, in red — what the literature lacks.
>
> Most studies use only one modality. Few report calibrated uncertainty. Pattern-matching ignores market regime. Lookahead bias is rarely audited. And there is no human in the loop — the user doesn't interact with the model.
>
> On the right, in green — how my project closes each gap.
>
> I combine all three modalities. I wrap every prediction in conformal uncertainty. I filter pattern matches by market regime. I enforce strict timestamp guards on news data. And I build an interactive dashboard where the user selects a chart region as part of the prediction pipeline.
>
> [pause]
>
> Each green item directly answers a red item. That alignment is the heart of my contribution.

---

## Slide 10 — Proposed Solution (~1 minute 30 seconds)

> Here's the system architecture, broken into seven phases.
>
> **Phase 1, the Foundation** — data ingestion. Pulling from Binance and from RSS news feeds, storing in monthly Parquet partitions.
>
> **Phase 2, Baselines** — the numerical models I just showed you, with walk-forward cross-validation.
>
> **Phase 3, the Models layer** — two parallel branches. The chart-CNN on the left, the news and sentiment NLP on the right.
>
> **Phase 4, Fusion** — combining all modalities with conformal uncertainty.
>
> **Phase 5, the Pattern Engine** — using FAISS for fast similarity search, plus continual learning so the model adapts as the market changes.
>
> **Phase 6, the Application layer** — a live dashboard built with FastAPI and TradingView's lightweight charts library, including Grad-CAM explanations of what the CNN is looking at.
>
> [pause]
>
> Each phase has a clear deliverable. Each phase produces something runnable.

---

## Slide 11 — Innovation (~1 minute)

> What is novel about my work compared to existing systems?
>
> Six things, briefly.
>
> First, **multimodal late-fusion** — combining numerical, visual, and textual features in a single classifier. Most published work fuses only two.
>
> Second, **conformal uncertainty** — every prediction comes with a calibrated 90% confidence interval. This is almost completely absent from crypto-prediction literature.
>
> Third, **regime-aware similarity search** — the pattern-matching engine filters historical matches by the current market regime, so a bull-market pattern doesn't contaminate a bear-market prediction.
>
> Fourth, **a human in the loop** — the user selects a chart region and the system finds historical analogues. No prior work treats the user as part of the pipeline.
>
> Fifth, **a zero-leakage news pipeline** — every article carries both publication and ingestion timestamps, with unit tests that fail the build if any training row uses information from after the prediction time.
>
> And sixth, **continual learning** — using Elastic Weight Consolidation and a replay buffer so the model adapts to new regimes without forgetting old patterns.

---

## Slide 12 — Feasibility and Progress (~1 minute)

> This is not just theory. Three of the seven phases are already complete.
>
> Phase 1, Data Ingestion — done. Phase 2, Baselines — done. Phase 3, Chart-CNN — done.
>
> Some concrete numbers from the work I've already completed.
>
> **Three hundred and seventy thousand candles** ingested across five pairs and six time intervals.
>
> **Seven different models** benchmarked side by side.
>
> **The best chart-CNN achieves 54% accuracy** on Bitcoin one-hour candles — that's 0.7 percentage points above the strongest numerical model, XGBoost, and the difference is statistically significant by the Diebold-Mariano test.
>
> And every real model significantly beats the naive baselines, with p-values essentially zero.
>
> [pause]
>
> So the project's foundations are solid. The remaining four phases build on a working, tested pipeline.

---

## Slide 13 — Research Methodology (~1 minute)

> Now the methodology, structured around Saunders' research onion.
>
> My **philosophy** is positivism — every output is a measurable metric that can be reproduced.
>
> My **approach** is deductive — I form a hypothesis, like *multimodal beats single-modality*, and test it against held-out data.
>
> My **methodological choice** is mixed — primarily quantitative experimentation, with some qualitative dashboard usability.
>
> My **strategy** is experimental — each new modality is an ablation arm against an established baseline.
>
> The **time horizon** is cross-sectional with rolling windows — I slice years of OHLCV data into expanding training windows and fixed test windows.
>
> And **data collection** is secondary — I use the public Binance REST and WebSocket APIs and freely-available RSS news feeds. Everything is reproducible from a fresh checkout of the repository.

---

## Slide 14 — Development Approach + Tools (~1 minute)

> On the left, my development approach.
>
> I use **Agile development with phase gates**. Each of the seven phases is a two-to-three week sprint with a runnable deliverable.
>
> Smoke tests are written first, in a test-driven style, so every phase has a one-minute sanity check.
>
> The walk-forward cross-validation harness is reused across phases, which means every model is directly comparable using the Diebold-Mariano test.
>
> [pause]
>
> On the right, the technology stack.
>
> For data — Python, pandas, PyArrow.
>
> For machine learning — scikit-learn, XGBoost, PyTorch.
>
> For NLP — FinBERT, CryptoBERT, and the Hugging Face transformers library.
>
> For search and the user interface — FAISS for vector similarity, FastAPI for the backend, React with TradingView's lightweight charts for the frontend.

---

## Slide 15 — Timeline (~1 minute)

> The full project runs over twelve months.
>
> The bars in **dark blue** are completed. The bars in **orange** are planned.
>
> Phase 1 was finished in April. Phase 2 in May. Phase 3 in June and July.
>
> The red NOW marker shows where I am today. From here:
>
> Phase 4, **News and Sentiment**, runs through August and September.
>
> Phase 5, **Multimodal Fusion**, October to November.
>
> Phase 6, the **Pattern Engine**, December to January.
>
> Phase 7, the **Live Dashboard**, in February.
>
> And **thesis writing and the viva** in March.
>
> Each milestone has a runnable deliverable, so even if a phase runs late, the project always has a working baseline to fall back on.

---

## Slide 16 — Conclusion (~30 seconds)

> To close.
>
> I've **established** that numerical baselines hit a 53% accuracy ceiling, replicated across five different cryptocurrency pairs.
>
> I've **demonstrated** that a chart-based CNN already adds 0.7 percentage points over XGBoost on Bitcoin, with a statistically significant Diebold-Mariano result.
>
> And **coming next** is the multimodal fusion that combines all three signals, plus calibrated uncertainty and an interactive dashboard.
>
> [pause]
>
> Thank you for listening. I'm happy to take any questions.

---

# Delivery Tips

## Body language
- **Stand still during the title and conclusion slides.** Move only when transitioning to a major section.
- **Look at three points in the room** — left, centre, right — and rotate every few sentences. Don't stare at the screen or your laptop.
- **Hands open and visible.** No pockets. No crossing arms.

## Voice
- **Slow down at the start.** Most students rush slide 1 because they're nervous. Give the title slide a full 30 seconds.
- **Pause after every key number** ("53% ceiling", "0.7 percentage points", "p-value zero point zero zero six"). Numbers carry weight only when they have space around them.
- **Drop your tone** at the end of every sentence. A rising tone sounds uncertain.

## Slide handling
- **Don't read off the slides.** Glance at them, then talk to the audience. If you need a memory aid, look at your notes/laptop, not the projection.
- **Move forward, not backward.** If you make a mistake, keep going — don't say "sorry, let me go back". The audience won't notice.

## Likely Q&A questions and short answers

| Question | One-line answer |
| --- | --- |
| Why crypto and not stocks? | Crypto is 24/7 with rich news flow and is a harder testbed; findings transfer. |
| Why direction prediction, not price? | Direction is binary and easier to evaluate; price is dominated by drift. |
| Why CryptoBERT *and* FinBERT? | Crypto slang vs. macro news — they catch different signals; ensemble averages them. |
| What if news is sparse? | Aggregator returns zeros; sentiment baseline only evaluates bars with at least one article. |
| How do you prevent leakage? | Two-timestamp scheme — every row carries `published_at` and `ingested_at`; unit tests fail the build on any violation. |
| Why FAISS and not Elasticsearch? | FAISS is built for high-dimensional vector search; sub-millisecond k-NN at our scale. |
| What if the CNN doesn't beat XGBoost? | It already does on BTC; on other pairs it's tied — the headline is the *fusion* in Phase 5. |
| Sample size for DM test? | ~38,000 test predictions per pair × 6 folds — well above the 2,000-sample rule of thumb. |

## Final checklist (night before)
- [ ] PowerPoint file open and tested on the presentation laptop
- [ ] Backup PDF on a USB stick
- [ ] Charging cable + adapter for HDMI / USB-C
- [ ] Water bottle
- [ ] Print this script — single-sided, large font, slide numbers in the margin
- [ ] Set phone to Do Not Disturb 30 minutes before
- [ ] Arrive at the room 10 minutes early to test the projector

You've got this.
