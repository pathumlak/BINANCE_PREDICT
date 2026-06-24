# Viva / Defence — Answers to the Three Hardest Questions

These are the answers to the three questions the panel raised. Memorise the
**bold** lines. Practice each answer aloud at least 5 times before Thursday.

---

## Question 1 — "If your model is successful and shared worldwide and everyone profits, what happens to market cap?"

**Why they asked it:** they want to see if you understand *why your research is valuable as research, not as a get-rich product*. This is a question about market efficiency.

### Your answer

> "That is the most important question in quantitative finance, and it has a well-documented answer.
>
> If everyone deployed my model in production, **the edge would disappear**. This is called **self-defeating prophecy** or **arbitrage decay**. The reasoning is simple: if every trader is long when my model says 'up', there is nobody left on the other side of the trade. The predicted move gets priced in immediately, slippage destroys the entry price, and accuracy collapses back toward 50%.
>
> This effect is empirically validated. **McLean and Pontiff (2016) studied 97 published cross-sectional return predictors in finance and found that 32% to 58% of the out-of-sample return decayed after the strategy was published.** Once a model is public, capital floods in until the edge is competed away.
>
> So the answer to your question: **market cap would briefly inflate as capital piles in, then the predictable component of returns collapses to zero**. The model would still produce predictions, but accuracy would converge to 50% and Sharpe would converge to zero.
>
> This is precisely why my research contribution is **methodological**, not commercial. I am not trying to build a money printer. I am demonstrating that:
>   * Multimodal fusion adds statistically significant information over single-modality baselines.
>   * Calibrated uncertainty can be attached to crypto-direction predictions using conformal prediction.
>   * Regime-aware similarity search preserves more signal than naive pattern matching.
>
> Those findings are robust *whether or not anyone deploys the model*. The methodology is the contribution."

### Backup citations
* **McLean, R. D., & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return Predictability?" *Journal of Finance*, 71(1), 5–32. — *the canonical paper showing post-publication arbitrage decay.*
* **Schwert, G. W. (2003).** "Anomalies and market efficiency." *Handbook of Economics of Finance*, 1, 939–974.
* **Lo, A. W. (2004).** "The Adaptive Markets Hypothesis." *Journal of Portfolio Management*, 30(5), 15–29.

---

## Question 2 — "Overall accuracy is 53%, how can you say this is the best?"

**Why they asked it:** they want to see if you understand the *context* of your numbers. 53% sounds bad to a non-specialist. It's actually publishable.

### Your answer

> "53% accuracy sounds modest, but in crypto and equity direction prediction it is **state-of-the-art and statistically significant**. I have three concrete pieces of evidence.
>
> **First, the published literature.** Sezer and Ozbayoglu in 2018 reported 53.5% on stock prediction with their CNN-TA model. Livieris and colleagues in 2020 reported 52% to 56% on crypto with a CNN-LSTM hybrid. Atsalakis and Valavanis surveyed 100 published papers on stock-price forecasting in 2009 and found typical accuracy in the range of **51% to 55%**. My result of 53.4% sits squarely in that range. So I am not below the field — I am in line with peer-reviewed work.
>
> **Second, statistical significance.** The Diebold-Mariano test compares two forecasting models pairwise. My chart-CNN beats the naive majority baseline at a **p-value of essentially zero** — that's significance at the 0.001 level. It also beats XGBoost on Bitcoin one-hour candles at **p equals 0.006**, which is comfortably below the 0.05 threshold. The 3-point gap above naive is not luck.
>
> **Third, accuracy is the wrong question for trading.** Even **Renaissance Technologies**, widely considered the most successful hedge fund in history, has an estimated per-trade hit rate of around **50.7%**. Their Medallion fund returned 66% per year before fees for three decades on that tiny edge. The reason is that **information ratio** and **calibrated confidence** matter more than raw accuracy. A 53% model that is right on big moves and wrong on small ones can be far more profitable than a 70% model that is right only on noise.
>
> Critically, **53.4% is the floor, not the ceiling**. That is my single-modality result. Phase 5 fuses three modalities — numerical features, chart-CNN embeddings, and news sentiment. Each modality on its own is at the 53% level; the literature in multimodal medical imaging and multimodal NLP shows fusion typically adds 2–4 percentage points. So I expect the final fused model in the 55%–57% range, with statistically significant gains over each single-modality baseline."

### Numbers to memorise

| Source | Reported accuracy | Domain |
| --- | --- | --- |
| Sezer & Ozbayoglu (2018) | 53.5% | Stock CNN-TA |
| Livieris et al. (2020) | 52–56% | Crypto CNN-LSTM |
| Atsalakis & Valavanis (2009) | 51–55% (survey of 100 papers) | Stock forecasting |
| Renaissance Medallion (estimate) | ~50.7% per trade | Quant trading |
| **My chart-CNN on BTC 1h** | **53.4%** | **Crypto direction** |
| **DM p-value vs naive** | **~0.000** | (significant at 0.001) |
| **DM p-value vs XGBoost** | **0.006** | (significant at 0.01) |

---

## Question 3 — "Do you trust your own project?"

**Why they asked it:** confidence test. They want to see whether you have honest, *reasoned* belief — not blind faith and not undue self-doubt.

### Your answer

> "Yes — and I want to be specific about *what* I trust and *what* I do not claim.
>
> **What I trust about my methodology.** I trust the walk-forward cross-validation harness because it is mathematically incapable of leaking future information into the training set. I trust the Diebold-Mariano significance tests because they apply the standard small-sample correction. I trust the leakage guards on the news pipeline because every article carries both publication and ingestion timestamps with unit tests that fail the build on any violation.
>
> **What I trust about the results.** I trust the 53.4% accuracy because it is reproduced on five independent cryptocurrency pairs. I trust the chart-CNN edge over XGBoost because the DM test confirms statistical significance at p equals 0.006. And I trust my **negative findings** as much as my positive ones — for example, naive persistence loses money on one-hour crypto, GAF encoding underperforms candlestick rasters on altcoins. Reporting honest negative results is evidence that I am not cherry-picking.
>
> **What I trust about the research contribution.** The gap I am filling is genuinely under-explored. Each of the four novel elements — multimodal late-fusion, conformal uncertainty, regime-aware pattern matching, human-in-the-loop dashboard — has been validated independently in adjacent domains. Combining them on crypto is novel. The seven-phase architecture is feasible because three phases are already complete and the foundations are tested.
>
> **What I do not claim.** I do not claim my model will make anyone rich. That would contradict the efficient-market answer to the first question. I do not claim 53% is a high accuracy in absolute terms — it is a high accuracy *given the noisiness of crypto direction*. And I do not claim Phase 5 fusion will definitely beat single-modality baselines — that is a hypothesis, and the DM test will tell us at the end whether it does or not.
>
> So my trust is **calibrated**: high in the methodology, well-evidenced for the results so far, and appropriately uncertain about hypotheses that are still being tested. That is what a research thesis should look like."

---

## Final delivery tips

* **Pause before each answer.** Take a breath. Counting to two looks confident; rushing looks scared.
* **Cite a specific source by name** for each question. Names of papers and authors are remembered, vague phrases are not.
* **Repeat the question back in your own words** before answering. It buys you 3 seconds and proves you understood.
* **Use "let me give you a concrete example"** instead of arguing abstractly — examiners reward concreteness.
* **End every answer with a one-line summary** of what your contribution actually *is*. Do not let them forget it.

If they push further on any question, the universal escape route is:

> "That is a great point — and it is exactly the question Phase 5 / 6 / 7 of my research is designed to answer empirically. I would rather measure the effect than argue about it in advance."

That redirects an attack into a future deliverable. Powerful and honest.

You have the answers. You have the evidence. Walk in confident.
