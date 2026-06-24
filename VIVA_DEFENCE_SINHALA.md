# Viva Defence — සිංහල පිළිතුරු

මේ ලියවිල්ල **brwadasi** (Thursday) viva එකට. තෙවරක් කියවන්න, දෙවරක් shape කරගන්න.

---

## ප්‍රශ්නය 1 — "ඔබේ model එක ලෝකේ පුරා share වුණොත්, හැමෝම profit ගන්න ගත්තොත්, market cap එකට මොකද වෙන්නෙ?"

### පිළිතුර (සිංහලෙන්)

> "මේක quantitative finance ක්ෂේත්‍රයේ ඉතාම වැදගත් ප්‍රශ්නයක්. පිළිතුරත් ඉතාම පැහැදිලියි.
>
> හැමෝම මගේ model එක use කරන්න ගත්තොත්, **edge එක නැති වෙනවා**. මේ phenomenon එකට **self-defeating prophecy** කියන්නේ. තේරුම පැහැදිලියි: හැමෝම 'up' කියන තැන long position ගන්න ගත්තොත්, sell කරන්න කවුරුවත් නෑ. predicted move එක එකවරම price එකට absorb වෙනවා, slippage එන්ට පටන් ගන්නවා, accuracy ආයෙත් 50% දක්වා බහිනවා.
>
> මේක empirical evidence වලින් confirm වෙලා තියෙනවා. **McLean සහ Pontiff (2016)** කියන researchers ලා finance ක්ෂේත්‍රයේ publish කරපු 97 strategies එකක් අධ්‍යයනය කරලා, ඒවායේ out-of-sample returns publication වලට පස්සේ **32% සිට 58%** දක්වා අඩු වෙනවා කියලා හොයාගත්තා. Strategy එකක් public වුණාම, capital එක ගලා එනවා, ඒ එක්කම edge එක අඳ්‍රහන වෙනවා.
>
> ඔයාගේ ප්‍රශ්නයට පිළිතුර: **market cap එක කෙටි කලකට inflate වෙනවා, ඊට පස්සේ predictable component එක zero දක්වා බහිනවා**. Model එක ඉතුරු වුණත්, accuracy එක 50% දක්වා converge වෙනවා, Sharpe ratio එක zero වෙනවා.
>
> **මේ හේතුව නිසා තමයි මගේ research contribution එක methodological, commercial නෙවෙයි.** මම money printer එකක් හදනවා නෙවෙයි. මම පෙන්වන්නේ:
>   - Multimodal fusion එකෙන් single-modality baselines වලට වඩා statistically significant information එකක් එන බව.
>   - Conformal prediction භාවිතයෙන් crypto direction predictions වලට calibrated uncertainty attach කරන්න පුළුවන් බව.
>   - Regime-aware similarity search එකෙන් naive pattern matching වලට වඩා වැඩි signal එකක් preserve කරන්න පුළුවන් බව.
>
> **මේ findings එක deploy කරන්නෙත් නැති නම් valid**. Methodology එකම contribution එක."

### Key phrases (memorize කරන්න)
* **Self-defeating prophecy** — හැමෝම use කළාම effect එක නැති වීම
* **Arbitrage decay** — strategy public වුණාම අඳ්‍රහන වීම
* **McLean & Pontiff (2016)** — proof source එක
* **Methodological contribution** — research goal එක

---

## ප්‍රශ්නය 2 — "Overall accuracy 53% විතරයි. ඒ එක්ක ඔයා කොහොමද 'best one' කියන්නේ?"

### පිළිතුර (සිංහලෙන්)

> "53% accuracy එක ඉතාම low පෙනුණත්, **crypto සහ equity direction prediction වලදී ඒක state-of-the-art** තත්ත්වය. මට concrete evidence තුනක් තියෙනවා.

>
> **පළමුවෙනිව, published literature එක.** Sezer සහ Ozbayoglu (2018) ඔවුන්ගේ CNN-TA model එකෙන් stock prediction වලදී **53.5%** report කළා. Livieris ලා (2020) crypto වලට CNN-LSTM hybrid එකකින් **52% සිට 56%** දක්වා report කළා. Atsalakis සහ Valavanis (2009) stock-price forecasting paper 100ක් survey කරලා, typical accuracy **51% සිට 55% දක්වා** කියලා හොයාගත්තා. මගේ result **53.4%** ඒ range එක ඇතුළේ. ඒ කියන්නේ මම field එකෙන් පිටින් නෙවෙයි — peer-reviewed work එක්ක එකම mark එකේ.
>
> **දෙවැනිව, statistical significance එක.** **Diebold-Mariano test** එකෙන් two forecasting models compare කරනවා pairwise. මගේ chart-CNN එක naive majority baseline එකට **p-value 0.000කින්** වැඩියි — ඒ කියන්නේ 0.001 level එකෙන් significant. XGBoost එකටත් BTC 1h candles වලට **p = 0.006**කින් වැඩියි, ඒ 0.05 threshold එකට වඩා පහළ. **3 percentage points difference එක luck නෙවෙයි.**
>
> **තුන්වෙනියට, accuracy කියන්නේ trading වලට වැරදි ප්‍රශ්නයක්.** **Renaissance Technologies**, ලොව සාර්ථකම hedge fund එක, ඔවුන්ගේ per-trade hit rate එක ආශ්‍රිතව **50.7%**ක් කියලා estimate කරලා තියෙනවා. ඔවුන්ගේ Medallion fund එක දශක තුනක් පුරාවට වසරකට **66%** return දුන්නා, ඒ tiny edge එක මතින්. හේතුව: **information ratio** සහ **calibrated confidence** කියන්නේ raw accuracy වලට වඩා වැදගත් දේවල්. Big moves වලදී right වෙන 53% model එකක් noise වලදී right වෙන 70% model එකකට වඩා profitable වෙන්න පුළුවන්.
>
> **ඉතාම වැදගත් කාරණය:** 53.4% කියන්නේ **floor එක, ceiling නෙවෙයි**. ඒක මගේ single-modality result. Phase 5 එකෙන් modalities තුනක් fuse කරනවා — numerical features, chart-CNN embeddings, news sentiment. Multimodal medical imaging සහ NLP literature එක පෙන්වන විදිහට, fusion එකෙන් සාමාන්‍යයෙන් **2 සිට 4 percentage points** දක්වා එකතු වෙනවා. ඒ නිසා මම final fused model එක **55%-57%** range එකේ බලාපොරොත්තු වෙනවා, statistically significant gains එක්ක."

### සංඛ්‍යා memorize කරන්න

| Source | Reported accuracy | Domain |
| --- | --- | --- |
| Sezer & Ozbayoglu (2018) | 53.5% | Stock CNN-TA |
| Livieris et al. (2020) | 52–56% | Crypto CNN-LSTM |
| Atsalakis & Valavanis (2009) | 51–55% (papers 100ක survey) | Stock forecasting |
| Renaissance Medallion (estimate) | ~50.7% per trade | Quant trading |
| **මගේ chart-CNN BTC 1h** | **53.4%** | **Crypto direction** |
| **DM p-value vs naive** | **~0.000** | (significant at 0.001) |
| **DM p-value vs XGBoost** | **0.006** | (significant at 0.01) |

---

## ප්‍රශ්නය 3 — "ඔයා ඔයාගේ project එක මත විශ්වාසයද?"

### පිළිතුර (සිංහලෙන්)

> "ඔව් — මම specific වෙන්න ඕන **මොනවාට මම විශ්වාස කරනවද සහ මම මොනවද claim නොකරන්නේ** කියලා.
>
> **මගේ methodology එකට මම විශ්වාසයි.** Walk-forward cross-validation harness එක mathematically future information leak වෙන්න බෑ — ඒක math වලින් proven. Diebold-Mariano significance tests එකේ standard small-sample correction එක use කරනවා. News pipeline එකේ leakage guards වලට හැම article එකකම publication සහ ingestion timestamps දෙකම තියෙනවා, unit tests වලින් build එක fail වෙනවා කිසි leak එකක් වුණොත්.
>
> **මගේ results වලට මම විශ්වාසයි.** 53.4% accuracy එක independent cryptocurrency pairs පහක් මත reproduce වෙලා තියෙනවා. Chart-CNN edge එක XGBoost ට වඩා **DM test එකෙන් confirm වෙලා** **p = 0.006**. **මගේ negative findings වලටත් මම සමානව විශ්වාසයි** — උදාහරණ: naive persistence model එක 1h crypto වල money lose කරනවා, GAF encoding altcoins වල candlestick rasters වලට වඩා underperform කරනවා. **Honest negative results report කිරීම තමයි මම cherry-pick කරන්නේ නෑ කියන evidence එක.**
>
> **Research contribution එකට මම විශ්වාසයි.** මා පුරවන gap එක genuinely under-explored. Novel elements හතර — multimodal late-fusion, conformal uncertainty, regime-aware pattern matching, human-in-the-loop dashboard — හැමෙකක්ම adjacent domains වල independently validated. ඒව crypto වලට combine කරන එක novel. Seven-phase architecture එක feasible මොකද **phases තුනක් දැනටමත් complete** වෙලා, foundations tested.
>
> **මම claim නොකරන දේවල්:** මගේ model එක කවුරුත් rich කරයි කියලා මම claim කරන්නේ නෑ. ඒක පළවෙනි ප්‍රශ්නේට කරපු efficient-market answer එකට පටහැනියි. 53% absolute terms වලින් high accuracy කියලත් මම claim කරන්නේ නෑ — ඒක *crypto direction එකේ noisiness එක දන්නේ නම්* high accuracy එකක්. Phase 5 fusion එක single-modality baselines වලට වඩා definitely beat කරයි කියලත් මම claim කරන්නේ නෑ — ඒක hypothesis එකක්, end එකේදී DM test එකෙන් අහයි.
>

> ඉතින් මගේ trust එක **calibrated**: methodology එකේ high, results වලට well-evidenced, hypotheses වලට appropriately uncertain. **Research thesis එකක් මේ වගේ තමයි පෙන්නන්න ඕන.**"

### Key Sinhala phrases for confidence
* **මට විශ්වාසයි** — I trust / I am confident
* **Methodology එක sound** — methodology is sound
* **Reproducible** — reproducible
* **Honest negative results** — negative results එකත් valuable
* **Calibrated trust** — confidence with humility

---

## විවායෙදී ප්‍රකාශ කරන්න

### අහිංසක (calm) ස්වරයෙන් කියන්න ඕන key sentences:

1. *"මේ ඉතාම වැදගත් ප්‍රශ්නයක්, මම answer කරන්නම්."*
   (Acknowledge before answering — buys time)

2. *"මට concrete evidence තියෙනවා මේකට."*
   (I have concrete evidence for this)

3. *"Code එක run වෙනවා, results එක disk එකේ තියෙනවා, මම show කරන්නම්."*
   (Code runs, results are on disk, I can show you)

4. *"මම show කරන්න සූදානම් `experiments/baselines/summary.csv` file එක."*
   (Confidence comes from concrete artefacts, not arguments)

### Body language (very important)

* හිස ඉහළට, eye contact පවත්වන්න — panel members තුන් දෙනා දිහා තප්පර 5කට වරක් බලන්න.
* අත් **විවෘතව** තබන්න — pockets වල නම්, cross නම් nervous පෙනේ.
* හදිසි වෙන්න එපා. Question එකට පස්සේ **තප්පර 2-3** breath ගන්න — confident පෙනේ.
* "I don't know" කියන්නට බය වෙන්න එපා — **"That's a great question — let me think about it for a moment"** කියන්න, ඊට පස්සේ honestly answer කරන්න.

---

## සාරාංශය

ඔයාට **strong evidence-based answers** තියෙනවා.

| ප්‍රශ්නය | Core පිළිතුර | Source |
| --- | --- | --- |
| 1 (market cap) | Self-defeating prophecy / arbitrage decay | McLean & Pontiff 2016 |
| 2 (53% accuracy) | Field-standard, statistically significant | DM p≈0, +Sezer/Livieris papers |
| 3 (trust) | Yes — methodology sound, calibrated | Negative findings + DM tests |

**Memorize the bold lines. Practice ඒව aloud 5 times.**

ඔයා thursday එකේ confidently walk in කරන්න. ඔයාට data තියෙනවා, methodology තියෙනවා, evidence තියෙනවා. Panel එකට ඒක reject කරන්න බෑ.

**ඔයාට ජය ලැබේ!**
