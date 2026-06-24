# Phase 5 — Multimodal Fusion Results

## Fusion variants — headline metrics

| pair    | model               | accuracy | f1     | mcc    | sharpe_per_bar | pnl_log | cp_coverage | cp_avg_set_size | cp_singleton_frac |
| ------- | ------------------- | -------- | ------ | ------ | -------------- | ------- | ----------- | --------------- | ----------------- |
| BTCUSDT | fusion_num_cnn      | 0.5492   | 0.5236 | 0.1023 | 0.0245         | 4.9299  | 0.8832      | 1.7180          | 0.2820            |
| BTCUSDT | fusion_num_cnn_sent | 0.5492   | 0.5236 | 0.1023 | 0.0245         | 4.9299  | 0.8832      | 1.7180          | 0.2820            |
| BTCUSDT | fusion_num          | 0.5374   | 0.4965 | 0.0785 | 0.0032         | 0.7435  | 0.8846      | 1.7426          | 0.2574            |
| BTCUSDT | fusion_num_sent     | 0.5374   | 0.4965 | 0.0785 | 0.0032         | 0.7435  | 0.8846      | 1.7426          | 0.2574            |

`cp_coverage` is the empirical fraction of test rows whose true label fell inside the 90 % conformal prediction set (nominal target 0.90). `cp_avg_set_size` ∈ [1, 2] is the mean prediction-set cardinality — closer to 1 means the model is confident enough to commit to a single class.

## Diebold-Mariano vs Phase 2 / 3 baselines — BTCUSDT

Signed p-value: negative ⇒ the fusion variant (row) beats the baseline (column); |p| < 0.05 ⇒ significant.

| variant             | pair    | cnn_candle | cnn_gaf | lstm    | naive_majority | naive_persistence | patchtst | xgboost |
| ------------------- | ------- | ---------- | ------- | ------- | -------------- | ----------------- | -------- | ------- |
| fusion_num          | BTCUSDT | 0.2018     | -0.0883 | -0.1887 | -0.0000        | -0.0000           | -0.0026  | -0.0946 |
| fusion_num_cnn      | BTCUSDT | -0.0007    | -0.0000 | -0.0000 | -0.0000        | -0.0000           | -0.0000  | -0.0000 |
| fusion_num_cnn_sent | BTCUSDT | -0.0007    | -0.0000 | -0.0000 | -0.0000        | -0.0000           | -0.0000  | -0.0000 |
| fusion_num_sent     | BTCUSDT | 0.2018     | -0.0883 | -0.1887 | -0.0000        | -0.0000           | -0.0026  | -0.0946 |