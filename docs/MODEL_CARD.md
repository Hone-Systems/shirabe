# Model card: Shirabe BERT-Mini

## Architecture and training

The deployed model is a fine-tuned [Google BERT-Mini](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4): four bidirectional transformer encoder layers, four attention heads per layer, hidden width 256, and a two-class learned head. The pinned pretrained revision, exact parameter count and artifact hash are in `artifacts/transformer/manifest.json`. All encoder parameters were fine-tuned locally; this is transfer learning, not pretraining from scratch.

Input: an English abstract of 80–800 regex words, tokenized into at most 512 uncased WordPieces including CLS/SEP. Longer inputs are truncated with a visible warning. No title, author, journal, citation count, field or publication year is supplied to the model. Topic words are supplied, so this model measures both content and writing; it does not isolate style.

Train on 5,798 abstracts from 2016–2018, select the lowest-loss epoch on 1,945 abstracts from 2019, calibrate on 1,986 from 2020, then evaluate once on 2,070 from 2021. AdamW, learning rate 3e-5, weight decay .01, batch 32, four epochs, seed 42. Architecture and settings were recorded before training in `docs/TRANSFORMER_PLAN.md`. The four-epoch run selected epoch four. Training used a local RTX 4070 Ti SUPER; inference uses CPU ONNX Runtime. A machine restart interrupted the cross-field evaluation; completed runs were recovered from saved best checkpoints and epoch logs, then remaining runs completed with the same configuration.

## Outcome and evaluation

The target is a citation count at or above the 75th percentile of the retained sampled field/year cohort, observed in September 2026. Citations measure attention, not scientific truth, replication, practical usefulness or commercial success. The temporal partition is retrospective, not an as-of-publication forecasting simulation. Even training labels use later citation counts, and pretrained text may contain historical paper-related information.

The main test AUC is **0.729** (500-bootstrap 95% interval **0.704–0.752**), versus **0.697** for the previous linear style model. Average precision is **0.452**, versus a positive prevalence of **0.258**. Brier score is **0.169**. These are test-set discrimination and probability-error measurements, not guarantees for individual papers. Bootstrap intervals fix the trained model and omit retraining, sampling and cluster uncertainty.

`artifacts/transformer/report.json` contains actual epoch losses, calibration bins, baselines, per-field scores, six independent leave-one-field-out training runs and a discipline probe on learned activations. Excluded-field AUC ranges from 0.636 to 0.744. The discipline probe achieves 68.6% accuracy versus 17.7% majority baseline, confirming substantial topic information. Shuffled-token AUC is 0.716 versus 0.729 with original ordering. A shuffled-token test preserves vocabulary while changing order; it is a diagnostic, not a causal estimate of the value of syntax. The earlier linear model's shuffled-label control and discipline probe are explicitly retained under `baseline_*` keys; they are not transformer results.

## Interpretation

For pooled activation h, the displayed head arithmetic is `p = sigmoid(intercept + Σ hᵢ × coefficientᵢ)`, with held-out calibration folded into coefficients and intercept. The 256 signed contributions sum to the calibrated logit. These are learned latent channels, not named linguistic features. Their train-relative z values are diagnostics; the classification head uses raw pooled activations. The intercept is the calibrated head bias at zero activation, not the empirical prior or the score at mean training features.

Token brightness shows final-layer CLS attention averaged across four heads. The JSON export includes all four layers' CLS-to-token attention and five CLS state vectors (embedding plus four layers). Attention is not causal token importance. Special-token attention is included in the exported matrices; the visible token strip excludes special tokens. Decorative particle movement does not measure processing speed or training activity.

## Scope and reproducibility

Only English abstracts and the six sampled fields were evaluated. Full papers are accepted only to extract a reviewable abstract. Language detection and PDF extraction are heuristic. Near duplicates, citation confounds, unfamiliar fields and modern LLM-edited text remain limitations. Exact training-abstract hashes and repetitive inputs trigger notices; these are not validated out-of-distribution detectors.

The exported ONNX graph, tokenizer, calibrated head and evaluation records ship in the repository; inference needs no GPU or network. PyTorch-to-ONNX parity and exact head arithmetic are checked. Retraining requires the ignored raw abstract cache or fresh collection, plus `uv run --group training python scripts/train_transformer.py`. `--resume` reuses completed runs only when checkpoint configuration matches. New OpenAlex collection can change the dataset. `make reproduce` remains exact offline reproduction of the **previous linear baseline**, not transformer retraining.

The original baseline and its numeric feature matrix remain available. See [archived baseline card](BASELINE_MODEL_CARD.md), [data card](DATA_CARD.md) and [third-party notices](THIRD_PARTY.md).
