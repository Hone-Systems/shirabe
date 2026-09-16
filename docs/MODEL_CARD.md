# Model card: Shirabe style model

## Intended use

An exploratory signal about citation attention from English scientific abstracts, for researchers investigating the writing/impact hypothesis. Do not interpret a score as a probability that a finding is correct, replicable, useful, or commercially successful. Do not rank people or make hiring/funding decisions from it. An abstract can be rewritten to move the score without changing the science; the model does not estimate a causal effect of rewriting.

## Representation and training

57 fixed features: 18 structural statistics, 10 rhetorical-category rates, and 29 function-word rates. Structural statistics include word/sentence counts, sentence length, word length, lexical diversity, numeric/punctuation rates, and rough passive/past-tense/adverb proxies. They are deliberately simple heuristics. Dictionary categories are promotion, hedging, certainty, limitations, evidence, future, negation, first person, causal and contrast. Dictionaries are hand-authored, case-insensitive exact lexical matches, not context-sensitive discourse labels. Some terms have ambiguous scientific meanings (e.g. “significant”).

No learned topic-word vocabulary, authors, institution, journal, title, citations, year, or field enters the deployed model. Word lengths, abbreviations and word diversity can still encode subject. Rhetorical category overlaps contribute to every matched category; the color inspector uses the first matching category. Tokens are regex words/numbers/punctuation, not BPE/LLM tokens; there are no fabricated transformer layers.

Fit the standardizer on 2016–2018 only; clip standardized values at ±5; fit L2 logistic regression. Choose C from {0.001, 0.01, 0.1, 1, 10} by 2019 log loss. Fit a separate sigmoid calibration on the base model's 2020 logits, using unweighted logistic regression (C=1e6). Combine calibration slope and offset into the published coefficients and intercept. Do not refit on the 2021 test set. Calibration follows the [independent calibration-data principle](https://scikit-learn.org/stable/modules/calibration.html).

For every prediction, `z = clip((x - mean) / scale, -5, 5)` and `p = sigmoid(intercept + Σ zᵢ × coefficientᵢ)`. Shown contributions exactly sum to the calibrated logit. They are not SHAP, token attributions, causal effects, or percentage-point deltas. The intercept at average features is different from the empirical prior.

## Evaluation

See `artifacts/report.json` for exact values, selected C, source/data hashes, environment versions, optimizer iteration count and timestamps.

- 2016–2018 train, 2019 validation, 2020 calibration, 2021 test. Global exact deduplication by ID/DOI/normalized title/normalized abstract before labeling and fitting. Near duplicates/related follow-on papers remain possible.
- Positive outcome: citations at or above the 75th percentile (`numpy.quantile(method="higher")`) of each retained field/year sample. Ties remain together. Labels for test cohorts use their eventual citation outcomes; those outcomes do not enter features, parameter choice, standardization or calibration.
- This is a retrospective temporal partition, **not an as-of-2020 forecasting simulation**: even training outcomes and metadata were observed in September 2026. Citation exposure differs across years. Normalizing within year/field addresses scale, not every horizon or cohort effect.
- Baselines: empirical training prior, length-only calibrated logistic, rhetorical-rate-only calibrated logistic, calibrated lexical TF-IDF logistic, calibrated shallow boosted trees on style, and journal/field one-hot logistic. Model settings and fit code are public. Journal/field baseline is uncalibrated; comparisons therefore mix calibration policies, disclosed in code.
- Six leave-one-field-out experiments independently tune and calibrate on the other fields, then evaluate the omitted field's 2021 papers. This is stronger than just disaggregating the deployed model's test set; it still does not prove invariance or transfer outside the six fields.
- A separate field-prediction probe quantifies residual discipline information. Its high accuracy is a limitation, not evidence that topic words entered the feature vector.
- 20 training-label permutations are negative controls; reported raw-score test AUC is near chance. No selection uses their test results.
- Bootstrap AUC intervals use 500 resamples of test observations with fixed trained weights. They omit variation from training/data collection, cluster dependence, and label construction. Individual-paper confidence intervals are not claimed.
- Learning curves fit separate models at 10/25/50/75/100% of the training data, using a fresh train-only scaler and validation-year evaluation. Optimization curves are fresh fits with iteration budgets, not live epoch logs; early convergence can repeat final values.

The architecture/splits/regularization search were fixed before viewing outcomes. An initial 12,157-record run revealed preprocessing inconsistency during contract checks: whitespace-based length filtering and metadata language labels admitted unsuitable records. The published run uses the same regex word count as inference and deterministic language detection, yielding 11,799 records. No features or hyperparameters were chosen using the improved test result.

## Input limits and uncertainty

Only English article abstracts with 80–800 regex words are supported. Language detection is heuristic and can reject legitimate English text. Longer papers are accepted as documents only to extract a reviewable abstract. Full-paper writing, translated text, unusual formats, unfamiliar disciplines and modern LLM-edited prose are unvalidated.

Warn if more than 15% of features fall outside the training 1st–99th percentile ranges, if clipping applies, if text is highly repetitive, or if an exact text hash matches a training abstract. These are diagnostic checks, not a validated out-of-distribution detector. A missing warning is not proof of applicability.

The sampled corpus is small relative to science, constrained by indexed abstract availability and six fields. Citation attention reflects networks, venue, topic, access, self-citation, fashions, and errors as well as intellectual contribution. Neither this observational model nor higher citation counts establish the user's broader concept of “success.”

## Compute and artifacts

CPU-only, seed 42, two numerical threads; roughly 11 seconds for the full run after data collection. $0 RunPod/cloud spend. No GPU or external inference model required. Portable JSON weights avoid loading untrusted pickle. Numeric features allow offline reproduction; full lexical-baseline reproduction requires recollection or the local raw cache. Source hashes bind the report to the feature/training code.
