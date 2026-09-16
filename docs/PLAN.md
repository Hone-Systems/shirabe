# Shirabe: execution and evidence

The brief is ../idea.md. Build a new open-source repo, train on real historic papers, evaluate whether manner of writing generalizes across topics, expose training and inference in a polished analytical app, accept paper uploads, visualize the actual representation and model. Local CPU first; no cloud spend needed for an interpretable baseline.

## Prespecified experiment (before outcomes are inspected)
- Fetch seeded, unranked samples of 400 English article records with abstracts per field/year from OpenAlex, 2016–2021. Fields: biology, computer science, economics, medicine, physics, social sciences. Exclude retractions. Cache raw responses, preserve provenance and deduplicate IDs, DOIs, normalized titles and abstracts globally.
- Outcome: citation count at collection at or above the 75th percentile of the retained sampled field/year cohort. Ties kept together; actual prevalence reported. This is a citation-attention proxy, never proof of truth, replication, or commercial success. Citation horizons differ; year cohorts partly address this, not fully. Labels use outcome data only; no labels/metadata enter the style representation.
- Fit 2016–2018; select regularization on 2019; fit probability calibration on 2020; untouched test 2021. Keep the deployed model identical to the evaluated model. Seed 42.
- Fixed, inspectable rhetorical dictionary rates + structural/function-word features. No topic vocabulary, names, author, institution, journal, citations, year, or field as predictors. Style may still encode discipline: measure it, do not claim invariance.
- Compare prior-only, length-only, lexical TF-IDF, and nonlinear style baselines; evaluate cross-field transfer by leaving each field out of all fitting/tuning/calibration. Report ROC AUC with bootstrap CI, average precision, Brier, log loss, calibration, per-field results, learning curves and label-permutation negative control. No test-guided tuning.
- Serve transparent linear style model with exact signed log-odds decomposition, token-category inspection, standardized feature values and input distribution flags. Upload PDF/TXT: extract an abstract then require editable confirmation before analysis. Support abstract text directly; do not silently score whole papers using an abstract model.

## Design direction
Precise / restrained / analytical. Dark instrument console: #0c1117 background, #131c25 panels, ice #dce7ed text, cyan #80d8d0 emphasis, amber #e8ba78 caveats. IBM Plex Sans + IBM Plex Mono, local font assets. 4/8px spacing, 2px corners, thin borders, quiet transitions. Two real routes: / (analysis) and /training. Header + tab navigation; analysis split input/result with token inspector below; training overview metrics followed by ROC/calibration, learning history, comparisons, cohorts and reproducibility. No provided design system was attached. Other considered directions: white laboratory/blue; neutral graphite/orange; navy/pink observatory. Dark cyan instrument console best fits the brief.

## Completion checklist
- [x] Reproducible data acquisition + source/license notes, immutable snapshot hashes.
- [x] Trained, portable real model and honest full evaluation.
- [x] Backend text/PDF/TXT inference and training metrics; safe bounded upload handling.
- [x] Two working analytical pages with real visualizations, errors and mobile layouts.
- [x] Meaningful tests, build, live end-to-end browser QA and independent frontend review.
- [x] README, model/data cards, reproducible commands, screenshots, license, clean secret scan, Git commit and public repository.

Final evidence: see VERIFICATION.md. Final retained corpus is 11,799 after matching inference word counting and language detection; the initial sample/filter pass is documented in MODEL_CARD.md. CPU model, routes, feature inspection, tests, browser QA, documentation and release artifacts are complete.
