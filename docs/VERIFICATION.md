# Verification record

Verified September 16, 2026 against model `style-c301b70645` and the final production build.

## Automated evidence

- `uv run ruff check shirabe scripts tests`: PASS.
- `uv run ruff format --check shirabe scripts tests`: 11 files formatted.
- `npm run format:check --prefix web`: PASS.
- `uv run pytest -q`: **19 passed**. Valid/invalid input, English/length scope, exact model arithmetic, all held-out outcome metrics recomputed from audit records, split/label/dedup invariants, known-training-text warning, PDF/TXT review flow, rotated stamps/author footnotes, scanned/encrypted/corrupt/oversized uploads, parser timeout, missing model, and file-serving boundaries.
- `uv run python scripts/reproduce_style.py`: PASS. Offline numeric-data and source hashes verified; refit chooses the same regularization, reproduces all coefficients/intercept and held-out AUC/Brier within numerical tolerance.
- `npm run build --prefix web`: TypeScript and production build PASS.
- `npm run test:e2e` in web: **8 passed**, four workflows each on desktop and Pixel 7. Runs against a fresh real backend and the production build, not a mocked successful inference. Includes upload/review/analyze, error/retry, representation, real downloads, route state preservation, training charts/filters, overflow and Axe audits.

Two dependency deprecation warnings in FastAPI/Starlette's test client remain; they do not affect these results. They are visible rather than globally suppressed.

## Independent frontend review

The factory-frontend skill's accessibility, interaction-state, visual hierarchy and polish reviews were performed independently. Final outcomes:

- Zero Axe violations at desktop 1440px and mobile 390px on both routes, including filtered tokens and expanded feature/chart/model tables.
- Keyboard focus/skip link, named scrollable regions, selected-state semantics, input errors, native details, reduced motion and route heading focus verified.
- Visual layout checked at 320, 390, 768, 1024 and 1440px; no document horizontal overflow. Final chart-label separation checked visually; token words wrap as units; chart buttons do not stretch icons.
- Real user flow covers default, pending, failure, recovery and success. Stale results/filters/source notices are reset on input change. Draft/result survives top-navigation round trips. Exports download valid JSON generated from actual results.
- Minor remaining polish limitation: the cohort table's horizontal-scroll hint is at the bottom of its scrolling container. The table remains visibly bounded, labeled and keyboard scrollable.

Final screenshots are under `docs/screenshots/`. The example visible there is a synthetic demonstration abstract, not a published outcome anecdote.

## Real paper and privacy checks

Downloaded Vaswani et al.'s “Attention Is All You Need” (arXiv 1706.03762) into ignored `artifacts/private/`. The 15-page PDF extracted a reviewable **163-word abstract**, excluded its rotated arXiv stamp and author-contribution footnotes, and completed local inference with HTTP 200. This establishes one real PDF workflow, not universal PDF-layout extraction accuracy; editable review remains required.

Staged-file secret-pattern scan found no matches. Raw responses, reconstructed abstracts, downloaded PDF, virtualenv, node_modules, logs and browser traces are excluded from version control. All committed images use the synthetic example. Model weights and numeric feature data contain no full paper text.

## Requirement-to-evidence audit

| Requirement from idea.md | Authoritative evidence |
|---|---|
| New Japanese-style named project and Git repo in code directory | Shirabe directory, README naming, Git history |
| Model trained on historic papers with observed outcomes | OpenAlex manifest; 11,799-row feature matrix and predictions; scripts/train.py; model/report JSON |
| Learn manner of writing rather than topic-word identity | Fixed features.py; topic-word replacement invariance test; discipline probe and leave-field-out results |
| Likelihood signal on a new paper | Real `/api/predict` output; PDF review→inference test; explicit citation-attention target |
| Explain training and measured performance | `/training` live page, report export, ROC/calibration/learning/convergence, all baseline/transfer results |
| Show representation/model internals | Live token masking/categories; 57 raw/standardized feature rows; exact weighted-logit decomposition |
| Analytical app with paper upload | Production React/FastAPI pages; desktop/mobile screenshots; successful TXT, synthetic PDF and real PDF QA |
| Use affordable compute | CPU-only measured run; $0 cloud spend; offline make reproduce |
| Open-source safely | AGPL license, dependency notices, ignored paper corpus, staged secret scan, public source link |

Scientific limitations are part of the result, not hidden unfinished features: the trained outcome is retrospective citation attention, English abstracts are the validated text unit, and subject independence or future real-world success is not established. The broader hypothesis is investigable with the delivered experiment, not proven by it.
