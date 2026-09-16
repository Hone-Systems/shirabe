# Data card

## Source and selection

OpenAlex Works API, retrieved September 16, 2026 (timestamps per request in `data/provenance.json`). The live request succeeds without credentials in this environment. No user account data or RunPod secrets were accessed.

36 cohorts: 2016–2021 inclusive × six OpenAlex primary-topic fields:

| ID | Field |
|---|---|
| 13 | Biochemistry, Genetics and Molecular Biology |
| 17 | Computer Science |
| 20 | Economics, Econometrics and Finance |
| 27 | Medicine |
| 31 | Physics and Astronomy |
| 33 | Social Sciences |

Each query requests a random sample of 400, seed 42, 200 records per page, filtered to English articles with abstracts and `is_retracted:false`. Sampling is not ranked by citation count. The selected fields are intentionally broad, not a representative sample of every scientific domain. OpenAlex type/field/language/retraction metadata may contain errors and reflects the collection snapshot, not publication-time knowledge.

Reconstruct the abstract by sorting inverted-index positions. Retain 80–800 words using the exact inference regex and English according to `langdetect` with seed 42. Deduplicate globally by work ID, DOI, whitespace/case-normalized title and whitespace/case-normalized abstract SHA-256. No fuzzy/semantic duplicate detection is claimed. Preserve citations and source metadata for labeling and confound comparisons, never deployed-model features.

Of 14,400 requested records, 2,414 failed word-length screening, 170 failed the language check, and 17 were exact duplicates; 11,799 were retained. Filtering and abstract availability introduce selection bias. Cohort thresholds and observed positive rates appear in the report and UI. Low-citation papers are called the lower-citation group, not “failed research.”

## Files and reproduction

- `data/provenance.json`: each exact API URL, retrieval timestamp, raw response hash, final corpus hash, counts, exclusions and seed. No credentials.
- `data/features.npz`: public numeric matrix (`x`, `y`, `years`, `fields`) in the same row order as predictions. Load with `allow_pickle=False`. 57-column order is `model.json.feature_names`. No text/title/author payloads. The report records its SHA-256.
- `artifacts/predictions.jsonl`: work ID, DOI, year, field, citation snapshot, binary label, split, probability. These allow independent recomputation of metrics and cohort membership.
- `data/raw/` and `data/papers.jsonl`: private-to-the-local-checkout cache, ignored by Git. Full corpus abstracts are not redistributed in this project.

`make reproduce` checks all source/feature hashes and refits the deployed style model without network. `make train` reuses existing raw cache or fetches it, then regenerates the full experiment. A fresh fetch from a changing index may not match the original dataset hash even with identical query seeds. To reproduce every baseline exactly, retain the original local cache; the repository's numeric matrix alone cannot reproduce the TF-IDF baseline.

## Provenance and rights

OpenAlex describes its dataset as [CC0](https://openalex.org/). Underlying abstracts and papers can retain their own copyrights; this project does not claim to relicense them. Publication PDFs used during manual QA remain in ignored `artifacts/private/`. The committed test PDF and example abstract are synthetic fixtures written for this project, clearly labeled as such in the app.

## Scope of outcome

Citation counts are a single September 2026 snapshot. Within-cohort percentile labels partially control field and exposure time but do not estimate a fixed five-year outcome or prospectively demonstrate future performance. Labels derived from each evaluation cohort's outcome distribution make this a retrospective rank-classification experiment, not a calibrated universal probability across all science.
