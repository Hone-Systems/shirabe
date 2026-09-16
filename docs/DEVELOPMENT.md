# Setup and development

## Local app

Run `make setup` and `make run` from the repository root. The server binds to `127.0.0.1:8787`; API documentation is at `/docs`.

| Route | View |
|---|---|
| `/` | Experimental wording-to-outcomes analysis |
| `/training` | Training histories, controls and external evaluation |
| `/evidence` | Paper rubrics, research sources and exports |
| `/?view=citation` | Earlier citation model |
| `/training?view=citation` | Citation-model diagnostics |

The legacy citation model is included. Outcome inference also needs dependencies from `uv sync --group training --group research`, the locally trained checkpoint referenced by `artifacts/outcome-training/served.json`, and credentials for cloud topic masking. Training weights and original paper caches under `artifacts/private/` are intentionally excluded from version control. A fresh clone does not contain everything needed to reproduce the outcome screenshots.

The current research client reads `OPENAI_API_KEY` from the environment and can also read credentials from the sibling `../senki` checkout. In that local integration, nonempty Senki `.env` values take precedence. Keep credentials out of source control. Input preparation and uncached topic masking can make paid model calls; the current outcome workflow uses a $50 ledger ceiling.

English PDF/TXT uploads are limited to 10 MB. The outcome PDF route reads all pages within a 200-page limit and requires text review before analysis. Scanned PDFs need OCR elsewhere. Topic masking sends original text to a cloud model; source text and responses are privately cached. The local transformer processes all resulting chunks.

For UI development, run the API on 8787 and `npm run dev --prefix web`; Vite serves the UI on 5187 and proxies `/api`.

## Outcome research and training

Follow the [paper-grading skill](../skills/shirabe-grade-paper/SKILL.md) and [experiment protocol](WORDING_EXPERIMENT.md). Agents research post-publication outcomes and justify 20 granular answers with sources. Unknown answers are masked, not converted into negative labels. Publish a validated annotation with:

```sh
uv run python scripts/agent_grade.py publish artifacts/private/agent-grades/paper.json
```

The Evidence view automatically reads published annotations from `data/outcomes/`. Downloaded sources and draft grades remain private. Challenge and external cohorts must stay out of fitting and model selection.

In the configured training environment with required source caches:

```sh
uv run --no-sync python -m scripts.train_expansion
uv run --no-sync python -m scripts.train_expansion --convergence
uv run --no-sync python -m scripts.evaluate_external
```

Public histories, controls and evaluation reports are in `artifacts/outcome-training/`; frozen labels, input caches and checkpoints are in `artifacts/private/outcome-training/`. The serving pointer identifies the experimental checkpoint by hash. No outcome model has passed validation for useful success/failure discrimination.

## Verification

```sh
make test
cd web && npx playwright install chromium && cd ..
make e2e
```

Outcome browser tests replay saved real measurements to avoid paid inference calls. README screenshots show actual cached inference and measured training/evidence reports, captured from the running app.

## Legacy experiments

```sh
make reproduce  # Offline linear-model reproduction from public numeric features
make train      # Fetch OpenAlex samples, then fit the original comparisons
uv run --group training python scripts/train_transformer.py
```

Transformer retraining needs the raw abstract cache; add `--resume` to reuse completed checkpoints. Restart the API after changing the legacy model. Current OpenAlex collection can differ from the original snapshot.

See the [legacy model card](MODEL_CARD.md), [data card](DATA_CARD.md), and [data provenance](../data/provenance.json) for its target, metrics and limitations. Citation results do not establish scientific validity or outcome-prediction performance.

## API and source map

- `shirabe/api.py`: API, bounded uploads and production UI.
- `shirabe/outcome_model.py`: local outcome inference and measured telemetry.
- `scripts/train_expansion.py`: outcome fitting, controls and convergence checks.
- `scripts/evaluate_external.py`: evaluation of frozen models on external papers.
- `web/src/`: analysis, training and evidence interfaces.

Key routes: `GET /api/outcome-model`, `POST /api/outcome-predict/stream` (NDJSON), `POST /api/outcome-predict`, `GET /api/outcome-training`, `GET /api/evidence`, and `POST /api/extract?full_text=true` (multipart `file`). The local research app has no public-account or multi-user upload system.
