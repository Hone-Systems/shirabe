# Shirabe grading contract

Run from the Shirabe checkout, normally `/home/hone/code/shirabe`. Use `uv run --no-sync python scripts/agent_grade.py`; source downloads/drafts belong under `artifacts/private/agent-grades/`.

Progress:
```
uv run --no-sync python scripts/agent_grade.py progress --id PAPER_ID --title "Paper title" --year 2017 --field-id 17 --field "Computer Science" --stage searching --note "Checking independent follow-up studies"
```
Stages: `reading_original`, `searching`, `reading_sources`, `answering_rubric`, `validating_evidence`, `complete`, `error`. Update during meaningful work, not for every browser call. The assignment may specify split/cohort; default is `challenge` / `agent_pilot`.

Write one JSON object, then publish:
```
uv run --no-sync python scripts/agent_grade.py publish artifacts/private/agent-grades/PAPER_ID.json
```
Required shape:
```json
{
  "id": "challenge-attention",
  "title": "Attention Is All You Need",
  "doi": "https://doi.org/10.48550/arXiv.1706.03762",
  "year": 2017,
  "field_id": 17,
  "field": "Computer Science",
  "split": "challenge",
  "sampling_cohort": "agent_pilot",
  "as_of": "2026-09-16",
  "original_source": {"url": "https://arxiv.org/pdf/1706.03762", "scope": "full_text", "read": true},
  "sources": [
    {"source_id": "S1", "url": "https://...", "title": "Source title", "date": "2019", "date_basis": "publication", "kind": "primary_study", "read": true, "independent": true, "finding": "Concise factual paraphrase of relevant finding."}
  ],
  "answers": [
    {"id": "claim_supported", "answer": "yes", "rationale": "Explain the evidence-backed judgment.", "evidence": [{"source_id": "S1", "date": "2019", "explanation": "How the source supports this answer."}]}
  ],
  "search_log": [{"query": "Actual search query", "purpose": "What it checked", "urls_read": ["https://..."]}],
  "search_gaps": ["Remaining uncertainty, if any."],
  "summary": "Brief account of the observed outcome and its limits.",
  "researcher": "Codex research agent"
}
```

The example shows one answer only for brevity. The artifact must contain all 20 IDs from `shirabe/outcome_rubric.py`, exactly once. Every yes/no needs at least one evidence item referencing a read source. Dates may be `YYYY` or `YYYY-MM-DD`. For an undated official artifact observed live, use the as-of date and `date_basis: "observed_as_of"`; do not invent a publication date. A source cannot be later than the as-of date. Do not supply an overall numeric score; the application computes it from the answers.

Sources and rationales become visible in the explorer. Do not put API keys, copyrighted full text, or long source quotations in this artifact. Use links and factual paraphrases. Optional additional research notes can remain in the private draft directory.

Optional `central_claims`: short string array identifying the original claims established before follow-up research. The publisher preserves these for users to inspect.

When metadata identifies the wrong article, preserve the assigned ID and add `identity_resolution` with assigned/resolved identifiers, an explanation, and `training_eligible: false`. Do not pair a corrected outcome with uncorrected training text. If the intended original cannot be read, set `original_source.read: false`, `scope: "unavailable"`, and all 20 answers to `unknown`; the publisher also requires the explicit training exclusion and explanation. This produces `review_status: "identity_unresolved"`, not a completed evidentiary grade. Never claim to have read an inaccessible original.
