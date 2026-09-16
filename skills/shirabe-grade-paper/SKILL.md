---
name: shirabe-grade-paper
description: Research a scientific paper's actual post-publication outcomes and produce an evidence-backed granular rubric grade for Shirabe. Use for autonomous paper grading, outcome-label dataset creation, and reassessment of low-evidence labels.
---

# Grade a paper's observed outcomes

Assess what happened to this particular contribution after publication, through the requested as-of date (default today). This is an outcome-labeling task, not a prediction or a judgment of how the paper sounds. Do not infer success from confident, restrained, simple or promotional wording. Do not tune grades to satisfy a famous-paper expectation.

Read [the output contract](references/output-contract.md), then the current questions in `shirabe/outcome_rubric.py` in the Shirabe checkout. Use the supplied paper assignment; if no paper is supplied, ask for its title/DOI or file. Resolve and read its original text to establish identity and central claims. Read the full paper where accessible; label an abstract-only assessment explicitly. Keep all available input text; do not apply arbitrary character caps. A large context window does not substitute for selecting relevant sources.

If title, DOI and original text disagree, record `identity_resolution` with assigned and resolved identifiers plus an explanation, and set its `training_eligible` to false. Preserve the assigned ID for tracking; do not silently attach the grade to mismatched cached text.

Check article type as well as identity. A research highlight, news item or summary of another team’s paper must not pair the summary writer’s language with the researchers’ outcomes. Preserve the record for audit and explicitly exclude it from training; do not silently substitute the underlying study.

Before researching follow-ups, record the original central claims as a short `central_claims` array. Assess those same claims throughout; do not redefine the central contribution after seeing which component survived.

Research adaptively. Follow the paper's references forward through independent replications, extensions, critiques, retractions/corrections, reviews, official implementations and documented adoption. Search by title/DOI and by the actual names of its claims, methods, datasets and follow-up studies. Read the sources, not just search snippets. Follow new leads rather than exhausting a fixed query list. Do not filter discovery by web-page update date; establish the dates of the findings themselves. A fixed four-year window applies only if explicitly requested.

Use primary research for scientific claims and official artifacts/documentation for adoption. Distinguish substantive reuse from a passing citation; distinguish negative attention from validation. An independent group means a different research team, not merely a different website. A product is optional for theoretical work. A retraction is not automatically proof that every result was false: inspect the reason. Later supersession does not erase earlier successful adoption. Record contradictory findings and bounded claims rather than forcing a single story.

Answer every question with `yes`, `no`, `unknown`, or `not_applicable`:
- `yes` / `no`: a defensible factual answer grounded in a source you read; cite it and explain the connection to this paper.
- `unknown`: evidence is missing or conflicting. Failure to find adoption is not proof of nonadoption.
- `not_applicable`: the question logically does not apply; explain why. Do not use it to hide missing evidence.

Use concise paraphrases with source IDs. Do not fabricate quotations to satisfy a checker. The mechanical aggregator computes the index; never invent a standalone 0–100 grade. Coverage and unknown-answer bounds accompany it. Record observation horizon and sampling cohort: lifetime landmark checks are not an unbiased training sample.

For live Shirabe work, announce each substantial stage with `scripts/agent_grade.py progress`, and publish the completed artifact through `scripts/agent_grade.py publish`. The UI reads these records automatically. Keep model inputs, source downloads and draft notes private; publish factual annotations, source metadata and the search trail. Do not edit model weights, training labels for other papers, or application code as part of a grading assignment.

Stop when the main lines of independent evidence and contrary evidence have been checked and further searches repeat the same leads. Explain remaining unknowns. If a rubric question is structurally unanswerable, report the issue rather than quietly changing its definition. A second independent grade can be requested for label-quality evaluation; do not show that grader the first result.

Apply the question literally. A later repair of a limitation does not alone establish that the original contribution survived that limitation. Conditional theoretical plausibility does not establish empirical reproduction. Research prompted by a disputed claim may count as uptake without validating the claim or demonstrating utility from applying it. For advantages, name the particular advantage and comparison conditions; do not generalize a quality gain into a speed gain or universal superiority.
