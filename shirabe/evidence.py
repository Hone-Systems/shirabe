"""Read-only public evaluation dataset; never loads private source text or credentials."""

import json
from collections import Counter

from shirabe.inference import ROOT
from shirabe.outcome_rubric import QUESTIONS, VERSION


def evaluation_data():
    folder = ROOT / "data/outcomes"
    records = []
    for path in (folder / "labels").glob("*.json"):
        row = json.loads(path.read_text())
        if row.get("retrieval_version", 0) >= 2 and row.get("repeat") == 0:
            records.append(row)
    latest = {}
    for row in records:
        if row["id"] not in latest or row.get("retrieval_version", 0) > latest[row["id"]].get(
            "retrieval_version", 0
        ):
            latest[row["id"]] = row
    previous = {}
    for row in records:
        if row.get("researcher_kind") == "autonomous_agent":
            continue
        if row["id"] not in previous or row.get("retrieval_version", 0) > previous[row["id"]].get(
            "retrieval_version", 0
        ):
            previous[row["id"]] = row
    for paper_id, row in latest.items():
        if row.get("researcher_kind") == "autonomous_agent" and paper_id in previous:
            old = previous[paper_id]
            row["previous_review"] = {
                "known_answers": old["aggregate"]["known_answers"],
                "coverage": old["aggregate"]["coverage"],
                "score": old["aggregate"]["score"],
                "followup_end": old["followup_end"],
            }
    records = list(latest.values())
    records.sort(key=lambda r: (r["split"] != "challenge", r["field_id"], r["year"], r["id"]))
    progress = [json.loads(p.read_text()) for p in (folder / "progress").glob("*.json")]
    selection = (
        json.loads((folder / "selection.json").read_text()) if (folder / "selection.json").exists() else {}
    )
    planned_ids = {p["id"] for p in selection.get("papers", [])}
    for manifest in (folder / "expansion").glob("*-manifest.json"):
        planned_ids.update(p["id"] for p in json.loads(manifest.read_text()).get("papers", []))
    for manifest in (folder / "external").glob("*-manifest.json"):
        planned_ids.update(p["id"] for p in json.loads(manifest.read_text()).get("papers", []))
    augmentation = folder / "expansion/historical-augmentation.json"
    if augmentation.exists():
        planned_ids.update(p["id"] for p in json.loads(augmentation.read_text())["papers"])
    states = Counter(a["answer"] for r in records for a in r["research"]["answers"])
    return {
        "rubric_version": VERSION,
        "questions": [{"id": i, "dimension": d, "question": q} for i, d, q in QUESTIONS],
        "records": records,
        "progress": progress,
        "planned_papers": len(planned_ids),
        "summary": {
            "papers": len(records),
            "scored": sum(r["aggregate"]["score"] is not None for r in records),
            "answer_counts": dict(states),
            "full_text_candidates": sum(r["input_scope"] == "full_text_candidate" for r in records),
            "agent_reviewed": sum(r.get("researcher_kind") == "autonomous_agent" for r in records),
        },
        "status": "research pilot; labels are LLM-assisted and require audit",
        "method": {
            "horizon": "Agent grades: observed through the stated as-of date. Earlier provisional batches used a four-year window.",
            "unknown": "Missing evidence is not a negative outcome",
            "score": "Equal mean of four dimension indices, shown only with sufficient evidence; not a probability",
            "split": "Challenge papers are excluded from model fitting and selection",
            "baseline": "Analyze serves the experimental wording outcome transformer. The earlier citation model remains available through Citation baseline.",
        },
    }
