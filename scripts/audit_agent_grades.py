"""Audit all assigned agent grades and save a completion summary when none are missing."""

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from shirabe.outcome_rubric import IDS, aggregate

folder = root / "data/outcomes"
manifest = json.loads((folder / "agent_run.json").read_text())
records = [json.loads(p.read_text()) for p in (folder / "labels").glob("agent-*.json")]
ids = [r["id"] for r in records]
assert len(ids) == len(set(ids)), "Duplicate agent records"
assert not set(ids) - set(manifest["paper_ids"]), "Unexpected assignments"
missing = set(manifest["paper_ids"]) - set(ids)
for r in records:
    a = r["research"]["answers"]
    assert len(a) == 20 and {x["id"] for x in a} == set(IDS)
    assert aggregate(a) == r["aggregate"], r["title"]
    sources = {s["source_id"]: s for s in r["source_manifest"]}
    assert len(sources) == len(r["source_manifest"])
    for s in sources.values():
        if len(s["date"]) == 10:
            date.fromisoformat(s["date"])
        assert s["date"] <= r["observed_through"]
    for x in a:
        assert x["rationale"]
        if x["answer"] in ("yes", "no"):
            assert x["evidence"]
            for e in x["evidence"]:
                s = sources[e["source_id"]]
                assert s["read"] is True and s["date"] == e["date"] and e["explanation"]
nonchallenge = [r for r in records if r["split"] != "challenge"]
summary = {
    "run_id": manifest["run_id"],
    "expected": len(manifest["paper_ids"]),
    "reviewed": len(records),
    "missing_ids": sorted(missing),
    "complete": not missing,
    "answers": dict(Counter(a["answer"] for r in records for a in r["research"]["answers"])),
    "aggregate_eligible": sum(r["aggregate"]["score"] is not None for r in records),
    "no_known_answers": sum(r["aggregate"]["known_answers"] == 0 for r in records),
    "training_exclusions": [
        {"id": r["id"], "title": r["title"], "reason": r["identity_resolution"]["explanation"]}
        for r in records
        if (r.get("identity_resolution") or {}).get("training_eligible") is False
    ],
    "non_challenge": {
        "papers": len(nonchallenge),
        "answers": dict(Counter(a["answer"] for r in nonchallenge for a in r["research"]["answers"])),
        "aggregate_eligible": sum(r["aggregate"]["score"] is not None for r in nonchallenge),
    },
    "source_entries": sum(len(r["source_manifest"]) for r in records),
    "logged_searches": sum(len(r["research_plan"]["queries"]) for r in records),
    "limitations": [
        "Single-agent judgments; independent repeat agreement is not measured.",
        "Age and field opportunities differ. Agent lifetime outcomes and earlier four-year grades are not directly comparable experiments.",
        "Known question answers from the same source can be correlated.",
        "Challenge papers are excluded from fitting; see artifacts/outcome-training/report.json for diagnostic fit status.",
    ],
}
print(json.dumps(summary, indent=2))
if not missing:
    (folder / "agent_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
