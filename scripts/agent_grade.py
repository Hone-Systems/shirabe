"""Validate and publish autonomous research-agent grades to the live evidence explorer."""

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shirabe.outcome_rubric import IDS, VERSION, aggregate

OUT = ROOT / "data/outcomes"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def key(paper_id):
    return hashlib.sha256(json.dumps(paper_id, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def mark(paper, stage, **extra):
    write(
        OUT / "progress" / f"{key(paper['id'])}.json",
        {k: paper[k] for k in ["id", "title", "year", "field_id", "field", "split", "sampling_cohort"]}
        | {
            "stage": stage,
            "updated_at": datetime.now(UTC).isoformat(),
            "researcher_kind": "autonomous_agent",
            **extra,
        },
    )


def publish(data):
    for name in [
        "id",
        "title",
        "doi",
        "year",
        "field_id",
        "field",
        "split",
        "sampling_cohort",
        "as_of",
        "original_source",
        "sources",
        "answers",
        "search_log",
        "search_gaps",
        "summary",
        "researcher",
    ]:
        if name not in data:
            raise ValueError(f"Missing field: {name}")
    as_of = datetime.strptime(data["as_of"], "%Y-%m-%d").date()
    if as_of > datetime.now(UTC).date():
        raise ValueError("As-of date cannot be in the future")
    unresolved_original = not data["original_source"].get("read")
    if unresolved_original:
        resolution = data.get("identity_resolution") or {}
        if (
            data["original_source"].get("scope") != "unavailable"
            or resolution.get("training_eligible") is not False
            or not resolution.get("explanation")
            or any(a.get("answer") != "unknown" for a in data["answers"])
        ):
            raise ValueError(
                "Unread original requires unavailable scope, explicit training exclusion and all unknown answers"
            )
    sources = {s["source_id"]: s for s in data["sources"]}
    if len(sources) != len(data["sources"]):
        raise ValueError("Duplicate source IDs")
    for source in sources.values():
        if not source["url"].startswith(("https://", "http://")):
            raise ValueError("Source requires a public URL")
        if source.get("date_basis") not in ("publication", "observed_as_of"):
            raise ValueError("Source needs publication/observed_as_of date basis")
        if not re.fullmatch(r"\d{4}(?:-\d{2}-\d{2})?", source["date"]):
            raise ValueError("Source date must be YYYY or YYYY-MM-DD")
        if source["date"] > data["as_of"]:
            raise ValueError("Source date beyond as-of date")
    for answer in data["answers"]:
        if answer["id"] not in IDS or not answer.get("rationale"):
            raise ValueError("Every rubric answer needs an ID and rationale")
        if answer["answer"] not in ("yes", "no"):
            continue
        if not answer["evidence"]:
            raise ValueError("Binary answer requires evidence")
        for evidence in answer["evidence"]:
            source = sources.get(evidence["source_id"])
            if not source or source.get("read") is not True:
                raise ValueError("Evidence must reference a source actually read")
            if not evidence.get("explanation"):
                raise ValueError("Explain the evidence connection")
            if evidence["date"] != source["date"]:
                raise ValueError("Evidence date must match its cited source")
            if int(source["date"][:4]) < data["year"]:
                raise ValueError("Outcome evidence predates original paper")
    score = aggregate(data["answers"])
    record = {
        k: data[k] for k in ["id", "title", "doi", "year", "field_id", "field", "split", "sampling_cohort"]
    }
    record.update(
        {
            "rubric_version": VERSION,
            "retrieval_version": 4,
            "researcher_kind": "autonomous_agent",
            "review_status": "identity_unresolved" if unresolved_original else "reviewed",
            "repeat": 0,
            "labeled_at": datetime.now(UTC).isoformat(),
            "observed_through": data["as_of"],
            "followup_end": data["as_of"],
            "followup_years": as_of.year - data["year"],
            "horizon_kind": "observed_through_as_of",
            "input_scope": data["original_source"]["scope"],
            "original_source": data["original_source"],
            "central_claims": data.get("central_claims", []),
            "identity_resolution": data.get("identity_resolution"),
            "source_manifest": [
                s | {"characters": None, "method": s.get("kind", "agent_read_source"), "truncated": False}
                for s in data["sources"]
            ],
            "research": {
                "answers": data["answers"],
                "search_gaps": data["search_gaps"],
                "summary": data["summary"],
            },
            "research_plan": {
                "queries": [s["query"] for s in data["search_log"]],
                "search_log": data["search_log"],
            },
            "validation_rejected": [],
            "aggregate": score,
            "llm": {
                "model": data["researcher"],
                "usage": {"input_tokens": None, "output_tokens": None},
                "truncated": False,
            },
        }
    )
    # Preserve complete reviewed annotations; quote-free schema keeps paper text private.
    write(OUT / "labels" / f"agent-{key(data['id'])}.json", record)
    mark(record, "complete", known_answers=score["known_answers"], score=score["score"])
    return record


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("publish")
    p.add_argument("artifact", type=Path)
    p = sub.add_parser("progress")
    for name, kind, default in [
        ("id", str, None),
        ("title", str, None),
        ("year", int, None),
        ("field-id", int, None),
        ("field", str, None),
        ("stage", str, None),
        ("note", str, ""),
        ("split", str, "challenge"),
        ("sampling-cohort", str, "agent_pilot"),
    ]:
        p.add_argument("--" + name, type=kind, default=default, required=default is None)
    args = parser.parse_args()
    if args.command == "publish":
        record = publish(json.loads(args.artifact.read_text()))
        print(
            json.dumps(
                {
                    "published": record["id"],
                    "score": record["aggregate"]["score"],
                    "coverage": record["aggregate"]["coverage"],
                    "known": record["aggregate"]["known_answers"],
                }
            )
        )
    else:
        mark(vars(args), args.stage, note=args.note)
        print(json.dumps({"id": args.id, "stage": args.stage}))


if __name__ == "__main__":
    main()
