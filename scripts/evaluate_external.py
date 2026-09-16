"""Untouched cancer-cohort evaluation of already selected checkpoints; no optimization."""

import hashlib
import json

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts import train_outcomes as base
from scripts.train_expansion import control_fit, paired_interval, predict, prevalence, summarize
from shirabe.research_clients import write_json

ROOT = base.ROOT
OUT = ROOT / "artifacts/outcome-training"


def main():
    manifest = json.loads((ROOT / "data/outcomes/external/cancer-manifest.json").read_text())
    planned = {r["id"] for r in manifest["papers"]}
    indexed = {
        r["id"]: r
        for p in (ROOT / "data/outcomes/labels").glob("agent-*.json")
        if (r := json.loads(p.read_text()))
    }
    if planned - set(indexed):
        raise ValueError(f"External grades incomplete: {len(planned - set(indexed))} remain")
    records = [
        indexed[i]
        for i in sorted(planned)
        if (indexed[i].get("identity_resolution") or {}).get("training_eligible") is not False
        and indexed[i]["aggregate"]["known_answers"] > 0
    ]
    if any(r["split"] != "external" for r in records):
        raise ValueError("External assignment changed")
    report = json.loads((OUT / "report.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Scheduled model selection has not finished")
    artifact_round = report.get("artifact_round", "round-2")
    frozen = json.loads(
        (ROOT / "artifacts/private/outcome-training" / artifact_round / "frozen-labels.json").read_text()
    )
    selected_ids = set(next(r for r in report["runs"] if r["id"] == "wording")["training_ids"])
    train = [r for r in frozen if r["id"] in selected_ids]
    if selected_ids & planned:
        raise ValueError("External papers leaked into training")
    if any(r["id"] in planned for r in frozen):
        raise ValueError("External cohort was present during original dataset snapshot")
    original_signature = report["labels_sha256"]
    base.OUT = OUT / "external-preparation"
    inputs = base.prepare(
        train + records, {"status": "preparing", "stage": "External wording preparation", "runs": []}
    )
    prior, supported = prevalence(train)
    result = {
        "cohort": "rpcb_completed_external_2021",
        "planned_papers": len(planned),
        "scorable_papers": len(records),
        "label_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        "training_label_sha256": original_signature,
        "selection": manifest.get("selection", "Completed replication subset; feasibility selection bias"),
        "runs": [],
        "controls": {},
        "inputs": [
            {
                "id": r["id"],
                "scope": inputs[r["id"]]["input_scope"],
                "source_sha256": inputs[r["id"]]["source_sha256"],
            }
            for r in records
        ],
        "note": "External cohort labels and text were never used for optimization, checkpoint selection, hyperparameter tuning or calibration. The model architecture and training protocol were fixed before external evaluation. A completed-experiment subset is not representative of all cancer research.",
    }
    for name in ("prevalence", "always_yes", "rhetorical", "metadata"):
        probabilities = (
            np.tile(prior if name == "prevalence" else np.ones(20), (len(records), 1))
            if name in ("prevalence", "always_yes")
            else control_fit(train, records, inputs, name)[0]
        )
        result["controls"][name] = summarize(records, probabilities, supported)
    tokenizer = AutoTokenizer.from_pretrained(base.BASE, revision=base.REVISION, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    for experiment in report["experiments"]:
        if experiment["fraction"] != 1:
            continue
        path = (
            ROOT
            / "artifacts/private/outcome-training"
            / artifact_round
            / experiment["experiment_id"]
            / "weights.pt"
        )
        if hashlib.sha256(path.read_bytes()).hexdigest() != experiment["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed")
        model = AutoModelForSequenceClassification.from_pretrained(
            base.BASE,
            revision=base.REVISION,
            num_labels=20,
            local_files_only=True,
            attn_implementation="eager",
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        encoded = {
            r["id"]: base.chunks(tokenizer, inputs[r["id"]][experiment["id"] + "_text"] or "[SUBJECT]")
            for r in records
        }
        metric = summarize(records, predict(model, encoded, records, device), supported)
        row = {
            "id": experiment["id"],
            "seed": experiment["seed"],
            "checkpoint_sha256": experiment["checkpoint_sha256"],
            "metrics": metric,
            "comparisons": {
                name: paired_interval(metric, control) for name, control in result["controls"].items()
            },
        }
        result["runs"].append(row)
        print(row["id"], row["seed"], metric["brier"], flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    write_json(OUT / "external-cancer.json", result)
    # Attach measured external results without altering weights, selection or original held-out data.
    current = json.loads((OUT / "report.json").read_text())
    if current["labels_sha256"] != original_signature:
        raise ValueError("Training report changed during external evaluation")
    current["external_evaluation"] = {k: v for k, v in result.items() if k not in ("inputs",)}
    for row in result["runs"]:
        for run in current["runs"]:
            if run["id"] == row["id"] and row["seed"] == 42:
                run["metrics"]["external"] = row["metrics"]
                run["constant_positive_baseline"]["external"] = {
                    "brier": result["controls"]["always_yes"]["brier"]
                }
    write_json(OUT / "report.json", current)
    write_json(OUT / "rounds" / artifact_round / "report.json", current)
    print("EXTERNAL COMPLETE", len(records), flush=True)


if __name__ == "__main__":
    main()
