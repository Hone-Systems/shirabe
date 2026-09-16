"""Frozen expansion experiment: paper-level controls, learning curves, repeated seeds.

Never inspect the fresh test split until every scheduled fit is selected on validation.
"""

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter

import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts import train_outcomes as base
from shirabe.outcome_rubric import IDS, QUESTIONS
from shirabe.research_clients import write_json

ROOT = base.ROOT
PUBLIC = ROOT / "artifacts/outcome-training"
ROUND = PUBLIC / "rounds/round-2"
PRIVATE = ROOT / "artifacts/private/outcome-training/round-2"
SEEDS = [42, 43, 44]
FRACTIONS = [1 / 3, 2 / 3, 1.0]
VARIANTS = ["wording", "original", "topic"]


def normalize_doi(value):
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", value.strip().lower()).rstrip(".")


def load_records(allow_incomplete=False):
    manifests = [
        json.loads(p.read_text()) for p in sorted((ROOT / "data/outcomes/expansion").glob("*-manifest.json"))
    ]
    planned = {r["id"]: r for m in manifests for r in m["papers"]}
    fresh_ids = sorted(planned)
    historical = json.loads((ROOT / "data/outcomes/expansion/historical-augmentation.json").read_text())[
        "papers"
    ]
    targeted_ids = {r["id"] for r in historical}
    planned.update({r["id"]: r for r in historical})
    records = [
        json.loads(p.read_text()) for p in sorted((ROOT / "data/outcomes/labels").glob("agent-*.json"))
    ]
    indexed = {r["id"]: r for r in records}
    missing = sorted(set(planned) - set(indexed))
    if missing and not allow_incomplete:
        raise ValueError(f"Expansion incomplete: {len(missing)} of {len(planned)} papers remain")
    unique, aliases = {}, []
    for r in records:
        key = normalize_doi(r.get("doi") or r["id"])
        if r["id"] in planned:
            assignment = planned[r["id"]]
            if r["split"] != assignment["split"]:
                raise ValueError("Published split differs from frozen manifest: " + r["id"])
            bucket = (
                int(hashlib.sha256((assignment.get("split_key") or key).encode()).hexdigest()[:8], 16) % 10
            )
            expected = (
                "test"
                if bucket < 2
                else "validation"
                if bucket < 4
                else "calibration"
                if bucket == 4
                else "train"
            )
            if r["id"] not in targeted_ids and r["split"] != expected:
                raise ValueError("Split hash mismatch: " + r["id"])
        if key in unique:
            prior = unique[key]
            if prior["split"] != r["split"]:
                raise ValueError("Duplicate original crosses splits: " + key)
            aliases.append({"id": r["id"], "canonical_id": prior["id"], "doi": key})
            continue
        unique[key] = r
    all_records = list(unique.values())
    usable = [
        r
        for r in all_records
        if (r.get("identity_resolution") or {}).get("training_eligible") is not False
        and r["aggregate"]["known_answers"] > 0
    ]
    return (
        all_records,
        usable,
        {
            "planned": len(planned),
            "reviewed": len(set(planned) & set(indexed)),
            "missing": missing,
            "aliases": aliases,
            "fresh_ids": fresh_ids,
            "targeted_training_ids": sorted(targeted_ids),
        },
    )


def counts(records):
    c = Counter(a["answer"] for r in records for a in r["research"]["answers"])
    return {
        "papers": len(records),
        "yes": c["yes"],
        "no": c["no"],
        "unknown": c["unknown"],
        "not_applicable": c["not_applicable"],
    }


def prevalence(records):
    y, m = zip(*(base.targets(r) for r in records), strict=True)
    y, m = np.array(y), np.array(m)
    support = m.sum(0)
    return ((y * m).sum(0) + 1) / (support + 2), support > 0


def summarize(records, probabilities, supervised):
    paper_brier, paper_loss, flat_y, flat_p, rows = [], [], [], [], []
    heads = {q: {"y": [], "p": []} for q in IDS}
    for r, raw in zip(records, probabilities, strict=True):
        y, mask = base.targets(r)
        mask &= supervised
        p = np.clip(np.array(raw), 1e-7, 1 - 1e-7)
        brier = float(np.mean((p[mask] - y[mask]) ** 2)) if mask.any() else None
        loss = (
            float(-np.mean(y[mask] * np.log(p[mask]) + (1 - y[mask]) * np.log1p(-p[mask])))
            if mask.any()
            else None
        )
        if mask.any():
            paper_brier.append(brier)
            paper_loss.append(loss)
            flat_y.extend(y[mask].tolist())
            flat_p.extend(p[mask].tolist())
        for j in np.flatnonzero(mask):
            heads[IDS[j]]["y"].append(y[j])
            heads[IDS[j]]["p"].append(p[j])
        rows.append(
            {
                "id": r["id"],
                "title": r["title"],
                "split": r["split"],
                "cohort": r["sampling_cohort"],
                "brier": brier,
                "log_loss": loss,
                "probabilities": [float(v) if supervised[j] else None for j, v in enumerate(p)],
                "known_labels": {
                    a["id"]: a["answer"] for a in r["research"]["answers"] if a["answer"] in ("yes", "no")
                },
            }
        )
    per_head = {}
    for q, d in heads.items():
        y, p = np.array(d["y"]), np.array(d["p"])
        both = len(set(y)) == 2
        per_head[q] = {
            "papers": len(y),
            "yes": int(y.sum()),
            "no": int(len(y) - y.sum()),
            "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
            "auc": float(roc_auc_score(y, p)) if both else None,
            "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)) if both else None,
        }
    aucs = [v["auc"] for v in per_head.values() if v["auc"] is not None]
    return {
        "known_items": len(flat_y),
        "scorable_papers": len(paper_brier),
        "positive_items": int(sum(flat_y)),
        "negative_items": int(len(flat_y) - sum(flat_y)),
        "brier": float(np.mean(paper_brier)) if paper_brier else None,
        "log_loss": float(np.mean(paper_loss)) if paper_loss else None,
        "mean_probability": float(np.mean(flat_p)) if flat_p else None,
        "auc": float(np.mean(aucs)) if aucs else None,
        "auc_heads": len(aucs),
        "metric_weighting": "Equal weight per paper over its known supported questions; AUC is macro across heads with both classes",
        "per_head": per_head,
        "papers": rows,
    }


def paired_interval(model, control, seed=731, repeats=2000):
    controls = {r["id"]: r for r in control["papers"]}
    differences = [
        controls[r["id"]]["brier"] - r["brier"]
        for r in model["papers"]
        if r["brier"] is not None and controls[r["id"]]["brier"] is not None
    ]
    if not differences:
        return {"papers": 0, "mean": None, "lower": None, "upper": None}
    d = np.array(differences)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(d, (repeats, len(d)), replace=True).mean(1)
    return {
        "papers": len(d),
        "mean": float(d.mean()),
        "lower": float(np.quantile(sampled, 0.025)),
        "upper": float(np.quantile(sampled, 0.975)),
        "method": "Paired paper-cluster percentile bootstrap, 2000 draws; positive favors model; conditional on this selected cohort and these labels",
    }


def rhetorical_features(text):
    words = re.findall(r"[a-z]+", text.lower())
    n = max(1, len(words))
    sentences = max(1, len(re.findall(r"[.!?](?:\s|$)", text)))
    features = {
        "log_words": np.log1p(len(words)),
        "words_per_sentence": len(words) / sentences,
        "mean_word_length": sum(map(len, words)) / n,
        "question_rate": text.count("?") / sentences,
        "semicolon_rate": text.count(";") / sentences,
    }
    for name, lexicon in {
        "hedging": "may might could suggest suggests possibly perhaps potentially likely unlikely appears seem seems",
        "confidence": "clearly certainly prove proves demonstrate demonstrates establish establishes definitive",
        "promotion": "novel groundbreaking remarkable unprecedented significant superior simple straightforward",
        "qualification": "however although despite unless limitation limitations except conditional",
        "discourse": "therefore because hence consequently moreover furthermore nevertheless",
        "first_person": "we our us",
    }.items():
        features[name] = sum(w in set(lexicon.split()) for w in words) / n
    return features


def control_fit(train, records, inputs, mode):
    def features(r):
        if mode == "rhetorical":
            return rhetorical_features(inputs[r["id"]]["wording_text"])
        return {
            "year": float(r["year"]),
            "field": str(r["field_id"]),
            "cohort": r["sampling_cohort"],
            "scope": inputs[r["id"]]["input_scope"],
        }

    vectorizer = DictVectorizer(sparse=False)
    x = vectorizer.fit_transform([features(r) for r in train])
    xx = vectorizer.transform([features(r) for r in records])
    prior, supported = prevalence(train)
    out = np.tile(prior, (len(records), 1))
    labels = [base.targets(r) for r in train]
    for j in range(len(IDS)):
        mask = np.array([m[j] for _, m in labels])
        y = np.array([v[j] for v, _ in labels])[mask]
        if len(set(y)) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        model.fit(x[mask], y)
        out[:, j] = model.predict_proba(xx)[:, 1]
    return out, supported


def predict(model, encoded, records, device):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for r in records:
            logits = []
            for chunk in encoded[r["id"]]:
                ids = torch.tensor([chunk], device=device)
                logits.append(model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0].cpu())
            predictions.append(torch.sigmoid(torch.stack(logits).mean(0)).numpy())
    return np.array(predictions)


def save(report):
    write_json(PUBLIC / "report.json", report)
    write_json(ROUND / "report.json", report)


def build_report(all_records, records, expansion):
    train = [r for r in records if r["split"] == "train"]
    _, supervised = prevalence(train)
    qcounts = [
        {
            "id": i,
            "dimension": d,
            "question": q,
            **dict(Counter(a["answer"] for r in train for a in r["research"]["answers"] if a["id"] == i)),
        }
        for i, d, q in QUESTIONS
    ]
    dataset = {
        "agent_records": len(all_records),
        "included_papers": len(records),
        "excluded_identity_or_type": sum(
            (r.get("identity_resolution") or {}).get("training_eligible") is False for r in all_records
        ),
        "splits": {
            s: counts([r for r in records if r["split"] == s])
            for s in ["train", "validation", "calibration", "test", "challenge"]
        },
        "questions": qcounts,
        "supported_heads": [q for q, m in zip(IDS, supervised, strict=True) if m],
    }
    return {
        "run_id": "agent-rubric-expansion-round-2",
        "labels_sha256": hashlib.sha256(json.dumps(all_records, sort_keys=True).encode()).hexdigest(),
        "status": "preparing",
        "stage": "Preparing expansion inputs",
        "architecture": {"name": "BERT-Mini", "layers": 4, "heads": 4, "hidden_size": 256, "outputs": 20},
        "config": {
            "epochs": 8,
            "learning_rate": 3e-5,
            "max_chunk_tokens": 512,
            "seeds": SEEDS,
            "fractions": FRACTIONS,
            "base_model": base.BASE,
            "revision": base.REVISION,
        },
        "dataset": dataset,
        "expansion": expansion,
        "runs": [],
        "experiments": [],
        "controls": {},
        "promotion": {
            "eligible": False,
            "reason": "Expansion experiment is not yet evaluated; Analyze serves the citation baseline.",
        },
        "rl_readiness": base.rl_readiness(dataset),
        "limitations": [
            "Selected social-science replication cohorts are not representative of all science.",
            "A failed replication concerns a specified result, not every claim or all scientific utility.",
            "All available original text is chunked; full text and abstract input scopes remain visible.",
            "Unknown and N/A targets are masked. Answers within a paper are correlated.",
            "Outcome labels are agent judgments; independent checks do not establish ground truth.",
            "Metric averages give each scorable paper equal weight. AUC averages only heads with both classes.",
            "Legacy test papers were evaluated previously; fresh expansion performance is reported separately.",
            "No calibrated overall success percentage is established.",
        ],
    }


def fit_one(train, val, inputs, tokenizer, variant, fraction, seed, report, device):
    ordered = sorted(train, key=lambda r: hashlib.sha256((str(seed) + r["id"]).encode()).hexdigest())
    selected = ordered[: max(1, int(np.ceil(len(ordered) * fraction)))]
    prior, supported = prevalence(selected)
    name = f"{variant}-n{len(selected)}-seed{seed}"
    out = PRIVATE / name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "metrics.json"
    checkpoint = out / "weights.pt"
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "labels": report["labels_sha256"],
                "config": report["config"],
                "ids": [r["id"] for r in selected],
                "variant": variant,
                "seed": seed,
                "inputs": report["inputs"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get("fingerprint") == fingerprint and checkpoint.exists():
            return previous, checkpoint
    torch.manual_seed(seed)
    random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        base.BASE, revision=base.REVISION, num_labels=20, local_files_only=True, attn_implementation="eager"
    ).to(device)
    if report["config"].get("prior_initialization"):
        with torch.no_grad():
            model.classifier.bias.copy_(
                torch.tensor(np.log(prior / (1 - prior)), dtype=model.classifier.bias.dtype, device=device)
            )
    report["architecture"]["parameters"] = sum(p.numel() for p in model.parameters())
    encoded = {
        r["id"]: base.chunks(tokenizer, inputs[r["id"]][variant + "_text"] or "[SUBJECT]")
        for r in selected + val
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    run = {
        "id": variant,
        "experiment_id": name,
        "seed": seed,
        "fraction": fraction,
        "training_papers": len(selected),
        "training_ids": [r["id"] for r in selected],
        "history": [],
        "steps": [],
        "chunks": sum(map(len, encoded.values())),
        "best_epoch": None,
        "metrics": {},
        "fingerprint": fingerprint,
        "constant_positive_baseline": {},
        "prevalence": prior.tolist(),
        "supported": supported.tolist(),
    }
    if fraction == 1 and seed == 42:
        report["runs"].append(run)
    best = float("inf")
    for epoch in range(1, report["config"]["epochs"] + 1):
        started = time.perf_counter()
        model.train()
        rows = list(selected)
        random.shuffle(rows)
        for r in rows:
            optimizer.zero_grad(set_to_none=True)
            y, m = base.targets(r)
            y = torch.tensor(y, dtype=torch.float32, device=device)
            m = torch.tensor(m, device=device)
            cs = encoded[r["id"]]
            paper_loss = 0
            for chunk in cs:
                ids = torch.tensor([chunk], device=device)
                loss = base.masked_loss(
                    model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits, y, m
                ) / len(cs)
                loss.backward()
                paper_loss += loss.item()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1))
            optimizer.step()
            run["steps"].append(
                {"step": len(run["steps"]) + 1, "epoch": epoch, "loss": paper_loss, "gradient_norm": norm}
            )
        tr = summarize(selected, predict(model, encoded, selected, device), supported)
        va = summarize(val, predict(model, encoded, val, device), supported)
        run["history"].append(
            {
                "epoch": epoch,
                "train_loss": tr["log_loss"],
                "validation_loss": va["log_loss"],
                "train_brier": tr["brier"],
                "validation_brier": va["brier"],
                "seconds": time.perf_counter() - started,
            }
        )
        if va["log_loss"] is not None and va["log_loss"] < best:
            best = va["log_loss"]
            run["best_epoch"] = epoch
            torch.save(model.state_dict(), checkpoint)
        report["stage"] = f"{name}: epoch {epoch}/{report['config']['epochs']}"
        save(report)
        print(report["stage"], flush=True)
        patience = report["config"].get("patience")
        if patience and run["best_epoch"] is not None and epoch - run["best_epoch"] >= patience:
            break
    if not checkpoint.exists():
        raise ValueError("No scorable validation targets; no checkpoint selected")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    run["metrics"]["validation"] = summarize(val, predict(model, encoded, val, device), supported)
    run["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_json(result_path, run)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return run, checkpoint


def main():
    global ROUND, PRIVATE
    parser = argparse.ArgumentParser()
    parser.add_argument("--convergence", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.convergence:
        previous = json.loads((ROUND / "report.json").read_text())
        all_records = json.loads((PRIVATE / "frozen-labels.json").read_text())
        records = [
            r
            for r in all_records
            if (r.get("identity_resolution") or {}).get("training_eligible") is not False
            and r["aggregate"]["known_answers"] > 0
        ]
        expansion = previous["expansion"]
        ROUND = PUBLIC / "rounds/round-3"
        PRIVATE = ROOT / "artifacts/private/outcome-training/round-3"
    else:
        all_records, records, expansion = load_records(args.allow_incomplete)

    def configure(report):
        report["artifact_round"] = "round-3" if args.convergence else "round-2"
        if args.convergence:
            report["run_id"] = "agent-rubric-convergence-round-3"
            report["config"].update(epochs=32, patience=5, prior_initialization=True, fractions=[1.0])
            report["test_status"] = (
                "The earlier test is now a previously inspected development holdout. The cancer cohort remains untouched external evaluation."
            )
            report["limitations"].append(report["test_status"])
        return report

    report = configure(build_report(all_records, records, expansion))
    if args.prepare_only:
        # Keep last measured dashboard intact while labels are still arriving.
        base.OUT = ROUND / "preparation"
        prepared = base.prepare(records, report)
        print(
            json.dumps(
                {
                    "prepared": len(prepared),
                    "reviewed": expansion["reviewed"],
                    "planned": expansion["planned"],
                }
            )
        )
        return
    if args.allow_incomplete:
        raise ValueError("Incomplete cohorts may be prepared, never fitted")
    ROUND.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    save(report)
    inputs = base.prepare(records, report)
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 42
    excluded = []
    for r in records:
        languages = detect_langs(inputs[r["id"]]["original_text"])
        if languages[0].lang != "en" and languages[0].prob > 0.9:
            excluded.append(
                {
                    "id": r["id"],
                    "reason": "Confidently non-English input for English BERT",
                    "language": languages[0].lang,
                }
            )
    excluded_ids = {r["id"] for r in excluded}
    records = [r for r in records if r["id"] not in excluded_ids]
    report = configure(build_report(all_records, records, expansion))
    report["input_quality_exclusions"] = excluded
    report["label_audit"] = json.loads((ROOT / "data/outcomes/expansion/label-audit.json").read_text())
    if report["label_audit"]["papers"] != 6:
        raise ValueError("Scheduled blind audit is incomplete")
    report["inputs"] = [
        {
            "id": r["id"],
            "split": r["split"],
            "scope": inputs[r["id"]]["input_scope"],
            "characters": len(inputs[r["id"]]["original_text"]),
            "redacted_fraction": inputs[r["id"]]["redacted_character_fraction"],
            "wording_version": inputs[r["id"]].get("wording_version"),
            "wording_sha256": hashlib.sha256(inputs[r["id"]]["wording_text"].encode()).hexdigest(),
            "topic_sha256": hashlib.sha256(inputs[r["id"]]["topic_text"].encode()).hexdigest(),
            "sha256": inputs[r["id"]]["source_sha256"],
            "version_note": inputs[r["id"]]["provenance"].get(
                "version_note", "Source version not independently established"
            ),
        }
        for r in records
    ]
    write_json(PRIVATE / "frozen-labels.json", all_records)
    splits = {
        s: [r for r in records if r["split"] == s]
        for s in ["train", "validation", "calibration", "test", "challenge"]
    }
    if not counts(splits["train"])["no"]:
        raise ValueError("Still no negative training targets; expansion has not resolved readiness")
    prior, supported = prevalence(splits["train"])
    tokenizer = AutoTokenizer.from_pretrained(base.BASE, revision=base.REVISION, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    schedule = ([] if args.convergence else [(v, f, 42) for f in FRACTIONS[:-1] for v in VARIANTS]) + [
        (v, 1.0, s) for s in SEEDS for v in VARIANTS
    ]
    trained = []
    report["status"] = "training"
    for variant, fraction, seed in schedule:
        run, path = fit_one(
            splits["train"], splits["validation"], inputs, tokenizer, variant, fraction, seed, report, device
        )
        if fraction == 1 and seed == 42 and not any(r["id"] == variant for r in report["runs"]):
            report["runs"].append(run)
        report["experiments"].append({k: v for k, v in run.items() if k not in ("steps", "fingerprint")})
        if fraction == 1:
            trained.append((run, path))
        save(report)
    # All scheduled fits and checkpoints are fixed before accessing fresh test outcomes.
    report["stage"] = "Evaluating frozen fits and controls"
    report["status"] = "evaluating"
    save(report)
    cohorts = {
        **splits,
        "fresh_test": [r for r in splits["test"] if r["id"] in expansion["fresh_ids"]],
        "fresh_validation": [r for r in splits["validation"] if r["id"] in expansion["fresh_ids"]],
    }
    for name in ("prevalence", "always_yes", "rhetorical", "metadata"):
        predictions = (
            np.tile(prior if name == "prevalence" else np.ones(len(IDS)), (len(records), 1))
            if name in ("prevalence", "always_yes")
            else control_fit(splits["train"], records, inputs, name)[0]
        )
        mapping = {r["id"]: p for r, p in zip(records, predictions, strict=True)}
        report["controls"][name] = {
            s: summarize(rs, [mapping[r["id"]] for r in rs], supported) for s, rs in cohorts.items()
        }
    for run, path in trained:
        model = AutoModelForSequenceClassification.from_pretrained(
            base.BASE,
            revision=base.REVISION,
            num_labels=20,
            local_files_only=True,
            attn_implementation="eager",
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        encoded = {
            r["id"]: base.chunks(tokenizer, inputs[r["id"]][run["id"] + "_text"] or "[SUBJECT]")
            for r in records
        }
        probs = predict(model, encoded, records, device)
        mapping = {r["id"]: p for r, p in zip(records, probs, strict=True)}
        run["metrics"] = {
            s: summarize(rs, [mapping[r["id"]] for r in rs], supported) for s, rs in cohorts.items()
        }
        run["constant_positive_baseline"] = {
            s: {"brier": m["brier"], "log_loss": m["log_loss"]}
            for s, m in report["controls"]["always_yes"].items()
        }
        run["fresh_test_comparisons"] = {
            name: paired_interval(run["metrics"]["fresh_test"], report["controls"][name]["fresh_test"])
            for name in report["controls"]
        }
        for entry in report["experiments"]:
            if entry["experiment_id"] == run["experiment_id"]:
                entry.update(
                    {"metrics": run["metrics"], "fresh_test_comparisons": run["fresh_test_comparisons"]}
                )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        save(report)
    # Topic/cohort leakage probe uses training/validation only, never fresh test for tuning.
    xtrain = [inputs[r["id"]]["wording_text"] for r in splits["train"]]
    xval = [inputs[r["id"]]["wording_text"] for r in splits["validation"]]
    ytrain = [r["sampling_cohort"] for r in splits["train"]]
    yval = [r["sampling_cohort"] for r in splits["validation"]]
    if len(set(ytrain)) > 1 and xval:
        probe = make_pipeline(
            TfidfVectorizer(min_df=1, max_features=10000, ngram_range=(1, 2)),
            LogisticRegression(max_iter=1000, C=1, random_state=42),
        )
        probe.fit(xtrain, ytrain)
        pred = probe.predict(xval)
        report["leakage_probe"] = {
            "target": "sampling cohort (proxy for subject/domain)",
            "validation_accuracy": float(np.mean(pred == np.array(yval))),
            "majority_accuracy": float(np.mean(np.array(yval) == Counter(ytrain).most_common(1)[0][0])),
            "classes": len(set(ytrain)),
            "papers": len(yval),
            "limitation": "Not a proof of topic independence; cohort includes publication-era and document-format cues.",
        }
    wording = [r for r, _ in trained if r["id"] == "wording"]
    consistent = all(
        r["fresh_test_comparisons"]["prevalence"]["lower"] is not None
        and r["fresh_test_comparisons"]["prevalence"]["lower"] > 0
        for r in wording
    )
    report["conclusion"] = {
        "consistent_positive_lower_bound_vs_prevalence": consistent,
        "wording_fresh_test_brier": [r["metrics"]["fresh_test"]["brier"] for r in wording],
        "interpretation": "Promising narrow-cohort signal; external validation and topic controls still required."
        if consistent
        else "A reliable wording signal is not established: held-out improvement is uncertain or inconsistent. Do not interpret training loss as success accuracy.",
    }
    report["promotion"] = {
        "eligible": False,
        "reason": report["conclusion"]["interpretation"]
        + " Analyze serves the predetermined seed42 wording checkpoint as an explicitly experimental model at the user’s request.",
    }
    report["rl_readiness"]["reason"] = (
        "Mixed rewards are now present, but a reliable held-out wording signal and a separate RL benefit are not established. RL was not run."
    )
    candidate = next(r for r, _ in trained if r["id"] == "wording" and r["seed"] == 42)
    checkpoint = next(p for r, p in trained if r is candidate)
    from scripts.extract_wording import VERSION

    write_json(
        PUBLIC / "served.json",
        {
            "model_id": report["run_id"] + "-wording-seed42",
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "base_model": base.BASE,
            "revision": base.REVISION,
            "wording_version": VERSION,
            "supported": candidate["supported"],
            "question_counts": report["dataset"]["questions"],
            "training_papers": candidate["training_papers"],
            "experimental": True,
        },
    )
    report["serving"] = {"experimental": True, "variant": "wording", "seed": 42, "validated": False}
    report["status"] = "complete"
    report["stage"] = "Expansion experiment complete"
    save(report)
    print(json.dumps(report["conclusion"], indent=2))


if __name__ == "__main__":
    main()
