"""Diagnostic full-text/chunked BERT-Mini fit to agent rubric labels, never unknown=negative."""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.extract_wording import VERSION as WORDING_VERSION
from scripts.extract_wording import extract
from shirabe.outcome_rubric import IDS, QUESTIONS
from shirabe.research_clients import ResearchClients, write_json

OUT = ROOT / "artifacts/outcome-training"
PRIVATE = ROOT / "artifacts/private/outcome-training"
AGENT = ROOT / "artifacts/private/agent-grades"
BASE = "google/bert_uncased_L-4_H-256_A-4"
REVISION = "387825ce42dbb39b87911cdf8e383ee3b25184f8"
# Local original texts from the agent research. Follow-up documents are never predictor inputs.
LOCAL = {
    "W4247974275": "root-sources/belgium-original.txt",
    "W2292065295": "emri-original.pdf.txt",
    "W2883944593": "root-sources/opinion-original.txt",
    "W3158745291": "batch-2-sources/automotive-original.txt",
    "W3081066862": "root-sources/somalia-original.txt",
    "W4287324271": "batch-2-sources/territorial.txt",
    "W2604283208": "batch1-sources/dgcr8-original.txt",
    "W3164529735": "batch1-sources/dialog.txt",
    "W3007451419": "batch-2-sources/intraday-original.txt",
    "W2922482404": "auction-original.pdf.txt",
    "W2502906121": "cnp-original.pdf.txt",
    "W4412375198": "batch1-sources/beres-original.txt",
    "challenge-bert": "batch1-sources/bert-original.txt",
}


def targets(record):
    by_id = {a["id"]: a["answer"] for a in record["research"]["answers"]}
    return np.array([float(by_id[i] == "yes") for i in IDS]), np.array(
        [by_id[i] in ("yes", "no") for i in IDS]
    )


def chunks(tokenizer, text, size=512):
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    return [
        [tokenizer.cls_token_id, *ids[i : i + size - 2], tokenizer.sep_token_id]
        for i in range(0, len(ids), size - 2)
    ]


def masked_loss(logits, values, mask):
    if not mask.any():
        raise ValueError("No known targets for this paper")
    per = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, values.expand_as(logits), reduction="none"
    )
    return (per * mask).sum() / (mask.sum() * len(logits))


def prepare(records, report):
    clients = ResearchClients(ROOT.parent / "senki", 50)
    corpus = {p["id"]: p for p in map(json.loads, (ROOT / "data/papers.jsonl").read_text().splitlines())}
    old = {}
    for f in (ROOT / "artifacts/private/research/labels").glob("*.json"):
        r = json.loads(f.read_text())
        old[r["id"]] = r
    expansion_inputs = {}
    for f in sorted((AGENT / "expansion").glob("*-inputs.json"), key=lambda p: ("blind" in p.name, p.name)):
        expansion_inputs.update(json.loads(f.read_text()))
    cache = PRIVATE / "inputs"
    cache.mkdir(parents=True, exist_ok=True)

    def one(r):
        path = cache / (hashlib.sha256(r["id"].encode()).hexdigest() + ".json")
        assigned = expansion_inputs.get(r["id"])
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get("wording_version") == WORDING_VERSION and (
                not assigned
                or cached.get("provenance", {}).get("raw_sha256")
                == hashlib.sha256(Path(assigned["path"]).read_bytes()).hexdigest()
            ):
                return cached
        key = r["id"].split("/")[-1]
        local = LOCAL.get(key)
        url = r["original_source"]["url"]
        scope = r["input_scope"]
        if assigned:
            if not assigned.get("identity_verified"):
                raise ValueError("Original identity is unverified: " + r["id"])
            text = Path(assigned["path"]).read_text()
            scope = assigned["scope"]
            provenance = {
                "url": assigned["url"],
                "local_source": assigned["path"],
                "raw_sha256": hashlib.sha256(Path(assigned["path"]).read_bytes()).hexdigest(),
                "identity_verified": True,
                "version_note": assigned.get("version_note", "Version not independently established"),
            }
        elif local:
            text = (AGENT / local).read_text()
            provenance = {"local_source": local, "url": url}
            scope = "full_text_extraction"
        elif "abstract" in scope:
            text = corpus.get(r["id"], {}).get("abstract") or old.get(r["id"], {}).get("original_text", "")
            provenance = {"url": url, "method": "indexed_original_abstract"}
            scope = "abstract"
        else:
            source = clients.fetch(url)
            text = source["text"]
            provenance = {k: v for k, v in source.items() if k != "text"}
            if len(text) < 500 or "just a moment" in text[:200].lower():
                text = corpus.get(r["id"], {}).get("abstract", "")
                scope = "abstract_fallback"
            else:
                scope = "full_text_extraction_unverified_version"
        if len(text) < 100:
            raise ValueError("No usable original input: " + r["id"])
        wording = extract(r | {"original_text": text, "input_scope": scope}, clients)
        out = {
            k: wording[k]
            for k in [
                "original_text",
                "wording_text",
                "topic_text",
                "redacted_character_fraction",
                "source_sha256",
                "input_scope",
            ]
        }
        out.update(id=r["id"], provenance=provenance, wording_version=WORDING_VERSION)
        write_json(path, out)
        return out

    result = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(one, r): r for r in records}
        for f in as_completed(futures):
            row = f.result()
            result[row["id"]] = row
            report["stage"] = f"Preparing wording inputs {len(result)}/{len(records)}"
            report["prepared_inputs"] = len(result)
            write_json(OUT / "report.json", report)
    return result


def evaluate(model, encoded, records, supervised, device):
    model.eval()
    all_p = []
    all_y = []
    per_paper = []
    with torch.inference_mode():
        for r in records:
            cs = encoded[r["id"]]
            logits = []
            for c in cs:
                ids = torch.tensor([c], device=device)
                logits.append(model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0].cpu())
            prob = torch.sigmoid(torch.stack(logits).mean(0)).numpy()
            y, m = targets(r)
            m = m & supervised
            all_p.extend(prob[m].tolist())
            all_y.extend(y[m].tolist())
            per_paper.append(
                {
                    "id": r["id"],
                    "title": r["title"],
                    "split": r["split"],
                    "probabilities": [float(v) if supervised[i] else None for i, v in enumerate(prob)],
                    "known_labels": {
                        a["id"]: a["answer"] for a in r["research"]["answers"] if a["answer"] in ("yes", "no")
                    },
                }
            )
    p = np.clip(np.array(all_p), 1e-7, 1 - 1e-7)
    y = np.array(all_y)
    return {
        "known_items": len(y),
        "positive_items": int(y.sum()),
        "negative_items": int(len(y) - y.sum()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()) if len(y) else None,
        "brier": float(((p - y) ** 2).mean()) if len(y) else None,
        "mean_probability": float(p.mean()) if len(y) else None,
        "auc": None,
        "auc_reason": "Single-class or tiny diagnostic cohort; discrimination is not established.",
        "papers": per_paper,
    }


def positive_baseline(metric):
    n = metric["known_items"]
    return {
        "log_loss": float(
            -(metric["positive_items"] * np.log(1 - 1e-7) + metric["negative_items"] * np.log(1e-7)) / n
        )
        if n
        else None,
        "brier": metric["negative_items"] / n if n else None,
        "note": "Train-only empirical positive prevalence is 1; available heads only.",
    }


def rl_readiness(dataset):
    train = dataset["splits"]["train"]
    return {
        "status": "not_run",
        "eligible": False,
        "training_papers": train["papers"],
        "positive_targets": train["yes"],
        "negative_targets": train["no"],
        "reason": (
            f"Only {train['papers']} training papers and {train['yes']} yes / {train['no']} no targets. No contrasting training rewards; RL would reinforce always-yes predictions."
            if not train["yes"] or not train["no"]
            else "Both outcomes are present, but paper-level reward coverage and held-out improvement must be assessed before an RL experiment."
        ),
        "requirements": [
            "Independent training papers with source-backed positive and negative outcomes; unknown remains masked.",
            "A fixed reward definition and evaluation split isolated from reward fitting; challenge papers stay held out.",
            "A learning-curve study across paper counts and seeds that establishes useful generalization over supervised and constant baselines.",
        ],
        "note": "Rubric answers from one paper are correlated, not independent training examples. RL adds no new outcome evidence to these labels. No universal sample-count cutoff is assumed.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    all_records = sorted(
        [json.loads(p.read_text()) for p in (ROOT / "data/outcomes/labels").glob("agent-*.json")],
        key=lambda r: r["id"],
    )
    records = [
        r
        for r in all_records
        if (r.get("identity_resolution") or {}).get("training_eligible") is not False
        and r["aggregate"]["known_answers"] > 0
    ]
    splits = {
        s: [r for r in records if r["split"] == s]
        for s in ["train", "validation", "calibration", "test", "challenge"]
    }
    qcounts = [
        {
            "id": i,
            "dimension": d,
            "question": q,
            **dict(
                Counter(
                    a["answer"] for r in splits["train"] for a in r["research"]["answers"] if a["id"] == i
                )
            ),
        }
        for i, d, q in QUESTIONS
    ]
    supervised = np.array([sum(targets(r)[1][i] for r in splits["train"]) > 0 for i in range(20)])
    report = {
        "run_id": "agent-rubric-bert-mini-v1",
        "labels_sha256": hashlib.sha256(json.dumps(all_records, sort_keys=True).encode()).hexdigest(),
        "status": "preparing",
        "stage": "Preparing original-text inputs",
        "started_at": datetime.now(UTC).isoformat(),
        "architecture": {"name": "BERT-Mini", "layers": 4, "heads": 4, "hidden_size": 256, "outputs": 20},
        "config": {
            "epochs": args.epochs,
            "learning_rate": 3e-5,
            "seed": 42,
            "max_chunk_tokens": 512,
            "chunking": "All tokens, nonoverlapping chunks; no prefix truncation. Equal paper weight, masked per-item loss.",
            "base_model": BASE,
            "revision": REVISION,
        },
        "dataset": {
            "agent_records": len(all_records),
            "included_papers": len(records),
            "excluded_identity_or_type": 2,
            "without_known_targets": len(all_records) - len(records) - 2,
            "splits": {
                s: {
                    "papers": len(rs),
                    "yes": sum(a["answer"] == "yes" for r in rs for a in r["research"]["answers"]),
                    "no": sum(a["answer"] == "no" for r in rs for a in r["research"]["answers"]),
                }
                for s, rs in splits.items()
            },
            "questions": qcounts,
            "supported_heads": [i for i, m in zip(IDS, supervised, strict=True) if m],
        },
        "runs": [],
        "promotion": {
            "eligible": False,
            "reason": "Training labels contain no negative outcomes. Fit is diagnostic, not a validated success predictor; served citation baseline is unchanged.",
        },
        "limitations": [
            "All non-challenge known targets are positive; constant-positive predictions are a decisive control.",
            "Unknown and not-applicable items are masked, never converted to no.",
            "Challenge papers never enter fitting, checkpoint selection or calibration.",
            "Mixed full text and abstract inputs; source versions and topic redaction are not yet validated.",
            "No calibrated success percentage or validated topic-independent performance can be inferred from this sample.",
            "Unsupported question heads are excluded from metrics and reported as null.",
        ],
    }
    report["rl_readiness"] = rl_readiness(report["dataset"])
    write_json(OUT / "report.json", report)
    inputs = prepare(records, report)
    report["inputs"] = [
        {
            "id": r["id"],
            "split": r["split"],
            "scope": inputs[r["id"]]["input_scope"],
            "characters": len(inputs[r["id"]]["original_text"]),
            "redacted_fraction": inputs[r["id"]]["redacted_character_fraction"],
            "sha256": inputs[r["id"]]["source_sha256"],
        }
        for r in records
    ]
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report["device"] = device
    report["status"] = "training"
    started = time.perf_counter()
    for variant in ["wording", "original", "topic"]:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE, revision=REVISION, num_labels=20, local_files_only=True, attn_implementation="eager"
        ).to(device)
        report["architecture"]["parameters"] = sum(p.numel() for p in model.parameters())
        encoded = {
            r["id"]: chunks(tokenizer, inputs[r["id"]][variant + "_text"] or "[SUBJECT]") for r in records
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
        run = {
            "id": variant,
            "history": [],
            "steps": [],
            "chunks": sum(len(v) for v in encoded.values()),
            "best_epoch": None,
            "metrics": {},
        }
        report["runs"].append(run)
        best = float("inf")
        checkpoint = PRIVATE / (variant + ".pt")
        for epoch in range(1, args.epochs + 1):
            model.train()
            rows = list(splits["train"])
            random.shuffle(rows)
            loss_sum = 0
            epoch_start = time.perf_counter()
            for r in rows:
                optimizer.zero_grad(set_to_none=True)
                y, m = targets(r)
                y = torch.tensor(y, dtype=torch.float32, device=device)
                m = torch.tensor(m, device=device)
                cs = encoded[r["id"]]
                paper_loss = 0.0
                for c in cs:
                    ids = torch.tensor([c], device=device)
                    loss = masked_loss(
                        model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits, y, m
                    ) / len(cs)
                    loss.backward()
                    paper_loss += loss.item()
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                optimizer.step()
                loss_sum += paper_loss
                run["steps"].append(
                    {"step": len(run["steps"]) + 1, "epoch": epoch, "loss": paper_loss, "gradient_norm": norm}
                )
            train = evaluate(model, encoded, splits["train"], supervised, device)
            val = evaluate(model, encoded, splits["validation"], supervised, device)
            row = {
                "epoch": epoch,
                "optimization_loss": loss_sum / len(rows),
                "train_loss": train["log_loss"],
                "validation_loss": val["log_loss"],
                "train_brier": train["brier"],
                "validation_brier": val["brier"],
                "mean_probability": val["mean_probability"],
                "seconds": time.perf_counter() - epoch_start,
            }
            run["history"].append(row)
            if val["log_loss"] is not None and val["log_loss"] < best:
                best = val["log_loss"]
                run["best_epoch"] = epoch
                torch.save(model.state_dict(), checkpoint)
            report["stage"] = f"{variant}: epoch {epoch}/{args.epochs}"
            write_json(OUT / "report.json", report)
            print(variant, json.dumps(row), flush=True)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        run["metrics"] = {s: evaluate(model, encoded, rs, supervised, device) for s, rs in splits.items()}
        run["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        run["constant_positive_baseline"] = {s: positive_baseline(v) for s, v in run["metrics"].items()}
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        write_json(OUT / "report.json", report)
    report["status"] = "complete"
    report["stage"] = "Diagnostic fit complete"
    report["seconds"] = time.perf_counter() - started
    report["finished_at"] = datetime.now(UTC).isoformat()
    write_json(OUT / "report.json", report)
    print("COMPLETE", OUT / "report.json", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        path = OUT / "report.json"
        if path.exists():
            report = json.loads(path.read_text())
            report.update(status="error", stage=type(exc).__name__)
            write_json(path, report)
        raise
