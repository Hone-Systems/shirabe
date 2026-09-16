"""Fine-tune BERT-Mini with chronological selection/calibration and export CPU ONNX."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import hashlib
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import model_info
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train import metrics

BASE = "google/bert_uncased_L-4_H-256_A-4"
SEED = 42
EPOCHS = 4
LR = 3e-5
MAX_LENGTH = 512
BATCH = 32
OUT = ROOT / "artifacts/transformer"
PRIVATE = ROOT / "artifacts/transformer-training"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)


def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def predict(model, encoded, mask):
    ds = TensorDataset(*(encoded[k][mask] for k in ["input_ids", "attention_mask", "token_type_ids"]))
    output = []
    model.eval()
    with torch.inference_mode():
        for ids, attention, types in DataLoader(ds, batch_size=64):
            logits = model(
                input_ids=ids.to(DEVICE), attention_mask=attention.to(DEVICE), token_type_ids=types.to(DEVICE)
            ).logits
            output.extend((logits[:, 1] - logits[:, 0]).float().cpu().numpy().tolist())
    return np.asarray(output)


def train(encoded, y, train_mask, val_mask, revision, name):
    seed()
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, revision=revision, num_labels=2, attn_implementation="eager"
    ).to(DEVICE)
    checkpoint = PRIVATE / f"{name}.pt"
    history_path = PRIVATE / f"{name}.json"
    config = {
        "base": BASE,
        "revision": revision,
        "seed": SEED,
        "epochs": EPOCHS,
        "lr": LR,
        "batch": BATCH,
        "max_length": MAX_LENGTH,
        "data_sha256": hashlib.sha256((ROOT / "data/papers.jsonl").read_bytes()).hexdigest(),
    }
    if "--resume" in sys.argv and history_path.exists() and checkpoint.exists():
        saved = json.loads(history_path.read_text())
        if saved["config"] != config:
            raise ValueError(f"Checkpoint configuration changed: {name}")
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
        model.eval()
        print("RESUMED", name, flush=True)
        return model, saved["history"], saved["best_epoch"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    ds = TensorDataset(
        *(encoded[k][train_mask] for k in ["input_ids", "attention_mask", "token_type_ids"]),
        torch.tensor(y[train_mask], dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    history = []
    best = float("inf")
    best_epoch = 0
    checkpoint = PRIVATE / f"{name}.pt"
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        started = time.perf_counter()
        for ids, attention, types, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                loss = model(
                    input_ids=ids.to(DEVICE),
                    attention_mask=attention.to(DEVICE),
                    token_type_ids=types.to(DEVICE),
                    labels=labels.to(DEVICE),
                ).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(ids)
            seen += len(ids)
        val_logits = predict(model, encoded, val_mask)
        val_p = 1 / (1 + np.exp(-val_logits))
        vl = log_loss(y[val_mask], val_p)
        row = {
            "epoch": epoch,
            "n": int(train_mask.sum()),
            "train_loss": loss_sum / seen,
            "validation_loss": float(vl),
            "validation_auc": float(roc_auc_score(y[val_mask], val_p)),
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(name, json.dumps(row), flush=True)
        if vl < best:
            best = vl
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
    history_path.write_text(
        json.dumps({"config": config, "history": history, "best_epoch": best_epoch}, indent=2)
    )
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()
    return model, history, best_epoch


class ExportHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = self.model.classifier(out.pooler_output)
        return (
            logits[:, 1] - logits[:, 0],
            out.pooler_output,
            torch.stack([a[:, :, 0, :] for a in out.attentions], dim=1),
            torch.stack([h[:, 0, :] for h in out.hidden_states], dim=1),
        )


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((ROOT / "artifacts/report.json").read_text())
    papers = [json.loads(line) for line in (ROOT / "data/papers.jsonl").read_text().splitlines()]
    assert hashlib.sha256((ROOT / "data/papers.jsonl").read_bytes()).hexdigest() == baseline["dataset_sha256"]
    d = np.load(ROOT / "data/features.npz", allow_pickle=False)
    y = d["y"]
    years = d["years"]
    fields = d["fields"]
    texts = [p["abstract"] for p in papers]
    tr, va, ca, te = years <= 2018, years == 2019, years == 2020, years == 2021
    revision = model_info(BASE).sha
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=revision, use_fast=True)
    encoded = tokenizer(
        texts, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt"
    )
    print(
        json.dumps(
            {
                "device": DEVICE,
                "gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else None,
                "revision": revision,
                "papers": len(papers),
            }
        ),
        flush=True,
    )
    model, history, best_epoch = train(encoded, y, tr, va, revision, "main")
    cal_logits = predict(model, encoded, ca)
    calibrator = LogisticRegression(C=1e6, max_iter=2000).fit(cal_logits[:, None], y[ca])
    slope = float(calibrator.coef_[0, 0])
    offset = float(calibrator.intercept_[0])
    test_logits = predict(model, encoded, te)
    probs = calibrator.predict_proba(test_logits[:, None])[:, 1]
    test = metrics(y[te], probs, True)
    print("MAIN TEST", json.dumps(test), flush=True)
    wrapper = ExportHead(model).eval().cpu()
    dummy = {k: v[:1, :128] for k, v in encoded.items()}
    torch.onnx.export(
        wrapper,
        tuple(dummy[k] for k in ["input_ids", "attention_mask", "token_type_ids"]),
        str(OUT / "model.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["margin", "pooled", "cls_attention", "layer_cls"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "margin": {0: "batch"},
            "pooled": {0: "batch"},
            "cls_attention": {0: "batch", 3: "sequence"},
            "layer_cls": {0: "batch"},
        },
        opset_version=17,
    )
    tokenizer.save_pretrained(OUT)
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(
        str(OUT / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    with torch.inference_mode():
        pt = wrapper(*(dummy[k] for k in ["input_ids", "attention_mask", "token_type_ids"]))
    onnx = sess.run(None, {k: v.numpy().astype(np.int64) for k, v in dummy.items()})
    maxdiff = float(np.max(abs(pt[0].numpy() - onnx[0])))
    assert maxdiff < 1e-4
    coeff = (model.classifier.weight[1] - model.classifier.weight[0]).detach().cpu().numpy() * slope
    intercept = float((model.classifier.bias[1] - model.classifier.bias[0]).detach().cpu()) * slope + offset
    assert np.allclose(onnx[1] @ coeff + intercept, onnx[0] * slope + offset, atol=1e-5)
    manifest = {
        "model_id": "bert-mini-" + hashlib.sha256((OUT / "model.onnx").read_bytes()).hexdigest()[:10],
        "architecture": "BERT-Mini",
        "base_model": BASE,
        "base_revision": revision,
        "layers": 4,
        "heads": 4,
        "hidden_size": 256,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "max_length": MAX_LENGTH,
        "calibration": {"slope": slope, "offset": offset},
        "coef": coeff.tolist(),
        "intercept": intercept,
        "train_prevalence": float(y[tr].mean()),
        "model_sha256": hashlib.sha256((OUT / "model.onnx").read_bytes()).hexdigest(),
        "dataset_sha256": baseline["dataset_sha256"],
        "best_epoch": best_epoch,
        "training_text_hashes": [
            hashlib.sha256(" ".join(texts[i].lower().split()).encode()).hexdigest() for i in np.where(tr)[0]
        ],
        "onnx_parity_max_abs_error": maxdiff,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Retain full test records for independent evaluation.
    (OUT / "test_predictions.jsonl").write_text(
        "".join(
            json.dumps(
                {"id": papers[i]["id"], "label": int(y[i]), "probability": float(p), "margin": float(logit)}
            )
            + "\n"
            for i, p, logit in zip(np.where(te)[0], probs, test_logits, strict=True)
        )
    )
    field_results = []
    for field in sorted(set(fields)):
        sub = fields[te] == field
        field_results.append(
            {
                "field_id": int(field),
                "field": next(p["field"] for p in papers if p["field_id"] == field),
                **metrics(y[te][sub], probs[sub], True),
            }
        )
    # Shuffled-token diagnostic: preserve token counts; alter order. Test only, not used for selection.
    rng = np.random.default_rng(SEED)
    shuffled = {k: v.clone() for k, v in encoded.items()}
    for i in np.where(te)[0]:
        length = int(shuffled["attention_mask"][i].sum())
        permutation = rng.permutation(length - 2) + 1
        shuffled["input_ids"][i, 1 : length - 1] = encoded["input_ids"][i, permutation]
    model.to(DEVICE)
    pooled_parts = []
    with torch.inference_mode():
        for start in range(0, len(y), 64):
            args = {k: v[start : start + 64].to(DEVICE) for k, v in encoded.items()}
            pooled_parts.append(model.bert(**args).pooler_output.cpu().numpy())
    pooled_all = np.vstack(pooled_parts)
    scaler = StandardScaler().fit(pooled_all[tr])
    probe = LogisticRegression(C=0.1, max_iter=2000).fit(scaler.transform(pooled_all[tr]), fields[tr])
    discipline_probe = {
        "accuracy": float(probe.score(scaler.transform(pooled_all[te]), fields[te])),
        "majority_baseline": float(max(np.unique(fields[te], return_counts=True)[1]) / te.sum()),
        "description": "Separate field classifier on learned pooled activations; above-chance performance shows topic information remains.",
    }
    manifest["activation_mean"] = pooled_all[tr].mean(axis=0).tolist()
    manifest["activation_scale"] = pooled_all[tr].std(axis=0).clip(min=1e-8).tolist()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shuffled_logits = predict(model, shuffled, te)
    shuffled_metrics = metrics(y[te], calibrator.predict_proba(shuffled_logits[:, None])[:, 1])
    del shuffled, model, wrapper
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    transfers = []
    for field in sorted(set(fields)):
        model, _, epoch = train(
            encoded, y, tr & (fields != field), va & (fields != field), revision, f"without-{field}"
        )
        cl = predict(model, encoded, ca & (fields != field))
        cal = LogisticRegression(C=1e6, max_iter=2000).fit(cl[:, None], y[ca & (fields != field)])
        tl = predict(model, encoded, te & (fields == field))
        pr = cal.predict_proba(tl[:, None])[:, 1]
        transfers.append(
            {
                "field_id": int(field),
                "field": next(p["field"] for p in papers if p["field_id"] == field),
                "best_epoch": epoch,
                **metrics(y[te & (fields == field)], pr, True),
            }
        )
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("TRANSFER", json.dumps(transfers[-1]), flush=True)
    fpr, tpr, _ = roc_curve(y[te], probs)
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int))
    reliability = [
        {"predicted": float(probs[ix].mean()), "observed": float(y[te][ix].mean()), "n": len(ix)}
        for ix in np.array_split(np.argsort(probs), 10)
    ]
    report = {
        **baseline,
        "model_id": manifest["model_id"],
        "architecture": "4-layer BERT-Mini → pooled CLS → learned classification head → held-out sigmoid calibration",
        "model_kind": "transformer",
        "transformer": {k: v for k, v in manifest.items() if k not in ["coef", "training_text_hashes"]},
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_count": 256,
        "test": test,
        "uncalibrated_test": metrics(y[te], 1 / (1 + np.exp(-test_logits))),
        "comparisons": [{"name": "BERT-Mini · fine-tuned", "role": "deployed", **test}]
        + [
            {**row, "role": "previous linear baseline" if row["role"] == "deployed" else row["role"]}
            for row in baseline["comparisons"]
        ],
        "field_results": field_results,
        "field_transfer": transfers,
        "roc": [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in keep],
        "reliability": reliability,
        "epoch_history": history,
        "best_epoch": best_epoch,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LR,
            "weight_decay": 0.01,
            "batch_size": BATCH,
            "epochs": EPOCHS,
        },
        "features": [
            {
                "name": f"latent_{i:03d}",
                "label": f"Latent {i + 1:03d}",
                "coefficient": float(c),
                "mean": float(manifest["activation_mean"][i]),
                "scale": float(manifest["activation_scale"][i]),
            }
            for i, c in enumerate(coeff)
        ],
        "compute": {
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else None,
            "cloud_cost_usd": 0,
            "duration_seconds": time.perf_counter() - started,
            "python": sys.version.split()[0],
            "sklearn": baseline["compute"]["sklearn"],
            "torch": torch.__version__,
        },
        "shuffled_token_test": shuffled_metrics,
        "limitations": baseline["limitations"]
        + [
            "This transformer sees topic words as well as style. Cross-field transfer does not establish topic independence.",
            "Pretraining on external text may contain historical paper-related information; publication-time knowledge separation is not guaranteed.",
            "Inputs longer than 512 WordPieces are truncated. Attention weights are not causal token importance.",
        ],
        "baseline_model_id": baseline["model_id"],
    }
    # Baseline-only diagnostics must not be presented as transformer measurements.
    report["baseline_discipline_probe"] = report.pop("discipline_probe")
    report["discipline_probe"] = discipline_probe
    report["baseline_negative_control"] = report.pop("negative_control")
    report.pop("learning_curve")
    report.pop("convergence")
    report.pop("selected_c")
    report.pop("candidates")
    report.pop("optimizer_iterations")
    report["limitations"] = [v for v in report["limitations"] if not v.startswith("Topic words are excluded")]
    report["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT / "scripts/train_transformer.py"]
    }
    report["baseline_feature_matrix_sha256"] = report.pop("feature_matrix_sha256")
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("COMPLETE", manifest["model_id"], json.dumps(test), flush=True)


if __name__ == "__main__":
    main()
