"""Train on historical abstracts and export an inspectable, pickle-free model."""

import hashlib
import json
import os
import platform
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from shirabe.features import FEATURE_NAMES, LEXICONS, STRUCTURAL, WORD_RE, feature_label, matrix

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = 42


def metrics(y, p, bootstrap=False):
    result = {
        "n": len(y),
        "positives": int(sum(y)),
        "prevalence": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
    }
    if bootstrap:
        rng = np.random.default_rng(SEED)
        aucs = []
        for _ in range(500):
            ix = rng.integers(0, len(y), len(y))
            if len(np.unique(y[ix])) == 2:
                aucs.append(roc_auc_score(y[ix], p[ix]))
        result["auc_ci95"] = np.quantile(aucs, [0.025, 0.975]).tolist()
    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -700, 700)))


def fit_style(x, y, train, validation, calibration, columns=None):
    x = x if columns is None else x[:, columns]
    scaler = StandardScaler().fit(x[train])
    z = np.clip(scaler.transform(x), -5, 5)
    candidates = []
    for c in [0.001, 0.01, 0.1, 1, 10]:
        model = LogisticRegression(C=c, max_iter=2000, random_state=SEED).fit(z[train], y[train])
        candidates.append((log_loss(y[validation], model.predict_proba(z[validation])[:, 1]), c, model))
    _, c, model = min(candidates, key=lambda item: item[0])
    calibrator = LogisticRegression(C=1e6, max_iter=2000).fit(
        model.decision_function(z[calibration]).reshape(-1, 1), y[calibration]
    )
    slope, offset = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
    coef = model.coef_[0] * slope
    intercept = float(model.intercept_[0] * slope + offset)
    probabilities = sigmoid(z @ coef + intercept)
    return {
        "scaler": scaler,
        "model": model,
        "z": z,
        "coef": coef,
        "intercept": intercept,
        "p": probabilities,
        "c": c,
        "slope": slope,
        "offset": offset,
        "candidates": [{"c": val, "validation_log_loss": float(loss)} for loss, val, _ in candidates],
    }


def main():
    start = time.perf_counter()
    data_path = ROOT / "data/papers.jsonl"
    papers = [json.loads(line) for line in data_path.read_text().splitlines()]
    groups = defaultdict(list)
    for i, paper in enumerate(papers):
        groups[(paper["field_id"], paper["year"])].append(i)
    y = np.zeros(len(papers), dtype=int)
    cohorts = []
    for (field, year), indices in sorted(groups.items()):
        values = np.array([papers[i]["citations"] for i in indices])
        threshold = float(np.quantile(values, 0.75, method="higher"))
        y[indices] = values >= threshold
        cohorts.append(
            {
                "field_id": field,
                "field": papers[indices[0]]["field"],
                "year": year,
                "n": len(indices),
                "threshold": threshold,
                "positives": int(sum(y[indices])),
                "prevalence": float(np.mean(y[indices])),
            }
        )
    years = np.array([p["year"] for p in papers])
    fields = np.array([p["field_id"] for p in papers])
    train, validation, calibration, test = years <= 2018, years == 2019, years == 2020, years == 2021
    texts = [p["abstract"] for p in papers]
    print(f"Extracting features from {len(papers)} papers", flush=True)
    assert all(80 <= len(WORD_RE.findall(text)) <= 800 for text in texts), "Corpus and inference scope differ"
    x = matrix(texts)
    np.savez_compressed(ROOT / "data/features.npz", x=x, y=y, years=years, fields=fields)
    fitted = fit_style(x, y, train, validation, calibration)
    p = fitted["p"]
    print("Style model:", metrics(y[test], p[test]), flush=True)
    comparisons = [
        {"name": "Style · calibrated linear", "role": "deployed", **metrics(y[test], p[test], True)},
        {
            "name": "Training prevalence",
            "role": "prior baseline",
            **metrics(y[test], np.full(sum(test), np.mean(y[train]))),
        },
    ]
    length = fit_style(x, y, train, validation, calibration, [0, 1, 2])
    comparisons.append(
        {"name": "Length only", "role": "confound baseline", **metrics(y[test], length["p"][test])}
    )
    rhetorical = fit_style(
        x, y, train, validation, calibration, list(range(len(STRUCTURAL), len(STRUCTURAL) + len(LEXICONS)))
    )
    comparisons.append(
        {"name": "Rhetoric only", "role": "ablation", **metrics(y[test], rhetorical["p"][test])}
    )
    vectorizer = TfidfVectorizer(
        max_features=15000, min_df=3, max_df=0.95, stop_words="english", sublinear_tf=True
    )
    lexical_train = vectorizer.fit_transform(np.array(texts)[train])
    lexical_val = vectorizer.transform(np.array(texts)[validation])
    choices = []
    for c in [0.1, 1, 10]:
        m = LogisticRegression(C=c, max_iter=1000).fit(lexical_train, y[train])
        choices.append((log_loss(y[validation], m.predict_proba(lexical_val)[:, 1]), m))
    lexical = min(choices, key=lambda item: item[0])[1]
    lc = LogisticRegression(C=1e6).fit(
        lexical.decision_function(vectorizer.transform(np.array(texts)[calibration])).reshape(-1, 1),
        y[calibration],
    )
    lp = lc.predict_proba(
        lexical.decision_function(vectorizer.transform(np.array(texts)[test])).reshape(-1, 1)
    )[:, 1]
    comparisons.append(
        {"name": "Topic words · TF–IDF", "role": "content baseline, not deployed", **metrics(y[test], lp)}
    )
    nonlinear = HistGradientBoostingClassifier(
        max_iter=150,
        max_leaf_nodes=7,
        l2_regularization=10,
        learning_rate=0.05,
        early_stopping=False,
        random_state=SEED,
    ).fit(x[train], y[train])
    nc = LogisticRegression(C=1e6).fit(
        nonlinear.decision_function(x[calibration]).reshape(-1, 1), y[calibration]
    )
    comparisons.append(
        {
            "name": "Style · boosted trees",
            "role": "nonlinear comparison",
            **metrics(y[test], nc.predict_proba(nonlinear.decision_function(x[test]).reshape(-1, 1))[:, 1]),
        }
    )
    meta = np.array([[str(paper["field_id"]), paper.get("source_id") or "missing"] for paper in papers])
    encoder = OneHotEncoder(handle_unknown="ignore")
    mx = encoder.fit_transform(meta[train])
    mm = LogisticRegression(C=1, max_iter=1000).fit(mx, y[train])
    comparisons.append(
        {
            "name": "Journal + field",
            "role": "metadata confound, not deployed",
            **metrics(y[test], mm.predict_proba(encoder.transform(meta[test]))[:, 1]),
        }
    )
    print("Baselines complete", flush=True)
    transfers, field_results = [], []
    for field in sorted(set(fields)):
        name = next(paper["field"] for paper in papers if paper["field_id"] == field)
        mask = test & (fields == field)
        field_results.append({"field": name, "field_id": int(field), **metrics(y[mask], p[mask], True)})
        transfer = fit_style(
            x, y, train & (fields != field), validation & (fields != field), calibration & (fields != field)
        )
        transfers.append(
            {
                "field": name,
                "field_id": int(field),
                "selected_c": transfer["c"],
                **metrics(y[mask], transfer["p"][mask], True),
            }
        )
        print(f"Leave-field-out {name}: {transfers[-1]['roc_auc']:.3f}", flush=True)
    # Measure residual discipline information, without using discipline to predict impact.
    discipline = LogisticRegression(C=0.1, max_iter=2000).fit(fitted["z"][train], fields[train])
    discipline_accuracy = float(discipline.score(fitted["z"][test], fields[test]))
    rng = np.random.default_rng(SEED)
    negative = []
    for _ in range(20):
        shuffled = y[train].copy()
        rng.shuffle(shuffled)
        nm = LogisticRegression(C=fitted["c"], max_iter=1000).fit(fitted["z"][train], shuffled)
        negative.append(float(roc_auc_score(y[test], nm.predict_proba(fitted["z"][test])[:, 1])))
    learning = []
    indices = np.where(train)[0]
    rng.shuffle(indices)
    for fraction in [0.1, 0.25, 0.5, 0.75, 1.0]:
        chosen = indices[: int(len(indices) * fraction)]
        sc = StandardScaler().fit(x[chosen])
        zt, zv = np.clip(sc.transform(x[chosen]), -5, 5), np.clip(sc.transform(x[validation]), -5, 5)
        m = LogisticRegression(C=fitted["c"], max_iter=2000).fit(zt, y[chosen])
        learning.append(
            {
                "n": len(chosen),
                "train_auc": float(roc_auc_score(y[chosen], m.predict_proba(zt)[:, 1])),
                "validation_auc": float(roc_auc_score(y[validation], m.predict_proba(zv)[:, 1])),
            }
        )
    convergence = []
    for iterations in [1, 2, 4, 8, 16, 32, 64, 128]:
        m = LogisticRegression(C=fitted["c"], max_iter=iterations).fit(fitted["z"][train], y[train])
        convergence.append(
            {
                "iteration_budget": iterations,
                "iterations_used": int(m.n_iter_[0]),
                "train_loss": float(log_loss(y[train], m.predict_proba(fitted["z"][train])[:, 1])),
                "validation_loss": float(
                    log_loss(y[validation], m.predict_proba(fitted["z"][validation])[:, 1])
                ),
            }
        )
    fpr, tpr, _ = roc_curve(y[test], p[test])
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int))
    reliability = []
    for indices in np.array_split(np.argsort(p[test]), 10):
        outcomes = y[test][indices]
        reliability.append(
            {
                "predicted": float(np.mean(p[test][indices])),
                "observed": float(np.mean(outcomes)),
                "n": len(indices),
            }
        )
    scaler = fitted["scaler"]
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT / "shirabe/features.py", ROOT / "scripts/train.py"]
    }
    model_id = (
        "style-"
        + hashlib.sha256(
            (
                hashlib.sha256(data_path.read_bytes()).hexdigest() + json.dumps(source_hashes, sort_keys=True)
            ).encode()
        ).hexdigest()[:10]
    )
    artifact = {
        "schema_version": 1,
        "model_id": model_id,
        "feature_names": FEATURE_NAMES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": fitted["coef"].tolist(),
        "intercept": fitted["intercept"],
        "clip": 5,
        "train_q01": np.quantile(x[train], 0.01, axis=0).tolist(),
        "train_q99": np.quantile(x[train], 0.99, axis=0).tolist(),
        "train_prevalence": float(np.mean(y[train])),
        "training_text_hashes": [
            hashlib.sha256(" ".join(texts[i].lower().split()).encode()).hexdigest()
            for i in np.where(train)[0]
        ],
        "calibration": {"slope": fitted["slope"], "offset": fitted["offset"]},
    }
    feature_report = [
        {
            "name": name,
            "label": feature_label(name),
            "coefficient": float(fitted["coef"][i]),
            "mean": float(scaler.mean_[i]),
            "scale": float(scaler.scale_[i]),
        }
        for i, name in enumerate(FEATURE_NAMES)
    ]
    report = {
        "model_id": model_id,
        "trained_at": datetime.now(UTC).isoformat(),
        "status": "completed",
        "architecture": "Fixed style features → train-only standardization → clipped linear logit → held-out sigmoid calibration",
        "feature_count": len(FEATURE_NAMES),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "feature_matrix_sha256": hashlib.sha256((ROOT / "data/features.npz").read_bytes()).hexdigest(),
        "seed": SEED,
        "target": "At or above the sampled field/year cohort's 75th citation percentile at collection",
        "splits": [
            {"name": name, "years": year, "n": int(sum(mask)), "positive_rate": float(np.mean(y[mask]))}
            for name, year, mask in [
                ("Train", "2016–2018", train),
                ("Validation", "2019", validation),
                ("Calibration", "2020", calibration),
                ("Test", "2021", test),
            ]
        ],
        "test": metrics(y[test], p[test], True),
        "uncalibrated_test": metrics(y[test], fitted["model"].predict_proba(fitted["z"][test])[:, 1]),
        "comparisons": comparisons,
        "cohorts": cohorts,
        "field_results": field_results,
        "field_transfer": transfers,
        "discipline_probe": {
            "accuracy": discipline_accuracy,
            "majority_baseline": float(max(Counter(fields[test]).values()) / sum(test)),
            "description": "Train a separate classifier to infer field from the same style features. Above-chance accuracy demonstrates remaining topic/discipline signal.",
        },
        "negative_control": {
            "runs": 20,
            "mean_auc": float(np.mean(negative)),
            "min_auc": min(negative),
            "max_auc": max(negative),
            "aucs": negative,
        },
        "learning_curve": learning,
        "convergence": convergence,
        "roc": [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in keep],
        "reliability": reliability,
        "features": feature_report,
        "selected_c": fitted["c"],
        "candidates": fitted["candidates"],
        "optimizer_iterations": int(fitted["model"].n_iter_[0]),
        "compute": {
            "device": "CPU",
            "cloud_cost_usd": 0,
            "duration_seconds": time.perf_counter() - start,
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        },
        "limitations": [
            "Citations measure attention, not truth, replication, practical success, or researcher quality.",
            "Retrospective citation snapshot; unequal horizons and changing field composition remain confounds. This is not a historical as-of backtest.",
            "Only English abstracts of indexed articles, 80–800 words, in six fields; full-text writing and other languages are unvalidated.",
            "Topic words are excluded, but style can still identify discipline. Cross-field transfer tests do not prove subject invariance.",
            "The labels are relative to retained sampled cohorts, not all papers in each field. Ties can raise the positive rate above 25%.",
            "Journal prestige, author networks, access, and scientific content can explain associations. Rewriting is not a causal intervention.",
            "Bootstrap intervals describe sampling variability for this test sample, not certainty about a single paper's future.",
        ],
    }
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    (out / "model.json").write_text(json.dumps(artifact, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    predictions = [
        {
            "id": paper["id"],
            "doi": paper["doi"],
            "year": paper["year"],
            "field_id": paper["field_id"],
            "citations": paper["citations"],
            "label": int(y[i]),
            "split": "train"
            if train[i]
            else "validation"
            if validation[i]
            else "calibration"
            if calibration[i]
            else "test",
            "probability": float(p[i]),
        }
        for i, paper in enumerate(papers)
    ]
    (out / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
    # Test exact JSON inference parity before declaring export complete.
    exported = sigmoid(
        np.clip((x - np.array(artifact["mean"])) / np.array(artifact["scale"]), -5, 5)
        @ np.array(artifact["coef"])
        + artifact["intercept"]
    )
    assert np.allclose(exported, p, atol=1e-12)
    print(
        json.dumps(
            {"model_id": model_id, "test": report["test"], "duration": report["compute"]["duration_seconds"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
