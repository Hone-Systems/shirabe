import hashlib
import json

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from shirabe.features import FEATURE_NAMES, extract, tokenize
from shirabe.inference import ROOT, StyleModel


def test_exact_contributions_and_export_parity(abstract):
    model = StyleModel()
    prediction = model.predict(abstract)
    assert prediction["logit"] == pytest.approx(
        prediction["intercept"] + sum(f["contribution"] for f in prediction["features"]), abs=1e-12
    )
    assert prediction["probability"] == pytest.approx(1 / (1 + np.exp(-prediction["logit"])))
    assert 0 < prediction["probability"] < 1
    assert len(prediction["features"]) == len(FEATURE_NAMES)
    assert not prediction["seen_in_training"]


def test_no_content_identity_in_feature_vector():
    # Same structure and shape, different subject vocabulary: feature vectors must be identical.
    assert (
        np.array_equal(
            extract("We tested planets and found signals."), extract("We tested proteins and found neurons.")
        )
        is False
    )  # Word shape differs.
    assert np.array_equal(
        extract("We tested planets and found signals."), extract("We tested enzymes and found tissues.")
    )


def test_token_offsets_preserve_input_and_categories():
    text = "We may offer a novel result; however, further evidence is needed."
    tokens = tokenize(text)
    assert all(text[t["start"] : t["end"]] == t["text"] for t in tokens)
    assert next(t for t in tokens if t["text"] == "novel")["category"] == "promotion"
    assert next(t for t in tokens if t["text"] == "may")["category"] == "hedging"


def test_report_recomputed_from_all_test_predictions():
    report = json.loads((ROOT / "artifacts/report.json").read_text())
    rows = [json.loads(line) for line in (ROOT / "artifacts/predictions.jsonl").read_text().splitlines()]
    test = [r for r in rows if r["split"] == "test"]
    assert len(test) == report["test"]["n"]
    assert all(r["year"] == 2021 for r in test)
    y, p = [r["label"] for r in test], [r["probability"] for r in test]
    assert roc_auc_score(y, p) == pytest.approx(report["test"]["roc_auc"])
    assert brier_score_loss(y, p) == pytest.approx(report["test"]["brier"])
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["doi"] for r in rows if r["doi"]}) == len([r for r in rows if r["doi"]])
    thresholds = {(c["field_id"], c["year"]): c["threshold"] for c in report["cohorts"]}
    for row in rows:
        assert row["label"] == (row["citations"] >= thresholds[(row["field_id"], row["year"])])
        assert row["split"] == (
            "train"
            if row["year"] <= 2018
            else {2019: "validation", 2020: "calibration", 2021: "test"}[row["year"]]
        )
    model = json.loads((ROOT / "artifacts/model.json").read_text())
    assert model["model_id"] == report["model_id"]
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    assert report["dataset_sha256"] == provenance["dataset_sha256"]


def test_exact_seen_training_warning(abstract):
    model = StyleModel()
    model.training_hashes.add(hashlib.sha256(" ".join(abstract.lower().split()).encode()).hexdigest())
    assert model.predict(abstract)["seen_in_training"]


@pytest.mark.parametrize("text", ["", "A short abstract.", "research " * 801])
def test_reject_out_of_scope_lengths(text):
    with pytest.raises(ValueError, match="80–800"):
        StyleModel().predict(text)
