import json

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from shirabe.transformer import TransformerModel


@pytest.fixture(scope="module")
def transformer():
    return TransformerModel()


def test_head_arithmetic_and_attention(transformer, abstract):
    result = transformer.predict(abstract)
    assert len(result["features"]) == 256
    assert (
        abs(result["intercept"] + sum(f["contribution"] for f in result["features"]) - result["logit"]) < 1e-5
    )
    attention = np.asarray(result["transformer"]["cls_attention"])
    assert attention.shape == (4, 4, result["transformer"]["sequence_length"])
    assert np.allclose(attention.sum(axis=-1), 1, atol=1e-5)
    assert np.asarray(result["transformer"]["layer_cls"]).shape == (5, 256)
    assert all(t["end"] > t["start"] and 0 <= t["attention"] <= 1 for t in result["tokens"])
    assert result["probability"] == pytest.approx(transformer.predict(abstract)["probability"], abs=1e-7)


def test_truncation_and_dynamic_sequence(transformer, abstract):
    short = transformer.predict(abstract)
    long = transformer.predict(" ".join([abstract] * 4))
    assert short["transformer"]["sequence_length"] < 512
    assert long["transformer"]["sequence_length"] == 512
    assert long["transformer"]["full_sequence_length"] > 512
    assert any("WordPieces" in w for w in long["warnings"])
    assert abs(long["intercept"] + sum(f["contribution"] for f in long["features"]) - long["logit"]) < 1e-5


def test_report_metrics_and_identity(transformer):
    report = json.loads((transformer.path / "report.json").read_text())
    records = [
        json.loads(line) for line in (transformer.path / "test_predictions.jsonl").read_text().splitlines()
    ]
    labels = [v["label"] for v in records]
    probs = [v["probability"] for v in records]
    assert report["model_id"] == transformer.artifact["model_id"]
    assert report["test"]["roc_auc"] == pytest.approx(roc_auc_score(labels, probs))
    assert report["test"]["brier"] == pytest.approx(brier_score_loss(labels, probs))
    assert report["best_epoch"] == min(report["epoch_history"], key=lambda r: r["validation_loss"])["epoch"]
