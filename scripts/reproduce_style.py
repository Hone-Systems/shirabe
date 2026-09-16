"""Offline verification of the shipped style model from the public numeric matrix."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train import fit_style, metrics


def main():
    report = json.loads((ROOT / "artifacts/report.json").read_text())
    model = json.loads((ROOT / "artifacts/model.json").read_text())
    path = ROOT / "data/features.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["feature_matrix_sha256"]
    for source, digest in report["source_sha256"].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest, f"Source changed: {source}"
    with np.load(path, allow_pickle=False) as data:
        x, y, years = data["x"], data["y"], data["years"]
    fit = fit_style(x, y, years <= 2018, years == 2019, years == 2020)
    assert fit["c"] == report["selected_c"]
    assert np.allclose(fit["coef"], model["coef"], atol=1e-10)
    assert np.isclose(fit["intercept"], model["intercept"], atol=1e-10)
    test = metrics(y[years == 2021], fit["p"][years == 2021])
    assert abs(test["roc_auc"] - report["test"]["roc_auc"]) < 1e-12
    assert abs(test["brier"] - report["test"]["brier"]) < 1e-10
    print(json.dumps({"verified": True, "model_id": model["model_id"], "test": test}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
