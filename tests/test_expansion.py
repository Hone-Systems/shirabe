"""Paper-level evaluation prevents prolific labels from inflating confidence."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from scripts.train_expansion import normalize_doi, paired_interval, prevalence, summarize  # noqa: E402
from shirabe.outcome_rubric import IDS  # noqa: E402


def paper(id_, states):
    return {
        "id": id_,
        "title": id_,
        "split": "test",
        "sampling_cohort": "synthetic",
        "research": {"answers": [{"id": q, "answer": states.get(q, "unknown")} for q in IDS]},
    }


def test_paper_weighting_not_correlated_answer_weighting():
    records = [paper("one", {IDS[0]: "no"}), paper("many", dict.fromkeys(IDS, "yes"))]
    result = summarize(records, np.ones((2, 20)), np.ones(20, dtype=bool))
    assert result["brier"] == pytest.approx(0.5, abs=1e-6)
    assert result["known_items"] == 21
    assert result["scorable_papers"] == 2
    assert result["auc_heads"] == 1
    assert result["per_head"][IDS[1]]["auc"] is None


def test_prevalence_uses_only_known_training_targets():
    records = [paper("one", {IDS[0]: "yes"}), paper("two", {IDS[0]: "no"})]
    prior, supported = prevalence(records)
    assert prior[0] == 0.5
    assert supported.sum() == 1
    result = summarize(records, np.tile(prior, (2, 1)), supported)
    assert result["brier"] == 0.25
    assert result["papers"][0]["probabilities"][1] is None


def test_paired_bootstrap_aligns_by_paper_id():
    model = {"papers": [{"id": "a", "brier": 0.1}, {"id": "b", "brier": 0.2}]}
    control = {"papers": [{"id": "b", "brier": 0.4}, {"id": "a", "brier": 0.3}]}
    result = paired_interval(model, control)
    assert result["papers"] == 2
    assert result["mean"] == pytest.approx(0.2)
    assert result["lower"] == pytest.approx(0.2)
    assert result["upper"] == pytest.approx(0.2)


def test_doi_normalization_keeps_versions_groupable():
    assert normalize_doi("HTTPS://DOI.ORG/10.1/ABC") == "10.1/abc"
    assert normalize_doi("doi:10.1/AbC") == "10.1/abc"
