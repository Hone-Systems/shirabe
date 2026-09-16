from copy import deepcopy

import pytest

from scripts import agent_grade
from shirabe.outcome_rubric import IDS, aggregate


def answers(state="unknown"):
    return [{"id": id_, "answer": state, "rationale": "Evidence assessment", "evidence": []} for id_ in IDS]


def test_missing_evidence_is_not_failure():
    result = aggregate(answers())
    assert result["score"] is None
    assert result["lower_bound"] == 0
    assert result["upper_bound"] == 1
    assert result["known_answers"] == 0


def test_missingness_bounds_and_dimension_weighting():
    rows = answers()
    for i, row in enumerate(rows):
        if i % 5 < 3:
            row.update(answer="yes" if i < 10 else "no", evidence=[{"source_id": "s1"}])
    result = aggregate(rows)
    assert result["score"] == 0.5
    assert result["coverage"] == 0.6
    assert result["lower_bound"] == pytest.approx(0.3)
    assert result["upper_bound"] == pytest.approx(0.7)


def test_duplicate_questions_and_unsupported_binary_rejected():
    rows = answers()
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match="Exactly one"):
        aggregate(rows)
    rows = answers("yes")
    with pytest.raises(ValueError, match="require evidence"):
        aggregate(rows)
    assert aggregate(answers("not_applicable"))["score"] is None


def grade():
    rows = answers()
    rows[0].update(
        answer="yes",
        evidence=[{"source_id": "s1", "date": "2020", "explanation": "Independent follow-up supports claim"}],
    )
    return {
        "id": "test-paper",
        "title": "Test",
        "doi": "https://doi.org/example",
        "year": 2017,
        "field_id": 17,
        "field": "Computer Science",
        "split": "challenge",
        "sampling_cohort": "agent_pilot",
        "as_of": "2026-09-16",
        "original_source": {"read": True, "scope": "full_text", "url": "https://example.org/original"},
        "sources": [
            {
                "source_id": "s1",
                "url": "https://example.org/followup",
                "date": "2020",
                "date_basis": "publication",
                "read": True,
            }
        ],
        "answers": rows,
        "search_log": [{"query": "paper independent replication"}],
        "search_gaps": ["Remaining questions unresolved"],
        "summary": "Limited evidence",
        "researcher": "test",
    }


def test_publisher_rejects_unread_and_mismatched_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_grade, "OUT", tmp_path)
    data = grade()
    for change, message in [(False, "actually read")]:
        bad = deepcopy(data)
        bad["sources"][0]["read"] = change
        with pytest.raises(ValueError, match=message):
            agent_grade.publish(bad)
    bad = deepcopy(data)
    bad["answers"][0]["evidence"][0]["date"] = "2019"
    with pytest.raises(ValueError, match="date must match"):
        agent_grade.publish(bad)
    assert not list(tmp_path.rglob("*.json"))
    record = agent_grade.publish(data)
    assert record["aggregate"]["score"] is None
    assert record["researcher_kind"] == "autonomous_agent"
    assert len(list((tmp_path / "labels").glob("*.json"))) == 1
    assert len(list((tmp_path / "progress").glob("*.json"))) == 1


def test_explorer_prefers_agent_review_and_keeps_comparison(tmp_path, monkeypatch):
    import json

    from shirabe import evidence

    folder = tmp_path / "data/outcomes"
    monkeypatch.setattr(agent_grade, "OUT", folder)
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    reviewed = agent_grade.publish(grade())
    old = deepcopy(reviewed)
    old.update(retrieval_version=3, researcher_kind="fixed_batch")
    old["aggregate"]["known_answers"] = 0
    (folder / "labels/previous.json").write_text(json.dumps(old))
    private = tmp_path / "artifacts/private"
    private.mkdir(parents=True)
    (private / "source.json").write_text('{"secret":"private-source-sentinel"}')
    response = evidence.evaluation_data()
    assert len(response["records"]) == 1
    assert response["summary"]["agent_reviewed"] == 1
    assert response["records"][0]["previous_review"]["known_answers"] == 0
    assert response["records"][0]["aggregate"]["known_answers"] == 1
    assert "private-source-sentinel" not in json.dumps(response)


def test_unresolved_original_can_publish_only_excluded_unknowns(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_grade, "OUT", tmp_path)
    data = grade()
    data["original_source"].update(read=False, scope="unavailable")
    data["identity_resolution"] = {"training_eligible": False, "explanation": "DOI resolves to another paper"}
    with pytest.raises(ValueError, match="all unknown"):
        agent_grade.publish(data)
    data["answers"] = answers()
    record = agent_grade.publish(data)
    assert record["review_status"] == "identity_unresolved"
    assert record["identity_resolution"]["training_eligible"] is False
    assert record["aggregate"]["known_answers"] == 0


def test_wording_keeps_appendices_and_rhetorical_claims():
    from scripts.extract_wording import original_body, redact

    text = "Title\nAbstract\nWe demonstrate a simple method using quantum foobar.\nReferences\nA citation.\nAppendix A\nMore experimental details."
    body = original_body(text)
    assert "Appendix A" in body and "More experimental details." in body
    masked, _, accepted = redact(
        body, ["We demonstrate a simple method", "quantum foobar", "More experimental details."]
    )
    assert "We demonstrate a simple method" in masked
    assert "quantum foobar" not in masked
    assert accepted == ["quantum foobar"]
