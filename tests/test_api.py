import subprocess

import pymupdf

from shirabe.api import MAX_UPLOAD, abstract_candidate


def test_health_and_prediction(client, abstract):
    health = client.get("/api/health")
    assert health.status_code == 200
    response = client.post("/api/predict", json={"text": abstract})
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == health.json()["model_id"]
    assert data["input_scope"] == "abstract"
    assert 0 < data["probability"] < 1


def test_invalid_and_missing_model(client, abstract):
    assert client.post("/api/predict", json={"text": "too short"}).status_code == 422
    assert client.post("/api/predict", json={"text": "x" * 20001}).status_code == 422
    model = client.app.state.model
    client.app.state.model = None
    try:
        assert client.get("/api/health").status_code == 503
        assert client.post("/api/predict", json={"text": abstract}).status_code == 503
    finally:
        client.app.state.model = model


def test_non_english_rejected(client):
    text = (
        "Nous avons étudié les effets de cette méthode sur les résultats des participants. Les observations montrent une différence importante entre les groupes. Cependant, les conclusions restent limitées par la taille de notre échantillon. "
    ) * 4
    response = client.post("/api/predict", json={"text": text})
    assert response.status_code == 422
    assert "English" in response.json()["detail"]


def test_text_upload_is_review_only(client, abstract):
    text = (
        f"Test paper\nAuthors\nAbstract\n{abstract}\nKeywords: learning\n1. Introduction\nNot the abstract."
    )
    response = client.post("/api/extract", files={"file": ("paper.txt", text.encode(), "text/plain")})
    assert response.status_code == 200
    data = response.json()
    assert data["requires_review"]
    assert data["text"] == abstract
    assert "probability" not in data


def test_pdf_upload_extracts_abstract(client, abstract):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 50, 550, 750),
        "A synthetic test paper\nAbstract\n"
        + abstract
        + "\n1. Introduction\nBody text should not be scored.",
        fontsize=11,
    )
    pdf = doc.tobytes()
    doc.close()
    response = client.post("/api/extract", files={"file": ("paper.pdf", pdf, "application/pdf")})
    assert response.status_code == 200, response.text
    data = response.json()
    assert "Body text" not in data["text"]
    assert data["pages"] == 1
    assert data["requires_review"]
    assert client.post("/api/predict", json={"text": data["text"]}).status_code == 200


def test_scanned_encrypted_corrupt_and_wrong_files(client):
    doc = pymupdf.open()
    doc.new_page()
    pdf = doc.tobytes()
    response = client.post("/api/extract", files={"file": ("scan.pdf", pdf)})
    assert response.status_code == 422 and "OCR" in response.json()["detail"]
    protected = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="test")
    response = client.post("/api/extract", files={"file": ("protected.pdf", protected)})
    assert response.status_code == 422 and "password" in response.json()["detail"]
    doc.close()
    for name, content, status in [
        ("bad.pdf", b"%PDF-invalid", 422),
        ("fake.pdf", b"not a pdf", 422),
        ("app.exe", b"bad", 415),
        ("empty.txt", b"", 422),
        ("binary.txt", b"\xff\xff", 422),
    ]:
        assert client.post("/api/extract", files={"file": (name, content)}).status_code == status


def test_upload_size_limit(client):
    response = client.post("/api/extract", files={"file": ("large.txt", b"x" * (MAX_UPLOAD + 1))})
    assert response.status_code == 413
    response = client.post("/api/extract", files={"file": ("large.txt", b"x" * (MAX_UPLOAD + 100000))})
    assert response.status_code == 413


def test_pdf_timeout_recoverable(client, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("pdf-worker", 15)

    monkeypatch.setattr(subprocess, "run", timeout)
    response = client.post("/api/extract", files={"file": ("paper.pdf", b"%PDF-test")})
    assert response.status_code == 422
    assert "safely" in response.json()["detail"]


def test_unknown_routes_and_no_arbitrary_file_serving(client):
    assert client.get("/api/unknown").status_code == 404
    assert client.get("/data/papers.jsonl").status_code == 404
    assert client.get("/artifacts/model.json").status_code == 404


def test_long_text_never_silently_claimed_as_abstract():
    text, notice = abstract_candidate("Long body text. " * 1000)
    assert "Could not isolate" in notice
    assert len(text) <= 12000


def test_pdf_rotated_stamp_and_author_footnote_excluded(client, abstract):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 500), "Abstract\n" + abstract, fontsize=11)
    page.insert_text((20, 600), "arXiv:stamp", rotate=90)
    page.insert_text((50, 600), "* Equal contribution. Author notes are not part of the abstract.")
    data = doc.tobytes()
    doc.close()
    result = client.post("/api/extract", files={"file": ("paper.pdf", data)})
    assert result.status_code == 200
    assert "arXiv" not in result.json()["text"]
    assert "Author notes" not in result.json()["text"]
    assert "abstract" not in result.json()["text"].lower()
