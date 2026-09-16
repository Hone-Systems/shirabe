import asyncio
import json
import queue
import re
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from shirabe.evidence import evaluation_data
from shirabe.inference import ROOT, StyleModel
from shirabe.outcome_model import outcome_model
from shirabe.transformer import TransformerModel

MAX_UPLOAD = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app):
    transformer = ROOT / "artifacts/transformer"
    if (transformer / "report.json").exists():
        app.state.model = TransformerModel(transformer)
        app.state.report_path = transformer / "report.json"
    else:
        app.state.model = StyleModel() if (ROOT / "artifacts/model.json").exists() else None
        app.state.report_path = ROOT / "artifacts/report.json"
    yield


app = FastAPI(title="Shirabe", version="0.1.0", lifespan=lifespan)


class BodyLimit:
    """Bound request streams, including chunked multipart bodies, before parsing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        size = 0

        async def limited_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > MAX_UPLOAD + 65536:
                raise HTTPException(413, "Upload must be 10 MB or smaller.")
            return message

        await self.app(scope, limited_receive, send)


app.add_middleware(BodyLimit)


class PredictInput(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class OutcomeInput(BaseModel):
    text: str = Field(min_length=100, max_length=3_000_000)


@app.get("/api/outcome-model")
def outcome_metadata():
    try:
        return outcome_model.metadata()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from None


@app.get("/api/outcome-example")
def outcome_example(paper: str = "attention"):
    if paper != "attention":
        raise HTTPException(404, "Unknown example")
    import hashlib

    key = hashlib.sha256(b"challenge-attention").hexdigest()
    path = ROOT / "artifacts/private/outcome-training/inputs" / (key + ".json")
    if not path.exists():
        raise HTTPException(503, "The original example is not available yet")
    source = json.loads(path.read_text())
    return {
        "title": "Attention Is All You Need",
        "text": source["original_text"],
        "source_url": "https://arxiv.org/abs/1706.03762",
        "held_out": True,
    }


@app.post("/api/outcome-predict")
def outcome_predict(body: OutcomeInput):
    try:
        return outcome_model.predict(body.text.strip())
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from None
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from None


@app.post("/api/outcome-predict/stream")
async def outcome_predict_stream(body: OutcomeInput):
    events = queue.Queue()
    cancelled = threading.Event()

    def progress(event):
        if cancelled.is_set():
            raise RuntimeError("Analysis cancelled")
        events.put({"type": "progress", **event})

    def work():
        try:
            result = outcome_model.predict(body.text.strip(), progress=progress)
            events.put({"type": "result", "result": result})
        except Exception as exc:
            events.put({"type": "error", "detail": str(exc)})

    async def stream():
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        try:
            while True:
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue
                yield json.dumps(event, allow_nan=False) + "\n"
                if event["type"] in ("result", "error"):
                    break
        finally:
            cancelled.set()

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
def health():
    loaded = app.state.model is not None
    return JSONResponse(
        {
            "status": "ready" if loaded else "model_missing",
            "model_id": app.state.model.artifact["model_id"] if loaded else None,
        },
        status_code=200 if loaded else 503,
    )


@app.get("/api/report")
def report():
    path = app.state.report_path
    if not path.exists():
        raise HTTPException(503, "No training run exists yet. Run scripts/train.py first.")
    return json.loads(path.read_text())


@app.get("/api/outcome-training")
def outcome_training():
    path = ROOT / "artifacts/outcome-training/report.json"
    if not path.exists():
        return {"status": "not_started", "stage": "No outcome training run yet", "runs": []}
    report = json.loads(path.read_text())
    labels = [
        r
        for p in (ROOT / "data/outcomes/labels").glob("agent-*.json")
        if (r := json.loads(p.read_text())).get("split") != "external"
    ]
    if len(labels) > report.get("dataset", {}).get("agent_records", 0):
        eligible = [
            r
            for r in labels
            if r["split"] == "train"
            and (r.get("identity_resolution") or {}).get("training_eligible") is not False
            and r["aggregate"]["known_answers"] > 0
        ]
        report["pending_dataset"] = {
            "papers": len(eligible),
            "yes": sum(a["answer"] == "yes" for r in eligible for a in r["research"]["answers"]),
            "no": sum(a["answer"] == "no" for r in eligible for a in r["research"]["answers"]),
        }
    return report


@app.get("/api/evidence")
def evidence():
    return evaluation_data()


@app.get("/api/provenance")
def provenance():
    path = ROOT / "data/provenance.json"
    if not path.exists():
        raise HTTPException(503, "Dataset provenance is unavailable.")
    return json.loads(path.read_text())


@app.post("/api/predict")
def predict(body: PredictInput):
    if app.state.model is None:
        raise HTTPException(503, "Model is unavailable. Run scripts/train.py and restart the server.")
    try:
        return app.state.model.predict(body.text.strip())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def abstract_candidate(text):
    # Handles inline headings, numbered introductions and structured abstracts.
    text = text.replace("\x00", "")
    heading = re.search(r"(?:^|\n)\s*(?:abstract|summary)\s*[:.\-—]?\s*", text, re.I)
    if not heading:
        # Some PDF extractors merge title and Abstract onto a single line.
        heading = re.search(r"\bAbstract\s*[:.\-—]\s*", text)
    if heading:
        tail = text[heading.end() :]
        end = re.search(
            r"(?:\n\s*(?:\d+[.\s]+)?(?:introduction|keywords|key words|index terms|references)\b|\bKeywords\s*:|\n\s*[*∗†‡©]|\n\s*Copyright\b)",
            tail,
            re.I,
        )
        candidate = tail[: end.start()] if end else tail
        candidate = re.sub(r"(?<=\w)-\n(?=\w)", "", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if 60 <= len(candidate.split()) <= 850:
            return candidate, "Abstract heading detected. Review the extracted text before analysis."
    if len(text.split()) <= 850:
        return re.sub(
            r"\s+", " ", text
        ).strip(), "Review this text and remove any title, authors, or references before analysis."
    return text[
        :12000
    ], "Could not isolate the abstract reliably. Select just the 80–800 word abstract below before analysis."


def extract_pdf(content, full_text=False):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "shirabe.pdf_worker", *(["--full-text"] if full_text else [])],
            input=content,
            capture_output=True,
            timeout=15,
            cwd=ROOT,
            check=False,
        )
        payload = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, ValueError):
        raise HTTPException(
            422, "PDF extraction could not finish safely. Paste its abstract instead."
        ) from None
    if "error" in payload:
        raise HTTPException(422, payload["error"])
    return payload


@app.post("/api/extract")
async def extract_upload(file: Annotated[UploadFile, File()], full_text: bool = False):
    try:
        content = await file.read(MAX_UPLOAD + 1)
        if len(content) > MAX_UPLOAD:
            raise HTTPException(413, "Upload must be 10 MB or smaller.")
        suffix = Path(file.filename or "").suffix.lower()
        pages = None
        if suffix == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise HTTPException(422, "The file is not a valid PDF. Upload a PDF or a UTF-8 text file.")
            payload = await run_in_threadpool(extract_pdf, content, full_text)
            text, pages = payload["text"], payload["pages"]
        elif suffix == ".txt":
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                raise HTTPException(422, "Save this text file as UTF-8 and try again.") from None
        else:
            raise HTTPException(415, "Choose a PDF or UTF-8 .txt file.")
        if full_text:
            if len(text) > 3_000_000:
                raise HTTPException(
                    413,
                    "Extracted text exceeds 3 million characters. Split the document explicitly; no text was silently clipped.",
                )
            candidate, notice = (
                text,
                "Complete extracted text. Review document identity and remove publisher notices before analysis; all supplied body tokens will be processed.",
            )
        else:
            candidate, notice = abstract_candidate(text)
        if not candidate.strip():
            raise HTTPException(422, "The file contains no readable text.")
        return {
            "text": candidate,
            "notice": notice,
            "filename": Path(file.filename or "upload").name,
            "pages": pages,
            "requires_review": True,
        }
    finally:
        await file.close()


# One process serves the production UI and API. Development uses Vite's local proxy.
DIST = ROOT / "web/dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/favicon.svg", include_in_schema=False)
    def favicon():
        return FileResponse(DIST / "favicon.svg")

    @app.get("/", include_in_schema=False)
    @app.get("/training", include_in_schema=False)
    @app.get("/evidence", include_in_schema=False)
    def index():
        return FileResponse(DIST / "index.html")
