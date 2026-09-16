import json
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from shirabe.inference import ROOT, StyleModel

MAX_UPLOAD = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app):
    app.state.model = StyleModel() if (ROOT / "artifacts/model.json").exists() else None
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
    path = ROOT / "artifacts/report.json"
    if not path.exists():
        raise HTTPException(503, "No training run exists yet. Run scripts/train.py first.")
    return json.loads(path.read_text())


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


def extract_pdf(content):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "shirabe.pdf_worker"],
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
async def extract_upload(file: Annotated[UploadFile, File()]):
    try:
        content = await file.read(MAX_UPLOAD + 1)
        if len(content) > MAX_UPLOAD:
            raise HTTPException(413, "Upload must be 10 MB or smaller.")
        suffix = Path(file.filename or "").suffix.lower()
        pages = None
        if suffix == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise HTTPException(422, "The file is not a valid PDF. Upload a PDF or a UTF-8 text file.")
            payload = await run_in_threadpool(extract_pdf, content)
            text, pages = payload["text"], payload["pages"]
        elif suffix == ".txt":
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                raise HTTPException(422, "Save this text file as UTF-8 and try again.") from None
        else:
            raise HTTPException(415, "Choose a PDF or UTF-8 .txt file.")
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
    def index():
        return FileResponse(DIST / "index.html")
