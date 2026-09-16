"""Cached paid-provider boundary; credentials stay in environment or a local Senki checkout."""

import ast
import hashlib
import json
import os
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pymupdf
import tiktoken
from bs4 import BeautifulSoup
from dotenv import dotenv_values

from shirabe.inference import ROOT

CACHE = ROOT / "artifacts/private/research"
MODEL = "gpt-5.6-luna"
LOCK = threading.Lock()
ENCODER = tiktoken.get_encoding("o200k_base")


def credentials(senki=None):
    keys = [
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "DATAFORSEO_USERNAME",
        "DATAFORSEO_PASSWORD",
        "FIRECRAWL_API_KEY",
    ]
    result = {k: os.environ.get(k, "") for k in keys}
    if senki:
        values = dotenv_values(Path(senki) / ".env")
        for k in keys:
            if values.get(k):
                result[k] = values[k]
        # Read only literal credential defaults; never execute the other application's config.
        tree = ast.parse((Path(senki) / "ai/config.py").read_text())
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in keys and not result[target.id]:
                    args = node.value.args
                    if len(args) > 1 and isinstance(args[1], ast.Constant) and isinstance(args[1].value, str):
                        result[target.id] = args[1].value
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


class ResearchClients:
    def __init__(self, senki=None, budget=10):
        self.keys = credentials(senki)
        self.budget = budget
        CACHE.mkdir(parents=True, exist_ok=True)
        self.ledger = CACHE / "usage.jsonl"
        self.reserved = 0.0

    def spent(self):
        return (
            sum(json.loads(line)["cost_usd"] for line in self.ledger.read_text().splitlines())
            if self.ledger.exists()
            else 0.0
        )

    def reserve(self, amount):
        with LOCK:
            if self.spent() + self.reserved + amount > self.budget:
                raise RuntimeError("Research budget cap reached; completed calls are cached")
            self.reserved += amount

    def record(self, provider, key, cost, reserved, **kwargs):
        with LOCK:
            self.reserved -= reserved
            with self.ledger.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "at": datetime.now(UTC).isoformat(),
                            "provider": provider,
                            "cache_key": key,
                            "cost_usd": cost,
                            **kwargs,
                        }
                    )
                    + "\n"
                )

    def llm(self, operation, prompt, schema, repeat=0):
        tokens = len(ENCODER.encode(prompt, disallowed_special=()))
        if tokens > 1_000_000:
            raise ValueError(f"Input is {tokens} tokens; explicit splitting required. Nothing was truncated.")
        body = {
            "model": MODEL,
            "input": [{"role": "user", "content": prompt}],
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 24000,
            "store": False,
            "text": {"format": {"type": "json_schema", "name": operation, "strict": True, "schema": schema}},
        }
        key = digest({"request": body, "repeat": repeat})
        path = CACHE / "llm" / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
        else:
            reserve = (tokens * 0.4 + 24000 * 1.8) / 1e6
            self.reserve(reserve)
            try:
                response = httpx.post(
                    "https://api.openai.com/v1/responses",
                    json=body,
                    headers={"Authorization": "Bearer " + self.keys["OPENAI_API_KEY"]},
                    timeout=600,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"OpenAI HTTP {response.status_code}: {(response.json().get('error') or {}).get('code')}"
                    )
                data = response.json()
                usage = data.get("usage") or {}
                it = usage.get("input_tokens", 0)
                ot = usage.get("output_tokens", 0)
                cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
                ir, cr, orr = (0.4, 0.04, 1.8) if it > 272000 else (0.2, 0.02, 1.2)
                cost = ((it - cached) * ir + cached * cr + ot * orr) / 1e6
                self.record(
                    "openai",
                    key,
                    cost,
                    reserve,
                    operation=operation,
                    model=data.get("model"),
                    usage=usage,
                    input_characters=len(prompt),
                    truncated=False,
                )
                reserve = 0
                write_json(path, data)
            finally:
                if reserve:
                    # Conservatively reserve the potential spend for failed/ambiguous requests.
                    self.record(
                        "openai", key, reserve, reserve, operation=operation, status="failed_or_unknown"
                    )
        if data.get("status") != "completed":
            raise ValueError(f"Incomplete LLM response {data.get('status')}; not accepted as a label")
        text = "".join(
            c["text"]
            for item in data["output"]
            for c in item.get("content", [])
            if c.get("type") == "output_text"
        )
        return json.loads(text), {
            "cache_key": key,
            "model": data.get("model"),
            "usage": data.get("usage"),
            "input_tokens_estimate": tokens,
            "truncated": False,
        }

    def search(self, query, year, cutoff):
        # Discovery dates often reflect page updates, not study publication. Apply the
        # follow-up window to cited evidence, never exclude old studies by SERP freshness.
        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": 10,
            "include_usage": True,
            "include_published_date": True,
        }
        key = digest({"provider": "tavily_search", "payload": payload})
        path = CACHE / "search" / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
        else:
            self.reserve(0.02)
            try:
                r = httpx.post(
                    "https://api.tavily.com/search",
                    json={"api_key": self.keys["TAVILY_API_KEY"], **payload},
                    timeout=90,
                )
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPError:
                self.record("tavily_search", key, 0.02, 0.02, status="failed_or_unknown")
                raise
            self.record(
                "tavily_search",
                key,
                float((data.get("usage") or {}).get("credits", 1)) * 0.008,
                0.02,
                cost_basis="conservative credit estimate",
            )
            write_json(path, data)
        return [
            {
                "url": v["url"],
                "title": v.get("title"),
                "description": v.get("content"),
                "timestamp": v.get("published_date"),
            }
            for v in data.get("results", [])
        ]

    def search_dataforseo(self, query, year, cutoff):
        payload = {
            "keyword": query,
            "location_code": 2840,
            "language_code": "en",
            "depth": 10,
            "search_param": f"tbs=cdr:1,cd_min:1/1/{year + 1},cd_max:12/31/{cutoff}",
        }
        key = digest(payload)
        path = CACHE / "search" / f"{key}.json"
        data = json.loads(path.read_text()) if path.exists() else None
        for attempt in range(3):
            if data and all(t.get("status_code") == 20000 for t in data.get("tasks", [])):
                break
            self.reserve(0.02)
            try:
                r = httpx.post(
                    "https://api.dataforseo.com/v3/serp/google/organic/live/advanced",
                    json=[payload],
                    auth=(self.keys["DATAFORSEO_USERNAME"], self.keys["DATAFORSEO_PASSWORD"]),
                    timeout=90,
                )
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPError:
                self.record("dataforseo", key, 0.02, 0.02, status="failed_or_unknown")
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
                continue
            self.record("dataforseo", key, float(data.get("cost") or 0), 0.02)
            write_json(path, data)
            if not all(t.get("status_code") == 20000 for t in data.get("tasks", [])):
                time.sleep(2**attempt)
        items = []
        for task in data.get("tasks", []):
            if task.get("status_code") != 20000:
                raise RuntimeError(f"DataForSEO task status {task.get('status_code')}")
            for result in task.get("result") or []:
                for item in result.get("items") or []:
                    if item.get("type") in ("organic", "featured_snippet") and item.get("url"):
                        items.append({k: item.get(k) for k in ["url", "title", "description", "timestamp"]})
        return items

    def fetch(self, url):
        key = digest(url)
        path = CACHE / "pages" / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        text = ""
        method = ""
        status = None
        try:
            r = httpx.get(
                url,
                follow_redirects=True,
                timeout=40,
                headers={"User-Agent": "ShirabeResearch/0.2 (+https://github.com/Hone-Systems/shirabe)"},
            )
            status = r.status_code
            if r.status_code == 200 and r.content.startswith(b"%PDF-"):
                with pymupdf.open(stream=r.content, filetype="pdf") as doc:
                    text = "\n\n".join(page.get_text(sort=True) for page in doc)
                    method = "pdf_all_pages"
            elif r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                text = (soup.find("article") or soup.find("main") or soup).get_text("\n", strip=True)
                method = "html_main_text"
        except (httpx.HTTPError, ValueError, pymupdf.FileDataError):
            pass
        # No length clipping: extracted article content is kept in full.
        if len(text) < 500 or any(
            v in text[:500].lower() for v in ["just a moment", "access denied", "enable javascript"]
        ):
            self.reserve(0.012)
            try:
                r = httpx.post(
                    "https://api.tavily.com/extract",
                    json={"api_key": self.keys["TAVILY_API_KEY"], "urls": [url], "extract_depth": "advanced"},
                    timeout=90,
                )
                r.raise_for_status()
                data = r.json()
                self.record("tavily", key, 0.0024, 0.012)
                results = data.get("results") or []
                if results:
                    text = results[0].get("raw_content", "")
                    method = "tavily_complete_extraction"
            except httpx.HTTPError:
                self.record("tavily", key, 0.012, 0.012, status="failed_or_unknown")
        result = {
            "url": url,
            "text": text,
            "method": method,
            "http_status": status,
            "characters": len(text),
            "truncated": False,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "retrieved_at": datetime.now(UTC).isoformat(),
        }
        write_json(path, result)
        return result
