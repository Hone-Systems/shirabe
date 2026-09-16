"""Reproducible, resumable OpenAlex sample; never rank/select by citation outcome."""

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from langdetect import DetectorFactory, LangDetectException, detect

from shirabe.features import WORD_RE

DetectorFactory.seed = 42

FIELDS = {
    13: "Biochemistry, Genetics and Molecular Biology",
    17: "Computer Science",
    20: "Economics, Econometrics and Finance",
    27: "Medicine",
    31: "Physics and Astronomy",
    33: "Social Sciences",
}


def reconstruct(index):
    pairs = [(pos, word) for word, positions in (index or {}).items() for pos in positions]
    return " ".join(word for _, word in sorted(pairs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-cohort", type=int, default=400)
    args = parser.parse_args()
    raw_dir = ROOT / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources, records, seen, excluded = [], [], set(), {}
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        for field, name in FIELDS.items():
            for year in range(2016, 2022):
                for page in range(1, (args.per_cohort + 199) // 200 + 1):
                    params = {
                        "filter": f"publication_year:{year},primary_topic.field.id:{field},has_abstract:true,language:en,type:article,is_retracted:false",
                        "sample": args.per_cohort,
                        "seed": 42,
                        "per-page": 200,
                        "page": page,
                        "select": "id,doi,title,publication_year,cited_by_count,abstract_inverted_index,primary_topic,primary_location,language,is_retracted",
                    }
                    path = raw_dir / f"{field}-{year}-{args.per_cohort}-{page}.json"
                    if not path.exists():
                        for attempt in range(6):
                            try:
                                response = client.get("https://api.openalex.org/works", params=params)
                                response.raise_for_status()
                                payload = response.json()
                                assert "results" in payload
                                path.write_text(
                                    json.dumps(
                                        {
                                            "retrieved_at": datetime.now(UTC).isoformat(),
                                            "url": str(response.url),
                                            "payload": payload,
                                        }
                                    )
                                )
                                break
                            except (httpx.HTTPError, ValueError, AssertionError):
                                if attempt == 5:
                                    raise
                                time.sleep(2**attempt)
                        time.sleep(0.3)
                    cached = json.loads(path.read_text())
                    sources.append(
                        {
                            "file": str(path.relative_to(ROOT)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "url": cached["url"],
                            "retrieved_at": cached["retrieved_at"],
                        }
                    )
                    for work in cached["payload"]["results"]:
                        abstract = reconstruct(work.get("abstract_inverted_index"))
                        if not 80 <= len(WORD_RE.findall(abstract)) <= 800:
                            excluded["length"] = excluded.get("length", 0) + 1
                            continue
                        try:
                            language = detect(abstract)
                        except LangDetectException:
                            language = "unknown"
                        if language != "en":
                            excluded["language"] = excluded.get("language", 0) + 1
                            continue
                        title = " ".join((work.get("title") or "").lower().split())
                        keys = {
                            work["id"],
                            "title:" + title,
                            "text:" + hashlib.sha256(" ".join(abstract.lower().split()).encode()).hexdigest(),
                        }
                        if work.get("doi"):
                            keys.add(work["doi"].lower())
                        if keys & seen:
                            excluded["duplicate"] = excluded.get("duplicate", 0) + 1
                            continue
                        seen.update(keys)
                        source = (work.get("primary_location") or {}).get("source") or {}
                        records.append(
                            {
                                "id": work["id"],
                                "doi": work.get("doi"),
                                "title": work["title"],
                                "year": year,
                                "field_id": field,
                                "field": name,
                                "citations": work["cited_by_count"],
                                "abstract": abstract,
                                "source_id": source.get("id"),
                                "source_name": source.get("display_name"),
                            }
                        )
                print(f"{field} {year}: cumulative {len(records)} retained", flush=True)
    output = ROOT / "data/papers.jsonl"
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (ROOT / "data/provenance.json").write_text(
        json.dumps(
            {
                "provider": "OpenAlex",
                "requested_per_cohort": args.per_cohort,
                "seed": 42,
                "records": len(records),
                "excluded": excluded,
                "dataset_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "sources": sources,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
