"""Isolated PDF text extraction with process time and memory limits. No OCR."""

import json
import resource
import sys

resource.setrlimit(resource.RLIMIT_AS, (1_500_000_000, 1_500_000_000))
resource.setrlimit(resource.RLIMIT_CPU, (12, 12))
import pymupdf

try:
    content = sys.stdin.buffer.read(10 * 1024 * 1024 + 1)
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("PDF exceeds 10 MB.")
    with pymupdf.open(stream=content, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError("This PDF is password-protected. Upload an unlocked PDF or paste its abstract.")
        if len(doc) > 200:
            raise ValueError("This PDF has more than 200 pages. Paste the paper's abstract instead.")
        lines = []
        for i in range(min(3, len(doc))):
            for block in doc[i].get_text("dict", sort=True)["blocks"]:
                for line in block.get("lines", []):
                    # Exclude rotated margin stamps, which can splice arXiv IDs into prose.
                    if abs(line["dir"][1]) > 0.1 or line["dir"][0] < 0:
                        continue
                    lines.append("".join(span["text"] for span in line["spans"]))
        text = "\n".join(lines)
        if len(text.strip()) < 100:
            raise ValueError("No readable text found. Scanned PDFs need OCR; paste the abstract instead.")
        print(json.dumps({"text": text[:60000], "pages": len(doc)}))
except Exception as exc:
    print(
        json.dumps(
            {
                "error": str(exc)
                if isinstance(exc, ValueError)
                else "Could not read this PDF. It may be corrupt; try pasting the abstract."
            }
        )
    )
    sys.exit(1)
