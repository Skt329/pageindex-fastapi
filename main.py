import os
import uuid
import tempfile
import httpx
import json
import re
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="PageIndex Tree Builder — Gemini 2.5 Flash Structured Output")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_key(x_api_key: str):
    expected = os.getenv("SERVER_API_KEY", "")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Gemini 2.5 Flash structured output ───────────────────────────────────────

NODE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title":      {"type": "STRING"},
        "node_id":    {"type": "STRING"},
        "page_index": {"type": "INTEGER"},
        "text":       {"type": "STRING"},
        "nodes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title":      {"type": "STRING"},
                    "node_id":    {"type": "STRING"},
                    "page_index": {"type": "INTEGER"},
                    "text":       {"type": "STRING"},
                    "nodes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "title":      {"type": "STRING"},
                                "node_id":    {"type": "STRING"},
                                "page_index": {"type": "INTEGER"},
                                "text":       {"type": "STRING"},
                                "nodes":      {"type": "ARRAY", "items": {"type": "OBJECT"}}
                            },
                            "required": ["title", "node_id", "page_index", "text", "nodes"]
                        }
                    }
                },
                "required": ["title", "node_id", "page_index", "text", "nodes"]
            }
        }
    },
    "required": ["title", "node_id", "page_index", "text", "nodes"]
}

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": NODE_SCHEMA
}


def call_gemini_structured(prompt: str) -> list:
    """
    Call Gemini 2.5 Flash with responseSchema for guaranteed structured JSON output.
    Free tier: 10 RPM, 250 RPD. No credit card required.
    Model: gemini-2.5-flash (stable, not deprecated)
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA
        }
    }

    response = httpx.post(url, json=body, timeout=180)
    data = response.json()

    if "error" in data:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini error: {data['error']['message']}"
        )

    raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    return json.loads(raw)


# ── PDF text extraction ───────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: str) -> tuple:
    """Extract text page by page with [PAGE N] markers."""
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[PAGE {i+1}]\n{text.strip()}")
    return "\n\n".join(pages), total_pages


def find_toc_pages(raw_text: str) -> str:
    """
    Find and extract the Table of Contents section.
    Returns up to 4 pages worth of TOC content.
    """
    patterns = [
        r"CONTENTS",
        r"TABLE OF CONTENTS",
        r"CONTENT",
        r"INDEX",
    ]
    for pattern in patterns:
        match = re.search(
            r"(\[PAGE \d+\].*?" + pattern + r".*?)(?=\[PAGE \d+\])",
            raw_text, re.DOTALL | re.IGNORECASE
        )
        if match:
            start = raw_text.find(match.group(0))
            remaining = raw_text[start:]
            markers = [m.start() for m in re.finditer(r'\[PAGE \d+\]', remaining)]
            if len(markers) >= 5:
                return remaining[:markers[4]]
            return remaining[:6000]
    return ""


def count_all_nodes(nodes: list) -> int:
    """Recursively count all nodes including nested."""
    count = 0
    for node in nodes:
        count += 1
        if node.get("nodes"):
            count += count_all_nodes(node["nodes"])
    return count


# ── Routes ────────────────────────────────────────────────────────────────────

class UploadRequest(BaseModel):
    file_url: str
    file_name: Optional[str] = "document.pdf"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "gemini-2.5-flash",
        "output": "structured JSON via responseSchema",
        "mode": "stateless — stores nothing"
    }


@app.post("/doc/")
async def build_document_tree(
    req: UploadRequest,
    x_api_key: str = Header(...)
):
    """
    Downloads PDF → extracts full text with [PAGE N] markers →
    builds complete PageIndex-style hierarchical tree using
    Gemini 2.5 Flash with responseSchema (guaranteed valid JSON).
    Returns tree + raw_text for Lamatic to save to Supabase.
    """
    verify_key(x_api_key)

    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    # Download PDF
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        r = await client.get(req.file_url)
        if r.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot download: HTTP {r.status_code}"
            )
        content_type = r.headers.get("content-type", "")
        if "text/html" in content_type:
            raise HTTPException(
                status_code=400,
                detail=(
                    "URL returned HTML not PDF. "
                    "For Google Drive use: "
                    "https://drive.google.com/uc?export=download&id=FILE_ID"
                )
            )
        pdf_bytes = r.content

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        raw_text, total_pages = extract_pdf_text(tmp_path)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from PDF")

        toc_text = find_toc_pages(raw_text)
        toc_section = ""
        if toc_text:
            toc_section = f"""
===TABLE OF CONTENTS (use this as the primary structure reference)===
{toc_text}
===END TABLE OF CONTENTS===
"""

        # Send up to 80k chars of document — covers most academic papers/theses
        doc_sample = raw_text[:80000] if len(raw_text) > 80000 else raw_text

        prompt = f"""You are implementing the PageIndex framework for vectorless RAG retrieval.
Your task: build a COMPLETE hierarchical tree index of this document.

Document: {req.file_name}
Total pages: {total_pages}
{toc_section}
Document text with [PAGE N] page markers:
{doc_sample}

RULES:
1. Build 3 levels: chapters → sections (1.1, 1.2) → subsections (1.1.1, 1.1.2)
2. Include ALL chapters, ALL numbered sections, ALL subsections from the document
3. Use [PAGE N] markers to set accurate page_index values for every node
4. Each node's "text" field = 2-3 sentences describing actual content of that section
5. node_id format: top-level "0001","0002" — children "0001_01","0001_02" — grandchildren "0001_01_01"
6. Front matter (title, abstract, acknowledgements) = flat top-level nodes with empty nodes array
7. Every chapter that has numbered sections MUST have those as child nodes
8. Every section that has numbered subsections MUST have those as grandchild nodes

For a thesis the output should have ~30-60 total nodes across all levels."""

        tree = call_gemini_structured(prompt)
        total_node_count = count_all_nodes(tree)
        doc_id = f"pi-{uuid.uuid4().hex[:16]}"

        return {
            "doc_id": doc_id,
            "file_name": req.file_name,
            "tree": tree,
            "raw_text": raw_text,
            "tree_node_count": total_node_count,
            "top_level_nodes": len(tree),
            "total_pages": total_pages,
            "status": "completed"
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Tree parse error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        os.unlink(tmp_path)
