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

app = FastAPI(title="PageIndex Tree Builder — Gemini 2.5 Flash Two-Pass")

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


def gemini(prompt: str, schema: dict, max_tokens: int = 65536) -> any:
    """
    Call Gemini 2.5 Flash with responseSchema structured output.
    Explicitly sets maxOutputTokens to 65536 — the API default is only 8192
    which silently truncates large JSON outputs mid-string.
    thinkingBudget=0 disables chain-of-thought to save output tokens.
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
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "thinkingConfig": {"thinkingBudget": 0}
        }
    }

    resp = httpx.post(url, json=body, timeout=180)
    data = resp.json()

    if "error" in data:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini error: {data['error']['message']}"
        )

    candidates = data.get("candidates", [])
    if not candidates:
        raise HTTPException(status_code=500, detail="Gemini returned no candidates")

    finish = candidates[0].get("finishReason", "")
    if finish == "MAX_TOKENS":
        raise HTTPException(
            status_code=500,
            detail="Gemini output was truncated — document is too large for one pass"
        )

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise HTTPException(status_code=500, detail="Gemini returned empty content")

    return json.loads(parts[0]["text"].strip())


# ── Schemas ───────────────────────────────────────────────────────────────────

# PASS 1: skeleton tree — just structure, no summaries
# Small output — always fits in token budget
SKELETON_NODE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title":      {"type": "STRING"},
        "node_id":    {"type": "STRING"},
        "page_index": {"type": "INTEGER"},
        "nodes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title":      {"type": "STRING"},
                    "node_id":    {"type": "STRING"},
                    "page_index": {"type": "INTEGER"},
                    "nodes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "title":      {"type": "STRING"},
                                "node_id":    {"type": "STRING"},
                                "page_index": {"type": "INTEGER"},
                                "nodes":      {"type": "ARRAY", "items": {"type": "OBJECT"}}
                            },
                            "required": ["title", "node_id", "page_index", "nodes"]
                        }
                    }
                },
                "required": ["title", "node_id", "page_index", "nodes"]
            }
        }
    },
    "required": ["title", "node_id", "page_index", "nodes"]
}

SKELETON_SCHEMA = {"type": "ARRAY", "items": SKELETON_NODE_SCHEMA}

# PASS 2: summaries — one string per node_id
SUMMARY_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "node_id": {"type": "STRING"},
            "text":    {"type": "STRING"}
        },
        "required": ["node_id", "text"]
    }
}


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: str) -> tuple:
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    total = len(reader.pages)
    pages = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        if t.strip():
            pages.append(f"[PAGE {i+1}]\n{t.strip()}")
    return "\n\n".join(pages), total


def find_toc(raw_text: str) -> str:
    """Extract the Table of Contents pages from raw text."""
    for pattern in [r"CONTENTS", r"TABLE OF CONTENTS", r"CONTENT"]:
        m = re.search(
            r"(\[PAGE \d+\].*?" + pattern + r".*?)(?=\[PAGE \d+\])",
            raw_text, re.DOTALL | re.IGNORECASE
        )
        if m:
            start = raw_text.find(m.group(0))
            rest = raw_text[start:]
            markers = [x.start() for x in re.finditer(r'\[PAGE \d+\]', rest)]
            end = markers[4] if len(markers) >= 5 else len(rest)
            return rest[:end]
    return raw_text[:4000]  # fallback: first 4k chars


def get_page_content(raw_text: str, page_index: int, pages_ahead: int = 3) -> str:
    """Fetch actual PDF content for a page range from raw_text."""
    content = ""
    for p in range(page_index, page_index + pages_ahead):
        marker = f"[PAGE {p}]"
        next_marker = f"[PAGE {p + 1}]"
        if marker in raw_text:
            start = raw_text.index(marker) + len(marker)
            end = raw_text.index(next_marker) if next_marker in raw_text else start + 2500
            content += raw_text[start:end].strip() + "\n\n"
    return content.strip()


# ── Tree helpers ──────────────────────────────────────────────────────────────

def collect_all_nodes(nodes: list) -> list:
    """Flatten all nodes from nested tree into a list."""
    result = []
    for n in nodes:
        result.append(n)
        if n.get("nodes"):
            result.extend(collect_all_nodes(n["nodes"]))
    return result


def apply_summaries(nodes: list, summary_map: dict) -> list:
    """Merge summaries into tree nodes in-place."""
    for n in nodes:
        nid = n.get("node_id", "")
        n["text"] = summary_map.get(nid, f"Section: {n.get('title', '')}")
        if n.get("nodes"):
            n["nodes"] = apply_summaries(n["nodes"], summary_map)
    return nodes


def count_nodes(nodes: list) -> int:
    count = 0
    for n in nodes:
        count += 1
        if n.get("nodes"):
            count += count_nodes(n["nodes"])
    return count


# ── Main route ────────────────────────────────────────────────────────────────

class UploadRequest(BaseModel):
    file_url: str
    file_name: Optional[str] = "document.pdf"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "gemini-2.5-flash",
        "approach": "two-pass structured output",
        "pass1": "TOC → skeleton tree (structure only)",
        "pass2": "page content → node summaries"
    }


@app.post("/doc/")
async def build_document_tree(
    req: UploadRequest,
    x_api_key: str = Header(...)
):
    """
    Two-pass PageIndex tree building:

    Pass 1: Send TOC text only → Gemini returns skeleton tree
            (titles, node_ids, page_indexes, nested structure)
            Small output → never truncated

    Pass 2: For each node, fetch actual page content from raw_text
            → Gemini generates 2-sentence summaries for all nodes at once
            This is the actual PageIndex RAG content

    Result: Full hierarchical tree with real summaries grounded in PDF content.
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
        if "text/html" in r.headers.get("content-type", ""):
            raise HTTPException(
                status_code=400,
                detail="URL returned HTML. For Google Drive use: https://drive.google.com/uc?export=download&id=FILE_ID"
            )
        pdf_bytes = r.content

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        raw_text, total_pages = extract_pdf_text(tmp_path)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from PDF")

        toc_text = find_toc(raw_text)

        # ── PASS 1: skeleton tree from TOC ────────────────────────────────────
        # Only sends TOC text (~1-4 pages), output is just titles+ids+pages
        # Will never truncate — a 100-node skeleton is ~3000 tokens output

        pass1_prompt = f"""You are implementing the PageIndex framework for hierarchical document indexing.

Document: {req.file_name}
Total pages: {total_pages}

Table of Contents / document structure:
{toc_text}

Build a COMPLETE hierarchical skeleton tree. Rules:
- 3 levels: chapters → sections (1.1, 1.2) → subsections (1.1.1)
- Include ALL chapters, ALL numbered sections, ALL subsections visible in the TOC
- Front matter (title page, abstract, acknowledgements etc.) = flat top-level nodes with empty nodes array
- page_index = page number where that section starts (use the numbers shown in the TOC)
- node_id format: top-level "0001","0002" | children "0001_01","0001_02" | grandchildren "0001_01_01"
- node_ids must be unique across the ENTIRE tree
- Do NOT include "text" summaries — just title, node_id, page_index, nodes"""

        skeleton = gemini(pass1_prompt, SKELETON_SCHEMA, max_tokens=16384)

        # ── PASS 2: summaries from actual page content ─────────────────────────
        # Collect all nodes, fetch their actual page text, batch summarize

        all_nodes = collect_all_nodes(skeleton)

        # Build content snippets for each node
        node_contents = []
        for node in all_nodes:
            content = get_page_content(raw_text, node.get("page_index", 1), pages_ahead=2)
            if content:
                node_contents.append(
                    f'node_id: {node["node_id"]}\ntitle: {node["title"]}\ncontent:\n{content[:1500]}'
                )

        # Batch all summaries in one call
        pass2_prompt = f"""For each section below, write a 2-sentence summary of what it actually contains.
Base your summary ONLY on the provided content text, not on the title alone.

Document: {req.file_name}

Sections:
{"---".join(node_contents[:40])}

Return one object per node_id with the summary in the "text" field."""

        summaries = gemini(pass2_prompt, SUMMARY_SCHEMA, max_tokens=16384)
        summary_map = {s["node_id"]: s["text"] for s in summaries}

        # Merge summaries into skeleton
        final_tree = apply_summaries(skeleton, summary_map)

        doc_id = f"pi-{uuid.uuid4().hex[:16]}"

        return {
            "doc_id": doc_id,
            "file_name": req.file_name,
            "tree": final_tree,
            "raw_text": raw_text,
            "tree_node_count": count_nodes(final_tree),
            "top_level_nodes": len(final_tree),
            "total_pages": total_pages,
            "status": "completed"
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        os.unlink(tmp_path)
