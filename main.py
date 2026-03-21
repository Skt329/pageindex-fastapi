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

app = FastAPI(title="PageIndex Tree Builder — Groq + llama-3.3-70b")

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


# ── Groq call — JSON mode, no schema constraints ─────────────────────────────
def call_groq(prompt: str) -> str:
    """
    Groq: completely free forever, no credits, no expiry.
    Model: llama-3.3-70b-versatile — 128k context, excellent instruction following.
    JSON mode: forces valid JSON output, no schema constraints that break nesting.
    Free limits: 6000 tokens/min, 500 requests/day on free tier.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a document analysis expert. You always return valid JSON exactly as requested, with no extra text, no markdown fences, no explanation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 8000,
            "response_format": {"type": "json_object"}
        },
        timeout=120
    )

    data = resp.json()
    if "error" in data:
        raise HTTPException(status_code=500, detail=f"Groq error: {data['error']['message']}")

    return data["choices"][0]["message"]["content"].strip()


# ── PDF extraction ────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: str) -> tuple:
    """Extract full text with [PAGE N] markers."""
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    total = len(reader.pages)
    pages = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        if t.strip():
            pages.append(f"[PAGE {i+1}]\n{t.strip()}")
    return "\n\n".join(pages), total


def extract_toc_pages(raw_text: str) -> str:
    """
    Extract the actual Table of Contents pages.
    Finds [PAGE N] markers that contain 'CONTENTS' or 'CHAPTER' patterns.
    Returns text of those pages + next 4 pages.
    """
    lines = raw_text.split("\n")
    toc_start_idx = None

    # Find the page marker just before the TOC
    current_page_idx = None
    for i, line in enumerate(lines):
        if re.match(r'\[PAGE \d+\]', line):
            current_page_idx = i
        # Look for TOC indicators in current page content
        if current_page_idx is not None and re.search(r'\bCONTENTS\b|\bCHAPTER[-\s]+1\b', line, re.IGNORECASE):
            # Check if this looks like a TOC (has page numbers pattern like "1-8" or just "1")
            context = "\n".join(lines[current_page_idx:current_page_idx+30])
            if re.search(r'CHAPTER.*\d+[-\d]*\s*$', context, re.MULTILINE | re.IGNORECASE):
                toc_start_idx = current_page_idx
                break

    if toc_start_idx is None:
        # Fallback: find first occurrence of CHAPTER-1 or similar
        match = re.search(r'\[PAGE \d+\]', raw_text)
        if match:
            # Try to find pages 12-17 directly as most theses have TOC there
            toc_match = re.search(r'(\[PAGE 1[0-9]\].*?CHAPTER)', raw_text, re.DOTALL)
            if toc_match:
                toc_start_idx = raw_text.rfind('\n[PAGE', 0, toc_match.start()) + 1

    if toc_start_idx is None:
        return raw_text[10000:18000]  # fallback: middle section

    toc_text_start = "\n".join(lines[toc_start_idx:])

    # Find all [PAGE N] positions from here
    page_positions = [m.start() for m in re.finditer(r'\[PAGE \d+\]', toc_text_start)]

    # Return up to 5 pages of TOC content (covers multi-page TOCs)
    if len(page_positions) >= 6:
        return toc_text_start[:page_positions[5]]
    return toc_text_start[:8000]


def get_page_content(raw_text: str, page_index: int, pages: int = 3) -> str:
    """Fetch actual PDF content for a given page range."""
    content = ""
    for p in range(page_index, page_index + pages):
        m1 = f"[PAGE {p}]"
        m2 = f"[PAGE {p+1}]"
        if m1 in raw_text:
            s = raw_text.index(m1) + len(m1)
            e = raw_text.index(m2) if m2 in raw_text else s + 2500
            content += raw_text[s:e].strip() + "\n\n"
    return content.strip()


def flatten_tree(nodes: list) -> list:
    """Flatten nested tree to list."""
    result = []
    for n in nodes:
        result.append(n)
        if n.get("nodes"):
            result.extend(flatten_tree(n["nodes"]))
    return result


def enrich_summaries(tree: list, raw_text: str) -> list:
    """Add real page content as page_content field to each node."""
    for node in tree:
        page = node.get("page_index", 1)
        content = get_page_content(raw_text, page, pages=2)
        node["page_content"] = content[:1500] if content else node.get("text", "")
        if node.get("nodes"):
            node["nodes"] = enrich_summaries(node["nodes"], raw_text)
    return tree


def count_nodes(nodes: list) -> int:
    c = 0
    for n in nodes:
        c += 1
        if n.get("nodes"):
            c += count_nodes(n["nodes"])
    return c


# ── Routes ────────────────────────────────────────────────────────────────────

class UploadRequest(BaseModel):
    file_url: str
    file_name: Optional[str] = "document.pdf"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "llama-3.3-70b-versatile via Groq",
        "cost": "completely free forever",
        "mode": "stateless — stores nothing"
    }


@app.post("/doc/")
async def build_document_tree(
    req: UploadRequest,
    x_api_key: str = Header(...)
):
    """
    PageIndex tree building with Groq (free):
    1. Extract full PDF text with page markers
    2. Find and extract TOC pages
    3. Single-pass tree building from TOC + context
    4. Enrich each node with actual page content
    """
    verify_key(x_api_key)

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    # Download PDF
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        r = await client.get(req.file_url)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Cannot download: HTTP {r.status_code}")
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

        toc_text = extract_toc_pages(raw_text)
        # Also include first chapter start for context
        first_chapter_text = get_page_content(raw_text, 18, pages=3)

        # Single pass — build complete tree from TOC
        # Return JSON object with "tree" key containing the array
        prompt = f"""You are implementing the PageIndex document indexing framework.

Document: {req.file_name}
Total pages: {total_pages}

=== TABLE OF CONTENTS (pages extracted from document) ===
{toc_text}

=== START OF CHAPTER 1 (for page number calibration) ===
{first_chapter_text[:2000]}

Build a COMPLETE hierarchical tree index following the EXACT structure from the Table of Contents above.

CRITICAL RULES:
1. Every chapter in the TOC must appear as a top-level node with all its sections as child nodes
2. Every numbered section (1.1, 1.2, 2.3.1 etc) must appear as a child or grandchild node
3. page_index: use the page numbers shown in the TOC (convert roman numerals: Chapter 1 starts at page 1 of main text = PDF page 18 based on sample above)
4. text: 1-2 sentence description of what that section covers
5. nodes: must contain all child sections — NEVER leave nodes as empty array if TOC shows subsections exist
6. node_id: "0001" for top level, "0001_01" for children, "0001_01_01" for grandchildren

Return a JSON object with a single "tree" key containing the array:
{{
  "tree": [
    {{
      "title": "Chapter 1 - Introduction",
      "node_id": "0001",
      "page_index": 18,
      "text": "Introduces spacecraft propulsion, catalysts, and green propellants.",
      "nodes": [
        {{
          "title": "1.1 Background",
          "node_id": "0001_01",
          "page_index": 18,
          "text": "Overview of spacecraft engine types and propulsion catalyst role.",
          "nodes": []
        }},
        {{
          "title": "1.2 Decomposition of Hydrogen Peroxide",
          "node_id": "0001_02",
          "page_index": 20,
          "text": "Thermal and catalytic decomposition reactions of H2O2.",
          "nodes": []
        }}
      ]
    }}
  ]
}}"""

        raw = call_groq(prompt)
        parsed = json.loads(raw)

        # Handle both {"tree": [...]} and direct array
        if isinstance(parsed, dict) and "tree" in parsed:
            tree = parsed["tree"]
        elif isinstance(parsed, list):
            tree = parsed
        else:
            # Try to find an array value in the dict
            for v in parsed.values():
                if isinstance(v, list):
                    tree = v
                    break
            else:
                raise ValueError(f"Unexpected response shape: {list(parsed.keys())}")

        # Enrich with actual page content from raw_text
        tree = enrich_summaries(tree, raw_text)

        doc_id = f"pi-{uuid.uuid4().hex[:16]}"

        return {
            "doc_id": doc_id,
            "file_name": req.file_name,
            "tree": tree,
            "raw_text": raw_text,
            "tree_node_count": count_nodes(tree),
            "top_level_nodes": len(tree),
            "total_pages": total_pages,
            "status": "completed"
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        os.unlink(tmp_path)
