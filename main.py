import os
import base64
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


def call_groq(prompt: str) -> str:
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


def extract_toc_pages(raw_text: str) -> str:
    lines = raw_text.split("\n")
    toc_start_idx = None
    current_page_idx = None
    for i, line in enumerate(lines):
        if re.match(r'\[PAGE \d+\]', line):
            current_page_idx = i
        if current_page_idx is not None and re.search(r'\bCONTENTS\b|\bCHAPTER[-\s]+1\b', line, re.IGNORECASE):
            context = "\n".join(lines[current_page_idx:current_page_idx+30])
            if re.search(r'CHAPTER.*\d+[-\d]*\s*$', context, re.MULTILINE | re.IGNORECASE):
                toc_start_idx = current_page_idx
                break

    if toc_start_idx is None:
        toc_match = re.search(r'(\[PAGE 1[0-9]\].*?CHAPTER)', raw_text, re.DOTALL)
        if toc_match:
            toc_start_idx_pos = raw_text.rfind('\n[PAGE', 0, toc_match.start()) + 1
            lines_before = raw_text[:toc_start_idx_pos].split("\n")
            toc_start_idx = len(lines_before)

    if toc_start_idx is None:
        return raw_text[10000:18000]

    toc_text_start = "\n".join(lines[toc_start_idx:])
    page_positions = [m.start() for m in re.finditer(r'\[PAGE \d+\]', toc_text_start)]
    if len(page_positions) >= 6:
        return toc_text_start[:page_positions[5]]
    return toc_text_start[:8000]


def get_page_content(raw_text: str, page_index: int, pages: int = 3) -> str:
    content = ""
    for p in range(page_index, page_index + pages):
        m1 = f"[PAGE {p}]"
        m2 = f"[PAGE {p+1}]"
        if m1 in raw_text:
            s = raw_text.index(m1) + len(m1)
            e = raw_text.index(m2) if m2 in raw_text else s + 2500
            content += raw_text[s:e].strip() + "\n\n"
    return content.strip()


def detect_page_offset(raw_text: str, toc_chapter1_page: int = 1) -> int:
    patterns = [
        r'\[PAGE (\d+)\]\s*\n.*?Chapter\s+1\b',
        r'\[PAGE (\d+)\]\s*\n\s*1\s*\n.*?INTRODUCTION',
        r'\[PAGE (\d+)\]\s*\n.*?INTRODUCTION',
        r'\[PAGE (\d+)\]\s*\n\s*Chapter\s+1',
    ]
    for pattern in patterns:
        m = re.search(pattern, raw_text, re.IGNORECASE | re.DOTALL)
        if m:
            pdf_page = int(m.group(1))
            offset = pdf_page - toc_chapter1_page
            return max(0, offset)
    roman_pages = re.findall(r'\[PAGE (\d+)\]\n[ivxlcdmIVXLCDM]+\s', raw_text)
    if roman_pages:
        return len(roman_pages)
    return 0


def clean_tree(nodes: list) -> list:
    """
    PageIndex approach: tree nodes contain ONLY navigation data.
    Strip any page_content, description, or extra fields the LLM may have added.
    The tree is a pure navigation index — titles + short summaries (text field).
    Raw content is fetched separately using page_index at query time.
    """
    allowed = {"node_id", "title", "page_index", "text", "nodes"}
    cleaned = []
    for node in nodes:
        clean_node = {k: v for k, v in node.items() if k in allowed}
        # Ensure text is short — truncate to 200 chars if LLM made it too long
        if "text" in clean_node and len(clean_node["text"]) > 300:
            clean_node["text"] = clean_node["text"][:300].rsplit(" ", 1)[0] + "..."
        # Recurse into children
        if "nodes" in clean_node and isinstance(clean_node["nodes"], list):
            clean_node["nodes"] = clean_tree(clean_node["nodes"])
        cleaned.append(clean_node)
    return cleaned


def count_nodes(nodes: list) -> int:
    c = 0
    for n in nodes:
        c += 1
        if n.get("nodes"):
            c += count_nodes(n["nodes"])
    return c


class UploadRequest(BaseModel):
    file_url: str
    file_name: Optional[str] = "document.pdf"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "llama-3.3-70b-versatile via Groq",
        "approach": "PageIndex — tree is navigation only, raw_text is content store",
        "mode": "stateless — stores nothing"
    }


@app.post("/doc/")
async def build_document_tree(
    req: UploadRequest,
    x_api_key: str = Header(...)
):
    """
    PageIndex tree building:
    1. Extract full PDF text with [PAGE N] markers → stored as raw_text
    2. Build navigation tree from TOC (titles + short descriptions only)
    3. NO page_content embedded in tree — raw_text is the content store
    4. At query time, tree search picks node_ids → page_index → raw_text fetch
    """
    verify_key(x_api_key)

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    if req.file_url.startswith("data:"):
        # Data URI — decode base64 payload directly
        try:
            header, encoded = req.file_url.split(",", 1)
            pdf_bytes = base64.b64decode(encoded)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid data URI: {e}")
    else:
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
        first_chapter_text = get_page_content(raw_text, 18, pages=3)

        # Detect front-matter offset
        offset = detect_page_offset(raw_text, toc_chapter1_page=1)

        prompt = f"""You are implementing the PageIndex document indexing framework.

Document: {req.file_name}
Total pages: {total_pages}
Detected front-matter offset: {offset} pages (roman numeral pages before Chapter 1)

=== TABLE OF CONTENTS (extracted from document) ===
{toc_text}

=== START OF CHAPTER 1 (for page number calibration) ===
{first_chapter_text[:2000]}

Build a COMPLETE hierarchical navigation tree following the EXACT structure from the Table of Contents.

CRITICAL RULES:
1. Every chapter must appear as a top-level node with ALL its sections as children
2. Every numbered section (1.1, 1.2, 2.3.1 etc) must appear as child/grandchild node
3. page_index: use EXACTLY the page numbers printed in TOC — do NOT add any offset
4. text: 1-2 sentence description of what that section covers — this is used for navigation.
   Keep it SHORT (under 200 chars). Do NOT include actual page content, quotes, or data.
   Example good text: "Covers thermal and catalytic decomposition reactions of H2O2."
   Example bad text: "H2O2 → H2O + 1/2O2. The enthalpy is -136.11 kJ/mol..."
5. nodes: must contain ALL child sections from TOC — never use empty array if subsections exist
6. node_id: "0001" top level, "0001_01" children, "0001_01_01" grandchildren
7. ONLY include these 5 fields per node: title, node_id, page_index, text, nodes
   DO NOT add: page_content, content, description, summary, raw_text, or any other field

Return a JSON object with a single "tree" key:
{{
  "tree": [
    {{
      "title": "Chapter 1 - Introduction",
      "node_id": "0001",
      "page_index": 1,
      "text": "Introduces spacecraft propulsion, green propellants, and catalysts overview.",
      "nodes": [
        {{
          "title": "1.1 Background",
          "node_id": "0001_01",
          "page_index": 1,
          "text": "Overview of spacecraft engine types and the role of propulsion catalysts.",
          "nodes": []
        }}
      ]
    }}
  ]
}}"""

        raw = call_groq(prompt)
        parsed = json.loads(raw)

        if isinstance(parsed, dict) and "tree" in parsed:
            tree = parsed["tree"]
        elif isinstance(parsed, list):
            tree = parsed
        else:
            for v in parsed.values():
                if isinstance(v, list):
                    tree = v
                    break
            else:
                raise ValueError(f"Unexpected response shape: {list(parsed.keys())}")

        # Clean tree — strip any page_content or extra fields LLM may have added
        # This enforces the PageIndex principle: tree = navigation only
        tree = clean_tree(tree)

        doc_id = f"pi-{uuid.uuid4().hex[:16]}"

        return {
            "doc_id": doc_id,
            "file_name": req.file_name,
            "tree": tree,           # navigation index only — no page content
            "raw_text": raw_text,   # content store — fetched by page_index at query time
            "tree_node_count": count_nodes(tree),
            "top_level_nodes": len(tree),
            "total_pages": total_pages,
            "page_offset_detected": offset,
            "status": "completed"
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        os.unlink(tmp_path)
