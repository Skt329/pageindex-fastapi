import os
import uuid
import tempfile
import httpx
import json
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="PageIndex Tree Builder — Stateless")

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


def mistral():
    import openai
    return openai.OpenAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        base_url="https://api.mistral.ai/v1"
    )


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text page by page with [PAGE N] markers — these are used
    during retrieval to fetch exact page content after tree search."""
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[PAGE {i+1}]\n{text.strip()}")
    return "\n\n".join(pages)


def build_tree(text: str, file_name: str) -> list:
    """Build PageIndex-style hierarchical tree from document text."""
    client = mistral()
    # Use first 12000 chars — enough for structure analysis
    sample_text = text[:12000]

    prompt = f"""You are a document analysis expert. Analyse this document and create a hierarchical tree index like a detailed Table of Contents, optimised for LLM-based retrieval.

Document: {file_name}

Document text (first portion):
{sample_text}

Return ONLY a valid JSON array. Each node must have exactly these fields:
- "title": section title (string)
- "node_id": unique zero-padded id like "0001", "0002" (string)
- "page_index": page number where this section starts (integer, 1-based)
- "text": 2-3 sentence summary of what this section contains (string)
- "nodes": array of child nodes with same structure (empty array if no children)

Rules:
- Maximum 15 top-level nodes
- Maximum 3 levels of nesting
- node_ids must be unique across entire tree
- Return ONLY the JSON array, no markdown fences, no explanation"""

    response = client.chat.completions.create(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        temperature=0.1
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


class UploadRequest(BaseModel):
    file_url: str
    file_name: Optional[str] = "document.pdf"


@app.get("/health")
def health():
    return {"status": "ok", "mode": "stateless — tree builder only"}


@app.post("/doc/")
async def build_document_tree(
    req: UploadRequest,
    x_api_key: str = Header(...)
):
    """
    Downloads PDF, extracts text with page markers, builds PageIndex tree.
    Returns tree + raw_text to caller (Lamatic Flow 1).
    This server stores NOTHING — Lamatic saves to Supabase.
    """
    verify_key(x_api_key)

    if not os.getenv("MISTRAL_API_KEY"):
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not set")

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        r = await client.get(req.file_url)
        if r.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot download file: HTTP {r.status_code}"
            )
        pdf_bytes = r.content

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        raw_text = extract_pdf_text(tmp_path)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")

        tree = build_tree(raw_text, req.file_name)
        doc_id = f"pi-{uuid.uuid4().hex[:16]}"

        return {
            "doc_id": doc_id,
            "file_name": req.file_name,
            "tree": tree,
            "raw_text": raw_text,          # stored in Supabase, used for page fetch
            "tree_node_count": len(tree),
            "status": "completed"
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Tree parse error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        os.unlink(tmp_path)