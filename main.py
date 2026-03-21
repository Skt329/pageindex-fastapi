import os
import uuid
import tempfile
import httpx
import json
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="PageIndex Self-Hosted — Mistral")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DOC_STORE = {}


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
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[PAGE {i+1}]\n{text.strip()}")
    return "\n\n".join(pages)


def build_tree(text: str, file_name: str) -> list:
    client = mistral()
    sample_text = text[:12000]
    prompt = f"""You are a document analysis expert. Analyse this document and create a hierarchical tree index like a detailed Table of Contents optimised for RAG retrieval.

Document: {file_name}

Document text:
{sample_text}

Return ONLY a valid JSON array. Each node must have:
- "title": section title (string)
- "node_id": unique id like "0001", "0002" etc (string)
- "page_index": estimated page number (integer)
- "text": 2-3 sentence summary of this section (string)
- "nodes": array of child nodes (can be empty array)

Return a maximum of 15 top-level nodes with up to 3 children each.
Return ONLY the JSON array, no markdown, no explanation."""

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
    raw = raw.strip()
    return json.loads(raw)


def search_tree(tree: list, query: str, raw_text: str) -> list:
    client = mistral()
    tree_summary = json.dumps([
        {
            "node_id": n.get("node_id"),
            "title": n.get("title"),
            "page": n.get("page_index"),
            "summary": n.get("text", "")[:200],
            "children": [c.get("title") for c in n.get("nodes", [])]
        }
        for n in tree
    ], indent=2)

    prompt = f"""You are a document retrieval expert using tree-based reasoning.

Query: {query}

Document tree index:
{tree_summary}

Which node_ids are most relevant to answer this query?
Reason step by step over the tree, then return ONLY a JSON array of 2-3 most relevant node_ids.
Example: ["0003", "0007"]
Return ONLY the JSON array, no explanation."""

    response = client.chat.completions.create(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.1
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    selected_ids = json.loads(raw)

    def find_nodes(nodes, ids):
        result = []
        for n in nodes:
            if n.get("node_id") in ids:
                result.append(n)
            result.extend(find_nodes(n.get("nodes", []), ids))
        return result

    retrieved = find_nodes(tree, selected_ids)

    for node in retrieved:
        page = node.get("page_index", 1)
        page_marker = f"[PAGE {page}]"
        if page_marker in raw_text:
            start = raw_text.find(page_marker)
            end = raw_text.find(f"[PAGE {page + 1}]", start)
            node["relevant_content"] = raw_text[start:end if end > 0 else start + 2000]

    return retrieved


class UploadRequest(BaseModel):
    file_url: str
    file_name: Optional[str] = "document.pdf"


class QueryRequest(BaseModel):
    doc_id: str
    query: str
    messages: Optional[List[dict]] = []


class DeleteRequest(BaseModel):
    doc_id: str


# ── HEALTH ───────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": "mistral",
        "docs_in_memory": len(DOC_STORE)
    }


# ── FLOW 1: Upload document ──────────────────────────────────

@app.post("/doc/")
async def upload_document(
    req: UploadRequest,
    x_api_key: str = Header(...)
):
    verify_key(x_api_key)

    if not os.getenv("MISTRAL_API_KEY"):
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not set on server")

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        r = await client.get(req.file_url)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Cannot download file: HTTP {r.status_code}")
        pdf_bytes = r.content

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        raw_text = extract_pdf_text(tmp_path)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        tree = build_tree(raw_text, req.file_name)

        doc_id = f"self-{uuid.uuid4().hex[:16]}"
        DOC_STORE[doc_id] = {
            "tree": tree,
            "raw_text": raw_text,
            "file_name": req.file_name,
            "status": "completed",
            "tree_node_count": len(tree)
        }
        return {"doc_id": doc_id}

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Tree parse error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tree build error: {str(e)}")
    finally:
        os.unlink(tmp_path)


# ── FLOW 1: Check status ─────────────────────────────────────

@app.get("/doc/{doc_id}/")
def get_status(doc_id: str, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    doc = DOC_STORE.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "doc_id": doc_id,
        "status": doc["status"],
        "retrieval_ready": True,
        "tree_node_count": doc.get("tree_node_count", 0)
    }


# ── FLOW 2: Chat with document ───────────────────────────────

@app.post("/chat/completions")
async def chat(
    req: QueryRequest,
    x_api_key: str = Header(...)
):
    verify_key(x_api_key)

    doc = DOC_STORE.get(req.doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found. Upload first via POST /doc/")

    try:
        retrieved = search_tree(doc["tree"], req.query, doc["raw_text"])

        context_parts = []
        for node in retrieved:
            title = node.get("title", "")
            content = node.get("relevant_content") or node.get("text", "")
            page = node.get("page_index", "")
            context_parts.append(f"[Section: {title} | Page: {page}]\n{content}")
        context = "\n\n---\n\n".join(context_parts)

        client = mistral()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Answer questions based ONLY on the "
                    "retrieved document sections. Cite section titles and page numbers.\n\n"
                    f"Retrieved context:\n{context}"
                )
            }
        ]
        for m in (req.messages or [])[:-1]:
            messages.append(m)
        messages.append({"role": "user", "content": req.query})

        completion = client.chat.completions.create(
            model="mistral-small-latest",
            messages=messages,
            max_tokens=1500,
            temperature=0.3
        )

        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": completion.choices[0].message.content
                },
                "finish_reason": "stop"
            }],
            "retrieved_nodes": retrieved,
            "tree_used": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── FLOW 3: List all documents ───────────────────────────────

@app.get("/docs/")
def list_docs(x_api_key: str = Header(...)):
    verify_key(x_api_key)
    docs = []
    for doc_id, doc in DOC_STORE.items():
        docs.append({
            "doc_id": doc_id,
            "file_name": doc.get("file_name", "unknown"),
            "status": doc.get("status", "unknown"),
            "tree_node_count": doc.get("tree_node_count", 0)
        })
    return {
        "documents": docs,
        "total": len(docs)
    }


# ── FLOW 4: Get tree structure ───────────────────────────────

@app.get("/doc/{doc_id}/tree")
def get_tree(doc_id: str, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    doc = DOC_STORE.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "doc_id": doc_id,
        "file_name": doc.get("file_name", ""),
        "tree": doc["tree"],
        "tree_node_count": doc.get("tree_node_count", 0)
    }


# ── DELETE document ──────────────────────────────────────────

@app.delete("/doc/{doc_id}/")
def delete_doc(doc_id: str, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    if doc_id not in DOC_STORE:
        raise HTTPException(status_code=404, detail="Document not found")
    del DOC_STORE[doc_id]
    return {"deleted": True, "doc_id": doc_id}