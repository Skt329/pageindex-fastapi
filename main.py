import os
import uuid
import tempfile
import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="PageIndex Self-Hosted API — Mistral")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store: doc_id -> {tree, file_name, status}
DOC_STORE = {}


# ── Auth ─────────────────────────────────────────────────────
def verify_key(x_api_key: str):
    expected = os.getenv("SERVER_API_KEY", "")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Mistral client (OpenAI-compatible) ───────────────────────
def get_mistral_client():
    """
    Mistral exposes an OpenAI-compatible REST API.
    We use the openai library, just pointing base_url at Mistral.
    Free tier model: mistral-small-latest
    """
    import openai
    return openai.OpenAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        base_url="https://api.mistral.ai/v1"
    )


# ── Models ───────────────────────────────────────────────────
class UploadRequest(BaseModel):
    file_url: str
    file_name: Optional[str] = "document.pdf"

class QueryRequest(BaseModel):
    doc_id: str
    query: str
    messages: Optional[List[dict]] = []


# ── Routes ───────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "provider": "mistral"}


@app.post("/doc/")
async def upload_document(
    req: UploadRequest,
    x_api_key: str = Header(...)
):
    verify_key(x_api_key)

    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not set on server")

    # Download PDF
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        r = await client.get(req.file_url)
        if r.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Could not download file. HTTP {r.status_code}"
            )
        pdf_bytes = r.content

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        # PageIndex internally uses the openai library.
        # We patch the environment so it calls Mistral instead.
        # CHATGPT_API_KEY is PageIndex's env var for the API key.
        os.environ["CHATGPT_API_KEY"] = mistral_key

        # Patch openai module's base URL before PageIndex imports it
        import openai
        openai.base_url = "https://api.mistral.ai/v1/"
        openai.api_key = mistral_key

        from pageindex import PageIndex
        pi = PageIndex(openai_api_key=mistral_key)
        tree = pi.get_tree(pdf_path=tmp_path)

        doc_id = f"self-{uuid.uuid4().hex[:16]}"
        DOC_STORE[doc_id] = {
            "tree": tree,
            "file_name": req.file_name,
            "status": "completed"
        }
        return {"doc_id": doc_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tree build error: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.get("/doc/{doc_id}/")
def get_doc_status(doc_id: str, x_api_key: str = Header(...)):
    verify_key(x_api_key)

    doc = DOC_STORE.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "doc_id": doc_id,
        "status": doc["status"],
        "retrieval_ready": doc["status"] == "completed",
        "tree_node_count": len(doc.get("tree", [])) if isinstance(doc.get("tree"), list) else 0
    }


@app.post("/chat/completions")
async def chat_with_doc(
    req: QueryRequest,
    x_api_key: str = Header(...)
):
    verify_key(x_api_key)

    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not set")

    doc = DOC_STORE.get(req.doc_id)
    if not doc:
        raise HTTPException(
            status_code=404,
            detail="Document not found. Upload first via POST /doc/"
        )

    tree = doc["tree"]

    try:
        # Patch openai to use Mistral
        import openai
        openai.base_url = "https://api.mistral.ai/v1/"
        openai.api_key = mistral_key
        os.environ["CHATGPT_API_KEY"] = mistral_key

        from pageindex import PageIndex
        pi = PageIndex(openai_api_key=mistral_key)

        # PageIndex tree search — LLM reasons over tree to find relevant nodes
        retrieved = pi.retrieve(tree=tree, query=req.query)

        # Build context from retrieved nodes
        context_parts = []
        nodes = retrieved if isinstance(retrieved, list) else [retrieved]
        for node in nodes:
            title = node.get("title", "")
            text = node.get("text", "") or node.get("relevant_content", "")
            page = node.get("page_index", "")
            if text:
                context_parts.append(f"[Section: {title} | Page: {page}]\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        # Generate final answer with Mistral
        client = get_mistral_client()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Answer ONLY from the retrieved "
                    "document sections below. Cite section title and page number.\n\n"
                    f"Context:\n{context}"
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

        answer = completion.choices[0].message.content

        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop"
                }
            ],
            "retrieved_nodes": nodes,
            "tree_used": True,
            "provider": "mistral-small-latest"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
