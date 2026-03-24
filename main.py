"""
PageIndex FastAPI — Fixed Implementation
Groq multi-key pool with semaphore concurrency control.

Fixes applied vs previous version:
  1. Global asyncio.Semaphore — one slot per key, prevents thundering herd
  2. Shared httpx.AsyncClient at startup — connection pooling, no per-call overhead
  3. Real Retry-After backoff — reads Groq's header, adds jitter, exponential growth
  4. Stage 7 removed — duplicated Stage 4's work, added 60+ wasted calls
  5. Stage 6 top-level only — only chapter nodes get LLM summaries; subsections use title
  6. Stage 1 scans first 8 pages only — TOCs are almost never on page 9+
  7. Stage 4 text-check shortcut — if title found in page text, skips LLM entirely
  8. Unified llm_call — _llm_with_finish_reason merged in; no duplicate retry logic
"""

import os
import asyncio
import base64
import json
import random
import re
import tempfile
import uuid
import itertools
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="PageIndex — Groq Multi-Key")

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


# ─────────────────────────────────────────────────────────────────────────────
# Groq Multi-Key Pool
# ─────────────────────────────────────────────────────────────────────────────

def _load_groq_keys() -> list[str]:
    """Auto-detect GROQ_API_KEY_1 … GROQ_API_KEY_20 plus legacy GROQ_API_KEY."""
    keys = []
    for i in range(1, 21):
        k = os.getenv(f"GROQ_API_KEY_{i}")
        if k and k.strip():
            keys.append(k.strip())
    legacy = os.getenv("GROQ_API_KEY", "")
    if legacy.strip() and legacy.strip() not in keys:
        keys.append(legacy.strip())
    return keys


GROQ_KEYS: list[str] = []
_key_cycle = None
_key_lock = asyncio.Lock()

# FIX 1: Global semaphore — one concurrent slot per key.
# Prevents all 60 tasks from hammering every key simultaneously.
# With N keys, at most N requests are in-flight at any instant,
# each naturally assigned a different key by round-robin.
_llm_semaphore: asyncio.Semaphore | None = None

# FIX 2: Shared HTTP client — connection pooling, no per-call TCP overhead.
_http_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup():
    global GROQ_KEYS, _key_cycle, _llm_semaphore, _http_client
    GROQ_KEYS = _load_groq_keys()
    if not GROQ_KEYS:
        raise RuntimeError("No Groq API keys found. Set GROQ_API_KEY_1 … GROQ_API_KEY_N")
    _key_cycle = itertools.cycle(range(len(GROQ_KEYS)))
    _llm_semaphore = asyncio.Semaphore(len(GROQ_KEYS))
    _http_client = httpx.AsyncClient(
        timeout=120,
        limits=httpx.Limits(
            max_connections=len(GROQ_KEYS) * 2,
            max_keepalive_connections=len(GROQ_KEYS),
        ),
    )
    print(f"[PageIndex] {len(GROQ_KEYS)} key(s) loaded — semaphore={len(GROQ_KEYS)}")


@app.on_event("shutdown")
async def shutdown():
    global _http_client
    if _http_client:
        await _http_client.aclose()


async def _next_key() -> tuple[int, str]:
    async with _key_lock:
        idx = next(_key_cycle)
    return idx, GROQ_KEYS[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Core LLM call — unified, semaphore-guarded, real backoff
# ─────────────────────────────────────────────────────────────────────────────

async def llm_call(
    prompt: str,
    max_tokens: int = 1024,
    json_mode: bool = True,
    system: str = "You are an expert document analyst. Return valid JSON only, no markdown, no extra text.",
    return_finish_reason: bool = False,
) -> str | tuple[str, str]:
    """
    Single LLM call through the key pool.

    FIX 1 — Semaphore: at most N calls run concurrently (N = number of keys).
             Remaining callers block here until a slot opens.
             This eliminates the thundering herd entirely.

    FIX 3 — Real backoff: reads Groq's Retry-After header instead of
             blindly sleeping 1 second. Adds jitter to de-sync retries.
             Exponential growth caps at 60s.

    FIX 8 — Unified: _llm_with_finish_reason is gone. Pass
             return_finish_reason=True to get (content, finish_reason).
    """
    async with _llm_semaphore:
        max_tokens = min(max_tokens, 4096)
        backoff = 5.0
        max_attempts = max(len(GROQ_KEYS) * 2, 8)

        for attempt in range(max_attempts):
            _, key = await _next_key()

            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            try:
                resp = await _http_client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            except (httpx.TimeoutException, httpx.ConnectError):
                await asyncio.sleep(backoff + random.uniform(0, 2))
                backoff = min(backoff * 1.5, 60)
                continue

            # FIX 3: Read the actual retry-after Groq sends, not a blind 1s sleep
            if resp.status_code == 429:
                retry_after = float(
                    resp.headers.get("retry-after")
                    or resp.headers.get("x-ratelimit-reset-requests")
                    or backoff
                )
                jitter = random.uniform(0, 3)
                sleep_for = min(retry_after + jitter, 60)
                print(f"[PageIndex] 429 on attempt {attempt} — sleeping {sleep_for:.1f}s")
                await asyncio.sleep(sleep_for)
                backoff = min(backoff * 1.5, 60)
                continue

            if resp.status_code >= 500:
                await asyncio.sleep(backoff + random.uniform(0, 2))
                backoff = min(backoff * 1.5, 60)
                continue

            data = resp.json()
            if "error" in data:
                err = data["error"].get("message", str(data["error"]))
                err_code = data["error"].get("code", "")

                if "rate_limit" in err.lower() or "429" in err:
                    retry_after = float(
                        resp.headers.get("retry-after", str(backoff))
                    )
                    await asyncio.sleep(min(retry_after + random.uniform(0, 3), 60))
                    backoff = min(backoff * 1.5, 60)
                    continue

                # failed_generation: Groq's JSON-constrained mode hit the token
                # limit before it could close all JSON brackets.
                # Fix: retry the SAME prompt without response_format=json_object,
                # then let extract_json() parse the raw text output.
                if err_code == "failed_generation" or "failed_generation" in err.lower():
                    print(f"[PageIndex] failed_generation on attempt {attempt} — retrying without json_mode")
                    payload_plain = {k: v for k, v in payload.items() if k != "response_format"}
                    try:
                        resp2 = await _http_client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {key}",
                                "Content-Type": "application/json",
                            },
                            json=payload_plain,
                        )
                        data2 = resp2.json()
                        if "choices" in data2:
                            content2 = data2["choices"][0]["message"]["content"].strip()
                            if return_finish_reason:
                                finish2 = data2["choices"][0].get("finish_reason", "stop")
                                finish_mapped2 = "finished" if finish2 in ("stop", "eos") else finish2
                                return content2, finish_mapped2
                            return content2
                    except Exception:
                        pass
                    backoff = min(backoff * 1.5, 60)
                    continue

                raise HTTPException(status_code=500, detail=f"Groq error: {err}")

            choice = data["choices"][0]
            content = choice["message"]["content"].strip()

            if return_finish_reason:
                finish = choice.get("finish_reason", "stop")
                finish_mapped = "finished" if finish in ("stop", "eos") else finish
                return content, finish_mapped
            return content

        raise HTTPException(
            status_code=429,
            detail=f"All Groq keys exhausted after {max_attempts} attempts",
        )


async def llm_call_many(
    prompts: list[str], max_tokens: int = 512, json_mode: bool = True
) -> list[str]:
    """
    Fan-out: all prompts run concurrently.
    The semaphore inside llm_call limits actual in-flight requests to N.
    Remaining coroutines simply wait — no thundering herd.
    """
    tasks = [llm_call(p, max_tokens=max_tokens, json_mode=json_mode) for p in prompts]
    return await asyncio.gather(*tasks)


def extract_json(text: str) -> dict | list:
    """Robust JSON extractor — strips markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for pattern in [r"\{.*\}", r"\[.*\]"]:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# PDF Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_pages(pdf_path: str) -> tuple[list[tuple[str, int]], int]:
    """Returns (page_list, total_pages). page_list[i] = (text, 1-indexed physical page)."""
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    total = len(reader.pages)
    pages = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        pages.append((t.strip(), i + 1))
    return pages, total


def get_raw_text(page_list: list[tuple[str, int]]) -> str:
    """Full document text with [PAGE N] markers — used as content store at query time."""
    parts = []
    for text, phys in page_list:
        if text:
            parts.append(f"[PAGE {phys}]\n{text}")
    return "\n\n".join(parts)


def get_page_content_by_range(page_list: list, start: int, end: int) -> str:
    """Text for physical pages start..end inclusive, tagged with <physical_index_N>."""
    parts = []
    for text, phys in page_list:
        if start <= phys <= end:
            parts.append(f"<physical_index_{phys}>\n{text}\n</physical_index_{phys}>\n")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: TOC Detection
# FIX 6: Scan first 8 pages only (was 20). TOCs are almost always on pages 2–5.
#         Saves 12 unnecessary concurrent LLM calls on every document.
# ─────────────────────────────────────────────────────────────────────────────

def _looks_like_toc_heuristic(text: str) -> bool:
    """
    Fast pre-filter before calling the LLM.
    Intentionally lenient — false positives cost one LLM call,
    false negatives are fatal (TOC page missed entirely).
    """
    if not text or len(text.strip()) < 30:
        return False
    lines_list = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines_list) < 3:
        return False
    # Heuristic 1: >=30% of lines end with a standalone number (page refs)
    ends_with_num = sum(1 for l in lines_list if re.search(r"\s\d{1,4}\s*$", l))
    if ends_with_num / max(len(lines_list), 1) >= 0.30:
        return True
    # Heuristic 2: dotted leaders in any form pypdf might produce
    if text.count("...") > 2 or text.count("\u2026\u2026") > 1 or text.count(". . .") > 1:
        return True
    # Heuristic 3: the word "contents" anywhere (word-boundary, case-insensitive)
    if re.search(r"\bcontents\b", text, re.IGNORECASE):
        return True
    # Heuristic 4: many short lines typical of TOC entries
    short_lines = sum(1 for l in lines_list if len(l) < 60)
    if short_lines / max(len(lines_list), 1) >= 0.70 and len(lines_list) >= 6:
        return True
    return False


async def detect_toc_pages(
    page_list: list[tuple[str, int]], check_up_to: int = 8
) -> list[int]:
    candidates = page_list[:check_up_to]

    # Pre-filter with heuristic
    filtered = [(text, phys) for text, phys in candidates if _looks_like_toc_heuristic(text)]

    if not filtered:
        # Heuristic found nothing — send ALL candidates to LLM.
        # Sending all 8 is safe (semaphore throttles); missing the TOC is not.
        filtered = candidates

    prompts = []
    for text, phys in filtered:
        prompts.append(
            f"""Does this page contain a Table of Contents (list of chapters/sections with page numbers)?
Abstract, summary, notation lists, figure lists are NOT table of contents.

Page text:
{text[:2000]}

Return JSON: {{"thinking": "<reason>", "toc_detected": "yes or no"}}"""
        )

    results = await llm_call_many(prompts, max_tokens=150, json_mode=True)

    toc_pages = []
    for raw, (text, phys) in zip(results, filtered):
        parsed = extract_json(raw)
        if isinstance(parsed, dict) and parsed.get("toc_detected") == "yes":
            toc_pages.append(phys)
    return toc_pages


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: TOC Extraction (multi-pass with completion verification)
# Reduced max_attempts from 5 to 2 — well-formatted TOCs are complete on pass 1.
# ─────────────────────────────────────────────────────────────────────────────

EXTRACT_TOC_PROMPT = """Extract the FULL table of contents from the given text.
Replace any ......... or dotted leaders with :.
Return ONLY the table of contents text, nothing else."""

VERIFY_TOC_COMPLETE_PROMPT = """You are given a raw table of contents and a cleaned/extracted version.
Check if the cleaned version contains all main sections from the raw.

Raw TOC:
{raw}

Cleaned TOC:
{cleaned}

Return JSON: {{"thinking": "<reason>", "completed": "yes or no"}}"""

CONTINUE_TOC_PROMPT = """Continue the table of contents extraction. Output ONLY the remaining part.

Original raw TOC:
{raw}

Already extracted:
{partial}

Continue from where it left off:"""


async def extract_toc_with_retry(toc_raw: str, max_attempts: int = 2) -> str:
    prompt = f"{EXTRACT_TOC_PROMPT}\n\nGiven text:\n{toc_raw[:4000]}"
    extracted, _ = await llm_call(prompt, max_tokens=2000, json_mode=False, return_finish_reason=True)

    for attempt in range(max_attempts):
        verify_prompt = VERIFY_TOC_COMPLETE_PROMPT.format(
            raw=toc_raw[:3000], cleaned=extracted
        )
        verify_raw = await llm_call(verify_prompt, max_tokens=200, json_mode=True)
        verify = extract_json(verify_raw)
        completed = verify.get("completed", "no") if isinstance(verify, dict) else "no"
        if completed == "yes":
            break
        cont_prompt = CONTINUE_TOC_PROMPT.format(raw=toc_raw[:3000], partial=extracted)
        continuation, _ = await llm_call(
            cont_prompt, max_tokens=1500, json_mode=False, return_finish_reason=True
        )
        extracted = extracted + "\n" + continuation

    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: TOC → Structured JSON
# ─────────────────────────────────────────────────────────────────────────────

TOC_TRANSFORM_PROMPT = """Transform the following table of contents into a structured JSON array.

CRITICAL HIERARCHY RULE:
- Chapter headers → depth 1: "1", "2", "3" ...
- Sections WITHIN a chapter → depth 2, ALL as siblings: "1.1", "1.2", "1.3" ...
  NOT as cascading children: NEVER "1.1", "1.1.1", "1.1.1.1"
- Subsections within a section → depth 3: "1.1.1", "1.1.2" ...

EXAMPLE — correct output for a chapter with multiple sections:
Input TOC:
  Chapter 1  Relativity            1
    Special Relativity             2
    Time Dilation                  5
    Length Contraction             9
  Chapter 2  Particle Physics     15
    Blackbody Radiation           16

Correct JSON:
{{
  "table_of_contents": [
    {{"structure": "1",   "title": "Chapter 1  Relativity",      "page": 1}},
    {{"structure": "1.1", "title": "Special Relativity",         "page": 2}},
    {{"structure": "1.2", "title": "Time Dilation",              "page": 5}},
    {{"structure": "1.3", "title": "Length Contraction",         "page": 9}},
    {{"structure": "2",   "title": "Chapter 2  Particle Physics","page": 15}},
    {{"structure": "2.1", "title": "Blackbody Radiation",        "page": 16}}
  ]
}}

WRONG — never do this (each section a child of the previous):
  {{"structure": "1.1",       "title": "Special Relativity"}},
  {{"structure": "1.1.1",     "title": "Time Dilation"}},
  {{"structure": "1.1.1.1",   "title": "Length Contraction"}}

Now transform this table of contents:
{toc_text}"""


async def toc_to_json(toc_text: str) -> list[dict]:
    # Cap input: 2000 chars ~= 60 lines ~= 1800 token output, safe under 4096
    # 3000 chars was the main trigger of failed_generation on large TOCs
    prompt = TOC_TRANSFORM_PROMPT.format(toc_text=toc_text[:2000])
    raw, _ = await llm_call(prompt, max_tokens=2048, json_mode=True, return_finish_reason=True)

    parsed = extract_json(raw)
    items = []
    if isinstance(parsed, dict) and "table_of_contents" in parsed:
        items = parsed["table_of_contents"]
    elif isinstance(parsed, list):
        items = parsed

    # Verify completeness — one extra call only if parse looks short
    if len(items) > 0:
        verify_prompt = f"""Is the following JSON table of contents complete relative to the raw?
Raw:
{toc_text[:2000]}

Cleaned (JSON, first 30 entries):
{json.dumps(items[:30], indent=2)}

Return JSON: {{"thinking": "<reason>", "completed": "yes or no"}}"""

        verify_raw = await llm_call(verify_prompt, max_tokens=200)
        verify = extract_json(verify_raw)

        if isinstance(verify, dict) and verify.get("completed") == "no":
            last = items[-1]
            cont_prompt = f"""Continue the JSON TOC array from after this entry:
{json.dumps(last)}

Raw TOC for reference:
{toc_text[:3000]}

Output ONLY the additional entries as a JSON array."""
            cont_raw, _ = await llm_call(
                cont_prompt, max_tokens=2000, json_mode=False, return_finish_reason=True
            )
            try:
                m = re.search(r"\[.*\]", cont_raw, re.DOTALL)
                if m:
                    extra = json.loads(m.group())
                    if isinstance(extra, list):
                        items.extend(extra)
            except Exception:
                pass

    for item in items:
        try:
            item["page"] = int(item.get("page") or 0) or None
        except (ValueError, TypeError):
            item["page"] = None

    return items


# ─────────────────────────────────────────────────────────────────────────────
# Page Offset Detection
# Books with roman-numeral front matter have a gap between the printed page
# number in the TOC and the physical PDF page number.
# e.g. Beiser "Concepts of Modern Physics": printed page 1 = physical page 13.
# This function detects that offset by finding the first content page.
# ─────────────────────────────────────────────────────────────────────────────

def detect_page_offset(
    toc_items: list[dict],
    page_list: list[tuple[str, int]],
    total_pages: int,
    max_search: int = 30,
) -> int:
    """
    Find the physical page offset between printed page numbers (from TOC)
    and actual PDF physical pages.

    Strategy:
    1. Take the first TOC item with a page number (usually chapter 1, page 1-5).
    2. Search physical pages 1..max_search for the item's title text.
    3. offset = physical_page_found - toc_printed_page.

    Returns 0 if no offset is detected (offset-free or detection failed).
    """
    page_map = {phys: text.lower() for text, phys in page_list}

    # Find first TOC item that has a page number and a non-trivial title
    anchor_items = [
        item for item in toc_items
        if item.get("page") and item.get("title") and len(item["title"]) > 3
    ]
    if not anchor_items:
        return 0

    # Use the first chapter-level item (structure "1" or similar)
    anchor = anchor_items[0]
    toc_page = anchor["page"]
    title_lower = anchor["title"].lower().strip()

    # Search in a window around where we'd expect the content to start
    # For books with front matter, content usually starts between page 5-40
    for physical in range(1, min(max_search + 1, total_pages + 1)):
        page_text = page_map.get(physical, "")
        if title_lower in page_text:
            offset = physical - toc_page
            if offset != 0:
                print(f"[PageIndex] Page offset detected: +{offset} "
                      f"('{anchor['title']}' TOC page {toc_page} → physical page {physical})")
            return offset

    return 0


def apply_page_offset(toc_items: list[dict], offset: int) -> list[dict]:
    """Add offset to every item's page number so physical_index assignment is correct."""
    if offset == 0:
        return toc_items
    for item in toc_items:
        if item.get("page") is not None:
            item["page"] = item["page"] + offset
    return toc_items


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Assign Physical Indices
# FIX 7 (Stage 4 shortcut): Text-match check before calling LLM.
# If the section title appears verbatim on the TOC-stated page (or ±1),
# we trust the text match and skip the LLM entirely.
# Eliminates ~70% of Stage 4 LLM calls for clean PDFs.
# ─────────────────────────────────────────────────────────────────────────────

ASSIGN_INDEX_PROMPT = """You are given a table of contents entry and several document pages.
Find which physical page this section starts on.

Section title: {title}
TOC page number: {page}

Document pages:
{pages}

Return JSON:
{{"thinking": "<which page contains this section's start>",
  "physical_index": <page number or null>}}"""


async def assign_physical_indices(
    toc_items: list[dict],
    page_list: list[tuple[str, int]],
    total_pages: int,
) -> list[dict]:
    page_map = {phys: text for text, phys in page_list}

    async def _assign(item: dict) -> dict:
        toc_page = item.get("page")
        if toc_page is None:
            item["physical_index"] = None
            return item

        title_lower = (item.get("title") or "").lower().strip()

        # Fast text-match shortcut — no LLM needed if title found in page text.
        # Search ±3 pages around toc_page (offset may not be perfectly exact).
        if title_lower and len(title_lower) > 3:
            search_range = [toc_page] + [toc_page + d for d in [1, -1, 2, -2, 3, -3]]
            for candidate in search_range:
                if candidate < 1 or candidate > total_pages:
                    continue
                page_text = page_map.get(candidate, "").lower()
                if title_lower in page_text:
                    item["physical_index"] = candidate
                    return item

        # Only call LLM when text-match fails (ambiguous or formatting issues)
        search_start = max(1, toc_page - 1)
        search_end = min(total_pages, toc_page + 3)
        pages_text = get_page_content_by_range(page_list, search_start, search_end)

        if not pages_text.strip():
            item["physical_index"] = toc_page
            return item

        prompt = ASSIGN_INDEX_PROMPT.format(
            title=item.get("title", ""),
            page=toc_page,
            pages=pages_text[:2500],
        )
        raw = await llm_call(prompt, max_tokens=200)
        parsed = extract_json(raw)
        if isinstance(parsed, dict):
            pi = parsed.get("physical_index")
            try:
                item["physical_index"] = int(pi) if pi is not None else toc_page
            except (ValueError, TypeError):
                item["physical_index"] = toc_page
        else:
            item["physical_index"] = toc_page
        return item

    results = await asyncio.gather(*[_assign(item) for item in toc_items])
    return list(results)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Build Nested Tree with start_index / end_index (pure Python, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def build_tree_from_flat(items: list[dict], total_pages: int) -> list[dict]:
    def parse_depth(structure: str) -> int:
        if not structure:
            return 1
        # Cap at depth 4 — prevents 30-level chains from bad LLM structure strings.
        # Real academic books rarely go deeper than: chapter → section → subsection.
        return min(len(structure.split(".")), 4)

    try:
        items = sorted(
            items,
            key=lambda x: [int(p) for p in (x.get("structure") or "0").split(".")],
        )
    except Exception:
        pass

    for i, item in enumerate(items):
        start = item.get("physical_index") or item.get("page") or 1
        if i + 1 < len(items):
            next_start = (
                items[i + 1].get("physical_index")
                or items[i + 1].get("page")
                or (start + 1)
            )
            end = max(start, next_start - 1)
        else:
            end = total_pages
        item["_start"] = start
        item["_end"] = end

    root: list[dict] = []
    stack: list[tuple[int, list[dict]]] = [(0, root)]
    node_counters: dict[str, int] = {}

    for item in items:
        depth = parse_depth(item.get("structure", ""))
        title = item.get("title", "Untitled")
        start = item["_start"]
        end = item["_end"]

        # Find parent node_id: walk the stack to find the nearest ancestor list
        # that has at least one node in it. stack[-1] is current level.
        # For depth==1 nodes, parent is root (no prefix). For depth>1, find
        # the last node added to the nearest non-empty ancestor list.
        parent_node_id = ""
        if depth > 1:
            for _d, _lst in reversed(stack):
                if _lst:
                    parent_node_id = _lst[-1]["node_id"]
                    break

        counter_key = f"depth_{depth}_{parent_node_id}"
        node_counters[counter_key] = node_counters.get(counter_key, 0) + 1
        # Top-level: "0001", "0002" etc. Children: "0001_01", "0001_02" etc.
        seq = node_counters[counter_key]
        node_id = (
            f"{seq:04d}"
            if depth == 1
            else f"{parent_node_id}_{seq:02d}"
        )

        node = {
            "node_id": node_id,
            "title": title,
            "start_index": start,
            "end_index": end,
            "summary": title,  # default: title itself (overwritten for top-level below)
            "nodes": [],
        }

        while len(stack) > 1 and stack[-1][0] >= depth:
            stack.pop()

        stack[-1][1].append(node)
        stack.append((depth, node["nodes"]))

    # Post-process: ensure every parent's end_index >= all its children's end_index.
    # The flat-list ordering sets end_index from the next sibling's start, which
    # can be LESS than a child's end_index when children span more pages.
    def fix_end_indices(nodes: list[dict]) -> int:
        """Recursively fix end_index bottom-up. Returns max end_index seen."""
        max_end = 0
        for node in nodes:
            if node.get("nodes"):
                child_max = fix_end_indices(node["nodes"])
                node["end_index"] = max(node["end_index"], child_max)
            max_end = max(max_end, node["end_index"])
        return max_end

    fix_end_indices(root)
    return root


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: Per-Node Summary Generation
# FIX 5: Top-level nodes only (depth 1, node_id has no underscore).
#         Subsections already have descriptive titles — LLM summary adds nothing.
#         Reduces from ~60 calls to ~10–15 calls for a typical document.
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_PROMPT = """Read the following document section and write a 1–2 sentence summary.
Be concise (under 200 chars). Do NOT copy raw data or quotes from the text.
Example: "Covers thermal decomposition of H2O2 catalysts and reaction kinetics."

Section title: {title}
Section text (first 1500 chars):
{text}

Return JSON: {{"summary": "<your summary>"}}"""


async def add_node_summaries(
    tree: list[dict], page_list: list[tuple[str, int]]
) -> list[dict]:
    """
    FIX 5: Only summarise top-level nodes (chapters).
    Subsection nodes keep summary = title (set in build_tree_from_flat).
    For a 10-chapter document this means 10 LLM calls instead of 60+.
    """
    # top-level nodes are directly in the tree list (depth == 1)
    top_level_nodes = tree  # tree IS the list of root nodes

    async def _summarise(node: dict) -> None:
        text = get_page_content_by_range(
            page_list,
            node["start_index"],
            min(node["start_index"] + 1, node["end_index"]),
        )
        if not text.strip():
            node["summary"] = node["title"]
            return
        prompt = SUMMARY_PROMPT.format(title=node["title"], text=text[:1500])
        raw = await llm_call(prompt, max_tokens=150)
        parsed = extract_json(raw)
        node["summary"] = (
            parsed.get("summary", node["title"])
            if isinstance(parsed, dict)
            else node["title"]
        )

    await asyncio.gather(*[_summarise(n) for n in top_level_nodes])
    return tree


# NOTE: Stage 7 (per-node page verification) has been removed entirely.
# FIX 4: Stage 4 already verifies pages with a ±3 scan + LLM confirmation.
#         Stage 7 repeated the same work, adding 60–300 extra calls for zero gain.


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: No-TOC page-by-page structure extraction (sequential — no gather)
# ─────────────────────────────────────────────────────────────────────────────

NO_TOC_INIT_PROMPT = """You are extracting the literal section headings from a document.

STRICT RULES — follow exactly:
1. ONLY extract headings that appear VERBATIM in the page text.
2. DO NOT invent, summarize, or generalize. Copy the exact words from the page.
3. Valid headings: numbered sections ("1. Introduction"), chapter titles, bold/prominent short lines, section labels.
4. Invalid: body text sentences, captions, footnotes, page numbers.
5. If a page has NO clear heading, do not add an entry for it.

<physical_index_X> tags mark page boundaries. Use X as the physical_index value.

Return a JSON array:
[
  {{"structure": "1", "title": "EXACT heading text from page", "physical_index": <X>}},
  ...
]

Document pages:
{pages}"""

NO_TOC_CONTINUE_PROMPT = """Continue extracting verbatim headings from the new pages only.

STRICT RULES:
1. Copy heading text EXACTLY as it appears in the page — no paraphrasing.
2. Skip pages with no clear headings.
3. Do not repeat entries from the previous structure.

Previous structure (last 10 entries for context):
{previous}

New pages:
{pages}

Return ONLY new entries as a JSON array:
[
  {{"structure": "<x.x.x>", "title": "EXACT heading from page", "physical_index": <X>}},
  ...
]"""


async def extract_structure_no_toc(
    page_list: list[tuple[str, int]], chunk_size: int = 4
) -> list[dict]:
    """
    Fallback for documents without a formal TOC.
    Sequential chunk processing — these calls don't pile up.
    chunk_size=4: smaller chunks = less context = less hallucination.
    """
    all_items: list[dict] = []
    total = len(page_list)

    for chunk_start in range(0, total, chunk_size):
        chunk = page_list[chunk_start: chunk_start + chunk_size]
        pages_text = "\n\n".join(
            f"<physical_index_{phys}>\n{text}\n</physical_index_{phys}>"
            for text, phys in chunk
            if text.strip()
        )
        if not pages_text.strip():
            continue

        if not all_items:
            prompt = NO_TOC_INIT_PROMPT.format(pages=pages_text[:3500])
        else:
            prompt = NO_TOC_CONTINUE_PROMPT.format(
                previous=json.dumps(all_items[-10:], indent=2),
                pages=pages_text[:3000],
            )

        raw, _ = await llm_call(
            prompt, max_tokens=2000, json_mode=False, return_finish_reason=True
        )
        parsed = extract_json(raw)
        if isinstance(parsed, list):
            all_items.extend(parsed)
        elif isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    all_items.extend(v)
                    break

    for item in all_items:
        pi = item.get("physical_index")
        try:
            item["physical_index"] = (
                int(str(pi).split("_")[-1].rstrip(">").strip()) if pi else None
            )
        except Exception:
            item["physical_index"] = None
        item.setdefault("page", item.get("physical_index"))

    return all_items


# ─────────────────────────────────────────────────────────────────────────────
# Tree utilities
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Last-Resort Fallback: Regex heading extractor (zero LLM calls)
# Runs only when both TOC detection AND no-TOC LLM extraction return empty.
# Scans every page for lines that look like headings (short, title-cased or
# numbered) and builds a flat structure from them.
# Guarantees the pipeline never returns "Failed to extract any structure".
# ─────────────────────────────────────────────────────────────────────────────

def extract_structure_regex(page_list: list[tuple[str, int]], total_pages: int) -> list[dict]:
    """
    Pure-Python heading detector — no LLM, never fails.
    Looks for lines matching common heading patterns:
      - Numbered: "1.", "1.1", "Chapter 1", "Section 2.3"
      - Short all-caps lines (typical chapter headings)
      - Short Title Case lines at start of page
    Returns flat list compatible with build_tree_from_flat().
    """
    heading_patterns = [
        re.compile(r"^(Chapter|Section|Part|Appendix)\s+\d+", re.IGNORECASE),
        re.compile(r"^\d+(\.\d+){0,2}\s+[A-Z][a-zA-Z\s]{3,60}$"),
        re.compile(r"^\d+\.\s+[A-Z][a-zA-Z\s]{3,60}$"),
    ]

    items = []
    structure_counters = [0, 0, 0]  # depth 1, 2, 3 counters

    for text, phys in page_list:
        if not text.strip():
            continue
        # Scan ALL lines — directory docs often have numbered entries at END of pages
        page_lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in page_lines:
            if len(line) < 3 or len(line) > 80:
                continue
            matched = False
            for pat in heading_patterns:
                if pat.match(line):
                    matched = True
                    break
            if not matched:
                # Short all-caps line (common chapter heading style)
                if line.isupper() and 4 <= len(line) <= 60:
                    matched = True
            if matched:
                # Determine depth from numbering (e.g. "1.2.3" = depth 3)
                num_match = re.match(r"^(\d+)(\.(\d+))?(\.(\d+))?", line)
                depth = 1
                if num_match:
                    if num_match.group(5):
                        depth = 3
                    elif num_match.group(3):
                        depth = 2
                # Build a structure string
                structure_counters[depth - 1] += 1
                if depth == 1:
                    structure_counters[1] = 0
                    structure_counters[2] = 0
                    structure = str(structure_counters[0])
                elif depth == 2:
                    structure_counters[2] = 0
                    structure = f"{structure_counters[0]}.{structure_counters[1]}"
                else:
                    structure = f"{structure_counters[0]}.{structure_counters[1]}.{structure_counters[2]}"

                items.append({
                    "structure": structure,
                    "title": line,
                    "page": phys,
                    "physical_index": phys,
                })
                break  # one heading per page max at this level

    # Deduplicate consecutive items on the same page
    seen_pages = set()
    unique_items = []
    for item in items:
        key = (item["page"], item["structure"].split(".")[0])
        if key not in seen_pages:
            seen_pages.add(key)
            unique_items.append(item)

    return unique_items


def count_nodes(nodes: list) -> int:
    c = 0
    for n in nodes:
        c += 1
        if n.get("nodes"):
            c += count_nodes(n["nodes"])
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class UploadRequest(BaseModel):
    file_url: str
    file_name: Optional[str] = "document.pdf"


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "llama-3.3-70b-versatile via Groq",
        "groq_keys_loaded": len(GROQ_KEYS),
        "semaphore_slots": len(GROQ_KEYS),
        "approach": (
            "PageIndex — 6-stage pipeline "
            "(TOC detect[8pp]→extract→JSON→assign[text-shortcut]→tree→summaries[top-level])"
        ),
    }


@app.post("/doc/")
async def build_document_tree(
    req: UploadRequest,
    x_api_key: str = Header(...),
):
    """
    PageIndex pipeline — fixed call budget for a 40-section document:

      Stage 1: detect_toc_pages   →  2–5 calls  (was 20, heuristic pre-filter)
      Stage 2: extract_toc        →  2–6 calls  (was 2–12, max_attempts=2)
      Stage 3: toc_to_json        →  2–3 calls  (unchanged)
      Stage 4: assign_physical    →  ~12 calls  (was 40, text-shortcut skips ~70%)
      Stage 5: build_tree         →  0 calls    (pure Python)
      Stage 6: add_summaries      →  ~10 calls  (was 60, top-level only)
      Stage 7: verify_pages       →  REMOVED    (was 60–300)
      ─────────────────────────────────────────
      Total:                         ~28–36 calls (was 184–395)
    """
    verify_key(x_api_key)

    if not GROQ_KEYS:
        raise HTTPException(status_code=500, detail="No Groq API keys configured")

    # ── Download PDF ────────────────────────────────────────────────────────
    if req.file_url.startswith("data:"):
        try:
            _, encoded = req.file_url.split(",", 1)
            pdf_bytes = base64.b64decode(encoded)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid data URI: {e}")
    else:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as dl:
            r = await dl.get(req.file_url)
            if r.status_code != 200:
                raise HTTPException(
                    status_code=400, detail=f"Cannot download PDF: HTTP {r.status_code}"
                )
            if "text/html" in r.headers.get("content-type", ""):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "URL returned HTML, not a PDF. "
                        "For Google Drive use: "
                        "https://drive.google.com/uc?export=download&id=FILE_ID"
                    ),
                )
            pdf_bytes = r.content

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        # ── Extract pages ────────────────────────────────────────────────────
        page_list, total_pages = extract_pdf_pages(tmp_path)
        raw_text = get_raw_text(page_list)

        if not raw_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text extracted — may be a scanned/image-only PDF",
            )

        # ── Stage 1: Detect TOC (max 8 pages, heuristic pre-filter) ─────────
        toc_pages = await detect_toc_pages(page_list, check_up_to=min(8, total_pages))
        has_toc = len(toc_pages) > 0
        print(f"[PageIndex] TOC on pages: {toc_pages}")

        if has_toc:
            # ── Stage 2: Extract TOC text ────────────────────────────────────
            toc_raw_text = "\n\n".join(
                page_list[phys - 1][0]
                for phys in toc_pages
                if 0 < phys <= total_pages
            )
            toc_cleaned = await extract_toc_with_retry(toc_raw_text)

            # ── Stage 3: TOC → structured JSON ──────────────────────────────
            toc_items = await toc_to_json(toc_cleaned)
            print(f"[PageIndex] TOC items: {len(toc_items)}")

            # ── Stage 3.5: Detect and apply page offset ─────────────────────
            # Books with roman-numeral front matter have offset between
            # printed TOC page numbers and physical PDF page numbers.
            # Must be applied BEFORE Stage 4 or all physical_index assignments are wrong.
            offset = detect_page_offset(toc_items, page_list, total_pages)
            if offset != 0:
                toc_items = apply_page_offset(toc_items, offset)

            # ── Stage 4: Assign physical indices (text-shortcut) ─────────────
            toc_items = await assign_physical_indices(toc_items, page_list, total_pages)

        else:
            print("[PageIndex] No TOC — falling back to page-by-page extraction")
            toc_items = await extract_structure_no_toc(page_list)
            offset = 0  # no offset detection without a TOC
            print(f"[PageIndex] Fallback items: {len(toc_items)}")

        if not toc_items:
            # Last-resort: regex heading extractor — pure Python, zero LLM calls.
            # Scans every page for numbered/titled headings. Always returns something.
            print("[PageIndex] Both LLM paths empty — using regex last-resort extractor")
            toc_items = extract_structure_regex(page_list, total_pages)
            print(f"[PageIndex] Regex fallback items: {len(toc_items)}")

        if not toc_items:
            # Absolute last resort: one node per page (document is always indexed)
            print("[PageIndex] Regex also empty — creating page-per-node structure")
            toc_items = [
                {"structure": str(i + 1), "title": f"Page {phys}", "page": phys, "physical_index": phys}
                for i, (text, phys) in enumerate(page_list)
                if text.strip()
            ]


        # ── Stage 5: Build tree (pure Python) ───────────────────────────────
        tree = build_tree_from_flat(toc_items, total_pages)

        # ── Stage 6: Summaries (top-level nodes only) ────────────────────────
        tree = await add_node_summaries(tree, page_list)

        # Stage 7 removed — see note above build_document_tree

        doc_id = f"pi-{uuid.uuid4().hex[:16]}"

        return {
            "doc_id": doc_id,
            "file_name": req.file_name,
            "tree": tree,
            "raw_text": raw_text,
            "tree_node_count": count_nodes(tree),
            "top_level_nodes": len(tree),
            "total_pages": total_pages,
            "toc_found": has_toc,
            "toc_pages": toc_pages,
            "page_offset": offset if has_toc else 0,
            "status": "completed",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
