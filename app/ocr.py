"""OCR for scanned (image-only) PDF pages via Gemini vision.

Books with no text layer (scanned image PDFs) yield nothing from pymupdf, so the
ingester routes their pages here: each page is rendered to an image and transcribed
by a vision model. Concurrency is capped low (vision requests are heavy and throttle
above ~8) with retry-on-timeout for the occasional straggler. Cost is ~$0.0003/page
(gemini-2.5-flash-lite); a 250-page book is ~$0.08 and ~15-20 min.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os

from .config import settings


def _cache_path(pdf_path: str) -> str:
    """Per-PDF OCR cache file, keyed by file content hash (stable across re-uploads)."""
    h = hashlib.sha1(open(pdf_path, "rb").read()).hexdigest()[:16]
    d = os.path.join(settings.lightrag_data_dir, ".ocr_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{h}.json")


def load_ocr_cache(pdf_path: str) -> dict[int, str]:
    """{page_no: text} previously OCR'd for this PDF (empty if none)."""
    p = _cache_path(pdf_path)
    if os.path.exists(p):
        try:
            return {int(k): v for k, v in json.load(open(p)).items()}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_ocr_cache(pdf_path: str, results: dict[int, str]) -> None:
    """Merge newly-OCR'd pages into the cache so a later re-ingest skips re-OCR."""
    merged = load_ocr_cache(pdf_path)
    merged.update({k: v for k, v in results.items() if v})
    try:
        json.dump({str(k): v for k, v in merged.items()}, open(_cache_path(pdf_path), "w"))
    except Exception:  # noqa: BLE001
        pass

GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_PROMPT = (
    "Transcribe ALL text on this page exactly, in natural reading order. Handle "
    "multiple columns, sidebars, headers and tables. Output only the transcribed "
    "text — no commentary, no markdown fences."
)


async def ocr_pages(pdf_path: str, page_numbers: list[int], progress_cb=None) -> dict[int, str]:
    """OCR the given pages of a PDF concurrently. Returns {page_number: text}; a page
    that fails after retries maps to "" (skipped, not fatal). progress_cb(done, total)
    is called per page so callers can surface OCR progress."""
    import fitz  # pymupdf
    from openai import AsyncOpenAI

    cache = load_ocr_cache(pdf_path)
    todo = [p for p in page_numbers if not cache.get(p)]
    reused = len(page_numbers) - len(todo)
    if reused:
        print(f"   OCR cache: reusing {reused} page(s); OCR'ing {len(todo)} new.", flush=True)
    if not todo:  # fully recovered from cache — no API calls
        return {p: cache.get(p, "") for p in page_numbers}

    client = AsyncOpenAI(base_url=GEMINI_OPENAI_URL, api_key=settings.gemini_api_key)
    doc = fitz.open(pdf_path)
    sem = asyncio.Semaphore(settings.ocr_concurrency)
    results: dict[int, str] = {}
    done = {"n": 0}
    total = len(todo)

    async def _one(pno: int) -> None:
        async with sem:
            pix = doc[pno].get_pixmap(matrix=fitz.Matrix(settings.ocr_zoom, settings.ocr_zoom))
            b64 = base64.b64encode(pix.tobytes("png")).decode()
            for attempt in range(settings.ocr_max_retries + 1):
                try:
                    coro = client.chat.completions.create(
                        model=settings.ocr_model, temperature=0,
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": _PROMPT},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{b64}"}}]}])
                    r = await asyncio.wait_for(coro, settings.ocr_per_page_timeout)
                    results[pno] = r.choices[0].message.content or ""
                    break
                except Exception:  # noqa: BLE001 — timeout/transient: retry, then give up on this page
                    if attempt == settings.ocr_max_retries:
                        results[pno] = ""
            done["n"] += 1
            if progress_cb:
                progress_cb(done["n"], total)
            if done["n"] % 25 == 0 or done["n"] == total:
                print(f"   OCR: {done['n']}/{total} pages…", flush=True)

    try:
        await asyncio.gather(*(_one(p) for p in todo))
    finally:
        doc.close()
    save_ocr_cache(pdf_path, results)  # persist so a failed KG run never re-OCRs
    failed = sum(1 for p in todo if not results.get(p))
    if failed:
        print(f"   ⚠️  OCR: {failed}/{total} page(s) failed after retries (skipped).", flush=True)
    return {p: (cache.get(p) or results.get(p, "")) for p in page_numbers}
