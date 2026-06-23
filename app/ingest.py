"""Ingest rulebook PDFs into a per-game index (one index per game).

The --game slug becomes the model name you select in PUM (e.g. "forbidden-lands").

LightRAG backend (default):
    python -m app.ingest --game forbidden-lands books/ForbiddenLands/
    python -m app.ingest --list
    python -m app.ingest --delete forbidden-lands
Requires OPENAI_API_KEY (embeddings) + GEMINI_API_KEY (KG extraction) in .env.

Gemini File Search backend (BACKEND=gemini_filesearch): same commands; builds a
managed File Search store instead. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import os
import re
import shutil
import sys
import time

from .config import settings
from .transform import slugify

# Tuned ingest concurrency (validated: ~1.7x faster, no quality loss).
_LIGHTRAG_INGEST_KW = dict(
    llm_model_max_async=32, embedding_func_max_async=16, embedding_batch_num=100,
)


class _Progress:
    """Writes ingest progress to a JSON file (atomically) that the UI polls. This is
    what lets the UI run the ingest *detached* — no live stdout pipe to stall or crash."""

    def __init__(self, path: str | None = None):
        self.path = path
        self.state: dict = {"phase": "starting"}

    def update(self, **kw) -> None:
        self.state.update(kw)
        if not self.path:
            return
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.state, f)
            os.replace(tmp, self.path)  # atomic: UI never reads a half-written file
        except Exception:  # noqa: BLE001
            pass


_PROGRESS = _Progress()  # module singleton; path set from --progress-file


class _ChunkProgressHandler(logging.Handler):
    """Turns LightRAG's chunk/phase log lines into progress-file updates."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if m := re.search(r"Chunk \d+ of (\d+)", msg):
            _PROGRESS.state["chunk_done"] = _PROGRESS.state.get("chunk_done", 0) + 1
            _PROGRESS.update(phase="extract", chunk_total=int(m.group(1)))
        elif "Phase 2" in msg or "Phase 3" in msg or "merging" in msg.lower():
            _PROGRESS.update(phase="merge")
        elif "embedding" in msg and "vectors" in msg:
            _PROGRESS.update(phase="embed")


def _expand_pdfs(paths: list[str]) -> list[str]:
    """Accept files, globs, or directories (recursing for *.pdf)."""
    out: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "**", "*.pdf"), recursive=True))
        elif any(ch in p for ch in "*?["):
            out += sorted(glob.glob(p, recursive=True))
        else:
            out.append(p)
    seen, uniq = set(), []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def _fmt(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def _extract_books(pdfs: list[str]) -> list[tuple[str, str, list]]:
    """Per-book page text as (name, path, [(page_no, text), ...]). Image-only pages
    (no text layer) come back as text="" and are OCR'd later in _resolve_books."""
    import fitz  # pymupdf
    books = []
    for path in pdfs:
        doc = fitz.open(path)
        name = os.path.basename(path)
        pages = []
        for i, page in enumerate(doc):
            t = page.get_text()
            pages.append((i, t if t.strip() else ""))
        doc.close()
        books.append((name, path, pages))
    return books


async def _resolve_books(books: list[tuple[str, str, list]]) -> list[tuple[str, str]]:
    """Turn per-page extraction into (name, text), OCRing any image-only pages first
    (with a loud cost/time warning). Only the scanned pages are OCR'd, so a mostly-text
    PDF only pays for its image pages."""
    resolved = []
    for name, path, pages in books:
        need = [pno for pno, t in pages if not t]
        if need and len(need) / max(len(pages), 1) < settings.ocr_min_scanned_fraction:
            # Just a few image pages in a mostly-text book — title/art/dividers, no useful
            # text. Skip OCR entirely; ingest the text pages (those image pages drop out).
            print(f"ℹ️  {name}: {len(need)} of {len(pages)} page(s) are image-only (likely "
                  f"title/art) — skipping OCR, ingesting the text.", flush=True)
            need = []
        if need:
            from .ocr import load_ocr_cache, ocr_pages
            cache = load_ocr_cache(path)
            uncached = [p for p in need if not cache.get(p)]
            if not uncached:  # fully recoverable from a prior OCR run — no API, no wait
                print(f"✓  {name}: {len(need)} scanned page(s) recovered from OCR cache — "
                      f"skipping OCR, going straight to KG.", flush=True)
                ocr_text = await ocr_pages(path, need)  # returns cached text, no API calls
                pages = [(pno, (ocr_text.get(pno, "") if not t else t)) for pno, t in pages]
            elif not settings.ocr_scanned:
                print(f"⚠️  {name}: {len(uncached)}/{len(pages)} page(s) have no text layer "
                      f"(scanned image) — SKIPPING those pages. Set OCR_SCANNED=true to OCR them.",
                      flush=True)
            elif not settings.gemini_api_key:
                print(f"⚠️  {name}: {len(uncached)} scanned page(s) but GEMINI_API_KEY is unset — "
                      f"can't OCR; skipping those pages.", flush=True)
            else:
                mins = max(1, round(len(uncached) * 4.5 / 60))
                cached_note = (f" ({len(need) - len(uncached)} already cached)"
                               if len(uncached) < len(need) else "")
                print(f"⚠️  {name}: {len(uncached)} scanned page(s){cached_note}, no text layer. OCR via "
                      f"{settings.ocr_model} — est ~{mins} min, ~${len(uncached) * 0.0003:.2f} "
                      f"(concurrency {settings.ocr_concurrency}). Disable with OCR_SCANNED=false.",
                      flush=True)
                _PROGRESS.update(phase="ocr", book_name=name, ocr_done=0, ocr_total=len(uncached))
                ocr_text = await ocr_pages(
                    path, need,
                    progress_cb=lambda d, tot: _PROGRESS.update(phase="ocr", ocr_done=d, ocr_total=tot))
                pages = [(pno, (ocr_text.get(pno, "") if not t else t)) for pno, t in pages]
        text = "\n\n".join(f"[{name} p{pno + 1}]\n{t}" for pno, t in pages if t.strip())
        if text.strip():
            resolved.append((name, text))
        else:
            print(f"⚠️  {name}: no usable text after extraction/OCR — skipped.", flush=True)
    return resolved


async def _ingest_flow(slug: str, books: list[tuple[str, str, list]]) -> None:
    resolved = await _resolve_books(books)
    if not resolved:
        print("No ingestable text found in the provided PDF(s).")
        _PROGRESS.update(phase="error",
                         message="No ingestable text (PDFs empty, or scanned with OCR off).")
        return
    total_chars = sum(len(t) for _, t in resolved)
    print(f"  {total_chars:,} chars across {len(resolved)} book(s). Building knowledge graph "
          f"for '{slug}' (LLM entity extraction per chunk — progress below).")
    await _lightrag_ingest(slug, resolved)


def _setup_ingest_logging() -> None:
    """Show LightRAG's own progress logs (entity/relation phases, chunk counts)
    while silencing HTTP-client noise. Robust across terminals and log files."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "nano-vectordb",
                  "lightrag.kg.shared_storage"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("lightrag").setLevel(logging.INFO)


# --- LightRAG ---------------------------------------------------------------

def _lightrag_dir(slug: str) -> str:
    return os.path.join(settings.lightrag_data_dir, slug)


async def _clear_stale_for_retry(rag, file_names: set[str]) -> int:
    """Recovery for failed imports: before (re)ingesting, delete any prior NON-processed
    doc (failed/processing/pending) whose filename we're about to ingest, plus any
    duplicate-marker records. Otherwise LightRAG's duplicate-filename guard fails every
    retry with 'File name already exists'. Processed docs are left untouched (LightRAG's
    content-dedup skips them, no re-cost); the LLM extraction cache is preserved so the
    retry is cheap."""
    from lightrag.base import DocStatus
    targets = {os.path.basename(n) for n in file_names}
    cleared = 0
    for status in DocStatus:
        if status == DocStatus.PROCESSED:
            continue
        try:
            docs = await rag.get_docs_by_status(status)
        except Exception:  # noqa: BLE001
            continue
        for doc_id, st in docs.items():
            fp = os.path.basename(getattr(st, "file_path", "") or "")
            summary = getattr(st, "content_summary", "") or ""
            if fp in targets or summary.startswith("[DUPLICATE"):
                try:
                    await rag.adelete_by_doc_id(doc_id)  # keeps LLM cache by default
                    cleared += 1
                except Exception as e:  # noqa: BLE001
                    print(f"  (couldn't clear stale doc {doc_id}: {str(e)[:80]})", flush=True)
    return cleared


async def _lightrag_ingest(slug: str, books: list[tuple[str, str]]) -> None:
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from .lightrag_backend import (
        _ingest_llm_func, make_embed_func, ingest_token_tracker, embed_usage, ingest_skipped)

    embed_provider = settings.lightrag_embedding_provider
    embed_model = settings.embedding_model_effective
    embed_func = make_embed_func(embed_provider, embed_model)
    # Detect the REAL vector size from the model itself (one tiny probe) rather than
    # guessing from the name — so any embedding model works (incl. non-1536/3072
    # OpenRouter ids). Falls back to the configured dim if the probe fails.
    embed_dim = settings.lightrag_embedding_dim
    try:
        embed_dim = len((await embed_func(["dimension probe"]))[0])
        if embed_dim != settings.lightrag_embedding_dim:
            print(f"Detected embedding dimension: {embed_dim} (for {embed_model}).", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Could not probe embedding dim ({str(e)[:80]}); using {embed_dim}.", flush=True)
    wd = _lightrag_dir(slug)
    os.makedirs(wd, exist_ok=True)
    rag = LightRAG(
        working_dir=wd, llm_model_func=_ingest_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embed_dim, max_token_size=8192, func=embed_func),
        **_LIGHTRAG_INGEST_KW)
    await rag.initialize_storages()
    await initialize_pipeline_status()

    logging.getLogger("lightrag").addHandler(_ChunkProgressHandler())  # → progress file

    # Self-heal: clear prior failed/incomplete attempts at these filenames so retries
    # don't hit "File name already exists". This is what makes a failed import recoverable.
    cleared = await _clear_stale_for_retry(rag, {name for name, _ in books})
    if cleared:
        print(f"Recovered {cleared} stale failed/duplicate entr(ies) — these books will re-ingest "
              f"cleanly (LLM cache reused).", flush=True)

    ingest_token_tracker.reset()
    embed_usage["tokens"] = 0
    embed_usage["calls"] = 0
    ingest_skipped["count"] = 0

    def _snap():  # running totals (LLM in/out + embedding tokens + skipped chunks)
        return (ingest_token_tracker.prompt_tokens, ingest_token_tracker.completion_tokens,
                embed_usage["tokens"], ingest_skipped["count"])

    skip_kg = settings.lightrag_skip_kg
    audit = {"slug": slug, "model": settings.lightrag_ingest_llm_model,
             "reasoning": settings.lightrag_ingest_reasoning_effort,
             "embedding_model": embed_model, "embedding_provider": embed_provider,
             "embedding_dim": embed_dim,  # DETECTED per-index dim, for query-time
             "skip_kg": skip_kg, "concurrency": _LIGHTRAG_INGEST_KW, "books": []}
    if skip_kg:
        print("RAG-only mode (skip_kg): no knowledge-graph extraction — embeddings only. "
              "Queries will use vector search (naive mode).", flush=True)
    # Safety: a cost ceiling against a model we can't price (e.g. a custom endpoint, or an
    # OpenRouter model missing from its live pricing) silently never enforces. Say so up
    # front. OpenRouter models we CAN price live don't trip this; nor does the default path.
    if settings.ingest_max_cost_usd > 0 and not skip_kg and not ingest_cost_is_priceable():
        print(f"⚠️  Cost ceiling ${settings.ingest_max_cost_usd:.2f} is set, but '"
              f"{settings.lightrag_ingest_llm_model}' has no price available — the ceiling CANNOT "
              f"enforce and the run will report $0. Watch your provider's dashboard for real spend.",
              flush=True)
    total_t = time.monotonic()
    for i, (name, text) in enumerate(books, 1):
        verb = "embedding chunks" if skip_kg else "extracting knowledge graph"
        print(f"\n[{i}/{len(books)}] {name}  ({len(text):,} chars) — {verb}...", flush=True)
        _PROGRESS.update(phase="extract", book_i=i, book_n=len(books), book_name=name,
                         chunk_done=0, chunk_total=0)
        t = time.monotonic()
        b0 = _snap()
        if skip_kg:
            # process_options='!' = skip entity/relation extraction; chunks still get
            # embedded into the vector store so naive retrieval works.
            track = await rag.apipeline_enqueue_documents(text, file_paths=name, process_options="!")
            await rag.apipeline_process_enqueue_documents()
        else:
            await rag.ainsert(text, file_paths=name)  # LightRAG logs phase progress
        dt = time.monotonic() - t
        b1 = _snap()
        din, dout, demb, dskip = (b1[0] - b0[0], b1[1] - b0[1], b1[2] - b0[2], b1[3] - b0[3])
        skip_note = f" | ⚠️ {dskip} chunk(s) skipped" if dskip else ""
        print(f"   ✓ {name} done in {_fmt(dt)}  "
              f"[LLM in {din:,} / out {dout:,} | embed {demb:,} tok]{skip_note}", flush=True)
        audit["books"].append({"name": name, "chars": len(text), "seconds": round(dt),
                               "llm_input_tokens": din, "llm_output_tokens": dout,
                               "embedding_tokens": demb, "skipped_chunks": dskip})

        # Hard cost ceiling: stop cleanly if the running spend crosses the limit.
        if settings.ingest_max_cost_usd > 0:
            sofar = _cost_breakdown({
                "llm_input": ingest_token_tracker.prompt_tokens,
                "llm_output": ingest_token_tracker.completion_tokens,
                "embedding": embed_usage["tokens"]})["total_usd"]
            if sofar >= settings.ingest_max_cost_usd:
                print(f"\n🛑 Cost ceiling reached: ${sofar:.2f} ≥ "
                      f"${settings.ingest_max_cost_usd:.2f} (INGEST_MAX_COST_USD). Stopping after "
                      f"{i}/{len(books)} book(s); those are kept. Raise the ceiling to continue.",
                      flush=True)
                break

    await rag.finalize_storages()
    total = time.monotonic() - total_t
    audit["total_seconds"] = round(total)
    audit["tokens"] = {
        "llm_input": ingest_token_tracker.prompt_tokens,
        "llm_output": ingest_token_tracker.completion_tokens,
        "embedding": embed_usage["tokens"],
        "embedding_calls": embed_usage["calls"]}
    audit["cost"] = _cost_breakdown(audit["tokens"])
    audit["skipped_chunks"] = ingest_skipped["count"]
    with open(os.path.join(wd, "ingest_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n✅ Ingest complete: {len(books)} book(s) in {_fmt(total)}", flush=True)
    if ingest_skipped["count"]:
        print(f"   ⚠️  {ingest_skipped['count']} chunk(s) skipped on empty/filtered LLM responses "
              f"(likely mature-content safety filtering) — books were kept, not failed.", flush=True)
    _print_cost(audit["tokens"], audit["cost"])
    print(f"   Full audit -> {wd}/ingest_audit.json")
    _PROGRESS.update(phase="done", returncode=0, skipped=ingest_skipped["count"])


# STATIC price estimates, USD per 1M tokens, for the direct providers (Gemini/OpenAI
# have no public pricing API). These DRIFT — verify before relying on them. The date
# is surfaced to the user. Token COUNTS are always exact; only the $ rate is an estimate.
# OpenRouter models are priced LIVE instead (see _openrouter_prices), so they're current.
_PRICES_AS_OF = "2026-06-21"
_PRICES = {
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "text-embedding-3-large": {"input": 0.13},
    "text-embedding-3-small": {"input": 0.02},
    "gemini-embedding-001": {"input": 0.15},  # default embedder — so its cost line isn't $0
}

_OR_PRICE_CACHE: dict = {}  # OpenRouter model id -> {input,output} per 1M
_OR_PRICE_FETCHED = False    # True once we've attempted the fetch (success OR failure)


def _openrouter_prices() -> dict:
    """OpenRouter's LIVE per-model pricing (USD per 1M tokens), fetched at most once per
    process from its public models API. Returns {} on any failure (offline etc.) so callers
    fall back to 'unknown' rather than breaking. The attempt is cached even on failure, so
    an offline run doesn't re-hit the 10s timeout on every book."""
    global _OR_PRICE_FETCHED
    if _OR_PRICE_FETCHED:
        return _OR_PRICE_CACHE
    _OR_PRICE_FETCHED = True
    try:
        import urllib.request
        data = json.load(urllib.request.urlopen(
            "https://openrouter.ai/api/v1/models", timeout=10))
        for m in data.get("data", []):
            p = m.get("pricing") or {}
            try:
                _OR_PRICE_CACHE[m["id"]] = {"input": float(p["prompt"]) * 1e6,
                                            "output": float(p["completion"]) * 1e6}
            except (KeyError, ValueError, TypeError):
                continue
    except Exception:  # noqa: BLE001
        pass
    return _OR_PRICE_CACHE


def _ingest_llm_rate() -> tuple[dict | None, str]:
    """Resolve the ingest LLM's rate (USD/1M) and its source.
    Returns (rate|None, source) where source is 'openrouter-live' | 'static' | 'unknown'."""
    model = settings.lightrag_ingest_llm_model
    if settings.lightrag_ingest_llm_provider == "openrouter":
        live = _openrouter_prices().get(model)
        return (live, "openrouter-live") if live else (None, "unknown")
    if model in _PRICES:
        return _PRICES[model], "static"
    return None, "unknown"


def ingest_cost_is_priceable() -> bool:
    """True if we can actually price the ingest LLM (so the cost ceiling can enforce)."""
    return _ingest_llm_rate()[0] is not None


def _cost_breakdown(tok: dict) -> dict:
    # Price the model ACTUALLY used. LLM: live for OpenRouter, static for direct providers,
    # else unknown ($0 placeholder, ceiling can't enforce). Embeddings: always a direct
    # provider, so static; embedding_model_effective is the real model (gemini-embedding-001
    # on the default path, not the raw OpenAI field).
    llm, llm_src = _ingest_llm_rate()
    emb_model = settings.embedding_model_effective
    emb = _PRICES.get(emb_model)
    llm_r = llm or {"input": 0, "output": 0}
    emb_r = emb or {"input": 0}
    in_cost = tok["llm_input"] / 1e6 * llm_r.get("input", 0)
    out_cost = tok["llm_output"] / 1e6 * llm_r.get("output", 0)
    emb_cost = tok["embedding"] / 1e6 * emb_r.get("input", 0)
    return {"llm_input_usd": round(in_cost, 4), "llm_output_usd": round(out_cost, 4),
            "embedding_usd": round(emb_cost, 4), "total_usd": round(in_cost + out_cost + emb_cost, 4),
            "rates_per_1m": {"llm": llm_r, "embedding": emb_r},
            "rates_known": {"llm": llm is not None, "embedding": emb is not None},
            "llm_rate_source": llm_src}


def _print_cost(tok: dict, cost: dict) -> None:
    src = cost.get("llm_rate_source", "static")
    print(f"   Tokens — KG LLM ({settings.lightrag_ingest_llm_model}): "
          f"{tok['llm_input']:,} in + {tok['llm_output']:,} out | "
          f"embeddings ({settings.embedding_model_effective}): {tok['embedding']:,}")
    print(f"   Est. cost — LLM ${cost['llm_input_usd']:.4f} in + ${cost['llm_output_usd']:.4f} out "
          f"+ embed ${cost['embedding_usd']:.4f}  =  ${cost['total_usd']:.4f}")
    # Prominent provenance: live OpenRouter rates vs dated static estimates.
    if tok["llm_output"] and src == "openrouter-live":
        print(f"   Rates: LLM = live OpenRouter pricing (fetched this run); embeddings = static "
              f"estimate as of {_PRICES_AS_OF}. Verify before relying on costs.", flush=True)
    elif tok["llm_output"] and src == "unknown":
        print(f"   ⚠️  No price for ingest model '{settings.lightrag_ingest_llm_model}' — the LLM "
              f"cost above is a $0 PLACEHOLDER, not an estimate. Check your provider's dashboard.",
              flush=True)
    else:
        print(f"   Rates: static estimates as of {_PRICES_AS_OF} — prices change; verify current "
              f"rates with your provider.", flush=True)


def lightrag_list() -> None:
    d = settings.lightrag_data_dir
    if not os.path.isdir(d):
        print(f"No indexes yet ({d}/).")
        return
    games = [n for n in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, n))]
    print("Installed games (use the slug as the PUM model):")
    for g in games:
        print(f"  {g}")


def lightrag_delete(slug: str) -> None:
    wd = _lightrag_dir(slug)
    if not os.path.isdir(wd):
        sys.exit(f"No index for {slug!r}.")
    shutil.rmtree(wd)
    print(f"Deleted index for {slug}.")


def lightrag_cmd(game: str, pdfs: list[str], cleanup_source: bool = False) -> None:
    # The whole body is wrapped so --cleanup-source deletes the caller's throwaway
    # temp PDFs on EVERY exit path — success, ingest error, or an early sys.exit (e.g.
    # missing key). NOT set for the CLI path, so a user's own book folder is untouched.
    try:
        # Embedding key depends on the embedding provider.
        ep = settings.lightrag_embedding_provider
        if ep == "gemini":
            need, have = "GEMINI_API_KEY", settings.gemini_api_key
        elif ep == "openrouter":
            need, have = "OPENROUTER_API_KEY", settings.openrouter_api_key or settings.lightrag_llm_api_key
        else:
            need, have = "OPENAI_API_KEY", settings.openai_api_key
        if not have:
            _PROGRESS.update(phase="error", message=f"{need} is required (embeddings).")
            sys.exit(f"{need} is required for embeddings (LIGHTRAG_EMBEDDING_PROVIDER="
                     f"{settings.lightrag_embedding_provider}).")
        _setup_ingest_logging()
        slug = slugify(game)
        print(f"Extracting text from {len(pdfs)} PDF(s)...")
        books = _extract_books(pdfs)
        try:
            asyncio.run(_ingest_flow(slug, books))
        except Exception as e:  # noqa: BLE001 — record failure for the UI poller, then re-raise
            _PROGRESS.update(phase="error", message=str(e)[:200])
            raise
        print(f"\nIn PUM, set the model to: {slug}")
    finally:
        if cleanup_source:  # frees the bulk (copied PDFs) regardless of how we exit
            for f in pdfs:
                try:
                    os.remove(f)
                except Exception:  # noqa: BLE001
                    pass


# --- Gemini File Search -----------------------------------------------------

def _fs_client():
    from google import genai
    if not settings.gemini_api_key:
        sys.exit("GEMINI_API_KEY is not set (.env).")
    return genai.Client(api_key=settings.gemini_api_key)


def _fs_find_store(client, slug: str):
    for st in client.file_search_stores.list():
        title = getattr(st, "display_name", "") or ""
        if slugify(title) == slug:
            return st
    return None


def fs_list(client) -> None:
    stores = list(client.file_search_stores.list())
    if not stores:
        print("No File Search stores yet.")
        return
    for st in stores:
        title = getattr(st, "display_name", "") or st.name.split("/")[-1]
        print(f"  {slugify(title):32} {getattr(st, 'active_documents_count', '?')} docs")


def fs_delete(client, slug: str) -> None:
    st = _fs_find_store(client, slug)
    if not st:
        sys.exit(f"No store with slug {slug!r}.")
    client.file_search_stores.delete(name=st.name, config={"force": True})
    print(f"Deleted {slug}.")


def fs_cmd(client, game: str, pdfs: list[str]) -> None:
    from google.genai import types
    slug = slugify(game)
    st = _fs_find_store(client, slug) or client.file_search_stores.create(
        config=types.CreateFileSearchStoreConfig(display_name=slug))
    print(f"Store {slug} -> {st.name}")
    for path in pdfs:
        print(f"  uploading {path} ...", end=" ", flush=True)
        op = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=st.name, file=path,
            config=types.UploadToFileSearchStoreConfig(display_name=os.path.basename(path)))
        deadline = time.monotonic() + 600
        while not op.done and time.monotonic() < deadline:
            time.sleep(3)
            op = client.operations.get(op)
        print("done" if op.done else "TIMEOUT (still indexing)")
    print(f"\nDone. In PUM, set the model to: {slug}")


# --- dispatch ---------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Ingest PDFs into a per-game index.")
    p.add_argument("--game", help="Game slug (becomes the PUM model name).")
    p.add_argument("--list", action="store_true", help="List installed games.")
    p.add_argument("--delete", metavar="SLUG", help="Delete a game's index.")
    p.add_argument("--backend", default=settings.backend,
                   help="lightrag (default) or gemini_filesearch.")
    p.add_argument("--progress-file", metavar="PATH",
                   help="Write structured progress JSON here (used by the admin UI poller).")
    p.add_argument("--cleanup-source", action="store_true",
                   help="Delete the source PDFs when done (admin UI uses this for its "
                        "throwaway temp copies — do NOT use on your own book folder).")
    p.add_argument("pdfs", nargs="*", help="PDF files, globs, or folders.")
    args = p.parse_args()

    if args.progress_file:
        _PROGRESS.path = args.progress_file

    if args.backend == "lightrag":
        if args.list:
            lightrag_list()
        elif args.delete:
            lightrag_delete(slugify(args.delete))
        elif args.game and args.pdfs:
            pdfs = _expand_pdfs(args.pdfs)
            if not pdfs:
                sys.exit("No PDFs found.")
            lightrag_cmd(args.game, pdfs, cleanup_source=args.cleanup_source)
        else:
            p.print_help()
    else:  # gemini_filesearch
        client = _fs_client()
        if args.list:
            fs_list(client)
        elif args.delete:
            fs_delete(client, slugify(args.delete))
        elif args.game and args.pdfs:
            pdfs = _expand_pdfs(args.pdfs)
            if not pdfs:
                sys.exit("No PDFs found.")
            fs_cmd(client, args.game, pdfs)
        else:
            p.print_help()


if __name__ == "__main__":
    main()
