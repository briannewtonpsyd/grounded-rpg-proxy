"""End-to-end test: ingest a small fixture PDF, then query it and assert the answer
is grounded in the source.

This is the PR-review gate. "It compiles and renders" has repeatedly missed real bugs
(the ingest used the wrong embedder; per-index dim mismatches) that only an actual
ingest→query round-trip catches. Run it before opening/merging a PR that touches
ingest, retrieval, embeddings, or settings:

    python tests/e2e.py

Uses RAG-only (cheap, ~a cent) and a throwaway data dir, so it never touches your real
games. Requires an embedding+LLM key for the configured provider (Gemini by default);
skips cleanly (exit 0) if none is set.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURE = os.path.join(HERE, "fixtures", "sample_rulebook.pdf")
sys.path.insert(0, ROOT)


def main() -> int:
    # Isolate: throwaway data dir + RAG-only (fast/cheap). Set BEFORE importing config.
    tmp = tempfile.mkdtemp(prefix="e2e_data_")
    os.environ["LIGHTRAG_DATA_DIR"] = tmp
    os.environ["LIGHTRAG_SKIP_KG"] = "true"

    from app.config import settings
    prov = settings.lightrag_embedding_provider
    key = {"gemini": settings.gemini_api_key,
           "openrouter": settings.openrouter_api_key or settings.lightrag_llm_api_key,
           }.get(prov, settings.openai_api_key)
    if not key:
        print(f"SKIP: no key for embedding provider '{prov}' — set it to run the e2e test.")
        shutil.rmtree(tmp, ignore_errors=True)
        return 0

    import logging
    logging.getLogger("lightrag").setLevel(logging.ERROR)  # quiet the ingest/query chatter

    from app.ingest import lightrag_cmd
    from app.lightrag_backend import backend
    from app.schemas import ChatMessage

    # Distinctive facts only present in the fixture.
    checks = [
        ("Who rules the city of Brasshollow?", "quillon"),
        ("On what die roll is a critical hit scored?", "20"),
    ]
    failures = []
    try:
        print(f"Ingesting fixture (provider={prov}, RAG-only) …")
        lightrag_cmd("e2e-test", [FIXTURE])
        backend.rescan()
        if "e2e-test" not in backend._games:
            print("FAIL: game was not registered after ingest.")
            return 1

        # Run ALL queries in ONE event loop — LightRAG binds its worker pools to the
        # loop of the first call, so a fresh asyncio.run() per query throws
        # "bound to a different event loop". One loop keeps the cached index valid.
        async def run_checks():
            results = []
            for q, expected in checks:
                out = await backend.ask([ChatMessage(role="user", content=q)], "e2e-test")
                results.append((q, expected, (out.answer or "").lower()))
            return results

        for q, expected, ans in asyncio.run(run_checks()):
            ok = expected in ans
            print(f"  {'✓' if ok else '✗'} {q!r} -> expected {expected!r} | {ans[:80]!r}")
            if not ok:
                failures.append(q)
    finally:
        try:
            backend.evict()
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print(f"\nFAIL: {len(failures)} grounding check(s) failed.")
        return 1
    print("\nPASS: ingest → query grounded correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
