"""LightRAG backend — KG + vector RAG over per-game local indexes.

Each game is a LightRAG index under `lightrag_data/<slug>/` (built by the ingest
CLI). The OpenAI `model` field selects the game; indexes are lazy-loaded on first
use. Grounding comes from `mix` mode (knowledge graph local+global + vector) with
optional reranking — which surfaces salient entities (e.g. the Blood Mist) and
real named NPCs that pure-vector retrieval buries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from .backend_base import AskOutcome
from .config import settings
from .schemas import ChatMessage
from .transform import extract_citations, slugify

log = logging.getLogger("nblm.lightrag")

GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
# Attribution headers OpenRouter recommends (surfaces this app in its dashboard/
# rankings). Optional — harmless if the OpenAI client/library ignores them.
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/briannewtonpsyd/grounded-rpg-proxy",
    "X-Title": "Grounded RPG Proxy",
}
_INDEX_MARKERS = ("vdb_chunks.json", "kv_store_full_docs.json",
                  "graph_chunk_entity_relation.graphml")

# Ingest-time token accounting (one process per ingest; reset at start). The LLM
# tracker captures Gemini KG-extraction tokens; embed_usage captures OpenAI
# embedding tokens. Read by app.ingest after the run to compute cost.
from lightrag.utils import TokenTracker  # noqa: E402
ingest_token_tracker = TokenTracker()
embed_usage = {"tokens": 0, "calls": 0}
ingest_skipped = {"count": 0}  # chunks skipped on empty/filtered LLM responses


# --- model functions (module-level; configured from settings) ----------------

def _resolve_llm(provider: str, model: str, reasoning: str):
    """(kind, model, base_url, api_key, extra_kwargs). reasoning_effort is only
    applied for gemini (the only provider here where it's both valid + effective)."""
    if provider == "gemini":
        extra = {"reasoning_effort": reasoning} if reasoning else {}
        return "openai", model, GEMINI_OPENAI_URL, settings.gemini_api_key, extra
    if provider == "openai":
        return "openai", model, None, settings.openai_api_key, {}
    if provider == "anthropic":
        return "anthropic", model, None, settings.anthropic_api_key, {}
    if provider == "openrouter":
        key = settings.openrouter_api_key or settings.lightrag_llm_api_key
        return "openai", model, OPENROUTER_URL, key, {"extra_headers": dict(OPENROUTER_HEADERS)}
    # custom (or any other value): an OpenAI-compatible endpoint. base_url is REQUIRED for
    # 'custom' — without it the OpenAI client silently falls back to api.openai.com, which
    # is surprising. Fail clearly instead.
    if provider == "custom" and not settings.lightrag_llm_base_url:
        raise ValueError("provider=custom needs a base URL (Settings → Custom endpoint base "
                         "URL) — an OpenAI-compatible endpoint, e.g. http://localhost:11434/v1.")
    return "openai", model, settings.lightrag_llm_base_url or None, settings.lightrag_llm_api_key, {}


def _provider_config_issue(provider: str) -> str | None:
    """A human-readable config problem for a provider, or None if it looks usable. Lets the
    Test button (and callers) report 'X key not set' instead of a cryptic auth/404 error."""
    need = {
        "gemini": (settings.gemini_api_key, "Gemini key"),
        "openai": (settings.openai_api_key, "OpenAI key"),
        "anthropic": (settings.anthropic_api_key, "Anthropic key"),
        "openrouter": (settings.openrouter_api_key or settings.lightrag_llm_api_key, "OpenRouter key"),
    }
    if provider in need:
        val, label = need[provider]
        return None if val else f"{label} is not set (Settings → {label})."
    if provider == "custom":
        if not settings.lightrag_llm_base_url:
            return "Custom provider needs a base URL (Settings → Custom endpoint base URL)."
        if not settings.lightrag_llm_api_key:
            return "Custom provider needs an endpoint key (Settings → Custom endpoint key)."
    return None


async def _call_llm(provider, model, reasoning, prompt, system_prompt, history_messages, **kw):
    kind, model, base_url, api_key, extra = _resolve_llm(provider, model, reasoning)
    kw.update(extra)
    if kind == "anthropic":
        from lightrag.llm.anthropic import anthropic_complete_if_cache
        return await anthropic_complete_if_cache(
            model, prompt, system_prompt=system_prompt,
            history_messages=history_messages or [], api_key=api_key, **kw)
    from lightrag.llm.openai import openai_complete_if_cache
    return await openai_complete_if_cache(
        model, prompt, system_prompt=system_prompt,
        history_messages=history_messages or [], base_url=base_url, api_key=api_key, **kw)


async def test_llm(provider: str, model: str, reasoning: str = "none") -> tuple[bool, str]:
    """One tiny round-trip to validate a provider/model/key from the dashboard.
    Returns (ok, detail) and never raises — on failure `detail` is the error text."""
    issue = _provider_config_issue(provider)
    if issue:  # clear message instead of a cryptic auth error / silent fallback
        return False, issue
    if not model.strip():
        return False, "No model id set."
    try:
        out = await _call_llm(provider, model, reasoning,
                              "Reply with the single word: OK.",
                              "You are a connectivity check.", [])
        return True, ((out or "").strip()[:200] or "(empty response)")
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"[:300]


async def _llm_func(prompt, system_prompt=None, history_messages=None, **kw):
    """Query-time LLM (keyword extraction + generation)."""
    kw.pop("keyword_extraction", None)
    return await _call_llm(settings.lightrag_llm_provider, settings.lightrag_llm_model,
                           settings.lightrag_reasoning_effort, prompt, system_prompt,
                           history_messages, **kw)


async def _ingest_llm_func(prompt, system_prompt=None, history_messages=None, **kw):
    """Ingest-time LLM (KG entity/relation extraction) — separate model + reasoning.
    Pins temperature and accumulates token usage for cost accounting."""
    kw.pop("keyword_extraction", None)
    kw.setdefault("temperature", settings.lightrag_ingest_temperature)
    kw["token_tracker"] = ingest_token_tracker
    per_req = settings.lightrag_ingest_timeout_s
    if per_req:
        # Client-level per-request timeout → APITimeoutError on a hang, which LightRAG's
        # tenacity retries (3 attempts) before this handler skips the chunk.
        kw["timeout"] = int(per_req)
    # Last-resort net for hangs OUTSIDE the HTTP layer. Must exceed the full retry
    # sequence (3 attempts of per_req + tenacity's ~12s exponential backoff) so it
    # never pre-empts a legitimate retry.
    backstop = per_req * 4 + 60 if per_req else None
    try:
        coro = _call_llm(settings.lightrag_ingest_llm_provider, settings.lightrag_ingest_llm_model,
                         settings.lightrag_ingest_reasoning_effort, prompt, system_prompt,
                         history_messages, **kw)
        return await (asyncio.wait_for(coro, backstop) if backstop else coro)
    except Exception as e:  # noqa: BLE001
        # Skip just this chunk's extraction (return empty) instead of letting it fail
        # the WHOLE book, on any of:
        #   - empty/filtered content (InvalidResponseError) — Gemini safety-filtering
        #     mature source text, the common case for e.g. Vampire;
        #   - retries exhausted, incl. timeouts retried 3x then given up (RetryError);
        #   - a hang outside the HTTP layer that hit the asyncio backstop (TimeoutError).
        # Re-raise anything else (auth/config errors) so real problems still surface.
        name, msg = type(e).__name__, str(e)
        if name in ("RetryError", "InvalidResponseError", "TimeoutError") \
                or "InvalidResponseError" in msg or "empty content" in msg.lower():
            ingest_skipped["count"] += 1
            log.warning("Ingest: skipping a chunk (%s): %s", name, msg[:140])
            return ""
        raise


def embedding_provider_for(model: str, recorded: str = "") -> str:
    """Provider an index uses: the value recorded at ingest, else inferred from the
    model name (keeps pre-provider indexes working — they were all OpenAI)."""
    if recorded:
        return recorded
    return "gemini" if "gemini" in (model or "").lower() else "openai"


def make_embed_func(provider: str, model: str):
    """Build an async embedding function bound to a provider+model. Used both at ingest
    (global settings) and at query time (resolved per-index, so an OpenAI-built index is
    always queried with OpenAI vectors and a Gemini-built one with Gemini vectors).

    We call the provider's client directly rather than LightRAG's openai_embed, whose
    decorator is hardwired to 1536d and rejects 3072d output as a vector-count mismatch.
    Retries on rate limits / transient errors so free-tier ingests slow down rather than
    fail. The outer EmbeddingFunc(embedding_dim=...) validates the result shape."""
    async def _embed(texts):
        import asyncio as _aio
        import numpy as np
        from openai import AsyncOpenAI
        if provider == "gemini":
            client = AsyncOpenAI(base_url=GEMINI_OPENAI_URL, api_key=settings.gemini_api_key)
        elif provider == "openrouter":
            client = AsyncOpenAI(base_url=OPENROUTER_URL,
                                 api_key=settings.openrouter_api_key or settings.lightrag_llm_api_key)
        else:
            client = AsyncOpenAI(api_key=settings.openai_api_key)
        last = None
        for attempt in range(5):  # backoff: handles free-tier 429s without failing the run
            try:
                resp = await client.embeddings.create(
                    model=model, input=texts, encoding_format="float")
                try:
                    embed_usage["tokens"] += getattr(resp.usage, "total_tokens", 0) or 0
                    embed_usage["calls"] += 1
                except Exception:  # noqa: BLE001
                    pass
                return np.array([d.embedding for d in resp.data], dtype=np.float32)
            except Exception as e:  # noqa: BLE001
                last = e
                msg = str(e).lower()
                if "429" in msg or "rate" in msg or "quota" in msg or "timeout" in msg or "503" in msg:
                    await _aio.sleep(min(2 ** attempt, 30))  # 1,2,4,8,16s
                    continue
                raise
        raise last
    return _embed


# Default query/ingest embed func from global settings (used unless an index records
# a different provider). Indexes resolve their own via make_embed_func at load time.
_embed_func = make_embed_func(settings.lightrag_embedding_provider, settings.embedding_model_effective)


async def _rerank_func(query, documents, top_n=None, **kw):
    from lightrag.rerank import cohere_rerank, jina_rerank
    fn = jina_rerank if settings.rerank_provider == "jina" else cohere_rerank
    return await fn(query, documents, top_n=top_n,
                    api_key=settings.rerank_api_key, model=settings.rerank_model)


class LightRAGBackend:
    name = "lightrag"

    def __init__(self) -> None:
        self._games: dict[str, str] = {}      # slug -> working_dir
        self._skip_kg_games: set[str] = set()  # slugs built RAG-only (query naive)
        self._instances: dict[str, object] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._pipeline_inited = False

    def _discover(self) -> None:
        self._games = {}
        self._skip_kg_games = set()  # slugs whose index was built RAG-only (query naive)
        data_dir = settings.lightrag_data_dir
        if os.path.isdir(data_dir):
            for name in sorted(os.listdir(data_dir)):
                wd = os.path.join(data_dir, name)
                if os.path.isdir(wd) and any(
                    os.path.exists(os.path.join(wd, m)) for m in _INDEX_MARKERS
                ):
                    slug = slugify(name)
                    self._games[slug] = wd
                    try:
                        if json.load(open(os.path.join(wd, "ingest_audit.json"))).get("skip_kg"):
                            self._skip_kg_games.add(slug)
                    except Exception:  # noqa: BLE001
                        pass

    def rescan(self) -> None:
        """Re-discover games on disk (after ingest/delete) — no restart needed."""
        self._discover()
        log.info("LightRAG games: %s", list(self._games) or "(none)")

    def evict(self, slug: str | None = None) -> None:
        """Drop cached LightRAG instance(s) so the next query reloads from disk."""
        if slug is None:
            self._instances.clear()
        else:
            self._instances.pop(slug, None)

    async def start(self) -> None:
        self._discover()
        log.info("LightRAG games discovered: %s", list(self._games) or "(none)")
        if not settings.openai_api_key:
            log.warning("OPENAI_API_KEY not set — embeddings will fail.")
        if settings.lightrag_enable_rerank and not settings.rerank_api_key:
            log.warning("Rerank enabled but RERANK_API_KEY not set — rerank disabled.")

    async def stop(self) -> None:
        for rag in self._instances.values():
            try:
                await rag.finalize_storages()
            except Exception:  # noqa: BLE001
                pass
        self._instances.clear()

    # --- selection ----------------------------------------------------------

    def resolve(self, model: str | None) -> str:
        default = ""
        if settings.gemini_default_game and slugify(settings.gemini_default_game) in self._games:
            default = slugify(settings.gemini_default_game)
        elif self._games:
            default = next(iter(self._games))
        if not model or model == settings.model_name:
            return default
        s = slugify(model)
        if s in self._games:
            return s
        log.info("Unknown model %r — falling back to default game", model)
        return default

    def model_cards(self) -> list[tuple[str, str]]:
        cards = [(settings.model_name, "Configured default game")]
        cards += [(slug, slug) for slug in self._games]
        return cards

    def title_for(self, game_id: str) -> str | None:
        return game_id or None

    # --- index lifecycle ----------------------------------------------------

    def _build(self, working_dir: str):
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc
        # Query with the SAME embedding provider/model/DIM the index was built with
        # (recorded in its audit) — NOT the global setting, so games built with different
        # embedders (e.g. a 1536d OpenAI-small game next to a 3072d Gemini game) each query
        # correctly. Legacy indexes (no recorded model) predate the provider option → OpenAI -3-large.
        model, provider = "text-embedding-3-large", "openai"
        dim = settings.lightrag_embedding_dim
        try:
            a = json.load(open(os.path.join(working_dir, "ingest_audit.json")))
            if a.get("embedding_model"):
                model = a["embedding_model"]
            provider = embedding_provider_for(model, a.get("embedding_provider", ""))
            # dim: recorded value, else infer from the model (1536 only for *-small).
            dim = a.get("embedding_dim") or (1536 if "small" in model else 3072)
        except Exception:  # noqa: BLE001
            pass
        kwargs = dict(
            working_dir=working_dir,
            llm_model_func=_llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=dim, max_token_size=8192, func=make_embed_func(provider, model)),
        )
        if settings.lightrag_enable_rerank and settings.rerank_api_key:
            kwargs["rerank_model_func"] = _rerank_func
        return LightRAG(**kwargs)

    async def _get(self, slug: str):
        if slug in self._instances:
            return self._instances[slug]
        if slug not in self._games:
            raise RuntimeError(
                f"No LightRAG index for game {slug!r}. Run `python -m app.ingest`.")
        lock = self._locks.setdefault(slug, asyncio.Lock())
        async with lock:
            if slug in self._instances:
                return self._instances[slug]
            rag = self._build(self._games[slug])
            await rag.initialize_storages()
            if not self._pipeline_inited:
                from lightrag.kg.shared_storage import initialize_pipeline_status
                await initialize_pipeline_status()
                self._pipeline_inited = True
            self._instances[slug] = rag
            log.info("Loaded LightRAG index for %s", slug)
            return rag

    # --- chat ---------------------------------------------------------------

    def _query(self, messages: list[ChatMessage]) -> str:
        """Focused retrieval query: recent scene tail + the instruction."""
        users = [m.text() for m in messages if m.role == "user" and m.text()]
        if not users:
            return ""
        instruction = users[-1]
        tail = "\n\n".join(users[:-1])[-settings.lightrag_query_context_chars:]
        q = (f"{tail}\n\n{instruction}" if tail else instruction).strip()
        return q + settings.lightrag_retrieval_lore_bias

    async def ask(self, messages: list[ChatMessage], game_id: str) -> AskOutcome:
        from lightrag import QueryParam
        rag = await self._get(game_id)
        query = self._query(messages)

        # A RAG-only index has no graph, so force naive (vector) retrieval + no
        # KG-steering regardless of the global query mode.
        rag_only = game_id in self._skip_kg_games
        mode = "naive" if rag_only else settings.lightrag_query_mode

        parts = []
        if settings.lightrag_generation_directive:
            parts.append(settings.lightrag_generation_directive)
        if settings.lightrag_kg_steering and not rag_only:
            steer = await self._steer(rag, query)
            if steer:
                parts.append(steer)
        user_prompt = "\n\n".join(parts)

        enable_rerank = bool(settings.lightrag_enable_rerank and settings.rerank_api_key)
        ans = await rag.aquery(query, param=QueryParam(
            mode=mode, enable_rerank=enable_rerank,
            top_k=settings.lightrag_top_k, chunk_top_k=settings.lightrag_chunk_top_k,
            user_prompt=user_prompt, include_references=True))
        raw = (ans or "").strip()
        # An empty answer means the generation LLM failed (LightRAG swallows the error and
        # returns empty; a genuine no-context answer is non-empty, e.g. "Sorry, …[no-context]").
        # Probe the Query LLM to confirm and surface a real failure instead of a silent blank
        # 200 — so the user sees e.g. an invalid/unavailable model. (Probe = no shared state,
        # so this is concurrency-safe, unlike capturing the error in a module global.)
        if not raw:
            ok, detail = await test_llm(settings.lightrag_llm_provider, settings.lightrag_llm_model,
                                        settings.lightrag_reasoning_effort)
            if not ok:
                log.warning("Query LLM unreachable (%s · %s): %s",
                            settings.lightrag_llm_provider, settings.lightrag_llm_model, detail)
                msg = (f"⚠️ Grounding proxy: couldn't generate a response. The Query LLM "
                       f"(provider={settings.lightrag_llm_provider}, model={settings.lightrag_llm_model}) "
                       f"failed — {detail}. Check the Query LLM provider/model in the dashboard "
                       f"Settings (use 'Test query LLM connection').")
                return AskOutcome(answer=msg, query_sent=f"[lightrag {mode} game={game_id} ERROR] {query[:200]}",
                                  game_id=game_id, backend=self.name)
        return AskOutcome(
            answer=raw,
            query_sent=f"[lightrag {mode} game={game_id}] {query[:300]}",
            game_id=game_id, citations=extract_citations(raw), backend=self.name)

    async def _steer(self, rag, scene: str) -> str:
        """Fetch canonical NPCs/entities for the scene and tell generation to prefer them."""
        from lightrag import QueryParam
        try:
            if settings.lightrag_steering_cheap:
                canon = await rag.aquery(scene, param=QueryParam(
                    mode="local", top_k=30, only_need_context=True))
            else:
                canon = await rag.aquery(
                    "List the named NPCs, factions and notable entities relevant to this "
                    "scene, each with a one-line role.\n\n" + scene,
                    param=QueryParam(mode="local", top_k=30))
        except Exception:  # noqa: BLE001
            return ""
        if not (canon and str(canon).strip()):
            return ""
        return ("CANONICAL ENTITIES from the source material relevant to this scene — when your "
                "response involves characters, inhabitants, factions or places, PREFER these real "
                "named entities over inventing new ones, and stay consistent with them:\n\n"
                + str(canon))


backend = LightRAGBackend()
