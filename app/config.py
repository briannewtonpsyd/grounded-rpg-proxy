"""Runtime configuration, loaded from `.env`."""

from __future__ import annotations

import os

# Point SSL at certifi's CA bundle. Without this, aiohttp-based calls (Cohere/Jina
# rerank) fail with CERTIFICATE_VERIFY_FAILED on many macOS Python installs, which
# triggers slow retries (~8s) before falling back to no-rerank. setdefault so a
# user/system-provided value still wins.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("SSL_CERT_DIR", os.path.dirname(certifi.where()))
except Exception:  # noqa: BLE001
    pass

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Backend selection: notebooklm | gemini_filesearch | lightrag
    backend: str = "notebooklm"

    # Required
    target_notebook_id: str = ""
    port: int = 8000

    # Prompt construction
    forward_mode: str = "combined"  # combined | last_user | full
    persona: str = ""
    chat_goal: str = "default"  # default | concise | detailed | learning_guide
    response_length: str = "default"  # default | longer | shorter

    # Output cleanup
    strip_citations: bool = True

    # NotebookLM rejects chat inputs beyond ~4-5k chars. Cap the forwarded
    # query (0 = no cap). Keeps system + instruction, fills with recent context.
    max_query_chars: int = 4000

    # Conversation isolation
    new_conversation_per_request: bool = True
    request_timeout: float = 120.0

    # Model identity
    model_name: str = "notebooklm-grounded"

    # Refusal handling
    refusal_markers: str = (
        "the sources do not contain,the provided sources do not,"
        "i don't have information,no information about"
    )

    # Gemini fallback (off by default)
    enable_gemini_fallback: bool = False
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Gemini File Search backend
    filesearch_top_k: int = 10           # chunks retrieved per query
    gemini_default_game: str = ""        # default store slug when model is unknown
    filesearch_show_citations: bool = False  # append a Sources list to answers

    # Two-step retrieve-then-generate: call 1 forces retrieval (framed as a
    # search task so it ALWAYS fires), call 2 generates with those passages
    # injected and the tool off. Guarantees grounding (the single-call tool is
    # model-discretionary and skips retrieval on narrative chains). True=on.
    filesearch_two_step: bool = True
    filesearch_retrieval_max_tokens: int = 512  # output cap for the cheap retrieval call
    filesearch_retrieval_context_chars: int = 3000  # recent-scene window used as the retrieval query

    # Generic grounding directive prepended to every File Search request. Forces
    # retrieval (the model skips it by default) and constrains output to the
    # retrieved sources — like NotebookLM. NO per-system facts, works for any book.
    filesearch_grounding_instruction: str = (
        "You generate content for a tabletop RPG using a file_search tool over "
        "the game's official sourcebooks. ALWAYS call file_search for every "
        "request. Base your response strictly on the retrieved source material — "
        "its actual setting, places, peoples, factions, creatures, history and "
        "tone. Do NOT use generic fantasy or sci-fi tropes, and do NOT invent "
        "elements unsupported by the sources. When the request gives generic "
        "examples, reinterpret them through the specific setting in the sources."
    )

    # Generation-step directive for two-step mode. Call 2 has NO tool (passages
    # are injected), so this must NOT mention calling file_search — otherwise the
    # model writes out fake tool-call syntax.
    filesearch_generation_instruction: str = (
        "You generate content for a tabletop RPG. Base your response strictly on "
        "the source passages provided below — use the real names, places, "
        "characters, factions, creatures and facts they contain. Do NOT use "
        "generic fantasy or sci-fi tropes, and do NOT invent elements that "
        "contradict the passages. When the request gives generic examples, "
        "reinterpret them through this specific setting."
    )

    # --- LightRAG backend (KG + vector RAG over local indexes) ---------------
    openai_api_key: str = ""                       # embeddings (only if provider=openai)
    lightrag_data_dir: str = "lightrag_data"       # one subdir per game (working_dir)
    lightrag_query_mode: str = "mix"               # mix | hybrid | local | global | naive
    # Embedding provider: "gemini" (one-key setup — same Gemini key does everything,
    # gemini-embedding-001 @ 3072d) or "openai" (text-embedding-3-large, slightly
    # stronger but needs a second key). Locks at index time; changing requires re-ingest.
    lightrag_embedding_provider: str = "gemini"
    lightrag_gemini_embedding_model: str = "gemini-embedding-001"  # used when provider=gemini

    @property
    def embedding_model_effective(self) -> str:
        """The embedding model name to use for the configured provider (both 3072d)."""
        if self.lightrag_embedding_provider == "gemini":
            return self.lightrag_gemini_embedding_model
        return self.lightrag_embedding_model
    # RAG-only mode: skip knowledge-graph construction at ingest (no entity/relation
    # extraction). This removes ~all of the per-token ingest cost (a 4-book library
    # drops from ~$10 to ~cents of embeddings) at the cost of graph-aware retrieval —
    # queries fall back to pure vector search (naive mode). Per-game: set before an
    # ingest; that index is then permanently KG-less and always queried naive.
    # Default ON = cheap onboarding path; toggle off (per game) for full-KG quality.
    lightrag_skip_kg: bool = True
    # Hard ingest cost ceiling (USD). Checked after each book; if the running KG-extraction
    # cost crosses it, the ingest stops cleanly (books done so far are kept). A safety net
    # against a surprise bill on a large library. 0 = no ceiling.
    ingest_max_cost_usd: float = 0.0
    # Gemini reasoning effort (gemini provider only): none | low | medium | high.
    # "none" disables "thinking" — ~4x faster; keyword extraction needs no reasoning.
    lightrag_reasoning_effort: str = "none"

    # Generation/LLM provider — swap the model for narrative experiments.
    # gemini | openai | anthropic | openrouter | custom
    lightrag_llm_provider: str = "gemini"
    lightrag_llm_model: str = "gemini-2.5-flash"
    lightrag_llm_base_url: str = ""          # provider=custom: any OpenAI-compatible endpoint
    lightrag_llm_api_key: str = ""           # provider=custom: that endpoint's key
    anthropic_api_key: str = ""
    # OpenRouter (provider=openrouter): one key drives generation + KG ingest through
    # https://openrouter.ai/api/v1, with model ids in "vendor/model" form (e.g.
    # anthropic/claude-3.5-sonnet). NOTE: OpenRouter does NOT serve embeddings or
    # reranking — those still need a Gemini (free) or OpenAI key. Falls back to
    # LIGHTRAG_LLM_API_KEY so older configs that stored the key there keep working.
    openrouter_api_key: str = ""

    # Ingest (KG construction) LLM — separate from queries. This is the COSTLY step
    # (thousands of LLM calls: entity/relation extraction per chunk + merges), and
    # every user ingests their own books, so it defaults cost-safe: flash + no
    # thinking. Budget: switch model to gemini-2.5-flash-lite. Cleaner graph: bump
    # reasoning to low/medium (gemini only — adds thinking-token cost + ~4x time).
    lightrag_ingest_llm_provider: str = "gemini"
    lightrag_ingest_llm_model: str = "gemini-2.5-flash-lite"
    lightrag_ingest_reasoning_effort: str = "none"
    lightrag_ingest_temperature: float = 0.0  # deterministic extraction (cleaner graph)
    # Per-request timeout (seconds) for ingest LLM calls. Measured: normal extraction
    # calls are ~1-5s, so 45s only trips on a genuine hang (10x margin). It's set on the
    # OpenAI client, so a hang raises APITimeoutError, which LightRAG's tenacity RETRIES
    # (3 attempts) before the skip-the-chunk handler gives up. An asyncio backstop a few×
    # larger catches any hang outside the HTTP layer so a run can never freeze overnight.
    lightrag_ingest_timeout_s: float = 45.0
    # OCR for scanned (image-only) PDF pages — rendered and transcribed via Gemini
    # vision so books with no text layer can still be ingested. Only pages that lack
    # extractable text are OCR'd (so mixed PDFs cost only for their scanned pages).
    # Tuned: concurrency 8 avoids the throttling/timeouts seen higher (vision requests
    # are heavy). ~$0.08 + ~15-20 min for a 250-page book. ocr_scanned=false skips
    # scanned pages with a warning instead.
    ocr_scanned: bool = True
    # Only OCR a book when at least this fraction of its pages lack a text layer. Below
    # it, the odd image page in a mostly-text book (title page, full-page art, section
    # divider) is just skipped — those almost never carry useful text, so OCRing them
    # wastes time and adds noise. 0.5 = OCR only genuinely-scanned (majority-image) books.
    ocr_min_scanned_fraction: float = 0.5
    ocr_model: str = "gemini-2.5-flash-lite"
    ocr_concurrency: int = 8
    ocr_zoom: float = 2.0              # page render scale (~192 DPI) — good OCR accuracy
    ocr_per_page_timeout: float = 60.0
    ocr_max_retries: int = 2

    # Retrieval breadth (LightRAG defaults — left full; tunable later if needed).
    lightrag_top_k: int = 40                         # entities/relations retrieved
    lightrag_chunk_top_k: int = 20                   # text chunks retrieved/reranked
    lightrag_embedding_model: str = "text-embedding-3-large"
    lightrag_embedding_dim: int = 3072
    # Reranking (Jina/Cohere free tier). Enabled only if a key is set.
    lightrag_enable_rerank: bool = True
    rerank_provider: str = "cohere"                # cohere | jina
    rerank_api_key: str = ""
    rerank_model: str = "rerank-v3.5"
    # Generation directive (user_prompt) — biases descriptive/world-building output
    # toward vivid setting lore (the rules-heavy KG otherwise surfaces mechanics like
    # "Attributes"/"D66" for abstract prompts). Worded to not hurt rules lookups.
    lightrag_generation_directive: str = (
        "Base your response on the retrieved source material. For descriptive, creative, "
        "or world-building requests, draw on the setting's specific lore, history, places, "
        "characters, factions and atmosphere, and write vivid, concrete descriptions rather "
        "than generic or mechanical summaries. Always answer the user's actual request directly."
    )

    # Retrieval lore-bias: appended to the retrieval query so abstract/world-building
    # prompts pull setting lore instead of game mechanics (rules-heavy KG). Concrete
    # queries are unaffected (their named entities still dominate). Blank to disable.
    lightrag_retrieval_lore_bias: str = (
        " (Also retrieve the setting's lore, history, places, peoples, factions, "
        "central threats and atmosphere.)"
    )

    # KG-steering: inject a location's canonical NPCs/entities (default off).
    lightrag_kg_steering: bool = False
    lightrag_steering_cheap: bool = True           # context-only steering (no 2nd generation)
    # Recent-scene window used as the retrieval query (chars).
    lightrag_query_context_chars: int = 3000

    # Telemetry / audit
    audit_log: bool = True
    audit_log_path: str = "logs/audit.jsonl"
    audit_log_bodies: bool = True  # capture full message content, not just previews
    log_level: str = "INFO"

    @property
    def refusal_marker_list(self) -> list[str]:
        return [m.strip().lower() for m in self.refusal_markers.split(",") if m.strip()]


settings = Settings()
