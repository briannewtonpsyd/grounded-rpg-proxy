# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **OpenRouter embeddings** — set `LIGHTRAG_EMBEDDING_PROVIDER=openrouter` (e.g.
  `openai/text-embedding-3-large`) for a true one-key-for-everything OpenRouter setup
  (generation + embeddings). (OpenRouter still has no reranking.)
- **End-to-end test** (`tests/e2e.py`) — ingests a tiny fixture PDF and asserts the
  query is grounded; the gate that runs before merging ingest/embedding changes.

### Changed
- **Embedding settings reworked** — the embedding **provider** and **model** are now
  side by side, and the model field adapts to the provider: Gemini is fixed, OpenAI
  offers the two `-3-small/-large` options, and OpenRouter is **typeable** (pick a
  suggestion or type any embedding id).
- **Clearer game-name entry** — friendlier label, "spaces & capitals are fine", and a
  **live preview** of the resulting PUM model name as you type
  (`Vampire: The Masquerade` → `vampire-the-masquerade`).

### Fixed
- **OpenRouter embedding model wouldn't stick** — typing a custom OpenRouter embedding
  id and clicking away reverted it to a suggestion. OpenRouter now uses a plain text
  box (typed value sticks, no Enter needed), and the real vector dimension is detected
  at ingest, so any embedding model works.
- **Dashboard ingest used the wrong embedder** — a UI-selected embedding provider
  could silently fall back to the Gemini default (the index then wouldn't query). The
  ingest now uses the on-screen selection, and each game's embedding
  provider/model/**dimension** is resolved **per-index**, so games built with different
  embedders coexist and query correctly.
- **"Enter a game name first" false error** — typing the game name and immediately
  clicking Ingest could fail because the typed value hadn't synced to the server yet.
  Now waits a beat and re-checks before failing, so you don't have to re-type it.
- Corrected docs that wrongly said OpenRouter has no embeddings.

## [0.1.0] — 2026-06-21

First public release. A small local **OpenAI-compatible** proxy that grounds
[PUM Companion](https://jeansenvaars.itch.io/pum-companion)'s AI in your own RPG
rulebooks (via [LightRAG](https://github.com/HKUDS/LightRAG) — knowledge-graph +
vector retrieval).

### Added
- **One-key setup.** Runs on a single free Google Gemini key by default (embeddings
  + generation); on the free tier it can't be billed.
- **Cheap-by-default grounding.** RAG-only (vector) ingest by default (~pennies per
  library); the knowledge graph is an opt-in quality upgrade.
- **Admin dashboard** (browser) — connect PUM, add/ingest/delete games by dropping in
  PDFs with live progress, edit settings and API keys, and test a grounded query.
- **Per-game indexes.** Each game is a separate index and a PUM "model"; switch the
  grounded source by changing the model in PUM.
- **Scanned-PDF OCR** via Gemini vision — books with no text layer are detected and
  transcribed automatically (results cached so a retry never re-OCRs).
- **Multiple LLM providers** for generation/extraction: `gemini`, `openai`,
  `anthropic`, `openrouter`, and `custom` (any OpenAI-compatible endpoint, incl.
  local models like Ollama / LM Studio).
- **OpenRouter support** — one key for many vendor models, a "Test connection" button,
  and **live per-model pricing** fetched from OpenRouter so cost estimates are current.
- **Cost transparency & safety** — every ingest reports exact token counts and an
  estimated cost (live for OpenRouter, dated static estimates for direct providers),
  with an optional `INGEST_MAX_COST_USD` hard ceiling and clear free-tier guidance.
- **Resilient ingests** — run detached and survive a browser refresh or app restart;
  failed imports recover cleanly; copied PDFs are cleaned up on every exit path.

### Notes
- Cost figures and static price estimates are as of the release date — prices change;
  verify with your provider. Token *counts* are always exact.
- `gemini_filesearch` and `notebooklm` backends ship but are **deprecated** in favor
  of `lightrag`.

[Unreleased]: https://github.com/briannewtonpsyd/grounded-rpg-proxy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/briannewtonpsyd/grounded-rpg-proxy/releases/tag/v0.1.0
