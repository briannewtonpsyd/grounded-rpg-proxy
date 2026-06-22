# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/briannewtonpsyd/grounded-rpg-proxy/releases/tag/v0.1.0
