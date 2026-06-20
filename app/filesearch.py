"""Gemini File Search backend — managed RAG over uploaded rulebook PDFs.

Each "game" is a File Search store (created/populated by `python -m app.ingest`).
At query time we send PUM's full messages (system + the large growing narrative +
the instruction) to Gemini with the File Search tool pointed at that game's store:
Gemini retrieves the relevant rules/lore chunks AND reads the full narrative AND
generates — fast, with citations in grounding_metadata.

Unlike the NotebookLM backend, there is no ~4k input cap: the narrative is just
ordinary context tokens in Gemini's ~1M window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .backend_base import AskOutcome
from .config import settings
from .schemas import ChatMessage
from .transform import slugify

log = logging.getLogger("nblm.filesearch")


@dataclass
class StoreEntry:
    name: str       # "fileSearchStores/xxxx"
    title: str      # display_name
    slug: str


class FileSearchBackend:
    name = "gemini_filesearch"

    def __init__(self) -> None:
        self._client = None
        self._entries: list[StoreEntry] = []
        self._slug_to_name: dict[str, str] = {}
        self._name_set: set[str] = set()

    async def start(self) -> None:
        from google import genai  # lazy

        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required for the gemini_filesearch backend."
            )
        self._client = genai.Client(api_key=settings.gemini_api_key)
        await self._load_registry()

    async def stop(self) -> None:
        self._client = None

    async def _load_registry(self) -> None:
        try:
            stores = await self._client.aio.file_search_stores.list()
        except Exception:  # noqa: BLE001 — registry is best-effort
            log.warning("Could not list File Search stores", exc_info=True)
            return
        async for st in stores:
            title = getattr(st, "display_name", "") or ""
            slug = slugify(title) if title else slugify(st.name.split("/")[-1])
            if slug in self._slug_to_name and self._slug_to_name[slug] != st.name:
                slug = f"{slug}-{st.name.split('/')[-1][:4]}"
            self._slug_to_name[slug] = st.name
            self._name_set.add(st.name)
            self._entries.append(StoreEntry(name=st.name, title=title, slug=slug))
        log.info("Loaded %d File Search stores", len(self._entries))

    # --- selection ----------------------------------------------------------

    def resolve(self, model: str | None) -> str:
        default = self._slug_to_name.get(
            slugify(settings.gemini_default_game)
        ) if settings.gemini_default_game else (
            self._entries[0].name if self._entries else ""
        )
        if not model or model == settings.model_name:
            return default
        if model in self._name_set:
            return model
        slug = slugify(model)
        if slug in self._slug_to_name:
            return self._slug_to_name[slug]
        log.info("Unknown model %r — falling back to default store", model)
        return default

    def model_cards(self) -> list[tuple[str, str]]:
        cards = [(settings.model_name, "Configured default game")]
        cards += [(e.slug, e.title or "(untitled)") for e in self._entries]
        return cards

    def title_for(self, store_name: str) -> str | None:
        for e in self._entries:
            if e.name == store_name:
                return e.title or None
        return None

    # --- chat ---------------------------------------------------------------

    def _to_contents(self, messages: list[ChatMessage]):
        from google.genai import types

        system_bits = [m.text() for m in messages if m.role == "system" and m.text()]
        system = "\n\n".join(system_bits) or None
        contents = []
        for m in messages:
            if m.role == "system":
                continue
            role = "model" if m.role == "assistant" else "user"
            body = m.text()
            if body:
                contents.append(types.Content(role=role, parts=[types.Part(text=body)]))
        return system, contents

    async def ask(self, messages: list[ChatMessage], store_name: str) -> AskOutcome:
        from google.genai import types

        if self._client is None:
            raise RuntimeError("Backend not started")
        if not store_name:
            raise RuntimeError(
                "No File Search store selected — run `python -m app.ingest` first."
            )

        system, contents = self._to_contents(messages)

        if settings.filesearch_two_step:
            passages, citations = await self._retrieve(messages, store_name)
            gen_system = self._grounded_system(system, passages)
            cfg = types.GenerateContentConfig(system_instruction=gen_system)  # no tool
            resp = await self._client.aio.models.generate_content(
                model=settings.gemini_model, contents=contents, config=cfg
            )
            answer = (resp.text or "").strip()
            mode = f"2step/{len(passages)}p"
        else:
            directive = settings.filesearch_grounding_instruction
            sys = f"{directive}\n\n{system}" if (directive and system) else (directive or system)
            tool = types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[store_name], top_k=settings.filesearch_top_k
                )
            )
            cfg = types.GenerateContentConfig(system_instruction=sys, tools=[tool])
            resp = await self._client.aio.models.generate_content(
                model=settings.gemini_model, contents=contents, config=cfg
            )
            answer = (resp.text or "").strip()
            citations = _extract_citations(resp)
            mode = "1call"

        if settings.filesearch_show_citations and citations:
            answer += "\n\nSources:\n" + "\n".join(f"- {c}" for c in citations)

        last_user = next(
            (m.text() for m in reversed(messages) if m.role == "user" and m.text()), ""
        )
        return AskOutcome(
            answer=answer,
            query_sent=f"[file_search {mode} store={store_name}] {last_user[:400]}",
            game_id=store_name,
            citations=citations,
            backend=self.name,
        )

    def _grounded_system(self, system: str | None, passages: list[str]) -> str:
        parts = [settings.filesearch_generation_instruction]
        if passages:
            parts.append(
                "Relevant passages retrieved from the official source material — base your "
                "response on these and stay consistent with them (use the real names, places, "
                "characters and facts they contain; do not invent alternatives):\n\n"
                + "\n\n".join(passages)
            )
        else:
            parts.append("(No relevant passages were found in the source material.)")
        if system:
            parts.append(system)
        return "\n\n".join(p for p in parts if p)

    def _retrieval_query(self, messages: list[ChatMessage]) -> str:
        """Focused query for the retrieval call: the recent scene + the task.

        Feeding the whole growing story makes the model not search; the current
        scene (where the active places/NPCs are named) triggers retrieval reliably.
        """
        users = [m.text() for m in messages if m.role == "user" and m.text()]
        if not users:
            return ""
        instruction = users[-1]
        context = "\n\n".join(users[:-1])
        tail = context[-settings.filesearch_retrieval_context_chars :]
        return (f"{tail}\n\n{instruction}" if tail else instruction).strip()

    async def _retrieve(self, messages, store_name: str) -> tuple[list[str], list[str]]:
        """Call 1: force retrieval by framing a focused query as a search task."""
        from google.genai import types

        query = self._retrieval_query(messages)
        retrieval_system = (
            "You are a retrieval assistant for a tabletop RPG. Use the file_search tool to find "
            "ALL passages in the official source material relevant to the named places, characters, "
            "factions, creatures, rules and topics in the following text. ALSO always search for "
            "the setting's most defining and iconic features, its central threat or phenomenon, "
            "dominant factions, and core history — the things that make this world unique. ALWAYS "
            "call file_search, and search for specific proper nouns by name. Then briefly list what you found."
        )
        tool = types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=[store_name], top_k=settings.filesearch_top_k
            )
        )
        cfg = types.GenerateContentConfig(
            system_instruction=retrieval_system,
            tools=[tool],
            max_output_tokens=settings.filesearch_retrieval_max_tokens,
            # Retrieval is pure search — no reasoning needed. Disabling thinking is
            # faster, cheaper, and avoids burning the output budget before the
            # tool call (which truncated retrieval and missed salient chunks).
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        resp = await self._client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=[types.Content(role="user", parts=[types.Part(text=query)])],
            config=cfg,
        )
        passages: list[str] = []
        citations: list[str] = []
        seen: set[str] = set()
        for cand in resp.candidates or []:
            gm = getattr(cand, "grounding_metadata", None)
            for ch in (getattr(gm, "grounding_chunks", None) or []) if gm else []:
                rc = getattr(ch, "retrieved_context", None)
                if not rc:
                    continue
                text = (getattr(rc, "text", "") or "").strip()
                title = (getattr(rc, "title", None) or "").split("/")[-1]
                page = getattr(rc, "page_number", None)
                label = title + (f" p.{page}" if page else "")
                if text:
                    passages.append(f"[{label}] {text}")
                if label and label not in seen:
                    seen.add(label)
                    citations.append(label)
        return passages, citations


def _extract_citations(resp) -> list[str]:
    """Pull source titles/pages from grounding_metadata, de-duplicated."""
    out: list[str] = []
    seen = set()
    try:
        for cand in resp.candidates or []:
            gm = getattr(cand, "grounding_metadata", None)
            if not gm:
                continue
            for chunk in getattr(gm, "grounding_chunks", None) or []:
                rc = getattr(chunk, "retrieved_context", None)
                if not rc:
                    continue
                title = getattr(rc, "title", None) or getattr(rc, "uri", None) or ""
                page = getattr(rc, "page_number", None)
                label = f"{title}" + (f" (p.{page})" if page else "")
                if label and label not in seen:
                    seen.add(label)
                    out.append(label)
    except Exception:  # noqa: BLE001 — citations are best-effort
        log.debug("citation extraction failed", exc_info=True)
    return out


backend = FileSearchBackend()
