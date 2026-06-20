"""Thin async wrapper around notebooklm-py for the proxy's needs.

Supports selecting the grounded notebook *per request* via the OpenAI `model`
field: each notebook is advertised as a model (a slug of its title), so the
target source can be switched straight from PUM's model picker — no restart.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from notebooklm import ChatGoal, ChatResponseLength, NotebookLMClient

from .backend_base import AskOutcome
from .config import settings
from .schemas import ChatMessage
from .transform import build_query, slugify

log = logging.getLogger("nblm.backend")

# notebooklm-py 0.7.1 exposes DEFAULT, CUSTOM, LEARNING_GUIDE.
_GOALS = {
    "default": ChatGoal.DEFAULT,
    "custom": ChatGoal.CUSTOM,
    "learning_guide": ChatGoal.LEARNING_GUIDE,
}
_LENGTHS = {
    "default": ChatResponseLength.DEFAULT,
    "longer": ChatResponseLength.LONGER,
    "shorter": ChatResponseLength.SHORTER,
}


@dataclass
class NotebookEntry:
    id: str
    title: str
    slug: str


class NotebookBackend:
    """Owns a single NotebookLMClient bound to the app's event loop."""

    name = "notebooklm"

    def __init__(self) -> None:
        self._client: NotebookLMClient | None = None
        self._cm = None
        self._last_conv: dict[str, str | None] = {}
        self._configured: set[str] = set()
        self._lock = asyncio.Lock()
        # model-selection registry
        self._entries: list[NotebookEntry] = []
        self._slug_to_id: dict[str, str] = {}
        self._id_set: set[str] = set()

    async def start(self) -> None:
        self._cm = NotebookLMClient.from_storage()
        self._client = await self._cm.__aenter__()
        await self._load_registry()

    async def stop(self) -> None:
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)
            self._cm = None
            self._client = None

    async def _load_registry(self) -> None:
        """List notebooks once and build the model<->notebook maps."""
        try:
            notebooks = await self._client.notebooks.list()
        except Exception:  # noqa: BLE001 — registry is best-effort
            log.warning("Could not list notebooks for model registry", exc_info=True)
            return
        for nb in notebooks:
            nb_id = getattr(nb, "id", None)
            if not nb_id:
                continue
            title = getattr(nb, "title", "") or ""
            slug = slugify(title)
            if slug in self._slug_to_id and self._slug_to_id[slug] != nb_id:
                slug = f"{slug}-{nb_id[:4]}"  # disambiguate collisions
            self._slug_to_id[slug] = nb_id
            self._id_set.add(nb_id)
            self._entries.append(NotebookEntry(id=nb_id, title=title, slug=slug))
        log.info("Loaded %d notebooks into the model registry", len(self._entries))

    # --- model selection ----------------------------------------------------

    def resolve(self, model: str | None) -> str:
        """Map an incoming OpenAI `model` value to a notebook id."""
        default = settings.target_notebook_id
        if not model or model == settings.model_name:
            return default
        if model in self._id_set:
            return model
        slug = slugify(model)
        if slug in self._slug_to_id:
            return self._slug_to_id[slug]
        log.info("Unknown model %r — falling back to default notebook", model)
        return default

    def title_for(self, nb_id: str) -> str | None:
        for e in self._entries:
            if e.id == nb_id:
                return e.title or None
        return None

    def model_cards(self) -> list[tuple[str, str]]:
        """(model_id, title) pairs for /v1/models — default alias first."""
        cards: list[tuple[str, str]] = [
            (settings.model_name, "Configured default notebook")
        ]
        cards += [(e.slug, e.title or "(untitled)") for e in self._entries]
        return cards

    # --- chat ---------------------------------------------------------------

    async def _ensure_configured(self, nb_id: str) -> None:
        if nb_id in self._configured:
            return
        goal = _GOALS.get(settings.chat_goal, ChatGoal.DEFAULT)
        if settings.persona and settings.chat_goal == "default":
            goal = ChatGoal.CUSTOM
        try:
            await self._client.chat.configure(
                nb_id,
                goal=goal,
                response_length=_LENGTHS.get(
                    settings.response_length, ChatResponseLength.DEFAULT
                ),
                custom_prompt=settings.persona or None,
            )
        except Exception:  # noqa: BLE001 — persona is best-effort
            log.warning("Could not configure persona on %s", nb_id, exc_info=True)
        self._configured.add(nb_id)

    async def ask(self, messages: list[ChatMessage], notebook_id: str) -> AskOutcome:
        if self._client is None:
            raise RuntimeError("Backend not started")
        query = build_query(messages, settings.forward_mode, settings.max_query_chars)

        async def _do() -> AskOutcome:
            await self._ensure_configured(notebook_id)
            if settings.new_conversation_per_request and self._last_conv.get(notebook_id):
                try:
                    await self._client.chat.delete_conversation(
                        notebook_id, self._last_conv[notebook_id]
                    )
                except Exception:  # noqa: BLE001 — non-fatal cleanup
                    log.debug("delete_conversation failed", exc_info=True)
                self._last_conv[notebook_id] = None

            result = await asyncio.wait_for(
                self._client.chat.ask(notebook_id, query),
                timeout=settings.request_timeout,
            )
            conv = getattr(result, "conversation_id", None)
            self._last_conv[notebook_id] = conv
            return AskOutcome(
                answer=getattr(result, "answer", "") or "",
                query_sent=query,
                game_id=notebook_id,
                turn_number=getattr(result, "turn_number", None),
                backend=self.name,
            )

        if settings.new_conversation_per_request:
            async with self._lock:
                return await _do()
        return await _do()


backend = NotebookBackend()
