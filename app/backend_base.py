"""Shared backend interface so the proxy is agnostic to the grounding engine.

A backend turns PUM's OpenAI messages (system + narrative + instruction) into a
source-grounded answer for a selected "game" (a NotebookLM notebook, a Gemini
File Search store, etc.). The proxy front-end, telemetry, citation cleanup and
model<->game selection are identical across backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .schemas import ChatMessage


@dataclass
class AskOutcome:
    answer: str
    query_sent: str = ""          # what was actually forwarded (for the audit log)
    game_id: str = ""             # notebook id / file-search store name
    citations: list[str] = field(default_factory=list)
    turn_number: int | None = None  # NotebookLM only
    backend: str = ""


@runtime_checkable
class Backend(Protocol):
    name: str

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    def resolve(self, model: str | None) -> str:
        """Map an incoming OpenAI `model` value to a game id."""

    def model_cards(self) -> list[tuple[str, str]]:
        """(model_id, title) pairs advertised on /v1/models."""

    def title_for(self, game_id: str) -> str | None: ...

    async def ask(self, messages: list[ChatMessage], game_id: str) -> AskOutcome: ...
