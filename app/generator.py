"""Optional Gemini generator — refusal fallback today, double-scaffold seam later.

Lazily imports google-genai so the proxy runs fine when the fallback is off.

Future "double-scaffold" idea (parked): instead of only firing on a refusal,
route generative PUM chains through here deliberately — call NotebookLM first to
retrieve grounded facts, then pass those facts + the chain instruction to Gemini
for the actual narrative writing. `generate()` already takes the full message
list plus optional grounding, so that mode slots in without reshaping callers.
"""

from __future__ import annotations

import logging

from .config import settings
from .schemas import ChatMessage

log = logging.getLogger("nblm.gemini")

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai  # lazy: only needed when fallback is on

        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


async def generate(messages: list[ChatMessage], grounding: str | None = None) -> str:
    """Generate a completion with Gemini from the original OpenAI messages.

    `grounding` is optional NotebookLM-retrieved context to ground the answer
    (used by the future double-scaffold path).
    """
    from google.genai import types  # lazy

    client = _get_client()

    system_bits = [m.text() for m in messages if m.role == "system" and m.text()]
    if grounding:
        system_bits.append(
            "Relevant grounded material from the source notebook:\n" + grounding
        )
    system_instruction = "\n\n".join(system_bits) or None

    contents = []
    for m in messages:
        if m.role == "system":
            continue
        role = "model" if m.role == "assistant" else "user"
        body = m.text()
        if body:
            contents.append(types.Content(role=role, parts=[types.Part(text=body)]))

    resp = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    return (resp.text or "").strip()
