"""Pure helpers: build the NotebookLM query, clean output, detect refusals."""

from __future__ import annotations

import re

from .schemas import ChatMessage

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    slug = _SLUG_RE.sub("-", (title or "").lower()).strip("-")
    return slug or "untitled"

# [1]  [2, 3]  [5-7]  [1-3, 5] -> numeric citation markers (NotebookLM/File Search).
_CITATION_RE = re.compile(r"\s*\[\d+(?:\s*[-,]\s*\d+)*\]")
# [Forbidden_Lands_GMs_Guide.pdf p205] -> file/source citations (LightRAG).
_FILE_CITATION_RE = re.compile(
    r"\s*\[[^\]]*?\.(?:pdf|txt|md|docx|epub)[^\]]*?\]", re.IGNORECASE)


# Clearly-mechanical jargon — a high count on a descriptive/world-building answer
# signals retrieval drift (the rules-heavy KG surfacing mechanics instead of lore).
_MECH_TERMS = ("d66", "willpower point", "skill point", "dice pool", "base die",
               "gear die", "artifact die", "2d6", "attribute score", "re-roll",
               "reroll", "game mechanic")


def mechanics_score(text: str) -> int:
    low = (text or "").lower()
    return sum(low.count(t) for t in _MECH_TERMS)


def extract_citations(text: str) -> list[str]:
    """Pull file/source citations (e.g. '[book.pdf p205]') for the audit log."""
    out, seen = [], set()
    for m in _FILE_CITATION_RE.finditer(text or ""):
        c = m.group().strip().strip("[]").strip()
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _system_text(messages: list[ChatMessage]) -> str:
    return "\n\n".join(m.text() for m in messages if m.role == "system" and m.text())


def _last_user_text(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.text():
            return m.text()
    return ""


def _all_user_text(messages: list[ChatMessage]) -> str:
    """All user messages joined in order.

    PUM packs the game context into an earlier user message and the terse
    instruction into the last one, so taking only the last drops the context.
    """
    return "\n\n".join(m.text() for m in messages if m.role == "user" and m.text())


def _full_transcript(messages: list[ChatMessage]) -> str:
    lines = []
    for m in messages:
        body = m.text()
        if body:
            lines.append(f"{m.role.capitalize()}: {body}")
    return "\n\n".join(lines)


_TRUNC_MARKER = "[…earlier story context truncated to fit…]\n\n"


def build_query(messages: list[ChatMessage], mode: str, max_chars: int = 0) -> str:
    """Collapse the OpenAI message array into a single NotebookLM query string.

    NotebookLM chat rejects inputs beyond ~4-5k chars, but PUM sends the whole
    growing story. When `max_chars` is set and `combined` overflows, we keep the
    system instruction and the final user instruction intact and fill the rest
    with the *tail* (most recent) of the earlier context.
    """
    if mode == "last_user":
        q = _last_user_text(messages)
        return q[:max_chars] if max_chars and len(q) > max_chars else q

    if mode == "full":
        q = _full_transcript(messages)
        return q[-max_chars:] if max_chars and len(q) > max_chars else q

    # default: "combined" — system instruction + user content (context + task).
    system = _system_text(messages)
    users = [m.text() for m in messages if m.role == "user" and m.text()]
    if not users:
        return system
    instruction = users[-1]
    context = "\n\n".join(users[:-1])

    head = f"Instruction:\n{system}\n\n---\n\n" if system else ""
    full = head + (context + "\n\n" if context else "") + instruction
    if not max_chars or len(full) <= max_chars:
        return full

    # Overflow: keep head + instruction, fill remaining budget with context tail.
    budget = max_chars - len(head) - len(instruction) - len(_TRUNC_MARKER) - 2
    if budget <= 0:
        return (head + instruction)[:max_chars]
    ctx_tail = context[-budget:]
    return f"{head}{_TRUNC_MARKER}{ctx_tail}\n\n{instruction}"


# Trailing "### References\n- book.pdf ..." block (LightRAG appends this).
_REF_SECTION_RE = re.compile(r"\s*#{1,4}\s*References\b.*\Z", re.IGNORECASE | re.DOTALL)


def strip_citations(text: str) -> str:
    text = _REF_SECTION_RE.sub("", text)
    text = _FILE_CITATION_RE.sub("", text)
    return _CITATION_RE.sub("", text).strip()


def is_refusal(text: str, markers: list[str]) -> bool:
    low = text.lower()
    return any(marker in low for marker in markers)
