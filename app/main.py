"""FastAPI app: OpenAI-compatible surface in front of NotebookLM."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .audit import AuditRecord, full_messages, summarize_messages
from .audit import write as audit_write
from .backends import make_backend
from .config import settings
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ModelCard,
    ModelList,
    ResponseMessage,
    Usage,
)
from .transform import is_refusal, mechanics_score, strip_citations

backend = make_backend()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("nblm.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.target_notebook_id:
        log.warning("TARGET_NOTEBOOK_ID is empty — set it in .env before sending requests.")
    await backend.start()
    try:  # remove orphaned ingest temp dirs (copied PDFs) from prior runs
        from .admin_ui import sweep_stale_ingest_tmpdirs
        freed = sweep_stale_ingest_tmpdirs()
        if freed:
            log.info("Cleaned %.0f MB of stale ingest temp files.", freed / 1e6)
    except Exception:  # noqa: BLE001
        pass
    log.info("Backend ready. forward_mode=%s strip_citations=%s fresh_conv=%s",
             settings.forward_mode, settings.strip_citations,
             settings.new_conversation_per_request)
    try:
        yield
    finally:
        await backend.stop()


app = FastAPI(title="NotebookLM Proxy for PUM Companion", lifespan=lifespan)

# Admin dashboard at /admin (NiceGUI, same process). Best-effort: the proxy must
# run even if NiceGUI isn't installed.
try:
    from .admin_ui import init_admin
    init_admin(app)
except Exception:  # noqa: BLE001
    logging.getLogger("nblm.api").warning("Admin UI unavailable", exc_info=True)


def _new_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def _error(status: int, message: str, etype: str = "proxy_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": etype, "code": status}},
    )


@dataclass
class AnswerResult:
    final: str
    raw: str = ""
    query: str = ""
    refusal: bool = False
    fallback_used: bool = False
    turn_number: int | None = None


async def _answer(req: ChatCompletionRequest, game_id: str, rec: AuditRecord) -> AnswerResult:
    """Run a request through the selected backend (+ optional Gemini fallback)."""
    outcome = await backend.ask(req.messages, game_id)
    rec.query_sent = outcome.query_sent
    if outcome.citations:
        rec.request_params = {**(rec.request_params or {}), "citations": outcome.citations}

    raw = outcome.answer
    text = strip_citations(raw) if settings.strip_citations else raw
    refused = is_refusal(text, settings.refusal_marker_list)

    res = AnswerResult(
        final=text, raw=raw, query=outcome.query_sent, refusal=refused,
        turn_number=outcome.turn_number,
    )
    if refused:
        log.info("Refusal detected (turn=%s): %.80s", outcome.turn_number, text)
        if settings.enable_gemini_fallback:
            from .generator import generate  # lazy

            log.info("Falling back to Gemini.")
            res.final = await generate(req.messages)
            res.fallback_used = True
    return res


# --- Endpoints --------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "notebook": bool(settings.target_notebook_id)}


@app.get("/admin/audit")
async def audit_recent(n: int = 20):
    """Last n audited requests (newest last) — for eyeballing what PUM sends."""
    from .audit import read_recent

    return {"records": read_recent(n)}


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    cards = [ModelCard(id=mid, created=now) for mid, _title in backend.model_cards()]
    return ModelList(data=cards)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    raw = await request.json()
    try:
        req = ChatCompletionRequest.model_validate(raw)
    except Exception as e:  # noqa: BLE001
        return _error(400, f"Invalid request body: {e}", "invalid_request_error")

    notebook_id = backend.resolve(req.model)
    started = time.monotonic()
    rec = AuditRecord(
        id=_new_id(),
        ts=datetime.now(timezone.utc).isoformat(),
        client_model=req.model,
        notebook_id=notebook_id,
        notebook_title=backend.title_for(notebook_id),
        stream=req.stream,
        forward_mode=settings.forward_mode,
        request_params={k: v for k, v in raw.items() if k != "messages"},
        message_summary=summarize_messages(req.messages),
        messages=full_messages(req.messages) if settings.audit_log_bodies else None,
    )

    try:
        result = await _answer(req, notebook_id, rec)
    except TimeoutError:
        rec.status, rec.error = 504, "timeout"
        rec.latency_ms = int((time.monotonic() - started) * 1000)
        audit_write(rec)
        return _error(504, "NotebookLM timed out.", "timeout")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        hint = ""
        if any(w in msg.lower() for w in ("auth", "401", "403", "cookie", "session")):
            hint = " (session may have expired — re-run `notebooklm login`)"
        log.exception("Backend call failed")
        rec.status, rec.error = 502, msg
        rec.latency_ms = int((time.monotonic() - started) * 1000)
        audit_write(rec)
        return _error(502, f"NotebookLM request failed: {msg}{hint}", "upstream_error")

    content = result.final
    rec.raw_answer = result.raw
    rec.final_answer = content
    rec.mechanics_score = mechanics_score(content)
    rec.citations_stripped = settings.strip_citations
    rec.refusal = result.refusal
    rec.fallback_used = result.fallback_used
    rec.turn_number = result.turn_number
    rec.status = 200
    rec.latency_ms = int((time.monotonic() - started) * 1000)
    audit_write(rec)
    log.info("model=%s nb=%s turn=%s refusal=%s %dms", req.model,
             rec.notebook_title or notebook_id, result.turn_number,
             result.refusal, rec.latency_ms)

    model = req.model or settings.model_name
    if req.stream:
        return StreamingResponse(
            _stream(content, model), media_type="text/event-stream"
        )

    resp = ChatCompletionResponse(
        id=_new_id(),
        created=int(time.time()),
        model=model,
        choices=[Choice(message=ResponseMessage(content=content))],
        usage=Usage(),
    )
    return JSONResponse(content=resp.model_dump())


# --- Fake streaming ---------------------------------------------------------
# NotebookLM returns a complete answer, so we synthesize OpenAI SSE chunks.

def _chunk_text(text: str, size: int = 80):
    for i in range(0, len(text), size):
        yield text[i : i + size]


async def _stream(content: str, model: str):
    cid = _new_id()
    created = int(time.time())
    base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}

    def frame(delta: dict, finish=None) -> str:
        payload = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(payload)}\n\n"

    yield frame({"role": "assistant"})
    for piece in _chunk_text(content):
        yield frame({"content": piece})
    yield frame({}, finish="stop")
    yield "data: [DONE]\n\n"
