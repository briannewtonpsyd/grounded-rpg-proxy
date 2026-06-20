"""Request/response audit trail — JSONL, one record per /v1/chat/completions call.

Captures exactly what PUM sends (model, full messages), how the proxy
transformed it (the query actually sent to NotebookLM), and what came back
(raw answer, cleaned answer, refusal/fallback flags, timing). This is the lens
for the "see what's possible" phase: which PUM chains send what, and how
NotebookLM responds.

Tail it live:
    tail -f logs/audit.jsonl | jq .
Or with the bundled viewer:
    python -m app.audit            # pretty-print recent records
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import settings

_lock = threading.Lock()
_dir_ready = False


@dataclass
class AuditRecord:
    id: str
    ts: str
    latency_ms: int | None = None
    # request
    client_model: str | None = None
    notebook_id: str | None = None
    notebook_title: str | None = None
    stream: bool = False
    forward_mode: str | None = None
    request_params: dict[str, Any] | None = None  # raw body minus messages
    message_summary: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] | None = None  # full bodies (optional)
    query_sent: str | None = None
    # response
    raw_answer: str | None = None
    final_answer: str | None = None
    mechanics_score: int = 0  # mechanics-jargon count (drift signal on descriptive prompts)
    citations_stripped: bool = False
    refusal: bool = False
    fallback_used: bool = False
    turn_number: int | None = None
    status: int | None = None
    error: str | None = None


def summarize_messages(messages) -> list[dict[str, Any]]:
    """Compact, always-safe-to-log view: role + length + short preview."""
    out = []
    for m in messages:
        body = m.text()
        out.append(
            {
                "role": m.role,
                "chars": len(body),
                "preview": body[:160].replace("\n", " "),
            }
        )
    return out


def full_messages(messages) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": m.text()} for m in messages]


def _ensure_dir(path: str) -> None:
    global _dir_ready
    if _dir_ready:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    _dir_ready = True


def write(record: AuditRecord) -> None:
    if not settings.audit_log:
        return
    path = settings.audit_log_path
    line = json.dumps(asdict(record), ensure_ascii=False, default=str)
    with _lock:
        _ensure_dir(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_recent(n: int = 20) -> list[dict[str, Any]]:
    """Return the last n audit records as parsed dicts (newest last)."""
    path = settings.audit_log_path
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()[-n:]
    except FileNotFoundError:
        return []
    out = []
    for ln in lines:
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def _tail(n: int = 20) -> None:
    """Pretty-print the last n audit records to the console."""
    path = settings.audit_log_path
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()[-n:]
    except FileNotFoundError:
        print(f"No audit log at {path} yet.")
        return
    for ln in lines:
        r = json.loads(ln)
        title = r.get("notebook_title") or r.get("notebook_id")
        flags = []
        if r.get("refusal"):
            flags.append("REFUSAL")
        if r.get("fallback_used"):
            flags.append("FALLBACK")
        if r.get("error"):
            flags.append("ERROR")
        flag_str = (" [" + ",".join(flags) + "]") if flags else ""
        print(f"\n=== {r['ts']}  {r.get('client_model')} → {title}  "
              f"{r.get('latency_ms')}ms  turn={r.get('turn_number')}{flag_str}")
        for m in r.get("message_summary", []):
            print(f"   {m['role']:>9} ({m['chars']:>5}c): {m['preview']}")
        if r.get("query_sent"):
            print(f"   QUERY: {r['query_sent'][:300]}")
        if r.get("error"):
            print(f"   ERROR: {r['error']}")
        else:
            print(f"   ANSWER: {(r.get('final_answer') or '')[:400]}")


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    _tail(n)
