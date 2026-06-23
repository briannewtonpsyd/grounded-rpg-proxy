"""NiceGUI admin dashboard, mounted into the proxy at /admin.

Setup + administer the whole thing from a browser: see installed games, add a
game by uploading PDFs (with live ingest progress), check settings, and test a
grounded query. Mounted into the existing FastAPI app — same process/port.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile

from nicegui import ui

from .backends import make_backend
from .config import settings
from .schemas import ChatMessage
from .transform import slugify, strip_citations

backend = make_backend()

_INGEST_TMP_PREFIX = "rpg_ingest_"
_README_URL = "https://github.com/briannewtonpsyd/grounded-rpg-proxy#readme"


def sweep_stale_ingest_tmpdirs(max_age_hours: float = 2.0) -> int:
    """Delete orphaned ingest temp dirs (copied PDFs + progress/log). An ingest's temp
    dir is normally removed when it finishes, but if the browser tab closes or the run
    is killed before the UI sees completion, it leaks. Sweep on startup, skipping any
    dir touched recently (a long detached ingest that outlived a restart is still active).
    Returns bytes freed."""
    import time
    root = tempfile.gettempdir()
    cutoff = time.time() - max_age_hours * 3600
    freed = 0
    try:
        entries = os.listdir(root)
    except Exception:  # noqa: BLE001
        return 0
    for name in entries:
        if not name.startswith(_INGEST_TMP_PREFIX):
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            # newest mtime of any file inside → "recently active?"
            newest = max([os.path.getmtime(path)] +
                         [os.path.getmtime(os.path.join(dp, f))
                          for dp, _, fs in os.walk(path) for f in fs] or [0])
            if newest > cutoff:
                continue  # still active — leave it
            for dp, _, fs in os.walk(path):
                for f in fs:
                    try:
                        freed += os.path.getsize(os.path.join(dp, f))
                    except Exception:  # noqa: BLE001
                        pass
            shutil.rmtree(path, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
    return freed


# --- data helpers -----------------------------------------------------------

def _pdf_scan_info(path: str) -> dict:
    """Fast (no OCR) check of how many pages lack a text layer, so the UI can label a
    dropped PDF as text-OK vs scanned-needs-OCR. Returns {pages, scanned}."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(path)
        pages = doc.page_count
        scanned = sum(1 for p in doc if not p.get_text().strip())
        doc.close()
        return {"pages": pages, "scanned": scanned}
    except Exception:  # noqa: BLE001
        return {"pages": 0, "scanned": 0}


def _list_games() -> list[dict]:
    d = settings.lightrag_data_dir
    out = []
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        wd = os.path.join(d, name)
        if not os.path.isdir(wd) or name.startswith("."):  # skip .ocr_cache and other dotdirs
            continue
        audit, ds = {}, {}
        try:
            audit = json.load(open(os.path.join(wd, "ingest_audit.json")))
        except Exception:  # noqa: BLE001
            pass
        try:
            ds = json.load(open(os.path.join(wd, "kv_store_doc_status.json")))
        except Exception:  # noqa: BLE001
            pass
        # Count what actually SUCCEEDED (doc_status), not what was attempted (audit).
        # Skip duplicate-collision marker records — they're artifacts, not real books.
        st = [(os.path.basename(v.get("file_path", "") or "?"), v.get("status"))
              for v in ds.values()
              if not (v.get("content_summary") or "").startswith("[DUPLICATE")]
        by_name = {}  # one entry per book; a 'processed' run supersedes a stale 'failed'
        for n, s in st:
            if n not in by_name or s == "processed":
                by_name[n] = s
        processed = sum(1 for s in by_name.values() if s == "processed")
        failed = sum(1 for s in by_name.values() if s == "failed")
        processing = sum(1 for s in by_name.values() if s in ("processing", "pending"))
        names = [(n, s) for n, s in by_name.items()] \
            or [(b.get("name", "?"), "processed") for b in audit.get("books", [])]
        out.append({
            "game": name,
            "books": processed if by_name else len(audit.get("books", [])),
            "failed": failed,
            "processing": processing,
            "book_names": names,
            "ingest_time": f"{audit.get('total_seconds', '?')}s" if audit else "—",
        })
    return out


_ENV_PATH = ".env"


def _update_env(updates: dict) -> None:
    """Update/insert KEY=value lines in .env, preserving comments + order."""
    lines = open(_ENV_PATH).read().splitlines() if os.path.exists(_ENV_PATH) else []
    out, seen = [], set()
    for ln in lines:
        if "=" in ln and not ln.lstrip().startswith("#"):
            k = ln.split("=", 1)[0].strip()
            if k in updates:
                out.append(f"{k}={updates[k]}")
                seen.add(k)
                continue
        out.append(ln)
    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}={v}")
    with open(_ENV_PATH, "w") as f:
        f.write("\n".join(out) + "\n")


# --- UI ---------------------------------------------------------------------

def init_admin(app) -> None:
    @ui.refreshable
    def games_table() -> None:
        rows = _list_games()
        if not rows:
            ui.label("No games installed yet. Add one below.").classes("text-grey")
            return
        for g in rows:
            with ui.row().classes("items-center w-full no-wrap"):
                with ui.column().classes("gap-0 grow"):
                    ui.label(g["game"]).classes("text-bold")
                    total_b = g["books"] + g["failed"] + g.get("processing", 0)
                    # Colored status summary: ok (green) · processing (amber) · failed (red)
                    with ui.row().classes("items-center gap-2 text-caption"):
                        ui.label(f"{g['books']} ok").classes("text-positive")
                        if g.get("processing"):
                            ui.label(f"· {g['processing']} processing").classes("text-warning")
                        if g["failed"]:
                            ui.label(f"· {g['failed']} failed").classes("text-negative text-bold")
                        ui.label(f"· {total_b} book(s) · ingest {g['ingest_time']}").classes("text-grey")
                    # Per-book list, each colored by status — failed books stand out in red.
                    if g["book_names"]:
                        _col = {"processed": "text-grey", "failed": "text-negative text-bold",
                                "processing": "text-warning"}
                        _ico = {"processed": "✓", "failed": "✗", "processing": "⏳"}
                        with ui.row().classes("items-center gap-x-3 gap-y-0").style("flex-wrap:wrap"):
                            for bn, bs in g["book_names"]:
                                ui.label(f"{_ico.get(bs, '•')} {bn}").classes(
                                    "text-caption " + _col.get(bs, "text-grey"))
                ui.button(icon="delete", on_click=lambda g=g: _confirm_delete(g["game"])) \
                    .props("flat round color=negative")
            ui.separator()

    async def _confirm_delete(game: str) -> None:
        with ui.dialog() as d, ui.card():
            ui.label(f"Delete '{game}'? This permanently removes its index.")
            with ui.row():
                ui.button("Cancel", on_click=d.close).props("flat")
                ui.button("Delete", on_click=lambda: d.submit("yes")).props("color=negative")
        if await d != "yes":
            return
        backend.evict(game)  # drop cached instance
        wd = os.path.join(settings.lightrag_data_dir, game)
        if os.path.isdir(wd):
            shutil.rmtree(wd, ignore_errors=True)
        backend.rescan()
        ui.notify(f"Deleted {game}", type="warning")
        games_table.refresh()

    @ui.page("/")
    async def dashboard():
        ui.dark_mode(True)
        ui.colors(primary="#6e94a8", dark="#1e1f24", dark_page="#15161a")
        ui.add_head_html("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=Inter:wght@400;500;600&display=swap');
        body, .q-page, .nicegui-content { background:#15161a !important;
            font-family:'Inter',system-ui,sans-serif; color:#d7d8db; }
        .q-header { background:#1b1c20 !important; border-bottom:1px solid #2a2b30; }
        .q-card { background:#1e1f24 !important; border:1px solid #2c2d33;
            border-radius:16px; box-shadow:none !important; }
        .q-table, .q-table__card { background:transparent !important; color:#d7d8db; }
        .q-table tbody td, .q-table thead th { border-color:#2c2d33 !important; }
        .serif { font-family:'EB Garamond',Georgia,serif; }
        .q-field__native, .q-field__label { color:#d7d8db; }
        </style>""")
        with ui.header().classes("items-center"):
            ui.label("🎲 Grounded RPG — Admin").classes("text-h6 serif")
            ui.space()
            ui.link("📖 Instructions & costs", _README_URL, new_tab=True) \
                .classes("text-caption").style("color:#cfd8dc")
        with ui.column().classes("w-full max-w-4xl mx-auto gap-4 p-4"):

            # PUM connection
            with ui.card().classes("w-full"):
                ui.label("Connect PUM Companion").classes("text-subtitle1 text-bold serif")
                ui.label("AI Settings → Text Generation → Other (OpenAI-compatible):")
                ui.code(f"Base URL:  http://localhost:{settings.port}/v1").classes("w-full")
                ui.label("Model: the game name from the table below (e.g. forbidden-lands)")

            # Installed games
            with ui.card().classes("w-full"):
                ui.label("Installed games").classes("text-subtitle1 text-bold serif")
                games_table()
                ui.button("Refresh", on_click=games_table.refresh).props("flat dense")

            # Add a game
            with ui.card().classes("w-full"):
                ui.label("Add a game").classes("text-subtitle1 text-bold serif")
                ui.label("Knowledge-graph extraction runs LLM calls per chunk — minutes per book.")
                slug_in = ui.input("Game name", placeholder="e.g. Forbidden Lands").classes("w-72")
                slug_preview = ui.label("").classes("text-caption text-grey")

                def _update_slug_preview():
                    v = (slug_in.value or "").strip()
                    if v:
                        slug_preview.text = f"→ used in PUM as model:  {slugify(v)}"
                    else:
                        slug_preview.text = ("Spaces & capitals are fine — it becomes a simple "
                                             "name like “forbidden-lands”.")
                slug_in.on_value_change(lambda _: _update_slug_preview())
                _update_slug_preview()
                rag_only_sw = ui.switch("RAG-only (cheap): skip knowledge graph",
                                        value=settings.lightrag_skip_kg)
                ui.label("On = embeddings only (~cents/library, no graph; vector search). "
                         "Off = full knowledge graph (~$/library, richer grounding). Set per game.") \
                    .classes("text-caption text-grey")
                tmpdir = {"path": None}
                staged = []  # [{name, pages, scanned}] — populated as files are dropped

                @ui.refreshable
                def staged_list():
                    if not staged:
                        return
                    ui.label("Staged:").classes("text-caption text-bold")
                    for f in staged:
                        sc, pg = f["scanned"], max(f["pages"], 1)
                        if not sc or sc / pg < settings.ocr_min_scanned_fraction:
                            # All text, or just a few image pages (title/art) that we skip.
                            note = f" ({sc} image/art page(s) skipped)" if sc else ""
                            ui.label(f"✓ {f['name']} — {f['pages']} page(s), text OK{note}") \
                                .classes("text-caption text-positive")
                        else:  # genuinely a scanned image PDF — will OCR
                            mins = max(1, round(sc * 4.5 / 60))
                            ui.label(f"⚠️ {f['name']} — SCANNED image PDF: {sc}/{f['pages']} pages need "
                                     f"OCR (~{mins} min, ~${sc * 0.0003:.2f})").classes("text-caption text-warning")

                async def on_upload(e):
                    if not tmpdir["path"]:
                        tmpdir["path"] = tempfile.mkdtemp(prefix="rpg_ingest_")
                    dest = os.path.join(tmpdir["path"], e.file.name)
                    await e.file.save(dest)
                    staged.append({"name": e.file.name, **_pdf_scan_info(dest)})
                    staged_list.refresh()

                ui.upload(on_upload=on_upload, multiple=True, auto_upload=True) \
                    .props('accept=.pdf label="Drop rulebook PDFs"').classes("w-full")
                staged_list()
                # Progress UI (hidden until an ingest starts). The bar tracks book
                # index + per-chunk extraction; the detailed log is collapsed below.
                with ui.column().classes("w-full gap-1").style("display:none") as progress_box:
                    status = ui.label("").classes("text-caption serif")
                    bar = ui.linear_progress(value=0.0, show_value=False).props("rounded size=14px") \
                        .classes("w-full")
                with ui.expansion("Detailed log").classes("w-full") as log_exp:
                    log = ui.log(max_lines=400).classes("w-full h-48")

                async def do_ingest():
                    raw = (slug_in.value or "").strip()
                    if not raw:
                        # Type-then-click race: the just-typed name may still be in flight
                        # to the server when the button fires. Give it a beat and re-check
                        # before failing, so the user doesn't have to re-enter it.
                        await asyncio.sleep(0.25)
                        raw = (slug_in.value or "").strip()
                    if not raw:
                        ui.notify("Enter a game name first", type="negative")
                        return
                    if not tmpdir["path"]:
                        ui.notify("Upload at least one PDF", type="negative")
                        return
                    if not emb_model_value():
                        ui.notify("Pick or type an embedding model first", type="negative")
                        return
                    slug = slugify(raw)  # safe: raw is non-empty, so never the 'untitled' fallback
                    # Explicit OK before OCR — but only when it's non-trivial. A couple of
                    # image pages (covers/art) aren't worth a confirmation; hundreds are.
                    scanned_books = [f for f in staged if f["scanned"]
                                     and f["scanned"] / max(f["pages"], 1) >= settings.ocr_min_scanned_fraction]
                    pp = sum(f["scanned"] for f in scanned_books)
                    if scanned_books and settings.ocr_scanned and pp > 15:
                        mins = max(1, round(pp * 4.5 / 60))
                        with ui.dialog() as d, ui.card():
                            ui.label("Scanned PDF(s) — OCR needed").classes("text-bold serif")
                            for f in scanned_books:
                                ui.label(f"• {f['name']}: {f['scanned']} scanned page(s)").classes("text-caption")
                            ui.label(f"OCR via {settings.ocr_model}: ~{mins} min, ~${pp * 0.0003:.2f} total.") \
                                .classes("text-caption text-warning")
                            with ui.row():
                                ui.button("Cancel", on_click=lambda: d.submit("no")).props("flat")
                                ui.button("OK, OCR them", on_click=lambda: d.submit("yes")) \
                                    .props("color=primary unelevated")
                        if await d != "yes":
                            ui.notify("Ingest cancelled", type="warning")
                            return

                    def ui_safe(fn):
                        # The browser tab can disconnect mid-run. With the detached design that
                        # never stalls the ingest — but guard UI writes so a poll tick can't raise.
                        try:
                            fn()
                        except Exception:  # noqa: BLE001
                            pass

                    run_dir = tmpdir["path"]
                    tmpdir["path"] = None          # next upload starts a fresh staging dir
                    staged.clear()
                    ui_safe(staged_list.refresh)
                    prog_path = os.path.join(run_dir, "progress.json")
                    log_path = os.path.join(run_dir, "ingest.log")
                    ui_safe(lambda: progress_box.style("display:flex"))
                    ui_safe(lambda: setattr(bar, "value", 0.0))
                    ui_safe(lambda: setattr(status, "text", "Starting…"))
                    ui_safe(log.clear)

                    # Launch DETACHED: its own session, output → a log file, progress → a JSON
                    # file. The UI never drains its stdout, so it can't stall the subprocess
                    # (pipe block) or crash on a disconnect — the ingest even survives a UI
                    # restart. The UI is now just a poller over those two files.
                    logf = open(log_path, "w")
                    # Pass the config the detached ingest must use EXPLICITLY, read from the
                    # on-screen controls (env vars override .env). This honors what the user
                    # has selected even if they haven't clicked Save, and avoids relying on a
                    # .env write landing in time — otherwise the index can be built with the
                    # wrong embedder (silently falling back to the Gemini default) and won't query.
                    _ep = emb_prov.value
                    _em = emb_model_value()
                    _emodel_key = ("LIGHTRAG_GEMINI_EMBEDDING_MODEL" if _ep == "gemini"
                                   else "LIGHTRAG_EMBEDDING_MODEL")
                    env = {**os.environ,
                           "LIGHTRAG_SKIP_KG": str(rag_only_sw.value).lower(),
                           "LIGHTRAG_EMBEDDING_PROVIDER": _ep,
                           _emodel_key: _em,
                           "LIGHTRAG_EMBEDDING_DIM": str(1536 if "small" in _em else 3072),
                           # KG ingest LLM (used only when not skip_kg) — honor its dropdowns too.
                           "LIGHTRAG_INGEST_LLM_PROVIDER": ing_prov.value,
                           "LIGHTRAG_INGEST_LLM_MODEL": ing_model.value,
                           "LIGHTRAG_INGEST_REASONING_EFFORT": ing_effort.value}
                    proc = subprocess.Popen(
                        [sys.executable, "-m", "app.ingest", "--game", slug, run_dir,
                         "--progress-file", prog_path, "--cleanup-source"],
                        stdout=logf, stderr=subprocess.STDOUT, start_new_session=True, env=env)

                    def book_frac(prog, f):
                        bi, bn = prog.get("book_i", 0), prog.get("book_n", 1)
                        return (bi - 1 + min(f, 1.0)) / max(bn, 1) if bi else 0.0

                    def apply_progress(prog):
                        ph = prog.get("phase")
                        bi, bn, nm = prog.get("book_i", 0), prog.get("book_n", 1), prog.get("book_name", "")
                        if ph == "ocr":
                            od, ot = prog.get("ocr_done", 0), prog.get("ocr_total", 1)
                            ui_safe(lambda: setattr(bar, "value", od / max(ot, 1)))
                            ui_safe(lambda: setattr(status, "text", f"⚠️ OCR'ing scanned pages ({od}/{ot})…"))
                        elif ph == "extract":
                            cd, ct = prog.get("chunk_done", 0), prog.get("chunk_total", 0)
                            ui_safe(lambda: setattr(bar, "value", book_frac(prog, 0.9 * (cd / ct if ct else 0))))
                            ui_safe(lambda: setattr(status, "text",
                                    f"Book {bi}/{bn}: {nm} — extracting graph ({cd}/{ct})"))
                        elif ph == "merge":
                            ui_safe(lambda: setattr(bar, "value", book_frac(prog, 0.93)))
                            ui_safe(lambda: setattr(status, "text", f"Book {bi}/{bn}: {nm} — building graph…"))
                        elif ph == "embed":
                            ui_safe(lambda: setattr(bar, "value", book_frac(prog, 0.97)))
                            ui_safe(lambda: setattr(status, "text", f"Book {bi}/{bn}: {nm} — embedding…"))

                    poll = {"timer": None, "off": 0, "fin": False}

                    def finalize(prog):
                        if poll["fin"]:
                            return
                        poll["fin"] = True
                        if poll["timer"]:
                            poll["timer"].cancel()
                        try:
                            logf.close()
                        except Exception:  # noqa: BLE001
                            pass
                        if prog.get("phase") == "done":
                            backend.evict(slug)   # drop stale instance
                            backend.rescan()      # pick up the new/updated game live
                            sk = prog.get("skipped", 0)
                            ui_safe(lambda: setattr(bar, "value", 1.0))
                            ui_safe(lambda: setattr(status, "text",
                                    "Ingest complete ✓" + (f" — {sk} chunk(s) skipped" if sk else "")))
                            ui_safe(lambda: ui.notify(
                                f"Ingest complete{f' ({sk} chunks skipped)' if sk else ''}",
                                type="warning" if sk else "positive"))
                        else:
                            msg = prog.get("message") or "see the detailed log"
                            ui_safe(lambda: setattr(status, "text", f"Ingest failed: {msg[:80]}"))
                            ui_safe(lambda: ui.notify(f"Ingest failed — {msg[:80]}", type="negative"))
                        shutil.rmtree(run_dir, ignore_errors=True)
                        ui_safe(games_table.refresh)

                    def tick():
                        try:  # tail new log lines (offset-based; cap per tick to avoid flooding)
                            with open(log_path) as f:
                                f.seek(poll["off"])
                                chunk = f.read()
                                poll["off"] = f.tell()
                            fresh = [l for l in chunk.splitlines() if l.strip() and "HTTP Request" not in l]
                            for ln in fresh[-40:]:
                                ui_safe(lambda l=ln: log.push(l))
                        except Exception:  # noqa: BLE001
                            pass
                        prog = {}
                        try:
                            prog = json.load(open(prog_path))
                        except Exception:  # noqa: BLE001
                            pass
                        apply_progress(prog)
                        if prog.get("phase") in ("done", "error"):
                            finalize(prog)
                        elif proc.poll() is not None:  # exited without a terminal phase → failure
                            finalize(prog if prog.get("phase") else {"phase": "error",
                                                                     "message": "process exited unexpectedly"})

                    poll["timer"] = ui.timer(1.0, tick)

                ui.button("Ingest", on_click=do_ingest).props("color=primary rounded unelevated")

            # Settings (editable)
            with ui.card().classes("w-full"):
                ui.label("Settings").classes("text-subtitle1 text-bold serif")
                ui.label("Most changes apply live; embedding model needs re-ingest + restart.") \
                    .classes("text-grey text-caption")
                with ui.grid(columns=2).classes("gap-3 w-full"):
                    prov = ui.select(["gemini", "openai", "anthropic", "openrouter", "custom"],
                                     value=settings.lightrag_llm_provider, label="Query LLM provider")
                    model = ui.input("Query LLM model", value=settings.lightrag_llm_model)
                    effort = ui.select(["none", "low", "medium", "high"],
                                       value=settings.lightrag_reasoning_effort,
                                       label="Query reasoning (gemini only)")
                    mode = ui.select(["mix", "hybrid", "local", "global", "naive"],
                                     value=settings.lightrag_query_mode, label="Query mode")
                    ing_prov = ui.select(["gemini", "openai", "anthropic", "openrouter", "custom"],
                                         value=settings.lightrag_ingest_llm_provider,
                                         label="Ingest/KG LLM provider")
                    ing_model = ui.input("Ingest/KG model", value=settings.lightrag_ingest_llm_model)
                    ing_effort = ui.select(["none", "low", "medium", "high"],
                                           value=settings.lightrag_ingest_reasoning_effort,
                                           label="Ingest reasoning (gemini only)")
                    rerankp = ui.select(["cohere", "jina"], value=settings.rerank_provider,
                                        label="Rerank provider")
                    cost_cap = ui.number("Ingest cost ceiling $ (0 = none)",
                                         value=settings.ingest_max_cost_usd, format="%.2f", min=0)

                # Embeddings: provider + model side by side. The model CONTROL depends on
                # provider — a dropdown for Gemini/OpenAI (fixed sets), and a plain free-text
                # box for OpenRouter, where any embedding id is valid. (A typeable select
                # discards uncommitted text on blur, so OpenRouter gets a real text input.)
                _EMB_CHOICES = {
                    "gemini": ["gemini-embedding-001"],
                    "openai": ["text-embedding-3-small", "text-embedding-3-large"],
                }
                _emb_cur = settings.embedding_model_effective
                _is_or = settings.lightrag_embedding_provider == "openrouter"
                _sel_opts = _EMB_CHOICES.get(settings.lightrag_embedding_provider, ["text-embedding-3-large"])
                if not _is_or and _emb_cur not in _sel_opts:
                    _sel_opts = list(dict.fromkeys(_sel_opts + [_emb_cur]))
                with ui.row().classes("items-center gap-3 w-full no-wrap"):
                    emb_prov = ui.select(["gemini", "openai", "openrouter"],
                                         value=settings.lightrag_embedding_provider,
                                         label="Embedding provider").classes("grow")
                    emb_sel = ui.select(_sel_opts, value=(_sel_opts[0] if _is_or else _emb_cur),
                                        label="Embedding model").classes("grow")
                    emb_txt = ui.input("Embedding model (OpenRouter id)",
                                       value=(_emb_cur if _is_or else "openai/text-embedding-3-large"),
                                       placeholder="e.g. openai/text-embedding-3-large").classes("grow")
                emb_note = ui.label("").classes("text-caption text-grey")

                def emb_model_value() -> str:
                    """The chosen embedding model, from whichever control the provider uses."""
                    raw = emb_txt.value if emb_prov.value == "openrouter" else emb_sel.value
                    return (raw or "").strip()

                def _set_emb_view():
                    p = emb_prov.value
                    emb_txt.visible = (p == "openrouter")
                    emb_sel.visible = (p != "openrouter")
                    emb_note.text = {
                        "gemini": "Gemini uses gemini-embedding-001 (3072d) — one model.",
                        "openai": "Pick an OpenAI embedding model. Re-ingest each game to change.",
                        "openrouter": "Type any OpenRouter embedding id (e.g. openai/text-embedding-3-large). "
                                      "The real vector size is detected at ingest. Re-ingest to change.",
                    }.get(p, "")

                def _on_emb_prov_change():
                    # Provider CHANGED: reset the now-active control to that provider's default.
                    p = emb_prov.value
                    if p == "openrouter":
                        emb_txt.value = "openai/text-embedding-3-large"
                    else:
                        opts = _EMB_CHOICES.get(p, ["text-embedding-3-large"])
                        emb_sel.set_options(opts, value=opts[0])
                    _set_emb_view()

                emb_prov.on_value_change(lambda _: _on_emb_prov_change())
                _set_emb_view()  # initial: show the right control with its saved value

                rerank_on = ui.switch("Reranking enabled", value=settings.lightrag_enable_rerank)
                ui.label("API keys (click the eye to reveal)").classes("text-grey mt-2")
                with ui.grid(columns=2).classes("gap-3 w-full"):
                    k_gemini = ui.input("Gemini key (default: embeddings + generation)",
                                        value=settings.gemini_api_key,
                                        password=True, password_toggle_button=True)
                    k_openai = ui.input("OpenAI key (embeddings / openai provider)",
                                        value=settings.openai_api_key,
                                        password=True, password_toggle_button=True)
                    k_anthropic = ui.input("Anthropic key (anthropic provider)",
                                           value=settings.anthropic_api_key,
                                           password=True, password_toggle_button=True)
                    k_orkey = ui.input("OpenRouter key (openrouter provider)",
                                       value=settings.openrouter_api_key,
                                       password=True, password_toggle_button=True)
                    k_rerank = ui.input("Rerank key (Cohere/Jina, optional)",
                                        value=settings.rerank_api_key,
                                        password=True, password_toggle_button=True)
                    k_llmurl = ui.input("Custom endpoint base URL (custom provider)",
                                        value=settings.lightrag_llm_base_url)
                    k_llmkey = ui.input("Custom endpoint key (custom provider)",
                                        value=settings.lightrag_llm_api_key,
                                        password=True, password_toggle_button=True)
                ui.label("OpenRouter: one key, any vendor/model (e.g. anthropic/claude-haiku-4.5, "
                         "openai/gpt-4o-mini) for generation AND embeddings (e.g. "
                         "openai/text-embedding-3-large). No reranking, so keep a Cohere/Jina key if "
                         "you use rerank. Slugs change: see openrouter.ai/models.") \
                    .classes("text-caption text-grey")

                # Test the *currently selected* provider/model/keys (any provider), without
                # persisting the typed values — snapshot → probe → restore; only Save applies.
                async def test_llm_conn():
                    from .lightrag_backend import test_llm
                    fields = {
                        "openrouter_api_key": k_orkey.value, "lightrag_llm_api_key": k_llmkey.value,
                        "lightrag_llm_base_url": k_llmurl.value, "gemini_api_key": k_gemini.value,
                        "openai_api_key": k_openai.value, "anthropic_api_key": k_anthropic.value,
                    }
                    snap = {k: getattr(settings, k) for k in fields}
                    for k, v in fields.items():
                        setattr(settings, k, v)
                    test_out.classes(replace="text-caption text-grey")
                    test_out.text = f"Testing {prov.value} · {model.value}…"
                    try:
                        ok, detail = await test_llm(prov.value, model.value, effort.value)
                    finally:
                        for k, v in snap.items():
                            setattr(settings, k, v)  # restore — Save is what applies changes
                    test_out.text = ("✓ " if ok else "✗ ") + detail
                    test_out.classes(replace="text-caption "
                                     + ("text-positive" if ok else "text-negative"))

                with ui.row().classes("items-center gap-3 mt-1"):
                    ui.button("Test query LLM connection", on_click=test_llm_conn).props("flat dense")
                    test_out = ui.label("").classes("text-caption")

                def save_settings():
                    emb_model = emb_model_value()
                    # 3072 unless it's a *-small model (gemini-embedding-001 & -large are 3072).
                    emb_dim = 1536 if "small" in emb_model else 3072
                    emb_changed = (emb_model != settings.embedding_model_effective
                                   or emb_prov.value != settings.lightrag_embedding_provider)
                    # Route the model to the right setting: Gemini has its own field; openai/
                    # openrouter share LIGHTRAG_EMBEDDING_MODEL.
                    emb_env = ({"LIGHTRAG_GEMINI_EMBEDDING_MODEL": emb_model}
                               if emb_prov.value == "gemini"
                               else {"LIGHTRAG_EMBEDDING_MODEL": emb_model})
                    _update_env({
                        "LIGHTRAG_LLM_PROVIDER": prov.value, "LIGHTRAG_LLM_MODEL": model.value,
                        "LIGHTRAG_REASONING_EFFORT": effort.value, "LIGHTRAG_QUERY_MODE": mode.value,
                        "LIGHTRAG_INGEST_LLM_PROVIDER": ing_prov.value,
                        "LIGHTRAG_INGEST_LLM_MODEL": ing_model.value,
                        "LIGHTRAG_INGEST_REASONING_EFFORT": ing_effort.value,
                        **emb_env, "LIGHTRAG_EMBEDDING_DIM": emb_dim,
                        "LIGHTRAG_EMBEDDING_PROVIDER": emb_prov.value,
                        "INGEST_MAX_COST_USD": cost_cap.value or 0,
                        "RERANK_PROVIDER": rerankp.value,
                        "LIGHTRAG_ENABLE_RERANK": str(rerank_on.value).lower(),
                        "OPENAI_API_KEY": k_openai.value, "GEMINI_API_KEY": k_gemini.value,
                        "RERANK_API_KEY": k_rerank.value, "ANTHROPIC_API_KEY": k_anthropic.value,
                        "LIGHTRAG_LLM_BASE_URL": k_llmurl.value, "LIGHTRAG_LLM_API_KEY": k_llmkey.value,
                        "OPENROUTER_API_KEY": k_orkey.value,
                    })
                    # live-apply query-time settings (read per request)
                    settings.lightrag_llm_provider = prov.value
                    settings.lightrag_llm_model = model.value
                    settings.lightrag_reasoning_effort = effort.value
                    settings.lightrag_query_mode = mode.value
                    settings.lightrag_ingest_llm_provider = ing_prov.value
                    settings.lightrag_ingest_llm_model = ing_model.value
                    settings.lightrag_ingest_reasoning_effort = ing_effort.value
                    settings.rerank_provider = rerankp.value
                    settings.lightrag_enable_rerank = rerank_on.value
                    settings.openai_api_key = k_openai.value
                    settings.gemini_api_key = k_gemini.value
                    settings.rerank_api_key = k_rerank.value
                    settings.anthropic_api_key = k_anthropic.value
                    settings.lightrag_llm_base_url = k_llmurl.value
                    settings.lightrag_llm_api_key = k_llmkey.value
                    settings.openrouter_api_key = k_orkey.value
                    if emb_prov.value == "gemini":
                        settings.lightrag_gemini_embedding_model = emb_model
                    else:
                        settings.lightrag_embedding_model = emb_model
                    settings.lightrag_embedding_dim = emb_dim
                    settings.lightrag_embedding_provider = emb_prov.value
                    settings.ingest_max_cost_usd = float(cost_cap.value or 0)
                    msg = "Saved to .env (applied live)."
                    if emb_changed:
                        msg += " Embedding changed — re-ingest each game + restart to apply."
                    ui.notify(msg, type="positive", timeout=6000)

                ui.button("Save settings", on_click=save_settings).props("color=primary rounded unelevated")

            # Test a query
            with ui.card().classes("w-full"):
                ui.label("Test a grounded query").classes("text-subtitle1 text-bold serif")
                games = [r["game"] for r in _list_games()]
                game_sel = ui.select(games, label="Game", value=games[0] if games else None).classes("w-72")
                q_in = ui.input("Question", placeholder="How does pushing a roll work?").classes("w-full")
                answer = ui.markdown().classes("w-full serif text-body1")

                async def run_test():
                    if not game_sel.value or not q_in.value:
                        return
                    answer.content = "_…thinking_"
                    import time
                    t = time.monotonic()
                    out = await backend.ask([ChatMessage(role="user", content=q_in.value)],
                                            backend.resolve(game_sel.value))
                    ans = strip_citations(out.answer)
                    answer.content = f"**{time.monotonic()-t:.1f}s** · cites: {len(out.citations)}\n\n{ans}"

                ui.button("Run", on_click=run_test).props("color=primary rounded unelevated")

    ui.run_with(app, mount_path="/admin", title="Grounded RPG Admin", storage_secret="rpg-admin")
