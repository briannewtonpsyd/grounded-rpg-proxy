"""Run the proxy. `python -m app` serves the web UI; `--native` opens it in a
desktop window (the proxy still serves PUM on the same port either way)."""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="app", description="Grounded RPG proxy + admin UI")
    p.add_argument("--native", action="store_true",
                   help="Open the admin dashboard in a desktop window (needs pywebview)")
    p.add_argument("--port", type=int, default=None, help="Port (default from .env / 8000)")
    args = p.parse_args()

    from .config import settings
    port = args.port or settings.port

    # First-run hint: a missing key is the #1 stumble. Point the user at the free
    # key + the dashboard (where they can paste it — no .env editing needed).
    needs_gemini = settings.lightrag_embedding_provider == "gemini" or settings.backend != "lightrag"
    if needs_gemini and not settings.gemini_api_key:
        print("\n" + "─" * 64)
        print("  No Gemini API key set yet.")
        print("  1) Get a FREE key (no credit card): https://aistudio.google.com/apikey")
        print("  2) Paste it into the dashboard → Settings → Gemini key → Save.")
        print("     (Or put GEMINI_API_KEY=... in your .env and restart.)")
        print("─" * 64)

    if not args.native:
        # Default: serve + open the admin UI in the user's browser. A real browser tab
        # reconnects cleanly, unlike the embedded webview (--native), which has been the
        # source of disconnect-related ingest failures.
        import threading
        import time
        import webbrowser

        import uvicorn
        url = f"http://127.0.0.1:{port}/admin/"

        def _open():
            time.sleep(1.5)
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=_open, daemon=True).start()
        print(f"\n  Admin dashboard : {url}\n  PUM endpoint    : http://127.0.0.1:{port}/v1\n")
        uvicorn.run("app.main:app", host="127.0.0.1", port=port)
        return

    # Native window: run the server in a background thread, open a webview to /admin.
    import threading
    import time
    import urllib.request

    import uvicorn
    import webview

    server = uvicorn.Server(uvicorn.Config(
        "app.main:app", host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()

    for _ in range(120):  # wait up to ~60s for startup (model/index load)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)

    webview.create_window("🎲 Grounded RPG", f"http://127.0.0.1:{port}/admin/",
                          width=1150, height=950)
    webview.start()  # blocks until the window closes


if __name__ == "__main__":
    main()
