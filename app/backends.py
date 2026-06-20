"""Backend factory — pick the grounding engine from settings.backend.

`lightrag` is the supported backend. `gemini_filesearch` and `notebooklm` are
DEPRECATED (kept for reference, may be removed) — see the README.
"""

from __future__ import annotations

import logging

from .backend_base import Backend
from .config import settings

log = logging.getLogger("nblm.api")
_DEPRECATED = {"gemini_filesearch", "notebooklm"}


def make_backend() -> Backend:
    if settings.backend in _DEPRECATED:
        log.warning("BACKEND=%s is DEPRECATED and may be removed — use 'lightrag'.",
                    settings.backend)
    if settings.backend == "lightrag":
        from .lightrag_backend import backend as lr_backend

        return lr_backend
    if settings.backend == "gemini_filesearch":
        from .filesearch import backend as fs_backend

        return fs_backend
    if settings.backend == "notebooklm":
        from .notebook import backend as nblm_backend

        return nblm_backend
    raise ValueError(
        f"Unknown BACKEND={settings.backend!r} (expected 'notebooklm', "
        "'gemini_filesearch', or 'lightrag')"
    )
