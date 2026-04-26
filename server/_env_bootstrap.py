"""Load project-root .env for local dev (not committed). Optional; no extra dependency."""
from __future__ import annotations

import os
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_loaded = False


def load_local_env() -> None:
    """
    Read KEY=value lines from <repo>/.env into os.environ (setdefault: won't override
    already-exported variables). If LLM_API_KEY is unset, copy HF_TOKEN so one Hub
    token works for both inference.py and the video LLM.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    path = _root / ".env"
    if not path.is_file():
        _sync_llm_key_from_hf_token()
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if v[:1] in ('"', "'") and len(v) >= 2 and v[-1] == v[0]:
            v = v[1:-1]
        if k:
            os.environ.setdefault(k, v)
    _sync_llm_key_from_hf_token()


def _sync_llm_key_from_hf_token() -> None:
    if not (os.environ.get("LLM_API_KEY") or "").strip():
        t = (os.environ.get("HF_TOKEN") or "").strip()
        if t:
            os.environ["LLM_API_KEY"] = t
