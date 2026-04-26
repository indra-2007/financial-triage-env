"""OpenEnv HTTP compliance smoke test.

Hits the shipping contract endpoints on a running OpenEnv server and asserts
the response shapes. Meant to be one of the first things a judge runs against
a deployed Space to sanity-check that the server actually speaks OpenEnv.

Run locally:   uvicorn server.app:app --port 7860  &  python -m scripts.check_openenv --base http://localhost:7860
Run on Space:  python -m scripts.check_openenv --base https://indra-dhanush-financial-triage-env.hf.space

Exit code 0 on success, 1 on any failed assertion. No external deps beyond
`requests` (already in requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

try:
    import requests
except ImportError:
    sys.stderr.write("requests not installed; run `pip install requests`\n")
    raise SystemExit(2)


_EXPECTED_OBS_KEYS = {
    "account", "bills", "debts", "risk", "current_day", "episode_length",
}


def _assert(ok: bool, label: str, detail: str = "") -> bool:
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {label}" + (f"   {detail}" if detail else ""))
    return ok


def run(base: str) -> int:
    base = base.rstrip("/")
    all_ok = True

    print(f"\n=== OpenEnv compliance check — {base} ===")
    sess = requests.Session()

    r = sess.get(f"{base}/health", timeout=30)
    all_ok &= _assert(r.status_code == 200, "/health returns 200", r.text[:80])

    r = sess.get(f"{base}/openapi.json", timeout=30)
    try:
        paths = set(r.json().get("paths", {}).keys())
    except Exception as e:
        paths = set()
        print(f"  openapi.json parse error: {e}")
    for p in ("/reset", "/step", "/state"):
        all_ok &= _assert(p in paths, f"openapi declares {p}")

    r = sess.post(f"{base}/reset", json={"task_id": "easy", "seed": 0}, timeout=60)
    all_ok &= _assert(r.status_code == 200, "POST /reset -> 200")
    body: Dict[str, Any] = r.json()
    obs = body.get("observation") or {}
    missing = _EXPECTED_OBS_KEYS - set(obs.keys())
    all_ok &= _assert(not missing, "observation has expected keys", f"missing={sorted(missing)}")
    all_ok &= _assert(
        isinstance(obs.get("episode_length"), int) and obs["episode_length"] > 0,
        "observation.episode_length is a positive int",
        str(obs.get("episode_length")),
    )

    step_payload = {"action": {"action_type": "do_nothing"}}
    r = sess.post(f"{base}/step", json=step_payload, timeout=60)
    all_ok &= _assert(r.status_code == 200,
                      "POST /step -> 200", f"body={r.text[:120]}")
    try:
        sb: Dict[str, Any] = r.json()
    except Exception:
        sb = {}
    all_ok &= _assert(
        isinstance(sb.get("reward"), (int, float)),
        "step returns a numeric reward",
        f"got {type(sb.get('reward')).__name__}",
    )
    all_ok &= _assert("done" in sb, "step returns a done flag")

    r = sess.get(f"{base}/state", timeout=30)
    all_ok &= _assert(r.status_code == 200, "GET /state -> 200", str(r.status_code))

    api_demo = f"{base}/api/demo"
    r = sess.post(f"{api_demo}/reset", json={"task_id": "hard", "seed": 1}, timeout=60)
    all_ok &= _assert(r.status_code == 200, "pitch API: POST /api/demo/reset -> 200")
    r = sess.post(f"{api_demo}/heuristic", timeout=60)
    all_ok &= _assert(r.status_code == 200, "pitch API: POST /api/demo/heuristic -> 200")
    if r.status_code == 200:
        action_body = {"action": r.json()["action"]}
        r = sess.post(f"{api_demo}/step", json=action_body, timeout=60)
        all_ok &= _assert(r.status_code == 200, "pitch API: POST /api/demo/step -> 200")
    r = sess.get(f"{api_demo}/score", timeout=30)
    all_ok &= _assert(r.status_code == 200, "pitch API: GET /api/demo/score -> 200")

    print(f"\n{'OK' if all_ok else 'FAILURES'} — {base}")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="Root URL of the OpenEnv server, e.g. https://.../hf.space")
    args = ap.parse_args()
    return run(args.base)


if __name__ == "__main__":
    raise SystemExit(main())
