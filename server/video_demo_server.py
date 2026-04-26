# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# Stateful demo API for the YouTube / pitch UI. OpenEnv's default HTTP /reset
# and /step on this repo create a fresh environment per request, so a local
# session is required for a multi-day walkthrough. Run:
#   python -m server.video_demo_server
# then open http://127.0.0.1:8088/
#
# For production / Spaces, use the standard app on 7860 or WebSocket client.

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from models import FinancialAction, ActionType, FinancialObservation
from server.my_env_environment import MyEnvironment

from inference import _heuristic_action, format_action

_lock = threading.Lock()
_env: Optional[MyEnvironment] = None
_last_obs: Optional[FinancialObservation] = None

STATIC = _ROOT / "video_demo"

app = FastAPI(
    title="Financial Triage — video demo (stateful)", version="0.1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResetBody(BaseModel):
    task_id: str = Field(default="medium", description="easy | medium | hard")
    seed: int = Field(default=42, ge=0)


class StepBody(BaseModel):
    action: Dict[str, Any]


def _get_env() -> MyEnvironment:
    if _env is None:
        raise HTTPException(400, "Call /api/demo/reset first")
    return _env


def _obs_to_json(obs: FinancialObservation) -> Dict[str, Any]:
    d = obs.model_dump()
    d["action_type_suggestions"] = [a.value for a in ActionType]
    return d


@app.post("/api/demo/reset")
def demo_reset(body: ResetBody) -> Dict[str, Any]:
    global _env, _last_obs
    with _lock:
        _env = MyEnvironment()
        obs = _env.reset(seed=body.seed, task_id=body.task_id)
        _last_obs = obs
        return {
            "observation": _obs_to_json(obs),
            "reward": obs.reward,
            "done": obs.done,
            "task_id": body.task_id,
            "seed": body.seed,
        }


@app.post("/api/demo/step")
def demo_step(body: StepBody) -> Dict[str, Any]:
    global _last_obs
    with _lock:
        e = _get_env()
        try:
            action = FinancialAction.model_validate(body.action)
        except Exception as ex:
            raise HTTPException(422, str(ex)) from ex
        obs = e.step(action)
        _last_obs = obs
        out: Dict[str, Any] = {
            "observation": _obs_to_json(obs),
            "reward": obs.reward,
            "done": obs.done,
        }
        if obs.done:
            out["episode_score"] = e.get_episode_score()
        return out


@app.post("/api/demo/heuristic")
def demo_heuristic() -> Dict[str, Any]:
    """Return the rule-based advisor action for the *current* observation (same as training baseline)."""
    with _lock:
        _get_env()
        if _last_obs is None:
            raise HTTPException(400, "No observation; reset first")
        if getattr(_last_obs, "done", False):
            raise HTTPException(400, "Episode finished; reset to continue")
        act = _heuristic_action(_last_obs)
        return {
            "action": act.model_dump(mode="json"),
            "action_text": format_action(act),
        }


@app.get("/api/demo/score")
def demo_score() -> Dict[str, Any]:
    with _lock:
        e = _get_env()
        if not e:
            raise HTTPException(400, "No episode")
        done = e._current_day >= e._episode_length
        if not done:
            return {"ready": False, "episode_score": None}
        return {"ready": True, "episode_score": e.get_episode_score()}


if STATIC.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(STATIC), html=True),
        name="static",
    )


def main() -> None:
    import uvicorn

    host = os.environ.get("VIDEO_DEMO_HOST", "127.0.0.1")
    port = int(os.environ.get("VIDEO_DEMO_PORT", "8088"))
    uvicorn.run("server.video_demo_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
