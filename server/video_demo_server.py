# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# Stateful demo API for the YouTube / pitch UI. OpenEnv's default HTTP /reset
# and /step on this repo create a fresh environment per request, so a local
# session is required for a multi-day walkthrough. Run:
#   python -m server.video_demo_server
# then open http://127.0.0.1:8088/
#
# The same /api/demo/* API is included on server.app (Hugging Face Space); the
# static UI is served at /demo/ there, and at / for this standalone server.

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import FinancialAction, ActionType, FinancialObservation
from server.my_env_environment import MyEnvironment

from inference import _heuristic_action, format_action

_lock = threading.Lock()
_env: Optional[MyEnvironment] = None
_last_obs: Optional[FinancialObservation] = None

STATIC = _ROOT / "video_demo"


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


def build_demo_api_router() -> APIRouter:
    r = APIRouter(prefix="/api/demo", tags=["stateful-pitch-demo"])

    @r.post("/reset")
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

    @r.post("/step")
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

    @r.post("/heuristic")
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

    @r.get("/score")
    def demo_score() -> Dict[str, Any]:
        with _lock:
            e = _get_env()
            if not e:
                raise HTTPException(400, "No episode")
            done = e._current_day >= e._episode_length
            if not done:
                return {"ready": False, "episode_score": None}
            return {"ready": True, "episode_score": e.get_episode_score()}

    return r


def _mount_static(app: FastAPI, at_path: str) -> None:
    if not STATIC.is_dir():
        return
    app.mount(
        at_path,
        StaticFiles(directory=str(STATIC), html=True),
        name="video_demo" + ("" if at_path == "/" else "_nested"),
    )


def include_video_demo_in_app(
    app: FastAPI, *, static_at: str, include_cors: bool = True
) -> None:
    """Attach stateful /api/demo/* and the pitch UI static files to an existing app.

    static_at: \"/\" for a dedicated local server, \"/demo\" (no trailing slash) on HF Space.
    """
    if include_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(build_demo_api_router())
    _mount_static(app, at_path=static_at if static_at == "/" else static_at)


def _create_standalone_app() -> FastAPI:
    app = FastAPI(title="Financial Triage — video demo (stateful)", version="0.1.0")
    include_video_demo_in_app(app, static_at="/", include_cors=True)
    return app


app = _create_standalone_app()


def main() -> None:
    import uvicorn

    host = os.environ.get("VIDEO_DEMO_HOST", "127.0.0.1")
    port = int(os.environ.get("VIDEO_DEMO_PORT", "8088"))
    uvicorn.run("server.video_demo_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
