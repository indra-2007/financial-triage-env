# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the My Env Environment.

This module creates an HTTP server that exposes the MyEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import MyAction, MyObservation
    from .my_env_environment import MyEnvironment
except ImportError:
    from models import MyAction, MyObservation
    from server.my_env_environment import MyEnvironment

try:
    from .video_demo_server import include_video_demo_in_app
except ImportError:
    from server.video_demo_server import include_video_demo_in_app

from fastapi import Request
from fastapi.responses import RedirectResponse


# Create the app with web interface and README integration
app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="financial_triage",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# Stateful pitch UI + /api/* (local session) — not in OpenEnv's stateless HTTP /step
include_video_demo_in_app(app, static_at="/demo", include_cors=False)


@app.get("/")
def read_root(request: Request):
    # Browsers: land on the interactive demo. API clients: keep JSON.
    if "text/html" in (request.headers.get("accept") or ""):
        return RedirectResponse(url="/demo/", status_code=302)
    return {"status": "ok", "message": "Environment is running on Hugging Face Spaces"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 7861
        python -m my_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 7860)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn my_env.server.app:app --host 0.0.0.0 --port 7860 --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
