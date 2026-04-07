# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the My Env Environment.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

from models import MyAction, MyObservation
from server.my_env_environment import MyEnvironment


# ✅ Create OpenEnv app (NO prefix)
app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="",   # VERY IMPORTANT → removes /my_env prefix
    max_concurrent_envs=1,
)


# ✅ Wrap with root FastAPI app (HF compatibility)
from fastapi import FastAPI

root_app = FastAPI()
root_app.mount("/", app)

app = root_app


# ✅ Entry point
def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()