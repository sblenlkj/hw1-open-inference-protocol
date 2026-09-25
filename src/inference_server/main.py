from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from inference_server.api.v2 import router as v2_router
from inference_server.application.runtime.model_runtime import ModelRuntime
from inference_server.core.settings import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = PROJECT_ROOT / settings.model_path

    runtime = ModelRuntime(model_path)
    runtime.load()
    app.state.model_runtime = runtime

    try:
        yield
    finally:
        runtime.unload()


app = FastAPI(
    title="Open Inference Protocol Demo",
    version=settings.server_version,
    lifespan=lifespan,
)

app.include_router(v2_router)


def run() -> None:
    import uvicorn

    uvicorn.run(
        "inference_server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
