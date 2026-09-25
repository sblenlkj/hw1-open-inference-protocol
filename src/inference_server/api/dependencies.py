from typing import Annotated, cast

from fastapi import Depends, Request

from inference_server.application.runtime.model_runtime import ModelRuntime
from inference_server.core.settings import Settings, settings


def get_model_runtime(request: Request) -> ModelRuntime:
    return cast(ModelRuntime, request.app.state.model_runtime)


def get_settings() -> Settings:
    return settings


ModelRuntimeDep = Annotated[ModelRuntime, Depends(get_model_runtime)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
