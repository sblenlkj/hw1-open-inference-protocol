from fastapi import APIRouter, HTTPException

from inference_server.api.dependencies import ModelRuntimeDep, SettingsDep
from inference_server.api.schemas import (
    InferenceRequest,
    InferenceResponse,
    LiveResponse,
    ModelMetadataResponse,
    ModelReadyResponse,
    ServerMetadataResponse,
)
from inference_server.application.mappers.inference import (
    numpy_outputs_to_response,
    request_to_numpy,
)
from inference_server.application.mappers.metadata import map_runtime_tensor

router = APIRouter()


@router.get("/v2", response_model=ServerMetadataResponse)
def server_metadata(settings: SettingsDep) -> ServerMetadataResponse:
    return ServerMetadataResponse(
        name=settings.server_name,
        version=settings.server_version,
        extensions=[],
    )


@router.get("/v2/health/live", response_model=LiveResponse)
def server_live() -> LiveResponse:
    return LiveResponse(live=True)


@router.get("/v2/health/ready", response_model=LiveResponse)
def server_ready(runtime: ModelRuntimeDep) -> LiveResponse:
    return LiveResponse(live=runtime.is_loaded)


@router.get(
    "/v2/models/{model_name}/ready",
    response_model=ModelReadyResponse,
)
def model_ready(
    model_name: str,
    runtime: ModelRuntimeDep,
    settings: SettingsDep,
) -> ModelReadyResponse:
    if model_name != settings.model_name:
        raise HTTPException(status_code=404, detail="Model not found")

    return ModelReadyResponse(
        name=settings.model_name,
        ready=runtime.is_loaded,
    )


@router.post(
    "/v2/models/{model_name}/infer",
    response_model=InferenceResponse,
)
def infer(
    model_name: str,
    payload: InferenceRequest,
    runtime: ModelRuntimeDep,
    settings: SettingsDep,
) -> InferenceResponse:
    if model_name != settings.model_name:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        feeds = request_to_numpy(
            payload,
            expected_inputs=list(runtime.session.get_inputs()),
        )
        output_names, outputs = runtime.infer(feeds)
        return numpy_outputs_to_response(
            model_name=settings.model_name,
            model_version=settings.model_version,
            request_id=payload.id,
            output_names=output_names,
            outputs=outputs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/v2/models/{model_name}",
    response_model=ModelMetadataResponse,
)
def model_metadata(
    model_name: str,
    runtime: ModelRuntimeDep,
    settings: SettingsDep,
) -> ModelMetadataResponse:
    if model_name != settings.model_name:
        raise HTTPException(status_code=404, detail="Model not found")

    session = runtime.session

    return ModelMetadataResponse(
        name=settings.model_name,
        versions=[settings.model_version],
        platform=settings.model_platform,
        inputs=[map_runtime_tensor(item) for item in session.get_inputs()],
        outputs=[map_runtime_tensor(item) for item in session.get_outputs()],
    )
