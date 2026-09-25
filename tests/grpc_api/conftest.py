from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import pytest

from inference_server.api.grpc.service import InferenceGrpcService
from inference_server.application.runtime.model_runtime import ModelRuntime
from inference_server.core.settings import settings
from inference_server.grpc_generated import open_inference_grpc_pb2_grpc as pb2_grpc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / settings.model_path


@pytest.fixture
def grpc_stub():
    runtime = ModelRuntime(MODEL_PATH)
    runtime.load()

    server = grpc.server(ThreadPoolExecutor())
    pb2_grpc.add_GRPCInferenceServiceServicer_to_server(
        InferenceGrpcService(runtime, settings),
        server,
    )

    port = server.add_insecure_port("localhost:0")
    server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = pb2_grpc.GRPCInferenceServiceStub(channel)

    try:
        yield stub
    finally:
        channel.close()
        server.stop(grace=0).wait()
        runtime.unload()
