from concurrent.futures import ThreadPoolExecutor

import grpc

from inference_server.api.grpc.service import InferenceGrpcService
from inference_server.application.runtime.model_runtime import ModelRuntime
from inference_server.core.settings import Settings
from inference_server.grpc_generated import open_inference_grpc_pb2_grpc as pb2_grpc


def create_grpc_server(
    runtime: ModelRuntime,
    settings: Settings,
) -> grpc.Server:
    server = grpc.server(ThreadPoolExecutor())

    pb2_grpc.add_GRPCInferenceServiceServicer_to_server(
        InferenceGrpcService(runtime, settings),
        server,
    )

    server.add_insecure_port(
        f"{settings.grpc_host}:{settings.grpc_port}"
    )

    return server
