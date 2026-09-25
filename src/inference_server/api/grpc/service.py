import grpc

from inference_server.application.mappers.inference import inputs_to_numpy
from inference_server.application.mappers.metadata import map_runtime_tensor
from inference_server.application.runtime.model_runtime import ModelRuntime
from inference_server.core.settings import Settings
from inference_server.grpc_generated import open_inference_grpc_pb2 as pb2
from inference_server.grpc_generated import open_inference_grpc_pb2_grpc as pb2_grpc
from inference_server.api.grpc.mappers import (
    numpy_output_to_proto,
    request_input_to_port,
)


class InferenceGrpcService(pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(self, runtime: ModelRuntime, settings: Settings) -> None:
        self._runtime = runtime
        self._settings = settings

    def ServerLive(
        self,
        request: pb2.ServerLiveRequest,
        context: grpc.ServicerContext,
    ) -> pb2.ServerLiveResponse:
        return pb2.ServerLiveResponse(live=True)

    def ServerReady(
        self,
        request: pb2.ServerReadyRequest,
        context: grpc.ServicerContext,
    ) -> pb2.ServerReadyResponse:
        return pb2.ServerReadyResponse(ready=self._runtime.is_loaded)

    def ModelReady(
        self,
        request: pb2.ModelReadyRequest,
        context: grpc.ServicerContext,
    ) -> pb2.ModelReadyResponse:
        self._validate_model(request.name, request.version, context)
        return pb2.ModelReadyResponse(ready=self._runtime.is_loaded)

    def ServerMetadata(
        self,
        request: pb2.ServerMetadataRequest,
        context: grpc.ServicerContext,
    ) -> pb2.ServerMetadataResponse:
        return pb2.ServerMetadataResponse(
            name=self._settings.server_name,
            version=self._settings.server_version,
            extensions=[],
        )

    def ModelMetadata(
        self,
        request: pb2.ModelMetadataRequest,
        context: grpc.ServicerContext,
    ) -> pb2.ModelMetadataResponse:
        self._validate_model(request.name, request.version, context)

        session = self._runtime.session
        response = pb2.ModelMetadataResponse(
            name=self._settings.model_name,
            versions=[self._settings.model_version],
            platform=self._settings.model_platform,
        )

        for runtime_input in session.get_inputs():
            metadata = map_runtime_tensor(runtime_input)
            response.inputs.add(
                name=metadata.name,
                datatype=metadata.datatype,
                shape=metadata.shape,
            )

        for runtime_output in session.get_outputs():
            metadata = map_runtime_tensor(runtime_output)
            response.outputs.add(
                name=metadata.name,
                datatype=metadata.datatype,
                shape=metadata.shape,
            )

        return response

    def ModelInfer(
        self,
        request: pb2.ModelInferRequest,
        context: grpc.ServicerContext,
    ) -> pb2.ModelInferResponse:
        self._validate_model(
            request.model_name,
            request.model_version,
            context,
        )

        if request.parameters:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "Inference parameters are not supported",
            )

        if request.raw_input_contents:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "raw_input_contents is not supported",
            )

        if request.outputs:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "Requested output filtering is not supported",
            )

        try:
            inputs = [request_input_to_port(item) for item in request.inputs]
            feeds = inputs_to_numpy(
                inputs,
                expected_inputs=list(self._runtime.session.get_inputs()),
            )
            output_names, outputs = self._runtime.infer(feeds)
        except NotImplementedError as exc:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        response = pb2.ModelInferResponse(
            model_name=self._settings.model_name,
            model_version=self._settings.model_version,
            id=request.id,
        )

        for name, array in zip(output_names, outputs, strict=True):
            response.outputs.append(numpy_output_to_proto(name, array))

        return response

    def _validate_model(
        self,
        name: str,
        version: str,
        context: grpc.ServicerContext,
    ) -> None:
        if name != self._settings.model_name:
            context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")

        if version and version != self._settings.model_version:
            context.abort(grpc.StatusCode.NOT_FOUND, "Model version not found")
