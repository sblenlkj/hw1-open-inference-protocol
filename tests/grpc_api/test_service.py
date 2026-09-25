import grpc
import pytest

from inference_server.grpc_generated import open_inference_grpc_pb2 as pb2
from inference_server.grpc_generated import open_inference_grpc_pb2_grpc as pb2_grpc


def test_server_ready(
    grpc_stub: pb2_grpc.GRPCInferenceServiceStub,
) -> None:
    response = grpc_stub.ServerReady(pb2.ServerReadyRequest())

    assert response.ready is True


def test_model_infer(
    grpc_stub: pb2_grpc.GRPCInferenceServiceStub,
) -> None:
    request = pb2.ModelInferRequest(
        model_name="titanic",
        model_version="1",
        id="grpc-test",
        inputs=[
            pb2.ModelInferRequest.InferInputTensor(
                name="input",
                datatype="FP32",
                shape=[2, 6],
                contents=pb2.InferTensorContents(
                    fp32_contents=[
                        1.0,
                        29.0,
                        0.0,
                        0.0,
                        100.0,
                        0.0,
                        3.0,
                        30.0,
                        0.0,
                        0.0,
                        8.0,
                        1.0,
                    ]
                ),
            )
        ],
    )

    response = grpc_stub.ModelInfer(request)

    assert response.model_name == "titanic"
    assert response.model_version == "1"
    assert response.id == "grpc-test"

    label, probabilities = response.outputs

    assert label.name == "label"
    assert list(label.shape) == [2]
    assert label.datatype == "INT64"
    assert list(label.contents.int64_contents) == [1, 0]

    assert probabilities.name == "probabilities"
    assert list(probabilities.shape) == [2, 2]
    assert probabilities.datatype == "FP32"
    assert len(probabilities.contents.fp32_contents) == 4


def test_unknown_model_returns_not_found(
    grpc_stub: pb2_grpc.GRPCInferenceServiceStub,
) -> None:
    request = pb2.ModelInferRequest(
        model_name="unknown",
        inputs=[
            pb2.ModelInferRequest.InferInputTensor(
                name="input",
                datatype="FP32",
                shape=[1, 6],
                contents=pb2.InferTensorContents(
                    fp32_contents=[1, 29, 0, 0, 100, 0]
                ),
            )
        ],
    )

    with pytest.raises(grpc.RpcError) as exc_info:
        grpc_stub.ModelInfer(request)

    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


def test_wrong_datatype_returns_invalid_argument(
    grpc_stub: pb2_grpc.GRPCInferenceServiceStub,
) -> None:
    request = pb2.ModelInferRequest(
        model_name="titanic",
        inputs=[
            pb2.ModelInferRequest.InferInputTensor(
                name="input",
                datatype="INT64",
                shape=[1, 6],
                contents=pb2.InferTensorContents(
                    int64_contents=[1, 29, 0, 0, 100, 0]
                ),
            )
        ],
    )

    with pytest.raises(grpc.RpcError) as exc_info:
        grpc_stub.ModelInfer(request)

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_wrong_shape_returns_invalid_argument(
    grpc_stub: pb2_grpc.GRPCInferenceServiceStub,
) -> None:
    request = pb2.ModelInferRequest(
        model_name="titanic",
        inputs=[
            pb2.ModelInferRequest.InferInputTensor(
                name="input",
                datatype="FP32",
                shape=[1, 5],
                contents=pb2.InferTensorContents(
                    fp32_contents=[1, 29, 0, 0, 100]
                ),
            )
        ],
    )

    with pytest.raises(grpc.RpcError) as exc_info:
        grpc_stub.ModelInfer(request)

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
