from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from inference_server.application.mappers.inference import (
    NUMPY_TO_OPEN_INFERENCE,
)
from inference_server.application.ports.inference import TensorData
from inference_server.grpc_generated import open_inference_grpc_pb2 as pb2


@dataclass(frozen=True, slots=True)
class GrpcInferenceInput:
    name: str
    shape: tuple[int, ...]
    datatype: str
    data: TensorData


def request_input_to_port(
    tensor: pb2.ModelInferRequest.InferInputTensor,
) -> GrpcInferenceInput:
    if tensor.parameters:
        raise NotImplementedError("Input parameters are not supported")

    datatype = tensor.datatype

    if datatype == "FP32":
        data = list(tensor.contents.fp32_contents)
    elif datatype == "FP64":
        data = list(tensor.contents.fp64_contents)
    elif datatype == "INT64":
        data = list(tensor.contents.int64_contents)
    elif datatype in {"INT8", "INT16", "INT32"}:
        data = list(tensor.contents.int_contents)
    elif datatype in {"UINT8", "UINT16", "UINT32"}:
        data = list(tensor.contents.uint_contents)
    elif datatype == "UINT64":
        data = list(tensor.contents.uint64_contents)
    elif datatype == "BOOL":
        data = list(tensor.contents.bool_contents)
    elif datatype == "BYTES":
        data = list(tensor.contents.bytes_contents)
    else:
        raise ValueError(f"Unsupported datatype: {datatype}")

    return GrpcInferenceInput(
        name=tensor.name,
        shape=tuple(tensor.shape),
        datatype=datatype,
        data=data,
    )


def numpy_output_to_proto(
    name: str,
    array: NDArray[np.generic],
) -> pb2.ModelInferResponse.InferOutputTensor:
    datatype = NUMPY_TO_OPEN_INFERENCE.get(array.dtype)
    if datatype is None:
        raise ValueError(f"Unsupported output dtype: {array.dtype}")

    output = pb2.ModelInferResponse.InferOutputTensor(
        name=name,
        datatype=datatype,
        shape=list(array.shape),
    )

    values = array.reshape(-1).tolist()

    if datatype == "FP32":
        output.contents.fp32_contents.extend(values)
    elif datatype == "FP64":
        output.contents.fp64_contents.extend(values)
    elif datatype == "INT64":
        output.contents.int64_contents.extend(values)
    elif datatype in {"INT8", "INT16", "INT32"}:
        output.contents.int_contents.extend(values)
    elif datatype in {"UINT8", "UINT16", "UINT32"}:
        output.contents.uint_contents.extend(values)
    elif datatype == "UINT64":
        output.contents.uint64_contents.extend(values)
    elif datatype == "BOOL":
        output.contents.bool_contents.extend(values)
    else:
        raise ValueError(f"Unsupported output datatype: {datatype}")

    return output
