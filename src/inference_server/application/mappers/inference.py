import numpy as np
from numpy.typing import NDArray

from inference_server.api.schemas import (
    InferenceRequest,
    InferenceResponse,
    InferenceTensorResponse,
)
from inference_server.application.ports.runtime import RuntimeTensorPort


OPEN_INFERENCE_TO_NUMPY: dict[str, np.dtype] = {
    "FP32": np.dtype(np.float32),
    "FP64": np.dtype(np.float64),
    "INT64": np.dtype(np.int64),
    "INT32": np.dtype(np.int32),
    "BOOL": np.dtype(np.bool_),
    "UINT8": np.dtype(np.uint8),
}

RUNTIME_TO_OPEN_INFERENCE: dict[str, str] = {
    "tensor(float)": "FP32",
    "tensor(double)": "FP64",
    "tensor(int64)": "INT64",
    "tensor(int32)": "INT32",
    "tensor(bool)": "BOOL",
    "tensor(uint8)": "UINT8",
}

NUMPY_TO_OPEN_INFERENCE: dict[np.dtype, str] = {
    np.dtype(np.float32): "FP32",
    np.dtype(np.float64): "FP64",
    np.dtype(np.int64): "INT64",
    np.dtype(np.int32): "INT32",
    np.dtype(np.bool_): "BOOL",
    np.dtype(np.uint8): "UINT8",
}


def _validate_shape(
    request_shape: list[int],
    runtime_shape: list[int | str | None],
) -> None:
    if len(request_shape) != len(runtime_shape):
        raise ValueError(
            f"Expected rank {len(runtime_shape)}, got {len(request_shape)}"
        )

    for index, (actual, expected) in enumerate(
        zip(request_shape, runtime_shape, strict=True)
    ):
        if isinstance(expected, int) and actual != expected:
            raise ValueError(
                f"Dimension {index} must be {expected}, got {actual}"
            )


def request_to_numpy(
    request: InferenceRequest,
    expected_inputs: list[RuntimeTensorPort],
) -> dict[str, NDArray[np.generic]]:
    expected_by_name = {item.name: item for item in expected_inputs}
    input_names = [item.name for item in request.inputs]
    actual_names = set(input_names)
    expected_names = set(expected_by_name)

    if len(input_names) != len(actual_names):
        raise ValueError("Duplicate input names are not allowed")

    if actual_names != expected_names:
        raise ValueError(
            f"Expected inputs {sorted(expected_names)}, got {sorted(actual_names)}"
        )

    feeds: dict[str, NDArray[np.generic]] = {}

    for tensor in request.inputs:
        expected = expected_by_name[tensor.name]

        expected_datatype = RUNTIME_TO_OPEN_INFERENCE.get(expected.type)
        if expected_datatype is None:
            raise ValueError(f"Unsupported runtime datatype: {expected.type}")

        if tensor.datatype != expected_datatype:
            raise ValueError(
                f"Input {tensor.name!r} expects datatype "
                f"{expected_datatype}, got {tensor.datatype}"
            )

        _validate_shape(tensor.shape, list(expected.shape))

        dtype = OPEN_INFERENCE_TO_NUMPY.get(tensor.datatype)
        if dtype is None:
            raise ValueError(f"Unsupported datatype: {tensor.datatype}")

        array = np.asarray(tensor.data, dtype=dtype)

        expected_size = int(np.prod(tensor.shape))
        if array.size != expected_size:
            raise ValueError(
                f"Tensor {tensor.name!r} has {array.size} values, "
                f"but shape {tensor.shape} requires {expected_size}"
            )

        feeds[tensor.name] = array.reshape(tensor.shape)

    return feeds


def numpy_outputs_to_response(
    *,
    model_name: str,
    model_version: str,
    request_id: str | None,
    output_names: list[str],
    outputs: list[NDArray[np.generic]],
) -> InferenceResponse:
    response_outputs: list[InferenceTensorResponse] = []

    for name, array in zip(output_names, outputs, strict=True):
        datatype = NUMPY_TO_OPEN_INFERENCE.get(array.dtype)
        if datatype is None:
            raise ValueError(f"Unsupported output dtype: {array.dtype}")

        response_outputs.append(
            InferenceTensorResponse(
                name=name,
                shape=list(array.shape),
                datatype=datatype,
                data=array.reshape(-1).tolist(),
            )
        )

    return InferenceResponse(
        model_name=model_name,
        model_version=model_version,
        id=request_id,
        outputs=response_outputs,
    )
