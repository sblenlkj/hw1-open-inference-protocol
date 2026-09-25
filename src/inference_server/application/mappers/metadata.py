from inference_server.api.schemas import TensorMetadata
from inference_server.application.ports.runtime import RuntimeTensorPort


ONNX_TO_OPEN_INFERENCE_DATATYPE: dict[str, str] = {
    "tensor(float)": "FP32",
    "tensor(double)": "FP64",
    "tensor(int64)": "INT64",
    "tensor(int32)": "INT32",
    "tensor(bool)": "BOOL",
    "tensor(uint8)": "UINT8",
}


def _map_dimension(dimension: int | str | None) -> int:
    if isinstance(dimension, int):
        return dimension
    return -1


def map_runtime_tensor(tensor: RuntimeTensorPort) -> TensorMetadata:
    datatype = ONNX_TO_OPEN_INFERENCE_DATATYPE.get(tensor.type)
    if datatype is None:
        raise ValueError(
            f"Unsupported runtime tensor type: {tensor.type}"
        )

    return TensorMetadata(
        name=tensor.name,
        datatype=datatype,
        shape=[_map_dimension(dimension) for dimension in tensor.shape],
    )
