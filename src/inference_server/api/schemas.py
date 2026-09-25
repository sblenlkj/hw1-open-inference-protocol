from typing import Annotated, TypeAlias

from pydantic import BaseModel, Field


TensorScalar: TypeAlias = int | float | bool | str
TensorData: TypeAlias = list[TensorScalar] | list[list[TensorScalar]]


class ServerMetadataResponse(BaseModel):
    name: str
    version: str
    extensions: list[str]


class LiveResponse(BaseModel):
    live: bool


class ModelReadyResponse(BaseModel):
    name: str
    ready: bool


class TensorMetadata(BaseModel):
    name: str
    datatype: str
    shape: list[int]


class ModelMetadataResponse(BaseModel):
    name: str
    versions: list[str]
    platform: str
    inputs: list[TensorMetadata]
    outputs: list[TensorMetadata]


ShapeDimension: TypeAlias = Annotated[int, Field(ge=0)]


class InferenceTensorRequest(BaseModel):
    name: str
    shape: list[ShapeDimension]
    datatype: str
    data: TensorData


class InferenceRequest(BaseModel):
    id: str | None = None
    inputs: list[InferenceTensorRequest]


class InferenceTensorResponse(BaseModel):
    name: str
    shape: list[int]
    datatype: str
    data: list[TensorScalar]


class InferenceResponse(BaseModel):
    model_name: str
    model_version: str
    id: str | None = None
    outputs: list[InferenceTensorResponse]
