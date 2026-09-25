from collections.abc import Sequence
from typing import Protocol, TypeAlias


TensorScalar: TypeAlias = int | float | bool | str
TensorData: TypeAlias = (
    Sequence[TensorScalar]
    | Sequence[Sequence[TensorScalar]]
)


class InferenceInputPort(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def shape(self) -> Sequence[int]: ...

    @property
    def datatype(self) -> str: ...

    @property
    def data(self) -> TensorData: ...
