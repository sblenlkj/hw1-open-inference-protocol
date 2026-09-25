from collections.abc import Sequence
from typing import Protocol


class RuntimeTensorPort(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def type(self) -> str: ...

    @property
    def shape(self) -> Sequence[int | str | None]: ...
