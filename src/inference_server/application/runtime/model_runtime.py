from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray


class ModelRuntime:
    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._session: ort.InferenceSession | None = None

    def load(self) -> None:
        self._session = ort.InferenceSession(
            self._model_path,
            providers=["CPUExecutionProvider"],
        )

    def unload(self) -> None:
        self._session = None

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def session(self) -> ort.InferenceSession:
        if self._session is None:
            raise RuntimeError("Model is not loaded")
        return self._session

    def infer(
        self,
        feeds: dict[str, NDArray[np.generic]],
    ) -> tuple[list[str], list[NDArray[np.generic]]]:
        session = self.session
        output_names = [item.name for item in session.get_outputs()]
        outputs = session.run(output_names, feeds)
        return output_names, outputs
