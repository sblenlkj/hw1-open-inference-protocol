from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    server_name: str = "titanic-inference-server"
    server_version: str = "0.1.0"

    model_name: str = "titanic"
    model_version: str = "1"
    model_platform: str = "onnx_onnxv1"
    model_path: str = "models/titanic.onnx"


settings = Settings()
