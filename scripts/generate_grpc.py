from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTO_ROOT = PROJECT_ROOT / "proto"
OUTPUT_ROOT = PROJECT_ROOT / "src"
PROTO_FILE = (
    PROTO_ROOT
    / "inference_server"
    / "grpc_generated"
    / "open_inference_grpc.proto"
)


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{PROTO_ROOT}",
            f"--python_out={OUTPUT_ROOT}",
            f"--pyi_out={OUTPUT_ROOT}",
            f"--grpc_python_out={OUTPUT_ROOT}",
            str(PROTO_FILE),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
