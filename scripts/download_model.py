from pathlib import Path
from urllib.request import urlretrieve

MODEL_URL = (
    "https://huggingface.co/Prahsant/argus-doors-titanic/"
    "resolve/main/model.onnx?download=true"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "titanic.onnx"


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model from:\n{MODEL_URL}")
    urlretrieve(MODEL_URL, MODEL_PATH)

    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"Saved to: {MODEL_PATH}")
    print(f"Size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
