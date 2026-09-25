from pathlib import Path

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "titanic.onnx"


def main() -> None:
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )

    print("=== RUNTIME INPUTS ===")
    for item in session.get_inputs():
        print(f"name={item.name!r}, type={item.type}, shape={item.shape}")

    print("\n=== RUNTIME OUTPUTS ===")
    for item in session.get_outputs():
        print(f"name={item.name!r}, type={item.type}, shape={item.shape}")

    # Features:
    # [pclass, age, sibsp, parch, fare, sex_male]
    batch = np.array(
        [
            [1.0, 29.0, 0.0, 0.0, 100.0, 0.0],
            [3.0, 30.0, 0.0, 0.0, 8.0, 1.0],
        ],
        dtype=np.float32,
    )

    labels, probabilities = session.run(
        ["label", "probabilities"],
        {"input": batch},
    )

    print("\n=== INPUT BATCH ===")
    print(batch)

    print("\n=== LABELS ===")
    print(labels)

    print("\n=== PROBABILITIES ===")
    print(probabilities)


if __name__ == "__main__":
    main()
