from pathlib import Path

import onnx
from onnx import TensorProto

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "titanic.onnx"


def tensor_type_name(elem_type: int) -> str:
    return TensorProto.DataType.Name(elem_type)


def value_info_to_str(value_info: onnx.ValueInfoProto) -> str:
    tensor_type = value_info.type.tensor_type
    shape = []

    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(str(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append("?")

    return (
        f"name={value_info.name!r}, "
        f"dtype={tensor_type_name(tensor_type.elem_type)}, "
        f"shape=[{', '.join(shape)}]"
    )


def main() -> None:
    model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(model)

    graph = model.graph

    print("Model is valid ONNX.\n")

    print("=== INPUTS ===")
    for item in graph.input:
        print(value_info_to_str(item))

    print("\n=== OUTPUTS ===")
    for item in graph.output:
        print(value_info_to_str(item))

    print("\n=== GRAPH NODES ===")
    for index, node in enumerate(graph.node, start=1):
        print(
            f"{index:02d}. op={node.op_type!r} "
            f"name={node.name!r} "
            f"inputs={list(node.input)} "
            f"outputs={list(node.output)}"
        )

        if node.attribute:
            print("    attributes:")
            for attribute in node.attribute:
                value = onnx.helper.get_attribute_value(attribute)
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                print(f"      - {attribute.name} = {value}")

    print(f"\nTotal nodes: {len(graph.node)}")
    print(f"Initializers (stored tensors/weights): {len(graph.initializer)}")


if __name__ == "__main__":
    main()
