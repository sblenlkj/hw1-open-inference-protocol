from fastapi.testclient import TestClient


def test_server_metadata(client: TestClient) -> None:
    response = client.get("/v2")

    assert response.status_code == 200
    assert response.json() == {
        "name": "titanic-inference-server",
        "version": "0.1.0",
        "extensions": [],
    }


def test_model_metadata(client: TestClient) -> None:
    response = client.get("/v2/models/titanic")

    assert response.status_code == 200
    assert response.json() == {
        "name": "titanic",
        "versions": ["1"],
        "platform": "onnx_onnxv1",
        "inputs": [
            {
                "name": "input",
                "datatype": "FP32",
                "shape": [-1, 6],
            }
        ],
        "outputs": [
            {
                "name": "label",
                "datatype": "INT64",
                "shape": [-1],
            },
            {
                "name": "probabilities",
                "datatype": "FP32",
                "shape": [-1, 2],
            },
        ],
    }


def test_unknown_model_metadata(client: TestClient) -> None:
    response = client.get("/v2/models/unknown")

    assert response.status_code == 404
