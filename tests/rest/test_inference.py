from fastapi.testclient import TestClient


VALID_PAYLOAD = {
    "id": "test-1",
    "inputs": [
        {
            "name": "input",
            "shape": [2, 6],
            "datatype": "FP32",
            "data": [
                [1, 29, 0, 0, 100, 0],
                [3, 30, 0, 0, 8, 1],
            ],
        }
    ],
}


def test_infer_batch(client: TestClient) -> None:
    response = client.post(
        "/v2/models/titanic/infer",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    body = response.json()
    assert body["model_name"] == "titanic"
    assert body["model_version"] == "1"
    assert body["id"] == "test-1"

    label, probabilities = body["outputs"]

    assert label == {
        "name": "label",
        "shape": [2],
        "datatype": "INT64",
        "data": [1, 0],
    }

    assert probabilities["name"] == "probabilities"
    assert probabilities["shape"] == [2, 2]
    assert probabilities["datatype"] == "FP32"
    assert len(probabilities["data"]) == 4


def test_infer_rejects_wrong_data_size(client: TestClient) -> None:
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [2, 6],
                "datatype": "FP32",
                "data": [1, 2, 3],
            }
        ]
    }

    response = client.post(
        "/v2/models/titanic/infer",
        json=payload,
    )

    assert response.status_code == 400
    assert "requires 12" in response.json()["detail"]


def test_infer_rejects_unsupported_datatype(client: TestClient) -> None:
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 6],
                "datatype": "BYTES",
                "data": ["a", "b", "c", "d", "e", "f"],
            }
        ]
    }

    response = client.post(
        "/v2/models/titanic/infer",
        json=payload,
    )

    assert response.status_code == 400
    assert "expects datatype FP32" in response.json()["detail"]


def test_infer_rejects_unknown_input_name(client: TestClient) -> None:
    payload = {
        "inputs": [
            {
                "name": "wrong",
                "shape": [1, 6],
                "datatype": "FP32",
                "data": [1, 29, 0, 0, 100, 0],
            }
        ]
    }

    response = client.post(
        "/v2/models/titanic/infer",
        json=payload,
    )

    assert response.status_code == 400
    assert "Expected inputs" in response.json()["detail"]


def test_infer_rejects_wrong_feature_dimension(client: TestClient) -> None:
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 5],
                "datatype": "FP32",
                "data": [1, 29, 0, 0, 100],
            }
        ]
    }

    response = client.post(
        "/v2/models/titanic/infer",
        json=payload,
    )

    assert response.status_code == 400
    assert "Dimension 1 must be 6" in response.json()["detail"]


def test_infer_rejects_wrong_input_datatype(client: TestClient) -> None:
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 6],
                "datatype": "INT64",
                "data": [1, 29, 0, 0, 100, 0],
            }
        ]
    }

    response = client.post(
        "/v2/models/titanic/infer",
        json=payload,
    )

    assert response.status_code == 400
    assert "expects datatype FP32" in response.json()["detail"]


def test_infer_rejects_empty_inputs(client: TestClient) -> None:
    response = client.post(
        "/v2/models/titanic/infer",
        json={"inputs": []},
    )

    assert response.status_code == 400
    assert "Expected inputs" in response.json()["detail"]


def test_infer_rejects_duplicate_input_names(client: TestClient) -> None:
    tensor = {
        "name": "input",
        "shape": [1, 6],
        "datatype": "FP32",
        "data": [1, 29, 0, 0, 100, 0],
    }

    response = client.post(
        "/v2/models/titanic/infer",
        json={"inputs": [tensor, tensor]},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Duplicate input names are not allowed"


def test_infer_rejects_negative_shape_dimension(client: TestClient) -> None:
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [-1, 6],
                "datatype": "FP32",
                "data": [1, 29, 0, 0, 100, 0],
            }
        ]
    }

    response = client.post(
        "/v2/models/titanic/infer",
        json=payload,
    )

    assert response.status_code == 422


def test_infer_rejects_missing_inputs_field(client: TestClient) -> None:
    response = client.post(
        "/v2/models/titanic/infer",
        json={"id": "missing-inputs"},
    )

    assert response.status_code == 422


def test_infer_rejects_malformed_json(client: TestClient) -> None:
    response = client.post(
        "/v2/models/titanic/infer",
        content='{"inputs": [',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422


def test_infer_unknown_model(client: TestClient) -> None:
    response = client.post(
        "/v2/models/unknown/infer",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 404
