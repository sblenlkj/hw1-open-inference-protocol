from fastapi.testclient import TestClient


def test_server_live(client: TestClient) -> None:
    response = client.get("/v2/health/live")

    assert response.status_code == 200
    assert response.json() == {"live": True}


def test_server_ready(client: TestClient) -> None:
    response = client.get("/v2/health/ready")

    assert response.status_code == 200
    assert response.json() == {"live": True}


def test_model_ready(client: TestClient) -> None:
    response = client.get("/v2/models/titanic/ready")

    assert response.status_code == 200
    assert response.json() == {
        "name": "titanic",
        "ready": True,
    }


def test_unknown_model_is_not_ready(client: TestClient) -> None:
    response = client.get("/v2/models/unknown/ready")

    assert response.status_code == 404
