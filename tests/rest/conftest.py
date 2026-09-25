from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from inference_server.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
