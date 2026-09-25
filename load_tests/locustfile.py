from locust import HttpUser, between, task


INFERENCE_PAYLOAD = {
    "id": "load-test",
    "inputs": [
        {
            "name": "input",
            "shape": [1, 6],
            "datatype": "FP32",
            "data": [[1, 29, 0, 0, 100, 0]],
        }
    ],
}


class InferenceUser(HttpUser):
    wait_time = between(0, 0)

    @task
    def infer(self) -> None:
        with self.client.post(
            "/v2/models/titanic/infer",
            json=INFERENCE_PAYLOAD,
            name="/v2/models/{model}/infer",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"Unexpected status: {response.status_code}"
                )
