# Load testing

The load test targets the REST inference endpoint using a valid batch-size-1 request.

Start the production Docker image with the assignment resource limits:

```bash
docker run --rm \
  --name inference-server \
  --cpus=4 \
  --memory=16g \
  -p 8000:8000 \
  -p 8081:8081 \
  inference-server:local
```

Install development dependencies:

```bash
uv sync
```

Run the predefined Locust scenarios:

```bash
bash load_tests/run_load_tests.sh
```

The scenarios use 1, 5, 20, and 50 concurrent users for 30 seconds each.
CSV results are written to `load_tests/results/`.

Key report metrics:

- requests per second;
- average latency;
- p50 latency;
- p95 latency;
- p99 latency;
- request failures.
