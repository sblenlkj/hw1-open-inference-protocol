# Open Inference Protocol server

Small inference service implementing the [KServe Open Inference Protocol](https://github.com/kserve/open-inference-protocol).

The service uses a small Titanic classification model in ONNX format and exposes the standard V2 REST endpoints for metadata, health/readiness, and inference.

## Run locally

Install dependencies and start the FastAPI/Uvicorn server:

```bash
uv sync
uv run inference-server
```

The service will be available at:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

## Run with Docker

`requirements.txt` is generated from `uv.lock` and is not stored in Git.

Export production dependencies:

```bash
uv export \
  --frozen \
  --no-dev \
  --no-emit-project \
  -o requirements.txt
```

Build and run the image:

```bash
docker build -t inference-server:local .
docker run --rm -p 8000:8000 inference-server:local
```

Check readiness:

```bash
curl http://localhost:8000/v2/health/ready
```
