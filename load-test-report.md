# Load Test Report

Tool: Locust  
Endpoint: `POST /v2/models/{model}/infer`  
Model: Titanic ONNX  
Batch size: 1  
Input datatype: FP32  
Input shape: `[1, 6]`

| Concurrent users | Requests | RPS | Avg latency, ms | p50, ms | p95, ms | p99, ms | Max, ms | Failures |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12,744 | 489.9 | 2.0 | 2 | 3 | 3 | 62.4 | 0 |
| 5 | 28,280 | 974.7 | 5.0 | 5 | 7 | 10 | 61.4 | 0 |
| 20 | 31,908 | 1,099.6 | 17.6 | 17 | 25 | 42 | 71.3 | 0 |
| 50 | 32,026 | 1,103.3 | 43.8 | 42 | 61 | 85 | 124.8 | 0 |

## Conclusion

The server remained stable in all tested scenarios: no failed requests or Locust exceptions were recorded.

Throughput increased significantly from 1 to 20 concurrent users and reached approximately 1.1k requests per second. Increasing concurrency from 20 to 50 users produced almost no additional throughput, while response latency increased noticeably.

This indicates that around 20 concurrent users the service is already close to the observed throughput ceiling for this test setup.
