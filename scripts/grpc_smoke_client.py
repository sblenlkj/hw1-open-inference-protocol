import grpc

from inference_server.grpc_generated import open_inference_grpc_pb2 as pb2
from inference_server.grpc_generated import open_inference_grpc_pb2_grpc as pb2_grpc


GRPC_TARGET = "localhost:8081"


def main() -> None:
    with grpc.insecure_channel(GRPC_TARGET) as channel:
        stub = pb2_grpc.GRPCInferenceServiceStub(channel)

        ready = stub.ServerReady(
            pb2.ServerReadyRequest(),
            timeout=5,
        )
        print("=== SERVER READY ===")
        print(ready)

        request = pb2.ModelInferRequest(
            model_name="titanic",
            model_version="1",
            id="grpc-smoke",
            inputs=[
                pb2.ModelInferRequest.InferInputTensor(
                    name="input",
                    datatype="FP32",
                    shape=[2, 6],
                    contents=pb2.InferTensorContents(
                        fp32_contents=[
                            1.0,
                            29.0,
                            0.0,
                            0.0,
                            100.0,
                            0.0,
                            3.0,
                            30.0,
                            0.0,
                            0.0,
                            8.0,
                            1.0,
                        ]
                    ),
                )
            ],
        )

        response = stub.ModelInfer(request, timeout=5)

        print("\n=== MODEL INFER ===")
        print(response)


if __name__ == "__main__":
    main()
