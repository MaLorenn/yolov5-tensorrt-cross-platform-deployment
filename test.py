from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch


WARMUP_RUNS = 20
TEST_RUNS = 100
INPUT_SHAPE = (1, 3, 640, 640)

ONNX_PATH = Path("yolov5n.onnx")
FP32_ENGINE_PATH = Path("yolov5n_fp32.engine")
FP16_ENGINE_PATH = Path("yolov5n_fp16.engine")

LOGGER = trt.Logger(trt.Logger.WARNING)


def print_statistics(name, times_ms):
    times_ms = np.asarray(times_ms, dtype=np.float64)

    print("\n" + "=" * 60)
    print(name)
    print(f"平均推理时间：{times_ms.mean():.3f} ms")
    print(f"最小推理时间：{times_ms.min():.3f} ms")
    print(f"最大推理时间：{times_ms.max():.3f} ms")
    print(f"P50 推理时间：{np.percentile(times_ms, 50):.3f} ms")
    print(f"P95 推理时间：{np.percentile(times_ms, 95):.3f} ms")
    print(f"理论吞吐率：{1000.0 / times_ms.mean():.2f} FPS")

    return {
        "name": name,
        "mean_ms": times_ms.mean(),
        "p95_ms": np.percentile(times_ms, 95),
        "fps": 1000.0 / times_ms.mean(),
    }


def benchmark_pytorch(input_tensor):
    model = torch.hub.load(
        ".",
        "custom",
        path="yolov5n.pt",
        source="local",
        autoshape=False
    )

    model = model.to("cuda").eval()

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(input_tensor)

        torch.cuda.synchronize()

        times_ms = []

        for _ in range(TEST_RUNS):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            model(input_tensor)
            end_event.record()

            end_event.synchronize()
            times_ms.append(
                start_event.elapsed_time(end_event)
            )

    return print_statistics(
        "PyTorch CUDA FP32",
        times_ms
    )


def benchmark_onnxruntime(input_array):
    providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=providers
    )

    print("\nONNX Runtime 实际 Providers：")
    print(session.get_providers())

    input_name = session.get_inputs()[0].name

    for _ in range(WARMUP_RUNS):
        session.run(
            None,
            {input_name: input_array}
        )

    times_ms = []

    for _ in range(TEST_RUNS):
        start = time.perf_counter()

        session.run(
            None,
            {input_name: input_array}
        )

        end = time.perf_counter()

        times_ms.append(
            (end - start) * 1000
        )

    return print_statistics(
        "ONNX Runtime CUDA",
        times_ms
    )


class TensorRTEngine:
    def __init__(self, engine_path):
        self.engine_path = Path(engine_path)

        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"没有找到 Engine：{self.engine_path}"
            )

        self.runtime = trt.Runtime(LOGGER)

        with self.engine_path.open("rb") as file:
            self.engine = (
                self.runtime.deserialize_cuda_engine(
                    file.read()
                )
            )

        if self.engine is None:
            raise RuntimeError(
                f"Engine 加载失败：{self.engine_path}"
            )

        self.context = (
            self.engine.create_execution_context()
        )

        self.input_name = None
        self.output_name = None

        for index in range(
            self.engine.num_io_tensors
        ):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name

        self.output_shape = tuple(
            self.engine.get_tensor_shape(
                self.output_name
            )
        )

        self.stream = torch.cuda.Stream()

    def benchmark(self, input_tensor, display_name):
        output_tensor = torch.empty(
            self.output_shape,
            dtype=torch.float32,
            device="cuda"
        )

        self.context.set_tensor_address(
            self.input_name,
            input_tensor.data_ptr()
        )

        self.context.set_tensor_address(
            self.output_name,
            output_tensor.data_ptr()
        )

        with torch.cuda.stream(self.stream):
            for _ in range(WARMUP_RUNS):
                success = (
                    self.context.execute_async_v3(
                        stream_handle=(
                            self.stream.cuda_stream
                        )
                    )
                )

                if not success:
                    raise RuntimeError(
                        "TensorRT 预热失败"
                    )

        self.stream.synchronize()

        times_ms = []

        for _ in range(TEST_RUNS):
            start_event = torch.cuda.Event(
                enable_timing=True
            )
            end_event = torch.cuda.Event(
                enable_timing=True
            )

            with torch.cuda.stream(self.stream):
                start_event.record(self.stream)

                success = (
                    self.context.execute_async_v3(
                        stream_handle=(
                            self.stream.cuda_stream
                        )
                    )
                )

                if not success:
                    raise RuntimeError(
                        "TensorRT 推理失败"
                    )

                end_event.record(self.stream)

            end_event.synchronize()

            times_ms.append(
                start_event.elapsed_time(end_event)
            )

        return print_statistics(
            display_name,
            times_ms
        )


def print_summary(results):
    print("\n" + "=" * 72)
    print("统一性能对比")
    print("=" * 72)

    print(
        f"{'后端':<25}"
        f"{'平均延迟/ms':>15}"
        f"{'P95/ms':>12}"
        f"{'理论FPS':>15}"
    )

    print("-" * 72)

    for result in results:
        print(
            f"{result['name']:<25}"
            f"{result['mean_ms']:>15.3f}"
            f"{result['p95_ms']:>12.3f}"
            f"{result['fps']:>15.2f}"
        )


def main():
    required_files = [
        ONNX_PATH,
        FP32_ENGINE_PATH,
        FP16_ENGINE_PATH,
        Path("yolov5n.pt")
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"缺少文件：{file_path}"
            )

    torch.manual_seed(0)
    np.random.seed(0)

    input_array = np.random.rand(
        *INPUT_SHAPE
    ).astype(np.float32)

    input_tensor = torch.from_numpy(
        input_array
    ).to("cuda")

    print("输入形状：", input_array.shape)
    print("预热次数：", WARMUP_RUNS)
    print("测试次数：", TEST_RUNS)

    results = []

    results.append(
        benchmark_pytorch(input_tensor)
    )

    results.append(
        benchmark_onnxruntime(input_array)
    )

    fp32_engine = TensorRTEngine(
        FP32_ENGINE_PATH
    )

    results.append(
        fp32_engine.benchmark(
            input_tensor,
            "TensorRT FP32"
        )
    )

    fp16_engine = TensorRTEngine(
        FP16_ENGINE_PATH
    )

    results.append(
        fp16_engine.benchmark(
            input_tensor,
            "TensorRT FP16"
        )
    )

    print_summary(results)


if __name__ == "__main__":
    main()