import torch
import torch.nn as nn
import tensorrt as trt


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


# =========================
# 1. PyTorch 推理
# =========================

model = SimpleNet()
model.load_state_dict(
    torch.load("simple.pth", map_location="cpu")
)
model.eval()
model.cuda()

input_tensor = torch.load(
    "input.pt",
    map_location="cuda"
).contiguous()

with torch.no_grad():
    pytorch_output = model(input_tensor)

torch.cuda.synchronize()

print("PyTorch输出：")
print(pytorch_output)


# =========================
# 2. TensorRT 推理
# =========================

logger = trt.Logger(trt.Logger.WARNING)

with open("simple.engine", "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine_data)

if engine is None:
    raise RuntimeError("Engine加载失败")

context = engine.create_execution_context()

if context is None:
    raise RuntimeError("ExecutionContext创建失败")

trt_output = torch.empty(
    (1, 5),
    device="cuda",
    dtype=torch.float32
).contiguous()

context.set_tensor_address(
    "input",
    input_tensor.data_ptr()
)

context.set_tensor_address(
    "output",
    trt_output.data_ptr()
)

stream = torch.cuda.current_stream()

success = context.execute_async_v3(
    stream_handle=stream.cuda_stream
)

if not success:
    raise RuntimeError("TensorRT推理失败")

torch.cuda.synchronize()

print("\nTensorRT输出：")
print(trt_output)


# =========================
# 3. 比较误差
# =========================

absolute_error = torch.abs(
    pytorch_output - trt_output
)

max_error = absolute_error.max().item()
mean_error = absolute_error.mean().item()

print("\n最大绝对误差：", max_error)
print("平均绝对误差：", mean_error)

if torch.allclose(
    pytorch_output,
    trt_output,
    rtol=1e-4,
    atol=1e-5
):
    print("\n验证通过：PyTorch与TensorRT输出基本一致")
else:
    print("\n验证失败：输出误差超出容许范围")