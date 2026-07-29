import tensorrt as trt
import torch


ENGINE_FILE = "simple.engine"

# TensorRT 日志记录器
logger = trt.Logger(trt.Logger.WARNING)


# 读取并反序列化 Engine
with open(ENGINE_FILE, "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine_data)

if engine is None:
    raise RuntimeError("TensorRT Engine 加载失败")

print("Engine 加载成功")


# 创建推理上下文
context = engine.create_execution_context()

if context is None:
    raise RuntimeError("推理上下文创建失败")


# 查看 Engine 的输入输出信息
print("\nEngine 输入输出信息：")

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    shape = engine.get_tensor_shape(name)
    dtype = engine.get_tensor_dtype(name)

    print(
        f"名称: {name}, "
        f"类型: {mode}, "
        f"形状: {shape}, "
        f"数据类型: {dtype}"
    )


# 创建 GPU 输入张量
input_tensor = torch.randn(
    1,
    10,
    device="cuda",
    dtype=torch.float32
).contiguous()


# 创建 GPU 输出张量
output_tensor = torch.empty(
    1,
    5,
    device="cuda",
    dtype=torch.float32
).contiguous()


# 将输入输出的 GPU 内存地址交给 TensorRT
context.set_tensor_address(
    "input",
    input_tensor.data_ptr()
)

context.set_tensor_address(
    "output",
    output_tensor.data_ptr()
)


# 获取 PyTorch 当前 CUDA Stream
stream = torch.cuda.current_stream()


# 执行 TensorRT 推理
success = context.execute_async_v3(
    stream_handle=stream.cuda_stream
)

if not success:
    raise RuntimeError("TensorRT 推理执行失败")


# 等待 GPU 运算完成
torch.cuda.synchronize()


print("\n输入数据：")
print(input_tensor)

print("\nTensorRT 输出结果：")
print(output_tensor)

print("\nTensorRT GPU 推理完成")