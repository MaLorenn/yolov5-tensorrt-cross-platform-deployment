import tensorrt as trt


ONNX_FILE = "simple.onnx"
ENGINE_FILE = "simple.engine"


# TensorRT日志
logger = trt.Logger(trt.Logger.WARNING)


# 创建Builder
builder = trt.Builder(logger)


# 创建Network
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)


# 创建ONNX解析器
parser = trt.OnnxParser(network, logger)


# 读取ONNX文件
with open(ONNX_FILE, "rb") as f:
    onnx_data = f.read()


# 解析ONNX
if not parser.parse(onnx_data):
    print("ONNX解析失败")

    for i in range(parser.num_errors):
        print(parser.get_error(i))

    exit()


print("ONNX解析成功")


# 创建配置
config = builder.create_builder_config()


# 设置工作空间大小
config.set_memory_pool_limit(
    trt.MemoryPoolType.WORKSPACE,
    1 << 30
)


# 构建Engine
engine = builder.build_serialized_network(
    network,
    config
)


if engine is None:
    print("Engine生成失败")
    exit()


# 保存Engine
with open(ENGINE_FILE, "wb") as f:
    f.write(engine)


print("TensorRT Engine生成完成")