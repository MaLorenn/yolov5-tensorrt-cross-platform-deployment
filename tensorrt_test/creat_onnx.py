import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


# 固定随机数，保证每次运行结果一致
torch.manual_seed(0)

model = SimpleNet()
model.eval()

# 保存 PyTorch 模型权重
torch.save(model.state_dict(), "simple.pth")

# 创建固定输入
x = torch.randn(1, 10)

# 保存输入，后面 PyTorch 和 TensorRT 都使用它
torch.save(x, "input.pt")

# 导出 ONNX
torch.onnx.export(
    model,
    x,
    "simple.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13
)

print("已生成：")
print("simple.pth")
print("input.pt")
print("simple.onnx")