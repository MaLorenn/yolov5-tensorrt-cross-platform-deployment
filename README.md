# YOLOv5 TensorRT 跨平台部署与推理优化（PC 端）

本分支保存项目的 **PC Ubuntu 端代码、模型转换程序、TensorRT Engine 构建脚本与性能测试结果**。

项目整体目标是建立一条完整的跨平台部署链路：

```text
PyTorch 模型
    ↓
导出 ONNX
    ↓
PC 端验证 ONNX
    ↓
PC 端构建并测试 TensorRT FP32 / FP16 Engine
    ↓
将 ONNX 模型传到 Jetson Nano
    ↓
Nano 端重新构建 Engine 并运行 C++ 实时推理
```

> PC 与 Jetson Nano 的 GPU 架构、CPU 架构和 TensorRT 版本不同。  
> 两端可以共享 `.onnx`，但不应直接共用 `.engine`。

---

## 1. 分支说明

本仓库长期保留两个分支：

```text
pc      PC Ubuntu 端
nano    Jetson Nano 端
```

当前 README 属于：

```text
pc
```

在 PC 上检查当前分支：

```bash
cd ~/yolov5-tensorrt-cross-platform-deployment
git status
```

正常应显示：

```text
位于分支 pc
```

---

## 2. PC 环境

当前测试环境：

| 项目 | 版本 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 显存 | 8 GB |
| NVIDIA Driver | 580.159.03 |
| Driver CUDA Version | 13.0 |
| Python | 3.10.20 |
| PyTorch | 2.13.0+cu130 |
| ONNX Runtime | 1.23.2 |
| TensorRT | 10.3.0 |
| TensorRT Python 包 | tensorrt-cu12 |

ONNX Runtime 可用 Provider：

```text
TensorrtExecutionProvider
CUDAExecutionProvider
CPUExecutionProvider
```

注意：

- 当前系统没有单独安装 CUDA Toolkit。
- PyTorch 自带 CUDA 13.0 运行环境。
- TensorRT 通过 Python 包安装。
- 不要仅根据 `nvidia-smi` 判断是否安装了完整 CUDA Toolkit。

---

## 3. 主要文件说明

仓库中包含原始 YOLOv5 工程以及本项目新增的 TensorRT 测试程序。

重点关注：

```text
export.py
```

作用：

- 将 PyTorch `.pt` 模型导出为 ONNX。
- 控制输入尺寸、batch size 和 opset。
- 为 Jetson Nano 导出时建议使用固定输入尺寸和 opset 12。

```text
tensorrt_test/
```

作用：

- 构建 TensorRT Engine。
- 检查 Engine 输入输出。
- 运行 TensorRT 推理。
- 比较 PyTorch、ONNX Runtime 与 TensorRT 性能。
- 比较 FP32 与 FP16 输出误差和最终检测结果。

```text
yolov5n.pt
```

PyTorch 原始模型。

```text
yolov5n.onnx
```

跨平台交换模型。该文件可以传到 Nano。

```text
yolov5n_fp32.engine
yolov5n_fp16.engine
```

PC 专用 TensorRT Engine。

> 不要把 PC 生成的 Engine 直接拿到 Nano 上运行。

---

## 4. 激活 Python 环境

进入项目目录：

```bash
cd ~/yolov5-tensorrt-cross-platform-deployment
```

激活此前使用的 Conda 环境：

```bash
conda activate tensorrt
```

检查 Python：

```bash
python --version
```

检查 PyTorch 与 GPU：

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

检查 TensorRT：

```bash
python - <<'PY'
import tensorrt as trt
print("TensorRT:", trt.__version__)
PY
```

检查 ONNX Runtime：

```bash
python - <<'PY'
import onnxruntime as ort
print("ONNX Runtime:", ort.__version__)
print("Available Providers:", ort.get_available_providers())
PY
```

---

## 5. 导出 YOLOv5n ONNX

为了兼容 Jetson Nano TensorRT 7.1.3，推荐参数：

```text
batch size = 1
输入尺寸 = 640 × 640
固定输入尺寸
ONNX opset = 12
不集成 NMS
```

示例：

```bash
python export.py \
  --weights yolov5n.pt \
  --include onnx \
  --imgsz 640 640 \
  --batch-size 1 \
  --opset 12
```

导出后检查文件：

```bash
ls -lh yolov5n.onnx
```

检查 ONNX 模型：

```bash
python - <<'PY'
import onnx

model = onnx.load("yolov5n.onnx")
onnx.checker.check_model(model)

print("ONNX 模型检查通过")
print("IR version:", model.ir_version)
print("Opset:", model.opset_import[0].version)
PY
```

本项目使用的模型输入输出：

```text
输入：  1 × 3 × 640 × 640
输出：  1 × 25200 × 85
```

输出维度解释：

```text
1       batch size，一次输入 1 张图片
25200   YOLOv5 在三个检测尺度上生成的候选框总数
85      每个候选框的数据
```

COCO 模型的 85 个数包括：

```text
4 个边界框参数
1 个目标置信度
80 个类别概率
```

---

## 6. ONNX Runtime 验证

运行 ONNX Runtime 前，先确认实际 Provider：

```bash
python - <<'PY'
import onnxruntime as ort

session = ort.InferenceSession(
    "yolov5n.onnx",
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)

print("实际 Providers：", session.get_providers())
print("输入：", session.get_inputs()[0].name, session.get_inputs()[0].shape)
print("输出：", session.get_outputs()[0].name, session.get_outputs()[0].shape)
PY
```

如果输出中包含：

```text
CUDAExecutionProvider
```

说明 ONNX Runtime 可以使用 GPU。

---

## 7. 构建 PC TensorRT Engine

本项目已经成功构建：

```text
yolov5n_fp32.engine
yolov5n_fp16.engine
```

如果需要重新构建，应运行 `tensorrt_test/` 中对应的 Engine 构建脚本。

由于脚本文件可能继续调整，先查看目录：

```bash
ls -la tensorrt_test
```

再根据脚本中的命令行参数运行，例如：

```bash
python tensorrt_test/构建脚本.py
```

构建完成后检查：

```bash
ls -lh *.engine
```

当前实验结果：

| Engine | 文件大小 | 构建耗时 |
|---|---:|---:|
| FP32 | 约 10.45 MiB | 约 48.46 s |
| FP16 | 约 5.91 MiB | 约 171.4 s |

FP16 Engine 文件大小相比 FP32 减少约 43%。

注意：

- Engine 构建时间不代表推理时间。
- FP16 Engine 的输入输出接口仍可能显示 FP32。
- TensorRT 内部仍可以使用 FP16 计算。

---

## 8. PC 端性能测试

统一测试条件：

```text
模型：YOLOv5n
输入：1 × 3 × 640 × 640
预热：20 次
正式测试：100 次
```

测试结果：

| 推理后端 | 平均推理时间 | P95 | 理论吞吐率 |
|---|---:|---:|---:|
| PyTorch CUDA FP32 | 2.716 ms | 2.950 ms | 368.16 FPS |
| ONNX Runtime CUDA | 4.066 ms | 4.442 ms | 245.96 FPS |
| TensorRT FP32 | 1.169 ms | 1.172 ms | 855.63 FPS |
| TensorRT FP16 | 0.657 ms | 0.664 ms | 1521.77 FPS |

主要结论：

```text
TensorRT FP32 相比 PyTorch CUDA FP32 加速约 2.32 倍
TensorRT FP16 相比 PyTorch CUDA FP32 加速约 4.13 倍
TensorRT FP16 相比 TensorRT FP32 加速约 1.78 倍
TensorRT FP16 平均延迟相比 FP32 降低约 43.8%
```

ONNX Runtime CUDA 当前测试慢于 PyTorch，主要可能因为普通 `session.run()` 包含：

- CPU NumPy 输入到 GPU 的传输；
- 运行时调度；
- 输出返回 CPU；
- 测试口径和 CUDA Event 纯 GPU 计时不同。

因此，不应简单把该结果解释为 ONNX Runtime 的 GPU 计算本身一定更慢。

---

## 9. FP32 与 FP16 精度验证

真实 `bus.jpg` 测试中：

```text
置信度阈值：0.25
NMS IoU 阈值：0.45
```

FP32 与 FP16 都得到：

```text
3 个 person
1 个 bus
```

两者：

- 候选框数量一致；
- NMS 后数量一致；
- 类别一致；
- 检测框差异通常只有零点几像素；
- 置信度仅有轻微变化；
- 没有出现 NaN 或 Inf。

结论：

```text
FP16 显著提升 PC 纯推理速度，
同时没有对最终检测结果产生明显影响。
```

---

## 10. 图片预处理流程

模型输入预处理流程：

```text
读取 BGR 图片
    ↓
Letterbox 等比例缩放与填充
    ↓
BGR → RGB
    ↓
HWC → CHW
    ↓
uint8 → float32
    ↓
像素值 0～255 → 0～1
    ↓
增加 batch 维度
```

`bus.jpg` 示例：

```text
原始尺寸：1080 × 810 × 3
缩放比例：0.5925925926
缩放后：640 × 480
左右填充：各 80 像素
最终输入：1 × 3 × 640 × 640
```

推理后还要完成：

```text
置信度筛选
    ↓
NMS
    ↓
坐标从 640 × 640 恢复到原始图片
    ↓
绘制检测框
```

---

## 11. 将 ONNX 传到 Jetson Nano

可以使用 `scp`：

```bash
scp yolov5n.onnx nano@JETSON_IP:/home/nano/yolov5-tensorrt-cross-platform-deployment/
```

把 `JETSON_IP` 替换为 Nano 的实际 IP 地址。

例如：

```bash
scp yolov5n.onnx nano@192.168.1.100:/home/nano/yolov5-tensorrt-cross-platform-deployment/
```

可以直接传输：

```text
ONNX 模型
测试图片和视频
类别文件
配置文件
通用 C++ 源码
```

不要直接传输并复用：

```text
PC 生成的 .engine
PC TensorRT 10 专用运行代码
PC CUDA / TensorRT 软件包
```

---

## 12. PC 分支的 Git 操作

### 查看状态

```bash
cd ~/yolov5-tensorrt-cross-platform-deployment
git status
```

确保当前是：

```text
位于分支 pc
```

### 提交普通修改

```bash
git add -A
git commit -m "描述本次PC端修改"
git push
```

### 只提交一个文件

```bash
git add tensorrt_test/文件名.py
git commit -m "更新PC端TensorRT测试脚本"
git push
```

### 被 `.gitignore` 忽略的新模型

例如新生成的 Engine 没有被 Git 跟踪：

```bash
git add -f yolov5n_fp16.engine
git commit -m "更新PC端FP16 Engine"
git push
```

### 什么时候需要 `git pull`

如果以下情况发生，修改前先执行：

```bash
git pull
```

适用情况：

- 在 GitHub 网页端修改过文件；
- 其他电脑修改过 `pc` 分支；
- Codex 或其他工具向 `pc` 分支提交过；
- 本地很久没有同步；
- `git push` 提示远程领先。

如果 PC 是唯一修改 `pc` 分支的设备，而且远程没有其他提交，则不需要机械地每次 `pull`。

---

## 13. 常见问题

### 13.1 TensorRT 无法导入

确认 Conda 环境：

```bash
conda activate tensorrt
python -c "import tensorrt as trt; print(trt.__version__)"
```

### 13.2 ONNX Runtime 没用 GPU

检查：

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

再检查 Session：

```python
print(session.get_providers())
```

只有可用 Provider 不代表实际 Session 一定使用该 Provider。

### 13.3 PC Engine 能否复制到 Nano

不能作为正式方案。

正确做法：

```text
PC 导出 ONNX
Nano 接收 ONNX
Nano 使用自己的 TensorRT 重新构建 Engine
```

### 13.4 为什么理论 FPS 很高，真实摄像头 FPS 却低

纯推理性能不包含：

- 摄像头采集；
- 图片预处理；
- CPU-GPU 数据传输；
- 后处理；
- NMS；
- 绘图与显示；
- 线程同步。

部署项目必须区分：

```text
纯模型推理延迟
端到端延迟
系统实际 FPS
```

---

## 14. 当前 PC 端结论

本项目在 PC 端已经完成：

- PyTorch 模型加载；
- YOLOv5n ONNX 导出；
- ONNX 输入输出检查；
- ONNX Runtime CUDA 验证；
- TensorRT FP32 Engine 构建；
- TensorRT FP16 Engine 构建；
- Engine 输入输出检查；
- PyTorch、ONNX Runtime、TensorRT 性能对比；
- FP32 与 FP16 原始输出误差分析；
- 真实图片预处理、NMS 与坐标恢复；
- FP32 与 FP16 最终检测结果对比。

当前 PC 端推荐：

```text
推理后端：TensorRT
计算精度：FP16
模型输入：640 × 640
```
