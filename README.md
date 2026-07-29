# YOLOv5 TensorRT 跨平台部署与推理优化（Jetson Nano 端）

本分支保存项目的 **Jetson Nano C++ TensorRT 实时推理、多线程流水线与性能实验代码**。

本项目在 PC Ubuntu 上完成模型导出和验证，再在 Jetson Nano 上使用本机 TensorRT 重新构建 Engine，接入 USB 摄像头运行 YOLOv5n 实时目标检测。

```text
PC：PyTorch → ONNX
                ↓
Jetson Nano：ONNX → TensorRT Engine → C++ 实时推理
```

> PC 与 Nano 的 TensorRT 版本和 GPU 架构不同，PC 生成的 `.engine` 不应直接在 Nano 上使用。

---

## 1. 分支说明

本仓库长期保留两个分支：

```text
pc      PC Ubuntu 端
nano    Jetson Nano 端
```

当前 README 属于：

```text
nano
```

检查分支：

```bash
cd ~/yolov5-tensorrt-cross-platform-deployment
git status
```

正常应显示：

```text
位于分支 nano
```

---

## 2. Jetson Nano 环境

当前设备：

| 项目 | 信息 |
|---|---|
| 设备 | NVIDIA Jetson Nano 4GB |
| 系统 | Ubuntu 18.04.6 LTS |
| L4T | R32.4.4 |
| 内核 | 4.9.140-tegra |
| 架构 | ARM64 / aarch64 |
| CUDA | 10.2 |
| nvcc | 10.2.89 |
| TensorRT | 7.1.3.0 |
| Python OpenCV | 3.2.0 |
| C++ OpenCV | 4.1.1 |

`trtexec` 路径：

```bash
/usr/src/tensorrt/bin/trtexec
```

注意：

- 当前属于较老的 JetPack 4.x 软件栈。
- 不要随意升级 Ubuntu、CUDA 或 TensorRT。
- 版本升级很可能破坏当前已经可用的部署环境。

---

## 3. Python 环境注意事项

Nano 上存在两个 Python：

默认 Python：

```text
/home/nano/.pyenv/shims/python3
Python 3.8.19
```

系统 Python：

```text
/usr/bin/python3
Python 3.6.9
```

当前 TensorRT Python 绑定对应系统 Python 3.6。

运行 TensorRT Python 程序时使用：

```bash
/usr/bin/python3 程序名.py
```

不要直接使用：

```bash
python3 程序名.py
```

否则可能出现：

```text
ModuleNotFoundError: No module named 'tensorrt'
```

---

## 4. 摄像头环境

摄像头设备：

```text
/dev/video0
```

设备名称：

```text
USB 2.0 Camera
```

已验证：

```text
分辨率：640 × 480
帧率：约 30.5 FPS
```

查看摄像头设备：

```bash
ls -l /dev/video*
```

查看详细信息：

```bash
v4l2-ctl --device=/dev/video0 --all
```

C++ 打开摄像头时建议明确使用 V4L2：

```cpp
cv::VideoCapture camera(0, cv::CAP_V4L2);
```

通过 SSH 运行时默认不使用：

```cpp
cv::imshow(...)
```

因为 SSH 无图形界面环境可能报错：

```text
Gtk-WARNING: cannot open display
```

本项目默认采用：

- 终端日志；
- 性能统计；
- 必要时保存检测结果；
- 不弹出 GUI 窗口。

---

## 5. 内存、Swap 与磁盘

当前物理内存：

```text
约 3.9 GB
```

当前 Swap：

```text
约 11～12 GB
```

组成大致包括：

```text
/mnt/swapfile：6 GB
/var/swapfile：4 GB
zram：约 2 GB
```

根分区：

```text
总容量约 117 GB
可用约 65 GB
```

外接存储：

```text
总容量约 58 GB
可用约 55 GB
```

当前内存、Swap 和磁盘空间能够支持：

- C++ 编译；
- TensorRT Engine 构建；
- 模型和日志保存；
- 连续运行测试。

---

## 6. 目录说明

当前 Nano 端主要结构：

```text
yolov5-tensorrt-cross-platform-deployment/
├── camera.py
├── bus.jpg
├── bus_result.jpg
├── yolov5n.onnx
├── yolov5n_fp32.engine
├── yolov5n_fp16.engine
└── trt_cpp/
    ├── CMakeLists.txt
    ├── include/
    │   └── logger.hpp
    ├── src/
    │   ├── main.cpp
    │   ├── main_serial_profiled.cpp
    │   ├── main_multithread.cpp
    │   ├── main_multithread_3stage_final.cpp
    │   ├── main_multithread_4stage_optimized.cpp
    │   ├── main_multithread_blocking_queue.cpp
    │   ├── main_multithread_queue_backup.cpp
    │   ├── main_serial_gui_backup.cpp
    │   └── test.cpp
    └── build/
```

重点文件：

```text
main_serial_profiled.cpp
```

串行性能基线，用于统计：

- 摄像头采集；
- 预处理；
- TensorRT 推理；
- 后处理；
- 端到端 FPS 和延迟。

```text
main_multithread_4stage_optimized.cpp
```

当前四阶段高帧率流水线版本。

流水线结构：

```text
采集线程
    ↓
预处理线程
    ↓
推理线程
    ↓
后处理 / 结果线程
```

队列容量均为 1，队列满时丢弃旧数据，以降低帧积压。

---

## 7. 从 ONNX 构建 TensorRT Engine

进入项目目录：

```bash
cd ~/yolov5-tensorrt-cross-platform-deployment
```

确认 ONNX 文件：

```bash
ls -lh yolov5n.onnx
```

### 构建 FP32 Engine

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov5n.onnx \
  --saveEngine=yolov5n_fp32.engine
```

### 构建 FP16 Engine

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov5n.onnx \
  --saveEngine=yolov5n_fp16.engine \
  --fp16
```

注意：

- Nano TensorRT 7.1.3 对 ONNX 算子版本兼容性有限。
- 推荐 PC 导出固定尺寸、batch=1、opset=12 的 ONNX。
- 如果解析失败，先检查 ONNX opset，不要随意升级 Nano TensorRT。

---

## 8. 使用 trtexec 测试纯推理性能

### FP32

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=yolov5n_fp32.engine \
  --warmUp=1000 \
  --duration=10
```

### FP16

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=yolov5n_fp16.engine \
  --warmUp=1000 \
  --duration=10
```

当前实验结果：

| 精度 | Engine 大小 | GPU 平均延迟 | Host 平均延迟 | 吞吐率 |
|---|---:|---:|---:|---:|
| FP32 | 14 MB | 57.63 ms | 58.93 ms | 16.97 qps |
| FP16 | 7.1 MB | 46.61 ms | 47.91 ms | 20.87 qps |

结论：

```text
FP16 Engine 大小减少约 49%
GPU 推理延迟降低约 19.1%
吞吐率提升约 23.0%
```

Jetson Nano 虽然没有 Tensor Core，但 FP16 纯推理仍有一定提升。

---

## 9. 编译 C++ 程序

进入 C++ 工程：

```bash
cd ~/yolov5-tensorrt-cross-platform-deployment/trt_cpp
```

如果 `build/` 已存在：

```bash
cd build
cmake ..
make -j4
```

如果需要重新建立编译目录：

```bash
cd ~/yolov5-tensorrt-cross-platform-deployment/trt_cpp
mkdir -p build
cd build
cmake ..
make -j4
```

编译成功后，查看可执行文件：

```bash
ls -lh
```

当前常用可执行文件：

```text
yolov5_trt
yolov5_trt_multithread
```

如果修改了 `CMakeLists.txt` 或新增了源文件，重新执行：

```bash
cmake ..
make -j4
```

---

## 10. 运行串行版本

进入编译目录：

```bash
cd ~/yolov5-tensorrt-cross-platform-deployment/trt_cpp/build
```

运行：

```bash
./yolov5_trt
```

如果当前 `CMakeLists.txt` 将串行性能版本编译成其他名字，以 `ls` 输出为准。

串行流程：

```text
采集一帧
    ↓
Letterbox 预处理
    ↓
TensorRT 推理
    ↓
NMS 后处理
    ↓
再采集下一帧
```

串行基线结果：

```text
平均 FPS：6.73
平均端到端延迟：148.61 ms
```

串行版本主要用于建立性能基准，不是最终高帧率版本。

---

## 11. 运行四阶段多线程版本

进入编译目录：

```bash
cd ~/yolov5-tensorrt-cross-platform-deployment/trt_cpp/build
```

运行：

```bash
./yolov5_trt_multithread
```

当前最优高帧率结构：

```text
四阶段流水线
采集队列容量：1
预处理队列容量：1
结果队列容量：1
队列满时丢弃旧帧或旧结果
```

实验结果：

```text
平均 FPS：11.20
平均端到端延迟：257.68 ms
处理比例：38.07%
```

相比串行版本：

```text
FPS：6.73 → 11.20，提升约 66.4%
延迟：148.61 → 257.68 ms，增加约 73.4%
```

为什么 FPS 提高但延迟也提高：

- 流水线允许多个阶段并行工作；
- 单位时间内处理完成的帧更多；
- 一帧仍要经过多个阶段和队列；
- 因此吞吐率提高不代表单帧端到端延迟一定降低。

需要区分：

```text
吞吐率 / FPS
单帧端到端延迟
摄像头帧处理比例
```

---

## 12. FP32 与 FP16 完整流水线结果

测试设置：

```text
采集队列容量：1
预处理队列容量：1
结果队列容量：1
队列满时丢弃旧数据
处理帧数：300
摄像头：640 × 480
模型输入：640 × 640
```

结果：

| 指标 | FP32 | FP16 |
|---|---:|---:|
| 平均 FPS | 11.23 | 11.31 |
| 预处理 | 64.28 ms | 64.39 ms |
| 推理 | 64.37 ms | 63.54 ms |
| 后处理 | 19.34 ms | 18.84 ms |
| 端到端延迟 | 257.88 ms | 256.14 ms |
| 处理比例 | 38.17% | 38.46% |

完整流水线中：

```text
FPS 仅提升约 0.7%
推理耗时降低约 1.3%
端到端延迟降低约 0.7%
```

这与 `trtexec` 的纯推理提升差别很大。

原因：

- CPU 预处理仍约 64 ms；
- CPU-GPU 数据传输；
- 后处理；
- 摄像头采集；
- 线程调度与同步；
- GPU 推理不是唯一瓶颈。

因此，优化部署系统时不能只看 Engine 的纯推理时间。

---

## 13. 功耗模式测试

查看当前模式：

```bash
sudo nvpmodel -q
```

### MAXN 模式

```bash
sudo nvpmodel -m 0
```

再确认：

```bash
sudo nvpmodel -q
```

### 5W 模式

```bash
sudo nvpmodel -m 1
```

实验结果：

| 指标 | MAXN | 5W |
|---|---:|---:|
| 平均 FPS | 11.23 | 5.91 |
| 预处理 | 64.28 ms | 95.30 ms |
| 推理 | 64.37 ms | 90.76 ms |
| 后处理 | 19.34 ms | 62.06 ms |
| 端到端延迟 | 257.88 ms | 411.50 ms |
| 处理比例 | 38.17% | 19.99% |

5W 相比 MAXN：

```text
FPS 下降约 47.4%
推理耗时增加约 41.0%
预处理耗时增加约 48.3%
端到端延迟增加约 59.6%
```

正式运行推荐：

```text
功耗模式：MAXN
```

因为 5W 同时限制 CPU 和 GPU，导致预处理、推理与后处理全部变慢。

---

## 14. 系统监控

使用：

```bash
tegrastats
```

可以观察：

- CPU 使用率；
- GPU 使用率；
- 内存；
- Swap；
- CPU / GPU 温度；
- 功耗。

保存监控日志：

```bash
tegrastats > tegrastats.log
```

后台运行：

```bash
tegrastats > tegrastats.log 2>&1 &
```

查找进程：

```bash
ps aux | grep tegrastats
```

停止：

```bash
sudo killall tegrastats
```

---

## 15. 当前推荐运行配置

当前最适合的配置：

```text
功耗模式：MAXN
TensorRT 精度：FP16
模型输入：640 × 640
摄像头分辨率：640 × 480
流水线：四阶段多线程
采集队列：1
预处理队列：1
结果队列：1
队列满时：丢弃旧帧
运行方式：SSH 无界面
```

继续优化应优先关注：

1. CPU Letterbox 预处理；
2. CPU-GPU 数据传输；
3. 后处理和 NMS；
4. 内存复用；
5. CUDA Stream 和异步拷贝；
6. 摄像头采集格式和数据转换；
7. 避免不必要的 `cv::Mat` 拷贝。

---

## 16. Nano 分支的 Git 操作

进入仓库：

```bash
cd ~/yolov5-tensorrt-cross-platform-deployment
```

检查：

```bash
git status
```

正常应显示：

```text
位于分支 nano
```

### 提交所有修改

```bash
git add -A
git commit -m "描述本次Nano端修改"
git push
```

### 只提交一个源文件

```bash
git add trt_cpp/src/main_multithread_4stage_optimized.cpp
git commit -m "优化Nano四阶段推理流水线"
git push
```

### 什么时候需要 git pull

以下情况修改前执行：

```bash
git pull
```

- 在 GitHub 网页端修改过 `nano` 分支；
- 其他设备提交过 `nano` 分支；
- Codex 或其他工具修改过远程分支；
- 本地很久没有同步；
- `git push` 提示远程分支领先。

如果 Nano 是唯一修改 `nano` 分支的设备，不需要机械地每次运行 `git pull`。

---

## 17. 常见故障排查

### 17.1 无法打开图形窗口

报错：

```text
Gtk-WARNING: cannot open display
```

原因：

- 通过 SSH 运行；
- 程序调用 `imshow`；
- 当前没有图形显示环境。

解决：

- 使用无界面版本；
- 删除或关闭 `imshow`、`waitKey`；
- 通过日志和保存结果验证。

### 17.2 摄像头打不开

检查：

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```

代码建议：

```cpp
cv::VideoCapture camera(0, cv::CAP_V4L2);
```

### 17.3 TensorRT Engine 报设备不一致

警告类似：

```text
Using an engine plan file across different models of devices is not recommended
```

原因：

- Engine 不是在当前 Nano 上构建；
- Engine 与当前 GPU / TensorRT 环境不完全匹配。

解决：

```text
重新用当前 Nano 的 TensorRT 从 ONNX 构建 Engine
```

### 17.4 Python 无法导入 TensorRT

使用：

```bash
/usr/bin/python3 程序名.py
```

而不是默认 `python3`。

### 17.5 编译报找不到 TensorRT 或 OpenCV

检查：

```bash
pkg-config --modversion opencv4
ls /usr/include/aarch64-linux-gnu/NvInfer.h
ls /usr/lib/aarch64-linux-gnu/libnvinfer.so
```

然后检查 `CMakeLists.txt` 中的头文件和库路径。

---

## 18. 当前 Nano 端结论

已经完成：

- Jetson Nano 环境验证；
- USB 摄像头读取；
- YOLOv5n ONNX 接收；
- Nano 本机 TensorRT Engine 构建；
- C++ TensorRT Runtime 推理；
- Letterbox 预处理；
- NMS 后处理；
- 坐标恢复；
- 摄像头实时目标检测；
- SSH 无界面运行；
- 串行性能基线；
- 三阶段和四阶段多线程流水线；
- 有限长度队列和丢旧帧策略；
- FP32 与 FP16 对比；
- MAXN 与 5W 功耗模式对比。

核心实验结论：

```text
四阶段流水线可把 FPS 从 6.73 提升到约 11.20，
但端到端延迟从 148.61 ms 增加到约 257.68 ms。
```

```text
FP16 能显著改善纯 TensorRT 推理，
但完整流水线仅有很小提升，
说明系统瓶颈同时存在于 CPU 预处理、数据传输和后处理。
```

```text
5W 模式会明显降低完整系统性能，
当前正式运行应使用 MAXN。
```
