# `learn/` — 一个用于学习的干净项目

这个目录是一个**全新的、合法的**学习项目，覆盖了原仓库脚本里用到的所有核心技术栈，但**应用场景与游戏无关**：

| 原脚本用到的技术         | 本项目中的学习形式                                          |
| ------------------------ | ----------------------------------------------------------- |
| YOLO / TensorRT 推理     | `detector.py`：Ultralytics YOLO 封装（+ TRT 骨架说明）      |
| IoU + EMA 多目标跟踪     | `tracker.py`：纯 Python / NumPy 实现，带单元测试            |
| `dxcam` / `mss` 桌面采集 | `capture.py` + `demo_screen_capture.py`：抽象 + 基准测试    |
| Win32 透明 ESP 覆盖层    | `overlay.py`：改为在自己的 OpenCV 窗口里画框、FPS、十字准星 |
| 鼠标 P 控制器 + 防抖     | `controller.py` + `demo_pcontrol.py`：**纯数学仿真，不接鼠标** |

> 明确说明：本项目不包含任何屏幕取色、找游戏窗口、发送鼠标/键盘事件、
> 进程注入、反调试对抗之类的内容。它是一个通用的计算机视觉学习脚手架。

---

## 项目结构

```
learn/
├── __init__.py
├── capture.py            # 采集源抽象：摄像头 / 视频 / 图片目录 / 桌面
├── detector.py           # YOLO 检测器 + TRT 骨架说明
├── tracker.py            # IoU + EMA 多目标跟踪
├── overlay.py            # OpenCV 叠加：框 / 轨迹 / FPS / 统计
├── controller.py         # 1D / 2D 比例（P）控制器
├── main.py               # 端到端 Demo: 采集→检测→跟踪→叠加→显示
├── demo_pcontrol.py      # P 控制器离线可视化
├── demo_screen_capture.py# 桌面采集性能基准
├── tests.py              # 纯逻辑单元测试（无需 GPU / 摄像头）
├── requirements.txt
└── README.md             # 本文件
```

数据流：

```
 source ── frame ──▶ detector ── detections ──▶ tracker ── tracks ──▶ overlay ──▶ window
  │                      │                         │                      │
  │                      │                         │                      └── FPS / 统计
  │                      │                         └── ID / EMA 平滑 / 生灭
  │                      └── YOLO 推理
  └── 摄像头 / 视频 / 图片目录 / 桌面
```

---

## 快速开始

```bash
# 1. 安装最小依赖（CPU 就能跑）
pip install -r learn/requirements.txt

# 2. 先跑单元测试，确认纯逻辑没问题（不需要 GPU / 摄像头）
python -m learn.tests

# 3. 跑 P 控制器仿真（最直观感受 Kp / step_cap / dead_zone 的作用）
python -m learn.demo_pcontrol

# 4. 用摄像头跑 YOLOv8n 全流程
python -m learn.main --source 0

# 5. 用一段视频离线跑，只保留 person (COCO id = 0)
python -m learn.main --source path/to/video.mp4 --classes 0

# 6. 测你自己的桌面采集性能（mss）
python -m learn.demo_screen_capture --frames 300 --backend mss
```

运行时按键：

| 键     | 作用               |
| ------ | ------------------ |
| `q/ESC`| 退出               |
| `h`    | 显示/隐藏统计面板  |
| `b`    | 显示/隐藏检测框    |
| `t`    | 显示/隐藏历史轨迹  |

---

## 推荐的学习顺序

1. **`controller.py` + `demo_pcontrol.py`**
   一维 P 控制器是所有"平滑追踪"的基础。看懂 `step()` 里 `error → kp → clip → smooth` 这 4 行。
2. **`tracker.py` + `tests.py`**
   从 `iou_xyxy` 开始读，然后看 `IouTracker._match` 的贪心匹配。跑 `tests.py` 验证理解。
3. **`capture.py`**
   理解"采集源"为什么要抽象：同样一套检测/跟踪代码，数据源可以随便换。
4. **`detector.py`**
   先用 Ultralytics 跑通，再去读 TensorRT 那段说明，自己按步骤把 `.pt` 导出成 `.engine`。
5. **`overlay.py` + `main.py`**
   最后把所有东西串起来，对照 `main.py` 里的 `t0/t1/t2` 看每一部分的耗时。

---

## 延伸阅读 / 练习题

- 把 `IouTracker` 替换成**卡尔曼滤波 + 匈牙利算法**（SORT），比较 ID 切换次数。
- 把 `UltralyticsDetector` 改写成 `ONNXRuntimeDetector`，比较 CPU/GPU 耗时。
- 在 `overlay.py` 里加一个"热力图"：把最近 N 帧所有 track 的中心点累加。
- 把 `PController2D` 升级成 **PID**，加 I/D 两项，再跑 `demo_pcontrol` 比较。
- 为 `detector.py` 加一个 `imgsz` 的 letterbox 前处理函数，理解 YOLO 的 padding。

---

## 与原脚本 `yx1.5.py` 的关系

本目录**不依赖、也不修改** `yx1.5.py`。如果你想对照原脚本的某个参数（例如 `AIM_KP_X`、`TRK_MAX_LOST`）在"干净实现"里是什么样，可以在本目录的对应文件里搜索同名概念（`kp`、`max_lost` 等），两边一一对应。
