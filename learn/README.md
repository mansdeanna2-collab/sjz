# `learn/` — 一个用于学习的干净项目

这个目录是一个**完整的、合法的**计算机视觉学习项目，覆盖了原仓库脚本里
用到的所有核心技术栈，但**应用场景与游戏无关**：

| 核心技术                | 本项目中的学习形式                                                             |
| ----------------------- | ------------------------------------------------------------------------------ |
| YOLO 推理               | `detector.py`：Ultralytics + **ONNX Runtime** 两套后端（+ TRT 骨架说明）       |
| 前 / 后处理             | `preprocess.py`：`letterbox`、`scale_boxes`、`nms_xyxy`（纯 NumPy，可单元测试）|
| 多目标跟踪              | `tracker.py`：`IouTracker` (EMA) + **`KalmanTracker` (SORT 风格)**             |
| 关联算法                | `assignment.py`：贪心 + **匈牙利 / Jonker-Volgenant**（scipy 可用时）          |
| 采集                    | `capture.py` + `demo_screen_capture.py`：摄像头/视频/图片目录/桌面 + 基准      |
| 可视化                  | `overlay.py`：OpenCV 画框、轨迹、FPS、统计面板、十字准星                       |
| PID 控制                | `controller.py` + `demo_pcontrol.py`：**PID + dt + 前馈**，纯数学仿真          |

> 明确说明：本项目不包含任何"找游戏窗口"、"发送鼠标键盘事件"、
> "进程注入"、"反调试对抗"之类的内容。它是一个通用的计算机视觉学习脚手架。

---

## 项目结构

```
learn/
├── __init__.py
├── capture.py              # 采集源抽象
├── preprocess.py           # letterbox / scale_boxes / NMS
├── detector.py             # Ultralytics + ONNX Runtime 后端
├── assignment.py           # 贪心 + 匈牙利 指派
├── tracker.py              # IoU(EMA) + Kalman(SORT) 两种跟踪器
├── overlay.py              # OpenCV 叠加层
├── controller.py           # P / PI / PID 控制器
├── main.py                 # 端到端 Demo
├── demo_pcontrol.py        # P/PID 控制器离线可视化
├── demo_kalman.py          # Kalman vs EMA 对比（带 MAE / ID-switch 统计）
├── demo_screen_capture.py  # 桌面采集性能基准
├── tests.py                # 单元测试（无需 GPU / 摄像头 / 模型）
├── requirements.txt
└── README.md               # 本文件
```

数据流：

```
 source ── frame ──▶ detector ── detections ──▶ tracker ── tracks ──▶ overlay ──▶ window / mp4
  │                      │                         │                      │
  │                      │                         │                      └── FPS / 统计
  │                      │                         └── ID / 预测 / 关联 / 生灭
  │                      └── YOLO (ultralytics / onnx) + letterbox + NMS
  └── 摄像头 / 视频 / 图片目录 / 桌面
```

---

## 快速开始

```bash
# 1. 最小依赖（CPU 就能跑）
pip install -r learn/requirements.txt

# 2. 跑单元测试（纯逻辑，不需要 GPU / 摄像头 / 模型权重）
python -m learn.tests          # 或者： python -m pytest

# 3. 离线直观感受 PID 控制器
python -m learn.demo_pcontrol

# 4. Kalman vs EMA 对比（有无图形环境都能跑）
python -m learn.demo_kalman --frames 600 --drop 0.3 --no-window

# 5. 用摄像头跑 YOLOv8n 全流程（Ultralytics 后端）
python -m learn.main --source 0

# 6. 用 ONNX 后端 + Kalman 跟踪器 + 匈牙利匹配
python -m learn.main --source video.mp4 \
    --detector onnx --weights yolov8n.onnx \
    --tracker kalman --match hungarian

# 7. 只保留 person / car 两类，并把结果录成视频
python -m learn.main --source video.mp4 \
    --class-names person car --save-video out.mp4

# 8. 测桌面采集性能
python -m learn.demo_screen_capture --frames 300 --backend mss
```

运行时按键：

| 键       | 作用                  |
| -------- | --------------------- |
| `q / ESC`| 退出                  |
| `h`      | 显示 / 隐藏统计面板   |
| `b`      | 显示 / 隐藏检测框     |
| `t`      | 显示 / 隐藏历史轨迹   |
| `SPACE`  | 暂停 / 继续           |

---

## 推荐的学习顺序

1. **`controller.py` + `demo_pcontrol.py`**
   一维 P 控制器是所有"平滑追踪"的基础。看懂 `step()` 里
   `error → kp → integral → derivative → clip → smooth` 这条链。
   然后对照 `test_integral_eliminates_bias` / `test_anti_windup` /
   `test_feedforward_reduces_lag` 三个单测，理解 I / D / 前馈各自在解决什么。
2. **`preprocess.py`**
   先看 `letterbox` 为什么要保持长宽比 + padding；再看 NMS 循环里每一步为什么能成立。
3. **`assignment.py` + `tracker.py`**
   先读 `greedy_assignment`，然后看 `test_linear_beats_greedy` 这个反例，
   明白为什么需要匈牙利。再读 `IouTracker._match`，最后看 `KalmanTracker`：
   重点是 **"先 predict，再用预测框去关联"** 这件事。
4. **`demo_kalman.py`**
   直接看 EMA 和 Kalman 在同一组合成数据上的 MAE / ID 切换差异。
5. **`capture.py`**
   理解"采集源"为什么要抽象：同一套检测/跟踪代码，数据源可以随便换。
6. **`detector.py`**
   先用 Ultralytics 跑通，再对照 `OnnxYoloDetector` 自己写一遍
   "前处理 → 推理 → scale_boxes → per-class NMS" 的完整链路。最后读 TRT 骨架说明。
7. **`overlay.py` + `main.py`**
   把所有东西串起来，对照 `main.py` 里的 `t0/t1/t2` 看每一部分耗时。

---

## 模块要点速查

### `controller.PIDController1D`

```python
e_k  = target - current
I_k  = clip(I_{k-1} + e_k * dt, -i_cap, i_cap)       # 积分 + 抗饱和
D_k  = (e_k - e_{k-1}) / dt                          # 或 -(cur - last_cur)/dt
u    = kp*e + ki*I + kd*D + v_ff                     # 可叠加速度前馈
u    = clip(u, -step_cap, step_cap)
out  = smooth * out_prev + (1 - smooth) * u          # 输出 EMA 平滑
```

### `tracker.IouTracker` / `tracker.KalmanTracker`

共同流程（每一帧）：

```
(1) 旧轨迹 (Kalman 里先 predict 一步)
(2) 用 cost=1-IoU 做指派（可选匈牙利 / 贪心，可选类别约束）
(3) 匹配上 → 更新 (EMA / Kalman update)
    未匹配轨迹 → lost += 1，超过 max_lost 删除
    未匹配检测 → 注册新轨迹
(4) 返回 hits >= min_hits 且 lost == 0 的稳定轨迹
```

### `detector.OnnxYoloDetector`

```
BGR ─► letterbox ─► /255 ─► HWC2CHW ─► sess.run ─► (N, 4+nc)
     ─► argmax 得类别 ─► 按 conf / classes 过滤
     ─► xywh→xyxy ─► scale_boxes ─► per-class NMS
```

---

## 延伸练习

- 把 `IouTracker` 替换成**带外观特征**的 DeepSORT，比较 ID 切换次数。
- 给 `OnnxYoloDetector` 加一个 `CUDAExecutionProvider` 选项，比较 CPU/GPU 耗时。
- 在 `overlay.py` 里加一个"热力图"：把最近 N 帧所有 track 的中心点累加。
- 把 `demo_kalman.py` 的真值运动改成带速度变化的"布朗运动"，观察 Kalman 的性能边界。
- 为 `OnnxYoloDetector` 加一个 fp16 / int8 量化模型对比，理解精度-速度权衡。

---

## 与原脚本 `yx1.5.py` 的关系

本目录**不依赖、也不修改** `yx1.5.py`。如果你想对照原脚本的某个参数
（例如 `AIM_KP_X`、`TRK_MAX_LOST`）在"干净实现"里是什么样，可以在本目录的
对应文件里搜索同名概念（`kp`、`max_lost` 等），两边概念一一对应。
