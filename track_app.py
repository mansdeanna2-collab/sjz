# -*- coding: utf-8 -*-
"""
track_app.py — 单文件端到端：TensorRT 检测 + SORT 卡尔曼跟踪 + PID 可视化跟随
================================================================================
这是一个**独立可运行**的脚本，把 learn/ 里验证过的几个最稳的组件
抽出来合并到一起。用 TensorRT 引擎 (默认 test.engine) 做 YOLOv8 风格检测。

明确说明：本脚本**完全在自己的 OpenCV 窗口里可视化**，不找任何游戏窗口、
不发任何系统鼠标/键盘事件、不做进程注入、不与反作弊系统对抗。

典型用法：

    # 摄像头
    python track_app.py --source 0 --engine test.engine

    # 视频文件，只看前两类
    python track_app.py --source demo.mp4 --engine test.engine --classes 0 1

    # 桌面截屏（需 pip install mss）
    python track_app.py --source screen --engine test.engine

    # 结果录成 mp4
    python track_app.py --source demo.mp4 --engine test.engine --save-video out.mp4

运行时按键：
    q / ESC   退出
    h         显示 / 隐藏统计
    b         显示 / 隐藏检测框
    t         显示 / 隐藏历史轨迹
    p         显示 / 隐藏 PID 跟随小圆点
    SPACE     暂停 / 继续

依赖：
    numpy, opencv-python, tensorrt, pycuda           # 必需
    scipy                                             # 可选：启用匈牙利匹配
    mss                                               # 可选：桌面截屏
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# =============================================================================
# 0) 常量：COCO80 类名（YOLOv8 官方默认）。换了权重请在 CLI 里改或改这里。
# =============================================================================

COCO80_NAMES: Sequence[str] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


# =============================================================================
# 1) 几何 / 前后处理
# =============================================================================

def letterbox(
    img: np.ndarray,
    new_shape: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """等比缩放到 new_shape 并居中 padding 到 stride 倍数。"""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) % stride
    dh = (new_shape - new_unpad[1]) % stride
    # 没到目标大小的部分用 padding 补齐到 new_shape
    dw_full = new_shape - new_unpad[0]
    dh_full = new_shape - new_unpad[1]
    # 我们直接按 new_shape 方形输入（YOLOv8 导出时一般 imgsz=640 是方形）
    dw = dw_full
    dh = dh_full
    dw /= 2
    dh /= 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (float(left), float(top))


def scale_boxes(
    boxes_xyxy: np.ndarray,
    ratio: float,
    pad: Tuple[float, float],
    orig_shape: Tuple[int, int, int],
) -> np.ndarray:
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    b = boxes_xyxy.astype(np.float32, copy=True)
    pw, ph = pad
    b[:, [0, 2]] -= pw
    b[:, [1, 3]] -= ph
    b[:, :4] /= max(ratio, 1e-9)
    h, w = orig_shape[:2]
    b[:, 0] = np.clip(b[:, 0], 0, w - 1)
    b[:, 1] = np.clip(b[:, 1], 0, h - 1)
    b[:, 2] = np.clip(b[:, 2], 0, w - 1)
    b[:, 3] = np.clip(b[:, 3], 0, h - 1)
    return b


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> np.ndarray:
    if boxes.size == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        iw = np.clip(xx2 - xx1, 0, None)
        ih = np.clip(yy2 - yy1, 0, None)
        inter = iw * ih
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou < iou_thr]
    return np.array(keep, dtype=int)


def nms_xyxy_per_class(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float = 0.45
) -> np.ndarray:
    if boxes.size == 0:
        return np.array([], dtype=int)
    keep_all = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        local_keep = nms_xyxy(boxes[idx], scores[idx], iou_thr=iou_thr)
        keep_all.extend(idx[local_keep].tolist())
    keep_all = np.array(keep_all, dtype=int)
    if keep_all.size == 0:
        return keep_all
    return keep_all[np.argsort(-scores[keep_all])]


def iou_matrix(a: Sequence, b: Sequence) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    A = np.asarray(a, dtype=np.float32)
    B = np.asarray(b, dtype=np.float32)
    ax1, ay1, ax2, ay2 = A[:, 0:1], A[:, 1:2], A[:, 2:3], A[:, 3:4]
    bx1, by1, bx2, by2 = B[:, 0], B[:, 1], B[:, 2], B[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    aa = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    bb = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = aa + bb - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out.astype(np.float32)


# =============================================================================
# 2) 指派算法（匈牙利 + 贪心回退）
# =============================================================================

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lap  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def greedy_assignment(cost: np.ndarray, max_cost: float = np.inf):
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    n, m = cost.shape
    flat = [(cost[i, j], i, j) for i in range(n) for j in range(m)]
    flat.sort()
    ur, uc = set(), set()
    r, c = [], []
    for v, i, j in flat:
        if v >= max_cost:
            break
        if i in ur or j in uc:
            continue
        ur.add(i); uc.add(j)
        r.append(i); c.append(j)
    return np.array(r, dtype=int), np.array(c, dtype=int)


def linear_assignment(cost: np.ndarray, max_cost: float = np.inf):
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if not _HAS_SCIPY:
        return greedy_assignment(cost, max_cost=max_cost)
    big = float(max_cost if np.isfinite(max_cost) else cost.max() + 1.0) * 10.0 + 1.0
    work = np.where(np.isfinite(cost), cost, big)
    r, c = _scipy_lap(work)
    keep = cost[r, c] < max_cost
    return r[keep], c[keep]


# =============================================================================
# 3) SORT 风格 Kalman 跟踪器
# =============================================================================

@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    conf: float
    cls: int
    label: str


@dataclass
class Track:
    tid: int
    cls: int
    label: str
    xyxy: Tuple[float, float, float, float]
    conf: float
    hits: int = 1
    lost: int = 0
    age: int = 1
    history: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def center(self):
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _xyxy_to_z(xyxy):
    x1, y1, x2, y2 = xyxy
    w = max(1e-6, x2 - x1); h = max(1e-6, y2 - y1)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w * h, w / h], dtype=np.float64)


def _x_to_xyxy(x):
    cx, cy, s, r = float(x[0]), float(x[1]), max(1e-6, float(x[2])), max(1e-6, float(x[3]))
    w = float(np.sqrt(s * r)); h = s / w
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class _Kalman7:
    """SORT 论文里的 7 维恒速卡尔曼，纯 NumPy。"""

    def __init__(self):
        self.F = np.eye(7)
        for i, j in ((0, 4), (1, 5), (2, 6)):
            self.F[i, j] = 1.0
        self.H = np.zeros((4, 7))
        for i in range(4):
            self.H[i, i] = 1.0
        self.Q = np.eye(7); self.Q[4:, 4:] *= 0.01; self.Q[-1, -1] *= 0.01
        self.R = np.eye(4); self.R[2:, 2:] *= 10.0
        self.P = np.eye(7) * 10.0; self.P[4:, 4:] *= 1000.0
        self.x = np.zeros((7,))

    def init(self, z):
        self.x[:4] = z; self.x[4:] = 0.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.x[2] < 0:
            self.x[2] = 1e-6

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P


@dataclass
class _KTrack(Track):
    kf: Optional[_Kalman7] = None


class KalmanTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost: int = 30,
        min_hits: int = 3,
        history_len: int = 30,
        match: str = "hungarian",
        class_constrained: bool = False,
        conf_alpha: float = 0.2,
    ):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.history_len = history_len
        self.match = match
        self.class_constrained = class_constrained
        self.conf_alpha = conf_alpha
        self._tracks: Dict[int, _KTrack] = {}
        self._ids = count(1)

    def update(self, detections: List[Detection]) -> List[Track]:
        det_list = list(detections)
        track_ids = list(self._tracks.keys())

        for tid in track_ids:
            tr = self._tracks[tid]
            tr.kf.predict()
            tr.xyxy = _x_to_xyxy(tr.kf.x)

        matches, un_t, un_d = self._associate(track_ids, det_list)

        for tid, di in matches:
            tr = self._tracks[tid]; det = det_list[di]
            tr.kf.update(_xyxy_to_z(det.xyxy))
            tr.xyxy = _x_to_xyxy(tr.kf.x)
            tr.conf = (1 - self.conf_alpha) * tr.conf + self.conf_alpha * det.conf
            tr.cls = det.cls; tr.label = det.label
            tr.hits += 1; tr.lost = 0; tr.age += 1
            tr.history.append(tr.center)
            if len(tr.history) > self.history_len:
                tr.history.pop(0)

        for tid in un_t:
            self._tracks[tid].lost += 1
            self._tracks[tid].age += 1

        for tid in list(self._tracks.keys()):
            if self._tracks[tid].lost > self.max_lost:
                del self._tracks[tid]

        for di in un_d:
            det = det_list[di]
            new_id = next(self._ids)
            kf = _Kalman7(); kf.init(_xyxy_to_z(det.xyxy))
            self._tracks[new_id] = _KTrack(
                tid=new_id, cls=det.cls, label=det.label,
                xyxy=det.xyxy, conf=det.conf, kf=kf,
            )

        return [t for t in self._tracks.values()
                if t.hits >= self.min_hits and t.lost == 0]

    def _associate(self, track_ids, det_list):
        if not track_ids or not det_list:
            return [], list(track_ids), list(range(len(det_list)))
        trk = [self._tracks[tid].xyxy for tid in track_ids]
        det = [d.xyxy for d in det_list]
        iou = iou_matrix(trk, det)
        if self.class_constrained:
            tc = np.array([self._tracks[tid].cls for tid in track_ids])
            dc = np.array([d.cls for d in det_list])
            iou = np.where(tc[:, None] == dc[None, :], iou, 0.0)
        cost = 1.0 - iou
        max_cost = 1.0 - self.iou_threshold
        if self.match == "hungarian":
            r, c = linear_assignment(cost, max_cost=max_cost)
        else:
            r, c = greedy_assignment(cost, max_cost=max_cost)
        matches = [(track_ids[int(i)], int(j)) for i, j in zip(r, c)]
        mt = set(int(i) for i in r); md = set(int(j) for j in c)
        un_t = [track_ids[i] for i in range(len(track_ids)) if i not in mt]
        un_d = [j for j in range(len(det_list)) if j not in md]
        return matches, un_t, un_d


# =============================================================================
# 4) PID 控制器（只用来驱动窗口里的一个"跟随小圆点"）
# =============================================================================

@dataclass
class PID2D:
    kp: float = 0.2
    ki: float = 0.0
    kd: float = 0.05
    step_cap: float = 80.0
    dead_zone: float = 1.0
    smooth: float = 0.25
    i_cap: float = 200.0

    _ex: float = 0.0; _ey: float = 0.0
    _ix: float = 0.0; _iy: float = 0.0
    _ox: float = 0.0; _oy: float = 0.0
    _init: bool = False

    def reset(self):
        self._ex = self._ey = 0.0
        self._ix = self._iy = 0.0
        self._ox = self._oy = 0.0
        self._init = False

    def step(self, cur_xy, tgt_xy, dt: float = 1.0):
        cx, cy = cur_xy; tx, ty = tgt_xy
        if dt <= 0: dt = 1e-6
        return self._axis(cx, tx, dt, axis="x"), self._axis(cy, ty, dt, axis="y")

    def _axis(self, cur, tgt, dt, axis):
        err = tgt - cur
        last_e = self._ex if axis == "x" else self._ey
        integ = self._ix if axis == "x" else self._iy
        last_o = self._ox if axis == "x" else self._oy
        if abs(err) < self.dead_zone:
            out = self.smooth * last_o
        else:
            integ += err * dt
            integ = max(-self.i_cap, min(self.i_cap, integ))
            deriv = (err - last_e) / dt if self._init else 0.0
            raw = self.kp * err + self.ki * integ + self.kd * deriv
            raw = max(-self.step_cap, min(self.step_cap, raw))
            out = self.smooth * last_o + (1 - self.smooth) * raw
        if axis == "x":
            self._ex = err; self._ix = integ; self._ox = out
        else:
            self._ey = err; self._iy = integ; self._oy = out
        self._init = True
        return out


# =============================================================================
# 5) TensorRT 检测器（兼容 TRT 8 / 10）
# =============================================================================

class TrtYoloDetector:
    """
    最小可用的 TensorRT YOLOv8 推理封装。

    假设：
        输入  (1, 3, H, W)   float32 / float16   RGB 0~1
        输出  (1, 4+nc, N) 或 (1, N, 4+nc)       4=xywh（中心），nc 个类别分数（已 sigmoid）
    """

    def __init__(
        self,
        engine_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[Sequence[int]] = None,
        names: Optional[Sequence[str]] = None,
    ):
        try:
            import tensorrt as trt  # type: ignore
            import pycuda.autoinit  # type: ignore  # noqa: F401
            import pycuda.driver as cuda  # type: ignore
        except ImportError as e:
            raise ImportError(
                "需要安装 tensorrt 和 pycuda:  pip install pycuda\n"
                "TensorRT 请按 NVIDIA 官方指引安装并匹配 CUDA 版本。"
            ) from e

        self._trt = trt
        self._cuda = cuda

        if not Path(engine_path).is_file():
            raise FileNotFoundError(f"engine 文件不存在: {engine_path}")

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            engine = rt.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"无法反序列化 engine: {engine_path}")
        self._engine = engine
        self._context = engine.create_execution_context()
        self._stream = cuda.Stream()

        # 区分 TRT 10（新 API）/ TRT 8（旧 API）
        self._is_trt10 = hasattr(engine, "num_io_tensors")
        self._bindings_info: List[dict] = []
        if self._is_trt10:
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape = tuple(engine.get_tensor_shape(name))
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                self._bindings_info.append(dict(
                    name=name, is_input=is_input, shape=shape, dtype=dtype,
                ))
        else:
            for i in range(engine.num_bindings):
                name = engine.get_binding_name(i)
                is_input = engine.binding_is_input(i)
                shape = tuple(engine.get_binding_shape(i))
                dtype = trt.nptype(engine.get_binding_dtype(i))
                self._bindings_info.append(dict(
                    name=name, is_input=is_input, shape=shape, dtype=dtype, index=i,
                ))

        # 取输入形状。若是动态（-1），就用 640x640 兜底，并在 set_input_shape 里设定。
        input_info = next(b for b in self._bindings_info if b["is_input"])
        in_shape = list(input_info["shape"])
        if any(d < 0 for d in in_shape):
            in_shape = [1, 3, 640, 640]
            if self._is_trt10:
                self._context.set_input_shape(input_info["name"], tuple(in_shape))
            else:
                self._context.set_binding_shape(
                    input_info.get("index", 0), tuple(in_shape)
                )
        self._input_name = input_info["name"]
        self._input_shape = tuple(in_shape)
        self._input_dtype = input_info["dtype"]
        self.imgsz = int(in_shape[-1])

        # 读取/更新输出形状（动态网络可能要在 set_input_shape 之后才能拿到最终输出形状）
        self._outputs: List[dict] = []
        for b in self._bindings_info:
            if b["is_input"]:
                continue
            if self._is_trt10:
                shape = tuple(self._context.get_tensor_shape(b["name"]))
            else:
                shape = tuple(self._context.get_binding_shape(b.get("index")))
            self._outputs.append(dict(name=b["name"], shape=shape, dtype=b["dtype"]))

        # 预分配 host / device 缓冲
        self._d_input = cuda.mem_alloc(int(np.prod(in_shape)) * np.dtype(self._input_dtype).itemsize)
        self._out_bufs = []
        for o in self._outputs:
            size = int(abs(np.prod(o["shape"]))) * np.dtype(o["dtype"]).itemsize
            d = cuda.mem_alloc(size)
            h = cuda.pagelocked_empty(int(abs(np.prod(o["shape"]))), dtype=o["dtype"])
            self._out_bufs.append(dict(d=d, h=h, shape=o["shape"], name=o["name"]))

        if self._is_trt10:
            self._context.set_tensor_address(self._input_name, int(self._d_input))
            for o in self._out_bufs:
                self._context.set_tensor_address(o["name"], int(o["d"]))

        self.conf = conf
        self.iou = iou
        self.classes = set(classes) if classes else None
        self.names = tuple(names) if names is not None else tuple(COCO80_NAMES)
        self._last_preproc = None

    # ---- 前处理 ----
    def _preprocess(self, frame_bgr: np.ndarray):
        img_lb, r, pad = letterbox(frame_bgr, new_shape=self.imgsz)
        x = img_lb[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, 0~1
        x = np.transpose(x, (2, 0, 1))[None]               # NCHW
        x = np.ascontiguousarray(x, dtype=self._input_dtype)
        self._last_preproc = (r, pad, frame_bgr.shape)
        return x

    # ---- 推理 ----
    def _infer(self, x: np.ndarray) -> List[np.ndarray]:
        cuda = self._cuda
        cuda.memcpy_htod_async(self._d_input, x, self._stream)
        if self._is_trt10:
            self._context.execute_async_v3(stream_handle=self._stream.handle)
        else:
            bindings = [0] * len(self._bindings_info)
            for b in self._bindings_info:
                if b["is_input"]:
                    bindings[b["index"]] = int(self._d_input)
                else:
                    for o in self._out_bufs:
                        if o["name"] == b["name"]:
                            bindings[b["index"]] = int(o["d"])
                            break
            self._context.execute_async_v2(
                bindings=bindings, stream_handle=self._stream.handle
            )
        outs = []
        for o in self._out_bufs:
            cuda.memcpy_dtoh_async(o["h"], o["d"], self._stream)
            outs.append(o)
        self._stream.synchronize()
        # 复制成 numpy ndarray 并 reshape
        results: List[np.ndarray] = []
        for o in outs:
            arr = np.array(o["h"]).reshape(o["shape"])
            results.append(arr)
        return results

    # ---- 后处理 ----
    def _postprocess(self, pred: np.ndarray) -> List[Detection]:
        r, pad, orig_shape = self._last_preproc
        if pred.ndim != 3:
            raise RuntimeError(f"unexpected YOLO output shape: {pred.shape}")
        # 统一到 (N, C)：C = 4 + nc
        _, d1, d2 = pred.shape
        if d1 <= d2:  # (1, C, N)
            pred = np.transpose(pred, (0, 2, 1))
        p = pred[0]  # (N, C)
        boxes_xywh = p[:, :4]
        cls_scores = p[:, 4:]
        cls_ids = cls_scores.argmax(axis=1)
        cls_conf = cls_scores.max(axis=1)

        keep = cls_conf > self.conf
        if self.classes is not None:
            keep &= np.isin(cls_ids, list(self.classes))
        if not keep.any():
            return []

        boxes_xywh = boxes_xywh[keep]
        cls_ids = cls_ids[keep]
        cls_conf = cls_conf[keep]

        xy = boxes_xywh[:, :2]; wh = boxes_xywh[:, 2:]
        x1y1 = xy - wh / 2; x2y2 = xy + wh / 2
        boxes = np.concatenate([x1y1, x2y2], axis=1)
        boxes = scale_boxes(boxes, r, pad, orig_shape)
        idx = nms_xyxy_per_class(boxes, cls_conf, cls_ids, iou_thr=self.iou)

        out: List[Detection] = []
        for i in idx:
            x1, y1, x2, y2 = boxes[i].tolist()
            k = int(cls_ids[i])
            label = self.names[k] if 0 <= k < len(self.names) else str(k)
            out.append(Detection(
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
                conf=float(cls_conf[i]), cls=k, label=label,
            ))
        return out

    def __call__(self, frame_bgr: np.ndarray) -> List[Detection]:
        x = self._preprocess(frame_bgr)
        outs = self._infer(x)
        # 取第一个 3D 输出当作检测结果（YOLOv8 导出时通常只有一个）
        pred = next((o for o in outs if o.ndim == 3), outs[0])
        return self._postprocess(pred)


# =============================================================================
# 6) 采集源
# =============================================================================

class _SrcBase:
    def __iter__(self): return self
    def __next__(self) -> np.ndarray: raise NotImplementedError
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


class VideoSrc(_SrcBase):
    def __init__(self, s):
        self.cap = cv2.VideoCapture(s)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {s}")
    def __next__(self):
        ok, f = self.cap.read()
        if not ok: raise StopIteration
        return f
    def close(self): self.cap.release()


class FolderSrc(_SrcBase):
    _E = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    def __init__(self, folder):
        self.paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in self._E)
        if not self.paths:
            raise RuntimeError(f"目录里没有图片: {folder}")
        self._i = 0
    def __next__(self):
        if self._i >= len(self.paths): raise StopIteration
        img = cv2.imread(str(self.paths[self._i])); self._i += 1
        return img if img is not None else next(self)


class ScreenSrc(_SrcBase):
    def __init__(self, monitor: int = 1):
        try:
            import mss  # type: ignore
        except ImportError as e:
            raise ImportError("需要先安装 mss: pip install mss") from e
        self._sct = mss.mss()
        self._mon = self._sct.monitors[monitor]
    def __next__(self):
        shot = np.array(self._sct.grab(self._mon))  # BGRA
        return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
    def close(self): self._sct.close()


def open_source(spec: str) -> _SrcBase:
    if spec.startswith("screen"):
        parts = spec.split(":")
        mon = int(parts[1]) if len(parts) > 1 else 1
        return ScreenSrc(monitor=mon)
    if spec.isdigit():
        return VideoSrc(int(spec))
    p = Path(spec)
    if p.is_dir():
        return FolderSrc(spec)
    if p.is_file():
        return VideoSrc(spec)
    raise ValueError(f"无法识别的采集源: {spec}")


# =============================================================================
# 7) 可视化
# =============================================================================

_PALETTE = [(255, 64, 64), (64, 255, 64), (64, 128, 255),
            (255, 192, 0), (255, 64, 255), (0, 255, 255),
            (128, 255, 128), (255, 128, 64)]


def color_for(i: int): return _PALETTE[i % len(_PALETTE)]


def draw_tracks(frame, tracks, show_history=True):
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.xyxy)
        color = color_for(t.cls)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"#{t.tid} {t.label} {t.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if show_history and len(t.history) >= 2:
            pts = np.array(t.history, dtype=np.int32)
            cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)


def draw_crosshair(frame, size=12, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1, cv2.LINE_AA)


def draw_stats(frame, lines):
    x, y = 10, 10; pad = 6; line_h = 18; w = 260
    h = line_h * len(lines) + pad * 2
    sub = frame[y:y + h, x:x + w]
    if sub.size:
        overlay = np.zeros_like(sub)
        frame[y:y + h, x:x + w] = cv2.addWeighted(sub, 0.45, overlay, 0.55, 0)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x + pad, y + pad + (i + 1) * line_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


# =============================================================================
# 8) 选一个"best track"给 PID 跟随（离图像中心最近、大且置信度高）
# =============================================================================

def pick_best_track(tracks: List[Track], img_w: int, img_h: int) -> Optional[Track]:
    if not tracks:
        return None
    cx0, cy0 = img_w / 2.0, img_h / 2.0
    best = None
    best_score = -1e18
    for t in tracks:
        x1, y1, x2, y2 = t.xyxy
        w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
        tcx, tcy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = math.hypot(tcx - cx0, tcy - cy0)
        size = math.sqrt(w * h)
        score = t.conf * 300 + size * 0.6 - dist * 0.4
        if score > best_score:
            best_score = score; best = t
    return best


# =============================================================================
# 9) 主入口
# =============================================================================

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="0/1/... 摄像头；或视频文件；或图片目录；或 'screen'")
    ap.add_argument("--engine", default="test.engine",
                    help="TensorRT .engine 权重路径")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--classes", type=int, nargs="*", default=None)
    ap.add_argument("--class-names-file", default=None,
                    help="可选，一行一个类名的文本文件；不给则用 COCO80")
    # 跟踪
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--max-lost", type=int, default=30)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--match", choices=["greedy", "hungarian"], default="hungarian")
    ap.add_argument("--class-constrained", action="store_true")
    # PID 跟随小圆点
    ap.add_argument("--pid-kp", type=float, default=0.25)
    ap.add_argument("--pid-ki", type=float, default=0.0)
    ap.add_argument("--pid-kd", type=float, default=0.05)
    ap.add_argument("--pid-step-cap", type=float, default=80.0)
    # 输出
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--save-video", default=None)
    ap.add_argument("--save-fps", type=float, default=30.0)
    return ap.parse_args(argv)


def _load_names(path: Optional[str]) -> Sequence[str]:
    if path is None:
        return COCO80_NAMES
    with open(path, "r", encoding="utf-8") as f:
        return tuple(line.strip() for line in f if line.strip())


def main(argv=None) -> int:
    args = parse_args(argv)

    names = _load_names(args.class_names_file)
    detector = TrtYoloDetector(
        engine_path=args.engine,
        conf=args.conf,
        iou=args.iou,
        classes=args.classes,
        names=names,
    )
    tracker = KalmanTracker(
        iou_threshold=args.iou_threshold,
        max_lost=args.max_lost,
        min_hits=args.min_hits,
        match=args.match,
        class_constrained=args.class_constrained,
    )
    pid = PID2D(
        kp=args.pid_kp, ki=args.pid_ki, kd=args.pid_kd,
        step_cap=args.pid_step_cap, dead_zone=1.0, smooth=0.25,
    )

    show_stats = True; show_boxes = True; show_history = True; show_pid = True
    paused = False

    marker_xy: Optional[Tuple[float, float]] = None
    writer: Optional[cv2.VideoWriter] = None

    t_prev = time.perf_counter()
    fps_ema = 0.0
    frame = None

    with open_source(args.source) as src:
        it = iter(src)
        while True:
            if not paused:
                try:
                    frame = next(it)
                except StopIteration:
                    break
            if frame is None:
                break

            t0 = time.perf_counter()
            detections = detector(frame)
            t1 = time.perf_counter()
            tracks = tracker.update(detections)
            t2 = time.perf_counter()

            now = time.perf_counter()
            dt = max(1e-6, now - t_prev); t_prev = now
            inst_fps = 1.0 / dt
            fps_ema = 0.9 * fps_ema + 0.1 * inst_fps if fps_ema > 0 else inst_fps

            render = frame.copy()
            h, w = render.shape[:2]

            if show_boxes:
                draw_tracks(render, tracks, show_history=show_history)
            draw_crosshair(render)

            best = pick_best_track(tracks, w, h)
            if best is not None:
                tgt = best.center
                if marker_xy is None:
                    marker_xy = (w / 2.0, h / 2.0)
                dx, dy = pid.step(marker_xy, tgt, dt=max(dt * 60.0, 1e-3))
                marker_xy = (marker_xy[0] + dx, marker_xy[1] + dy)
            else:
                pid.reset()
                marker_xy = None

            if show_pid and marker_xy is not None:
                mx, my = int(marker_xy[0]), int(marker_xy[1])
                cv2.circle(render, (mx, my), 8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(render, (mx, my), 2, (0, 255, 255), -1, cv2.LINE_AA)
                if best is not None:
                    tx, ty = int(best.center[0]), int(best.center[1])
                    cv2.line(render, (mx, my), (tx, ty),
                             (0, 200, 200), 1, cv2.LINE_AA)

            if show_stats:
                draw_stats(render, [
                    f"FPS      : {fps_ema:5.1f}",
                    f"Det time : {(t1 - t0) * 1000:5.1f} ms",
                    f"Trk time : {(t2 - t1) * 1000:5.1f} ms",
                    f"Dets     : {len(detections)}",
                    f"Tracks   : {len(tracks)}",
                    f"Engine   : {os.path.basename(args.engine)}",
                    f"Best     : {('#'+str(best.tid)+' '+best.label) if best else '-'}",
                ])

            if args.save_video:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save_video, fourcc,
                                             float(args.save_fps), (w, h))
                writer.write(render)

            if args.no_window:
                continue

            cv2.imshow("track_app", render)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("h"): show_stats = not show_stats
            elif key == ord("b"): show_boxes = not show_boxes
            elif key == ord("t"): show_history = not show_history
            elif key == ord("p"): show_pid = not show_pid
            elif key == ord(" "): paused = not paused

    if writer is not None:
        writer.release()
    if not args.no_window:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
