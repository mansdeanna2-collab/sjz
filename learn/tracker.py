# -*- coding: utf-8 -*-
"""
tracker.py — 多目标跟踪器
===========================================
本模块提供两种跟踪器，接口完全一致：

1. :class:`IouTracker`
    * IoU 在"上一帧轨迹"和"当前帧检测"之间做关联
    * 关联方式可选：贪心 (``match="greedy"``) 或最优指派 (``match="hungarian"``)
    * 可选 ``class_constrained=True``：不同类别的检测与轨迹互不相认
    * 匹配上的轨迹：bbox 做 EMA 平滑 + 置信度 EMA
    * 丢失超过 ``max_lost`` 帧则移除

2. :class:`KalmanTracker`
    * SORT 风格：每条轨迹内部维护一个 **恒速卡尔曼滤波器**
      （状态 = [cx, cy, s=w*h, r=w/h, vx, vy, vs]，观测 = [cx, cy, s, r]）
    * 关联时拿滤波器的**预测**作为"轨迹当前位置"，而不是上一帧观测
    * 比 :class:`IouTracker` 在遮挡 / 快速运动下 ID 更稳

对应原脚本里的概念：
    TRK_IOU_MATCH_BASE  -> iou_threshold
    TRK_MAX_LOST        -> max_lost
    TRK_MIN_HITS        -> min_hits
    EMA_ALPHA_*         -> ema_alpha / conf_alpha
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .assignment import greedy_assignment, linear_assignment


# --------------------------- 基础几何 ---------------------------

def iou_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """计算两个 (x1, y1, x2, y2) 框的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(
    a_boxes: Sequence[Tuple[float, float, float, float]],
    b_boxes: Sequence[Tuple[float, float, float, float]],
) -> np.ndarray:
    """批量版 IoU。返回形状 (len(a), len(b)) 的矩阵。"""
    if len(a_boxes) == 0 or len(b_boxes) == 0:
        return np.zeros((len(a_boxes), len(b_boxes)), dtype=np.float32)
    a = np.asarray(a_boxes, dtype=np.float32)
    b = np.asarray(b_boxes, dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out.astype(np.float32)


# --------------------------- 通用 Track 数据结构 ---------------------------

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
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


# --------------------------- IoU 跟踪器 ---------------------------

class IouTracker:
    """IoU + EMA 跟踪器。

    参数：
        iou_threshold:     关联阈值，IoU < 此值不认为是一对
        max_lost:          连续丢失多少帧就删除
        min_hits:          累计命中多少帧才展示（防抖生）
        ema_alpha:         bbox 的 EMA 权重（新观测占比）
        conf_alpha:        conf 的 EMA 权重
        history_len:       轨迹历史点数上限
        match:             "greedy" 或 "hungarian"
        class_constrained: True 时只有同类别的轨迹/检测才能配对
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_lost: int = 8,
        min_hits: int = 2,
        ema_alpha: float = 0.5,
        conf_alpha: float = 0.2,
        history_len: int = 30,
        match: str = "greedy",
        class_constrained: bool = False,
    ) -> None:
        if match not in ("greedy", "hungarian"):
            raise ValueError(f"未知的 match 模式: {match}")
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.ema_alpha = ema_alpha
        self.conf_alpha = conf_alpha
        self.history_len = history_len
        self.match_kind = match
        self.class_constrained = class_constrained
        self._tracks: Dict[int, Track] = {}
        self._ids = count(1)

    # ---------- 外部接口 ----------
    def update(self, detections) -> List[Track]:
        det_list = list(detections)
        track_ids = list(self._tracks.keys())

        matches, unmatched_tracks, unmatched_dets = self._match(track_ids, det_list)

        for tid, di in matches:
            tr = self._tracks[tid]
            det = det_list[di]
            tr.xyxy = self._ema_box(tr.xyxy, det.xyxy)
            tr.conf = (1 - self.conf_alpha) * tr.conf + self.conf_alpha * det.conf
            tr.cls = det.cls
            tr.label = det.label
            tr.hits += 1
            tr.lost = 0
            tr.age += 1
            tr.history.append(tr.center)
            if len(tr.history) > self.history_len:
                tr.history.pop(0)

        for tid in unmatched_tracks:
            self._tracks[tid].lost += 1
            self._tracks[tid].age += 1

        for tid in list(self._tracks.keys()):
            if self._tracks[tid].lost > self.max_lost:
                del self._tracks[tid]

        for di in unmatched_dets:
            det = det_list[di]
            new_id = next(self._ids)
            self._tracks[new_id] = Track(
                tid=new_id,
                cls=det.cls,
                label=det.label,
                xyxy=det.xyxy,
                conf=det.conf,
            )

        return [t for t in self._tracks.values()
                if t.hits >= self.min_hits and t.lost == 0]

    # ---------- 内部工具 ----------
    def _match(self, track_ids, det_list):
        if not track_ids or not det_list:
            return [], list(track_ids), list(range(len(det_list)))

        trk_boxes = [self._tracks[tid].xyxy for tid in track_ids]
        det_boxes = [d.xyxy for d in det_list]
        iou_mat = iou_matrix(trk_boxes, det_boxes)

        if self.class_constrained:
            trk_cls = np.array([self._tracks[tid].cls for tid in track_ids])
            det_cls = np.array([d.cls for d in det_list])
            same = trk_cls[:, None] == det_cls[None, :]
            iou_mat = np.where(same, iou_mat, 0.0)

        # cost = 1 - IoU；要求 cost < 1 - iou_threshold 才算关联
        cost = 1.0 - iou_mat
        max_cost = 1.0 - self.iou_threshold

        if self.match_kind == "greedy":
            rows, cols = greedy_assignment(cost, max_cost=max_cost)
        else:
            rows, cols = linear_assignment(cost, max_cost=max_cost)

        matches = [(track_ids[i], int(j)) for i, j in zip(rows, cols)]
        matched_t = set(int(i) for i in rows)
        matched_d = set(int(j) for j in cols)
        unmatched_tracks = [track_ids[i] for i in range(len(track_ids)) if i not in matched_t]
        unmatched_dets = [j for j in range(len(det_list)) if j not in matched_d]
        return matches, unmatched_tracks, unmatched_dets

    def _ema_box(self, old, new):
        a = self.ema_alpha
        return tuple(a * n + (1 - a) * o for o, n in zip(old, new))


# --------------------------- 卡尔曼跟踪器 (SORT 风格) ---------------------------

def _xyxy_to_z(xyxy):
    """(x1, y1, x2, y2) -> SORT 观测向量 z = [cx, cy, s, r]，s=w*h，r=w/h。"""
    x1, y1, x2, y2 = xyxy
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return np.array([cx, cy, w * h, w / h], dtype=np.float64)


def _x_to_xyxy(x):
    """状态向量 x=[cx,cy,s,r,vx,vy,vs] 的前 4 维转回 xyxy。"""
    cx, cy, s, r = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    s = max(1e-6, s)
    r = max(1e-6, r)
    w = float(np.sqrt(s * r))
    h = s / w
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class _KalmanBoxFilter:
    """一个简化的恒速模型卡尔曼滤波器（实现 SORT 论文里的 7 维状态）。

    状态  x = [cx, cy, s, r, vx, vy, vs]
    观测  z = [cx, cy, s, r]
    """

    def __init__(self):
        self.F = np.eye(7)
        for i, j in ((0, 4), (1, 5), (2, 6)):
            self.F[i, j] = 1.0
        self.H = np.zeros((4, 7))
        for i in range(4):
            self.H[i, i] = 1.0

        # 过程 / 观测噪声协方差（与 SORT 默认接近）
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01
        self.Q[-1, -1] *= 0.01

        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0

        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0  # 初始速度未知，给大不确定度

        self.x = np.zeros((7,))

    def init(self, z):
        self.x[:4] = z
        self.x[4:] = 0.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # s (面积) 不能为负
        if self.x[2] < 0:
            self.x[2] = 1e-6
        return self.x.copy()

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P


@dataclass
class _KTrack(Track):
    kf: Optional[_KalmanBoxFilter] = None


class KalmanTracker:
    """SORT 风格的卡尔曼跟踪器（纯 NumPy，不依赖 filterpy）。"""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost: int = 30,
        min_hits: int = 3,
        history_len: int = 30,
        match: str = "hungarian",
        class_constrained: bool = False,
        conf_alpha: float = 0.2,
    ) -> None:
        if match not in ("greedy", "hungarian"):
            raise ValueError(f"未知的 match 模式: {match}")
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.history_len = history_len
        self.match_kind = match
        self.class_constrained = class_constrained
        self.conf_alpha = conf_alpha
        self._tracks: Dict[int, _KTrack] = {}
        self._ids = count(1)

    def update(self, detections) -> List[Track]:
        det_list = list(detections)
        track_ids = list(self._tracks.keys())

        # 1) 先让每条轨迹向前预测一步
        for tid in track_ids:
            tr = self._tracks[tid]
            assert tr.kf is not None
            tr.kf.predict()
            tr.xyxy = _x_to_xyxy(tr.kf.x)

        # 2) 用预测出的 bbox 和本帧检测算关联
        matches, unmatched_tracks, unmatched_dets = self._match(track_ids, det_list)

        for tid, di in matches:
            tr = self._tracks[tid]
            det = det_list[di]
            assert tr.kf is not None
            tr.kf.update(_xyxy_to_z(det.xyxy))
            tr.xyxy = _x_to_xyxy(tr.kf.x)
            tr.conf = (1 - self.conf_alpha) * tr.conf + self.conf_alpha * det.conf
            tr.cls = det.cls
            tr.label = det.label
            tr.hits += 1
            tr.lost = 0
            tr.age += 1
            tr.history.append(tr.center)
            if len(tr.history) > self.history_len:
                tr.history.pop(0)

        for tid in unmatched_tracks:
            self._tracks[tid].lost += 1
            self._tracks[tid].age += 1

        for tid in list(self._tracks.keys()):
            if self._tracks[tid].lost > self.max_lost:
                del self._tracks[tid]

        for di in unmatched_dets:
            det = det_list[di]
            new_id = next(self._ids)
            kf = _KalmanBoxFilter()
            kf.init(_xyxy_to_z(det.xyxy))
            self._tracks[new_id] = _KTrack(
                tid=new_id,
                cls=det.cls,
                label=det.label,
                xyxy=det.xyxy,
                conf=det.conf,
                kf=kf,
            )

        return [t for t in self._tracks.values()
                if t.hits >= self.min_hits and t.lost == 0]

    def _match(self, track_ids, det_list):
        if not track_ids or not det_list:
            return [], list(track_ids), list(range(len(det_list)))

        trk_boxes = [self._tracks[tid].xyxy for tid in track_ids]
        det_boxes = [d.xyxy for d in det_list]
        iou_mat = iou_matrix(trk_boxes, det_boxes)

        if self.class_constrained:
            trk_cls = np.array([self._tracks[tid].cls for tid in track_ids])
            det_cls = np.array([d.cls for d in det_list])
            same = trk_cls[:, None] == det_cls[None, :]
            iou_mat = np.where(same, iou_mat, 0.0)

        cost = 1.0 - iou_mat
        max_cost = 1.0 - self.iou_threshold

        if self.match_kind == "greedy":
            rows, cols = greedy_assignment(cost, max_cost=max_cost)
        else:
            rows, cols = linear_assignment(cost, max_cost=max_cost)

        matches = [(track_ids[i], int(j)) for i, j in zip(rows, cols)]
        matched_t = set(int(i) for i in rows)
        matched_d = set(int(j) for j in cols)
        unmatched_tracks = [track_ids[i] for i in range(len(track_ids)) if i not in matched_t]
        unmatched_dets = [j for j in range(len(det_list)) if j not in matched_d]
        return matches, unmatched_tracks, unmatched_dets


def build_tracker(kind: str = "iou", **kwargs):
    """工厂：``kind`` = ``"iou"`` | ``"kalman"``。"""
    kind = kind.lower()
    if kind == "iou":
        return IouTracker(**kwargs)
    if kind == "kalman":
        return KalmanTracker(**kwargs)
    raise ValueError(f"未知的 tracker 类型: {kind}")


__all__ = [
    "Track",
    "IouTracker",
    "KalmanTracker",
    "build_tracker",
    "iou_xyxy",
    "iou_matrix",
]
