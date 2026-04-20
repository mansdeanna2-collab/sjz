# -*- coding: utf-8 -*-
"""
tracker.py — 轻量多目标跟踪器（IoU 匹配 + EMA 平滑）
===========================================
复刻原脚本中的核心跟踪思路，但以通用、可阅读的方式重新实现：

    1. 用 IoU 在"上一帧轨迹"和"当前帧检测"之间做贪心匹配
    2. 匹配到的轨迹：用 EMA 对 bbox 做平滑（减少抖动）
    3. 没匹配到的检测：注册为新轨迹（需要累计 min_hits 次再展示）
    4. 没匹配到的轨迹：累计 lost 次数，超过 max_lost 就删除

对应原脚本里的参数：
    TRK_IOU_MATCH_BASE, TRK_MAX_LOST, TRK_MIN_HITS, EMA_ALPHA_*
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Dict, List, Optional, Tuple

import numpy as np


def iou_xyxy(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
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
    history: List[Tuple[float, float]] = field(default_factory=list)  # 中心点轨迹

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


class IouTracker:
    """IoU + EMA 的轻量跟踪器，API 仿照 SORT。"""

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_lost: int = 8,
        min_hits: int = 2,
        ema_alpha: float = 0.5,
        conf_alpha: float = 0.2,
        history_len: int = 30,
    ):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.ema_alpha = ema_alpha          # bbox EMA 权重（新观测占比）
        self.conf_alpha = conf_alpha        # 置信度 EMA 权重（新观测占比）
        self.history_len = history_len
        self._tracks: Dict[int, Track] = {}
        self._ids = count(1)

    # ---------- 外部接口 ----------
    def update(self, detections) -> List[Track]:
        """
        detections: 一个可迭代对象，每项必须有 .xyxy / .conf / .cls / .label
        返回：稳定展示（hits >= min_hits 且 lost == 0）的 Track 列表
        """
        track_ids = list(self._tracks.keys())
        det_list = list(detections)

        # 1) 构造 IoU 代价矩阵并贪心匹配
        matches, unmatched_tracks, unmatched_dets = self._match(track_ids, det_list)

        # 2) 更新匹配上的轨迹
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

        # 3) 未匹配的旧轨迹：lost+1
        for tid in unmatched_tracks:
            self._tracks[tid].lost += 1
            self._tracks[tid].age += 1

        # 4) 删除丢失太久的
        for tid in list(self._tracks.keys()):
            if self._tracks[tid].lost > self.max_lost:
                del self._tracks[tid]

        # 5) 未匹配的检测：新建轨迹
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

        # 6) 返回稳定轨迹
        return [t for t in self._tracks.values()
                if t.hits >= self.min_hits and t.lost == 0]

    # ---------- 内部工具 ----------
    def _match(self, track_ids, det_list):
        if not track_ids or not det_list:
            return [], list(track_ids), list(range(len(det_list)))

        iou_mat = np.zeros((len(track_ids), len(det_list)), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(det_list):
                iou_mat[i, j] = iou_xyxy(self._tracks[tid].xyxy, det.xyxy)

        matches: List[Tuple[int, int]] = []
        used_t, used_d = set(), set()
        # 贪心：每次选全局最大 IoU
        flat = [(iou_mat[i, j], i, j)
                for i in range(iou_mat.shape[0])
                for j in range(iou_mat.shape[1])]
        flat.sort(reverse=True)
        for v, i, j in flat:
            if v < self.iou_threshold:
                break
            if i in used_t or j in used_d:
                continue
            matches.append((track_ids[i], j))
            used_t.add(i)
            used_d.add(j)

        unmatched_tracks = [track_ids[i] for i in range(len(track_ids)) if i not in used_t]
        unmatched_dets = [j for j in range(len(det_list)) if j not in used_d]
        return matches, unmatched_tracks, unmatched_dets

    def _ema_box(self, old, new):
        a = self.ema_alpha
        return tuple(a * n + (1 - a) * o for o, n in zip(old, new))
