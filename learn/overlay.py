# -*- coding: utf-8 -*-
"""
overlay.py — OpenCV 画面叠加
===========================================
把检测/跟踪结果画在视频帧上，并在角落显示 FPS、计数等统计信息。

相比原脚本用 Win32 透明窗叠加在别的进程上，
这里全部画在"我们自己的"视频显示窗口里，简单、跨平台、也方便理解。
"""
from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

# 给不同类别挑一组稳定、对比度高的颜色
_PALETTE = [
    (255, 64, 64),   (64, 255, 64),   (64, 128, 255),
    (255, 192, 0),   (255, 64, 255),  (0, 255, 255),
    (128, 255, 128), (255, 128, 64),
]


def color_for(idx: int) -> Tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


def draw_track(frame: np.ndarray, track, show_history: bool = True) -> None:
    """画一个跟踪框 + 标签 + 轨迹线。"""
    x1, y1, x2, y2 = map(int, track.xyxy)
    color = color_for(track.cls)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"#{track.tid} {track.label} {track.conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if show_history and len(track.history) >= 2:
        pts = np.array(track.history, dtype=np.int32)
        cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)


def draw_tracks(frame: np.ndarray, tracks: Iterable, show_history: bool = True) -> None:
    for t in tracks:
        draw_track(frame, t, show_history=show_history)


def draw_crosshair(frame: np.ndarray, size: int = 12,
                   color=(255, 255, 255)) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1, cv2.LINE_AA)


def draw_stats(frame: np.ndarray, lines) -> None:
    """左上角多行统计文字（带半透明底板）。"""
    x, y = 10, 10
    pad = 6
    line_h = 18
    w = 220
    h = line_h * len(lines) + pad * 2
    # 半透明底板
    sub = frame[y:y + h, x:x + w]
    if sub.size:
        overlay = np.zeros_like(sub)
        frame[y:y + h, x:x + w] = cv2.addWeighted(sub, 0.45, overlay, 0.55, 0)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x + pad, y + pad + (i + 1) * line_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
