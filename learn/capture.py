# -*- coding: utf-8 -*-
"""
capture.py — 统一的视频帧来源
===========================================
把"从哪拿到一帧图像"这件事抽象成一个迭代器，
底层可以是：摄像头 / 视频文件 / 一张图片 / 图片目录 / 桌面截屏。

这样上层的检测/跟踪/绘制代码完全不用关心数据是从哪儿来的，
便于你对比不同采集方式的性能（延迟、帧率、CPU/GPU 占用）。

用法：
    with open_source("0") as src:          # 摄像头 0
        for frame in src:
            ...

    with open_source("video.mp4") as src:  # 视频文件
        ...

    with open_source("screen") as src:     # 桌面全屏（需要 mss）
        ...
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


class _BaseSource:
    """所有采集源的公共接口。"""

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class VideoSource(_BaseSource):
    """OpenCV VideoCapture 封装，支持摄像头下标或视频文件路径。"""

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {src}")

    def __next__(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise StopIteration
        return frame

    def close(self) -> None:
        self.cap.release()


class ImageFolderSource(_BaseSource):
    """遍历一个图片目录，适合做离线评测。"""

    _EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, folder: str):
        self.paths = sorted(
            p for p in Path(folder).iterdir() if p.suffix.lower() in self._EXTS
        )
        if not self.paths:
            raise RuntimeError(f"目录里没有图片: {folder}")
        self._i = 0

    def __next__(self) -> np.ndarray:
        if self._i >= len(self.paths):
            raise StopIteration
        img = cv2.imread(str(self.paths[self._i]))
        self._i += 1
        if img is None:
            return next(self)
        return img


class ScreenSource(_BaseSource):
    """桌面截屏（需要 mss）。只抓你自己的桌面用于学习采集性能。"""

    def __init__(self, monitor: int = 1):
        try:
            import mss  # 延迟导入，避免不用时强依赖
        except ImportError as e:
            raise ImportError("需要先安装 mss: pip install mss") from e
        self._sct = mss.mss()
        self._mon = self._sct.monitors[monitor]

    def __next__(self) -> np.ndarray:
        shot = np.array(self._sct.grab(self._mon))  # BGRA
        return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self._sct.close()


def open_source(spec: str) -> _BaseSource:
    """
    根据字符串规格打开一个采集源。

    spec 规则：
        "0", "1", ...            -> 摄像头下标
        "screen" / "screen:2"    -> 桌面截屏（可选指定 monitor 序号）
        路径（文件）              -> 视频文件
        路径（目录）              -> 图片目录
    """
    if spec.startswith("screen"):
        parts = spec.split(":")
        monitor = int(parts[1]) if len(parts) > 1 else 1
        return ScreenSource(monitor=monitor)

    if spec.isdigit():
        return VideoSource(int(spec))

    path = Path(spec)
    if path.is_dir():
        return ImageFolderSource(spec)
    if path.is_file():
        return VideoSource(spec)

    raise ValueError(f"无法识别的采集源: {spec}")


class FPSMeter:
    """滑动窗口 FPS 计数器，上层显示和日志都可以用。"""

    def __init__(self, window: int = 30):
        self.window = window
        self._stamps: list[float] = []

    def tick(self) -> float:
        now = time.perf_counter()
        self._stamps.append(now)
        if len(self._stamps) > self.window:
            self._stamps.pop(0)
        if len(self._stamps) < 2:
            return 0.0
        return (len(self._stamps) - 1) / (self._stamps[-1] - self._stamps[0])
