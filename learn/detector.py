# -*- coding: utf-8 -*-
"""
detector.py — 目标检测器封装
===========================================
提供一个与框架无关的 Detector 接口：
    detections = detector(frame_bgr)
每个 detection 是一个 dict:
    {"xyxy": (x1, y1, x2, y2), "conf": float, "cls": int, "label": str}

默认实现用 Ultralytics YOLO（最容易跑起来，`pip install ultralytics`），
并给出如何切换到 TensorRT .engine 的说明和接口占位。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    xyxy: tuple
    conf: float
    cls: int
    label: str


class UltralyticsDetector:
    """
    基于 Ultralytics YOLO 的检测器。

    参数：
        weights: 本地权重或名称，例如 "yolov8n.pt"（首次会自动下载）
        conf:    置信度阈值
        iou:     NMS IoU 阈值
        classes: 只保留哪些类别 id（None 表示全部）
        device:  "cpu" / "cuda" / "cuda:0" 等
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        device: Optional[str] = None,
        imgsz: int = 640,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "需要先安装 ultralytics: pip install ultralytics"
            ) from e
        self._model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.device = device
        self.imgsz = imgsz
        self.names = self._model.names  # {id: name}

    def __call__(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        out: List[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            out.append(
                Detection(
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    conf=float(c),
                    cls=int(k),
                    label=str(self.names.get(int(k), str(int(k)))),
                )
            )
        return out


# -------------------------------------------------------------------------
# 进阶占位：TensorRT 引擎推理
# -------------------------------------------------------------------------
# 学习 TensorRT 的最小骨架：用 pycuda + tensorrt 直接跑 .engine。
# 你自己先用 `trtexec` 或 ultralytics 的 export 生成 engine。
#
# 这里只给出"如何搭骨架"的说明，而不是可运行的完整实现，
# 避免对运行环境（CUDA/驱动版本）做过多假设。
#
# 关键步骤（伪代码）：
#   1) 反序列化 engine
#       import tensorrt as trt
#       logger = trt.Logger(trt.Logger.WARNING)
#       with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
#           engine = rt.deserialize_cuda_engine(f.read())
#       context = engine.create_execution_context()
#
#   2) 分配 host / device 缓冲区，绑定每个输入/输出 Tensor
#
#   3) 前处理： letterbox 到 imgsz，BGR->RGB，/255，HWC->CHW，拷到 device
#
#   4) context.execute_v2(bindings=...) 或 execute_async_v3
#
#   5) 后处理： 从 (num, 4+1+nc) 解 xyxy+conf+cls，再做 NMS
#      （可用 torchvision.ops.nms；输入输出只要在同一设备就行）
#
# 建议先把 UltralyticsDetector 跑通，再用 TRTDetector 做性能对比。
# -------------------------------------------------------------------------


def build_detector(kind: str = "ultralytics", **kwargs):
    """工厂函数，便于 main 里切换实现。"""
    if kind == "ultralytics":
        return UltralyticsDetector(**kwargs)
    raise ValueError(f"未知的 detector 类型: {kind}")
