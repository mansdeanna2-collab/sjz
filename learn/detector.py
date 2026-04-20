# -*- coding: utf-8 -*-
"""
detector.py — 目标检测器封装
===========================================
提供与框架无关的 Detector 接口：``detections = detector(frame_bgr)``，
其中每个 detection 是一个带 ``xyxy / conf / cls / label`` 的 dataclass。

目前提供两个真实实现 + 一个占位说明：

* :class:`UltralyticsDetector`   —— 基于 ultralytics YOLO，最易上手。
* :class:`OnnxYoloDetector`      —— 基于 onnxruntime 加载 YOLOv8 导出的
  ONNX 权重，自己做 letterbox + NMS。用来学习"不依赖 ultralytics 的推理链路"。
* TensorRT 骨架说明见文件尾部注释。

所有后端共用 :mod:`learn.preprocess` 里的 ``letterbox`` / ``scale_boxes``
/ ``nms_xyxy_per_class``，前后处理可以独立单测。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .preprocess import letterbox, nms_xyxy_per_class, scale_boxes


@dataclass
class Detection:
    xyxy: tuple
    conf: float
    cls: int
    label: str


# ---------------------------------------------------------------------------
# 后端 1：Ultralytics YOLO（高层 API，最简单）
# ---------------------------------------------------------------------------

class UltralyticsDetector:
    """基于 Ultralytics YOLO 的检测器（``pip install ultralytics``）。"""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        device: Optional[str] = None,
        imgsz: int = 640,
    ) -> None:
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
            out.append(Detection(
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
                conf=float(c),
                cls=int(k),
                label=str(self.names.get(int(k), str(int(k)))),
            ))
        return out


# ---------------------------------------------------------------------------
# 后端 2：ONNX Runtime（YOLOv8 ONNX）
# ---------------------------------------------------------------------------
# 适合学习的原因：
#   * 没有 ultralytics 的"黑盒 predict"，所有前 / 后处理都写在 Python 里。
#   * 能跑在任何装了 onnxruntime 的机器上（CPU 就能跑）。
#   * 导出 ONNX：`yolo export model=yolov8n.pt format=onnx imgsz=640`。
# ---------------------------------------------------------------------------

# COCO 80 类（YOLOv8 官方权重默认类名），如果你换了权重请自行覆盖。
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


class OnnxYoloDetector:
    """加载一个 YOLOv8 的 ONNX 权重，用 onnxruntime 推理。

    假设输入是 (1, 3, H, W)、float32、0~1，输出是 (1, 84, N)
    或 (1, N, 84)（YOLOv8 默认格式：4 bbox + 80 类分数）。
    两种布局会自动识别。
    """

    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        classes: Optional[List[int]] = None,
        names: Optional[Sequence[str]] = None,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as e:
            raise ImportError(
                "需要先安装 onnxruntime: pip install onnxruntime"
            ) from e

        if providers is None:
            providers = ["CPUExecutionProvider"]
        self._sess = ort.InferenceSession(weights, providers=list(providers))
        self._input_name = self._sess.get_inputs()[0].name

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = set(classes) if classes else None
        self.names = tuple(names) if names is not None else tuple(COCO80_NAMES)

    def __call__(self, frame_bgr: np.ndarray) -> List[Detection]:
        img_lb, ratio, pad = letterbox(frame_bgr, new_shape=self.imgsz)
        x = img_lb[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, 0~1
        x = np.transpose(x, (2, 0, 1))[None]               # HWC -> NCHW

        outputs = self._sess.run(None, {self._input_name: x})
        pred = outputs[0]
        if pred.ndim != 3:
            raise RuntimeError(f"unexpected YOLO output shape: {pred.shape}")
        if pred.shape[1] in (84, 85) and pred.shape[1] < pred.shape[2]:
            pred = np.transpose(pred, (0, 2, 1))  # -> (1, N, C)
        pred = pred[0]  # (N, C)

        # YOLOv8: 前 4 列 xywh（center），后 nc 列是类别分数（已含 sigmoid）
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]
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

        # xywh (center) -> xyxy
        xy = boxes_xywh[:, :2]
        wh = boxes_xywh[:, 2:]
        x1y1 = xy - wh / 2
        x2y2 = xy + wh / 2
        boxes_xyxy = np.concatenate([x1y1, x2y2], axis=1)

        boxes_xyxy = scale_boxes(boxes_xyxy, ratio, pad, frame_bgr.shape)
        keep_idx = nms_xyxy_per_class(boxes_xyxy, cls_conf, cls_ids, iou_thr=self.iou)

        out: List[Detection] = []
        for i in keep_idx:
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            k = int(cls_ids[i])
            label = self.names[k] if 0 <= k < len(self.names) else str(k)
            out.append(Detection(
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
                conf=float(cls_conf[i]),
                cls=k,
                label=label,
            ))
        return out


# ---------------------------------------------------------------------------
# 后端 3：TensorRT 引擎推理（骨架说明）
# ---------------------------------------------------------------------------
# 关键步骤（伪代码）：
#   1) 反序列化 engine
#       import tensorrt as trt
#       logger = trt.Logger(trt.Logger.WARNING)
#       with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
#           engine = rt.deserialize_cuda_engine(f.read())
#       context = engine.create_execution_context()
#   2) 分配 host / device 缓冲区，绑定每个输入/输出 Tensor
#   3) 前处理：letterbox / BGR->RGB / /255 / HWC->CHW / 拷到 device
#   4) context.execute_v2(bindings=...) 或 execute_async_v3
#   5) 后处理：复用 learn.preprocess 里的 scale_boxes / nms_xyxy_per_class
# ---------------------------------------------------------------------------


def build_detector(kind: str = "ultralytics", **kwargs):
    """工厂：``kind`` = ``"ultralytics"`` | ``"onnx"``。"""
    kind = kind.lower()
    if kind == "ultralytics":
        return UltralyticsDetector(**kwargs)
    if kind == "onnx":
        return OnnxYoloDetector(**kwargs)
    raise ValueError(f"未知的 detector 类型: {kind}")


__all__ = [
    "Detection",
    "UltralyticsDetector",
    "OnnxYoloDetector",
    "build_detector",
    "COCO80_NAMES",
]
