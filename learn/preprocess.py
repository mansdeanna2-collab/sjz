# -*- coding: utf-8 -*-
"""
preprocess.py — YOLO 风格的前 / 后处理工具
===========================================
把"图像如何进神经网络"和"网络输出的框如何回到原图坐标"这两件事
独立出来，便于阅读、复用和测试。

函数：
    letterbox(img, new_shape=640, color=(114,114,114), stride=32)
        等比缩放 + 居中 padding 到固定大小，返回 (img, ratio, (pad_w, pad_h))。

    scale_boxes(boxes_xyxy, ratio, pad, orig_shape)
        把推理时坐标系下的框映射回原图坐标系。

    nms_xyxy(boxes, scores, iou_thr=0.45)
        类别无关的 NMS（纯 NumPy），返回保留的下标。

这些函数没有外部依赖（numpy + opencv-python），适合在 CPU 环境里做原型。
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def letterbox(
    img: np.ndarray,
    new_shape=640,
    color: Tuple[int, int, int] = (114, 114, 114),
    stride: int = 32,
    scaleup: bool = True,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    等比缩放到 ``new_shape`` 并 padding 到 stride 的整数倍。

    返回：
        img_out: BGR ndarray
        ratio:   原图到 letterbox 的缩放系数（长宽都用这一个）
        (pw, ph): 左 / 上的 padding 像素数（宽、高方向各一）

    典型用法：
        img_lb, ratio, (pw, ph) = letterbox(img, new_shape=640)
        # 推理得到 boxes_xyxy（在 img_lb 坐标系里）
        boxes_xyxy = scale_boxes(boxes_xyxy, ratio, (pw, ph), img.shape)
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    # 按 stride 取模，保持输出尺寸是 stride 的倍数
    dw = dw % stride
    dh = dh % stride
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
    """把 letterbox 坐标系下的 xyxy 映射回原图尺寸。"""
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    boxes = boxes_xyxy.astype(np.float32, copy=True)
    pw, ph = pad
    boxes[:, [0, 2]] -= pw
    boxes[:, [1, 3]] -= ph
    boxes[:, :4] /= max(ratio, 1e-9)
    h, w = orig_shape[:2]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


def nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thr: float = 0.45,
) -> np.ndarray:
    """
    类别无关的 NMS（纯 NumPy）。

    参数：
        boxes:  shape (N, 4) 的 xyxy 框
        scores: shape (N,) 的分数
        iou_thr: IoU 阈值，>= 此值的低分框会被抑制

    返回：
        keep: 保留下来的下标，按分数降序
    """
    if boxes.size == 0:
        return np.array([], dtype=int)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thr: float = 0.45,
) -> np.ndarray:
    """按类别分别做 NMS，然后合并结果。"""
    if boxes.size == 0:
        return np.array([], dtype=int)
    keep_all = []
    for c in np.unique(classes):
        mask = classes == c
        idx = np.where(mask)[0]
        local_keep = nms_xyxy(boxes[idx], scores[idx], iou_thr=iou_thr)
        keep_all.extend(idx[local_keep].tolist())
    # 按 score 降序
    keep_all = np.array(keep_all, dtype=int)
    return keep_all[np.argsort(-scores[keep_all])]


__all__ = ["letterbox", "scale_boxes", "nms_xyxy", "nms_xyxy_per_class"]
