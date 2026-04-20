# -*- coding: utf-8 -*-
"""
demo_kalman.py — Kalman vs EMA 跟踪器的可视化对比
===========================================
构造一个"有噪声、偶尔丢失观测"的合成数据流：
    * 一个小球做曲线运动，每帧给一个带噪声的 bbox 检测
    * 以 ``drop_prob`` 的概率随机丢失这一帧的观测（模拟遮挡）

分别喂给 :class:`IouTracker`（EMA 平滑） 和 :class:`KalmanTracker`
（SORT 风格），把两者预测的轨迹叠加到同一画布上，直观观察：

    * 无遮挡时两者都能跟上
    * 有遮挡时，Kalman 能根据速度继续预测，EMA 会"原地停住"
    * 快速运动时，Kalman 的 ID 更稳，不容易断

运行：
    python -m learn.demo_kalman
    python -m learn.demo_kalman --frames 600 --drop 0.3 --no-window   # 无图形环境

按键：q / ESC 退出
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .tracker import IouTracker, KalmanTracker, Track


@dataclass
class _FakeDet:
    xyxy: Tuple[float, float, float, float]
    conf: float = 0.9
    cls: int = 0
    label: str = "ball"


def _gt_bbox(t: float, w: int, h: int, size: float = 40.0) -> Tuple[float, float, float, float]:
    cx = w * 0.5 + 0.35 * w * math.cos(t * 0.9)
    cy = h * 0.5 + 0.35 * h * math.sin(t * 1.3) + 0.05 * h * math.sin(t * 4.2)
    s = size
    return (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)


def _noisy(bbox, sigma: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    nx = np.random.normal(0.0, sigma, size=4)
    return (x1 + nx[0], y1 + nx[1], x2 + nx[2], y2 + nx[3])


def _draw_track(canvas, tracks: List[Track], color, label: str) -> None:
    for tr in tracks:
        x1, y1, x2, y2 = map(int, tr.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{label}#{tr.tid}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if len(tr.history) >= 2:
            pts = np.array(tr.history, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)


def run(
    frames: int = 600,
    drop_prob: float = 0.25,
    noise_sigma: float = 3.0,
    no_window: bool = False,
    seed: int = 42,
) -> dict:
    """运行仿真。返回一个统计字典（MAE、ID 切换次数）。"""
    W, H = 800, 600
    random.seed(seed)
    np.random.seed(seed)

    iou_trk = IouTracker(iou_threshold=0.1, max_lost=30, min_hits=1, ema_alpha=0.6)
    kal_trk = KalmanTracker(iou_threshold=0.1, max_lost=30, min_hits=1,
                            match="hungarian")

    mae_iou = 0.0
    mae_kal = 0.0
    count_iou = 0
    count_kal = 0
    iou_ids = set()
    kal_ids = set()

    for i in range(frames):
        t = i * 0.03
        gt = _gt_bbox(t, W, H)
        dets: List[_FakeDet] = []
        if random.random() >= drop_prob:
            dets.append(_FakeDet(xyxy=_noisy(gt, noise_sigma)))

        out_iou = iou_trk.update(dets)
        out_kal = kal_trk.update(dets)

        gt_c = ((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2)
        if out_iou:
            c = out_iou[0].center
            mae_iou += math.hypot(c[0] - gt_c[0], c[1] - gt_c[1])
            count_iou += 1
            iou_ids.add(out_iou[0].tid)
        if out_kal:
            c = out_kal[0].center
            mae_kal += math.hypot(c[0] - gt_c[0], c[1] - gt_c[1])
            count_kal += 1
            kal_ids.add(out_kal[0].tid)

        if no_window:
            continue

        canvas = np.full((H, W, 3), 30, dtype=np.uint8)
        # ground-truth
        x1, y1, x2, y2 = map(int, gt)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), 1)
        cv2.putText(canvas, "GT", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        # detection (if any)
        for d in dets:
            dx1, dy1, dx2, dy2 = map(int, d.xyxy)
            cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), (80, 80, 80), 1)

        _draw_track(canvas, out_iou, (80, 220, 80), "EMA")
        _draw_track(canvas, out_kal, (80, 160, 255), "KF")

        info = [
            f"frame {i+1}/{frames}  drop_prob={drop_prob:.2f}  noise={noise_sigma:.1f}",
            f"EMA     ids={len(iou_ids):<3} mae={mae_iou/max(count_iou,1):5.2f} px",
            f"Kalman  ids={len(kal_ids):<3} mae={mae_kal/max(count_kal,1):5.2f} px",
            "q / ESC to quit",
        ]
        for k, line in enumerate(info):
            cv2.putText(canvas, line, (12, 24 + k * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

        cv2.imshow("kalman vs ema", canvas)
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord("q")):
            break

    if not no_window:
        cv2.destroyAllWindows()

    return {
        "frames": frames,
        "drop_prob": drop_prob,
        "noise_sigma": noise_sigma,
        "mae_ema": mae_iou / max(count_iou, 1),
        "mae_kalman": mae_kal / max(count_kal, 1),
        "ids_ema": len(iou_ids),
        "ids_kalman": len(kal_ids),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--drop", type=float, default=0.25,
                    help="每帧丢失观测的概率 (0~1)")
    ap.add_argument("--noise", type=float, default=3.0,
                    help="检测 bbox 的高斯噪声标准差")
    ap.add_argument("--no-window", action="store_true",
                    help="无窗口模式，仅打印统计（适合 SSH / CI）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    stats = run(
        frames=args.frames,
        drop_prob=args.drop,
        noise_sigma=args.noise,
        no_window=args.no_window,
        seed=args.seed,
    )

    print("---- summary ----")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:12s}: {v:.3f}")
        else:
            print(f"  {k:12s}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
