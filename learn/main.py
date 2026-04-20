# -*- coding: utf-8 -*-
"""
main.py — 端到端 Demo： 采集 -> 检测 -> 跟踪 -> 叠加 -> 显示
===========================================
典型用法：
    # 用笔记本摄像头跑 YOLOv8n，全 COCO 类别
    python -m learn.main --source 0

    # 用一段视频文件，只保留 person (class 0)
    python -m learn.main --source /path/to/video.mp4 --classes 0

    # 用一个图片目录离线评测
    python -m learn.main --source /path/to/images_dir

按键：
    q / ESC  退出
    h        显示/隐藏统计
    b        显示/隐藏检测框
    t        显示/隐藏历史轨迹
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2

from .capture import FPSMeter, open_source
from .detector import build_detector
from .overlay import draw_crosshair, draw_stats, draw_tracks
from .tracker import IouTracker


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0",
                   help="0/1/... 摄像头下标；或视频文件；或图片目录；或 'screen'")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="cpu / cuda / cuda:0")
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="只保留这些类别 id（留空则全部）")
    p.add_argument("--no-window", action="store_true",
                   help="无窗口模式，仅在终端打印 FPS（适合 SSH 环境）")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    detector = build_detector(
        "ultralytics",
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        classes=args.classes,
        device=args.device,
        imgsz=args.imgsz,
    )
    tracker = IouTracker(iou_threshold=0.25, max_lost=8, min_hits=2, ema_alpha=0.5)
    fps = FPSMeter(window=30)

    show_stats = True
    show_boxes = True
    show_history = True

    last_log = time.time()

    with open_source(args.source) as src:
        for frame in src:
            t0 = time.perf_counter()
            detections = detector(frame)
            t1 = time.perf_counter()
            tracks = tracker.update(detections)
            t2 = time.perf_counter()

            cur_fps = fps.tick()

            if show_boxes:
                draw_tracks(frame, tracks, show_history=show_history)
            draw_crosshair(frame)

            if show_stats:
                draw_stats(frame, [
                    f"FPS      : {cur_fps:5.1f}",
                    f"Det time : {(t1 - t0) * 1000:5.1f} ms",
                    f"Trk time : {(t2 - t1) * 1000:5.1f} ms",
                    f"Dets     : {len(detections)}",
                    f"Tracks   : {len(tracks)}",
                ])

            if args.no_window:
                if time.time() - last_log > 1.0:
                    print(f"FPS={cur_fps:5.1f} dets={len(detections)} "
                          f"tracks={len(tracks)}")
                    last_log = time.time()
                continue

            cv2.imshow("learn-demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("h"):
                show_stats = not show_stats
            elif key == ord("b"):
                show_boxes = not show_boxes
            elif key == ord("t"):
                show_history = not show_history

    if not args.no_window:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
