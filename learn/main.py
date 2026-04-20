# -*- coding: utf-8 -*-
"""
main.py — 端到端 Demo： 采集 -> 检测 -> 跟踪 -> 叠加 -> 显示
===========================================
典型用法：

    # 用笔记本摄像头跑 YOLOv8n，全 COCO 类别
    python -m learn.main --source 0

    # 用一段视频文件，只保留 person (class 0)
    python -m learn.main --source /path/to/video.mp4 --classes 0

    # 只保留按类名过滤的类别
    python -m learn.main --source video.mp4 --class-names person car

    # 使用 ONNX 后端 + 卡尔曼跟踪器，用匈牙利匹配
    python -m learn.main --source video.mp4 \\
        --detector onnx --weights yolov8n.onnx \\
        --tracker kalman --match hungarian

    # 把渲染后的视频写到文件
    python -m learn.main --source video.mp4 --save-video out.mp4

按键：
    q / ESC  退出
    h        显示/隐藏统计
    b        显示/隐藏检测框
    t        显示/隐藏历史轨迹
    SPACE    暂停 / 继续
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

import cv2

from .capture import FPSMeter, open_source
from .detector import build_detector
from .overlay import draw_crosshair, draw_stats, draw_tracks
from .tracker import build_tracker


def _parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser()
    # 数据源
    p.add_argument("--source", default="0",
                   help="0/1/... 摄像头下标；或视频文件；或图片目录；或 'screen'")
    # 检测器
    p.add_argument("--detector", choices=["ultralytics", "onnx"], default="ultralytics")
    p.add_argument("--weights", default="yolov8n.pt",
                   help="权重路径；ultralytics 会自动下载，onnx 需你自己导出")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="cpu / cuda / cuda:0 (ultralytics)")
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="只保留这些类别 id")
    p.add_argument("--class-names", nargs="*", default=None,
                   help="只保留这些类名（如 person car）；解析在检测后进行")
    # 跟踪器
    p.add_argument("--tracker", choices=["iou", "kalman"], default="iou")
    p.add_argument("--match", choices=["greedy", "hungarian"], default="greedy")
    p.add_argument("--iou-threshold", type=float, default=0.25)
    p.add_argument("--max-lost", type=int, default=None,
                   help="丢失多少帧就移除（默认 iou=8, kalman=30）")
    p.add_argument("--min-hits", type=int, default=None,
                   help="展示前累计命中数（默认 iou=2, kalman=3）")
    p.add_argument("--class-constrained", action="store_true",
                   help="不同类别的轨迹/检测互不相认")
    # 输出
    p.add_argument("--no-window", action="store_true",
                   help="无窗口模式，仅在终端打印 FPS（适合 SSH 环境）")
    p.add_argument("--save-video", default=None,
                   help="把渲染结果写到这个视频文件")
    p.add_argument("--save-fps", type=float, default=30.0,
                   help="保存视频的帧率（读取源帧率不可得时使用）")
    return p.parse_args(argv)


def _build_tracker_with_defaults(args):
    kwargs = dict(
        iou_threshold=args.iou_threshold,
        match=args.match,
        class_constrained=args.class_constrained,
    )
    if args.tracker == "iou":
        kwargs["max_lost"] = args.max_lost if args.max_lost is not None else 8
        kwargs["min_hits"] = args.min_hits if args.min_hits is not None else 2
    else:
        kwargs["max_lost"] = args.max_lost if args.max_lost is not None else 30
        kwargs["min_hits"] = args.min_hits if args.min_hits is not None else 3
    return build_tracker(args.tracker, **kwargs)


def _build_detector(args):
    if args.detector == "ultralytics":
        return build_detector(
            "ultralytics",
            weights=args.weights,
            conf=args.conf,
            iou=args.iou,
            classes=args.classes,
            device=args.device,
            imgsz=args.imgsz,
        )
    return build_detector(
        "onnx",
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=args.classes,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    detector = _build_detector(args)
    tracker = _build_tracker_with_defaults(args)
    fps = FPSMeter(window=30)

    name_filter = set(args.class_names) if args.class_names else None

    show_stats = True
    show_boxes = True
    show_history = True
    paused = False

    writer: Optional[cv2.VideoWriter] = None
    last_log = time.time()

    with open_source(args.source) as src:
        frame_iter = iter(src)
        frame = None
        while True:
            if not paused:
                try:
                    frame = next(frame_iter)
                except StopIteration:
                    break

            if frame is None:
                break

            t0 = time.perf_counter()
            detections = detector(frame)
            if name_filter is not None:
                detections = [d for d in detections if d.label in name_filter]
            t1 = time.perf_counter()
            tracks = tracker.update(detections)
            t2 = time.perf_counter()

            cur_fps = fps.tick()

            render = frame.copy()
            if show_boxes:
                draw_tracks(render, tracks, show_history=show_history)
            draw_crosshair(render)

            if show_stats:
                draw_stats(render, [
                    f"FPS      : {cur_fps:5.1f}",
                    f"Det time : {(t1 - t0) * 1000:5.1f} ms",
                    f"Trk time : {(t2 - t1) * 1000:5.1f} ms",
                    f"Dets     : {len(detections)}",
                    f"Tracks   : {len(tracks)}",
                    f"Det/Trk  : {args.detector}/{args.tracker}",
                ])

            if args.save_video:
                if writer is None:
                    h, w = render.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save_video, fourcc,
                                             float(args.save_fps), (w, h))
                writer.write(render)

            if args.no_window:
                if time.time() - last_log > 1.0:
                    print(f"FPS={cur_fps:5.1f} dets={len(detections)} "
                          f"tracks={len(tracks)}")
                    last_log = time.time()
                continue

            cv2.imshow("learn-demo", render)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("h"):
                show_stats = not show_stats
            elif key == ord("b"):
                show_boxes = not show_boxes
            elif key == ord("t"):
                show_history = not show_history
            elif key == ord(" "):
                paused = not paused

    if writer is not None:
        writer.release()
    if not args.no_window:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
