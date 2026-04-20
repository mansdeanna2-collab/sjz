# -*- coding: utf-8 -*-
"""
demo_screen_capture.py — 桌面采集性能基准
===========================================
在你的桌面上截 N 帧，测平均帧率、单帧耗时分布。
用于理解 dxcam / mss 等工具的延迟特性，不做任何"找游戏窗口"的事情。

运行：
    python -m learn.demo_screen_capture --frames 300 --backend mss
"""
from __future__ import annotations

import argparse
import statistics
import time


def bench_mss(frames: int):
    import mss
    import numpy as np
    with mss.mss() as sct:
        mon = sct.monitors[1]
        times = []
        for _ in range(frames):
            t0 = time.perf_counter()
            _ = np.array(sct.grab(mon))
            times.append(time.perf_counter() - t0)
    return times


def bench_dxcam(frames: int):
    # Windows-only，未安装就跳过
    import dxcam
    cam = dxcam.create()
    cam.start(target_fps=240)
    times = []
    for _ in range(frames):
        t0 = time.perf_counter()
        _ = cam.get_latest_frame()
        times.append(time.perf_counter() - t0)
    cam.stop()
    return times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--backend", choices=["mss", "dxcam"], default="mss")
    args = ap.parse_args()

    if args.backend == "mss":
        ts = bench_mss(args.frames)
    else:
        ts = bench_dxcam(args.frames)

    avg = statistics.mean(ts) * 1000
    p50 = statistics.median(ts) * 1000
    p95 = sorted(ts)[int(len(ts) * 0.95)] * 1000
    fps = 1.0 / statistics.mean(ts)
    print(f"[{args.backend}] frames={len(ts)}  avg={avg:.2f} ms  "
          f"p50={p50:.2f} ms  p95={p95:.2f} ms  fps={fps:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
