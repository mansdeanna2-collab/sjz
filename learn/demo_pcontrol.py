# -*- coding: utf-8 -*-
"""
demo_pcontrol.py — P 控制器离线仿真
===========================================
让一个"光标"追一个在屏幕上做圆周运动的"目标"，观察：
    * Kp 过大 -> 过冲、振荡
    * Kp 过小 -> 跟不上、滞后
    * dead_zone 太小 -> 静止时抖
    * step_cap 太大 -> 远距离一步就跳过去

运行：
    python -m learn.demo_pcontrol

按键：q 退出 / 上下箭头调整 Kp / 左右箭头调整 step_cap
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .controller import PController2D

# 方向键 keycode 在不同平台（Win / Linux / macOS）上返回值不同，
# 所以把所有已知可能值都列出来，避免写死某一个平台。
KEY_UP    = (2490368, 0x260000, 82, 65362)
KEY_DOWN  = (2621440, 0x280000, 84, 65364)
KEY_LEFT  = (2424832, 0x250000, 81, 65361)
KEY_RIGHT = (2555904, 0x270000, 83, 65363)


def main() -> int:
    W, H = 800, 600
    cur = np.array([W / 2, H / 2], dtype=np.float32)
    kp = 0.15
    cap = 20.0
    ctrl = PController2D(kp_x=kp, kp_y=kp, step_cap=cap,
                         dead_zone=1.0, smooth=0.25)

    traj_cur: list = []
    traj_tgt: list = []
    t0 = time.perf_counter()

    while True:
        t = time.perf_counter() - t0
        # 目标做一个变速圆周 + 小正弦扰动，模拟真实世界难以预测的运动
        r = 180 + 40 * math.sin(t * 0.7)
        tgt = np.array([
            W / 2 + r * math.cos(t * 1.3),
            H / 2 + r * math.sin(t * 1.1) + 30 * math.sin(t * 3.0),
        ], dtype=np.float32)

        dx, dy = ctrl.step(tuple(cur), tuple(tgt))
        cur = cur + np.array([dx, dy], dtype=np.float32)

        traj_cur.append(cur.copy())
        traj_tgt.append(tgt.copy())
        if len(traj_cur) > 200:
            traj_cur.pop(0)
            traj_tgt.pop(0)

        canvas = np.full((H, W, 3), 30, dtype=np.uint8)

        # 画历史轨迹
        if len(traj_tgt) >= 2:
            pts = np.array(traj_tgt, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (120, 120, 200), 1, cv2.LINE_AA)
        if len(traj_cur) >= 2:
            pts = np.array(traj_cur, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (120, 220, 120), 1, cv2.LINE_AA)

        # 画目标 / 光标
        cv2.circle(canvas, tuple(tgt.astype(int)), 8, (64, 64, 255), -1)
        cv2.circle(canvas, tuple(cur.astype(int)), 6, (64, 255, 64), -1)
        cv2.line(canvas, tuple(cur.astype(int)), tuple(tgt.astype(int)),
                 (180, 180, 180), 1, cv2.LINE_AA)

        err = float(np.linalg.norm(tgt - cur))
        texts = [
            f"Kp       : {kp:.3f}   (up / down)",
            f"step_cap : {cap:.1f}   (right / left)",
            f"error    : {err:6.2f} px",
            "q to quit",
        ]
        for i, s in enumerate(texts):
            cv2.putText(canvas, s, (12, 24 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        cv2.imshow("p-controller demo", canvas)
        key = cv2.waitKeyEx(16)
        if key in (27, ord("q")):
            break
        elif key in KEY_UP:
            kp = min(1.0, kp + 0.01)
        elif key in KEY_DOWN:
            kp = max(0.01, kp - 0.01)
        elif key in KEY_RIGHT:
            cap = min(200.0, cap + 2.0)
        elif key in KEY_LEFT:
            cap = max(1.0, cap - 2.0)
        ctrl = PController2D(kp_x=kp, kp_y=kp, step_cap=cap,
                             dead_zone=1.0, smooth=0.25)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
