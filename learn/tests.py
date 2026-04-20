# -*- coding: utf-8 -*-
"""
最小单元测试 —— 只测纯逻辑（IoU、跟踪器、P 控制器），
不依赖 YOLO / OpenCV / CUDA，任何电脑都能跑：

    python -m learn.tests
"""
from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from .controller import PController1D, PController2D
from .tracker import IouTracker, iou_xyxy


@dataclass
class FakeDet:
    xyxy: tuple
    conf: float = 0.9
    cls: int = 0
    label: str = "obj"


class TestIoU(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint(self):
        self.assertEqual(iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_half_overlap(self):
        # 两个 10x10，水平错位 5 像素 -> 交 50，并 150 -> 1/3
        self.assertAlmostEqual(iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3, places=4)


class TestTracker(unittest.TestCase):
    def test_birth_and_stability(self):
        trk = IouTracker(min_hits=2, max_lost=3, ema_alpha=0.5)
        det = FakeDet((10, 10, 30, 30))
        # 第 1 帧：hits=1 < min_hits，不展示
        self.assertEqual(len(trk.update([det])), 0)
        # 第 2 帧：hits=2，展示
        out = trk.update([FakeDet((12, 12, 32, 32))])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].tid, 1)

    def test_lost_and_remove(self):
        trk = IouTracker(min_hits=1, max_lost=2)
        trk.update([FakeDet((0, 0, 10, 10))])
        for _ in range(3):
            trk.update([])  # 连续丢失
        # 超过 max_lost 应被清理
        self.assertEqual(len(trk._tracks), 0)

    def test_two_targets_get_different_ids(self):
        trk = IouTracker(min_hits=1)
        out = trk.update([
            FakeDet((0, 0, 10, 10)),
            FakeDet((100, 100, 120, 120)),
        ])
        self.assertEqual(len({t.tid for t in out}), 2)


class TestPController(unittest.TestCase):
    def test_converges(self):
        ctrl = PController1D(kp=0.3, step_cap=50, dead_zone=0.5, smooth=0.0)
        x = 0.0
        for _ in range(200):
            x += ctrl.step(x, 100.0)
        self.assertLess(abs(x - 100.0), 1.0)

    def test_dead_zone(self):
        ctrl = PController1D(kp=0.5, step_cap=10, dead_zone=2.0, smooth=0.0)
        # 误差在死区内，输出应为 0
        self.assertEqual(ctrl.step(0.0, 1.0), 0.0)

    def test_step_cap(self):
        ctrl = PController1D(kp=1.0, step_cap=5.0, dead_zone=0.0, smooth=0.0)
        # 大误差应被限幅
        self.assertAlmostEqual(ctrl.step(0.0, 100.0), 5.0)

    def test_2d(self):
        ctrl = PController2D(kp_x=0.3, kp_y=0.3, step_cap=50,
                             dead_zone=0.5, smooth=0.0)
        cx, cy = 0.0, 0.0
        for _ in range(200):
            dx, dy = ctrl.step((cx, cy), (50.0, -50.0))
            cx += dx
            cy += dy
        self.assertTrue(math.hypot(cx - 50.0, cy + 50.0) < 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
