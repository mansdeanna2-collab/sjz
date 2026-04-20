# -*- coding: utf-8 -*-
"""
最小单元测试 —— 只测纯逻辑，不依赖 YOLO / GPU / 摄像头。

运行：
    python -m learn.tests        # 直接 stdlib unittest
    python -m pytest learn/      # 或 pytest（可选）
"""
from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import numpy as np

from .assignment import greedy_assignment, linear_assignment
from .controller import (
    PController1D,
    PController2D,
    PIDController1D,
    PIDController2D,
)
from .preprocess import (
    letterbox,
    nms_xyxy,
    nms_xyxy_per_class,
    scale_boxes,
)
from .tracker import (
    IouTracker,
    KalmanTracker,
    iou_matrix,
    iou_xyxy,
)


@dataclass
class FakeDet:
    xyxy: tuple
    conf: float = 0.9
    cls: int = 0
    label: str = "obj"


# --------------------------- 几何 ---------------------------

class TestIoU(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint(self):
        self.assertEqual(iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_half_overlap(self):
        self.assertAlmostEqual(iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)),
                               1 / 3, places=4)

    def test_matrix_shape_and_values(self):
        a = [(0, 0, 10, 10), (20, 20, 30, 30)]
        b = [(0, 0, 10, 10), (25, 25, 35, 35), (100, 100, 110, 110)]
        m = iou_matrix(a, b)
        self.assertEqual(m.shape, (2, 3))
        self.assertAlmostEqual(float(m[0, 0]), 1.0, places=4)
        self.assertEqual(float(m[0, 2]), 0.0)
        self.assertGreater(float(m[1, 1]), 0.0)


# --------------------------- IoU 跟踪器 ---------------------------

class TestIouTracker(unittest.TestCase):
    def test_birth_and_stability(self):
        trk = IouTracker(min_hits=2, max_lost=3, ema_alpha=0.5)
        det = FakeDet((10, 10, 30, 30))
        self.assertEqual(len(trk.update([det])), 0)
        out = trk.update([FakeDet((12, 12, 32, 32))])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].tid, 1)

    def test_lost_and_remove(self):
        trk = IouTracker(min_hits=1, max_lost=2)
        trk.update([FakeDet((0, 0, 10, 10))])
        for _ in range(3):
            trk.update([])
        self.assertEqual(len(trk._tracks), 0)

    def test_two_targets_get_different_ids(self):
        trk = IouTracker(min_hits=1)
        out = trk.update([
            FakeDet((0, 0, 10, 10)),
            FakeDet((100, 100, 120, 120)),
        ])
        self.assertEqual(len({t.tid for t in out}), 2)

    def test_class_constrained_prevents_swap(self):
        """不同类别的检测即便 IoU 高也不应匹配到同一条轨迹。"""
        trk = IouTracker(min_hits=1, class_constrained=True, iou_threshold=0.1)
        trk.update([FakeDet((0, 0, 10, 10), cls=0, label="a")])
        out = trk.update([FakeDet((1, 1, 11, 11), cls=1, label="b")])
        # 应该有两条轨迹而不是一条被改类
        self.assertEqual(len(trk._tracks), 2)

    def test_hungarian_vs_greedy_same_easy_case(self):
        trk_g = IouTracker(min_hits=1, match="greedy")
        trk_h = IouTracker(min_hits=1, match="hungarian")
        dets = [
            FakeDet((0, 0, 10, 10)),
            FakeDet((50, 50, 60, 60)),
            FakeDet((100, 100, 110, 110)),
        ]
        for _ in range(3):
            out_g = trk_g.update(dets)
            out_h = trk_h.update(dets)
        self.assertEqual({t.tid for t in out_g}, {t.tid for t in out_h})


# --------------------------- Kalman 跟踪器 ---------------------------

class TestKalmanTracker(unittest.TestCase):
    def test_constant_velocity_keeps_id(self):
        trk = KalmanTracker(iou_threshold=0.1, min_hits=1, max_lost=10)
        ids = []
        for k in range(12):
            x = 10 + k * 3
            out = trk.update([FakeDet((x, 10, x + 20, 30))])
            self.assertEqual(len(out), 1)
            ids.append(out[0].tid)
        self.assertEqual(len(set(ids)), 1)  # 一直是同一个 ID

    def test_predicts_through_occlusion(self):
        """短暂遮挡（无观测）期间 Kalman 依旧保留轨迹，再现后仍是同一 ID。"""
        trk = KalmanTracker(iou_threshold=0.1, min_hits=1, max_lost=20)
        box = (10, 10, 30, 30)
        for _ in range(5):  # 先稳定地观测几帧（静止目标）
            trk.update([FakeDet(box)])
        first_id = list(trk._tracks.keys())[0]

        for _ in range(5):  # 短暂遮挡：无检测
            trk.update([])

        out = trk.update([FakeDet(box)])  # 目标在原位置再出现
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].tid, first_id)

    def test_removes_after_max_lost(self):
        """超过 max_lost 帧无观测，轨迹被移除。"""
        trk = KalmanTracker(iou_threshold=0.1, min_hits=1, max_lost=3)
        trk.update([FakeDet((0, 0, 10, 10))])
        for _ in range(5):
            trk.update([])
        self.assertEqual(len(trk._tracks), 0)


# --------------------------- 指派问题 ---------------------------

class TestAssignment(unittest.TestCase):
    def test_greedy_trivial(self):
        cost = np.array([[0.1, 0.9], [0.8, 0.2]])
        r, c = greedy_assignment(cost, max_cost=1.0)
        self.assertEqual(set(zip(r.tolist(), c.tolist())), {(0, 0), (1, 1)})

    def test_linear_beats_greedy(self):
        """经典例子：贪心会被迫做次优，而匈牙利取得最优解。"""
        cost = np.array([
            [0.1, 0.2],
            [0.05, 0.3],
        ])
        # 贪心：先挑 (1,0)=0.05 -> 再挑 (0,1)=0.2, 总 0.25
        gr, gc = greedy_assignment(cost, max_cost=1.0)
        greedy_total = float(cost[gr, gc].sum())
        lr, lc = linear_assignment(cost, max_cost=1.0)
        lap_total = float(cost[lr, lc].sum())
        self.assertLessEqual(lap_total, greedy_total)

    def test_max_cost_filter(self):
        cost = np.array([[0.9, 0.95], [0.99, 0.5]])
        r, c = linear_assignment(cost, max_cost=0.6)
        # 只有 (1,1)=0.5 < 0.6，应该只配上这一对
        self.assertEqual(list(zip(r.tolist(), c.tolist())), [(1, 1)])

    def test_empty(self):
        for fn in (greedy_assignment, linear_assignment):
            r, c = fn(np.zeros((0, 0)))
            self.assertEqual(r.size, 0)
            self.assertEqual(c.size, 0)


# --------------------------- 前 / 后处理 ---------------------------

class TestPreprocess(unittest.TestCase):
    def test_letterbox_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out, ratio, pad = letterbox(img, new_shape=640, stride=32)
        self.assertEqual(out.shape[2], 3)
        # H, W 都应该是 stride 的整数倍
        self.assertEqual(out.shape[0] % 32, 0)
        self.assertEqual(out.shape[1] % 32, 0)
        self.assertGreater(ratio, 0)
        self.assertGreaterEqual(pad[0], 0)
        self.assertGreaterEqual(pad[1], 0)

    def test_letterbox_scale_boxes_roundtrip(self):
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        _, ratio, pad = letterbox(img, new_shape=640)
        # 在原图里的一个框先映射到 letterbox，再映射回来
        gt = np.array([[10, 20, 110, 220]], dtype=np.float32)
        lb = gt.copy()
        lb[:, [0, 2]] = lb[:, [0, 2]] * ratio + pad[0]
        lb[:, [1, 3]] = lb[:, [1, 3]] * ratio + pad[1]
        back = scale_boxes(lb, ratio, pad, img.shape)
        self.assertTrue(np.allclose(back, gt, atol=1e-4))

    def test_nms_removes_duplicates(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],    # 与上重合度高
            [100, 100, 120, 120],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        keep = nms_xyxy(boxes, scores, iou_thr=0.5)
        self.assertEqual(set(keep.tolist()), {0, 2})

    def test_nms_per_class_is_independent(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        # 同类：会被 NMS 掉一个
        keep_same = nms_xyxy_per_class(boxes, scores, np.array([0, 0]), 0.5)
        self.assertEqual(keep_same.size, 1)
        # 异类：互不影响
        keep_diff = nms_xyxy_per_class(boxes, scores, np.array([0, 1]), 0.5)
        self.assertEqual(set(keep_diff.tolist()), {0, 1})


# --------------------------- P / PID 控制器 ---------------------------

class TestPController(unittest.TestCase):
    def test_converges(self):
        ctrl = PController1D(kp=0.3, step_cap=50, dead_zone=0.5, smooth=0.0)
        x = 0.0
        for _ in range(200):
            x += ctrl.step(x, 100.0)
        self.assertLess(abs(x - 100.0), 1.0)

    def test_dead_zone(self):
        ctrl = PController1D(kp=0.5, step_cap=10, dead_zone=2.0, smooth=0.0)
        self.assertEqual(ctrl.step(0.0, 1.0), 0.0)

    def test_step_cap(self):
        ctrl = PController1D(kp=1.0, step_cap=5.0, dead_zone=0.0, smooth=0.0)
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


class TestPIDController(unittest.TestCase):
    def test_converges_with_derivative(self):
        ctrl = PIDController1D(kp=0.3, ki=0.0, kd=0.05,
                               step_cap=50, dead_zone=0.1, smooth=0.0)
        x = 0.0
        for _ in range(300):
            x += ctrl.step(x, 100.0, dt=1.0)
        self.assertLess(abs(x - 100.0), 1.0)

    def test_integral_eliminates_bias(self):
        """在一个"常数外力"下，纯 P 会有稳态误差，PI 应能消除。"""
        target = 10.0
        bias = 0.5  # 每步额外干扰

        x = 0.0
        p_only = PIDController1D(kp=0.1, ki=0.0, kd=0.0,
                                 step_cap=5.0, dead_zone=0.0, smooth=0.0)
        for _ in range(500):
            x += p_only.step(x, target, dt=1.0) - bias
        p_err = abs(x - target)

        x = 0.0
        pi = PIDController1D(kp=0.1, ki=0.02, kd=0.0, i_cap=200.0,
                             step_cap=5.0, dead_zone=0.0, smooth=0.0)
        for _ in range(2000):
            x += pi.step(x, target, dt=1.0) - bias
        pi_err = abs(x - target)

        # PI 的稳态误差应显著小于 P
        self.assertLess(pi_err, p_err)
        self.assertLess(pi_err, 1.0)

    def test_anti_windup(self):
        ctrl = PIDController1D(kp=0.1, ki=1.0, kd=0.0,
                               step_cap=1.0, dead_zone=0.0, smooth=0.0,
                               i_cap=5.0)
        # 长时间大误差但输出被限幅，积分不能无限增长
        for _ in range(1000):
            ctrl.step(0.0, 100.0, dt=1.0)
        self.assertLessEqual(abs(ctrl._integral), 5.0 + 1e-6)

    def test_feedforward_reduces_lag(self):
        """给定一个以常速移动的目标，速度前馈应减小稳态误差。"""
        v = 0.5
        x1, x2 = 0.0, 0.0
        target = 0.0
        no_ff = PIDController1D(kp=0.2, step_cap=100, dead_zone=0.0, smooth=0.0)
        with_ff = PIDController1D(kp=0.2, step_cap=100, dead_zone=0.0, smooth=0.0)
        for _ in range(400):
            target += v
            x1 += no_ff.step(x1, target, dt=1.0)
            x2 += with_ff.step(x2, target, dt=1.0, v_ff=v)
        self.assertLess(abs(x2 - target), abs(x1 - target))

    def test_2d_pid(self):
        ctrl = PIDController2D(kp_x=0.3, kp_y=0.3, step_cap=50,
                               dead_zone=0.5, smooth=0.0)
        cx, cy = 0.0, 0.0
        for _ in range(200):
            dx, dy = ctrl.step((cx, cy), (50.0, -50.0), dt=1.0)
            cx += dx
            cy += dy
        self.assertLess(math.hypot(cx - 50.0, cy + 50.0), 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
