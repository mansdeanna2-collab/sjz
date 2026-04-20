# -*- coding: utf-8 -*-
"""
controller.py — 比例（P）控制器
===========================================
复刻原脚本里 AIM_KP_X / AIM_KP_Y / AIM_STEP_CAP_* / AIM_DEAD_ZONE 等概念，
但**不接鼠标**，只做纯数学上的控制器——可用于模拟"把 A 点移到 B 点"的任何场景：
    - 云台/相机追踪仿真
    - 机器人关节控制的入门
    - 游戏 AI / 可视化里让一个光标追一个目标

核心公式：
    error = target - current
    step  = Kp * error
    step  = clip(step, -cap, +cap)       # 步长限幅（防过冲）
    if |error| < dead_zone: step = 0     # 死区，静止时不抖
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PController1D:
    kp: float = 0.2
    step_cap: float = 32.0
    dead_zone: float = 1.0
    smooth: float = 0.25  # 输出一阶低通平滑（EMA）

    _last: float = 0.0

    def reset(self):
        self._last = 0.0

    def step(self, current: float, target: float) -> float:
        err = target - current
        if abs(err) < self.dead_zone:
            raw = 0.0
        else:
            raw = self.kp * err
            raw = max(-self.step_cap, min(self.step_cap, raw))
        # 输出平滑
        out = self.smooth * self._last + (1 - self.smooth) * raw
        self._last = out
        return out


@dataclass
class PController2D:
    """二维版本，x/y 各自独立调参，方便比较过冲/稳态。"""
    kp_x: float = 0.2
    kp_y: float = 0.2
    step_cap: float = 32.0
    dead_zone: float = 1.0
    smooth: float = 0.25

    def __post_init__(self):
        self._x = PController1D(self.kp_x, self.step_cap, self.dead_zone, self.smooth)
        self._y = PController1D(self.kp_y, self.step_cap, self.dead_zone, self.smooth)

    def reset(self):
        self._x.reset()
        self._y.reset()

    def step(self, current_xy, target_xy):
        cx, cy = current_xy
        tx, ty = target_xy
        return self._x.step(cx, tx), self._y.step(cy, ty)
