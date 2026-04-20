# -*- coding: utf-8 -*-
"""
controller.py — PID 控制器（教学版）
===========================================
本模块提供一维 / 二维的 **P / PI / PID** 控制器，
以及用于离散时间系统的可选：
    - 基于 ``dt`` 的积分与微分（帧率解耦）
    - 速度前馈（feedforward），抵消已知目标运动
    - 输出步长限幅（step cap）+ 死区（dead zone）+ 一阶 EMA 平滑
    - 积分抗饱和（clamping anti-windup）

它完全是纯数学实现，不发送任何系统事件（鼠标 / 键盘 / 网络），
可用于任何"让量 A 收敛到量 B"的仿真场景：相机云台、机器人关节、
UI 动画、数值求解等。

约定：
    current, target, error 都是**实数**或等长向量；
    step(current, target[, dt=1.0, v_ff=0.0]) 返回**本步输出增量**，
    调用方负责把它累加到被控对象上。

离散 PID 公式：
    e_k  = target - current
    I_k  = I_{k-1} + e_k * dt
    D_k  = (e_k - e_{k-1}) / dt
    u_k  = kp * e_k + ki * I_k + kd * D_k + v_ff
    u_k  = clip(u_k, -step_cap, +step_cap)
    out  = smooth * out_{k-1} + (1 - smooth) * u_k      # 输出 EMA
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class PIDController1D:
    """一维 PID 控制器。把 ki / kd 设为 0 即退化为纯 P 控制器。"""

    kp: float = 0.2
    ki: float = 0.0
    kd: float = 0.0
    step_cap: float = 32.0
    dead_zone: float = 1.0
    smooth: float = 0.25  # 输出一阶低通（EMA），0 表示不平滑
    i_cap: Optional[float] = None  # 积分项绝对值上限（抗饱和），None 表示不限
    deriv_on_measurement: bool = False  # True: 微分作用在 current 而不是 error 上，抗目标跳变

    _last_err: float = field(default=0.0, init=False, repr=False)
    _last_cur: float = field(default=0.0, init=False, repr=False)
    _integral: float = field(default=0.0, init=False, repr=False)
    _last_out: float = field(default=0.0, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def reset(self) -> None:
        self._last_err = 0.0
        self._last_cur = 0.0
        self._integral = 0.0
        self._last_out = 0.0
        self._initialized = False

    def step(
        self,
        current: float,
        target: float,
        dt: float = 1.0,
        v_ff: float = 0.0,
    ) -> float:
        """
        返回本步应累加到 ``current`` 上的增量。

        参数：
            current: 当前值
            target:  目标值
            dt:      本步耗时（秒）。设为 1.0 等价于"按帧计步"。
            v_ff:    速度前馈项，已知目标速度时可直接加上，减少滞后。
        """
        if dt <= 0:
            dt = 1e-6

        err = float(target) - float(current)

        if abs(err) < self.dead_zone:
            self._last_err = err
            self._last_cur = float(current)
            self._initialized = True
            # 死区内不改变内部状态的动量，直接让输出衰减
            out = self.smooth * self._last_out
            self._last_out = out
            return out

        self._integral += err * dt
        if self.i_cap is not None:
            if self._integral > self.i_cap:
                self._integral = self.i_cap
            elif self._integral < -self.i_cap:
                self._integral = -self.i_cap

        if not self._initialized:
            deriv = 0.0
        elif self.deriv_on_measurement:
            deriv = -(float(current) - self._last_cur) / dt
        else:
            deriv = (err - self._last_err) / dt

        raw = (
            self.kp * err
            + self.ki * self._integral
            + self.kd * deriv
            + float(v_ff)
        )

        if raw > self.step_cap:
            raw = self.step_cap
        elif raw < -self.step_cap:
            raw = -self.step_cap

        out = self.smooth * self._last_out + (1 - self.smooth) * raw

        self._last_err = err
        self._last_cur = float(current)
        self._last_out = out
        self._initialized = True
        return out


@dataclass
class PController1D:
    """向后兼容的纯 P 控制器（包一层 PIDController1D）。"""

    kp: float = 0.2
    step_cap: float = 32.0
    dead_zone: float = 1.0
    smooth: float = 0.25

    def __post_init__(self) -> None:
        self._pid = PIDController1D(
            kp=self.kp,
            ki=0.0,
            kd=0.0,
            step_cap=self.step_cap,
            dead_zone=self.dead_zone,
            smooth=self.smooth,
        )

    def reset(self) -> None:
        self._pid.reset()

    def step(self, current: float, target: float) -> float:
        return self._pid.step(current, target, dt=1.0)


@dataclass
class PIDController2D:
    """二维 PID：x / y 各自独立的一维控制器。"""

    kp_x: float = 0.2
    kp_y: float = 0.2
    ki: float = 0.0
    kd: float = 0.0
    step_cap: float = 32.0
    dead_zone: float = 1.0
    smooth: float = 0.25
    i_cap: Optional[float] = None
    deriv_on_measurement: bool = False

    def __post_init__(self) -> None:
        self._x = PIDController1D(
            kp=self.kp_x, ki=self.ki, kd=self.kd,
            step_cap=self.step_cap, dead_zone=self.dead_zone,
            smooth=self.smooth, i_cap=self.i_cap,
            deriv_on_measurement=self.deriv_on_measurement,
        )
        self._y = PIDController1D(
            kp=self.kp_y, ki=self.ki, kd=self.kd,
            step_cap=self.step_cap, dead_zone=self.dead_zone,
            smooth=self.smooth, i_cap=self.i_cap,
            deriv_on_measurement=self.deriv_on_measurement,
        )

    def reset(self) -> None:
        self._x.reset()
        self._y.reset()

    def step(
        self,
        current_xy: Tuple[float, float],
        target_xy: Tuple[float, float],
        dt: float = 1.0,
        v_ff_xy: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[float, float]:
        cx, cy = current_xy
        tx, ty = target_xy
        vx, vy = v_ff_xy
        return (
            self._x.step(cx, tx, dt=dt, v_ff=vx),
            self._y.step(cy, ty, dt=dt, v_ff=vy),
        )


@dataclass
class PController2D:
    """向后兼容的纯 P 二维控制器。"""

    kp_x: float = 0.2
    kp_y: float = 0.2
    step_cap: float = 32.0
    dead_zone: float = 1.0
    smooth: float = 0.25

    def __post_init__(self) -> None:
        self._pid = PIDController2D(
            kp_x=self.kp_x, kp_y=self.kp_y, ki=0.0, kd=0.0,
            step_cap=self.step_cap, dead_zone=self.dead_zone,
            smooth=self.smooth,
        )

    def reset(self) -> None:
        self._pid.reset()

    def step(self, current_xy, target_xy):
        return self._pid.step(current_xy, target_xy, dt=1.0)
