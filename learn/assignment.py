# -*- coding: utf-8 -*-
"""
assignment.py — 指派问题（线性分配）求解
===========================================
在多目标跟踪里，"把 N 条历史轨迹"配到"M 个当前检测"上，
本质是一个二分图带权指派问题：

    最小化   sum_{(i,j) in M} cost[i, j]
    约束：每个 i、每个 j 最多被选一次

这里提供两个求解器：

* :func:`linear_assignment`  —— 最优解，底层调用 scipy（若可用），
  否则退回到"合理"的贪心实现。
* :func:`greedy_assignment`  —— 纯贪心，实现简单、速度快，
  便于教学对比。

两者接口一致：返回两条一一对应的下标数组 ``(row_idx, col_idx)``。

教学要点：
    * 贪心每次选全局最小 cost 的一对，O(N*M*log(N*M))。
      在 cost 方差大时结果接近最优；方差小时容易错配。
    * 匈牙利 / Jonker-Volgenant 算法能保证总代价最优，但实现较复杂。
      这里直接复用 scipy，不重复造轮子。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lap  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - only hit when scipy missing
    _HAS_SCIPY = False


def greedy_assignment(
    cost: np.ndarray,
    max_cost: float = np.inf,
) -> Tuple[np.ndarray, np.ndarray]:
    """贪心指派：每次挑全局最小 cost 的 (i, j)，过滤 >= ``max_cost``。"""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    n, m = cost.shape
    flat = [(cost[i, j], i, j) for i in range(n) for j in range(m)]
    flat.sort()

    used_r, used_c = set(), set()
    row_idx, col_idx = [], []
    for v, i, j in flat:
        if v >= max_cost:
            break
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        row_idx.append(i)
        col_idx.append(j)
    return np.array(row_idx, dtype=int), np.array(col_idx, dtype=int)


def linear_assignment(
    cost: np.ndarray,
    max_cost: float = np.inf,
) -> Tuple[np.ndarray, np.ndarray]:
    """最优指派（scipy 可用时是匈牙利 / JV 算法），否则退回贪心。

    会额外应用 ``max_cost`` 过滤：scipy 返回的某些配对若 cost >= max_cost
    会被丢弃。这样可以同时表达"谁配谁"和"谁不该配"两件事。
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    if not _HAS_SCIPY:
        return greedy_assignment(cost, max_cost=max_cost)

    n, m = cost.shape
    # scipy 要求非负有限；把 >=max_cost 的位置抬到一个大值，
    # 让它"不得已"才配上，然后在外层再过滤掉。
    big = float(max_cost if np.isfinite(max_cost) else cost.max() + 1.0) * 10.0 + 1.0
    work = np.where(np.isfinite(cost), cost, big)
    row_idx, col_idx = _scipy_lap(work)
    keep = cost[row_idx, col_idx] < max_cost
    return row_idx[keep], col_idx[keep]


__all__ = ["greedy_assignment", "linear_assignment"]
