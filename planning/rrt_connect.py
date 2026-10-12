"""双向 RRT-Connect 规划器 + 路径后处理（planning.rrt_connect）。

原理
====
RRT-Connect（Kuffner & LaValle 2000）同时维护两棵树：tree_start 以 q_start
为根、tree_goal 以 q_goal 为根。每轮：

1. 采样一个目标点：以 goal_bias 的概率直接取"另一棵树的根"（加速两树相遇），
   否则在关节限位盒内均匀采样；
2. 让当前树朝该目标扩展一步（EXTEND）：找到最近节点，沿连线走 step_size，
   若这段边无碰撞则加入新节点；
3. 若扩展成功，让另一棵树朝"新节点"做贪心连接（CONNECT）：反复扩展，
   直到被障碍卡住（失败）或到达目标（成功）；
4. 两棵树交换角色，重复直到相遇或超过 max_iters。

相遇时沿 parent 数组回溯两条链，拼接成 q_start → q_goal 的路径。

实现要点（踩过的坑）
====================
* 随机源用 np.random.default_rng(seed)，且 **plan() 每次调用都按 seed 重新播种**，
  这样"同 seed + 同查询"结果完全可复现；若只播种一次，多次 plan 会让 rng
  前进，结果漂移。
* 贪心连接必须能判断"到达"（距离 <= connect_eps）并限制节点数，
  否则当两棵树根几乎重合时最近节点不变会死循环。
* 距离用 6 维关节空间欧氏范数（rad）。step_size 默认 0.15 rad。
* **穿墙陷阱**：碰撞检测的离散采样步长必须比 step_size 更细。
  这里用 edge_resolution=0.02 rad 反推每条边的采样段数（2..64 段），
  比 is_edge_valid 的默认 12 段更保守，细障碍物（例如 2.4 cm 厚的墙）
  才不会在两端采样点都恰好合法时被漏判。
* 先做一次"起点直连终点"的快路径：无障碍时直接返回两点，省掉整棵树。
* shortcut 只做随机两点直连并保持端点不变；densify 把每段切成 <= max_step
  的小段，方便 control.follow_joint_path 逐段跟踪。
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

__all__ = ["RRTConnect", "shortcut", "densify", "joint_path_length"]


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def joint_path_length(path) -> float:
    """路径的 6 维关节空间总长度（rad）。"""
    if path is None or len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        total += float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))
    return total


def _edge_steps(a, b, resolution: float = 0.02, max_steps: int = 64) -> int:
    d = float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))
    return int(np.clip(int(np.ceil(d / max(resolution, 1e-9))), 2, max_steps))


def densify(path, max_step: float = 0.05) -> list:
    """把路径每段线性插值切成步长 <= max_step 的小段。

    返回的列表首尾与原路径首尾**完全相等**（原样拷贝），方便上层做端点断言。
    """
    if path is None:
        return None
    pts = [np.asarray(p, dtype=np.float64).reshape(-1).copy() for p in path]
    if len(pts) <= 1:
        return pts
    max_step = float(max_step)
    if not np.isfinite(max_step) or max_step <= 0:
        max_step = 0.05
    out = [pts[0].copy()]
    for a, b in zip(pts[:-1], pts[1:]):
        d = float(np.linalg.norm(b - a))
        k = max(1, int(np.ceil(d / max_step)))
        for i in range(1, k + 1):
            out.append(a + (b - a) * (i / k))
    out[0] = pts[0].copy()
    out[-1] = pts[-1].copy()
    return out


def shortcut(path, checker, iters: int = 200, seed: int = 0) -> list:
    """随机短路：不断尝试把路径上随机两点直接连起来，能连就删掉中间节点。

    Parameters
    ----------
    path : list[np.ndarray]
    checker : planning.collision.CollisionChecker
    iters : int
        随机尝试次数（每次只删一段，200 次足够把 RRT 的锯齿磨平）。
    seed : int
        随机种子，保证可复现。
    """
    if path is None:
        return None
    pts = [np.asarray(p, dtype=np.float64).reshape(-1).copy() for p in path]
    if len(pts) <= 2:
        return pts
    rng = np.random.default_rng(int(seed))
    cur = pts
    for _ in range(max(0, int(iters))):
        n = len(cur)
        if n <= 2:
            break
        i = int(rng.integers(0, n - 1))
        j = int(rng.integers(i + 1, n))
        if j == i + 1:
            continue
        steps = _edge_steps(cur[i], cur[j])
        if checker.is_edge_valid(cur[i], cur[j], steps=steps):
            cur = cur[: i + 1] + cur[j:]
    out = [p.copy() for p in cur]
    out[0] = pts[0].copy()
    out[-1] = pts[-1].copy()
    return out


# --------------------------------------------------------------------------- #
# 内部：RRT 树（预分配数组，避免每次 nearest 都 vstack）
# --------------------------------------------------------------------------- #
class _Tree:
    __slots__ = ("q", "parent", "n", "capacity", "dim")

    def __init__(self, root: np.ndarray, capacity: int = 64):
        root = np.asarray(root, dtype=np.float64).reshape(-1)
        self.dim = int(root.size)
        self.capacity = max(int(capacity), 64)
        self.q = np.empty((self.capacity, self.dim), dtype=np.float64)
        self.parent = np.full(self.capacity, -1, dtype=np.int64)
        self.n = 0
        self.add(root, -1)

    def _grow(self) -> None:
        cap = self.capacity * 2
        q = np.empty((cap, self.dim), dtype=np.float64)
        q[: self.n] = self.q[: self.n]
        p = np.full(cap, -1, dtype=np.int64)
        p[: self.n] = self.parent[: self.n]
        self.q, self.parent, self.capacity = q, p, cap

    def add(self, q: np.ndarray, parent: int) -> int:
        if self.n >= self.capacity:
            self._grow()
        self.q[self.n] = q
        self.parent[self.n] = parent
        self.n += 1
        return self.n - 1

    def nearest(self, target: np.ndarray) -> int:
        arr = self.q[: self.n]
        d = arr - target
        return int(np.einsum("ij,ij->i", d, d).argmin())


# --------------------------------------------------------------------------- #
# RRT-Connect
# --------------------------------------------------------------------------- #
class RRTConnect:
    """双向 RRT-Connect。

    Parameters
    ----------
    checker : CollisionChecker
        提供 is_state_valid / is_edge_valid。
    q_min, q_max : array-like (6,)
        采样边界，通常取 core.config.ARM_JOINT_LIMITS 的两列。
    step_size : float
        扩展步长（rad）。
    max_iters : int
        双向主循环最大轮数。
    seed : int
        随机种子；plan() 每次调用都会用它重新播种，保证可复现。
    goal_bias : float
        以该概率直接把"另一棵树的根"当作扩展目标。
    """

    def __init__(self, checker, q_min, q_max, step_size: float = 0.15,
                 max_iters: int = 4000, seed: int = 0, goal_bias: float = 0.1):
        self.checker = checker
        self.q_min = np.asarray(q_min, dtype=np.float64).reshape(-1).copy()
        self.q_max = np.asarray(q_max, dtype=np.float64).reshape(-1).copy()
        if self.q_min.size != self.q_max.size:
            raise ValueError("q_min / q_max 维度不一致")
        self.dim = int(self.q_min.size)
        self.step_size = float(step_size)
        if not np.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError("step_size 必须为正数")
        self.max_iters = int(max_iters)
        self.seed = int(seed)
        self.goal_bias = float(np.clip(goal_bias, 0.0, 1.0))
        #: 碰撞离散采样步长（rad），比 step_size 细，防止穿墙
        self.edge_resolution = 0.02
        self.connect_eps = 1e-9
        #: plan() 结束后的统计信息（iters / nodes / time），给 smoke 打印用
        self.last_stats: dict = {}

    # ------------------------------------------------------------------ #
    def _valid(self, a: np.ndarray, b: np.ndarray) -> bool:
        return bool(self.checker.is_edge_valid(a, b, steps=_edge_steps(a, b, self.edge_resolution)))

    def _sample(self, rng) -> np.ndarray:
        return rng.uniform(self.q_min, self.q_max)

    def _extend(self, tree: _Tree, target: np.ndarray):
        """朝 target 走一步。返回 (新节点下标, 是否恰好到达)；被挡返回 (-1, False)。"""
        i = tree.nearest(target)
        delta = target - tree.q[i]
        d = float(np.linalg.norm(delta))
        if d <= self.connect_eps:
            return i, True
        newq = tree.q[i] + delta * (min(self.step_size, d) / d)
        if not self._valid(tree.q[i], newq):
            return -1, False
        return tree.add(newq, i), bool(d <= self.step_size)

    def _connect(self, tree: _Tree, target: np.ndarray):
        """贪心连接：反复 _extend 直到到达或被挡。返回 (节点下标, 是否到达)。"""
        while True:
            i = tree.nearest(target)
            delta = target - tree.q[i]
            d = float(np.linalg.norm(delta))
            if d <= self.connect_eps:
                return i, True
            newq = tree.q[i] + delta * (min(self.step_size, d) / d)
            if not self._valid(tree.q[i], newq):
                return -1, False
            i = tree.add(newq, i)
            if d <= self.step_size:  # 这一步正好落到 target 上
                return i, True

    def _path_from(self, tree: _Tree, idx: int) -> list:
        out = []
        while idx != -1:
            out.append(tree.q[idx].copy())
            idx = int(tree.parent[idx])
        out.reverse()
        return out

    # ------------------------------------------------------------------ #
    def plan(self, q_start, q_goal) -> Optional[list]:
        """规划 q_start → q_goal。成功返回 [q_start, ..., q_goal]，失败返回 None。"""
        t0 = time.perf_counter()
        self.last_stats = {"iters": 0, "nodes_start": 0, "nodes_goal": 0, "time": 0.0}
        qs = np.asarray(q_start, dtype=np.float64).reshape(-1)
        qg = np.asarray(q_goal, dtype=np.float64).reshape(-1)
        if qs.size != self.dim or qg.size != self.dim:
            raise ValueError("q_start / q_goal 维度应为 %d" % self.dim)
        if not self.checker.is_state_valid(qs) or not self.checker.is_state_valid(qg):
            self.last_stats["time"] = time.perf_counter() - t0
            self.last_stats["reason"] = "invalid_endpoint"
            return None

        # 快路径：起点直连终点
        if self._valid(qs, qg):
            self.last_stats = {
                "iters": 0, "nodes_start": 1, "nodes_goal": 1,
                "time": time.perf_counter() - t0, "direct": True,
            }
            return [qs.copy(), qg.copy()]

        rng = np.random.default_rng(self.seed)
        tree_s = _Tree(qs, capacity=min(self.max_iters + 2, 4096))
        tree_g = _Tree(qg, capacity=min(self.max_iters + 2, 4096))
        ta, tb = tree_s, tree_g
        max_nodes = self.max_iters * 2 + 64
        found = None

        for it in range(self.max_iters):
            q_rand = tb.q[0] if rng.random() < self.goal_bias else self._sample(rng)
            ia, _reached = self._extend(ta, q_rand)
            if ia >= 0:
                ib, connected = self._connect(tb, ta.q[ia])
                if connected:
                    if ta is tree_s:
                        path = self._path_from(tree_s, ia) + self._path_from(tree_g, ib)[::-1][1:]
                    else:
                        path = self._path_from(tree_s, ib) + self._path_from(tree_g, ia)[::-1][1:]
                    found = path
                    break
            ta, tb = tb, ta
            if ta.n + tb.n > max_nodes:
                break

        stats = {
            "iters": it + 1,
            "nodes_start": tree_s.n,
            "nodes_goal": tree_g.n,
            "time": time.perf_counter() - t0,
            "direct": False,
        }
        if found is None:
            stats["reason"] = "max_iters"
            self.last_stats = stats
            return None

        # 端点用原始数组，避免多次运算带来的浮点漂移
        found[0] = qs.copy()
        found[-1] = qg.copy()
        self.last_stats = stats
        return found
