"""关节空间规划器（planning.planner）：满足 core.interfaces.PlannerProtocol。

职责
====
* plan_joint(q_goal)：以"当前臂关节角"为起点，用 RRT-Connect 规划到 q_goal，
  再做 shortcut + densify，返回可直接交给 control.follow_joint_path 的路径。
* plan_pose(pos, quat=None, n_seeds=20)：多组 IK 种子求逆解，对每个"无碰撞
  且确实到达目标位姿"的解调用 plan_joint，返回第一条成功路径。

实现要点
========
* 起点取自 env.data.qpos[arm_qpos_idx]，即真实当前位形；规划过程只读 env.data，
  读写都在 CollisionChecker 的私有 MjData 上，绝不推进仿真。
* RRT 单次失败很常见（尤其窄通道），这里对 plan_joint 做 n_attempts 次重试，
  每次换一个随机种子（seed, seed+1, ...），显著提高成功率且整体仍可复现。
* plan_pose 的种子策略：先试当前位形（q0=现在），再在关节限位内均匀撒
  n_seeds-1 个随机种子。ctrl-builder 的实现里 q0 给出时会走"不做随机重启的
  单次 DLS"，所以显式给 q0 比自己随机重启更快也更可控。
* quat=None 时 IK 是欠约束的：control.kinematics.ik 的 ok 标志可能因为姿态
  误差置 False 而位置其实已达。因此本模块**不信任 ok**，而用 kin.fk 复核
  "位置误差 < pos_tol（默认 5 mm）"；只有显式给 quat 时才额外检查姿态误差
  （四元数夹角 < rot_tol rad）。
* 路径后处理后会再逐段 is_edge_valid 复核一遍（防御性），任何一段非法就换下
  一个 RRT 种子重来，而不是把坏路径交出去。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from core.config import ARM_JOINT_LIMITS
from core.interfaces import PlannerProtocol

from .collision import CollisionChecker
from .rrt_connect import RRTConnect, densify, joint_path_length, shortcut

__all__ = ["JointPlanner"]


def _edge_steps(a, b, resolution: float = 0.02, max_steps: int = 64) -> int:
    d = float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))
    return int(np.clip(int(np.ceil(d / max(resolution, 1e-9))), 2, max_steps))


def quat_angle_error(qa, qb) -> float:
    """两个 wxyz 四元数之间的最小夹角（rad）。"""
    a = np.asarray(qa, dtype=np.float64).reshape(-1)[:4]
    b = np.asarray(qb, dtype=np.float64).reshape(-1)[:4]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float(np.pi)
    d = abs(float(np.dot(a, b)) / (na * nb))
    return 2.0 * float(np.arccos(np.clip(d, -1.0, 1.0)))


class JointPlanner:
    """RRT-Connect 关节空间规划器（PlannerProtocol 的结构化实现）。

    Parameters
    ----------
    env : MuJoCoArmEnv
    kin : control.kinematics.Kinematics
        需要 fk / ik / clamp（契约见 INTERFACES 2.2）。
    checker : CollisionChecker | None
        不给则用 env 现场构造一个。
    """

    def __init__(self, env, kin, checker: Optional[CollisionChecker] = None):
        self.env = env
        self.kin = kin
        self.checker = checker if checker is not None else CollisionChecker(env)
        self.q_min = np.asarray(ARM_JOINT_LIMITS[:, 0], dtype=np.float64).copy()
        self.q_max = np.asarray(ARM_JOINT_LIMITS[:, 1], dtype=np.float64).copy()
        self.n_arm = int(self.q_min.size)
        self._qpos_idx = np.asarray(env.arm_qpos_idx, dtype=int).reshape(-1)

        # --- RRT 参数 ----------------------------------------------------- #
        self.seed = 0
        self.step_size = 0.15
        self.max_iters = 4000
        self.goal_bias = 0.1
        self.n_attempts = 3          # 换种子重试次数
        self.shortcut_iters = 200
        self.max_step = 0.05         # densify 步长（rad）

        # --- IK / 位姿容差 ------------------------------------------------ #
        self.ik_iters = 300
        self.ik_tol = 1e-3
        self.pos_tol = 5e-3          # 5 mm
        self.rot_tol = 0.05          # rad

        self.last_stats: dict = {}

    # ------------------------------------------------------------------ #
    def current_qpos(self) -> np.ndarray:
        """当前 6 个机械臂关节角（只读 env.data）。"""
        return np.asarray(self.env.data.qpos, dtype=np.float64)[self._qpos_idx].copy()

    def _clamp(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        clamp = getattr(self.kin, "clamp", None)
        if callable(clamp):
            q = np.asarray(clamp(q), dtype=np.float64).reshape(-1)
        if q.size != self.n_arm:
            q = q[: self.n_arm]
        return np.clip(q, self.q_min, self.q_max)

    def _verify_path(self, path) -> bool:
        if path is None or len(path) == 0:
            return False
        for a, b in zip(path[:-1], path[1:]):
            if not self.checker.is_edge_valid(a, b, steps=_edge_steps(a, b)):
                return False
        return True

    # ------------------------------------------------------------------ #
    def plan_joint(self, q_goal: np.ndarray) -> Optional[list]:
        """规划"当前位形 → q_goal"的关节空间路径。

        Returns
        -------
        list[np.ndarray] | None
            首元素严格等于当前关节角、末元素严格等于（限位钳制后的）q_goal；
            失败返回 None。
        """
        q_goal = np.asarray(q_goal, dtype=np.float64).reshape(-1)
        if q_goal.size != self.n_arm:
            return None
        q_goal = self._clamp(q_goal)
        q_start = self.current_qpos()

        for k in range(max(1, int(self.n_attempts))):
            planner = RRTConnect(
                self.checker, self.q_min, self.q_max,
                step_size=self.step_size, max_iters=self.max_iters,
                seed=self.seed + k, goal_bias=self.goal_bias,
            )
            raw = planner.plan(q_start, q_goal)
            if raw is None:
                self.last_stats = {"ok": False, "attempt": k, **planner.last_stats}
                continue
            smooth = shortcut(raw, self.checker, iters=self.shortcut_iters, seed=self.seed + k)
            path = densify(smooth, self.max_step)
            path[0] = q_start.copy()
            path[-1] = q_goal.copy()
            if not self._verify_path(path):
                self.last_stats = {"ok": False, "attempt": k, "reason": "verify_failed"}
                continue
            self.last_stats = {
                "ok": True,
                "attempt": k,
                "waypoints": len(path),
                "raw_waypoints": len(raw),
                "length": joint_path_length(path),
                **planner.last_stats,
            }
            return path

        self.last_stats.setdefault("ok", False)
        self.last_stats["waypoints"] = 0
        return None

    # ------------------------------------------------------------------ #
    def plan_pose(self, pos, quat=None, n_seeds: int = 20) -> Optional[list]:
        """多 IK 种子求逆解并规划到该位姿；返回第一条可行路径。

        Parameters
        ----------
        pos : array-like (3,) TCP 目标位置（世界系）
        quat : array-like (4,) | None  目标姿态 wxyz；None 表示只约束位置
        n_seeds : int  IK 种子数（含首个"当前位形"种子）
        """
        pos = np.asarray(pos, dtype=np.float64).reshape(-1)
        if pos.size != 3:
            return None
        q_start = self.current_qpos()

        rng = np.random.default_rng(self.seed)
        seeds = [q_start]
        for _ in range(max(0, int(n_seeds) - 1)):
            seeds.append(rng.uniform(self.q_min, self.q_max))

        tried_ik = 0
        tried_plan = 0
        # 第一枪交给 control 内部的多次随机重启（q0=None，约 2~60 ms，不容易漏解），
        # 之后逐个用显式种子热身精化（q0 给定约 0.3~2 ms）。见 ctrl-builder 的实测建议。
        attempts = [None] + list(seeds)
        for q0 in attempts:
            try:
                q_sol, ok = self.kin.ik(
                    pos, target_quat=quat, q0=q0, max_iters=self.ik_iters, tol=self.ik_tol
                )
            except Exception:
                continue
            q_sol = np.asarray(q_sol, dtype=np.float64).reshape(-1)
            if q_sol.size != self.n_arm:
                continue
            tried_ik += 1
            q_sol = self._clamp(q_sol)
            if not self.checker.is_state_valid(q_sol):
                continue
            if not self._pose_reached(q_sol, pos, quat):
                continue
            tried_plan += 1
            path = self.plan_joint(q_sol)
            if path is not None:
                self.last_stats["pose_target"] = pos.tolist()
                self.last_stats["ik_tried"] = tried_ik
                self.last_stats["ik_planned"] = tried_plan
                return path

        self.last_stats = {
            "ok": False,
            "reason": "no_reachable_ik",
            "ik_tried": tried_ik,
            "ik_planned": tried_plan,
        }
        return None

    def _pose_reached(self, q: np.ndarray, pos: np.ndarray, quat) -> bool:
        """用 fk 复核解是否真的到达目标（不信任 ik 的 ok，见模块 docstring）。"""
        try:
            p, qq = self.kin.fk(q)
        except Exception:
            return False
        p = np.asarray(p, dtype=np.float64).reshape(-1)[:3]
        if float(np.linalg.norm(p - pos)) > self.pos_tol:
            return False
        if quat is not None:
            if quat_angle_error(qq, quat) > self.rot_tol:
                return False
        return True

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover - 调试友好
        return "JointPlanner(step_size=%.2f, max_iters=%d, is_PlannerProtocol=%s)" % (
            self.step_size, self.max_iters, isinstance(self, PlannerProtocol),
        )
