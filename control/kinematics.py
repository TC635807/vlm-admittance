"""control.kinematics —— UR5e + Robotiq 2F85 的正/逆运动学。

设计要点（务必先读）
--------------------
1. **私有临时 MjData**：所有运动学计算都在 `self._data = mujoco.MjData(env.model)`
   上完成，每次只把 `env.data.qpos` 复制进来、覆盖 6 个机械臂关节后做一次
   `mj_kinematics + mj_comPos`。因此 `fk/jacobian/ik` 绝对不会污染 `env.data`
   （smoke 里有专门断言）。夹爪的 8 个内部自由度从 `env.data` 原样复制，保持不动。
2. **只算轻量前向**：用 `mj_kinematics + mj_comPos` 而不是 `mj_forward`——后者会做
   碰撞检测，在 IK 内循环（每步数千次）里代价过高；site 位姿与雅可比只需前两者。
3. **fk 输出**：TCP site 的世界位姿，四元数为 **wxyz**（MuJoCo 约定）。
   MuJoCo 3.12 的 MjData 没有 `site_xquat`，用 `mju_mat2Quat` 从 `site_xmat` 转。
4. **jacobian 输出**：`(6, nv)` 世界系 site 雅可比 = `[jacp; jacr]`（`mj_jacSite`）。
   前 3 行平动、后 3 行转动；列对应 `qvel`（含夹爪自由度，其对应列约为 0）。
   IK 内只取机械臂列 `arm_dof_idx`。
5. **IK = DLS + 零空间 + 多次随机重启**：
   `dq = J^T (J J^T + lambda^2 I)^{-1} e + (I - J^+ J) * k * (q_center - q)`
   - 误差 `e = [目标位置 - 当前位置; rotvec(target_quat (*) conj(cur_quat))]`，
     四元数误差转旋转向量时走最短分支（|angle| <= pi）。
   - `lambda` 用简易 LM 自适应：误差下降则减半，否则乘 4（上限 1e3 视为卡死）。
   - 零空间项把关节往限位中心拉，冗余/退化方向上更稳。
   - 单步关节增量按 `max_step`（0.25 rad）限幅，防跳变。
   - **热身启动**：调用方给了 `q0`（例如上一步的解）时只做一次求解、不做随机重启，
     控制环每步调用才不会慢；`q0=None` 时才在关节限位内做多次随机重启。
6. **收敛判据**：`||[ep; eo]|| <= tol` —— 同时保证位置误差 <= tol(m) 与姿态误差
   <= tol(rad)。`target_quat=None` 时只约束位置（转动误差置 0，DLS 自动最小化关节移动）。

踩坑点
------
* `ARM_JOINT_LIMITS` 来自 ur5e.xml，是**关节角**限位，直接 clip 即可。
* 随机重启的 RNG 固定 `seed=0`，保证测试可复现。
* 关节角接近限位时 DLS 仍会给出解，因此 `clamp` 后误差可能不收敛 -> 返回 ok=False，
  调用方（MotionRunner）据此返回 `ik_fail`。
"""
from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from core.config import ARM_JOINT_LIMITS, HOME_QPOS
from core.types import ARM_NDOF, TCP_SITE

from .trajectory import normalize_quat, quat_conj, quat_mul, rotvec_from_quat

Array = np.ndarray


class Kinematics:
    """基于 MuJoCo 前向运动学的 TCP 正/逆解。

    参数
    ----
    env : 满足 core.interfaces.SimEnv 的环境对象（提供 `model/data` 及
          `arm_qpos_idx / arm_dof_idx`；两个下标属性缺失时回退为 `arange(6)`）
    tcp_site : str TCP site 名，默认 `core.types.TCP_SITE = "g_pinch"`

    可调属性：`n_restarts`（随机重启次数，默认 6）、`max_step`（单步关节限幅，rad）、
    `nullspace_gain`（零空间回中增益）。
    """

    def __init__(self, env, tcp_site: str = TCP_SITE) -> None:
        self.env = env
        self.model = env.model
        self.tcp_site = str(tcp_site)
        self.site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.tcp_site))
        if self.site_id < 0:
            raise ValueError(f"模型中找不到 TCP site: {self.tcp_site!r}")

        nq_fallback = np.arange(ARM_NDOF, dtype=int)
        self.arm_qpos_idx = np.asarray(
            getattr(env, "arm_qpos_idx", nq_fallback), dtype=int).reshape(-1)
        self.arm_dof_idx = np.asarray(
            getattr(env, "arm_dof_idx", nq_fallback), dtype=int).reshape(-1)
        if self.arm_qpos_idx.size != ARM_NDOF or self.arm_dof_idx.size != ARM_NDOF:
            raise ValueError(f"arm_qpos_idx/arm_dof_idx 必须各含 {ARM_NDOF} 个元素")

        self.q_min = np.array(ARM_JOINT_LIMITS[:, 0], dtype=float)
        self.q_max = np.array(ARM_JOINT_LIMITS[:, 1], dtype=float)
        self.q_center = 0.5 * (self.q_min + self.q_max)

        #: 随机重启配置
        self.n_restarts = 6
        self.max_step = 0.25          # rad/迭代
        self.nullspace_gain = 0.05
        self.lam0 = 1e-3
        self.rng_seed = 0

        # ---- 私有临时数据（关键：绝不改 env.data） ----
        self._data = mujoco.MjData(self.model)
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    # ------------------------------------------------------------------ #
    # 基础
    # ------------------------------------------------------------------ #
    def clamp(self, q: Array) -> Array:
        """把 6 维关节角钳制到 `ARM_JOINT_LIMITS`（返回新数组）。"""
        q = np.asarray(q, dtype=float).reshape(ARM_NDOF)
        return np.clip(q, self.q_min, self.q_max)

    def arm_qpos(self) -> Array:
        """从 env.data 读取当前 6 个机械臂关节角（只读）。"""
        return np.array(self.env.data.qpos, dtype=float)[self.arm_qpos_idx]

    def _sync(self, q: Array) -> None:
        """把 q 写进私有 data（先复制 env.data.qpos 以保留夹爪状态），再做前向运动学。"""
        d = self._data
        d.qpos[:] = self.env.data.qpos          # 只读复制：保留夹爪 8 个自由度
        d.qpos[self.arm_qpos_idx] = self.clamp(q)
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)

    def _eval(self, q: Array) -> tuple[Array, Array, Array]:
        """一次性取回 (pos(3), quat(4), J(6, nv))，供 IK 内循环复用。"""
        self._sync(q)
        d = self._data
        sid = self.site_id
        pos = np.array(d.site_xpos[sid], dtype=float)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(d.site_xmat[sid], dtype=float).reshape(9))
        mujoco.mj_jacSite(self.model, d, self._jacp, self._jacr, sid)
        J = np.vstack([self._jacp, self._jacr]).copy()
        return pos, quat, J

    # ------------------------------------------------------------------ #
    # 正运动学 / 雅可比
    # ------------------------------------------------------------------ #
    def fk(self, q: Optional[Array] = None) -> tuple[Array, Array]:
        """TCP 正解 -> (pos(3), quat wxyz(4))。q=None 用 env.data 当前关节角。"""
        if q is None:
            q = self.arm_qpos()
        pos, quat, _ = self._eval(q)
        return pos, normalize_quat(quat)

    def jacobian(self, q: Optional[Array] = None) -> Array:
        """TCP site 世界系雅可比 `(6, nv)`：前 3 行平动、后 3 行转动。"""
        if q is None:
            q = self.arm_qpos()
        _, _, J = self._eval(q)
        return J

    def jacobian_arm(self, q: Optional[Array] = None) -> Array:
        """只保留 6 个机械臂关节列的 `(6, 6)` 雅可比（IK 内部使用）。"""
        return self.jacobian(q)[:, self.arm_dof_idx]

    # ------------------------------------------------------------------ #
    # 逆运动学
    # ------------------------------------------------------------------ #
    @staticmethod
    def _error(p: Array, quat: Array, target_pos: Array, target_quat: Optional[Array]) -> Array:
        """位置 + 姿态误差拼成 6 维向量（姿态误差为最短旋转向量）。"""
        ep = target_pos - p
        if target_quat is None:
            eo = np.zeros(3)
        else:
            eo = rotvec_from_quat(quat_mul(target_quat, quat_conj(quat)))
        return np.concatenate([ep, eo])

    def _solve(self, seed: Array, target_pos: Array, target_quat: Optional[Array],
               max_iters: int, tol: float) -> tuple[Array, float]:
        """单个种子上的 DLS + 零空间 + LM 阻尼迭代，返回 (最优 q, 最优误差范数)。"""
        q = self.clamp(seed)
        p, quat, J = self._eval(q)
        err = self._error(p, quat, target_pos, target_quat)
        best_q = q.copy()
        best_norm = float(np.linalg.norm(err))
        lam = self.lam0
        I6 = np.eye(6)
        for _ in range(max_iters):
            if best_norm <= tol:
                break
            J6 = J[:, self.arm_dof_idx]
            A = J6 @ J6.T + (lam * lam) * I6
            try:
                dq = J6.T @ np.linalg.solve(A, err)
                # 零空间：把关节往限位中心拉
                Jpinv = J6.T @ np.linalg.inv(A)
                N = I6 - Jpinv @ J6
                dq = dq + N @ (self.nullspace_gain * (self.q_center - q))
            except np.linalg.LinAlgError:
                lam = min(lam * 10.0, 1e3)
                if lam >= 1e3:
                    break
                continue
            n = float(np.linalg.norm(dq))
            if n > self.max_step:
                dq = dq * (self.max_step / n)
            q_try = self.clamp(q + dq)
            p2, quat2, J2 = self._eval(q_try)
            err2 = self._error(p2, quat2, target_pos, target_quat)
            n2 = float(np.linalg.norm(err2))
            if n2 < best_norm - 1e-12:
                q, J, err = q_try, J2, err2
                best_q, best_norm = q_try.copy(), n2
                lam = max(lam * 0.5, 1e-6)
            else:
                lam = min(lam * 4.0, 1e3)
                if lam >= 1e3:
                    break
        return best_q, best_norm

    def ik(self, target_pos: Array, target_quat: Optional[Array] = None,
           q0: Optional[Array] = None, max_iters: int = 300,
           tol: float = 1e-3) -> tuple[Array, bool]:
        """阻尼最小二乘 IK。

        返回 `(q6, ok)`：`ok=True` 表示 `||[位置误差; 姿态误差]|| <= tol`。
        给定 `q0` 时以它为唯一（热身）种子；否则以当前关节角 / HOME / 随机角做多重启。
        姿态用四元数误差转旋转向量（短路分支）；关节限位用 ARM_JOINT_LIMITS 钳制。
        """
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        target_quat = None if target_quat is None else normalize_quat(target_quat)
        tol = float(tol)
        max_iters = int(max_iters)

        if q0 is not None:
            seeds = [self.clamp(np.asarray(q0, dtype=float).reshape(ARM_NDOF))]
        else:
            seeds = [self.arm_qpos(), self.clamp(HOME_QPOS)]
            rng = np.random.default_rng(self.rng_seed)
            for k in range(self.n_restarts):
                if k % 2 == 0:      # 限位内均匀采样：覆盖不同分支
                    seeds.append(self.clamp(rng.uniform(self.q_min, self.q_max)))
                else:               # HOME 附近小扰动：更容易收敛
                    seeds.append(self.clamp(HOME_QPOS + rng.uniform(-0.8, 0.8, ARM_NDOF)))

        best_q, best_norm = None, np.inf
        for seed in seeds:
            q, norm = self._solve(seed, target_pos, target_quat, max_iters, tol)
            if norm < best_norm:
                best_q, best_norm = q, norm
            if best_norm <= tol:
                break
        ok = bool(best_norm <= tol)
        if best_q is None:  # pragma: no cover - seeds 永不为空
            best_q = self.clamp(HOME_QPOS)
        return best_q, ok
