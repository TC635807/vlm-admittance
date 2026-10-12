"""planning.stub_env —— 规划模块自测用的最小测试替身（非交付功能）。

为什么会有这个文件
==================
planning 的契约依赖 sim.MuJoCoArmEnv 与 control.Kinematics。在它们就绪之前，
planning/smoke.py 需要一个"长得像契约"的最小环境来独立验证算法；这就是本文件。
它**不是** sim/control 的实现，只是测试替身：

* StubEnv：UR5e + 2F85（MjSpec.attach，prefix="g_"）+ 地面 + 桌面 + 4 个
  free-joint 物体 + 可选障碍墙。几何参数按 sim-builder 通报的真实场景抄写
  （桌面上表面 z=0.40；障碍 body "barrier" / geom "barrier_geom" 中心
  (0.25, 0, 0.52) 半尺寸 (0.012, 0.16, 0.12)），所以这里的绕障用例
  与真实场景同构。
* StubKinematics：用 mujoco.mj_jacSite + 阻尼最小二乘写的最小 IK，
  只为让 JointPlanner.plan_pose 有东西可调；真实实现由 ctrl-builder 提供。

物体初始就压在桌面上（微穿透），因此场景里 ncon>0 —— 这正好用来验证
CollisionChecker"只统计含机器人 geom 的接触对"这一判据。
"""
from __future__ import annotations

import numpy as np
import mujoco

from core.config import ARM_JOINT_LIMITS, CAMERAS, GRIPPER_XML, HOME_QPOS, SIM_DT, UR5E_XML
from core.types import ARM_JOINT_NAMES, TCP_SITE

__all__ = ["StubEnv", "StubKinematics", "build_stub_env", "DEFAULT_OBJECTS"]

TABLE_H = 0.40
DEFAULT_OBJECTS = [
    {"id": "box_red", "category": "box", "color": "red", "size": (0.02, 0.02, 0.02), "pos": (0.40, -0.15, TABLE_H + 0.0199)},
    {"id": "box_blue", "category": "box", "color": "blue", "size": (0.02, 0.02, 0.02), "pos": (0.40, 0.15, TABLE_H + 0.0199)},
    {"id": "box_green", "category": "box", "color": "green", "size": (0.02, 0.02, 0.02), "pos": (0.46, 0.0, TABLE_H + 0.0199)},
    {"id": "can_yellow", "category": "can", "color": "yellow", "size": (0.025, 0.03, 0.025), "pos": (0.34, 0.0, TABLE_H + 0.03)},
]


def build_stub_env(objects=None, with_barrier: bool = False, dt: float = SIM_DT, seed: int = 0) -> "StubEnv":
    """搭一个与真实场景同构的最小 MuJoCo 场景，返回 StubEnv。"""
    spec = mujoco.MjSpec.from_file(str(UR5E_XML))
    grip = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    spec.attach(grip, prefix="g_", site=spec.site("attachment_site"))
    try:  # 消除 attach 冲突警告（见 INTERFACES 1 节）
        spec.option.impratio = 10.0
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    except Exception:
        pass
    spec.body("base").pos = [0.0, 0.0, TABLE_H]

    # 地面
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3.0, 3.0, 0.1], pos=[0.0, 0.0, 0.0], rgba=[0.30, 0.30, 0.34, 1.0],
    )
    # 桌面（body "table"，geom "table_top"）+ 四条腿
    table = spec.worldbody.add_body(name="table", pos=[0.1, 0.0, 0.38])
    table.add_geom(
        name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.6, 0.6, 0.02], pos=[0.0, 0.0, 0.0], rgba=[0.55, 0.45, 0.32, 1.0],
    )
    for i, (lx, ly) in enumerate([(0.55, 0.55), (0.55, -0.55), (-0.55, 0.55), (-0.55, -0.55)]):
        table.add_geom(
            name="table_leg_%d" % i, type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.03, 0.03, 0.19], pos=[lx, ly, -0.21], rgba=[0.45, 0.36, 0.26, 1.0],
        )
    # 障碍墙（固定，无 free joint）
    if with_barrier:
        barrier = spec.worldbody.add_body(name="barrier", pos=[0.25, 0.0, TABLE_H + 0.12])
        barrier.add_geom(
            name="barrier_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.012, 0.16, 0.12], pos=[0.0, 0.0, 0.0], rgba=[0.80, 0.25, 0.20, 1.0],
        )
    # 物体（free joint）
    for obj in (objects if objects is not None else DEFAULT_OBJECTS):
        b = spec.worldbody.add_body(name=obj["id"], pos=list(obj["pos"]))
        b.add_freejoint(name=obj["id"] + "_free")
        sx, sy, sz = obj["size"]
        if obj["category"] == "can":
            b.add_geom(
                name=obj["id"] + "_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[sx, sz, 0.0], rgba=[0.9, 0.8, 0.2, 1.0], mass=0.1,
            )
        else:
            b.add_geom(
                name=obj["id"] + "_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[sx, sy, sz], rgba=[0.8, 0.3, 0.3, 1.0], mass=0.05,
            )
    model = spec.compile()
    return StubEnv(model, dt=dt, seed=seed)


class StubEnv:
    """最小 SimEnv 测试替身（只实现 planning 用到的部分）。"""

    def __init__(self, model, dt: float = SIM_DT, seed: int = 0):
        self.model = model
        self.data = mujoco.MjData(model)
        self.dt = float(dt)
        self.arm_qpos_idx = np.array(
            [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINT_NAMES],
            dtype=int,
        )
        self.arm_dof_idx = np.array(
            [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINT_NAMES],
            dtype=int,
        )
        self.camera_names = tuple(CAMERAS)
        self.seed = int(seed)
        self.home()
        self.reset(seed)

    # -- 状态 ------------------------------------------------------------ #
    def home(self) -> None:
        self.data.qpos[self.arm_qpos_idx] = HOME_QPOS
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed: int = 0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos_idx] = HOME_QPOS
        mujoco.mj_forward(self.model, self.data)
        return None

    def step(self, n: int = 1):
        for _ in range(int(n)):
            mujoco.mj_step(self.model, self.data)
        return None

    def set_arm_joint_targets(self, q: np.ndarray) -> None:
        q = np.clip(np.asarray(q, dtype=np.float64).reshape(-1), ARM_JOINT_LIMITS[:, 0], ARM_JOINT_LIMITS[:, 1])
        self.data.ctrl[self.arm_dof_idx] = q
        self.data.qpos[self.arm_qpos_idx] = q
        mujoco.mj_forward(self.model, self.data)

    def set_gripper(self, open_ratio: float) -> None:
        return None

    def get_gripper(self) -> float:
        return 1.0

    def site_pose(self, name: str):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError("no site %s" % name)
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[sid].copy(), self.data.site_xquat[sid].copy()

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# 测试替身运动学（mujoco.mj_jacSite + DLS）
# --------------------------------------------------------------------------- #
def _normalize_quat(q):
    q = np.asarray(q, dtype=np.float64).reshape(-1)[:4]
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def _quat_error_vec(cur, target) -> np.ndarray:
    """wxyz 四元数夹角旋转矢量（target 相对 cur）。"""
    a = _normalize_quat(target)
    b = _normalize_quat(cur)
    bw, bv = b[0], b[1:]
    w = a[0] * bw + float(np.dot(a[1:], bv))
    v = -a[0] * bv + bw * a[1:] - np.cross(a[1:], bv)
    if w < 0:
        w, v = -w, -v
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:
        return np.zeros(3)
    ang = 2.0 * np.arctan2(nv, w)
    return (ang / nv) * v


class StubKinematics:
    """最小 Kinematics 测试替身：fk / jacobian / ik / clamp。"""

    def __init__(self, env, tcp_site: str = TCP_SITE):
        self.env = env
        self.model = env.model
        self.data = mujoco.MjData(self.model)
        self.qpos_idx = np.asarray(env.arm_qpos_idx, dtype=int)
        self.dof_idx = np.asarray(env.arm_dof_idx, dtype=int)
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, tcp_site)
        if sid < 0:
            raise ValueError("模型里没有 site %s" % tcp_site)
        self.site_id = int(sid)
        self.q_min = np.asarray(ARM_JOINT_LIMITS[:, 0], dtype=np.float64).copy()
        self.q_max = np.asarray(ARM_JOINT_LIMITS[:, 1], dtype=np.float64).copy()
        self.n_restarts = 6
        self.ik_lam = 0.05

    def _set(self, q) -> None:
        self.data.qpos[:] = self.env.data.qpos[:]
        self.data.qpos[self.qpos_idx] = np.asarray(q, dtype=np.float64).reshape(-1)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def clamp(self, q) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=np.float64).reshape(-1), self.q_min, self.q_max)

    def _site_quat(self) -> np.ndarray:
        # MuJoCo 的 MjData 只有 site_xmat（3x3 旋转矩阵），没有 site_xquat；
        # 用 mju_mat2Quat 转成 wxyz 四元数。
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id].reshape(9))
        return quat

    def fk(self, q=None):
        qq = self.env.data.qpos[self.qpos_idx] if q is None else q
        self._set(qq)
        return self.data.site_xpos[self.site_id].copy(), self._site_quat()

    def jacobian(self, q=None):
        self._set(self.env.data.qpos[self.qpos_idx] if q is None else q)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        return np.vstack([jacp, jacr])

    def ik(self, target_pos, target_quat=None, q0=None, max_iters: int = 300, tol: float = 1e-3):
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(-1)[:3]
        if q0 is None:
            q0 = self.env.data.qpos[self.qpos_idx]
        q = self.clamp(q0)
        alpha, max_step = 0.6, 0.15
        err = np.zeros(6)
        for _ in range(int(max_iters)):
            self._set(q)
            p = self.data.site_xpos[self.site_id].copy()
            ep = target_pos - p
            eo = np.zeros(3) if target_quat is None else _quat_error_vec(
                self._site_quat(), target_quat
            )
            err = np.concatenate([ep, eo])
            if float(np.linalg.norm(err)) <= tol:
                return q.copy(), True
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            J = np.vstack([jacp, jacr])[:, self.dof_idx]
            dq = J.T @ np.linalg.solve(J @ J.T + (self.ik_lam ** 2) * np.eye(6), err)
            n = float(np.linalg.norm(dq))
            if n > max_step:
                dq = dq * (max_step / n)
            q = self.clamp(q + alpha * dq)
        return q.copy(), bool(float(np.linalg.norm(err)) <= tol)
