"""sim/env.py —— MuJoCoArmEnv：完整实现 core.interfaces.SimEnv 的环境封装。

职责
----
把 sim/scene.py 构建的 MuJoCo 模型包成一个"状态机"：
* reset(seed) 确定性重置（机械臂回 HOME_QPOS，物体按 seed 做可复现微扰）；
* step(n) 推进物理；
* get_state() 返回 core.types.SceneState（RobotState / ContactState / list[SceneObject]）；
* 渲染（懒加载 SceneRenderer）、关节目标、夹爪开关、末端接触力、位姿读写。

关键实现与踩坑点（本机实测）
----------------------------
* dt 默认 core.config.SIM_DT=0.002（500 Hz），直接写 model.opt.timestep。
* 夹爪方向**实测**：g_fingers_actuator 是肌腱驱动的位置执行器，ctrl ∈ [0,255]。
  ctrl=0  -> driver 关节 ≈0.003 rad，两指硅胶垫间距 ≈93 mm = **全开**；
  ctrl=255 -> driver ≈0.782 rad，间距 ≈8.4 mm = **全闭**。
  因此 open_ratio=1（全开）-> ctrl=0；open_ratio=0（全闭）-> ctrl=255，
  即 ctrl = (1 - open_ratio) * 255。get_gripper() 用 driver 关节角反算：
  ratio = 1 - clip(mean(driver)/0.8, 0, 1)（driver 关节范围 0..0.8）。
* contact_force_on_ee()：mj_contactForce 返回的 6 维力在**接触系**；
  实测（方块压桌面：geom1=table_top, geom2=box）它给的是**施加在 geom2 上的力**
  （world = frame^T @ f_local，方向 +z 支撑方块，4 个接触点之和 = 方块重力）。
  所以：ee 是 geom2 -> 取 +，ee 是 geom1 -> 取 -，得到"环境施加在末端上的力"。
  再平移到 g_pinch：torque_tcp = torque + (p_contact - p_tcp) x force。
* 只统计涉及 g_* body 的接触；无接触返回全 0。
* reset(seed) 确定性：seed=0（默认）物体精确落在配置位置，与 MjData(model) 的
  qpos0（= planning 私有临时 data 看到的布局）完全一致；seed!=0 时用
  numpy.random.default_rng(seed) 生成 ±1.5cm 的 xy 微扰，同一 seed 完全可复现。
* reset 瞬间物体底面与桌面上表面**恰好相切**，MuJoCo 此时不生成接触（ncon=0）；
  step 若干步物体沉底约 1e-4 m 后才出现 21 个 table_top<->物体 的静置接触
  （实测 step(20) -> ncon=21）。这是正常静置接触、不是穿透错误；
  判断"机器人是否碰撞"时只统计含机器人 geom 的接触对即可。
* 相机名固定 core.config.CAMERAS；wrist 相机在场景里挂在 g_base body 上。
"""
from __future__ import annotations

from typing import Iterable, Optional

from . import _glinit  # noqa: F401  必须在 import mujoco 之前

import mujoco
import numpy as np

from core.config import CAMERAS, GRIPPER_OPEN, HOME_QPOS, SIM_DT
from core.types import (
    ARM_ACTUATOR_NAMES,
    ARM_JOINT_NAMES,
    ARM_NDOF,
    GRIPPER_ACTUATOR_NAME,
    TCP_SITE,
    ContactState,
    RobotState,
    SceneObject,
    SceneState,
)

from .render import SceneRenderer
from .scene import (
    DEFAULT_OBJECTS,
    OBJECT_IDS,
    build_spec,
    normalize_objects,
    object_joint_name,
    object_rest_z,
)

#: 夹爪执行器 ctrl 上限（= MJCF ctrlrange 上界）
GRIPPER_CTRL_MAX: float = 255.0
#: driver 关节行程上界（2f85.xml: driver joint range 0..0.8）
GRIPPER_DRIVER_MAX: float = 0.8
#: reset 时物体 xy 的确定性随机微扰幅度（米）
OBJECT_RESET_JITTER: float = 0.015
#: 判定"夹住"物体时的夹爪闭合阈值（比例小于它才可能是夹持）
HOLD_GRIPPER_RATIO: float = 0.85


class MuJoCoArmEnv:
    """UR5e + Robotiq 2F85 场景环境，满足 core.interfaces.SimEnv。

    参数
    ----
    objects : 物体描述（None -> DEFAULT_OBJECTS，见 sim.scene）。
    with_barrier : True 时场景内加竖直挡板（planning 绕障测试用）。
    dt : 物理步长，默认 SIM_DT=0.002。
    render_size : (width, height) 渲染分辨率。
    seed : 初始 reset 用的随机种子。

    属性：model, data, dt, arm_qpos_idx(6,), arm_dof_idx(6,), camera_names。
    额外：with_barrier, render_size，以及 close()。
    """

    def __init__(self, objects: Optional[Iterable[dict]] = None,
                 with_barrier: bool = False, dt: float = SIM_DT,
                 render_size: tuple = (512, 512), seed: int = 0):
        self._objects_spec = normalize_objects(objects)
        self.with_barrier = bool(with_barrier)
        self.model = build_spec(self._objects_spec, self.with_barrier).compile()
        self.model.opt.timestep = float(dt)
        self.dt = float(dt)
        self.data = mujoco.MjData(self.model)
        self.render_size = (int(render_size[0]), int(render_size[1]))
        self.camera_names = tuple(CAMERAS)

        self._renderer: Optional[SceneRenderer] = None
        self._max_force = 0.0
        self._holding: Optional[str] = None
        self._gripper_cmd = float(GRIPPER_OPEN)
        self._resolve_ids()
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # 名称 -> 索引
    # ------------------------------------------------------------------ #
    def _resolve_ids(self) -> None:
        m = self.model

        def jid(name: str) -> int:
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if i < 0:
                raise ValueError("模型里缺少 joint %r" % (name,))
            return int(i)

        def aid(name: str) -> int:
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if i < 0:
                raise ValueError("模型里缺少 actuator %r" % (name,))
            return int(i)

        def sid(name: str) -> int:
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
            if i < 0:
                raise ValueError("模型里缺少 site %r" % (name,))
            return int(i)

        def bid(name: str) -> int:
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if i < 0:
                raise ValueError("模型里缺少 body %r" % (name,))
            return int(i)

        self.arm_qpos_idx = np.array([m.jnt_qposadr[jid(n)] for n in ARM_JOINT_NAMES], dtype=int)
        self.arm_dof_idx = np.array([m.jnt_dofadr[jid(n)] for n in ARM_JOINT_NAMES], dtype=int)
        self._arm_act_idx = np.array([aid(n) for n in ARM_ACTUATOR_NAMES], dtype=int)
        self._grip_act_idx = aid(GRIPPER_ACTUATOR_NAME)
        self._tcp_site_id = sid(TCP_SITE)
        self._driver_qpos_idx = np.array(
            [m.jnt_qposadr[jid("g_right_driver_joint")],
             m.jnt_qposadr[jid("g_left_driver_joint")]], dtype=int
        )
        # 夹爪 body 集合：名字以 "g_" 开头（attach prefix）
        self._ee_body_ids = frozenset(
            i for i in range(m.nbody)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("g_")
        )
        # 物体映射（顺序固定 = self._objects_spec 顺序）
        self._obj_ids = [o["id"] for o in self._objects_spec]
        self._obj_body = {}
        self._obj_qposadr = {}
        self._obj_dofadr = {}
        self._obj_meta = {}
        for o in self._objects_spec:
            oid = o["id"]
            j = jid(object_joint_name(oid))
            self._obj_body[oid] = bid(oid)
            self._obj_qposadr[oid] = int(m.jnt_qposadr[j])
            self._obj_dofadr[oid] = int(m.jnt_dofadr[j])
            self._obj_meta[oid] = o
        self._body_to_obj = {b: oid for oid, b in self._obj_body.items()}

    # ------------------------------------------------------------------ #
    # 重置 / 推进
    # ------------------------------------------------------------------ #
    def reset(self, seed: int = 0) -> SceneState:
        """确定性重置：机械臂回 HOME_QPOS，夹爪全开，物体按 seed 做可复现微扰。

        seed=0（默认）：物体精确位于配置位置（与 model.qpos0 / planning 的私有
        MjData 一致，便于规划与执行对齐）；seed!=0：用
        numpy.random.default_rng(seed) 生成 ±OBJECT_RESET_JITTER 的 xy 微扰。
        两种情况都可复现；z 精确落在桌面上表面（TABLE_HEIGHT + 半高），
        姿态为单位四元数、速度清零。

        注意：z 恰好等于 TABLE_HEIGHT + 半高，所以 reset 瞬间物体与桌面只是相切，
        MuJoCo 不生成接触（ncon=0）；step 若干步物体沉底约 1e-4 m 后才出现
        table_top<->物体 的静置接触（实测 21 个，step(20) 时稳定）。属正常现象。
        """
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.arm_qpos_idx] = HOME_QPOS
        d.qvel[self.arm_dof_idx] = 0.0
        d.ctrl[:] = 0.0
        d.ctrl[self._arm_act_idx] = HOME_QPOS
        self._max_force = 0.0
        self._holding = None

        rng = np.random.default_rng(int(seed))
        use_jitter = int(seed) != 0          # seed=0 保持与 model.qpos0 完全一致
        for o in self._objects_spec:
            oid = o["id"]
            adr = self._obj_qposadr[oid]
            if use_jitter:
                jit = rng.uniform(-OBJECT_RESET_JITTER, OBJECT_RESET_JITTER, size=2)
            else:
                jit = (0.0, 0.0)
            d.qpos[adr + 0] = float(o["pos"][0]) + float(jit[0])
            d.qpos[adr + 1] = float(o["pos"][1]) + float(jit[1])
            d.qpos[adr + 2] = object_rest_z(o)
            d.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
            dof = self._obj_dofadr[oid]
            d.qvel[dof:dof + 6] = 0.0

        self.set_gripper(GRIPPER_OPEN)
        mujoco.mj_forward(self.model, d)
        return self.get_state()

    def step(self, n: int = 1) -> SceneState:
        """推进 n 个物理步（mj_step），返回推进后的状态。"""
        for _ in range(int(n)):
            mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def home(self, *, open_gripper: bool = False) -> None:
        """机械臂瞬移回 HOME_QPOS（同时写 ctrl 目标）。

        注意：这里直接改 qpos 是**瞬移**（不是插值运动），便于确定性复位；
        需要轨迹运动请用 control 模块的 MotionRunner。

        P0 修复（带载瞬移丢件）：旧实现无条件 ``set_gripper(GRIPPER_OPEN)``，
        于是"复核前归位"这类瞬移会在机械臂已经跑到 0.6 m 外、手里还夹着物体时把夹爪
        张开，物体直接掉回桌面（实测方块被丢回原位 -> 复核 check_gone 判 subject_still_present）。
        现在默认**不动夹爪**；确实需要"回位顺带松开"的调用方显式传 ``open_gripper=True``。
        ``env.reset()`` 仍会把夹爪全开，不受这里影响。
        """
        d = self.data
        d.qpos[self.arm_qpos_idx] = HOME_QPOS
        d.qvel[self.arm_dof_idx] = 0.0
        d.ctrl[self._arm_act_idx] = HOME_QPOS
        if bool(open_gripper):
            self.set_gripper(GRIPPER_OPEN)
        mujoco.mj_forward(self.model, d)

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    def get_state(self) -> SceneState:
        pos, quat = self.site_pose(TCP_SITE)
        force, torque = self.contact_force_on_ee()
        f_norm = float(np.linalg.norm(force))
        if f_norm > self._max_force:
            self._max_force = f_norm
        self._holding = self._detect_holding()
        robot = RobotState(
            qpos=self.data.qpos[self.arm_qpos_idx].copy(),
            qvel=self.data.qvel[self.arm_dof_idx].copy(),
            tcp_pos=pos,
            tcp_quat=quat,
            gripper=self.get_gripper(),
            holding=self._holding,
        )
        contact = ContactState(force=force, torque=torque,
                               max_force=self._max_force, in_contact=f_norm > 1e-6)
        return SceneState(
            time=float(self.data.time),
            robot=robot,
            objects=self.list_objects(),
            contact=contact,
            joint_names=list(ARM_JOINT_NAMES),
            meta={
                "ncon": int(self.data.ncon),
                "with_barrier": self.with_barrier,
                "object_ids": list(self._obj_ids),
            },
        )

    def list_objects(self) -> list:
        """按固定顺序返回场景物体（core.types.SceneObject 列表）。"""
        out = []
        for oid in self._obj_ids:
            o = self._obj_meta[oid]
            out.append(SceneObject(
                id=oid,
                category=o["category"],
                color=o["color"],
                # 内部用私有读取器：这样 §5.7 的"毒丸测试"把公开 object_pose 换成抛异常后，
                # get_state()/reset() 仍能正常工作（行为与之前完全一致）。
                pose=self._object_pose_raw(oid),
                size=tuple(float(v) for v in o["size"]),
                graspable=bool(o["graspable"]),
                is_container=bool(o["is_container"]),
            ))
        return out

    # ------------------------------------------------------------------ #
    # 控制
    # ------------------------------------------------------------------ #
    def set_arm_joint_targets(self, q: np.ndarray) -> None:
        """设置 6 个机械臂关节的位置目标（写 data.ctrl，按 ctrlrange 夹紧）。"""
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != ARM_NDOF:
            raise ValueError("set_arm_joint_targets 需要 %d 维关节角，收到 %d" % (ARM_NDOF, q.size))
        lo = self.model.actuator_ctrlrange[self._arm_act_idx, 0]
        hi = self.model.actuator_ctrlrange[self._arm_act_idx, 1]
        self.data.ctrl[self._arm_act_idx] = np.clip(q, lo, hi)

    def set_gripper(self, open_ratio: float) -> None:
        """设置夹爪：0=全闭、1=全开（超过范围会被夹紧到 [0,1]）。

        实测映射：ctrl = (1 - open_ratio) * 255；ctrl=0 全开（指间距≈93mm），
        ctrl=255 全闭（≈8.4mm）。见模块 docstring。
        """
        r = float(np.clip(open_ratio, 0.0, 1.0))
        self._gripper_cmd = r
        self.data.ctrl[self._grip_act_idx] = (1.0 - r) * GRIPPER_CTRL_MAX

    def get_gripper(self) -> float:
        """由 driver 关节角反算夹爪开合比例：1=全开、0=全闭。"""
        drv = float(np.mean(self.data.qpos[self._driver_qpos_idx]))
        return float(1.0 - np.clip(drv / GRIPPER_DRIVER_MAX, 0.0, 1.0))

    def gripper_target(self) -> float:
        """最近一次 set_gripper 写入的目标比例（不是实测值）。"""
        return float(self._gripper_cmd)

    # ------------------------------------------------------------------ #
    # 接触 / 位姿
    # ------------------------------------------------------------------ #
    def contact_force_on_ee(self) -> tuple:
        """末端受力（世界系，作用于 TCP）：返回 (force(3), torque(3))。

        遍历 data.contact，只保留至少一个 geom 属于夹爪 body（名字以 g_ 开头）的接触。
        mj_contactForce 给出接触系中的 6 维 wrench，实测它是**施加在 geom2 上**的力
        （方块压桌面时 geom2=方块，结果 +z 支撑方块），故 ee 是 geom2 取正、
        ee 是 geom1 取负。最后把力矩从接触点平移到 g_pinch。无接触返回全 0。
        """
        force = np.zeros(3, dtype=float)
        torque = np.zeros(3, dtype=float)
        d = self.data
        tcp = d.site_xpos[self._tcp_site_id]
        buf = np.zeros(6, dtype=float)
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            ee1 = b1 in self._ee_body_ids
            ee2 = b2 in self._ee_body_ids
            if not ee1 and not ee2:
                continue
            mujoco.mj_contactForce(self.model, d, i, buf)
            rot = c.frame.reshape(3, 3)                # 行 = 接触法向/切向（世界系）
            f_w = rot.T @ buf[:3]
            t_w = rot.T @ buf[3:]
            if ee2:
                sign = 1.0
            elif ee1:
                sign = -1.0
            else:                                      # pragma: no cover
                continue
            force += sign * f_w
            torque += sign * t_w + np.cross(c.pos - tcp, sign * f_w)
        return force, torque

    def _detect_holding(self) -> Optional[str]:
        """尽力而为的夹持判定：夹爪已闭合且与某物体有接触 -> 该物体 id。"""
        if self.get_gripper() > HOLD_GRIPPER_RATIO:
            return None
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            ee1 = b1 in self._ee_body_ids
            ee2 = b2 in self._ee_body_ids
            if not (ee1 or ee2):
                continue
            o1 = self._body_to_obj.get(b1)
            o2 = self._body_to_obj.get(b2)
            if ee1 and o2:
                return o2
            if ee2 and o1:
                return o1
        return None

    def site_pose(self, name: str) -> tuple:
        """site 世界位姿：返回 (pos(3), quat wxyz(4))。"""
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if i < 0:
            raise ValueError("模型里不存在 site %r" % (name,))
        pos = self.data.site_xpos[i].copy()
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(self.data.site_xmat[i].reshape(9)))
        return pos, quat

    def body_pose(self, name: str) -> tuple:
        """body 世界位姿：返回 (pos(3), quat wxyz(4))。"""
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise ValueError("模型里不存在 body %r" % (name,))
        pos = self.data.xpos[i].copy()
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(self.data.xmat[i].reshape(9)))
        return pos, quat

    # ------------------------------------------------------------------ #
    # 物体
    # ------------------------------------------------------------------ #
    def _check_object(self, obj_id: str) -> str:
        obj_id = str(obj_id)
        if obj_id not in self._obj_qposadr:
            raise KeyError("未知物体 id %r，可用：%r" % (obj_id, self._obj_ids))
        return obj_id

    def _object_pose_raw(self, obj_id: str) -> np.ndarray:
        """【私有】读取物体真值位姿 (7,)。env 内部（list_objects/get_state/reset）用这个。

        与公开接口 object_pose() 的区别只有一个：**毒丸测试只替换公开接口**，
        这样 env 自身的状态刷新不会因为"禁止读真值"而被误伤（见 INTERFACES.md §5.7）。
        """
        oid = self._check_object(obj_id)
        adr = self._obj_qposadr[oid]
        return self.data.qpos[adr:adr + 7].copy()

    def _set_object_pose_raw(self, obj_id: str, pose) -> None:
        """【私有】直接写物体真值位姿 (7,)，速度清零后 mj_forward（见 _object_pose_raw 说明）。"""
        oid = self._check_object(obj_id)
        pose = np.asarray(pose, dtype=float).reshape(-1)
        if pose.size != 7:
            raise ValueError("set_object_pose 需要 7 维位姿，收到 %d" % pose.size)
        p = pose.copy()
        n = float(np.linalg.norm(p[3:7]))
        p[3:7] = p[3:7] / n if n > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])
        adr = self._obj_qposadr[oid]
        dof = self._obj_dofadr[oid]
        self.data.qpos[adr:adr + 7] = p
        self.data.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def object_pose(self, obj_id: str) -> np.ndarray:
        """物体位姿 (7,) = [x, y, z, qw, qx, qy, qz]。（真值接口：vlm 运行路径禁用）"""
        return self._object_pose_raw(obj_id)

    def set_object_pose(self, obj_id: str, pose: np.ndarray) -> None:
        """设置物体位姿 (7,)，清零速度后 mj_forward（真值接口：vlm 运行路径禁用）。

        四元数部分会做归一化（模长过小时退化为单位四元数），避免非法姿态。
        """
        self._set_object_pose_raw(obj_id, pose)

    # ------------------------------------------------------------------ #
    # 渲染
    # ------------------------------------------------------------------ #
    def _get_renderer(self) -> SceneRenderer:
        if self._renderer is None:
            self._renderer = SceneRenderer(self.model, width=self.render_size[0],
                                           height=self.render_size[1],
                                           camera_names=self.camera_names)
        return self._renderer

    def _check_camera(self, camera: str) -> str:
        if camera not in self.camera_names:
            raise ValueError("未知相机 %r，可用：%r" % (camera, self.camera_names))
        return camera

    def render(self, camera: str = "overview") -> np.ndarray:
        """渲染彩色图 (H, W, 3) uint8；首次调用会懒加载 Renderer。"""
        return self._get_renderer().rgb(self.data, self._check_camera(camera))

    def render_depth(self, camera: str = "overview") -> np.ndarray:
        """渲染深度图 (H, W) float32，单位米。"""
        return self._get_renderer().depth(self.data, self._check_camera(camera))

    def close(self) -> None:
        """释放渲染器 GL 资源（物理模型不受影响）。"""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "MuJoCoArmEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:                       # 尽力而为
        try:
            self.close()
        except Exception:
            pass
