"""control.motion —— 笛卡尔直线运动 + 导纳外环 + 差分 IK 的执行器。

满足 `core.interfaces.MotionProtocol`。

控制回路（move_to_pose / follow 的每一步）
----------------------------------------
1. 由 CartesianTrajectory 采样参考位姿 `(pos_ref, quat_ref)`；
2. 读 `env.contact_force_on_ee()` 得到世界系力/力矩 `wrench(6)`；
3. `AdmittanceController.update(x_ref, wrench) -> x_cmd(6)`；
4. `x_cmd` 加**任务空间积分补偿**得 `x_goal`（见下），再做差分 IK（上一步解热身）；
5. 关节命令加**速度前馈**（见下），`env.set_arm_joint_targets(q)` -> `env.step()`；
6. 轨迹跑完后再跑一段**收尾伺服**，直到末端到点或超时上限。

为什么需要积分补偿（实测踩坑）
------------------------------
Menagerie 的 UR5e 位置执行器只是 PD（kp=2000 / kv=400），**没有重力补偿**：
本机实测把 ctrl 固定在 HOME_QPOS 时 TCP 静止下沉约 **11 mm**，命令另一组关节角
稳态静差约 **14 mm**。纯 "IK -> ctrl -> step" 无法满足验收要求的末端误差 < 5 mm。
因此加了一个慢速任务空间积分外环：

  e = [p_cmd - p_act; rotvec(q_cmd (*) conj(q_act))]   # 都是世界系
  I <- clamp(I + integral_gain * dt * e)      # 按组限幅 integral_limit
  x_goal = p_cmd + I[:3],  quat_goal = exp(I[3:6]) (*) q_cmd

注意姿态修正是**左乘**（世界系）：误差由 q_cmd (*) conj(q_act) 给出，若右乘
（体坐标系）会在参考姿态转动较大时收敛到错误固定点（实测残留 0.217 rad）。

为什么需要速度前馈（实测踩坑）
------------------------------
同上的位置执行器有很强的速度反馈（bias = -kp*q - kv*qd，本模型 kv/kp = 0.2 s）。
跟踪斜坡时静差 ` (kv/kp)*qd，实测轨迹中段 TCP 滞后可达 28 mm。因此从模型直接读
`kv/kp = -biasprm[2]/gainprm[0]`（对非 position 执行器为 0，自动退化为无前馈），
用参考笛卡尔速度经雅可比伪逆映射成关节速度，做命令前馈：

  qd_ref = J^+ * [v_ref; w_ref]
  q_cmd  = q_ik + (kv/kp) * qd_ref

实测：轨迹中段最大误差 28 mm -> 12 mm，收尾后 < 1 mm。

其它关键设计
------------
* **转动通道用"相对增量"表达**：传给导纳的 `x_ref[3:6] = 0`，命令姿态为
  `quat_ref (*) quat_from_rotvec(x_cmd[3:6])`，避免绝对旋转向量在 ±pi 附近的奇异
  （UR5e 末端朝下时 rotvec 模长接近 pi，直接用绝对量会跳变）。平动仍是绝对位置。
* **收尾伺服**：轨迹结束后继续按最终参考位姿跑导纳+积分，直到平动误差
  <= `settle_tol` 且姿态误差 <= `settle_rot_tol`，或达到 `settle_time` 上限
  （默认 1.5 s）。这样 `move_to_pose` 返回时末端真正停在目标点（< 1 mm），
  而不是停在伺服滞后处。收尾同样受 timeout / force_limit 约束。
* **超时用仿真时间**（`steps * env.dt`）而不是墙钟：物理步进确定、可复现，
  测试不受机器负载影响。
* **force_limit** 默认取柔顺参数的 `force_limit`，|force| 一旦超过立即返回
  `error_code="force_limit"`（不继续压，抓取安全）。底层另有**独立硬上限**
  `force_limit_cap`（默认 200 N）：外部显式传入的 force_limit 超过它就被钳制并
  `warnings.warn`，不依赖调用方自律；`force_limit=None`（用柔顺参数自带值）与
  `force_limit_cap=float('inf')`（显式关闭）是保留的放开口子。
* **target_state()（UI 用）**：每个运动步把"参考目标 / 纯导纳输出 / 真实下发命令"原地写进
  预分配的 np.float32 数组，viewer 以 `100Hz 调用只做浅拷贝、不重算；运动结束后状态保留。
  `cmd_pos` 是字面意义的下发点（含 ~11-14mm 重力静差补偿），`adm_pos` 是纯导纳输出，
  `adm_delta = |cmd_pos - ref_pos|` 是「导纳让位 + 积分静差补偿」两部分之和。
  follow_joint_path / hold 没有笛卡尔参考，ref=cmd=adm=当前 TCP 位姿，skill_hint 固定为
  'joint_path' / 'hold'；中止后 hint 变 'halt'（仅在已有目标时，不改变 valid 语义）。
  该状态只读展示用，不参与任何控制/错误码判定。
  **判外力用 force_norm**（无接触恒为 0）：|adm_pos-ref_pos| 含导纳稳态滞后 (D/K)*v，
  自由空间也有 ~5mm 量级，不能当柔顺让位判据。
* **compliance=None**：默认构造的 runner 用 `core.config.DEFAULT_COMPLIANCE`
  （z_only）；若外部注入了自定义 AdmittanceController，则沿用其当前参数。
* **接触即停（方法 B，stop_on_force）**：每步 step 之后力 >= 阈值就冻结参考点（并重锚定到
  接触点实际位姿）+ 位置保持，然后 ok=True / stop_reason="contact" / metrics.force_at_stop /
  z_at_stop 正常返回，不中止不 halt；硬限与 _halt 仍在，默认 None 行为逐位不变。
  ⚠️ 注意：sim.contact_force_on_ee() 返回的是**作用在 EE 上**的力（下压时 +z），而冻结方程是
  M*xdd+D*xd+K(x-x_ref) = -f_ext，因此导纳在接触时会把参考继续推向接触面（实测未重锚定时
  11N -> 60N 失控）；本模式下改为"冻结参考 + 位置保持"，不依赖导纳让位（见 _freeze_and_return）。
* **接触段参考限速（方法 A，INTERFACES §6.1）**：`move_to_pose` / `follow` 支持
  `v_max>0` —— 参考点按 ‖p_ref - target‖ <= v_max*dt 逐控制步推进（忽略 duration）、
  **关闭速度前馈**（前馈是按 0.15m/s 算的速度超前项，接触段会把参考顶进接触面）、
  收尾段只把误差收到 settle_tol 不重新加速；`v_max=None` 与历史行为**逐位一致**
  （已用 4 条基线轨迹验证）。
  单步压入量 = v_max*dt（0.008 * 2ms = 16 µm）。**真实刚性桌面 60 mm 下降的首触力标定**
  （sim/env.py 桌面 z=0.40、z_only、force_limit=3.0，起始 TCP z=0.45 @ (0.30,0.02)）：
      v_max=None 171.2 N | 0.016 28.36 N | 0.008 13.42 N | 0.004 6.58 N | 0.0035 4.88 N
                       | 0.003 7.23 N | 0.0025 6.74 N | 0.002 3.44 N
  总体随 v_max 下降（>=0.008 近似正比），但 <=0.004 之后受"接触恰好落在哪一控制步"的
  相位影响，有 ±3 N 量级的离散，**不是单调函数**；要稳定 <=5 N 需 v_max<=0.002。
* **错误码**：`"timeout" | "force_limit" | "ik_fail" | "unreachable" | "invalid_path"`。
  目标位姿出发前先用冷启动 IK 预检，失败判 `unreachable`；执行中 IK 失败
  （含一次冷启动重试仍失败）判 `ik_fail`。
* **中止即停住（安全语义）**：任何非成功返回（timeout / force_limit / ik_fail /
  unreachable / invalid_path）在返回 MoveResult 之前，都会先把 6 个臂关节目标设成
  **当前关节角**并空步 `halt_steps`（默认 20）步，主动撤销残留的压入/运动指令。
  只改 ctrl 不 step 是不够的（接触仍按旧穿透演化）；实测不这么做时，一次失败的
  预检返回会让上一帧的压入指令继续生效，0.2 s 内末端力可冲到 7471 N 并把物体抛飞。
  成功路径不受影响。
* **follow_joint_path** 只做关节空间插值（供规划器/RRT 路径执行），**不做**笛卡尔
  导纳；实现为「折线细分 -> 弧长 min-jerk 时间律 -> 关节速度前馈 -> 末端收尾」，
  不再逐路点等待伺服（逐点等待要么慢、要么命令甩开伺服，实测对比见方法 docstring）。
  **所有 timeout 都是仿真时间**（`steps * env.dt`），不是墙钟；max_force 只统计不中止。
"""
from __future__ import annotations

import warnings
from typing import Optional

import mujoco
import numpy as np

from core.config import DEFAULT_COMPLIANCE, SIM_DT
from core.types import (ARM_ACTUATOR_NAMES, ARM_NDOF, ComplianceParams, MoveResult)

from .admittance import AdmittanceController
from .trajectory import (CartesianTrajectory, IDENTITY_QUAT, min_jerk,
                        normalize_quat, quat_conj, quat_from_rotvec, quat_mul,
                        rotvec_from_quat, slerp)
Array = np.ndarray


class _RateLimitedTraj:
    """方法 A（INTERFACES §6.1）：参考点按 ‖p_ref - target‖ <= v_max*dt 逐控制步推进。

    与 CartesianTrajectory 鸭子类型等价（.duration / .sample(t)），可直接喂给 follow 主循环，
    因此限速模式复用同一套导纳 + 积分 + IK + 力检查代码，不新增分支。

    设计
    ----
    * 构造时传入的 duration **被忽略**，总时长由 v_max 决定：
      n = ceil(length / (v_max*dt))，duration = n*dt（第 n 步正好到 target）。
    * 位置沿 p0->p1 恒速推进；姿态按同一归一化进度 s = k/n 做 slerp，因此角速度也恒定。
    * sample(t) 只做一次乘加（k = round(t/dt)），不重算轨迹、无迭代。
    * **关闭速度前馈**（由 follow 控制，见其 docstring）：前馈是按 0.15 m/s 量级算出来的
      速度超前项，接触段用它会把参考"顶"进接触面；限速本身已经把单步压入量压到
      v_max*dt（v_max=0.008、dt=2ms -> 16 µm），再叠加 kv/kp*v 的小超前没有必要且有害。
    """

    def __init__(self, p0, p1, dt: float, v_max: float,
                 rot0: Array | None = None, rot1: Array | None = None) -> None:
        self.p0 = np.asarray(p0, dtype=float).reshape(3).copy()
        self.p1 = np.asarray(p1, dtype=float).reshape(3).copy()
        self.rot0 = None if rot0 is None else normalize_quat(rot0)
        self.rot1 = None if rot1 is None else normalize_quat(rot1)
        self.dt = float(dt)
        self.v_max = float(v_max)
        self.length = float(np.linalg.norm(self.p1 - self.p0))
        step = max(self.v_max * self.dt, 1e-12)
        self.n = max(1, int(np.ceil(self.length / step))) if self.length > 1e-12 else 1
        self.duration = float(self.n) * self.dt
        self._step_s = 1.0 / float(self.n)

    def sample(self, t: float) -> tuple[Array, Array]:
        k = min(max(int(round(float(t) / self.dt)), 0), self.n)
        s = k * self._step_s
        pos = (self.p0 + (self.p1 - self.p0) * s).astype(float)
        if self.rot0 is not None and self.rot1 is not None:
            quat = slerp(self.rot0, self.rot1, s)
        elif self.rot0 is not None:
            quat = self.rot0.copy()
        else:
            quat = IDENTITY_QUAT.copy()
        return pos, quat


class MotionRunner:
    """轨迹跟踪 + 导纳执行器。

    参数
    ----
    env : 满足 core.interfaces.SimEnv 的环境
    kin : control.kinematics.Kinematics（或同接口对象）
    admittance : 可选，注入的 AdmittanceController；None 时按 env.dt 自建
                 DEFAULT_COMPLIANCE 实例
    """

    def __init__(self, env, kin, admittance: Optional[AdmittanceController] = None) -> None:
        self.env = env
        self.kin = kin
        self.dt = float(getattr(env, "dt", SIM_DT))
        if self.dt <= 0.0:
            raise ValueError(f"env.dt 必须 > 0，收到 {self.dt!r}")
        self._adm_auto = admittance is None
        self.adm = AdmittanceController(DEFAULT_COMPLIANCE, self.dt) if admittance is None else admittance
        if abs(float(self.adm.dt) - self.dt) > 1e-9:
            warnings.warn(
                f"注入的导纳 dt={self.adm.dt} 与 env.dt={self.dt} 不一致，"
                f"导纳按自身 dt 离散，可能与物理步不同频。", RuntimeWarning, stacklevel=2)
        self.arm_qpos_idx = np.asarray(
            getattr(kin, "arm_qpos_idx", getattr(env, "arm_qpos_idx", np.arange(ARM_NDOF))),
            dtype=int).reshape(-1)
        self.arm_dof_idx = np.asarray(
            getattr(kin, "arm_dof_idx", getattr(env, "arm_dof_idx", np.arange(ARM_NDOF))),
            dtype=int).reshape(-1)

        # --- 任务空间积分补偿（抵消位置伺服重力静差） ---
        self.integral_gain = 5.0                                   # 1/s
        self.integral_limit = np.array([0.06, 0.06, 0.06, 0.20, 0.20, 0.20], dtype=float)
        self._cart_int = np.zeros(6, dtype=float)

        # --- 速度前馈（抵消位置伺服 kv 的速度滞后） ---
        self.use_velocity_feedforward = hasattr(kin, "jacobian")
        self.vel_lead = self._actuator_vel_lead()

        # --- 导纳力符号（Lead 2026-10-12 裁决 b） ---
        #: 喂给 AdmittanceController 的 wrench 前乘的符号。
        #: **默认 -1.0 = 正确语义**：sim.contact_force_on_ee() 返回"作用在 EE 上的力"，
        #: 而冻结方程是 M*xdd+D*xd+K(x-x_ref) = -f_ext，直接喂原值会让导纳往接触面里压
        #: （实测自由空间无影响，接触时 11N -> 38N -> 60N 失控）；取负后 EE 沿 +f 方向
        #: 让位（离开接触面），接触力有界。
        #: 设成 +1.0 可恢复"修正前"的旧行为（仅用于 A/B 对照，不推荐）。
        self.adm_force_sign = -1.0

        # --- 接触即停（方法 B）：参考冻结后允许导纳让位多久（仿真秒） ---
        self.stop_settle_time = 0.2

        # --- 力上限的独立硬保护 ---
        #: 外部显式传入的 force_limit 的硬上限（N）。不依赖调用方自律：
        #: 超过即钳制并 warnings.warn；force_limit=None 走柔顺参数自带值不受此限；
        #: 设为 float('inf') 可显式关闭该保护（例如实验室里要压到 500N）。
        self.force_limit_cap = 200.0

        # --- 收尾伺服 ---
        self.settle_time = 1.5        # s，收尾上限
        self.settle_tol = 1e-3        # m，平动到位阈值
        self.settle_rot_tol = 0.01    # rad，转动到位阈值

        # --- UI：目标状态（viewer ~100Hz 读取；预分配 float32，读取零重算） ---
        #: 可选：当前阶段名（skills 层可设 'descend'/'lift'/...），没有就是空串
        self.skill_hint: str = ""
        self._target_state = {
            "valid": False,
            "ref_pos": np.zeros(3, dtype=np.float32),
            "ref_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "cmd_pos": np.zeros(3, dtype=np.float32),
            "cmd_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "adm_pos": np.zeros(3, dtype=np.float32),
            "adm_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "force_norm": 0.0,
            "wrench": np.zeros(6, dtype=np.float32),
            "adm_delta": 0.0,
            "skill_hint": "",
        }
        self._site_pose_fn = getattr(env, "site_pose", None)
        self._tcp_site_name = getattr(kin, "tcp_site", None)

        # --- 关节路径执行参数 ---
        self.joint_vel_limit = 1.0    # rad/s，关节路径推进速度（弧长/min-jerk 时间律）
        self.joint_settle_tol = 0.05  # rad，末端收尾到位阈值
        self.joint_settle_time = 0.06 # s，末端收尾时间上限（仿真时间）

        # --- 中止安全：任何非成功返回前先把机械臂停住（绝不保留压入指令） ---
        self.halt_steps = 20          # 中止时用于"停住"的仿真步数（建议 10~25）

        #: 最近一次 follow 的逐参考点跟踪误差（诊断用，不参与契约）
        self.last_track_errors: list[float] = []

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _actuator_vel_lead(self) -> Array:
        """读模型的 position 执行器 kv/kp 比率，作为关节速度前馈系数。

        位置执行器 tau = kp*(ctrl - q) - kv*qd，跟踪斜坡的静差约 (kv/kp)*qd，
        因此 ctrl = q_des + (kv/kp)*qd_des 可消掉速度滞后。若执行器不是 position
        （biasprm[2]==0）或模型不可用，系数为 0（自动退化为无前馈）。
        """
        lead = np.zeros(ARM_NDOF, dtype=float)
        model = getattr(self.env, "model", None)
        if model is None:
            return lead
        try:
            for i, name in enumerate(ARM_ACTUATOR_NAMES):
                aid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
                if aid < 0:
                    continue
                kp = float(model.actuator_gainprm[aid][0])
                kv = -float(model.actuator_biasprm[aid][2])
                if kp > 0.0 and kv > 0.0:
                    lead[i] = kv / kp
        except Exception:  # pragma: no cover - 模型异常时退化为无前馈
            return np.zeros(ARM_NDOF, dtype=float)
        return lead

    def _rate_limited_traj(self, traj, v_max: float) -> "_RateLimitedTraj":
        """把任意 traj 的端点采样出来，构造按 v_max 限速的直线参考（方法 A）。

        端点用 sample(0) / sample(duration) 取，兼容任何鸭子类型的轨迹对象。
        """
        p0, q0 = traj.sample(0.0)
        p1, q1 = traj.sample(float(traj.duration))
        return _RateLimitedTraj(p0, p1, self.dt, float(v_max), rot0=q0, rot1=q1)

    def _resolve_params(self, compliance: Optional[ComplianceParams]) -> ComplianceParams:
        if compliance is not None:
            return compliance
        if self._adm_auto:
            return DEFAULT_COMPLIANCE
        return self.adm.params

    def _resolve_force_limit(self, force_limit: Optional[float],
                             params: ComplianceParams) -> float:
        """解析有效力上限：外部显式值必须经过独立硬上限 force_limit_cap。

        - force_limit=None  -> 用柔顺参数自带 force_limit（显式放开的口子，不钳制）；
        - 显式给值          -> 超过 force_limit_cap 时钳到 cap 并 warnings.warn；
        - force_limit_cap=inf -> 关闭该保护（返回原值）。
        """
        if force_limit is None:
            return float(params.force_limit)
        value = float(force_limit)
        cap = float(self.force_limit_cap)
        if np.isfinite(cap) and value > cap:
            warnings.warn(
                f"force_limit={value:.1f}N 超过底层硬上限 force_limit_cap={cap:.1f}N，"
                f"已钳制到 {cap:.1f}N（如需放开请设 runner.force_limit_cap=float('inf')）",
                RuntimeWarning, stacklevel=2)
            return cap
        return value

    def _arm_q(self) -> Array:
        """读当前 6 个机械臂关节角（只读 env.data）。"""
        return np.array(self.env.data.qpos, dtype=float)[self.arm_qpos_idx]

    def _wrench(self) -> Array:
        """读 TCP 处的力/力矩（世界系），拼成 6 维。"""
        out = self.env.contact_force_on_ee()
        if out is None:
            return np.zeros(6, dtype=float)
        force, torque = out
        force = np.zeros(3) if force is None else np.asarray(force, dtype=float).reshape(-1)[:3]
        torque = np.zeros(3) if torque is None else np.asarray(torque, dtype=float).reshape(-1)[:3]
        return np.concatenate([force, torque])

    def _pose(self) -> tuple[Array, Array]:
        return self.kin.fk()

    def _tcp_pose(self) -> tuple[Array, Array]:
        """便宜的当前 TCP 位姿：优先 env.site_pose（仿真已算好，直接读），否则退回 fk。"""
        if self._site_pose_fn is not None and self._tcp_site_name:
            try:
                return self._site_pose_fn(self._tcp_site_name)
            except Exception:                               # pragma: no cover
                pass
        return self._pose()

    def target_state(self) -> dict:
        """当前下发目标的只读快照（供 MuJoCo viewer 每帧读取，~100Hz）。

        返回 dict 的**浅拷贝**：其中的 np.float32 数组是内部缓冲区的直接引用，
        不复制、不重算，所以调用非常廉价；调用方请只读、不要改写数组。

        键固定（UI/sim 按此并行开发）：
            valid      bool          是否已经有过目标（从未运动过为 False）
            ref_pos    (3,) float32  本步轨迹 reference 目标位置（未经导纳修正）
            ref_quat   (4,) float32  对应姿态 wxyz
            cmd_pos    (3,) float32  **真实下发点（字面）**：实际喂给 kin.ik 的命令位置
            cmd_quat   (4,) float32
            adm_pos    (3,) float32  AdmittanceController 单独输出（未叠加积分静差补偿）
            adm_quat   (4,) float32  对应姿态
            force_norm float        本步喂给导纳的外力模长 ‖f_ext[:3]‖（N）；无接触 == 0.0
            wrench     (6,) float32 本步喂给导纳的 6 维 wrench（世界系 [force; torque]）
            adm_delta  float         |cmd_pos - ref_pos| = 让位 + 积分静差补偿（米）
            skill_hint str           阶段名；无则空串；关节路径为 'joint_path'

        语义对照：cmd_pos 是机器人真被要求去的地方（自由空间里比 ref 偏移约 11-14mm，
        用来抵消位置伺服重力静差）；adm_delta 是"下发点相对参考"的总偏移。

        **判断"有没有外力/柔顺让位"必须用 force_norm，不能用 |adm_pos - ref_pos|**：
        自由空间下 adm_pos 与 ref_pos 之间仍存在导纳的稳态跟踪滞后 (D/K)*v
        （z_only: D=25, K=200, v≈0.04 m/s -> ~5mm；实测中位 5.52mm、最大 18.66mm），
        它是速度滞后再加 z 通道跟随残差，**不是柔顺让位**。无接触时 force_norm 恒为 0.0。
        对 joint_path / hold / halt 这类非导纳动作，wrench/force_norm 同样给出同一时刻
        env.contact_force_on_ee() 的真实读数（无接触即 0）。
        """
        return dict(self._target_state)

    def _update_target_state(self, ref_pos: Array, ref_quat: Array,
                             cmd_pos: Array, cmd_quat: Array,
                             skill_hint: Optional[str] = None,
                             adm_pos: Optional[Array] = None,
                             adm_quat: Optional[Array] = None,
                             wrench: Optional[Array] = None) -> None:
        """原地写入预分配的 float32 数组（不 new 大对象、不重算）；每个运动步调用一次。

        adm_pos/adm_quat 为纯导纳输出（未叠加积分补偿）；缺省时退化为与 cmd 相同
        （关节路径 / hold 这类没有笛卡尔导纳的动作即如此）。
        """
        st = self._target_state
        rp = np.asarray(ref_pos, dtype=np.float32)
        cp = np.asarray(cmd_pos, dtype=np.float32)
        cq = np.asarray(cmd_quat, dtype=np.float32)
        st["ref_pos"][:] = rp
        st["ref_quat"][:] = np.asarray(ref_quat, dtype=np.float32)
        st["cmd_pos"][:] = cp
        st["cmd_quat"][:] = cq
        st["adm_pos"][:] = cp if adm_pos is None else np.asarray(adm_pos, dtype=np.float32)
        st["adm_quat"][:] = cq if adm_quat is None else np.asarray(adm_quat, dtype=np.float32)
        if wrench is None:
            st["wrench"][:] = 0.0
            st["force_norm"] = 0.0
        else:
            ww = np.asarray(wrench, dtype=np.float32).reshape(-1)[:6]
            st["wrench"][:] = 0.0
            st["wrench"][:ww.size] = ww
            st["force_norm"] = float(np.linalg.norm(ww[:3]))
        st["adm_delta"] = float(np.linalg.norm(cp - rp))
        st["skill_hint"] = self.skill_hint if skill_hint is None else str(skill_hint)
        st["valid"] = True

    def _reset_integral(self) -> None:
        """每次笛卡尔运动开始时清零任务空间积分，避免跨动作 windup。"""
        self._cart_int = np.zeros(6, dtype=float)

    def _integral_step(self, x_cmd: Array, quat_cmd: Array) -> tuple[Array, Array]:
        """任务空间积分外环：把导纳命令 x_cmd 变成实际可达的 x_goal。"""
        p_act, q_act = self._pose()
        e_pos = x_cmd[:3] - p_act
        e_rot = rotvec_from_quat(quat_mul(quat_cmd, quat_conj(q_act)))
        self._cart_int = self._cart_int + (self.integral_gain * self.dt) * np.concatenate([e_pos, e_rot])
        for g0 in (0, 3):
            g = np.arange(g0, g0 + 3)
            n = float(np.linalg.norm(self._cart_int[g]))
            lim = float(self.integral_limit[g0])
            if n > lim:
                self._cart_int[g] *= lim / n
        x_goal = np.concatenate([x_cmd[:3] + self._cart_int[:3], x_cmd[3:6]])
        # 转动误差取的是世界系（q_cmd (*) conj(q_act)），因此修正也必须左乘到世界系：
        # quat_goal = exp(I_rot) (*) quat_cmd。若写成右乘（体坐标系），
        # 当参考姿态本身转动较大时会收敛到错误的固定点（实测残留 0.217 rad）。
        quat_goal = quat_mul(quat_from_rotvec(self._cart_int[3:6]), quat_cmd)
        return x_goal, quat_goal

    def _joint_vel_ref(self, traj, t: float, q: Array) -> Array:
        """参考笛卡尔速度经雅可比伪逆映射成关节速度（速度前馈用）。"""
        h = 0.5 * self.dt
        t0 = max(float(t) - h, 0.0)
        t1 = min(float(t) + h, float(traj.duration))
        span = t1 - t0
        if span < 1e-9:
            return np.zeros(ARM_NDOF, dtype=float)
        p0, q0 = traj.sample(t0)
        p1, q1 = traj.sample(t1)
        twist = np.concatenate([
            (np.asarray(p1) - np.asarray(p0)) / span,
            rotvec_from_quat(quat_mul(q1, quat_conj(q0))) / span,
        ])
        J = np.asarray(self.kin.jacobian(q), dtype=float)[:, self.arm_dof_idx]
        A = J @ J.T + 1e-4 * np.eye(6)
        try:
            return J.T @ np.linalg.solve(A, twist)
        except np.linalg.LinAlgError:  # pragma: no cover
            return np.zeros(ARM_NDOF, dtype=float)

    @staticmethod
    def _failure(code: str, message: str, max_force: float, steps: int, dt: float,
                 pos: Array | None = None, quat: Array | None = None) -> MoveResult:
        return MoveResult(
            ok=False, message=message, error_code=code,
            final_pos=None if pos is None else np.asarray(pos, dtype=float).copy(),
            final_quat=None if quat is None else np.asarray(quat, dtype=float).copy(),
            max_force=float(max_force), duration=float(steps) * float(dt), steps=int(steps),
        )

    def _halt(self) -> None:
        """中止即停住：把关节目标拉回**当前关节角**并空步 halt_steps 步。

        安全性来自两点：
        ① 目标 = 当前关节角 => 位置伺服不再继续往接触方向压，残留的压入指令被撤销；
        ② 仍然 step，让新命令真正作用到物理上（只改 ctrl 不 step，接触仍按旧穿透演化）。
        任何异常都吞掉——中止路径必须能返回 MoveResult，不能因清场再抛异常。
        """
        try:
            q_hold = np.asarray(self._arm_q(), dtype=float).reshape(ARM_NDOF)
        except Exception:                                   # pragma: no cover
            return
        try:
            for _ in range(max(1, int(self.halt_steps))):
                self.env.set_arm_joint_targets(q_hold)
                self.env.step()
        except Exception:                                   # pragma: no cover
            pass
        # 展示用：中止后把目标状态更新为"停住位置"；只有之前确实运动过才更新，
        # 保证"从未运动过 valid=False"不被预检失败破坏。
        try:
            if self._target_state["valid"]:
                p_now, q_now = self._tcp_pose()
                self._update_target_state(p_now, q_now, p_now, q_now, "halt",
                                          wrench=self._wrench())
        except Exception:                                   # pragma: no cover
            pass

    def _abort(self, code: str, message: str, max_force: float, steps: int) -> MoveResult:
        """非成功返回的统一出口：先 _halt() 停住，再用停住后的实际位姿构造 MoveResult。

        `steps/duration` 只统计运动步数，halt 步数不计入（安全动作不额外占上层预算）。
        """
        self._halt()
        try:
            pos, quat = self._pose()
        except Exception:                                   # pragma: no cover
            pos, quat = None, None
        return self._failure(code, f"{message} [已停住 halt={int(self.halt_steps)}步]",
                             max_force, steps, self.dt, pos, quat)

    def _apply_reference(self, pos_ref: Array, quat_ref: Array, flim: float, q_seed: Array,
                         steps: int, qd_ref: Optional[Array] = None):
        """执行一个控制步（导纳 -> 积分 -> IK -> 速度前馈 -> set targets -> step）。

        返回 `(q_next, fmax, failure)`；failure 非 None 表示本步失败，调用方直接返回它。
        """
        wrench = self._wrench()                      # 原始读数：|F|/metrics 沿用 sim 的约定
        fmax = float(np.linalg.norm(wrench[:3]))
        # 导纳入口换号（裁决 b）：默认 -1 -> EE 沿 +f 方向让位（见 adm_force_sign 的注释）
        adm_wrench = wrench * float(self.adm_force_sign)
        x_cmd = self.adm.update(np.concatenate([np.asarray(pos_ref, dtype=float), np.zeros(3)]), adm_wrench)
        quat_cmd = quat_mul(quat_ref, quat_from_rotvec(x_cmd[3:6]))
        x_goal, quat_goal = self._integral_step(x_cmd, quat_cmd)
        # cmd=真正下发给 IK 的（含积分补偿），adm=纯导纳输出（未叠加积分）
        self._update_target_state(pos_ref, quat_ref, x_goal[:3], quat_goal,
                                  adm_pos=x_cmd[:3], adm_quat=quat_cmd, wrench=wrench)

        q_ik, ok = self.kin.ik(x_goal[:3], quat_goal, q0=q_seed, max_iters=120, tol=5e-4)
        if not ok:                      # 热身失败 -> 冷启动多重启再试一次
            q_ik, ok = self.kin.ik(x_goal[:3], quat_goal)
        if not ok:
            return q_seed, fmax, self._abort(
                "ik_fail", f"IK 不可达 target={np.round(x_goal[:3], 4).tolist()}",
                fmax, steps)

        q_cmd = np.asarray(q_ik, dtype=float)
        if qd_ref is not None and self.use_velocity_feedforward:
            q_cmd = q_cmd + self.vel_lead * np.asarray(qd_ref, dtype=float)
            if hasattr(self.kin, "clamp"):       # 前馈偏移后仍需守住关节限位
                q_cmd = self.kin.clamp(q_cmd)

        self.env.set_arm_joint_targets(q_cmd)
        self.env.step()
        # 返回 IK 解（而非前馈偏移后的 q_cmd）作为下一步热身种子，保持姿态连续
        return np.asarray(q_ik, dtype=float), fmax, None

    # ------------------------------------------------------------------ #
    # 笛卡尔运动
    # ------------------------------------------------------------------ #
    def move_to_pose(self, pos: Array, quat: Optional[Array] = None,
                     duration: Optional[float] = None,
                     compliance: Optional[ComplianceParams] = None,
                     timeout: float = 8.0,
                     force_limit: Optional[float] = None,
                     v_max: Optional[float] = None,
                     stop_on_force: Optional[float] = None) -> MoveResult:
        """直线运动到目标位姿：min-jerk 位置 + slerp 姿态 + 导纳 + 积分/前馈。

        quat=None 时保持当前姿态；duration=None 时按直线距离/0.15 m/s 估算并钳制到
        [0.6, 6.0] s；force_limit=None 时用柔顺参数的 force_limit。
        轨迹结束后自动收尾到目标（误差 < settle_tol）。

        timeout 是**仿真时间**语义（steps * env.dt，不是墙钟），duration 是轨迹时长。

        stop_on_force>0 时启用方法 B「接触即停」（见 follow 的 docstring），默认 None 不变。
        v_max=None（默认）行为与历史完全一致；v_max>0 时走 INTERFACES §6.1 方法 A
        参考限速（忽略 duration、关闭速度前馈、收尾不重新加速），单步压入量 = v_max*dt。
        用法见 §6.2：细高物体（水瓶/圆柱）下降 v_max=0.008、timeout>=12、force_limit=3.0，
        若返回 error_code="force_limit" 视为"到位接触"，取 final_pos 作 grasp_z_effective。
        """
        q_now = self._arm_q()
        p0, q0 = self.kin.fk(q_now)
        p1 = np.asarray(pos, dtype=float).reshape(3)
        q1 = q0 if quat is None else np.asarray(quat, dtype=float).reshape(4)
        dist = float(np.linalg.norm(p1 - p0))
        if duration is None:
            duration = float(np.clip(max(0.6, dist / 0.15), 0.6, 6.0))
        _, ok = self.kin.ik(p1, q1, q0=None)     # 出发前预检可达性
        if not ok:
            return self._abort("unreachable", f"目标位姿不可达 pos={np.round(p1, 4).tolist()}", 0.0, 0)
        traj = CartesianTrajectory(p0, p1, duration, self.dt, rot0=q0, rot1=q1)
        return self.follow(traj, compliance=compliance, timeout=timeout,
                           force_limit=force_limit, v_max=v_max,
                           stop_on_force=stop_on_force)

    def follow(self, traj, compliance: Optional[ComplianceParams] = None,
               timeout: float = 8.0, force_limit: Optional[float] = None,
               v_max: Optional[float] = None,
               stop_on_force: Optional[float] = None) -> MoveResult:
        """执行一条轨迹（含导纳修正 + 收尾伺服）。

        timeout 是**仿真时间**上限（steps * env.dt，不是墙钟）；轨迹段与收尾段
        共用该预算，超出返回 error_code="timeout"。

        v_max=None（默认）：**与历史行为完全一致** —— min-jerk 时间参数化 + 速度前馈
            + 收尾 settle，老调用方零改动。
        v_max>0（INTERFACES §6.1 方法 A）：参考点按 ‖p_ref - target‖ <= v_max*dt 逐控制步
            推进（**忽略 duration**），把单步压入量压到 v_max*dt（v_max=0.008、dt=2ms
            -> 16 µm）。该模式下**关闭速度前馈**，原因：前馈是按 v_max 量级算出的速度
            超前项（q_cmd = q_ik + (kv/kp)*qd），接触段会把参考顶进接触面；限速已经把
            步进压到微米级，这个超前带来的跟踪收益 <1 mm 却增加冲击风险，故关闭。
            收尾段只把误差收到 settle_tol（不重新加速、不改变参考）。
            调用方必须按 dist/v_max + 1s 显式放大 timeout，否则会 timeout。
        stop_on_force>0（INTERFACES 方法 B「接触即停」）：每步 step 之后若
            ‖f_ext[:3]‖ >= stop_on_force，就**冻结参考点**（并把参考重锚定到接触点的
            实际位姿）并位置保持 stop_settle_time，然后正常返回：ok=True、
            stop_reason="contact"、final_pos=停住时的 TCP、metrics.force_at_stop /
            metrics.z_at_stop。**不中止、不 _halt**；硬限 force_limit 与 _halt 仍保留
            （真异常才中止），错误码集合不变。默认 None = 现有行为逐位不变。
            速度前馈同样关闭（与 v_max 同策略）。
            **判定顺序（Lead 2026-10-12 live 诊断）**：接触即停**先于**硬判据 —— 单步撞击可能
            同时满足 stop_on_force 与 force_limit，若硬限先判就会直接中止、没机会冻结
            （live g2: 18.66N 同时 >2.5N 和 >15N，结果被 force_limit 掐掉）。冻结后的安全网
            只在力**继续上涨**超过 max(硬限, 2.0*触发力) 时才中止；stop_on_force=None 时
            一切照旧（硬限先判、逐位不变）。
        实测（本机刚性桌面 z=0.40、z_only、60mm 下降、force_limit=50）：
            v_max=0.006 -> force_at_stop=11.19N（z=0.4047）
            v_max=0.002 -> force_at_stop= 3.44N（z=0.4047）  ← 落在 2.5`5N 区间
            v_max=0.006 + soft=15 -> 29.28N
          即：**刚性面上首触力 ≈ k_contact * v_max*dt**，要停在 2.5`5N 需 v_max<=0.002；
          柔顺接触（方块叠方块等）力是逐步爬升的，2.5N 阈值能在早期抓住。
        力符号（2026-10-12 裁决 b）：sim 的 f_ext 是"作用在 EE 上的力"，喂给导纳前乘以
        runner.adm_force_sign（默认 -1.0 = 让位方向正确的语义；+1.0 恢复旧行为仅供对照）。
        自由空间 f_ext=0，所以该符号对无接触运动**没有任何影响**（control.smoke 有逐位断言）。
        错误码集合不变：timeout / force_limit / ik_fail / unreachable / invalid_path。
        """
        if v_max is not None:
            if not (float(v_max) > 0.0):
                raise ValueError("v_max 必须 > 0 或 None，收到 %r" % (v_max,))
            traj = self._rate_limited_traj(traj, float(v_max))
            use_vel_ff = False
        else:
            use_vel_ff = True
        if stop_on_force is not None:
            if not (float(stop_on_force) > 0.0):
                raise ValueError("stop_on_force 必须 > 0 或 None，收到 %r" % (stop_on_force,))
            stop_on_force = float(stop_on_force)
            use_vel_ff = False       # 同 v_max 策略：任何"速度超前"项都可能在接触瞬间多压一点
        params = self._resolve_params(compliance)
        self.adm.set_params(params)
        flim = self._resolve_force_limit(force_limit, params)

        n_steps = max(1, int(round(float(traj.duration) / self.dt)))
        pos0, _ = traj.sample(0.0)
        self.adm.reset(np.concatenate([pos0, np.zeros(3)]))   # 转动通道用增量表达
        self._reset_integral()
        self.last_track_errors = []

        q_seed = self._arm_q()
        max_force = 0.0
        steps = 0

        # ---- 1) 轨迹段 ----
        for k in range(1, n_steps + 1):
            if steps * self.dt > timeout:
                return self._abort("timeout", f"超时 {steps * self.dt:.3f}s > {timeout}s", max_force, steps)
            t = min(k * self.dt, float(traj.duration))
            pos_ref, quat_ref = traj.sample(t)
            qd_ref = self._joint_vel_ref(traj, t, q_seed) if use_vel_ff else None
            q_seed, fmax, fail = self._apply_reference(pos_ref, quat_ref, flim, q_seed, steps, qd_ref)
            max_force = max(max_force, fmax)
            steps += 1
            pos_act, _ = self._pose()
            self.last_track_errors.append(float(np.linalg.norm(pos_act - pos_ref)))
            if fail is not None:
                return fail
            # 顺序很重要（Lead 2026-10-12 live 诊断）：接触即停**先判**。
            # 单步撞击可能同时 >= stop_on_force 和 > flim；若硬限先判就会直接中止、
            # 根本没机会冻结（live g2: 18.66N 同时满足 2.5N 与 15N，结果被 force_limit 掐掉）。
            if stop_on_force is not None:
                f_after = float(np.linalg.norm(self._wrench()[:3]))     # 本步 step 之后的力
                if f_after >= stop_on_force:
                    return self._freeze_and_return(pos_ref, quat_ref, flim, q_seed,
                                                   steps, max_force, f_after)
            if fmax > flim:
                return self._abort("force_limit", f"接触力 {fmax:.2f} N 超限 {flim:.2f} N", max_force, steps)

        # ---- 2) 收尾伺服：把末端真正停到最终参考位姿 ----
        pos_end, quat_end = traj.sample(float(traj.duration))
        for _ in range(max(1, int(round(self.settle_time / self.dt)))):
            pos_act, quat_act = self._pose()
            e_pos = float(np.linalg.norm(pos_act - np.asarray(pos_end)))
            e_rot = float(np.linalg.norm(rotvec_from_quat(quat_mul(np.asarray(quat_end), quat_conj(quat_act)))))
            if e_pos <= self.settle_tol and e_rot <= self.settle_rot_tol:
                break
            if steps * self.dt > timeout:
                return self._abort("timeout", f"收尾超时 {steps * self.dt:.3f}s > {timeout}s", max_force, steps)
            q_seed, fmax, fail = self._apply_reference(pos_end, quat_end, flim, q_seed, steps, None)
            max_force = max(max_force, fmax)
            steps += 1
            if fail is not None:
                return fail
            if stop_on_force is not None:          # 同上：接触即停先判
                f_after = float(np.linalg.norm(self._wrench()[:3]))
                if f_after >= stop_on_force:
                    pos_now, quat_now = self._pose()
                    return self._stop_result("contact", f_after, float(pos_now[2]),
                                             max_force, steps, pos_now, quat_now)
            if fmax > flim:
                return self._abort("force_limit", f"接触力 {fmax:.2f} N 超限 {flim:.2f} N", max_force, steps)
        pos, quat = self._pose()
        err = float(np.linalg.norm(pos - np.asarray(pos_end)))
        return MoveResult(
            ok=True,
            message=(f"轨迹完成 err={err * 1000:.2f}mm "
                     f"track_max={max(self.last_track_errors, default=0.0) * 1000:.2f}mm "
                     f"max_force={max_force:.2f}N"),
            error_code=None, final_pos=pos, final_quat=quat,
            max_force=max_force, duration=steps * self.dt, steps=steps)

    # ------------------------------------------------------------------ #
    # 关节空间
    # ------------------------------------------------------------------ #
    def _freeze_and_return(self, pos_ref: Array, quat_ref: Array, flim: float, q_seed: Array,
                           steps: int, max_force: float, force_at_stop: float) -> MoveResult:
        """接触即停：**冻结参考点**，只用导纳在 stop_settle_time 内让位，然后正常返回。

        - 不中止、不 _halt：ok=True + stop_reason="contact"，final_pos=停住时的 TCP。
        - 安全网：只有在力**继续上涨**超过 max(force_limit, 2.0*force_at_stop) 时才 _abort。
          触发值本身可能就高于硬限（单步撞击，如 live 的 18.66N > 15N），那不是"继续上涨"，
          不能因此中止，否则"接触即停"永远没机会返回。
        - 冻结 = 继续用同一个 (pos_ref, quat_ref) 调 _apply_reference（qd_ref=None），
          参考不再前进，但导纳/积分仍会让位，避免"停住后还压着"。
        """
        pos_stop, quat_stop = self._pose()
        z_at_stop = float(pos_stop[2])
        self._update_target_state(pos_stop, quat_stop, pos_stop, quat_stop, "contact_stop",
                                  adm_pos=pos_stop, adm_quat=quat_stop, wrench=self._wrench())
        for _ in range(max(0, int(round(self.stop_settle_time / self.dt)))):
            q_ik, ok = self.kin.ik(pos_stop, quat_stop, q0=q_seed, max_iters=120, tol=5e-4)
            if not ok:
                q_ik = q_seed
            self.env.set_arm_joint_targets(q_ik)
            self.env.step()
            steps += 1
            f_after = float(np.linalg.norm(self._wrench()[:3]))
            max_force = max(max_force, f_after)
            # 安全网只在"冻结后力**继续上涨**"时才中止：触发值本身可能就高于硬限
            # （单步撞击，如 live 的 18.66N > 15N），那不是继续上涨，不能因此中止。
            # 系数 2.0：接触后机械臂会因重力静差轻微下沉（实测 11N->19N），那是物理静置、
            # 不是失控；真正的失控是旧符号那种一路涨到 60~200N，2x 仍能兜住。
            if f_after > max(flim, 2.0 * force_at_stop):
                return self._abort("force_limit",
                                   f"接触即停后力继续上涨 {f_after:.2f} N (触发 {force_at_stop:.2f} N, "
                                   f"硬限 {flim:.2f} N)", max_force, steps)
        pos, quat = self._pose()
        return self._stop_result("contact", force_at_stop, z_at_stop, max_force, steps, pos, quat)

    def _stop_result(self, reason: str, force_at_stop: float, z_at_stop: float,
                     max_force: float, steps: int, pos: Array, quat: Array) -> MoveResult:
        """「接触即停」的成功返回：ok=True + stop_reason/metrics（不新增错误码、不走 _halt）。

        契约字段 stop_reason / metrics 定义在 core/types.MoveResult；该文件若还没声明这两个
        字段，这里用动态属性挂上（普通 dataclass 非 slots，读法 res.stop_reason /
        res.metrics["force_at_stop"] 两种情况下都成立），Lead 补齐声明后无需改动本函数。
        """
        res = MoveResult(
            ok=True,
            message=(f"接触即停 stop_reason={reason} force_at_stop={force_at_stop:.2f}N "
                     f"z_at_stop={z_at_stop:.4f} max_force={max_force:.2f}N"),
            error_code=None, final_pos=pos, final_quat=quat,
            max_force=float(max_force), duration=float(steps) * self.dt, steps=int(steps),
        )
        try:
            res.stop_reason = str(reason)
            metrics = getattr(res, "metrics", None)
            if not isinstance(metrics, dict):
                metrics = {}
                res.metrics = metrics
            metrics["force_at_stop"] = float(force_at_stop)
            metrics["z_at_stop"] = float(z_at_stop)
        except Exception:                                   # pragma: no cover
            pass
        return res

    def _densify_joint_path(self, path: list, max_step_rad: float, q_start: Array) -> Array:
        """把输入折线按每段 <= max_step_rad 细分，并前置当前关节角作为起点。"""
        step = max(float(max_step_rad), 1e-6)
        pts = [np.asarray(q_start, dtype=float).reshape(ARM_NDOF)]
        for wp in path:
            q_wp = self.kin.clamp(np.asarray(wp, dtype=float).reshape(ARM_NDOF))
            prev = pts[-1]
            n = max(1, int(np.ceil(float(np.linalg.norm(q_wp - prev)) / step)))
            for i in range(1, n + 1):
                pts.append(prev + (q_wp - prev) * (i / n))
        return np.asarray(pts, dtype=float)

    @staticmethod
    def _joint_point_at(arr: Array, cum: Array, s: float) -> Array:
        """按弧长 s 在关节折线上线性取点。"""
        s = float(np.clip(s, 0.0, float(cum[-1])))
        i = int(np.searchsorted(cum, s, side="right") - 1)
        i = min(max(i, 0), len(cum) - 2)
        width = float(cum[i + 1] - cum[i])
        w = 0.0 if width < 1e-12 else (s - float(cum[i])) / width
        return arr[i] + (arr[i + 1] - arr[i]) * w

    def follow_joint_path(self, path: list, max_step_rad: float = 0.05,
                          timeout: float = 20.0,
                          compliance: Optional[ComplianceParams] = None,
                          settle_time: Optional[float] = None,
                          settle_tol: Optional[float] = None,
                          force_limit: Optional[float] = None) -> MoveResult:
        """关节空间路径执行：折线细分 -> 弧长 min-jerk -> 关节速度前馈 -> 末端收尾。

        参数
        ----
        path : list[(6,)] 关节路点（规划器输出）；会前置当前关节角并按 <= max_step_rad
               细分（线性段下细分几何等价，保留参数以对齐契约/未来非线性插值）。
        timeout : **仿真时间**上限（`steps * env.dt`，**不是墙钟时间**）；超时返回
               `error_code="timeout"`；move_to_pose / follow 同样语义。
        compliance : 只取 force_limit 记入 message；不做导纳、不因力超限中止。
        force_limit : 可选；显式给出时会被底层硬上限 `force_limit_cap`（默认 200 N）
               钳制，并在该路径执行中启用"超力即中止"（默认 None = 只统计不中止，
               保持旧行为）。
        settle_time / settle_tol : 末端收尾参数，默认取实例属性
               `joint_settle_time=0.06 s` / `joint_settle_tol=0.05 rad`：
               路径推进结束后继续把关节伺服到终点，直到 `|q_act - q_end| <= settle_tol`
               或累计 `settle_time`（仿真时间）。

        推进速度取 `self.joint_vel_limit`（rad/s，默认 1.0），总仿真时长约
        弧长 / joint_vel_limit + settle_time。

        为什么不再逐路点等待伺服（实测：1.055 rad 路径 / 23 路点 / UR5e 位置执行器）
        --------------------------------------------------------------------------
        * 逐点等待 tol=0.02 rad、0.30 s/点：2888 步(5.78 s)，终点误差 0.020 rad —— 慢，
          50+ 路点时正是端到端遇到的 7458 步 / 14.9 s 超时来源；
        * 逐点等待 tol=0.05 rad、0.06 s/点：630 步(1.26 s)，但命令持续甩开伺服（没有
          "追不上就不推进"的约束），终点误差 0.134 rad —— 会抓空；
        * 本方案 1.0 rad/s + 关节速度前馈：528 步(1.06 s)，终点误差 0.012 rad。
        """
        if path is None or len(path) == 0:
            return self._abort("invalid_path", "关节路径为空", 0.0, 0)
        settle_time = self.joint_settle_time if settle_time is None else float(settle_time)
        settle_tol = self.joint_settle_tol if settle_tol is None else float(settle_tol)
        params = self._resolve_params(compliance)
        flim = self._resolve_force_limit(force_limit, params)
        # 显式传 force_limit 时才中止（默认 None 保持"只统计不中止"的旧行为）
        enforce_force = force_limit is not None

        arr = self._densify_joint_path(path, max_step_rad, self._arm_q())
        seg_len = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        length = float(cum[-1])
        q_end = arr[-1]
        max_force = 0.0
        steps = 0
        use_ff = bool(self.use_velocity_feedforward and np.any(self.vel_lead != 0.0))
        p_cur, q_cur = self._tcp_pose()           # 关节路径没有笛卡尔参考：ref=cmd=当前 TCP
        self._update_target_state(p_cur, q_cur, p_cur, q_cur, "joint_path",
                                  wrench=self._wrench())

        # ---- 1) 沿弧长按 min-jerk 时间律推进 ----
        if length > 1e-9:
            duration = max(length / max(float(self.joint_vel_limit), 1e-3), 1e-3)
            n_steps = max(1, int(round(duration / self.dt)))
            h = 0.5 * self.dt
            for k in range(1, n_steps + 1):
                if steps * self.dt > timeout:
                    return self._abort(
                        "timeout", f"关节路径超时 {steps * self.dt:.3f}s > {timeout}s",
                        max_force, steps)
                t = min(k * self.dt, duration)
                q_cmd = self._joint_point_at(arr, cum, length * min_jerk(t / duration))
                qd = np.zeros(ARM_NDOF, dtype=float)
                t0 = max(t - h, 0.0)
                t1 = min(t + h, duration)
                span = t1 - t0
                if span > 1e-12:
                    qd = (self._joint_point_at(arr, cum, length * min_jerk(t1 / duration))
                          - self._joint_point_at(arr, cum, length * min_jerk(t0 / duration))) / span
                ctrl = q_cmd + (self.vel_lead * qd if use_ff else 0.0)
                if hasattr(self.kin, "clamp"):
                    ctrl = self.kin.clamp(ctrl)
                max_force = max(max_force, float(np.linalg.norm(self._wrench()[:3])))
                self.env.set_arm_joint_targets(ctrl)
                self.env.step()
                steps += 1
                p_now, q_now = self._tcp_pose()
                self._update_target_state(p_now, q_now, p_now, q_now, "joint_path",
                                          wrench=self._wrench())
                if enforce_force and max_force > flim:
                    return self._abort(
                        "force_limit", f"关节路径接触力 {max_force:.2f} N 超限 {flim:.2f} N",
                        max_force, steps)

        # ---- 2) 末端收尾：把关节真正停到终点 ----
        dwell = 0.0
        while dwell < settle_time:
            if steps * self.dt > timeout:
                return self._abort(
                    "timeout", f"关节路径收尾超时 {steps * self.dt:.3f}s > {timeout}s",
                    max_force, steps)
            if float(np.linalg.norm(self._arm_q() - q_end)) <= settle_tol:
                break
            max_force = max(max_force, float(np.linalg.norm(self._wrench()[:3])))
            self.env.set_arm_joint_targets(q_end)
            self.env.step()
            steps += 1
            dwell += self.dt
            p_now, q_now = self._tcp_pose()
            self._update_target_state(p_now, q_now, p_now, q_now, "joint_path",
                                      wrench=self._wrench())
            if enforce_force and max_force > flim:
                return self._abort(
                    "force_limit", f"关节路径收尾接触力 {max_force:.2f} N 超限 {flim:.2f} N",
                    max_force, steps)

        pos, quat = self._pose()
        q_err = float(np.linalg.norm(self._arm_q() - q_end))
        return MoveResult(
            ok=True,
            message=(f"关节路径完成 {len(path)} 路点/{len(arr) - 1} 段 steps={steps} "
                     f"t_sim={steps * self.dt:.2f}s 终点关节误差={q_err:.4f}rad "
                     f"max_force={max_force:.2f}N (force_limit 参考 {flim:.2f})"),
            final_pos=pos, final_quat=quat, max_force=max_force,
            duration=steps * self.dt, steps=steps)
    def hold(self, duration: float) -> MoveResult:
        """保持当前关节角（位置伺服）duration 秒，返回期间的力峰值。"""
        n_steps = max(0, int(round(float(duration) / self.dt)))
        q_hold = self._arm_q()
        max_force = 0.0
        steps = 0
        p_cur, q_cur = self._tcp_pose()
        self._update_target_state(p_cur, q_cur, p_cur, q_cur, "hold",
                                  wrench=self._wrench())
        for _ in range(n_steps):
            max_force = max(max_force, float(np.linalg.norm(self._wrench()[:3])))
            self.env.set_arm_joint_targets(q_hold)
            self.env.step()
            steps += 1
            p_now, q_now = self._tcp_pose()
            self._update_target_state(p_now, q_now, p_now, q_now, "hold",
                                      wrench=self._wrench())
        pos, quat = self._pose()
        return MoveResult(
            ok=True, message=f"保持 {steps * self.dt:.3f}s max_force={max_force:.2f}N",
            final_pos=pos, final_quat=quat, max_force=max_force,
            duration=steps * self.dt, steps=steps)