"""control.admittance —— 六维笛卡尔导纳控制器（半隐式欧拉离散）。

被控方程（与 INTERFACES.md 冻结一致）
-------------------------------------
`M * x_dd + D * x_d + K * (x - x_ref) = -f_ext`

其中 x 为 6 维"位置3 + 旋转向量3"（平动/转动统一按线性导纳近似处理），
`f_ext = [force(3), torque(3)]` 为末端外力/力矩（世界系）。

半隐式欧拉（symplectic Euler，先更新速度、再用**新速度**更新位置）
---------------------------------------------------------------
`a_k      = ( -f_ext - D * v_k - K * (x_k - x_ref) ) / M`
`v_{k+1}  = v_k + dt * a_k`
`x_{k+1}  = x_k + dt * v_{k+1}`

踩坑点
------
1. **稳定性条件**：无阻尼时半隐式欧拉的特征多项式为
   `L^2 - (2 - dt^2 w^2) L + 1 = 0`，`w = sqrt(K/M)`，要求
   `dt * sqrt(K/M) < 2`（推荐 <= 0.5 保证精度）；dt=0.002、K=5000、M=5 时
   dt*w ~= 0.063，非常安全。构造/`set_params` 时会计算 `stability_ratio` 并在
   超限时 `warnings.warn`（只提示不抛错，避免打断控制环）。
2. **`axes[i]==0` 的轴不做导纳**：该轴 `x = x_ref`、速度清零，等价纯位置控制
   （z_only 模式就是这种用法）。`axes[i]==1` 但 `m[i] <= 0` 视为配置非法，
   同样退化为"跟随 x_ref"并给一次警告，绝不除零。
3. **单步修正限幅**：平动修正向量模 <= 0.05 m、转动修正向量模 <= 0.15 rad，
   防止接触瞬间弹飞。限幅后把该轴速度回算为 `delta/dt`（抗积分饱和），
   否则速度会继续累积、离开接触时爆冲。
4. **力符号约定**：严格按冻结方程取 `-f_ext`。若 sim 的 `contact_force_on_ee()`
   返回的是"环境作用在 EE 上的力"，则末端会**逆着该力**退让；需要顺着力退让时
   在调用处把 wrench 取负（或检查 sim 的符号约定），本类不做隐式猜测。
5. x 的转动通道在 MotionRunner 里以"相对姿态增量(rotvec)"使用（x_ref 转动部分为 0），
   避免绝对旋转向量在 ±pi 附近奇异；本控制器对"绝对 rotvec"和"增量 rotvec"两种
   用法都成立（方程是线性的）。
"""
from __future__ import annotations

import warnings

import numpy as np

from core.types import ComplianceParams

Array = np.ndarray

#: 单步修正限幅：前 3 维平动(m)，后 3 维转动(rad)
STEP_LIMITS = np.array([0.05, 0.05, 0.05, 0.15, 0.15, 0.15], dtype=float)


class AdmittanceController:
    """6 维导纳控制器。

    参数
    ----
    params : core.types.ComplianceParams（k/d/m/axes/force_limit）
    dt : float 采样周期(s)，应与物理步长 `env.dt` 一致
    """

    def __init__(self, params: ComplianceParams, dt: float) -> None:
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError(f"dt 必须 > 0，收到 {dt!r}")
        self._params = params
        self._x = np.zeros(6, dtype=float)          # 当前位置（位置3 + rotvec3）
        self._xd = np.zeros(6, dtype=float)         # 当前速度
        self._x_ref = np.zeros(6, dtype=float)      # 参考位置
        self._last_delta = np.zeros(6, dtype=float)
        self._clipped = False
        self._steps = 0
        self._enabled = np.zeros(6, dtype=bool)
        self.stability_ratio = 0.0
        self.stable = True
        self._refresh_params(self._x_ref)

    # ------------------------------------------------------------------ #
    # 参数
    # ------------------------------------------------------------------ #
    @property
    def params(self) -> ComplianceParams:
        return self._params

    def _refresh_params(self, x_ref: Array) -> None:
        """重算使能掩码与稳定性指标；非法轴(axes=1 但 m<=0)退化为跟随。"""
        p = self._params
        k = np.asarray(p.k, dtype=float).reshape(6)
        m = np.asarray(p.m, dtype=float).reshape(6)
        axes = np.asarray(p.axes, dtype=float).reshape(6)
        raw = axes != 0.0
        bad = raw & (m <= 0.0)
        if np.any(bad):
            warnings.warn(
                f"导纳轴 {np.where(bad)[0].tolist()} 的虚拟质量 m<=0，已退化为纯位置跟随",
                RuntimeWarning, stacklevel=2)
        self._enabled = raw & ~bad
        # 稳定性指标：dt * sqrt(K/M)，取所有使能轴最大值（K<=0 视为无穷稳定）
        ratio = 0.0
        for i in np.where(self._enabled)[0]:
            if k[i] > 0.0:
                ratio = max(ratio, self.dt * float(np.sqrt(k[i] / m[i])))
        self.stability_ratio = ratio
        self.stable = ratio < 2.0
        if not self.stable:
            warnings.warn(
                f"导纳离散可能发散：dt*sqrt(K/M)={ratio:.3f} >= 2（无阻尼稳定边界）。"
                f"请减小 dt 或降低刚度/提高虚拟质量。",
                RuntimeWarning, stacklevel=2)
        # 参数变化后，非使能轴直接吸附到参考
        if x_ref is not None:
            self._x[~self._enabled] = np.asarray(x_ref, dtype=float)[~self._enabled]

    def set_params(self, params: ComplianceParams) -> None:
        """切换柔顺参数；位置/速度状态保留（非使能轴会在下一次 update 吸附到 x_ref）。"""
        self._params = params
        self._refresh_params(self._x_ref)

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    def reset(self, x_ref: Array) -> None:
        """把内部状态复位到参考位置 x_ref(6)（位置3 + rotvec3），速度清零。"""
        x_ref = np.asarray(x_ref, dtype=float).reshape(6)
        self._x_ref = x_ref.copy()
        self._x = x_ref.copy()
        self._xd = np.zeros(6, dtype=float)
        self._last_delta = np.zeros(6, dtype=float)
        self._clipped = False
        self._steps = 0

    def state(self) -> dict:
        """返回当前内部状态快照（可直接塞进 SkillResult.metrics）。"""
        return {
            "x": self._x.tolist(),
            "xd": self._xd.tolist(),
            "x_ref": self._x_ref.tolist(),
            "enabled": self._enabled.astype(float).tolist(),
            "last_delta": self._last_delta.tolist(),
            "clipped": bool(self._clipped),
            "steps": int(self._steps),
            "dt": self.dt,
            "stability_ratio": float(self.stability_ratio),
            "stable": bool(self.stable),
            "params": self._params.to_json(),
        }

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def update(self, x_ref: Array, wrench: Array) -> Array:
        """推进一个 dt，返回修正后的命令位姿 x_cmd(6)。

        x_ref : (6,) 参考位姿（位置3 + rotvec3）
        wrench : (6,) 末端外力/力矩（世界系），顺序 [fx,fy,fz,mx,my,mz]
        """
        x_ref = np.asarray(x_ref, dtype=float).reshape(6)
        f = np.asarray(wrench, dtype=float).reshape(-1)
        if f.size < 6:
            f = np.concatenate([f, np.zeros(6 - f.size)])
        f = np.nan_to_num(f[:6], nan=0.0, posinf=0.0, neginf=0.0)

        p = self._params
        k = np.asarray(p.k, dtype=float).reshape(6)
        d = np.asarray(p.d, dtype=float).reshape(6)
        m = np.asarray(p.m, dtype=float).reshape(6)
        dt = self.dt
        en = self._enabled

        x = self._x
        v = self._xd
        self._x_ref = x_ref.copy()

        # --- 半隐式欧拉 ---
        acc = np.zeros(6, dtype=float)
        idx = np.where(en)[0]
        acc[idx] = (-f[idx] - d[idx] * v[idx] - k[idx] * (x[idx] - x_ref[idx])) / m[idx]
        v_new = np.zeros(6, dtype=float)
        v_new[idx] = v[idx] + dt * acc[idx]
        delta = np.zeros(6, dtype=float)
        delta[idx] = dt * v_new[idx]

        # --- 单步修正限幅（按组取模长，只统计使能轴）---
        self._clipped = False
        for g0 in (0, 3):
            gidx = np.arange(g0, g0 + 3)
            sel = gidx[en[gidx]]
            if sel.size == 0:
                continue
            norm = float(np.linalg.norm(delta[sel]))
            lim = float(STEP_LIMITS[g0])
            if norm > lim:
                delta[sel] *= lim / norm
                self._clipped = True

        x = x + delta
        # 抗饱和：限幅后用实际位移回算速度
        v_out = np.zeros(6, dtype=float)
        v_out[idx] = delta[idx] / dt
        # --- 非使能轴：纯位置跟随 ---
        x[~en] = x_ref[~en]
        v_out[~en] = 0.0

        self._x = x
        self._xd = v_out
        self._last_delta = x - self._x_ref
        self._steps += 1
        return self._x.copy()
