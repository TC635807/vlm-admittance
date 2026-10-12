"""control.trajectory —— 笛卡尔轨迹与四元数工具。

原理
----
* `min_jerk`：五次多项式 `10s^3 - 15s^4 + 6s^5`，满足 s(0)=0、s(1)=1，
  且速度与加速度在两端均为 0（C2 连续），用于点位运动抑制启停冲击。
* 位置插值：`p(s) = p0 + (p1 - p0) * min_jerk(s)`；姿态用 slerp 按**同一个** s
  插值，保证位置与姿态同时启停。
* 四元数约定：全部为 **wxyz**（与 MuJoCo 一致）；旋转向量 = 单位轴 * 转角(rad)。

踩坑点
------
1. slerp 必须走**最短弧**：若 `dot(q0, q1) < 0` 先把 q1 取反，否则末端会绕远路
   （甚至出现 360 度反转）。
2. 两四元数几乎重合时 `sin(theta0) -> 0` 会除零，退化用线性插值 + 归一化。
3. `rotvec_from_quat` 先把 w<0 的分支翻正，得到 |angle| <= pi 的最短表示；
   角度接近 0 时用 `2*v` 一阶近似，避免 `v/|v|` 除零。
4. `sample(t)` 对 t 做钳制（t<0 取 0，t>duration 取 duration），调用方传超界的 t
   也安全；`duration` 以 1e-9 兜底，避免除零。
5. `rot0/rot1` 均为 None 时不插值姿态，恒定返回单位四元数；只给 `rot0` 时
   恒定返回 `rot0`。
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray

#: 单位四元数（wxyz）
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


# --------------------------------------------------------------------------- #
# 标量 / 四元数基础
# --------------------------------------------------------------------------- #
def min_jerk(s: float) -> float:
    """五次最小加加速度时间律，输入 s 会被钳制到 [0, 1]。"""
    s = float(np.clip(float(s), 0.0, 1.0))
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def normalize_quat(q: Array) -> Array:
    """归一化四元数；模长过小时返回单位四元数（容错）。"""
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return IDENTITY_QUAT.copy()
    return q / n


def quat_mul(a: Array, b: Array) -> Array:
    """Hamilton 乘积 `a (x) b`（wxyz）。

    旋转复合语义：先施加 b 的旋转，再施加 a 的旋转（与矩阵乘法 R(a) R(b) 一致）。
    """
    a = np.asarray(a, dtype=float).reshape(4)
    b = np.asarray(b, dtype=float).reshape(4)
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: Array) -> Array:
    """共轭四元数（单位四元数前提下等于逆旋转）。"""
    q = np.asarray(q, dtype=float).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_rotvec(rv: Array) -> Array:
    """旋转向量（轴*角, rad）-> 四元数 wxyz。"""
    rv = np.asarray(rv, dtype=float).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        # 一阶展开：sin(theta/2)/theta -> 1/2
        return normalize_quat(np.array([1.0, 0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2]]))
    axis = rv / theta
    half = 0.5 * theta
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def rotvec_from_quat(q: Array) -> Array:
    """四元数 wxyz -> 旋转向量（|angle| <= pi 的最短表示）。"""
    q = normalize_quat(q)
    if q[0] < 0.0:                       # 翻到 w>=0 的分支，保证最短
        q = -q
    v = q[1:]
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:
        return 2.0 * v                   # 小角度一阶近似，避免除零
    angle = 2.0 * float(np.arctan2(nv, q[0]))
    return v / nv * angle


def slerp(q0: Array, q1: Array, s: float) -> Array:
    """球面线性插值（wxyz），走最短弧；s 会钳制到 [0, 1]。"""
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    s = float(np.clip(float(s), 0.0, 1.0))
    d = float(np.dot(q0, q1))
    if d < 0.0:                          # 最短弧
        q1 = -q1
        d = -d
    if d > 0.9995:                       # 近平行：线性插值 + 归一化
        return normalize_quat(q0 + s * (q1 - q0))
    theta0 = float(np.arccos(np.clip(d, -1.0, 1.0)))
    theta = theta0 * s
    sin_theta0 = float(np.sin(theta0))
    w0 = np.sin(theta0 - theta) / sin_theta0
    w1 = np.sin(theta) / sin_theta0
    return normalize_quat(w0 * q0 + w1 * q1)


# --------------------------------------------------------------------------- #
# 笛卡尔轨迹
# --------------------------------------------------------------------------- #
class CartesianTrajectory:
    """位置最小加加速度 + 姿态 slerp 的直线笛卡尔轨迹。

    参数
    ----
    p0, p1 : (3,) 起止位置（世界系）
    duration : float 轨迹总时长(s)，必须 > 0
    dt : float 采样周期(s)，仅记录/供调用方对齐（本类不参与推进）
    rot0, rot1 : (4,) wxyz 起止姿态；都给定才做姿态插值。
                 两者都给且相同 -> 姿态恒定；只给 rot0 -> 恒定 rot0。
    """

    def __init__(self, p0: Array, p1: Array, duration: float, dt: float,
                 rot0: Array | None = None, rot1: Array | None = None) -> None:
        self.p0 = np.asarray(p0, dtype=float).reshape(3).copy()
        self.p1 = np.asarray(p1, dtype=float).reshape(3).copy()
        self.rot0 = None if rot0 is None else normalize_quat(rot0)
        self.rot1 = None if rot1 is None else normalize_quat(rot1)
        self.duration = max(float(duration), 1e-9)
        self.dt = float(dt)
        #: 直线轨迹总长度，方便调用方估算速度
        self.length = float(np.linalg.norm(self.p1 - self.p0))

    # -- 采样 ---------------------------------------------------------------- #
    def sample(self, t: float) -> tuple[Array, Array]:
        """采样 t 时刻的 (pos(3), quat wxyz(4))；t 自动钳制到 [0, duration]。"""
        s = min_jerk(float(t) / self.duration)
        pos = (self.p0 + (self.p1 - self.p0) * s).astype(float)
        if self.rot0 is not None and self.rot1 is not None:
            quat = slerp(self.rot0, self.rot1, s)
        elif self.rot0 is not None:
            quat = self.rot0.copy()
        else:
            quat = IDENTITY_QUAT.copy()
        return pos, quat

    def sample_at(self, s: float) -> tuple[Array, Array]:
        """按归一化进度 s∈[0,1] 采样（等价于 sample(min_jerk(s)*duration)）。"""
        return self.sample(min_jerk(s) * self.duration)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (f"CartesianTrajectory(p0={np.round(self.p0, 4).tolist()}, "
                f"p1={np.round(self.p1, 4).tolist()}, duration={self.duration:.3f}, "
                f"length={self.length:.4f})")
