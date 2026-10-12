"""sim/overlay.py —— 只操作 Mujoco MjvScene 的目标点/参考点/轨迹绘制工具。

用途
----
给交互式 viewer 的 user_scn（mujoco.viewer 的 MjvScene）画“当前机械臂目标点”：
白色/青色小球 + 三根细胶囊表示姿态；当 ref_pos（导纳让位后的参考点）与 pos 不同时，
再画一个橙色 reference 小球并用细胶囊连起来，便于肉眼看出导纳修正量。

接口（冻结，ui-fixer 按此调用）
-------------------------------
    clear(scn) -> None
    draw_target(scn, pos, quat=None, *, ref_pos=None, radius=0.012, axes_len=0.06,
                show_ref=True, show_label=True, rgba=None) -> int   # 返回新增 geom 数
    draw_trail(scn, points, rgba=None, radius=0.004) -> int

原理与踩坑点（本机 MuJoCo 3.12 实测）
------------------------------------
* 只调用 mujoco.mjv_initGeom(geom, type, size, pos, mat, rgba) 写 scn.geoms[i]，
  再从 scn.ngeom 往后推进；**不 import mujoco.viewer、不建 GL 上下文**，
  因此可以在无显示环境（无 DISPLAY / 无 GL）下用 unittest 直接测。
* mjv_initGeom 的 mat 参数是**必填**的：本机传 None 会 TypeError
  （lead 提示“球体可传 None”在此版本不成立）。球体朝向无意义，传单位阵即可。
* mat 是 9 元素。实测 mju_quat2Mat 生成的 9 元素数组与 mjv_initGeom 之后读回的
  g.mat（按 numpy row-major reshape 成 3x3）完全一致，所以**用 mju_quat2Mat 生成
  mat 即可**，不要手写 mat、也不要猜列主序。约定自洽后：
  g.mat 的**第 3 列 = geom 局部 +z 在世界系的方向**（MJV/引擎的 capsule、cylinder
  都以局部 +z 为对称轴），三轴胶囊就是靠这个把 +z 转到目标方向。
* 读回时注意形状差异：g.mat 是 (3,3) ndarray，而 g.pos / g.size / g.rgba 是 (3,)/(4,)。
* mjv_initGeom 会**重排 size**（本机 3.12 实测）：sphere 传 (r,0,0) 读回 (r,r,r)；
  capsule 传 (radius, half_length, 0) 读回 (radius, radius, half_length)。
  所以传入用引擎约定，写测试断言时不要假设"传入即读回"。
* scn.maxgeom 可能很小甚至为 0（此时 len(scn.geoms)==0）：任何索引都会越界，
  所以每次写入前都要判 i < maxgeom，写满就停止并返回实际写入数量，不抛异常。
* MjvGeom.label 是 char[100]，写入超过 99 字符会被截断（这里统一截断到 LABEL_MAX）。
* 位置/尺寸用 float64、rgba 用 float32 传给 mjv_initGeom（与 C 端 const double*/float* 对齐）。
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import mujoco
import numpy as np

# --------------------------------------------------------------------------- #
# 默认样式
# --------------------------------------------------------------------------- #
#: 目标小球半径（米）
DEFAULT_RADIUS: float = 0.012
#: 三轴胶囊长度（米）
DEFAULT_AXES_LEN: float = 0.06
#: 轨迹胶囊半径（米）
DEFAULT_TRAIL_RADIUS: float = 0.004
#: 目标点颜色（青色）
RGBA_TARGET: tuple = (0.10, 0.90, 0.95, 1.0)
#: 导纳参考点颜色（橙色）
RGBA_REF: tuple = (0.95, 0.55, 0.10, 1.0)
#: 参考点与目标点之间的连线颜色（黄色）
RGBA_LINK: tuple = (0.95, 0.85, 0.20, 1.0)
#: 三轴颜色 x/y/z
RGBA_AXES: tuple = (
    (0.95, 0.20, 0.20, 1.0),
    (0.20, 0.95, 0.25, 1.0),
    (0.25, 0.45, 0.95, 1.0),
)
#: 轨迹颜色
RGBA_TRAIL: tuple = (1.00, 0.75, 0.20, 0.9)
#: 轴胶囊半径 = 目标半径 * AXES_RADIUS_SCALE
AXES_RADIUS_SCALE: float = 0.30
#: 连线胶囊半径 = 目标半径 * LINK_RADIUS_SCALE
LINK_RADIUS_SCALE: float = 0.35
#: MjvGeom.label 容量为 char[100]，安全上限 99
LABEL_MAX: int = 99
#: 判断两点是否重合的阈值（米）
EPS_DIST: float = 1e-9

_IDENTITY9 = np.eye(3, dtype=np.float64).reshape(9)
_Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
class _Sink:
    """按 scn.ngeom 顺序写 geom，并保证不越过 scn.maxgeom。"""

    __slots__ = ("scn", "maxgeom", "index", "added")

    def __init__(self, scn):
        self.scn = scn
        self.maxgeom = max(0, int(getattr(scn, "maxgeom", 0)))
        start = int(getattr(scn, "ngeom", 0))
        self.index = min(max(start, 0), self.maxgeom)
        self.added = 0

    def claim(self):
        """申请下一个 geom；容量不足返回 None（不越界、不抛异常）。"""
        if self.index >= self.maxgeom:
            return None
        geom = self.scn.geoms[self.index]
        self.index += 1
        self.added += 1
        return geom

    def close(self) -> int:
        self.scn.ngeom = self.index
        return self.added


def _as_vec3(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError("%s 需要 3 个分量，收到 %d 个" % (name, arr.size))
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s 含非有限值：%r" % (name, value))
    return arr


def _as_quat(value) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError("quat 需要 4 个分量 (wxyz)，收到 %d 个" % q.size)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < EPS_DIST:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def _as_rgba(value, default) -> np.ndarray:
    src = default if value is None else value
    arr = np.asarray(src, dtype=np.float32).reshape(-1)
    if arr.size == 3:
        arr = np.concatenate([arr, np.ones(1, dtype=np.float32)])
    if arr.size != 4:
        raise ValueError("rgba 需要 3 或 4 个分量，收到 %d 个" % arr.size)
    return arr


def _mat_from_quat(quat: np.ndarray) -> np.ndarray:
    """wxyz 四元数 -> 9 元素旋转矩阵（与 mjv_initGeom 的 mat 参数同约定）。"""
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat)
    return mat


def _rotate(quat: np.ndarray, vec) -> np.ndarray:
    """用四元数旋转向量（世界系），避免依赖 mat 的存储约定。"""
    out = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(out, np.asarray(vec, dtype=np.float64).reshape(3), quat)
    return out


def _quat_from_z_to(direction: np.ndarray) -> np.ndarray:
    """构造把局部 +z 转到 direction 的四元数（wxyz）。

    capsule/cylinder 在 MJV 中以局部 +z 为对称轴，因此画任意方向的胶囊时
    只需要这个旋转。direction 与 -z 近乎反向时绕 x 轴转 180 度。
    """
    d = direction / max(float(np.linalg.norm(direction)), EPS_DIST)
    c = float(np.dot(_Z_AXIS, d))
    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-12:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(_Z_AXIS, d)
    axis /= float(np.linalg.norm(axis))
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_axisAngle2Quat(quat, axis, float(np.arccos(np.clip(c, -1.0, 1.0))))
    return quat


def _decorate(geom, label: Optional[str]) -> None:
    """写标签（截断到 LABEL_MAX），并把 geom 标成 viewer 装饰元素。"""
    if label is not None:
        geom.label = str(label)[:LABEL_MAX]
    try:
        geom.category = int(mujoco.mjtCategory.mjCAT_DECOR)
    except Exception:                      # 该绑定没有 category 字段时忽略
        pass


def _write_sphere(sink: _Sink, pos: np.ndarray, radius: float, rgba: np.ndarray,
                  label: Optional[str]) -> bool:
    geom = sink.claim()
    if geom is None:
        return False
    size = np.array([float(radius), 0.0, 0.0], dtype=np.float64)
    mujoco.mjv_initGeom(geom, int(mujoco.mjtGeom.mjGEOM_SPHERE), size, pos,
                        _IDENTITY9, rgba)          # 球体朝向无意义 -> 单位阵
    _decorate(geom, label)
    return True


def _write_capsule(sink: _Sink, p0: np.ndarray, p1: np.ndarray, radius: float,
                   rgba: np.ndarray, label: Optional[str]) -> bool:
    """在 p0-p1 之间画一根胶囊（长度<=EPS 时跳过，返回 False）。"""
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length <= EPS_DIST:
        return False
    geom = sink.claim()
    if geom is None:
        return False
    quat = _quat_from_z_to(delta)
    size = np.array([float(radius), 0.5 * length, 0.0], dtype=np.float64)
    center = 0.5 * (p0 + p1)
    mujoco.mjv_initGeom(geom, int(mujoco.mjtGeom.mjGEOM_CAPSULE), size, center,
                        _mat_from_quat(quat), rgba)
    _decorate(geom, label)
    return True


# --------------------------------------------------------------------------- #
# 冻结接口
# --------------------------------------------------------------------------- #
def clear(scn) -> None:
    """清空场景里的所有 geom（只把 scn.ngeom 置 0，不释放缓冲）。"""
    scn.ngeom = 0


def draw_target(scn, pos, quat=None, *, ref_pos=None, radius: float = DEFAULT_RADIUS,
                axes_len: float = DEFAULT_AXES_LEN, show_ref: bool = True,
                show_label: bool = True, rgba=None) -> int:
    """在 scn 上画“当前目标点”，返回新增的 geom 数量。

    参数
    ----
    pos : (3,) 目标位置（世界系，米）
    quat : (4,) wxyz；给定时额外画三根细胶囊表示姿态（x 红 / y 绿 / z 蓝），
           不给则只画球。
    ref_pos : (3,) 导纳让位后的参考位置；与 pos 距离 > 1e-9 且 show_ref=True 时，
              额外画橙色 reference 小球 + 黄色连线胶囊。
    radius / axes_len : 小球半径 / 三轴胶囊长度。
    show_ref / show_label : 控制参考点与标签。
    rgba : 目标小球颜色，None 用 RGBA_TARGET（青色）。

    返回实际写入的 geom 数：1（球）+ 3（有 quat 时的三轴）
    + 1（参考球）+ 1（连线），容量不足时按实际截断数返回。
    """
    sink = _Sink(scn)
    p = _as_vec3(pos, "pos")
    color = _as_rgba(rgba, RGBA_TARGET)
    label = "target" if show_label else None

    _write_sphere(sink, p, radius, color, label)

    if quat is not None:
        q = _as_quat(quat)
        axis_radius = float(radius) * AXES_RADIUS_SCALE
        for k in range(3):
            local = np.zeros(3, dtype=np.float64)
            local[k] = 1.0
            tip = p + _rotate(q, local) * float(axes_len)
            _write_capsule(sink, p, tip, axis_radius,
                           _as_rgba(RGBA_AXES[k], RGBA_AXES[k]),
                           ("axis_%s" % "xyz"[k]) if show_label else None)

    if show_ref and ref_pos is not None:
        r = _as_vec3(ref_pos, "ref_pos")
        if float(np.linalg.norm(r - p)) > EPS_DIST:
            _write_sphere(sink, r, radius, _as_rgba(RGBA_REF, RGBA_REF),
                          "ref(admittance)" if show_label else None)
            _write_capsule(sink, r, p, float(radius) * LINK_RADIUS_SCALE,
                           _as_rgba(RGBA_LINK, RGBA_LINK),
                           "ref_link" if show_label else None)

    return sink.close()


def draw_trail(scn, points, rgba=None, radius: float = DEFAULT_TRAIL_RADIUS) -> int:
    """把一串目标点画成轨迹：每个点一个小球，相邻点之间一根胶囊。

    points : (N,3) 数组或 list（N=0 返回 0）。
    rgba   : 轨迹颜色，None 用 RGBA_TRAIL。
    radius : 轨迹球/胶囊半径。

    返回实际写入的 geom 数 = 点数 + 有效线段数（重合的相邻点不画线段）；
    容量不足时截断并返回实际数量。
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return 0
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points 需要 (N,3)，收到 shape=%r" % (pts.shape,))
    if not np.all(np.isfinite(pts)):
        raise ValueError("points 含非有限值")

    sink = _Sink(scn)
    color = _as_rgba(rgba, RGBA_TRAIL)
    for p in pts:
        _write_sphere(sink, p, radius, color, "trail")
    for k in range(pts.shape[0] - 1):
        _write_capsule(sink, pts[k], pts[k + 1], radius, color, "trail")
    return sink.close()
