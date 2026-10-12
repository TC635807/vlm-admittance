"""sim/vision.py —— 相机标定与像素 <-> 世界反投影（INTERFACES.md §4.1 冻结契约）。

用途：VLM 感知闭环。大模型只输出目标中心像素 (u,v)，这里把像素反投影到桌面平面
得到世界坐标；技能层只接收世界 xy，不接收物体真值位姿。

原理（针孔 + 竖直视场角）
------------------------
1. 相机位姿由 mj_forward 得到（**不硬编码任何相机位置**）：
   data.cam_xpos 是世界位置；data.cam_xmat 是 3x3 旋转矩阵，**列 = 相机局部
   x右 / y上 / z后 轴在世界系的分量**（MuJoCo 相机沿自身 -z 轴看）。
2. 焦距（像素）：f = (H/2) / tan(fovy/2)，fovy 取 model.cam_fovy（度）；
   正方形像素 => f_x = f_y = f。
3. world -> 相机系： p_c = R^T (P - cam_pos)，depth = -p_c[2]（>0 表示在相机前方）
       u = W/2 + f * p_c[0] / depth      （u 向右，原点在图像左上角）
       v = H/2 - f * p_c[1] / depth      （v 向下）
4. pixel -> ray：相机系方向 d_c = normalize([(u-W/2)/f, -(v-H/2)/f, -1])，
   再 d_w = R d_c；与水平面 z 求交得世界点。

实测校验（本机 MuJoCo 3.12）
---------------------------
* top 相机的 cam_xmat 列 = x右(-1,0,0)、y上(0,-1,0)、z后(0,0,1)（相机朝下）。
* 把 5 个物体中心投影成像素后在渲染图上采样：box_red->[255,108,98]、
  can_yellow->[255,255,131]、overview 下同样各自对上颜色；
  该像素的深度图数值 ≈ 预测 depth 减去"物体中心到迎光面"的距离（方块 0.02、
  圆柱 0.035，完全吻合）——说明 y 轴方向与 v 向下都取对了。
* 桌面点 project -> pixel -> pixel_to_plane 往返误差约 1e-15 m（见 vision_smoke）。

越界 / 相机后方的行为（契约要求，务必对齐）
------------------------------------------
* project(xyz)：点落在相机后方（depth<=0）返回 None；投影像素落在图像外
  （u<0 或 u>=W 或 v<0 或 v>=H）也返回 None；输入非有限值同样返回 None。
  含义：调用方拿不到像素 = "模型瞄偏了/该点不可见"。
* pixel_to_plane(u,v,z)：**不做图像越界检查**，始终按针孔模型求交，因此 u/v 越界时
  会返回图像外的世界点（允许，用于诊断与鲁棒性）；只有射线与水平面近乎平行
  （|dir_z| < 1e-12）或平面在相机后方（t <= 0）才返回 None。
* ray(u,v)：任意 u/v（含负数、超大值）都返回 (origin, 单位方向)，不抛异常。
  三者对越界像素都不崩溃。

其他实现约定
------------
* 相机位姿用**私有 MjData(env.model)** 做 mj_forward，绝不改动 env.data
  （与 planning/control 的私有临时 data 约定一致）。
* 腕部相机随机械臂运动：每次 project/ray 前 _sync() 会比较私有 data 与 env.data 的
  qpos，一旦不同就重新 mj_forward 刷新 cam_pos/cam_mat；也可显式 update()。
* 分辨率取 env.render_size（与 sim/render.py 一致，默认 512x512），
  保证像素坐标和实际渲染图完全对应。

公开接口（冻结，不得改签名）
----------------------------
    class CameraModel:
        __init__(env, camera="top")
        project(xyz) -> (u, v) | None
        ray(u, v) -> (origin(3,), dir(3,))
        pixel_to_plane(u, v, z) -> xyz(3,) | None
    属性: name, width, height, fovy(deg), cam_pos(3,), cam_mat(3,3)
    camera_model(env, camera="top") -> CameraModel
    pixel_to_table(env, u, v, camera="top", z=TABLE_HEIGHT) -> xyz(3,) | None
    project_point(env, xyz, camera="top") -> (u, v) | None
附加（供 verifier 复用，不属于冻结清单）:
    reprojection_errors(env, points, camera="top", z=None) -> np.ndarray
    reprojection_summary(env, points, camera="top", z=None) -> dict
"""
from __future__ import annotations

from typing import Optional, Sequence

import mujoco
import numpy as np

from core.config import RENDER_HEIGHT, RENDER_WIDTH, TABLE_HEIGHT

#: 判定 depth<=0（相机后方）的容差
DEPTH_EPS: float = 1e-12
#: 判定射线与水平面平行的容差
PARALLEL_EPS: float = 1e-12


def _as_vec3(value, name: str) -> Optional[np.ndarray]:
    """转成 (3,) float64；形状不对或含非有限值时返回 None（不抛异常）。"""
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        return None
    return arr


class CameraModel:
    """单个 MuJoCo 相机的针孔模型（标定来自 model.cam_fovy + mj_forward 位姿）。"""

    def __init__(self, env, camera: str = "top"):
        self.env = env
        self.model = env.model
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, str(camera))
        if cam_id < 0:
            raise ValueError("模型里不存在相机 %r" % (camera,))
        self.name = str(camera)
        self.cam_id = int(cam_id)
        self.fovy = float(self.model.cam_fovy[self.cam_id])
        if not (0.0 < self.fovy < 180.0):
            raise ValueError("相机 %r 的 fovy 非法：%r" % (self.name, self.fovy))

        size = getattr(env, "render_size", (RENDER_WIDTH, RENDER_HEIGHT))
        self.width = int(size[0])
        self.height = int(size[1])
        if self.width <= 0 or self.height <= 0:
            raise ValueError("渲染分辨率非法：%r" % (size,))

        #: 焦距（像素），正方形像素
        self._focal = (self.height / 2.0) / float(np.tan(np.radians(self.fovy) / 2.0))

        #: 相机世界位置（3,）；由 update() 从 mj_forward 结果刷新
        self.cam_pos = np.zeros(3, dtype=np.float64)
        #: 相机世界旋转矩阵（3,3，列 = x右 / y上 / z后）；由 update() 刷新
        self.cam_mat = np.eye(3, dtype=np.float64)

        # 私有临时 MjData：绝不动 env.data
        self._data = mujoco.MjData(self.model)
        self.update()

    # ------------------------------------------------------------------ #
    # 位姿刷新
    # ------------------------------------------------------------------ #
    def update(self) -> "CameraModel":
        """把 env.data.qpos 复制到私有 MjData 并 mj_forward，刷新 cam_pos/cam_mat。"""
        env_data = self.env.data
        d = self._data
        if d.qpos.shape != env_data.qpos.shape:
            raise ValueError("env.model 与已缓存的 MjData 不匹配（nq 不同）")
        d.qpos[:] = env_data.qpos
        d.qvel[:] = env_data.qvel
        mujoco.mj_forward(self.model, d)
        self.cam_pos = d.cam_xpos[self.cam_id].copy()
        self.cam_mat = d.cam_xmat[self.cam_id].reshape(3, 3).copy()
        return self

    def _sync(self) -> None:
        """qpos 变了（例如腕部相机随臂运动）就重新标定。"""
        if not np.array_equal(self._data.qpos, self.env.data.qpos):
            self.update()

    # ------------------------------------------------------------------ #
    # 冻结接口
    # ------------------------------------------------------------------ #
    def project(self, xyz) -> Optional[tuple]:
        """world -> 像素 (u, v)；u 向右、v 向下、原点左上。

        返回 None 的三种情况（契约规定，不抛异常）：
          * 点在相机后方（depth = -p_c[2] <= 0）；
          * 投影结果落在图像外（u<0 / u>=W / v<0 / v>=H）；
          * xyz 形状不是 (3,) 或含 NaN/Inf。
        """
        point = _as_vec3(xyz, "xyz")
        if point is None:
            return None
        self._sync()
        p_c = self.cam_mat.T @ (point - self.cam_pos)
        depth = -float(p_c[2])
        if depth <= DEPTH_EPS:
            return None
        u = self.width / 2.0 + self._focal * float(p_c[0]) / depth
        v = self.height / 2.0 - self._focal * float(p_c[1]) / depth
        if not np.isfinite(u) or not np.isfinite(v):
            return None
        if not (0.0 <= u < self.width and 0.0 <= v < self.height):
            return None
        return (float(u), float(v))

    def ray(self, u, v) -> tuple:
        """像素 (u, v) -> (origin(3,), 单位方向(3,))，均在世界系。

        不做越界检查：任意 u/v（负数、超大值、NaN 之外）都返回一条射线。
        """
        self._sync()
        try:
            fu = float(u)
            fv = float(v)
        except Exception:
            raise ValueError("u/v 必须是数值，收到 %r / %r" % (u, v))
        if not (np.isfinite(fu) and np.isfinite(fv)):
            raise ValueError("u/v 必须是有限数值，收到 %r / %r" % (u, v))
        x_c = (fu - self.width / 2.0) / self._focal
        y_c = -(fv - self.height / 2.0) / self._focal
        direction = self.cam_mat @ np.array([x_c, y_c, -1.0], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < DEPTH_EPS:
            raise ValueError("像素射线退化（u=%r, v=%r）" % (u, v))
        return self.cam_pos.copy(), direction / norm

    def pixel_to_plane(self, u, v, z: float) -> Optional[np.ndarray]:
        """像素 (u, v) 的射线与水平面 z 求交，返回世界点 (3,)；无交点返回 None。

        不做图像越界检查：u/v 越界时返回图像外的世界点（允许）。
        返回 None 只发生在：射线与水平面近乎平行（|dir_z| < 1e-12），
        或交点位于相机后方（t <= 0）；z 非法时也返回 None。
        """
        try:
            plane_z = float(z)
        except Exception:
            return None
        if not np.isfinite(plane_z):
            return None
        origin, direction = self.ray(u, v)
        dz = float(direction[2])
        if abs(dz) < PARALLEL_EPS:
            return None
        t = (plane_z - float(origin[2])) / dz
        if t <= 0.0:
            return None
        return origin + t * direction


# --------------------------------------------------------------------------- #
# 模块级便捷函数（冻结）
# --------------------------------------------------------------------------- #
def camera_model(env, camera: str = "top") -> CameraModel:
    """构造指定相机的 CameraModel（每次新建，标定即时来自 mj_forward）。"""
    return CameraModel(env, camera)


def pixel_to_table(env, u, v, camera: str = "top", z: float = TABLE_HEIGHT) -> Optional[np.ndarray]:
    """像素 -> 桌面平面世界坐标 (3,)；无交点（平行/平面在相机后方）返回 None。

    约定：u 右、v 下、原点左上；u/v 允许越界（返回图像外的世界点，不崩溃）。
    """
    return CameraModel(env, camera).pixel_to_plane(u, v, z)


def project_point(env, xyz, camera: str = "top") -> Optional[tuple]:
    """世界点 -> 像素 (u, v)；相机后方或像素越界返回 None（不崩溃）。"""
    return CameraModel(env, camera).project(xyz)


# --------------------------------------------------------------------------- #
# 供 verifier 复用的往返误差工具（附加，不属于冻结清单）
# --------------------------------------------------------------------------- #
def reprojection_errors(env, points, camera: str = "top",
                        z: Optional[float] = None) -> np.ndarray:
    """给定 world 点列表，返回每个点的往返误差（米，float64 数组）。

    流程：world -> project -> 像素 -> pixel_to_plane(平面 z) -> 世界点，
    误差 = 原点到反投影点的欧氏距离。
      * z=None：用每个点自身的 z 作为平面（要求点本来就在该水平面上）；
      * z=float：所有点统一用该平面（例如都按 TABLE_HEIGHT）。
    投影失败（相机后方 / 像素越界 / 输入非法）或求交失败时该点记 np.nan。
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros(0, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points 需要 (N,3)，收到 shape=%r" % (pts.shape,))
    cam = CameraModel(env, camera)
    out = np.full(pts.shape[0], np.nan, dtype=np.float64)
    for i, point in enumerate(pts):
        pixel = cam.project(point)
        if pixel is None:
            continue
        plane_z = float(point[2]) if z is None else float(z)
        back = cam.pixel_to_plane(pixel[0], pixel[1], plane_z)
        if back is None:
            continue
        out[i] = float(np.linalg.norm(back - point))
    return out


def reprojection_summary(env, points, camera: str = "top",
                         z: Optional[float] = None) -> dict:
    """reprojection_errors 的汇总（verifier 用）：均值/最大误差(米)与失败数。"""
    errors = reprojection_errors(env, points, camera=camera, z=z)
    ok = errors[np.isfinite(errors)]
    return {
        "camera": str(camera),
        "n": int(errors.size),
        "n_failed": int(errors.size - ok.size),
        "mean_m": float(ok.mean()) if ok.size else float("nan"),
        "max_m": float(ok.max()) if ok.size else float("nan"),
        "errors_m": [float(v) for v in errors],
    }
