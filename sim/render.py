"""sim/render.py —— 离屏渲染封装（懒加载 mujoco.Renderer）。

原理
----
mujoco.Renderer(model, height, width) 内部持有 GL 上下文与 Framebuffer，创建开销大
（本机 glfw 后端首次创建约几百毫秒），因此这里**懒加载**：第一次调用 rgb()/depth()
时才创建，之后复用；close() 释放 GL 资源。

踩坑点（实测）
--------------
* 必须在 import mujoco 之前设置 MUJOCO_GL=glfw（见 sim/_glinit.py）；本机 EGL 不可用。
* depth 渲染要先 enable_depth_rendering()，取完立刻 disable_depth_rendering()，
  否则后续 rgb() 会拿到 1 通道浮点深度图。
* Renderer.render() 返回内部缓冲的**引用**，再次 render 会覆盖它；本模块统一 .copy()
  后返回，调用方可放心持有。
* 深度单位是米；天空 / 超出远平面处为远平面值（MuJoCo 默认 far=100）。
* Renderer 尺寸受 GL 限制，宽高必须为正；模型必须在同进程先编译完成。
* **每帧尺寸可覆盖**：rgb()/depth() 接受可选 width/height（默认用构造尺寸=512）。
  每个 (w,h) 各自缓存一个 mujoco.Renderer（创建一次约百毫秒），切换尺寸不会反复重建；
  **默认行为（512）完全不变**（sim/test_env.py::test_render_shapes_and_depth 依赖它）。
* 请求尺寸大于 model.visual.global_ 的离屏缓冲时，本类会**按需抬高** offwidth/offheight
  （只影响离屏渲染，不动物理），否则 MuJoCo 会报
  "Image height H > framebuffer height F"（本机实测）。
* close() 关闭全部缓存尺寸的 renderer。
"""
from __future__ import annotations

from typing import Optional

from . import _glinit  # noqa: F401  必须在 import mujoco 之前

import mujoco
import numpy as np


class SceneRenderer:
    """懒加载的场景渲染器：RGB (uint8 HxWx3) 与深度 (float32 HxW，米)。"""

    def __init__(self, model, width: int = 512, height: int = 512,
                 camera_names: Optional[tuple] = None):
        self.model = model
        self.width = int(width)
        self.height = int(height)
        self._camera_names = tuple(camera_names) if camera_names else None
        #: (w, h) -> mujoco.Renderer 缓存；self._renderer 只指向"最近使用"的那个
        self._renderers: dict = {}
        self._renderer = None

    # -- 内部 --------------------------------------------------------------- #
    def _ensure(self, width: Optional[int] = None, height: Optional[int] = None):
        """取出（必要时创建）指定尺寸的 Renderer；缺省用构造尺寸。"""
        w = self.width if width is None else int(width)
        h = self.height if height is None else int(height)
        if w <= 0 or h <= 0:
            raise ValueError("渲染尺寸必须为正，收到 %r" % ((w, h),))
        # 离屏 framebuffer 必须 >= 请求尺寸，否则 MuJoCo 直接报错；按需抬高。
        # 注意：编译后的 MjModel 字段名是 vis（不是 XML 里的 visual），实测 3.12。
        vis = getattr(self.model, "vis", None)
        if vis is not None:
            g = vis.global_
            if w > int(g.offwidth) or h > int(g.offheight):
                g.offwidth = max(int(g.offwidth), w)
                g.offheight = max(int(g.offheight), h)
        key = (w, h)
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.model, height=h, width=w)
            self._renderers[key] = r
        self._renderer = r
        return r

    def _check_camera(self, camera: str) -> None:
        if not camera:
            raise ValueError("camera 不能为空")
        if self._camera_names is not None and camera not in self._camera_names:
            raise ValueError("未知相机 %r，可用：%r" % (camera, self._camera_names))
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            raise ValueError("模型里不存在相机 %r" % (camera,))

    # -- 对外 --------------------------------------------------------------- #
    def rgb(self, data, camera: str,
            width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        """渲染彩色图，返回 (H, W, 3) uint8。

        width/height 缺省 = 构造尺寸（默认 512x512），可逐次覆盖（如 1024 提升像素精度）。
        """
        self._check_camera(camera)
        r = self._ensure(width, height)
        r.update_scene(data, camera=camera)
        return np.asarray(r.render()).copy()

    def depth(self, data, camera: str,
              width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        """渲染深度图，返回 (H, W) float32，单位米（天空≈远平面 far）。

        width/height 语义同 rgb()。
        """
        self._check_camera(camera)
        r = self._ensure(width, height)
        r.enable_depth_rendering()
        try:
            r.update_scene(data, camera=camera)
            out = np.asarray(r.render()).copy()
        finally:
            r.disable_depth_rendering()
        return out.astype(np.float32, copy=False)

    def close(self) -> None:
        """释放全部尺寸的 GL 资源；之后可再次懒加载。"""
        for r in getattr(self, "_renderers", {}).values():
            try:
                r.close()
            except Exception:                              # pragma: no cover - 已释放
                pass
        self._renderers = {}
        self._renderer = None

    def __enter__(self) -> "SceneRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
