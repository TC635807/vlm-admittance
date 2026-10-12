"""sim 包：MuJoCo 场景构建（scene）/ 离屏渲染（render）/ 环境封装（env）。

这里不在包导入时 import mujoco 子模块，避免在未设 MUJOCO_GL 的情况下过早
初始化 GL 后端；常用类型通过 PEP 562 的模块级 __getattr__ 懒加载，
所以 'from sim import MuJoCoArmEnv' 与 'from sim.env import MuJoCoArmEnv' 等价。
"""
from __future__ import annotations

__all__ = ["scene", "render", "env", "smoke", "MuJoCoArmEnv", "SceneRenderer"]


def __getattr__(name):
    if name == "MuJoCoArmEnv":
        from .env import MuJoCoArmEnv
        return MuJoCoArmEnv
    if name == "SceneRenderer":
        from .render import SceneRenderer
        return SceneRenderer
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
