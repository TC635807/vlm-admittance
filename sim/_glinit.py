"""sim/_glinit.py —— 必须在 import mujoco 之前导入，用于选定离屏渲染后端。

本机实测：EGL 不可用，只能用 glfw（需要 DISPLAY）。这里用 setdefault，
所以外部显式设置的 MUJOCO_GL / DISPLAY 优先级更高，不会被覆盖。

为什么要单独一个模块：MuJoCo 的 GL 后端在 'import mujoco' 时选定，
之后再改环境变量无效，因此所有 sim 子模块都必须在 import mujoco 之前
先 import 本模块。
"""
from __future__ import annotations

import os

#: 本机实测可用的离屏渲染后端（EGL 在本机会报错）
os.environ.setdefault("MUJOCO_GL", "glfw")
#: 本机 DISPLAY=:0 可用；仅在未设置时兜底
os.environ.setdefault("DISPLAY", ":0")
