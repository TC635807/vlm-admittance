"""sim/scene.py —— MuJoCo 场景构建（UR5e + Robotiq 2F85 + 桌面 + 物体 + 3 相机）。

原理
----
1. MjSpec.from_file(UR5E_XML) 读入 UR5e（Menagerie）；MjSpec.from_file(GRIPPER_XML)
   读入 Robotiq 2F85，用 arm.attach(grip, prefix="g_", site=arm.site("attachment_site"))
   挂到法兰上。编译结果 nq=nv=14、nu=7、neq=3；TCP site = "g_pinch"，
   夹爪执行器 = "g_fingers_actuator"（ctrl 0..255）。
2. 地面/桌子/物体/相机/灯光**直接用 MjSpec 的 Python API 加到 arm spec 上**
   （worldbody.add_geom / add_body / add_light / add_camera，spec.add_texture /
   add_material）。这样所有名字保持干净（"box_red" / "table_top" / "overview" ...）。
3. wrist 相机在 attach 之后挂到已合并的夹爪 body 上（arm.body("g_base")）。

踩坑点（均在本机实测）
----------------------
* ur5e.xml 的 base body 在世界原点，必须抬到 core.config.ROBOT_BASE_POS=(0,0,0.40)
  （= 桌面上表面），否则 home 位形下 TCP 在 z≈0.33，会穿进桌面。
* 不要用 MjSpec.from_string + attach 来加 props：attach 之后子 spec 的元素名会被加上
  路径前缀（实测 mj_id2name 返回 "/box_red"、"tray_free" -> "/tray_free"），
  于是 mj_name2id(m, mjOBJ_BODY, "box_red") 返回 -1，破坏与 planning/skills 的契约。
  用 API 直接加就没有前缀。
* attach 时父子 spec 的 option.impratio / option.cone 不一致会打 Attach conflict
  警告；父子都设 impratio=10 / cone=elliptic 即可消除。
* MjsBody.add_camera / add_geom 的 pos/quat/size 必须是 float 序列，传字符串会 TypeError。
* MuJoCo 相机沿自身局部 **-z** 轴看；_look_at_quat 按此约定构造朝向。
* 物体全部使用 free joint（joint 名 "<id>_free"），body 名即 obj_id；
  托盘 is_container=True / graspable=False。
* 物体中心 z = TABLE_HEIGHT + sz，正好落在桌面上表面，静止时与 table_top 形成
  永久接触（这是正常接触，planning 只统计含机器人 geom 的接触对即可）。
* 日常物品场景（INTERFACES.md §5.5）另有一份 DAILY_OBJECTS / DAILY_OBJECT_IDS：
  水瓶(cylinder, cyan) 与眼镜盒(box, black)，与默认场景**互不影响**，用
  build_spec(DAILY_OBJECTS) 或 MuJoCoArmEnv(objects=DAILY_OBJECTS) 选用。
  COLOR_RGBA 为此新增 cyan / black 两色；normalize_objects 支持 category
  "bottle"->cylinder、"case"->box 的缺省 shape 推断（显式 shape 优先）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from . import _glinit  # noqa: F401  必须在 import mujoco 之前

import mujoco
import numpy as np

from core.config import (
    GRIPPER_XML,
    ROBOT_BASE_POS,
    SIM_DT,
    TABLE_HEIGHT,
    UR5E_XML,
)

# --------------------------------------------------------------------------- #
# 场景常量（planning / skills 会引用，改名会破坏集成）
# --------------------------------------------------------------------------- #
#: 机器人底座世界坐标（= 桌面上表面）
ROBOT_BASE = (float(ROBOT_BASE_POS[0]), float(ROBOT_BASE_POS[1]), float(ROBOT_BASE_POS[2]))
#: 桌面上表面高度（= core.config.TABLE_HEIGHT）
TABLE_TOP_Z: float = float(TABLE_HEIGHT)
#: 桌面：固定 body "table"，上表面 z = 0.40
TABLE_BODY: str = "table"
TABLE_GEOM: str = "table_top"
TABLE_CENTER: tuple = (0.10, 0.0, TABLE_TOP_Z - 0.02)
TABLE_HALF: tuple = (0.60, 0.60, 0.02)
#: 地面（plane，固定，属 world body）
FLOOR_GEOM: str = "floor"
#: with_barrier=True 时的竖直挡板（固定 body + 固定 geom，无 free joint）
BARRIER_BODY: str = "barrier"
BARRIER_GEOM: str = "barrier_geom"
BARRIER_POS: tuple = (0.25, 0.0, TABLE_TOP_Z + 0.12)
BARRIER_HALF: tuple = (0.012, 0.16, 0.12)

#: 相机名（契约固定）
CAMERA_OVERVIEW: str = "overview"
CAMERA_WRIST: str = "wrist"
CAMERA_TOP: str = "top"

#: 物体 id 的稳定顺序（list_objects() 按此顺序返回）
OBJECT_IDS: tuple = ("box_red", "box_blue", "box_green", "can_yellow", "tray")

#: 颜色 -> rgba（0..1）
COLOR_RGBA: dict = {
    "red": (0.85, 0.15, 0.12, 1.0),
    "blue": (0.15, 0.35, 0.85, 1.0),
    "green": (0.15, 0.70, 0.25, 1.0),
    "yellow": (0.90, 0.80, 0.15, 1.0),
    "gray": (0.55, 0.57, 0.60, 1.0),
    # §5.5 日常物品：水瓶（青）/ 眼镜盒（黑）
    "cyan": (0.25, 0.68, 0.88, 1.0),
    "black": (0.10, 0.10, 0.12, 1.0),
}

#: category -> 缺省 shape 推断（§5.5：bottle 用圆柱、case 用盒；显式 shape 优先）
CATEGORY_SHAPE: dict = {
    "box": "box", "cube": "box", "case": "box",
    "cylinder": "cylinder", "can": "cylinder", "bottle": "cylinder",
    "tray": "tray",
}

#: 默认物体布局。
#: size 是**半尺寸** (sx, sy, sz)，与 core.types.SceneObject.size 一致；
#: 圆柱 can_yellow 的 (0.015, 0.015, 0.035) 等价于“半径 0.015、半高 0.035”。
#: 摆放约定：三个方块 x≈0.35~0.45、y∈[-0.15,0.15]，彼此间距 >= 0.10 m；
#: 托盘靠 +y 一侧（y≈0.30），圆柱居中；物体中心 z = TABLE_HEIGHT + sz（正好落在桌面上）。
DEFAULT_OBJECTS: list = [
    {
        "id": "box_red", "category": "box", "color": "red", "shape": "box",
        "size": (0.02, 0.02, 0.02), "pos": (0.35, -0.14, TABLE_TOP_Z + 0.02),
        "graspable": True, "is_container": False,
    },
    {
        "id": "box_blue", "category": "box", "color": "blue", "shape": "box",
        "size": (0.02, 0.02, 0.02), "pos": (0.35, 0.14, TABLE_TOP_Z + 0.02),
        "graspable": True, "is_container": False,
    },
    {
        "id": "box_green", "category": "box", "color": "green", "shape": "box",
        "size": (0.02, 0.02, 0.02), "pos": (0.45, -0.05, TABLE_TOP_Z + 0.02),
        "graspable": True, "is_container": False,
    },
    {
        "id": "can_yellow", "category": "cylinder", "color": "yellow", "shape": "cylinder",
        "size": (0.015, 0.015, 0.035), "pos": (0.40, 0.02, TABLE_TOP_Z + 0.035),
        "graspable": True, "is_container": False,
    },
    {
        "id": "tray", "category": "tray", "color": "gray", "shape": "tray",
        "size": (0.09, 0.07, 0.012), "pos": (0.40, 0.30, TABLE_TOP_Z + 0.012),
        "graspable": False, "is_container": True,
    },
]

#: 托盘壁厚半值（几何细节）
TRAY_WALL_HALF: float = 0.004

#: 日常物品场景（INTERFACES.md §5.5）。**不改** DEFAULT_OBJECTS / OBJECT_IDS 与默认布局，
#: 这是一份独立布局，用 build_spec(DAILY_OBJECTS) / MuJoCoArmEnv(objects=DAILY_OBJECTS) 选用。
#: size 同样是半尺寸；圆柱的 (r, r, hh) 等价于半径 r、半高 hh。
#: 布局：方块沿用默认（x≈0.35/0.45）；水瓶 x≈0.45 y≈-0.18；眼镜盒 x≈0.35 y≈0.20；
#: 托盘沿用默认 (0.40, 0.30)。全部物体中心两两间距 >= 90 mm，水平投影互不重叠
#: （实测最小中心距 107.7 mm = box_red<->bottle_water；眼镜盒与托盘边缘最小间隙 24.0 mm，
#: 眼镜盒 y 由 §5.5 初值 0.20 调整为 0.18，Lead 2026-10-12 裁决，中心距 130.0 mm 仍 >= 90 mm）。
#: 物体中心 z = TABLE_HEIGHT + sz，正好稳落在桌面上表面。
DAILY_OBJECTS: list = [
    {
        "id": "box_red", "category": "box", "color": "red", "shape": "box",
        "size": (0.02, 0.02, 0.02), "pos": (0.35, -0.14, TABLE_TOP_Z + 0.02),
        "graspable": True, "is_container": False,
    },
    {
        "id": "box_green", "category": "box", "color": "green", "shape": "box",
        "size": (0.02, 0.02, 0.02), "pos": (0.45, -0.05, TABLE_TOP_Z + 0.02),
        "graspable": True, "is_container": False,
    },
    {
        # 水瓶：半径 32 mm、高 200 mm（半尺寸 (0.032, 0.032, 0.100)）
        "id": "bottle_water", "category": "bottle", "color": "cyan", "shape": "cylinder",
        "size": (0.032, 0.032, 0.100), "pos": (0.45, -0.18, TABLE_TOP_Z + 0.100),
        "graspable": True, "is_container": False,
    },
    {
        # 眼镜盒：136 x 52 x 24 mm（半尺寸 (0.068, 0.026, 0.012)）
        "id": "case_black", "category": "case", "color": "black", "shape": "box",
        "size": (0.068, 0.026, 0.012), "pos": (0.35, 0.18, TABLE_TOP_Z + 0.012),
        "graspable": True, "is_container": False,
    },
    {
        "id": "tray", "category": "tray", "color": "gray", "shape": "tray",
        "size": (0.09, 0.07, 0.012), "pos": (0.40, 0.30, TABLE_TOP_Z + 0.012),
        "graspable": False, "is_container": True,
    },
]

#: 日常场景的物体 id 稳定顺序（list_objects() 按此顺序返回）
DAILY_OBJECT_IDS: tuple = ("box_red", "box_green", "bottle_water", "case_black", "tray")


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def object_joint_name(obj_id: str) -> str:
    """物体 free joint 名。"""
    return obj_id + "_free"


def object_geom_name(obj_id: str) -> str:
    """物体主碰撞 geom 名（托盘的主 geom 是 "<id>_bottom"）。"""
    return obj_id + "_geom"


def object_rest_z(obj: dict) -> float:
    """物体静止在桌面上时 body 原点的 z（body 原点 = 几何中心）。"""
    return float(TABLE_TOP_Z + obj["size"][2])


def normalize_objects(objects: Optional[Iterable[dict]] = None) -> list:
    """把外部传入的物体描述规范化成内部统一格式（返回新 list，浅拷贝）。

    支持：
      * objects=None -> 复制 DEFAULT_OBJECTS；
      * size 给 2 元组时按“圆柱 (半径, 半高)”解释，自动补成 3 元组 (r, r, hh)；
      * 缺省 fill：shape 由 category 推断；graspable / is_container 由 category 推断；
        pos 只给 (x, y) 时 z 按桌面静止高度补齐。
    """
    src = DEFAULT_OBJECTS if objects is None else list(objects)
    out: list = []
    for raw in src:
        o = dict(raw)
        oid = str(o["id"])
        category = str(o.get("category", "box"))
        # shape 显式给出时优先；缺省按 category 推断（含 bottle->cylinder / case->box）
        shape = str(o.get("shape") or CATEGORY_SHAPE.get(category, "box"))
        size = tuple(float(v) for v in o["size"])
        if len(size) == 2:                      # 圆柱 (半径, 半高)
            size = (size[0], size[0], size[1])
        if len(size) != 3:
            raise ValueError("物体 %s 的 size 必须是 2 或 3 元组，收到 %r" % (oid, o["size"]))
        pos = tuple(float(v) for v in o["pos"])
        if len(pos) == 2:
            pos = (pos[0], pos[1], float(TABLE_TOP_Z + size[2]))
        if len(pos) != 3:
            raise ValueError("物体 %s 的 pos 必须是 2 或 3 元组，收到 %r" % (oid, o["pos"]))
        o.update(
            id=oid,
            category=category,
            color=str(o.get("color", "gray")),
            shape=shape,
            size=size,
            pos=pos,
            graspable=bool(o.get("graspable", category != "tray")),
            is_container=bool(o.get("is_container", category == "tray")),
        )
        out.append(o)
    ids = [o["id"] for o in out]
    if len(set(ids)) != len(ids):
        raise ValueError("物体 id 重复：%r" % (ids,))
    return out


def _look_at_quat(pos: Sequence[float], target: Sequence[float],
                  up: Sequence[float] = (0.0, 0.0, 1.0)) -> list:
    """构造“位于 pos、看向 target”的相机四元数（wxyz）。

    MuJoCo 相机沿自身局部 -z 轴看：令 z_cam = normalize(pos - target)，
    再取 x_cam = normalize(up x z_cam)、y_cam = z_cam x x_cam 组成右手系。
    up 与视线平行时（俯视）退化为参考向量 (0,-1,0)，避免叉积为零。
    """
    p = np.asarray(pos, dtype=float)
    t = np.asarray(target, dtype=float)
    f = t - p
    n = float(np.linalg.norm(f))
    if n < 1e-9:
        raise ValueError("相机 pos 与 target 重合，无法确定朝向")
    f /= n
    z_cam = -f
    u = np.asarray(up, dtype=float)
    if abs(float(np.dot(u, z_cam))) > 0.999:
        u = np.array([0.0, -1.0, 0.0])
    x_cam = np.cross(u, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    rot = np.column_stack([x_cam, y_cam, z_cam])           # 列 = 相机轴在世界系
    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot.reshape(9)))
    return [float(v) for v in quat]


def _rgba(o: dict) -> list:
    return list(COLOR_RGBA.get(o["color"], (0.6, 0.6, 0.6, 1.0)))


def _add_object(spec, o: dict) -> None:
    """向 spec.worldbody 添加一个带 free joint 的物体 body。"""
    oid = o["id"]
    sx, sy, sz = o["size"]
    body = spec.worldbody.add_body(name=oid, pos=list(o["pos"]))
    body.add_freejoint(name=object_joint_name(oid))
    rgba = _rgba(o)
    if o["shape"] == "box":
        body.add_geom(name=object_geom_name(oid), type=int(mujoco.mjtGeom.mjGEOM_BOX),
                      size=[sx, sy, sz], rgba=rgba, condim=4)
    elif o["shape"] == "cylinder":
        # MuJoCo cylinder size = (半径, 半高)
        body.add_geom(name=object_geom_name(oid), type=int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                      size=[sx, sz], rgba=rgba, condim=4)
    elif o["shape"] == "tray":
        th = TRAY_WALL_HALF
        bottom_half_z = 0.0055
        wall_half_z = sz - 0.001
        wall_center_z = sz - wall_half_z
        box = int(mujoco.mjtGeom.mjGEOM_BOX)
        body.add_geom(name=oid + "_bottom", type=box,
                      size=[sx, sy, bottom_half_z],
                      pos=[0.0, 0.0, -sz + bottom_half_z], rgba=rgba, condim=4)
        walls = [
            ("wall_xp", (th, sy - th, wall_half_z), (sx - th, 0.0, wall_center_z)),
            ("wall_xn", (th, sy - th, wall_half_z), (-sx + th, 0.0, wall_center_z)),
            ("wall_yp", (sx - 2 * th, th, wall_half_z), (0.0, sy - th, wall_center_z)),
            ("wall_yn", (sx - 2 * th, th, wall_half_z), (0.0, -sy + th, wall_center_z)),
        ]
        for part, size, off in walls:
            body.add_geom(name="%s_%s" % (oid, part), type=box,
                          size=list(size), pos=list(off), rgba=rgba, condim=4)
    else:
        raise ValueError("物体 %s 的 shape 不支持：%r" % (oid, o["shape"]))


def _add_props(spec, objects: list, with_barrier: bool) -> None:
    """用 MjSpec API 把地面/桌子/物体/相机/灯光直接加到 spec 上（名字无前缀）。"""
    # --- 纹理 / 材质（天空盒 + 棋盘地面 + 桌面 + 挡板） ---
    spec.add_texture(
        name="sky_tex", type=int(mujoco.mjtTexture.mjTEXTURE_SKYBOX),
        builtin=int(mujoco.mjtBuiltin.mjBUILTIN_GRADIENT),
        rgb1=[0.25, 0.45, 0.65], rgb2=[0.05, 0.07, 0.11], width=256, height=1536,
    )
    spec.add_texture(
        name="checker_tex", type=int(mujoco.mjtTexture.mjTEXTURE_2D),
        builtin=int(mujoco.mjtBuiltin.mjBUILTIN_CHECKER),
        mark=int(mujoco.mjtMark.mjMARK_EDGE),
        rgb1=[0.30, 0.32, 0.35], rgb2=[0.16, 0.18, 0.20], markrgb=[0.70, 0.70, 0.70],
        width=300, height=300,
    )
    spec.add_material(name="checker_mat", textures=["checker_tex"], texuniform=True,
                      texrepeat=[6.0, 6.0], reflectance=0.1)
    spec.add_material(name="table_mat", rgba=[0.62, 0.48, 0.32, 1.0], reflectance=0.05)
    spec.add_material(name="barrier_mat", rgba=[0.75, 0.75, 0.78, 1.0], reflectance=0.1)

    wb = spec.worldbody
    # --- 灯光 ---
    # 注意：MJCF 的 directional="true" 对应 light_type=mjLIGHT_DIRECTIONAL，
    # add_light 没有 directional 关键字，要用 type 参数。
    light_dir = int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    wb.add_light(name="key_light", type=light_dir, pos=[0.9, -1.0, 2.0], dir=[-0.35, 0.45, -1.0],
                 diffuse=[0.8, 0.8, 0.8], specular=[0.2, 0.2, 0.2])
    wb.add_light(name="fill_light", type=light_dir, pos=[-1.2, 0.8, 1.4], dir=[0.6, -0.4, -1.0],
                 diffuse=[0.35, 0.35, 0.35], specular=[0.0, 0.0, 0.0])
    # --- 地面 ---
    wb.add_geom(name=FLOOR_GEOM, type=int(mujoco.mjtGeom.mjGEOM_PLANE),
                size=[0.0, 0.0, 0.05], pos=[0.0, 0.0, 0.0], material="checker_mat")
    # --- 桌子：桌面 + 4 腿（固定 body "table"） ---
    box = int(mujoco.mjtGeom.mjGEOM_BOX)
    table = wb.add_body(name=TABLE_BODY, pos=[0.0, 0.0, 0.0])
    table.add_geom(name=TABLE_GEOM, type=box, size=list(TABLE_HALF),
                   pos=list(TABLE_CENTER), material="table_mat")
    tx, ty, tz = TABLE_CENTER
    hx, hy, hz = TABLE_HALF
    leg_hz = (TABLE_TOP_Z - 2 * hz) / 2.0
    leg_xy = [
        (tx - hx + 0.05, ty - hy + 0.05), (tx - hx + 0.05, ty + hy - 0.05),
        (tx + hx - 0.05, ty - hy + 0.05), (tx + hx - 0.05, ty + hy - 0.05),
    ]
    for i, (lx, ly) in enumerate(leg_xy):
        table.add_geom(name="table_leg_%d" % i, type=box,
                       size=[0.03, 0.03, leg_hz], pos=[lx, ly, leg_hz],
                       material="table_mat")
    # --- 可选挡板 ---
    if with_barrier:
        barrier = wb.add_body(name=BARRIER_BODY, pos=list(BARRIER_POS))
        barrier.add_geom(name=BARRIER_GEOM, type=box, size=list(BARRIER_HALF),
                         material="barrier_mat")
    # --- 物体 ---
    for o in objects:
        _add_object(spec, o)
    # --- 固定相机（wrist 在 attach 后挂到夹爪 body 上） ---
    overview_pos = (1.05, -0.95, 1.15)
    overview_target = (0.22, 0.05, 0.55)
    # top 相机：俯视工作台（z=1.32 时刚好框住全部物体 + 托盘，太远物体太小）
    top_pos = (0.36, 0.05, 1.32)
    top_target = (0.36, 0.05, TABLE_TOP_Z)
    wb.add_camera(name=CAMERA_OVERVIEW, pos=list(overview_pos),
                  quat=_look_at_quat(overview_pos, overview_target))
    wb.add_camera(name=CAMERA_TOP, pos=list(top_pos),
                  quat=_look_at_quat(top_pos, top_target))


# --------------------------------------------------------------------------- #
# 对外接口
# --------------------------------------------------------------------------- #
def build_spec(objects: Optional[Iterable[dict]] = None,
               with_barrier: bool = False) -> mujoco.MjSpec:
    """组装完整场景 spec：UR5e + Robotiq 2F85 + 桌面/地面/物体/灯光/相机。

    objects : 物体描述列表（键 id/category/color/size/pos/graspable/is_container）；
              传 None 用 DEFAULT_OBJECTS。
    with_barrier : True 时在 (0.25, 0, 0.52) 加竖直挡板
                   （半尺寸 0.012 x 0.16 x 0.12，z∈[0.40,0.64]），供 planning 绕障测试。

    返回尚未编译的 mujoco.MjSpec；编译后 nq=nv=14+7*len(objects)、nu=7、neq=3。
    """
    objs = normalize_objects(objects)

    arm = mujoco.MjSpec.from_file(str(UR5E_XML))
    # 父子 option 一致 -> 消除 Attach conflict 警告（实测有效）
    arm.option.impratio = 10.0
    arm.option.cone = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    arm.option.timestep = float(SIM_DT)
    # 底座抬到桌面上表面（否则 home 位形 TCP 穿桌）
    arm.body("base").pos = list(ROBOT_BASE)

    grip = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    arm.attach(grip, prefix="g_", site=arm.site("attachment_site"))

    _add_props(arm, objs, bool(with_barrier))

    # 离屏渲染 framebuffer：本机默认高度只有 480，渲染 512x512 会报
    # "Image height 512 > framebuffer height 480"，必须显式放大（实测）。
    arm.visual.global_.offwidth = 1024
    arm.visual.global_.offheight = 1024

    # 腕部相机：attach 之后加到合并后的夹爪 body 上。
    # g_base 的 +z 指向两指 / TCP，quat=(0,1,0,0) 让相机绕 x 轴转 180 度、
    # 沿 -z_cam = +z_gbase 看向指尖。
    # 实测：pos.z <= 0.06 时相机埋在夹爪 base 网格内部（depth 中位数只有 2.6cm），
    # 0.070 时刚好在两指之间、可见两指与桌面；fovy 必须放大到 ~110 度，
    # 否则指垫（横向 ±3cm）落在 45 度视锥外，画面里看不到夹爪。
    arm.body("g_base").add_camera(
        name=CAMERA_WRIST, pos=[0.0, 0.0, 0.070], quat=[0.0, 1.0, 0.0, 0.0],
        fovy=110.0,
    )
    return arm


def write_scene_xml(path, objects: Optional[Iterable[dict]] = None,
                    with_barrier: bool = False) -> str:
    """把 build_spec 的结果导出为 MJCF 文本（调试用），返回写入的绝对路径。"""
    spec = build_spec(objects, with_barrier)
    xml = spec.to_xml()
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(xml, encoding="utf-8")
    return str(p)
