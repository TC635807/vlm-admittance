"""agent.schema —— 工具调用 schema、系统提示词、解析与校验。

职责边界（只做决策层的「语法/语义安检」，不执行任何技能）：
- TOOLS 是 OpenAI function-calling 格式的工具表，参数与 INTERFACES.md §2.5 严格对齐；
- system_prompt() 给 VLM 讲清机器人、坐标系、导纳语义与失败重试策略；
- parse_skill_call() 把五花八门的模型输出（tool_calls / 纯文本内嵌 JSON / 代码块）
  统一成 core.types.SkillCall；非法输入一律 raise ValueError；
- validate_skill_call() 做白名单 + 类型/范围 + 场景一致性检查（物体是否存在、
  container 是否真的是容器），返回 (ok, msg) 而不是抛异常——决策环要把失败回灌给 VLM。

踩坑记录：
1. 模型的 arguments 在不同端点里有三种形态：JSON 字符串、已解析 dict、甚至是空串。
   必须三者都兼容，空串按 {} 处理。
2. 未知参数不要直接报错（模型经常多塞 helpful 字段），这里「白名单过滤」并写进
   reasoning，既不污染技能执行器，也保留可观测性。
3. 校验顺序：先白名单，再类型/范围，最后场景一致性；错误信息要具体到参数名，
   这样失败观察回灌给 VLM 后它才知道怎么改。
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

from core.config import DEFAULT_COMPLIANCE, SOFT_COMPLIANCE, STIFF_COMPLIANCE
from core.types import ComplianceParams, SceneState, SkillCall

#: 全部技能（与 INTERFACES.md §2.5 一致；v2 §5.3 追加 3 个控制面工具）
SKILL_NAMES: tuple[str, ...] = (
    "home", "move_to", "pick", "place", "push",
    "open_gripper", "close_gripper", "set_compliance", "stop", "ask_user",
    # ---- v2 §5.3：子目标闭环用的控制面工具 ----
    # plan_subgoals 只在规划阶段（taskplan.plan_subgoals）直接解析，执行循环里出现一律 invalid_call；
    # locate / subgoal_done 由 AgentLoop 特判，不交给 skills 执行器。
    "plan_subgoals", "locate", "subgoal_done", "verify_subgoal",
)

#: pick 的抓取位姿档位（INTERFACES.md §5.3）：auto=按类别自选，top=自上而下，side=侧向水平夹持
GRASP_PROFILES: tuple[str, ...] = ("auto", "top", "side")

#: 子目标类型（INTERFACES.md §5.2 的 SubGoal.goal_type）——决定独立复核走哪条几何判据
GOAL_TYPES: tuple[str, ...] = (
    "place_in_container", "place_on_top", "place_at", "pick", "free",
)

#: 目标颜色词表（与 sim/perception.py 的颜色分类保持一致；用于 locate/subgoal 的颜色提示）
OBJECT_COLORS: tuple[str, ...] = (
    "red", "blue", "green", "yellow", "gray", "black", "white", "cyan", "orange",
)

#: set_compliance 允许的档位
PROFILE_NAMES: tuple[str, ...] = ("stiff", "soft", "z_only")

#: place 的放置表面（与 container_id / on_top_of / anchor_id+relation 四选一）：
#: table=桌面/桌子/台面/地上，tray=托盘面
SURFACE_NAMES: tuple[str, ...] = ("table", "tray")

#: place 的横向关系（第 4 种目标 anchor_id+relation）：相对锚点的世界系水平方向，
#: right=+x, left=-x, front=+y, back=-y；beside=由技能自动选净距最大的方向。
RELATION_NAMES: tuple[str, ...] = ("right", "left", "front", "back", "beside")

#: target_pixel 的相机枚举（INTERFACES.md §4.3）；默认 top
PIXEL_CAMERAS: tuple[str, ...] = ("top", "overview", "wrist")
DEFAULT_PIXEL_CAMERA = "top"

#: §5.12-2：place 的 xy 落点是否算「落在容器内」的 xy 容差（米）
CONTAINER_XY_TOL_M: float = 0.010
#: §5.12-4：oracle 模式 target_id + target_xy 同时给时，xy 与真值的最大允许距离（米）
PICK_XY_TRUTH_TOL_M: float = 0.030

#: locate 的相机枚举（§5.10）：both = top+overview 双目三角定位（**默认**），
#: 单目时按桌面平面近似，高物体（水瓶等）会有最大约 7 cm 的视差误差。
LOCATE_CAMERAS: tuple[str, ...] = ("both", "top", "overview", "wrist")
DEFAULT_LOCATE_CAMERA = "both"

#: 每个技能允许的参数（白名单过滤用）
ALLOWED_ARGS: dict[str, tuple[str, ...]] = {
    "home": (),
    "move_to": ("pos", "quat", "compliance"),
    "pick": ("target_id", "target_xy", "target_pixel", "target_pixel_overview",
             "pixel_camera", "target_z_hint", "target_size",
             "approach_height", "close_ratio", "grasp_profile", "compliance"),
    "place": ("container_id", "surface", "on_top_of", "anchor_id", "relation", "distance",
              "target_pixel", "target_pixel_overview", "pixel_camera", "xy", "offset",
              "descend_until_contact", "compliance"),
    "push": ("target_id", "direction", "distance"),
    "open_gripper": (),
    "close_gripper": ("ratio",),
    "set_compliance": ("profile", "k_scale"),
    "stop": (),
    "ask_user": ("question",),
    # v2 §5.3 控制面工具
    "plan_subgoals": ("subgoals",),
    # pixel/bbox：模型自己量的像素与框（颜色分割失败时的兜底；bbox 取中心当像素）
    "locate": ("target", "camera", "color", "pixel", "bbox"),
    "subgoal_done": ("note",),
    # 复核专用（§5.11：结构化判断必须走 function calling，不能靠纯文本）
    "verify_subgoal": ("done", "confidence", "why"),
}

#: 代码围栏用（避免源码里出现反引号字面量，便于工具处理）
_FENCE = chr(96) * 3


# --------------------------------------------------------------------------- #
# 「纠正」记录（§5.12-5）
# --------------------------------------------------------------------------- #
#: validate_skill_call 把「已纠正」的参数临时挂在 SkillCall 上的属性名
CORRECTIONS_ATTR = "corrected_args"


def record_correction(call: Any, key: str, action: str, detail: str,
                      log: Optional[str] = None) -> None:
    """在 SkillCall 上记一条参数纠正（被丢 / 被接受 / 被替换）。

    契约冻结的 validate_skill_call 只能返回 (ok, msg)，没法带 metrics；
    所以纠正项先挂在 call 上，由 AgentLoop 用 take_corrections() 取走并写进
    SkillResult.metrics["corrected_args"]（§5.12-5），UI 据此计「已纠正」。
    """
    items = list(getattr(call, CORRECTIONS_ATTR, None) or [])
    entry: dict = {"key": str(key), "action": str(action), "detail": str(detail)}
    if log:
        entry["log"] = str(log)
    items.append(entry)
    setattr(call, CORRECTIONS_ATTR, items)


def take_corrections(call: Any) -> list[dict]:
    """取出（并清空）call 上累积的纠正项；没有则返回 []。"""
    items = list(getattr(call, CORRECTIONS_ATTR, None) or [])
    if items:
        try:
            object.__delattr__(call, CORRECTIONS_ATTR)
        except Exception:
            pass
    return items


def container_contains_xy(state: SceneState, obj: Any, xy: Any) -> bool:
    """xy 是否落在容器 obj 的水平范围内（AABB + 容差），§5.12-2 用。

    容器是托盘这类方盒时用 pose ± size（半尺寸）判 AABB；位姿/尺寸异常返回 False
    （保守：判不了就不放行，宁可拒绝也不把方块放到盘子外面）。
    """
    try:
        x, y = float(xy[0]), float(xy[1])
        cx, cy = float(obj.pose[0]), float(obj.pose[1])
        sx, sy = float(obj.size[0]), float(obj.size[1])
    except Exception:
        return False
    return (abs(x - cx) <= sx + CONTAINER_XY_TOL_M) and (abs(y - cy) <= sy + CONTAINER_XY_TOL_M)


# --------------------------------------------------------------------------- #
# 导纳档位
# --------------------------------------------------------------------------- #
def _copy_cp(cp: ComplianceParams) -> ComplianceParams:
    """复制 ComplianceParams，避免技能执行器改动 core.config 里的共享常量。"""
    return ComplianceParams(
        k=cp.k.copy(), d=cp.d.copy(), m=cp.m.copy(), axes=cp.axes.copy(),
        force_limit=float(cp.force_limit),
    )


def compliance_from_profile(profile: str, k_scale: Optional[float] = None) -> ComplianceParams:
    """档位 -> ComplianceParams：stiff=刚性 / soft=全向柔顺 / z_only=仅 z 轴导纳。

    语义约定（与 system_prompt 一致）：
    - stiff：axes 全 0，等同纯位置控制，用于精确快速移动；
    - soft：全 6 维导纳，k 小 -> 受力时明显退让，用于「轻一点/易碎」；
    - z_only：只有 z 平动参与导纳，用于沿重力方向的压入/贴合；
    k_scale 用 ComplianceParams.scaled 缩放刚度（d 按 sqrt 缩放以保持阻尼比）。
    """
    table = {"stiff": STIFF_COMPLIANCE, "soft": SOFT_COMPLIANCE, "z_only": DEFAULT_COMPLIANCE}
    if profile not in table:
        raise ValueError(f"未知导纳档位 {profile!r}，可选 {list(PROFILE_NAMES)}")
    cp = _copy_cp(table[profile])
    if k_scale is not None:
        cp = cp.scaled(float(k_scale))
    return cp


def _compliance_from_json(obj: dict) -> ComplianceParams:
    """把 compliance 对象 {"k":[6],"d":[6],"m":[6],"axes":[6],"force_limit":f} 转成类型。"""
    if not isinstance(obj, dict):
        raise ValueError(f"compliance 必须是对象，实际 {type(obj).__name__}")

    def _vec(key: str) -> Any:
        if key not in obj:
            raise ValueError(f"compliance 缺少字段 {key!r}（需要 6 维数组）")
        v = obj[key]
        if not isinstance(v, (list, tuple)) or len(v) != 6:
            raise ValueError(f"compliance.{key} 必须是长度 6 的数组")
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            raise ValueError(f"compliance.{key} 必须是数值数组")

    k, d, m, axes = _vec("k"), _vec("d"), _vec("m"), _vec("axes")
    for name, vec in (("k", k), ("d", d), ("m", m)):
        if any(x < 0 for x in vec):
            raise ValueError(f"compliance.{name} 不能为负")
    if any(x not in (0.0, 1.0) for x in axes):
        raise ValueError("compliance.axes 只允许 0.0/1.0（0=该轴不做导纳修正）")
    fl = float(obj.get("force_limit", 10.0))
    if fl <= 0:
        raise ValueError("compliance.force_limit 必须 > 0")
    import numpy as np

    return ComplianceParams(
        k=np.array(k), d=np.array(d), m=np.array(m), axes=np.array(axes), force_limit=fl,
    )


# --------------------------------------------------------------------------- #
# 工具 schema
# --------------------------------------------------------------------------- #
def _vec_schema(n: int, desc: str, nullable: bool = False) -> dict:
    s: dict[str, Any] = {
        "type": ["array", "null"] if nullable else "array",
        "items": {"type": "number"},
        "minItems": n,
        "maxItems": n,
        "description": desc,
    }
    return s


def _compliance_schema() -> dict:
    """compliance 对象 schema（每次返回新 dict，避免调用方互相污染）。"""
    return {
        "type": "object",
        "description": (
            "导纳参数。k/d/m/axes 均为 6 维数组，顺序 [x,y,z,rx,ry,rz]（世界系）；"
            "axes 用 1 启用 / 0 关闭该轴导纳；force_limit 是异常接触力阈值(N)。"
            "不确定时不要传，改用 set_compliance 档位。"
        ),
        "properties": {
            "k": _vec_schema(6, "刚度 (N/m 或 Nm/rad)，越大越硬"),
            "d": _vec_schema(6, "阻尼 (Ns/m)，抑制振荡"),
            "m": _vec_schema(6, "虚拟质量 (kg)，越大越迟钝"),
            "axes": _vec_schema(6, "每轴导纳开关，1=启用 0=关闭"),
            "force_limit": {"type": "number", "minimum": 0.1, "description": "异常接触力阈值(N)"},
        },
        "required": ["k", "d", "m", "axes"],
    }


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _subgoal_item_schema() -> dict:
    """plan_subgoals 里单个子目标的对象 schema（字段与 taskplan.SubGoal 对齐，§5.2）。"""
    color_prop = {
        "oneOf": [{"type": "string", "enum": list(OBJECT_COLORS)}, {"type": "null"}],
        "description": "颜色线索（perception 的颜色词表），不确定就传 null",
    }
    return {
        "type": "object",
        "properties": {
            "instruction": {"type": "string",
                            "description": "给机械臂的自然语言子任务，如「把红色方块放进托盘」"},
            "subject": {"type": "string",
                        "description": "被操作物体的描述性名称，如「红色方块」「水瓶」"},
            "goal_type": {
                "type": "string", "enum": list(GOAL_TYPES),
                "description": ("子目标类型：place_in_container=放进容器；place_on_top=叠到某物上；"
                                "place_at=放到指定位置；pick=只抓取；free=其它/说不清"),
            },
            "subject_color": dict(color_prop, description="被操作物体的颜色线索"),
            "container": {"type": ["string", "null"],
                          "description": "容器名称（place_in_container 用），如「托盘」"},
            "container_color": dict(color_prop, description="容器的颜色线索"),
            "anchor_color": dict(color_prop, description="叠放/横向锚点的颜色线索（place_on_top 用）"),
        },
        "required": ["instruction", "goal_type"],
    }


def _build_tools() -> list[dict]:
    """构造 TOOLS（函数只跑一次，模块导入时调用）。"""
    return [
        _tool(
            "home",
            "机械臂回到标称 HOME 位形并张开夹爪。任务开始、或需要重新观察全场时调用。",
            {}, [],
        ),
        _tool(
            "move_to",
            "把 TCP 直线移动到世界系目标位姿。pos 为 [x,y,z]（米）；quat 为 [w,x,y,z]，"
            "null 表示保持当前姿态；compliance 可为档位名 stiff/soft/z_only 或导纳参数对象。",
            {
                "pos": _vec_schema(3, "TCP 目标位置 [x,y,z]，米"),
                "quat": _vec_schema(4, "TCP 目标姿态 [w,x,y,z]，null=保持当前姿态", nullable=True),
                "compliance": {
                    "oneOf": [
                        {"type": "string", "enum": list(PROFILE_NAMES)},
                        _compliance_schema(),
                        {"type": "null"},
                    ],
                    "description": "可选：本次移动使用的导纳档位或参数对象",
                },
            },
            ["pos"],
        ),
        _tool(
            "pick",
            "抓取指定物体：先张开夹爪、对准物体上方，再合爪并抬升。"
            "指定目标三选一、必须且只能给一个：target_id（oracle 模式，物体 id）、"
            "target_xy=[x,y]（世界系桌面坐标）、target_pixel=[u,v]（无真值模式，物体中心像素）。"
            "target_id 必须是场景 objects 里 graspable=true 的物体 id。",
            {
                "target_id": {"type": "string", "description": "要抓取的物体 id，如 box_red；三选一（oracle 模式）"},
                "target_xy": _vec_schema(2, "世界系桌面目标 [x,y]（米）；与 target_id/target_pixel 三选一",
                                         nullable=True),
                "target_pixel": _vec_schema(2, "物体中心像素 [u,v]（原点左上角，u 右 v 下）；"
                                              "无真值模式用；与 target_id/target_xy 三选一", nullable=True),
                "target_pixel_overview": _vec_schema(
                    2, "同一物体在 overview 相机里的中心像素 [u,v]；与 target_pixel 成对给出时"
                       "走双目三角定位（§5.10）。高物体（水瓶/立盒）必须给，否则有最大 7 cm 视差",
                    nullable=True),
                "pixel_camera": {"oneOf": [{"type": "string", "enum": list(PIXEL_CAMERAS)},
                                           {"type": "null"}],
                                 "description": "target_pixel 所在相机，默认 top"},
                "target_z_hint": {"type": ["number", "null"],
                                  "description": "物体中心 z 估计（米），可选；缺省由技能按桌面推算"},
                "target_size": {
                    "type": ["array", "null"], "items": {"type": "number"},
                    "minItems": 2, "maxItems": 3,
                    "description": ("物体半尺寸估计 [sx,sy] 或 [sx,sy,sz]（米）；"
                                    "缺省由技能按类别先验。给对了闭合量会按真实尺寸算，抓空更少"),
                },
                "approach_height": {
                    "type": "number", "minimum": 0.01, "maximum": 0.5,
                    "description": "预抓取抬高距离(米)，默认 0.10",
                },
                "close_ratio": {
                    "type": "number", "minimum": 0.0, "maximum": 1.0,
                    "description": "夹爪闭合量：0=全闭 1=全开，默认 0.35（避免过压）",
                },
                "grasp_profile": {
                    "oneOf": [{"type": "string", "enum": list(GRASP_PROFILES)}, {"type": "null"}],
                    "description": ("抓取位姿档位：top=自上而下（**默认，所有物体都用它**，"
                                     "含圆柱/水瓶；2F85 张开 93 mm 足够夹住瓶身/罐体上半部）；"
                                     "side=从侧面水平夹（**只在任务明确要求侧抓/避开顶部、"
                                     "且物体周围有明显净空时才用**，否则接近时会推走物体）；"
                                     "auto=按类别自选（等价 top）。"),
                },
                "compliance": {"oneOf": [_compliance_schema(), {"type": "null"}],
                               "description": "抓取接触阶段的导纳参数对象"},
            },
            ["target_id"],
        ),
        _tool(
            "place",
            "把当前夹持物体放到目标位置：移动到目标上方再下放、松爪。"
            "放置目标四选一、必须且只能给一个："
            "container_id 给场景里 is_container=true 的容器 id（如 tray）；"
            "surface='table' 表示放到桌面（桌子/台面/地上），可选 xy 指定桌面落点；"
            "surface='tray' 表示放到最近的托盘面；"
            "on_top_of 给一个可抓物体 id 表示叠放在它上方（「蓝色方块放在红色方块上方」）；"
            "anchor_id + relation 表示放到某物体旁边的横向方位（relation=right/left/front/back/beside，"
            "可选 distance 中心距；「绿色方块放在蓝色方块右边」）。"
            "桌面不是场景物体、没有 id，不要用 container_id 表示桌面。",
            {
                "container_id": {
                    "type": ["string", "null"],
                    "description": "目标容器 id，如 tray；四选一（用其它三种时传 null 或省略）",
                },
                "surface": {
                    "oneOf": [{"type": "string", "enum": list(SURFACE_NAMES)}, {"type": "null"}],
                    "description": "放置表面：'table'=桌面/桌子/台面/地上；'tray'=托盘面；四选一",
                },
                "on_top_of": {
                    "type": ["string", "null"],
                    "description": "叠放锚点物体 id（必须 graspable=true，不能是托盘）；"
                                   "对应「X 上方 / X 上面 / 叠在 X 上」；四选一",
                },
                "anchor_id": {
                    "type": ["string", "null"],
                    "description": "横向方位锚点物体 id（必须 graspable=true）；与 relation 成对出现；四选一",
                },
                "target_pixel": _vec_schema(2, "落点像素 [u,v]（原点左上角，u 右 v 下）；"
                                              "可选，用来指明放置落点（无真值模式）", nullable=True),
                "target_pixel_overview": _vec_schema(
                    2, "同一个落点在 overview 相机里的像素 [u,v]；与 target_pixel 成对给出时走"
                       "双目三角定位（§5.10），可消除高物体/高落点的视差", nullable=True),
                "pixel_camera": {"oneOf": [{"type": "string", "enum": list(PIXEL_CAMERAS)},
                                           {"type": "null"}],
                                 "description": "target_pixel 所在相机，默认 top"},
                "relation": {
                    "oneOf": [{"type": "string", "enum": list(RELATION_NAMES)}, {"type": "null"}],
                    "description": "横向方位：right=+x / left=-x / front=+y / back=-y（世界系水平），"
                                   "beside=自动选净距最大方向；必须与 anchor_id 成对；四选一",
                },
                "distance": {
                    "type": ["number", "null"],
                    "description": "横向放置的可选中心距（米）；不填由技能按两者半宽自动算",
                },
                "xy": _vec_schema(2, "落点 [x,y]（世界系，米）；surface='table' 或 on_top_of 时可选，"
                                     "不填由技能自选空位", nullable=True),
                "offset": _vec_schema(3, "相对容器中心/上沿的放置偏移 [dx,dy,dz]，米，默认 [0,0,0.02]"),
                "descend_until_contact": {
                    "type": ["boolean", "null"],
                    "description": ("是否接触式落放（降到碰到为止再松爪）：true=开启；false=关闭；"
                                     "null/省略=自动（on_top_of 叠放与容器内落放自动开启）。"
                                     "叠放/放进容器时建议留空用自动。"),
                },
                "compliance": {"oneOf": [_compliance_schema(), {"type": "null"}],
                               "description": "放置接触阶段的导纳参数对象"},
            },
            ["container_id"],
        ),
        _tool(
            "push",
            "沿给定方向推动物体（非抓取操作）。direction 会被内部归一化，distance 单位为米。",
            {
                "target_id": {"type": "string", "description": "被推动的物体 id"},
                "direction": _vec_schema(3, "推动方向 [dx,dy,dz]（世界系，无需归一化）"),
                "distance": {"type": "number", "minimum": 0.001, "maximum": 0.5,
                             "description": "推动距离(米)"},
            },
            ["target_id", "direction", "distance"],
        ),
        _tool("open_gripper", "完全张开夹爪。抓取前或释放后调用。", {}, []),
        _tool(
            "close_gripper",
            "闭合夹爪。ratio 为闭合量归一化值，0=全闭、1=全开，默认 0.0。",
            {"ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                       "description": "0=全闭 1=全开，默认 0.0"}},
            [],
        ),
        _tool(
            "set_compliance",
            "切换导纳档位。stiff=刚性（精确/快速）；soft=全向柔顺（任务说「轻一点/小心/易碎」）；"
            "z_only=仅 z 轴柔顺（沿重力压入/贴合）。k_scale 可再整体缩放刚度（>0）。",
            {
                "profile": {"type": "string", "enum": list(PROFILE_NAMES),
                            "description": "导纳档位"},
                "k_scale": {"type": "number", "minimum": 0.01, "maximum": 100.0,
                            "description": "刚度整体缩放系数，可选"},
            },
            ["profile"],
        ),
        _tool("stop", "结束当前任务。任务完成、无法继续或连续失败时调用。", {}, []),
        _tool(
            "ask_user",
            "向人提问请求澄清（例如指令含糊、目标物体不存在）。question 为要问的问题。",
            {"question": {"type": "string", "description": "要询问用户的问题"}},
            ["question"],
        ),
        # ---- v2 §5.3：子目标闭环控制面工具 ----
        _tool(
            "plan_subgoals",
            "【只在规划阶段调用】把用户的复合任务拆成有序子目标列表（最多 4 条，按执行顺序）。"
            "「先 A 再 B」「A 放进托盘，B 叠到 A 上」这类多步任务用它。"
            "执行阶段不允许调用它——那时请直接调用具体技能（locate/pick/place/...）。",
            {
                "subgoals": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": _subgoal_item_schema(),
                    "description": "有序子目标列表，按执行先后排列",
                },
            },
            ["subgoals"],
        ),
        _tool(
            "locate",
            "语义标定：在相机图像里标定目标物体/容器的像素与包围框，结果会被记录并供后续 "
            "pick/place 复用（用户要求的「标定目标点」）。拿不准目标位置时先 locate，别凭记忆猜像素。"
            "返回 pixel_top / pixel_overview / bbox_top / bbox_overview 四个字段；"
            "camera='both'（默认）同时给 top 与 overview，高物体必须用它来消除视差。",
            {
                "target": {"type": "string",
                           "description": "要标定的目标描述，如「红色方块」「托盘」「水瓶」"},
                "camera": {"oneOf": [{"type": "string", "enum": list(LOCATE_CAMERAS)}, {"type": "null"}],
                           "description": "用哪个相机标定：both=top+overview 双目（默认，最稳，"
                                          "高物体必需）；也可只给 top/overview/wrist"},
                "color": {"oneOf": [{"type": "string", "enum": list(OBJECT_COLORS)}, {"type": "null"}],
                          "description": "目标的颜色线索，供颜色分割精修用；不确定就传 null"},
                "pixel": _vec_schema(2, "你自己从图里量到的目标中心像素 [u,v]（颜色分割失败时"
                                        "会直接采用它），可选", nullable=True),
                "bbox": {"type": ["array", "null"], "items": {"type": "number"},
                         "minItems": 4, "maxItems": 4,
                         "description": "你自己量到的目标包围框 [u1,v1,u2,v2]；系统取中心当像素，"
                                        "颜色分割失败时很有用"},
            },
            ["target"],
        ),
        _tool(
            "subgoal_done",
            "声明当前子目标已完成（模型自评）。系统仍会用图像独立复核，"
            "复核通过才算完成；请只在你确信动作已经做完时调用。",
            {"note": {"type": "string",
                      "description": "一句话说明完成依据，如「已在 top 图里确认红块落在托盘框内」"}},
            ["note"],
        ),
        _tool(
            "verify_subgoal",
            "【复核阶段专用】用工具调用回答「子目标是否真的完成」：done=true 表示你确信已完成，"
            "confidence 是 0~1 的把握，why 是一句话依据。系统复核子目标时会点名要求你调用它；"
            "执行阶段（抓取/放置流程中）不要调用。",
            {
                "done": {"type": "boolean", "description": "是否确信子目标已完成"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                               "description": "判断把握，0~1；< 0.6 视为证据不足"},
                "why": {"type": "string", "description": "一句话依据（看到图像里的什么现象）"},
            },
            ["done", "confidence", "why"],
        ),
    ]


#: OpenAI function-calling 工具表（覆盖 §2.5 全部 10 个技能）
TOOLS: list[dict] = _build_tools()


# --------------------------------------------------------------------------- #
# 系统提示词
# --------------------------------------------------------------------------- #
def system_prompt(include_poses: bool = True) -> str:
    """高层决策的系统提示词。

    include_poses=False 时追加「无真值感知模式」说明（只给像素瞄准方式，
    INTERFACES.md §4.3），与 serialize_state(include_poses=False) 配套。

    刻意不接收物体清单：场景每次都在 user 消息的 SCENE JSON 里给全，
    这样 system 段固定不变 -> 命中 prompt 缓存、也便于离线 MockVLM 复用。
    """
    base = (
        "你是 MuJoCo 仿真中 UR5e 六轴机械臂 + Robotiq 2F85 两指夹爪的高层决策器。"
        "你不做底层控制，只负责选择下一个工具调用（function call）。\n"
        "\n"
        "【坐标系与单位】\n"
        "- 使用 MuJoCo 世界坐标系：右手系，z 轴向上，长度单位米，角度弧度。\n"
        "- 机械臂底座固定在 [0, 0, 0.40]，桌面高度 0.40 m，工作区在桌面上方。\n"
        "- 物体位姿为 [x, y, z, qw, qx, qy, qz]（四元数 wxyz 在前），位置是物体几何中心。\n"
        "- TCP 是夹爪指尖中点（site g_pinch）。抓取方块通常从物体上方接近。\n"
        "\n"
        "【可用物体清单】\n"
        "- 每次 user 消息里的 SCENE JSON 给出当前 objects 列表（id/category/color/pose/size/"
        "graspable/is_container）以及 summary.graspable / summary.containers。\n"
        "- 只能操作列表里真实存在的 id；graspable=false 的物体（如托盘）不能抓，"
        "但 is_container=true 的可以当放置容器。\n"
        "\n"
        "【导纳控制语义】\n"
        "- 机械臂位置环外面套了一层导纳外环 M*x'' + D*x' + K*(x - x_ref) = -f_ext，"
        "接触受力时会沿被激活的轴产生柔顺退让。\n"
        "- k 是刚度（越大越硬、越不易被推偏）；d 是阻尼（抑制振荡）；m 是虚拟质量（越大越迟钝）；"
        "axes 是 6 维开关 [x,y,z,rx,ry,rz]，1=该轴参与导纳、0=该轴纯位置控制；"
        "force_limit 是异常接触力阈值(N)，超过会判force_limit错误。\n"
        "- 档位：stiff（刚性，axes 全 0，精确快速）；soft（全向柔顺，易碎/轻触场景）；"
        "z_only（仅 z 平动柔顺，适合下压贴合）。\n"
        "- 任务里出现「轻一点 / 轻拿轻放 / 小心 / 易碎 / 别压坏 / soft」-> 先 set_compliance(soft) "
        "或在 pick/place 里传 soft 档位；出现「快 / 精确 / 硬」-> stiff。\n"
        "- 接触力读数见 SCENE JSON 的 contact.force_norm / max_force / in_contact；"
        "力接近 force_limit 时应降低刚度而不是硬顶。\n"
        "\n"
        "【工具使用规则】\n"
        "- 必须且只能从给定工具里选一个调用；不要输出自然语言动作描述。\n"
        "- pick 的 target_id 必须存在且 graspable。\n"
        "- place 的放置目标四选一、必须且只能给一个：container_id 指定容器（必须 is_container=true）、"
        "surface 指定表面、on_top_of 指定一个可抓物体作为叠放锚点、"
        "或 anchor_id+relation 指定横向方位（放到某物体旁边）。\n"
        "- 任务说「在 X 右边 / X 左侧 / X 前面 / X 后面 / X 旁边（right of X / left of X / "
        "next to X / beside X）」-> place 用 anchor_id=X 的 id + relation；"
        "中英映射：右边/右侧/right -> right；左边/左侧/left -> left；前面/front -> front；"
        "后面/背后/behind -> back；旁边/一侧/beside/next to -> beside。"
        "方向是世界系水平方向（right=+x, left=-x, front=+y, back=-y），"
        "可选 distance=[米] 给中心距（不填由技能按两者半宽自动算）；anchor_id 必须 graspable。\n"
        "- 任务说「放到桌子上 / 桌面 / 台面 / 地上 / table」-> place 用 surface='table'，"
        "可选 xy=[x,y] 指定桌面落点（世界系米；不填由技能自选空位）；"
        "surface='tray' 表示最近的托盘面。桌面/地上的坐标含义：xy 为桌面上落点的世界系 x/y，"
        "z 由技能按桌面高度 0.40 m 自动决定。\n"
        "- 任务说「放在 X 上方 / X 上面 / X 上 / 叠在 X 上 / 摞在 X 上 / on top of X」-> place 用 "
        "on_top_of=X 的物体 id（X 必须是场景里 graspable=true 的物体；xy 可选，表示锚点顶面内的"
        "水平落点，z 由技能按锚点高度 + 被放物体半高计算）。托盘/桌面不能当锚点。\n"
        "- 放置关系表达不出来时（「旁边 / 左边 / 里面」等，或找不到锚点物体），不要猜、也不要用托盘"
        "兜底，改用 ask_user 问清目标。\n"
        "- 纠正类指令（「不是盘子里 / 放错了 / 重新放 / 不对 / 应该是…」）：目标物体沿用"
        "最近一次成功抓取的物体，place 到本次新指定的 surface/container/on_top_of。\n"
        "- SCENE JSON 的 objects[*].grasp_blocked / blocked_by（谁压着我、抓取是否受阻）"
        "与 supported_by / supporting（支撑关系），以及 relation_summary（中文摘要），"
        "都是现成观测：要抓的物体 grasp_blocked=true 时应先移走 blocked_by 里的物体，别硬抓。\n"
        "- pick/place 失败的 observation.metrics 里会有 failure_reason / obstruction_id / "
        "contact_force_norm / suggested_cameras：据此重新规划（移走 obstruction、换目标或换视角），"
        "不要重复同一参数硬试。\n"
        "- set_compliance 的 profile 只能是 stiff / soft / z_only。\n"
        "- 一次决策只输出一个工具调用，逐步推进；任务完成后调用 stop。\n"
        "\n"
        "【失败与重试策略】\n"
        "- 每步之后你会看到上一步的 observation（ok/message/error_code/metrics）。\n"
        "- error_code=force_limit：换成 soft 或更低刚度后重试，不要重复同样的硬接触。\n"
        "- error_code=ik_fail/unreachable：改为先用 home() 或 move_to 改变位形，再重新尝试；"
        "不要连续两次用完全相同的参数调用同一技能。\n"
        "- error_code=timeout：抬高预抓取高度（approach_height）或减少移动距离后重试。\n"
        "- 同一技能最多重试 2 次；仍失败则 ask_user 说明卡点，或 stop 结束。\n"
        "- 指令含糊（例如「抓那个东西」且有多个候选）时用 ask_user 澄清。"
    )
    if include_poses:
        return base
    return base + (
        "\n"
        "【无真值感知模式】\n"
        "- 本模式不会提供物体真实坐标/尺寸：objects 只给 id/color/category，也没有支撑关系真值；"
        "唯一的目标定位方式是从图像里读出像素。\n"
        "- 要抓/放某个物体时，输出它的中心像素 [u,v]：原点在图像左上角，u 向右、v 向下；"
        "配合 pixel_camera（不确定时用 top 相机并给估计值）。\n"
        "- pick 用 target_pixel=[u,v] + pixel_camera；place 用 target_pixel 指明落点，"
        "放置目标仍用 container_id / surface / on_top_of / anchor_id+relation 表达。\n"
        "- **禁止**在本模式输出 target_id / target_xy：它们依赖真值坐标，会被校验直接拒绝并浪费一步；"
        "pick 只给 target_pixel（+ pixel_camera），一次调用里绝不能把 target_id/target_xy 与 target_pixel 混用。\n"
        "- 反投影失败会返回可重试的 perception_error（带原始像素与相机），据此重新给像素或换相机，"
        "不要重复同一个坏像素。\n"
        "\n"
        "【工作流程（无真值模式固定三步）】\n"
        "- ① 先 locate(target, color, camera='top') 标定目标像素：它会返回 pixel 与 bbox，"
        "结果会被系统记录并供后续步骤复用；\n"
        "- ② 再用 pick(target_pixel=...) / place(...) 执行；\n"
        "- ③ 当前子目标做完后调用 subgoal_done(note=...) 声明完成，系统会用图像独立复核。\n"
        "- 拿不准就再 locate 一次（可换 overview 相机或换 color 线索），不要凭记忆猜像素；"
        "同一个坏像素连续失败两次会被系统判为重复错误并终止。\n"
        "\n"
        "【抓取档位 grasp_profile（重要）】\n"
        "- **默认一律用 top（自上而下）抓取，包括圆柱、水瓶这类细长物体**：2F85 夹爪张开 93 mm，"
        "足够夹住瓶身/罐体的上半部，不要去侧抓。\n"
        "- 只有任务**明确要求「从侧面抓 / 避开顶部」**、且物体**周围明显空旷**（有侧向接近的净空）时，"
        "才允许 grasp_profile='side'。\n"
        "- 理由：side 需要侧向净空，场景里常常没有；硬侧抓会在接近途中把物体推走。\n"
        "- 拿不准就用 top（或 auto，等价 top）。\n"
        "\n"
        "【日常物品识别要点】\n"
        "- 水瓶：细长（约 200 mm 高）半透明圆柱体、带盖，颜色偏 cyan/淡蓝；**默认自上而下（top）**"
        "夹住瓶身上半部即可（张开 93 mm 足够），不要侧抓。\n"
        "- 眼镜盒：扁长方形硬壳（约 136x52x24 mm），颜色偏 black/黑；面积大、高度低，"
        "自上而下夹长边中部即可（grasp_profile='auto'）。\n"
        "- 彩色方块（红/绿）：20 mm 立方体，grasp_profile='auto'（等价 top）自上而下夹。\n"
        "- 叠放（「放到 X 上面」）：place 用 on_top_of=X，descend_until_contact 留空即自动开启接触式落放；\n"
        "- 放进容器：落点像素在容器内时同样自动接触式落放，避免砸下去把方块弹飞。\n"
        "\n"
        "【高度与视差（重要）】\n"
        "- top 相机不是严格正交俯视：把物体**顶面**像素按桌面平面反投影会有系统视差——"
        "40 mm 方块约 6~9 mm，而水瓶（高约 200 mm）误差可达 68.6 mm（比瓶半径还大），"
        "而且加大分辨率也救不了。\n"
        "- 所以：**高物体（水瓶、立起来的盒子/罐）必须同时给两个像素**——"
        "target_pixel（top 相机）与 target_pixel_overview（overview 相机），"
        "系统会用双目三角定位求出真实三维点；只给一个像素时按桌面平面近似，最大约 7 cm 误差。\n"
        "- 矮方块（20 mm）给一个 top 像素就够了，不必凑 overview。\n"
        "- 拿不准就先 locate(target, camera='both')：它会一次返回 pixel_top / pixel_overview / "
        "bbox_top / bbox_overview，把这两个像素直接填进 pick/place 即可。\n"
    )


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def _from_dict(d: dict) -> dict:
    """把各种 dict 形态归一成 {name, arguments, reasoning}。"""
    if "tool_calls" in d and isinstance(d["tool_calls"], (list, tuple)) and d["tool_calls"]:
        return _from_dict(d["tool_calls"][0])
    if "function" in d and isinstance(d["function"], dict):
        f = d["function"]
        if "name" in f or "arguments" in f:
            out = {"name": f.get("name"), "arguments": f.get("arguments", {})}
            if d.get("reasoning") is not None:
                out["reasoning"] = d.get("reasoning")
            return out
    # {"type":"function","function":{...}} 已在上面处理；下面兼容我们自己的 SkillCall JSON
    if "skill" in d and "name" not in d:
        out = {"name": d.get("skill"), "arguments": d.get("args", {})}
        if d.get("reasoning") is not None:
            out["reasoning"] = d.get("reasoning")
        return out
    return d


def _balanced_json_objects(text: str) -> list[Any]:
    """扫描文本里所有「括号平衡」的 JSON 对象（正确处理字符串内的花括号）。"""
    out: list[Any] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    frag = text[start:i + 1]
                    try:
                        out.append(json.loads(frag))
                    except Exception:
                        pass
                    start = -1
    return out


def _from_text(text: str) -> dict:
    """从纯文本里提取工具调用：整段 JSON / 代码块 / 内嵌 JSON / name({...}) 形式。"""
    s = text.strip()
    if not s:
        raise ValueError("parse_skill_call: 输入为空字符串")

    # 1) 整段就是 JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return _from_dict(obj)
    except Exception:
        pass

    # 2) 代码围栏内是 JSON
    fence_re = re.compile(re.escape(_FENCE) + r"(?:json|JSON)?\s*(.*?)" + re.escape(_FENCE), re.S)
    for m in fence_re.finditer(text):
        try:
            obj = json.loads(m.group(1).strip())
        except Exception:
            continue
        if isinstance(obj, dict):
            return _from_dict(obj)

    # 3) name({...}) / name {...} 形式的文本调用
    call_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(\{.*?\})\s*\)", re.S)
    for m in call_re.finditer(text):
        try:
            args = json.loads(m.group(2))
        except Exception:
            continue
        if isinstance(args, dict):
            return {"name": m.group(1), "arguments": args}

    # 4) 任意平衡 JSON 对象，优先带 name/skill 字段的
    cands = [o for o in _balanced_json_objects(text) if isinstance(o, dict)]
    for o in cands:
        if "name" in o or "skill" in o or "function" in o:
            return _from_dict(o)
    if cands:
        return _from_dict(cands[0])
    raise ValueError("parse_skill_call: 文本中找不到可解析的工具调用 JSON")


def parse_skill_call(raw: Any) -> SkillCall:
    """把模型原始输出解析成 SkillCall；非法或未知技能 raise ValueError。

    兼容输入：
    (a) OpenAI tool_calls 元素 {"name": "pick", "arguments": "{\\"target_id\\": ...}"}
    (b) 完整的 tool_call {"function": {"name": ..., "arguments": ...}}
    (c) 响应壳 {"tool_calls": [...]}（取第一个）
    (d) 纯文本内嵌 JSON / 代码块 / tool_name({...})
    (e) SkillCall 自身的 JSON {"skill": ..., "args": {...}}
    """
    if isinstance(raw, dict):
        obj = _from_dict(raw)
    elif isinstance(raw, (list, tuple)):
        errs = []
        for it in raw:
            try:
                return parse_skill_call(it)
            except ValueError as e:
                errs.append(str(e))
        raise ValueError("parse_skill_call: 列表中没有任何可解析的调用；" + " | ".join(errs))
    elif isinstance(raw, str):
        obj = _from_text(raw)
    else:
        raise ValueError(f"parse_skill_call: 不支持的类型 {type(raw).__name__}")

    if not isinstance(obj, dict):
        raise ValueError(f"parse_skill_call: 解析结果不是对象，而是 {type(obj).__name__}")

    name = obj.get("name", obj.get("skill"))
    if not isinstance(name, str) or not name.strip():
        raise ValueError("parse_skill_call: 缺少工具名 name/skill")
    name = name.strip()
    if name not in SKILL_NAMES:
        raise ValueError(f"parse_skill_call: 未知技能 {name!r}，可用技能 {list(SKILL_NAMES)}")

    arguments = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
    if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
        arguments = {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception as e:
            raise ValueError(f"parse_skill_call: arguments 不是合法 JSON: {e}")
    if not isinstance(arguments, dict):
        raise ValueError(f"parse_skill_call: arguments 必须是对象，实际 {type(arguments).__name__}")

    allowed = ALLOWED_ARGS[name]
    args = {k: v for k, v in arguments.items() if k in allowed}
    unknown = sorted(set(arguments) - set(args))
    reasoning = str(obj.get("reasoning") or "").strip()
    if unknown:
        note = f"[parse] 忽略 {name} 的未知参数: {unknown}"
        reasoning = (reasoning + " " + note).strip() if reasoning else note

    compliance: Optional[ComplianceParams] = None
    if "compliance" in args:
        v = args["compliance"]
        if v is None:
            args.pop("compliance", None)
        elif isinstance(v, dict):
            compliance = _compliance_from_json(v)
        elif isinstance(v, str):
            if v not in PROFILE_NAMES:
                raise ValueError(f"parse_skill_call: compliance 档位 {v!r} 非法，可选 {list(PROFILE_NAMES)}")
            compliance = compliance_from_profile(v)
        else:
            raise ValueError("parse_skill_call: compliance 只能是档位名、参数对象或 null")

    if name == "set_compliance":
        profile = args.get("profile")
        k_scale = args.get("k_scale")
        if isinstance(k_scale, (int, float)) and not isinstance(k_scale, bool):
            compliance = compliance_from_profile(str(profile), float(k_scale)) if profile in PROFILE_NAMES else None
        elif profile in PROFILE_NAMES:
            compliance = compliance_from_profile(str(profile))

    return SkillCall(skill=name, args=args, compliance=compliance, reasoning=reasoning)


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num_vec(v: Any, n: int) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == n and all(_is_num(x) for x in v)


def validate_skill_call(call: SkillCall, state: SceneState) -> tuple[bool, str]:
    """对 SkillCall 做「白名单 + 类型/范围 + 场景一致性」三道检查。

    返回 (ok, msg)：ok=False 时 msg 是给 VLM 看的具体原因（回灌到 observation 里）。
    刻意不抛异常——决策环需要把失败当成正常反馈继续跑（最多 max_steps 步）。

    §5.12「冗余要纠正、矛盾才拒绝」（实机诊断驱动）：
    1. 无真值模式（给了 target_pixel/target_pixel_overview）：pick 的冗余 target_id/target_xy
       直接丢掉、以像素为准，并在 call 上记一条纠正（日志
       「[validate] 冗余 target_id 已忽略（无真值模式以像素为准）」）；
    2. place(container_id + xy)：落点在容器水平范围内 -> 接受 xy 为落点（container 只决定
       高度/越壁），记纠正；落点在容器外 -> 判矛盾拒绝。注意直接调用本函数时若只给
       target_pixel（未反投影成 xy）则无法判断内外，此时放行，交给 AgentLoop 反投影后再校验；
    3. 真矛盾（surface + container_id、on_top_of + container_id、锚点关系混用）仍拒绝；
    4. oracle 模式 pick(target_id + target_xy)：xy 与该物体真值距离 ≤ 30 mm 才接受（改以 xy 为准），
       超过就是自相矛盾，拒绝。

    纠正记录经 record_correction() 挂在 call 上，由 AgentLoop 用 take_corrections() 取出并写进
    SkillResult.metrics["corrected_args"]（§5.12-5）。
    """
    if not isinstance(call, SkillCall):
        return False, f"调用不是 SkillCall，而是 {type(call).__name__}"
    if call.skill not in SKILL_NAMES:
        return False, f"未知技能 {call.skill!r}，不在白名单 {list(SKILL_NAMES)}"
    args = call.args or {}
    if call.skill == "plan_subgoals":
        # 规划专用工具：只允许在 taskplan.plan_subgoals 的规划调用里解析；
        # 执行循环里出现一律 invalid_call（否则会让循环空转、也会绕过子目标编排）。
        return False, ("plan_subgoals 只能在规划阶段调用，执行循环里请直接给出一个具体技能"
                       "（home/move_to/pick/place/push/open_gripper/close_gripper/"
                       "set_compliance/locate/subgoal_done/stop/ask_user）")
    if call.skill == "verify_subgoal":
        # 复核专用工具（§5.11）：复核必须走 function calling，但它属于 SubGoalVerifier，
        # 执行循环里出现一律 invalid_call —— 否则模型可以自己给自己发「完成」判定（谎报成功）。
        done_arg = args.get("done")
        if not isinstance(done_arg, bool):
            return False, f"verify_subgoal.done 必须是布尔值，实际 {done_arg!r}"
        conf = args.get("confidence")
        if not _is_num(conf) or not (0.0 <= float(conf) <= 1.0):
            return False, f"verify_subgoal.confidence 必须在 [0, 1]，实际 {conf!r}"
        if not isinstance(args.get("why"), str):
            return False, f"verify_subgoal.why 必须是字符串，实际 {args.get('why')!r}"
        return False, ("verify_subgoal 只能在复核阶段（SubGoalVerifier）调用；"
                       "执行阶段若想声明子目标完成，请用 subgoal_done(note=...)")

    if call.compliance is not None and not isinstance(call.compliance, ComplianceParams):
        return False, "compliance 字段类型错误（应为 ComplianceParams）"

    if call.skill == "pick":
        # 目标三选一：target_id（oracle）/ target_xy（世界系桌面坐标）/ target_pixel（像素）
        tid = args.get("target_id")
        txy = args.get("target_xy")
        tpix = args.get("target_pixel")
        tpix_ov = args.get("target_pixel_overview")
        tid_given = isinstance(tid, str) and bool(tid.strip())
        txy_given = txy is not None
        tpix_given = tpix is not None
        tpix_ov_given = tpix_ov is not None
        # 像素族（§5.10）：target_pixel 与 target_pixel_overview 可成对给出（双目三角），
        # 只给 overview 也允许；二者算同一个「三选一」里的像素分支。
        pix_given = tpix_given or tpix_ov_given
        # §5.12-1：冗余但一致 -> 纠正而不是拒绝。无真值模式（给了像素）下 target_id 不含
        # 可执行信息，直接丢掉、以像素为准；若同时塞了 target_xy，也以像素为准。
        if pix_given and tid_given:
            args.pop("target_id", None)
            record_correction(call, "target_id", "dropped",
                              "同时给了像素；无真值模式下 target_id 不含可执行信息，以像素为准",
                              "[validate] 冗余 target_id 已忽略（无真值模式以像素为准）")
            tid, tid_given = None, False
        if pix_given and txy_given:
            args.pop("target_xy", None)
            record_correction(call, "target_xy", "dropped", "同时给了像素，以像素为准",
                              "[validate] 冗余 target_xy 已忽略（已有像素，以像素为准）")
            txy, txy_given = None, False
        given = [name for name, hit in (("target_id", tid_given),
                                        ("target_xy", txy_given),
                                        ("target_pixel/target_pixel_overview", pix_given)) if hit]
        # target_id + target_xy（无像素）不按「三选一冲突」处理：交给 §5.12-4 做 ≤30 mm 一致性校验
        id_xy_pair = tid_given and txy_given and not pix_given
        if len(given) > 1 and not id_xy_pair:
            return False, ("pick 的 target_id / target_xy / target_pixel（可带 target_pixel_overview）"
                           f"三选一、不能同时给：当前给了 {given}")
        if not given:
            return False, ("pick 需要 target_id / target_xy / target_pixel（可带 "
                           "target_pixel_overview）三选一（必须且只能给一个），当前都为空")
        if pix_given:
            if tpix_given and not _num_vec(tpix, 2):
                return False, f"pick.target_pixel 必须是 2 维数值数组 [u,v]，实际 {tpix!r}"
            if tpix_ov_given and not _num_vec(tpix_ov, 2):
                return False, (f"pick.target_pixel_overview 必须是 2 维数值数组 [u,v]，"
                                f"实际 {tpix_ov!r}")
            # 像素允许越界（模型会数错）：桌外/相机后方由 pixel_resolver 失败成 perception_error
        elif txy_given and tid_given:
            # §5.12-4：oracle 模式同时给 target_id 与 target_xy —— 两者一致（≤30 mm）才纠正后
            # 接受 xy；不一致说明模型自相矛盾，必须拒绝并给可读错误（不许悄悄用真值兜底）。
            obj = state.find(tid)
            if obj is None:
                ids = [o.id for o in state.objects]
                return False, f"pick 目标 {tid!r} 不存在；当前物体: {ids}"
            if not _num_vec(txy, 2):
                return False, f"pick.target_xy 必须是 2 维数值数组 [x,y]，实际 {txy!r}"
            if any(abs(float(v)) > 2.0 for v in txy):
                return False, f"pick.target_xy 超出合理工作区（|分量|>2 m）：{txy!r}"
            dist = math.hypot(float(txy[0]) - float(obj.pose[0]), float(txy[1]) - float(obj.pose[1]))
            if dist > PICK_XY_TRUTH_TOL_M:
                return False, (
                    f"pick 的 target_xy {[round(float(v), 4) for v in txy]} 与物体 {tid!r} 的真实位置"
                    f"相距 {dist * 1000:.0f} mm，超过 {PICK_XY_TRUTH_TOL_M * 1000:.0f} mm，参数互相矛盾："
                    f"请只给 target_id（用真值）或只给 target_xy（用你自己算出来的坐标）")
            args.pop("target_id", None)
            record_correction(call, "target_id", "dropped",
                              f"target_xy 与 {tid} 真值一致（{dist * 1000:.0f} mm），改用 xy",
                              f"[validate] target_id 与 target_xy 一致（{dist * 1000:.0f} mm ≤ "
                              f"{PICK_XY_TRUTH_TOL_M * 1000:.0f} mm），已改用 target_xy")
        elif txy_given:
            if not _num_vec(txy, 2):
                return False, f"pick.target_xy 必须是 2 维数值数组 [x,y]，实际 {txy!r}"
            if any(abs(float(v)) > 2.0 for v in txy):
                return False, f"pick.target_xy 超出合理工作区（|分量|>2 m）：{txy!r}"
        else:
            obj = state.find(tid)
            if obj is None:
                ids = [o.id for o in state.objects]
                return False, f"pick 目标 {tid!r} 不存在；当前物体: {ids}"
            if not obj.graspable:
                return False, f"pick 目标 {tid!r} 的 graspable=false，不能抓取"
        cam = args.get("pixel_camera")
        if cam is not None and cam not in PIXEL_CAMERAS:
            return False, f"pick.pixel_camera 只能是 {list(PIXEL_CAMERAS)} 之一或 null，实际 {cam!r}"
        zh = args.get("target_z_hint")
        if zh is not None and not _is_num(zh):
            return False, f"pick.target_z_hint 必须是数值或 null，实际 {zh!r}"
        ts = args.get("target_size")
        if ts is not None:
            if not isinstance(ts, (list, tuple)) or len(ts) not in (2, 3) \
                    or not all(_is_num(v) for v in ts):
                return False, f"pick.target_size 必须是 2~3 维数值数组 [sx,sy(,sz)]，实际 {ts!r}"
            if any(not (0.0 < float(v) <= 0.5) for v in ts):
                return False, f"pick.target_size 分量必须在 (0, 0.5] 米，实际 {ts!r}"
        ah = args.get("approach_height", 0.10)
        if not _is_num(ah) or not (0.01 <= float(ah) <= 0.5):
            return False, f"pick.approach_height 必须在 [0.01, 0.5] 米，实际 {ah!r}"
        cr = args.get("close_ratio", 0.35)
        if not _is_num(cr) or not (0.0 <= float(cr) <= 1.0):
            return False, f"pick.close_ratio 必须在 [0, 1]，实际 {cr!r}"
        gp = args.get("grasp_profile")
        if gp is not None and gp not in GRASP_PROFILES:
            return False, f"pick.grasp_profile 只能是 {list(GRASP_PROFILES)} 之一或 null，实际 {gp!r}"

    elif call.skill == "place":
        # 放置目标四选一：container_id（容器）/ surface（'table'/'tray'）/
        # on_top_of（叠放锚点）/ anchor_id+relation（横向方位），必须且只能给一个。
        cid = args.get("container_id")
        surface = args.get("surface")
        on_top = args.get("on_top_of")
        aid = args.get("anchor_id")
        relation = args.get("relation")
        distance = args.get("distance")
        xy = args.get("xy")
        tpix = args.get("target_pixel")
        pcam = args.get("pixel_camera")
        if tpix is not None and not _num_vec(tpix, 2):
            return False, f"place.target_pixel 必须是 2 维数值数组 [u,v]，实际 {tpix!r}"
        tpv = args.get("target_pixel_overview")
        if tpv is not None and not _num_vec(tpv, 2):
            return False, f"place.target_pixel_overview 必须是 2 维数值数组 [u,v]，实际 {tpv!r}"
        if pcam is not None and pcam not in PIXEL_CAMERAS:
            return False, f"place.pixel_camera 只能是 {list(PIXEL_CAMERAS)} 之一或 null，实际 {pcam!r}"
        cid_given = isinstance(cid, str) and bool(cid.strip())
        on_top_given = isinstance(on_top, str) and bool(on_top.strip())
        aid_given = isinstance(aid, str) and bool(aid.strip())
        if cid is not None and not cid_given:
            return False, f"place.container_id 必须是非空字符串或 null，实际 {cid!r}"
        if on_top is not None and not on_top_given:
            return False, f"place.on_top_of 必须是非空字符串或 null，实际 {on_top!r}"
        if aid is not None and not aid_given:
            return False, f"place.anchor_id 必须是非空字符串或 null，实际 {aid!r}"
        if surface is not None and surface not in SURFACE_NAMES:
            return False, f"place.surface 只能是 {list(SURFACE_NAMES)} 之一或 null，实际 {surface!r}"
        if relation is not None and relation not in RELATION_NAMES:
            return False, f"place.relation 只能是 {list(RELATION_NAMES)} 之一或 null，实际 {relation!r}"
        surface_given = surface is not None
        rel_given = relation is not None
        if aid_given != rel_given:
            return False, ("place 的 anchor_id 与 relation 必须成对出现"
                           "（横向方位 = anchor_id + relation）："
                           f"anchor_id={aid!r}, relation={relation!r}")
        lateral_given = aid_given and rel_given
        given = [name for name, hit in (("container_id", cid_given),
                                        ("surface", surface_given),
                                        ("on_top_of", on_top_given),
                                        ("anchor_id+relation", lateral_given)) if hit]
        if len(given) > 1:
            return False, ("place 的 container_id / surface / on_top_of / anchor_id+relation 四选一、"
                           f"不能同时给：当前给了 {given}")
        if not given:
            return False, ("place 需要 container_id / surface / on_top_of / anchor_id+relation 四选一"
                           "（必须且只能给一个），当前都为空")
        if distance is not None and not lateral_given:
            return False, (f"place.distance 只在 anchor_id+relation（横向方位）时可用，"
                           f"实际给了 {distance!r}")
        if lateral_given:
            if distance is not None and (not _is_num(distance) or not (0.0 < float(distance) <= 2.0)):
                return False, f"place.distance 必须在 (0, 2] 米，实际 {distance!r}"
            aobj = state.find(aid)
            if aobj is None:
                ids = [o.id for o in state.objects]
                return False, f"place 锚点 {aid!r} 不存在；当前物体: {ids}"
            if aobj.is_container or not aobj.graspable:
                return False, (f"place 锚点 {aid!r} 不可抓（graspable={aobj.graspable}, "
                               f"is_container={aobj.is_container}），不能作为方位锚点")
            if xy is not None:
                return False, "place.xy 只在 surface='table' 时可用；横向方位请用 distance"
        elif on_top_given:
            aobj = state.find(on_top)
            if aobj is None:
                ids = [o.id for o in state.objects]
                return False, f"place 锚点 {on_top!r} 不存在；当前物体: {ids}"
            if aobj.is_container or not aobj.graspable:
                return False, (f"place 锚点 {on_top!r} 不可抓（graspable={aobj.graspable}, "
                               f"is_container={aobj.is_container}），不能作为叠放锚点")
            if xy is not None:
                if not _num_vec(xy, 2):
                    return False, f"place.xy 必须是 2 维数值数组 [x,y] 或 null，实际 {xy!r}"
                if any(abs(float(v)) > 2.0 for v in xy):
                    return False, f"place.xy 超出合理工作区（|分量|>2 m）：{xy!r}"
        elif surface_given:
            if surface != "table" and xy is not None:
                return False, (f"place.xy 只在 surface='table' 或 on_top_of 时可用，"
                               f"实际 surface={surface!r}")
            if xy is not None:
                if not _num_vec(xy, 2):
                    return False, f"place.xy 必须是 2 维数值数组 [x,y] 或 null，实际 {xy!r}"
                if any(abs(float(v)) > 2.0 for v in xy):
                    return False, f"place.xy 超出合理工作区（|分量|>2 m）：{xy!r}"
        else:
            obj = state.find(cid)
            if obj is None:
                ids = [o.id for o in state.objects]
                return False, f"place 容器 {cid!r} 不存在；当前物体: {ids}"
            if not obj.is_container:
                return False, f"place 目标 {cid!r} 的 is_container=false，不能作为容器"
            if xy is not None:
                # §5.12-2：container_id + xy 不再一律判矛盾——落点落在容器范围内就接受
                # （container 只决定放置高度/越壁策略）；落在容器外才是真矛盾，拒绝。
                if not _num_vec(xy, 2):
                    return False, f"place.xy 必须是 2 维数值数组 [x,y] 或 null，实际 {xy!r}"
                if any(abs(float(v)) > 2.0 for v in xy):
                    return False, f"place.xy 超出合理工作区（|分量|>2 m）：{xy!r}"
                if container_contains_xy(state, obj, xy):
                    record_correction(
                        call, "xy", "accepted",
                        f"xy 落点落在容器 {obj.id} 范围内，接受为落点（container 决定高度/越壁）",
                        f"[validate] place 的 xy 落点在容器 {obj.id} 范围内，已接受为落点"
                        f"（container 只决定高度/越壁策略）")
                else:
                    return False, (
                        f"place 的 xy 落点 {[round(float(v), 4) for v in xy]} 不在容器 {obj.id!r} 范围内"
                        f"（容器中心 ({float(obj.pose[0]):.3f},{float(obj.pose[1]):.3f})，"
                        f"半宽 {float(obj.size[0]):.3f}/{float(obj.size[1]):.3f} m）；"
                        f"落点在容器外时 container_id 与 xy 互相矛盾——请改用 offset 给容器内偏移，"
                        f"或改 surface='table' 放到桌面")
        off = args.get("offset", [0.0, 0.0, 0.02])
        if not _num_vec(off, 3):
            return False, f"place.offset 必须是 3 维数值数组，实际 {off!r}"
        if any(abs(float(x)) > 0.5 for x in off):
            return False, f"place.offset 分量过大（|.|>0.5 m）：{off!r}"
        duc = args.get("descend_until_contact")
        if duc is not None and not isinstance(duc, bool):
            return False, f"place.descend_until_contact 必须是 true/false 或 null，实际 {duc!r}"

    elif call.skill == "move_to":
        pos = args.get("pos")
        if not _num_vec(pos, 3):
            return False, f"move_to 需要 3 维数值数组 pos，实际 {pos!r}"
        if any(abs(float(x)) > 2.0 for x in pos):
            return False, f"move_to.pos 超出合理工作区（|分量|>2 m）：{pos!r}"
        quat = args.get("quat")
        if quat is not None and not _num_vec(quat, 4):
            return False, f"move_to.quat 必须是 4 维数组 [w,x,y,z] 或 null，实际 {quat!r}"
        if quat is not None:
            n = sum(float(x) ** 2 for x in quat) ** 0.5
            if abs(n - 1.0) > 1e-2:
                return False, f"move_to.quat 不是单位四元数（模={n:.4f}）"

    elif call.skill == "push":
        tid = args.get("target_id")
        if not isinstance(tid, str) or not tid:
            return False, "push 需要非空字符串参数 target_id"
        if state.find(tid) is None:
            ids = [o.id for o in state.objects]
            return False, f"push 目标 {tid!r} 不存在；当前物体: {ids}"
        d = args.get("direction")
        if not _num_vec(d, 3):
            return False, f"push.direction 必须是 3 维数值数组，实际 {d!r}"
        if sum(float(x) ** 2 for x in d) ** 0.5 < 1e-6:
            return False, "push.direction 不能是零向量"
        dist = args.get("distance")
        if not _is_num(dist) or not (0.001 <= float(dist) <= 0.5):
            return False, f"push.distance 必须在 [0.001, 0.5] 米，实际 {dist!r}"

    elif call.skill == "close_gripper":
        r = args.get("ratio", 0.0)
        if not _is_num(r) or not (0.0 <= float(r) <= 1.0):
            return False, f"close_gripper.ratio 必须在 [0, 1]，实际 {r!r}"

    elif call.skill == "set_compliance":
        profile = args.get("profile")
        if profile not in PROFILE_NAMES:
            return False, f"set_compliance.profile 必须是 {list(PROFILE_NAMES)} 之一，实际 {profile!r}"
        ks = args.get("k_scale")
        if ks is not None and (not _is_num(ks) or not (0.01 <= float(ks) <= 100.0)):
            return False, f"set_compliance.k_scale 必须在 [0.01, 100]，实际 {ks!r}"
        if call.compliance is None:
            return False, "set_compliance 未解析出有效 compliance（profile 非法或缺失）"

    elif call.skill == "locate":
        target = args.get("target")
        if not isinstance(target, str) or not target.strip():
            return False, "locate 需要非空字符串参数 target（要标定的目标描述）"
        cam = args.get("camera")
        if cam is not None and cam not in LOCATE_CAMERAS:
            return False, f"locate.camera 只能是 {list(LOCATE_CAMERAS)} 之一或 null，实际 {cam!r}"
        color = args.get("color")
        if color is not None and (not isinstance(color, str) or not color.strip()):
            return False, f"locate.color 必须是非空字符串或 null，实际 {color!r}"
        lp = args.get("pixel")
        if lp is not None and not _num_vec(lp, 2):
            return False, f"locate.pixel 必须是 2 维数值数组 [u,v] 或 null，实际 {lp!r}"
        lb = args.get("bbox")
        if lb is not None and not _num_vec(lb, 4):
            return False, f"locate.bbox 必须是 4 维数值数组 [u1,v1,u2,v2] 或 null，实际 {lb!r}"

    elif call.skill == "subgoal_done":
        note = args.get("note")
        if not isinstance(note, str):
            return False, f"subgoal_done 需要字符串参数 note（可给空串），实际 {note!r}"

    elif call.skill == "ask_user":
        q = args.get("question")
        if not isinstance(q, str) or not q.strip():
            return False, "ask_user 需要非空字符串参数 question"

    elif call.skill in ("home", "open_gripper", "stop"):
        extra = sorted(args.keys())
        if extra:
            return False, f"{call.skill} 不接受参数，实际收到 {extra}"

    return True, "ok"
