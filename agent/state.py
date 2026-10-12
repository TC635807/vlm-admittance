"""agent.state —— 场景状态序列化 + 图像编码 + 多模态消息构造。

原理与踩坑（重要）：
1. serialize_state 只输出 **JSON-ready** 的纯 Python 对象（str/float/bool/None/list/dict）。
   必须显式 float()/int()：numpy 标量（np.float64 等）不是 JSON 可序列化类型，
   直接 json.dumps 会抛 TypeError，而这里的结果既要进 prompt 又要被 MockVLM 反向解析。
2. encode_image 走 PIL -> BytesIO(JPEG) -> base64：
   - 不落盘（CI/容器里没有可写目录也不会失败）；
   - JPEG 体积小，OpenAI 兼容端点普遍支持；
   - PIL 不可用时用「标准库 zlib 手写最小 PNG 编码器」兜底，绝不新增第三方依赖。
   注意：JPEG 不支持 alpha 通道，RGBA 必须先丢 alpha，否则 PIL 报错。
3. 图像输入可能是 (H,W,3) uint8（MuJoCo render 的返回值），也可能是 float 0~1 或 0~255，
   这里统一归一化到 uint8 RGB；灰度 (H,W) 会复制成 3 通道。
4. build_messages 的图像顺序固定为 sorted(images.keys())：
   同一场景多次调用产生逐字节一致的消息（可复现、可缓存、便于 CI diff）。
   文本块用 SCENE_JSON_BEGIN/END 标记包住场景 JSON，MockVLM 靠它做确定性解析。
"""
from __future__ import annotations

import base64
import io
import json
import math
import struct
import zlib
from typing import Any, Optional

import numpy as np

from core.config import RENDER_HEIGHT, RENDER_WIDTH, TABLE_HEIGHT
from core.types import ARM_JOINT_NAMES, SceneState

try:  # pillow 已装；仍然容错，保证无 PIL 环境可用
    from PIL import Image as _PILImage

    _HAS_PIL = True
except Exception:  # pragma: no cover - 仅在缺 PIL 的环境触发
    _PILImage = None
    _HAS_PIL = False

#: 场景 JSON 在 user 文本里的稳定标记，MockVLM 依赖它做「确定性」解析
SCENE_BEGIN = "SCENE_JSON_BEGIN"
SCENE_END = "SCENE_JSON_END"

#: fmt -> MIME
MIME_TYPES = {"jpeg": "image/jpeg", "jpg": "image/jpeg", "png": "image/png"}

#: 无真值模式（include_poses=False）附带的相机标定（INTERFACES.md §4.3）。
#: sim 侧真实 fovy/相机位姿由 demo/vision 注入 pixel_resolver；这里给与 sim/render.py 一致的默认，
#: 只用于告诉模型像素坐标系，不参与反投影计算。
CAMERA_CALIBRATION: dict[str, Any] = {
    "camera": "top",
    "width": int(RENDER_WIDTH),
    "height": int(RENDER_HEIGHT),
    "fovy": 45.0,
    "table_z": float(TABLE_HEIGHT),
}


# --------------------------------------------------------------------------- #
# numpy -> JSON
# --------------------------------------------------------------------------- #
def _finite(v: float) -> float:
    """非有限浮点（NaN/inf）归零：严格 JSON 不允许，VLM 也会被 NaN 带偏。"""
    return float(v) if math.isfinite(float(v)) else 0.0


def to_jsonable(obj: Any) -> Any:
    """把 numpy / dataclass 等递归转换成 json.dumps 可用的纯 Python 对象。

    踩坑：np.float32 不是 float 的子类，json.dumps 会失败；np.bool_ 同理；
    NaN/inf 虽被 json.dumps 默认放行，但会污染 prompt，这里统一归零。
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return _finite(obj)
    if isinstance(obj, np.generic):
        v = obj.item()
        return _finite(v) if isinstance(v, float) else v
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "to_json"):
        return to_jsonable(obj.to_json())
    return str(obj)


def _r3(v: Any) -> float:
    """安全取 3 位小数；None/NaN 归零，避免 JSON 里出现 NaN（严格 JSON 不允许）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f != f or f in (float("inf"), float("-inf")):  # NaN / inf
        return 0.0
    return round(f, 3)


# --------------------------------------------------------------------------- #
# 状态序列化
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 物体间关系推导（纯几何观测，task-45）
# --------------------------------------------------------------------------- #
#: 叠放判据：xy footprint 重叠面积 / 较小者面积
STACK_XY_OVERLAP_MIN = 0.60
#: |dz - (半高_a + 半高_b)| 容差（米）
STACK_Z_TOL_M = 0.015


def _aabb(obj: Any) -> tuple[float, float, float, float, float, float]:
    """物体（忽略旋转）的轴对齐包围盒 (xmin,xmax,ymin,ymax,zmin,zmax)。

    SceneObject.size 是半尺寸，pose 前 3 位是几何中心。
    """
    p = np.asarray(obj.pose, dtype=float)
    sx, sy, sz = float(obj.size[0]), float(obj.size[1]), float(obj.size[2])
    return (float(p[0]) - sx, float(p[0]) + sx,
            float(p[1]) - sy, float(p[1]) + sy,
            float(p[2]) - sz, float(p[2]) + sz)


def _xy_overlap_ratio(a: tuple, b: tuple) -> float:
    """两个 AABB 的 xy 重叠面积 / 较小者 footprint 面积；无重叠返回 0。"""
    ox = min(a[1], b[1]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[2], b[2])
    if ox <= 0.0 or oy <= 0.0:
        return 0.0
    area_a = (a[1] - a[0]) * (a[3] - a[2])
    area_b = (b[1] - b[0]) * (b[3] - b[2])
    smaller = min(area_a, area_b)
    return float(ox * oy / smaller) if smaller > 0 else 0.0


def derive_relations(state: SceneState) -> dict[str, dict]:
    """从 SceneState 的位姿/尺寸推导「谁压着谁」（纯几何，不做任何决策）。

    b 叠在 a 上（on_top_of）当且仅当：
      - xy footprint 重叠 >= STACK_XY_OVERLAP_MIN（按较小者面积），且
      - |dz - (sz_a + sz_b)| <= STACK_Z_TOL_M（中心距 ≈ 两个半高之和）。
    返回 {obj_id: {"supported_by": [...], "supporting": [...],
                   "grasp_blocked": bool, "blocked_by": [...]}}。
    """
    objs = list(state.objects)
    boxes = {o.id: _aabb(o) for o in objs}
    rel: dict[str, dict] = {o.id: {"supported_by": [], "supporting": [],
                                   "grasp_blocked": False, "blocked_by": []}
                            for o in objs}
    for b in objs:
        for a in objs:
            if a.id == b.id:
                continue
            if _xy_overlap_ratio(boxes[a.id], boxes[b.id]) < STACK_XY_OVERLAP_MIN:
                continue
            dz = float(b.pose[2]) - float(a.pose[2])
            if abs(dz - (float(a.size[2]) + float(b.size[2]))) > STACK_Z_TOL_M:
                continue
            rel[b.id]["supported_by"].append(a.id)
            rel[a.id]["supporting"].append(b.id)
    for o in objs:
        blocked = list(rel[o.id]["supporting"])
        rel[o.id]["blocked_by"] = blocked
        rel[o.id]["grasp_blocked"] = bool(blocked)
    return rel


def _relation_summary(relations: dict[str, dict], objs: list) -> str:
    """把关系压成中文人可读摘要（task-45 R2），例如
    「box_red 上方压着 box_blue（抓取受阻）；box_blue 叠在 box_red 上」。"""
    parts: list[str] = []
    for o in objs:
        r = relations.get(o.id) or {}
        supporting = r.get("supporting") or []
        if supporting:
            parts.append(f"{o.id} 上方压着 {'、'.join(supporting)}（抓取受阻）")
        supported_by = r.get("supported_by") or []
        if supported_by:
            parts.append(f"{o.id} 叠在 {'、'.join(supported_by)} 上")
    if not parts:
        return "当前没有物体互相压着，各物体抓取路径无遮挡"
    return "；".join(parts)


def obstruction_id(state: SceneState, call: Any) -> Optional[str]:
    """失败观测用：给出最可能的阻挡物 id（纯几何，不替 LLM 做决策）。

    - pick 失败：目标正上方压着谁（blocked_by 的第一个）；
    - place 失败：目标容器/锚点里或上面已经放着谁（supporting 的最后一个）。
    """
    try:
        rel = derive_relations(state)
    except Exception:  # 位姿/尺寸异常时不阻塞主流程
        return None
    args = getattr(call, "args", {}) or {}
    skill = getattr(call, "skill", "")
    if skill == "pick":
        blockers = (rel.get(str(args.get("target_id") or "")) or {}).get("blocked_by") or []
        return blockers[0] if blockers else None
    target = args.get("container_id") or args.get("anchor_id") or args.get("on_top_of")
    if target:
        on = (rel.get(str(target)) or {}).get("supporting") or []
        return on[-1] if on else None
    return None


def serialize_state(state: SceneState, task: str, extra: Optional[dict] = None,
                    include_poses: bool = True) -> dict:
    """把一次 SceneState 快照压成给 VLM 看的 JSON 字典。

    include_poses=False（INTERFACES.md §4.3 no-oracle 模式）：删除 objects[*].pose/size
    与全部 relations/支撑字段（这些等于真值），objects 只留 {id,color,category}，
    并附 camera_calibration {camera,width,height,fovy,table_z}。

    结构（契约冻结）::

        {
          "task": str,                 # 自然语言任务
          "time": float,               # 仿真时间(s)
          "joint_names": [str x6],
          "robot": {qpos, tcp_pos, tcp_quat, gripper, holding},
          "objects": [{id, category, color, pose, size, graspable, is_container,
                       supported_by, supporting, grasp_blocked, blocked_by}],
          "contact": {force, torque, force_norm, max_force, in_contact},
          "summary": {"graspable": [id...], "containers": [id...]},
          "relations": {id: {supported_by, supporting, grasp_blocked, blocked_by}},
          "relation_summary": str,     # 中文人可读：「box_red 上方压着 box_blue（抓取受阻）」
          "extra": {...}               # 进度/失败信息，调用方自定义
        }

    关系字段语义（task-45，纯几何观测）：
    - supported_by: 直接压在我上面的物体（我托着它们）；
    - supporting  : 直接被我托着的物体（我在它们下面）；
    - blocked_by  : == supporting，从夹爪进入方向看会挡住我抓取的物体；
    - grasp_blocked: blocked_by 非空（顶面被占用，抓取受阻）。

    设计取舍：不把 14 维 qpos 全塞进去——夹爪内部自由度对高层决策无意义，
    RobotState 只含 6 个臂关节（core.types 已如此约定）。
    """
    if include_poses:
        relations = derive_relations(state)
        objects_json = []
        for o in state.objects:
            obj = to_jsonable(o.to_json())
            r = relations.get(o.id) or {}
            obj["supported_by"] = list(r.get("supported_by") or [])
            obj["supporting"] = list(r.get("supporting") or [])
            obj["grasp_blocked"] = bool(r.get("grasp_blocked"))
            obj["blocked_by"] = list(r.get("blocked_by") or [])
            objects_json.append(obj)
    else:
        relations = None
        objects_json = [{"id": str(o.id), "color": str(o.color), "category": str(o.category)}
                        for o in state.objects]
    graspable = [o.id for o in state.objects if o.graspable]
    containers = [o.id for o in state.objects if o.is_container]
    out = {
        "task": str(task),
        "time": _r3(state.time),
        "joint_names": list(state.joint_names or ARM_JOINT_NAMES),
        "robot": to_jsonable(state.robot.to_json()),
        "objects": objects_json,
        "contact": to_jsonable(state.contact.to_json()),
        "summary": {"graspable": graspable, "containers": containers},
        "extra": to_jsonable(extra) if extra else {},
    }
    if include_poses:
        out["relations"] = relations
        out["relation_summary"] = _relation_summary(relations, state.objects)
    else:
        out["camera_calibration"] = dict(CAMERA_CALIBRATION)
    return out


# --------------------------------------------------------------------------- #
# 图像编码
# --------------------------------------------------------------------------- #
def _as_uint8_rgb(img: Any) -> np.ndarray:
    """归一化任意图像数组到 (H,W,3) uint8。"""
    arr = np.asarray(img)
    if arr.size == 0:
        raise ValueError("encode_image: 输入图像为空数组")
    if arr.dtype != np.uint8:
        a = arr.astype(np.float64)
        if np.issubdtype(arr.dtype, np.floating):
            finite = a[np.isfinite(a)]
            mx = float(finite.max()) if finite.size else 0.0
            if mx <= 1.0001:  # 0~1 浮点渲染结果
                a = a * 255.0
        a = np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(a, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]  # JPEG 不支持 alpha
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"encode_image: 期望 (H,W,3) 或 (H,W) 图像，实际 shape={arr.shape}")
    return np.ascontiguousarray(arr)


def _encode_png_py(rgb: np.ndarray) -> bytes:
    """纯标准库最小 PNG 编码器（PIL 缺失时的兜底）。

    格式 = 8 字节签名 + IHDR + IDAT(zlib) + IEND；
    每行前加 1 字节 filter type=0（None），色彩类型 2 = truecolor RGB，位深 8。
    """
    h, w, _ = rgb.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw.extend(rgb[y].tobytes())

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _chunk(b"IEND", b"")
    )


def encode_image_with_mime(img: Any, fmt: str = "jpeg", quality: int = 85) -> tuple[str, str]:
    """返回 (base64 字符串, MIME)。build_messages 用它给 data URI 贴正确的 MIME。

    为什么要一起返回 MIME：PIL 缺失时即使请求 jpeg 也只能退化成 PNG，
    调用方必须知道真实编码格式，否则 data URI 声明与实际字节不符。
    """
    fmt_key = str(fmt).lower().lstrip(".")
    if fmt_key not in MIME_TYPES:
        raise ValueError(f"encode_image: 不支持的格式 {fmt!r}，可选 {sorted(MIME_TYPES)}")
    if not (0 <= int(quality) <= 100):
        raise ValueError(f"encode_image: quality 必须在 0~100，实际 {quality}")
    rgb = _as_uint8_rgb(img)
    if _HAS_PIL:
        try:
            if fmt_key in ("jpeg", "jpg"):
                buf = io.BytesIO()
                _PILImage.fromarray(rgb).save(buf, format="JPEG", quality=int(quality))
                return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"
            buf = io.BytesIO()
            _PILImage.fromarray(rgb).save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii"), "image/png"
        except Exception:
            # PIL 存在但保存失败（少见，例如畸形 buffer）-> 落到纯 Python PNG
            pass
    png = _encode_png_py(rgb)
    return base64.b64encode(png).decode("ascii"), "image/png"


def encode_image(img: Any, fmt: str = "jpeg", quality: int = 85) -> str:
    """把图像编码成 base64 字符串（契约签名）。

    优先 PIL+JPEG（内存 BytesIO，不落盘）；PIL 不可用时退化为纯 Python PNG。
    JPEG 的 base64 一定以 "/9j/" 开头（0xFF 0xD8 0xFF 的 base64 前缀），smoke 用它断言。
    """
    b64, _mime = encode_image_with_mime(img, fmt=fmt, quality=quality)
    return b64


# --------------------------------------------------------------------------- #
# 消息构造
# --------------------------------------------------------------------------- #
def _content_to_text(content: Any) -> str:
    """把消息 content 压成纯文本（兼容 str 与 OpenAI 多模态 part 列表）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
                elif "text" in p:
                    parts.append(str(p["text"]))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)


def _user_text(task: str, scene: dict, n_images: int, cameras: list[str]) -> str:
    """拼 user 文本块：任务 + 关系摘要 + 场景 JSON + 图像清单 + 输出要求。

    「场景关系摘要」是中文人可读的一句（谁压着谁），随图像一起进每步 VLM message，
    真模型不必自己从位姿反推遮挡（task-45 R2）。
    """
    cam_line = ", ".join(cameras) if cameras else "(无图像)"
    summary = str(scene.get("relation_summary") or "")
    return (
        f"任务: {task}\n"
        f"场景关系摘要: {summary}\n"
        f"随附图像({n_images} 张，按此顺序): {cam_line}\n\n"
        f"{SCENE_BEGIN}\n"
        f"{json.dumps(scene, ensure_ascii=False)}\n"
        f"{SCENE_END}\n\n"
        "请根据任务、图像与场景 JSON 选择【一个】工具调用作为下一步动作。"
        "只输出工具调用，不要输出多余自然语言。"
    )


def _observation_text(item: dict) -> str:
    """把一条 SkillResult.to_observation() 渲染成历史观察文本。"""
    return "OBSERVATION_JSON " + json.dumps(to_jsonable(item), ensure_ascii=False)


def _normalize_history_item(item: Any) -> dict:
    """history 元素兼容两种形态：
    (a) 完整 chat message {role, content} -> 原样透传；
    (b) SkillResult.to_observation() 的裸 dict -> 包成 user 观察消息。
    """
    if isinstance(item, dict) and "role" in item and "content" in item:
        return {"role": str(item["role"]), "content": item["content"]}
    return {"role": "user", "content": _observation_text(to_jsonable(item))}


def build_messages(
    task: str,
    state: SceneState,
    images: dict[str, np.ndarray],
    history: Optional[list[dict]] = None,
    include_poses: bool = True,
) -> list[dict]:
    """构造 OpenAI 兼容的多模态 messages。

    结构::

        [ {role: system, content: system_prompt()},
          ...history（若有）...,
          {role: user, content: [ {type:text, ...},
                                  {type:image_url, image_url:{url:data:...;base64,...}},
                                  ... ]} ]

    图像顺序 = sorted(images)（不是 dict 插入顺序），保证可复现。
    """
    from .schema import system_prompt  # 延迟导入，避免模块级循环依赖

    scene = serialize_state(state, task, include_poses=include_poses)
    messages: list[dict] = [{"role": "system",
                             "content": system_prompt(include_poses=include_poses)}]
    for item in history or []:
        messages.append(_normalize_history_item(item))

    cam_names = sorted(str(k) for k in images.keys())
    parts: list[dict] = [{"type": "text", "text": _user_text(task, scene, len(cam_names), cam_names)}]
    for name in cam_names:
        b64, mime = encode_image_with_mime(images[name])
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    messages.append({"role": "user", "content": parts})
    return messages


def decode_data_uri(uri: str) -> bytes:
    """测试/调试辅助：把 data URI 还原成原始字节。"""
    if not isinstance(uri, str) or ";base64," not in uri:
        raise ValueError("不是合法的 data URI")
    return base64.b64decode(uri.split(";base64,", 1)[1])
