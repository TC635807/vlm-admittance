"""agent.taskplan —— 子目标规划 + 独立复核 + 逐子目标编排（INTERFACES.md §5.2 / §5.3）。

动机
----
单条自然语言任务常含多个动作（「把红色方块放进托盘，绿色方块叠到红方块上」）。
一次性丢给 AgentLoop 容易在中途丢失目标、也无法判断「做到哪一步了」。这里把任务拆成
有序子目标 SubGoal，每个子目标开一个**新的 AgentLoop epoch** 去跑，跑完由
SubGoalVerifier 用**独立证据**（图像几何 / 技能 metrics / VLM 视觉复核）判定是否真的完成，
不通过就带着上次失败原因重试，最多 SUBGOAL_MAX_ATTEMPTS 次。

信息隔离（§5.7 强制）
------------------
本模块**不读物体真值**：不调用 env.object_pose / set_object_pose，也不读
state.objects[*].pose。复核只用「渲染图 + sim/perception 的颜色/几何判据 + 技能 metrics
里的证据字段 + VLM 视觉复核」；连喂给复核 VLM 的 metrics 都只保留白名单证据键，
避免把诊断用的真值位姿泄漏进无真值模式的 prompt。

判定顺序（§5.2 冻结，绝不谎报成功）
------------------------------
a) place_in_container -> check_inside(rgb, subject_color, container_bbox)
b) place_on_top       -> check_stacked(rgb, subject_color, anchor_color)
c) pick               -> 技能 metrics 的 holding/grasp_gap_mm 优先；缺失则 check_gone(before, after)
d) place_at / free    -> 直接问 VLM 复核（带 top 图），confidence >= 0.6 才算 done
几何 ok            -> {"done": True,  "evidence": "geometry", ...}
几何否/不确定      -> VLM 复核 -> {"done": bool, "evidence": "vlm", "confidence": c}
两者都判否         -> {"done": False, "evidence": "none", "why": "..."}
任何异常           -> done=False（绝不上抛、绝不谎报成功）。

踩坑记录
-------
1. sim/perception.py（§5.1）是并行开发的模块，可能尚未就绪：这里**延迟导入**，并且
   perception=None 时优雅退化（几何判据不可用 -> 走 VLM 复核 -> 仍无证据就 done=False），
   绝不 ImportError 崩掉主流程。
2. check_inside / check_stacked / check_gone 的签名按 §5.1 冻结契约调用；测试里用
   stub perception 注入，不越界实现别人的模块。
3. VLM 的「自评」不算数：subgoal_done 只是终止当前 epoch 的信号；验收一律看 verify()
   的独立证据，模型说完成而证据否 -> 仍判未完成。
4. 每个子目标必须开新 epoch：AgentLoop 的 MockVLM 用 TASK_EPOCH_JSON 切分「当前任务」，
   复用同一个 loop 实例会串历史；loop_factory 每次都应造新实例。
5. locate 结果带缓存：同一 (target, color, camera) 只标定一次，避免复核时反复渲染/分割。
"""
from __future__ import annotations

import importlib
import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.types import SceneState, SkillCall

from .schema import (DEFAULT_LOCATE_CAMERA, DEFAULT_PIXEL_CAMERA, GOAL_TYPES, LOCATE_CAMERAS,
                     OBJECT_COLORS, parse_skill_call)
from .state import build_messages

#: 单个子目标最多尝试次数（§5.2 冻结）
SUBGOAL_MAX_ATTEMPTS = 3

#: 叠放复核（overview 斜视）：上层 bbox 底边与下层 bbox 顶边的垂直容差（px）
STACK_ADJACENT_TOL_PX: float = 8.0
#: 叠放复核（overview 斜视）：水平重叠 / 较小者宽度
STACK_OVERLAP_MIN: float = 0.60

#: 颜色词 -> 规范名（与 sim/perception.py 的颜色词表对齐）；长词优先，避免「浅蓝」被判成 blue
_COLOR_WORDS: dict[str, tuple[str, ...]] = {
    "red": ("red", "红"),
    "blue": ("blue", "蓝"),
    "green": ("green", "绿"),
    "yellow": ("yellow", "黄"),
    "gray": ("gray", "grey", "灰"),
    "black": ("black", "黑"),
    "white": ("white", "白"),
    "cyan": ("cyan", "浅蓝", "淡蓝", "青"),
    "orange": ("orange", "橙"),
}
_COLOR_CN = {"red": "红色", "blue": "蓝色", "green": "绿色", "yellow": "黄色",
             "gray": "灰", "black": "黑色", "white": "白色", "cyan": "淡蓝", "orange": "橙色"}

#: 颜色分割暂不可靠的颜色（sim-builder 正在修 COLOR_RULES；实机 gray 误检 494mm）：
#: 这类目标若模型自己给了 bbox/pixel，优先采用模型的结果
_UNRELIABLE_COLORS = ("gray", "white")

_PICK_WORDS = ("抓", "夹", "拿", "取", "pick", "grasp", "grab")
_PLACE_WORDS = ("放", "place", "put", "into")
_CONTAINER_WORDS = ("托盘", "盘子", "tray")
_STACK_WORDS = ("上方", "上面", "顶上", "叠", "摞", "on top", "stack")

#: 允许喂给「复核 VLM」的 metrics 证据键（白名单，杜绝真值位姿泄漏，§5.7）
_METRIC_EVIDENCE_KEYS = (
    "holding", "grasp_ok", "picked", "grabbed", "grasp_gap_mm", "grasp_closed_mm",
    "jaw_closed_mm", "grasp_evidence",
    # P1：无真值 pick 的物理随动证据（抬升 ok + 夹爪接触的物体已离地）。
    # 缺了这三个键，_geom_pick 的技能证据分支就是死代码（实测被本白名单过滤掉）。
    "grasp_lift_ok", "grasp_carry_evidence", "grasp_carry_bodies",
    "touchdown_z", "contact_force_norm", "obstruction_id", "failure_reason",
    "place_blocked", "target_id",
)
#: 这些子串出现在 metrics 键名里就一定丢弃（真值/坐标类诊断字段）
_METRIC_BANNED_SUBSTRINGS = ("pose", "truth", "oracle", "position", "xy", "quat", "qpos")

# --------------------------------------------------------------------------- #
# 共享像素缓存（无真值抓取验证 / 复核复用同一次标定）
# --------------------------------------------------------------------------- #
#: (color|"", camera) -> {"uv","bbox","source","target","color"}。托盘这类静止物在同一个
#: **任务**内不移动，所以任何一次 pick/place/locate 成功解析出的目标像素都可以喂给复核器复用，
#: 避免「实机 locate('托盘') 连续失败 -> place_in_container 的几何判据跑不了」。
#: 另外维护 _PIXEL_LOG（按时间顺序、含无颜色条目）：复核拿不到颜色线索时可按 target 文本
#: 或「最近一次 top 标定」回退到动作阶段实际用过的容器框。
_PIXEL_CACHE: dict[tuple[str, str], dict] = {}
_PIXEL_LOG: list[dict] = []
#: 缓存日志上限（够一次任务用；避免长会话内存增长）
_PIXEL_LOG_MAX = 64


def clear_pixel_cache() -> None:
    """清空共享像素缓存。

    **只在「新任务开始」时调用**（SubGoalOrchestrator.run 入口 / demo 换任务时）；
    绝不能在每个子目标 epoch 或每次 AgentLoop.run 开头清 —— 否则动作阶段（place/locate）
    标定好的容器框会在复核前丢掉，place_in_container 的几何判据就只剩 VLM 复核（实机踩过）。
    """
    _PIXEL_CACHE.clear()
    _PIXEL_LOG.clear()


def remember_pixel(color: Any, uv: Any, *, camera: str = "top", bbox: Any = None,
                   source: str = "resolve", target: Any = None) -> bool:
    """记住一次标定结果（颜色可缺省）；uv 不合法则忽略（返回 False）。"""
    if not (isinstance(uv, (list, tuple)) and len(uv) == 2):
        return False
    try:
        point = [round(float(uv[0]), 2), round(float(uv[1]), 2)]
    except (TypeError, ValueError):
        return False
    box = None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            box = [round(float(v), 2) for v in bbox]
        except (TypeError, ValueError):
            box = None
    cam = str(camera or "top")
    canon = canonical_color(color)
    entry = {
        "uv": point, "bbox": box, "source": str(source or "resolve"),
        "target": None if target is None else str(target),
        "color": canon, "camera": cam,
    }
    _PIXEL_CACHE[(canon or "", cam)] = dict(entry)
    _PIXEL_LOG.append(dict(entry))
    if len(_PIXEL_LOG) > _PIXEL_LOG_MAX:
        del _PIXEL_LOG[0:len(_PIXEL_LOG) - _PIXEL_LOG_MAX]
    return True


def recall_pixel(color: Any = None, *, camera: str = "top", target: Any = None,
                 allow_latest: bool = False) -> Optional[dict]:
    """取回上一次标定结果（没有则 None）。

    查找顺序：
      1) 颜色精确命中（color 给定时）；
      2) target 文本匹配（本次动作阶段 locate/place 用过的容器/目标；中英/包含都算）；
      3) allow_latest=True 时退回「最近一次该相机的带 bbox 标定」（容器框最后的兜底，
         调用方应把命中条目的 target/source 记进证据，保证可审计）。
    """
    cam = str(camera or "top")
    canon = canonical_color(color)
    if canon:
        hit = _PIXEL_CACHE.get((canon, cam))
        if hit:
            return dict(hit)
    if target is not None and str(target).strip():
        want = str(target).strip()
        for entry in reversed(_PIXEL_LOG):
            if entry.get("camera") != cam:
                continue
            have = str(entry.get("target") or "").strip()
            if have and (want in have or have in want):
                return dict(entry)
    if allow_latest:
        for entry in reversed(_PIXEL_LOG):
            if entry.get("camera") == cam and entry.get("bbox"):
                return dict(entry)
    return None


#: 纯文本兜底关键词（§5.11）：只在没有工具调用时使用；否定词优先，宁可不判完成也不谎报成功
_TEXT_POSITIVE = ("已完成", "完成", "已经", "成功", "done", "yes", "叠上", "落在",
                  "在托盘", "在容器", "夹住")
_TEXT_NEGATIVE = ("未完成", "没有", "还没", "不在", "仍在", "还在", "失败", "not done",
                  "未夹", "空了", "掉了", "掉下来")


def _text_verdict(text: str) -> Optional[tuple[bool, str]]:
    """从纯文本里读一个明确结论；含糊/空白返回 None（=无结论，不许猜成「未完成」）。

    否定词优先：只要出现否定表述就判未完成——这样绝不会把「还在地上」误判成成功。
    """
    t = str(text or "").strip()
    if not t:
        return None
    low = t.lower()
    if any(w in low for w in _TEXT_NEGATIVE):
        return False, f"VLM 复核未通过：{t[:120]}"
    if any(w in low for w in _TEXT_POSITIVE):
        return True, f"文本兜底判定已完成：{t[:120]}"
    return None


# --------------------------------------------------------------------------- #
# SubGoal
# --------------------------------------------------------------------------- #
@dataclass
class SubGoal:
    """一条子目标（§5.2 字段冻结）。

    id            : "g1"/"g2"…（由规划顺序生成）
    instruction   : 直接喂给 AgentLoop 的自然语言子任务
    goal_type     : place_in_container | place_on_top | place_at | pick | free（决定复核判据）
    subject       : 被操作物体的描述（人可读；不参与几何计算）
    subject_color : 被操作物体的颜色线索（喂 perception 做颜色分割）
    container     : 容器名称/描述（place_in_container 用）
    container_color/anchor_color : 颜色线索
    verify_hint   : 给人看的验收说明
    """
    id: str
    instruction: str
    goal_type: str
    subject: str
    subject_color: Optional[str] = None
    container: Optional[str] = None
    container_color: Optional[str] = None
    anchor_color: Optional[str] = None
    verify_hint: str = ""

    def to_json(self) -> dict:
        return {
            "id": self.id, "instruction": self.instruction, "goal_type": self.goal_type,
            "subject": self.subject, "subject_color": self.subject_color,
            "container": self.container, "container_color": self.container_color,
            "anchor_color": self.anchor_color, "verify_hint": self.verify_hint,
        }


def canonical_color(text: Any) -> Optional[str]:
    """把任意颜色描述规范成 perception 词表里的名字；认不出来返回 None。

    长别名优先（「浅蓝」-> cyan 而不是 blue；「灰」-> gray）。
    """
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip().lower()
    if t in OBJECT_COLORS:
        return t
    best: Optional[tuple[int, str]] = None
    for canon, aliases in _COLOR_WORDS.items():
        for a in aliases:
            if a in t and (best is None or len(a) > best[0]):
                best = (len(a), canon)
    return best[1] if best else None


def _colors_in_text(text: str) -> list[str]:
    """按首次出现位置返回文本里的颜色规范名（去重）。"""
    t = str(text).lower()
    hits: list[tuple[int, str]] = []
    for canon, aliases in _COLOR_WORDS.items():
        idxs = [t.find(a) for a in aliases]
        idxs = [i for i in idxs if i >= 0]
        if idxs:
            hits.append((min(idxs), canon))
    hits.sort()
    out: list[str] = []
    for _i, c in hits:
        if c not in out:
            out.append(c)
    return out


def bbox_top_midpoint(bbox: Any) -> Optional[list]:
    """bbox [u1,v1,u2,v2] -> 顶边中点 (u_center, v_top)；非法返回 None。

    「同一特征对应」的关键：top 视角里模型点的是物体**顶面中心**，overview 视角里对应的
    物理特征也是顶面中心（而不是模型容易点到的瓶身中部）。用 bbox 的顶边中点当这个特征，
    两个视角才在说同一个点，三角定位才准（实机：水瓶用模型点偏 25mm，用顶边中点只偏 4.2mm）。
    """
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    try:
        u1, v1, u2, v2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None
    if u2 < u1:
        u1, u2 = u2, u1
    if v2 < v1:
        v1, v2 = v2, v1
    return [round((u1 + u2) / 2.0, 2), round(v1, 2)]


def colors_in_text(text: Any) -> list[str]:
    """返回文本里出现过的颜色规范名（按首次出现位置、去重）；供 AgentLoop 做目标一致性检查。"""
    return _colors_in_text(str(text or ""))


def _has(text: str, words: tuple[str, ...]) -> bool:
    t = str(text).lower()
    return any(w in t for w in words)


def _verify_hint(sg: SubGoal) -> str:
    """按 goal_type 生成人可读的验收说明（也直接进复核 VLM 的 prompt）。"""
    subj = sg.subject or "被夹物体"
    if sg.goal_type == "place_in_container":
        return (f"{subj}（{sg.subject_color or '颜色未知'}）的色块质心应落在容器 "
                f"{sg.container or '目标容器'} 的框内（check_inside）")
    if sg.goal_type == "place_on_top":
        return (f"{subj}（{sg.subject_color or '颜色未知'}）应叠在 "
                f"{sg.anchor_color or '锚点'} 物体上方（check_stacked：质心落入下层框且重叠率达标）")
    if sg.goal_type == "pick":
        return (f"夹爪应已夹住 {subj}（技能 metrics 的 holding/grasp_gap_mm 优先；"
                f"缺失则看原位置该颜色色块是否消失 check_gone）")
    if sg.goal_type == "place_at":
        return f"{subj} 应已到达任务指定的位置（无固定几何判据，交 VLM 视觉复核）"
    return "自由形式子目标，没有固定几何判据，交 VLM 视觉复核"


def _subgoal_from_dict(d: Any, idx: int, log: Optional[Callable[[str], None]] = None) -> Optional[SubGoal]:
    """把 VLM 给的子目标 dict 转成 SubGoal；缺少 instruction 直接丢弃（返回 None）。"""
    if not isinstance(d, dict):
        return None
    instruction = str(d.get("instruction") or "").strip()
    if not instruction:
        if log:
            log(f"[taskplan] 第 {idx} 个子目标缺少 instruction，丢弃")
        return None
    raw_gt = d.get("goal_type")
    gt = str(raw_gt).strip().lower() if raw_gt is not None else "free"
    if gt not in GOAL_TYPES:
        if log:
            log(f"[taskplan] 子目标 goal_type={raw_gt!r} 非法，降级为 free（改走 VLM 复核）")
        gt = "free"
    subject = str(d.get("subject") or "").strip()
    subject_color = canonical_color(d.get("subject_color")) or canonical_color(subject)
    container = d.get("container")
    container = str(container).strip() if isinstance(container, str) and container.strip() else None
    sg = SubGoal(
        id=f"g{idx}",
        instruction=instruction,
        goal_type=gt,
        subject=subject or (instruction[:24]),
        subject_color=subject_color,
        container=container,
        container_color=canonical_color(d.get("container_color")),
        anchor_color=canonical_color(d.get("anchor_color")),
    )
    sg.verify_hint = _verify_hint(sg)
    return sg


def fallback_subgoals(task: str) -> list[SubGoal]:
    """无 LLM / 解析失败 / 空列表时的兜底：**单条**子目标，instruction 就是原任务（§5.2）。

    顺手做一点确定性推断（不猜物体 id、不读真值，只看任务文本）：
    - 有容器词 + 放置语义 -> place_in_container；
    - 有叠放词且提到 >=2 种颜色 -> place_on_top（第 1 色为被夹物，第 2 色为锚点）；
    - 只有抓取词没有放置语义 -> pick；
    - 其余 -> free（复核只能靠 VLM，没有 VLM 就诚实地判未完成）。
    """
    text = str(task or "").strip()
    colors = _colors_in_text(text)
    goal_type = "free"
    container: Optional[str] = None
    subject_color = colors[0] if colors else None
    anchor_color: Optional[str] = None

    if _has(text, _STACK_WORDS) and len(colors) >= 2:
        goal_type = "place_on_top"
        anchor_color = colors[1]
    elif _has(text, _CONTAINER_WORDS) and _has(text, _PLACE_WORDS + ("进", "入", "装")):
        goal_type = "place_in_container"
        for word in _CONTAINER_WORDS:          # 记录任务里点名的容器名（人可读即可）
            if word in text.lower():
                container = "托盘" if word in ("托盘", "tray") else word
                break
    elif _has(text, _PICK_WORDS) and not _has(text, _PLACE_WORDS):
        goal_type = "pick"

    subject = f"{_COLOR_CN.get(subject_color, subject_color)}物体" if subject_color else (text[:24] or "任务目标")
    sg = SubGoal(
        id="g1", instruction=text or "（空任务）", goal_type=goal_type, subject=subject,
        subject_color=subject_color, container=container, anchor_color=anchor_color,
    )
    sg.verify_hint = _verify_hint(sg)
    return [sg]


def _coerce_call(obj: Any) -> Optional[SkillCall]:
    """把 vlm.decide 的返回值统一成 SkillCall；dict/str 走 parse_skill_call，失败返回 None。"""
    if obj is None:
        return None
    if isinstance(obj, SkillCall):
        return obj
    try:
        return parse_skill_call(obj)
    except Exception:
        return None


def _parse_subgoal_list(raw: Any, cap: int,
                        log: Optional[Callable[[str], None]] = None) -> list[SubGoal]:
    """解析 plan_subgoals 的 subgoals 参数（list / JSON 字符串 / {"subgoals": [...]}）。"""
    data = raw
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            if log:
                log("[taskplan] subgoals 不是合法 JSON，回退")
            return []
    if isinstance(data, dict):
        data = data.get("subgoals", data.get("sub_goals"))
    if not isinstance(data, (list, tuple)):
        return []
    out: list[SubGoal] = []
    for d in data:
        if len(out) >= cap:
            if log:
                log(f"[taskplan] 子目标数超过上限 {cap}，截断")
            break
        sg = _subgoal_from_dict(d, len(out) + 1, log)
        if sg is not None:
            out.append(sg)
    return out


def plan_subgoals(vlm: Any, task: str, state: SceneState, images: dict,
                  *, max_subgoals: int = 4, log: Optional[Callable[[str], None]] = None,
                  include_poses: bool = True) -> list[SubGoal]:
    """优先让 VLM 调 plan_subgoals 工具拆解任务；失败/为空 -> fallback_subgoals(task)。

    include_poses=False（vlm 无真值模式）时规划 prompt 同样不带物体真值位姿，保持隔离（§5.7）。
    """
    _log = log if callable(log) else (lambda _m: None)
    cap = max(1, int(max_subgoals))
    try:
        if vlm is not None and state is not None:
            plan_task = (f"（规划阶段）请把下面这个复合任务拆成有序子目标，"
                         f"只调用 plan_subgoals 工具，最多 {cap} 条：{task}")
            messages = build_messages(plan_task, state, images or {}, history=None,
                                      include_poses=include_poses)
            call = _coerce_call(vlm.decide(messages))
            if call is not None and call.skill == "plan_subgoals":
                sgs = _parse_subgoal_list((call.args or {}).get("subgoals"), cap, _log)
                if sgs:
                    _log(f"[taskplan] VLM 规划出 {len(sgs)} 个子目标：" +
                         " / ".join(f"{s.id}:{s.goal_type}" for s in sgs))
                    return sgs
                _log("[taskplan] VLM 的 plan_subgoals 列表为空或不可解析，回退 fallback_subgoals")
            else:
                _log(f"[taskplan] VLM 未调用 plan_subgoals（返回 "
                     f"{getattr(call, 'skill', None) if call else None}），回退 fallback_subgoals")
    except Exception as e:  # 规划调用失败绝不影响主流程
        _log(f"[taskplan] 规划调用失败（{e}），回退 fallback_subgoals")
    return fallback_subgoals(task)


# --------------------------------------------------------------------------- #
# 独立复核
# --------------------------------------------------------------------------- #
class SubGoalVerifier:
    """子目标独立复核器（§5.2）。

    env        : SimEnv（只用 render / get_state，绝不读物体真值位姿）
    perception : sim.perception 模块（或同签名的 stub）；None 时延迟尝试 import，仍不可用则退化
    vlm        : 复核用决策器（decide(messages) -> SkillCall）；None 时不做 VLM 复核
    renderer   : 可选 SceneRenderer（rgb(data, camera)）；None 时用 env.render
    """

    def __init__(self, env: Any, *, perception: Any = None, vlm: Any = None,
                 renderer: Any = None, log: Optional[Callable[[str], None]] = None,
                 camera: str = DEFAULT_PIXEL_CAMERA, vlm_confidence_min: float = 0.6):
        self.env = env
        self.perception = perception
        self.vlm = vlm
        self.renderer = renderer
        self.log = log if callable(log) else (lambda _m: None)
        self.camera = str(camera or DEFAULT_PIXEL_CAMERA)
        #: VLM 复核的最低置信度（§5.2：confidence >= 0.6 才算 done）
        self.vlm_confidence_min = float(vlm_confidence_min)
        self._perception_resolved = perception is not None
        self._locate_cache: dict[tuple, dict] = {}
        self.locate_calls = 0        #: 真实执行的 locate 次数（缓存命中不计，便于测试）

    # -- 渲染 / 感知后端 ---------------------------------------------------- #
    def _get_perception(self) -> Any:
        """延迟解析 sim.perception（并行开发中，可能还不存在）。"""
        if self.perception is not None:
            return self.perception
        if not self._perception_resolved:
            self._perception_resolved = True
            try:
                self.perception = importlib.import_module("sim.perception")
                self.log("[taskplan] 已加载 sim.perception 作为几何判据后端")
            except Exception as e:
                self.perception = None
                self.log(f"[taskplan] sim.perception 不可用（{e}）；几何判据退化到 VLM 复核")
        return self.perception

    def _render(self, camera: Optional[str] = None) -> Any:
        """渲染一帧 RGB；失败返回 None（绝不抛）。"""
        cam = str(camera or self.camera)
        try:
            if self.renderer is not None:
                return self.renderer.rgb(getattr(self.env, "data", None), cam)
            return self.env.render(cam)
        except Exception as e:
            self.log(f"[taskplan] 渲染相机 {cam!r} 失败: {e}")
            return None

    @staticmethod
    def _safe_metrics(metrics: Optional[dict]) -> dict:
        """只保留白名单证据键，丢掉任何坐标/真值类诊断字段（§5.7）。"""
        if not isinstance(metrics, dict):
            return {}
        out: dict = {}
        for k, v in metrics.items():
            key = str(k).lower()
            if any(b in key for b in _METRIC_BANNED_SUBSTRINGS):
                continue
            if key in _METRIC_EVIDENCE_KEYS:
                out[str(k)] = v
        return out

    # -- locate（语义标定，带缓存） ---------------------------------------- #
    def locate(self, sg_or_text: Any, *, color: Optional[str] = None,
               camera: str = DEFAULT_LOCATE_CAMERA) -> dict:
        """标定目标的像素与包围框（§5.3 + §5.10 双目）。

        先让 VLM 调 locate 工具（可能补充/修正 target、color、camera），
        再用 perception.color_blob 逐相机精修出色块质心与 bbox。
        camera='both'（默认）会同时跑 top 与 overview，返回
        {"ok","pixel_top","pixel_overview","pixel_overview_top","bbox_top","bbox_overview",
         "refined","delta_px","pixel","bbox","target","camera","color","source","why"}
        （pixel_overview_top = bbox_overview 顶边中点 = 与 top 视角「同一物理特征：顶面中心」，
          三角定位时优先用它替代模型点）
        （"pixel"/"bbox" 是兼容旧调用方的别名：优先 top，其次 overview）。
        像素先由 color_blob 颜色分割给出，再用 sim.perception.refine_pixel 做颜色质心精修
        （refined=False 时严格保留原像素，绝不返回来路不明的错点）。
        注意：perception 的 gray/white 颜色规则目前在修（误检可达几十 cm），
        这类目标找不到时本函数诚实返回 ok=False，不做猜测。
        任何异常都转成 ok=False，绝不上抛。同一 (target,color,camera) 命中缓存。
        """
        target = sg_or_text.subject if isinstance(sg_or_text, SubGoal) else str(sg_or_text or "")
        target = str(target).strip()
        if color is None and isinstance(sg_or_text, SubGoal):
            color = sg_or_text.subject_color
        cam = str(camera or self.camera)
        key = (target, canonical_color(color) or color, cam)
        cached = self._locate_cache.get(key)
        if cached is not None:
            return dict(cached)
        try:
            res = self._locate_uncached(target, color, cam)
        except Exception as e:  # 兜底：任何异常都变成 ok=False
            res = {"ok": False, "pixel": None, "bbox": None, "target": target,
                   "camera": cam, "color": color, "source": "none", "why": f"locate 异常: {e}"}
        self.locate_calls += 1
        self._locate_cache[key] = dict(res)
        return dict(res)

    @staticmethod
    def _blob_fields(blob: Any) -> tuple[bool, Optional[list], Optional[list]]:
        """把 perception.color_blob 的返回（BlobResult 或 dict）归一成 (found, uv, bbox)。"""
        if isinstance(blob, dict):
            found, uv, bbox = blob.get("found"), blob.get("uv"), blob.get("bbox")
        else:
            found, uv, bbox = (getattr(blob, "found", None), getattr(blob, "uv", None),
                               getattr(blob, "bbox", None))
        norm_uv = norm_bbox = None
        if isinstance(uv, (list, tuple)) and len(uv) == 2:
            try:
                norm_uv = [round(float(uv[0]), 2), round(float(uv[1]), 2)]
            except (TypeError, ValueError):
                norm_uv = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                norm_bbox = [round(float(v), 2) for v in bbox]
            except (TypeError, ValueError):
                norm_bbox = None
        return bool(found), norm_uv, norm_bbox

    @staticmethod
    def _bbox_center(bbox: Any) -> Optional[list]:
        """[u1,v1,u2,v2] -> 中心像素；非法返回 None。"""
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return None
        try:
            xs = [float(bbox[0]), float(bbox[2])]
            ys = [float(bbox[1]), float(bbox[3])]
        except (TypeError, ValueError):
            return None
        return [round((min(xs) + max(xs)) / 2.0, 2), round((min(ys) + max(ys)) / 2.0, 2)]

    def _locate_one(self, rgb: Any, color: str, cam: str, *,
                    hint_uv: Optional[list] = None, model_bbox: Optional[list] = None) -> dict:
        """单相机定位：color_blob 找色块 -> refine_pixel 用颜色质心精修（§5.1）。

        返回 {"ok","uv","bbox","refined","delta_px","source","why"}。
        - color_blob 找不到时：模型自己给的 bbox（取中心）或 pixel 直接采用（source=vlm_bbox/vlm_pixel）；
        - gray/white 这类颜色规则暂不可靠时，模型给了 bbox 就优先用模型的；
        - refine_pixel 的契约是「refined=False 时 uv 必须原样等于 hint_uv」——这里严格遵守，
          绝不返回一个来路不明的错点。
        """
        perception = self._get_perception()
        out: dict = {"ok": False, "uv": None, "bbox": None, "refined": False,
                     "delta_px": None, "source": "none", "why": ""}
        blob_uv = blob_bbox = None
        blob_found = False
        if color:                      # 没颜色线索时不硬分割（避免瞎猜）
            try:
                blob = perception.color_blob(rgb, color)
                blob_found, blob_uv, blob_bbox = self._blob_fields(blob)
            except Exception as e:
                out["why"] = f"{cam} color_blob 异常: {e}"
        unreliable = canonical_color(color) in _UNRELIABLE_COLORS
        if model_bbox is not None and (not blob_found or unreliable):
            mc = self._bbox_center(model_bbox)
            if mc is not None:
                why = "颜色规则暂不可靠，采用模型自报 bbox" if blob_found else "颜色分割未命中，采用模型自报 bbox"
                return {"ok": True, "uv": mc, "bbox": [round(float(v), 2) for v in model_bbox],
                        "refined": False, "delta_px": None, "source": "vlm_bbox", "why": why}
        if not blob_found or blob_uv is None:
            if blob_found and blob_bbox is not None:
                # 分割只给了框（没有质心）-> 取 bbox 中心当像素（§5.3 兜底）
                mc = self._bbox_center(blob_bbox)
                if mc is not None:
                    return {"ok": True, "uv": mc,
                            "bbox": [round(float(v), 2) for v in blob_bbox],
                            "refined": False, "delta_px": None, "source": "blob_bbox",
                            "why": f"{cam} 颜色分割只给了 bbox，取中心当像素"}
            if hint_uv is not None:
                return {"ok": True, "uv": [round(float(hint_uv[0]), 2), round(float(hint_uv[1]), 2)],
                        "bbox": None, "refined": False, "delta_px": None,
                        "source": "vlm_pixel", "why": f"{cam} 颜色分割未命中，采用模型自报像素"}
            if out["why"]:
                return out
            out["why"] = f"{cam} 没找到 {color} 色块（可能被遮挡或在画面外）"
            return out
        uv, bbox = blob_uv, blob_bbox
        out.update(ok=True, uv=uv, bbox=bbox, source="color_blob")
        if hasattr(perception, "refine_pixel"):
            ref: Any = None
            try:
                try:       # §5.1 冻结签名：refine_pixel(rgb, color, hint_uv, *, camera, env)
                    ref = perception.refine_pixel(rgb, color, uv, camera=cam, env=self.env)
                except TypeError:      # 桩/旧实现不接受关键字 -> 退回位置参数
                    ref = perception.refine_pixel(rgb, color, uv)
            except Exception as e:     # 精修失败绝不影响主流程
                self.log(f"[taskplan] {cam} refine_pixel 异常（按未精修返回）: {e}")
                ref = None
            if isinstance(ref, dict):
                if ref.get("refined"):
                    ruv = ref.get("uv")
                    if isinstance(ruv, (list, tuple)) and len(ruv) == 2:
                        out["uv"] = [round(float(ruv[0]), 2), round(float(ruv[1]), 2)]
                    rbox = ref.get("bbox")
                    if isinstance(rbox, (list, tuple)) and len(rbox) == 4:
                        out["bbox"] = [round(float(v), 2) for v in rbox]
                    out["source"] = str(ref.get("source") or "color_blob")
                out["refined"] = bool(ref.get("refined"))
                if ref.get("delta_px") is not None:
                    try:
                        out["delta_px"] = round(float(ref["delta_px"]), 2)
                    except (TypeError, ValueError):
                        out["delta_px"] = None
        return out

    def _locate_uncached(self, target: str, color: Optional[str], cam: str) -> dict:
        cams = ["top", "overview"] if cam == "both" else [cam]
        out: dict = {"ok": False, "pixel": None, "bbox": None,
                     "pixel_top": None, "pixel_overview": None,
                     # 同一特征对应：overview bbox 顶边中点（= 顶面中心），三角定位优先用它
                     "pixel_overview_top": None,
                     "source_top": None, "source_overview": None,
                     "bbox_top": None, "bbox_overview": None,
                     "refined": False, "delta_px": None,
                     "refined_top": False, "refined_overview": False,
                     "delta_px_top": None, "delta_px_overview": None,
                     "target": target, "camera": cam, "color": color,
                     "source": "none", "why": ""}
        model_pixel: Optional[list] = None
        model_bbox: Optional[list] = None
        # 1) VLM locate 工具：语义标定（可补 color 线索 / 改相机 / 自报 pixel+bbox）
        if self.vlm is not None:
            call = None
            try:
                msgs = self._locate_messages(target, cam)
                call = _coerce_call(self.vlm.decide(msgs))
            except Exception as e:
                self.log(f"[taskplan] locate 的 VLM 调用失败: {e}")
            if call is not None and call.skill == "locate":
                a = call.args or {}
                t2 = str(a.get("target") or "").strip()
                if t2:
                    target = t2
                c2 = canonical_color(a.get("color"))
                if c2:
                    color = c2
                raw_px, raw_box = a.get("pixel"), a.get("bbox")
                if isinstance(raw_px, (list, tuple)) and len(raw_px) == 2:
                    try:
                        model_pixel = [round(float(raw_px[0]), 2), round(float(raw_px[1]), 2)]
                    except (TypeError, ValueError):
                        model_pixel = None
                if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
                    try:
                        model_bbox = [round(float(v), 2) for v in raw_box]
                    except (TypeError, ValueError):
                        model_bbox = None
                cam2 = a.get("camera")
                if isinstance(cam2, str) and cam2 in LOCATE_CAMERAS:
                    cam = cam2
                    cams = ["top", "overview"] if cam == "both" else [cam]
        # 注意：这里**不**从目标文本猜颜色 —— 调用方显式 color=None 就是「没有颜色线索」，
        # 必须诚实失败（不猜像素）。模型路径的文本解析放在 AgentLoop._run_locator 里（见那边注释）。
        out.update(target=target, color=color, camera=cam)
        # 2) perception.color_blob 逐相机颜色分割精修（§5.10：both 时 top+overview 都要）
        perception = self._get_perception()
        has_blob = perception is not None and hasattr(perception, "color_blob")
        if not has_blob:
            built = self._from_model_hint(out, model_pixel, model_bbox)
            if built is not None:
                return built
            out["why"] = "perception 不可用（sim.perception 未就绪），无法做颜色定位"
            return out
        # 2a) 没有颜色线索时**优先复用动作阶段实际用过的标定**（本次任务内 act->verify 不清缓存）：
        #     托盘这类目标模型常不给 color，但 place/locate 阶段已经标过它的框，直接拿来用，
        #     这样 place_in_container 的几何判据才跑得起来（实机 accept_v2_final2 的问题）。
        if not color and model_pixel is None and model_bbox is None:
            hit = recall_pixel(None, camera=str(cams[0]), target=target, allow_latest=True)
            if hit and hit.get("uv"):
                tag = "top" if cams[0] == "top" else "overview"
                out[f"pixel_{tag}"] = list(hit["uv"])
                out[f"bbox_{tag}"] = hit.get("bbox")
                out[f"source_{tag}"] = "cache"
                out["source"] = "cache"
                out["why"] = (f"没有颜色线索，复用动作阶段标定 {target!r}"
                              f"（target={hit.get('target')}, source={hit.get('source')}）")
                out["ok"] = True
                out["pixel"] = list(hit["uv"])
                out["bbox"] = hit.get("bbox")
                return out
            out["why"] = (f"缺少颜色线索，也无法复用缓存，无法在图像里定位 {target!r}；"
                          f"请在 locate 里给出 color 或自己量一个 pixel/bbox"
                          f"（perception 词表：{list(OBJECT_COLORS)}）")
            return out
        notes: list[str] = []
        for idx, c in enumerate(cams):
            rgb = self._render(c)
            if rgb is None:
                notes.append(f"{c} 相机渲染失败")
                continue
            one = self._locate_one(rgb, color or "", c,
                                   hint_uv=(model_pixel if idx == 0 else None),
                                   model_bbox=(model_bbox if idx == 0 else None))
            # 注意：top 的键名是 refined_top/delta_px_top（不能拼成 refined，会把顶层聚合键覆盖掉）
            tag = "top" if c == "top" else "overview"
            out[f"refined_{tag}"] = bool(one.get("refined"))
            out[f"delta_px_{tag}"] = one.get("delta_px")
            if one.get("ok"):
                out[f"pixel_{tag}"] = one.get("uv")
                out[f"bbox_{tag}"] = one.get("bbox")
                out[f"source_{tag}"] = one.get("source")
                if one.get("refined"):
                    notes.append(f"{c} 颜色质心精修 Δ={one.get('delta_px')}px")
                elif one.get("source") in ("vlm_bbox", "vlm_pixel"):
                    notes.append(f"{c} {one.get('why')}")
            else:
                notes.append(f"{c}: {one.get('why')}")
        # 注意：out["ok"] 要到下面的聚合块才计算，这里先按「有没有拿到像素」判断
        found_any = out["pixel_top"] is not None or out["pixel_overview"] is not None
        # 2b) 单相机失败 -> 换另一个相机再试一次（实机 top 找不到、overview 能找到）
        if not found_any and cam != "both" and model_pixel is None and model_bbox is None:
            other = "top" if cam != "top" else "overview"
            rgb = self._render(other)
            if rgb is not None:
                one = self._locate_one(rgb, color or "", other)
                if one.get("ok"):
                    tag = "top" if other == "top" else "overview"
                    out[f"pixel_{tag}"] = one.get("uv")
                    out[f"bbox_{tag}"] = one.get("bbox")
                    out[f"refined_{tag}"] = bool(one.get("refined"))
                    out[f"delta_px_{tag}"] = one.get("delta_px")
                    notes.append(f"换 {other} 相机重试成功")
                    found_any = True
                else:
                    notes.append(f"{other} 重试: {one.get('why')}")
        # 2c) 仍失败 -> 复用共享缓存里上一次成功标定的像素（同一 run 内托盘这类静止物不动）
        if not found_any:
            hit = recall_pixel(color, camera="top", target=target, allow_latest=(not color))
            if hit and hit.get("uv"):
                out["pixel_top"] = list(hit["uv"])
                out["bbox_top"] = hit.get("bbox")
                out["source"] = "cache"
                notes.append(f"复用上一次 {color} 标定 {hit['uv']}")
        out["pixel"] = out["pixel_top"] if out["pixel_top"] is not None else out["pixel_overview"]
        out["bbox"] = out["bbox_top"] if out["bbox_top"] is not None else out["bbox_overview"]
        out["ok"] = out["pixel"] is not None
        out["refined"] = bool(out["pixel_top"] is not None and out["refined_top"]) or bool(
            out["pixel_overview"] is not None and out["refined_overview"])
        top_delta = out["delta_px_top"] if out["pixel_top"] is not None else out["delta_px_overview"]
        out["delta_px"] = top_delta
        # 同一特征对应（§5.10 补）：有 bbox_overview 就给出「顶边中点」，供三角定位替换模型点
        _ov_box = out.get("bbox_overview")
        if _ov_box is None and cam == "overview":
            _ov_box = out.get("bbox_top")
        out["pixel_overview_top"] = bbox_top_midpoint(_ov_box) if _ov_box is not None else None
        if out["pixel_overview_top"] is not None:
            notes.append(f"overview 顶面中心(box 顶边中点)={out['pixel_overview_top']}"
                         f"（模型原点 pixel_overview={out.get('pixel_overview')}）")
        if out["ok"] and out.get("source") in (None, "none"):
            if out["refined"]:
                out["source"] = "refined"
            else:
                out["source"] = (out.get("source_top") or out.get("source_overview")
                                 or "color_blob")
        if not out["ok"]:
            out["source"] = "none"
        found_cams = [c for c, key in (("top", "pixel_top"), ("overview", "pixel_overview"))
                      if out.get(key) is not None]
        if out["ok"]:
            out["why"] = (f"颜色分割在 {'+'.join(found_cams)} 找到 {color} 色块"
                          + (f"；{'；'.join(notes)}" if notes else ""))
            # 注意：复核器**只读**共享像素缓存，不写 —— 缓存由动作阶段（AgentLoop 的
            # locate/place 解析成功时）写入。这样「本次动作阶段实际用过的容器框」才可靠，
            # 也避免复核自己的标定反过来污染下一次复核（实机 accept_v2_final2 的教训）。
        else:
            out["why"] = (f"颜色分割没找到 {color} 色块（可能被遮挡或在画面外）"
                          + (f"；{'；'.join(notes)}" if notes else ""))
        return out

    def _from_model_hint(self, out: dict, model_pixel: Optional[list],
                         model_bbox: Optional[list]) -> Optional[dict]:
        """没有颜色分割能力时的兜底：直接用模型自报的 bbox 中心 / pixel。

        返回填好的 out（就地修改）或 None（模型什么都没给）。
        """
        uv = self._bbox_center(model_bbox) if model_bbox is not None else None
        src = "vlm_bbox"
        if uv is None and model_pixel is not None:
            try:
                uv = [round(float(model_pixel[0]), 2), round(float(model_pixel[1]), 2)]
                src = "vlm_pixel"
            except (TypeError, ValueError, IndexError):
                uv = None
        if uv is None:
            return None
        box = None
        if model_bbox is not None:
            try:
                box = [round(float(v), 2) for v in model_bbox]
            except (TypeError, ValueError):
                box = None
        out.update(ok=True, pixel_top=uv, pixel=uv, bbox_top=box, bbox=box,
                   source=src, why="perception 不可用，采用模型自报的 bbox/pixel")
        if box is not None:
            out["pixel_overview_top"] = bbox_top_midpoint(box)
        return out

    def _locate_messages(self, target: str, cam: str) -> list[dict]:
        """构造 locate 用的消息；无场景时退化成纯文本（仍然能问）。both 时附 top+overview 两张图。"""
        cams = ["top", "overview"] if cam == "both" else [cam]
        task = (f"（语义标定）请在 {' + '.join(cams)} 相机图像里定位 {target!r}，"
                f"调用 locate(target=..., camera={cam!r}) 给出它的像素与颜色线索；"
                f"高物体（水瓶/立盒）务必同时给 top 与 overview 两个像素；只输出工具调用。")
        try:
            state = self.env.get_state()
        except Exception:
            state = None
        if state is None:
            return [{"role": "user", "content": task}]
        images = {}
        for c in cams:
            rgb = self._render(c)
            if rgb is not None:
                images[c] = rgb
        # include_poses=False：标定/复核都在无真值模式下工作（§5.7）
        return build_messages(task, state, images, include_poses=False)

    # -- verify ------------------------------------------------------------ #
    def verify(self, sg: Any, *, metrics: Optional[dict] = None, rgb_before: Any = None) -> dict:
        """复核一个子目标是否真的完成（§5.2 判定顺序）。

        返回 {"done": bool, "evidence": "geometry"|"vlm"|"none", "why": str, ...}。
        任何异常 -> done=False；**绝不**在没有证据时返回 done=True。
        """
        try:
            if not isinstance(sg, SubGoal):
                sg = fallback_subgoals(str(sg))[0]
            ev = self._safe_metrics(metrics)
            geo = self._geometry(sg, ev, rgb_before)
            if geo is not None and geo.get("done"):
                return geo
            geo_why = str((geo or {}).get("why") or "")
            vlm_res = self._vlm_review(sg, ev)
            if vlm_res is not None and vlm_res.get("done"):
                return vlm_res
            vlm_why = str((vlm_res or {}).get("why") or "")
            parts = []
            if geo_why:
                parts.append(f"几何判据：{geo_why}")
            if vlm_why:
                parts.append(f"VLM 复核：{vlm_why}")
            out: dict = {"done": False, "evidence": "none",
                         "why": "；".join(parts) or "没有可用的独立证据"}
            if geo_why:
                out["geometry_why"] = geo_why
            if vlm_res is not None:
                out["confidence"] = float(vlm_res.get("confidence", 0.0) or 0.0)
                if vlm_res.get("inconclusive"):
                    # §5.11：空回复/被截断 = 无结论，绝不能当成「模型说未完成」
                    out["inconclusive"] = True
                if vlm_why:
                    out["vlm_why"] = vlm_why
            return out
        except Exception as e:  # 复核本身出问题也绝不谎报成功
            return {"done": False, "evidence": "none", "why": f"复核异常: {e}"}

    def _geometry(self, sg: SubGoal, metrics: dict, rgb_before: Any) -> Optional[dict]:
        """几何/实测证据；返回 None 表示「判不了」（交给 VLM 复核）。"""
        if sg.goal_type == "place_in_container":
            return self._geom_inside(sg)
        if sg.goal_type == "place_on_top":
            return self._geom_stacked(sg)
        if sg.goal_type == "pick":
            return self._geom_pick(sg, metrics, rgb_before)
        return None                     # place_at / free -> 直接 VLM 复核

    def _geom_inside(self, sg: SubGoal) -> Optional[dict]:
        perception = self._get_perception()
        if perception is None or not hasattr(perception, "check_inside"):
            return None
        if not sg.subject_color:
            return None
        rgb = self._render(self.camera)
        if rgb is None:
            return None
        loc = self.locate(sg.container or "容器", color=sg.container_color, camera=self.camera)
        bbox = loc.get("bbox")
        if not loc.get("ok") or not bbox:
            return None                 # 容器框不出来 -> 不确定
        try:
            res = perception.check_inside(rgb, sg.subject_color, bbox)
        except Exception as e:
            return {"done": False, "evidence": "geometry", "why": f"check_inside 异常: {e}"}
        res = res if isinstance(res, dict) else {}
        ok = bool(res.get("ok"))
        why = str(res.get("why") or ("被夹物色块在容器框内" if ok else "被夹物色块不在容器框内"))
        return {"done": ok, "evidence": "geometry",
                "why": f"[相机={self.camera}｜判据=check_inside] {why}",
                "metrics": res.get("metrics")}

    @staticmethod
    def _blob_box(blob: Any) -> tuple[bool, Optional[list], Optional[list]]:
        """color_blob 返回 -> (found, uv, bbox)（dict / dataclass 都兼容）。"""
        if isinstance(blob, dict):
            found, uv, bbox = blob.get("found"), blob.get("uv"), blob.get("bbox")
        else:
            found, uv = getattr(blob, "found", None), getattr(blob, "uv", None)
            bbox = getattr(blob, "bbox", None)
        norm_uv = norm_box = None
        if isinstance(uv, (list, tuple)) and len(uv) == 2:
            try:
                norm_uv = [float(uv[0]), float(uv[1])]
            except (TypeError, ValueError):
                norm_uv = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                norm_box = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            except (TypeError, ValueError):
                norm_box = None
        return bool(found), norm_uv, norm_box

    @staticmethod
    def _boxes_vertically_adjacent(upper: list, lower: list) -> tuple[bool, str]:
        """上层 bbox 的底边 ≈ 下层 bbox 的顶边（±STACK_ADJACENT_TOL_PX）且水平重叠 ≥60%。"""
        u1, v1u, u2, v2u = upper
        l1, v1l, l2, v2l = lower
        gap = abs(v2u - v1l)
        ow = min(u2, l2) - max(u1, l1)
        wmin = min(max(u2 - u1, 1e-6), max(l2 - l1, 1e-6))
        ratio = max(ow, 0.0) / wmin
        ok = (gap <= STACK_ADJACENT_TOL_PX) and (ratio >= STACK_OVERLAP_MIN)
        why = (f"上层底边 {v2u:.0f} 与下层顶边 {v1l:.0f} 相邻差 {gap:.1f}px"
               f"（≤{STACK_ADJACENT_TOL_PX:.0f}），水平重叠 {ratio * 100:.0f}%"
               f"（≥{STACK_OVERLAP_MIN * 100:.0f}%）")
        return ok, why

    def _stacked_evidence_overview(self, sg: SubGoal, rgb: Any) -> tuple[Optional[dict], str]:
        """overview 斜视下判定叠放：返回 (判定 dict | None, 失败说明)。

        三条正面证据（绝不接受「只看到上层」）：
          ① 契约判据 check_stacked；
          ② bbox 几何：上层 bbox 底边 ≈ 下层 bbox 顶边（±8px）且水平重叠 ≥60%；
          ③ 遮挡工况：下层**完全找不到** 且 上层质心落在**动作阶段标定的锚点 bbox** 内。
        每条都会在 why 里写清用的是哪一条 + 相机。
        """
        perception = self._get_perception()
        if perception is None or not hasattr(perception, "color_blob"):
            return None, "perception 不可用"
        # ① 契约判据（check_stacked 在 overview 上也适用：质心落在下层框内 + 重叠率）
        if hasattr(perception, "check_stacked"):
            try:
                res = perception.check_stacked(rgb, sg.subject_color, sg.anchor_color)
                if isinstance(res, dict) and res.get("ok"):
                    why = str(res.get("why") or "上层质心在下层框内且重叠达标")
                    return ({"done": True, "evidence": "geometry",
                             "why": f"[相机=overview｜判据=check_stacked] {why}",
                             "metrics": res.get("metrics")}), ""
            except Exception as e:
                pass
        # ② / ③ 自己用 bbox 判（contract check_stacked 找不到下层时的补充）
        try:
            up = perception.color_blob(rgb, sg.subject_color)
        except Exception as e:
            return None, f"overview 上层分割异常: {e}"
        up_found, up_uv, up_box = self._blob_box(up)
        try:
            low = perception.color_blob(rgb, sg.anchor_color)
        except Exception as e:
            return None, f"overview 下层分割异常: {e}"
        low_found, low_uv, low_box = self._blob_box(low)
        if up_found and up_box is not None and low_found and low_box is not None:
            ok, why = self._boxes_vertically_adjacent(up_box, low_box)
            if ok:
                return ({"done": True, "evidence": "geometry",
                         "why": f"[相机=overview｜判据=bbox 相邻] {why}",
                         "metrics": {"upper_bbox": up_box, "lower_bbox": low_box}}), ""
            note = f"overview bbox 几何不成立：{why}"
        elif not up_found:
            note = "overview 上层色块也找不到"
        else:
            note = "overview 下层被完全遮挡（找不到）"
        # ③ 遮挡工况：下层找不到 + 上层质心落在动作阶段标定的锚点 bbox 内
        if up_found and up_uv is not None and not (low_found and low_box is not None):
            anchor_box = None
            for cam_try in ("overview", "top"):
                hit = recall_pixel(sg.anchor_color, camera=cam_try)
                if hit and hit.get("bbox"):
                    anchor_box = hit["bbox"]
                    break
            if anchor_box is not None:
                a1, av1, a2, av2 = (float(anchor_box[0]), float(anchor_box[1]),
                                    float(anchor_box[2]), float(anchor_box[3]))
                if a1 <= up_uv[0] <= a2 and av1 <= up_uv[1] <= av2:
                    return ({"done": True, "evidence": "geometry",
                             "why": (f"[相机=overview｜判据=下层被遮挡+上层落在锚点框内] "
                                     f"下层 {sg.anchor_color} 完全看不到，但上层质心 "
                                     f"{[round(up_uv[0], 1), round(up_uv[1], 1)]} 落在动作阶段"
                                     f"标定的锚点框 {anchor_box} 内"),
                             "metrics": {"upper_uv": up_uv, "anchor_bbox": anchor_box}}), ""
                note += f"；上层质心不在锚点框 {anchor_box} 内"
            else:
                note += "；没有锚点框可对（动作阶段未标定）"
        return None, note

    def _geom_stacked(self, sg: SubGoal) -> Optional[dict]:
        """叠放复核：先 top（契约判据），top 判不了再换 overview 斜视（遮挡工况）。

        设计动机（实机）：蓝块叠到绿块上严丝合缝后，俯视图里绿块被完全盖住 ->
        check_stacked 返回 lower_color_not_found，几何判据失明；必须用斜视视角补证据。
        两条都拿不到证据才回落到 VLM 复核；**绝不**「看到上层就算成功」。
        """
        perception = self._get_perception()
        if perception is None or not hasattr(perception, "check_stacked"):
            return None
        if not (sg.subject_color and sg.anchor_color):
            return None
        reasons: list[str] = []
        # ① top：契约判据
        rgb_top = self._render(self.camera)
        if rgb_top is not None:
            try:
                res = perception.check_stacked(rgb_top, sg.subject_color, sg.anchor_color)
            except Exception as e:
                res = {"ok": False, "why": f"check_stacked 异常: {e}"}
            res = res if isinstance(res, dict) else {}
            if res.get("ok"):
                why = str(res.get("why") or f"{sg.subject_color} 已叠在 {sg.anchor_color} 上")
                return {"done": True, "evidence": "geometry",
                        "why": f"[相机={self.camera}｜判据=check_stacked] {why}",
                        "metrics": res.get("metrics")}
            reasons.append(f"{self.camera}: {res.get('why') or 'check_stacked 未通过'}")
        else:
            reasons.append(f"{self.camera}: 渲染失败")
        # ② overview：斜视补证据（bbox 相邻 / 下层被遮挡+上层在锚点框内）
        if str(self.camera) != "overview":
            rgb_ov = self._render("overview")
            if rgb_ov is not None:
                verdict, note = self._stacked_evidence_overview(sg, rgb_ov)
                if verdict is not None:
                    return verdict
                if note:
                    reasons.append(f"overview: {note}")
            else:
                reasons.append("overview: 渲染失败")
        return {"done": False, "evidence": "geometry",
                "why": f"{sg.subject_color} 未叠在 {sg.anchor_color} 上（" + "；".join(reasons) + "）"}

    def _geom_pick(self, sg: SubGoal, metrics: dict, rgb_before: Any) -> Optional[dict]:
        # (c1) 技能实测 metrics 优先
        for key in ("holding", "grasp_ok", "picked", "grabbed"):
            if metrics.get(key) is True:
                ev = {k: metrics[k] for k in ("grasp_gap_mm", "grasp_evidence", "target_id") if k in metrics}
                return {"done": True, "evidence": "geometry",
                        "why": f"技能实测显示已夹住（metrics.{key}=True）", "metrics": ev}
        # (c1b) P1：技能的**物理随动证据**（抬升 ok + 夹爪接触的物体已离地）。
        #   为什么单列：无真值 pick 的 metrics 里既没有 holding 也没有 grasp_closed_mm，
        #   于是"技能实测优先"这条分支从来命不中，复核只能靠 check_gone；而 check_gone 是否
        #   通过取决于夹爪有没有把物体挡住（实测蓝块真抓起来后 top 面积 818->827，会漏判）。
        #   抬升 ok + 物体离地是物理事实，用它兜底避免把真抓判成失败。
        if (metrics.get("grasp_lift_ok") is True
                and metrics.get("grasp_carry_evidence") is True):
            return {"done": True, "evidence": "geometry",
                    "why": ("技能物理证据：抬升 ok 且夹爪接触的物体已离地"
                            f"（{metrics.get('grasp_carry_bodies')}）"),
                    "metrics": {k: metrics.get(k) for k in
                                ("grasp_gap_mm", "grasp_lift_ok", "grasp_carry_evidence",
                                 "grasp_carry_bodies")}}
        gap = metrics.get("grasp_gap_mm")
        closed = metrics.get("grasp_closed_mm", metrics.get("jaw_closed_mm"))
        if (isinstance(gap, (int, float)) and not isinstance(gap, bool)
                and isinstance(closed, (int, float)) and not isinstance(closed, bool)):
            if float(gap) > float(closed) + 1.0:      # 指间有物（> 全闭 + 1 mm）
                return {"done": True, "evidence": "geometry",
                        "why": f"指间间距 {float(gap):.1f} mm > 全闭 {float(closed):.1f} mm + 1 mm（指间有物）",
                        "metrics": {"grasp_gap_mm": gap, "grasp_closed_mm": closed}}
        if metrics.get("holding") is False:
            return {"done": False, "evidence": "geometry", "why": "技能实测 metrics.holding=False（未夹住）"}
        # (c2) metrics 无结论 -> 抓走后原位置色块应消失
        perception = self._get_perception()
        if perception is None or not hasattr(perception, "check_gone") or rgb_before is None:
            return None
        if not sg.subject_color:
            return None
        rgb_after = self._render(self.camera)
        if rgb_after is None:
            return None
        loc = self.locate(sg.subject or sg.instruction, color=sg.subject_color, camera=self.camera)
        hint = loc.get("pixel")
        if not hint:
            return None
        try:
            res = perception.check_gone(rgb_before, rgb_after, sg.subject_color, hint)
        except Exception as e:
            return {"done": False, "evidence": "geometry", "why": f"check_gone 异常: {e}"}
        res = res if isinstance(res, dict) else {}
        ok = bool(res.get("ok"))
        why = str(res.get("why") or ("原位置色块已消失（判定已抓走）" if ok else "原位置色块仍在"))
        return {"done": ok, "evidence": "geometry", "why": why, "metrics": res.get("metrics")}

    def _finish_vlm(self, done: bool, conf: float, why: str, source: str = "vlm") -> dict:
        """复核结论的统一出口：confidence < 阈值就算「证据不足」，不得算完成。"""
        if done and conf < self.vlm_confidence_min:
            return {"done": False, "evidence": "vlm", "confidence": conf, "source": source,
                    "why": f"VLM 认为完成但置信度 {conf:.2f} < {self.vlm_confidence_min:.2f}，不算数"}
        if done:
            return {"done": True, "evidence": "vlm", "confidence": conf, "source": source,
                    "why": why or "VLM 视觉复核通过"}
        return {"done": False, "evidence": "vlm", "confidence": conf, "source": source,
                "why": why or "VLM 复核未通过"}

    def _vlm_review(self, sg: SubGoal, metrics: dict) -> Optional[dict]:
        """问 VLM 复核（带图像），**优先 function calling**（§5.11）。

        端点实测：message.reasoning 占 completion token 预算，纯文本 content 可能为空，
        所以结构化判断必须走工具：首选 verify_subgoal(done, confidence, why)，
        兼容旧的 subgoal_done（视为 done=true/confidence=1），纯文本只作兜底；
        「回复为空 / 没有结构化结论」→ 标 inconclusive=True（**无结论**，不是「未完成」）。
        异常同样按无结论处理（绝不因此判子目标失败）。
        """
        if self.vlm is None:
            return None
        try:
            rgb = self._render(self.camera)
            images = {self.camera: rgb} if rgb is not None else {}
            state = self.env.get_state()
            task = self._review_task(sg, metrics)
            call = _coerce_call(self.vlm.decide(build_messages(task, state, images, include_poses=False)))
        except Exception as e:
            self.log(f"[taskplan] VLM 复核没有拿到结论（{e}）；本次视作无结论，可重试")
            return {"done": False, "evidence": "none", "confidence": 0.0, "inconclusive": True,
                    "why": f"VLM 复核未返回可用结论（{e}），可能被 token 截断，本次不计为未完成"}
        if call is None:
            return {"done": False, "evidence": "none", "confidence": 0.0, "inconclusive": True,
                    "why": "VLM 复核没有解析出工具调用，本次视作无结论"}

        args = call.args if isinstance(call.args, dict) else {}
        conf = args.get("confidence")
        conf_given = isinstance(conf, (int, float)) and not isinstance(conf, bool)
        if call.skill == "verify_subgoal":
            done = bool(args.get("done"))
            why = str(args.get("why") or "").strip()
            return self._finish_vlm(done, float(conf) if conf_given else (1.0 if done else 0.0),
                                    why, source="vlm_tool")
        if call.skill == "subgoal_done":                      # 兼容旧工具
            why = str(args.get("note") or "").strip()
            return self._finish_vlm(True, float(conf) if conf_given else 1.0, why, source="vlm_tool")

        # 纯文本兜底（§5.11：不作为首选；空文本 -> 无结论）
        text = " ".join(str(x) for x in (getattr(call, "reasoning", "") or "",
                                          args.get("content") or "", args.get("reply") or "",
                                          args.get("note") or "") if x).strip()
        verdict = _text_verdict(text)
        if verdict is not None:
            done, why = verdict
            conf_txt = float(conf) if conf_given else (
                self.vlm_confidence_min if done else 0.0)
            return self._finish_vlm(done, conf_txt, why, source="vlm_text")
        return {"done": False, "evidence": "none", "confidence": 0.0, "inconclusive": True,
                "why": "VLM 回复里没有结构化结论（纯文本为空或无法判读，可能被 token 截断），"
                       "本次视作无结论、不计为未完成"}

    def _review_task(self, sg: SubGoal, metrics: dict) -> str:
        m = json.dumps(metrics, ensure_ascii=False) if metrics else "无"
        return (f"（子目标复核）请判断这个子目标是否已经真的完成：{sg.instruction}\n"
                f"子目标类型：{sg.goal_type}\n"
                f"验收标准：{sg.verify_hint}\n"
                f"当前技能实测证据（已过滤，不含真值坐标）：{m}\n"
                f"请对照 {self.camera} 相机图像核对，并用 verify_subgoal 工具作答："
                f"verify_subgoal(done=true/false, confidence=0~1, why=一句话依据)。"
                f"必须走工具调用、不要只用文字回答；证据不足就给 done=false + 低 confidence，"
                f"不要为了结束而谎报完成。")


# --------------------------------------------------------------------------- #
# 逐子目标编排
# --------------------------------------------------------------------------- #
class SubGoalOrchestrator:
    """逐子目标跑 AgentLoop（每子目标新 epoch）-> 独立复核 -> 带原因重试（§5.3 末条）。

    loop_factory : Callable[[SubGoal, int], AgentLoop]（也兼容只收 SubGoal 或零参的工厂；
                   用 inspect.signature 判断应传几个参数），每次调用都应返回**新的** AgentLoop。
    verifier     : SubGoalVerifier（或同签名的 stub：verify(sg, metrics=..., rgb_before=...)）
    observe      : 复核前的「观察位」动作（可选）。不传则用 env.home()；失败只记日志不影响判定。
    run() 返回    : [{"subgoal": SubGoal, "attempts": int, "done": bool,
                      "evidence": str, "why": str}]（§5.2/§5.3 冻结字段）
    """

    def __init__(self, env: Any, vlm: Any, loop_factory: Callable, verifier: Any,
                 log: Optional[Callable[[str], None]] = None, *,
                 max_attempts: int = SUBGOAL_MAX_ATTEMPTS, include_poses: bool = True,
                 observe: Optional[Callable[[], Any]] = None):
        self.env = env
        self.vlm = vlm
        self.loop_factory = loop_factory
        self.verifier = verifier
        self.log = log if callable(log) else (lambda _m: None)
        self.max_attempts = max(1, int(max_attempts))
        #: 规划 prompt 是否带真值位姿（vlm 模式应传 False，§5.7）
        self.include_poses = bool(include_poses)
        #: 复核前的「观察位」钩子（demo 可注入 skills 的 home()/带载归位）；None -> env.home()
        self.observe = observe

    # -- 内部 -------------------------------------------------------------- #
    def _make_loop(self, sg: SubGoal, attempt: int) -> Any:
        """按工厂签名决定传参个数（约定 (subgoal, attempt)，兼容 (subgoal) 与 ()）。"""
        f = self.loop_factory
        if f is None:
            return None
        n = 2
        try:
            params = [p for p in inspect.signature(f).parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            n = len([p for p in params if p.default is p.empty])
        except (TypeError, ValueError):
            n = 2
        if n >= 2:
            return f(sg, attempt)
        if n == 1:
            return f(sg)
        return f()

    def _run_epoch(self, sg: SubGoal, attempt: int, instruction: str) -> list:
        try:
            loop = self._make_loop(sg, attempt)
        except Exception as e:
            self.log(f"[taskplan] {sg.id} loop_factory 调用失败: {e}")
            return []
        if loop is None:
            self.log(f"[taskplan] {sg.id} loop_factory 返回 None，跳过该子目标")
            return []
        try:
            return list(loop.run(instruction) or [])
        except Exception as e:
            self.log(f"[taskplan] {sg.id} AgentLoop.run 异常: {e}")
            return []

    @staticmethod
    def _collect_metrics(results: list) -> dict:
        """把该 epoch 各步的 metrics 合并（后出现的覆盖先出现的），只用于复核证据。"""
        merged: dict = {}
        for r in results or []:
            m = getattr(r, "metrics", None)
            if isinstance(m, dict):
                merged.update(m)
        return merged

    def _safe_render(self) -> Any:
        cam = str(getattr(self.verifier, "camera", DEFAULT_PIXEL_CAMERA))
        try:
            return self.env.render(cam)
        except Exception as e:
            self.log(f"[taskplan] 复核前渲染 {cam!r} 失败: {e}")
            return None

    def _observe(self, sg: SubGoal) -> None:
        """复核前先把机械臂移出工作区（观察位），让 top/overview 看得见结果。

        实机教训：叠放完成后夹爪还杵在叠放点上方，把整个区域挡住 —— 连 VLM 复核都「只见机械臂」。
        失败/超时**绝不允许**把子目标判死：只记日志，复核照常进行（复核仍是唯一判据）。
        优先用注入的 observe 钩子（demo 可传 skills 的 home / 带载归位）；否则退回 env.home()。
        钩子接受参数时把**子目标**传进去（P0：pick 类子目标的复核不能挪动机械臂，
        否则会把刚抓起来的物体丢掉 —— 钩子据此自行决定"不动"）。
        """
        try:
            if self.observe is not None:
                self._call_observe_hook(sg)
                self.log(f"[taskplan] {sg.id} 复核前观察位：已调用注入的 observe 钩子")
            elif self._observe_would_drop_held_object(sg):
                return                      # 见下：没注入钩子时也不许把刚抓起的物体挪丢
            elif hasattr(self.env, "home"):
                self.env.home()
                self.log(f"[taskplan] {sg.id} 复核前观察位：env.home()")
            else:
                self.log(f"[taskplan] {sg.id} 复核前观察位：环境不支持 home()，跳过")
        except Exception as e:
            self.log(f"[taskplan] {sg.id} 复核前观察位失败/超时（不影响复核）: {e}")

    def _observe_would_drop_held_object(self, sg: SubGoal) -> bool:
        """P0：没注入 observe 钩子时的兜底守卫 —— pick 子目标/仍持物时不做归位。

        为什么：默认兜底直接 `env.home()`。实测（verifier 探针）持物时瞬移会让物体
        留在原地掉回桌面（z 0.5393 -> 0.4199、carry True -> False），pick 子目标复核
        必然 subject_still_present。注入了钩子时由钩子自己负责（demo 的钩子会跳过）。
        """
        holding = None
        try:
            st = self.env.get_state()
            holding = getattr(getattr(st, "robot", None), "holding", None)
        except Exception:  # noqa: BLE001 - 查不到就当没持物
            holding = None
        goal_type = str(getattr(sg, "goal_type", "") or "")
        if goal_type == "pick" or holding:
            reason = "pick 子目标" if goal_type == "pick" else f"仍持物({holding})"
            self.log(f"[taskplan] {sg.id} 复核前观察位：跳过归位（{reason}）"
                     "—— 挪动机械臂会把刚抓起的物体丢掉")
            return True
        return False

    def _call_observe_hook(self, sg: SubGoal) -> Any:
        """调用注入的 observe 钩子；钩子能吃参数就把 sg 传进去，不能就零参调用。

        与 AgentLoop 调用带扩展形参的 pixel_resolver 同一套兼容策略：
        老钩子（lambda: ...）原样可用，新钩子（hook(sg)）能拿到子目标类型。
        """
        hook = self.observe
        try:
            params = inspect.signature(hook).parameters
            accepts = any(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                          for p in params.values()) or any(
                p.kind == p.VAR_POSITIONAL for p in params.values())
        except (TypeError, ValueError):
            accepts = False
        return hook(sg) if accepts else hook()

    def _safe_verify(self, sg: SubGoal, metrics: dict, rgb_before: Any) -> dict:
        try:
            v = self.verifier.verify(sg, metrics=metrics, rgb_before=rgb_before)
            if isinstance(v, dict):
                return v
            return {"done": False, "evidence": "none", "why": f"verifier 返回 {type(v).__name__}"}
        except Exception as e:
            return {"done": False, "evidence": "none", "why": f"verifier 异常: {e}"}

    # -- 主入口 ------------------------------------------------------------ #
    def run(self, task: str, subgoals: Optional[list] = None, *,
            state: Optional[SceneState] = None, images: Optional[dict] = None) -> list[dict]:
        """执行（或给定子目标后执行）整条任务，返回每个子目标的复核记录。"""
        clear_pixel_cache()      # 每 run 清一次共享像素缓存，避免跨任务串味
        sgs = list(subgoals) if subgoals else None
        if not sgs:
            st = state
            if st is None:
                try:
                    st = self.env.get_state()
                except Exception:
                    st = None
            if st is not None:
                sgs = plan_subgoals(self.vlm, task, st, images or {}, log=self.log,
                                    include_poses=self.include_poses)
            if not sgs:
                sgs = fallback_subgoals(task)
        self.log(f"[taskplan] 子目标编排开始：{len(sgs)} 个子目标 | 每目标最多 "
                 f"{self.max_attempts} 次尝试")
        records: list[dict] = []
        for sg in sgs:
            records.append(self._run_one(sg))
        done_n = sum(1 for r in records if r.get("done"))
        self.log(f"[taskplan] 子目标编排结束：{done_n}/{len(records)} 个通过")
        return records

    def _run_one(self, sg: SubGoal) -> dict:
        """跑一个子目标：失败 -> 带上原因重试，最多 max_attempts；全败诚实报 subgoal_failed。"""
        why = ""
        verdict: Optional[dict] = None
        for attempt in range(1, self.max_attempts + 1):
            if attempt == 1:
                instruction = sg.instruction
            else:
                instruction = (f"{sg.instruction}（上一次复核未通过：{why}；"
                                f"请再试一次，必要时先 locate 重新标定目标像素）")
            rgb_before = self._safe_render()
            t0 = time.monotonic()
            results = self._run_epoch(sg, attempt, instruction)
            metrics = self._collect_metrics(results)
            # 复核前让开视线：动作循环结束时夹爪常停在结果上方，会把 top/overview 挡死
            self._observe(sg)
            verdict = self._safe_verify(sg, metrics, rgb_before)
            elapsed = time.monotonic() - t0
            evidence = str(verdict.get("evidence") or "none")
            why = str(verdict.get("why") or "复核未通过")
            flag = "（无结论，可重试）" if verdict.get("inconclusive") else ""
            self.log(f"[taskplan] {sg.id} 第 {attempt}/{self.max_attempts} 次尝试 -> "
                     f"done={bool(verdict.get('done'))} evidence={evidence} ({elapsed:.1f}s) | {why}{flag}")
            if verdict.get("done"):
                if attempt > 1:
                    self.log(f"[taskplan] {sg.id} 成功（重试后），共 {attempt} 次尝试")
                    return {"subgoal": sg, "attempts": attempt, "done": True, "evidence": evidence,
                            "why": f"{why}；成功（重试后），共 {attempt} 次尝试"}
                self.log(f"[taskplan] {sg.id} 成功（一次通过）")
                return {"subgoal": sg, "attempts": 1, "done": True, "evidence": evidence,
                        "why": f"{why}；成功（一次通过）"}
        last = str((verdict or {}).get("why") or "复核未通过")
        self.log(f"[taskplan] {sg.id} 失败：{self.max_attempts} 次尝试全部未通过（subgoal_failed）")
        return {"subgoal": sg, "attempts": self.max_attempts, "done": False, "evidence": "none",
                "why": f"{last}；{self.max_attempts} 次尝试均未通过（subgoal_failed）"}
