"""agent.loop —— 感知 -> 决策 -> 校验 -> 执行 -> 回灌 的闭环。

数据流（每一步）::

    env.get_state()                 # 最新 SceneState
      + env.render(camera)          # 多视角 RGB
      -> build_messages(...)        # system + history + user(文本 JSON + data URI 图像)
      -> vlm.decide(messages)       # SkillCall
      -> validate_skill_call(...)   # 白名单/类型/场景一致性
      -> executor(call)             # skills 模块执行，返回 SkillResult
      -> history += [assistant(call JSON), user(OBSERVATION_JSON)]
    结束条件：skill == "stop"/"ask_user"、同一 skill+error_code 连续失败 >= 2 次后
    VLM 仍提议同一 skill、不可重试错误码第 2 次出现、或步数达 max_steps。

跨命令历史（task-25 / task-29）::
    history 可经构造参数或 run(history=...) 注入，run 结束后 self.history 为
    「注入历史 + 本次新增」，调用方（如交互式 UI）可读回保存并喂给下一条命令，
    这样 MockVLM/真模型能看到「上一任务抓了/放了什么」。历史超过
    max_history_messages 时按整组（assistant 调用 + user 观察）裁剪，不会拆散配对。

    纪元标记（TASK_EPOCH_JSON 加 {"task": "<task>"}）有两处：run 第一步的 prompt 里
    临时补一条（不落进 self.history），本 run 的第一条 assistant 消息里再内联写一份
    持久化。MockVLM 只把「最后一个标记之后」的观测当作当前任务状态
    （picked/placed/pick_fails/…），从而避免跨命令泄漏；内联写法不新增历史消息，
    因此历史仍严格 assistant/user 成对、长度断言不受影响。

健壮性约定（踩坑点）：
- 任何一步都不允许把整个 loop 崩掉：渲染失败跳过该相机；VLM 抛异常 -> 记一条失败结果并结束；
  校验失败 -> 记 error_code="invalid_call" 并继续（让 VLM 看到失败原因后自我纠正）；
  执行器抛异常 -> error_code="executor_exception"。
- history 只存 JSON 字符串消息（assistant 的调用 + user 的观察），
  这样既能原样发给 OpenAI 兼容端点，也能被 MockVLM 用括号平衡扫描反向解析。
- 每步 log 一行，便于无 GUI 环境下观察决策轨迹。
"""
from __future__ import annotations

import inspect
import json
import math
from typing import Any, Callable, Optional

from core.config import AGENT_MAX_STEPS, RENDER_HEIGHT, RENDER_WIDTH, TABLE_HEIGHT
from core.types import SceneState, SkillCall, SkillResult

from .schema import (DEFAULT_LOCATE_CAMERA, DEFAULT_PIXEL_CAMERA, record_correction,
                     take_corrections, validate_skill_call)
from .state import build_messages, obstruction_id, to_jsonable
from .taskplan import (bbox_top_midpoint, canonical_color, clear_pixel_cache, colors_in_text,
                       recall_pixel, remember_pixel)


#: P0：resolver 给的 z 与桌面平面相差小于这个值就不写 target_z_hint（保持旧行为）
Z_HINT_MIN_DELTA_M: float = 0.005
#: P0：locate 的 bbox 面积超过画面这个比例就认为不可信，不估尺寸
SIZE_BBOX_MAX_AREA_RATIO: float = 0.25
#: 同一特征对应：用 overview 色块框精修时，模型原点离框的最大距离（px）——超过就认为不是同一物体，不替换
_OVERVIEW_REFINE_MAX_OFFSET_PX: float = 40.0
#: P0：颜色规则不可靠的颜色不据其 bbox 估尺寸（sim-builder 正在修 COLOR_RULES）
_SIZE_UNRELIABLE_COLORS: tuple[str, ...] = ("gray", "white", "black")
#: P0/P1：腕部相机色块面积的诊断阈值（不再是主判据）。真机实测（Lead）：抓住 40mm 红方块时
#: wrist red_blob area=16384px（占满中央），空夹爪=0px；但**夹爪停在物体上方也能看到大片颜色**
#: （距顶 6cm -> 12527px），所以它单独用时是"近似恒真"的假阳性来源。
#: 现在主判据是 top 前后 check_gone（见 _top_gone_evidence），腕部用于：(a) 诊断指标；
#: (b) top 判据不可用时的兜底；(c) 物理证据例外的反向闸（腕部看不到目标色 -> 不放行）。
WRIST_GRASP_MIN_AREA_PX: int = 1500


class AgentLoop:
    """VLM 高层决策闭环。

    env       : 满足 core.interfaces.SimEnv（只用 get_state / render）
    executor  : core.interfaces.SkillExecutor（SkillCall -> SkillResult）
    vlm       : 满足 core.interfaces.VLMProtocol（decide(messages) -> SkillCall）
    cameras   : 每一步渲染的相机名，默认只看 overview（省 token / 省时间）
    max_steps : 最多决策步数，防止死循环
    log       : 日志函数（默认 print；传 None 可静音）
    history   : 跨命令历史种子（assistant 调用 + user 观察）；run(history=...) 优先于它，
                两者都没有时每次 run 清空（保持原行为）
    max_history_messages : 历史消息条数上限（默认 40），超出按整组裁剪
    pixel_resolver : 无真值模式：pixel_resolver(u,v,camera) -> 世界坐标 | None
    include_poses : 是否把物体真值位姿放进 prompt；False=no-oracle（默认 True，行为不变）
    locator   : 语义标定后端 locator(target, color, camera) -> dict（v2 §5.3）。
                收到 locate 调用时**不交给技能执行器**，而是走它，把结果记成
                SkillResult(skill="locate", metrics={"pixel":[u,v],"bbox":[...]}) 回灌；
                locator=None 时按**可重试**的 perception_error 处理。
    """

    #: 不可重试错误码：第 1 次出现允许重试 1 次，第 2 次出现立即终止闭环。
    NON_RETRYABLE_ERRORS: tuple[str, ...] = (
        "home_blocked", "stuck", "object_disturbed", "scene_reset",
    )

    #: 任务纪元标记：每个 run 的第一条 assistant 消息前缀，MockVLM 据此切分「当前任务」。
    #: 内容形如: TASK_EPOCH_JSON {"task": "<task 原文>"}
    EPOCH_PREFIX = "TASK_EPOCH_JSON"

    def __init__(self, env: Any, executor: Callable[[SkillCall], SkillResult], vlm: Any,
                 cameras: tuple[str, ...] = ("overview",),
                 max_steps: int = AGENT_MAX_STEPS, log: Optional[Callable[[str], None]] = print,
                 history: Optional[list[dict]] = None,
                 max_history_messages: int = 40,
                 pixel_resolver: Optional[Callable[[float, float, str], Any]] = None,
                 include_poses: bool = True,
                 locator: Optional[Callable[[str, Optional[str], str], Any]] = None):
        self.env = env
        self.executor = executor
        self.vlm = vlm
        self.cameras = tuple(cameras)
        self.max_steps = int(max_steps)
        self.log = log if callable(log) else (lambda _m: None)
        #: 无真值模式：pixel_resolver(u, v, camera) -> 世界坐标 [x,y(,z)] | None（INTERFACES §4.3）
        self.pixel_resolver = pixel_resolver
        #: 是否把物体真值位姿放进 prompt（False = no-oracle，配合 pixel_resolver，INTERFACES §4.4）
        self.include_poses = bool(include_poses)
        #: 语义标定后端 locator(target, color, camera) -> dict（v2 §5.3）；None = 未配置
        self.locator = locator
        #: ---------- 无真值抓取验证（P0：不许把「抓空」报成成功）----------
        #: 最近一次成功 locate 的 color（目标 -> color 的记忆表 + 最近值）
        self._locate_color: Optional[str] = None
        self._locate_colors: dict[str, str] = {}
        #: 被 §5.12-1 纠正掉的 target_id（仍可从 SCENE JSON 取它的 color 用于抓取验证）
        self._pick_relaxed_id: Optional[str] = None
        #: 当前夹持物体的颜色（用于 place 后的「真的放下了吗」腕部复核）；None = 不知道
        self._held_color: Optional[str] = None
        #: 抓取核验通过时记录的目标世界 xy（供日志/诊断；place 只注入 holding/holding_color）
        self._held_target_xy: Optional[list] = None
        #: 最近一次 pick 报 ok 且未被证伪（含「核验没做成」）-> place 时要按「手里有东西」处理，
        #: 否则 skills 会把持物当入口障碍撤退，撤退中把件弄掉（P0 实机根因）
        self._held_assumed: bool = False
        #: VLM 连续异常计数：允许连续 2 次 vlm_error 才中止（空回复/截断是瞬时的，别一步打死）
        self._vlm_error_streak: int = 0
        #: 跨命令历史种子（run(history=...) 优先）；None 表示不注入
        self._seed_history: Optional[list[dict]] = list(history) if history else None
        #: 历史消息条数上限（实例属性，便于调用方调整）
        self.max_history_messages = int(max_history_messages)
        self.history: list[dict] = []
        self.results: list[SkillResult] = []
        #: 连续失败检测：上一组 (skill, error_code, args 指纹) 及其连续次数
        self._repeat_key: Optional[tuple[str, str, str]] = None
        self._repeat_hits = 0
        #: 不可重试错误码在本轮任务里的累计出现次数
        self._fatal_hits: dict[str, int] = {}
        #: 同一失败已连续重复 >= 2 次时记下的 (skill, error_code, args 指纹)
        self._pending_repeat: Optional[tuple[str, str, str]] = None
        #: 本次 run 的纪元标记：只在第一条 assistant 消息里写一次
        self._epoch_task: Optional[str] = None
        self._epoch_marked = False

    # -- 内部工具 ---------------------------------------------------------- #
    def _render_all(self) -> dict[str, Any]:
        """渲染所有相机；单个相机失败只跳过并记录，不中断闭环。"""
        images: dict[str, Any] = {}
        for cam in self.cameras:
            try:
                images[cam] = self.env.render(cam)
            except Exception as e:  # 相机名不存在 / 渲染上下文问题
                self.log(f"[AgentLoop] 渲染相机 {cam!r} 失败，跳过: {e}")
        return images

    @staticmethod
    def _history_groups(messages: list[dict]) -> list[list[dict]]:
        """把历史切成「调用 + 观察」整组：assistant 起一组，其后的非 assistant 消息归入同组。"""
        groups: list[list[dict]] = []
        for m in messages:
            role = m.get("role") if isinstance(m, dict) else None
            if role == "assistant" or not groups:
                groups.append([m])
            else:
                groups[-1].append(m)
        return groups

    @classmethod
    def _has_epoch_marker(cls, msg: Any) -> bool:
        """一条历史消息里是否带任务纪元标记。"""
        if not isinstance(msg, dict):
            return False
        content = msg.get("content")
        return isinstance(content, str) and cls.EPOCH_PREFIX in content

    def _trim_history(self, messages: list[dict]) -> list[dict]:
        """把历史裁到 max_history_messages 条以内，只能整组丢（不拆散 assistant/user 配对）。

        保留「最早一组 + 最近若干整组」；额外硬约束：**最后一个纪元标记所在的整组及其后
        所有整组必须保留**——否则 MockVLM 会把上一纪元的历史误当成当前任务（D2/D3 复现）。
        纪元标记计入配额；当前纪元整段优先于条数上限。
        """
        limit = int(self.max_history_messages)
        msgs = list(messages)
        if limit <= 0 or len(msgs) <= limit:
            return msgs
        groups = self._history_groups(msgs)
        if len(groups) <= 1:
            return msgs                          # 只有一组，裁掉就没上下文了
        keep_from = len(groups)                  # 最后一个纪元标记所在组；无标记则为 len
        for i, g in enumerate(groups):
            if any(self._has_epoch_marker(m) for m in g):
                keep_from = i
        head: list[dict] = []
        k = 0
        while k < keep_from and len(head) < 2:   # 「最早 2 条」锚点（按整组取）
            head += groups[k]
            k += 1
        budget = max(limit - len(head), 0)
        tail_idx: list[int] = list(range(keep_from, len(groups)))   # 当前纪元整段必须保留
        used = sum(len(groups[i]) for i in tail_idx)
        i = keep_from - 1
        while i >= k:                            # 预算内再往前多带几组
            if used + len(groups[i]) > budget:
                break
            tail_idx.insert(0, i)
            used += len(groups[i])
            i -= 1
        if not tail_idx and k < len(groups):
            tail_idx = [len(groups) - 1]         # 至少保留最近一组，最新观察不能丢
        return head + [m for i in tail_idx for m in groups[i]]

    def _append_history(self, call: SkillCall, result: SkillResult) -> None:
        """把「动作」与「观察」都塞进 history，作为后续步骤的上下文（并按上限裁剪）。

        本次 run 的**第一条** assistant 消息前面带任务纪元标记 TASK_EPOCH_JSON，
        供 MockVLM / 真模型切分「当前任务 vs 上一条命令」。标记内联在已有消息里、
        不新增消息，assistant/user 仍严格成对（长度/配对断言不受影响）。
        """
        content = json.dumps(to_jsonable(call.to_json()), ensure_ascii=False)
        if self._epoch_task is not None and not self._epoch_marked:
            marker = json.dumps({"task": self._epoch_task}, ensure_ascii=False)
            content = f"{self.EPOCH_PREFIX} {marker}\n{content}"
            self._epoch_marked = True
        self.history.append({"role": "assistant", "content": content})
        self.history.append({
            "role": "user",
            "content": "OBSERVATION_JSON " + json.dumps(to_jsonable(result.to_observation()), ensure_ascii=False),
        })
        self.history = self._trim_history(self.history)

    def _attach_failure_observation(self, call: SkillCall, result: SkillResult,
                                    state: SceneState) -> None:
        """把技能失败上下文并入 result.metrics，随 observation 回灌给 VLM（U1/R3）。

        只做**观测补全**：失败原因 / 阻挡物 / 接触力 / 建议图像；
        **不**修改调用参数、也**不**自行重试——重新规划必须由 LLM 决定。
        skills 侧已给 obstruction_id 时原样保留，没有才用本地几何关系兜底。
        """
        metrics = dict(result.metrics or {})
        metrics.setdefault("failure_reason", result.message)
        if not metrics.get("obstruction_id"):
            blocker = obstruction_id(state, call)
            if blocker:
                metrics["obstruction_id"] = blocker
        try:
            force = state.contact.force
            metrics.setdefault(
                "contact_force_norm",
                round(float(sum(float(v) * float(v) for v in force)) ** 0.5, 3))
        except Exception:  # 接触力字段异常不影响主流程
            pass
        if self.cameras:
            metrics.setdefault("suggested_cameras", list(self.cameras))
        result.metrics = metrics

    @staticmethod
    def _args_signature(call: SkillCall) -> str:
        """调用参数的稳定指纹；参数一变就必须放行执行（D1：改正后的像素调用不能被误杀）。"""
        try:
            return json.dumps(to_jsonable(call.args or {}), ensure_ascii=False, sort_keys=True)
        except Exception:
            return repr(call.args)

    def _note_failure(self, skill: str, error_code: Optional[str],
                      args_sig: str = "") -> Optional[str]:
        """记录一步失败；返回「立即终止」的原因字符串，None 表示可继续。

        规则（防止 VLM 对着同一个必然失败的动作空转到 max_steps）：
        1) 同一 skill + 同一 error_code + **同一参数指纹**连续重复 >= 2 次：不立即结束，
           而是记下 _pending_repeat；下一步若 VLM 仍提议同一个 skill 且参数**完全相同**，
           才在执行前终止。参数变了（例如先给 target_id+target_pixel 被拒，再改成纯像素）
           一律放行执行——否则改正后的正确调用会被守卫误杀；
        2) error_code 属于 NON_RETRYABLE_ERRORS：第 1 次允许重试，第 2 次出现立即终止。
        成功步（error_code 为空）会清空以上状态。
        """
        if not error_code:
            self._repeat_key = None
            self._repeat_hits = 0
            self._pending_repeat = None
            return None

        key = (skill, error_code, args_sig)
        if key == self._repeat_key:
            self._repeat_hits += 1
        else:
            self._repeat_key = key
            self._repeat_hits = 1
        if self._repeat_hits >= 2:
            self._pending_repeat = key

        if error_code in self.NON_RETRYABLE_ERRORS:
            self._fatal_hits[error_code] = self._fatal_hits.get(error_code, 0) + 1
            if self._fatal_hits[error_code] >= 2:
                return (f"不可重试错误码 {error_code} 第 {self._fatal_hits[error_code]} 次出现"
                        f"（最多允许重试 1 次）")
        return None

    @staticmethod
    def _parse_uv(raw: Any, label: str):
        """把像素参数解析成 (u, v)；失败返回 (None, 错误信息)。"""
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return None, f"{label} 必须是 [u,v] 两个数，实际 {raw!r}"
        try:
            return (float(raw[0]), float(raw[1])), None
        except (TypeError, ValueError):
            return None, f"{label} 必须是数值 [u,v]，实际 {raw!r}"

    def _resolver_supports_dual(self) -> bool:
        """pixel_resolver 是否支持双目扩展（§5.10）。

        契约里 pixel_resolver 是 (u,v,camera)；双目三角定位需要一次拿到两个像素（§5.10）。
        为了不擅自改冻结签名，这里用签名探测，满足任意一条就认为支持双目：
        - 显式有 overview_uv 参数（如 lambda u,v,cam,overview_uv=None）；
        - 有 **kwargs（demo 的 _pixel_resolver(u, v, cam, *extra, **kw) 就是这种）；
        - 有 >= 4 个必填位置参数。
        否则退回单目（日志里标注 single_view）。
        """
        if self.pixel_resolver is None:
            return False
        try:
            params = inspect.signature(self.pixel_resolver).parameters
        except (TypeError, ValueError):
            return False
        if "overview_uv" in params:
            return True
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            return True
        required_positional = [p for p in params.values()
                               if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                               and p.default is p.empty]
        return len(required_positional) >= 4

    def _call_pixel_resolver(self, u: float, v: float, camera: str,
                             overview_uv: Optional[tuple] = None) -> Any:
        """调用 pixel_resolver：不支持双目时退回 §4.3 的三参签名。"""
        if overview_uv is None or not self._resolver_supports_dual():
            return self.pixel_resolver(u, v, camera)
        try:
            params = inspect.signature(self.pixel_resolver).parameters
        except (TypeError, ValueError):
            params = {}
        accepts_kw = ("overview_uv" in params
                      or any(p.kind == p.VAR_KEYWORD for p in params.values()))
        if accepts_kw:
            # §5.10 冻结调用形式：pixel_resolver(u, v, camera, overview_uv=(u2, v2))
            return self.pixel_resolver(u, v, camera, overview_uv=list(overview_uv))
        return self.pixel_resolver(u, v, camera, list(overview_uv))

    @staticmethod
    def _world_from_result(raw: Any, metrics: dict, u: float, v: float, camera: str):
        """把 resolver 返回值归一成 (世界坐标 seq, None) 或 (None, 错误信息)。

        支持 §4.3 的数组返回与 §5.10 的 dict 返回（{"ok","xyz","residual_mm","source","why"}）；
        residual>15 mm 之类的不可信判定由 perception.object_point 置 ok=False，这里原样上报为
        可重试的 perception_error。
        """
        if raw is None:
            return None, (f"像素 ({u:g},{v:g}) [{camera}] 反投影失败"
                          f"（可能落在桌外或在相机后方），请重新给像素或改用 top 相机")
        if isinstance(raw, dict):
            metrics["source"] = raw.get("source")
            if raw.get("residual_mm") is not None:
                metrics["residual_mm"] = raw.get("residual_mm")
            if not raw.get("ok", True):
                why = str(raw.get("why") or "双目三角定位不可信")
                res = raw.get("residual_mm")
                extra = f"（residual={float(res):.1f} mm）" if isinstance(res, (int, float)) else ""
                return None, f"像素 ({u:g},{v:g}) [{camera}] 定位失败: {why}{extra}"
            point = raw.get("xyz", raw.get("point"))
            if point is None:
                return None, f"pixel_resolver 返回的 dict 既不是 ok=False 也没有 xyz: {raw!r}"
            raw = point
        try:
            seq = [float(x) for x in raw]
        except (TypeError, ValueError):
            return None, f"pixel_resolver 返回了非数值结果 {raw!r}"
        if len(seq) < 2:
            return None, f"pixel_resolver 返回结果不足 2 维: {raw!r}"
        return seq, None

    def _resolve_pixel(self, call: SkillCall, state: SceneState) -> Optional[dict]:
        """把像素目标反投影成世界 xy 并改写调用（§4.3 + §5.10 + §5.12-1）。

        - 支持 target_pixel（top）与 target_pixel_overview（overview）成对给出（双目三角定位），
          只给 overview 也允许；resolver 不支持双目时退回单目并在日志标注 single_view。
        - §5.12-1：无真值模式以像素为准 —— pick 同时给 target_id/target_xy 时在这里丢掉它们，
          并记一条 corrected_args（validate_skill_call 里有同样的兜底，供直接调用方使用）。
        - 成功：原地把像素换成 target_xy（pick）/ xy（place）并返回 None；
          失败：返回 {"message","metrics"}，由调用方记成**可重试**的 perception_error。
        MockVLM 的 oracle 路径不带像素，这里直接返回 None，行为不变。
        """
        args = dict(call.args or {})
        top_raw = args.get("target_pixel")
        ov_raw = args.get("target_pixel_overview")
        if call.skill == "pick":
            # 每次 pick 调用都重算"模型点名的目标 id"（下方 target_id 被丢弃后仍要用它
            # 交叉校验抓取验证里"夹爪夹的是不是这个物体"；旧值不得跨调用残留）。
            tid_arg = args.get("target_id")
            self._pick_relaxed_id = (str(tid_arg) if isinstance(tid_arg, str)
                                     and tid_arg.strip() else None)
        if top_raw is None and ov_raw is None:
            return None
        camera = str(args.get("pixel_camera") or DEFAULT_PIXEL_CAMERA)
        metrics: dict = {"pixel_camera": camera}
        if top_raw is not None:
            metrics["target_pixel"] = list(top_raw) if isinstance(top_raw, (list, tuple)) else top_raw
        if ov_raw is not None:
            metrics["target_pixel_overview"] = (list(ov_raw) if isinstance(ov_raw, (list, tuple))
                                                else ov_raw)

        # §5.12-1：像素在场时，冗余的 target_id / target_xy 丢掉（无真值模式以像素为准）
        if call.skill == "pick":
            if isinstance(args.get("target_id"), str) and args["target_id"].strip():
                self._pick_relaxed_id = str(args["target_id"])   # 抓取验证要用它的 color
                args.pop("target_id", None)
                record_correction(call, "target_id", "dropped",
                                  "同时给了像素；无真值模式下 target_id 不含可执行信息，以像素为准",
                                  "[validate] 冗余 target_id 已忽略（无真值模式以像素为准）")
            if args.get("target_xy") is not None:
                args.pop("target_xy", None)
                record_correction(call, "target_xy", "dropped", "同时给了像素，以像素为准",
                                  "[validate] 冗余 target_xy 已忽略（已有像素，以像素为准）")

        top_uv = ov_uv = None
        if top_raw is not None:
            top_uv, err = self._parse_uv(top_raw, "target_pixel")
            if err:
                return {"message": err, "metrics": metrics}
        if ov_raw is not None:
            ov_uv, err = self._parse_uv(ov_raw, "target_pixel_overview")
            if err:
                return {"message": err, "metrics": metrics}
        # 主像素优先 top（俯视最稳）；只有 overview 时就用 overview（日志标注单目）
        if top_uv is not None:
            pu, pv, pcam = top_uv[0], top_uv[1], camera
        else:
            pu, pv, pcam = ov_uv[0], ov_uv[1], "overview"
        # 同一特征对应：双目三角时 overview 像素优先用「bbox 顶边中点（顶面中心）」，
        # 而不是模型容易点偏的瓶身中部；没有 bbox 就沿用模型点（不瞎猜）。
        if top_uv is not None and ov_uv is not None:
            alt, how = self._overview_top_uv(call, state, ov_uv)
            if alt is not None and (abs(alt[0] - ov_uv[0]) > 0.5 or abs(alt[1] - ov_uv[1]) > 0.5):
                metrics["overview_uv_model"] = [round(float(ov_uv[0]), 2), round(float(ov_uv[1]), 2)]
                metrics["overview_uv_used"] = list(alt)
                metrics["overview_uv_source"] = how
                self.log(f"[AgentLoop] 双目 overview 像素改用 bbox 顶边中点 {alt}（{how}），"
                         f"模型原点 {[round(float(ov_uv[0]), 2), round(float(ov_uv[1]), 2)]}"
                         f" —— 两视角对齐到「物体顶面中心」同一特征")
                ov_uv = (float(alt[0]), float(alt[1]))
        if self.pixel_resolver is None:
            return {"message": (f"当前没有像素反投影能力（pixel_resolver 未配置），无法把像素 "
                                f"({pu:g},{pv:g}) [{pcam}] 变成世界坐标；"
                                f"请改用 target_id/target_xy 或配置感知后端"),
                    "metrics": metrics}
        try:
            raw = self._call_pixel_resolver(pu, pv, pcam,
                                            overview_uv=(ov_uv if top_uv is not None else None))
        except Exception as e:  # 相机后方/标定异常等都不许崩
            return {"message": f"像素 ({pu:g},{pv:g}) [{pcam}] 反投影异常: {e}", "metrics": metrics}
        seq, err = self._world_from_result(raw, metrics, pu, pv, pcam)
        if err:
            return {"message": err, "metrics": metrics}

        x, y = seq[0], seq[1]
        z = seq[2] if len(seq) > 2 else float(TABLE_HEIGHT)
        new_args = {k: val for k, val in args.items()
                    if k not in ("target_pixel", "target_pixel_overview", "pixel_camera")}
        if call.skill == "pick":
            new_args["target_xy"] = [round(x, 6), round(y, 6)]
            # P0-a：三角/显式 z 作为 target_z_hint 传给技能（高物体预抓取高度）。
            # 只 pick 写；z 与桌面平面相差 < 5mm 时视为「就是桌面」，保持旧行为不写。
            if abs(float(z) - float(TABLE_HEIGHT)) >= Z_HINT_MIN_DELTA_M:
                new_args["target_z_hint"] = round(float(z), 6)
                metrics["z_hint"] = new_args["target_z_hint"]
                self.log(f"[AgentLoop] 用定位 z={z:.3f} 作为 target_z_hint"
                         f"（桌面 {TABLE_HEIGHT:.3f}，避免预抓取位落进高物体内部）")
            # P0-b：locate 的 bbox -> footprint 半宽，喂 target_size（闭合量按真实尺寸算）
            size_est = self._estimate_target_size(call, state)
            if size_est is not None:
                new_args["target_size"] = size_est
                metrics["target_size_est"] = list(size_est)
                self.log(f"[AgentLoop] 用 locate bbox 估尺寸 target_size={size_est}"
                         f"（闭合量按真实尺寸算）")
        else:
            # place 不写 z_hint / size（落点高度由技能按容器/锚点决定）
            new_args["xy"] = [round(x, 6), round(y, 6)]
        call.args = new_args
        dual_given = top_uv is not None and ov_uv is not None
        dual_used = dual_given and self._resolver_supports_dual()
        if dual_used and not metrics.get("source"):
            metrics["source"] = "triangulate"
        if not metrics.get("source"):
            metrics["source"] = "single_view"
        residual = metrics.get("residual_mm")
        # 成功的像素解析指标也要能观测（UI/verifier 会读 source/residual_mm）
        self._pixel_metrics = dict(metrics)
        self.log(f"[AgentLoop] 像素 ({pu:g},{pv:g}) [{pcam}] -> 世界 ({x:.3f}, {y:.3f}, {z:.3f})")
        if dual_used:
            self.log(f"[AgentLoop] 双目像素 top({top_uv[0]:g},{top_uv[1]:g}) + "
                     f"overview({ov_uv[0]:g},{ov_uv[1]:g}) source={metrics['source']}"
                     + (f" residual={float(residual):.1f}mm" if isinstance(residual, (int, float)) else ""))
        elif dual_given:
            self.log("[AgentLoop] 同时给了 top+overview 像素，但 pixel_resolver 不支持双目扩展，"
                     "已按 top 单目回退 source=single_view（高物体注意视差）")
        elif ov_uv is not None:
            self.log("[AgentLoop] 仅 overview 单目像素 source=single_view（高物体注意视差）")
        else:
            self.log(f"[AgentLoop] 单目 top 像素 source=single_view(z={z:.3f})")
        return None

    def _collect_corrections(self, call: SkillCall) -> list[dict]:
        """取出参数纠正并逐条打日志（§5.12-5）；没有纠正返回 []。"""
        items = take_corrections(call)
        for item in items:
            self.log(str(item.get("log") or f"[validate] 已纠正参数 {item.get('key')}: {item.get('detail')}"))
        return items

    # -- 无真值抓取验证（P0） ------------------------------------------------ #
    def _render_camera(self, camera: str) -> Any:
        """单独渲染某相机；失败返回 None（绝不抛）。"""
        try:
            return self.env.render(camera)
        except Exception as e:
            self.log(f"[AgentLoop] 渲染相机 {camera!r} 失败: {e}")
            return None

    def _pick_target_color(self, call: SkillCall, state: SceneState) -> tuple[Optional[str], str]:
        """抓取验证用的目标颜色 (color, 来源)，取不到返回 (None, "none")。

        优先级（P0 要求）：
          1) 最近一次 locate(target, color) 的 color —— 但若本次 pick 的 target_id 也在 SCENE 里
             有颜色且两者不一致，以 SCENE 的颜色为准（避免「先 locate 托盘再抓红块」串色）；
          2) 被 §5.12-1 纠正掉的 target_id（或仍在 args 里的 target_id）在 SCENE JSON 里的 color；
          3) 仍无 -> (None, "none")，调用方必须明确日志「抓取未验证（未知颜色）」。
        """
        args = call.args or {}
        tid = self._pick_relaxed_id
        if not (isinstance(tid, str) and tid):
            raw = args.get("target_id")
            tid = raw if isinstance(raw, str) and raw else None
        scene_color = None
        if tid:
            obj = state.find(tid)
            if obj is not None:
                scene_color = str(obj.color)
        if self._locate_color and (scene_color is None or self._locate_color == scene_color):
            return str(self._locate_color), "locate"
        if scene_color:
            return scene_color, f"scene:{tid}"
        if self._locate_color:
            return str(self._locate_color), "locate"
        return None, "none"

    def _infer_color_at(self, hint_uv: Any, state: SceneState, rgb: Any,
                        *, max_dist_px: float = 60.0) -> Optional[str]:
        """用模型给的像素反查该处物体的颜色（纯图像，绝不读真值）。

        为什么需要：实机里模型经常只给像素（不 locate、不点名 id），若因此跳过抓取验证，
        「抓空报成功」就治不住。候选颜色取 SCENE JSON 里出现过的颜色（模型本来就能看到
        id/color/category），对每个候选做一次 color_blob(hint_uv=hint)，取质心离 hint
        最近且 <= max_dist_px 的第一个命中。找不到返回 None（调用方按「未知颜色」处理）。
        颜色规则暂不可靠的颜色（gray/white/black）不参与推断，避免张冠李戴。
        """
        try:
            from sim import perception as _perception
        except Exception:
            return None
        cands: list[str] = []
        for obj in getattr(state, "objects", None) or []:
            canon = canonical_color(getattr(obj, "color", None))
            if canon and canon not in cands and canon not in _SIZE_UNRELIABLE_COLORS:
                cands.append(canon)
        try:
            hx, hy = float(hint_uv[0]), float(hint_uv[1])
        except (TypeError, ValueError, IndexError):
            return None
        for color in cands[:6]:
            try:
                blob = _perception.color_blob(rgb, color, hint_uv=[hx, hy])
            except Exception:
                continue
            if isinstance(blob, dict):
                found, uv = bool(blob.get("found")), blob.get("uv")
            else:
                found, uv = bool(getattr(blob, "found", None)), getattr(blob, "uv", None)
            if not (found and isinstance(uv, (list, tuple)) and len(uv) == 2):
                continue
            if math.hypot(float(uv[0]) - hx, float(uv[1]) - hy) <= max_dist_px:
                return color
        return None

    def _estimate_target_size(self, call: SkillCall, state: SceneState) -> Optional[list]:
        """用 locate 记住的 bbox 估物体 footprint 半宽 (hx, hy)；不可信/拿不到返回 None。

        不可信判据（P0-2）：
        - bbox 面积 > 画面 25%（明显把整片桌面框进来）；
        - bbox 退化（宽或高 <= 1px）；
        - 颜色属 gray/white/black（sim-builder 正在修 COLOR_RULES，误检可达几十 cm）；
        - 估出来的半宽不在 (2mm, 250mm] 这种正常物体尺度内。
        只用 sim.perception.bbox_to_world（不读任何物体真值）。
        """
        color, _src = self._pick_target_color(call, state)
        canon = canonical_color(color)
        if not canon or canon in _SIZE_UNRELIABLE_COLORS:
            return None
        hit = recall_pixel(canon, camera="top")
        bbox = (hit or {}).get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return None
        try:
            u1, v1, u2, v2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError):
            return None
        w, h = abs(u2 - u1), abs(v2 - v1)
        if w <= 1.0 or h <= 1.0:
            return None
        frame = float(max(RENDER_WIDTH, 1) * max(RENDER_HEIGHT, 1))
        if (w * h) / frame > SIZE_BBOX_MAX_AREA_RATIO:
            self.log(f"[AgentLoop] locate bbox 面积 {100.0 * (w * h) / frame:.0f}% > "
                     f"{100.0 * SIZE_BBOX_MAX_AREA_RATIO:.0f}% 画面，不据此估尺寸")
            return None
        try:
            from sim import perception as _perception
            world = _perception.bbox_to_world(self.env, list(bbox), camera="top")
        except Exception as e:
            self.log(f"[AgentLoop] bbox_to_world 失败（不估尺寸）: {e}")
            return None
        if not isinstance(world, dict) or not world.get("ok"):
            return None
        size = world.get("size_xy")
        if not (isinstance(size, (list, tuple)) and len(size) == 2):
            return None
        try:
            hx, hy = abs(float(size[0])) / 2.0, abs(float(size[1])) / 2.0
        except (TypeError, ValueError):
            return None
        if not (0.002 <= hx <= 0.25 and 0.002 <= hy <= 0.25):
            return None
        return [round(hx, 6), round(hy, 6)]

    def _overview_top_uv(self, call: SkillCall, state: SceneState,
                         model_uv: Any) -> tuple[Optional[list], str]:
        """同一特征对应：三角定位用的 overview 像素优先取「bbox 顶边中点（=顶面中心）」。

        为什么：top 视角里模型点的是顶面中心，overview 视角里模型常点到瓶身中部（水瓶实测低
        45px），两视角指的不是同一物理点 -> 三角偏 25mm -> 侧壁碰撞。用 bbox 顶边中点对齐。

        来源顺序（都不猜）：
          1) 共享缓存里该目标/颜色的 overview bbox（动作阶段 locate 标定过）；
          2) 本次 overview 图对该颜色做一次 color_blob —— 但要求模型原点贴着该框
             （<= _OVERVIEW_REFINE_MAX_OFFSET_PX），只做「精修」不推翻模型。
        返回 (uv, 来源说明)；拿不到返回 (None, "")，调用方沿用模型给的点。
        """
        color, _src = self._pick_target_color(call, state)
        bbox = None
        how = ""
        if color:
            hit = recall_pixel(color, camera="overview")
            bbox = (hit or {}).get("bbox")
            if bbox:
                how = f"缓存 overview bbox(target={hit.get('target')})"
        if bbox is None and color:
            try:
                from sim import perception as _perception
                img = self._render_camera("overview")
                blob = _perception.color_blob(img, color) if img is not None else None
            except Exception:
                blob = None
            if isinstance(blob, dict):
                found, bbox = bool(blob.get("found")), blob.get("bbox")
            else:
                found = bool(getattr(blob, "found", None))
                bbox = getattr(blob, "bbox", None)
            if not found:
                bbox = None
            if bbox is not None:
                # 只做「精修」：模型原点必须贴着这个框，否则可能有串色，放弃替换
                try:
                    u1, v1, u2, v2 = (float(bbox[0]), float(bbox[1]),
                                      float(bbox[2]), float(bbox[3]))
                    mx, my = float(model_uv[0]), float(model_uv[1])
                    dx = max(u1 - mx, 0.0, mx - u2)
                    dy = max(v1 - my, 0.0, my - v2)
                    if math.hypot(dx, dy) > _OVERVIEW_REFINE_MAX_OFFSET_PX:
                        return None, ""
                except Exception:
                    return None, ""
                how = "本次 overview 色块框"
        if bbox is None:
            return None, ""
        return bbox_top_midpoint(bbox), how

    def _pick_hint_uv(self, call: SkillCall) -> Optional[list]:
        """抓取验证用的 top 图像素：模型给的 top 像素 > 反投影后的世界点投回 top。"""
        pm = self._pixel_metrics or {}
        pix = pm.get("target_pixel")
        cam = str(pm.get("pixel_camera") or DEFAULT_PIXEL_CAMERA)
        if isinstance(pix, (list, tuple)) and len(pix) == 2 and cam == "top":
            try:
                return [float(pix[0]), float(pix[1])]
            except (TypeError, ValueError):
                return None
        xy = (call.args or {}).get("target_xy")
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            try:
                from sim.vision import project_point
                z = (call.args or {}).get("target_z_hint")
                z = float(z) if isinstance(z, (int, float)) else float(TABLE_HEIGHT)
                uv = project_point(self.env, [float(xy[0]), float(xy[1]), z], "top")
                if uv is not None:
                    return [float(uv[0]), float(uv[1])]
            except Exception as e:   # 投影失败不阻塞主流程（会记成「未验证」）
                self.log(f"[AgentLoop] 抓取验证：世界点投回 top 失败（{e}）")
        return None

    def _wrist_blob(self, color: str) -> tuple[Optional[dict], str]:
        """渲染 wrist 相机并在夹爪处找 color 色块（诊断/兜底信号）。

        返回 ({"found","uv","area_px"} | None, 错误说明)。None = 这次核验做不了。
        真机实测：抓住 40mm 方块时 area≈16384px，空夹爪 0px；但夹爪停在物体上方时也能看到
        大片颜色（6cm -> 12527px），所以它单独用时是"近似恒真"的假阳性来源，只能当诊断指标、
        top 判据不可用时的兜底、以及物理证据例外的反向闸（见 _verify_pick_grasp）。
        """
        try:
            from sim import perception as _perception
        except Exception as e:
            return None, f"sim.perception 不可用：{e}"
        img = self._render_camera("wrist")
        if img is None:
            return None, "wrist 相机渲染失败"
        try:
            blob = _perception.color_blob(img, color)
        except Exception as e:
            return None, f"wrist color_blob 异常: {e}"
        if isinstance(blob, dict):
            found, uv, area = blob.get("found"), blob.get("uv"), blob.get("area_px")
        else:
            found, uv = getattr(blob, "found", None), getattr(blob, "uv", None)
            area = getattr(blob, "area_px", None)
        norm_uv = None
        if isinstance(uv, (list, tuple)) and len(uv) == 2:
            try:
                norm_uv = [round(float(uv[0]), 2), round(float(uv[1]), 2)]
            except (TypeError, ValueError):
                norm_uv = None
        try:
            norm_area = int(area) if area is not None else 0
        except (TypeError, ValueError):
            norm_area = 0
        return {"found": bool(found), "uv": norm_uv, "area_px": norm_area}, ""

    def _lookup_held_color_by_wrist(self, state: SceneState) -> Optional[str]:
        """腕部反查「手里到底是什么颜色」：对 SCENE 里出现过的颜色逐个查 wrist 色块，
        面积 >= 阈值就认定夹爪里夹的是它。

        用于 target_xy/未知颜色路径（模型没给像素、也没有 locate）：这类 pick 无法在抓取时
        定颜色，但 place 之前腕部图能**正向**证明手里有东西 —— 有证据才注入 holding。
        找不到返回 None（调用方按「不注入」处理，绝不瞎猜）。
        """
        try:
            from sim import perception as _perception
        except Exception:
            return None
        img = self._render_camera("wrist")
        if img is None:
            return None
        cands: list[str] = []
        for obj in getattr(state, "objects", None) or []:
            canon = canonical_color(getattr(obj, "color", None))
            if canon and canon not in cands and canon not in _SIZE_UNRELIABLE_COLORS:
                cands.append(canon)
        for color in cands[:6]:
            try:
                blob = _perception.color_blob(img, color)
            except Exception:
                continue
            if isinstance(blob, dict):
                found, area = bool(blob.get("found")), int(blob.get("area_px") or 0)
            else:
                found = bool(getattr(blob, "found", None))
                area = int(getattr(blob, "area_px", None) or 0)
            if found and area >= WRIST_GRASP_MIN_AREA_PX:
                return color
        return None

    def _top_gone_evidence(self, color: str, hint_uv: Any, rgb_before: Any) -> tuple:
        """P1 主判据：top 视角前后对比，目标色块是否已离开原位（= 真被拿走）。

        返回 (判定, metrics)：判定 True/False/None。None = 这次判不了（无 before 图 /
        无 hint / top 渲染失败 / 感知异常），调用方按"降级"处理。

        为什么换掉腕部：腕部相机挂在夹爪根部俯视正下方，只要夹爪停在物体上方就能看到
        大片颜色（实测张开夹爪、方块留在桌面时：距顶 6 cm -> 12527 px、19 cm -> 1299 px），
        是"近似恒真"的假阳性来源。top 前后对比问的是"物体还在不在原位"，这才是
        「抓走了」的语义。
        """
        metrics: dict = {}
        if rgb_before is None or hint_uv is None:
            return None, metrics
        try:
            from sim import perception as _perception
            rgb_after = self._render_camera("top")
            if rgb_after is None:
                return None, metrics
            res = _perception.check_gone(rgb_before, rgb_after, color, hint_uv)
        except Exception as e:
            self.log(f"[AgentLoop] 抓取验证：top check_gone 不可用（{e}）")
            return None, metrics
        if not isinstance(res, dict) or res.get("ok") is None:
            return None, metrics
        metrics["grasp_top_check_gone"] = bool(res.get("ok"))
        metrics["grasp_top_why"] = res.get("why")
        m = res.get("metrics")
        if isinstance(m, dict):
            metrics["grasp_top_area_before_px"] = m.get("area_before_px")
            metrics["grasp_top_area_after_px"] = m.get("area_after_px")
        return bool(res.get("ok")), metrics

    def _verify_pick_grasp(self, color: str, color_src: str, hint_uv: Any, rgb_before: Any,
                           skill_metrics: Any = None) -> Optional[dict]:
        """技能自称抓取成功后独立确认「物体真的被拿走了」（P1：主判据 = top check_gone）。

        - top 前后对比：原位置色块消失/面积掉到 1/3 以下 -> grasp_verified=True；
        - top 前后对比：色块仍在原位 -> 默认判 grasp_not_confirmed（可重试，调用方改写，
          绝不 ok=True）；**例外**：技能自带物理随动证据（抬升 ok 且夹爪接触的物体已离地，
          `grasp_lift_ok` + `grasp_carry_evidence=True`）时判成功并记
          `grasp_verify_note`。为什么需要这个例外：top 是否看得见物体取决于夹爪有没有把它
          挡住 —— 实测蓝块被夹起来后 top 图里面积 818->827（没被挡），纯 check_gone 会把
          真抓成功判成失败；而"夹爪接触 + 物体离地"是物理事实，比像素更硬。
        - 该例外还有一道反向闸：腕部**可用且看不到目标颜色**时不允许放行 —— 物理证据只说
          明"手里夹着某个可动物体"，不说明夹的是目标物体；腕部看不到目标色块说明夹的多半
          是别的东西。腕部不可用时才只看物理证据。
        - top 判据做不了（无 before 图 / 无 hint / 渲染失败）-> 退回腕部色块（旧语义）；
        - 两者都做不了 -> 「未验证」（grasp_verified=None + grasp_unverified=True），
          不判成功也不判失败。
        腕部面积一律作为诊断指标保留（grasp_wrist_area_px / grasp_verify_area_px），
        不再单独决定成败。返回 None = 连「未验证」都不用记（调用方按未知颜色处理）。
        """
        metrics: dict = {"grasp_verify_color": color,
                         "grasp_verify_color_source": color_src}
        # 真值路径（技能用 metrics.moved >= 20mm 判成功）比任何图像判据都硬：再做图像复核
        # 只会引入假失败（实测蓝块真抓 Δz=+119.5mm，但 top 面积 818->805 会被 check_gone
        # 判成"仍在原位"）。所以直接采信技能的抬升位移。
        mv = (skill_metrics or {}).get("moved")
        if (isinstance(mv, (int, float)) and not isinstance(mv, bool)
                and float(mv) >= 0.02):
            metrics.update({"grasp_verify_camera": "skill_truth",
                            "grasp_verified": True,
                            "grasp_verify_note": f"技能真值抬升 {float(mv) * 1000:.0f}mm（oracle 路径）"})
            self.log(f"[AgentLoop] 抓取已验证 grasp_verified=True（技能真值抬升 "
                     f"{float(mv) * 1000:.0f}mm，oracle 路径不再做图像复核）")
            return {"ok": True, "message": "", "metrics": metrics}
        wrist, err = self._wrist_blob(color)
        if wrist is not None:
            metrics["grasp_wrist_area_px"] = int(wrist.get("area_px") or 0)
            metrics["grasp_wrist_uv"] = wrist.get("uv")
            metrics["grasp_wrist_found"] = bool(wrist.get("found"))
        metrics["grasp_skill_lift_ok"] = (skill_metrics or {}).get("grasp_lift_ok")
        metrics["grasp_skill_carry_evidence"] = (skill_metrics or {}).get("grasp_carry_evidence")
        carry_bodies = [str(b) for b in ((skill_metrics or {}).get("grasp_carry_bodies") or [])]
        metrics["grasp_skill_carry_bodies"] = carry_bodies
        target_id = getattr(self, "_pick_relaxed_id", None)
        bodies_ok = bool(carry_bodies)
        if bodies_ok and target_id:
            # 知道目标 id 时，夹爪接触的必须是它本人（防"夹着别的东西"被物理证据放行）
            bodies_ok = any(str(target_id).lower() == b.lower() for b in carry_bodies)
        has_physical = (bool((skill_metrics or {}).get("grasp_lift_ok"))
                        and (skill_metrics or {}).get("grasp_carry_evidence") is True)
        skill_physical = has_physical and bodies_ok
        # 物理证据只证明"手里夹着某个可动物体"，不证明夹的是**目标颜色**那个：
        # 腕部若可用且看不到目标色块，说明夹的多半是别的东西 -> 不许用物理证据放行。
        wrist_confirms_target = bool(wrist is not None and wrist.get("found")
                                     and int(wrist.get("area_px") or 0) >= WRIST_GRASP_MIN_AREA_PX)
        wrist_contradicts = bool(wrist is not None and not wrist_confirms_target)
        gone, gone_metrics = self._top_gone_evidence(color, hint_uv, rgb_before)
        metrics.update(gone_metrics)
        area_for_compat = int(metrics.get("grasp_wrist_area_px") or 0)
        physical_exception = (gone is False and skill_physical and not wrist_contradicts)
        if gone is True or physical_exception:
            metrics["grasp_verified"] = True
            metrics["grasp_verify_camera"] = "top"
            metrics["grasp_verify_area_px"] = area_for_compat
            if gone is True:
                self.log(f"[AgentLoop] 抓取已验证 grasp_verified=True（top check_gone："
                         f"{metrics.get('grasp_top_why')}；面积 "
                         f"{metrics.get('grasp_top_area_before_px')} -> "
                         f"{metrics.get('grasp_top_area_after_px')}）")
            else:
                metrics["grasp_verify_note"] = ("top 视角仍能看到色块（未被夹爪挡住），"
                                                "但技能有物理随动证据（抬升 ok 且夹爪接触的"
                                                "物体已离地），且腕部也看得到目标颜色")
                self.log("[AgentLoop] 抓取已验证 grasp_verified=True（top 色块仍可见，"
                         "但技能物理证据：lift_ok + carry_evidence=True + 腕部目标色块）")
            return {"ok": True, "message": "", "metrics": metrics}
        if gone is False:
            metrics["grasp_verified"] = False
            metrics["grasp_verify_camera"] = "top"
            metrics["grasp_verify_area_px"] = area_for_compat
            why = str(metrics.get("grasp_top_why") or "subject_still_present")
            if has_physical and not bodies_ok:
                reason = (f"技能报告夹爪接触的是 {carry_bodies or '（无）'}，"
                          f"不是目标 {target_id or color}")
            elif skill_physical and wrist_contradicts:
                reason = (f"技能虽有物理随动证据（lift_ok/carry=True），但腕部看不到 {color} "
                          f"目标色块（{metrics.get('grasp_wrist_area_px')}px，说明夹的可能是"
                          f"别的物体）")
            else:
                reason = (f"技能没有抬升/随动物理证据"
                          f"（lift_ok={metrics.get('grasp_skill_lift_ok')} "
                          f"carry={metrics.get('grasp_skill_carry_evidence')}）")
            msg = (f"抓取未确认（grasp_not_confirmed）：top 视角里 {color} 色块仍留在原位"
                   f"（{why}，面积 {metrics.get('grasp_top_area_before_px')} -> "
                   f"{metrics.get('grasp_top_area_after_px')}），{reason}，说明没被抓走。"
                   f"请核对接近高度/目标像素后重试")
            return {"ok": False, "message": msg, "metrics": metrics}
        # top 判据不可用 -> 退回腕部（保留旧语义，并标注降级原因）
        if wrist is not None:
            area = area_for_compat
            metrics["grasp_verify_camera"] = "wrist"
            metrics["grasp_verify_area_px"] = area
            metrics["grasp_verify_fallback"] = "top_unavailable"
            if wrist.get("found") and area >= WRIST_GRASP_MIN_AREA_PX:
                metrics["grasp_verified"] = True
                self.log(f"[AgentLoop] 抓取已验证 grasp_verified=True（top 判据不可用，"
                         f"退回腕部 {color} {area}px >= {WRIST_GRASP_MIN_AREA_PX}）")
                return {"ok": True, "message": "", "metrics": metrics}
            metrics["grasp_verified"] = False
            why = ("腕部没看到色块" if not wrist.get("found")
                   else f"腕部色块只有 {area}px（< {WRIST_GRASP_MIN_AREA_PX}）")
            msg = (f"抓取未确认（grasp_not_confirmed）：top 判据不可用，腕部里也没有 {color} "
                   f"物体（{why}，uv={wrist.get('uv')}）。请换像素、换夹持档位"
                   f"（grasp_profile）或换接近高度后重试")
            return {"ok": False, "message": msg, "metrics": metrics}
        # 两者都做不了 -> 明示「未验证」
        metrics.update({"grasp_verify_camera": "none",
                        "grasp_verified": None,
                        "grasp_unverified": True,
                        "grasp_verify_why": err or "腕部与 top 判据都不可用"})
        self.log(f"[AgentLoop] 抓取未验证（top check_gone 与腕部核验都不可用："
                 f"{metrics['grasp_verify_why']}）")
        return {"ok": True, "message": "", "metrics": metrics}

    def _check_place_release(self, result: SkillResult) -> None:
        """place 之后的第二个独立信号：腕部还看得见色块 -> 没放下（只写 metrics，不改 ok）。

        按 Lead 要求：release_verified 只做观测；放置是否成功仍由 skills 的几何校验
        与子目标复核决定，这里绝不越权改 ok/error_code。
        """
        color = self._held_color
        if not color:
            return
        wrist, err = self._wrist_blob(color)
        if wrist is None:
            self.log(f"[AgentLoop] 释放复核不可用（{err}）")
            return
        area = int(wrist.get("area_px") or 0)
        still_held = bool(wrist.get("found")) and area >= WRIST_GRASP_MIN_AREA_PX
        metrics = dict(result.metrics or {})
        metrics["release_check_camera"] = "wrist"
        metrics["release_check_color"] = color
        metrics["release_check_area_px"] = area
        metrics["release_verified"] = (not still_held)
        if still_held:
            metrics["release_note"] = (f"腕部仍看到 {color} 色块 {area}px：可能没放下去"
                                       f"或粘在夹爪上")
        result.metrics = metrics
        self.log(f"[AgentLoop] 释放复核：腕部 {color} {area}px -> "
                 f"release_verified={not still_held}")
        if not still_held:
            self._held_color = None
            self._held_target_xy = None
            self._held_assumed = False

    def _remember_target_pixel(self, call: SkillCall, state: SceneState) -> None:
        """把本次解析出的目标像素喂进共享缓存（供后续抓取验证 / 同 run 复核复用）。"""
        pm = self._pixel_metrics or {}
        cam = str(pm.get("pixel_camera") or DEFAULT_PIXEL_CAMERA)
        pairs = []
        if isinstance(pm.get("target_pixel"), (list, tuple)):
            pairs.append((cam, list(pm["target_pixel"])))
        if isinstance(pm.get("target_pixel_overview"), (list, tuple)):
            pairs.append(("overview", list(pm["target_pixel_overview"])))
        if not pairs:
            return
        tag = None
        if call.skill == "pick":
            color, _src = self._pick_target_color(call, state)
            tag = self._pick_relaxed_id or (call.args or {}).get("target_id") or color
        else:
            cid = (call.args or {}).get("container_id")
            obj = state.find(cid) if isinstance(cid, str) and cid else None
            color = str(obj.color) if obj is not None else None
            tag = cid if isinstance(cid, str) and cid else color
        if not color and not tag:
            return
        for camera, uv in pairs:
            remember_pixel(color, uv, camera=camera, source="pick/place", target=tag)

    def _run_locator(self, call: SkillCall) -> SkillResult:
        """处理 locate 调用（v2 §5.3）：走 locator 做语义标定，**绝不当技能执行**。

        结果统一成 SkillResult(skill="locate", metrics={"pixel":[u,v],"bbox":[...]})：
        - locator 未配置 / 抛异常 / 返回非 dict / 没拿到像素 -> 可重试的 perception_error，
          （原始请求写进 message/metrics，让 LLM 换个说法或换相机重试）；
        - 成功 -> ok=True，message 带上像素与框，日志里也能看到，供后续 pick 复用。
        """
        args = dict(call.args or {})
        target = str(args.get("target") or "").strip()
        # §5.10：locate 默认 both（top+overview 双目），高物体只靠单目会有最大 7 cm 视差
        camera = str(args.get("camera") or DEFAULT_LOCATE_CAMERA)
        raw_color = args.get("color")
        color = str(raw_color).strip() if isinstance(raw_color, str) and raw_color.strip() else None
        # 模型没给 color 时，从 target 文本解析（「红色方块」-> red）；解析不出就交给 locator
        # 的诚实失败路径（SubGoalVerifier.locate 收到 color=None 时不得猜像素）。
        if color is None:
            color = canonical_color(target)
        base_metrics: dict[str, Any] = {"target": target, "camera": camera}
        if color:
            base_metrics["color"] = color

        if self.locator is None:
            return SkillResult(
                ok=False, skill="locate", error_code="perception_error",
                message=(f"当前没有视觉定位后端（locator 未配置），无法定位 {target!r}；"
                         f"请改用像素链路（target_pixel + pixel_resolver）或稍后重试"),
                metrics=base_metrics)
        try:
            res = self.locator(target, color, camera)
        except Exception as e:  # 感知后端异常不许崩循环
            return SkillResult(ok=False, skill="locate", error_code="perception_error",
                               message=f"locate({target!r}, camera={camera}) 异常: {e}",
                               metrics=base_metrics)
        if not isinstance(res, dict):
            return SkillResult(ok=False, skill="locate", error_code="perception_error",
                               message=f"locator 返回了 {type(res).__name__}，应为 dict",
                               metrics=base_metrics)

        def _clean_uv(v: Any):
            if isinstance(v, (list, tuple)) and len(v) == 2:
                try:
                    return [round(float(v[0]), 2), round(float(v[1]), 2)]
                except (TypeError, ValueError):
                    return None
            return None

        def _clean_bbox(v: Any):
            if isinstance(v, (list, tuple)) and len(v) == 4:
                try:
                    return [round(float(x), 2) for x in v]
                except (TypeError, ValueError):
                    return None
            return None

        metrics = dict(base_metrics)
        legacy_pixel = _clean_uv(res.get("pixel", res.get("uv")))
        pixel_top = _clean_uv(res.get("pixel_top"))
        pixel_ov = _clean_uv(res.get("pixel_overview"))
        if pixel_top is None and pixel_ov is None and legacy_pixel is not None:
            if camera == "overview":
                pixel_ov = legacy_pixel
            else:
                pixel_top = legacy_pixel
        legacy_bbox = _clean_bbox(res.get("bbox"))
        bbox_top = _clean_bbox(res.get("bbox_top"))
        bbox_ov = _clean_bbox(res.get("bbox_overview"))
        if bbox_top is None and bbox_ov is None and legacy_bbox is not None:
            if camera == "overview":
                bbox_ov = legacy_bbox
            else:
                bbox_top = legacy_bbox
        metrics.update({
            # §5.10 冻结的四键
            "pixel_top": pixel_top, "pixel_overview": pixel_ov,
            "bbox_top": bbox_top, "bbox_overview": bbox_ov,
            # 兼容旧字段（UI/verifier 可能还在读 pixel/bbox：优先 top，其次 overview）
            "pixel": pixel_top if pixel_top is not None else pixel_ov,
            "bbox": bbox_top if bbox_top is not None else bbox_ov,
        })
        for key in ("source", "refined", "delta_px", "score", "why"):
            if key in res and key not in metrics:
                metrics[key] = res[key]

        ok = bool(res.get("ok", True)) and (pixel_top is not None or pixel_ov is not None)
        why = str(res.get("why") or res.get("message") or "").strip()
        if ok:
            # 记住这次标定：抓取验证（pick 后 check_gone）与同 run 内的复核都靠它复用
            if color:
                self._locate_color = str(color)
                self._locate_colors[target] = str(color)
            for _tag, _uv, _box in (("top", pixel_top, bbox_top),
                                    ("overview", pixel_ov, bbox_ov)):
                if _uv is not None:
                    remember_pixel(color, _uv, camera=_tag, bbox=_box,
                                   source=str(res.get("source") or "locate"), target=target)
        if ok:
            msg = (f"标定 {target!r} 成功：pixel_top={pixel_top} pixel_overview={pixel_ov} "
                   f"bbox_top={bbox_top} bbox_overview={bbox_ov}")
            if why:
                msg += f"（{why}）"
        else:
            msg = why or (f"标定 {target!r} 失败：top/overview 都没拿到像素"
                          f"（颜色分割没找到 {color or '目标'}）")
        return SkillResult(ok=ok, skill="locate", message=msg,
                            error_code=None if ok else "perception_error", metrics=metrics)

    # -- 主入口 ------------------------------------------------------------ #
    def run(self, task: str, history: Optional[list[dict]] = None) -> list[SkillResult]:
        """执行任务，返回每一步的 SkillResult 列表（含失败的步）。

        history: 跨命令历史（assistant 调用 + user 观察）。
        - 传了就用它当起点；没传则回退到构造参数 history；两者都没有 -> 清空（原行为）。
        - 结束后 self.history = 注入历史 + 本次新增，调用方可读回保存给下一条命令。
        """
        #: 本次任务文本（抓取核验用：任务只点名一种颜色时，抓错颜色要判失败）
        self._task_text = str(task)
        seed = history if history is not None else self._seed_history
        self.history = self._trim_history(list(seed) if seed else [])
        # 本 run 的任务纪元：写在本 run 第一条 assistant 消息里（供下一条命令区分）
        self._epoch_task = str(task)
        self._epoch_marked = False
        self.results = []
        self._repeat_key = None
        self._repeat_hits = 0
        self._fatal_hits = {}
        self._pending_repeat = None
        self._vlm_error_streak = 0
        self.log(f"[AgentLoop] 开始任务: {task!r} | vlm={type(self.vlm).__name__} | max_steps={self.max_steps}")

        for step in range(1, self.max_steps + 1):
            state: SceneState = self.env.get_state()
            images = self._render_all()
            # 第一步的 prompt 里临时补一条纪元标记（不写进 self.history）：
            # 否则本 run 还没写下标记，MockVLM 会把上一条命令的观测误当成当前任务。
            hist_for_prompt = self.history
            if step == 1 and not self._epoch_marked:
                marker = json.dumps({"task": task}, ensure_ascii=False)
                hist_for_prompt = self.history + [{
                    "role": "user",
                    "content": f"{self.EPOCH_PREFIX} {marker}",
                }]
            messages = build_messages(task, state, images, history=hist_for_prompt,
                                      include_poses=self.include_poses)

            # 1) 决策。空回复/被截断是瞬时的（该端点 reasoning 会吃光 completion 预算），
            #    允许连续 2 次 vlm_error 才中止；中间那一次 continue -> 下一步会**重新截图+
            #    重建 messages** 再问，既不重发旧 payload，也不往 history 里塞脏观察。
            try:
                call = self.vlm.decide(messages)
                self._vlm_error_streak = 0
            except Exception as e:
                self._vlm_error_streak += 1
                result = SkillResult(ok=False, skill="unknown", message=f"VLM 决策失败: {e}",
                                     error_code="vlm_error")
                self.results.append(result)
                self.log(f"[step {step}/{self.max_steps}] VLM 异常: {e}")
                if self._vlm_error_streak >= 2:
                    self.log(f"[AgentLoop] VLM 连续异常 {self._vlm_error_streak}/2，任务中止")
                    break
                self.log(f"[AgentLoop] VLM 连续异常 {self._vlm_error_streak}/2，继续重试同一观测"
                         "（下一步重新截图并重建 messages）")
                continue

            # 1a) no-oracle：target_pixel -> 世界 xy；失败记成可重试的 perception_error，
            #     把原始像素回灌给 LLM 纠错（不崩、不替 LLM 猜参数）。
            self._pixel_metrics = {}
            self._pick_relaxed_id = None
            pixel_fail = self._resolve_pixel(call, state)
            # §5.12-5：像素解析阶段的参数纠正（丢冗余 target_id/target_xy）取出并打日志
            corrections = self._collect_corrections(call)
            pixel_metrics = dict(getattr(self, "_pixel_metrics", None) or {})
            # 1b) 持物状态注入（P0）：腕部核验过的 pick 之后，place 要告诉 skills「手里有东西」。
            #     否则 skills 会把被夹持的物体当成入口障碍做「温和撤退」，撤退中把件弄掉
            #     （实机：无真值 pick 抬到 z=0.5397 -> place 撤退 -> 方块掉回桌面，place 还报成功）。
            #     模型不需要知道这两个键；oracle 模式（include_poses=True）一行不改。
            if call.skill == "place" and not self.include_poses and (self._held_assumed or self._held_color):
                if self._held_color is None:
                    looked = self._lookup_held_color_by_wrist(state)
                    if looked:
                        self._held_color = looked
                        self.log(f"[AgentLoop] 腕部反查确认手里夹着 {looked}（正面证据）")
                inject = dict(call.args or {})
                inject["holding"] = True
                if self._held_color:
                    inject["holding_color"] = str(self._held_color)
                call.args = inject
                self.log("[AgentLoop] 注入持物状态 holding=True"
                         + (f" holding_color={self._held_color}" if self._held_color else "（颜色未知）")
                         + (f"（抓取点 {self._held_target_xy}）" if self._held_target_xy else ""))
            if pixel_fail is not None:
                fail_metrics = dict(pixel_fail.get("metrics") or {})
                if corrections:
                    fail_metrics["corrected_args"] = corrections
                result = SkillResult(ok=False, skill=call.skill,
                                     message=pixel_fail["message"],
                                     error_code="perception_error",
                                     metrics=fail_metrics)
                self.results.append(result)
                self._append_history(call, result)
                self.log(f"[step {step}/{self.max_steps}] 像素反投影失败: {pixel_fail['message']}")
                reason = self._note_failure(call.skill, result.error_code,
                                               self._args_signature(call))
                if reason:
                    self.log(f"[AgentLoop] {reason}，任务中止")
                    break
                continue

            # 1b) 上一步已判定「同一失败连续重复 >= 2 次」：只有当 VLM 仍提议**同一个 skill
            #     且参数完全相同**时才终止；动作或参数有任何变化一律放行执行
            #     （D1：先 target_id+target_pixel 被拒、再改成纯像素时不能被误杀）。
            if self._pending_repeat is not None:
                p_skill, p_code, p_sig = self._pending_repeat
                if call.skill == p_skill and self._args_signature(call) == p_sig:
                    self.log(f"[AgentLoop] 同一调用 {p_skill}() 连续 {self._repeat_hits} 次返回"
                             f"相同错误 {p_code}，任务中止")
                    break
                self.log(f"[AgentLoop] 调用已改变（{p_skill} -> {call.skill} 或参数不同），"
                         f"放行执行不再套用重复守卫")
            self._pending_repeat = None

            # 2) 校验（validate 里还有 §5.12 的冗余纠正，一并取出）
            ok, msg = validate_skill_call(call, state)
            corrections += self._collect_corrections(call)
            if not ok:
                invalid_metrics = dict(pixel_metrics)
                if corrections:
                    invalid_metrics["corrected_args"] = corrections
                result = SkillResult(ok=False, skill=call.skill, message=msg, error_code="invalid_call",
                                     metrics=invalid_metrics)
                self.results.append(result)
                self._append_history(call, result)
                self.log(f"[step {step}/{self.max_steps}] 非法调用被拒: {call.skill} -> {msg}")
                reason = self._note_failure(call.skill, result.error_code,
                                               self._args_signature(call))
                if reason:
                    self.log(f"[AgentLoop] {reason}，任务中止")
                    break
                continue

            # 2b) 控制面工具（v2 §5.3）：locate / subgoal_done 不交给技能执行器。
            if call.skill == "locate":
                result = self._run_locator(call)
                self.results.append(result)
                self._append_history(call, result)
                self.log(f"[step {step}/{self.max_steps}] locate({call.args.get('target')!r}) -> "
                         f"ok={result.ok} pixel={result.metrics.get('pixel')} "
                         f"bbox={result.metrics.get('bbox')} msg={result.message!r}")
                reason = self._note_failure(call.skill, result.error_code,
                                            self._args_signature(call))
                if reason:
                    self.log(f"[AgentLoop] {reason}，任务中止")
                    break
                continue

            if call.skill == "subgoal_done":
                # 模型自评完成：本 epoch 到此为止，交给 SubGoalVerifier 用图像独立复核。
                note = str((call.args or {}).get("note") or "").strip()
                result = SkillResult(ok=True, skill="subgoal_done",
                                      message=note or "模型自评当前子目标完成（待独立复核）",
                                      metrics={"self_reported": True})
                self.results.append(result)
                self._append_history(call, result)
                self.log(f"[step {step}/{self.max_steps}] subgoal_done -> ok=True | "
                         f"模型自评完成: {result.message}")
                self.log("[AgentLoop] 模型自评子目标完成，结束本子目标 epoch"
                         "（自评不计入验收，由 SubGoalVerifier 独立复核）")
                break

            # 2c) P0 无真值抓取验证：执行**前**先定好颜色与 hint 像素，并抓一张 top 图作对比。
            #     颜色未知 / 没有像素时明确记「未验证」，绝不假装通过。
            self._remember_target_pixel(call, state)
            verify_color: Optional[str] = None
            verify_color_src = "none"
            rgb_before: Any = None
            if call.skill == "pick" and not self.include_poses:
                # 先抓一张 top 图：既是 check_gone 的 before，也用于「从像素反查颜色」
                rgb_probe = images.get("top")
                if rgb_probe is None:
                    rgb_probe = self._render_camera("top")
                verify_color, verify_color_src = self._pick_target_color(call, state)
                if verify_color is None:
                    hint_probe = self._pick_hint_uv(call)
                    model_gave_pixel = isinstance(
                        (self._pixel_metrics or {}).get("target_pixel"), (list, tuple))
                    if hint_probe is not None and rgb_probe is not None and model_gave_pixel:
                        inferred = self._infer_color_at(hint_probe, state, rgb_probe)
                        if inferred:
                            verify_color, verify_color_src = inferred, "pixel"
                            self.log(f"[AgentLoop] 抓取验证：按模型像素反查目标颜色 -> "
                                     f"{inferred}（hint={hint_probe}）")
                rgb_before = rgb_probe if verify_color else None
                # 3a) 任务只点名一种颜色时，先做「抓的是不是任务要的物体」一致性检查：
                #     像素指错物体（任务要红块、像素却指蓝块）必须说清楚，免得白抓一次还以为成功。
                if verify_color:
                    want_colors = colors_in_text(self._task_text)
                    if len(want_colors) == 1 and canonical_color(verify_color) != want_colors[0]:
                        result = SkillResult(
                            ok=False, skill="pick",
                            message=(f"抓取目标与任务不一致（grasp_not_confirmed）：任务要 "
                                     f"{want_colors[0]}，而像素/标定指向 {verify_color}"
                                     f"（来源 {verify_color_src}）；请重新核对目标像素，"
                                     f"或明确指定要抓的物体"),
                            error_code="grasp_not_confirmed",
                            metrics={"grasp_target_mismatch": True,
                                     "grasp_verify_color": verify_color,
                                     "grasp_verify_color_source": verify_color_src,
                                     "task_colors": list(want_colors)})
                        self.results.append(result)
                        self._append_history(call, result)
                        self.log(f"[AgentLoop] 目标不一致 -> grasp_not_confirmed：任务要 "
                                 f"{want_colors[0]}，像素/标定指向 {verify_color}")
                        reason = self._note_failure(call.skill, result.error_code,
                                                    self._args_signature(call))
                        if reason:
                            self.log(f"[AgentLoop] {reason}，任务中止")
                            break
                        continue

            # 3) 执行
            try:
                result = self.executor(call)
                if not isinstance(result, SkillResult):
                    raise TypeError(f"executor 返回了 {type(result).__name__}，应为 SkillResult")
            except Exception as e:
                result = SkillResult(ok=False, skill=call.skill, message=f"执行器异常: {e}",
                                     error_code="executor_exception")
            # 3b) 技能自称抓取成功 -> 独立确认"物体真的被拿走了"（P1）
            #     主判据 = top 前后 check_gone；top 不可用才退回腕部色块；
            #     技能带物理随动证据（抬升 ok + 夹爪接触的物体已离地 + 就是目标物）时，
            #     top"仍可见"不构成否决（蓝块真抓后 top 面积 818->827，纯像素会漏判）。
            if call.skill == "pick" and not self.include_poses and result.ok:
                if not verify_color:
                    # 技能自称成功、我们又证伪不了 -> place 时按「手里有东西」处理（P0：
                    # 否则 skills 会做入口清场撤退，撤退中把件弄掉，实机 diag 的根因）
                    self._held_assumed = True
                    self.log("[AgentLoop] 抓取未验证（未知颜色）：无法用腕部相机确认夹爪里"
                             "是否有物体；若后续放置异常请检查夹持")
                else:
                    hint = self._pick_hint_uv(call)      # 只有 top 辅助路径才需要
                    verdict = self._verify_pick_grasp(verify_color, verify_color_src,
                                                      hint, rgb_before,
                                                      skill_metrics=result.metrics)
                    if verdict is not None:
                        vmetrics = dict(result.metrics or {})
                        vmetrics.update(verdict["metrics"])
                        if verdict["ok"]:
                            result.metrics = vmetrics
                            self._held_assumed = True      # 技能自称成功且未被证伪
                            if vmetrics.get("grasp_verified") is True:
                                # P0：记住持物状态 —— 供 place 注入 holding（skills 据此跳过
                                # 入口清场撤退，避免撤退中把件弄掉）+ place 后释放复核复用
                                self._held_color = verify_color
                                held_xy = (call.args or {}).get("target_xy")
                                self._held_target_xy = (list(held_xy)
                                                        if isinstance(held_xy, (list, tuple))
                                                        and len(held_xy) >= 2 else None)
                        else:
                            vmetrics["skill_claimed_message"] = result.message
                            vmetrics["skill_claimed_ok"] = True
                            result = SkillResult(ok=False, skill="pick",
                                                 message=verdict["message"],
                                                 error_code="grasp_not_confirmed",
                                                 metrics=vmetrics)
                            self.log(f"[AgentLoop] 抓取未确认 -> grasp_not_confirmed："
                                     f"{verdict['message']}")

            # 3c) place 成功 -> 腕部复核「真的放下了吗」（P0-4，只写 metrics，不改 ok）
            if call.skill == "place" and not self.include_poses and result.ok:
                self._check_place_release(result)
            if corrections or pixel_metrics:
                metrics = dict(result.metrics or {})
                # 像素解析指标（source=triangulate|single_view、residual_mm、原始像素）可观测
                for key, val in pixel_metrics.items():
                    metrics.setdefault(key, val)
                if corrections:
                    # §5.12-5：纠正必须出现在 metrics.corrected_args 里（UI 计「已纠正」）
                    metrics["corrected_args"] = corrections
                result.metrics = metrics
            if not result.ok and call.skill in ("pick", "place"):
                # U1/R3：pick/place 失败必须把「原因 + obstruction_id + 接触力 + 建议图像」
                # 交回 LLM 重新规划（本循环只补观测，不代 LLM 改参数盲重试）。
                self._attach_failure_observation(call, result, state)
            self.results.append(result)
            self._append_history(call, result)

            preview = json.dumps(to_jsonable(call.args), ensure_ascii=False)
            if len(preview) > 160:
                preview = preview[:157] + "..."
            # 把模型的判断显性化（task-31 要求 D）：附 reasoning 摘要，用户一眼能看出
            # 是模型在决策而不是脚本在走流程。
            reason = str(getattr(call, "reasoning", "") or "").strip().replace("\n", " ")
            if len(reason) > 80:
                reason = reason[:80] + "…"
            self.log(f"[step {step}/{self.max_steps}] {call.skill}({preview}) -> "
                     f"ok={result.ok} err={result.error_code} msg={result.message!r}"
                     + (f"  | 依据: {reason}" if reason else ""))

            # 4) 结束条件：stop 或 ask_user 都终止闭环。
            #    ask_user 表示需要人工确认，继续循环只会反复问同一个问题直到 max_steps。
            if call.skill in ("stop", "ask_user"):
                if call.skill == "stop":
                    self.log("[AgentLoop] 收到 stop，任务结束")
                else:
                    self.log("[AgentLoop] 需要人工确认，任务中止")
                break

            # 4b) 同一失败连续重复 / 不可重试错误码 -> 立即终止，避免空转到 max_steps
            reason = self._note_failure(call.skill, result.error_code,
                                               self._args_signature(call))
            if reason:
                self.log(f"[AgentLoop] {reason}，任务中止")
                break

            if step == self.max_steps:
                self.log(f"[AgentLoop] 达到 max_steps={self.max_steps}，强制结束")

        ok_n = sum(1 for r in self.results if r.ok)
        self.log(f"[AgentLoop] 结束：共 {len(self.results)} 步，成功 {ok_n}，失败 {len(self.results) - ok_n}")
        return self.results
