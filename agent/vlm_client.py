"""agent.vlm_client —— OpenAI 兼容的多模态决策客户端 + 离线规则决策器 MockVLM。

设计要点：
1. VLMClient 只用标准库（urllib.request / json / base64），不引入 openai SDK。
   请求体含 tools + tool_choice="auto"，走 POST {base_url}/chat/completions。
   解析顺序：先 message.tool_calls（标准 function calling），失败再从 message.content
   里正则/括号平衡扫描抽 JSON（不少兼容端点会把 function call 塞进 content）。
2. 超时、HTTP 错误、网络错误分别 raise 明确异常类，调用方（AgentLoop）负责降级/重试。
3. MockVLM 是确定性规则决策器：从 messages 文本里反向解析 SCENE JSON 与历史
   observation，按 pick -> place 推进；接触力超阈值先 set_compliance(soft)。
   它让 demo 在没有 VLM_API_KEY 时也能离线跑通（CI 依赖这一点）。
   重要：MockVLM 只依据 scene JSON 里的 color / category / is_container / graspable
   字段做匹配，不硬编码任何具体 id（id 命名约定由 sim 决定，随时可能变）。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from core.config import VLM_API_KEY, VLM_BASE_URL, VLM_MAX_TOKENS, VLM_MODEL, VLM_TIMEOUT

#: 请求重试次数（网络层 + 空回复）。默认 4 次重试 = 最多 5 次尝试，够覆盖两次空回复。
#: （本机实测：走代理访问 ollama.com 会间歇性 TLS "unexpected eof"，同一秒重试往往成功；
#:   该端点还会把思维链放进 message.reasoning 吃光 completion 预算 -> content 为空。）
VLM_MAX_RETRIES = int(os.environ.get("VLM_MAX_RETRIES", "4"))
#: 重试退避（秒），长度不足时取最后一个值。
VLM_RETRY_BACKOFF = (0.6, 1.5, 3.0, 5.0)
#: 空回复重试时的 max_tokens 放大倍数与上限（reasoning 吃预算，翻倍通常就够）
VLM_EMPTY_TOKEN_GROWTH = 2.0
VLM_MAX_TOKENS_CAP = int(os.environ.get("VLM_MAX_TOKENS_CAP", "8192"))
#: 重试时追加的强制工具调用提示（只加在重试 payload 上，不动调用方 messages）
TOOL_HINT_TEXT = ("上一条回复没有可用的工具调用（可能被 completion 预算截断）。"
                  "这次**必须调用工具**：只输出一个 function call，不要输出自然语言说明。")
from core.types import SkillCall

from .schema import (PROFILE_NAMES, RELATION_NAMES, TOOLS, compliance_from_profile,
                     parse_skill_call)


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class VLMError(RuntimeError):
    """VLM 调用相关错误基类。

    每个异常都显式带两个稳定属性（不再依赖中文错误字符串做判据）：
      retryable  : 是否值得原样重试；
      error_code : 机器可读的错误码（empty_response / unparseable_content / bad_json /
                   no_choices / http_429 ...）。
    """

    def __init__(self, message: str = "", *, retryable: bool = False,
                 error_code: str = "vlm_error") -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.error_code = str(error_code)


class VLMNetworkError(VLMError):
    """连接失败 / 超时 / DNS 等网络层错误（默认可重试）。"""

    def __init__(self, message: str = "", *, retryable: bool = True,
                 error_code: str = "network") -> None:
        super().__init__(message, retryable=retryable, error_code=error_code)


class VLMResponseError(VLMError):
    """HTTP 非 2xx，或响应体里没有可解析的工具调用。"""

    def __init__(self, message: str = "", *, retryable: bool = False,
                 error_code: str = "bad_response") -> None:
        super().__init__(message, retryable=retryable, error_code=error_code)


#: 值得重试的 HTTP 状态码（限流与网关/服务端瞬时错误）。
RETRYABLE_HTTP_CODES = (408, 425, 429, 500, 502, 503, 504)


def _retryable_response(err: Any) -> bool:
    """错误是否值得重试（**稳定判据**：优先读异常自带的 retryable 属性）。

    2026-10-12 实机教训：老实现靠中文子串（『没有可解析的工具调用』）判断，而 decide() 抛的是
    『响应既没有 tool_calls，content 也为空』『content 里找不到合法工具调用: …』，两条都不匹配
    -> 空回复一次都不重试，任务直接死掉。现在每个 VLMError 都显式标注 retryable/error_code。
    传裸字符串时只保留 HTTP 状态码兜底（兼容旧调用）。
    """
    if isinstance(err, VLMError):
        return bool(getattr(err, "retryable", False))
    m = re.search(r"HTTP\s+(\d{3})", str(err))
    return bool(m) and int(m.group(1)) in RETRYABLE_HTTP_CODES


# --------------------------------------------------------------------------- #
# VLMClient
# --------------------------------------------------------------------------- #
class VLMClient:
    """OpenAI 兼容端点的多模态 function-calling 客户端。

    参数留空则读 core.config（环境变量 VLM_API_KEY / VLM_BASE_URL / VLM_MODEL ...）。
    available 表示是否配置了 api_key；没 key 时 decide() 直接抛 VLMError，
    由 make_vlm() 保证实际不会走到这里（无 key 返回 MockVLM）。
    """

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: Optional[float] = None,
                 log: Optional[Any] = None, max_retries: Optional[int] = None,
                 sleep: Optional[Any] = None):
        self.model = model or VLM_MODEL
        self.base_url = (base_url or VLM_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else VLM_API_KEY
        self.timeout = float(timeout if timeout is not None else VLM_TIMEOUT)
        self.max_tokens = int(VLM_MAX_TOKENS)
        #: 可选日志回调（demo 会传自己的 log；缺省静默）
        self.log = log
        self.max_retries = int(VLM_MAX_RETRIES if max_retries is None else max_retries)
        #: 最近一次 _post（传输层）实际尝试的次数（诊断用）
        self.last_attempts = 0
        #: 最近一次 decide（含空回复重试）实际尝试的次数（诊断/测试用）
        self.last_decide_attempts = 0
        #: 重试等待函数（测试可传 lambda _s: None 免得真睡）
        self._sleep = sleep if callable(sleep) else time.sleep
        #: 上一次 decide 实际用的 max_tokens（诊断/测试用）
        self.last_max_tokens = int(self.max_tokens)

    @property
    def available(self) -> bool:
        """是否配置了 API key（无 key -> 应改用 MockVLM）。"""
        return bool(self.api_key)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    # -- 底层请求 ---------------------------------------------------------- #
    def _post_once(self, data: bytes) -> dict:
        """发一次请求（不重试）。异常语义与原实现一致，便于单测直接替换。"""
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            raise VLMResponseError(f"HTTP {e.code} {e.reason} @ {self.endpoint}: {detail}",
                                    retryable=(e.code in RETRYABLE_HTTP_CODES),
                                    error_code=f"http_{e.code}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise VLMNetworkError(f"请求 {self.endpoint} 失败（超时 {self.timeout}s）: {e}") from e
        try:
            return json.loads(body)
        except Exception as e:
            # 响应被截断时也值得重试（该端点 reasoning 吃预算时会截断）
            raise VLMResponseError(f"响应不是合法 JSON: {e}; 前 200 字符: {body[:200]!r}",
                                    retryable=True, error_code="bad_json") from e

    def _post(self, payload: dict) -> dict:
        """带重试的请求。网络层错误（含本机代理间歇性 TLS EOF）与 429/5xx 自动重试。

        动机（实测）：走 http://127.0.0.1:7897 访问 ollama.com 时会间歇性
        "SSL: UNEXPECTED_EOF_WHILE_READING"，5 s 后被对端关闭；同一秒重试一次往往成功。
        """
        if not self.available:
            raise VLMError("未配置 api_key（VLM_API_KEY / OPENAI_API_KEY），无法调用远程 VLM")
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        attempts = max(1, int(self.max_retries) + 1)
        err: Optional[VLMError] = None
        for i in range(attempts):
            self.last_attempts = i + 1
            try:
                return self._post_once(data)
            except VLMNetworkError as e:
                if i + 1 >= attempts:
                    raise VLMNetworkError(f"{e}（已尝试 {attempts} 次）") from e
                err = e
            except VLMResponseError as e:
                if i + 1 >= attempts or not _retryable_response(str(e)):
                    raise
                err = e
            wait = VLM_RETRY_BACKOFF[min(i, len(VLM_RETRY_BACKOFF) - 1)]
            if self.log is not None:
                try:
                    self.log(f"[VLM] 第 {i + 1}/{attempts} 次请求失败（{type(err).__name__}: "
                             f"{str(err)[:120]}），{wait:.1f}s 后重试")
                except Exception:
                    pass
            time.sleep(wait)
        raise err if err is not None else VLMError("请求失败且无异常记录")  # pragma: no cover

    @staticmethod
    def _content_text(content: Any) -> str:
        """兼容 content 为字符串或 part 列表（部分端点会回多模态 part）。"""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    parts.append(str(p.get("text", "")))
                else:
                    parts.append(str(p))
            return "\n".join(parts)
        return str(content)

    def _log(self, text: str) -> None:
        """可选日志（demo 会传自己的 log）。"""
        if self.log is not None:
            try:
                self.log(text)
            except Exception:
                pass

    def _parse_response(self, obj: dict) -> SkillCall:
        """解析响应体 -> SkillCall；没有可用工具调用时抛**可重试**的 VLMResponseError。"""
        choices = obj.get("choices") or []
        if not choices:
            err = obj.get("error")
            raise VLMResponseError(
                f"响应缺少 choices: {json.dumps(err or obj, ensure_ascii=False)[:300]}",
                retryable=True, error_code="no_choices")
        msg = choices[0].get("message") or {}
        reasoning = self._content_text(msg.get("reasoning_content") or msg.get("reasoning") or "")

        # (a) 标准 tool_calls
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            first = tool_calls[0] if isinstance(tool_calls, (list, tuple)) else tool_calls
            call = parse_skill_call(first)
            if reasoning and not call.reasoning:
                call.reasoning = reasoning
            return call

        # (b) 兼容端点把 function call 塞进 content
        content = self._content_text(msg.get("content"))
        if content.strip():
            try:
                return parse_skill_call(content)
            except ValueError as e:
                raise VLMResponseError(
                    f"content 里找不到合法工具调用: {e}; content={content[:200]!r}",
                    retryable=True, error_code="unparseable_content") from e
        # (c) 空回复：reasoning 吃光 completion 预算（实测 finish_reason=length）
        raise VLMResponseError("响应既没有 tool_calls，content 也为空",
                                retryable=True, error_code="empty_response")

    def decide(self, messages: list[dict]) -> SkillCall:
        """发一次多模态请求（含**空回复重试**），返回解析后的 SkillCall。

        2026-10-12 实机根因：老实现只在传输层重试，响应解析失败（空 content / 无 tool_calls）
        直接抛出 -> AgentLoop 一步就 break，把「只差最后一个 place」的任务打死。
        现在：
          - 解析失败且异常标了 retryable 就重试；
          - **第 2 次起把 max_tokens 翻倍**（上限 VLM_MAX_TOKENS_CAP）并追加一条「必须调用工具」
            的提示 —— 该端点的 message.reasoning 会占满 completion 预算；
          - 重试只改本次 payload，绝不修改调用方传入的 messages；退避用 VLM_RETRY_BACKOFF。
        """
        attempts = max(1, int(self.max_retries) + 1)
        tokens = int(self.max_tokens)
        attempt_messages: list[dict] = list(messages or [])
        last_err: Optional[VLMError] = None
        for i in range(attempts):
            # 注意：_post 内部还会用自己的 last_attempts（传输层），两者分开记，互不覆盖
            self.last_decide_attempts = i + 1
            self.last_max_tokens = tokens
            payload = {
                "model": self.model,
                "messages": attempt_messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "temperature": 0.2,
                "max_tokens": tokens,
            }
            try:
                return self._parse_response(self._post(payload))
            except VLMResponseError as e:
                exhausted = (i + 1) >= attempts
                if not getattr(e, "retryable", False) or exhausted:
                    if i > 0:
                        raise VLMResponseError(
                            f"{e}（空回复/无工具调用共重试 {i} 次，最后 max_tokens={tokens}）",
                            retryable=False,
                            error_code=getattr(e, "error_code", "bad_response")) from e
                    raise
                last_err = e
                tokens = min(int(max(tokens * VLM_EMPTY_TOKEN_GROWTH, tokens + 1)),
                             VLM_MAX_TOKENS_CAP)
                attempt_messages = list(messages or []) + [{"role": "user", "content": TOOL_HINT_TEXT}]
                wait = VLM_RETRY_BACKOFF[min(i, len(VLM_RETRY_BACKOFF) - 1)]
                self._log(f"[VLM] 第 {i + 1}/{attempts} 次响应不可用"
                          f"（{getattr(e, 'error_code', 'bad_response')}: {str(e)[:80]}），"
                          f"提高 max_tokens 到 {tokens} 后 {wait:.1f}s 重试")
                self._sleep(wait)
        raise last_err if last_err is not None else VLMError("请求失败且无异常记录")  # pragma: no cover


# --------------------------------------------------------------------------- #
# MockVLM
# --------------------------------------------------------------------------- #
_COLOR_ALIASES: dict[str, tuple[str, ...]] = {
    "red": ("red", "红"),
    "blue": ("blue", "蓝"),
    "green": ("green", "绿"),
    "yellow": ("yellow", "黄"),
    "gray": ("gray", "grey", "灰"),
}
_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "box": ("box", "cube", "方块", "立方"),
    "can": ("can", "cylinder", "圆柱", "罐"),
    "tray": ("tray", "托盘", "盘子"),
}
_SOFT_WORDS = ("轻", "轻拿轻放", "轻一点", "小心", "易碎", "别压", "温柔", "gentle", "soft", "fragile", "careful")
_PLACE_WORDS = ("放", "place", "into", "put", "装", "丢进")

#: 意图词表 —— MockVLM 用它把「回家 / 松开 / 停止 / 只夹不放」从抓放流程里分流出去。
#: 这些词只做子串匹配（任务文本很短），命中即视为该意图。
_HOME_WORDS = ("回家", "回默认", "回原位", "复位", "初始位姿", "初始位置", "原位置",
               "回到默认", "home", "reset pose")
_OPEN_WORDS = ("松开", "张开", "打开夹爪", "松爪", "放开", "open gripper", "release")
_STOP_WORDS = ("停下", "停止", "别动", "stop", "cancel")
_PICK_WORDS = ("抓", "夹", "拿", "握", "取", "pick", "grasp", "grab")

#: 不可重试错误码 —— 与 agent.loop.AgentLoop.NON_RETRYABLE_ERRORS 保持一致。
#: 观察历史里一旦出现这些错误，MockVLM 立即 stop，不再产生同类调用。
_NON_RETRYABLE_ERRORS = ("home_blocked", "stuck", "object_disturbed", "scene_reset")

#: 放置表面词表：命中「桌子/桌面/台面/地上/table」-> place(surface='table')。
#: 「托盘/盘子」仍走原有的 container_id 路径（保持向后兼容）。
_SURFACE_WORDS: dict[str, tuple[str, ...]] = {
    "table": ("桌子", "桌面", "台面", "桌上", "地上", "table", "desk", "floor"),
    "tray": ("托盘", "盘子", "tray"),
}
#: 否定词：「不是盘子里」里的「盘子」要被排除，避免把被否定的表面当目标。
_NEGATION_WORDS = ("不是", "别", "不要", "不", "没")
#: 指代纠正词：命中且历史里有过成功 place -> 把刚放下的物体重新放到本次指定位置。
_CORRECTION_WORDS = ("不是", "别放", "放错", "不对", "重新", "应该是", "改成", "错了", "放反")
#: 会改变「是否夹持」的技能。判断纠正时是否需要重新 pick，只能看这些技能，
#: 末尾的 stop/home/set_compliance 等不能掩盖之前的 place（否则会漏 pick 直接下压）。
_HOLD_CHANGE_SKILLS = ("pick", "place", "open_gripper")

#: 叠放关系词（强信号）：命中即优先尝试 place(on_top_of=锚点)。
_STACK_WORDS = ("上方", "上面", "顶上", "顶部", "叠在", "叠放", "摞在", "摞放",
                "on top", "stack")
#: 放置关系词（宽口径）：出现却解析不出目标时必须 ask_user，绝不静默丢托盘。
_RELATION_WORDS = _STACK_WORDS + (
    "旁边", "左边", "右边", "下方", "下面", "底下", "里", "中", "上", "下", "左", "右",
    "beside", "next to", "left", "right", "under", "below", "inside", "into", "above",
)

#: 横向方位词 -> place 的 relation 值（与 skills 冻结接口一致：
#: right=+x, left=-x, front=+y, back=-y，beside=自动选净距最大方向）。
_DIRECTION_WORDS: dict[str, tuple[str, ...]] = {
    "right": ("右边", "右侧", "右", "right"),
    "left": ("左边", "左侧", "左", "left"),
    "front": ("前面", "前方", "前", "front"),
    "back": ("后面", "后方", "后", "back", "behind"),
    "beside": ("旁边", "一侧", "beside", "next to"),
}
#: 统一的**复合方向**别名表（中英）：命中一律判为「不支持」-> ask_user
#: （回显锚点 + 支持项 right/left/front/back/beside + 可直接给 xy）。
#: 规则：复合方向 > 纯方向；复合方向（含与叠放词相邻的斜向说法）> 纯叠放。
#: 刻意不含裸 "top"（避免误伤 "on top of X"）与裸 left/right/front/back（纯方向）。
_COMPOUND_DIRECTIONS = (
    # 英文
    "upper", "upper-left", "upper-right", "upper-front", "upper-back",
    "top-left", "top-right", "top-front", "top-back", "top left", "top right",
    "bottom-left", "bottom-right", "bottom left", "bottom right",
    "lower-left", "lower-right", "lower left", "lower right",
    "front-left", "front-right", "back-left", "back-right",
    "diagonal", "diagonally",
    # 中文
    "左上", "右上", "左下", "右下",
    "斜左", "斜右", "斜前", "斜后",
    "左上方", "右上方", "前上方", "后上方",
    "斜左上方", "斜右上方", "斜前上方", "斜后上方",
    "左前", "右前", "左后", "右后",
    "左下方", "右下方", "前下方", "后下方",
    "对角", "中间", "正上", "正下", "偏左", "偏右", "偏前", "偏后",
)
#: 悬挂指示词：没有点名具体物体/位置，必须先问清楚再动手（R1：「那边」不能先抓 box_red）。
_DEICTIC_WORDS = ("那边", "那儿", "这里", "这儿", "那里", "这个", "那个", "它")

_SCENE_RE = re.compile(r"SCENE_JSON_BEGIN\s*(\{.*?\})\s*SCENE_JSON_END", re.S)


def _messages_text(messages: list[dict]) -> str:
    """把 messages 里所有可见文本拼起来（MockVLM 的输入就是这个字符串）。"""
    chunks: list[str] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            chunks.append(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    chunks.append(str(p.get("text", "")))
                elif isinstance(p, dict) and "text" in p:
                    chunks.append(str(p["text"]))
    return "\n".join(chunks)


def _json_objects(text: str) -> list[Any]:
    """括号平衡扫描：抽出文本里所有合法 JSON 对象（忽略字符串内的花括号）。"""
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
                    try:
                        out.append(json.loads(text[start:i + 1]))
                    except Exception:
                        pass
                    start = -1
    return out


def _find_scene(text: str) -> Optional[dict]:
    """优先用 SCENE_JSON_BEGIN/END 标记；退化到「含 objects 字段的最后一个 JSON 对象」。"""
    m = _SCENE_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    cands = [o for o in _json_objects(text) if isinstance(o, dict) and "objects" in o]
    return cands[-1] if cands else None


def _find_observations(text: str) -> list[dict]:
    """历史 observation = 同时含 skill 与 ok 字段的 JSON 对象（按出现顺序）。"""
    return [o for o in _json_objects(text)
            if isinstance(o, dict) and "skill" in o and "ok" in o]


def _mentioned_colors(text: str) -> list[str]:
    """按在任务文本中首次出现的位置返回颜色规范名（red/blue/green/yellow/gray）。"""
    t = str(text).lower()
    hits: list[tuple[int, str]] = []
    for canon, aliases in _COLOR_ALIASES.items():
        idxs = [t.find(a) for a in aliases]
        idxs = [i for i in idxs if i >= 0]
        if idxs:
            hits.append((min(idxs), canon))
    hits.sort()
    return [c for _, c in hits]


def _mentioned_categories(text: str) -> list[str]:
    t = str(text).lower()
    hits: list[tuple[int, str]] = []
    for canon, aliases in _CATEGORY_ALIASES.items():
        idxs = [t.find(a) for a in aliases]
        idxs = [i for i in idxs if i >= 0]
        if idxs:
            hits.append((min(idxs), canon))
    hits.sort()
    return [c for _, c in hits]


def _wants_soft(task: str) -> bool:
    t = str(task).lower()
    return any(w in t for w in _SOFT_WORDS)


def _wants_home(task: str) -> bool:
    """任务是否要求回到默认 / 初始位姿。"""
    t = str(task).lower()
    return any(w in t for w in _HOME_WORDS)


def _wants_open(task: str) -> bool:
    """任务是否要求松开 / 打开夹爪。"""
    t = str(task).lower()
    return any(w in t for w in _OPEN_WORDS)


def _wants_stop(task: str) -> bool:
    """任务是否要求停止 / 取消。"""
    t = str(task).lower()
    return any(w in t for w in _STOP_WORDS)


def _pick_only(task: str) -> bool:
    """任务是否「只夹不放」：命中抓取词且未命中放置词。

    特例：「放开 / 松开」这类松开意图本身含「放」，会误命中 _PLACE_WORDS，
    但它们是 _OPEN_WORDS，必须在判定前先排除，否则会被当成抓取任务。
    """
    if _wants_open(task):
        return False
    t = str(task).lower()
    if not any(w in t for w in _PICK_WORDS):
        return False
    return not any(w in t for w in _PLACE_WORDS)


def _surface_mentions(task: str) -> list[tuple[int, str, bool]]:
    """返回 [(出现位置, surface, 是否被否定)]，按出现位置排序。"""
    t = str(task).lower()
    out: list[tuple[int, str, bool]] = []
    for surf, words in _SURFACE_WORDS.items():
        for w in words:
            start = 0
            while True:
                i = t.find(w, start)
                if i < 0:
                    break
                prefix = t[max(0, i - 4):i]
                out.append((i, surf, any(n in prefix for n in _NEGATION_WORDS)))
                start = i + len(w)
    out.sort(key=lambda item: item[0])
    return out


def _placement_surface(task: str) -> Optional[str]:
    """任务文本里指定的放置表面（排除被否定的说法），没指定返回 None。"""
    for _i, surf, negated in _surface_mentions(task):
        if not negated:
            return surf
    return None


def _wants_table_surface(task: str) -> bool:
    """任务是否要求放到桌面（桌子/桌面/台面/地上/table）。"""
    return _placement_surface(task) == "table"


def _object_mentions(task: str, objs: list[dict]) -> list[tuple[int, dict]]:
    """返回 [(物体在任务文本里首次出现的位置, 物体)]，按位置排序。

    先按颜色匹配；未被颜色命中的物体再用类别兜底，但会跳过「已被别的颜色词占用」
    的类别出现——否则「蓝色方块放在红色方块上方」里的两个「方块」会让 box_red/
    box_green/box_blue 都在同一位置命中，锚点解析就乱了。
    """
    t = str(task).lower()
    hits: list[tuple[int, dict]] = []
    matched: set[int] = set()
    for o in objs:
        pos: Optional[int] = None
        for alias in _COLOR_ALIASES.get(str(o.get("color", "")).lower(), ()):
            i = t.find(alias)
            if i >= 0:
                pos = i if pos is None else min(pos, i)
        if pos is not None:
            hits.append((pos, o))
            matched.add(id(o))
    color_aliases = tuple(a for aliases in _COLOR_ALIASES.values() for a in aliases)
    for o in objs:
        if id(o) in matched:
            continue
        best: Optional[int] = None
        for alias in _CATEGORY_ALIASES.get(str(o.get("category", "")).lower(), ()):
            start = 0
            while True:
                i = t.find(alias, start)
                if i < 0:
                    break
                # 若紧贴该类别词的（间隔 <= 2 字符）是别的颜色词，说明这个类别词属于那个
                # 颜色物体（如「蓝色方块」里的「方块」），本轮跳过。
                glued = False
                for color_alias in color_aliases:
                    j = t.rfind(color_alias, 0, i)
                    if j >= 0 and 0 <= i - (j + len(color_alias)) <= 2:
                        glued = True
                        break
                if not glued and (best is None or i < best):
                    best = i
                start = i + len(alias)
        if best is not None:
            hits.append((best, o))
    hits.sort(key=lambda item: item[0])
    return hits


def _first_relation_word(task: str) -> Optional[str]:
    """任务里出现的第一个放置关系词（只用于报错信息）。"""
    t = str(task).lower()
    for w in _RELATION_WORDS:
        if w in t:
            return w
    return None


def _resolve_anchor_at(task: str, objs: list[dict],
                        rel_pos: int) -> tuple[Optional[dict], Optional[str]]:
    """给定关系词位置，解析锚点物体：关系词后第一个物体优先（英文语序），
    否则取关系词前最近的物体（中文语序，且前面至少 2 个物体才敢认锚点）。"""
    mentions = _object_mentions(task, objs)
    before = [(p, o) for p, o in mentions if p < rel_pos]
    after = [(p, o) for p, o in mentions if p > rel_pos]
    if after:
        return after[0][1], f"关系词后的物体 {after[0][1].get('id')} 作锚点"
    if len(before) >= 2:
        return before[-1][1], f"关系词前最近的物体 {before[-1][1].get('id')} 作锚点"
    return None, None


def _stack_relation_pos(task: str) -> Optional[int]:
    """任务里最早的叠放关系词位置（上方/上面/叠/摞/on top）。"""
    t = str(task).lower()
    pos: Optional[int] = None
    for w in _STACK_WORDS:
        i = t.find(w)
        if i >= 0 and (pos is None or i < pos):
            pos = i
    return pos


def _direction_scan(task: str) -> Optional[tuple[int, Optional[str], bool, str]]:
    """扫描**纯**横向方位词（right/left/front/back/beside），返回 (位置, relation, False, 原词)。

    复合方向（左上/top-left/斜右上方…）由 _compound_scan 单独处理，且优先级更高；
    同位置取更长/更具体的词。
    """
    t = str(task).lower()
    best: Optional[tuple[int, Optional[str], bool, str]] = None

    def _better(pos: int, word: str) -> bool:
        if best is None:
            return True
        if pos != best[0]:
            return pos < best[0]
        return len(word) > len(best[3])      # 同位置 -> 更长/更具体者优先

    for rel, words in _DIRECTION_WORDS.items():
        for w in words:
            i = t.find(w)
            if i >= 0 and _better(i, w):
                best = (i, rel, False, w)
    return best


def _compound_scan(task: str) -> Optional[tuple[int, str]]:
    """命中复合方向（左上/斜右上方/top-left/upper-front…）返回 (位置, 原词)，否则 None。

    同位置取更长者（如「斜右上方」优先于「斜右」/「右上方」）。
    """
    t = str(task).lower()
    best: Optional[tuple[int, str]] = None
    for w in _COMPOUND_DIRECTIONS:
        i = t.find(w)
        if i < 0:
            continue
        if best is None or (i, -len(w)) < (best[0], -len(best[1])):
            best = (i, w)
    return best


def _has_deictic(task: str) -> bool:
    """任务是否用了悬挂指示词（那边/这里/它/这个…）而没有点名具体目标。"""
    t = str(task).lower()
    return any(w in t for w in _DEICTIC_WORDS)


def _has_place_intent(task: str) -> bool:
    """任务是否含放置语义（要求 A：只抓不放的任务不需要放置计划）。"""
    t = str(task).lower()
    if any(w in t for w in _PLACE_WORDS):
        return True
    if _placement_surface(task) is not None:
        return True
    if any(a in t for a in _CATEGORY_ALIASES["tray"]):
        return True
    if _stack_relation_pos(task) is not None:
        return True
    return _direction_scan(task) is not None or _compound_scan(task) is not None


def _describe_place(args: dict) -> str:
    """把 place 目标压成一行短描述（给 reasoning / 日志用）。"""
    if "on_top_of" in args:
        return f"on_top_of={args['on_top_of']}"
    if "anchor_id" in args:
        return f"anchor_id={args['anchor_id']}, relation={args.get('relation')}"
    if "surface" in args:
        return f"surface={args['surface']}"
    return f"container_id={args.get('container_id')}"


def _place_options(objs: list[dict]) -> str:
    """ask_user 时列出的可选放置目标（可抓物体 / 容器 / 桌子）。"""
    grasps = [str(o.get("id")) for o in objs
              if o.get("graspable", True) is not False and not o.get("is_container")]
    conts = [str(o.get("id")) for o in objs if o.get("is_container")]
    return "当前可选：" + "/".join(grasps + conts + ["桌子"])


def _resolve_stack(task: str, objs: list[dict]) -> tuple[Optional[dict], Optional[str], bool]:
    """解析叠放关系，返回 (锚点物体, 依据说明, 任务里是否出现强叠放词)。

    锚点用 _resolve_anchor_at 的稳健启发式（中文「A 放在 B 上方」与英文 "on top of X"
    都覆盖）；找不到锚点时返回 (None, None, True)，调用方必须 ask_user，不得兜底。
    """
    rel_pos = _stack_relation_pos(task)
    if rel_pos is None:
        return None, None, False
    anchor, why = _resolve_anchor_at(task, objs, rel_pos)
    if anchor is None:
        return None, None, True
    return anchor, why, True


def _wants_correction(task: str) -> bool:
    """任务是否含指代纠正词（不是/别放/放错/重新/应该是…）。"""
    t = str(task)
    return any(w in t for w in _CORRECTION_WORDS)


#: 任务纪元标记前缀（与 agent.loop.AgentLoop.EPOCH_PREFIX 一致）。
_EPOCH_MARK = "TASK_EPOCH_JSON"


def _current_epoch_text(text: str) -> str:
    """返回最后一个任务纪元标记之后的文本（= 当前任务的对话）。

    历史里没有标记时（手工构造的短历史）整段都算当前任务，行为与旧版一致。
    """
    s = str(text)
    idx = s.rfind(_EPOCH_MARK)
    return s[idx:] if idx >= 0 else s


def _object_explicit(task: str) -> bool:
    """任务是否明确点名了物体（颜色 / 可抓类别）。

    先把放置表面词（托盘/盘子/桌子/桌面/台面/桌上/地上/table/desk/floor/tray）
    抹掉，否则「放到盘子里」会被误当成点名了 tray 类别。
    """
    t = str(task).lower()
    for words in _SURFACE_WORDS.values():
        for w in words:
            t = t.replace(w, " ")
    return bool(_mentioned_colors(t) or _mentioned_categories(t))


def _scan_history(text: str, target_id: Optional[str] = None) -> dict:
    """扫描一段历史文本，按出现顺序配对 assistant 调用与 observation，返回统计量。

    target_id 非 None 时做「归属过滤」：只有目标一致（或调用里没写目标）的成功 pick /
    失败计数才算进当前任务，避免「上一条命令抓的别的物体」污染本次决策。

    返回键：picked / placed / holding / last_ok_skill / pick_target / last_place_args /
    fails_by_target / fails_unknown / n_obs。
    """
    pending_pick: Optional[str] = None
    pending_place: Optional[dict] = None
    matched = False
    facts: dict[str, Any] = {
        "picked": False, "placed": False, "holding": False,
        "last_ok_skill": None, "pick_target": None, "last_place_args": None,
        "fails_by_target": {}, "fails_unknown": 0, "n_obs": 0,
    }
    for obj in _json_objects(text):
        if not isinstance(obj, dict):
            continue
        if "ok" in obj and "skill" in obj:  # observation
            facts["n_obs"] += 1
            skill = str(obj.get("skill"))
            ok = bool(obj.get("ok"))
            metrics = obj.get("metrics") if isinstance(obj.get("metrics"), dict) else {}
            if ok:
                facts["last_ok_skill"] = skill
                if skill in _HOLD_CHANGE_SKILLS:
                    facts["holding"] = (skill == "pick")
            if skill == "pick":
                tid = pending_pick
                if not tid and isinstance(metrics.get("target_id"), str):
                    tid = metrics["target_id"]
                hit = target_id is None or tid is None or tid == target_id
                if ok:
                    if tid:
                        facts["pick_target"] = tid     # 跨命令指代用（不受归属过滤影响）
                    if hit:
                        facts["picked"] = True
                    matched = hit
                elif tid is None:
                    facts["fails_unknown"] += 1
                elif hit:
                    facts["fails_by_target"][tid] = facts["fails_by_target"].get(tid, 0) + 1
            elif skill == "place" and ok:
                facts["last_place_args"] = (dict(pending_place)
                                            if isinstance(pending_place, dict) else {})
                if matched:
                    facts["placed"] = True
            continue
        name = obj.get("skill", obj.get("name"))  # assistant 调用 / 纪元标记
        args = obj.get("args", obj.get("arguments"))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = None
        if name == "pick":
            tid = args.get("target_id") if isinstance(args, dict) else None
            pending_pick = tid if isinstance(tid, str) and tid else None
        elif name == "place":
            pending_place = args if isinstance(args, dict) else {}
        elif isinstance(obj.get("task"), str) and "objects" not in obj:
            pending_pick = None                      # 纪元标记：不跨纪元配对
            pending_place = None
        elif isinstance(name, str):
            pending_pick = None
            pending_place = None
    return facts


def _history_facts(text: str) -> tuple[Optional[str], Optional[str], Optional[dict]]:
    """兼容旧签名：整段历史的 (最近成功 pick 目标, 最近成功动作 skill, 最近 place 参数)。"""
    facts = _scan_history(text)
    return facts["pick_target"], facts["last_ok_skill"], facts["last_place_args"]


class MockVLM:
    """确定性规则决策器（离线 CI / 无 key 演示）。

    决策优先级（每次只返回一个调用）：
      0) 历史里出现不可重试错误码（home_blocked / stuck / object_disturbed /
         scene_reset）-> stop；否则按意图路由：回家 -> home（成功后再 stop）；停止 -> stop；
         松开夹爪 -> open_gripper（成功后再 stop）；只夹不放 -> pick 成功后直接 stop；
      1) contact.force_norm >= force_threshold 且尚未降刚度 -> set_compliance("soft")；
      2) 选出目标物体（颜色/类别关键词匹配 -> 无颜色/类别时复用历史最近一次成功 pick 的
         目标 -> 仍无则退化到第一个 graspable）；
      2b) 指代纠正（历史里有过成功 place 且本次文本含「不是/放错/重新/应该是…」）：
          若刚放下则按历史目标重新 pick；重新抓起后 place 到本次文本指定的 surface/container；
      3) 未成功抓取 -> pick(target_id)（连续失败先降刚度，再失败则 ask_user）；
      4) 已抓取、未放置 -> place(surface='table' | container_id)（pick_only 时跳过，直接 stop）；
      5) 其余 -> stop。
    任务含「轻/小心/易碎」时，pick/place 会自带 soft 导纳参数。
    """

    def __init__(self, force_threshold: float = 6.0, max_pick_retries: int = 3):
        self.force_threshold = float(force_threshold)
        self.max_pick_retries = int(max_pick_retries)

    # -- 选择 -------------------------------------------------------------- #
    def _select_target(self, task: str, objs: list[dict],
                       fallback_id: Optional[str] = None) -> Optional[dict]:
        """颜色优先、其次类别；任务没指定颜色/类别时优先复用历史最近一次成功抓取的目标，
        仍无则退化为场景列表里第一个 graspable。

        只用 scene JSON 的 color/category/graspable 字段，不硬编码任何 id。
        同分候选按 objects 列表顺序取第一个（列表顺序由 env 决定，稳定可复现）。
        """
        grasp = [o for o in objs if o.get("graspable", True) is not False]
        for color in _mentioned_colors(task):
            for o in grasp:
                if str(o.get("color", "")).lower() == color:
                    return o
        for cat in _mentioned_categories(task):
            for o in grasp:
                if str(o.get("category", "")).lower() == cat:
                    return o
        if fallback_id:
            for o in grasp:
                if str(o.get("id")) == fallback_id:
                    return o
        return grasp[0] if grasp else None

    def _select_container(self, task: str, objs: list[dict]) -> Optional[dict]:
        containers = [o for o in objs if o.get("is_container")]
        if not containers:
            return None
        t = str(task).lower()

        def _first_index(canon: str) -> int:
            idxs = [t.find(a) for a in _COLOR_ALIASES[canon]]
            idxs = [i for i in idxs if i >= 0]
            return min(idxs) if idxs else -1

        # 找「放/place」之后出现的颜色词作为容器的颜色线索
        pos = min([t.find(w) for w in _PLACE_WORDS if t.find(w) >= 0], default=-1)
        colors = [c for c in _mentioned_colors(task) if pos < 0 or _first_index(c) > pos]
        if not colors:
            colors = _mentioned_colors(task)
        for color in colors:
            for o in containers:
                if str(o.get("color", "")).lower() == color:
                    return o
        return containers[0]

    # -- 参数构造 ---------------------------------------------------------- #
    @staticmethod
    def _pick_args(target: dict, soft_task: bool) -> tuple[dict, Any]:
        """pick 的参数（+ 可选的 soft 导纳档位）。"""
        args: dict[str, Any] = {"target_id": str(target.get("id")), "approach_height": 0.10,
                                "close_ratio": 0.35}
        cp = None
        if soft_task:
            cp = compliance_from_profile("soft")
            args["compliance"] = cp.to_json()
        return args, cp

    def _explicit_container(self, task: str, objs: list[dict]) -> Optional[dict]:
        """任务里明确提到「托盘/盘子/tray」时才返回容器；否则 None。

        这是 task-31 要求 B 的关键：不再在没有任何放置语义时盲目退化到 containers[0]（托盘）。
        """
        t = str(task).lower()
        if not any(a in t for a in _CATEGORY_ALIASES["tray"]):
            return None
        return self._select_container(task, objs)

    def _placement_plan(self, task: str, objs: list[dict], soft_task: bool
                        ) -> tuple[Optional[dict], Any, Optional[str], Optional[str]]:
        """解析放置目标，返回 (args, compliance, ask_question, note)。

        args 非 None -> 可以放置；否则 ask_question 是应 ask_user 的内容。
        优先序：叠放 on_top_of > 横向 anchor_id+relation > surface='table' > 显式容器 container_id；
        任何解析不出来的情况都返回 ask_question（大声报错、回显已识别锚点，绝不静默丢托盘）。
        note 是写进 reasoning 的判定依据（要求 D）。
        """
        compound = _compound_scan(task)
        anchor, why, stack_hit = _resolve_stack(task, objs)
        lat = _direction_scan(task)
        args: Optional[dict[str, Any]] = None
        note: Optional[str] = None
        if compound is not None:
            # 复合方向优先级最高：> 纯方向、> 纯叠放。
            # 「斜右上方」既不能被当成 left/right，也不能被当成 on_top_of。
            cpos, cword = compound
            canchor, _cwhy = _resolve_anchor_at(task, objs, cpos)
            if canchor is not None:
                return None, None, (
                    f"我识别到锚点 {canchor.get('id')}，但「{cword}」这种方向我目前不支持；"
                    f"支持 {list(RELATION_NAMES)}，或直接给桌面坐标 xy=[x,y]。"), None
            return None, None, (
                f"我识别到复合方向「{cword}」，但没找到锚点物体。{_place_options(objs)}"), None
        if anchor is not None:
            if anchor.get("is_container") or anchor.get("graspable", True) is False:
                return None, None, (
                    f"我识别到锚点 {anchor.get('id')}，但它是容器/不可抓物，不能当叠放锚点"
                    f"（托盘/桌面不能当锚点）。{_place_options(objs)}"), None
            args = {"on_top_of": str(anchor.get("id"))}
            note = f"叠放依据：{why}"
        elif stack_hit:
            rel = _first_relation_word(task)
            return None, None, (
                f"我识别到叠放关系「{rel}」，但找不到锚点物体。请指明锚点。"
                f"{_place_options(objs)}"), None
        elif lat is not None:
            pos, relation, _unsupported, word = lat
            lanchor, lwhy = _resolve_anchor_at(task, objs, pos)
            if lanchor is None:
                return None, None, (
                    f"我识别到方位词「{word}」，但找不到锚点物体。请指明锚点。"
                    f"{_place_options(objs)}"), None
            if lanchor.get("is_container") or lanchor.get("graspable", True) is False:
                return None, None, (
                    f"我识别到锚点 {lanchor.get('id')}，但它不可抓（托盘/容器不能当方位锚点）。"
                    f"{_place_options(objs)}"), None
            args = {"anchor_id": str(lanchor.get("id")), "relation": str(relation)}
            note = f"横向方位依据：{lwhy}"
        elif _wants_table_surface(task):
            args = {"surface": "table"}
            note = "任务里点名了桌面"
        else:
            cont = self._explicit_container(task, objs)
            if cont is not None:
                args = {"container_id": str(cont.get("id")), "offset": [0.0, 0.0, 0.02]}
                note = f"任务里点名了容器 {cont.get('id')}（不再退化到默认托盘）"
        if args is None:
            rel = _first_relation_word(task)
            head = (f"我不确定要放到哪里：任务里的「{rel}」无法解析出容器/桌面/锚点物体"
                    if rel else "我不确定要放到哪里：没有可确认的放置目标（容器/桌面/锚点物体）")
            return None, None, f"{head}，请指明。{_place_options(objs)}", None
        cp = None
        if soft_task:
            cp = compliance_from_profile("soft")
            args["compliance"] = cp.to_json()
        return args, cp, None, note

    def _pick_decision(self, target: dict, soft_fails: int, cap_fails: int, comp_done: bool,
                       soft_task: bool, note: str = "") -> SkillCall:
        """共用的抓取决策：失败先降刚度，超过重试上限转 ask_user，否则 pick(target)。

        soft_fails: 与本目标相关的失败数（含目标未知的保守计数）——达到 1 就先切 soft；
        cap_fails : 明确归属本目标的失败数——只有它达到上限才转 ask_user，
                    避免「上一条命令抓别的物体失败」把新目标直接判死（D2）。
        note 用于在 reasoning 里标注上下文（如「纠正指令，重新抓取刚放下的 box_blue」）。
        """
        tag = f"{note}，" if note else ""
        if cap_fails >= self.max_pick_retries:
            return SkillCall(
                "ask_user",
                {"question": f"连续 {cap_fails} 次抓取 {target.get('id')} 失败，请确认物体位置或更换目标。"},
                reasoning=f"MockVLM: {tag}抓取重试次数达上限，转人工确认")
        if soft_fails >= 1 and not comp_done:
            cp = compliance_from_profile("soft")
            return SkillCall("set_compliance", {"profile": "soft"}, compliance=cp,
                             reasoning=f"MockVLM: {tag}抓取已失败 {soft_fails} 次，先降刚度再重试")
        pargs, pcp = self._pick_args(target, soft_task)
        return SkillCall("pick", pargs, compliance=pcp,
                         reasoning=f"MockVLM: {tag}选择目标 {target.get('id')}"
                                   f"（color={target.get('color')}, category={target.get('category')}）"
                                   + ("，轻拿轻放 -> soft" if soft_task else ""))

    @staticmethod
    def _placement_matches(want: Optional[dict], done: Optional[dict]) -> bool:
        """上一次成功 place 是否已经落在本次要求的放置目标上。"""
        if not want or not isinstance(done, dict):
            return False
        if "surface" in want:
            return done.get("surface") == want["surface"]
        if "on_top_of" in want:
            return done.get("on_top_of") == want["on_top_of"]
        return done.get("container_id") == want.get("container_id")

    # -- 决策 -------------------------------------------------------------- #
    def decide(self, messages: list[dict]) -> SkillCall:
        if not isinstance(messages, list):
            raise ValueError("MockVLM.decide 需要 messages 列表")
        text = _messages_text(messages)
        scene = _find_scene(text)
        if scene is None:
            return SkillCall("stop", {}, reasoning="MockVLM: 消息里找不到 SCENE JSON，无法决策，停止")

        task = str(scene.get("task", ""))
        objs = [o for o in scene.get("objects", []) if isinstance(o, dict)]
        contact = scene.get("contact") if isinstance(scene.get("contact"), dict) else {}
        try:
            force_norm = float(contact.get("force_norm") or 0.0)
        except (TypeError, ValueError):
            force_norm = 0.0

        # 跨命令历史（task-29）：状态统计只看「当前任务纪元」（最后一个 TASK_EPOCH_JSON
        # 之后）；整段历史另用于跨命令指代（最近成功 pick 的目标 / 最近 place 的参数）。
        text_cur = _current_epoch_text(text)
        obs = _find_observations(text_cur)
        last_skill = str(obs[-1].get("skill")) if obs else None
        comp_done = any(o.get("skill") == "set_compliance" and bool(o.get("ok")) for o in obs)
        soft_task = _wants_soft(task)

        facts_all = _scan_history(text)
        hist_target = facts_all["pick_target"]
        last_place_args = facts_all["last_place_args"]
        target = self._select_target(task, objs, fallback_id=hist_target)

        # 归属过滤：任务明确点名了物体时，只认同物体的历史观测，避免上一条命令
        # 抓过/放过的别的物体污染 picked / placed / pick_fails（验证者的 D2/D3）。
        explicit = _object_explicit(task)
        attrib_id = str(target.get("id")) if (explicit and target is not None) else None
        facts = _scan_history(text_cur, target_id=attrib_id)
        picked = facts["picked"]
        placed = facts["placed"]
        # holding 是「物理是否夹着东西」，不随命令切换重置：只看整段历史里最近一次
        # 改变夹持状态的动作（否则新命令会对着已夹好的物体再抓一次）。
        holding = facts_all["holding"]

        def _fails_for(tid: Optional[str]) -> tuple[int, int]:
            """(soft 触发计数, 硬上限计数)——目标未知的失败保守计入 soft，但不触发 ask_user。"""
            known = facts["fails_by_target"].get(tid, 0) if tid else 0
            return known + facts["fails_unknown"], known

        # 0) 终局检查 + 意图路由（最高优先级，放在接触力检查之前：卡死/高压时
        #    「松开夹爪」「停下」必须能立即生效）。
        # 0a) 不可重试错误：历史里出现过致命失败（home_blocked / stuck /
        #     object_disturbed / scene_reset）就立即 stop，不再产生同类调用，
        #     避免 AgentLoop 空转到 max_steps。
        fatal = [o for o in obs
                 if not bool(o.get("ok"))
                 and str(o.get("error_code") or "") in _NON_RETRYABLE_ERRORS]
        if fatal:
            last = fatal[-1]
            return SkillCall(
                "stop", {},
                reasoning=f"MockVLM: 上一步 {last.get('skill')} 返回不可重试错误 "
                          f"{last.get('error_code')}，停止重试")

        # 0b) 意图路由：回家 / 停止 / 松开夹爪 / 只夹不放。
        #     这些意图与 pick->place 无关，必须先分流，否则「回到默认位置」会被当成
        #     抓取任务、松开夹爪会被当成抓取任务，停止命令也会被继续执行。
        if _wants_home(task):
            if any(o.get("skill") == "home" and bool(o.get("ok")) for o in obs):
                return SkillCall("stop", {}, reasoning="MockVLM: 已回到默认位置，任务完成")
            return SkillCall("home", {}, reasoning="MockVLM: 任务要求回到默认位置")

        if _wants_stop(task):
            return SkillCall("stop", {}, reasoning="MockVLM: 任务要求停止，结束决策")

        pick_only = _pick_only(task)
        if _wants_open(task) and not pick_only:
            if any(o.get("skill") == "open_gripper" and bool(o.get("ok")) for o in obs):
                return SkillCall("stop", {}, reasoning="MockVLM: 夹爪已松开，任务完成")
            return SkillCall("open_gripper", {}, reasoning="MockVLM: 任务要求松开夹爪")

        # 1) 接触力超阈值 -> 先降刚度
        if force_norm >= self.force_threshold and not comp_done and last_skill != "set_compliance":
            cp = compliance_from_profile("soft")
            return SkillCall("set_compliance", {"profile": "soft"}, compliance=cp,
                             reasoning=f"MockVLM: 接触力 {force_norm:.2f}N 超阈值 "
                                       f"{self.force_threshold:.1f}N，先切 soft 导纳")

        if target is None:
            return SkillCall("ask_user", {"question": "场景里没有可抓取物体，请确认物体摆放。"},
                             reasoning="MockVLM: objects 中没有 graspable=true 的物体")

        # 2) 指代纠正 / 无上下文的纠正指令
        correction_words = _wants_correction(task)
        has_reference = (hist_target is not None) or (last_place_args is not None)
        if correction_words and not has_reference and not explicit:
            # D1：历史里没有任何可参考的抓取/放置（例如刚 :clear-history），任务又只有
            # 纠正/否定词 -> 不要猜物体，直接问人并给出候选清单。
            ids = [str(o.get("id")) for o in objs if o.get("graspable", True) is not False]
            hint = f"当前可选：{', '.join(ids)}" if ids else "当前场景没有可抓取物体"
            return SkillCall(
                "ask_user",
                {"question": f"你指的是哪个物体？没有上文可参考，请指明颜色/类别。{hint}"},
                reasoning="MockVLM: 纠正/否定指令但无历史可参考，不猜物体，转人工确认")
        # 历史里有过成功 place 且本次含纠正词 -> 把刚放下的物体重新放到本次指定的位置。
        correction = correction_words and last_place_args is not None
        force_place = False
        correction_place: Optional[tuple[Optional[dict], Any, Optional[str]]] = None
        if correction:
            want_args, want_cp, want_q, want_note = self._placement_plan(task, objs, soft_task)
            if want_args is None:
                # 新的放置目标表达不出来 -> 大声报错，不退回常规逻辑乱放（要求 B）
                return SkillCall("ask_user",
                                 {"question": want_q or "我不确定要放到哪里，请指明目标。"},
                                 reasoning="MockVLM: 纠正指令的放置目标无法解析，转人工确认")
            if self._placement_matches(want_args, last_place_args):
                return SkillCall("stop", {},
                                 reasoning="MockVLM: 上一步已放到本次指定位置，纠正完成")
            if not holding:
                # 物体已经放下（末尾可能还跟着 stop/home 等非夹持动作）-> 先按历史目标
                # 重新抓取，忽略本次文本里的颜色/类别；否则会直接 place 得到 nothing_held
                grab = target
                if hist_target:
                    found = next((o for o in objs if str(o.get("id")) == hist_target), None)
                    if found is not None:
                        grab = found
                if grab is None:
                    return SkillCall("ask_user",
                                     {"question": "历史里没有可复用的抓取目标，请指定物体。"},
                                     reasoning="MockVLM: 纠正指令但缺少可复用历史目标")
                soft_n, cap_n = _fails_for(str(grab.get("id")))
                return self._pick_decision(grab, soft_n, cap_n, comp_done, soft_task,
                                           note=f"纠正指令，重新抓取刚放下的 {grab.get('id')}")
            else:
                # 当前仍夹着物体 -> 直接走下面的放置分支
                force_place = True
                correction_place = (want_args, want_cp, want_note)

        # 3) 抓取阶段（force_place 表示「纠正指令下已经夹着物体」，直接跳到放置）
        if not picked and not force_place:
            planned: Optional[str] = None
            # 要求 A + R1：还没抓之前先确认放置目标与指代：
            # - 有放置语义（容器/表面/叠放/横向）-> 必须先能解析出目标；
            # - 只有悬挂指示词（那边/这里/它…）且没点名物体 -> 同样先问清楚，
            #   否则会先抓 box_red 再发现不知道放哪（半执行）。
            place_intent = _has_place_intent(task)
            deictic = (not explicit) and _has_deictic(task)
            if place_intent or deictic:
                plan_args, _plan_cp, plan_q, _plan_note = self._placement_plan(task, objs, soft_task)
                if plan_args is None:
                    question = plan_q or "我不确定要放到哪里，请先指明放置目标。"
                    if deictic and not place_intent:
                        question = ("我识别到你用了指示词，但任务没有点名具体物体/位置，"
                                    f"请指明要抓什么、放到哪里。{_place_options(objs)}")
                    return SkillCall(
                        "ask_user", {"question": question},
                        reasoning="MockVLM: 放置目标/指代无法确定，先问清楚再抓（不做半执行）")
                planned = _describe_place(plan_args)
            soft_n, cap_n = _fails_for(str(target.get("id")))
            call = self._pick_decision(target, soft_n, cap_n, comp_done, soft_task)
            if planned:
                call.reasoning = (call.reasoning + f"；已规划放置：{planned}").strip()
            return call

        # 4) 放置阶段
        if not placed or force_place:
            if pick_only and not force_place:
                return SkillCall("stop", {}, reasoning="MockVLM: 只要求夹住，不放置")
            note: Optional[str] = None
            if force_place and correction_place is not None:
                pargs, cp, note = correction_place
            else:
                pargs, cp, question, note = self._placement_plan(task, objs, soft_task)
                if pargs is None:
                    return SkillCall(
                        "ask_user",
                        {"question": question or "我不确定要放到哪里，请指明目标。"},
                        reasoning="MockVLM: 放置目标无法解析，转人工确认（不静默丢托盘）")
            on_top = pargs.get("on_top_of")
            surface = pargs.get("surface")
            aid = pargs.get("anchor_id")
            if isinstance(on_top, str):
                desc = f"叠放到 {on_top} 上方"
            elif isinstance(aid, str):
                desc = f"横向放到 {aid} 的 {pargs.get('relation')} 侧（anchor_id={aid}）"
            elif isinstance(surface, str):
                desc = f"放到{'桌面' if surface == 'table' else '托盘面'}（surface={surface!r}）"
            else:
                desc = f"放入容器 {pargs.get('container_id')}"
            if correction:
                desc = "纠正指令，" + desc
            return SkillCall("place", pargs, compliance=cp,
                             reasoning=f"MockVLM: {desc}"
                                       + (f"（{note}）" if note else "")
                                       + ("，轻放 -> soft" if soft_task else ""))

        # 5) 完成
        return SkillCall("stop", {}, reasoning="MockVLM: 目标已抓取并放置，任务完成")


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
def make_vlm(prefer_api: bool = True):
    """有 key 且 prefer_api=True -> VLMClient；否则 MockVLM（保证 demo 离线可跑）。"""
    client = VLMClient()
    if prefer_api and client.available:
        return client
    return MockVLM()
