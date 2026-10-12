"""PixelScriptVLM —— 离线脚本化像素决策器（INTERFACES.md §5.9，verifier 维护）。

用途：网络不可用时驱动**完整 vlm 闭环**（含像素→世界、颜色精修、子目标复核的代码路径）。
铁律：只读 SCENE_JSON 里的 objects[*].id/color/category，**绝不读 pose/size**；
一旦发现 scene 里出现真值字段就抛 SceneLeak（配合 tests/test_isolation_no_truth.py 的毒丸）。
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from core.types import SkillCall

_SCENE_RE = re.compile(r"SCENE_JSON_BEGIN\s*(\{.*?\})\s*SCENE_JSON_END", re.S)
_TRUTH_KEYS = ("pose", "size")


class SceneLeak(AssertionError):
    """SCENE 里出现物体真值字段（vlm 模式不该有）。"""


def messages_text(messages) -> str:
    chunks: list = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            chunks.append(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    chunks.append(str(part.get("text", "")))
                else:
                    chunks.append(str(part))
    return "\n".join(chunks)


def extract_scene(messages) -> Optional[dict]:
    """取 SCENE JSON；若 objects[*] 含 pose/size 立即抛 SceneLeak。"""
    m = _SCENE_RE.search(messages_text(messages))
    if not m:
        return None
    scene = json.loads(m.group(1))
    for o in scene.get("objects", []) or []:
        if not isinstance(o, dict):
            continue
        for bad in _TRUTH_KEYS:
            if bad in o:
                raise SceneLeak(f"scene 出现物体真值字段 {bad!r}（id={o.get('id')!r}）")
    return scene


def scene_brief(messages) -> list:
    """[(id, color, category)]；不读任何坐标。"""
    scene = extract_scene(messages) or {}
    out = []
    for o in scene.get("objects", []) or []:
        if isinstance(o, dict):
            out.append((str(o.get("id")), str(o.get("color")), str(o.get("category"))))
    return out


class PixelScriptVLM:
    """按剧本返回 SkillCall；剧本耗尽后 stop。每步记录 messages 与 scene 摘要。"""

    def __init__(self, script: list, *, log: Optional[Any] = None):
        self.script = list(script)
        self.i = 0
        self.log = log
        self.batches: list = []
        self.scenes: list = []

    def decide(self, messages) -> SkillCall:
        self.batches.append(messages)
        self.scenes.append(scene_brief(messages))     # 顺带做真值隔离检查
        if self.i < len(self.script):
            call = self.script[self.i]
            self.i += 1
            if callable(self.log):
                self.log(f"[PixelScriptVLM] step {self.i}: {call.skill} {call.args}")
            return call
        return SkillCall("stop", {}, reasoning="PixelScriptVLM: 剧本结束")
