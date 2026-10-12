"""agent 包 —— VLM 高层决策层（状态序列化 / 工具 schema / VLM 客户端 / 决策闭环）。

对外主要入口：
    from agent.state import serialize_state, encode_image, build_messages
    from agent.schema import TOOLS, system_prompt, parse_skill_call, validate_skill_call
    from agent.vlm_client import VLMClient, MockVLM, make_vlm, VLMError
    from agent.loop import AgentLoop

本包不依赖 mujoco，也不依赖任何第三方 SDK（仅 numpy + 可选 pillow + 标准库），
因此 python -m agent.smoke 无需 MUJOCO_GL 即可离线运行。
"""
from __future__ import annotations

__all__ = [
    "serialize_state", "encode_image", "build_messages",
    "TOOLS", "system_prompt", "parse_skill_call", "validate_skill_call",
    "VLMClient", "MockVLM", "make_vlm", "VLMError",
    "AgentLoop",
]
