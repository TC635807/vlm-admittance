"""真实 VLM 端点探针：连通性 / 图片输入 / function calling 三项能力逐项验证。

用法（不会打印任何密钥）:
    cd /home/tc63/mujoco/vlm_admittance
    AI_API_URL=... AI_API_KEY=... AI_MODEL=... PYTHONPATH=. python tools/vlm_probe.py
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from agent.schema import TOOLS, system_prompt
from agent.state import encode_image
from agent.vlm_client import VLMClient


def main() -> int:
    c = VLMClient()
    print(f"[probe] base_url={c.base_url}")
    print(f"[probe] model={c.model}  key_configured={c.available}  endpoint={c.endpoint}")
    if not c.available:
        print("[probe] 未配置 key，退出")
        return 2

    # --- 1) 文本连通性 ---------------------------------------------------- #
    t0 = time.time()
    try:
        obj = c._post({"model": c.model, "max_tokens": 64,
                       "messages": [{"role": "user", "content": "只回复两个字：收到"}]})
        msg = (obj.get("choices") or [{}])[0].get("message") or {}
        print(f"[probe] 1) 文本 OK ({time.time()-t0:.1f}s) model={obj.get('model')} reply={str(msg.get('content'))[:60]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] 1) 文本 FAIL: {type(exc).__name__}: {str(exc)[:400]}")
        return 1

    # --- 2) 图片输入 ------------------------------------------------------ #
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    img[:, :, 0] = 255          # 纯红图
    b64 = encode_image(img, fmt="jpeg", quality=70)
    t0 = time.time()
    try:
        obj2 = c._post({"model": c.model, "max_tokens": 64, "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图是什么颜色？只回答颜色名。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }]})
        m2 = (obj2.get("choices") or [{}])[0].get("message") or {}
        print(f"[probe] 2) 图片 OK ({time.time()-t0:.1f}s) reply={str(m2.get('content'))[:80]!r}")
        vision = True
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] 2) 图片 FAIL: {type(exc).__name__}: {str(exc)[:400]}")
        vision = False

    # --- 3) function calling ---------------------------------------------- #
    scene = {"task": "把红色方块放进托盘", "objects": [
        {"id": "box_red", "category": "box", "color": "red", "graspable": True, "is_container": False},
        {"id": "tray", "category": "tray", "color": "gray", "graspable": False, "is_container": True}]}
    t0 = time.time()
    try:
        obj3 = c._post({"model": c.model, "max_tokens": 256, "tools": TOOLS, "tool_choice": "auto",
                        "messages": [{"role": "system", "content": system_prompt()},
                                     {"role": "user", "content": "场景: " + json.dumps(scene, ensure_ascii=False) +
                                                                "\n请选择下一个动作。"}]})
        m3 = (obj3.get("choices") or [{}])[0].get("message") or {}
        tc = m3.get("tool_calls")
        print(f"[probe] 3) tools OK ({time.time()-t0:.1f}s) tool_calls={json.dumps(tc, ensure_ascii=False)[:400]}")
        print(f"[probe] 3) content={str(m3.get('content'))[:200]!r}")
        tools_ok = bool(tc)
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] 3) tools FAIL: {type(exc).__name__}: {str(exc)[:400]}")
        tools_ok = False

    print(f"[probe] 结论: 文本=OK  图片={'OK' if vision else 'FAIL'}  function_calling={'OK' if tools_ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
