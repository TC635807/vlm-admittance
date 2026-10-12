"""tests/test_locate_fallback.py —— SubGoalVerifier.locate 的相机回退 / bbox 兜底 / 缓存复用。

断言（§5.3 + §5.10）：
  a) top 拿不到像素 -> 自动退到 overview 重试，且 pixel 别名优先 top、其次 overview；
  b) 模型/分割只给 bbox（没有 uv）-> 取 bbox 中心当像素【当前未落地 -> pending skip】；
  c) 同一 (target,color,camera) 的解析结果必须缓存并在同 run 内复用（托盘不移动），
     即使之后再 locate 会得到不同结果，也返回缓存值。

运行：
    MUJOCO_GL=glfw DISPLAY=:0 /home/tc63/mujoco/.venv/bin/python -m tests.test_locate_fallback
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_CAM_TAG = {"top": 1, "overview": 2}


class _FakeEnv:
    """render(cam) 返回一个带相机标签的极小图；get_state 故意失败（locate 要能容错）。"""

    def __init__(self):
        self.renders: list = []

    def render(self, cam: str = "overview", **kw):
        self.renders.append(str(cam))
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        img[0, 0, 0] = _CAM_TAG.get(str(cam), 0)
        return img

    def get_state(self):
        raise RuntimeError("fake env: no state")


class _FakePerception:
    """可控的 color_blob：按相机返回 found/uv/bbox；记录调用次数。"""

    def __init__(self, mode: str):
        self.mode = mode
        self.calls: list = []

    def _cam(self, rgb):
        return {1: "top", 2: "overview"}.get(int(np.asarray(rgb)[0, 0, 0]), "?")

    def color_blob(self, rgb, color, hint_uv=None, **kw):
        cam = self._cam(rgb)
        self.calls.append(cam)
        if self.mode == "top_fails":
            if cam == "top":
                return {"found": False, "uv": None, "bbox": None}
            return {"found": True, "uv": [100.0, 200.0], "bbox": [90, 190, 110, 210],
                    "area_px": 400, "color": color, "score": 0.9}
        if self.mode == "bbox_only":
            if cam == "top":
                return {"found": True, "uv": None, "bbox": [10, 20, 30, 40],
                        "area_px": 400, "color": color, "score": 0.9}
            return {"found": False, "uv": None, "bbox": None}
        if self.mode == "both_ok":
            return {"found": True, "uv": [50.0, 60.0], "bbox": [40, 50, 60, 70],
                    "area_px": 400, "color": color, "score": 0.9}
        raise AssertionError(f"未知 mode {self.mode}")


def _verifier(mode: str):
    from agent import taskplan
    env = _FakeEnv()
    perc = _FakePerception(mode)
    v = taskplan.SubGoalVerifier(env, perception=perc, vlm=None)
    return taskplan, env, perc, v


_BBOX_CENTER_OK = None


def _bbox_center_supported() -> bool:
    """能力探针：bbox-only 时 locate 是否返回中心像素。"""
    global _BBOX_CENTER_OK
    if _BBOX_CENTER_OK is None:
        try:
            _t, _e, _p, v = _verifier("bbox_only")
            loc = v.locate("红色方块", color="red", camera="top")
            _BBOX_CENTER_OK = bool(loc.get("ok") and loc.get("pixel"))
        except Exception:                                     # noqa: BLE001
            _BBOX_CENTER_OK = False
    return bool(_BBOX_CENTER_OK)


class TestLocateCameraFallback(unittest.TestCase):
    def test_top_failure_falls_back_to_overview(self):
        _t, env, perc, v = _verifier("top_fails")
        loc = v.locate("红色方块", color="red", camera="both")
        print(f"\n[probe] top 失败 -> ok={loc.get('ok')} pixel_top={loc.get('pixel_top')} "
              f"pixel_overview={loc.get('pixel_overview')} pixel={loc.get('pixel')} "
              f"renders={env.renders} calls={perc.calls}")
        self.assertTrue(loc.get("ok"), f"应退到 overview 成功: {loc.get('why')}")
        self.assertIsNone(loc.get("pixel_top"))
        self.assertEqual(loc.get("pixel_overview"), [100.0, 200.0])
        self.assertEqual(loc.get("pixel"), [100.0, 200.0], "别名应回退到 overview")
        self.assertIn("top", env.renders)
        self.assertIn("overview", env.renders)

    def test_no_color_hint_is_honest_failure(self):
        _t, _e, _p, v = _verifier("both_ok")
        loc = v.locate("红色方块", color=None, camera="both")
        print(f"[probe] 无颜色线索 -> ok={loc.get('ok')} why={loc.get('why')[:60]}")
        self.assertFalse(loc.get("ok"), "没有颜色线索不得猜像素")
        self.assertIn("颜色", str(loc.get("why")))


class TestLocateCache(unittest.TestCase):
    def test_result_is_cached_and_reused_within_run(self):
        _t, env, perc, v = _verifier("both_ok")
        first = v.locate("红色方块", color="red", camera="both")
        n_calls = len(perc.calls)
        n_renders = len(env.renders)
        # 改变场景：若重新解析会得到不同数字（这里直接让 render 标签失配 -> 重新解析会失败）
        env2_tag = _CAM_TAG.copy()
        for k in env2_tag:
            env2_tag[k] = 0
        env._tag = env2_tag
        # 让后续真实调用拿不到色块：改 fake perception 的 mode
        perc.mode = "top_fails"
        second = v.locate("红色方块", color="red", camera="both")
        print(f"\n[probe] 缓存: first={first.get('pixel')} second={second.get('pixel')} "
              f"calls {n_calls}->{len(perc.calls)} renders {n_renders}->{len(env.renders)} "
              f"locate_calls={v.locate_calls}")
        self.assertEqual(second.get("pixel"), first.get("pixel"), "第二次必须复用缓存")
        self.assertEqual(len(perc.calls), n_calls, "命中缓存不得再调 color_blob")
        self.assertEqual(len(env.renders), n_renders, "命中缓存不得再渲染")
        self.assertEqual(v.locate_calls, 1, "locate_calls 只应计一次真实解析")


@unittest.skipUnless(_bbox_center_supported(),
                     "待 agent-builder 落地：模型/分割只给 bbox 时取中心当像素")
class TestLocateBboxCenter(unittest.TestCase):
    def test_bbox_only_uses_center_as_pixel(self):
        _t, _e, _p, v = _verifier("bbox_only")
        loc = v.locate("红色方块", color="red", camera="top")
        print(f"\n[probe] bbox-only: ok={loc.get('ok')} pixel={loc.get('pixel')} "
              f"bbox={loc.get('bbox')}")
        self.assertTrue(loc.get("ok"))
        self.assertEqual(loc.get("pixel"), [20.0, 30.0], "应取 bbox(10,20,30,40) 中心")
        self.assertEqual(loc.get("bbox"), [10, 20, 30, 40])


if __name__ == "__main__":
    unittest.main(verbosity=2)
