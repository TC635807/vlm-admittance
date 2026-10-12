"""tests/test_vlm_argguard.py —— 参数冗余「纠正而非拒绝」（INTERFACES.md §5.12）。

实机诊断（outputs/diag_vla_1.log）驱动，冻结策略：
  1. vlm 模式：pick(target_id + target_pixel/target_xy) -> 丢 target_id 以像素为准，必须接受；
  2. place(container_id + xy)：落点在容器内 -> 接受 xy；容器外 -> 拒绝且错误可读；
  3. 真矛盾（surface=table + container_id、on_top_of + container_id）-> 必须拒绝；
  4. oracle 模式：pick(target_id + target_xy)，|xy - 真值| <= 30mm -> 接受；>30mm -> 拒绝；
  5. 所有纠正必须在 SkillResult.metrics 留 corrected_args。

运行：
    MUJOCO_GL=glfw DISPLAY=:0 /home/tc63/mujoco/.venv/bin/python -m tests.test_vlm_argguard
"""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.types import (ContactState, RobotState, SceneObject,       # noqa: E402
                        SceneState, SkillCall, SkillResult)
from agent.schema import validate_skill_call                          # noqa: E402


def _obj(i, c, col, pos, size, graspable=True, is_container=False):
    return SceneObject(i, c, col, np.array([pos[0], pos[1], pos[2], 1.0, 0, 0, 0]),
                       tuple(size), graspable, is_container)


def _state():
    objs = [
        _obj("box_red", "box", "red", (0.35, -0.14, 0.42), (0.02, 0.02, 0.02)),
        _obj("box_blue", "box", "blue", (0.35, 0.14, 0.42), (0.02, 0.02, 0.02)),
        _obj("tray", "tray", "gray", (0.40, 0.30, 0.412), (0.09, 0.07, 0.012), False, True),
    ]
    return SceneState(0.0, RobotState(np.zeros(6), np.zeros(6), np.zeros(3),
                                      np.array([1.0, 0, 0, 0])),
                      objs, ContactState(np.zeros(3), np.zeros(3), 0.0, False))


_MODE_KEYS = ("no_oracle", "vlm", "vlm_mode", "include_poses", "has_truth", "oracle")


def _validate(call, state, *, no_oracle=None):
    """自适应调用 validate_skill_call：支持模式参数就传，否则 2 参调用（旧签名）。"""
    sig = inspect.signature(validate_skill_call)
    kwargs = {}
    if no_oracle is not None:
        for name in _MODE_KEYS:
            if name in sig.parameters:
                kwargs[name] = (not no_oracle) if name in ("include_poses", "has_truth", "oracle") \
                    else bool(no_oracle)
                break
    return validate_skill_call(call, state, **kwargs), kwargs


class TestRedundantArgsCorrected(unittest.TestCase):
    """§5.12-1 vlm 模式：点名 + 像素 = 冗余但一致 -> 纠正后接受。"""

    def test_pick_id_plus_pixel_is_accepted_in_vlm_mode(self):
        call = SkillCall("pick", {"target_id": "box_red", "target_pixel": [267, 125],
                                  "pixel_camera": "top"})
        (ok, msg), kwargs = _validate(call, _state(), no_oracle=True)
        print(f"\n[probe] vlm pick(id+pixel): ok={ok} kwargs={kwargs} msg={msg}")
        self.assertTrue(ok, "§5.12-1：vlm 模式下 target_id+target_pixel 应纠正后接受，"
                            f"实际被拒: {msg}")
        dropped = "target_id" not in (call.args or {}) or "冗余" in msg or "忽略" in msg
        self.assertTrue(dropped, "纠正必须丢掉 target_id（或在消息里标注）")

    def test_pick_id_plus_xy_is_accepted_in_vlm_mode(self):
        call = SkillCall("pick", {"target_id": "box_red", "target_xy": [0.35, -0.14]})
        (ok, msg), _kw = _validate(call, _state(), no_oracle=True)
        print(f"[probe] vlm pick(id+xy): ok={ok} msg={msg}")
        self.assertTrue(ok, f"§5.12-1：vlm 模式下 target_id+target_xy 应接受: {msg}")

    def test_corrected_args_recorded_in_metrics(self):
        """纠正必须落到 SkillResult.metrics.corrected_args。"""
        from agent.loop import AgentLoop
        captured = {}

        class _Exec:
            def __call__(self, call):
                captured["args"] = dict(call.args or {})
                return SkillResult(True, call.skill, "ok", metrics={"seen": True})

        class _StubEnv:
            def get_state(self):
                return _state()

            def render(self, camera="overview"):
                raise RuntimeError("stub")

        class _VLM:
            def __init__(self):
                self.n = 0

            def decide(self, messages):
                self.n += 1
                if self.n == 1:
                    return SkillCall("pick", {"target_id": "box_red",
                                              "target_pixel": [267, 125],
                                              "pixel_camera": "top"})
                return SkillCall("stop", {})

        loop = AgentLoop(_StubEnv(), executor=_Exec(), vlm=_VLM(), cameras=(), max_steps=2,
                         log=None, include_poses=False,
                         pixel_resolver=lambda u, v, cam: np.array([0.35, -0.14, 0.4]))
        results = loop.run("把红色方块放到桌子上")
        first = results[0]
        print(f"[probe] corrected_args: args={captured.get('args')} metrics={first.metrics}")
        self.assertNotIn("target_id", captured.get("args", {}),
                         "执行器收到的调用里不应再有 target_id")
        self.assertIn("corrected_args", first.metrics or {},
                      "§5.12-5：纠正必须写进 SkillResult.metrics.corrected_args")


class TestContainerXyAgreement(unittest.TestCase):
    """§5.12-2 容器内 -> 接受；容器外 -> 拒绝。"""

    def test_xy_inside_container_accepted(self):
        call = SkillCall("place", {"container_id": "tray", "xy": [0.402, 0.297]})
        (ok, msg), _kw = _validate(call, _state())
        print(f"\n[probe] place(tray, xy 内): ok={ok} msg={msg}")
        self.assertTrue(ok, f"容器内落点应接受: {msg}")

    def test_xy_outside_container_rejected_readable(self):
        call = SkillCall("place", {"container_id": "tray", "xy": [0.05, -0.40]})
        (ok, msg), _kw = _validate(call, _state())
        print(f"[probe] place(tray, xy 外): ok={ok} msg={msg}")
        self.assertFalse(ok, "容器外落点必须拒绝")
        self.assertTrue(any(k in msg for k in ("容器", "托盘", "范围", "外")),
                        f"拒绝信息应可读并指明容器/范围: {msg}")


class TestTrueConflictsRejected(unittest.TestCase):
    """§5.12-3 真矛盾保持拒绝。"""

    def test_surface_table_plus_container(self):
        (ok, msg), _kw = _validate(SkillCall("place", {"surface": "table",
                                                      "container_id": "tray"}), _state())
        print(f"\n[probe] place(table+tray): ok={ok} msg={msg}")
        self.assertFalse(ok)

    def test_on_top_of_plus_container(self):
        (ok, msg), _kw = _validate(SkillCall("place", {"on_top_of": "box_red",
                                                      "container_id": "tray"}), _state())
        print(f"[probe] place(on_top_of+tray): ok={ok} msg={msg}")
        self.assertFalse(ok)


class TestOracleThirtyMmRule(unittest.TestCase):
    """§5.12-4 oracle：|xy-真值| <=30mm 接受，>30mm 拒绝。"""

    def _pick(self, xy):
        return SkillCall("pick", {"target_id": "box_red", "target_xy": list(xy)})

    def test_within_30mm_accepted(self):
        xy = [0.35 + 0.020, -0.14 - 0.010]          # 22.4mm
        (ok, msg), _kw = _validate(self._pick(xy), _state(), no_oracle=False)
        print(f"\n[probe] oracle pick(22mm): ok={ok} msg={msg}")
        self.assertTrue(ok, f"≤30mm 应接受: {msg}")

    def test_beyond_30mm_rejected(self):
        xy = [0.35 + 0.060, -0.14]                  # 60mm
        (ok, msg), _kw = _validate(self._pick(xy), _state(), no_oracle=False)
        print(f"[probe] oracle pick(60mm): ok={ok} msg={msg}")
        self.assertFalse(ok, ">30mm 必须拒绝")
        self.assertTrue(any(k in msg for k in ("30", "mm", "真值", "偏差")),
                        f"拒绝信息应给出偏差口径: {msg}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
