"""tests/test_grasp_verification.py —— 抓取成功必须经物理/夹爪核实（防「抓空报成功」）。

实机证据：真模型 vlm 模式 pick 报 ok=True（指间距 45.4mm）但真值 box_red 完全没动
（Δxy=9.6mm、Δz=0），place 还报 ok=True —— 抓空被报成成功。
夹具标定（Lead 实测）：空夹爪 ratio 1.0/0.6/0.35/0.0 -> gap 92.69/58.91/37.78/10.28 mm。

断言（真值只用于断言，绝不进 prompt）：
  a) 故意偏 15~20px 的像素 -> 物理抓空（真值 z 不变）-> 必须给可重试 grasp_not_confirmed，绝不 ok=True；
  b) 正确像素 -> 真值确认被抬起（Δz > 100mm）-> metrics.grasp_verified=True；
  c) 颜色未知（既不 locate 也无 target_id）-> 不报错、日志/指标明示「抓取未验证」，仍按旧行为执行。

grasp_not_confirmed / grasp_verified 尚未落地时整类 skip（pending），落地后自动生效。

运行：
    MUJOCO_GL=glfw DISPLAY=:0 /home/tc63/mujoco/.venv/bin/python -m tests.test_grasp_verification
"""
from __future__ import annotations

import argparse
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.types import SkillCall                              # noqa: E402
from agent.loop import AgentLoop                              # noqa: E402
from sim.vision import pixel_to_table, project_point          # noqa: E402
from tests.pixel_script_vlm import PixelScriptVLM             # noqa: E402

_LOOP_SRC = (_ROOT / "agent" / "loop.py").read_text(encoding="utf-8")
_LIB_SRC = (_ROOT / "skills" / "library.py").read_text(encoding="utf-8")
# P0 落在 agent/loop.py（无真值抓取核验），skills 侧可能先给 grasp_failed —— 两边都要 grep
_GRASP_VERIFY = any(k in (_LOOP_SRC + _LIB_SRC)
                    for k in ("grasp_not_confirmed", "grasp_verified", "grasp_failed"))
_SKIP = ("抓取核实未落地（agent/loop.py 与 skills/library.py 里都没有 "
         "grasp_not_confirmed / grasp_verified / grasp_failed）—— 交付后自动生效")

TARGET = "box_red"


def _make_app(seed: int = 0):
    from demo.interactive import InteractiveApp
    args = argparse.Namespace(vlm="mock", perception="vlm", with_barrier=False, seed=seed,
                              realtime=False, headless=True, with_images=False,
                              max_steps=6, self_test=False)
    return InteractiveApp(args)


def _run_pick(call: SkillCall, seed: int = 0):
    """跑一次 vlm 闭环 pick，返回 (results, truth_before, truth_after, logs)。"""
    app = _make_app(seed)
    before = np.asarray(app.env.object_pose(TARGET), float).copy()
    logs: list = []
    loop = AgentLoop(app.env, executor=app.skills, vlm=PixelScriptVLM([call]), cameras=(),
                     max_steps=4, log=lambda m: logs.append(str(m)),
                     pixel_resolver=lambda u, v, cam: pixel_to_table(app.env, u, v, camera=cam),
                     include_poses=False)
    results = loop.run("把红色方块拿起来")
    after = np.asarray(app.env.object_pose(TARGET), float).copy()
    return results, before, after, logs, app


def _truth_pixel(app):
    return project_point(app.env, np.asarray(app.env.object_pose(TARGET)[:3], float),
                         camera="top")


@unittest.skipUnless(_GRASP_VERIFY, _SKIP)
class TestGraspVerification(unittest.TestCase):
    def test_wrong_pixel_grasps_air_and_must_report_retryable_failure(self):
        app0 = _make_app(0)
        px = _truth_pixel(app0)
        self.assertIsNotNone(px, "box_red 应能被 top 相机看到")
        bad = [float(px[0] + 18.0), float(px[1] - 18.0)]        # 约 25px ≈ 38mm，必然抓空
        results, before, after, logs, _app = _run_pick(
            SkillCall("pick", {"target_pixel": bad, "pixel_camera": "top"}))
        dz = float(after[2] - before[2])
        first = results[0]
        print(f"\n[probe] 偏像素 {bad} -> ok={first.ok} err={first.error_code} "
              f"Δz={dz * 1000:+.1f}mm Δxy={float(np.linalg.norm(after[:2] - before[:2])) * 1000:.1f}mm")
        self.assertLessEqual(abs(dz), 0.002, f"前提：这次应当物理抓空，实测 Δz={dz * 1000:.1f}mm")
        self.assertFalse(first.ok, "抓空绝不能报 ok=True")
        # skills 可能先自己返回 grasp_failed，loop 侧核验给 grasp_not_confirmed —— 两者都是诚实失败
        self.assertIn(first.error_code, ("grasp_not_confirmed", "grasp_failed"),
                      f"抓空应给 grasp_not_confirmed/grasp_failed，实际 {first.error_code}: "
                      f"{first.message}")
        nonretry = getattr(AgentLoop, "NON_RETRYABLE_ERRORS", ()) or ()
        self.assertNotIn(first.error_code, nonretry, "grasp_not_confirmed 必须可重试")

    def test_correct_pixel_lifts_object_and_reports_verified(self):
        app0 = _make_app(0)
        px = _truth_pixel(app0)
        results, before, after, logs, _app = _run_pick(
            SkillCall("pick", {"target_pixel": [float(px[0]), float(px[1])],
                               "pixel_camera": "top"}))
        dz = float(after[2] - before[2])
        first = results[0]
        print(f"[probe] 正确像素 -> ok={first.ok} err={first.error_code} Δz={dz * 1000:+.1f}mm "
              f"metrics.grasp_verified={ (first.metrics or {}).get('grasp_verified') }")
        self.assertTrue(first.ok, f"正确像素应抓起: {first.error_code} {first.message}")
        self.assertGreater(dz, 0.1, f"真值确认应被抬起 >100mm，实测 {dz * 1000:.1f}mm")
        self.assertIs(bool((first.metrics or {}).get("grasp_verified")), True,
                      f"必须标记 grasp_verified=True，metrics={first.metrics}")

    def test_unknown_color_is_honest_unverified_but_still_executes(self):
        app0 = _make_app(0)
        xy = np.asarray(app0.env.object_pose(TARGET)[:3], float)[:2]
        results, before, after, logs, _app = _run_pick(
            SkillCall("pick", {"target_xy": [float(xy[0]), float(xy[1])]}))
        first = results[0]
        metrics = first.metrics or {}
        text = " ".join(logs) + " " + str(metrics) + " " + str(first.message)
        print(f"[probe] 无颜色线索 -> ok={first.ok} err={first.error_code} "
              f"grasp_verified={metrics.get('grasp_verified')} 日志含未验证={'未验证' in text}")
        self.assertTrue(first.ok, f"无颜色线索时仍应按旧行为执行: {first.error_code} {first.message}")
        self.assertIsNot(bool(metrics.get("grasp_verified")), True,
                         "颜色未知时不得声称已核实")
        self.assertTrue(("未验证" in text) or (metrics.get("grasp_unverified") is True)
                        or (metrics.get("grasp_verified") is None),
                        "日志/指标必须明示「抓取未验证」")


if __name__ == "__main__":
    if not _GRASP_VERIFY:
        print(f"[PENDING] {_SKIP}")
    unittest.main(verbosity=2)
