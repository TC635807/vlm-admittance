"""tests/test_isolation_no_truth.py —— 毒丸隔离测试（INTERFACES.md §5.7 / §5.8-2）。

目标：vlm 闭环运行路径**不得**读物体真值（env.object_pose / set_object_pose / state.objects[*].pose）。

三组断言：
  A. 字面毒丸：object_pose/set_object_pose 一律抛异常，用 PixelScriptVLM 跑完整 vlm 闭环必须成功。
     （§5.7 的字面要求；当前实现是否满足见运行输出，红就是红。）
  B. 调用栈毒丸：只允许 sim/env.py 自身的快照组装调用 object_pose；agent/skills/planning/control/
     sim/perception 等运行路径**直接**调用一律抛异常。闭环必须成功。
  C. 消息隔离：vlm 模式送进模型的 messages / SCENE JSON 里不存在 pose/size 与真值坐标三元组。

运行：
    cd /home/tc63/mujoco/vlm_admittance
    MUJOCO_GL=glfw DISPLAY=:0 /home/tc63/mujoco/.venv/bin/python -m tests.test_isolation_no_truth
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.types import SkillCall                              # noqa: E402
from agent.loop import AgentLoop                              # noqa: E402
from sim.env import MuJoCoArmEnv                              # noqa: E402
from sim.vision import pixel_to_table, project_point          # noqa: E402

from tests.pixel_script_vlm import (PixelScriptVLM,           # noqa: E402
                                    SceneLeak, extract_scene, messages_text)

POISON = "POISON: 运行路径读了物体真值"
_FLOAT_RE = re.compile(r"-?\d+\.\d+")
_TRIP: dict = {}


def _make_app(seed: int = 0):
    from demo.interactive import InteractiveApp
    args = argparse.Namespace(vlm="mock", perception="vlm", with_barrier=False, seed=seed,
                              realtime=False, headless=True, with_images=False,
                              max_steps=10, self_test=False)
    return InteractiveApp(args)


def _install_poison(env, *, allow_sim_env: bool):
    """返回 restore()。allow_sim_env=True 时只放行 sim/env.py 内部的快照组装。"""
    orig_pose, orig_set = env.object_pose, env.set_object_pose

    def _allowed():
        if not allow_sim_env:
            return False
        for fr in inspect.stack()[2:]:
            if str(fr.filename).endswith("sim/env.py"):
                return True
        return False

    def poisoned_pose(obj_id=None, *a, **k):
        if _allowed():
            return orig_pose(obj_id, *a, **k)
        _TRIP["where"] = "".join(
            f"  {f.filename}:{f.lineno} {f.function}\n" for f in inspect.stack()[1:6])
        raise AssertionError(POISON)

    def poisoned_set(obj_id=None, *a, **k):
        if _allowed():
            return orig_set(obj_id, *a, **k)
        _TRIP["where"] = "".join(
            f"  {f.filename}:{f.lineno} {f.function}\n" for f in inspect.stack()[1:6])
        raise AssertionError(POISON)

    env.object_pose = poisoned_pose
    env.set_object_pose = poisoned_set

    def restore():
        env.object_pose = orig_pose
        env.set_object_pose = orig_set

    return restore


def _script(pixel=(262, 318)):
    """一次 pick + 一次 place（都用像素，不点名 id）。"""
    return [
        SkillCall("pick", {"target_pixel": list(pixel), "pixel_camera": "top"}),
        SkillCall("place", {"surface": "table", "target_pixel": list(pixel),
                            "pixel_camera": "top"}),
    ]


def _run_closed_loop(app, script):
    vlm = PixelScriptVLM(script)
    loop = AgentLoop(app.env, executor=app.skills, vlm=vlm, cameras=(), max_steps=6, log=None,
                     pixel_resolver=lambda u, v, cam: pixel_to_table(app.env, u, v, camera=cam),
                     include_poses=False)
    results = loop.run("把红色方块放到桌子上")
    return vlm, results


def _truth_pixel(app, obj_id: str = "box_red"):
    """真值目标中心的 top 投影（tests 允许读真值；只用于造脚本像素，绝不进 prompt）。"""
    pose = np.asarray(app.env.object_pose(obj_id), float)
    px = project_point(app.env, pose[:3], camera="top")
    assert px is not None, f"{obj_id} 应能被 top 相机看到"
    return [float(px[0]), float(px[1])]


def _scan_truth(batches, truths):
    """结构键 + 数值双扫描：返回 (struct_hits, numeric_hits)。"""
    struct_hits, numeric_hits = [], []
    for bi, messages in enumerate(batches):
        scene = extract_scene(messages)                 # scene 里出现 pose/size -> SceneLeak
        if scene is None:
            struct_hits.append((bi, "missing_scene"))
            continue
        blob = json.dumps(scene, ensure_ascii=False)
        for key in ("relations", "supported_by"):
            if re.search(r'"%s"\s*:' % key, blob):
                struct_hits.append((bi, key))
        nums = [float(x) for x in _FLOAT_RE.findall(messages_text(messages))]
        for oid, t in truths.items():
            for i in range(len(nums) - 2):
                if np.allclose(np.array(nums[i:i + 3]), t, atol=1e-3):
                    numeric_hits.append((bi, oid))
    return struct_hits, numeric_hits


class TestPoisonLiteral(unittest.TestCase):
    def test_literal_poison_loop_must_succeed(self):
        _TRIP.clear()
        app = _make_app()
        px = _truth_pixel(app)                       # 真值像素：让这次 pick 真的成功（隔离才是重点）
        restore = _install_poison(app.env, allow_sim_env=False)
        try:
            vlm, results = _run_closed_loop(app, _script(px))
        except AssertionError as exc:
            self.fail("字面毒丸：运行路径仍调用 object_pose/set_object_pose（异常从闭环里冒出）\n"
                      f"抛出点调用栈:\n{_TRIP.get('where', '')}")
        finally:
            restore()
        print(f"\n[probe] 字面毒丸: {[(r.skill, r.ok, r.error_code) for r in results]}")
        self.assertTrue(results and all(r.ok for r in results),
                        f"字面毒丸下闭环未成功: {[(r.skill, r.error_code, r.message) for r in results]}")


class TestPoisonStackAware(unittest.TestCase):
    def test_runtime_path_never_directly_reads_truth(self):
        _TRIP.clear()
        app = _make_app()
        px = _truth_pixel(app)
        restore = _install_poison(app.env, allow_sim_env=True)
        try:
            vlm, results = _run_closed_loop(app, _script(px))
        finally:
            restore()
        print(f"[probe] 栈感知毒丸: {[(r.skill, r.ok, r.error_code) for r in results]}")
        if _TRIP.get("where"):
            self.fail("运行路径直接调用 object_pose/set_object_pose（§5.7 禁止）:\n"
                      + _TRIP["where"])
        self.assertTrue(results and all(r.ok for r in results),
                        f"闭环未成功: {[(r.skill, r.error_code, r.message) for r in results]}")
        self.assertGreaterEqual(len(vlm.batches), 3)


class TestMessageIsolation(unittest.TestCase):
    def test_vlm_messages_have_no_truth(self):
        app = _make_app()
        truths = {o.id: np.asarray(o.pose[:3], float) for o in app.env.list_objects()}
        px = _truth_pixel(app)
        vlm, results = _run_closed_loop(app, _script(px))
        struct_hits, numeric_hits = _scan_truth(vlm.batches, truths)
        print(f"[probe] 消息隔离: 批次数={len(vlm.batches)} 结构命中={struct_hits} "
              f"数值命中={numeric_hits}")
        self.assertEqual(struct_hits, [])
        self.assertEqual(numeric_hits, [])
        self.assertTrue(results and all(r.ok for r in results))

    def test_scene_leak_detector_works(self):
        fake = [{"role": "user", "content": 'SCENE_JSON_BEGIN {"objects":[{"id":"box_red",'
                                            '"pose":[0.35,-0.14,0.42],"size":[0.02,0.02,0.02]}]} '
                                            'SCENE_JSON_END'}]
        with self.assertRaises(SceneLeak):
            extract_scene(fake)


class TestWrongObjectPixelHonestFailure(unittest.TestCase):
    """脚本像素故意指向**错误物体**：必须诚实失败，且消息依旧不含真值。

    这条同时覆盖两件事：
      * 信息隔离：失败路径也不得把真值位姿喂进 prompt/state；
      * P0 抓取核验：抓空/抓错不得报 ok=True（grasp_not_confirmed 或 grasp_failed）。
    """

    def test_wrong_object_pixel_is_honest_and_isolated(self):
        app = _make_app()
        truths = {o.id: np.asarray(o.pose[:3], float) for o in app.env.list_objects()}
        target = "box_red"
        before = np.asarray(app.env.object_pose(target), float).copy()
        # 任务要的是 box_red，像素却指向 box_blue（真值只用来造脚本，不进 prompt）
        wrong_px = _truth_pixel(app, "box_blue")
        vlm, results = _run_closed_loop(app, _script(wrong_px))
        after = np.asarray(app.env.object_pose(target), float).copy()
        dz = float(after[2] - before[2])
        first = results[0]
        struct_hits, numeric_hits = _scan_truth(vlm.batches, truths)
        print(f"\n[probe] 错物体像素 {wrong_px}: ok={first.ok} err={first.error_code} "
              f"Δz({target})={dz * 1000:+.1f}mm 结构命中={struct_hits} 数值命中={numeric_hits}")
        self.assertFalse(first.ok, f"指向错误物体却报成功: {first.message}")
        self.assertIn(first.error_code, ("grasp_not_confirmed", "grasp_failed"),
                      f"必须是诚实失败，实际 {first.error_code}: {first.message}")
        self.assertLessEqual(abs(dz), 0.002, f"目标 {target} 不应被抬起，实测 Δz={dz * 1000:.1f}mm")
        self.assertEqual(struct_hits, [])
        self.assertEqual(numeric_hits, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
