"""tests/test_vlm_perception.py —— VLM 感知闭环独立验证（INTERFACES.md §4.5）。

只测这条新链路（不跑全量套件）：
  (1) 无真值：--perception vlm 下真正送进模型的 messages 做结构键扫描 + 数值扫描，必须无物体真值；
      对照组 oracle 必须仍带真值；MockVLM 默认路径不受影响。
  (2) 标定精度：sim.vision.reprojection_summary，>=5 个桌面点，top/overview 两相机 max/mean (mm)。
  (4) 反例：像素落桌外/相机后方/解析器异常 -> 不得崩溃，返回可重试 perception_error（含原始像素），
      且下一步 LLM 能在 messages 里看到该错误与像素以便纠错。

运行：
    cd /home/tc63/mujoco/vlm_admittance
    MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m tests.test_vlm_perception
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import TABLE_HEIGHT                      # noqa: E402
from core.types import SkillCall, SkillResult             # noqa: E402
from agent.loop import AgentLoop                          # noqa: E402
from agent.vlm_client import MockVLM                      # noqa: E402
from sim.env import MuJoCoArmEnv                          # noqa: E402
from sim.vision import (CameraModel, pixel_to_table,      # noqa: E402
                        reprojection_summary)

_FLOAT_RE = re.compile(r"-?\d+\.\d+")


def _make_app(perception: str = "vlm", vlm: str = "mock"):
    from demo.interactive import InteractiveApp
    args = argparse.Namespace(vlm=vlm, perception=perception, with_barrier=False, seed=0,
                              realtime=False, headless=True, with_images=False,
                              max_steps=12, self_test=False)
    return InteractiveApp(args)


def _messages_text(messages) -> str:
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


def _scene_json(messages):
    txt = _messages_text(messages)
    m = re.search(r"SCENE_JSON_BEGIN\s*(\{.*?\})\s*SCENE_JSON_END", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


class _RecordingVLM:
    """记录每条 messages，再委托给 MockVLM（不联网）。"""

    def __init__(self, delegate=None):
        self.delegate = delegate or MockVLM()
        self.batches: list = []

    def decide(self, messages):
        self.batches.append(messages)
        return self.delegate.decide(messages)


class TestNoOracleMessages(unittest.TestCase):
    def test_vlm_mode_messages_have_no_ground_truth(self):
        app = _make_app(perception="vlm")
        self.assertEqual(app.perception_mode, "vlm")
        rec = _RecordingVLM()
        app.vlm = rec
        results = app.run_task("把蓝色方块放到桌子上")
        self.assertTrue(results, "任务应至少产生一步")
        self.assertTrue(all(r.ok for r in results), [(r.skill, r.message) for r in results])
        self.assertTrue(rec.batches)
        truths = {o.id: np.asarray(o.pose[:3], float) for o in app.env.list_objects()}

        struct_hits: list = []
        numeric_hits: list = []
        for bi, messages in enumerate(rec.batches):
            scene = _scene_json(messages)
            self.assertIsNotNone(scene, f"第 {bi} 批 messages 缺少 SCENE JSON")
            for o in scene.get("objects", []):
                for key in ("pose", "size"):
                    if key in o:
                        struct_hits.append((bi, o.get("id"), key))
            blob = json.dumps(scene, ensure_ascii=False)
            for key in ("relations", "supported_by"):
                if re.search(r'"%s"\s*:' % key, blob):
                    struct_hits.append((bi, "scene", key))
            # 数值扫描：任何连续三元组若与某物体真值 xyz 在 1e-3 内吻合 -> 命中
            nums = [float(x) for x in _FLOAT_RE.findall(_messages_text(messages))]
            for oid, t in truths.items():
                for i in range(len(nums) - 2):
                    tri = np.array(nums[i:i + 3])
                    if np.allclose(tri, t, atol=1e-3):
                        numeric_hits.append((bi, oid, [round(v, 4) for v in tri]))
        print(f"\n[probe] vlm 模式: 批次数={len(rec.batches)} 结构命中={struct_hits} "
              f"数值命中={numeric_hits}")
        self.assertEqual(struct_hits, [], "no-oracle messages 里出现 pose/size/relations 键")
        self.assertEqual(numeric_hits, [], "no-oracle messages 里出现物体真值坐标三元组")

    def test_oracle_mode_still_has_ground_truth(self):
        app = _make_app(perception="oracle")
        self.assertEqual(app.perception_mode, "oracle")
        truth = next(o for o in app.env.list_objects() if o.id == "box_blue")  # 移动前取真值
        rec = _RecordingVLM()
        app.vlm = rec
        results = app.run_task("把蓝色方块放到桌子上")
        self.assertTrue(results and all(r.ok for r in results))
        scene = _scene_json(rec.batches[0])
        self.assertIsNotNone(scene)
        objs = {o["id"]: o for o in scene.get("objects", [])}
        self.assertIn("box_blue", objs)
        self.assertIn("pose", objs["box_blue"], "oracle 模式必须带真值位姿")
        self.assertIn("size", objs["box_blue"])
        self.assertTrue(np.allclose(np.asarray(objs["box_blue"]["pose"][:3], float),
                                    np.asarray(truth.pose[:3], float), atol=1e-3))
        print(f"[probe] oracle 模式: box_blue pose={objs['box_blue']['pose'][:3]}（带真值 ✅）")

    def test_mockvlm_default_path_unaffected(self):
        app = _make_app(perception="auto")     # MockVLM -> oracle
        self.assertEqual(app.perception_mode, "oracle")
        self.assertIsInstance(app.vlm, MockVLM)
        results = app.run_task("把蓝色方块放到桌子上")
        print(f"[probe] MockVLM 默认路径: {[(r.skill, r.ok) for r in results]}")
        self.assertTrue(all(r.ok for r in results))


class TestCalibrationAccuracy(unittest.TestCase):
    def test_reprojection_roundtrip_two_cameras(self):
        env = MuJoCoArmEnv()
        pts = [[0.20, -0.10, TABLE_HEIGHT], [0.50, -0.10, TABLE_HEIGHT],
               [0.20, 0.10, TABLE_HEIGHT], [0.50, 0.10, TABLE_HEIGHT],
               [0.35, 0.00, TABLE_HEIGHT], [0.55, 0.20, TABLE_HEIGHT],
               [0.25, 0.25, TABLE_HEIGHT]]
        self.assertGreaterEqual(len(pts), 5)
        for cam in ("top", "overview"):
            with self.subTest(camera=cam):
                s = reprojection_summary(env, pts, camera=cam, z=TABLE_HEIGHT)
                print(f"[probe] {cam}: n={s['n']} failed={s['n_failed']} "
                      f"mean={s['mean_m'] * 1000:.6f}mm max={s['max_m'] * 1000:.6f}mm")
                self.assertEqual(s["n_failed"], 0)
                self.assertLess(s["max_m"], 0.005, "往返误差应远小于 5mm")


class _ScriptedVLM:
    """按脚本返回 SkillCall，并记录每一步 messages。"""

    def __init__(self, calls):
        self.calls = list(calls)
        self.batches: list = []

    def decide(self, messages):
        self.batches.append(messages)
        if self.calls:
            return self.calls.pop(0)
        return SkillCall("stop", {}, reasoning="scripted done")


class TestPixelPerceptionErrors(unittest.TestCase):
    def test_bad_pixel_yields_retryable_perception_error(self):
        app = _make_app(perception="vlm")
        vlm = _ScriptedVLM([SkillCall("pick", {"target_pixel": [9999.0, 9999.0],
                                                "pixel_camera": "top"})])
        loop = AgentLoop(app.env, executor=app.skills, vlm=vlm, cameras=(),
                         max_steps=3, log=None, pixel_resolver=lambda u, v, cam: None,
                         include_poses=False)
        results = loop.run("把蓝色方块放到桌子上")
        self.assertGreaterEqual(len(results), 2, "出错后必须继续下一步（可重试），不是崩掉")
        err = results[0]
        print(f"\n[probe] 坏像素: {[(r.skill, r.ok, r.error_code) for r in results]}")
        self.assertFalse(err.ok)
        self.assertEqual(err.error_code, "perception_error")
        self.assertEqual(list(err.metrics.get("target_pixel", [])), [9999.0, 9999.0],
                         "原始像素必须写进 metrics")
        self.assertIn("9999", err.message)
        # 第二步的 messages 必须能看到该错误与像素，供 LLM 纠错
        step2 = _messages_text(vlm.batches[1])
        self.assertIn("perception_error", step2)
        self.assertIn("9999", step2)

    def test_resolver_exception_is_caught(self):
        app = _make_app(perception="vlm")

        def boom(u, v, cam):
            raise RuntimeError("calibration exploded")

        vlm = _ScriptedVLM([SkillCall("pick", {"target_pixel": [10.0, 20.0]})])
        loop = AgentLoop(app.env, executor=app.skills, vlm=vlm, cameras=(),
                         max_steps=3, log=None, pixel_resolver=boom, include_poses=False)
        results = loop.run("把蓝色方块放到桌子上")
        self.assertEqual(results[0].error_code, "perception_error")
        self.assertIn("calibration exploded", results[0].message,
                      "异常原因应出现在可重试错误里")
        print(f"[probe] 解析器异常: {[(r.skill, r.ok, r.error_code) for r in results]}")

    def test_out_of_image_and_behind_camera_do_not_crash(self):
        env = MuJoCoArmEnv()
        for cam in ("top", "overview"):
            c = CameraModel(env, cam)
            for (u, v) in ((-10000.0, -10000.0), (10000.0, 10000.0), (-1.0, 99999.0)):
                with self.subTest(camera=cam, pixel=(u, v)):
                    ray = c.ray(u, v)                    # 任意像素都不许抛异常
                    self.assertEqual(np.asarray(ray[0]).shape, (3,))
                    self.assertAlmostEqual(float(np.linalg.norm(ray[1])), 1.0, places=6)
                    _ = pixel_to_table(env, u, v, camera=cam)   # 允许返回 None 或图外点
            # 相机后方的世界点 project 必须返回 None（不崩）
            behind = c.cam_pos + c.cam_mat[:, 2] * 1.0    # 相机局部 +z = 视线反方向 -> 后方
            self.assertIsNone(c.project(behind), f"{cam}: 相机后方的点应 project 为 None")
        # NaN 像素：§4.1 契约只承诺「负数/超大值」不抛；NaN 由 ray() raise ValueError。
        # 管线层面不会崩：interactive._pixel_resolver 用 try/except 兜住并返回 None
        # -> AgentLoop 记成可重试 perception_error（见 test_resolver_exception_is_caught）。
        with self.assertRaises(ValueError):
            pixel_to_table(env, float("nan"), 0.0, camera="top")
        print("[probe] 越界/相机后方像素不崩溃；NaN 在 raw API 按契约 raise ✅")


if __name__ == "__main__":
    unittest.main(verbosity=2)
