"""sim/test_overlay.py —— overlay 绘制工具的回归测试（标准库 unittest）。

运行：
    cd /home/tc63/mujoco/vlm_admittance && /home/tc63/mujoco/.venv/bin/python -m unittest sim.test_overlay -v

本模块只构造 MjvScene 并直接读写 geom，不建 GL 上下文、不 import mujoco.viewer，
因此**不需要 MUJOCO_GL / DISPLAY**，可在完全无显示的环境跑。
"""
from __future__ import annotations

import unittest

import mujoco
import numpy as np

from sim import overlay

SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)


def _mat3(geom) -> np.ndarray:
    """读回 geom 的旋转矩阵（本机绑定里 g.mat 是 (3,3) ndarray）。"""
    return np.asarray(geom.mat, dtype=float).reshape(3, 3)


class OverlayTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 极简空模型：MjvScene 只需要一个 model 来分配缓冲，无需真实场景/GL
        cls.model = mujoco.MjModel.from_xml_string("<mujoco/>")
        cls.scn = mujoco.MjvScene(cls.model, maxgeom=64)

    def setUp(self):
        overlay.clear(self.scn)

    # ------------------------------------------------------------------ #
    def test_clear_resets_ngeom(self):
        self.scn.ngeom = 5
        overlay.clear(self.scn)
        self.assertEqual(self.scn.ngeom, 0)

    def test_draw_target_ball_only(self):
        pos = np.array([0.10, -0.20, 0.55])
        n = overlay.draw_target(self.scn, pos)
        self.assertEqual(n, 1)
        self.assertEqual(self.scn.ngeom, 1)
        g = self.scn.geoms[0]
        self.assertEqual(int(g.type), SPHERE)
        self.assertTrue(np.allclose(np.asarray(g.pos), pos))
        self.assertAlmostEqual(float(g.size[0]), overlay.DEFAULT_RADIUS, places=6)
        self.assertEqual(g.label, "target")

    def test_draw_target_with_axes_orientation(self):
        pos = np.array([0.40, 0.02, 0.62])
        # 绕 z 轴 +90 度：局部 x->世界 y, 局部 y->世界 -x, 局部 z->世界 z
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), np.pi / 2)
        n = overlay.draw_target(self.scn, pos, quat)
        self.assertEqual(n, 4)                       # 球 + 3 轴
        self.assertEqual(self.scn.ngeom, 4)
        self.assertEqual([int(self.scn.geoms[i].type) for i in range(4)],
                         [SPHERE, CAPSULE, CAPSULE, CAPSULE])
        expected_dirs = [np.array([0.0, 1.0, 0.0]),
                         np.array([-1.0, 0.0, 0.0]),
                         np.array([0.0, 0.0, 1.0])]
        axes_len = overlay.DEFAULT_AXES_LEN
        for k, direction in enumerate(expected_dirs):
            g = self.scn.geoms[1 + k]
            # 中心应在 pos + dir * axes_len / 2
            self.assertTrue(np.allclose(np.asarray(g.pos), pos + direction * axes_len * 0.5,
                                        atol=1e-9))
            # mat 第 3 列 = geom 局部 +z 在世界系的方向
            self.assertTrue(np.allclose(_mat3(g)[:, 2], direction, atol=1e-9))
            # mjv_initGeom 会把 capsule 传入的 (radius, half_length, 0)
            # 重排成 (radius, radius, half_length)（本机 3.12 实测），所以按此断言
            size = np.asarray(g.size, dtype=float)
            self.assertAlmostEqual(
                float(size[0]), overlay.DEFAULT_RADIUS * overlay.AXES_RADIUS_SCALE, places=6)
            self.assertAlmostEqual(float(size[2]), axes_len * 0.5, places=6)
            self.assertEqual(g.label, "axis_%s" % "xyz"[k])

    def test_draw_target_with_reference_point(self):
        pos = np.array([0.40, 0.02, 0.62])
        ref = np.array([0.428, 0.02, 0.62])
        n = overlay.draw_target(self.scn, pos, ref_pos=ref)
        self.assertEqual(n, 3)                       # 目标球 + 参考球 + 连线
        self.assertEqual([int(self.scn.geoms[i].type) for i in range(3)],
                         [SPHERE, SPHERE, CAPSULE])
        self.assertEqual(self.scn.geoms[1].label, "ref(admittance)")
        self.assertTrue(np.allclose(np.asarray(self.scn.geoms[1].pos), ref))
        link = self.scn.geoms[2]
        self.assertEqual(link.label, "ref_link")
        self.assertTrue(np.allclose(np.asarray(link.pos), 0.5 * (pos + ref), atol=1e-9))
        self.assertTrue(np.allclose(_mat3(link)[:, 2], (pos - ref) / np.linalg.norm(pos - ref),
                                    atol=1e-9))

    def test_ref_skipped_when_same_or_disabled(self):
        pos = np.array([0.40, 0.02, 0.62])
        self.assertEqual(overlay.draw_target(self.scn, pos, ref_pos=pos.copy()), 1)
        overlay.clear(self.scn)
        self.assertEqual(overlay.draw_target(self.scn, pos, ref_pos=pos + 0.05,
                                             show_ref=False), 1)
        overlay.clear(self.scn)
        self.assertEqual(overlay.draw_target(self.scn, pos, ref_pos=None), 1)

    def test_show_label_false_writes_no_label(self):
        n = overlay.draw_target(self.scn, np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]),
                                show_label=False)
        self.assertEqual(n, 4)
        for i in range(4):
            self.assertEqual(self.scn.geoms[i].label, "")

    def test_custom_radius_and_rgba(self):
        rgba = (0.7, 0.1, 0.9, 0.5)
        overlay.draw_target(self.scn, np.zeros(3), radius=0.05, rgba=rgba)
        g = self.scn.geoms[0]
        self.assertAlmostEqual(float(g.size[0]), 0.05, places=6)
        self.assertTrue(np.allclose(np.asarray(g.rgba), rgba, atol=1e-6))

    def test_appends_after_existing_geoms(self):
        overlay.draw_target(self.scn, np.array([0.0, 0.0, 0.1]))
        n = overlay.draw_target(self.scn, np.array([0.0, 0.0, 0.2]))
        self.assertEqual(n, 1)
        self.assertEqual(self.scn.ngeom, 2)
        self.assertTrue(np.allclose(np.asarray(self.scn.geoms[1].pos), [0.0, 0.0, 0.2]))

    # ------------------------------------------------------------------ #
    def test_maxgeom_truncation_does_not_overflow(self):
        scn = mujoco.MjvScene(self.model, maxgeom=2)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        n = overlay.draw_target(scn, np.zeros(3), quat, ref_pos=np.array([0.05, 0.0, 0.0]))
        self.assertEqual(n, 2)                       # 球 + 第一根轴，写满即停
        self.assertEqual(scn.ngeom, 2)
        self.assertLessEqual(scn.ngeom, scn.maxgeom)
        # 已经写满：后续绘制必须返回 0 且不越界
        self.assertEqual(overlay.draw_trail(scn, np.zeros((3, 3))), 0)
        self.assertEqual(overlay.draw_target(scn, np.zeros(3)), 0)
        self.assertEqual(scn.ngeom, 2)

    def test_maxgeom_zero_and_one(self):
        scn0 = mujoco.MjvScene(self.model, maxgeom=0)
        self.assertEqual(len(scn0.geoms), 0)
        self.assertEqual(overlay.draw_target(scn0, np.zeros(3), np.array([1.0, 0, 0, 0])), 0)
        self.assertEqual(overlay.draw_trail(scn0, np.zeros((2, 3))), 0)
        self.assertEqual(scn0.ngeom, 0)

        scn1 = mujoco.MjvScene(self.model, maxgeom=1)
        self.assertEqual(overlay.draw_target(scn1, np.zeros(3), np.array([1.0, 0, 0, 0])), 1)
        self.assertEqual(scn1.ngeom, 1)
        self.assertEqual(overlay.draw_trail(scn1, np.zeros((2, 3))), 0)

    # ------------------------------------------------------------------ #
    def test_draw_trail(self):
        pts = np.array([[0.0, 0.0, 0.0],
                        [0.05, 0.0, 0.0],
                        [0.05, 0.05, 0.0],
                        [0.05, 0.05, 0.05]])
        n = overlay.draw_trail(self.scn, pts)
        self.assertEqual(n, 7)                       # 4 个点球 + 3 段胶囊
        self.assertEqual(self.scn.ngeom, 7)
        types = [int(self.scn.geoms[i].type) for i in range(7)]
        self.assertEqual(types, [SPHERE, SPHERE, SPHERE, SPHERE,
                                 CAPSULE, CAPSULE, CAPSULE])
        self.assertTrue(np.allclose(np.asarray(self.scn.geoms[0].pos), pts[0]))
        seg = self.scn.geoms[4]
        self.assertTrue(np.allclose(np.asarray(seg.pos), 0.5 * (pts[0] + pts[1]), atol=1e-9))
        self.assertTrue(np.allclose(_mat3(seg)[:, 2], [1.0, 0.0, 0.0], atol=1e-9))
        self.assertEqual(seg.label, "trail")

    def test_draw_trail_degenerate_and_empty(self):
        self.assertEqual(overlay.draw_trail(self.scn, []), 0)
        self.assertEqual(overlay.draw_trail(self.scn, np.zeros((1, 3))), 1)
        overlay.clear(self.scn)
        # 相邻重复点不生成线段
        n = overlay.draw_trail(self.scn, np.array([[0.0, 0.0, 0.0],
                                                   [0.0, 0.0, 0.0],
                                                   [0.1, 0.0, 0.0]]))
        self.assertEqual(n, 4)                       # 3 个球 + 1 段有效胶囊

    # ------------------------------------------------------------------ #
    def test_label_is_truncated_to_99(self):
        g = self.scn.geoms[0]
        mujoco.mjv_initGeom(g, SPHERE, np.array([0.01, 0.0, 0.0]), np.zeros(3),
                            np.eye(3).reshape(9), np.array([1, 1, 1, 1], np.float32))
        overlay._decorate(g, "x" * 300)
        self.assertEqual(len(g.label), overlay.LABEL_MAX)
        self.assertEqual(overlay.LABEL_MAX, 99)

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            overlay.draw_target(self.scn, np.zeros(2))
        with self.assertRaises(ValueError):
            overlay.draw_target(self.scn, np.zeros(3), np.zeros(3))
        with self.assertRaises(ValueError):
            overlay.draw_trail(self.scn, np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            overlay.draw_target(self.scn, np.zeros(3), rgba=(1.0, 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
