"""sim/test_vision.py —— sim/vision.py 回归测试（标准库 unittest）。

运行：
    cd /home/tc63/mujoco/vlm_admittance && /home/tc63/mujoco/.venv/bin/python -m unittest sim.test_vision -v

除最后的渲染一致性用例需要 GL（不可用时自动 skip）外，其余用例不需要 MUJOCO_GL/DISPLAY。
"""
from __future__ import annotations

import unittest

import mujoco
import numpy as np

from core.config import RENDER_HEIGHT, RENDER_WIDTH, TABLE_HEIGHT

from sim.env import MuJoCoArmEnv
from sim.vision import (
    CameraModel,
    camera_model,
    pixel_to_table,
    project_point,
    reprojection_errors,
    reprojection_summary,
)

TABLE_XY = [(0.35, -0.14), (0.35, 0.14), (0.45, -0.05), (0.40, 0.02), (0.40, 0.30)]


class VisionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MuJoCoArmEnv()
        cls.env.reset(seed=0)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(seed=0)

    # ------------------------------------------------------------------ #
    def test_camera_attributes_and_orthonormal_matrix(self):
        for name in ("top", "overview", "wrist"):
            cam = camera_model(self.env, name)
            self.assertEqual(cam.name, name)
            self.assertEqual((cam.width, cam.height), tuple(self.env.render_size))
            self.assertEqual((cam.width, cam.height), (RENDER_WIDTH, RENDER_HEIGHT))
            self.assertTrue(0.0 < cam.fovy < 180.0)
            self.assertEqual(cam.cam_pos.shape, (3,))
            self.assertEqual(cam.cam_mat.shape, (3, 3))
            rot = cam.cam_mat
            self.assertTrue(np.allclose(rot.T @ rot, np.eye(3), atol=1e-12))
            self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, places=12)

    def test_top_camera_axes_convention(self):
        cam = camera_model(self.env, "top")
        # 列 = x右 / y上 / z后；top 相机朝下，所以 z后 = 世界 +z
        self.assertTrue(np.allclose(cam.cam_mat[:, 0], [-1.0, 0.0, 0.0], atol=1e-12))
        self.assertTrue(np.allclose(cam.cam_mat[:, 1], [0.0, -1.0, 0.0], atol=1e-12))
        self.assertTrue(np.allclose(cam.cam_mat[:, 2], [0.0, 0.0, 1.0], atol=1e-12))
        forward = cam.cam_mat @ np.array([0.0, 0.0, -1.0])       # 视线方向
        self.assertLess(float(forward[2]), -0.99)

    def test_roundtrip_errors_below_1mm(self):
        points = np.array([[x, y, TABLE_HEIGHT] for x, y in TABLE_XY])
        for name in ("top", "overview"):
            errors = reprojection_errors(self.env, points, camera=name, z=TABLE_HEIGHT)
            self.assertEqual(errors.shape, (len(TABLE_XY),))
            self.assertTrue(np.all(np.isfinite(errors)), "%s 有投影失败的点" % name)
            self.assertLess(float(errors.max()), 1e-3, "%s 往返误差超 1mm" % name)

    def test_backprojected_z_matches_plane(self):
        cam = camera_model(self.env, "top")
        for x, y in TABLE_XY:
            uv = cam.project(np.array([x, y, TABLE_HEIGHT]))
            self.assertIsNotNone(uv)
            back = cam.pixel_to_plane(uv[0], uv[1], TABLE_HEIGHT)
            self.assertIsNotNone(back)
            self.assertLess(abs(float(back[2]) - TABLE_HEIGHT), 1e-6)
            self.assertTrue(np.allclose(back[:2], [x, y], atol=1e-9))

    def test_default_plane_is_table_height(self):
        cam = camera_model(self.env, "top")
        uv = cam.project(np.array([0.40, 0.02, TABLE_HEIGHT]))
        back = pixel_to_table(self.env, uv[0], uv[1], "top")
        self.assertLess(abs(float(back[2]) - TABLE_HEIGHT), 1e-6)

    # ------------------------------------------------------------------ #
    def test_behind_camera_returns_none(self):
        cam = camera_model(self.env, "top")
        behind = np.array([cam.cam_pos[0], cam.cam_pos[1], cam.cam_pos[2] + 0.3])
        self.assertIsNone(cam.project(behind))
        # 平面在相机上方 -> 交点 t<=0 -> None
        self.assertIsNone(cam.pixel_to_plane(cam.width / 2, cam.height / 2, cam.cam_pos[2] + 0.3))

    def test_out_of_image_behaviour(self):
        cam = camera_model(self.env, "top")
        self.assertIsNone(cam.project(np.array([5.0, 5.0, TABLE_HEIGHT])))
        # 越界像素：ray 与 pixel_to_plane 不崩溃，且落在给定平面上
        origin, direction = cam.ray(-50.0, 1e6)
        self.assertEqual(origin.shape, (3,))
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=12)
        back = cam.pixel_to_plane(5000.0, 5000.0, TABLE_HEIGHT)
        self.assertIsNotNone(back)
        self.assertLess(abs(float(back[2]) - TABLE_HEIGHT), 1e-6)
        self.assertGreater(float(np.linalg.norm(back - cam.cam_pos)), 1.0)   # 落在远处图像外

    def test_non_finite_input_does_not_crash(self):
        cam = camera_model(self.env, "top")
        self.assertIsNone(cam.project(np.array([np.nan, 0.0, 0.4])))
        self.assertIsNone(cam.project(np.zeros(2)))
        self.assertIsNone(cam.project([np.inf, 0.0, 0.4]))

    def test_ray_geometry_is_consistent(self):
        cam = camera_model(self.env, "top")
        origin, direction = cam.ray(300.0, 200.0)
        self.assertTrue(np.allclose(origin, cam.cam_pos))
        back = cam.pixel_to_plane(300.0, 200.0, TABLE_HEIGHT)
        delta = back - origin
        self.assertLess(float(np.linalg.norm(np.cross(delta, direction))), 1e-9)  # 共线
        self.assertGreater(float(np.dot(delta, direction)), 0.0)                  # 在视线正方向

    # ------------------------------------------------------------------ #
    def test_pose_comes_from_mj_forward_not_hardcoded(self):
        priv = mujoco.MjData(self.env.model)
        priv.qpos[:] = self.env.data.qpos
        mujoco.mj_forward(self.env.model, priv)
        for name in ("top", "overview", "wrist"):
            cam = camera_model(self.env, name)
            cid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            self.assertTrue(np.allclose(cam.cam_pos, priv.cam_xpos[cid], atol=1e-12))
            self.assertTrue(np.allclose(cam.cam_mat, priv.cam_xmat[cid].reshape(3, 3), atol=1e-12))

    def test_wrist_camera_follows_arm_motion(self):
        cam = camera_model(self.env, "wrist")
        before = cam.cam_pos.copy()
        self.env.set_arm_joint_targets(np.array([-1.5708, -1.30, 1.80, -1.5708, -1.30, 0.4]))
        self.env.step(300)
        cam.project(np.array([0.4, 0.02, TABLE_HEIGHT + 0.3]))     # 触发 _sync
        self.assertGreater(float(np.linalg.norm(cam.cam_pos - before)), 1e-3)
        # 刷新后仍与独立 mj_forward 一致
        priv = mujoco.MjData(self.env.model)
        priv.qpos[:] = self.env.data.qpos
        mujoco.mj_forward(self.env.model, priv)
        cid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        self.assertTrue(np.allclose(cam.cam_pos, priv.cam_xpos[cid], atol=1e-12))

    def test_unknown_camera_raises(self):
        with self.assertRaises(ValueError):
            camera_model(self.env, "no_such_camera")

    # ------------------------------------------------------------------ #
    def test_reprojection_helpers(self):
        points = np.array([[x, y, TABLE_HEIGHT] for x, y in TABLE_XY])
        errors = reprojection_errors(self.env, points, camera="top", z=TABLE_HEIGHT)
        self.assertTrue(np.all(errors < 1e-9))
        summary = reprojection_summary(self.env, points, camera="top", z=TABLE_HEIGHT)
        self.assertEqual(summary["n"], len(TABLE_XY))
        self.assertEqual(summary["n_failed"], 0)
        self.assertLess(summary["max_m"], 1e-3)
        # 图像外的点记为 nan 并计入 n_failed
        bad = np.array([[5.0, 5.0, TABLE_HEIGHT]])
        summary_bad = reprojection_summary(self.env, bad, camera="top", z=TABLE_HEIGHT)
        self.assertEqual(summary_bad["n_failed"], 1)
        self.assertTrue(np.isnan(summary_bad["errors_m"][0]))
        with self.assertRaises(ValueError):
            reprojection_errors(self.env, np.zeros((3, 2)))

    def test_resolution_matches_renderer_output(self):
        try:
            rgb = self.env.render("top")
        except Exception as exc:                      # 无 GL 环境自动跳过
            self.skipTest("渲染不可用：%s" % exc)
        self.assertEqual(rgb.shape, (self.env.render_size[1], self.env.render_size[0], 3))
        cam = camera_model(self.env, "top")
        self.assertEqual((cam.width, cam.height), (rgb.shape[1], rgb.shape[0]))
        # 物体中心投影处的像素颜色应与其颜色一致（红：R 主导）
        uv = cam.project(self.env.object_pose("box_red")[:3])
        self.assertIsNotNone(uv)
        r, g, b = (int(v) for v in rgb[int(round(uv[1])), int(round(uv[0]))])
        self.assertGreater(r, g + 20)
        self.assertGreater(r, b + 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
