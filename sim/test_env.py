"""sim/test_env.py —— sim 模块回归测试（标准库 unittest，无 GUI）。

运行：
    cd /home/tc63/mujoco/vlm_admittance && MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m unittest sim.test_env -v
"""
from __future__ import annotations

import unittest

from . import _glinit  # noqa: F401  必须在 import mujoco 之前

import mujoco
import numpy as np

from core.config import CAMERAS, HOME_QPOS, SIM_DT, TABLE_HEIGHT
from core.interfaces import SimEnv
from core.types import ARM_JOINT_NAMES, TCP_SITE

from .env import GRIPPER_CTRL_MAX, MuJoCoArmEnv
from .scene import DEFAULT_OBJECTS, OBJECT_IDS, normalize_objects, object_rest_z

#: 压桌位形（IK 求得，TCP=(0.40,0.02,0.40) 且夹爪朝下）——用于验证接触力方向
PRESS_QPOS = np.array([-3.4328, -1.4240, 2.4135, -2.5603, -1.5708, 4.0684])


class SimEnvTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MuJoCoArmEnv()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(seed=0)

    # ------------------------------------------------------------------ #
    def test_model_layout_matches_contract(self):
        m = self.env.model
        self.assertEqual((m.nq, m.nv, m.nu, m.neq), (14 + 7 * len(DEFAULT_OBJECTS),
                                                     14 + 6 * len(DEFAULT_OBJECTS), 7, 3))
        self.assertEqual(self.env.dt, SIM_DT)
        self.assertEqual(self.env.camera_names, tuple(CAMERAS))
        self.assertEqual(self.env.arm_qpos_idx.tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(self.env.arm_dof_idx.tolist(), [0, 1, 2, 3, 4, 5])
        for name in ARM_JOINT_NAMES:
            self.assertGreaterEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name), 0)
        self.assertGreaterEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE), 0)
        self.assertGreaterEqual(
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "g_fingers_actuator"), 0)

    def test_satisfies_sim_env_protocol(self):
        self.assertIsInstance(self.env, SimEnv)
        for attr in ("model", "data", "dt", "arm_qpos_idx", "arm_dof_idx", "camera_names"):
            self.assertTrue(hasattr(self.env, attr), attr)
        for meth in ("reset", "step", "get_state", "render", "render_depth",
                     "set_arm_joint_targets", "set_gripper", "get_gripper",
                     "contact_force_on_ee", "site_pose", "body_pose", "object_pose",
                     "set_object_pose", "list_objects", "home"):
            self.assertTrue(callable(getattr(self.env, meth, None)), meth)

    def test_reset_is_deterministic(self):
        a = [o.pose.copy() for o in self.env.reset(seed=7).objects]
        b = [o.pose.copy() for o in self.env.reset(seed=7).objects]
        c = [o.pose.copy() for o in self.env.reset(seed=8).objects]
        for pa, pb in zip(a, b):
            self.assertTrue(np.allclose(pa, pb))
        self.assertFalse(all(np.allclose(pa, pc) for pa, pc in zip(a, c)))
        self.assertTrue(np.allclose(self.env.data.qpos[self.env.arm_qpos_idx], HOME_QPOS))

    def test_reset_seed0_matches_model_defaults(self):
        # seed=0 必须与 model.qpos0 一致，否则 planning 私有 MjData 看到的布局会偏离实际
        self.env.reset(seed=0)
        for o in self.env._objects_spec:
            adr = self.env._obj_qposadr[o["id"]]
            for k in range(3):
                self.assertAlmostEqual(float(self.env.data.qpos[adr + k]), float(o["pos"][k]), places=6)
        for o in self.env._objects_spec:
            adr = self.env._obj_qposadr[o["id"]]
            for k in range(7):
                self.assertAlmostEqual(float(self.env.data.qpos[adr + k]),
                                       float(self.env.model.qpos0[adr + k]), places=6)

    def test_objects_rest_on_table_in_fixed_order(self):
        state = self.env.reset(seed=0)
        self.assertEqual([o.id for o in state.objects], list(OBJECT_IDS))
        for o in state.objects:
            obj = [d for d in DEFAULT_OBJECTS if d["id"] == o.id][0]
            self.assertAlmostEqual(o.pose[2], object_rest_z(normalize_objects([obj])[0]), places=4)
            self.assertAlmostEqual(o.pose[2], TABLE_HEIGHT + o.size[2], places=4)
        by_id = {o.id: o for o in state.objects}
        self.assertTrue(by_id["tray"].is_container)
        self.assertFalse(by_id["tray"].graspable)
        self.assertTrue(by_id["box_red"].graspable and by_id["can_yellow"].graspable)
        # 三个方块彼此间距 >= 0.10 m
        import itertools
        boxes = [by_id[i] for i in ("box_red", "box_blue", "box_green")]
        for a, b in itertools.combinations(boxes, 2):
            self.assertGreaterEqual(float(np.linalg.norm(a.pose[:2] - b.pose[:2])), 0.10)

    # ------------------------------------------------------------------ #
    def test_gripper_direction_and_mapping(self):
        # ctrl 映射：open_ratio=1 -> 0；open_ratio=0 -> 255（实测：0=全开、255=全闭）
        self.env.set_gripper(1.0)
        self.assertAlmostEqual(float(self.env.data.ctrl[self.env._grip_act_idx]), 0.0)
        self.env.set_gripper(0.0)
        self.assertAlmostEqual(float(self.env.data.ctrl[self.env._grip_act_idx]), GRIPPER_CTRL_MAX)
        # 物理上：全开时两指分离远大于全闭
        self.env.set_gripper(1.0)
        self.env.step(400)
        open_ratio = self.env.get_gripper()
        self.env.set_gripper(0.0)
        self.env.step(400)
        closed_ratio = self.env.get_gripper()
        self.assertGreater(open_ratio, 0.95)
        self.assertLess(closed_ratio, 0.15)
        # set_gripper 会夹紧到 [0, 1]
        self.env.set_gripper(2.0)
        self.assertEqual(self.env.gripper_target(), 1.0)

    def test_arm_joint_targets_and_home(self):
        q = HOME_QPOS + np.array([0.0, 0.1, -0.1, 0.0, 0.1, 0.0])
        self.env.set_arm_joint_targets(q)
        self.assertTrue(np.allclose(self.env.data.ctrl[self.env._arm_act_idx], q))
        self.env.step(200)
        self.assertFalse(np.allclose(self.env.data.qpos[self.env.arm_qpos_idx], HOME_QPOS))
        self.env.home()
        self.assertTrue(np.allclose(self.env.data.qpos[self.env.arm_qpos_idx], HOME_QPOS))
        with self.assertRaises(ValueError):
            self.env.set_arm_joint_targets(np.zeros(3))

    # ------------------------------------------------------------------ #
    def test_object_pose_roundtrip(self):
        pose = np.array([0.30, -0.05, 0.55, 1.0, 0.0, 0.0, 0.0])
        self.env.set_object_pose("box_red", pose)
        self.assertTrue(np.allclose(self.env.object_pose("box_red"), pose))
        found = [o for o in self.env.list_objects() if o.id == "box_red"][0]
        self.assertTrue(np.allclose(found.pose, pose))
        with self.assertRaises(KeyError):
            self.env.object_pose("nope")

    def test_site_and_body_pose(self):
        pos, quat = self.env.site_pose(TCP_SITE)
        self.assertEqual(pos.shape, (3,))
        self.assertEqual(quat.shape, (4,))
        self.assertAlmostEqual(float(np.linalg.norm(quat)), 1.0, places=6)
        bpos, bquat = self.env.body_pose("g_base")
        self.assertAlmostEqual(float(np.linalg.norm(bpos - pos)), 0.145, places=2)
        with self.assertRaises(ValueError):
            self.env.site_pose("not_a_site")

    # ------------------------------------------------------------------ #
    def test_contact_force_zero_at_rest(self):
        self.env.reset(seed=0)
        force, torque = self.env.contact_force_on_ee()
        self.assertTrue(np.allclose(force, 0.0))
        self.assertTrue(np.allclose(torque, 0.0))
        self.assertFalse(self.env.get_state().contact.in_contact)

    def test_contact_force_direction_when_pressing_table(self):
        # 直接把夹爪瞬移进桌面，mj_forward 后接触力应为"环境施加在末端上的力"，
        # 即竖直向上（+z），实测约 +1.1e3 N。
        self.env.data.qpos[self.env.arm_qpos_idx] = PRESS_QPOS
        mujoco.mj_forward(self.env.model, self.env.data)
        force, torque = self.env.contact_force_on_ee()
        self.assertGreater(float(np.linalg.norm(force)), 1.0)
        self.assertGreater(float(force[2]), 0.0)

    # ------------------------------------------------------------------ #
    def test_render_shapes_and_depth(self):
        rgb = self.env.render("overview")
        self.assertEqual(rgb.shape, (self.env.render_size[1], self.env.render_size[0], 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertGreater(float(rgb.std()), 1.0)
        depth = self.env.render_depth("wrist")
        self.assertEqual(depth.shape, (self.env.render_size[1], self.env.render_size[0]))
        self.assertEqual(depth.dtype, np.float32)
        self.assertGreater(float(np.median(depth)), 0.0)
        with self.assertRaises(ValueError):
            self.env.render("no_such_camera")

    def test_with_barrier_scene(self):
        env = MuJoCoArmEnv(with_barrier=True)
        try:
            bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "barrier")
            gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "barrier_geom")
            self.assertGreaterEqual(bid, 0)
            self.assertGreaterEqual(gid, 0)
            self.assertEqual(int(env.model.body_jntnum[bid]), 0)          # 固定，无关节
            pos, quat = env.body_pose("barrier")
            self.assertAlmostEqual(float(pos[2]), TABLE_HEIGHT + 0.12, places=5)
        finally:
            env.close()
        gid0 = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "barrier_geom")
        self.assertEqual(gid0, -1)                                        # 默认场景无挡板

    def test_object_layout_matches_lead_spec(self):
        by_id = {o["id"]: o for o in normalize_objects()}
        self.assertEqual(by_id["box_red"]["color"], "red")
        self.assertEqual(by_id["box_blue"]["color"], "blue")
        self.assertEqual(by_id["box_green"]["color"], "green")
        self.assertEqual(by_id["can_yellow"]["category"], "cylinder")
        self.assertEqual(by_id["can_yellow"]["size"], (0.015, 0.015, 0.035))
        self.assertEqual(by_id["tray"]["size"], (0.09, 0.07, 0.012))


if __name__ == "__main__":
    unittest.main(verbosity=2)
