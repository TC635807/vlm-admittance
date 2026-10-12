"""planning.smoke —— 规划模块自检（标准库 unittest，无 GUI 交互）。

运行
----
    cd /home/tc63/mujoco/vlm_admittance
    MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m planning.smoke

覆盖内容
--------
1. CollisionChecker：限位/NaN 拒绝、夹爪内部忽略对、以及"只统计含机器人 geom 的
   接触对"（场景里物体压桌子的永久接触不能导致所有位形非法）、
   **不污染 env.data**（前后 qpos/qvel 完全一致）。
2. RRTConnect / shortcut / densify：用一个解析式 2D 障碍做确定性单测
   （不依赖 MuJoCo），验证绕障、路径有效、种子可复现、densify 步长。
3. JointPlanner.plan_joint：从 HOME_QPOS 规划到"另一组可行关节角"，
   断言端点误差、每段 is_edge_valid、规划耗时 < 2 s。
4. JointPlanner.plan_pose：多 IK 种子到某个 TCP 位置，断言路径有效且末端到位。
5. 绕障用例：with_barrier=True 场景里找出一个"无墙时直连可行、有墙时直连
   恰好被 barrier_geom 挡死"的目标位形，断言 RRT 能找到合法绕障路径，
   且整条路径不出现任何 barrier 接触。

环境来源
--------
优先用真实 sim.MuJoCoArmEnv / control.Kinematics；两者未就绪时自动退化为
planning/stub_env.py 里的最小测试替身，并在输出里打印 WARNING。
sim-builder / ctrl-builder 就绪后重跑本文件即为真环境验收。
"""
from __future__ import annotations

import time
import unittest

import mujoco
import numpy as np

from core.config import ARM_JOINT_LIMITS, HOME_QPOS
from core.interfaces import PlannerProtocol

from planning.collision import CollisionChecker
from planning.planner import JointPlanner
from planning.rrt_connect import RRTConnect, densify, joint_path_length, shortcut


# --------------------------------------------------------------------------- #
# 环境 / 运动学：真实模块优先，未就绪则用 stub
# --------------------------------------------------------------------------- #
def make_env(with_barrier: bool):
    try:
        from sim.env import MuJoCoArmEnv

        try:
            env = MuJoCoArmEnv(objects=None, with_barrier=with_barrier, seed=0)
        except TypeError:  # 兼容只接受关键字的实现
            env = MuJoCoArmEnv(with_barrier=with_barrier)
        return env, "sim.env.MuJoCoArmEnv"
    except Exception as exc:  # noqa: BLE001 - 未就绪就退化到 stub
        from planning.stub_env import build_stub_env

        return build_stub_env(with_barrier=with_barrier), "planning.stub_env [sim 未就绪: %r]" % (exc,)


def make_kin(env):
    try:
        from control.kinematics import Kinematics

        return Kinematics(env), "control.kinematics.Kinematics"
    except Exception as exc:  # noqa: BLE001
        from planning.stub_env import StubKinematics

        return StubKinematics(env), "planning.stub_env.StubKinematics [control 未就绪: %r]" % (exc,)


def _set_home(env) -> np.ndarray:
    env.home()
    mujoco.mj_forward(env.model, env.data)
    return np.asarray(env.data.qpos, dtype=np.float64)[np.asarray(env.arm_qpos_idx, dtype=int)].copy()


def _has_barrier_contact(checker, q) -> bool:
    for c in checker.debug_contacts(q):
        name = str(c["geom1"]) + "|" + str(c["geom2"])
        if "barrier" in name and c["dist"] < 0:
            return True
    return False


def _direct_line_blocking(checker, q1, q2, steps: int = 25):
    """沿直线采样，找第一个非法点，返回 (是否被挡, 是否由 barrier 造成, 接触列表)。"""
    for k in range(1, int(steps)):
        s = k / steps
        q = q1 + s * (q2 - q1)
        if checker.is_state_valid(q):
            continue
        contacts = checker.debug_contacts(q)
        bad = [c for c in contacts if c["dist"] < 0 and not c["ignored"]]
        by_barrier = any(
            "barrier" in (str(c["geom1"]) + "|" + str(c["geom2"])) for c in bad
        )
        return True, by_barrier, contacts
    return False, False, []


def _find_feasible_goal(checker, kin, rng, q_start, budget: int = 2000, min_dist: float = 0.5):
    """撒随机关节角，挑一个无碰撞、离起点够远、TCP 在桌面之上的目标位形。"""
    for _ in range(int(budget)):
        qg = rng.uniform(checker.q_min, checker.q_max)
        if not checker.is_state_valid(qg):
            continue
        if float(np.linalg.norm(qg - q_start)) < min_dist:
            continue
        tcp = np.asarray(kin.fk(qg)[0], dtype=np.float64)
        if tcp[2] < 0.45:  # 别选贴着桌面的位形，避免"擦桌子"式退化
            continue
        return qg, tcp
    return None, None


def _find_barrier_goal(free_checker, bar_checker, kin, rng, q_start, budget: int = 4000):
    """找一个"无墙可直连、有墙被 barrier 挡死"的目标位形（绕障用例）。"""
    for _ in range(int(budget)):
        qg = rng.uniform(free_checker.q_min, free_checker.q_max)
        if not free_checker.is_state_valid(qg) or not bar_checker.is_state_valid(qg):
            continue
        tcp = np.asarray(kin.fk(qg)[0], dtype=np.float64)
        # 目标 TCP 要在墙的 +x 一侧、且低于墙顶（必须跨越/绕过障碍）
        if tcp[0] < 0.28 or not (0.42 <= tcp[2] <= 0.70):
            continue
        if bar_checker.is_edge_valid(q_start, qg, steps=25):
            continue
        blocked, by_barrier, contacts = _direct_line_blocking(bar_checker, q_start, qg, 25)
        if not (blocked and by_barrier):
            continue
        if not free_checker.is_edge_valid(q_start, qg, steps=25):
            continue
        return qg, tcp, contacts
    return None, None, []


# --------------------------------------------------------------------------- #
# 解析式 2D 障碍：给 RRT 做与 MuJoCo 无关的确定性单测
# --------------------------------------------------------------------------- #
class _SlabChecker:
    """2D 检查器：竖墙 |x| < 0.06 且 |y| < 0.5 视为障碍。"""

    def is_state_valid(self, q) -> bool:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        return not (abs(q[0]) < 0.06 and abs(q[1]) < 0.5)

    def is_edge_valid(self, q1, q2, steps: int = 12) -> bool:
        q1 = np.asarray(q1, dtype=np.float64).reshape(-1)
        q2 = np.asarray(q2, dtype=np.float64).reshape(-1)
        steps = max(1, int(steps))
        if not self.is_state_valid(q1) or not self.is_state_valid(q2):
            return False
        for k in range(1, steps + 1):
            if not self.is_state_valid(q1 + (q2 - q1) * (k / steps)):
                return False
        return True


# --------------------------------------------------------------------------- #
class PlanningSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_free, cls.env_free_src = make_env(False)
        cls.env_bar, cls.env_bar_src = make_env(True)
        cls.checker_free = CollisionChecker(cls.env_free)
        cls.checker_bar = CollisionChecker(cls.env_bar)
        cls.kin, cls.kin_src = make_kin(cls.env_free)
        cls.planner_free = JointPlanner(cls.env_free, cls.kin, cls.checker_free)
        cls.planner_bar = JointPlanner(cls.env_bar, cls.kin, cls.checker_bar)
        print("\n[planning.smoke] env(free) = %s" % cls.env_free_src)
        print("[planning.smoke] env(barrier) = %s" % cls.env_bar_src)
        print("[planning.smoke] kin = %s" % cls.kin_src)
        for tag, src in (("free", cls.env_free_src), ("barrier", cls.env_bar_src), ("kin", cls.kin_src)):
            if "stub" in src:
                print("[planning.smoke][WARNING] %s 使用测试替身，非最终验收：%s" % (tag, src))
        print("[planning.smoke] checker = %r" % (cls.checker_free,))

    @classmethod
    def tearDownClass(cls):
        for env in (getattr(cls, "env_free", None), getattr(cls, "env_bar", None)):
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    def test_01_collision_checker(self):
        chk = self.checker_free
        q_start = _set_home(self.env_free)
        self.assertTrue(chk.is_state_valid(q_start), "HOME_QPOS 应当无碰撞且在限位内")

        bad = q_start.copy()
        bad[0] += 100.0
        self.assertFalse(chk.is_state_valid(bad), "超限位应当非法")
        self.assertFalse(chk.is_state_valid(np.full(6, np.nan)), "NaN 应当非法")
        self.assertFalse(chk.is_state_valid(np.zeros(5)), "维度不对应当非法")

        # 场景里物体压在桌面上 —— 这些永久接触不能把所有位形判非法。
        # reset 瞬间物体与桌面恰好相切（dist=0，MuJoCo 不生成接触）；step 若干步
        # 让它们真正沉到桌面上，就会出现永久接触（真场景实测 ncon=21，下沉约 1.6e-4 m）。
        # 这里给 30 步（约 0.06 s）的余量，避免不同实现的接触生成时序差异。
        probe = mujoco.MjData(self.env_free.model)
        probe.qpos[:] = self.env_free.data.qpos[:]
        mujoco.mj_forward(self.env_free.model, probe)
        ncon_reset = int(probe.ncon)
        self.env_free.set_arm_joint_targets(q_start)
        self.env_free.step(30)
        probe.qpos[:] = self.env_free.data.qpos[:]
        mujoco.mj_forward(self.env_free.model, probe)
        ncon_settled = int(probe.ncon)
        print("[smoke] 静态接触数（reset 瞬间=%d，沉底后=%d），HOME 仍判合法=%s"
              % (ncon_reset, ncon_settled, chk.is_state_valid(q_start)))
        self.assertGreater(ncon_settled, 0, "物体落到桌面后应产生静态接触")
        self.assertTrue(chk.is_state_valid(q_start),
                        "物体与桌面的永久接触不应让机械臂位形非法")
        # 复位，避免影响后续用例
        self.env_free.reset(0)
        mujoco.mj_forward(self.env_free.model, self.env_free.data)
        q_start = _set_home(self.env_free)

        # 自环边
        self.assertEqual(chk.is_edge_valid(q_start, q_start), True)
        self.assertFalse(chk.is_edge_valid(q_start, bad), "含超限端的边必须非法")

        # 默认忽略对：夹爪内部成对忽略，机械臂连杆之间不忽略
        names = chk._geom_names
        bodies = chk._geom_body_names
        def first(pred):
            for g in range(len(names)):
                if pred(names[g], bodies[g]):
                    return g
            return None
        gr = first(lambda n, b: b == "g_right_pad")
        gl = first(lambda n, b: b == "g_left_pad")
        ua = first(lambda n, b: b == "upper_arm_link")
        fa = first(lambda n, b: b == "forearm_link")
        self.assertIsNotNone(gr); self.assertIsNotNone(gl)
        self.assertTrue(chk._ignore_mask[gr, gl], "夹爪左右 pad 的接触应默认忽略")
        self.assertFalse(chk._ignore_mask[ua, fa], "机械臂连杆之间的接触不能忽略")

        # 自定义忽略对：按 body 名也能命中
        chk2 = CollisionChecker(self.env_free, ignore_pairs={("forearm_link", "upper_arm_link")})
        self.assertTrue(chk2._ignore_mask[ua, fa], "ignore_pairs 应按 body 名生效")

        # 不污染 env.data
        snap_q = self.env_free.data.qpos.copy()
        snap_v = self.env_free.data.qvel.copy()
        for _ in range(30):
            chk.is_state_valid(q_start)
            chk.is_edge_valid(q_start, q_start + 0.05)
        self.assertTrue(np.array_equal(self.env_free.data.qpos, snap_q), "env.data.qpos 被污染")
        self.assertTrue(np.array_equal(self.env_free.data.qvel, snap_v), "env.data.qvel 被污染")

    # ------------------------------------------------------------------ #
    def test_02_rrt_connect_fake_checker(self):
        fake = _SlabChecker()
        start = np.array([-0.8, 0.0])
        goal = np.array([0.8, 0.0])
        self.assertFalse(fake.is_edge_valid(start, goal), "解析障碍：直连必须被挡")

        def build(seed):
            return RRTConnect(fake, [-1.0, -1.0], [1.0, 1.0],
                              step_size=0.1, max_iters=4000, seed=seed, goal_bias=0.1)

        path = build(1).plan(start, goal)
        self.assertIsNotNone(path, "RRT-Connect 应能绕过解析障碍")
        self.assertTrue(np.allclose(path[0], start))
        self.assertTrue(np.allclose(path[-1], goal))
        for a, b in zip(path[:-1], path[1:]):
            self.assertTrue(fake.is_edge_valid(a, b, steps=32), "RRT 输出边非法")
        self.assertGreater(len(path), 2, "绕障路径应当不止两个点")
        print("[smoke] 解析障碍：RRT waypoints=%d len=%.3f" % (len(path), joint_path_length(path)))

        # 种子可复现
        p2 = build(1).plan(start, goal)
        self.assertEqual(len(p2), len(path))
        for a, b in zip(p2, path):
            self.assertTrue(np.allclose(a, b), "同 seed 的 RRT 结果应完全一致")

        # shortcut 只减不增，且保持端点与有效性
        smooth = shortcut(path, fake, iters=200, seed=0)
        self.assertLessEqual(len(smooth), len(path))
        self.assertTrue(np.allclose(smooth[0], start))
        self.assertTrue(np.allclose(smooth[-1], goal))
        for a, b in zip(smooth[:-1], smooth[1:]):
            self.assertTrue(fake.is_edge_valid(a, b, steps=32))
        print("[smoke] shortcut：%d -> %d waypoints" % (len(path), len(smooth)))

        # densify 步长与端点
        dense = densify(smooth, max_step=0.05)
        self.assertTrue(np.allclose(dense[0], start))
        self.assertTrue(np.allclose(dense[-1], goal))
        for a, b in zip(dense[:-1], dense[1:]):
            self.assertLessEqual(float(np.linalg.norm(b - a)), 0.05 + 1e-9)
        print("[smoke] densify：%d -> %d waypoints (max_step=0.05)" % (len(smooth), len(dense)))

    # ------------------------------------------------------------------ #
    def test_03_plan_joint_home_to_feasible_goal(self):
        self.assertTrue(isinstance(self.planner_free, PlannerProtocol),
                        "JointPlanner 应满足 core.interfaces.PlannerProtocol")
        q_start = _set_home(self.env_free)
        rng = np.random.default_rng(3)
        q_goal, tcp = _find_feasible_goal(self.checker_free, self.kin, rng, q_start)
        self.assertIsNotNone(q_goal, "找不到可行目标位形")

        t0 = time.perf_counter()
        path = self.planner_free.plan_joint(q_goal)
        elapsed = time.perf_counter() - t0

        self.assertIsNotNone(path, "plan_joint 失败，stats=%s" % (self.planner_free.last_stats,))
        self.assertTrue(np.allclose(path[0], q_start, atol=1e-9), "路径起点应为当前关节角")
        self.assertTrue(np.allclose(path[-1], q_goal, atol=1e-9), "路径终点应为目标关节角")
        for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
            self.assertTrue(self.checker_free.is_edge_valid(a, b, steps=20),
                            "第 %d 段路径非法" % i)
        print("[smoke] plan_joint: %.3f s, waypoints=%d, len=%.3f rad, goal TCP=%s"
              % (elapsed, len(path), joint_path_length(path), np.round(tcp, 3).tolist()))
        self.assertLess(elapsed, 2.0, "规划耗时应 < 2 s，实测 %.3f s" % elapsed)

    # ------------------------------------------------------------------ #
    def test_04_plan_pose(self):
        q_start = _set_home(self.env_free)
        rng = np.random.default_rng(4)
        q_ref, tcp = _find_feasible_goal(self.checker_free, self.kin, rng, q_start)
        self.assertIsNotNone(q_ref, "找不到参考位形")
        target = np.asarray(self.kin.fk(q_ref)[0], dtype=np.float64)

        t0 = time.perf_counter()
        path = self.planner_free.plan_pose(target, quat=None, n_seeds=40)
        elapsed = time.perf_counter() - t0
        self.assertIsNotNone(path, "plan_pose 失败，stats=%s" % (self.planner_free.last_stats,))
        self.assertTrue(np.allclose(path[0], q_start, atol=1e-9))
        for a, b in zip(path[:-1], path[1:]):
            self.assertTrue(self.checker_free.is_edge_valid(a, b, steps=20))
        end_tcp = np.asarray(self.kin.fk(path[-1])[0], dtype=np.float64)
        err = float(np.linalg.norm(end_tcp - target))
        print("[smoke] plan_pose: %.3f s, waypoints=%d, |p_end - p_target|=%.4f m, stats=%s"
              % (elapsed, len(path), err, self.planner_free.last_stats))
        self.assertLess(err, self.planner_free.pos_tol + 1e-9, "plan_pose 末端未到位")
        self.assertLess(elapsed, 2.0, "plan_pose 耗时应 < 2 s")

    # ------------------------------------------------------------------ #
    def test_05_barrier_detour(self):
        q_start = _set_home(self.env_bar)
        _set_home(self.env_free)
        rng = np.random.default_rng(7)
        t0 = time.perf_counter()
        q_goal, tcp, contacts = _find_barrier_goal(
            self.checker_free, self.checker_bar, self.kin, rng, q_start
        )
        search = time.perf_counter() - t0
        self.assertIsNotNone(
            q_goal,
            "找不到被障碍挡住的绕障用例（场景/障碍几何可能变了），search=%.2fs" % search,
        )
        # 无墙时直连可行
        self.assertTrue(self.checker_free.is_edge_valid(q_start, q_goal, steps=25),
                        "无墙场景直连本应可行")
        # 有墙时直连被 barrier_geom 挡死
        blocked, by_barrier, contacts = _direct_line_blocking(
            self.checker_bar, q_start, q_goal, 25
        )
        self.assertTrue(blocked and by_barrier,
                        "直线关节插值应被 barrier 挡住，blocked=%s by_barrier=%s" % (blocked, by_barrier))
        print("[smoke] 绕障用例：goal TCP=%s，直连被挡（%s）"
              % (np.round(tcp, 3).tolist(),
                 [c["geom1"] + "-" + c["geom2"] for c in contacts if "barrier" in str(c["geom1"]) + str(c["geom2"])][:2]))

        t0 = time.perf_counter()
        path = self.planner_bar.plan_joint(q_goal)
        elapsed = time.perf_counter() - t0
        self.assertIsNotNone(path, "绕障规划失败，stats=%s" % (self.planner_bar.last_stats,))
        self.assertTrue(np.allclose(path[0], q_start, atol=1e-9))
        self.assertTrue(np.allclose(path[-1], q_goal, atol=1e-9))
        for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
            self.assertTrue(self.checker_bar.is_edge_valid(a, b, steps=20),
                            "绕障路径第 %d 段非法" % i)
        # 整条路径不得出现 barrier 接触
        n_barrier = sum(1 for q in path if self.checker_bar.debug_contacts(q) and
                        any(("barrier" in (str(c["geom1"]) + "|" + str(c["geom2"])) and c["dist"] < 0)
                            for c in self.checker_bar.debug_contacts(q)))
        direct = float(np.linalg.norm(q_goal - q_start))
        length = joint_path_length(path)
        print("[smoke] 绕障规划: %.3f s, waypoints=%d, len=%.3f vs 直连=%.3f (ratio=%.2f), "
              "barrier 接触样本数=%d"
              % (elapsed, len(path), length, direct, length / max(direct, 1e-9), n_barrier))
        self.assertEqual(n_barrier, 0, "绕障路径上仍有 barrier 接触")
        self.assertLess(elapsed, 2.0, "绕障规划耗时应 < 2 s，实测 %.3f s" % elapsed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
