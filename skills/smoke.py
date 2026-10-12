"""skills.smoke —— skills 模块自检（标准库 unittest，无 GUI 交互）。

覆盖
----
1. TestConstants：PLACE_TOUCH_COMPLIANCE 的 force_limit == 6.0（守卫式下降力阈值）、
   仅 z 轴使能；PICK_APPROACH_DRIFT_TOL == 0.008。
2. TestHardRecover：hard_recover 三级分支 'retreat' / 'home_joint' / 'scene_reset'
   （stub env/runner/planner 触发），并校验冻结签名；确认第 3 级会调用
   env.reset(seed=0) 且清空 holding。
3. TestPickDriftGuard：预抓取位移动把物体碰动 > 8 mm 时 pick 返回 object_disturbed，
   且不再下降/合爪；< 8 mm 时不触发该错误码。
4. TestGuardedPlace：放置下降因 force_limit(6 N) 中止时按"轻触到位"继续开爪并返回 ok。
5. TestTablePlace（task-24 要求 1/2）：surface='table' 显式 xy / 自选空位两条路径，
   以及判据对 dz 偏大、xy 偏大的拒绝；container_id 路径保持可用。
6. TestDirectionalRetreat（task-24 要求 3）：撤退方向 = +反力（环境施加在末端上的力，
   不是 -反力）、多方向去重、力上限 <= 80 N、时长为 2.5 s、脱身成功后 |F| <= 5 N。

运行：
    cd /home/tc63/mujoco/vlm_admittance
    MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m skills.smoke
断言失败时进程退出码非 0。
"""
from __future__ import annotations

import inspect
import unittest

import mujoco
import numpy as np

from core.config import (GRASP_COMPLIANCE, GRIPPER_GAP_CLOSED_MM, GRIPPER_GAP_OPEN_MM,
                         HOME_QPOS, STIFF_COMPLIANCE, TABLE_HEIGHT)
from core.types import MoveResult, SceneObject, SkillCall

from skills.library import (GRASP_CLOSE_PRIOR, GRASP_JAW_GAP_MIN_MM,
                            GRASP_JAW_OPEN_MARGIN_MM, HOLDING_UNNAMED,
                            approach_quat, approach_quat_dir,
                            PICK_APPROACH_DRIFT_TOL, PICK_DEFAULT_HALF, PICK_MAX_ATTEMPTS,
                            PLACE_CONTACT_LAST10_M, PLACE_CONTACT_START_M, PLACE_CONTACT_STEP_M,
                            PLACE_CONTACT_STOP_FORCE_N, PLACE_DESCENT_SOFT_COMPLIANCE,
                            CONTACT_VMAX, CONTACT_VMAX_TIMEOUT_MIN_S,
                            GRASP_SIZE_GAP_TOL_MM, PICK_CONTACT_BACKOFF_M,
                            PICK_CRUISE_VMAX_SLENDER,
                            PLACE_VMAX,
                            PLACE_DESCENT_SOFT_K_SCALE, PLACE_DESCENT_SOFT_WINDOW_M,
                            PICK_SLENDER_FORCE_LIMIT_N, PICK_VMAX_BOX, PICK_VMAX_SLENDER,
                            PLACE_MIN_GRASP_HEIGHT_M, PLACE_PICK_DESCEND_COMPLIANCE,
                            PLACE_APPROACH_VMAX, PLACE_PICK_STOP_FORCE_N,
                            PLACE_SLENDER_FLOOR_MIN_M,
                            PLACE_SIDE_SOFT_COMPLIANCE,
                            PLACE_SIDE_STOP_FORCE_N, VMAX_TIMEOUT_MIN_S,
                            PLACE_GRASPABLE_BAND, PLACE_SEAT_MAX_M,
                            RETREAT_FORCE_CAP_N,
                            PLACE_STACK_MARGIN_M, PLACE_STACK_SLIDE_MAX_M,
                            PLACE_TABLE_MARGIN_M, PLACE_TOUCH_COMPLIANCE,
                            PLACE_XY_TOL_M, RETREAT_DURATION_S, RETREAT_FORCE_HEADROOM,
                            RETREAT_MAX_FORCE_N, RETREAT_MIN_FORCE_N, SkillExecutor)


# --------------------------------------------------------------------------- #
# stub：只实现 SkillExecutor 用到的接口
# --------------------------------------------------------------------------- #
def _make_obj(obj_id, pose, size, category="box", color="red",
              graspable=True, is_container=False):
    return SceneObject(id=obj_id, category=category, color=color,
                       pose=np.asarray(pose, float), size=tuple(size),
                       graspable=graspable, is_container=is_container)


class _StubEnv:
    """最小 env：物体位姿可被测试脚本直接搬动，接触力可注入。"""

    def __init__(self, objects=(), force=0.0, events=None, gripper=1.0):
        self.dt = 0.002
        self._objects = {o.id: o for o in objects}
        self.force = float(force)
        self.reset_seeds = []
        self.gripper = float(gripper)
        self.gripper_cmds = []
        self.events = events if events is not None else []
        self.arm_targets = []
        self.step_steps = 0
        self.home_calls = 0
        #: TCP 位置（task-44 的几何接触判据用：把物体摆到 TCP 附近即模拟"接触"）
        self.tcp_pos = np.array([0.35, 0.0, 0.60])
        #: task-48：模拟指间有物——夹爪最多合到该开合度（0=空手可全合）
        self.jaw_block_ratio = 0.0
        #: task-59：接触式落放用——TCP 降到 contact_z 以下时报"接触力"（模拟触底）
        self.contact_z = None
        self.contact_force = 6.0
        #: task-59：True 时 runner.move_to_pose 会同步更新 tcp_pos（配合 contact_z 使用）
        self.track_moves = False
        #: 合爪"被物体挡住"时的接触力（None=不模拟）：夹爪合到 jaw_block_ratio 附近时报这个力
        self.close_force = None

    def list_objects(self):
        return list(self._objects.values())

    def object_pose(self, obj_id):
        return self._objects[str(obj_id)].pose.copy()

    def set_object_pose(self, obj_id, pose):
        self._objects[str(obj_id)].pose = np.asarray(pose, float).copy()

    def move_object(self, obj_id, delta):
        """测试辅助：把物体平移 delta（模拟接近过程碰飞）。"""
        o = self._objects[str(obj_id)]
        pose = o.pose.copy()
        pose[:3] += np.asarray(delta, float)
        o.pose = pose

    def site_pose(self, name):
        return np.asarray(self.tcp_pos, float).copy(), np.array([1.0, 0.0, 0.0, 0.0])

    def contact_force_on_ee(self):
        if (self.close_force is not None and float(self.jaw_block_ratio) > 0.0
                and float(self.gripper) <= float(self.jaw_block_ratio) + 0.02):
            # 夹爪合到"被物体挡住"的开合度附近 -> 有力（模拟真夹住）
            return np.array([0.0, 0.0, max(float(self.force), float(self.close_force))]), \
                np.zeros(3)
        if self.contact_z is not None and float(self.tcp_pos[2]) <= float(self.contact_z):
            return np.array([0.0, 0.0, max(float(self.force), float(self.contact_force))]), \
                np.zeros(3)
        return np.array([0.0, 0.0, self.force]), np.zeros(3)

    def set_gripper(self, ratio):
        #: jaw_block_ratio 模拟"指间有物"：夹爪合不到指令开合度以下（被物体挡住）
        self.gripper = max(float(ratio), float(self.jaw_block_ratio))
        self.gripper_cmds.append(float(ratio))
        self.events.append("gripper:%.2f" % float(ratio))

    def get_gripper(self):
        return float(self.gripper)

    def set_arm_joint_targets(self, q):
        """skills 的"停住"安全闸会调它；记录事件用于断言顺序。"""
        self.arm_targets.append(np.asarray(q, float).reshape(-1).copy())
        self.events.append("freeze")

    def step(self, n=1):
        self.step_steps += int(n)
        return None

    def reset(self, seed=0):
        self.reset_seeds.append(int(seed))
        self.force = 0.0
        self.events.append("reset")
        return None

    def home(self):
        self.home_calls += 1
        self.events.append("env.home")


class _StubRunner:
    """记录每次笛卡尔/关节调用的 stub runner。

    fail_when(compliance) 返回 True 时，move_to_pose 直接返回该错误码的失败。
    on_move(runner) 在成功后回调（可用于搬动物体）；on_path(runner) 在关节路径后回调。
    """

    def __init__(self, fail_when=None, error_code=None, max_force=0.0,
                 on_move=None, on_path=None, path_ok=True, events=None, env=None):
        self.calls = []
        self.joint_calls = []
        self.env = env
        self.events = events if events is not None else []
        self.fail_when = fail_when
        self.error_code = error_code
        self.max_force = float(max_force)
        self.on_move = on_move
        self.on_path = on_path
        self.path_ok = bool(path_ok)

    def move_to_pose(self, pos, quat=None, duration=None, compliance=None,
                     timeout=8.0, force_limit=None, v_max=None, stop_on_force=None):
        # §6.1：v_max / stop_on_force 是 ctr-builder 陆续加的形参；如实记录，供用例断言
        self.calls.append({"pos": np.asarray(pos, float).copy(), "compliance": compliance,
                           "duration": None if duration is None else float(duration),
                           "timeout": float(timeout), "force_limit": force_limit,
                           "v_max": None if v_max is None else float(v_max),
                           "stop_on_force": stop_on_force})
        if (getattr(self, "stop_on_force_z", None) is not None and stop_on_force
                and float(np.asarray(pos, float)[2]) <= float(self.stop_on_force_z)):
            # ctrl-builder 的 contact-stop：力到阈值 -> 冻结参考并**正常返回**
            stop_z = np.array([float(np.asarray(pos, float)[0]), float(np.asarray(pos, float)[1]),
                               float(self.stop_on_force_z)])
            if self.env is not None:
                self.env.tcp_pos = stop_z.copy()
            res = MoveResult(True, "stub contact stop", None,
                             max_force=(float(self.max_force) or 2.6),
                             final_pos=stop_z.copy())
            res.stop_reason = "contact"
            return res
        self.events.append("move_to_pose")
        if self.env is not None and getattr(self.env, "track_moves", False):
            self.env.tcp_pos = np.asarray(pos, float).copy()   # 模拟"手臂按命令到位"
        # 连续下降到接触面：运动到接触高度就停（模拟 v_max 参考逐步推进 + 看门狗中止），
        # 而不是一路到命令 z —— 这样 touchdown_z 才是真实接触高度。
        _cz = getattr(self.env, "contact_z", None) if self.env is not None else None
        if (_cz is not None and float(np.asarray(pos, float)[2]) <= float(_cz)
                and float(getattr(self.env, "contact_force", 0.0)) > 0.0):
            stop_z = np.array([float(np.asarray(pos, float)[0]),
                               float(np.asarray(pos, float)[1]), float(_cz)])
            if self.env is not None:
                self.env.tcp_pos = stop_z.copy()
            return MoveResult(False, "stub contact stop", "force_limit",
                              max_force=float(self.env.contact_force),
                              final_pos=stop_z.copy())
        if self.fail_when is not None and self.fail_when(compliance):
            # §6.1：_abort 的 final_pos 必须可用（上层据此取接触高度）
            return MoveResult(False, "stub force_limit", self.error_code,
                              max_force=self.max_force,
                              final_pos=np.asarray(pos, float).copy())
        if self.on_move is not None:
            self.on_move(self)
        return MoveResult(True, "stub ok", None, max_force=self.max_force,
                          final_pos=np.asarray(pos, float).copy())

    def follow_joint_path(self, path, max_step_rad=0.05, timeout=20.0, compliance=None):
        self.joint_calls.append({"path": path, "max_step_rad": float(max_step_rad),
                                 "timeout": float(timeout), "compliance": compliance})
        self.events.append("follow_joint_path")
        if self.on_path is not None:
            self.on_path(self)
        if self.path_ok:
            return MoveResult(True, "stub path", None)
        return MoveResult(False, "stub path fail", "invalid_path")


class _StubPlanner:
    def __init__(self, pose_path=None, joint_path=None, on_plan_joint=None):
        self.pose_path = pose_path
        self.joint_path = joint_path
        self.pose_calls = []
        self.joint_calls = []
        self.on_plan_joint = on_plan_joint

    def plan_pose(self, pos, quat=None, n_seeds=20):
        self.pose_calls.append(np.asarray(pos, float).copy())
        return self.pose_path

    def plan_joint(self, q_goal):
        self.joint_calls.append(np.asarray(q_goal, float).copy())
        if self.on_plan_joint is not None:
            self.on_plan_joint(self)
        return self.joint_path


class _StubKin:
    """最小 kin：ik 恒有解（供 _escape_contact 求"抬高位形"）。"""

    def ik(self, target_pos, target_quat=None, q0=None, max_iters=300, tol=1e-3):
        return np.zeros(6, dtype=float), True

    def fk(self, q=None):
        return np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0])

    def clamp(self, q):
        return np.asarray(q, dtype=float)


class _NoIkKin(_StubKin):
    """IK 无解：用来跳过"短抬升位形"回退，专门测 env.home() 安全门。"""

    def ik(self, target_pos, target_quat=None, q0=None, max_iters=300, tol=1e-3):
        return np.zeros(6, dtype=float), False


class _StubChecker:
    """最小 collision checker：只需要 _ignore_mask / _geom_body_names / _is_gripper_geom。"""

    class _Model:
        def __init__(self, ngeom):
            self.ngeom = int(ngeom)

    def __init__(self, ngeom=4, obj_body="box_blue", grip_ids=(1, 2)):
        self.model = self._Model(ngeom)
        self._geom_body_names = [
            obj_body if g == 0 else ("g_pad%d" % g if g in grip_ids else "arm_geom")
            for g in range(ngeom)
        ]
        self._is_gripper_geom = np.array([g in grip_ids for g in range(ngeom)], dtype=bool)
        self._ignore_mask = np.zeros((ngeom, ngeom), dtype=bool)


class _HeldAwarePlanner(_StubPlanner):
    """plan_joint 只有在「物体<->夹爪」接触被忽略时才给路径（模拟 N1 的根因）。"""

    def __init__(self, checker, obj_geom=0, grip_geom=1):
        super().__init__(joint_path=None)
        self.checker = checker
        self._obj_geom = int(obj_geom)
        self._grip_geom = int(grip_geom)

    def plan_joint(self, q_goal):
        self.joint_calls.append(np.asarray(q_goal, float).copy())
        if self.checker._ignore_mask[self._obj_geom, self._grip_geom]:
            return [HOME_QPOS.copy()]
        return None


def _executor(env, runner, planner, kin=None):
    return SkillExecutor(env, kin if kin is not None else _StubKin(), runner, planner,
                         log=lambda *a, **k: None)


# --------------------------------------------------------------------------- #
# 1) 常量
# --------------------------------------------------------------------------- #
class TestConstants(unittest.TestCase):
    def test_place_touch_compliance(self):
        self.assertAlmostEqual(float(PLACE_TOUCH_COMPLIANCE.force_limit), 6.0, places=6)
        self.assertAlmostEqual(float(PLACE_TOUCH_COMPLIANCE.k[2]), 120.0, places=6)
        self.assertAlmostEqual(float(PLACE_TOUCH_COMPLIANCE.d[2]), 20.0, places=6)
        axes_on = list(np.asarray(PLACE_TOUCH_COMPLIANCE.axes) != 0.0)
        self.assertEqual(axes_on, [False, False, True, False, False, False])

    def test_pick_drift_tol(self):
        self.assertAlmostEqual(float(PICK_APPROACH_DRIFT_TOL), 0.008, places=9)


# --------------------------------------------------------------------------- #
# 2) hard_recover 三级
# --------------------------------------------------------------------------- #
class TestHardRecover(unittest.TestCase):
    def test_signature_frozen(self):
        params = list(inspect.signature(SkillExecutor.hard_recover).parameters.values())
        self.assertEqual([p.name for p in params], ["self", "quat", "reset_if_force_above"])
        self.assertIsNone(params[1].default)
        self.assertAlmostEqual(float(params[2].default), 30.0, places=6)

    def test_level1_retreat(self):
        env = _StubEnv(force=0.0)
        runner = _StubRunner()
        planner = _StubPlanner(joint_path=[HOME_QPOS])
        ex = _executor(env, runner, planner)
        self.assertEqual(ex.hard_recover(), "retreat")
        self.assertEqual(len(planner.joint_calls), 0)   # 第 1 级成功则不动关节
        self.assertEqual(env.reset_seeds, [])

    def _stuck_env(self, force=300.0):
        """卡死场景：有个物体就贴在 TCP 上，几何判据会一直判有接触（第 1 级过不了）。"""
        blocker = _make_obj("blocker", [0.30, 0.0, 0.20], (0.02, 0.02, 0.02))
        env = _StubEnv([blocker], force=force)
        env.tcp_pos = np.array([0.30, 0.0, 0.20])
        return env, blocker

    def test_level2_home_joint(self):
        """第 1 级温和脱身清不掉（几何上一直接触），第 2 级回 HOME 后力降到 2 N。"""
        env, blocker = self._stuck_env(300.0)

        def on_path(_r, _env=env, _o=blocker):
            _env.move_object("blocker", [0.0, 0.0, 0.50])   # 第 2 级把物体带离
            _env.force = 2.0

        planner = _StubPlanner(joint_path=[HOME_QPOS.copy()])
        runner = _StubRunner(on_path=on_path)
        ex = _executor(env, runner, planner)
        self.assertEqual(ex.hard_recover(reset_if_force_above=30.0), "home_joint")
        last = runner.joint_calls[-1]
        self.assertAlmostEqual(last["max_step_rad"], 0.12, places=6)
        self.assertAlmostEqual(last["timeout"], 40.0, places=6)
        self.assertIs(last["compliance"], STIFF_COMPLIANCE)
        self.assertEqual(env.reset_seeds, [])

    def test_level3_no_plan(self):
        env, _b = self._stuck_env(300.0)
        runner = _StubRunner()
        planner = _StubPlanner(joint_path=None)         # plan_joint -> None
        ex = _executor(env, runner, planner)
        ex.holding = "held_dummy"                       # 不能设成 blocker——持物接触会被排除
        self.assertEqual(ex.hard_recover(), "scene_reset")
        self.assertEqual(env.reset_seeds, [0])
        self.assertIsNone(ex.holding)

    def test_level3_force_still_high(self):
        env, _b = self._stuck_env(300.0)                # 回 HOME 后力仍不降
        runner = _StubRunner()
        planner = _StubPlanner(joint_path=[HOME_QPOS.copy()])
        ex = _executor(env, runner, planner)
        ex.holding = "held_dummy"
        self.assertEqual(ex.hard_recover(reset_if_force_above=30.0), "scene_reset")
        self.assertEqual(env.reset_seeds, [0])
        self.assertIsNone(ex.holding)


# --------------------------------------------------------------------------- #
# 3) pick 接近漂移护栏
# --------------------------------------------------------------------------- #
class TestPickDriftGuard(unittest.TestCase):
    TARGET = "cyl_blue"

    def _setup(self, shift_m):
        obj = _make_obj(self.TARGET, [0.35, 0.0, 0.43], (0.015, 0.015, 0.05),
                        category="cylinder", color="blue")
        env = _StubEnv([obj])
        state = {"done": False}

        def on_move(runner):
            if not state["done"]:
                state["done"] = True
                env.move_object(self.TARGET, [shift_m, 0.0, 0.0])

        runner = _StubRunner(on_move=on_move)
        planner = _StubPlanner(pose_path=None)          # 退化到笛卡尔
        ex = _executor(env, runner, planner)
        return env, runner, planner, ex

    def test_drift_above_tol_aborts_before_descend(self):
        env, runner, planner, ex = self._setup(0.033)
        res = ex(SkillCall("pick", {"target_id": self.TARGET}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "object_disturbed")
        self.assertGreater(res.metrics["drift"], PICK_APPROACH_DRIFT_TOL)
        self.assertAlmostEqual(res.metrics["drift"], 0.033, places=3)
        # 只发生 1 次笛卡尔（接近）：没有下降（GRASP 档 kz=200）、没有合爪；
        # 力已为 0 时 _escape_contact 只 freeze，不再多走一次笛卡尔
        self.assertEqual(len(runner.calls), 1)
        self.assertFalse(any(c["compliance"] is not None
                             and abs(float(c["compliance"].k[2])
                                     - float(GRASP_COMPLIANCE.k[2])) < 1e-9
                             for c in runner.calls))
        self.assertEqual(env.gripper_cmds, [1.0])

    def test_drift_below_tol_keeps_going(self):
        env, runner, planner, ex = self._setup(0.005)
        res = ex(SkillCall("pick", {"target_id": self.TARGET}))
        self.assertNotEqual(res.error_code, "object_disturbed")
        # 确实执行了下降（GRASP 档 force_limit 出现）
        self.assertTrue(any(c["compliance"] is not None
                            and float(c["compliance"].force_limit)
                            == float(GRASP_COMPLIANCE.force_limit)
                            for c in runner.calls))


# --------------------------------------------------------------------------- #
# 4) 守卫式放置下降
# --------------------------------------------------------------------------- #
class TestGuardedPlace(unittest.TestCase):
    """task-59：容器内落放默认走"接触式下降"；接触后松爪，metrics 报 touchdown_z / 接触力。"""

    def _setup(self, with_contact=True):
        # 物体"已经落到托盘上表面"（0.45+0.02=0.47）——stub 没有物理，直接把终态摆好，
        # 这样 task-24 的四条判据（xy/底面/力/夹爪）都能通过。
        box = _make_obj("box_red", [0.35, 0.0, 0.47], (0.02, 0.02, 0.02))
        tray = _make_obj("tray", [0.35, 0.0, 0.44], (0.10, 0.08, 0.01),
                         category="tray", color="gray",
                         graspable=False, is_container=True)
        env = _StubEnv([box, tray])
        # 接触式下降：TCP 降到"目标高度"(support 0.45 + 半高 0.02 + 余量 0.01 = 0.48)时报接触
        env.contact_z = 0.48 if with_contact else None
        env.track_moves = True
        runner = _StubRunner(env=env)
        planner = _StubPlanner(pose_path=None)
        ex = _executor(env, runner, planner)
        ex.holding = "box_red"
        return env, runner, planner, ex

    def test_contact_descent_reports_touchdown(self):
        env, runner, planner, ex = self._setup()
        res = ex(SkillCall("place", {"container_id": "tray"}))
        self.assertTrue(res.ok, res.message)
        self.assertIsNone(res.error_code)
        self.assertTrue(res.metrics.get("touched"))
        self.assertTrue(res.metrics.get("descend_until_contact"))   # 容器内默认开启
        self.assertAlmostEqual(res.metrics["touchdown_z"], 0.48, delta=0.002)  # 步长 1.5mm
        self.assertGreaterEqual(res.metrics["contact_force_norm"], 2.5)
        # 每一步都用 6 N 守卫档（force_limit 未变），且是分步下降
        touch_calls = [c for c in runner.calls if c["compliance"] is PLACE_TOUCH_COMPLIANCE]
        self.assertGreater(touch_calls, [])
        self.assertTrue(all(abs(float(c["compliance"].force_limit) - 6.0) < 1e-9
                            for c in touch_calls))
        self.assertIsNone(ex.holding)              # N2: 释放后 holding 必须清空

    def test_place_descent_uses_vmax_0_01(self):
        """lead 裁决：叠放/容器内落放**整段下降都限速**（接近段 PLACE_APPROACH_VMAX、最后 10mm CONTACT_VMAX），
        看门狗 2.5N 与硬限不变。"""
        env, runner, planner, ex = self._setup()
        res = ex(SkillCall("place", {"container_id": "tray"}))
        self.assertTrue(res.ok, res.message)
        steps = [c for c in runner.calls if c["compliance"] is PLACE_TOUCH_COMPLIANCE
                 or c["compliance"] is PLACE_DESCENT_SOFT_COMPLIANCE]
        self.assertTrue(steps)
        slow = [c for c in steps if c["v_max"] is not None]
        self.assertTrue(slow, "落放下降没有限速")
        self.assertTrue(any(abs(float(c["v_max"]) - PLACE_APPROACH_VMAX) < 1e-9
                            for c in slow), [c["v_max"] for c in slow])
        self.assertTrue(any(abs(float(c["v_max"]) - CONTACT_VMAX) < 1e-9 for c in slow))
        # 最后 10mm 那步仍带 2.5N 看门狗（硬限不变）
        for c in slow:
            if c["compliance"] is PLACE_DESCENT_SOFT_COMPLIANCE:
                self.assertAlmostEqual(float(c["force_limit"]), PLACE_CONTACT_STOP_FORCE_N, places=9)

    def test_contact_step_is_force_watchdogged_and_reports_peak(self):
        """每一步下降都带 4 N 看门狗；接触后立刻停住，metrics 记 peak_contact_force。"""
        env, runner, planner, ex = self._setup()
        res = ex(SkillCall("place", {"container_id": "tray"}))
        self.assertTrue(res.ok, res.message)
        # 阶段2（限速那段）带 2.5N 看门狗；阶段1 是原速段（力限沿用旧档）
        slow = [c for c in runner.calls if c["compliance"] is PLACE_DESCENT_SOFT_COMPLIANCE]
        self.assertTrue(slow)
        self.assertTrue(all(abs(float(c["force_limit"]) - PLACE_CONTACT_STOP_FORCE_N) < 1e-9
                            for c in slow), [c["force_limit"] for c in slow])
        self.assertIn("peak_contact_force", res.metrics)
        self.assertGreaterEqual(res.metrics["peak_contact_force"],
                                res.metrics["contact_force_norm"])
        # 接触后立刻"停住"：freeze 事件里应包含接触之后追加的那次
        self.assertGreaterEqual(env.events.count("freeze"), 1)

    def test_no_contact_env_falls_through_to_normal_validation(self):
        """环境完全读不到接触力（桩/几何真值）时：到几何地板就停住并正常释放/校验，
        不再误判成 place_contact_miss（否则正常放置会被判死）。"""
        env, runner, planner, ex = self._setup(with_contact=False)   # 一直没有接触
        res = ex(SkillCall("place", {"container_id": "tray"}))
        self.assertTrue(res.ok, res.message)
        self.assertTrue(res.metrics.get("contact_miss"))              # 观测标记，不是错误
        self.assertEqual(env.reset_seeds, [])

    def test_contact_miss_when_floor_unreachable(self):
        """只有"几何地板落在 120mm 窗口之外"（目标异常深）才报可重试 place_contact_miss。"""
        import skills.library as lib
        env, runner, planner, ex = self._setup(with_contact=False)
        o1, o2 = lib.PLACE_CONTACT_START_M, lib.PLACE_CONTACT_MAX_M
        lib.PLACE_CONTACT_START_M = 0.05               # 起点只高 5cm
        lib.PLACE_CONTACT_MAX_M = 0.01                 # 行程上限 1cm -> 到不了阶段2 边界
        try:
            res = ex(SkillCall("place", {"container_id": "tray"}))
        finally:
            lib.PLACE_CONTACT_START_M, lib.PLACE_CONTACT_MAX_M = o1, o2
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "place_contact_miss")       # 可重试，不是 scene_reset
        # 用了上面 monkeypatch 的小行程上限（默认是 PLACE_CONTACT_MAX_M=120mm）
        self.assertGreaterEqual(res.metrics["descend_travel_mm"], 10.0)
        self.assertIn("dxy_target", res.metrics)                      # 早退路径 metrics 也要完整
        self.assertEqual(env.reset_seeds, [])

    def test_container_wall_hit_is_place_blocked(self):
        """容器内：远高于目标就接触 -> place_blocked + obstruction_id=容器 id。"""
        env, runner, planner, ex = self._setup()
        env.contact_z = 0.53                       # 目标上方 50 mm 就"撞到"（且在下降起点之下）
        res = ex(SkillCall("place", {"container_id": "tray"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "place_blocked")
        self.assertEqual(res.metrics["obstruction_id"], "tray")
        self.assertIn("换一个落点像素", res.message)

    def test_place_descent_uses_stop_on_force_contact(self):
        """ctrl-builder 的 stop_on_force：落放下降力到阈值 -> 冻结参考并**正常返回**（ok+contact），
        上层必须当"到位接触"处理（停住 -> 释放 -> 走原校验），而不是报错。"""
        env, runner, planner, ex = self._setup()
        runner.stop_on_force_z = 0.49                   # 目标上方 10mm 就"接触停住"
        res = ex(SkillCall("place", {"container_id": "tray"}))
        self.assertTrue(res.ok, res.message)
        self.assertTrue(res.metrics.get("touched"))
        self.assertIsNotNone(res.metrics.get("touchdown_z"))
        calls = [c for c in runner.calls if c["compliance"] is PLACE_TOUCH_COMPLIANCE
                 or c["compliance"] is PLACE_DESCENT_SOFT_COMPLIANCE]
        self.assertTrue(calls)
        self.assertTrue(all(c["stop_on_force"] is True for c in calls),
                        [c["stop_on_force"] for c in calls])

    def test_last_20mm_descent_uses_soft_profile_hard_limit_unchanged(self):
        """硬化：最后 20 mm 下降改用软档（k 缩 PLACE_DESCENT_SOFT_K_SCALE），但**硬限一个都不放宽**。"""
        env, runner, planner, ex = self._setup()
        res = ex(SkillCall("place", {"container_id": "tray"}))
        self.assertTrue(res.ok, res.message)
        soft = [c for c in runner.calls if c["compliance"] is PLACE_DESCENT_SOFT_COMPLIANCE]
        self.assertTrue(soft, "最后 20mm 没有走软档")
        self.assertGreater(res.metrics.get("soft_descent_steps", 0), 0)
        # 硬限不变：档位 force_limit 与 PLACE_TOUCH_COMPLIANCE 完全一致，且每步 2.5N 看门狗不变
        for c in soft:
            self.assertAlmostEqual(float(c["compliance"].force_limit),
                                   float(PLACE_TOUCH_COMPLIANCE.force_limit), places=9)
            self.assertAlmostEqual(float(c["force_limit"]),
                                   float(PLACE_CONTACT_STOP_FORCE_N), places=9)
        # 仍是 z_only，且 k 正好是按比例缩小的值
        self.assertEqual(float(soft[0]["compliance"].k[0]), 0.0)
        self.assertEqual(float(soft[0]["compliance"].k[1]), 0.0)
        self.assertAlmostEqual(float(soft[0]["compliance"].k[2]),
                               float(PLACE_TOUCH_COMPLIANCE.k[2]) * PLACE_DESCENT_SOFT_K_SCALE,
                               places=6)
        # 只有"目标上方 PLACE_DESCENT_SOFT_WINDOW_M 以内"才换软档
        tgt_z = float(res.metrics["support_z"]) + 0.02 + 0.010      # 容器: support + 半高 + 0.01
        self.assertTrue(all(float(c["pos"][2]) <= tgt_z + PLACE_DESCENT_SOFT_WINDOW_M + 1e-6
                            for c in soft),
                        [round(float(c["pos"][2]), 4) for c in soft])

    def test_descend_until_contact_can_be_disabled(self):
        """显式 descend_until_contact=False -> 回到原来的"守卫式下降"语义。"""
        env, runner, planner, ex = self._setup()
        res = ex(SkillCall("place", {"container_id": "tray",
                                     "descend_until_contact": False}))
        self.assertTrue(res.ok, res.message)
        self.assertFalse(res.metrics.get("descend_until_contact"))
        touch_calls = [c for c in runner.calls if c["compliance"] is PLACE_TOUCH_COMPLIANCE]
        self.assertGreaterEqual(len(touch_calls), 1)   # 两段式：巡航 + 最后 10mm
        self.assertNotIn("touchdown_z", res.metrics)

    def test_other_descent_error_still_fails(self):
        box = _make_obj("box_red", [0.35, 0.0, 0.44], (0.02, 0.02, 0.02))
        tray = _make_obj("tray", [0.35, 0.0, 0.44], (0.10, 0.08, 0.01),
                         category="tray", color="gray",
                         graspable=False, is_container=True)
        env = _StubEnv([box, tray])
        runner = _StubRunner(fail_when=lambda c: c is PLACE_TOUCH_COMPLIANCE,
                             error_code="timeout", max_force=0.0)
        ex = _executor(env, runner, _StubPlanner(pose_path=None))
        ex.holding = "box_red"
        res = ex(SkillCall("place", {"container_id": "tray"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "timeout")   # 步进硬错误沿用既有语义


# --------------------------------------------------------------------------- #
# 5) S1: _retreat 永不依赖单次笛卡尔移动（ik_fail -> 关节空间）
# --------------------------------------------------------------------------- #
class TestRetreatTask44(unittest.TestCase):
    """task-44：撤退不再是"多方向 × 力上限逐级抬高"的暴力阶梯，也不再用瞬时 |F| 判成功。"""

    def _blocked_env(self, force=50.0):
        """构造"物体就贴在 TCP 上"的卡死场景（几何接触判据会一直判有接触）。"""
        blocker = _make_obj("blocker", [0.30, 0.0, 0.20], (0.02, 0.02, 0.02))
        env = _StubEnv([blocker], force=force)
        env.tcp_pos = np.array([0.30, 0.0, 0.20])
        return env, blocker

    def test_retreat_force_limit_capped_at_45(self):
        """R1：任何撤退的 force_limit <= RETREAT_FORCE_CAP_N(45)，且只走一步。"""
        for force in (0.0, 20.0, 50.0, 300.0, 5000.0):
            env, _b = self._blocked_env(force)
            runner = _StubRunner()
            _executor(env, runner, _StubPlanner())._retreat()
            self.assertEqual(len(runner.calls), 1)          # 只有一步，没有 5 方向阶梯
            lim = float(runner.calls[0]["force_limit"])
            self.assertLessEqual(lim, RETREAT_FORCE_CAP_N)
            self.assertLess(lim, 60.0)                      # 日志里不得再出现 60+ N
            self.assertAlmostEqual(float(runner.calls[0]["duration"]),
                                   RETREAT_DURATION_S, places=6)

    def test_retreat_has_no_builtin_joint_fallback(self):
        env, _b = self._blocked_env(50.0)
        runner = _StubRunner()
        _executor(env, runner, _StubPlanner(joint_path=[HOME_QPOS.copy()]))._retreat()
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(len(runner.joint_calls), 0)        # 关节回退只在终态清场里出现

    def test_escaped_is_geometric_not_instant_force(self):
        """R4：松爪瞬间力掉到 0 但物体仍贴在 TCP 上 -> 必须判"未脱身"（假成功消失）。"""
        env, _b = self._blocked_env(0.0)                    # 力为 0，但几何上仍接触
        ex = _executor(env, _StubRunner(), _StubPlanner())
        self.assertFalse(ex._escaped({"blocker"}))
        env.move_object("blocker", [0.0, 0.0, 0.30])        # 真正离开
        self.assertTrue(ex._escaped({"blocker"}))

    def test_escape_contact_repeatedly_never_fakes_success(self):
        """R4/R6：同一场景重复 4 次，不得出现假脱身成功。"""
        env, _b = self._blocked_env(50.0)
        runner = _StubRunner(fail_when=lambda c: True, error_code="force_limit")
        ex = _executor(env, runner, _StubPlanner(joint_path=None))
        for i in range(4):
            self.assertFalse(ex._escape_contact(None, "blocker"),
                             "第 %d 次不应假报脱身成功" % (i + 1))
            self.assertEqual(ex.last_escape_level, "blocked")

    def test_escape_contact_clear_and_joint_lift(self):
        env = _StubEnv([])
        runner = _StubRunner()
        ex = _executor(env, runner, _StubPlanner())
        self.assertTrue(ex._escape_contact(None, None))
        self.assertEqual(ex.last_escape_level, "clear")
        self.assertEqual(len(runner.calls), 0)

        env2, _b2 = self._blocked_env(30.0)
        runner2 = _StubRunner(fail_when=lambda c: True, error_code="force_limit")
        runner2.on_path = lambda r: env2.move_object("blocker", [0.0, 0.0, 0.40])
        ex2 = _executor(env2, runner2, _StubPlanner(joint_path=[HOME_QPOS.copy()]))
        self.assertTrue(ex2._escape_contact(None, "blocker"))
        self.assertEqual(ex2.last_escape_level, "lifted")

    def test_ensure_clear_single_gentle_retreat_only(self):
        """R2：入口不干净只允许一次温和撤退；仍不干净立即 False（禁止升级阶梯）。"""
        env, _b = self._blocked_env(50.0)
        runner = _StubRunner()
        ex = _executor(env, runner, _StubPlanner(joint_path=[HOME_QPOS.copy()]))
        self.assertFalse(ex._ensure_clear(None, obj_id="blocker"))
        self.assertEqual(len(runner.calls), 1)              # 只有一次撤退
        self.assertEqual(len(runner.joint_calls), 0)

    def test_finish_failure_reports_obstruction_before_cleanup(self):
        """R5：受阻物必须在"还卡着"的时刻取；脱身清空之后依然要能报出来。"""
        env, _b = self._blocked_env(50.0)
        runner = _StubRunner()
        runner.on_path = lambda r: env.move_object("blocker", [0.0, 0.0, 0.40])
        ex = _executor(env, runner, _StubPlanner(joint_path=[HOME_QPOS.copy()]))
        extra, override, note = ex._finish_failure(None, "blocker")
        self.assertTrue(extra["escaped"])
        self.assertIsNone(override)
        self.assertEqual(extra["obstruction_id"], "blocker")   # 脱身后仍保留
        self.assertEqual(extra["obstruction_force"], 50.0)
        self.assertIn("blocker", note)

    def test_obstruction_reported(self):
        """R5：受阻物 id 被报出（读不到则 None）。"""
        env, _b = self._blocked_env(50.0)
        ex = _executor(env, _StubRunner(), _StubPlanner())
        obs, note = ex._obstruction()
        self.assertEqual(obs["obstruction_id"], "blocker")
        self.assertIn("blocker", note)
        self.assertIn("suggest_image", obs)
        env2 = _StubEnv([])
        ex2 = _executor(env2, _StubRunner(), _StubPlanner())
        obs2, note2 = ex2._obstruction()
        self.assertIsNone(obs2["obstruction_id"])
        self.assertEqual(note2, "")


# --------------------------------------------------------------------------- #
# 6) S3: home 后按夹爪重判 holding；S4: 瞬移前置条件
# --------------------------------------------------------------------------- #
class TestHomeSafety(unittest.TestCase):
    def test_s3_holding_cleared_when_gripper_open(self):
        env = _StubEnv(gripper=0.95)
        ex = _executor(env, _StubRunner(), _StubPlanner(joint_path=[HOME_QPOS.copy()]))
        ex.holding = "box_red"
        res = ex(SkillCall("home"))
        self.assertTrue(res.ok)
        self.assertIsNone(ex.holding)
        self.assertGreaterEqual(res.metrics["gripper"], 0.8)

    def test_s3_holding_kept_when_gripper_closed(self):
        env = _StubEnv(gripper=0.3)
        ex = _executor(env, _StubRunner(), _StubPlanner(joint_path=[HOME_QPOS.copy()]))
        ex.holding = "can_yellow"
        res = ex(SkillCall("home"))
        self.assertTrue(res.ok)
        self.assertEqual(ex.holding, "can_yellow")

    def test_s4_home_blocked_when_pressed_and_closed(self):
        events = []
        env = _StubEnv(force=200.0, events=events, gripper=0.2)
        runner = _StubRunner(fail_when=lambda c: True, error_code="ik_fail", events=events)
        planner = _StubPlanner(joint_path=None)
        ex = _executor(env, runner, planner)
        res = ex(SkillCall("home"))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "home_blocked")
        self.assertEqual(env.home_calls, 0)
        self.assertGreaterEqual(events.count("freeze"), 2)
        self.assertIn("move_to_pose", events)      # 尝试过 _retreat

    def test_s4_home_allowed_when_clear(self):
        env = _StubEnv(force=0.0, gripper=1.0)
        planner = _StubPlanner(joint_path=None)
        ex = _executor(env, _StubRunner(), planner)
        ex.holding = "box_red"
        res = ex(SkillCall("home"))
        self.assertTrue(res.ok)
        self.assertEqual(env.home_calls, 1)
        self.assertIsNone(ex.holding)              # 夹爪张开 -> holding 清空


# --------------------------------------------------------------------------- #
# 7) N1: 带载归位（holding != None 且 plan_joint 因接触判非法返回 None）
# --------------------------------------------------------------------------- #
class TestLoadedHome(unittest.TestCase):
    def test_loaded_home_retries_with_ignored_gripper_contacts(self):
        checker = _StubChecker(obj_body="box_blue")
        planner = _HeldAwarePlanner(checker)
        env = _StubEnv(force=0.0, gripper=0.44)
        ex = _executor(env, _StubRunner(), planner)
        ex.holding = "box_blue"
        res = ex(SkillCall("home"))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(len(planner.joint_calls), 2)       # 第一次 None，忽略后成功
        # 规划窗口结束后 mask 必须还原，不能污染后续规划
        self.assertFalse(bool(planner.checker._ignore_mask.any()))
        # 夹爪仍闭合 -> 物体没被丢下
        self.assertEqual(ex.holding, "box_blue")

    def test_loaded_home_still_blocks_when_retry_fails(self):
        checker = _StubChecker(obj_body="box_blue")
        planner = _StubPlanner(joint_path=None)
        planner.checker = checker
        env = _StubEnv(force=200.0, gripper=0.44)
        runner = _StubRunner(fail_when=lambda c: True, error_code="ik_fail")
        ex = _executor(env, runner, planner)
        ex.holding = "box_blue"
        res = ex(SkillCall("home"))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "home_blocked")
        self.assertEqual(env.home_calls, 0)                 # 安全门保留：不盲目瞬移

    def test_loaded_home_allowed_when_force_safe(self):
        """P0 变更：带载（夹爪未张开）时**一律禁止**瞬移兜底，即使接触力安全。

        旧行为（task-68）"带载 + 力安全 -> 放行瞬移"实测会让机械臂瞬间跑到 0.6 m 外、
        手里夹着的方块被丢回桌面 -> 复核 subject_still_present。现在改判 home_blocked，
        持物状态保留，由上层走关节路径或先放下物体。"""
        env = _StubEnv(force=0.6, gripper=0.44)
        planner = _StubPlanner(joint_path=None)
        ex = _executor(env, _StubRunner(), planner)
        ex.holding = "box_blue"
        res = ex(SkillCall("home"))
        self.assertFalse(res.ok, res.message)
        self.assertEqual(res.error_code, "home_blocked")
        self.assertEqual(env.home_calls, 0, "带载禁止瞬移")
        self.assertEqual(ex.holding, "box_blue")            # 物体没被丢下
        self.assertTrue(res.metrics.get("loaded"))

    def test_loaded_home_blocked_when_force_high(self):
        """反例：带载但接触力 30N -> 仍须 home_blocked（安全门不许放宽）。"""
        env = _StubEnv(force=30.0, gripper=0.44)
        planner = _StubPlanner(joint_path=None)
        ex = _executor(env, _StubRunner(), planner)
        ex.holding = "box_blue"
        res = ex(SkillCall("home"))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "home_blocked")
        self.assertEqual(env.home_calls, 0)

    def test_unloaded_home_blocked_when_gripper_closed(self):
        """反例：**无载荷**且开合 0.44 -> 旧门原样，仍须 home_blocked。"""
        env = _StubEnv(force=0.6, gripper=0.44)
        planner = _StubPlanner(joint_path=None)
        ex = _executor(env, _StubRunner(), planner)
        ex.holding = None
        res = ex(SkillCall("home"))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "home_blocked")
        self.assertEqual(env.home_calls, 0)

    def test_unloaded_home_does_not_retry(self):
        checker = _StubChecker(obj_body="box_blue")
        planner = _HeldAwarePlanner(checker)
        env = _StubEnv(force=0.0, gripper=1.0)
        ex = _executor(env, _StubRunner(), planner)
        ex.holding = None
        res = ex(SkillCall("home"))
        self.assertTrue(res.ok)
        self.assertEqual(len(planner.joint_calls), 1)       # 未带载不重试


# --------------------------------------------------------------------------- #
# 8) N2: 松开夹爪后 holding 刷新
# --------------------------------------------------------------------------- #
class TestHoldingRefresh(unittest.TestCase):
    def test_open_gripper_clears_holding(self):
        env = _StubEnv(gripper=0.3)
        ex = _executor(env, _StubRunner(), _StubPlanner())
        ex.holding = "box_blue"
        res = ex(SkillCall("open_gripper"))
        self.assertTrue(res.ok)
        self.assertIsNone(ex.holding)
        self.assertGreaterEqual(res.metrics["gripper"], 0.8)

    def test_close_gripper_keeps_holding(self):
        env = _StubEnv(gripper=1.0)
        ex = _executor(env, _StubRunner(), _StubPlanner())
        ex.holding = "box_blue"
        ex(SkillCall("close_gripper", {"ratio": 0.3}))
        self.assertEqual(ex.holding, "box_blue")


# --------------------------------------------------------------------------- #
# 9) N3: pick 失败终态自己清干净（不留给下一次任务 / 不触发 scene_reset）
# --------------------------------------------------------------------------- #
class TestPickTerminalCleanup(unittest.TestCase):
    def _setup(self, force=46.6):
        obj = _make_obj("can_yellow", [0.40, 0.0, 0.42], (0.015, 0.015, 0.05),
                        category="cylinder", color="yellow")
        env = _StubEnv([obj], force=force)
        # 关节路径执行后接触力清零（模拟退到"抬高位形"后脱困）
        runner = _StubRunner(on_path=lambda r: setattr(env, "force", 0.0))
        planner = _StubPlanner(pose_path=None, joint_path=[np.zeros(6)])
        ex = _executor(env, runner, planner)
        return env, runner, planner, ex

    def test_pick_failure_cleans_contact_without_reset(self):
        """task-44：失败终态按几何判据清场（物体离开 TCP），不触发 scene_reset。"""
        obj = _make_obj("can_yellow", [0.30, 0.0, 0.20], (0.015, 0.015, 0.05),
                        category="cylinder", color="yellow")
        env = _StubEnv([obj], force=0.0)
        env.tcp_pos = np.array([0.30, 0.0, 0.20])            # 入口就与 TCP 接触
        runner = _StubRunner()
        runner.on_path = lambda r: env.move_object("can_yellow", [0.0, 0.0, 0.40])
        ex = _executor(env, runner, _StubPlanner(pose_path=None, joint_path=[np.zeros(6)]))
        res = ex(SkillCall("pick", {"target_id": "can_yellow"}))
        self.assertFalse(res.ok)
        self.assertTrue(res.metrics.get("escaped"))
        self.assertEqual(env.reset_seeds, [])                # 未触发 scene_reset
        self.assertEqual(ex._objects_touching_ee(), set())   # 终态几何上无接触
        self.assertIn("obstruction_id", res.metrics)         # R5 观测字段

    def test_escape_is_noop_when_already_clear(self):
        env, runner, planner, ex = self._setup(force=0.0)
        self.assertTrue(ex._escape_contact(None, "can_yellow"))
        self.assertEqual(len(runner.calls), 0)               # 已干净：不额外动手臂/夹爪
        self.assertEqual(ex.last_escape_level, "clear")


# --------------------------------------------------------------------------- #
# 10) task-24 要求 1/2：桌面放置（显式 xy / 自选空位）+ 判据加强
# --------------------------------------------------------------------------- #
class TestTablePlace(unittest.TestCase):
    def _setup(self, box_pose, extra=()):
        box = _make_obj("box_red", box_pose, (0.02, 0.02, 0.02))
        env = _StubEnv([box, *extra])
        runner = _StubRunner()
        planner = _StubPlanner(pose_path=None)
        ex = _executor(env, runner, planner)
        ex.holding = "box_red"
        return env, runner, planner, ex

    def test_table_place_with_explicit_xy(self):
        xy = [0.42, -0.30]
        env, runner, planner, ex = self._setup([xy[0], xy[1], TABLE_HEIGHT + 0.02])
        res = ex(SkillCall("place", {"surface": "table", "xy": xy}))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics["target_kind"], "table")
        self.assertEqual(res.metrics["spot"], "explicit_xy")
        self.assertAlmostEqual(res.metrics["target_xy"][0], xy[0], places=4)
        self.assertAlmostEqual(res.metrics["target_xy"][1], xy[1], places=4)
        self.assertAlmostEqual(res.metrics["xy"][0], xy[0], places=4)
        self.assertAlmostEqual(res.metrics["support_z"], TABLE_HEIGHT, places=4)
        self.assertAlmostEqual(res.metrics["dz"], 0.02, places=6)
        self.assertAlmostEqual(res.metrics["bottom_gap"], 0.0, places=6)
        self.assertLessEqual(res.metrics["force"], 5.0)
        # 下降目标 z = TABLE_HEIGHT + 半高 + 余量
        descent = [c for c in runner.calls if abs(float(c["pos"][2]) - (TABLE_HEIGHT + 0.02
                                                                       + PLACE_TABLE_MARGIN_M)) < 1e-9]
        self.assertEqual(len(descent), 1)

    def test_table_place_auto_spot_is_free(self):
        tray = _make_obj("tray", [0.40, 0.30, TABLE_HEIGHT + 0.012], (0.09, 0.07, 0.012),
                         category="tray", color="gray", graspable=False, is_container=True)
        box = _make_obj("box_red", [0.35, -0.14, TABLE_HEIGHT + 0.02], (0.02, 0.02, 0.02))
        env = _StubEnv([box, tray])
        ex = _executor(env, _StubRunner(), _StubPlanner(pose_path=None))
        ex.holding = "box_red"
        spot, note = ex._choose_table_spot(box)
        self.assertTrue(str(note).startswith("grasp_aware"), note)
        # task-28：落点必须落在"实测可抓带"里（放下后还能再抓起来）
        self.assertLessEqual(ex._band_distance(spot), 1e-9, note)
        for o in env.list_objects():
            if o.id == "box_red":
                continue
            d = float(np.linalg.norm(spot - np.asarray(o.pose[:2], float)))
            self.assertGreaterEqual(d, 0.02 + max(o.size[0], o.size[1]) + 0.08 - 1e-9)
        # 把物体摆到该空位后，放置应成功且 metrics 记录最终 xy
        env.set_object_pose("box_red", np.array([spot[0], spot[1], TABLE_HEIGHT + 0.02,
                                                 1.0, 0.0, 0.0, 0.0]))
        res = ex(SkillCall("place", {"surface": "table"}))
        self.assertTrue(res.ok, res.message)
        self.assertTrue(str(res.metrics["spot"]).startswith("grasp_aware"))
        self.assertIn("spot_reason", res.metrics)
        self.assertAlmostEqual(res.metrics["target_xy"][0], float(spot[0]), places=4)
        self.assertAlmostEqual(res.metrics["xy"][1], float(spot[1]), places=4)

    def test_table_place_rejects_big_dz(self):
        # 物体停在桌面上方 50 mm（例如被别的物体顶着）-> 旧判据会误报成功，新判据必须拒绝
        env, runner, planner, ex = self._setup([0.42, -0.30, TABLE_HEIGHT + 0.02 + 0.05])
        res = ex(SkillCall("place", {"surface": "table", "xy": [0.42, -0.30]}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "place_mismatch")
        self.assertGreater(res.metrics["bottom_gap"], PLACE_SEAT_MAX_M)
        self.assertIn("支撑面", res.message)

    def test_table_place_rejects_xy_offset(self):
        env, runner, planner, ex = self._setup([0.52, -0.30, TABLE_HEIGHT + 0.02])
        res = ex(SkillCall("place", {"surface": "table", "xy": [0.42, -0.30]}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "place_mismatch")
        self.assertGreater(res.metrics["dxy_target"], PLACE_XY_TOL_M)

    def test_container_place_rejects_stacked_above_rim(self):
        """旧实现把"摞在盘内另一物体上"（用户实测 dz=59 mm）判为成功；新判据必须拒绝。"""
        tray = _make_obj("tray", [0.40, 0.0, TABLE_HEIGHT + 0.012], (0.09, 0.07, 0.012),
                         category="tray", color="gray", graspable=False, is_container=True)
        # 盘口 = 0.424；物体底面 0.450（高出盘口 26 mm = 搁在盘内另一物体上）
        box = _make_obj("box_red", [0.40, 0.0, 0.47], (0.02, 0.02, 0.02))
        env = _StubEnv([box, tray])
        env.contact_z = 0.454          # 容器内接触式下降的目标高度 0.424+0.02+0.01
        env.track_moves = True
        ex = _executor(env, _StubRunner(env=env), _StubPlanner(pose_path=None))
        ex.holding = "box_red"
        res = ex(SkillCall("place", {"container_id": "tray"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "place_mismatch")
        self.assertFalse(res.metrics["seat_ok"])
        self.assertGreater(res.metrics["bottom_vs_table"], 0.02)

    def test_container_path_still_supported(self):
        """冻结接口：container_id 与 surface 二选一，容器路径必须继续可用。"""
        tray = _make_obj("tray", [0.40, 0.0, TABLE_HEIGHT + 0.012], (0.09, 0.07, 0.012),
                         category="tray", color="gray", graspable=False, is_container=True)
        # 托盘上表面 = 0.412+0.012 = 0.424；方块落在其上的中心 = 0.424+0.02 = 0.444
        box = _make_obj("box_red", [0.40, 0.0, TABLE_HEIGHT + 0.044], (0.02, 0.02, 0.02))
        env = _StubEnv([box, tray])
        env.contact_z = 0.454          # 容器内接触式下降的目标高度 0.424+0.02+0.01
        env.track_moves = True
        ex = _executor(env, _StubRunner(env=env), _StubPlanner(pose_path=None))
        ex.holding = "box_red"
        res = ex(SkillCall("place", {"container_id": "tray"}))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics["target_kind"], "tray")
        self.assertAlmostEqual(res.metrics["touchdown_z"], 0.454, delta=0.002)


# --------------------------------------------------------------------------- #
# 11) task-24 要求 3：沿 +反力方向撤退 / 多方向 / 力不失控
# --------------------------------------------------------------------------- #
class TestRetreatDirectionSafety(unittest.TestCase):
    """task-68 根因 1：力还在噪声量级（<= threshold，等于没接触）时，撤退方向必须纯 +z，
    且任何候选方向的 z 都不得为负（实机 0.6N 朝下噪声 -> 主动下压 8cm -> 32.3N）。"""

    def test_escape_dir_never_points_down(self):
        for f_vec in ([0.0, 0.0, -0.6], [0.013, -0.006, -0.6], [0.1, 0.0, -5.0]):
            d = SkillExecutor._escape_dir(np.asarray(f_vec, float), 0)
            self.assertGreaterEqual(float(d[2]), -1e-9,
                                    f"方向朝下: {np.round(d, 3).tolist()} (f={f_vec})")

    def test_retreat_with_downward_noise_goes_up(self):
        env = _StubEnv(force=-0.6, gripper=0.44)      # 朝下 0.6N 噪声（复刻实机）
        env.tcp_pos = np.array([0.35, 0.0, 0.50])
        runner = _StubRunner()
        ex = _executor(env, runner, _StubPlanner(pose_path=None))
        ex._retreat(None)
        retreats = [c for c in runner.calls if c["compliance"] is not None
                     and int(np.count_nonzero(c["compliance"].axes)) == 6]
        self.assertTrue(retreats)
        z_cmd = float(np.asarray(retreats[-1]["pos"], float)[2])
        self.assertGreater(z_cmd, 0.50, f"撤退没有向上: z={z_cmd:.4f}")


class TestDirectionalRetreat(unittest.TestCase):
    def test_escape_dir_runs_along_plus_force(self):
        """contract: contact_force_on_ee 返回**环境施加在末端上的力**，所以退开方向 = +f̂。"""
        np.testing.assert_allclose(SkillExecutor._escape_dir(np.array([0.0, 0.0, 30.0]), 0),
                                   [0.0, 0.0, 1.0], atol=1e-9)
        np.testing.assert_allclose(SkillExecutor._escape_dir(np.array([25.0, 0.0, 0.0]), 0),
                                   [1.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(SkillExecutor._escape_dir(np.array([25.0, 0.0, 0.0]), 1),
                                   [0.0, 0.0, 1.0], atol=1e-9)
        # |f|≈0 -> 纯向上
        np.testing.assert_allclose(SkillExecutor._escape_dir(np.zeros(3), 0),
                                   [0.0, 0.0, 1.0], atol=1e-9)

    def test_retreat_first_target_along_force(self):
        env = _StubEnv(force=50.0)                     # f = +z
        runner = _StubRunner()
        ex = _executor(env, runner, _StubPlanner(joint_path=[HOME_QPOS.copy()]))
        ex._retreat()
        p0 = np.asarray(runner.calls[0]["pos"], float)
        np.testing.assert_allclose(p0, np.array([0.35, 0.0, 0.60]) + np.array([0.0, 0.0, 0.08]),
                                   atol=1e-9)
        self.assertAlmostEqual(float(runner.calls[0]["duration"]), RETREAT_DURATION_S, places=6)
        self.assertLessEqual(float(runner.calls[0]["force_limit"]), RETREAT_MAX_FORCE_N)

    def test_no_escalating_staircase(self):
        """R1/R6：旧版 5 方向 × 力上限逐级抬高（40→68.6 N）已删除；现在只有一步且 <=45 N。"""
        blocker = _make_obj("blocker", [0.30, 0.0, 0.20], (0.02, 0.02, 0.02))
        env = _StubEnv([blocker], force=200.0)     # 高力也不该抬高上限
        env.tcp_pos = np.array([0.30, 0.0, 0.20])
        runner = _StubRunner()
        ex = _executor(env, runner, _StubPlanner(joint_path=[HOME_QPOS.copy()]))
        ex._retreat()
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(len(runner.joint_calls), 0)
        lims = [float(c["force_limit"]) for c in runner.calls]
        self.assertTrue(all(lim <= RETREAT_FORCE_CAP_N for lim in lims), lims)
        self.assertTrue(all(lim <= 45.0 for lim in lims), lims)
        self.assertTrue(all(abs(float(c["duration"]) - RETREAT_DURATION_S) < 1e-6
                            for c in runner.calls))

    def test_finish_failure_resets_only_after_all_fail(self):
        """只有温和脱身 + 关节抬升都清不掉才 scene_reset，并返回 error_code='scene_reset'。"""
        blocker = _make_obj("blocker", [0.30, 0.0, 0.20], (0.02, 0.02, 0.02))
        env = _StubEnv([blocker], force=60.0)           # 几何上一直贴着 TCP
        env.tcp_pos = np.array([0.30, 0.0, 0.20])
        runner = _StubRunner(fail_when=lambda c: True, error_code="force_limit")
        planner = _StubPlanner(joint_path=None)
        ex = _executor(env, runner, planner)
        extra, override, note = ex._finish_failure(None, "box_red")
        self.assertFalse(extra["escaped"])
        self.assertEqual(override, "scene_reset")
        self.assertEqual(extra["escape_level"], "blocked")
        self.assertEqual(env.reset_seeds, [0])
        self.assertIn("重置场景", note)


# --------------------------------------------------------------------------- #
# 12) task-28：桌面落点偏好"可抓带" + 任意落点再抓鲁棒性
# --------------------------------------------------------------------------- #
class TestGraspAwareSpot(unittest.TestCase):
    def _default_scene(self):
        objs = [
            _make_obj("box_blue", [0.35, 0.14, TABLE_HEIGHT + 0.02], (0.02, 0.02, 0.02), color="blue"),
            _make_obj("box_red", [0.35, -0.14, TABLE_HEIGHT + 0.02], (0.02, 0.02, 0.02)),
            _make_obj("box_green", [0.45, -0.05, TABLE_HEIGHT + 0.02], (0.02, 0.02, 0.02), color="green"),
            _make_obj("can_yellow", [0.40, 0.02, TABLE_HEIGHT + 0.035], (0.015, 0.015, 0.035),
                      category="cylinder", color="yellow"),
            _make_obj("tray", [0.40, 0.30, TABLE_HEIGHT + 0.012], (0.09, 0.07, 0.012),
                      category="tray", color="gray", graspable=False, is_container=True),
        ]
        return _StubEnv(objs)

    def test_spot_lands_in_graspable_band(self):
        """task-28 要求 1：自选落点必须优先落在实测可抓带内（而不是"只要有空"）。"""
        env = self._default_scene()
        held = env._objects["box_blue"]
        ex = _executor(env, _StubRunner(), _StubPlanner(pose_path=None))
        spot, note = ex._choose_table_spot(held)
        self.assertLessEqual(ex._band_distance(spot), 1e-9,
                             "落点不在可抓带内: %s (%s)" % (np.round(spot, 3).tolist(), note))
        bx0, bx1, by0, by1 = PLACE_GRASPABLE_BAND
        self.assertLessEqual(bx0, float(spot[0]))
        self.assertLessEqual(float(spot[0]), bx1)
        # reason 里要写清排序依据，便于核对
        self.assertIn("可抓带", note)
        self.assertIn("净距", note)

    def test_spot_prefers_band_over_bare_clearance(self):
        """同一个场景里，旧锚点 (0.44,-0.30) 在带外；新选择必须更靠近带。"""
        env = self._default_scene()
        held = env._objects["box_blue"]
        ex = _executor(env, _StubRunner(), _StubPlanner(pose_path=None))
        spot, _note = ex._choose_table_spot(held)
        self.assertLess(ex._band_distance(spot), ex._band_distance(np.array([0.44, -0.30])))


class TestPickRobustness(unittest.TestCase):
    def _setup(self):
        obj = _make_obj("box_blue", [0.35, 0.14, TABLE_HEIGHT + 0.02], (0.02, 0.02, 0.02), color="blue")
        env = _StubEnv([obj])
        # 物体永远抬不起来（stub 无物理）：所有尝试都失败，用来检查重试覆盖与失败信息
        ex = _executor(env, _StubRunner(), _StubPlanner(pose_path=None))
        return env, ex

    def test_attempt_plan_capped_at_three(self):
        """task-44/R3：盲重试 10 组 -> 3 组，仍保留 深度/闭合量/偏航 的代表性组合。"""
        env, ex = self._setup()
        res = ex(SkillCall("pick", {"target_id": "box_blue"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "grasp_failed")
        self.assertEqual(PICK_MAX_ATTEMPTS, 3)
        self.assertEqual(int(res.metrics["attempts"]), PICK_MAX_ATTEMPTS)
        reasons = " ".join(res.metrics["reasons"])
        self.assertIn("yaw=90", reasons)                    # 90° 偏航变体（第 3 组）
        self.assertIn("dz=+6mm", reasons)                  # +0.3h 略浅
        self.assertIn("原因", res.message)                  # 给出明确失败原因
        self.assertNotIn("仅抬升", res.message)
        # R5/U1：失败观测里必须带受阻物字段（这里无接触 -> None）
        self.assertIn("obstruction_id", res.metrics)
        self.assertIn("suggest_image", res.metrics)

    def test_guarded_close_stops_when_force_high(self):
        """guarded close：合爪时力超阈值就停住，不再继续捏（60 N 远超 45 N 阈值）。"""
        env = _StubEnv(force=60.0)
        ex = _executor(env, _StubRunner(), _StubPlanner())
        self.assertFalse(ex._guarded_close(0.30))
        # 停住 = 把夹爪目标改回"当前开合"，而不是继续往 0.30 捏
        self.assertEqual(float(env.gripper_cmds[-1]), float(env.gripper))

    def test_guarded_close_completes_when_clear(self):
        env = _StubEnv(force=0.0)
        ex = _executor(env, _StubRunner(), _StubPlanner())
        self.assertTrue(ex._guarded_close(0.30))


# --------------------------------------------------------------------------- #
# 13) task-32：叠放 on_top_of
# --------------------------------------------------------------------------- #
class TestStackPlace(unittest.TestCase):
    def _setup(self, held_pose=None, anchor_pose=(0.35, -0.14, TABLE_HEIGHT + 0.02),
               on_move=None, anchor_id="box_red", held="box_blue"):
        anchor = _make_obj(anchor_id, list(anchor_pose), (0.02, 0.02, 0.02), color="red")
        if held_pose is None:
            # 目标叠放位：锚点顶面 + 被夹物半高 + 余量
            held_pose = [anchor_pose[0], anchor_pose[1],
                         anchor_pose[2] + 0.02 + 0.02 + PLACE_STACK_MARGIN_M]
        box = _make_obj(held, list(held_pose), (0.02, 0.02, 0.02), color="blue")
        env = _StubEnv([box, anchor])
        # task-59：叠放默认走接触式下降；stub 用 contact_z 模拟"叠到位时触到锚点顶面"
        env.contact_z = float(held_pose[2])
        env.track_moves = True
        runner = _StubRunner(on_move=on_move, env=env)
        ex = _executor(env, runner, _StubPlanner(pose_path=None))
        ex.holding = held
        return env, ex, runner

    def test_stack_success(self):
        env, ex, runner = self._setup()
        res = ex(SkillCall("place", {"on_top_of": "box_red"}))
        self.assertTrue(res.ok, res.message)
        self.assertIsNone(res.error_code)
        # lead 两段式：阶段1 原速走到"目标上方 10mm"，xy 就是目标 xy
        descent = runner.calls[1]
        np.testing.assert_allclose(descent["pos"][:2], [0.35, -0.14], atol=1e-9)
        self.assertAlmostEqual(float(descent["pos"][2]),
                               TABLE_HEIGHT + 0.04 + 0.02 + PLACE_STACK_MARGIN_M
                               + PLACE_CONTACT_LAST10_M, places=6)
        self.assertAlmostEqual(res.metrics["touchdown_z"],
                               TABLE_HEIGHT + 0.064, delta=0.002)
        self.assertGreaterEqual(res.metrics["contact_force_norm"], 2.5)
        self.assertEqual(res.metrics["anchor"], "box_red")
        self.assertAlmostEqual(res.metrics["anchor_top_z"], TABLE_HEIGHT + 0.04, places=4)
        self.assertAlmostEqual(res.metrics["stack_z"], TABLE_HEIGHT + 0.064, places=4)
        self.assertAlmostEqual(res.metrics["dz"], 0.024, places=4)
        self.assertEqual(res.metrics["xy_offset"], [0.0, 0.0])
        self.assertAlmostEqual(res.metrics["bottom_gap"], PLACE_STACK_MARGIN_M, places=4)
        self.assertTrue(res.metrics["footprint_ok"])
        self.assertTrue(res.metrics["stable_ok"])

    def test_stack_offset_moves_target(self):
        env, ex, runner = self._setup()
        res = ex(SkillCall("place", {"on_top_of": "box_red", "xy": [0.01, -0.01]}))
        self.assertEqual(res.metrics["xy_offset"], [0.01, -0.01])
        np.testing.assert_allclose(runner.calls[1]["pos"][:2], [0.36, -0.15], atol=1e-9)
        self.assertTrue(res.ok, res.message)

    def test_model_xy_absolute_in_no_oracle_stack(self):
        """P0：无真值链路给了 xy 时，落点必须直接用该 xy（绝对），绝不能叠加锚点位姿。

        实测 bug：模型给 xy=[0.397,0.303]（托盘里红块的位置），旧实现叠加锚点位姿后变成
        (0.794,0.607) -> 工作空间边角 -> unreachable，g2 叠放必失败。
        """
        env, ex, runner = self._setup()
        ex.holding = HOLDING_UNNAMED            # 无真值（不知道 id）
        model_xy = [0.397, 0.303]
        res = ex(SkillCall("place", {"on_top_of": "box_red", "xy": model_xy,
                                     "holding": True}))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics.get("xy_source"), "model_absolute")
        self.assertEqual(res.metrics.get("entry_clear_skipped"), "holding_arg")
        np.testing.assert_allclose(runner.calls[0]["pos"][:2], model_xy, atol=1e-9)
        self.assertNotAlmostEqual(float(runner.calls[0]["pos"][0]), 0.397 + 0.35, places=3)

    def test_oracle_stack_xy_still_relative_offset(self):
        """oracle 叠放语义不变：xy 仍是"相对锚点的水平偏移"（task-32 冻结接口）。"""
        env, ex, runner = self._setup()        # holding=box_blue(已知 id) -> oracle 路径
        res = ex(SkillCall("place", {"on_top_of": "box_red", "xy": [0.01, -0.01]}))
        self.assertEqual(res.metrics.get("xy_source"), "anchor_offset")
        np.testing.assert_allclose(runner.calls[0]["pos"][:2], [0.36, -0.15], atol=1e-9)
        self.assertTrue(res.ok, res.message)

    def test_stack_anchor_missing(self):
        env, ex, _ = self._setup()
        res = ex(SkillCall("place", {"on_top_of": "ghost"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "object_not_found")

    def test_stack_anchor_held_by_us(self):
        env, ex, _ = self._setup()
        ex.holding = "box_red"
        res = ex(SkillCall("place", {"on_top_of": "box_red"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "anchor_held")

    def test_stack_anchor_not_on_surface(self):
        env, ex, _ = self._setup(anchor_pose=(0.35, -0.14, 0.50))
        res = ex(SkillCall("place", {"on_top_of": "box_red"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "anchor_not_on_surface")

    def test_stack_anchor_not_graspable(self):
        tray = _make_obj("tray", [0.35, -0.14, TABLE_HEIGHT + 0.012], (0.09, 0.07, 0.012),
                         category="tray", color="gray", graspable=False, is_container=True)
        box = _make_obj("box_blue", [0.35, -0.14, TABLE_HEIGHT + 0.036], (0.02, 0.02, 0.02), color="blue")
        env = _StubEnv([box, tray])
        ex = _executor(env, _StubRunner(), _StubPlanner(pose_path=None))
        ex.holding = "box_blue"
        res = ex(SkillCall("place", {"on_top_of": "tray"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "not_graspable")

    def test_stack_rejects_slide_after_release(self):
        holder = {}

        def on_move(r):
            # 接触式下降有很多步；用"目标 z 明显高于接触面"识别释放后的抬升段
            p = np.asarray(r.calls[-1]["pos"], float)
            if float(p[2]) > 0.56:
                holder["env"].move_object("box_blue", [0.03, 0.0, 0.0])

        env, ex, runner = self._setup(on_move=on_move)
        holder["env"] = env
        res = ex(SkillCall("place", {"on_top_of": "box_red"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "place_mismatch")
        self.assertFalse(res.metrics["stable_ok"])
        self.assertGreater(res.metrics["slide"], PLACE_STACK_SLIDE_MAX_M)

    def test_stack_rejects_xy_outside_footprint(self):
        env, ex, runner = self._setup(held_pose=[0.40, -0.14, TABLE_HEIGHT + 0.064])
        res = ex(SkillCall("place", {"on_top_of": "box_red"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "place_mismatch")
        self.assertFalse(res.metrics["footprint_ok"])

    def test_modes_are_exclusive(self):
        env, ex, _ = self._setup()
        res = ex(SkillCall("place", {"container_id": "tray", "on_top_of": "box_red"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "bad_args")

    def test_container_path_unchanged(self):
        tray = _make_obj("tray", [0.40, 0.0, TABLE_HEIGHT + 0.012], (0.09, 0.07, 0.012),
                         category="tray", color="gray", graspable=False, is_container=True)
        box = _make_obj("box_red", [0.40, 0.0, TABLE_HEIGHT + 0.044], (0.02, 0.02, 0.02))
        env = _StubEnv([box, tray])
        env.contact_z = 0.454
        env.track_moves = True
        ex = _executor(env, _StubRunner(env=env), _StubPlanner(pose_path=None))
        ex.holding = "box_red"
        res = ex(SkillCall("place", {"container_id": "tray"}))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics["target_kind"], "tray")


# --------------------------------------------------------------------------- #
# 14) task-36：横向关系放置 anchor_id + relation
# --------------------------------------------------------------------------- #
class TestSideRelationPlace(unittest.TestCase):
    def _setup(self, held_pose=(0.40, 0.14, TABLE_HEIGHT + 0.02),
               anchor=(0.35, 0.14, TABLE_HEIGHT + 0.02), extra=(),
               held="box_green", anchor_id="box_blue"):
        a = _make_obj(anchor_id, list(anchor), (0.02, 0.02, 0.02), color="blue")
        h = _make_obj(held, list(held_pose), (0.02, 0.02, 0.02), color="green")
        env = _StubEnv([h, a, *extra])
        ex = _executor(env, _StubRunner(), _StubPlanner(pose_path=None))
        ex.holding = held
        return env, ex

    def test_right_is_plus_x(self):
        env, ex = self._setup()
        res = ex(SkillCall("place", {"anchor_id": "box_blue", "relation": "right"}))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics["relation_dir"], [1.0, 0.0])   # right = +x
        self.assertAlmostEqual(res.metrics["expected_dist"], 0.05, places=6)  # .02+.02+.01
        self.assertAlmostEqual(res.metrics["actual_dist"], 0.05, places=3)
        self.assertGreater(res.metrics["side_dot"], 0.0)
        self.assertTrue(res.metrics["side_ok"])

    def test_left_is_minus_x(self):
        env, ex = self._setup(held_pose=(0.30, 0.14, TABLE_HEIGHT + 0.02))
        res = ex(SkillCall("place", {"anchor_id": "box_blue", "relation": "left"}))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics["relation_dir"], [-1.0, 0.0])
        self.assertGreater(res.metrics["side_dot"], 0.0)

    def test_front_is_plus_y(self):
        env, ex = self._setup(held_pose=(0.35, 0.19, TABLE_HEIGHT + 0.02))
        res = ex(SkillCall("place", {"anchor_id": "box_blue", "relation": "front"}))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics["relation_dir"], [0.0, 1.0])   # front = +y
        self.assertGreater(res.metrics["side_dot"], 0.0)

    def test_beside_picks_clearest_direction(self):
        # 右/左/前 各摆一个障碍，只剩 back(-y) 空 -> beside 必须选 back
        extras = [
            _make_obj("o_r", [0.44, 0.0, TABLE_HEIGHT + 0.02], (0.02, 0.02, 0.02)),
            _make_obj("o_l", [0.26, 0.0, TABLE_HEIGHT + 0.02], (0.02, 0.02, 0.02)),
            _make_obj("o_f", [0.35, 0.09, TABLE_HEIGHT + 0.02], (0.02, 0.02, 0.02)),
        ]
        env, ex = self._setup(held_pose=(0.35, -0.05, TABLE_HEIGHT + 0.02),
                              anchor=(0.35, 0.0, TABLE_HEIGHT + 0.02), extra=extras)
        res = ex(SkillCall("place", {"anchor_id": "box_blue", "relation": "beside"}))
        self.assertEqual(res.metrics["relation"], "beside")
        self.assertEqual(res.metrics["relation_dir"], [0.0, -1.0])
        self.assertIn("beside 自动选", res.metrics["relation_reason"])
        self.assertTrue(res.ok, res.message)

    def test_unsupported_relation_rejected(self):
        env, ex = self._setup()
        res = ex(SkillCall("place", {"anchor_id": "box_blue", "relation": "diagonal"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "bad_args")

    def test_anchor_id_without_relation_rejected(self):
        env, ex = self._setup()
        res = ex(SkillCall("place", {"anchor_id": "box_blue"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "bad_args")

    def test_distance_override(self):
        env, ex = self._setup(held_pose=(0.47, 0.14, TABLE_HEIGHT + 0.02))
        res = ex(SkillCall("place",
                           {"anchor_id": "box_blue", "relation": "right", "distance": 0.12}))
        self.assertAlmostEqual(res.metrics["expected_dist"], 0.12, places=6)
        self.assertTrue(res.ok, res.message)

    def test_wrong_side_rejected(self):
        env, ex = self._setup(held_pose=(0.30, 0.14, TABLE_HEIGHT + 0.02))
        res = ex(SkillCall("place", {"anchor_id": "box_blue", "relation": "right"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "place_mismatch")
        self.assertLessEqual(res.metrics["side_dot"], 0.0)

    def test_distance_out_of_range_rejected(self):
        env, ex = self._setup(held_pose=(0.60, 0.14, TABLE_HEIGHT + 0.02))
        res = ex(SkillCall("place", {"anchor_id": "box_blue", "relation": "right"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "place_mismatch")
        self.assertFalse(res.metrics["side_ok"])


# --------------------------------------------------------------------------- #
# 15) task-36：中止归还 return_held_object
# --------------------------------------------------------------------------- #
class TestReturnHeldObject(unittest.TestCase):
    def _setup(self, held_pose=(0.35, 0.14, TABLE_HEIGHT + 0.02), last_pick_pose="auto",
               runner=None):
        obj = _make_obj("box_blue", list(held_pose), (0.02, 0.02, 0.02), color="blue")
        env = _StubEnv([obj])
        ex = _executor(env, runner if runner is not None else _StubRunner(),
                       _StubPlanner(pose_path=None))
        ex.holding = "box_blue"
        if last_pick_pose == "auto":
            ex.last_pick_pose = np.asarray(obj.pose, float).copy()
        elif last_pick_pose is not None:
            ex.last_pick_pose = np.asarray(last_pick_pose, float)
        return env, ex

    def test_return_to_pick_pose(self):
        env, ex = self._setup()
        res = ex.return_held_object()
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics["mode"], "pick_pose")
        self.assertIsNone(ex.holding)
        self.assertLessEqual(res.metrics["bottom_gap"], 0.05)   # 落到桌面，不悬空

    def test_return_to_table_spot_when_no_pick_pose(self):
        env, ex = self._setup(last_pick_pose=None)
        res = ex.return_held_object()
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics["mode"], "table_spot")
        self.assertIsNone(ex.holding)
        self.assertLessEqual(res.metrics["bottom_gap"], 0.05)

    def test_return_drop_when_unreachable(self):
        runner = _StubRunner(fail_when=lambda c: True, error_code="unreachable")
        env, ex = self._setup(runner=runner)
        res = ex.return_held_object()
        self.assertEqual(res.metrics["mode"], "drop")
        self.assertIsNone(ex.holding)
        self.assertFalse(res.metrics["suspended"])

    def test_return_reports_suspended(self):
        runner = _StubRunner(fail_when=lambda c: True, error_code="unreachable")
        env, ex = self._setup(held_pose=(0.35, 0.14, 0.60), last_pick_pose=None, runner=runner)
        res = ex.return_held_object()
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "return_incomplete")
        self.assertTrue(res.metrics["suspended"])
        self.assertIsNone(ex.holding)

    def test_return_without_holding(self):
        env, ex = self._setup()
        ex.holding = None
        res = ex.return_held_object()
        self.assertTrue(res.ok)
        self.assertEqual(res.metrics["mode"], "none")


# --------------------------------------------------------------------------- #
# 16) task-48：pick(target_xy) 无真值路径 + 夹爪实测(jaw_gap)判据
# --------------------------------------------------------------------------- #
class TestPickNoOracle(unittest.TestCase):
    TARGET = (0.35, -0.14)

    def _setup(self, obj_pose=(0.35, -0.14, TABLE_HEIGHT + 0.02), runner=None):
        obj = _make_obj("box_blue", list(obj_pose), (0.02, 0.02, 0.02), color="blue")
        env = _StubEnv([obj])
        ex = _executor(env, runner if runner is not None else _StubRunner(),
                       _StubPlanner(pose_path=None))
        return env, ex

    def test_slender_pick_descent_uses_vmax_and_contact_semantics(self):
        """§6.2#2：圆柱/水瓶的 pick 下降 v_max=0.008 + force_limit=3.0 + timeout>=12；
        返回 force_limit 解释为"到位接触"：取 final_pos 作 grasp_z_effective，就地合爪继续。"""
        env, ex = self._setup()
        env.jaw_block_ratio = 0.66                    # 指间被卡在 64.2mm ≈ 瓶径（正面证据通过）
        env.close_force = 20.0                       # 合爪被瓶身挡住 -> 有接触力
        # 方案B（顶面语义，lead 修正）：目标 z = hint - 0.03 = 0.43
        ex.runner = _SlenderPhase2Fail(0.43)          # 只在最后 10mm（目标 z=0.43）中止
        res = ex(SkillCall("pick", {"target_xy": [0.35, -0.14], "target_z_hint": 0.46,
                                    "target_category": "bottle",
                                    "target_size": [0.032, 0.032, 0.10]}))
        self.assertTrue(res.ok, res.message)                     # 接触中止被解释为到位，不是失败
        calls = [c for c in ex.runner.calls if c["compliance"] is PLACE_PICK_DESCEND_COMPLIANCE]
        # 两段式：巡航(中速+3N) + 最后 10mm(限速 CONTACT_VMAX) + 中止后回退 2mm
        self.assertGreaterEqual(len(calls), 3)
        self.assertAlmostEqual(float(calls[0]["v_max"]), PICK_CRUISE_VMAX_SLENDER, places=9)
        last = calls[1]
        self.assertAlmostEqual(float(last["v_max"]), CONTACT_VMAX, places=9)
        self.assertAlmostEqual(float(last["force_limit"]), PICK_SLENDER_FORCE_LIMIT_N, places=9)
        self.assertGreaterEqual(float(last["timeout"]), CONTACT_VMAX_TIMEOUT_MIN_S)
        self.assertAlmostEqual(float(calls[-1]["pos"][2]), 0.43 + PICK_CONTACT_BACKOFF_M, places=6)
        # 接触高度取 final_pos（桩里 final_pos = 命令位姿）
        self.assertAlmostEqual(float(res.metrics["grasp_z_effective"]), 0.43, places=4)
        self.assertGreaterEqual(float(res.metrics["touchdown_force"]), PICK_SLENDER_FORCE_LIMIT_N)

    def test_box_pick_descent_keeps_legacy_no_low_force_limit(self):
        """§6.2#1：方块/低矮物体保持现状（v_max<=0.03）且**不开低力限**。"""
        env, ex = self._setup()
        env.jaw_block_ratio = 0.45
        res = ex(SkillCall("pick", {"target_xy": [0.35, -0.14]}))
        self.assertTrue(res.ok, res.message)
        calls = [c for c in ex.runner.calls
                 if np.allclose(np.asarray(c["pos"], float)[:2], [0.35, -0.14], atol=1e-9)
                 and float(np.asarray(c["pos"], float)[2]) <= TABLE_HEIGHT + PICK_DEFAULT_HALF[2] + 1e-9]
        self.assertTrue(calls)
        for c in calls:
            self.assertIsNone(c["force_limit"])                  # 不开低力限
            self.assertIsNone(c["v_max"])                        # 也不限速（实测 0.03 会伤方块收尾）

    def test_top_descent_huge_force_is_anomaly_not_contact(self):
        """兜底硬上限：单步力爆表时按异常中止（force_limit），不许当成"接触"顺手合爪。"""
        env, ex = self._setup()
        env.jaw_block_ratio = 0.45
        ex.runner = _StubRunner(max_force=2910.3)
        res = ex(SkillCall("pick", {"target_xy": [0.35, -0.14]}))
        self.assertFalse(res.ok, "单步力爆表被当成接触成功了")
        self.assertIn(res.error_code, ("force_limit", "grasp_failed"))
        self.assertIn("force_limit", str(res.metrics.get("reasons")))

    def test_target_xy_grasps_and_reports_jaw_gap(self):
        env, ex = self._setup()
        env.jaw_block_ratio = 0.45                 # 指间有 45 mm 物体 -> 合不到位
        res = ex(SkillCall("pick", {"target_xy": list(self.TARGET)}))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics.get("grasp_evidence"), "jaw_gap+lift+carry")
        self.assertTrue(res.metrics.get("grasp_lift_ok"))
        self.assertIsNone(res.metrics.get("grasp_carry_evidence"),
                          "桩环境没有接触信息 -> 随动证据为 None（不据此否决）")
        self.assertGreater(res.metrics["grasp_gap_mm"],
                           GRIPPER_GAP_CLOSED_MM + GRASP_JAW_GAP_MIN_MM)
        self.assertLess(res.metrics["grasp_gap_mm"],
                        GRIPPER_GAP_OPEN_MM - GRASP_JAW_OPEN_MARGIN_MM)
        self.assertEqual(ex.holding, HOLDING_UNNAMED)

    def test_target_xy_uses_arg_point_not_truth(self):
        """R-a：真值物体被放到别处，目标点仍必须只来自入参。"""
        env, ex = self._setup(obj_pose=(0.50, 0.20, TABLE_HEIGHT + 0.02))
        env.jaw_block_ratio = 0.45
        res = ex(SkillCall("pick", {"target_xy": list(self.TARGET),
                                    "target_z_hint": 0.425}))
        self.assertTrue(res.ok, res.message)
        # task-59 后下降段是"守卫式接触下降"（小步），最后一步正好停在模型给的 z
        # pick 下降段 = 落点正好在目标 xy/z 的那次调用（方块走"保持现状"档，不传 v_max）
        descent = [c for c in ex.runner.calls
                   if np.allclose(np.asarray(c["pos"], float)[:2], list(self.TARGET), atol=1e-9)
                   and float(np.asarray(c["pos"], float)[2]) <= 0.43 + 1e-9]
        self.assertTrue(descent, "没有对目标 xy 发起下降")
        np.testing.assert_allclose(descent[-1]["pos"][:2], list(self.TARGET), atol=1e-9)
        # 统一口径：max(顶面 0.425 - 0.03, 桌面+0.03=0.43) = 0.43（下限保护生效）
        self.assertAlmostEqual(float(descent[-1]["pos"][2]), 0.43, places=6)
        self.assertIn("grasp_z_effective", res.metrics)   # 实测 TCP 高度（桩里 tcp_pos 是静态的）
        self.assertIn("touchdown_force", res.metrics)
        self.assertEqual(res.metrics.get("oracle_nearest_id"), "box_blue")
        self.assertAlmostEqual(res.metrics.get("oracle_xy_error_mm"),
                               float(np.hypot(0.50 - 0.35, 0.20 + 0.14)) * 1000, places=1)

    def test_target_xy_default_z_uses_table_plus_half(self):
        env, ex = self._setup()
        env.jaw_block_ratio = 0.45
        res = ex(SkillCall("pick", {"target_xy": list(self.TARGET)}))
        self.assertTrue(res.ok, res.message)
        descent = [c for c in ex.runner.calls
                   if np.allclose(np.asarray(c["pos"], float)[:2], list(self.TARGET), atol=1e-9)
                   and float(np.asarray(c["pos"], float)[2]) <= TABLE_HEIGHT + PICK_DEFAULT_HALF[2] + 1e-9]
        self.assertTrue(descent)
        self.assertAlmostEqual(float(descent[-1]["pos"][2]),
                               TABLE_HEIGHT + PICK_DEFAULT_HALF[2], places=6)

    def test_tall_hint_without_target_id_enters_slender_branch(self):
        """lead 要求：无 target_id、只给 target_xy / target_z_hint(高出桌面>=80mm) / 2 元 target_size
        -> 必须进细高分支（限速接触下降 + 方案 B 高度 + 受控合到接触）。"""
        env, ex = self._setup()
        env.jaw_block_ratio = 0.78          # = 模型 2 元 size 反算出的指令开合（被瓶身挡住）
        env.close_force = 20.0
        env.track_moves = True
        ex.runner = _StubRunner(env=env)
        res = ex(SkillCall("pick", {"target_xy": [0.45, -0.18], "target_z_hint": 0.5218,
                                    "target_size": [0.0402, 0.0402],
                                    "grasp_profile": "top"}))
        # lead 修正语义：z_hint 是**顶面** -> 夹持高度 = hint - 0.03 = 0.4918
        self.assertAlmostEqual(float(res.metrics["grasp_z_effective"]), 0.4918, places=4)
        calls = [c for c in ex.runner.calls if c["compliance"] is PLACE_PICK_DESCEND_COMPLIANCE]
        self.assertGreaterEqual(len(calls), 2, "没走细高分支的两段式接触下降")
        self.assertAlmostEqual(float(calls[0]["v_max"]), PICK_CRUISE_VMAX_SLENDER, places=9)
        self.assertAlmostEqual(float(calls[0]["force_limit"]), PICK_SLENDER_FORCE_LIMIT_N, places=9)
        self.assertAlmostEqual(float(calls[1]["v_max"]), CONTACT_VMAX, places=9)
        self.assertTrue(res.ok, res.message)
        self.assertTrue(res.metrics["grasp_positive_evidence"])

    def test_target_xy_and_target_id_are_exclusive(self):
        env, ex = self._setup()
        res = ex(SkillCall("pick", {"target_id": "box_blue", "target_xy": [0.35, -0.14]}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "bad_args")

    def test_stuck_open_jaws_are_not_reported_as_grasped(self):
        """反例：夹爪卡在全开（合爪没生效）时，不能因为"间距大"就判抓住。"""
        env, ex = self._setup()
        env.set_gripper = lambda r: None      # 合爪指令无效，get_gripper 恒为 1.0
        res = ex(SkillCall("pick", {"target_xy": list(self.TARGET)}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "grasp_failed")
        self.assertEqual(res.metrics.get("grasp_evidence"), "jaw_gap")
        self.assertGreater(res.metrics["grasp_gap_mm"], GRIPPER_GAP_OPEN_MM - 1.0)
        self.assertIn("指间距", " ".join(res.metrics.get("reasons", [])))

    def test_empty_table_is_not_reported_as_grasped(self):
        """关键反例：目标点空无一物时，夹爪会走到指令开合度——这不算抓住。"""
        env, ex = self._setup(obj_pose=(0.60, -0.30, TABLE_HEIGHT + 0.02))
        env.jaw_block_ratio = 0.0                  # 空手，能合到指令开合度
        res = ex(SkillCall("pick", {"target_xy": [0.35, 0.14]}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "grasp_failed")
        self.assertEqual(res.metrics.get("grasp_evidence"), "jaw_gap")
        self.assertLessEqual(res.metrics["grasp_gap_mm"],
                             res.metrics["grasp_commanded_gap_mm"] + GRASP_JAW_GAP_MIN_MM)

    def test_target_id_path_unchanged(self):
        """旧路径回归：target_id 成功时仍按"抬起 > 20 mm"判据、holding=id、不报 jaw_gap。"""
        state = {"n": 0}

        def on_move(_r):
            state["n"] += 1
            if state["n"] == 3:               # 第 3 次 = 合爪后的抬升
                env.move_object("box_blue", [0.0, 0.0, 0.12])

        env, ex = self._setup(runner=_StubRunner(on_move=on_move))
        res = ex(SkillCall("pick", {"target_id": "box_blue"}))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(ex.holding, "box_blue")
        self.assertGreaterEqual(res.metrics["moved"], 0.02)
        self.assertIsNone(res.metrics.get("grasp_evidence"))
        self.assertNotIn("grasp_gap_mm", res.metrics)

    def test_obstruction_contacts_only_gives_none_without_contacts(self):
        """R-c：无真值模式的受阻物只用接触对反查；拿不到（stub 无 contact 数据）给 None。"""
        env, ex = self._setup()
        obs, note = ex._obstruction(contacts_only=True)
        self.assertIsNone(obs["obstruction_id"])
        self.assertEqual(note, "")


# --------------------------------------------------------------------------- #
# 17) task-59：grasp_profile（auto/top/side）+ close_ratio 类别先验
# --------------------------------------------------------------------------- #
class _SlenderPhase2Fail(_StubRunner):
    """只在"最后 10mm"那一步返回 force_limit（模拟接触中止），巡航段正常。"""

    def __init__(self, phase2_z, **kw):
        super().__init__(**kw)
        self.phase2_z = float(phase2_z)

    def move_to_pose(self, pos, quat=None, duration=None, compliance=None,
                     timeout=8.0, force_limit=None, v_max=None):
        r = super().move_to_pose(pos, quat, duration=duration, compliance=compliance,
                                 timeout=timeout, force_limit=force_limit, v_max=v_max)
        p = np.asarray(pos, float)
        if abs(float(p[2]) - self.phase2_z) < 1e-9:
            return MoveResult(False, "stub force_limit", "force_limit", max_force=3.2,
                              final_pos=p.copy())
        return r


class _PickyKin(_StubKin):
    """只让"沿 +x 后退到 x≈0.30"的预抓取位不可解，用来验证 side 会自动换方向。"""

    def ik(self, target_pos, target_quat=None, q0=None, max_iters=300, tol=1e-3):
        p = np.asarray(target_pos, float).reshape(-1)[:3]
        if abs(float(p[0]) - 0.30) < 1e-6:
            return np.zeros(6, dtype=float), False
        return np.zeros(6, dtype=float), True


class TestGraspProfile(unittest.TestCase):
    def _setup(self, obj_pose=(0.45, -0.18, TABLE_HEIGHT + 0.035), block=0.7):
        obj = _make_obj("cyl", list(obj_pose), (0.032, 0.032, 0.10),
                        category="cylinder", color="yellow")
        env = _StubEnv([obj])
        env.jaw_block_ratio = block
        ex = _executor(env, _StubRunner(), _StubPlanner(pose_path=None))
        return env, ex

    def test_approach_quat_dir_top_matches_legacy(self):
        """approach_quat_dir([0,0,-1], roll) 必须与既有 approach_quat(roll) 等价。"""
        for yaw in (0.0, 0.3, 1.5707963267948966):
            np.testing.assert_allclose(approach_quat_dir((0.0, 0.0, -1.0), yaw),
                                       approach_quat(yaw), atol=1e-12)

    def test_approach_quat_dir_side_points_plus_x(self):
        q = approach_quat_dir((1.0, 0.0, 0.0))
        out = np.zeros(3)
        mujoco.mju_rotVecQuat(out, np.array([0.0, 0.0, 1.0]), q)
        np.testing.assert_allclose(out, [1.0, 0.0, 0.0], atol=1e-9)

    def test_side_profile_approaches_horizontally(self):
        env, ex = self._setup()
        res = ex(SkillCall("pick", {"target_xy": [0.45, -0.18], "target_z_hint": 0.42,
                                    "target_size": [0.032, 0.032, 0.10],
                                    "grasp_profile": "side"}))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics["grasp_profile"], "side")
        staging = ex.runner.calls[0]             # 先在预抓取位正上方落脚（避开贴桌面横扫）
        np.testing.assert_allclose(staging["pos"], [0.45 - 0.10, -0.18, 0.42 + 0.10], atol=1e-9)
        pre = ex.runner.calls[1]                 # 再竖直到预抓取高度
        np.testing.assert_allclose(pre["pos"], [0.45 - 0.10, -0.18, 0.42], atol=1e-9)
        first = ex.runner.calls[2]               # 第 1 步：同一高度、x 方向推进
        self.assertAlmostEqual(float(first["pos"][2]), 0.42, places=9)
        self.assertGreater(float(first["pos"][0]), 0.35)

    def test_side_direction_auto_switches_when_plus_x_unreachable(self):
        """实测 +x 对近处物体预抓取位不可达 -> 必须自动换成 ±y 等可解方向。"""
        obj = _make_obj("cyl", [0.40, 0.02, TABLE_HEIGHT + 0.035], (0.032, 0.032, 0.10),
                        category="cylinder", color="yellow")
        env = _StubEnv([obj])
        env.jaw_block_ratio = 0.7
        ex = _executor(env, _StubRunner(), _StubPlanner(pose_path=None), kin=_PickyKin())
        res = ex(SkillCall("pick", {"target_xy": [0.40, 0.02], "target_z_hint": 0.435,
                                    "target_size": [0.032, 0.032, 0.10],
                                    "grasp_profile": "side"}))
        self.assertTrue(res.ok, res.message)
        staging = ex.runner.calls[0]["pos"]      # 换成 +y：先到预抓取位正上方
        pre = ex.runner.calls[1]["pos"]
        self.assertAlmostEqual(float(pre[0]), 0.40, places=9)
        self.assertAlmostEqual(float(pre[1]), 0.02 - 0.10, places=9)
        self.assertAlmostEqual(float(pre[2]), 0.435, places=9)
        self.assertAlmostEqual(float(staging[2]), 0.435 + 0.10, places=9)

    def test_side_without_size_uses_cylinder_prior(self):
        env, ex = self._setup(block=0.65)
        res = ex(SkillCall("pick", {"target_xy": [0.45, -0.18], "target_z_hint": 0.42,
                                    "grasp_profile": "side"}))
        self.assertTrue(res.ok, res.message)
        # 圆柱先验 0.55 -> 指令指间距 = 8.4 + 0.55*(93-8.4)
        self.assertAlmostEqual(res.metrics["grasp_commanded_gap_mm"],
                               GRIPPER_GAP_CLOSED_MM + GRASP_CLOSE_PRIOR["cylinder"]
                               * (GRIPPER_GAP_OPEN_MM - GRIPPER_GAP_CLOSED_MM), places=1)

    def test_bottle_prior(self):
        env, ex = self._setup(block=0.85)
        res = ex(SkillCall("pick", {"target_xy": [0.45, -0.18], "target_z_hint": 0.42,
                                    "target_category": "bottle",
                                    "grasp_profile": "side"}))
        self.assertTrue(res.ok, res.message)
        self.assertAlmostEqual(res.metrics["grasp_commanded_gap_mm"],
                               GRIPPER_GAP_CLOSED_MM + GRASP_CLOSE_PRIOR["bottle"]
                               * (GRIPPER_GAP_OPEN_MM - GRIPPER_GAP_CLOSED_MM), places=1)

    def test_box_prior_is_default_for_top(self):
        env, ex = self._setup(block=0.45)
        res = ex(SkillCall("pick", {"target_xy": [0.45, -0.18], "target_z_hint": 0.42}))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics["grasp_profile"], "auto")
        self.assertAlmostEqual(res.metrics["grasp_commanded_gap_mm"],
                               GRIPPER_GAP_CLOSED_MM + GRASP_CLOSE_PRIOR["box"]
                               * (GRIPPER_GAP_OPEN_MM - GRIPPER_GAP_CLOSED_MM), places=1)

    def test_explicit_close_ratio_wins(self):
        env, ex = self._setup(block=0.45)
        res = ex(SkillCall("pick", {"target_xy": [0.45, -0.18], "target_z_hint": 0.42,
                                    "close_ratio": 0.30}))
        self.assertAlmostEqual(res.metrics["grasp_commanded_gap_mm"],
                               GRIPPER_GAP_CLOSED_MM + 0.30
                               * (GRIPPER_GAP_OPEN_MM - GRIPPER_GAP_CLOSED_MM), places=1)

    def test_side_falls_back_to_top_without_clearance(self):
        """lead 裁决：side 需要 >=10cm 净空；净空不足时 oracle 路径自动退回 top（不推物体）。"""
        tgt = _make_obj("cyl", [0.40, 0.02, TABLE_HEIGHT + 0.035], (0.032, 0.032, 0.10),
                        category="cylinder", color="yellow")
        near = _make_obj("box_red", [0.40, -0.02, TABLE_HEIGHT + 0.02], (0.02, 0.02, 0.02))
        env = _StubEnv([tgt, near])
        runner = _StubRunner()
        ex = _executor(env, runner, _StubPlanner(pose_path=None))
        res = ex(SkillCall("pick", {"target_id": "cyl", "grasp_profile": "side",
                                    "target_z_hint": 0.435}))
        self.assertEqual(res.metrics.get("grasp_profile"), "side")     # 请求仍是 side
        self.assertIn("退回 top", str(res.metrics.get("side_fallback_reason")))
        self.assertEqual(res.metrics.get("grasp_profile_effective"), "top")
        # 退回 top 后：预抓取位在目标正上方（xy 与目标一致、z + approach）
        np.testing.assert_allclose(runner.calls[0]["pos"][:2], [0.40, 0.02], atol=1e-9)
        self.assertAlmostEqual(float(runner.calls[0]["pos"][2]), 0.435 + 0.10, places=9)

    def test_side_advance_uses_soft_axis_and_watchdog(self):
        """2910.3N 安全红线：side 的水平推进必须用 6 轴软档 + 显式 6N 看门狗（z_only 会硬顶）。"""
        env, ex = self._setup()
        res = ex(SkillCall("pick", {"target_xy": [0.45, -0.18], "target_z_hint": 0.42,
                                    "target_size": [0.032, 0.032, 0.10],
                                    "grasp_profile": "side"}))
        self.assertTrue(res.ok, res.message)
        adv = [c for c in ex.runner.calls if c["compliance"] is PLACE_SIDE_SOFT_COMPLIANCE]
        self.assertTrue(adv, "side 推进段没有用软档")
        for c in adv:
            self.assertEqual(int(np.count_nonzero(c["compliance"].axes)), 6)
            self.assertAlmostEqual(float(c["force_limit"]), PLACE_SIDE_STOP_FORCE_N, places=9)
            self.assertAlmostEqual(float(c["compliance"].force_limit),
                                   PLACE_SIDE_STOP_FORCE_N, places=9)

    def test_side_advance_blocked_falls_back_to_top(self):
        """无真值模式：side 推进被挡 -> 自动退回 top-down 重试（不再直接失败）。"""
        env, ex = self._setup(block=0.75)      # 指间被卡在 71.9mm（> 指令 58mm + 4mm）
        ex.runner = _StubRunner(fail_when=lambda c: c is PLACE_SIDE_SOFT_COMPLIANCE,
                                error_code="force_limit", max_force=0.0)
        res = ex(SkillCall("pick", {"target_xy": [0.45, -0.18], "target_z_hint": 0.42,
                                    "target_size": [0.032, 0.032, 0.10],
                                    "grasp_profile": "side"}))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics.get("grasp_profile_effective"), "top")
        self.assertIn("退回 top", str(res.metrics.get("side_fallback_reason")))

    def test_side_below_floor_is_refused_before_advance(self):
        """预抓取高度离桌面不足 -> 直接拒绝进入侧向推进（根本不发 side 推进指令）。"""
        env, ex = self._setup(block=0.75)
        low = TABLE_HEIGHT + 0.005
        res = ex(SkillCall("pick", {"target_xy": [0.45, -0.18], "target_z_hint": low,
                                    "target_size": [0.032, 0.032, 0.10],
                                    "grasp_profile": "side"}))
        self.assertTrue(res.ok, res.message)
        self.assertIn("距桌面", str(res.metrics.get("side_fallback_reason")))
        self.assertIn("退回 top-down", str(res.metrics.get("side_fallback_reason")))
        self.assertEqual([c for c in ex.runner.calls
                          if c["compliance"] is PLACE_SIDE_SOFT_COMPLIANCE], [])

    def test_cylinder_grasp_height_clamped_and_uses_hint(self):
        """圆柱/水瓶默认抓上半身：高度用 target_z_hint，且不低于桌面 + PLACE_MIN_GRASP_HEIGHT_M。"""
        env, ex = self._setup(block=0.6)
        low = TABLE_HEIGHT + 0.005
        # 用**矮**圆柱（半高 35mm，非细高）来验"不低于桌面+25mm"这条下限；细高物体走方案 B
        res = ex(SkillCall("pick", {"target_xy": [0.45, -0.18], "target_z_hint": low,
                                    "target_category": "cylinder",
                                    "target_size": [0.015, 0.015, 0.035]}))
        self.assertTrue(res.ok, res.message)
        # 统一口径后：矮圆柱走"顶面-30mm"的下限保护 = TABLE_HEIGHT + PLACE_SLENDER_FLOOR_MIN_M
        self.assertAlmostEqual(float(ex.runner.calls[0]["pos"][2]),
                               TABLE_HEIGHT + PLACE_SLENDER_FLOOR_MIN_M + 0.10, places=9)

    def test_slender_grip_positive_evidence_passes(self):
        """细高物体：gap 贴近直径（64mm，实测≈59）-> 正面证据通过。"""
        env, ex = self._setup(block=0.60)
        env.close_force = 20.0              # 合爪被瓶身挡住 -> 有接触力（正面证据）
        res = ex(SkillCall("pick", {"target_xy": [0.45, -0.18], "target_z_hint": 0.50,
                                    "target_category": "bottle",
                                    "target_size": [0.032, 0.032, 0.10]}))
        self.assertTrue(res.ok, res.message)
        self.assertTrue(res.metrics["grasp_positive_evidence"])
        self.assertLessEqual(float(res.metrics["grasp_gap_vs_size_mm"]), GRASP_SIZE_GAP_TOL_MM)

    def test_slender_grip_positive_evidence_gate(self):
        """lead 要求：细高物体只夹到空气/掌部顶住（gap=84.6 vs 直径 64）必须判失败。"""
        env, ex = self._setup(block=0.90)
        res = ex(SkillCall("pick", {"target_xy": [0.45, -0.18], "target_z_hint": 0.50,
                                    "target_category": "bottle",
                                    "target_size": [0.032, 0.032, 0.10]}))
        self.assertFalse(res.ok, "只夹到空气却判成功了")
        self.assertFalse(bool(res.metrics.get("grasp_positive_evidence")))
        self.assertGreater(float(res.metrics["grasp_gap_vs_size_mm"]), GRASP_SIZE_GAP_TOL_MM)

    def test_unsupported_profile_rejected(self):
        env, ex = self._setup()
        res = ex(SkillCall("pick", {"target_xy": [0.45, -0.18], "grasp_profile": "diagonal"}))
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "bad_args")

    def test_oracle_top_path_unchanged(self):
        """oracle 路径（target_id）默认仍是 top-down：预抓取在正上方、下降沿 -z。"""
        obj = _make_obj("box_blue", [0.35, 0.14, TABLE_HEIGHT + 0.02], (0.02, 0.02, 0.02))
        env = _StubEnv([obj])
        runner = _StubRunner()
        runner.on_move = lambda r: (env.move_object("box_blue", [0.0, 0.0, 0.12])
                                    if len(r.calls) == 3 else None)
        ex = _executor(env, runner, _StubPlanner(pose_path=None))
        res = ex(SkillCall("pick", {"target_id": "box_blue"}))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics["grasp_profile"], "auto")
        np.testing.assert_allclose(runner.calls[0]["pos"],
                                   [0.35, 0.14, TABLE_HEIGHT + 0.02 + 0.10], atol=1e-9)
        self.assertNotIn("grasp_evidence", res.metrics)


# --------------------------------------------------------------------------- #
# 18) P0：place 入口清场不得把"手里抓着的东西"当障碍物
# --------------------------------------------------------------------------- #
class TestPlaceEntryHeldGuard(unittest.TestCase):
    def _setup(self, grip=0.45):
        # 手里那个物体：位置就是 EE 当前所在处（距离判据下算"与 EE 接触"）
        held = _make_obj("box_red", [0.30, -0.10, TABLE_HEIGHT + 0.02], (0.02, 0.02, 0.02))
        env = _StubEnv([held], gripper=grip)
        env.tcp_pos = np.array([0.30, -0.10, TABLE_HEIGHT + 0.02])
        runner = _StubRunner(env=env)
        ex = _executor(env, runner, _StubPlanner(pose_path=None))
        ex.holding = HOLDING_UNNAMED          # 无真值抓取后的占位（不知道 id）
        return env, runner, ex

    @staticmethod
    def _retreat_calls(runner):
        """撤退档 = 6 轴全使能的 RETREAT_COMPLIANCE（放置/接近用的都是 z_only）。"""
        return [c for c in runner.calls
                if c["compliance"] is not None
                and int(np.count_nonzero(c["compliance"].axes)) == 6]

    def test_holding_true_skips_entry_clear_and_places(self):
        env, runner, ex = self._setup()
        res = ex(SkillCall("place", {"surface": "table", "xy": [0.30, -0.10],
                                     "holding": True}))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics.get("entry_clear_skipped"), "holding_arg")
        self.assertEqual(self._retreat_calls(runner), [])      # 没有发生撤退
        self.assertIsNone(ex.holding)                          # 正常释放

    def test_single_touching_object_with_mid_gap_skips_without_arg(self):
        """兜底判据：只有一个物体接触 + 夹爪卡在中位 -> 认定是手持物，同样跳过清场。"""
        env, runner, ex = self._setup(grip=0.45)               # gap ≈ 46 mm（中位）
        res = ex(SkillCall("place", {"surface": "table", "xy": [0.30, -0.10]}))
        self.assertTrue(res.ok, res.message)
        self.assertEqual(res.metrics.get("entry_clear_skipped"), "held_heuristic")
        self.assertEqual(self._retreat_calls(runner), [])

    def test_full_closed_gripper_keeps_old_behavior(self):
        """holding=None 且夹爪全闭（非中位）-> 旧行为不变：仍按障碍物清场 -> stuck。"""
        env, runner, ex = self._setup(grip=0.0)                # gap ≈ 8.4 mm（全闭）
        res = ex(SkillCall("place", {"surface": "table", "xy": [0.30, -0.10]}))
        self.assertFalse(res.ok)
        # 旧行为：入口仍按"障碍物"处理（清场 -> 失败）；失败码按既有语义（stuck / scene_reset）
        self.assertIn(res.error_code, ("stuck", "scene_reset"))
        self.assertIsNone(res.metrics.get("entry_clear_skipped"))
        # 入口温和撤退至少发生一次（之后终态清场可能再来一次）
        self.assertGreaterEqual(len(self._retreat_calls(runner)), 1)

    def test_two_touching_objects_do_not_trigger_heuristic(self):
        """兜底判据要求"恰好一个"接触物体；多一个就不算手持物。"""
        env, runner, ex = self._setup(grip=0.45)
        env._objects["box_blue"] = _make_obj("box_blue", [0.32, -0.10, TABLE_HEIGHT + 0.02],
                                             (0.02, 0.02, 0.02), color="blue")
        res = ex(SkillCall("place", {"surface": "table", "xy": [0.30, -0.10]}))
        self.assertFalse(res.ok)
        self.assertIn(res.error_code, ("stuck", "scene_reset"))
        self.assertIsNone(res.metrics.get("entry_clear_skipped"))

    def test_oracle_path_unchanged(self):
        """oracle（holding=已知 id）：由 exclude_held 排除，本来就不触发清场，语义不变。"""
        held = _make_obj("box_red", [0.30, -0.10, TABLE_HEIGHT + 0.02], (0.02, 0.02, 0.02))
        env = _StubEnv([held], gripper=0.45)
        env.tcp_pos = np.array([0.30, -0.10, TABLE_HEIGHT + 0.02])
        runner = _StubRunner(env=env)
        ex = _executor(env, runner, _StubPlanner(pose_path=None))
        ex.holding = "box_red"
        res = ex(SkillCall("place", {"surface": "table", "xy": [0.30, -0.10]}))
        self.assertTrue(res.ok, res.message)
        self.assertIsNone(res.metrics.get("entry_clear_skipped"))   # 走的不是新分支
        self.assertEqual(self._retreat_calls(runner), [])


# --------------------------------------------------------------------------- #
# 19) F3（verifier 独立发现）：side 回退 top-down 不得抛 UnboundLocalError，撤退力必须被 45N 压住
# --------------------------------------------------------------------------- #
class TestSideFallbackSafety(unittest.TestCase):
    def test_side_fallback_never_raises_and_retreat_capped(self):
        obj = _make_obj("box_blue", [0.350, 0.136, TABLE_HEIGHT + 0.02],
                        (0.0201, 0.0216, 0.02), color="blue")
        env = _StubEnv([obj])
        env.jaw_block_ratio = 0.45
        # 所有"接近段"（STIFF，axes 全 0）都被挡 -> side 回退 top-down 后也失败 ->
        # 每次迭代都在下降段之前 continue -> 复刻 F3（循环结束后引用未绑定的 descend_metrics）
        runner = _StubRunner(fail_when=lambda c: (c is not None
                                                   and int(np.count_nonzero(c.axes)) == 0),
                             error_code="force_limit", max_force=36.4)
        ex = _executor(env, runner, _StubPlanner(pose_path=None))
        res = ex(SkillCall("pick", {"target_xy": [0.350, 0.136], "target_z_hint": 0.4567,
                                    "target_size": [0.0201, 0.0216],
                                    "grasp_profile": "side"}))
        self.assertNotEqual(res.error_code, "exception", res.message)
        self.assertNotIn("UnboundLocalError", res.message)
        self.assertIn(res.error_code, ("grasp_failed", "object_disturbed", "stuck", "stuck"))
        # 每一次撤退的力上限都 <= RETREAT_FORCE_CAP_N（45N）
        retreats = [c for c in runner.calls if c["compliance"] is not None
                     and int(np.count_nonzero(c["compliance"].axes)) == 6]
        self.assertTrue(retreats, "没有发生撤退？")
        for c in retreats:
            self.assertLessEqual(float(c["force_limit"]), RETREAT_FORCE_CAP_N + 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
