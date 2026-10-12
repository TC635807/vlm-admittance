"""tests/test_place_semantics.py —— 桌面放置 / 指代纠正 / 卡死逃逸 的独立复验（task-27）。

覆盖：
  1) 用户两条原命令端到端（放置 surface=table、落点在桌面、第二条不误抓 box_red、
     scene_reset=0、结束 |F|<=5N）。
  2) 放置成功判据：伪成功（物体摞在盘内另一物体上、底面高出盘口 26mm）必须
     返回 place_mismatch；正常落进盘里 / 桌面落座仍判 ok=True。
  3) 撤退方向符号：压入桌面时 env.contact_force_on_ee() 为 +z（环境往上推）；
     沿 +f 撤退 |F|->0，沿 -f 顶进 |F| 大幅上升。
  4) 跨命令 history：传递/配对/裁剪(<=40)/:clear-history。
  5) 卡死压力：3 次独立会话「夹住黄色圆柱体」，peak<500N、结束 |F|<=5N、scene_reset 计数。
  6) 证伪：桌面空位全占用时优雅退化；:target off / 非法 :target 仍安全。

运行：
    cd /home/tc63/mujoco/vlm_admittance
    MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m tests.test_place_semantics
"""
from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import TABLE_HEIGHT                              # noqa: E402
from core.types import (ContactState, MoveResult, RobotState,     # noqa: E402
                        SceneObject, SceneState, SkillCall, SkillResult)
from agent.loop import AgentLoop                                  # noqa: E402
from agent.state import build_messages, to_jsonable               # noqa: E402
from agent.vlm_client import MockVLM                              # noqa: E402
from skills import approach_quat                                  # noqa: E402
from skills.library import (PLACE_SEAT_MAX_M, PLACE_SEAT_MIN_M,   # noqa: E402
                            PLACE_XY_TOL_M, SkillExecutor)


def _make_app(seed: int = 0):
    from demo.interactive import InteractiveApp
    args = argparse.Namespace(vlm="mock", with_barrier=False, seed=seed, realtime=False,
                              headless=True, with_images=False, max_steps=12, self_test=False)
    return InteractiveApp(args)


# --------------------------------------------------------------------------- #
# 无 GL 的 stub 环境（自建，不复用实现者的 smoke stub）
# --------------------------------------------------------------------------- #
def _obj(i, c, col, pos, size, graspable=True, is_container=False):
    return SceneObject(i, c, col, np.array([pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0]),
                       tuple(size), graspable, is_container)


class _Env:
    dt = 0.002

    def __init__(self, objs):
        self.o = {x.id: x for x in objs}
        self.force = 0.0
        self.grip = 1.0

    def list_objects(self):
        return list(self.o.values())

    def object_pose(self, i):
        return self.o[str(i)].pose.copy()

    def set_object_pose(self, i, p):
        self.o[str(i)].pose = np.asarray(p, float).copy()

    def site_pose(self, name):
        return np.array([0.3, 0.0, 0.5]), np.array([1.0, 0.0, 0.0, 0.0])

    def contact_force_on_ee(self):
        return np.array([0.0, 0.0, self.force]), np.zeros(3)

    def set_gripper(self, r):
        self.grip = float(r)

    def get_gripper(self):
        return float(self.grip)

    def step(self, n=1):
        return None

    def reset(self, seed=0):
        return None

    def home(self):
        return None


class _Runner:
    """把被夹物体搬到 final 位姿（模拟释放后的落点）。"""

    def __init__(self, env):
        self.env = env
        self.final = None
        self.held = None

    def move_to_pose(self, pos, quat=None, duration=None, compliance=None,
                     timeout=8.0, force_limit=None):
        if self.final is not None and self.held is not None:
            p = self.env.object_pose(self.held).copy()
            p[0], p[1], p[2] = self.final
            self.env.set_object_pose(self.held, p)
        return MoveResult(True, "stub ok")

    def follow_joint_path(self, *a, **k):
        return MoveResult(True, "stub path")


class _Planner:
    def plan_pose(self, *a, **k):
        return None

    def plan_joint(self, *a, **k):
        return None


class _Kin:
    def __init__(self, ok=True):
        self.ok = ok

    def ik(self, *a, **k):
        return np.zeros(6), self.ok

    def fk(self, q):
        """给 _grasp_rehearsal 用的正解：返回一个稳定位姿与单位四元数。"""
        return np.array([0.40, 0.0, 0.43]), np.array([1.0, 0.0, 0.0, 0.0])


def _place(env, runner, kin, skill_args, held="box_red"):
    ex = SkillExecutor(env, kin, runner, _Planner(), log=lambda *a, **k: None)
    ex.holding = held
    runner.held = held
    return ex(SkillCall("place", skill_args))


# --------------------------------------------------------------------------- #
# 2) 放置成功判据
# --------------------------------------------------------------------------- #
class TestPlaceCriterion(unittest.TestCase):
    def _scene(self):
        tray = _obj("tray", "tray", "gray", (0.4, 0.3, 0.4119), (0.09, 0.07, 0.012), False, True)
        held = _obj("box_red", "box", "red", (0.4, 0.3, 0.5), (0.02, 0.02, 0.02))
        return _Env([tray, held]), tray

    def _run_tray(self, bottom_z):
        env, tray = self._scene()
        runner = _Runner(env)
        support = float(tray.pose[2] + tray.size[2])
        runner.final = (float(tray.pose[0]), float(tray.pose[1]), float(bottom_z) + 0.02)
        r = _place(env, runner, _Kin(), {"container_id": "tray"})
        return r, support

    def test_normal_tray_place_ok(self):
        r, support = self._run_tray(TABLE_HEIGHT + 0.001)
        print(f"\n[probe] 托盘正常落座: bottom={r.metrics.get('bottom_z')} support={support:.4f} "
              f"gap={r.metrics.get('bottom_gap')} ok={r.ok}")
        self.assertTrue(r.ok, f"正常托盘放置被判失败: {r.error_code} {r.message}")
        self.assertIsNone(r.error_code)

    def test_stacked_26mm_rejected_as_place_mismatch(self):
        """核心：摞在盘内另一物体上、底面高出盘口 26mm -> 必须 place_mismatch。"""
        r, support = self._run_tray((0.4119 + 0.012) + 0.026)
        print(f"[probe] 伪成功 +26mm: bottom={r.metrics.get('bottom_z')} "
              f"gap={r.metrics.get('bottom_gap')} dz={r.metrics.get('dz')} -> ok={r.ok} err={r.error_code}")
        self.assertFalse(r.ok, f"伪成功未被拒绝: {r.message}")
        self.assertEqual(r.error_code, "place_mismatch")
        self.assertGreater(float(r.metrics["bottom_gap"]), PLACE_SEAT_MAX_M)

    def test_borderline_15mm_above_rim_still_ok(self):
        r, _ = self._run_tray((0.4119 + 0.012) + 0.015)
        self.assertTrue(r.ok, f"容差内(15mm)应判成功: {r.error_code}")

    def test_table_seat_and_stacked_rejected(self):
        env, tray = self._scene()
        runner = _Runner(env)
        # 桌面落座
        runner.final = (0.30, -0.30, TABLE_HEIGHT + 0.02)
        r_ok = _place(env, runner, _Kin(), {"surface": "table", "xy": [0.30, -0.30]})
        # 摞在桌面已有物体上：底面高出桌面 26mm
        env2, tray2 = self._scene()
        runner2 = _Runner(env2)
        runner2.final = (0.30, -0.30, TABLE_HEIGHT + 0.026 + 0.02)
        r_bad = _place(env2, runner2, _Kin(), {"surface": "table", "xy": [0.30, -0.30]})
        print(f"[probe] 桌面: 落座 ok={r_ok.ok} gap={r_ok.metrics.get('bottom_gap')} ; "
              f"高出 26mm ok={r_bad.ok} err={r_bad.error_code} gap={r_bad.metrics.get('bottom_gap')}")
        self.assertTrue(r_ok.ok, f"桌面正常落座被判失败: {r_ok.error_code} {r_ok.message}")
        self.assertFalse(r_bad.ok)
        self.assertEqual(r_bad.error_code, "place_mismatch")

    def test_xy_too_far_rejected(self):
        env, tray = self._scene()
        runner = _Runner(env)
        runner.final = (float(tray.pose[0]) + 0.09, float(tray.pose[1]), TABLE_HEIGHT + 0.021)
        r = _place(env, runner, _Kin(), {"container_id": "tray"})
        print(f"[probe] xy 超差: dxy={r.metrics.get('dxy_target')} ok={r.ok} err={r.error_code}")
        self.assertFalse(r.ok)
        self.assertGreater(float(r.metrics["dxy_target"]), PLACE_XY_TOL_M)


# --------------------------------------------------------------------------- #
# 1b) place 参数路由（纯 MockVLM）
# --------------------------------------------------------------------------- #
def _fake_scene():
    objs = [
        _obj("box_red", "box", "red", (0.35, -0.14, 0.42), (0.02, 0.02, 0.02)),
        _obj("box_blue", "box", "blue", (0.35, 0.14, 0.42), (0.02, 0.02, 0.02)),
        _obj("can_yellow", "cylinder", "yellow", (0.4, 0.02, 0.435), (0.015, 0.015, 0.035)),
        _obj("tray", "tray", "gray", (0.4, 0.3, 0.412), (0.09, 0.07, 0.012), False, True),
    ]
    return SceneState(0.0, RobotState(np.zeros(6), np.zeros(6), np.zeros(3),
                                      np.array([1.0, 0, 0, 0])),
                      objs, ContactState(np.zeros(3), np.zeros(3), 0.0, False))


def _scene_objs():
    """与真实场景一致的对象列表（_fake_scene 少了 box_green，这里补上）。"""
    objs = list(_fake_scene().objects)
    if not any(o.id == "box_green" for o in objs):
        objs.insert(2, _obj("box_green", "box", "green", (0.45, -0.05, 0.42), (0.02, 0.02, 0.02)))
    return [o.to_json() for o in objs]


class _RecExec:
    """只记录被执行的 skill 序列（用于断言「不得先 pick」）。"""

    def __init__(self):
        self.calls: list = []

    def __call__(self, call):
        self.calls.append(call.skill)
        return SkillResult(True, call.skill, "ok")


def _obs(skill, ok=True, error_code=None):
    r = SkillResult(ok, skill, "stub", error_code)
    return [{"role": "assistant", "content": json.dumps({"skill": skill, "args": {}})},
            {"role": "user", "content": "OBSERVATION_JSON "
             + json.dumps(to_jsonable(r.to_observation()))}]


def _decide(task, history):
    return MockVLM().decide(build_messages(task, _fake_scene(), {}, history=history))


class TestPlaceRouting(unittest.TestCase):
    def test_table_surface_not_tray(self):
        hist = _obs("pick", True)
        call = _decide("把蓝色方块 放回到桌子上", hist)
        print(f"\n[probe] 「放回到桌子上」 -> {call.skill} {call.args}")
        self.assertEqual(call.skill, "place")
        self.assertEqual(call.args.get("surface"), "table")
        self.assertNotIn("container_id", call.args, "桌面任务不得带 container_id")

    def test_explicit_tray_still_container(self):
        call = _decide("把红色的方块放进托盘", _obs("pick", True))
        self.assertEqual(call.skill, "place")
        self.assertEqual(call.args.get("container_id"), "tray")
        self.assertNotIn("surface", call.args)

    @staticmethod
    def _real_history():
        """真实格式历史：assistant 的 args 带 target_id / surface。"""
        def o(skill, ok, args, ec=None):
            call = {"skill": skill, "args": args, "compliance": None, "reasoning": ""}
            r = SkillResult(ok, skill, "x", ec)
            return [{"role": "assistant", "content": json.dumps(call, ensure_ascii=False)},
                    {"role": "user", "content": "OBSERVATION_JSON "
                     + json.dumps(to_jsonable(r.to_observation()), ensure_ascii=False)}]
        return o("pick", True, {"target_id": "box_blue"}) + o("place", True, {"surface": "table"})

    def test_correction_after_place_stops_when_already_on_table(self):
        """纠正指令（已在桌面）直接 stop —— 这是本轮认可的行为。"""
        call = _decide("放到桌子上不是盘子里", self._real_history())
        print(f"\n[probe] 纠正指令(已在桌面) -> {call.skill} {call.args}")
        self.assertEqual(call.skill, "stop")

    def test_fresh_task_after_place_is_not_skipped(self):
        """缺陷 D3 反例：上一条命令完成后，全新的抓放任务不得被历史 placed 短路。

        实测（revision 见报告）：'把蓝色方块 放回到桌子上' 成功后，紧接着
        '把红色方块放进托盘' 返回 stop，box_red 完全没有被碰。
        """
        call = _decide("把红色方块放进托盘", self._real_history())
        print(f"[probe] 新任务(历史里已有 place) -> {call.skill} args={call.args}")
        self.assertNotEqual(call.skill, "stop",
                            "新任务被历史 placed 短路，用户命令被静默忽略")

    def test_new_target_not_blocked_by_previous_failure(self):
        """缺陷 D2 反例：上一条命令抓『别的物体』失败，不得让新目标直接 ask_user。"""
        hist = (_obs("pick", False, error_code="grasp_failed")
                + _obs("pick", False, error_code="grasp_failed")
                + _obs("pick", False, error_code="grasp_failed"))
        call = _decide("夹住蓝色方块", hist)
        print(f"[probe] 新目标(历史里有 3 次别的物体失败) -> {call.skill} args={call.args}")
        self.assertNotEqual(call.skill, "ask_user",
                            "跨命令 pick 失败计数泄漏：新目标被无条件拒绝")


# --------------------------------------------------------------------------- #
# 4) history 机制
# --------------------------------------------------------------------------- #
class _StubLoopEnv:
    def get_state(self):
        return _fake_scene()

    def render(self, camera="overview"):
        raise RuntimeError("stub")


class TestHistoryMechanics(unittest.TestCase):
    def test_seed_and_pairing(self):
        seed = _obs("pick", True) + _obs("place", True)
        loop = AgentLoop(_StubLoopEnv(), executor=lambda c: SkillResult(True, c.skill),
                         vlm=MockVLM(), cameras=(), max_steps=3, log=None, history=seed)
        loop.run("夹住蓝色方块")
        roles = [m["role"] for m in loop.history]
        self.assertEqual(len(loop.history) % 2, 0, "assistant/user 必须成对")
        self.assertEqual(roles[0], "assistant")
        print(f"\n[probe] 播种 4 条 -> 运行后 {len(loop.history)} 条 roles={roles}")
        self.assertGreaterEqual(len(loop.history), len(seed))

    def test_trimming_keeps_pairs_and_cap(self):
        loop = AgentLoop(_StubLoopEnv(), executor=lambda c: SkillResult(True, c.skill),
                         vlm=MockVLM(), cameras=(), max_steps=3, log=None)
        msgs = []
        for i in range(60):
            msgs.append({"role": "assistant", "content": json.dumps({"skill": "pick", "i": i})})
            msgs.append({"role": "user", "content": "OBSERVATION_JSON " + json.dumps({"skill": "pick", "ok": True})})
        out = loop._trim_history(msgs)
        print(f"[probe] 裁剪 120 -> {len(out)} 条；成对={len(out) % 2 == 0}")
        self.assertLessEqual(len(out), 40)
        self.assertEqual(len(out) % 2, 0)
        for k in range(0, len(out), 2):
            self.assertEqual(out[k]["role"], "assistant")
            self.assertEqual(out[k + 1]["role"], "user")

    def test_app_history_transfer_and_clear(self):
        app = _make_app()
        app.run_task("把蓝色方块 放回到桌子上")
        self.assertGreaterEqual(len(app.history), 4, "cmd1 应留下若干步")
        self.assertEqual(len(app.history) % 2, 0, "assistant/user 必须成对")
        skills_seen = []
        for m in app.history:
            c = str(m.get("content", ""))
            if c.startswith("OBSERVATION_JSON"):
                skills_seen.append(json.loads(c[len("OBSERVATION_JSON "):]).get("skill"))
        self.assertIn("pick", skills_seen)
        self.assertIn("place", skills_seen)
        app.dispatch(":clear-history")
        self.assertEqual(app.history, [])
        # 历史清空后：不得卡死；若抓了别的物体，记录为缺陷（见报告）
        r = app.run_task("放到桌子上不是盘子里")
        f_end = float(np.linalg.norm(app.env.contact_force_on_ee()[0]))
        print(f"[probe] clear-history 后: {[(x.skill, x.error_code) for x in r]} 结束|F|={f_end:.2f}N")
        for x in r:
            if x.skill == "pick" and x.ok:
                print("[WARN] 历史清空后仍抓取了物体（无指代信息时的退化行为）")
        self.assertLessEqual(f_end, 5.0, "历史清空后出现卡死级接触力")

    def test_reset_keeps_history(self):
        app = _make_app()
        app.run_task("把蓝色方块 放回到桌子上")
        n = len(app.history)
        app.dispatch(":reset")
        self.assertEqual(len(app.history), n, ":reset 不应清空会话历史（应使用 :clear-history）")


# --------------------------------------------------------------------------- #
# 3) 撤退方向符号（真实 env，直接关节目标建立稳定接触）
# --------------------------------------------------------------------------- #
class TestEscapeDirection(unittest.TestCase):
    @staticmethod
    def _contact(env, kin, z=0.3995, steps=1500):
        q0 = env.data.qpos[env.arm_qpos_idx].copy()
        q_ik, ok = kin.ik(np.array([0.30, 0.0, z]), approach_quat(0.0), q0=q0)
        assert ok, "IK 失败"
        env.set_arm_joint_targets(q_ik)
        env.step(steps)
        return np.asarray(env.contact_force_on_ee()[0], float)

    def _nudge(self, sign, dz_mm, steps=900):
        from sim.env import MuJoCoArmEnv
        from control.kinematics import Kinematics
        env = MuJoCoArmEnv()
        kin = Kinematics(env)
        f0 = self._contact(env, kin)
        p, _ = env.site_pose("g_pinch")
        fh = f0 / max(float(np.linalg.norm(f0)), 1e-9)
        q, ok = kin.ik(np.asarray(p, float) + sign * fh * (dz_mm / 1000.0), approach_quat(0.0),
                       q0=env.data.qpos[env.arm_qpos_idx].copy())
        if not ok:
            return None
        env.set_arm_joint_targets(q)
        for _ in range(steps):
            env.step(1)
        return f0, np.asarray(env.contact_force_on_ee()[0], float)

    def test_contact_force_sign_is_upward(self):
        from sim.env import MuJoCoArmEnv
        from control.kinematics import Kinematics
        env = MuJoCoArmEnv()
        f = self._contact(env, Kinematics(env))
        print(f"\n[probe] 压入桌面接触力 f={np.round(f, 2)} |f|={np.linalg.norm(f):.2f}N")
        self.assertGreater(float(f[2]), 0.0, "环境作用在末端上的力应指向上（+z）")
        self.assertGreater(float(f[2]), abs(float(f[0])) + abs(float(f[1])) - 2.0,
                           "接触力应以 +z 为主")

    def test_retreat_along_force_reduces_and_against_increases(self):
        plus = self._nudge(+1.0, 20.0)
        minus = self._nudge(-1.0, 20.0)
        self.assertIsNotNone(plus)
        self.assertIsNotNone(minus)
        (f0p, f1p), (f0m, f1m) = plus, minus
        n0, np_, nm = float(np.linalg.norm(f0p)), float(np.linalg.norm(f1p)), float(np.linalg.norm(f1m))
        print(f"[probe] 基线 |F|={n0:.2f}N ; 沿 +f 20mm -> {np_:.2f}N ; 沿 -f 20mm -> {nm:.2f}N")
        self.assertLess(np_, 1.0, "沿 +f 撤退应脱离接触（|F|->0）")
        self.assertGreater(nm, n0 + 10.0, "沿 -f 顶进应显著加深（不是降低）")


# --------------------------------------------------------------------------- #
# 5) 卡死压力测试（3 个独立会话）
# --------------------------------------------------------------------------- #
class TestPressure(unittest.TestCase):
    def test_three_independent_cylinder_sessions(self):
        import demo.interactive as di
        for k in range(1, 4):
            app = _make_app()
            logs: list = []
            orig = app._log
            app._log = lambda m, _l=logs, _o=orig: (_l.append(str(m)), _o(m))[1]
            F: list = []
            base = app.env.step

            def w(n=1, _b=base, _a=app):
                st = _b(n)
                F.append(float(np.linalg.norm(_a.env.contact_force_on_ee()[0])))
                return st

            app.env.step = w  # type: ignore[method-assign]
            res = app.run_task("夹住黄色圆柱体")
            arr = np.asarray(F)
            peak = float(arr.max()) if len(arr) else 0.0
            n_over = int((arr > 100.0).sum())
            resets = sum(1 for l in logs if "scene_reset" in l or "场景已重置" in l)
            f_end = float(np.linalg.norm(app.env.contact_force_on_ee()[0]))
            print(f"\n[probe] run{k}: peak={peak:.1f}N >100N帧={n_over} scene_reset={resets} "
                  f"final|F|={f_end:.3f}N errs={[(r.skill, r.error_code) for r in res]}")
            self.assertLess(peak, 500.0, f"run{k} 峰值 {peak:.1f}N 超限")
            self.assertLessEqual(f_end, 5.0, f"run{k} 结束 |F|={f_end:.2f}N")
            self.assertEqual(resets, 0, f"run{k} 出现 scene_reset")


# --------------------------------------------------------------------------- #
# 7) 证伪
# --------------------------------------------------------------------------- #
class TestExtraFalsifications(unittest.TestCase):
    def test_table_full_degrades_gracefully(self):
        """桌面空位全被占用：应优雅退化（返回锚点+说明），不抛异常。"""
        tray = _obj("tray", "tray", "gray", (0.4, 0.3, 0.412), (0.09, 0.07, 0.012), False, True)
        held = _obj("box_red", "box", "red", (0.4, 0.3, 0.5), (0.02, 0.02, 0.02))
        blockers = []
        n = 0
        xs = np.arange(0.20, 0.61, 0.05)
        ys = np.arange(-0.35, 0.36, 0.05)
        for x in xs:
            for y in ys:
                n += 1
                blockers.append(_obj(f"blk{n}", "box", "gray", (float(x), float(y), 0.42),
                                     (0.03, 0.03, 0.02)))
        env = _Env([tray, held] + blockers)
        ex = SkillExecutor(env, _Kin(ok=True), _Runner(env), _Planner(), log=lambda *a, **k: None)
        xy, note = ex._choose_table_spot(held, approach_quat(0.0))
        print(f"\n[probe] 桌面塞满 {n} 个物体 -> 选点 {np.round(xy, 3)} note={note!r}")
        self.assertEqual(len(xy), 2, "应返回二维点")
        self.assertTrue(np.all(np.isfinite(xy)))
        self.assertIn("fallback_anchor", note)

    def test_choose_spot_respects_clearance_when_free(self):
        tray = _obj("tray", "tray", "gray", (0.4, 0.3, 0.412), (0.09, 0.07, 0.012), False, True)
        held = _obj("box_red", "box", "red", (0.4, 0.3, 0.5), (0.02, 0.02, 0.02))
        env = _Env([tray, held])
        ex = SkillExecutor(env, _Kin(ok=True), _Runner(env), _Planner(), log=lambda *a, **k: None)
        xy, note = ex._choose_table_spot(held, approach_quat(0.0))
        print(f"[probe] 空桌面选点 {np.round(xy, 3)} note={note!r}")
        self.assertNotIn("fallback_anchor", note, f"空桌面不应退化: {note}")
        for o in env.list_objects():
            if o.id == held.id:
                continue
            d = float(np.linalg.norm(np.asarray(xy, float) - np.asarray(o.pose[:2], float)))
            self.assertGreater(d, 0.02 + max(o.size[0], o.size[1]) + 0.08 - 1e-9,
                               f"空位与 {o.id} 净距不足")

    def test_target_switch_still_safe_and_place_works(self):
        app = _make_app()
        app.dispatch(":target bogus")
        self.assertEqual(app.target_mode, "on")
        app.dispatch(":target off")
        self.assertEqual(app.target_mode, "off")
        app.run_task("把蓝色方块 放回到桌子上")
        f_end = float(np.linalg.norm(app.env.contact_force_on_ee()[0]))
        z = float(app.env.object_pose("box_blue")[2])
        print(f"[probe] :target off 下放置: |F|={f_end:.2f}N box_blue z={z:.4f}")
        self.assertLessEqual(f_end, 5.0)
        self.assertLess(abs(z - (TABLE_HEIGHT + 0.02)), 0.02, "物体应落在桌面高度")


class _OkExec:
    def __call__(self, call):
        return SkillResult(True, call.skill, "ok")


class TestEpochMarkerMessageShape(unittest.TestCase):
    @staticmethod
    def _marked_history():
        call = {"skill": "pick", "args": {"target_id": "box_blue"},
                "compliance": None, "reasoning": ""}
        r = SkillResult(True, "pick", "x")
        marked = ("TASK_EPOCH_JSON "
                  + json.dumps({"task": "把蓝色方块 放回到桌子上"}, ensure_ascii=False)
                  + "\n" + json.dumps(call, ensure_ascii=False))
        return [{"role": "assistant", "content": marked},
                {"role": "user", "content": "OBSERVATION_JSON "
                 + json.dumps(to_jsonable(r.to_observation()), ensure_ascii=False)}]

    def test_multimodal_messages_stay_valid_openai(self):
        img = (np.random.default_rng(0).random((48, 48, 3)) * 255).astype(np.uint8)
        msgs = build_messages("夹住蓝色方块", _fake_scene(), {"overview": img},
                              history=self._marked_history())
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[-1], "user")
        # OpenAI 允许连续 user（历史末尾 + 本次任务）；不允许连续 assistant
        for a, b in zip(roles, roles[1:]):
            self.assertFalse(a == b == "assistant", "不得出现连续 assistant 消息")
        for m in msgs:
            c = m["content"]
            self.assertTrue(isinstance(c, (str, list)), f"content 类型非法: {type(c)}")
            if isinstance(c, list):
                for part in c:
                    self.assertIsInstance(part, dict)
                    self.assertIn("type", part)
        found = 0
        for m in msgs:
            c = m["content"]
            if isinstance(c, str) and "TASK_EPOCH_JSON" in c:
                found += 1
                self.assertEqual(m["role"], "assistant", "标记只应出现在 assistant 消息里")
                head, body = c.split("\n", 1)
                payload = json.loads(head.split("TASK_EPOCH_JSON ", 1)[1])
                self.assertEqual(payload["task"], "把蓝色方块 放回到桌子上")
                self.assertEqual(json.loads(body)["skill"], "pick", "标记不得破坏后续 JSON")
        self.assertGreaterEqual(found, 1, "历史里的纪元标记应原样保留")
        blob = json.dumps(msgs, ensure_ascii=False)
        self.assertIn("data:image/", blob, "应带合法图像 data URI")
        json.loads(blob)

    def test_vlm_client_request_body_serializable(self):
        from agent.schema import TOOLS
        from agent.vlm_client import VLMClient
        img = (np.random.default_rng(1).random((32, 32, 3)) * 255).astype(np.uint8)
        msgs = build_messages("夹住蓝色方块", _fake_scene(), {"overview": img},
                              history=self._marked_history())
        captured = {}
        client = VLMClient(api_key="dummy")

        def fake_post(payload):
            captured["dump"] = json.dumps(payload, ensure_ascii=False)
            captured["payload"] = payload
            return {"choices": [{"message": {"tool_calls": [
                {"function": {"name": "stop", "arguments": "{}"}}]}}]}

        client._post = fake_post  # type: ignore[method-assign]
        call = client.decide(msgs)
        self.assertEqual(call.skill, "stop")
        payload = captured["payload"]
        self.assertEqual(payload["model"], client.model)
        self.assertEqual(len(payload["tools"]), len(TOOLS))
        back = json.loads(captured["dump"])
        self.assertEqual(back["messages"][0]["role"], "system")
        self.assertTrue(any("TASK_EPOCH_JSON" in json.dumps(m, ensure_ascii=False)
                            for m in back["messages"]), "标记应可 json 往返")


class TestEpochTrimBoundary(unittest.TestCase):
    def _loop(self):
        return AgentLoop(_StubLoopEnv(), executor=_OkExec(), vlm=MockVLM(),
                         cameras=(), max_steps=6, log=None, max_history_messages=40)

    def test_last_epoch_group_and_successors_kept(self):
        loop = self._loop()
        msgs = []
        for i in range(40):
            content = json.dumps({"skill": "pick", "i": i})
            if i == 35:
                content = "TASK_EPOCH_JSON " + json.dumps({"task": "t35"}) + "\n" + content
            msgs.append({"role": "assistant", "content": content})
            msgs.append({"role": "user", "content": "OBSERVATION_JSON {}"})
        out = loop._trim_history(msgs)
        print(f"\n[probe] 裁剪 80 条(含 i=35 纪元) -> {len(out)} 条")
        self.assertLessEqual(len(out), 40)
        self.assertEqual(len(out) % 2, 0)
        for k in range(0, len(out), 2):
            self.assertEqual(out[k]["role"], "assistant")
            self.assertEqual(out[k + 1]["role"], "user")
        idx = [json.loads(m["content"].split("\n")[-1]).get("i")
               for m in out if m["role"] == "assistant"]
        self.assertIn(35, idx, "最后一个纪元标记所在组必须保留")
        self.assertIn(39, idx, "纪元之后的观测必须保留")
        tail = sorted(i for i in idx if i != 0)
        self.assertEqual(tail, list(range(tail[0], 40)), "最近若干组应连续保留（含纪元及其后）")
        self.assertLessEqual(tail[0], 35, "纪元组及其后不得被裁掉")

    def test_eight_commands_no_cross_pollution(self):
        loop = self._loop()
        tasks = ["夹住蓝色方块", "把红色方块放进托盘", "夹住黄色圆柱体", "把绿色方块放进托盘",
                 "夹住红色方块", "把蓝色方块放进托盘", "夹住绿色方块", "把黄色圆柱体放进托盘"]
        hist: list = []
        first = []
        for t in tasks:
            res = loop.run(t, history=hist)
            hist = list(loop.history)
            first.append(res[0].skill if res else None)
        print(f"[probe] 8 条命令首步: {list(zip(tasks, first))}; 末次 history={len(hist)} 条")
        for t, f in zip(tasks, first):
            self.assertEqual(f, "pick", f"{t} 首步应 pick，实际 {f}（历史污染）")


class TestOwnershipFilter(unittest.TestCase):
    @staticmethod
    def _fails(tid, n=3):
        out = []
        for _ in range(n):
            out.append({"role": "assistant",
                        "content": json.dumps({"skill": "pick", "args": {"target_id": tid}})})
            r = SkillResult(False, "pick", "x", "grasp_failed")
            out.append({"role": "user", "content": "OBSERVATION_JSON "
                        + json.dumps(to_jsonable(r.to_observation()))})
        return out

    def test_other_target_failures_do_not_block(self):
        call = _decide("夹住蓝色方块", self._fails("can_yellow"))
        print(f"\n[probe] can_yellow 失败 x3 -> 夹住蓝色方块: {call.skill} {call.args}")
        self.assertEqual(call.skill, "pick")
        self.assertEqual(call.args.get("target_id"), "box_blue")

    def test_same_target_failures_still_ask_user(self):
        call = _decide("夹住蓝色方块", self._fails("box_blue"))
        print(f"[probe] box_blue 失败 x3 -> 夹住蓝色方块: {call.skill}")
        self.assertEqual(call.skill, "ask_user", "同目标连续失败仍应转人工确认")

    def test_new_epoch_does_not_inherit_old_failures(self):
        hist = self._fails("box_blue") + [{
            "role": "assistant",
            "content": 'TASK_EPOCH_JSON {"task": "夹住蓝色方块"}\n'
                       + json.dumps({"skill": "pick", "args": {"target_id": "box_blue"}})}]
        call = _decide("夹住蓝色方块", hist)
        print(f"[probe] 新纪元标记后: {call.skill}")
        self.assertEqual(call.skill, "pick", "新纪元不应继承上一纪元的失败计数")

    def test_clear_history_style_question_is_conservative(self):
        call = _decide("放到桌子上不是盘子里", [])
        print(f"[probe] 无历史纠正指令: {call.skill}")
        self.assertIn(call.skill, ("ask_user", "stop"))
        self.assertNotEqual(call.args.get("target_id"), "box_red")


# --------------------------------------------------------------------------- #
# 8) on_top_of：真实叠放几何（独立复算）+ 大声失败 + 技能边界 + schema
# --------------------------------------------------------------------------- #
def _epoch_pick_blue():
    call = {"skill": "pick", "args": {"target_id": "box_blue"}}
    r = SkillResult(True, "pick", "ok")
    return [{"role": "assistant",
             "content": 'TASK_EPOCH_JSON {"task": "x"}\n' + json.dumps(call, ensure_ascii=False)},
            {"role": "user", "content": "OBSERVATION_JSON "
             + json.dumps(to_jsonable(r.to_observation()), ensure_ascii=False)}]


class TestStackGeometryReal(unittest.TestCase):
    def test_on_top_of_geometry_recomputed(self):
        app = _make_app()
        logs: list = []
        orig = app._log
        app._log = lambda m, _l=logs, _o=orig: (_l.append(str(m)), _o(m))[1]
        res = app.run_task("蓝色方块放在红色方块上方")
        obj = {o.id: o for o in app.env.list_objects()}
        blue = app.env.object_pose("box_blue")[:3]
        red = app.env.object_pose("box_red")[:3]
        red_top = float(red[2] + obj["box_red"].size[2])
        blue_bottom = float(blue[2] - obj["box_blue"].size[2])
        gap = blue_bottom - red_top
        dx = abs(float(blue[0] - red[0]))
        dy = abs(float(blue[1] - red[1]))
        # place 调用的 args（从 history 的 assistant 消息里取）
        place_args = None
        for m in app.history:
            c = str(m.get("content", ""))
            if m.get("role") == "assistant" and '"place"' in c:
                try:
                    place_args = json.loads(c.split("\n")[-1]).get("args")
                except Exception:
                    pass
        f_end = float(np.linalg.norm(app.env.contact_force_on_ee()[0]))
        resets = sum(1 for l in logs if "scene_reset" in l or "场景已重置" in l)
        print(f"\n[probe] 真实叠放独立复算: blue_bottom={blue_bottom:.4f} red_top={red_top:.4f} "
              f"gap={gap * 1000:+.1f}mm dx={dx * 1000:.1f}mm dy={dy * 1000:.1f}mm "
              f"|F|={f_end:.3f}N resets={resets} args={place_args}")
        self.assertEqual([r.skill for r in res], ["pick", "place", "stop"])
        self.assertTrue(all(r.ok for r in res))
        self.assertIsNotNone(place_args)
        self.assertEqual(place_args.get("on_top_of"), "box_red")
        self.assertNotIn("container_id", place_args)
        self.assertLess(abs(gap), 0.008, f"底面与锚点顶面差 {gap * 1000:.1f}mm 超 ±8mm")
        self.assertLessEqual(dx, obj["box_red"].size[0] + 0.02, "xy 超出锚点 footprint")
        self.assertLessEqual(dy, obj["box_red"].size[1] + 0.02, "xy 超出锚点 footprint")
        self.assertLessEqual(f_end, 5.0)
        self.assertEqual(resets, 0, "叠放不应触发 scene_reset")


class TestLoudFailurePhrases(unittest.TestCase):
    AMBIGUOUS = ["把它放到旁边去", "放到那个东西上面", "挪一挪", "放好一点",
                 "蓝色方块放过去", "放到上面",
                 "把蓝色方块放在红色方块和绿色方块中间"]

    def test_no_silent_tray_fallback(self):
        for ph in self.AMBIGUOUS:
            with self.subTest(phrase=ph):
                call = _decide(ph, _epoch_pick_blue())
                self.assertNotEqual(
                    (call.skill, call.args.get("container_id")), ("place", "tray"),
                    f"「{ph}」静默兜底到托盘")
                self.assertEqual(call.skill, "ask_user", f"「{ph}」应大声失败: {call.skill} {call.args}")
                self.assertIn("请指明", call.args.get("question", "") + "请指明")
                print(f"[probe] 大声失败: 「{ph}」 -> ask_user")

    def test_tray_above_phrase_judgement(self):
        """「放到盘子上面」：锚点是不可抓容器，不得当作 on_top_of；实测 ask_user。"""
        call = _decide("放到盘子上面", _epoch_pick_blue())
        print(f"[probe] 「放到盘子上面」 -> {call.skill} {call.args}")
        self.assertNotEqual(call.skill, "place", "不得把不可抓的盘子当锚点叠放")
        self.assertIn(call.skill, ("ask_user", "stop"))

    def test_explicit_targets_still_resolve(self):
        cases = {"把蓝色方块放进托盘": {"container_id": "tray"},
                 "放到盘子里": {"container_id": "tray"},
                 "蓝色方块放在红色方块上方": {"on_top_of": "box_red"},
                 "把蓝色方块放到桌子上": {"surface": "table"},
                 "蓝色方块放回到桌子上": {"surface": "table"}}
        for task, want in cases.items():
            with self.subTest(task=task):
                call = _decide(task, _epoch_pick_blue())
                self.assertEqual(call.skill, "place", f"{task} -> {call.skill}")
                for k, v in want.items():
                    self.assertEqual(call.args.get(k), v, f"{task} 应给出 {k}={v}")
                print(f"[probe] 合法目标: 「{task}」 -> {call.args}")


class _StackEnv(_Env):
    def __init__(self, objs):
        super().__init__(objs)
        self.held = None
        self._slip = None

    def step(self, n=1):
        if self._slip is not None and self.held is not None:
            p = self.object_pose(self.held).copy()
            p[:3] = p[:3] + np.asarray(self._slip, float)
            self.set_object_pose(self.held, p)
            self._slip = None
        return None


class _StackRunner(_Runner):
    def __init__(self, env, final=None, slip=None):
        super().__init__(env)
        self.final = final
        self.slip = slip
        self.calls = 0

    def move_to_pose(self, pos, quat=None, duration=None, compliance=None,
                     timeout=8.0, force_limit=None):
        self.calls += 1
        if self.final is not None and self.held is not None:
            p = self.env.object_pose(self.held).copy()
            p[0], p[1], p[2] = self.final
            self.env.set_object_pose(self.held, p)
        if self.calls >= 3 and self.slip is not None:
            self.env._slip = self.slip          # 释放后（观测阶段）人为滑移
        return MoveResult(True, "stub ok")


class TestStackSkillBoundaries(unittest.TestCase):
    def _scene(self, anchor_pose=(0.35, -0.14, 0.42), anchor_graspable=True):
        anchor = _obj("box_red", "box", "red", anchor_pose, (0.02, 0.02, 0.02),
                      graspable=anchor_graspable)
        moved = _obj("box_blue", "box", "blue", (0.35, 0.14, 0.50), (0.02, 0.02, 0.02))
        return _StackEnv([anchor, moved]), anchor, moved

    def _place(self, env, runner, args, held="box_blue"):
        ex = SkillExecutor(env, _Kin(), runner, _Planner(), log=lambda *a, **k: None)
        ex.holding = held
        runner.held = held
        env.held = held
        return ex(SkillCall("place", args))

    def test_anchor_missing(self):
        env, anchor, moved = self._scene()
        r = self._place(env, _StackRunner(env), {"on_top_of": "nope"})
        self.assertFalse(r.ok)
        self.assertEqual(r.error_code, "object_not_found")

    def test_anchor_not_graspable(self):
        env, anchor, moved = self._scene(anchor_graspable=False)
        r = self._place(env, _StackRunner(env), {"on_top_of": "box_red"})
        self.assertFalse(r.ok)
        self.assertEqual(r.error_code, "not_graspable")

    def test_anchor_held_by_self(self):
        env, anchor, moved = self._scene()
        r = self._place(env, _StackRunner(env), {"on_top_of": "box_red"}, held="box_red")
        self.assertFalse(r.ok)
        self.assertEqual(r.error_code, "anchor_held")

    def test_anchor_not_on_surface(self):
        env, anchor, moved = self._scene(anchor_pose=(0.35, -0.14, 0.50))
        r = self._place(env, _StackRunner(env), {"on_top_of": "box_red"})
        self.assertFalse(r.ok)
        self.assertEqual(r.error_code, "anchor_not_on_surface")

    def test_xy_outside_footprint_rejected(self):
        env, anchor, moved = self._scene()
        runner = _StackRunner(env, final=(0.35 + 0.10, -0.14, 0.44 + 0.02))
        r = self._place(env, runner, {"on_top_of": "box_red"})
        print(f"\n[probe] xy 超出 footprint: ok={r.ok} err={r.error_code} "
              f"footprint_ok={r.metrics.get('footprint_ok')}")
        self.assertFalse(r.ok)
        self.assertEqual(r.error_code, "place_mismatch")
        self.assertFalse(r.metrics.get("footprint_ok"))

    def test_release_slide_rejected(self):
        env, anchor, moved = self._scene()
        runner = _StackRunner(env, final=(0.35, -0.14, 0.44 + 0.02), slip=(0.03, 0.0, 0.0))
        r = self._place(env, runner, {"on_top_of": "box_red"})
        print(f"[probe] 释放后滑移 30mm: ok={r.ok} err={r.error_code} slide={r.metrics.get('slide_m')} "
              f"stable_ok={r.metrics.get('stable_ok')}")
        self.assertFalse(r.ok, "释放后 30mm 滑移应被稳定性判据拒绝")
        self.assertEqual(r.error_code, "place_mismatch")
        self.assertFalse(r.metrics.get("stable_ok"))

    def test_release_drop_rejected(self):
        env, anchor, moved = self._scene()
        runner = _StackRunner(env, final=(0.35, -0.14, 0.44 + 0.02), slip=(0.0, 0.0, -0.03))
        r = self._place(env, runner, {"on_top_of": "box_red"})
        self.assertFalse(r.ok)
        self.assertEqual(r.error_code, "place_mismatch")

    def test_stable_stack_accepted(self):
        env, anchor, moved = self._scene()
        runner = _StackRunner(env, final=(0.35, -0.14, 0.44 + 0.02), slip=None)
        r = self._place(env, runner, {"on_top_of": "box_red"})
        print(f"[probe] 稳定叠放: ok={r.ok} err={r.error_code} gap={r.metrics.get('bottom_gap')}")
        self.assertTrue(r.ok, f"稳定叠放应成功: {r.error_code} {r.message}")

    def test_two_targets_rejected(self):
        env, anchor, moved = self._scene()
        r = self._place(env, _StackRunner(env),
                        {"on_top_of": "box_red", "container_id": "tray"})
        self.assertFalse(r.ok)
        self.assertEqual(r.error_code, "bad_args")


class TestPlaceToolSchema(unittest.TestCase):
    def test_tools_expose_on_top_of_three_way(self):
        from agent.schema import TOOLS
        place = next(t for t in TOOLS if (t.get("function") or t).get("name") == "place")
        fn = place.get("function", place)
        props = fn["parameters"]["properties"]
        print(f"\n[probe] place 工具参数: {sorted(props)}")
        self.assertIn("on_top_of", props)
        self.assertIn("container_id", props)
        self.assertIn("surface", props)
        self.assertIn("on_top_of", json.dumps(fn, ensure_ascii=False))

    def test_validate_accepts_on_top_of_only_and_null_container(self):
        from agent.schema import validate_skill_call
        state = _fake_scene()
        ok, msg = validate_skill_call(SkillCall("place", {"container_id": None,
                                                          "on_top_of": "box_red"}), state)
        print(f"[probe] validate(container_id=null, on_top_of=box_red) -> {ok} {msg}")
        self.assertTrue(ok, f"合法叠放调用被拒: {msg}")

    def test_validate_rejects_two_and_none(self):
        from agent.schema import validate_skill_call
        state = _fake_scene()
        ok2, msg2 = validate_skill_call(SkillCall("place", {"container_id": "tray",
                                                            "on_top_of": "box_red"}), state)
        ok0, msg0 = validate_skill_call(SkillCall("place", {}), state)
        print(f"[probe] 三选一: two={ok2} ({msg2}) ; none={ok0} ({msg0})")
        self.assertFalse(ok2, "同时给两个目标应被拒")
        self.assertFalse(ok0, "一个都不给应被拒")

    def test_messages_with_on_top_of_history_stay_valid(self):
        img = (np.random.default_rng(2).random((40, 40, 3)) * 255).astype(np.uint8)
        hist = _epoch_pick_blue() + [
            {"role": "assistant", "content": json.dumps({"skill": "place",
                                                         "args": {"on_top_of": "box_red"}})},
            {"role": "user", "content": "OBSERVATION_JSON {}"}]
        msgs = build_messages("蓝色方块放在红色方块上方", _fake_scene(), {"overview": img}, history=hist)
        blob = json.dumps(msgs, ensure_ascii=False)
        self.assertIn("on_top_of", blob)
        json.loads(blob)
        for a, b in zip(msgs, msgs[1:]):
            self.assertFalse(a["role"] == b["role"] == "assistant")


# --------------------------------------------------------------------------- #
# 9) task-38：横向关系 / 先规划后执行 / 中止归还
# --------------------------------------------------------------------------- #
class TestLateralPlanning(unittest.TestCase):
    def test_all_relations_resolve_en_cn(self):
        cases = [("set box_green right of box_blue", "right"),
                 ("set box_green left of box_blue", "left"),
                 ("set box_green in front of box_blue", "front"),
                 ("set box_green behind box_blue", "back"),
                 ("set box_green beside box_blue", "beside"),
                 ("把绿色方块放在蓝色方块右边", "right"),
                 ("把绿色方块放在蓝色方块左边", "left"),
                 ("把绿色方块放在蓝色方块前面", "front"),
                 ("把绿色方块放在蓝色方块后面", "back"),
                 ("把绿色方块放在蓝色方块旁边", "beside")]
        for task, rel in cases:
            with self.subTest(task=task):
                args, _cp, q, _note = MockVLM()._placement_plan(task, _scene_objs(), False)
                self.assertIsNone(q, f"{task} 不应报错: {q}")
                self.assertEqual(args, {"anchor_id": "box_blue", "relation": rel})
                print(f"[probe] {task} -> {args}")

    def _sequence(self, phrase):
        ex = _RecExec()
        loop = AgentLoop(_StubLoopEnv(), executor=ex, vlm=MockVLM(), cameras=(),
                         max_steps=6, log=None)
        res = loop.run(phrase)
        return [r.skill for r in res], ex.calls

    def test_unsupported_phrases_only_ask_user_no_pick(self):
        for ph in ["斜后方", "中间", "左上", "随便放", "把它放到旁边去",
                   "把蓝色方块放在红色方块和绿色方块中间", "put box_green somewhere"]:
            with self.subTest(phrase=ph):
                seq, calls = self._sequence(ph)
                print(f"[probe] 「{ph}」 -> {seq} (executor={calls})")
                self.assertEqual(seq, ["ask_user"], f"「{ph}」应只 ask_user")
                self.assertNotIn("pick", calls, f"「{ph}」不得先 pick")

    def test_na_bian_must_not_pick_first_RED(self):
        """R1（保留红）：裸「那边」目前 intent=False -> 先 pick 再 ask_user（半执行）。"""
        seq, calls = self._sequence("那边")
        print(f"\n[FINDING] 「那边」 -> {seq} (executor={calls})")
        self.assertNotIn("pick", calls, "「那边」不得先抓后拒（当前会 pick）")
        self.assertEqual(seq, ["ask_user"])

    def test_diagonal_left_up_must_not_become_left_RED(self):
        """R2（保留红）：『左上』应视为不支持（_LATERAL_HINTS 含 左上），
        现被同位置的『左』抢先解析成 relation=left 并执行。"""
        args, _cp, q, _note = MockVLM()._placement_plan("把绿色方块放在蓝色方块左上", _scene_objs(), False)
        print(f"[FINDING] 『左上』 -> args={args} q={q}")
        self.assertIsNone(args, f"『左上』不应解析成可执行目标（当前 {args}）")

    def test_error_message_echoes_anchor_and_support_list(self):
        args, _cp, q, _note = MockVLM()._placement_plan("把绿色方块放在蓝色方块斜后方", _scene_objs(), False)
        print(f"[probe] 斜后方 -> q={q}")
        self.assertIsNone(args)
        self.assertIn("box_blue", q, "报错必须回显已识别锚点")
        self.assertIn("right", q)
        self.assertIn("left", q)
        self.assertIn("beside", q)


class TestLateralGeometryReal(unittest.TestCase):
    def test_right_of_independent_geometry(self):
        app = _make_app()
        logs: list = []
        orig = app._log
        app._log = lambda m, _l=logs, _o=orig: (_l.append(str(m)), _o(m))[1]
        res = app.run_task("set box_green right of box_blue")
        obj = {o.id: o for o in app.env.list_objects()}
        bg = app.env.object_pose("box_green")[:3]
        bb = app.env.object_pose("box_blue")[:3]
        half = max(obj["box_blue"].size[0], obj["box_blue"].size[1]) + \
            max(obj["box_green"].size[0], obj["box_green"].size[1])
        delta = np.asarray(bg[:2], float) - np.asarray(bb[:2], float)
        dist = float(np.linalg.norm(delta))
        f_end = float(np.linalg.norm(app.env.contact_force_on_ee()[0]))
        resets = sum(1 for l in logs if "scene_reset" in l or "场景已重置" in l)
        plan_in_reasoning = any("已规划放置" in str(m.get("content", "")) for m in app.history
                                if m.get("role") == "assistant")
        print(f"\n[probe] right-of 独立复算: delta={np.round(delta, 4)} 方向={'+x' if delta[0] > 0 else '?'} "
              f"中心距={dist * 1000:.1f}mm 期望≥{half * 1000 + 10:.0f}mm |F|={f_end:.3f} resets={resets} "
              f"plan_in_step1={plan_in_reasoning}")
        self.assertEqual([r.skill for r in res], ["pick", "place", "stop"])
        self.assertTrue(all(r.ok for r in res))
        self.assertGreater(delta[0], 0.0, "right 应为 +x")
        self.assertLess(abs(float(delta[1])), 0.005, "right 不应有 y 偏移")
        self.assertGreaterEqual(dist, half + 0.010 - 0.004, "中心距应≈半宽和+10mm")
        self.assertLessEqual(dist, half + 0.010 + 0.040, "中心距超出容差")
        self.assertLessEqual(f_end, 5.0)
        self.assertEqual(resets, 0)
        self.assertIsNone(app.skills.holding)
        self.assertTrue(plan_in_reasoning, "第 1 步 reasoning 应含放置计划")

    def test_beside_picks_max_clearance_independently(self):
        app = _make_app()
        res = app.run_task("set box_green beside box_blue")
        self.assertTrue(all(r.ok for r in res), [(r.skill, r.message) for r in res])
        m = res[1].metrics
        obj = {o.id: o for o in app.env.list_objects()}
        anchor = obj["box_blue"]
        held = obj["box_green"]
        margin = 0.010
        best_name, best_clear = None, -9.9
        clearances = {}
        for name, vec in {"right": (1.0, 0.0), "left": (-1.0, 0.0),
                          "front": (0.0, 1.0), "back": (0.0, -1.0)}.items():
            axis_x = abs(vec[0]) > 0.5
            ha = float(anchor.size[0] if axis_x else anchor.size[1])
            hh = float(held.size[0] if axis_x else held.size[1])
            p = np.asarray(anchor.pose[:2], float) + np.asarray(vec, float) * (ha + hh + margin)
            x0, x1, y0, y1 = -0.47, 0.67, -0.57, 0.57
            on = (x0 + hh <= p[0] <= x1 - hh and y0 + hh <= p[1] <= y1 - hh)
            clear = -1.0 if not on else min(
                (float(np.linalg.norm(p - np.asarray(o.pose[:2], float))) - hh
                 - float(max(o.size[0], o.size[1]))
                 for o in app.env.list_objects()
                 if o.id not in (anchor.id, held.id)), default=9.9)
            clearances[name] = round(clear * 1000, 1)
            if clear > best_clear:
                best_name, best_clear = name, clear
        chosen = None
        for name, vec in {"right": (1.0, 0.0), "left": (-1.0, 0.0),
                          "front": (0.0, 1.0), "back": (0.0, -1.0)}.items():
            if np.allclose(np.asarray(m.get("relation_dir"), float), vec, atol=1e-6):
                chosen = name
        print(f"[probe] beside 独立复算净距={clearances} -> 最大={best_name}; "
              f"技能选={chosen} ({m.get('spot')})")
        self.assertEqual(chosen, best_name, "beside 应选净距最大方向")

    def test_distance_override(self):
        from core.types import SkillCall as SC
        app = _make_app()
        r1 = app.skills(SC("pick", {"target_id": "box_green"}))
        self.assertTrue(r1.ok, r1.message)
        r2 = app.skills(SC("place", {"anchor_id": "box_blue", "relation": "right",
                                     "distance": 0.12}))
        bb = app.env.object_pose("box_blue")[:3]
        bg = app.env.object_pose("box_green")[:3]
        dist = float(np.linalg.norm(np.asarray(bg[:2], float) - np.asarray(bb[:2], float)))
        print(f"[probe] distance=0.12: ok={r2.ok} expected={r2.metrics.get('expected_dist')} "
              f"actual={r2.metrics.get('actual_dist')} 独立复算={dist * 1000:.1f}mm")
        self.assertTrue(r2.ok, f"{r2.error_code} {r2.message}")
        self.assertAlmostEqual(dist, 0.12, delta=0.025)


class TestReturnSemantics(unittest.TestCase):
    def test_abort_returns_held_object(self):
        app = _make_app()
        logs: list = []
        orig = app._log
        app._log = lambda m, _l=logs, _o=orig: (_l.append(str(m)), _o(m))[1]
        app.run_task("夹住蓝色方块")
        self.assertEqual(app.skills.holding, "box_blue", "pick-only 成功后应保持夹持")
        r = app.run_task("把它放到那边去")
        self.assertEqual([x.skill for x in r], ["ask_user"])
        obj = {o.id: o for o in app.env.list_objects()}
        pose = app.env.object_pose("box_blue")
        bottom = float(pose[2] - obj["box_blue"].size[2])
        gap = bottom - TABLE_HEIGHT
        ret = [l for l in logs if "return_held_object metrics" in l or "归还" in l]
        print(f"\n[probe] 中止归还: holding={app.skills.holding} bottom_gap={gap * 1000:.1f}mm "
              f"logs={ret[:2]}")
        self.assertIsNone(app.skills.holding, "中止后必须归还并清空 holding")
        self.assertLess(gap, 0.05, "物体不得悬空（底面离桌应 <50mm）")
        self.assertTrue(any("returned_to_xy" in l and "mode" in l for l in logs),
                        "应打印归还位置与 metrics(mode/returned_to_xy)")

    def test_pick_only_success_keeps_holding_and_no_return(self):
        app = _make_app()
        called = {"n": 0}
        orig = app.skills.return_held_object

        def spy(*a, **k):
            called["n"] += 1
            return orig(*a, **k)

        app.skills.return_held_object = spy  # type: ignore[method-assign]
        r = app.run_task("夹住蓝色方块")
        print(f"[probe] pick-only: {[(x.skill, x.ok) for x in r]} holding={app.skills.holding} "
              f"return_calls={called['n']}")
        self.assertEqual([x.skill for x in r], ["pick", "stop"])
        self.assertEqual(app.skills.holding, "box_blue", "成功结束的 pick-only 必须保留夹持")
        self.assertEqual(called["n"], 0, "成功结束不得调用 return_held_object")


# --------------------------------------------------------------------------- #
# 10) task-43：复合方向红断言（N1/N2 同类）+ 防误伤
# --------------------------------------------------------------------------- #
def _scene5_state():
    """5 物体场景（含 box_green），供复合方向用例使用。"""
    st = _fake_scene()
    if not any(o.id == "box_green" for o in st.objects):
        st.objects.insert(2, _obj("box_green", "box", "green", (0.45, -0.05, 0.42),
                                  (0.02, 0.02, 0.02)))
    return st


class _StubLoopEnv5:
    def get_state(self):
        return _scene5_state()

    def render(self, camera="overview"):
        raise RuntimeError("stub")


def _sequence_phrase(phrase):
    ex = _RecExec()
    loop = AgentLoop(_StubLoopEnv5(), executor=ex, vlm=MockVLM(), cameras=(),
                     max_steps=6, log=None)
    res = loop.run(phrase)
    return [r.skill for r in res], ex.calls


def _decide_fresh(task):
    return MockVLM().decide(build_messages(task, _scene5_state(), {}, history=[]))


class TestCompoundDirectionRed(unittest.TestCase):
    """复合方向必须「大声失败」：只有 ask_user、回显锚点、列出支持项。"""

    COMPOUND = [
        ("put box_green top-left of box_blue", "top-left"),
        ("put box_green top-right of box_blue", "top-right"),
        ("put box_green bottom-left of box_blue", "bottom-left"),
        ("put box_green bottom-right of box_blue", "bottom-right"),
        ("put box_green upper-left of box_blue", "upper-left"),
        ("put box_green upper-right of box_blue", "upper-right"),
        ("put box_green lower-left of box_blue", "lower-left"),
        ("put box_green lower-right of box_blue", "lower-right"),
        ("put box_green upper-front of box_blue", "upper-front"),
        ("put box_green back-right of box_blue", "back-right"),
        ("把绿色方块放在蓝色方块斜右上方", "斜右上方"),
        ("把绿色方块放在蓝色方块斜左上方", "斜左上方"),
        ("把绿色方块放在蓝色方块斜前上方", "斜前上方"),
        ("把绿色方块放在蓝色方块左上方", "左上方"),
        ("把绿色方块放在蓝色方块右上方", "右上方"),
    ]

    def test_no_pick_or_place_only_ask_user(self):
        reds = []
        for ph, tag in self.COMPOUND:
            with self.subTest(direction=tag):
                seq, calls = _sequence_phrase(ph)
                if seq != ["ask_user"]:
                    reds.append((tag, seq))
                self.assertEqual(seq, ["ask_user"],
                                 f"「{tag}」应只 ask_user，当前 {seq}（会先抓/放）")
                self.assertNotIn("pick", calls, f"「{tag}」不得 pick")
                self.assertNotIn("place", calls, f"「{tag}」不得 place")
        if reds:
            print(f"\n[RED] 复合方向仍会执行: {reds}")

    def test_message_echoes_anchor_and_supports(self):
        reds = []
        for ph, tag in self.COMPOUND:
            with self.subTest(direction=tag):
                call = _decide_fresh(ph)
                if call.skill != "ask_user":
                    reds.append((tag, call.skill, call.args))
                self.assertEqual(call.skill, "ask_user", f"「{tag}」当前 {call.skill} {call.args}")
                q = str(call.args.get("question", ""))
                self.assertIn("box_blue", q, f"「{tag}」报错必须回显锚点 box_blue")
                for rel in ("right", "left", "front", "back", "beside"):
                    self.assertIn(rel, q, f"「{tag}」报错必须列出支持项 {rel}")
        if reds:
            print(f"[RED] 复合方向未回显锚点/支持项: {reds}")


class TestCompoundNoOverBlock(unittest.TestCase):
    """防误伤：简单方向与指示词+可解析目标必须照常 pick/place。"""

    def test_simple_stack_and_lateral_still_execute(self):
        cases = [("set box_green on top of box_blue", {"on_top_of": "box_blue"}),
                 ("set box_green right of box_blue", {"anchor_id": "box_blue", "relation": "right"})]
        for ph, want in cases:
            with self.subTest(phrase=ph):
                seq, _calls = _sequence_phrase(ph)
                print(f"[probe] 防误伤 {ph!r} -> {seq}")
                self.assertEqual(seq, ["pick", "place", "stop"], f"{ph} 被误伤: {seq}")
                args, _cp, q, _note = MockVLM()._placement_plan(ph, _scene_objs(), False)
                self.assertIsNone(q)
                self.assertEqual(args, want)

    def test_deictic_with_resolvable_target_still_executes(self):
        seq, _calls = _sequence_phrase("把它放到桌子上")
        print(f"[probe] 防误伤 '把它放到桌子上' -> {seq}")
        self.assertEqual(seq, ["pick", "place", "stop"],
                         "指示词 + 可解析目标 不得被复合方向规则拦住")

    def test_pick_only_unaffected(self):
        seq, _calls = _sequence_phrase("夹住蓝色方块")
        print(f"[probe] 防误伤 '夹住蓝色方块' -> {seq}")
        self.assertEqual(seq, ["pick", "stop"], "纯抓取不得被指示词/方向规则误伤")


if __name__ == "__main__":
    unittest.main(verbosity=2)
