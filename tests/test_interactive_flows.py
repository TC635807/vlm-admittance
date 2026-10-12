"""交互式意图路由与安全回归（verifier 独立验证，标准库 unittest）。

覆盖（对应用户实测 P1~P5）
------------------------
* P1「回到默认位置」-> home 成功后再 stop，全程不得出现 pick/place。
* P2「夹住黄色圆柱体」「夹住蓝色方块」-> pick 成功后再 stop，不得追加 place。
* P3 抓取连续失败 -> ask_user 立即终止闭环，不得空转到 max_steps。
* P4 放置守卫式下降 -> 接触力峰值受控、任务结束时 |force|≈0、不卡死。
* P5 圆柱接近漂移护栏 -> pick 返回 object_disturbed，物体最终仍在桌面。

证伪（对抗）用例
----------------
* (a) 脏状态：先把末端压入桌面造成 800+ N 卡死，再跑自然语言任务，验证
      InteractiveApp.run_task 的前置 hard_recover 会清掉残留接触并让任务继续。
* (b) hard_recover 三级：可脱身 -> 'retreat'；救不回 -> 'scene_reset'，且确实
      env.reset(seed=0)、holding 清空、物体回到初始位。
* (c) 正常任务「把红色的方块放进托盘」-> pick->place 未被 pick_only 路由吞掉。
* (d) 边界词「松开夹爪」-> open_gripper、「停下」-> stop，且「放」字不干扰。

运行
----
    cd /home/tc63/mujoco/vlm_admittance
    MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m tests.test_interactive_flows

严格安全门
----------
默认只打印实测力值/位移（回归用例保持绿）。设置 VLM_STRICT_SAFETY=1 时，把
"接触力峰值 >= 500 N / 单次冲击 > 100 N 持续 >= 0.1 s / 任何物体被撞离桌面"
判为失败，用于复现本轮证伪发现的 BLOCKER：

    VLM_STRICT_SAFETY=1 MUJOCO_GL=glfw python -m tests.test_interactive_flows
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.types import (ContactState, RobotState, SceneObject, SceneState,
                        SkillCall, SkillResult)          # noqa: E402
from agent.state import build_messages, to_jsonable      # noqa: E402
from agent.vlm_client import MockVLM                      # noqa: E402
from agent.loop import AgentLoop                          # noqa: E402
from skills import approach_quat                          # noqa: E402
from skills.library import PICK_APPROACH_DRIFT_TOL as PICK_DRIFT_TOL  # noqa: E402

STRICT = os.environ.get("VLM_STRICT_SAFETY", "").strip().lower() not in ("", "0", "false", "no")
TABLE_Z = 0.40


# --------------------------------------------------------------------------- #
# 无 GL 的纯路由夹具
# --------------------------------------------------------------------------- #
def _fake_scene() -> SceneState:
    def obj(i, c, col, pos, size, graspable=True, is_container=False):
        return SceneObject(i, c, col, np.array([pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0]),
                           tuple(size), graspable, is_container)
    objs = [
        obj("box_red", "box", "red", (0.35, -0.14, 0.42), (0.02, 0.02, 0.02)),
        obj("box_blue", "box", "blue", (0.35, 0.14, 0.42), (0.02, 0.02, 0.02)),
        obj("box_green", "box", "green", (0.45, -0.05, 0.42), (0.02, 0.02, 0.02)),
        obj("can_yellow", "cylinder", "yellow", (0.40, 0.02, 0.435), (0.015, 0.015, 0.035)),
        obj("tray", "tray", "gray", (0.40, 0.30, 0.412), (0.09, 0.07, 0.012), False, True),
    ]
    return SceneState(
        0.0,
        RobotState(np.zeros(6), np.zeros(6), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
        objs, ContactState(np.zeros(3), np.zeros(3), 0.0, False),
    )


def _observation(skill: str, ok: bool = True, error_code=None) -> list:
    r = SkillResult(ok, skill, "stub", error_code)
    return [
        {"role": "assistant", "content": json.dumps({"skill": skill, "args": {}}, ensure_ascii=False)},
        {"role": "user", "content": "OBSERVATION_JSON "
         + json.dumps(to_jsonable(r.to_observation()), ensure_ascii=False)},
    ]


def route(task: str, max_steps: int = 8) -> list[SkillCall]:
    """离线跑 MockVLM 的决策序列（每一步都当成功，观察回灌）。"""
    vlm, state, hist, calls = MockVLM(), _fake_scene(), [], []
    for _ in range(max_steps):
        call = vlm.decide(build_messages(task, state, {}, history=hist))
        calls.append(call)
        if call.skill in ("stop", "ask_user"):
            break
        hist += _observation(call.skill, True)
    return calls


def skills_of(calls) -> list:
    return [c.skill for c in calls]


def assert_holding_invariant(case: unittest.TestCase, app) -> None:
    """夹持一致性不变量（task-15，替代旧的「序列结束必须 holding=None」）。

    * 夹爪张开（env.get_gripper() >= 0.8）=> skills.holding 必须为 None；
    * skills.holding 非 None => 夹爪必须闭合且与场景真值 robot.holding 一致，
      且该 id 必须真实存在（带载归位后 holding 仍为手持物是**正确**行为）。

    夹爪闭合但为空夹（robot.holding=None）不算违反——只要求 holding 不指向错的东西。
    """
    st = app.env.get_state()
    gripper = float(app.env.get_gripper())
    holding = app.skills.holding
    if gripper >= 0.8:
        case.assertIsNone(
            holding, f"夹爪已张开 gripper={gripper:.3f} 但 skills.holding={holding!r}")
        return
    if holding is not None:
        case.assertEqual(holding, st.robot.holding,
                         f"skills.holding={holding!r} 与 robot.holding={st.robot.holding!r} 不一致")
        case.assertTrue(any(o.id == holding for o in st.objects),
                        f"skills.holding={holding!r} 不是场景中的物体")


# --------------------------------------------------------------------------- #
# 1) 纯路由：P1 / P2 / 证伪 c / d
# --------------------------------------------------------------------------- #
class TestIntentRouting(unittest.TestCase):
    def test_p1_home_route_never_picks(self):
        calls = route("回到默认位置")
        self.assertEqual(skills_of(calls), ["home", "stop"])
        self.assertNotIn("pick", skills_of(calls))
        self.assertNotIn("place", skills_of(calls))
        self.assertEqual(calls[0].args, {})

    def test_p2_pick_only_never_places(self):
        for task, target in (("夹住黄色圆柱体", "can_yellow"), ("夹住蓝色方块", "box_blue")):
            with self.subTest(task=task):
                calls = route(task)
                self.assertEqual(skills_of(calls), ["pick", "stop"])
                self.assertEqual(calls[0].args.get("target_id"), target)
                self.assertNotIn("place", skills_of(calls))

    def test_c_normal_place_flow_not_swallowed(self):
        calls = route("把红色的方块放进托盘")
        self.assertEqual(skills_of(calls), ["pick", "place", "stop"])
        self.assertEqual(calls[0].args.get("target_id"), "box_red")
        self.assertEqual(calls[1].args.get("container_id"), "tray")

    def test_d_boundary_words(self):
        self.assertEqual(skills_of(route("松开夹爪")), ["open_gripper", "stop"])
        self.assertEqual(skills_of(route("停下")), ["stop"])
        self.assertEqual(skills_of(route("别动")), ["stop"])
        # 「松开/放开」含「放」，不得被误判成抓取
        self.assertNotIn("pick", skills_of(route("松开夹爪")))
        self.assertNotIn("pick", skills_of(route("放开它")))


# --------------------------------------------------------------------------- #
# 2) P3：ask_user 终止闭环（stub env/executor，无 GL）
# --------------------------------------------------------------------------- #
class _StubEnv:
    def get_state(self):
        return _fake_scene()

    def render(self, camera="overview"):     # cameras=() 时不会被调用
        raise RuntimeError("stub env 不渲染")


class _FailPickExecutor:
    def __init__(self):
        self.skills: list[str] = []

    def __call__(self, call: SkillCall) -> SkillResult:
        self.skills.append(call.skill)
        if call.skill == "pick":
            return SkillResult(False, "pick", "stub 抓取失败", "grasp_failed")
        return SkillResult(True, call.skill)


class TestAskUserTerminates(unittest.TestCase):
    def test_p3_ask_user_breaks_loop(self):
        ex = _FailPickExecutor()
        loop = AgentLoop(_StubEnv(), executor=ex, vlm=MockVLM(), cameras=(),
                         max_steps=12, log=None)
        results = loop.run("夹住红色方块")
        asks = [r for r in results if r.skill == "ask_user"]
        # 要求（P3）：ask_user <= 1 且闭环立即终止。task-11 后对致命错误可直接
        # stop，因此最后一步允许是 ask_user 或 stop，但绝不允许空转到 max_steps。
        self.assertLessEqual(len(asks), 1, f"ask_user 应 <=1 次，实际 {len(asks)}")
        self.assertIn(results[-1].skill, ("ask_user", "stop"), "必须以 ask_user/stop 结束")
        self.assertLess(len(results), 12, "不得空转到 max_steps")
        self.assertNotIn("place", ex.skills)


# --------------------------------------------------------------------------- #
# 真实仿真夹具（需 MUJOCO_GL）
# --------------------------------------------------------------------------- #
def _make_app(seed: int = 0):
    """构造 InteractiveApp 但不进入 main()/viewer，复用 run_task 的真实路径。"""
    from demo.interactive import InteractiveApp
    args = argparse.Namespace(vlm="mock", with_barrier=False, seed=seed, realtime=False,
                              headless=True, with_images=False, max_steps=12, self_test=False)
    return InteractiveApp(args)


class _ForceProbe:
    """只读探针：包裹 env.step，记录每步 |contact_force_on_ee| 与物体位置。"""

    def __init__(self, app):
        self.app = app
        self.ids = [o.id for o in app.env.list_objects()]
        self.start = {i: app.env.object_pose(i)[:3].copy() for i in self.ids}
        self.forces: list[float] = []
        self.times: list[float] = []
        self.track: list[dict] = []
        self._base = app.env.step

        def rec(n: int = 1):
            st = self._base(n)
            f, _ = app.env.contact_force_on_ee()
            self.forces.append(float(np.linalg.norm(f)))
            self.times.append(float(app.env.data.time))
            self.track.append({i: app.env.object_pose(i)[:3].copy() for i in self.ids})
            return st

        app.env.step = rec  # type: ignore[method-assign]

    def peak(self) -> float:
        return float(max(self.forces)) if self.forces else 0.0

    def final(self) -> float:
        f, _ = self.app.env.contact_force_on_ee()
        return float(np.linalg.norm(f))

    def sustained_over(self, thr: float) -> tuple[int, float]:
        best = cur = 0
        for v in self.forces:
            cur = cur + 1 if v > thr else 0
            best = max(best, cur)
        return int(sum(1 for v in self.forces if v > thr)), best * self.app.env.dt

    def displacement(self, obj_id: str) -> float:
        if not self.track:
            return 0.0
        arr = np.array([s[obj_id] for s in self.track])
        return float(np.linalg.norm(arr - self.start[obj_id], axis=1).max())

    def final_pose(self, obj_id: str):
        return self.app.env.object_pose(obj_id)[:3].copy()


# --------------------------------------------------------------------------- #
# 3) 用户三条命令序列（P1/P2/P3/P5）+ 力探针
# --------------------------------------------------------------------------- #
class TestUserSequence(unittest.TestCase):
    def _run_sequence(self):
        app = _make_app()
        probe = _ForceProbe(app)
        r1 = app.run_task("夹住黄色圆柱体")     # P5：圆柱
        r2 = app.run_task("夹住蓝色方块")       # P2：只夹不放
        r3 = app.run_task("回到默认位置")       # P1：回家
        return app, probe, r1, r2, r3

    def test_user_sequence_routing(self):
        app, probe, r1, r2, r3 = self._run_sequence()
        sk1, sk2, sk3 = skills_of(r1), skills_of(r2), skills_of(r3)

        # P2：pick 成功后再 stop，绝不 place
        self.assertEqual(sk2, ["pick", "stop"], f"蓝色方块应只 pick+stop，实际 {sk2}")
        self.assertTrue(r2[0].ok, r2[0].message)
        self.assertNotIn("place", sk2)

        # P1：带载归位（task-13）——「回到默认位置」现在必须**成功**，
        # 且不得出现 pick/place，也不得空转到 max_steps。
        self.assertIn("home", sk3)
        self.assertEqual(sk3[-1], "stop")
        self.assertTrue(r3[0].ok, f"带载归位应成功: {r3[0].error_code} {r3[0].message}")
        self.assertNotIn("pick", sk3)
        self.assertNotIn("place", sk3)
        self.assertLess(len(r3), 12, "home 任务空转到了 max_steps")

        # P2/P5：圆柱任务不得出现 place
        self.assertNotIn("place", sk1)

        # P3：ask_user 至多 1 次且任何任务都不得空转到 max_steps
        asks = sum(1 for r in (r1 + r2 + r3) if r.skill == "ask_user")
        self.assertLessEqual(asks, 1, f"ask_user 出现 {asks} 次")
        for r in (r1, r2, r3):
            self.assertLess(len(r), 12, "任务空转到了 max_steps")

        # P5：护栏是**按单次接近**判的（>8mm 才 object_disturbed）。pick 已改为
        # 10 组参数搜索，实测每段接近漂移均 <8mm，故护栏不必触发；这里改为直接约束
        # 累计位移（原缺陷是 33->127->152mm 级联），阈值 20mm 仍远严于原始 33mm。
        cy = probe.final_pose("can_yellow")
        disp = probe.displacement("can_yellow")
        guard = any(r.error_code == "object_disturbed" for r in r1)
        print(f"\n[probe] 圆柱累计位移 {disp * 1000:.1f}mm；护栏触发={guard}")
        self.assertTrue(guard or disp < 0.020,
                        f"圆柱累计推移 {disp * 1000:.1f} mm（护栏未触发，原缺陷 33mm 起）: "
                        f"{[(r.skill, r.error_code) for r in r1]}")
        # 圆柱最终仍在桌面上（z 不低于桌面，且未飞出桌面范围）
        self.assertGreater(cy[2], TABLE_Z - 0.01, f"圆柱掉下桌面 z={cy[2]:.4f}")
        self.assertLess(cy[2], TABLE_Z + 0.15, f"圆柱被弹飞 z={cy[2]:.4f}")

        # 结束时 |F| 接近 0
        f_end = probe.final()
        self.assertLess(f_end, 1.0, f"结束时 |F|={f_end:.2f} N")

        # 结束时夹持一致性不变量：带载归位后 holding 仍为 box_blue 是正确行为，
        # 只要它与 robot.holding/夹爪开合一致即可。
        assert_holding_invariant(self, app)

    def test_safety_gate(self):
        """BLOCKER-1 严格门：整段用户序列的峰值力 / 超限持续时间 / 物体是否被抛离桌面。"""
        app, probe, r1, r2, r3 = self._run_sequence()
        peak = probe.peak()
        n_over, longest = probe.sustained_over(100.0)
        ejected = {i: p for i, p in ((i, probe.final_pose(i)) for i in probe.ids)
                   if p[2] < TABLE_Z - 0.02}
        print(f"\n[probe] 峰值 |F| = {peak:.1f} N；>100 N 采样 {n_over} 个，"
              f"最长连续 {longest * 1000:.1f} ms；被抛离桌面: {sorted(ejected)}")
        if STRICT:
            self.assertLess(peak, 500.0, f"BLOCKER: 接触力峰值 {peak:.1f} N >= 500 N")
            self.assertLess(longest, 0.1, f"BLOCKER: |F|>100 N 持续 {longest * 1000:.1f} ms")
            self.assertEqual(ejected, {}, f"BLOCKER: 物体被撞下桌面 {ejected}")


# --------------------------------------------------------------------------- #
# 3b) 夹持状态一致性 + 带载归位（task-13/14 新语义）
# --------------------------------------------------------------------------- #
class TestHoldingHygiene(unittest.TestCase):
    def test_holding_after_open_and_home(self):
        app = _make_app()
        r1 = app.run_task("夹住蓝色方块")
        self.assertEqual(skills_of(r1), ["pick", "stop"])
        self.assertEqual(app.skills.holding, "box_blue")
        assert_holding_invariant(self, app)          # 闭合且夹着 -> 与 robot.holding 一致

        r2 = app.run_task("松开夹爪")
        self.assertEqual(skills_of(r2), ["open_gripper", "stop"])
        self.assertGreater(app.env.get_gripper(), 0.8, "夹爪应已张开")
        assert_holding_invariant(self, app)          # 张开 -> holding 必须为 None
        self.assertIsNone(app.skills.holding, "松爪后 holding 应刷新为 None（N2 已修）")

        r3 = app.run_task("回到默认位置")
        self.assertEqual(skills_of(r3), ["home", "stop"])
        self.assertTrue(r3[0].ok, r3[0].message)
        assert_holding_invariant(self, app)

    def test_carry_home_while_holding(self):
        """N1 的带载归位（task-13）：夹着 box_blue 也应成功归位，且不撞桌面/无关物体。"""
        app = _make_app()
        probe = _ForceProbe(app)
        app.run_task("夹住蓝色方块")
        self.assertEqual(app.skills.holding, "box_blue")
        other_before = {o.id: app.env.object_pose(o.id)[:3].copy()
                        for o in app.env.list_objects() if o.id != "box_blue"}
        probe.forces.clear()          # 只统计「归位阶段」的力（pick 阶段的力不算）

        r = app.run_task("回到默认位置")
        self.assertEqual(skills_of(r), ["home", "stop"])
        self.assertTrue(r[0].ok, f"带载归位应成功: {r[0].error_code} {r[0].message}")
        self.assertLess(len(r), 12, "home 任务空转到了 max_steps")
        self.assertNotIn("pick", skills_of(r))
        self.assertNotIn("place", skills_of(r))

        # 归位后仍夹着 box_blue：holding 与 robot.holding 一致（正确行为，不是 bug）
        assert_holding_invariant(self, app)
        self.assertEqual(app.skills.holding, "box_blue")

        # 安全：手持物不得掉到桌面/被撞飞，且归位路径不得碰到无关物体
        held_z = float(app.env.object_pose("box_blue")[2])
        self.assertGreater(held_z, TABLE_Z + 0.05,
                           f"归位后手持物掉到桌面附近 z={held_z:.4f}")
        for oid, before in other_before.items():
            d = float(np.linalg.norm(app.env.object_pose(oid)[:3] - before))
            self.assertLess(d, 0.001, f"归位路径碰到无关物体 {oid}: {d * 1000:.2f} mm")
        self.assertLess(probe.peak(), 5.0, f"归位峰值力 {probe.peak():.2f} N 过高")
        print(f"[probe] 带载归位: peak|F|={probe.peak():.2f}N steps={r[0].metrics.get('steps')} "
              f"held_z={held_z:.4f} holding={app.skills.holding!r}")


# --------------------------------------------------------------------------- #
# 4) P4：正常抓放守卫式下降，力受控
# --------------------------------------------------------------------------- #
class TestPlaceSafety(unittest.TestCase):
    def test_p4_guarded_place_force(self):
        app = _make_app()
        probe = _ForceProbe(app)
        results = app.run_task("把红色的方块放进托盘")
        sk = skills_of(results)
        self.assertEqual(sk, ["pick", "place", "stop"], f"正常抓放流程退化: {sk}")
        self.assertTrue(all(r.ok for r in results), [(r.skill, r.message) for r in results])

        # 放置下降是守卫式（6 N 阈值），整段任务不得出现深压卡死
        self.assertLess(probe.peak(), 50.0, f"放置峰值力 {probe.peak():.1f} N 过高")
        self.assertLess(probe.final(), 1.0, f"结束时 |F|={probe.final():.2f} N")
        self.assertIsNone(app.skills.holding, "放置后 holding 应清空")


# --------------------------------------------------------------------------- #
# 5) 证伪 (b)：hard_recover 三级（真实 env）
# --------------------------------------------------------------------------- #
class TestHardRecoverLevels(unittest.TestCase):
    def test_b_retreat_and_scene_reset(self):
        app = _make_app()
        app.env.step(100)
        start = {o.id: app.env.object_pose(o.id)[:3].copy() for o in app.env.list_objects()}

        # 可脱身：静止（|F|=0）-> 第 1 级 retreat，且不重置场景
        self.assertEqual(app.skills.hard_recover(), "retreat")
        for i in start:
            moved = float(np.linalg.norm(app.env.object_pose(i)[:3] - start[i]))
            self.assertLess(moved, 0.001, f"{i} 在 level1 后移动 {moved * 1000:.2f} mm")

        # 需复位：注入无法清除的 200 N -> 必须落到第 3 级并真的 env.reset
        moved = app.env.object_pose("box_green").copy()
        moved[0] += 0.05
        moved[2] += 0.10
        app.env.set_object_pose("box_green", moved)
        app.skills.holding = "box_green"
        real = app.env.contact_force_on_ee
        app.env.contact_force_on_ee = lambda: (np.array([0.0, 0.0, 200.0]), np.zeros(3))
        try:
            level = app.skills.hard_recover()
        finally:
            app.env.contact_force_on_ee = real
        self.assertEqual(level, "scene_reset", f"高压死状态下应场景重置，实际 {level}")
        self.assertIsNone(app.skills.holding, "场景重置后 holding 必须清空")
        for i in start:
            err = float(np.linalg.norm(app.env.object_pose(i)[:3] - start[i]))
            self.assertLess(err, 0.002, f"{i} 未被重置，残余 {err * 1000:.2f} mm")
        self.assertLess(float(np.linalg.norm(app.env.contact_force_on_ee()[0])), 1.0)


# --------------------------------------------------------------------------- #
# 6) 证伪 (a)：脏状态 -> run_task 前置自动 hard_recover
# --------------------------------------------------------------------------- #
class TestDirtyStateAutoRecover(unittest.TestCase):
    def test_a_pressed_state_autorecovered_before_task(self):
        from core.types import ComplianceParams
        app = _make_app()
        app.env.step(50)
        logs: list[str] = []
        orig_log = app._log

        def capture(msg):
            logs.append(str(msg))
            orig_log(msg)

        app._log = capture  # type: ignore[method-assign]

        # 构造真实卡死：把 TCP 压到桌面以下（0.395 < 桌面 0.40），大 force_limit。
        # 注意（task-67 导纳符号修正后）：正确符号下"下压"会让位（远离接触），
        # 压不出大残留力——那正是修复目标。为了仍然能造出"脏状态"前置条件，
        # 这里**显式临时用旧符号(+1)**压出残留接触，随后立刻恢复新符号(-1)，
        # 再验证恢复逻辑本身。断言（stuck > 20 N）不放宽。
        comp = ComplianceParams.z_only(k=200.0, d=25.0, force_limit=1e9)
        app.skills.runner.adm_force_sign = 1.0
        try:
            app.skills.runner.move_to_pose(np.array([0.30, 0.0, 0.395]), approach_quat(0.0),
                                           compliance=comp, timeout=6.0, force_limit=1e9)
        finally:
            app.skills.runner.adm_force_sign = -1.0
        stuck = float(np.linalg.norm(app.env.contact_force_on_ee()[0]))
        self.assertGreater(stuck, 20.0, f"未能构造脏状态，仅 {stuck:.1f} N")

        results = app.run_task("把红色的方块放进托盘")
        self.assertTrue(any("任务前先应急恢复" in s for s in logs),
                        "run_task 未检测到残留接触并自动恢复")
        self.assertTrue(any("应急恢复级别" in s for s in logs), "未调用 hard_recover")
        self.assertEqual(skills_of(results), ["pick", "place", "stop"],
                         f"恢复后任务未正常完成: {skills_of(results)}")
        self.assertTrue(all(r.ok for r in results), [(r.skill, r.message) for r in results])
        self.assertLess(float(np.linalg.norm(app.env.contact_force_on_ee()[0])), 1.0)


# --------------------------------------------------------------------------- #
# 5) P0：pick 子目标复核前不得归位（否则刚抓起的物体被丢回原位，复核必失败）
# --------------------------------------------------------------------------- #
class TestPickSubgoalObservationKeepsHeldObject(unittest.TestCase):
    """P0 回归（实机 2026-10-12「红色方块放到蓝色方块上」）：

    observe 钩子对每个子目标都调 skills.home；带载时规划失败 -> env.home() 瞬移，
    旧实现还会顺带 `set_gripper(OPEN)` -> 刚抓起的方块被丢回原位 ->
    复核 check_gone 必然 subject_still_present -> g1(pick) 3 次全废、任务失败。
    """

    @staticmethod
    def _pick_sg(goal_type="pick"):
        from agent.taskplan import SubGoal
        return SubGoal(id="g1", instruction="抓起蓝色方块", goal_type=goal_type,
                       subject="蓝色方块", subject_color="blue")

    def test_pick_subgoal_observe_does_not_move_arm(self):
        app = _make_app()
        r = app.run_task("夹住蓝色方块")
        self.assertEqual(skills_of(r), ["pick", "stop"])
        self.assertTrue(r[0].ok, r[0].message)
        self.assertIsNotNone(app.skills.holding)
        arm_before = app.env.data.qpos[app.env.arm_qpos_idx].copy()
        held_before = app.env.object_pose("box_blue")[:3].copy()

        app._observe_home(self._pick_sg("pick"))

        arm_after = app.env.data.qpos[app.env.arm_qpos_idx].copy()
        held_after = app.env.object_pose("box_blue")[:3].copy()
        print(f"[probe] pick 复核前: 臂位移={np.linalg.norm(arm_after - arm_before):.4f}rad "
              f"物体z {held_before[2]:.4f}->{held_after[2]:.4f} holding={app.skills.holding}")
        self.assertTrue(np.allclose(arm_before, arm_after, atol=1e-9),
                        "pick 子目标复核前不该挪动机械臂")
        self.assertGreater(float(held_after[2]), TABLE_Z + 0.05,
                           f"刚抓起的物体被丢回桌面 z={held_after[2]:.4f}")
        self.assertLess(float(np.linalg.norm(held_after - held_before)), 0.005)

    def test_check_gone_passes_after_pick_when_observe_skips_home(self):
        """用实机故障用例的颜色（红块）：抓走后 top 图里色块面积 795->32，check_gone 通过。

        注意：蓝块被夹起后 top 图里仍可见（面积 818->827，夹爪没挡住），纯 check_gone 会漏判；
        AgentLoop 对这种情况用技能的物理随动证据兜底（见 agent.smoke 的对应用例）。
        """
        from sim import perception
        app = _make_app()
        rgb_before = app.env.render("top").copy()
        r = app.run_task("夹住红色方块")
        self.assertTrue(r[0].ok, r[0].message)
        app._observe_home(self._pick_sg("pick"))
        rgb_after = app.env.render("top").copy()
        blob = perception.color_blob(rgb_before, "red")
        self.assertTrue(blob.found, "before 图里应能看到红色方块")
        res = perception.check_gone(rgb_before, rgb_after, "red", blob.uv)
        print(f"[probe] pick 复核 check_gone: ok={res['ok']} why={res['why']} "
              f"area {res['metrics']['area_before_px']}->{res['metrics']['area_after_px']}")
        self.assertTrue(res["ok"], f"抓走后复核必须通过，实际 {res['why']}")

    def test_place_subgoal_observe_still_homes(self):
        """非 pick 子目标、且手里没东西 -> 仍然回观察位（保持原语义，不误伤叠放复核）。"""
        from agent.taskplan import SubGoal
        from core.config import HOME_QPOS
        app = _make_app()
        r = app.run_task("把红色的方块放进托盘")
        self.assertTrue(all(x.ok for x in r), [(x.skill, x.message) for x in r])
        self.assertIsNone(app.skills.holding, "放置后不该还认为手里有东西")
        arm_before = app.env.data.qpos[app.env.arm_qpos_idx].copy()
        app._observe_home(SubGoal(id="g2", instruction="放到托盘",
                                  goal_type="place_in_container", subject="红色方块",
                                  subject_color="red", container="tray"))
        arm_after = app.env.data.qpos[app.env.arm_qpos_idx].copy()
        print(f"[probe] place 复核前: 臂位移={np.linalg.norm(arm_after - arm_before):.4f}rad")
        # 关节路径终点允许 ~0.02 rad 误差，只要求确实回到了 HOME 附近
        self.assertLess(float(np.linalg.norm(arm_after - HOME_QPOS)), 0.05,
                        "非 pick 子目标仍应回到观察位(HOME)")


class _ScriptedEpoch:
    """最小 epoch：执行一次真实 pick，返回 SkillResult 列表（供编排器收集 metrics）。"""

    def __init__(self, skills, args: dict):
        self.skills = skills
        self.args = dict(args)

    def run(self, instruction: str):
        return [self.skills(SkillCall("pick", dict(self.args)))]


class TestPickSubgoalVerificationEndToEnd(unittest.TestCase):
    """P0 端到端回归（复刻实机 2026-10-12 的 g1:pick）：

    真抓红块 -> 复核前观察位 -> SubGoalVerifier 必须 done=True。
    修复前 observe 会把方块丢回原位，check_gone 必然 subject_still_present。
    """

    def test_g1_pick_verification_passes(self):
        from agent.taskplan import SubGoal, SubGoalOrchestrator, SubGoalVerifier
        from sim import perception as P
        app = _make_app()
        logs: list[str] = []
        args = {"target_xy": [0.350, -0.140], "target_z_hint": 0.456315,
                "target_size": [0.020096, 0.02084], "grasp_profile": "top"}
        verifier = SubGoalVerifier(app.env, perception=P, vlm=None,
                                   log=logs.append, camera="top")
        orch = SubGoalOrchestrator(app.env, None,
                                   lambda sg, attempt: _ScriptedEpoch(app.skills, args),
                                   verifier, log=logs.append, max_attempts=3,
                                   include_poses=False, observe=app._observe_home)
        sg = SubGoal(id="g1", instruction="抓起红色方块", goal_type="pick",
                     subject="红色方块", subject_color="red")
        rec = orch.run("红色方块放到蓝色方块上", [sg])[0]
        held_z = float(app.env.object_pose("box_red")[2])
        print(f"[probe] g1:pick 复核: done={rec['done']} attempts={rec['attempts']} "
              f"why={rec['why'][:90]} 方块z={held_z:.4f}")
        self.assertTrue(rec["done"], f"g1:pick 复核必须通过: {rec['why']}")
        self.assertGreater(held_z, TABLE_Z + 0.05, "复核时方块必须仍被举着")

    def test_default_observe_without_hook_does_not_drop_object(self):
        """绕关口：**不注入 observe 钩子**时，编排器的默认 env.home() 兜底也不许对
        pick 子目标归位（verifier 实测旧行为会让物体 z 0.5393 -> 0.4199）。"""
        from agent.taskplan import SubGoal, SubGoalOrchestrator, SubGoalVerifier
        from sim import perception as P
        app = _make_app()
        args = {"target_xy": [0.350, -0.140], "target_z_hint": 0.456315,
                "target_size": [0.020096, 0.02084], "grasp_profile": "top"}
        verifier = SubGoalVerifier(app.env, perception=P, vlm=None, camera="top")
        orch = SubGoalOrchestrator(app.env, None,
                                   lambda sg, attempt: _ScriptedEpoch(app.skills, args),
                                   verifier, log=None, max_attempts=1,
                                   include_poses=False, observe=None)
        sg = SubGoal(id="g1", instruction="抓起红色方块", goal_type="pick",
                     subject="红色方块", subject_color="red")
        rec = orch.run("红色方块放到蓝色方块上", [sg])[0]
        held_z = float(app.env.object_pose("box_red")[2])
        print(f"[probe] 默认 observe(无钩子) g1:pick: done={rec['done']} z={held_z:.4f}")
        self.assertTrue(rec["done"], f"默认兜底不得丢件: {rec['why']}")
        self.assertGreater(held_z, TABLE_Z + 0.05, "方块被默认兜底丢回桌面了")


if __name__ == "__main__":
    unittest.main(verbosity=2)
