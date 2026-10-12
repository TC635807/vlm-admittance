"""tests/test_target_overlay.py —— 「当前机械臂目标点」渲染的独立验证（标准库 unittest）。

覆盖 task-19 的全部必测项：
  1) 无显示环境几何：MjvScene + overlay 的 ngeom/类型/位置/边界(maxgeom=0/1/2)、
     label 截断、mjv_initGeom 的 mat 必填 与 size 重排（独立断言，不照抄实现者）。
  2) 顺序正确性：clear -> runner.target_state -> draw_target -> viewer.sync
     （运行时探针 + 源码静态顺序检查）。
  3) 真 GL 窗口端到端：真 launch_passive 窗口跑抓放任务，统计每帧 user_scn.ngeom，
     并验证 :target off 后 ngeom 恒为 0。
  4) target_state 语义：自由空间积分补偿、注入外力后的纯导纳让位量 F/k。
  5) 容错：draw_target 抛异常 / overlay 缺失时只提示一条且任务照常完成。
  6) 证伪：700 帧无 ngeom 泄漏/越界；:target bogus 与无参数 :target；
     valid=False 时静默跳过（绝不画 (0,0,0) 假目标）。

task-18 追加语义（本文件一并覆盖）
---------------------------------
  * _pure_yield(state) = |adm_pos - ref_pos|；无 adm_pos 时退回 adm_delta（legacy）。
  * 橙色 ref 自动触发条件 = pure_yield > 5mm（'on' 模式则始终显示基线 ref）；
    紫红 adm 标记（radius 0.008, _ADM_MARKER_RGBA）= adm_pos 且 pure_yield > 1mm。
  * 新接口 / legacy 两条路径都覆盖；并证伪「自由空间运动不得误报让位」。

运行（无显示环境也能跑；真窗口用例在无法启动 viewer 时自动 skip）：
    cd /home/tc63/mujoco/vlm_admittance
    MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m tests.test_target_overlay
严格让位门（复现「自由空间 pure_yield 误报」finding）：
    VLM_STRICT_TARGET_YIELD=1 MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m tests.test_target_overlay
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np
import mujoco

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim import overlay                                   # noqa: E402
from sim.env import MuJoCoArmEnv                          # noqa: E402
from control.kinematics import Kinematics                 # noqa: E402
from control.admittance import AdmittanceController       # noqa: E402
from control.motion import MotionRunner                   # noqa: E402
from core.config import DEFAULT_COMPLIANCE                # noqa: E402
from skills import approach_quat                          # noqa: E402

SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)

#: 严格门（可选）：把「自由空间运动时 pure_yield 误报」判为失败，用于复现 finding。
STRICT_YIELD = os.environ.get("VLM_STRICT_TARGET_YIELD", "").strip().lower() not in (
    "", "0", "false", "no")


def _mat3(geom) -> np.ndarray:
    return np.asarray(geom.mat, dtype=float).reshape(3, 3)


def _empty_model():
    return mujoco.MjModel.from_xml_string("<mujoco/>")


def _make_app(headless: bool = True):
    from demo.interactive import InteractiveApp
    args = argparse.Namespace(vlm="mock", with_barrier=False, seed=0, realtime=False,
                              headless=headless, with_images=False, max_steps=12, self_test=False)
    return InteractiveApp(args)


def _capture_log(app) -> list:
    lines: list = []
    orig = app._log

    def cap(msg):
        lines.append(str(msg))
        orig(msg)

    app._log = cap  # type: ignore[method-assign]
    return lines


# --------------------------------------------------------------------------- #
# 1) 无显示环境几何单测
# --------------------------------------------------------------------------- #
class TestOverlayGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _empty_model()

    def test_clear_resets_ngeom_only(self):
        scn = mujoco.MjvScene(self.model, maxgeom=8)
        scn.ngeom = 5
        overlay.clear(scn)
        self.assertEqual(scn.ngeom, 0)

    def test_maxgeom_zero_no_crash(self):
        scn = mujoco.MjvScene(self.model, maxgeom=0)
        self.assertEqual(len(scn.geoms), 0)
        n = overlay.draw_target(scn, np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(n, 0)
        self.assertEqual(scn.ngeom, 0)
        self.assertEqual(overlay.draw_trail(scn, [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), 0)
        with self.assertRaises(IndexError):
            _ = scn.geoms[0]

    def test_maxgeom_one_and_two_truncate_without_overflow(self):
        pos = np.array([0.11, -0.22, 0.55])
        scn1 = mujoco.MjvScene(self.model, maxgeom=1)
        n1 = overlay.draw_target(scn1, pos, np.array([1.0, 0.0, 0.0, 0.0]),
                                 ref_pos=pos + np.array([0.0, 0.0, 0.1]))
        self.assertEqual((n1, scn1.ngeom), (1, 1))
        self.assertEqual(int(scn1.geoms[0].type), SPHERE)
        self.assertTrue(np.allclose(np.asarray(scn1.geoms[0].pos), pos, atol=1e-9))

        scn2 = mujoco.MjvScene(self.model, maxgeom=2)
        n2 = overlay.draw_target(scn2, pos, np.array([1.0, 0.0, 0.0, 0.0]),
                                 ref_pos=pos + np.array([0.0, 0.0, 0.1]))
        self.assertEqual((n2, scn2.ngeom), (2, 2))
        self.assertEqual([int(scn2.geoms[i].type) for i in range(2)], [SPHERE, CAPSULE])

    def test_return_value_is_actual_written_and_stable(self):
        scn = mujoco.MjvScene(self.model, maxgeom=3)
        pos = np.array([0.4, 0.02, 0.62])
        for _ in range(50):
            overlay.clear(scn)                     # 每帧先 clear（与 UI 一致）
            n = overlay.draw_target(scn, pos, np.array([1.0, 0.0, 0.0, 0.0]),
                                    ref_pos=pos + np.array([0.0, 0.0, 0.05]))
            self.assertEqual(n, 3)                 # 需要 6 个，容量 3 -> 实际 3
            self.assertEqual(scn.ngeom, 3)
            self.assertLessEqual(scn.ngeom, scn.maxgeom)

    def test_mjv_initgeom_mat_is_required(self):
        """独立验证「mat 必填」：本机 3.12 传 None 必须抛异常。"""
        scn = mujoco.MjvScene(self.model, maxgeom=4)
        geom = scn.geoms[0]
        with self.assertRaises(TypeError):
            mujoco.mjv_initGeom(geom, SPHERE, np.array([0.01, 0.0, 0.0]),
                                np.array([0.0, 0.0, 0.5]), None,
                                np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))

    def test_size_is_rearranged_on_readback(self):
        """独立验证「size 会被重排」：sphere -> (r,r,r)；capsule -> (r,r,half_len)。"""
        scn = mujoco.MjvScene(self.model, maxgeom=8)
        pos = np.array([0.0, 0.0, 0.5])
        overlay.draw_target(scn, pos, np.array([1.0, 0.0, 0.0, 0.0]))
        sph_size = np.asarray(scn.geoms[0].size, dtype=float)
        self.assertTrue(np.allclose(sph_size, [overlay.DEFAULT_RADIUS] * 3, atol=1e-9),
                        f"sphere size 读回 {sph_size}")
        cap_size = np.asarray(scn.geoms[1].size, dtype=float)
        r_ax = overlay.DEFAULT_RADIUS * overlay.AXES_RADIUS_SCALE
        # 注意：geom 缓冲是 float32，比较用 places=6 而不是 9
        self.assertAlmostEqual(float(cap_size[0]), r_ax, places=6)
        self.assertAlmostEqual(float(cap_size[1]), r_ax, places=6)
        self.assertAlmostEqual(float(cap_size[2]), overlay.DEFAULT_AXES_LEN / 2.0, places=6)

    def test_capsule_orientation_and_center(self):
        """三轴胶囊：第 3 列 = 局部 +z 的世界方向 = 轴的方向；中心 = p 到 tip 的中点。"""
        scn = mujoco.MjvScene(self.model, maxgeom=8)
        pos = np.array([0.40, 0.02, 0.62])
        overlay.draw_target(scn, pos, np.array([1.0, 0.0, 0.0, 0.0]))
        for k, expect in enumerate(np.eye(3)):
            g = scn.geoms[1 + k]
            axis_world = _mat3(g)[:, 2]
            self.assertTrue(np.allclose(axis_world, expect, atol=1e-6),
                            f"轴 {k} 方向 {axis_world} != {expect}")
            center = np.asarray(g.pos, dtype=float)
            want = pos + expect * (overlay.DEFAULT_AXES_LEN / 2.0)
            self.assertTrue(np.allclose(center, want, atol=1e-9))

    def test_ref_point_and_link_layout(self):
        scn = mujoco.MjvScene(self.model, maxgeom=16)
        pos = np.array([0.2, 0.1, 0.5])
        ref = pos + np.array([0.0, 0.0, 0.03])
        n = overlay.draw_target(scn, pos, np.array([1.0, 0.0, 0.0, 0.0]), ref_pos=ref)
        self.assertEqual(n, 6)                      # 球 + 3 轴 + ref 球 + 连线
        self.assertEqual(int(scn.geoms[4].type), SPHERE)
        self.assertTrue(np.allclose(np.asarray(scn.geoms[4].pos), ref, atol=1e-9))
        self.assertEqual(int(scn.geoms[5].type), CAPSULE)
        # 重合的 ref 不画
        scn2 = mujoco.MjvScene(self.model, maxgeom=16)
        self.assertEqual(overlay.draw_target(scn2, pos, None, ref_pos=pos.copy()), 1)

    def test_label_truncated_to_99(self):
        scn = mujoco.MjvScene(self.model, maxgeom=2)
        overlay.draw_target(scn, np.zeros(3))
        self.assertEqual(scn.geoms[0].label, "target")
        long = "x" * 250
        overlay._decorate(scn.geoms[0], long)
        self.assertLessEqual(len(scn.geoms[0].label), overlay.LABEL_MAX)
        self.assertEqual(len(scn.geoms[0].label), overlay.LABEL_MAX)
        self.assertLessEqual(overlay.LABEL_MAX, 99)

    def test_trail_counts_and_bounds(self):
        scn = mujoco.MjvScene(self.model, maxgeom=64)
        pts = np.array([[0.0, 0.0, 0.5], [0.05, 0.0, 0.5], [0.05, 0.05, 0.5]])
        n = overlay.draw_trail(scn, pts)
        self.assertEqual(n, 5)                      # 3 球 + 2 段
        self.assertEqual(overlay.draw_trail(scn, []), 0)
        with self.assertRaises(ValueError):
            overlay.draw_trail(scn, np.zeros((2, 2)))


# --------------------------------------------------------------------------- #
# 2) 顺序正确性
# --------------------------------------------------------------------------- #
class TestOverlayOrder(unittest.TestCase):
    def tearDown(self):
        import demo.interactive as di
        di._overlay_clear = self._saved_clear
        di._overlay_draw_target = self._saved_draw

    def setUp(self):
        import demo.interactive as di
        self.di = di
        self._saved_clear = di._overlay_clear
        self._saved_draw = di._overlay_draw_target

    def test_runtime_order_clear_state_draw(self):
        app = _make_app()
        order: list = []
        self.di._overlay_clear = lambda scn: order.append("clear")
        self.di._overlay_draw_target = lambda *a, **k: (order.append("draw"), 1)[1]

        def fake_state():
            order.append("target_state")
            return {"valid": True, "cmd_pos": np.zeros(3), "cmd_quat": np.array([1.0, 0, 0, 0]),
                    "ref_pos": np.zeros(3), "adm_pos": np.zeros(3),
                    "adm_quat": np.array([1.0, 0, 0, 0]), "adm_delta": 0.0}

        app.runner.target_state = fake_state  # type: ignore[method-assign]
        app._update_target_overlay(object())
        self.assertEqual(order, ["clear", "target_state", "draw"])

    def test_off_mode_only_clears(self):
        app = _make_app()
        order: list = []
        self.di._overlay_clear = lambda scn: order.append("clear")
        self.di._overlay_draw_target = lambda *a, **k: (order.append("draw"), 1)[1]
        app.target_mode = "off"
        app._update_target_overlay(object())
        self.assertEqual(order, ["clear"])

    def test_main_loop_source_order_before_sync(self):
        src = (_ROOT / "demo" / "interactive.py").read_text(encoding="utf-8")
        tail = src[src.index("viewer = mujoco.viewer.launch_passive"):]
        i_draw = tail.index("_update_target_overlay(viewer.user_scn)")
        i_sync = tail.index("viewer.sync()")
        self.assertLess(i_draw, i_sync, "主循环必须在 viewer.sync() 之前画目标点")
        seg = tail[max(0, i_draw - 300):i_draw]
        self.assertIn("with self.lock", seg, "绘制应发生在 self.lock 临界区内")


# --------------------------------------------------------------------------- #
# 2b) 让位标记规则：唯一触发条件 = force_norm > _YIELD_FORCE_N（task-21）
# --------------------------------------------------------------------------- #
def _record_draw(app, state, mode="axes"):
    """录下 _update_target_overlay 对 draw_target 的每次调用（不真的画）。"""
    import demo.interactive as di
    calls: list = []
    saved_clear, saved_draw = di._overlay_clear, di._overlay_draw_target
    di._overlay_clear = lambda scn: None

    def rec(scn, pos, quat=None, **kw):
        calls.append({"pos": np.asarray(pos, dtype=float).copy(), "quat": quat, "kw": dict(kw)})
        return 1

    di._overlay_draw_target = rec
    app.target_mode = mode
    app.runner.target_state = lambda: state
    try:
        app._update_target_overlay(object())
    finally:
        di._overlay_clear, di._overlay_draw_target = saved_clear, saved_draw
    return calls


def _state(force_norm=None, adm_off=0.0, quat_delta=0.0, include_force=True):
    """构造 target_state 快照；force_norm=None 表示无外力（0.0）。"""
    q_adm = np.array([1.0, 0.0, 0.0, 0.0])
    if abs(quat_delta) > 0.0:
        q_adm = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_adm, np.array([0.0, 0.0, 1.0]), float(quat_delta))
    st = {"valid": True,
          "cmd_pos": np.array([0.0, 0.0, 0.612]), "cmd_quat": np.array([1.0, 0, 0, 0]),
          "ref_pos": np.array([0.0, 0.0, 0.60]), "ref_quat": np.array([1.0, 0, 0, 0]),
          "adm_pos": np.array([0.0, 0.0, 0.60 + adm_off]), "adm_quat": q_adm,
          "adm_delta": 0.012}
    if include_force:
        st["force_norm"] = 0.0 if force_norm is None else float(force_norm)
    return st


def _draw_flags_for_states(app, states, mode="axes"):
    """按 UI 真实规则逐帧判定，返回 [(has_ref_marker, has_adm_marker, n_calls), ...]。"""
    import demo.interactive as di
    saved_clear, saved_draw = di._overlay_clear, di._overlay_draw_target
    di._overlay_clear = lambda scn: None
    cur: list = []
    adm_rgb = tuple(di._ADM_MARKER_RGBA[:3])

    def rec(scn, pos, quat=None, **kw):
        rgba = kw.get("rgba")
        is_adm = rgba is not None and tuple(np.asarray(rgba, dtype=float)[:3]) == adm_rgb
        cur.append((bool(kw.get("show_ref")), bool(is_adm)))
        return 1

    di._overlay_draw_target = rec
    app.target_mode = mode
    out = []
    try:
        for st in states:
            cur.clear()
            app.runner.target_state = lambda _s=st: _s  # type: ignore[method-assign]
            app._update_target_overlay(object())
            out.append((any(c[0] for c in cur), any(c[1] for c in cur), len(cur)))
    finally:
        di._overlay_clear, di._overlay_draw_target = saved_clear, saved_draw
    return out


class TestAdmMarkerLogic(unittest.TestCase):
    def setUp(self):
        import demo.interactive as di
        self.di = di
        self.thr = float(di._YIELD_FORCE_N)

    def test_force_threshold_is_sane(self):
        self.assertGreater(self.thr, 0.0)
        self.assertLessEqual(self.thr, 2.5)

    def test_free_space_cyan_only_no_markers(self):
        for fn in (0.0, self.thr * 0.5):
            with self.subTest(force_norm=fn):
                calls = _record_draw(_make_app(), _state(force_norm=fn, adm_off=0.05), mode="axes")
                self.assertEqual(len(calls), 1, "无外力时只画青色目标")
                self.assertFalse(calls[0]["kw"]["show_ref"])

    def test_contact_above_threshold_shows_ref_and_magenta(self):
        calls = _record_draw(_make_app(), _state(force_norm=self.thr + 5.0, adm_off=0.04), mode="axes")
        self.assertEqual(len(calls), 2, "外力超阈值应补画橙色 ref + 紫红 adm")
        self.assertTrue(calls[0]["kw"]["show_ref"])
        self.assertEqual(tuple(np.asarray(calls[1]["kw"]["rgba"])[:3]),
                         tuple(self.di._ADM_MARKER_RGBA[:3]))
        self.assertAlmostEqual(float(calls[1]["kw"]["radius"]), 0.008, places=9)
        self.assertFalse(calls[1]["kw"]["show_ref"])

    def test_threshold_is_strict_greater(self):
        at = _record_draw(_make_app(), _state(force_norm=self.thr), mode="axes")
        self.assertEqual(len(at), 1, "force_norm == 阈值时不应触发（判据是严格 >）")
        self.assertFalse(at[0]["kw"]["show_ref"])
        above = _record_draw(_make_app(), _state(force_norm=self.thr + 0.01), mode="axes")
        self.assertEqual(len(above), 2)

    def test_on_mode_no_ref_without_force(self):
        calls = _record_draw(_make_app(), _state(force_norm=0.0, adm_off=0.05), mode="on")
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["kw"]["show_ref"], "无外力时 on 模式也不得显示橙色 ref")

    def test_sphere_mode_contact_marks_adm(self):
        calls = _record_draw(_make_app(), _state(force_norm=self.thr + 5.0), mode="sphere")
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["quat"], "sphere 模式不画三轴")
        self.assertTrue(calls[0]["kw"]["show_ref"])

    def test_translation_offset_alone_never_triggers(self):
        """旧误报根因回归：|adm-ref|=50mm 但 force_norm=0 -> 绝不显示让位标记。"""
        calls = _record_draw(_make_app(), _state(force_norm=0.0, adm_off=0.05), mode="axes")
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["kw"]["show_ref"])

    def test_rotation_offset_alone_never_triggers(self):
        app = _make_app()
        st = _state(force_norm=0.0, quat_delta=0.2)
        self.assertAlmostEqual(app._pure_rot_yield(st), 0.2, places=4)
        calls = _record_draw(app, st, mode="axes")
        self.assertEqual(len(calls), 1, "只有旋转差、无外力时也不显示让位标记")
        self.assertFalse(calls[0]["kw"]["show_ref"])

    def test_rotation_axes_shown_when_contact(self):
        calls = _record_draw(_make_app(), _state(force_norm=self.thr + 5.0, quat_delta=0.2), mode="axes")
        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(calls[1]["quat"], "接触且旋转让位大时紫红点应带 adm_quat 三轴")

    def test_rotation_tools_missing_skips_silently(self):
        import demo.interactive as di
        st = _state(force_norm=self.thr + 5.0, quat_delta=0.2)
        saved = (di._quat_mul, di._quat_conj, di._rotvec_from_quat)
        di._quat_mul = di._quat_conj = di._rotvec_from_quat = None
        try:
            app = _make_app()
            self.assertEqual(app._pure_rot_yield(st), 0.0)
            calls = _record_draw(app, st, mode="axes")
        finally:
            di._quat_mul, di._quat_conj, di._rotvec_from_quat = saved
        self.assertEqual(len(calls), 2, "工具缺失只跳过旋转三轴，不抛异常")
        self.assertIsNone(calls[1]["quat"])

    def test_legacy_missing_force_norm_no_markers_one_notice(self):
        app = _make_app()
        logs = _capture_log(app)
        st = _state(include_force=False, adm_off=0.05)
        calls = _record_draw(app, st, mode="axes")
        calls2 = _record_draw(app, st, mode="axes")
        self.assertEqual(len(calls), 1, "缺 force_norm 时只画青色目标")
        self.assertFalse(calls[0]["kw"]["show_ref"])
        self.assertEqual(len(calls2), 1)
        self.assertEqual(len([l for l in logs if "force_norm" in l]), 1, "legacy 只提示一次")

    def test_without_adm_pos_no_magenta(self):
        st = _state(force_norm=self.thr + 5.0)
        st.pop("adm_pos")
        calls = _record_draw(_make_app(), st, mode="axes")
        self.assertEqual(len(calls), 1, "无 adm_pos -> 只有青色目标（+橙色 ref）")
        self.assertTrue(calls[0]["kw"]["show_ref"])


# 3) valid=False：静默跳过，绝不画 (0,0,0) 假目标（重点证伪）
# --------------------------------------------------------------------------- #
class TestValidFalseNoFakeTarget(unittest.TestCase):
    def test_fresh_runner_invalid_and_nothing_drawn(self):
        model = _empty_model()
        scn = mujoco.MjvScene(model, maxgeom=64)
        app = _make_app()
        self.assertFalse(app.runner.target_state()["valid"], "从未运动的 runner 应 valid=False")
        app._update_target_overlay(scn)
        self.assertEqual(scn.ngeom, 0, "valid=False 时不得画任何（尤其 (0,0,0) 假目标）")
        # 再确认没有 draw 被调用
        import demo.interactive as di
        called = {"n": 0}
        saved = di._overlay_draw_target
        di._overlay_draw_target = lambda *a, **k: called.__setitem__("n", called["n"] + 1)
        try:
            app._update_target_overlay(scn)
        finally:
            di._overlay_draw_target = saved
        self.assertEqual(called["n"], 0)
        self.assertEqual(scn.ngeom, 0)

    def test_zeros_never_drawn_even_if_draw_called(self):
        """补充：即使 valid 为真但 cmd_pos 缺失/None，也应跳过而不是画原点。"""
        app = _make_app()
        model = _empty_model()
        scn = mujoco.MjvScene(model, maxgeom=64)
        app.runner.target_state = lambda: {"valid": True, "cmd_pos": None,
                                           "cmd_quat": None, "adm_delta": 0.0}
        app._update_target_overlay(scn)
        self.assertEqual(scn.ngeom, 0)


# --------------------------------------------------------------------------- #
# 4) target_state 语义
# --------------------------------------------------------------------------- #
class _MoveSampler:
    """跑一次笛卡尔 move，逐步采样 runner.target_state()（只读）。"""

    def __init__(self, inject=None):
        self.env = MuJoCoArmEnv()
        self.kin = Kinematics(self.env)
        self.runner = MotionRunner(self.env, self.kin,
                                   AdmittanceController(DEFAULT_COMPLIANCE, self.env.dt))
        self.samples: list = []
        base = self.env.step

        def wrapped(n: int = 1):
            st = base(n)
            s = self.runner.target_state()
            self.samples.append({k: (np.array(v, dtype=float) if isinstance(v, np.ndarray) else v)
                                 for k, v in s.items()})
            return st

        self.env.step = wrapped  # type: ignore[method-assign]
        if inject is not None:
            self.env.contact_force_on_ee = lambda: (np.asarray(inject, float), np.zeros(3))

    def run(self):
        p, _ = self.env.site_pose("g_pinch")
        return self.runner.move_to_pose(p + np.array([0.05, 0.0, 0.05]),
                                        approach_quat(0.0), timeout=8.0)

    def metrics(self):
        v = [s for s in self.samples if s["valid"]]
        cmd_ref = np.array([np.linalg.norm(s["cmd_pos"] - s["ref_pos"]) for s in v])
        adm_ref = np.array([np.linalg.norm(s["adm_pos"] - s["ref_pos"]) for s in v])
        delta = np.array([float(s["adm_delta"]) for s in v])
        return v, cmd_ref, adm_ref, delta

    def rot_yields(self):
        """用 UI 的真实实现算逐步姿态让位角（rad）。"""
        import demo.interactive as di
        v = [s for s in self.samples if s["valid"]]
        return np.array([di.InteractiveApp._pure_rot_yield(s) for s in v])


class TestTargetStateSemantics(unittest.TestCase):
    def test_free_space_integral_compensation(self):
        s = _MoveSampler()
        r = s.run()
        self.assertTrue(r.ok, r.message)
        v, cmd_ref, adm_ref, delta = s.metrics()
        self.assertGreater(len(v), 50)
        m_cmd, m_adm, m_delta = (float(np.median(cmd_ref)), float(np.median(adm_ref)),
                                 float(np.median(delta)))
        print(f"\n[probe] 自由空间: |cmd-ref| med={m_cmd * 1000:.2f}mm  "
              f"|adm-ref| med={m_adm * 1000:.2f}mm  adm_delta med={m_delta * 1000:.2f}mm")
        # 真实下发点含积分静差补偿（mm 级）
        self.assertGreater(m_cmd, 0.005, f"cmd-ref={m_cmd * 1000:.2f}mm 未体现静差补偿")
        # adm_delta 明显大于纯导纳让位 |adm-ref|
        self.assertGreater(m_delta, m_adm,
                           f"adm_delta={m_delta * 1000:.2f}mm 应大于 |adm-ref|={m_adm * 1000:.2f}mm")

    def test_injected_force_pure_compliance(self):
        for fz in (8.0, -8.0):
            with self.subTest(fz=fz):
                s = _MoveSampler(inject=np.array([0.0, 0.0, fz]))
                r = s.run()
                self.assertTrue(r.ok, r.message)
                v, cmd_ref, adm_ref, delta = s.metrics()
                m_adm = float(np.median(adm_ref))
                z = np.array([(x["adm_pos"] - x["ref_pos"])[2] for x in v])
                k_z = float(DEFAULT_COMPLIANCE.k[2])
                expect = abs(fz) / k_z
                print(f"[probe] 注入 Fz={fz:+.1f}N: |adm-ref| med={m_adm * 1000:.2f}mm "
                      f"(F/k={expect * 1000:.2f}mm), adm-ref z 中位={np.median(z) * 1000:+.2f}mm")
                self.assertAlmostEqual(m_adm, expect, delta=0.008,
                                       msg=f"纯导纳让位应≈F/k={expect * 1000:.1f}mm")
                self.assertLess(float(np.median(z)) * fz, 0.0,
                                "让位方向应与外力相反（外力 +z 时 adm-ref z 为负）")

    def test_free_space_settles_to_no_yield(self):
        s = _MoveSampler()
        r = s.run()
        self.assertTrue(r.ok, r.message)
        v, cmd_ref, adm_ref, delta = s.metrics()
        tail = adm_ref[-50:]
        print(f"\n[probe] 自由空间停稳后 pure_yield: tail med={np.median(tail) * 1000:.2f}mm "
              f"max={tail.max() * 1000:.2f}mm")
        self.assertLess(float(tail.max()), 0.002, "停稳后 pure_yield 应≈0（无让位）")

    def test_free_space_motion_zero_marker_frames(self):
        """按 UI 实际规则：自由空间运动必须 0 帧出现橙色 ref / 紫红 adm（青色 target 允许）。"""
        from demo.interactive import _YIELD_FORCE_N
        s = _MoveSampler()
        r = s.run()
        self.assertTrue(r.ok, r.message)
        v = [x for x in s.samples if x["valid"]]
        frames = _draw_flags_for_states(_make_app(), v, mode="axes")
        n_ref = sum(1 for f in frames if f[0])
        n_adm = sum(1 for f in frames if f[1])
        fn = np.array([float(x["force_norm"]) for x in v])
        print(f"\n[probe] 自由空间 {len(frames)} 帧: force_norm max={fn.max():.6f}N "
              f"橙色ref标记={n_ref}帧 紫红adm标记={n_adm}帧（阈值 {_YIELD_FORCE_N}N）")
        self.assertEqual(n_ref, 0, f"自由空间出现橙色 ref {n_ref}/{len(frames)} 帧")
        self.assertEqual(n_adm, 0, f"自由空间出现紫红 adm {n_adm}/{len(frames)} 帧")

    def test_free_space_rotation_yield_is_zero(self):
        """自由空间 z_only 不使能转动：姿态让位必须≈0，不得触发旋转标记。"""
        s = _MoveSampler()
        r = s.run()
        self.assertTrue(r.ok, r.message)
        rot = s.rot_yields()
        print(f"[probe] 自由空间 rot_yield: med={np.median(rot):.4f} max={rot.max():.4f} rad")
        self.assertLess(float(rot.max()), 0.05, "自由空间不应出现旋转让位")

    def test_real_contact_yield_exceeds_threshold(self):
        """接触阶段应出现标记：真实压桌面时 pure_yield 必须 >5mm（力上限 20N）。"""
        from core.types import ComplianceParams
        env = MuJoCoArmEnv()
        kin = Kinematics(env)
        runner = MotionRunner(env, kin, AdmittanceController(ComplianceParams.z_only(), env.dt))
        rows: list = []
        base = env.step

        def wrapped(n: int = 1):
            st = base(n)
            s = runner.target_state()
            rows.append((float(np.linalg.norm(s["adm_pos"] - s["ref_pos"])),
                         float(np.linalg.norm(env.contact_force_on_ee()[0]))))
            return st

        env.step = wrapped  # type: ignore[method-assign]
        comp = ComplianceParams.z_only(k=120.0, d=20.0, force_limit=20.0)
        r = runner.move_to_pose(np.array([0.30, 0.0, 0.395]), approach_quat(0.0),
                                compliance=comp, timeout=6.0, force_limit=20.0)
        py = np.array([a for a, _ in rows])
        force = np.array([b for _, b in rows])
        print(f"[probe] 真实接触: pure_yield max={py.max() * 1000:.1f}mm "
              f"force max={force.max():.1f}N ok={r.ok} err={r.error_code}")
        self.assertGreater(float(force.max()), 1.0, "未形成真实接触")
        self.assertGreater(float(py.max()), 0.005, "接触阶段 pure_yield 应>5mm（标记应出现）")


# --------------------------------------------------------------------------- #
# 4b) force_norm 语义独立复核 + UI 规则帧级验证
# --------------------------------------------------------------------------- #
class TestForceNormSemantics(unittest.TestCase):
    @staticmethod
    def _run(inject, action):
        from core.types import ComplianceParams
        from core.config import HOME_QPOS
        env = MuJoCoArmEnv()
        kin = Kinematics(env)
        runner = MotionRunner(env, kin, AdmittanceController(ComplianceParams.z_only(), env.dt))
        rows: list = []
        base = env.step

        def w(n: int = 1):
            st = base(n)
            s = runner.target_state()
            rows.append({k: (np.array(v, float) if isinstance(v, np.ndarray) else v)
                         for k, v in s.items()})
            return st

        env.step = w  # type: ignore[method-assign]
        if inject is not None:
            env.contact_force_on_ee = lambda: (np.asarray(inject, float),
                                               np.array([0.1, 0.2, 0.3]))
        if action == "move":
            p, _ = env.site_pose("g_pinch")
            r = runner.move_to_pose(p + np.array([0.04, 0.0, 0.04]), approach_quat(0.0), timeout=8.0)
        elif action == "joint":
            path = [HOME_QPOS.copy() + np.array([0.01 * k, 0.0, 0.0, 0.0, 0.0, 0.0])
                    for k in range(6)]
            r = runner.follow_joint_path(path)
        else:
            r = runner.hold(0.3)
        return r, rows

    def test_free_space_force_norm_is_zero(self):
        r, rows = self._run(None, "move")
        self.assertTrue(r.ok, r.message)
        v = [s for s in rows if s["valid"]]
        fn = np.array([float(s["force_norm"]) for s in v])
        wn = np.array([float(np.linalg.norm(np.asarray(s["wrench"])[:3])) for s in v])
        print(f"\n[probe] 自由空间 force_norm max={fn.max():.9f}N  wrench[:3] max={wn.max():.9f}")
        self.assertEqual(float(fn.max()), 0.0, "无接触自由运动 force_norm 必须恒为 0")
        self.assertEqual(float(wn.max()), 0.0)

    def test_injected_force_norm_equals_norm(self):
        for f in (np.array([0.0, 0.0, 8.0]), np.array([3.0, 4.0, 0.0])):
            with self.subTest(f=f):
                r, rows = self._run(f, "move")
                self.assertTrue(r.ok, r.message)
                v = [s for s in rows if s["valid"]]
                fn = np.array([float(s["force_norm"]) for s in v])
                exp = float(np.linalg.norm(f))
                print(f"[probe] 注入 {f} -> force_norm med={np.median(fn):.4f} 期望={exp:.4f}")
                self.assertTrue(np.allclose(fn, exp, atol=1e-5))
                for s in v[:5]:
                    self.assertTrue(np.allclose(np.asarray(s["wrench"])[:3], f, atol=1e-5))
                    self.assertTrue(np.allclose(np.asarray(s["wrench"])[3:],
                                                [0.1, 0.2, 0.3], atol=1e-5))

    def test_joint_path_and_hold_wrench_is_real(self):
        for action, f in (("joint", np.array([0.0, 0.0, 5.0])),
                          ("hold", np.array([2.0, 0.0, 0.0]))):
            with self.subTest(action=action):
                r, rows = self._run(f, action)
                v = [s for s in rows if s["valid"]]
                self.assertGreater(len(v), 3, f"{action} 采样太少")
                fn = np.array([float(s["force_norm"]) for s in v])
                hints = set(str(s["skill_hint"]) for s in v)
                expect_hint = {"joint": "joint_path", "hold": "hold"}[action]
                print(f"[probe] {action}: hints={sorted(hints)} force_norm med={np.median(fn):.4f}")
                self.assertIn(expect_hint, hints)
                self.assertTrue(np.allclose(fn, float(np.linalg.norm(f)), atol=1e-5),
                                f"{action} 阶段 force_norm 应等于同一时刻真实读数")


class TestUIRuleFrames(unittest.TestCase):
    @unittest.skipUnless(STRICT_YIELD, "set VLM_STRICT_TARGET_YIELD=1")
    def test_strict_gate_free_zero_and_contact_positive(self):
        from demo.interactive import _YIELD_FORCE_N
        s = _MoveSampler()
        r = s.run()
        self.assertTrue(r.ok, r.message)
        v = [x for x in s.samples if x["valid"]]
        f0 = _draw_flags_for_states(_make_app(), v, mode="axes")
        self.assertEqual(sum(1 for f in f0 if f[0] or f[1]), 0, "自由空间出现让位标记")
        s2 = _MoveSampler(inject=np.array([0.0, 0.0, 8.0]))
        r2 = s2.run()
        self.assertTrue(r2.ok, r2.message)
        v2 = [x for x in s2.samples if x["valid"]]
        f2 = _draw_flags_for_states(_make_app(), v2, mode="axes")
        ref2 = sum(1 for f in f2 if f[0])
        print(f"\n[probe] 严格门: 自由 {len(f0)}帧/标记0  接触 {len(f2)}帧/ref {ref2}帧 "
              f"(阈值 {_YIELD_FORCE_N}N)")
        self.assertGreater(ref2, 0, "接触阶段未显示橙色 ref")

    def test_real_grasp_contact_frames_show_ref(self):
        from core.types import SkillCall
        from demo.interactive import _YIELD_FORCE_N
        app = _make_app()
        states: list = []
        base = app.env.step

        def w(n: int = 1):
            st = base(n)
            s = app.runner.target_state()
            states.append({k: (np.array(v, float) if isinstance(v, np.ndarray) else v)
                           for k, v in s.items()})
            return st

        app.env.step = w  # type: ignore[method-assign]
        r = app.skills(SkillCall("pick", {"target_id": "box_red"}))
        self.assertTrue(r.ok, r.message)
        v = [x for x in states if x["valid"]]
        frames = _draw_flags_for_states(app, v, mode="axes")
        thr = float(_YIELD_FORCE_N)
        contact = [i for i, x in enumerate(v) if float(x.get("force_norm") or 0.0) > thr]
        missing = [i for i in contact if not frames[i][0]]
        peak = max(float(x.get("force_norm") or 0.0) for x in v)
        print(f"\n[probe] 真实抓取(box_red): force_norm 峰值={peak:.2f}N 阈值={thr}N "
              f"接触帧={len(contact)} 未显示ref的接触帧={len(missing)}")
        self.assertGreater(len(contact), 0,
                           f"真实抓取未产生 >{thr}N 的接触帧（峰值仅 {peak:.2f}N）")
        self.assertEqual(missing, [], "真实抓取的接触帧必须显示橙色 ref 标记")


# --------------------------------------------------------------------------- #
# 5) 容错
# --------------------------------------------------------------------------- #
class TestFaultTolerance(unittest.TestCase):
    def _valid_state(self):
        return {"valid": True, "cmd_pos": np.zeros(3), "cmd_quat": np.array([1.0, 0, 0, 0]),
                "ref_pos": np.zeros(3), "adm_delta": 0.0}

    def test_draw_raises_only_one_notice_and_task_ok(self):
        import demo.interactive as di
        saved = di._overlay_draw_target
        app = _make_app()
        logs = _capture_log(app)
        di._overlay_draw_target = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        app.runner.target_state = self._valid_state  # type: ignore[method-assign]
        try:
            app._update_target_overlay(object())
            app._update_target_overlay(object())
        finally:
            di._overlay_draw_target = saved
        notices = [l for l in logs if "叠加" in l]
        self.assertEqual(len(notices), 1, f"只应提示一条，实际 {notices}")
        res = app.run_task("夹住蓝色方块")
        self.assertEqual([r.skill for r in res], ["pick", "stop"])
        self.assertTrue(all(r.ok for r in res), [(r.skill, r.message) for r in res])

    def test_overlay_unavailable_only_one_notice_and_task_ok(self):
        import demo.interactive as di
        sc, sd = di._overlay_clear, di._overlay_draw_target
        app = _make_app()
        logs = _capture_log(app)
        di._overlay_clear = None
        di._overlay_draw_target = None
        try:
            app._update_target_overlay(object())
            app._update_target_overlay(object())
        finally:
            di._overlay_clear, di._overlay_draw_target = sc, sd
        self.assertEqual(len([l for l in logs if "叠加" in l]), 1)
        res = app.run_task("夹住蓝色方块")
        self.assertEqual([r.skill for r in res], ["pick", "stop"])
        self.assertTrue(all(r.ok for r in res))


# --------------------------------------------------------------------------- #
# 6) 证伪
# --------------------------------------------------------------------------- #
class TestFalsifications(unittest.TestCase):
    def test_700_frames_no_leak_or_overflow(self):
        model = _empty_model()
        pos = np.array([0.4, 0.02, 0.62])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        for mg, expect in ((64, 6), (5, 5)):
            scn = mujoco.MjvScene(model, maxgeom=mg)
            counts = []
            for f in range(700):
                overlay.clear(scn)
                p = pos + np.array([0.0, 0.0, 0.001 * (f % 7)])
                counts.append(overlay.draw_target(scn, p, quat,
                                                  ref_pos=p + np.array([0.0, 0.0, 0.02])))
                self.assertLessEqual(scn.ngeom, scn.maxgeom, "ngeom 越界")
                self.assertEqual(scn.ngeom, min(expect, mg))
            self.assertEqual(set(counts), {min(expect, mg)})
            self.assertLessEqual(scn.ngeom, scn.maxgeom)

    def test_target_bogus_and_bare(self):
        app = _make_app()
        logs = _capture_log(app)
        self.assertEqual(app.target_mode, "on")
        app.dispatch(":target bogus")
        self.assertEqual(app.target_mode, "on", "非法模式不得改变状态")
        self.assertTrue(any("只支持" in l for l in logs), logs)
        app.dispatch(":target")
        self.assertTrue(any("目标点显示: on" in l for l in logs), logs)
        for mode in ("off", "axes", "sphere", "on"):
            app.dispatch(f":target {mode}")
            self.assertEqual(app.target_mode, mode)
        app.dispatch(":target OFF")
        self.assertEqual(app.target_mode, "off", "模式应大小写不敏感")

    def test_target_off_keeps_ngeom_zero(self):
        import demo.interactive as di
        app = _make_app()
        scn = mujoco.MjvScene(_empty_model(), maxgeom=64)
        app.target_mode = "off"
        app.runner.target_state = lambda: {"valid": True, "cmd_pos": np.array([0.1, 0.2, 0.3]),
                                           "cmd_quat": np.array([1.0, 0, 0, 0]),
                                           "ref_pos": np.zeros(3), "adm_delta": 0.0}
        for _ in range(50):
            app._update_target_overlay(scn)
            self.assertEqual(scn.ngeom, 0)


# --------------------------------------------------------------------------- #
# 7) 真 GL 窗口端到端
# --------------------------------------------------------------------------- #
# 硬性规则（用户投诉桌面被弹黑窗）：自动化测试**一律不得**启动 MuJoCo 窗口。
# 真正的窗口 E2E 只能由人显式开启：DSH_ALLOW_WINDOW_TESTS=1。
_ALLOW_WINDOW_TESTS = os.environ.get("DSH_ALLOW_WINDOW_TESTS") == "1"


@unittest.skipUnless(_ALLOW_WINDOW_TESTS,
                     "硬性规则：自动化测试不得开 MuJoCo 窗口"
                     "（人工验证请设 DSH_ALLOW_WINDOW_TESTS=1）")
class TestRealWindow(unittest.TestCase):
    def test_real_window_frames_task_and_target_off(self):
        try:
            import mujoco.viewer  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"mujoco.viewer 不可用: {exc}")
        app = _make_app(headless=False)
        try:
            viewer = __import__("mujoco.viewer", fromlist=["viewer"]).launch_passive(
                app.env.model, app.env.data)
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"launch_passive 失败（无显示环境?）: {exc}")

        import demo.interactive as di
        from core.types import ComplianceParams, SkillCall
        thr = float(di._YIELD_FORCE_N)
        real_draw = di._overlay_draw_target
        phase = {"name": "init"}
        marker = {"ref": False, "adm": False}
        results: list = []

        def spy(scn, pos, quat=None, **kw):
            rgba = kw.get("rgba")
            is_adm = (rgba is not None
                      and tuple(np.asarray(rgba, dtype=float)[:3])
                      == tuple(di._ADM_MARKER_RGBA[:3]))
            marker["ref"] = marker["ref"] or bool(kw.get("show_ref"))
            marker["adm"] = marker["adm"] or is_adm
            return real_draw(scn, pos, quat, **kw)

        di._overlay_draw_target = spy

        def worker():
            try:
                p, _ = app.env.site_pose("g_pinch")
                phase["name"] = "free"
                app.skills(SkillCall("move_to", {"pos": (p + np.array([0.04, 0.0, 0.04])).tolist(),
                                                 "planned": True}))
                phase["name"] = "task"
                results.extend(app.run_task("把红色的方块放进托盘"))
                phase["name"] = "press"
                app.skills.runner.move_to_pose(
                    np.array([0.30, 0.0, 0.395]), approach_quat(0.0),
                    compliance=ComplianceParams.z_only(k=120.0, d=20.0, force_limit=20.0),
                    timeout=6.0, force_limit=20.0)
            except Exception as exc:  # noqa: BLE001
                results.append(exc)
            finally:
                phase["name"] = "done"

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        stats: dict = {}

        def record():
            key = phase["name"]
            s = stats.setdefault(key, {"frames": 0, "ref": 0, "adm": 0, "max_ngeom": 0})
            s["frames"] += 1
            s["ref"] += 1 if marker["ref"] else 0
            s["adm"] += 1 if marker["adm"] else 0
            s["max_ngeom"] = max(s["max_ngeom"], int(viewer.user_scn.ngeom))

        off_frames = off_bad = 0
        t0 = time.time()
        try:
            while th.is_alive() or stats.get("done", {"frames": 0})["frames"] < 20:
                marker["ref"] = marker["adm"] = False
                with app.lock:
                    app._update_target_overlay(viewer.user_scn)
                    record()
                    viewer.sync()
                if time.time() - t0 > 150:
                    break
                time.sleep(0.01)
            app.target_mode = "off"
            for _ in range(60):
                marker["ref"] = marker["adm"] = False
                with app.lock:
                    app._update_target_overlay(viewer.user_scn)
                    n = int(viewer.user_scn.ngeom)
                    viewer.sync()
                off_frames += 1
                off_bad += 1 if (n != 0 or marker["ref"] or marker["adm"]) else 0
                time.sleep(0.005)
        finally:
            di._overlay_draw_target = real_draw
            try:
                viewer.close()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.4)

        free = stats.get("free", {"frames": 0, "ref": 0, "adm": 0, "max_ngeom": 0})
        task = stats.get("task", {"frames": 0, "ref": 0, "adm": 0, "max_ngeom": 0})
        press = stats.get("press", {"frames": 0, "ref": 0, "adm": 0, "max_ngeom": 0})
        print(f"\n[probe] 真窗口 阈值={thr}N free={free} task={task} press={press} "
              f"off_frames={off_frames} off_bad={off_bad}")
        self.assertGreater(free["frames"], 10, "自由空间相位帧数太少")
        self.assertEqual(free["ref"], 0, f"自由空间出现橙色 ref {free['ref']} 帧")
        self.assertEqual(free["adm"], 0, f"自由空间出现紫红 adm {free['adm']} 帧")
        self.assertGreater(task["ref"], 0,
                           f"真实抓取接触帧未出现橙色 ref（阈值 {thr}N，task={task}）")
        self.assertEqual(off_bad, 0, f":target off 后仍有 {off_bad}/{off_frames} 帧异常")
        self.assertTrue(results and all(hasattr(r, "ok") and r.ok for r in results),
                        f"窗口下抓放任务失败: {results}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
