"""tests/test_grasp_height.py —— 无真值 top 抓取高度回归 + 1mm 被挡容差 + side 语义（task-66 验证）。

背景（用户实测回归）：z_hint 语义变成「顶面」后，方块分支没跟着减 0.03，
TCP 停在方块顶上方 16mm、合在空气里 -> ok=True 但没抓住。
修复：非 side 时 z_center = max(z_hint - 0.03, TABLE_HEIGHT + 0.03)；方块被挡容差 4mm -> 1mm。

真值只用于断言，不进 prompt；调用形状与 live 一致（target_xy / target_z_hint / target_size / grasp_profile）。

运行：
    MUJOCO_GL=glfw DISPLAY=:0 /home/tc63/mujoco/.venv/bin/python -m tests.test_grasp_height
"""
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import TABLE_HEIGHT                      # noqa: E402
from core.types import SkillCall                          # noqa: E402


def _make_app(scene: str = "default"):
    from demo.interactive import InteractiveApp
    args = argparse.Namespace(vlm="mock", perception="oracle", scene=scene, planner=None,
                              with_barrier=False, with_images=False, cameras=None,
                              realtime=False, headless=True, max_steps=4, seed=0,
                              self_test=False, panels="none")
    return InteractiveApp(args)


def _pick(app, kwargs: dict):
    return app.skills(SkillCall("pick", dict(kwargs)))


class TestTopGraspHeight(unittest.TestCase):
    """① 方块：live 形状 -> 夹持高度落在顶面下方 3cm（而不是顶面上方 16mm）。"""

    def test_live_shape_box_blue_grasps_for_real(self):
        app = _make_app()
        before = np.asarray(app.env.object_pose("box_blue"), float).copy()
        res = _pick(app, {"target_xy": [0.350, 0.136], "target_z_hint": 0.4567,
                          "target_size": [0.0201, 0.0216], "grasp_profile": "top"})
        after = np.asarray(app.env.object_pose("box_blue"), float).copy()
        z = (res.metrics or {}).get("grasp_z_effective")
        dz = float(after[2] - before[2])
        print(f"\n[probe] 方块: ok={res.ok} err={res.error_code} grasp_z_effective={z} "
              f"Δz={dz * 1000:+.1f}mm msg={res.message[:60]}")
        self.assertTrue(res.ok, f"应真的抓起: {res.error_code} {res.message}")
        self.assertIsNotNone(z, "需要 grasp_z_effective 证据")
        self.assertGreaterEqual(float(z) + 1e-9, 0.42, f"夹持高度 {z} 偏低")
        self.assertLessEqual(float(z), 0.435, f"夹持高度 {z} 偏高（旧回归会停在 0.456）")
        self.assertGreater(dz, 0.1, f"真值应被抬起 >100mm，实测 {dz * 1000:.1f}mm")

    def test_slender_bottle_height_not_regressed(self):
        app = _make_app(scene="daily")
        res = _pick(app, {"target_xy": [0.45, -0.18], "target_z_hint": 0.614,
                          "target_size": [0.032, 0.032], "grasp_profile": "top"})
        z = (res.metrics or {}).get("grasp_z_effective")
        print(f"[probe] 水瓶: ok={res.ok} err={res.error_code} grasp_z_effective={z} "
              f"（期望≈0.584）")
        if z is None:
            self.skipTest(f"水瓶没走到下降接触段: {res.error_code} {res.message[:80]}")
        self.assertAlmostEqual(float(z), 0.584, delta=0.010,
                               msg=f"细高物体夹持高度应≈z_hint-0.03=0.584，实测 {z}")


class TestBlockedTolerance1mm(unittest.TestCase):
    """③ 1mm 容差：真夹住（40mm vs 指令 37.2mm，差 2.8mm）不得判失败；空抓必须判失败。"""

    def test_positive_real_grasp_margin_over_1mm(self):
        app = _make_app()
        res = _pick(app, {"target_xy": [0.350, 0.136], "target_z_hint": 0.4567,
                          "target_size": [0.0201, 0.0216], "grasp_profile": "top"})
        m = res.metrics or {}
        gap = m.get("grasp_gap_mm")
        cmd = m.get("grasp_commanded_gap_mm")
        print(f"\n[probe] 正例: ok={res.ok} 实测指间距={gap}mm 指令={cmd}mm 差="
              f"{None if gap is None or cmd is None else round(gap - cmd, 2)}mm "
              f"（容差 1mm）")
        self.assertTrue(res.ok, f"真夹住不该被判失败: {res.error_code} {res.message}")
        self.assertIsNotNone(gap)
        self.assertIsNotNone(cmd)
        self.assertGreater(float(gap) - float(cmd), 1.0,
                           "真夹住时实测间距必须比指令大过 1mm 容差")

    def test_negative_air_grasp_is_not_grasped(self):
        app = _make_app()
        before = np.asarray(app.env.object_pose("box_blue"), float).copy()
        res = _pick(app, {"target_xy": [0.300, -0.320], "target_z_hint": 0.4567,
                          "target_size": [0.0201, 0.0216], "grasp_profile": "top"})
        after = np.asarray(app.env.object_pose("box_blue"), float).copy()
        m = res.metrics or {}
        gap, cmd = m.get("grasp_gap_mm"), m.get("grasp_commanded_gap_mm")
        print(f"[probe] 反例(空处): ok={res.ok} err={res.error_code} 实测={gap} 指令={cmd} "
              f"Δz(box_blue)={(after[2] - before[2]) * 1000:+.1f}mm")
        self.assertFalse(res.ok, "空抓绝不能报成功")
        if gap is not None and cmd is not None:
            self.assertLessEqual(float(gap) - float(cmd), 1.0,
                                 "空抓时实测间距不该超过指令 1mm（否则 1mm 容差失效）")


class TestSideProfileUnchanged(unittest.TestCase):
    """④ side：夹持高度=可见点（不 -0.03）；与 top 分支的高度差应≈3cm。

    用**方块**做对照（圆柱/水瓶在 top 分支本来也用 z_hint，分不出来）。
    """

    @staticmethod
    def _first_target_z(app, kwargs):
        seen: list = []
        real = app.skills.runner.move_to_pose

        def spy(target, quat, *a, **k):
            seen.append(np.asarray(target, float).copy())
            return real(target, quat, *a, **k)

        app.skills.runner.move_to_pose = spy
        try:
            res = _pick(app, kwargs)
        finally:
            app.skills.runner.move_to_pose = real
        return (float(seen[0][2]) if seen else None), res

    def test_side_uses_z_hint_without_minus_3cm(self):
        kw = {"target_xy": [0.350, 0.136], "target_z_hint": 0.4567,
              "target_size": [0.0201, 0.0216]}
        z_side, r_side = self._first_target_z(_make_app(), dict(kw, grasp_profile="side"))
        z_top, r_top = self._first_target_z(_make_app(), dict(kw, grasp_profile="top"))
        print(f"\n[probe] side 首目标 z={z_side}（{r_side.error_code}） top 首目标 z={z_top}"
              f"（{r_top.error_code}） z_hint=0.4567")
        self.assertIsNotNone(z_side, "side 分支应有接近目标")
        self.assertIsNotNone(z_top, "top 分支应有接近目标")
        self.assertGreater(z_side - z_top, 0.02,
                         f"side 高度应比 top 高约 3cm（side 不减 0.03），实测差 "
                         f"{None if None in (z_side, z_top) else round(z_side - z_top, 4)}m")

    def test_side_on_bottle_still_runs(self):
        app = _make_app(scene="daily")
        res = _pick(app, {"target_xy": [0.45, -0.18], "target_z_hint": 0.614,
                          "target_size": [0.032, 0.032], "grasp_profile": "side"})
        print(f"[probe] 水瓶 side: ok={res.ok} err={res.error_code} "
              f"msg={res.message[:70]}")
        self.assertIn(res.error_code, (None, "grasp_failed"), "side 失败只能是有语义的失败")


class TestLiftAndCarryGates(unittest.TestCase):
    """P1 回归：抬升失败 / 抬升后物体不随动，绝不能报"已抓起"。

    历史缺陷（实机 2026-10-12）：gap 判据过了就 return ok=True，抬升
    `move_to_pose` 的 r.ok 被完全忽略 —— 实测把 40mm 方块当 20mm 给尺寸时，
    guarded close 47N 停在 0.46、抬升 force_limit 47N、方块只动 +1mm，仍报成功。
    """

    def test_lift_force_limit_is_not_reported_as_grasped(self):
        app = _make_app()
        before = np.asarray(app.env.object_pose("box_red"), float).copy()
        res = _pick(app, {"target_xy": [0.350, -0.140], "target_z_hint": 0.423,
                          "target_size": [0.010, 0.010], "grasp_profile": "top"})
        after = np.asarray(app.env.object_pose("box_red"), float).copy()
        dz = float(after[2] - before[2])
        m = res.metrics or {}
        print(f"\n[probe] 抬升门: ok={res.ok} err={res.error_code} Δz={dz * 1000:+.1f}mm "
              f"lift_ok={m.get('grasp_lift_ok')} carry={m.get('grasp_carry_evidence')} "
              f"gap={m.get('grasp_gap_mm')} 指令={m.get('grasp_commanded_gap_mm')}")
        self.assertFalse(res.ok, "抬升失败/物体没抬起来绝不能报成功")
        self.assertLess(dz, 0.02, "断言前提：该配置本来就没把方块拿起来")
        self.assertIn(res.error_code, ("grasp_failed", "force_limit"))

    def test_carry_evidence_reflects_real_physics(self):
        """随动证据（夹爪接触的物体且不被支撑面托着）在真场景里必须说真话。"""
        app = _make_app()
        sk = app.skills
        self.assertEqual(sk._carry_evidence(), (False, []), "空载时不该有任何随动证据")

        ok = _pick(app, {"target_xy": [0.350, -0.140], "target_z_hint": 0.456315,
                         "target_size": [0.020096, 0.02084], "grasp_profile": "top"})
        self.assertTrue(ok.ok, f"这个配置本应真抓起: {ok.error_code} {ok.message}")
        carry, bodies = sk._carry_evidence()
        print(f"[probe] 随动证据: carry={carry} bodies={bodies} holding={sk.holding}")
        self.assertTrue(carry, "真抓住方块时随动证据必须为 True")
        self.assertIn("box_red", bodies)

        # 复核前归位（瞬移）把物体放回桌面后：没有接触 -> 随动证据变 False
        app.env.home()
        app.env.step(300)
        carry2, bodies2 = sk._carry_evidence()
        print(f"[probe] home 之后: carry={carry2} bodies={bodies2}")
        self.assertFalse(carry2, "物体已落回桌面，随动证据必须为 False")

    def test_unnamed_holding_resolves_to_contacted_object(self):
        """P0：无真值 holding='unnamed' 必须能反查出真实手持物（否则带载归位永远回落瞬移）。"""
        app = _make_app()
        res = _pick(app, {"target_xy": [0.350, -0.140], "target_z_hint": 0.456315,
                          "target_size": [0.020096, 0.02084], "grasp_profile": "top"})
        self.assertTrue(res.ok, f"这个配置本应真抓起: {res.error_code} {res.message}")
        self.assertEqual(app.skills.holding, "unnamed")
        names = app.skills.planner.checker._geom_body_names
        targets = app.skills._resolve_ignore_targets("unnamed", names)
        print(f"[probe] unnamed -> 忽略目标 {sorted(targets)}")
        self.assertEqual(targets, {"box_red"})
        # 走一遍忽略上下文：必须在原 mask 基础上真的置位（旧实现 0 个 -> 归位规划必然失败），
        # 出栈后逐位还原（真实 checker 的基线 mask 里夹爪<->夹爪 本来就是忽略的）。
        mask = app.skills.planner.checker._ignore_mask
        before = mask.copy()
        with app.skills._ignore_contacts_with("unnamed"):
            self.assertGreater(int(mask.sum()), int(before.sum()),
                               "unnamed 情况下必须真的忽略夹爪<->手持物接触对")
        self.assertTrue(np.array_equal(mask, before), "出栈后 mask 必须逐位还原")


if __name__ == "__main__":
    unittest.main(verbosity=2)
