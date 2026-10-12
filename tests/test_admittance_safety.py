"""tests/test_admittance_safety.py —— 导纳符号 + stop_on_force 优先序 + 冻结窗安全网（task-67 独立复核）。

独立实现（不照抄 control/smoke）：用**真实 sim.env.MuJoCoArmEnv** + 自带的力注入包装
（只覆盖 contact_force_on_ee），真值/真物理都在。

覆盖：
  1) 自由空间 sign=-1 与 +1 **逐位一致**（f_ext=0 -> 符号无影响）；
     真接触下 sign=-1 的峰值力**不随压入增大**且小于 +1，且末端**向上让位**。
  2) 注入 20N(>flim=15) + stop_on_force=2.5 -> ok=True / stop_reason=contact；
     stop_on_force=None -> 同场景必须 force_limit 中止。
  3) 冻结窗安全网：触发后力**不再上涨**（静置 11N < net=15N）不得误杀；
     触发后力**继续上涨**（3N -> 20N > net）必须中止。

运行：
    MUJOCO_GL=glfw DISPLAY=:0 /home/tc63/mujoco/.venv/bin/python -m tests.test_admittance_safety
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import TABLE_HEIGHT, SIM_DT               # noqa: E402
from core.types import ComplianceParams                    # noqa: E402
from control.kinematics import Kinematics                  # noqa: E402
from control.motion import MotionRunner                    # noqa: E402
from sim.env import MuJoCoArmEnv                           # noqa: E402
from skills import approach_quat                           # noqa: E402


class _ForceEnv(MuJoCoArmEnv):
    """真实环境 + 可注入/可编程的 EE 力读数（只覆盖 contact_force_on_ee）。

    注意：基类 __init__ 里就会 reset()->get_state()->contact_force_on_ee()，
    所以注入字段必须是**类属性默认值**，不能在 super().__init__ 之后才设。
    """

    force_override = None                  # 常量或 callable(call_index) -> |F| 的幅值
    _force_calls = 0

    def contact_force_on_ee(self):
        if self.force_override is not None:
            self._force_calls = int(getattr(self, "_force_calls", 0)) + 1
            f = (self.force_override(self._force_calls)
                 if callable(self.force_override) else float(self.force_override))
            return (np.array([0.0, 0.0, float(f)]), np.zeros(3))
        return super().contact_force_on_ee()


def _runner(env, sign: float = -1.0) -> MotionRunner:
    kin = Kinematics(env, "g_pinch")
    run = MotionRunner(env, kin)
    run.adm_force_sign = float(sign)
    return run


def _tcp(env):
    pos, quat = env.site_pose("g_pinch")
    return np.asarray(pos, float), np.asarray(quat, float)


class TestAdmittanceSignFreeSpace(unittest.TestCase):
    def test_free_space_bit_identical_for_both_signs(self):
        outs = []
        for sign in (-1.0, 1.0):
            env = MuJoCoArmEnv()
            run = _runner(env, sign)
            p0, q0 = _tcp(env)
            target = p0 + np.array([0.0, 0.0, 0.05])
            res = run.move_to_pose(target, q0, compliance=ComplianceParams.z_only(force_limit=8.0),
                                   timeout=2.0)
            outs.append((res, env))
        a, b = outs[0][0], outs[1][0]
        same = (a.final_pos.tobytes() == b.final_pos.tobytes())
        print(f"\n[probe] 自由空间 sign±1: bitwise_pos_equal={same} steps={a.steps}/{b.steps} "
              f"maxF={a.max_force:.3f}/{b.max_force:.3f} pos0={a.final_pos[0]:.12f}")
        self.assertTrue(a.ok and b.ok)
        self.assertLess(a.max_force, 1e-6, "自由空间不该有接触力")
        self.assertTrue(same, "自由空间下力符号必须对轨迹逐位无影响")
        self.assertEqual(a.steps, b.steps)


class TestAdmittanceSignUnderContact(unittest.TestCase):
    def _press(self, sign: float):
        env = MuJoCoArmEnv()
        run = _runner(env, sign)
        p0, q0 = _tcp(env)
        # 先落到桌面上方 2cm（避免长距离下降被判不可达），再压进桌面 1cm
        stage = np.array([p0[0], p0[1], float(TABLE_HEIGHT) + 0.02])
        r0 = run.move_to_pose(stage, q0, compliance=ComplianceParams.z_only(force_limit=50.0),
                              timeout=3.0)
        p1, q1 = _tcp(env)
        target = np.array([p1[0], p1[1], float(TABLE_HEIGHT) - 0.01])
        res = run.move_to_pose(target, q1, compliance=ComplianceParams.z_only(force_limit=50.0),
                               timeout=3.0, v_max=0.006)
        if not r0.ok and not res.ok:
            print(f"[probe]   阶段一也失败: {r0.error_code} {r0.message[:60]}")
        return res, target

    def test_force_does_not_run_away_and_yields_upward(self):
        res_neg, target = self._press(-1.0)
        res_pos, _ = self._press(1.0)
        print(f"\n[probe] 真接触: sign=-1 maxF={res_neg.max_force:.2f}N final_z={res_neg.final_pos[2]:.4f}"
              f" | sign=+1 maxF={res_pos.max_force:.2f}N final_z={res_pos.final_pos[2]:.4f}"
              f" | 目标 z={target[2]:.4f}")
        self.assertLessEqual(res_neg.max_force, 60.0,
                             f"正确符号下峰值力应有界，实测 {res_neg.max_force:.2f}N")
        self.assertLess(res_neg.max_force, res_pos.max_force,
                        f"正确符号(-1)的峰值力应小于旧符号(+1)：{res_neg.max_force:.2f} vs "
                        f"{res_pos.max_force:.2f}")
        self.assertGreater(float(res_neg.final_pos[2]), float(target[2]),
                           "被压时应向上让位（远离接触）")


class TestStopOnForcePrecedence(unittest.TestCase):
    def _run(self, *, stop_on_force, flim=15.0, force=20.0, timeout=5.0):
        env = _ForceEnv()
        env.force_override = float(force)
        run = _runner(env, -1.0)
        p0, q0 = _tcp(env)
        res = run.move_to_pose(p0 + np.array([0.0, 0.0, -0.03]), q0,
                               compliance=ComplianceParams.z_only(force_limit=flim),
                               timeout=timeout, stop_on_force=stop_on_force)
        return res, env

    def test_injected_20N_with_stop_on_force_returns_contact(self):
        res, _env = self._run(stop_on_force=2.5)
        m = getattr(res, "metrics", {}) or {}
        print(f"\n[probe] 20N>flim15 + stop=2.5: ok={res.ok} err={res.error_code} "
              f"stop_reason={getattr(res, 'stop_reason', None)} "
              f"force_at_stop={m.get('force_at_stop')}")
        self.assertTrue(res.ok, f"接触即停必须先于硬限: {res.error_code} {res.message}")
        self.assertEqual(getattr(res, "stop_reason", None), "contact")
        self.assertGreaterEqual(float(m.get("force_at_stop", 0.0)), 2.5)

    def test_injected_20N_without_stop_on_force_aborts_on_hard_limit(self):
        res, _env = self._run(stop_on_force=None)
        print(f"[probe] 20N>flim15 + stop=None: ok={res.ok} err={res.error_code}")
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "force_limit")


class TestFreezeWindowSafetyNet(unittest.TestCase):
    """触发后力不再上涨 -> 不误杀；触发后继续上涨 -> 必须中止。"""

    def _schedule(self, before: float, after: float, n_before: int = 40):
        def sched(i):
            return float(before) if i <= n_before else float(after)
        return sched

    def _run(self, sched):
        env = _ForceEnv()
        env.force_override = sched
        run = _runner(env, -1.0)
        p0, q0 = _tcp(env)
        res = run.move_to_pose(p0 + np.array([0.0, 0.0, -0.03]), q0,
                               compliance=ComplianceParams.z_only(force_limit=15.0),
                               timeout=5.0, stop_on_force=2.5)
        return res

    def test_normal_rest_11N_is_not_killed(self):
        res = self._run(self._schedule(3.0, 11.0))     # net = max(15, 2*3)=15 > 11
        m = getattr(res, "metrics", {}) or {}
        print(f"\n[probe] 静置 3->11N: ok={res.ok} err={res.error_code} "
              f"stop_reason={getattr(res, 'stop_reason', None)} maxF={res.max_force:.2f}N")
        self.assertTrue(res.ok, f"正常静置（11N < net 15N）不得被安全网误杀: {res.error_code}")
        self.assertEqual(getattr(res, "stop_reason", None), "contact")
        self.assertGreater(m.get("force_at_stop", 0.0), 0.0)

    def test_rising_force_above_net_aborts(self):
        res = self._run(self._schedule(3.0, 20.0))     # net = max(15, 2*3)=15 < 20 -> 中止
        print(f"[probe] 上涨 3->20N: ok={res.ok} err={res.error_code} maxF={res.max_force:.2f}N")
        self.assertFalse(res.ok, "冻结后力继续上涨超过安全网必须中止")
        self.assertEqual(res.error_code, "force_limit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
