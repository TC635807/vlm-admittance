"""tests/test_contact_place.py —— 接触式落放 / 叠放 / 容器内落放（INTERFACES.md §5.8-4）。

  * 方块叠方块：上层底面 - 下层顶面 高度差 <= 5 mm（真值仅用于断言）；
  * 容器内落放：3 个不同起点，成功 >= 2（成功 = place ok 且物体落在托盘 footprint 内、未穿桌面）。

运行：
    cd /home/tc63/mujoco/vlm_admittance
    MUJOCO_GL=glfw DISPLAY=:0 /home/tc63/mujoco/.venv/bin/python -m tests.test_contact_place
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


def _make_app(seed: int = 0):
    from demo.interactive import InteractiveApp
    args = argparse.Namespace(vlm="mock", perception="oracle", with_barrier=False, seed=seed,
                              realtime=False, headless=True, with_images=False,
                              max_steps=12, self_test=False)
    return InteractiveApp(args)


def _size(env, oid):
    return next(o for o in env.list_objects() if o.id == oid).size


class TestStackContact(unittest.TestCase):
    def test_green_stacked_on_red_gap_le_5mm(self):
        app = _make_app()
        r_pick = app.skills(SkillCall("pick", {"target_id": "box_green"}))
        self.assertTrue(r_pick.ok, f"pick 失败: {r_pick.error_code} {r_pick.message}")
        r_place = app.skills(SkillCall("place", {"on_top_of": "box_red"}))
        self.assertTrue(r_place.ok, f"叠放失败: {r_place.error_code} {r_place.message}")
        red = app.env.object_pose("box_red")[:3]
        green = app.env.object_pose("box_green")[:3]
        red_top = float(red[2] + _size(app.env, "box_red")[2])
        green_bottom = float(green[2] - _size(app.env, "box_green")[2])
        gap = green_bottom - red_top
        print(f"\n[probe] 叠放: red_top={red_top:.4f} green_bottom={green_bottom:.4f} "
              f"gap={gap * 1000:+.2f}mm metrics={ {k: r_place.metrics.get(k) for k in ('bottom_gap','stable_ok','footprint_ok')} }")
        self.assertLessEqual(abs(gap), 0.005, f"顶面高度差 {gap * 1000:.2f}mm > 5mm")
        self.assertGreater(float(green[2]), TABLE_HEIGHT, "上层方块掉到桌面以下")

    def test_bottom_never_below_table(self):
        app = _make_app()
        app.skills(SkillCall("pick", {"target_id": "box_green"}))
        app.skills(SkillCall("place", {"on_top_of": "box_red"}))
        green = app.env.object_pose("box_green")[:3]
        bottom = float(green[2] - _size(app.env, "box_green")[2])
        self.assertGreaterEqual(bottom, TABLE_HEIGHT - 0.002,
                                f"叠放后底面 {bottom * 1000:.1f}mm 穿到桌面下")


class TestContainerPlace(unittest.TestCase):
    STARTS = [(0.30, -0.10), (0.40, -0.08), (0.33, 0.06)]

    def test_three_starts_at_least_two_success(self):
        ok_n = 0
        for k, (x, y) in enumerate(self.STARTS):
            app = _make_app(seed=0)
            app.env.set_object_pose("box_red", np.array([x, y, TABLE_HEIGHT + 0.02, 1.0, 0, 0, 0]))
            app.env.step(30)
            r_pick = app.skills(SkillCall("pick", {"target_id": "box_red"}))
            r_place = app.skills(SkillCall("place", {"container_id": "tray"})) if r_pick.ok else r_pick
            tray = app.env.object_pose("tray")[:3]
            obj = app.env.object_pose("box_red")[:3]
            tray_half = _size(app.env, "tray")
            dxy = float(np.linalg.norm(obj[:2] - tray[:2]))
            inside = dxy <= float(np.hypot(tray_half[0], tray_half[1]))
            on_table = float(obj[2] - _size(app.env, "box_red")[2]) >= TABLE_HEIGHT - 0.002
            good = bool(r_pick.ok and getattr(r_place, "ok", False) and inside and on_table)
            ok_n += 1 if good else 0
            pm = getattr(r_place, "metrics", None) or {}
            print(f"[probe] start{k}=({x},{y}) pick={r_pick.ok} "
                  f"place={getattr(r_place, 'ok', False)} err={getattr(r_place, 'error_code', None)} "
                  f"dxy={dxy * 1000:.0f}mm inside={inside} on_table={on_table} "
                  f"touchdown_z={pm.get('touchdown_z')} 峰值接触力={pm.get('contact_force_norm')}N "
                  f"-> {'OK' if good else 'FAIL'}")   # skill-fixer 真机对照：touchdown 峰值 18.6N（偏高）
        print(f"[probe] 容器内落放成功 {ok_n}/3")
        self.assertGreaterEqual(ok_n, 2, f"容器内落放仅成功 {ok_n}/3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
