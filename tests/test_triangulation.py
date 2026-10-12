"""tests/test_triangulation.py —— 双目三角定位与俯视视差（INTERFACES.md §5.10）。

  (1) 视差复现（今天就能跑，只依赖 sim/vision）：水瓶**顶面**像素按 z=TABLE_HEIGHT 反投影，
      误差应 >= 60 mm（Lead 实测 68.6 mm）；对照：瓶**中心**像素误差 <= 1 mm。
  (2) 双像素 triangulate/object_point 后误差 <= 5 mm（水瓶），方块 <= 3 mm。
  (3) 两个像素里有一个是错的 -> residual_mm 明显变大且 ok=False（不把垃圾三角当真值）。

运行：
    MUJOCO_GL=glfw DISPLAY=:0 /home/tc63/mujoco/.venv/bin/python -m tests.test_triangulation
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import TABLE_HEIGHT                      # noqa: E402
from sim.env import MuJoCoArmEnv                          # noqa: E402
from sim.vision import project_point, pixel_to_table      # noqa: E402

try:
    from sim import perception as perception              # noqa: E402
except Exception:                                         # noqa: BLE001
    perception = None

_HAS_TRI = perception is not None and hasattr(perception, "triangulate") \
    and hasattr(perception, "object_point")
_SKIP_TRI = "sim.perception.triangulate/object_point 未落地（§5.10 pending）"

BOTTLE_XY = (0.45, -0.18)
BOTTLE_HALF = (0.032, 0.032, 0.100)       # 半径 32mm、高 200mm
BOX_XY = (0.30, 0.10)
BOX_HALF = (0.02, 0.02, 0.02)


def _bottle_env():
    objs = [dict(id="bottle_water", category="cylinder", color="yellow", shape="cylinder",
                 size=BOTTLE_HALF, pos=(BOTTLE_XY[0], BOTTLE_XY[1], TABLE_HEIGHT + BOTTLE_HALF[2]))]
    env = MuJoCoArmEnv(objects=objs)
    env.step(30)
    return env


def _box_env():
    objs = [dict(id="box_red", category="box", color="red", shape="box",
                 size=BOX_HALF, pos=(BOX_XY[0], BOX_XY[1], TABLE_HEIGHT + BOX_HALF[2]))]
    env = MuJoCoArmEnv(objects=objs)
    env.step(30)
    return env


class TestParallaxRepro(unittest.TestCase):
    """① 俯视视差：高物体顶面像素按桌面平面反投影会产生系统性偏移。"""

    def test_top_face_plane_backprojection_ge_60mm(self):
        env = _bottle_env()
        truth = np.array(BOTTLE_XY, float)
        z_top = TABLE_HEIGHT + 2 * BOTTLE_HALF[2]
        px = project_point(env, np.array([truth[0], truth[1], z_top]), camera="top")
        self.assertIsNotNone(px, "水瓶顶面应能被 top 相机看到")
        world = pixel_to_table(env, px[0], px[1], camera="top", z=TABLE_HEIGHT)
        self.assertIsNotNone(world)
        err = float(np.linalg.norm(np.asarray(world[:2], float) - truth)) * 1000.0
        print(f"\n[probe] 水瓶顶面 top_px={tuple(round(v, 1) for v in px)} "
              f"-> 平面反投影 {np.round(world[:2], 4)} 误差={err:.1f}mm")
        self.assertGreaterEqual(err, 60.0, f"视差复现失败：{err:.1f}mm < 60mm")

    def test_center_pixel_has_no_parallax(self):
        env = _bottle_env()
        truth = np.array(BOTTLE_XY, float)
        px = project_point(env, np.array([truth[0], truth[1], TABLE_HEIGHT]), camera="top")
        world = pixel_to_table(env, px[0], px[1], camera="top", z=TABLE_HEIGHT)
        err = float(np.linalg.norm(np.asarray(world[:2], float) - truth)) * 1000.0
        print(f"[probe] 对照组（瓶中心高度像素）误差={err:.3f}mm")
        self.assertLessEqual(err, 1.0, "桌面高度像素本身不应有视差")


@unittest.skipUnless(_HAS_TRI, _SKIP_TRI)
class TestTwoViewTriangulation(unittest.TestCase):
    def _two_pixels(self, env, truth_xy, z):
        a = project_point(env, np.array([truth_xy[0], truth_xy[1], z]), camera="top")
        b = project_point(env, np.array([truth_xy[0], truth_xy[1], z]), camera="overview")
        return a, b

    def test_bottle_two_view_error_le_5mm(self):
        env = _bottle_env()
        truth = np.array(BOTTLE_XY, float)
        z_top = TABLE_HEIGHT + 2 * BOTTLE_HALF[2]
        a, b = self._two_pixels(env, truth, z_top)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        out = perception.object_point(env, top_uv=a, overview_uv=b)
        err = float(np.linalg.norm(np.asarray(out["xyz"], float)[:2] - truth)) * 1000.0
        print(f"\n[probe] 水瓶双目: source={out.get('source')} residual={out.get('residual_mm')} "
              f"err={err:.2f}mm")
        self.assertTrue(out.get("ok"), f"双目应可信: {out}")
        self.assertLessEqual(err, 5.0, f"双目误差 {err:.2f}mm > 5mm")

    def test_box_two_view_error_le_3mm(self):
        env = _box_env()
        truth = np.array(BOX_XY, float)
        z_top = TABLE_HEIGHT + 2 * BOX_HALF[2]
        a, b = self._two_pixels(env, truth, z_top)
        out = perception.object_point(env, top_uv=a, overview_uv=b)
        err = float(np.linalg.norm(np.asarray(out["xyz"], float)[:2] - truth)) * 1000.0
        print(f"[probe] 方块双目: residual={out.get('residual_mm')} err={err:.2f}mm")
        self.assertTrue(out.get("ok"), f"双目应可信: {out}")
        self.assertLessEqual(err, 3.0, f"方块双目误差 {err:.2f}mm > 3mm")

    def test_wrong_pixel_gives_large_residual_and_not_ok(self):
        env = _bottle_env()
        truth = np.array(BOTTLE_XY, float)
        z_top = TABLE_HEIGHT + 2 * BOTTLE_HALF[2]
        a, b = self._two_pixels(env, truth, z_top)
        bad_b = (float(b[0]) + 80.0, float(b[1]) - 60.0)          # 明显错像素
        out = perception.triangulate(env, a, "top", bad_b, "overview")
        good = perception.triangulate(env, a, "top", b, "overview")
        print(f"[probe] 错像素: residual={out.get('residual_mm')} ok={out.get('ok')} ; "
              f"正确像素 residual={good.get('residual_mm')} ok={good.get('ok')}")
        self.assertFalse(out.get("ok"), "错像素三角不得判 ok=True")
        self.assertGreater(float(out.get("residual_mm", 0.0)), 15.0)
        self.assertTrue(good.get("ok"))
        self.assertLess(float(good.get("residual_mm", 1e9)), 15.0)


if __name__ == "__main__":
    if not _HAS_TRI:
        print(f"[PENDING] {_SKIP_TRI} —— 双像素用例会 skip；视差复现用例今天即可运行")
    unittest.main(verbosity=2)
