"""tests/test_perception.py —— sim/perception.py 单元验证（INTERFACES.md §5.8-1）。

标定/几何事实（Lead 实测）：top 相机 512x512、fovy 45° -> 桌面处 **1.489 mm/px**；
overview 为斜视（du 2.895 / dv 6.882 mm/px），精定位只测 top。

断言口径（不放宽）：
  * color_blob 合成色块质心误差 <= 2 px；
  * bbox_to_world 与**独立解析解**（本文件自带射线-平面求交）尺寸误差 <= 3 mm（约 ±1 px）；
  * 越界 / 单色 / 噪声图不崩；
  * refine_pixel 失败时 uv 必须**原样**等于粗像素（可安全回退）。

sim/perception.py 尚未落地时整类 skip（pending），并打印原因。

运行：
    cd /home/tc63/mujoco/vlm_admittance
    MUJOCO_GL=glfw DISPLAY=:0 /home/tc63/mujoco/.venv/bin/python -m tests.test_perception
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
from sim.vision import CameraModel                        # noqa: E402

_IMPORT_ERROR = ""
perception = None
try:                                                      # §5.1 pending 时整类 skip
    from sim import perception as perception
except Exception as exc:                                  # noqa: BLE001
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

_PENDING = perception is None
_SKIP_REASON = f"sim.perception 未落地（§5.1 pending: {_IMPORT_ERROR}）"

TOP_MM_PER_PX = 1.489          # Lead 实测（512x512, fovy 45°, 桌面处）


# --------------------------------------------------------------------------- #
# 合成图
# --------------------------------------------------------------------------- #
BG = (128, 128, 128)
COLORS_RGB = {"red": (220, 40, 40), "blue": (40, 60, 220), "green": (40, 200, 60),
               "yellow": (230, 220, 40), "gray": (128, 128, 128)}


def _canvas(w=512, h=512):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = BG
    return img


def _disc(img, cu, cv, r, color="red"):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    mask = (xx - cu) ** 2 + (yy - cv) ** 2 <= r * r
    img[mask] = COLORS_RGB[color]
    return img


def _rect(img, u1, v1, u2, v2, color="blue"):
    img[v1:v2, u1:u2] = COLORS_RGB[color]
    return img


def _analytic_corners(env, bbox, camera="top", z=TABLE_HEIGHT):
    """独立实现：四角射线与 z 平面求交（不调用 perception.bbox_to_world）。"""
    cam = CameraModel(env, camera)
    u1, v1, u2, v2 = [float(v) for v in bbox]
    pts = []
    for (u, v) in ((u1, v1), (u1, v2), (u2, v1), (u2, v2)):
        origin, d = cam.ray(u, v)
        origin = np.asarray(origin, float)
        d = np.asarray(d, float)
        t = (float(z) - origin[2]) / d[2]
        pts.append(origin + t * d)
    pts = np.asarray(pts)
    size = np.array([pts[:, 0].max() - pts[:, 0].min(),
                      pts[:, 1].max() - pts[:, 1].min()])
    center = np.array([pts[:, 0].mean(), pts[:, 1].mean()])
    return center, size, pts


# --------------------------------------------------------------------------- #
# 不依赖 sim.perception 的标定自检（今天就能跑）
# --------------------------------------------------------------------------- #
class TestAnalyticScale(unittest.TestCase):
    def test_top_camera_mm_per_px_matches_lead_datum(self):
        env = MuJoCoArmEnv()
        cam = CameraModel(env, "top")
        cu = cam.width / 2.0
        cv = cam.height / 2.0
        box = (cu - 50, cv - 50, cu + 50, cv + 50)        # 100x100 px 框
        center, size, _pts = _analytic_corners(env, box, "top", TABLE_HEIGHT)
        mm_per_px_u = float(size[0]) * 1000.0 / 100.0
        mm_per_px_v = float(size[1]) * 1000.0 / 100.0
        print(f"\n[probe] top 中心 100px 框 -> 世界尺寸 {np.round(size * 1000, 2)} mm "
              f"(u {mm_per_px_u:.3f} mm/px, v {mm_per_px_v:.3f} mm/px)")
        self.assertAlmostEqual(mm_per_px_u, TOP_MM_PER_PX, delta=0.06,
                               msg="top 相机 u 方向 mm/px 与 Lead 实测 1.489 不符")
        self.assertAlmostEqual(mm_per_px_v, TOP_MM_PER_PX, delta=0.06)

    def test_overview_is_oblique(self):
        env = MuJoCoArmEnv()
        cam = CameraModel(env, "overview")
        cu, cv = cam.width / 2.0, cam.height / 2.0
        _c, size, _p = _analytic_corners(env, (cu - 50, cv - 50, cu + 50, cv + 50), "overview")
        print(f"[probe] overview 中心 100px 框 -> {np.round(size * 1000, 2)} mm（斜视，仅对照）")
        self.assertGreater(float(size[1]) * 1000.0 / 100.0, 3.0, "overview 应为明显斜视")


# --------------------------------------------------------------------------- #
# sim.perception 单元（pending 时 skip）
# --------------------------------------------------------------------------- #
@unittest.skipUnless(not _PENDING, _SKIP_REASON)
class TestColorBlob(unittest.TestCase):
    def test_disc_centroid_within_2px(self):
        img = _disc(_canvas(), 200, 150, 24)
        r = perception.color_blob(img, "red", (200, 150))
        self.assertTrue(r.found, f"未找到红块: {r}")
        err = float(np.hypot(r.uv[0] - 200.0, r.uv[1] - 150.0))
        print(f"[probe] disc centroid err={err:.2f}px area={r.area_px}")
        self.assertLessEqual(err, 2.0)
        self.assertGreaterEqual(r.area_px, 0.7 * np.pi * 24 ** 2)

    def test_rect_bbox_within_2px(self):
        img = _rect(_canvas(), 100, 200, 170, 260)
        r = perception.color_blob(img, "blue", (135, 230))
        self.assertTrue(r.found)
        u1, v1, u2, v2 = r.bbox
        print(f"[probe] rect bbox={r.bbox}（期望约 (100,200,170,260)）")
        for got, want in ((u1, 100), (v1, 200), (u2, 170), (v2, 260)):
            self.assertLessEqual(abs(float(got) - want), 2.0)

    def test_absent_color_returns_not_found(self):
        img = _disc(_canvas(), 300, 300, 20, "red")
        r = perception.color_blob(img, "green", (300, 300))
        self.assertFalse(r.found)

    def test_single_color_and_noise_do_not_crash(self):
        flat = np.zeros((512, 512, 3), dtype=np.uint8)
        flat[:, :] = COLORS_RGB["red"]
        r1 = perception.color_blob(flat, "red")
        self.assertIsInstance(r1.found, bool)
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
        r2 = perception.color_blob(noise, "blue", (10, 10))
        self.assertIsInstance(r2.found, bool)
        r3 = perception.color_blob(_canvas(), "blue", (-50, 9999))     # 越界 hint
        self.assertIsInstance(r3.found, bool)
        print(f"[probe] 单色/噪声/越界 hint 均未崩: flat={r1.found} noise={r2.found} oob={r3.found}")


@unittest.skipUnless(not _PENDING, _SKIP_REASON)
class TestRefinePixel(unittest.TestCase):
    def test_refine_snaps_to_centroid(self):
        img = _disc(_canvas(), 260, 200, 22)
        out = perception.refine_pixel(img, "red", (268, 206), camera="top")
        err = float(np.hypot(out["uv"][0] - 260.0, out["uv"][1] - 200.0))
        print(f"[probe] refine: uv={out['uv']} refined={out['refined']} "
              f"delta={out['delta_px']:.2f}px err={err:.2f}px")
        self.assertTrue(out["refined"], f"应精修成功: {out}")
        self.assertLessEqual(err, 2.0)
        self.assertEqual(out["source"], "color_blob")

    def test_refine_failure_returns_hint_verbatim(self):
        img = _disc(_canvas(), 260, 200, 22, "red")
        hint = (30.0, 480.0)                                  # 该处没有红色
        out = perception.refine_pixel(img, "green", hint, camera="top")
        print(f"[probe] refine 失败: uv={out['uv']} refined={out['refined']} source={out['source']}")
        self.assertFalse(out["refined"])
        self.assertEqual(tuple(float(v) for v in out["uv"]), hint,
                         "refine 失败必须原样返回 hint")
        self.assertEqual(out["source"], "vlm_raw")


@unittest.skipUnless(not _PENDING, _SKIP_REASON)
class TestBboxToWorld(unittest.TestCase):
    def test_size_matches_independent_analytic_solution(self):
        env = MuJoCoArmEnv()
        cam = CameraModel(env, "top")
        cu, cv = cam.width / 2.0, cam.height / 2.0
        bbox = (cu - 40, cv - 30, cu + 40, cv + 30)
        want_c, want_s, _pts = _analytic_corners(env, bbox, "top", TABLE_HEIGHT)
        got = perception.bbox_to_world(env, bbox, camera="top", z=TABLE_HEIGHT)
        self.assertTrue(got.get("ok"), f"bbox_to_world 失败: {got}")
        got_s = np.asarray(got["size_xy"], float)
        got_c = np.asarray(got["center_xy"], float)
        dsize = float(np.linalg.norm(got_s - want_s)) * 1000.0
        dcen = float(np.linalg.norm(got_c - want_c)) * 1000.0
        print(f"[probe] bbox_to_world size={np.round(got_s * 1000, 2)}mm "
              f"解析解={np.round(want_s * 1000, 2)}mm |Δsize|={dsize:.3f}mm |Δcenter|={dcen:.3f}mm")
        self.assertLessEqual(dsize, 3.0, "尺寸误差 >3mm（约 ±1px）")
        self.assertLessEqual(dcen, 3.0)

    def test_degenerate_bbox_no_crash(self):
        env = MuJoCoArmEnv()
        for bbox in ((0, 0, 0, 0), (500, 500, 511, 511), (-10, -10, 5, 5)):
            with self.subTest(bbox=bbox):
                out = perception.bbox_to_world(env, bbox, camera="top", z=TABLE_HEIGHT)
                self.assertIn("ok", out)


@unittest.skipUnless(not _PENDING, _SKIP_REASON)
class TestImageSpaceChecks(unittest.TestCase):
    def test_check_inside_and_stacked(self):
        img = _canvas()
        _rect(img, 120, 120, 400, 400, "gray")          # 容器
        _disc(img, 250, 250, 18, "red")                # 主体在容器内
        inside = perception.check_inside(img, "red", (120, 120, 400, 400))
        print(f"[probe] check_inside={inside.get('ok')} why={inside.get('why')}")
        self.assertTrue(inside.get("ok"))
        img2 = _canvas()
        _rect(img2, 150, 300, 360, 460, "blue")         # 下层
        _disc(img2, 255, 380, 30, "red")               # 上层压在下层框内
        st = perception.check_stacked(img2, "red", "blue")
        print(f"[probe] check_stacked={st.get('ok')} why={st.get('why')}")
        self.assertTrue(st.get("ok"))

    def test_check_gone_and_overlap_no_crash(self):
        before = _disc(_canvas(), 200, 200, 20, "red")
        after = _canvas()
        gone = perception.check_gone(before, after, "red", (200, 200))
        self.assertIn("ok", gone)
        self.assertTrue(gone.get("ok"))
        self.assertGreaterEqual(perception.overlap_ratio((0, 0, 10, 10), (5, 5, 15, 15)), 0.0)
        self.assertTrue(perception.point_in_bbox((5, 5), (0, 0, 10, 10)))


if __name__ == "__main__":
    if _PENDING:
        print(f"[PENDING] {_SKIP_REASON} —— 需要 sim/perception.py 落地后本模块才会真正执行")
    unittest.main(verbosity=2)
