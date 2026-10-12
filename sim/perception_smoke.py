"""sim/perception_smoke.py —— sim/perception.py 自检（INTERFACES.md §5.1 / §5.8 / §5.10）。

运行：
    cd /home/tc63/mujoco/vlm_admittance && MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m sim.perception_smoke

四组数字：
  (1) 合成图色块质心误差 px（9 种颜色，含 refine_pixel 精修）；
  (2) top 相机 + 已知 40 mm 方块 bbox_to_world 的 size_xy 误差 mm（换算基准 1.489 mm/px）；
  (3) 俯视视差 vs 双目三角（§5.10）：水瓶 top-only 反投影误差（应 >=60 mm）、
      水瓶双像素三角误差（应 <=5 mm）、方块双像素三角误差（应 <=3 mm）；
  (4) check_inside / check_stacked / check_gone 的正反例结论。
另做越界/单色/噪声/空图鲁棒性检查与 512x512 连通域耗时。

信息隔离：本脚本不调用 env.object_pose / set_object_pose / state.objects[*].pose。
第 (2)(3) 组的"真值"是脚本自己构造/传入的场景布局常量（DAILY_OBJECTS 仅作为建场输入），
第 (3) 组中 overview 的像素取"同一 3D 点在该相机的投影"，模拟 VLM 在 overview 图上点同一特征。
退出码 0 = 通过。
"""
from __future__ import annotations

import sys
import time

from . import _glinit  # noqa: F401  必须在 import mujoco 之前

import numpy as np

from core.config import TABLE_HEIGHT

from . import perception as P
from .env import MuJoCoArmEnv
from .scene import DAILY_OBJECTS, DEFAULT_OBJECTS
from .vision import camera_model, pixel_to_table, project_point

#: 合成图背景：偏棕的桌面色（刻意不属于 9 种可识别颜色，避免污染灰度判定）
BG = (110, 85, 60)
#: 合成色块的"真值"颜色（脚本自定，与 scene.COLOR_RGBA 无关）
PALETTE = {
    "red": (220, 40, 35),
    "green": (40, 190, 60),
    "blue": (35, 60, 200),
    "yellow": (230, 205, 40),
    "cyan": (60, 190, 225),
    "orange": (235, 120, 30),
    "white": (245, 245, 245),
    "black": (18, 18, 20),
    "gray": (128, 128, 132),
}
#: 第 (2) 组用自己构造的 40 mm 红方块（半尺寸 0.02），放在 top 相机轴线上
BOX_HALF = 0.02
BOX_XY = (0.36, 0.05)

CENTROID_TOL_PX = 2.0      # §5.8 要求 <= 2 px
SIZE_TOL_MM = 3.0          # §5.8 要求 <= 3 mm


def _new_image(h=240, w=320):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = BG
    return img


def _draw_circle(img, cu, cv, r, rgb):
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    m = (xs - float(cu)) ** 2 + (ys - float(cv)) ** 2 <= float(r) ** 2
    img[m] = rgb


def _draw_rect(img, cu, cv, half_w, half_h, rgb):
    u1, u2 = int(round(cu - half_w)), int(round(cu + half_w))
    v1, v2 = int(round(cv - half_h)), int(round(cv + half_h))
    img[max(v1, 0):v2 + 1, max(u1, 0):u2 + 1] = rgb
    return (u1, v1, u2, v2)


# --------------------------------------------------------------------------- #
def section1_synthetic_centroid():
    print("[1] 合成图色块质心误差（阈值 <= %.1f px）" % CENTROID_TOL_PX)
    img = _new_image()
    # (颜色, 形状, 真值中心, 尺寸, 粗像素偏移)
    cases = [
        ("red",    "circle", (50.4, 45.3), 16.0, (5.0, -4.0)),
        ("green",  "rect",   (130.7, 45.6), (17.0, 13.0), (-6.0, 3.0)),
        ("blue",   "circle", (215.2, 44.8), 14.0, (4.0, 5.0)),
        ("yellow", "circle", (285.5, 45.1), 12.0, (-5.0, -5.0)),
        ("cyan",   "rect",   (50.6, 150.4), (15.0, 11.0), (6.0, -3.0)),
        ("orange", "circle", (130.3, 150.7), 13.0, (-4.0, 6.0)),
        ("white",  "rect",   (215.4, 150.2), (14.0, 12.0), (5.0, 4.0)),
        ("black",  "rect",   (285.1, 150.5), (13.0, 11.0), (-6.0, -4.0)),
        ("gray",   "circle", (160.5, 210.3), 15.0, (4.0, -5.0)),
    ]
    for color, shape, (cu, cv), size, off in cases:
        if shape == "circle":
            _draw_circle(img, cu, cv, size, PALETTE[color])
        else:
            _draw_rect(img, cu, cv, size[0], size[1], PALETTE[color])

    max_err = 0.0
    max_refine_err = 0.0
    for color, shape, (cu, cv), size, off in cases:
        hint = (cu + off[0], cv + off[1])
        blob = P.color_blob(img, color, hint, radius_px=45)
        assert blob.found, "合成图 %s 未找到色块" % color
        err = float(np.hypot(blob.uv[0] - cu, blob.uv[1] - cv))
        ref = P.refine_pixel(img, color, hint, camera="top")
        assert ref["refined"], "%s refine 未成功" % color
        rerr = float(np.hypot(ref["uv"][0] - cu, ref["uv"][1] - cv))
        max_err = max(max_err, err)
        max_refine_err = max(max_refine_err, rerr)
        print("    %-7s %-6s center=(%.1f,%.1f) -> blob=(%.2f,%.2f) err=%.3f px | "
              "refine=(%.2f,%.2f) err=%.3f px delta=%.2f px area=%d score=%.3f"
              % (color, shape, cu, cv, blob.uv[0], blob.uv[1], err,
                 ref["uv"][0], ref["uv"][1], rerr, ref["delta_px"], blob.area_px, blob.score))
        assert err <= CENTROID_TOL_PX and rerr <= CENTROID_TOL_PX
    print("    质心误差: color_blob max=%.3f px, refine_pixel max=%.3f px（面积加权亚像素）"
          % (max_err, max_refine_err))

    # 颜色不存在 -> refined=False 且 uv 必须原样等于 hint
    green_only = _new_image(80, 80)
    _draw_circle(green_only, 40.0, 40.0, 12.0, PALETTE["green"])
    miss = P.refine_pixel(green_only, "red", (33.0, 27.0))
    print("    反例: 图里没有红色 -> refined=%s source=%s uv=%s（必须等于 hint）"
          % (miss["refined"], miss["source"], miss["uv"]))
    assert miss["refined"] is False and miss["source"] == "vlm_raw"
    assert miss["uv"] == (33.0, 27.0) and miss["delta_px"] == 0.0

    # 放大链路：zoom_window -> crop_upscale -> map_zoom_uv 往返
    win = P.zoom_window((cu, cv), size_px=40, scale=8)
    zoom = P.crop_upscale(img, win)
    back = P.map_zoom_uv(win["uv_zoom"], win)
    print("    zoom_window=%s" % {k: win[k] for k in ("u0", "v0", "size", "scale")})
    print("    crop_upscale=%s ; map_zoom_uv(uv_zoom)=%s（应等于 (%.1f, %.1f)）"
          % (zoom.shape, tuple(round(v, 2) for v in back), cu, cv))
    assert zoom.shape == (40 * 8, 40 * 8, 3)
    assert np.allclose(back, (cu, cv), atol=1e-9)
    return max_err, max_refine_err


# --------------------------------------------------------------------------- #
def section2_bbox_to_world():
    print("[2] top 相机 + 已知 40 mm 方块：bbox_to_world 的 size_xy 误差（阈值 %.1f mm）"
          % SIZE_TOL_MM)
    env = MuJoCoArmEnv(objects=[dict(id="box_red", category="box", color="red", shape="box",
                                     size=(BOX_HALF, BOX_HALF, BOX_HALF),
                                     pos=(BOX_XY[0], BOX_XY[1], TABLE_HEIGHT + BOX_HALF))])
    try:
        env.reset(seed=0)
        cam = camera_model(env, "top")
        focal = (cam.height / 2.0) / np.tan(np.radians(cam.fovy) / 2.0)
        mm_per_px = (cam.cam_pos[2] - TABLE_HEIGHT) / focal * 1000.0
        print("    换算基准: top 相机 %.2f mm/px（f=%.2f px, fovy=%.1f, 相机高 %.3f m, 桌面 z=%.2f）"
              % (mm_per_px, focal, cam.fovy, cam.cam_pos[2], TABLE_HEIGHT))
        rgb = env.render("top")

        blob = P.color_blob(rgb, "red")
        assert blob.found, "真实渲染图里没找到红方块"
        width_px = blob.bbox[2] - blob.bbox[0] + 1
        height_px = blob.bbox[3] - blob.bbox[1] + 1
        world = P.bbox_to_world(env, blob.bbox, camera="top")
        assert world["ok"]
        err_mm = [abs(world["size_xy"][0] - 2 * BOX_HALF) * 1000.0,
                  abs(world["size_xy"][1] - 2 * BOX_HALF) * 1000.0]
        print("    (a) 真实渲染: bbox=%s（%d x %d px，理论 %.1f px）-> size_xy=(%.2f, %.2f) mm "
              "误差=(%.2f, %.2f) mm"
              % (blob.bbox, width_px, height_px, 2 * BOX_HALF * 1000.0 / mm_per_px,
                 world["size_xy"][0] * 1000.0, world["size_xy"][1] * 1000.0, err_mm[0], err_mm[1]))
        print("        center_xy=(%.4f, %.4f)（构造真值 (%.2f, %.2f)）"
              % (world["center_xy"][0], world["center_xy"][1], BOX_XY[0], BOX_XY[1]))
        assert max(err_mm) <= SIZE_TOL_MM, "真实渲染 bbox_to_world 尺寸误差超 %.1f mm" % SIZE_TOL_MM

        # (b) 已知标定回算：把 40 mm 足迹四角用 project 成像素取整 -> bbox -> world
        corners = [(BOX_XY[0] + BOX_HALF, BOX_XY[1] - BOX_HALF),
                   (BOX_XY[0] - BOX_HALF, BOX_XY[1] - BOX_HALF),
                   (BOX_XY[0] - BOX_HALF, BOX_XY[1] + BOX_HALF),
                   (BOX_XY[0] + BOX_HALF, BOX_XY[1] + BOX_HALF)]
        px = [cam.project(np.array([x, y, TABLE_HEIGHT])) for x, y in corners]
        u1 = int(round(min(p[0] for p in px)))
        u2 = int(round(max(p[0] for p in px)))
        v1 = int(round(min(p[1] for p in px)))
        v2 = int(round(max(p[1] for p in px)))
        world2 = P.bbox_to_world(env, (u1, v1, u2, v2), camera="top")
        est = P.estimate_size(env, (u1, v1, u2, v2), camera="top")
        err2 = [abs(world2["size_xy"][0] - 2 * BOX_HALF) * 1000.0,
                abs(world2["size_xy"][1] - 2 * BOX_HALF) * 1000.0]
        print("    (b) 已知标定回算: 真值足迹像素 %s -> bbox=%s size_xy=(%.2f, %.2f) mm "
              "误差=(%.2f, %.2f) mm"
              % ([tuple(round(v, 2) for v in p) for p in px], (u1, v1, u2, v2),
                 world2["size_xy"][0] * 1000.0, world2["size_xy"][1] * 1000.0, err2[0], err2[1]))
        print("        estimate_size: half_xy=(%.4f, %.4f) long=%.2f mm short=%.2f mm"
              % (est["half_xy"][0], est["half_xy"][1],
                 est["long_side"] * 1000.0, est["short_side"] * 1000.0))
        assert max(err2) <= SIZE_TOL_MM

        # (c) refine_pixel 在真实图上的亚像素精度（用构造常量算期望像素）
        expected = cam.project(np.array([BOX_XY[0], BOX_XY[1], TABLE_HEIGHT + 2 * BOX_HALF]))
        coarse = (expected[0] + 8.0, expected[1] - 6.0)
        ref = P.refine_pixel(rgb, "red", coarse, camera="top", env=env)
        rerr = float(np.hypot(ref["uv"][0] - expected[0], ref["uv"][1] - expected[1]))
        print("    (c) 真实图 refine: 期望像素=(%.2f,%.2f) 粗像素=(%.2f,%.2f) -> 精修=(%.2f,%.2f) "
              "误差=%.2f px (%.2f mm) Δ=%.2f px"
              % (expected[0], expected[1], coarse[0], coarse[1],
                 ref["uv"][0], ref["uv"][1], rerr, rerr * mm_per_px, ref["delta_px"]))
        assert ref["refined"] and rerr <= CENTROID_TOL_PX
        return mm_per_px
    finally:
        env.close()


# --------------------------------------------------------------------------- #
def _bbox_center(blob):
    return ((blob.bbox[0] + blob.bbox[2]) / 2.0, (blob.bbox[1] + blob.bbox[3]) / 2.0)


def section3_stereo():
    """§5.10：俯视视差与双目三角。水瓶 200 mm / 方块 40 mm，取自 DAILY_OBJECTS 的建场参数。"""
    print("[3] 俯视视差 vs 双目三角（§5.10）")
    objs = [dict(DAILY_OBJECTS[0]), dict(DAILY_OBJECTS[2])]      # box_red + bottle_water
    box, bottle = objs[0], objs[1]
    box_top = np.array([box["pos"][0], box["pos"][1], box["pos"][2] + box["size"][2]])
    bot_top = np.array([bottle["pos"][0], bottle["pos"][1], bottle["pos"][2] + bottle["size"][2]])
    print("    建场常量: 方块顶面中心 %s (40x40 mm) / 水瓶顶面中心 %s (半径 %.0f mm, 高 %.0f mm)"
          % (np.round(box_top, 3).tolist(), np.round(bot_top, 3).tolist(),
             bottle["size"][0] * 1000.0, 2 * bottle["size"][2] * 1000.0))

    env = MuJoCoArmEnv(objects=objs)
    try:
        env.reset(seed=0)
        cam_top = camera_model(env, "top")
        cam_ov = camera_model(env, "overview")
        rgb_top = env.render("top")
        cyan = P.color_blob(rgb_top, "cyan")
        red = P.color_blob(rgb_top, "red")
        assert cyan.found and red.found, "top 图里没同时找到水瓶与方块"

        # ---- (3a) 水瓶 top-only：复现视差 ----
        plane_bot = pixel_to_table(env, cyan.uv[0], cyan.uv[1], "top")
        err_bot_single = float(np.hypot(plane_bot[0] - bot_top[0], plane_bot[1] - bot_top[1])) * 1000.0
        analytic = float(np.hypot(bot_top[0] - cam_top.cam_pos[0],
                                  bot_top[1] - cam_top.cam_pos[1]))             * (bot_top[2] - TABLE_HEIGHT) / (cam_top.cam_pos[2] - bot_top[2]) * 1000.0
        single = P.object_point(env, top_uv=cyan.uv)
        print("    (3a) 水瓶 top-only: cyan 质心 uv=(%.2f,%.2f) -> 按 z=%.2f 反投影 (%s)"
              % (cyan.uv[0], cyan.uv[1], TABLE_HEIGHT, np.round(plane_bot[:2], 4).tolist()))
        print("         误差 = %.2f mm（解析视差 %.2f mm）[要求 >=60 mm]；source=%s"
              % (err_bot_single, analytic, single["source"]))
        assert err_bot_single >= 60.0, "水瓶 top-only 视差未复现（%.2f mm）" % err_bot_single
        assert single["source"] == "single_view" and abs(single["z_used"] - TABLE_HEIGHT) < 1e-12

        # ---- (3b) 水瓶 双像素三角 ----
        ov_bot = cam_ov.project(bot_top)
        tri_bot = P.triangulate(env, cyan.uv, "top", ov_bot, "overview")
        err_bot_dual = float(np.linalg.norm(tri_bot["xyz"] - bot_top)) * 1000.0
        print("    (3b) 水瓶 双像素三角: top=(%.2f,%.2f) + overview=(%.2f,%.2f) -> xyz=%s"
              % (cyan.uv[0], cyan.uv[1], ov_bot[0], ov_bot[1], np.round(tri_bot["xyz"], 4).tolist()))
        print("         残差 = %.3f mm，误差 = %.3f mm [要求 <=5 mm]" % (tri_bot["residual_mm"], err_bot_dual))
        assert tri_bot["ok"] and err_bot_dual <= 5.0, "水瓶双像素三角误差 %.3f mm" % err_bot_dual
        dual = P.object_point(env, top_uv=cyan.uv, overview_uv=ov_bot)
        print("         object_point(top+overview): source=%s residual=%.3f mm xyz=%s"
              % (dual["source"], dual["residual_mm"], np.round(dual["xyz"], 4).tolist()))
        assert dual["source"] == "triangulate" and dual["ok"]

        # ---- (3c) 方块 双像素三角 + top-only 对照 ----
        ov_box = cam_ov.project(box_top)
        tri_box = P.triangulate(env, red.uv, "top", ov_box, "overview")
        err_box_dual = float(np.linalg.norm(tri_box["xyz"] - box_top)) * 1000.0
        plane_box = pixel_to_table(env, red.uv[0], red.uv[1], "top")
        err_box_single = float(np.hypot(plane_box[0] - box_top[0], plane_box[1] - box_top[1])) * 1000.0
        print("    (3c) 方块 双像素三角: top=(%.2f,%.2f) + overview=(%.2f,%.2f) -> 残差=%.3f mm，误差=%.3f mm [要求 <=3 mm]"
              % (red.uv[0], red.uv[1], ov_box[0], ov_box[1], tri_box["residual_mm"], err_box_dual))
        print("         对照 top-only 误差 = %.2f mm（矮方块的小视差，预期 6~9 mm）；"
              "bbox 中心代替质心: 双像素 %.3f mm"
              % (err_box_single,
                 float(np.linalg.norm(P.triangulate(env, _bbox_center(red), "top", ov_box,
                                                    "overview")["xyz"] - box_top)) * 1000.0))
        assert tri_box["ok"] and err_box_dual <= 3.0, "方块双像素三角误差 %.3f mm" % err_box_dual
        assert 3.0 <= err_box_single <= 12.0

        # ---- 纯几何自检 + 高度/失败路径 ----
        geo = P.triangulate(env, cam_top.project(bot_top), "top", ov_bot, "overview")
        print("    纯几何自检（同一 3D 点两视角投影）: 残差=%.2e mm 误差=%.2e mm"
              % (geo["residual_mm"], float(np.linalg.norm(geo["xyz"] - bot_top)) * 1000.0))
        assert geo["ok"] and geo["residual_mm"] < 1e-6
        print("    height_from_top_z(%.2f) = %.3f m（真值 %.3f m）"
              % (bot_top[2], P.height_from_top_z(bot_top[2]), 2 * bottle["size"][2]))
        assert abs(P.height_from_top_z(bot_top[2]) - 2 * bottle["size"][2]) < 1e-12

        bad = P.triangulate(env, cyan.uv, "top", (ov_bot[0] + 60.0, ov_bot[1]), "overview")
        print("    反例 不一致像素对(+60px): ok=%s why=%s residual=%.1f mm" % (bad["ok"], bad["why"], bad["residual_mm"]))
        assert bad["ok"] is False and bad["why"] == "residual_too_large"
        none = P.object_point(env)
        print("    反例 无像素: ok=%s source=%s why=%s" % (none["ok"], none["source"], none["why"]))
        assert none["ok"] is False and none["source"] == "none"
        single_hint = P.object_point(env, top_uv=cyan.uv, z_hint=bot_top[2])
        print("    单目 + z_hint=%.2f: source=%s 误差=%.3f mm"
              % (bot_top[2], single_hint["source"],
                 float(np.linalg.norm(single_hint["xyz"] - bot_top)) * 1000.0))
        assert abs(float(np.linalg.norm(single_hint["xyz"] - bot_top)) * 1000.0) < 5.0
        return err_bot_single, err_bot_dual, err_box_dual
    finally:
        env.close()


def section4_checks():
    print("[4] check_inside / check_stacked / check_gone 正反例")
    results = {}

    # --- check_inside ---
    img = _new_image(200, 240)
    container = _draw_rect(img, 120, 100, 60, 45, (128, 128, 132))      # 灰托盘
    inner = _draw_rect(img, 120, 100, 50, 35, BG)                      # 挖空成框
    _draw_circle(img, 120.0, 100.0, 12.0, PALETTE["red"])              # 红块在框内
    ok_in = P.check_inside(img, "red", container)
    img2 = _new_image(200, 240)
    _draw_rect(img2, 120, 100, 60, 45, (128, 128, 132))
    _draw_rect(img2, 120, 100, 50, 35, BG)
    _draw_circle(img2, 215.0, 30.0, 12.0, PALETTE["red"])              # 红块在框外
    no_in = P.check_inside(img2, "red", container)
    print("    正例 check_inside: ok=%s why=%s subject_uv=%s" % (ok_in["ok"], ok_in["why"], ok_in["metrics"]["subject_uv"]))
    print("    反例 check_inside: ok=%s why=%s subject_uv=%s" % (no_in["ok"], no_in["why"], no_in["metrics"]["subject_uv"]))
    assert ok_in["ok"] is True and no_in["ok"] is False
    results["inside"] = (ok_in["ok"], no_in["ok"])

    # --- check_stacked ---
    img = _new_image(220, 220)
    lower = _draw_rect(img, 110, 110, 40, 40, PALETTE["red"])          # 红块
    _draw_rect(img, 110, 110, 16, 16, PALETTE["green"])                # 绿块压中间 -> 红被切成 4 片
    ok_st = P.check_stacked(img, "green", "red")
    img2 = _new_image(220, 220)
    _draw_rect(img2, 110, 110, 40, 40, PALETTE["red"])
    _draw_rect(img2, 25, 25, 12, 12, PALETTE["green"])                 # 绿块远离红块
    no_st = P.check_stacked(img2, "green", "red")
    # 遮挡分片：绿条横跨整个红块 -> 红被切成上下两片（4 邻域不连通），
    # 必须靠 check_stacked 内部的"相邻下层块并框"才能判成功
    img3 = _new_image(220, 220)
    _draw_rect(img3, 110, 110, 40, 40, PALETTE["red"])
    _draw_rect(img3, 110, 110, 45, 16, PALETTE["green"])
    ok_split = P.check_stacked(img3, "green", "red")

    print("    正例 check_stacked: ok=%s why=%s overlap=%.3f merged=%d lower_bbox=%s"
          % (ok_st["ok"], ok_st["why"], ok_st["metrics"]["overlap"],
             ok_st["metrics"]["lower_merged_components"], ok_st["metrics"]["lower_bbox"]))
    print("    正例 check_stacked(遮挡分片): ok=%s why=%s overlap=%.3f merged=%d lower_bbox=%s"
          % (ok_split["ok"], ok_split["why"], ok_split["metrics"]["overlap"],
             ok_split["metrics"]["lower_merged_components"], ok_split["metrics"]["lower_bbox"]))
    print("    反例 check_stacked: ok=%s why=%s overlap=%.3f"
          % (no_st["ok"], no_st["why"], no_st["metrics"]["overlap"]))
    assert ok_st["ok"] is True and no_st["ok"] is False
    assert ok_split["ok"] is True
    assert ok_split["metrics"]["lower_merged_components"] >= 2, "遮挡分片应触发并框"
    results["stacked"] = (ok_st["ok"], ok_split["ok"], no_st["ok"])

    # --- check_gone ---
    before = _new_image(160, 160)
    _draw_circle(before, 80.0, 80.0, 15.0, PALETTE["red"])
    after_gone = _new_image(160, 160)
    after_same = before.copy()
    after_shrunk = _new_image(160, 160)
    _draw_circle(after_shrunk, 80.0, 80.0, 7.0, PALETTE["red"])        # 只剩不到 1/3 面积
    g1 = P.check_gone(before, after_gone, "red", (80, 80))
    g2 = P.check_gone(before, after_same, "red", (80, 80))
    g3 = P.check_gone(before, after_shrunk, "red", (80, 80))
    print("    正例 check_gone(消失): ok=%s why=%s area %d -> %d"
          % (g1["ok"], g1["why"], g1["metrics"]["area_before_px"], g1["metrics"]["area_after_px"]))
    print("    反例 check_gone(还在): ok=%s why=%s area %d -> %d"
          % (g2["ok"], g2["why"], g2["metrics"]["area_before_px"], g2["metrics"]["area_after_px"]))
    print("    正例 check_gone(面积<1/3): ok=%s why=%s ratio=%.3f"
          % (g3["ok"], g3["why"], g3["metrics"]["ratio"]))
    assert g1["ok"] is True and g2["ok"] is False and g3["ok"] is True
    results["gone"] = (g1["ok"], g2["ok"], g3["ok"])
    return results


# --------------------------------------------------------------------------- #
def section5_real_render_regression():
    """T7：真渲染回归（default 5 物体 + daily 5 物体），伪 VLM 像素 = 真值投影 ±8 px。

    参考真值来自**脚本自己传入的建场布局常量**（DEFAULT_OBJECTS / DAILY_OBJECTS），
    不调用 env.object_pose / state.objects[*].pose。
    列含义：
      d_px   = |refine_pixel.uv - 真值体心投影|
      floor  = |真值顶面中心投影 - 真值体心投影|，即 §5.10 俯视视差在像素上的"几何地板"
               （方块 40 mm 高 20 mm、水瓶高 100 mm；这部分与颜色判别无关）
      色差   = d_px - floor，颜色判别/分割真正贡献的误差
    判定（Lead 裁决 a）：红/蓝/绿 色差 = d_px - floor <= 1px 且 d_px<=max(3, floor+1)；黄 d_px<=5px；
          灰/白/黑这类与桌面同色的目标允许 refined=False（但不得给出错点：refined=True 时 d_px<=3）。
    """
    print("[5] 真渲染回归（default + daily 共 10 个物体，伪 VLM 像素 = 真值投影 ±8 px）")
    rng = np.random.default_rng(20261012)
    fails = []
    for scene, objs in (("default", DEFAULT_OBJECTS), ("daily", DAILY_OBJECTS)):
        env = MuJoCoArmEnv(objects=objs)
        try:
            env.reset(seed=0)
            env.step(60)                                  # 与 tools/vision_accuracy.py 一致
            rgb = env.render("top")
            print("    -- scene=%s --" % scene)
            print("      %-13s%-7s%-16s%-16s%-16s%7s%7s%7s  %s"
                  % ("object", "color", "truth_px", "coarse(+/-8px)", "refined_px",
                     "d_px", "floor", "色差", "refined"))
            for spec in objs:
                oid, color = spec["id"], str(spec["color"])
                center = np.asarray(spec["pos"], float)
                top = center + np.array([0.0, 0.0, float(spec["size"][2])])
                px = project_point(env, center, camera="top")
                pt = project_point(env, top, camera="top")
                floor = float(np.hypot(pt[0] - px[0], pt[1] - px[1]))
                coarse = (px[0] + float(rng.uniform(-8.0, 8.0)),
                          px[1] + float(rng.uniform(-8.0, 8.0)))
                out = P.refine_pixel(rgb, color, coarse, camera="top", env=env)
                d_px = float(np.hypot(out["uv"][0] - px[0], out["uv"][1] - px[1]))
                color_err = d_px - floor
                ok = True
                if color in ("gray", "white", "black"):
                    ok = (not out["refined"]) or d_px <= 3.0
                elif color == "yellow":
                    ok = d_px <= max(5.0, floor + 1.0)
                else:
                    # Lead 裁决 (a)：颜色判别的贡献 = 色差 = d_px - floor 必须 <= 1 px
                    ok = (color_err <= 1.0) and (d_px <= max(3.0, floor + 1.0))
                if not ok:
                    fails.append("%s/%s" % (scene, oid))
                print("      %-13s%-7s(%6.1f,%6.1f) (%6.1f,%6.1f) (%6.1f,%6.1f)%7.2f%7.2f%7.2f  %s%s"
                      % (oid, color, px[0], px[1], coarse[0], coarse[1],
                         out["uv"][0], out["uv"][1], d_px, floor, color_err,
                         out["refined"], "" if ok else "  <-- FAIL"))
            # 说明：灰/白/黑允许 refined=False，此时 uv 原样等于粗像素（上层用 VLM bbox 兜底）
        finally:
            env.close()
    print("    结论: 红/蓝/绿 色差<=1px 且 d_px<=max(3, floor+1)；黄 d_px<=5px；"
          "灰/白/黑 refined=False 或 d_px<=3 -> %s" % ("全部通过" if not fails else "FAIL: " + ",".join(fails)))
    assert not fails, "真渲染回归未通过：%s" % fails
    return True


def section6_robustness():
    print("[6] 鲁棒性（越界/单色/噪声/空图/退化输入，均不得崩）")
    uniform = np.zeros((64, 64, 3), dtype=np.uint8)
    uniform[:, :] = (110, 85, 60)
    noise = (np.random.default_rng(0).random((96, 96, 3)) * 255).astype(np.uint8)
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    flat = np.zeros((32, 32), dtype=np.uint8)          # 灰度单通道也要能处理
    checks = [
        ("单色图找红", P.color_blob(uniform, "red").found),
        ("噪声图找绿", P.color_blob(noise, "green").found),
        ("空图找红", P.color_blob(empty, "red").found),
        ("单通道图找红", P.color_blob(flat, "red").found),
        ("未知颜色", P.color_blob(uniform, "purple").found),
        ("越界 hint", P.color_blob(uniform, "red", (-500.0, 900.0)).found),
    ]
    for name, found in checks:
        print("    %-26s found=%s" % (name, found))
    assert not any(found for _n, found in checks)
    # bbox_to_world 用真实 env（远处/退化框不崩）
    env = MuJoCoArmEnv()
    try:
        env.reset(seed=0)
        w_far = P.bbox_to_world(env, (5000, 5000, 5010, 5010), camera="top")
        w_bad = P.bbox_to_world(env, (10, 10, 10, 10), camera="top")
        w_none = P.bbox_to_world(env, (10, 10), camera="top")
        print("    图像外框 bbox_to_world: ok=%s center=%s assumed_z=%s"
              % (w_far["ok"], np.round(w_far["center_xy"], 3).tolist(), w_far["assumed_z"]))
        print("    退化框(单像素)      : ok=%s size_xy=%s" % (w_bad["ok"], w_bad["size_xy"]))
        print("    非法框(2 元素)      : ok=%s why=%s" % (w_none["ok"], w_none.get("why")))
        big = np.zeros((512, 512, 3), dtype=np.uint8)
        big[:, :] = BG
        _draw_rect(big, 256, 256, 100, 100, PALETTE["red"])
        t0 = time.perf_counter()
        b = P.color_blob(big, "red")
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print("    512x512 连通域耗时: %.2f ms（RLE+并查集，非 O(n^2)）area=%d" % (dt_ms, b.area_px))
    finally:
        env.close()
    return dt_ms


def main() -> int:
    print("=" * 78)
    print("sim.perception_smoke —— 纯图像感知（INTERFACES.md §5.1）")
    print("=" * 78)
    e1, e2 = section1_synthetic_centroid()
    mmpp = section2_bbox_to_world()
    bot_single, bot_dual, box_dual = section3_stereo()
    results = section4_checks()
    section5_real_render_regression()
    dt_ms = section6_robustness()
    print("-" * 78)
    print("汇总:")
    print("  (1) 合成图质心最大误差: color_blob %.3f px / refine_pixel %.3f px（阈值 %.1f px）"
          % (e1, e2, CENTROID_TOL_PX))
    print("  (2) top 相机尺寸换算基准 %.3f mm/px；bbox_to_world 的 size_xy 误差见上（阈值 %.1f mm）"
          % (mmpp, SIZE_TOL_MM))
    print("  (3) §5.10 视差/三角: 水瓶 top-only %.2f mm（要求 >=60）| 水瓶双像素 %.3f mm（要求 <=5）"
          "| 方块双像素 %.3f mm（要求 <=3）" % (bot_single, bot_dual, box_dual))
    print("  (4) 判定正反例: inside=%s stacked=%s gone=%s %s"
          % (results["inside"], results["stacked"], results["gone"], "全部符合预期"))
    print("  (5) 真渲染回归（10 物体）见上表：红/蓝/绿 色差<=1px、黄 d_px<=5px、灰/白/黑 honesty")
    print("  (6) 越界/单色/噪声/空图/退化输入均未崩溃；512x512 连通域 %.2f ms" % dt_ms)
    print("PERCEPTION SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
