"""sim/vision_smoke.py —— sim/vision.py 的自检脚本（INTERFACES.md §4.1 验收）。

运行：
    cd /home/tc63/mujoco/vlm_admittance && MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m sim.vision_smoke

检查项：
  (1) 桌面已知点 world -> project -> pixel -> pixel_to_table 往返误差 < 1 mm，
      top 与 overview 两个相机各测 >=5 个点；
  (2) 反投影点的 z 与给定平面一致（< 1e-6 m）；
  (3) 相机后方 / 像素越界的行为与 docstring 一致（project 返回 None、
      pixel_to_plane 仍按针孔模型求交、ray 任意像素不崩溃）；
  (4) 标定确实来自 mj_forward（cam_pos 与独立算出的 cam_xpos 一致；
      wrist 相机随机械臂运动自动更新，说明没有硬编码相机位置）；
  (5) 附加交叉验证（需要 GL，失败则跳过）：把物体中心投影像素在渲染图上采样，
      颜色应与物体颜色一致 —— 用来证明 u/v 方向与 fovy 标定没有反。
  (6) verifier 工具 reprojection_summary 的输出。

退出码 0 = 通过。
"""
from __future__ import annotations

import sys

from . import _glinit  # noqa: F401  必须在 import mujoco 之前

import mujoco
import numpy as np

from core.config import HOME_QPOS, TABLE_HEIGHT

from .env import MuJoCoArmEnv
from .vision import (
    CameraModel,
    camera_model,
    pixel_to_table,
    project_point,
    reprojection_summary,
)

ROUNDTRIP_TOL: float = 1e-3      # 1 mm（契约）
PLANE_TOL: float = 1e-6          # 反投影 z 与平面一致
MIN_POINTS: int = 5              # verifier §4.5(2) 要求 >=5 个桌面点

#: 用于往返测试的桌面点 xy（物体中心 + 2 个额外点），z 统一取 TABLE_HEIGHT
TABLE_XY = [
    (0.35, -0.14), (0.35, 0.14), (0.45, -0.05), (0.40, 0.02), (0.40, 0.30),
    (0.30, 0.10), (0.50, 0.20),
]

#: 渲染图颜色交叉验证（物体 id -> 主导通道判据）
COLOR_CHECKS = {
    "box_red": lambda r, g, b: r > g + 20 and r > b + 20,
    "box_blue": lambda r, g, b: b > r + 20 and b > g + 20,
    "box_green": lambda r, g, b: g > r + 20 and g > b + 20,
    "can_yellow": lambda r, g, b: r > b + 20 and g > b + 20,
}


def _roundtrip(cam: CameraModel, points: np.ndarray):
    """返回 (pixels, backs, errors)；投影失败的点记 nan。"""
    pixels, backs, errors = [], [], []
    for p in points:
        uv = cam.project(p)
        if uv is None:
            pixels.append(None)
            backs.append(None)
            errors.append(float("nan"))
            continue
        back = cam.pixel_to_plane(uv[0], uv[1], TABLE_HEIGHT)
        pixels.append(uv)
        backs.append(back)
        errors.append(float(np.linalg.norm(back - p)) if back is not None else float("nan"))
    return pixels, backs, np.array(errors, dtype=float)


def main() -> int:
    print("=" * 78)
    print("sim.vision_smoke —— 相机标定与像素<->世界反投影（MuJoCo %s）" % mujoco.__version__)
    print("=" * 78)
    env = MuJoCoArmEnv()
    env.reset(seed=0)

    points = np.array([[x, y, TABLE_HEIGHT] for x, y in TABLE_XY], dtype=float)
    ok = True

    # ------------------------------------------------------------------ #
    print("[1][2] 桌面点往返误差与反投影 z（阈值：%.0e m / %.0e m）" % (ROUNDTRIP_TOL, PLANE_TOL))
    for cam_name in ("top", "overview"):
        cam = camera_model(env, cam_name)
        pixels, backs, errors = _roundtrip(cam, points)
        valid = np.isfinite(errors)
        n_ok = int(valid.sum())
        max_err = float(errors[valid].max()) if n_ok else float("nan")
        z_err = max((abs(float(b[2]) - TABLE_HEIGHT) for b in backs if b is not None),
                    default=float("nan"))
        print("   %-8s fovy=%.1f 分辨率=%dx%d 有效点=%d/%d 最大往返误差=%.3e m(%.3e mm) "
              "最大 z 偏差=%.3e m"
              % (cam_name, cam.fovy, cam.width, cam.height, n_ok, points.shape[0],
                 max_err, max_err * 1000.0, z_err))
        for (x, y), uv, back, e in zip(TABLE_XY, pixels, backs, errors):
            print("      world=(%.2f, %.2f, %.3f) -> uv=%s -> back=%s err=%.3e m"
                  % (x, y, TABLE_HEIGHT,
                     ("(%.2f, %.2f)" % uv) if uv else "None",
                     np.round(back, 6).tolist() if back is not None else "None", e))
        assert n_ok >= MIN_POINTS, "%s 有效点不足 %d 个" % (cam_name, MIN_POINTS)
        assert max_err < ROUNDTRIP_TOL, "%s 往返误差 %.3e 超过 %.0e" % (cam_name, max_err, ROUNDTRIP_TOL)
        assert z_err < PLANE_TOL, "%s 反投影 z 偏差 %.3e 超过 %.0e" % (cam_name, z_err, PLANE_TOL)

    # ------------------------------------------------------------------ #
    print("[3] 相机后方 / 像素越界行为（不得崩溃）")
    cam = camera_model(env, "top")
    behind = np.array([cam.cam_pos[0], cam.cam_pos[1], cam.cam_pos[2] + 0.2])
    print("   project(相机后方点 %s) = %s" % (np.round(behind, 3).tolist(), cam.project(behind)))
    assert cam.project(behind) is None
    far = np.array([5.0, 5.0, TABLE_HEIGHT])
    print("   project(远处桌外点 %s) = %s" % (np.round(far, 3).tolist(), cam.project(far)))
    assert cam.project(far) is None, "图像外的点 project 应返回 None"
    o, d = cam.ray(-50.0, 1e6)
    print("   ray(-50, 1e6) origin=%s dir=%s |dir|=%.6f"
          % (np.round(o, 3).tolist(), np.round(d, 4).tolist(), float(np.linalg.norm(d))))
    assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-12
    outside = pixel_to_table(env, 5000.0, 5000.0, "top")
    print("   pixel_to_table(5000, 5000) = %s（图像外仍按针孔求交，允许）"
          % np.round(outside, 3).tolist())
    assert outside is not None and abs(float(outside[2]) - TABLE_HEIGHT) < PLANE_TOL
    above = pixel_to_table(env, 256, 256, "top", z=cam.cam_pos[2] + 0.5)
    print("   pixel_to_table(z=相机上方平面) = %s（t<=0 -> None）" % above)
    assert above is None
    assert cam.ray(0.0, 0.0)[1].shape == (3,)

    # ------------------------------------------------------------------ #
    print("[4] 标定来自 mj_forward（不硬编码相机位置）")
    priv = mujoco.MjData(env.model)
    priv.qpos[:] = env.data.qpos
    mujoco.mj_forward(env.model, priv)
    for cam_name in ("top", "overview", "wrist"):
        c = camera_model(env, cam_name)
        cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        ref = priv.cam_xpos[cid]
        err = float(np.linalg.norm(c.cam_pos - ref))
        print("   %-8s cam_pos=%s（与独立 mj_forward 差 %.2e m）"
              % (cam_name, np.round(c.cam_pos, 4).tolist(), err))
        assert err < 1e-12, "%s 的 cam_pos 与 mj_forward 不一致" % cam_name
    wrist = camera_model(env, "wrist")
    before = wrist.cam_pos.copy()
    env.set_arm_joint_targets(HOME_QPOS + np.array([0.0, 0.25, -0.25, 0.0, 0.2, 0.0]))
    env.step(300)
    wrist.project(np.array([0.4, 0.02, TABLE_HEIGHT + 0.3]))       # 触发 _sync
    moved = float(np.linalg.norm(wrist.cam_pos - before))
    print("   wrist 相机随臂移动: |Δcam_pos|=%.4f m（> 0 说明位姿是实时算的）" % moved)
    assert moved > 1e-3, "wrist 相机位姿没有随机械臂更新（可能硬编码了）"

    # ------------------------------------------------------------------ #
    print("[5] 附加交叉验证：投影像素在渲染图上的颜色（需要 GL，失败则跳过）")
    try:
        env.reset(seed=0)
        for cam_name in ("top", "overview"):
            rgb = env.render(cam_name)
            c = camera_model(env, cam_name)
            checked = 0
            for oid, pred in COLOR_CHECKS.items():
                pose = env.object_pose(oid)
                uv = c.project(pose[:3])
                if uv is None:
                    print("      %s/%s 中心不在图像内，跳过" % (cam_name, oid))
                    continue
                u_i, v_i = int(round(uv[0])), int(round(uv[1]))
                if not (0 <= u_i < c.width and 0 <= v_i < c.height):
                    continue
                r, g, b = (int(v) for v in rgb[v_i, u_i])
                good = bool(pred(r, g, b))
                checked += 1
                print("      %-8s %-10s uv=(%.1f,%.1f) rgb=[%d,%d,%d] 颜色判据=%s"
                      % (cam_name, oid, uv[0], uv[1], r, g, b, "OK" if good else "FAIL"))
                assert good, "%s/%s 的投影像素颜色不符，u/v 方向或 fovy 标定可能反了" % (cam_name, oid)
            assert checked >= 4, "%s 颜色交叉验证点数不足" % cam_name
    except Exception as exc:                      # GL/显示不可用时只告警不失败
        if isinstance(exc, AssertionError):
            raise
        print("      跳过（需要 GL）：%s: %s" % (type(exc).__name__, exc))

    # ------------------------------------------------------------------ #
    print("[6] verifier 工具 reprojection_summary（>=5 个桌面点，供 §4.5(2) 用）")
    for cam_name in ("top", "overview"):
        summary = reprojection_summary(env, points, camera=cam_name, z=TABLE_HEIGHT)
        print("   %-8s n=%d n_failed=%d mean=%.3e m max=%.3e m"
              % (cam_name, summary["n"], summary["n_failed"],
                 summary["mean_m"], summary["max_m"]))
        assert summary["n"] >= MIN_POINTS
        assert summary["n_failed"] == 0
        assert summary["max_m"] < ROUNDTRIP_TOL

    print("VISION SMOKE OK")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
