"""sim/smoke.py —— sim 模块无 GUI 自检（验收脚本）。

运行：
    cd /home/tc63/mujoco/vlm_admittance && MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m sim.smoke

流程：
    1) 打印 model 的 nq/nv/nu/neq/dt 与三个相机名；
    2) reset(seed=0) 后用 SceneRenderer 渲染 overview/wrist/top 三张 PNG 存到 outputs/，
       并额外存一张 overview 深度预览图；
    3) 打印静止（无接触）时的末端力读数，应约等于 0；
    4) 设置一组指向 can_yellow 上方、夹爪朝下的关节角（由 IK 求得），step 若干步；
    5) 打印 TCP 位姿、夹爪开合比例与物体列表；
    6) 自检断言：SimEnv 协议、三张 PNG 存在且非纯色、静止力≈0。

退出码 0 = 通过。
"""
from __future__ import annotations

import sys

from . import _glinit  # noqa: F401  必须在 import mujoco 之前

import numpy as np

from core.config import OUTPUT_DIR
from core.interfaces import SimEnv

from .env import MuJoCoArmEnv

#: 演示位形：TCP 到 (0.40, 0.02, 0.62)（正好在 can_yellow 上方），g_base 的 +z 朝下。
#: 该关节角由 scipy least_squares 对 g_pinch site 位置 + g_base 朝向求 IK 得到。
DEMO_QPOS = np.array([-3.4328, -1.8123, 2.0699, -1.8284, -1.5708, 4.7228])


def _save_png(path, rgb) -> str:
    from PIL import Image

    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(str(path))
    return str(path)


def _save_depth_png(path, depth, far: float = 2.0) -> str:
    from PIL import Image

    norm = np.clip(1.0 - np.asarray(depth, dtype=np.float32) / float(far), 0.0, 1.0)
    Image.fromarray((norm * 255.0).astype(np.uint8), mode="L").save(str(path))
    return str(path)


def main() -> int:
    import mujoco  # 仅用于打印版本

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("sim.smoke —— MuJoCo %s | 场景 UR5e + Robotiq 2F85 + 桌面 + 物体" % mujoco.__version__)
    print("=" * 72)

    env = MuJoCoArmEnv()
    m = env.model
    print("[1] 模型: nq=%d nv=%d nu=%d neq=%d dt=%.4f" % (m.nq, m.nv, m.nu, m.neq, env.dt))
    print("    相机 camera_names = %r" % (env.camera_names,))
    print("    arm_qpos_idx = %s, arm_dof_idx = %s" % (env.arm_qpos_idx, env.arm_dof_idx))
    print("    SimEnv 协议满足: %s" % isinstance(env, SimEnv))
    assert isinstance(env, SimEnv), "MuJoCoArmEnv 不满足 core.interfaces.SimEnv"

    state = env.reset(seed=0)

    print("[2] 渲染三张 RGB 到 outputs/ 并检查深度：")
    png_paths = []
    for cam in env.camera_names:
        rgb = env.render(cam)
        path = OUTPUT_DIR / ("sim_%s.png" % cam)
        _save_png(path, rgb)
        png_paths.append(path)
        depth = env.render_depth(cam)
        finite = depth[np.isfinite(depth)]
        # 只有落在 [0, 2)m 内的像素才算"场景内"
        inside = finite[(finite > 0.0) & (finite < 2.0)]
        print("    %-8s rgb=%s uint8  std=%.1f | depth=%s 场景内像素=%.1f%% 中位=%.3fm 最近=%.3fm"
              % (cam, rgb.shape, float(rgb.std()),
                 depth.shape, 100.0 * inside.size / max(finite.size, 1),
                 float(np.median(inside)) if inside.size else float("nan"),
                 float(inside.min()) if inside.size else float("nan")))
        assert rgb.shape == (env.render_size[1], env.render_size[0], 3), "RGB 形状不对"
        assert rgb.std() > 1.0, "%s 渲染图近似纯色，可能没渲染到东西" % cam
        assert inside.size > 0, "%s 深度图没有场景内像素" % cam
        assert path.exists() and path.stat().st_size > 0
    dpath = _save_depth_png(OUTPUT_DIR / "sim_depth_overview.png", env.render_depth("overview"))
    print("    已保存: %s" % ", ".join(str(p) for p in png_paths))
    print("    深度预览: %s" % dpath)

    force, torque = env.contact_force_on_ee()
    print("[3] 静止时末端力读数: force=%s |force|=%.6f N  torque=%s" %
          (np.round(force, 6), float(np.linalg.norm(force)), np.round(torque, 6)))
    assert float(np.linalg.norm(force)) < 1e-6, "静止无接触时末端力应约为 0"
    print("    in_contact=%s" % state.contact.in_contact)

    print("[4] 设置关节目标（指向 can_yellow 上方）并 step 1500 步：")
    print("    q_target = %s" % np.round(DEMO_QPOS, 4))
    env.set_arm_joint_targets(DEMO_QPOS)
    env.set_gripper(1.0)
    state = env.step(1500)
    err = float(np.linalg.norm(state.robot.tcp_pos - np.array([0.40, 0.02, 0.62])))
    print("    step 后 time=%.3fs  q=%s" % (state.time, np.round(state.robot.qpos, 4)))
    print("    TCP pos=%s  quat(wxyz)=%s" % (np.round(state.robot.tcp_pos, 4),
                                             np.round(state.robot.tcp_quat, 4)))
    print("    TCP 与 IK 目标距离 = %.4f m；夹爪 open_ratio=%.3f" % (err, state.robot.gripper))
    f2, t2 = env.contact_force_on_ee()
    print("    运动后末端力: force=%s |force|=%.4f N in_contact=%s" %
          (np.round(f2, 4), float(np.linalg.norm(f2)), state.contact.in_contact))

    print("[5] 物体列表 list_objects()（顺序固定 %r）：" % (list(env._obj_ids),))
    for o in state.objects:
        print("    %s" % o.to_json())
    print("    holding=%s  ncon=%d" % (state.robot.holding, state.meta["ncon"]))

    print("[6] 自检通过：nq/nv/nu 与契约一致、3 张 PNG 已生成、静止力≈0。")
    print("SMOKE OK")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
