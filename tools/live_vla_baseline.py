"""Lead 实机诊断：真模型 + --perception vlm 链路跑一条任务。

只用于调试：**会读真值**（真值绝不进 prompt，仅用于事后判定「是识别错还是动作错」）。
运行时链路（AgentLoop / skills / perception）仍是无真值的。

用法:
    cd /home/tc63/mujoco/vlm_admittance
    MUJOCO_GL=glfw DISPLAY=:0 PYTHONPATH=. /home/tc63/mujoco/.venv/bin/python \
        tools/live_vla_baseline.py --task "把红色方块放到盘子里，绿色方块放到红方块上" --max-steps 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from core.config import DEFAULT_COMPLIANCE
from control.admittance import AdmittanceController
from control.kinematics import Kinematics
from control.motion import MotionRunner
from planning.collision import CollisionChecker
from planning.planner import JointPlanner
from sim.env import MuJoCoArmEnv
from sim.vision import pixel_to_table
from skills import SkillExecutor

try:                                    # §5.1 感知增强（可能尚未落地）
    from sim.perception import refine_pixel as _refine_pixel
except Exception:                        # noqa: BLE001
    _refine_pixel = None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-refine", action="store_true", help="不做颜色质心精修")
    ap.add_argument("--scene", choices=["default", "daily"], default="default")
    args = ap.parse_args(argv)

    objects = None
    if args.scene == "daily":
        from sim.scene import DAILY_OBJECTS
        objects = DAILY_OBJECTS

    env = MuJoCoArmEnv(objects=objects, seed=args.seed)
    kin = Kinematics(env)
    adm = AdmittanceController(DEFAULT_COMPLIANCE, env.dt)
    runner = MotionRunner(env, kin, adm)
    planner = JointPlanner(env, kin, CollisionChecker(env))
    skills = SkillExecutor(env, kin, runner, planner)

    from agent.loop import AgentLoop
    from agent.vlm_client import make_vlm
    vlm = make_vlm(prefer_api=True)
    print("[diag] vlm = %s (available=%s)" % (type(vlm).__name__, getattr(vlm, "available", "-")))

    truth0 = {o.id: np.array(env.object_pose(o.id)[:3], dtype=float) for o in env.list_objects()}
    colors = {o.id: getattr(o, "color", "?") for o in env.list_objects()}
    print("[diag] 真值(调试用) " + json.dumps(
        {k: [round(float(x), 3) for x in v] for k, v in truth0.items()}, ensure_ascii=False))

    calls = []

    def pixel_resolver(u, v, cam):
        raw = (float(u), float(v))
        refined_note = ""
        if not args.no_refine and _refine_pixel is not None:
            try:
                img = env.render(cam)
                color = None
                if calls:
                    txt = str(calls[-1].args)
                    for c in ("red", "blue", "green", "yellow", "cyan", "black", "gray"):
                        if c in txt:
                            color = c
                            break
                if color:
                    r = _refine_pixel(img, color, raw, camera=cam, env=env)
                    if r.get("refined"):
                        refined_note = " 精修 %s d=%.2fpx" % (
                            tuple(round(float(x), 1) for x in r["uv"]), float(r.get("delta_px", 0.0)))
                        u, v = r["uv"]
                    else:
                        refined_note = " 精修失败(用原像素)"
            except Exception as exc:      # noqa: BLE001
                refined_note = " 精修异常 %s" % exc
        p = pixel_to_table(env, float(u), float(v), cam)
        note = ""
        if p is not None:
            d = {oid: float(np.linalg.norm(np.array(p[:2]) - t[:2])) for oid, t in truth0.items()}
            near = min(d, key=d.get)
            note = " | 距最近真值 %s=%.1fmm" % (near, 1000 * d[near])
            print("[diag] 像素 (%.0f,%.0f)[%s]%s -> 世界 (%.3f,%.3f)%s" % (
                u, v, cam, refined_note, p[0], p[1], note))
        else:
            print("[diag] 像素 (%.0f,%.0f)[%s]%s -> 反投影失败" % (u, v, cam, refined_note))
        return p

    # 调试用：每执行完一个技能就打印一次真值（绝不进 prompt，只帮 Lead 判断"识别错还是动作错"）
    def exec_with_truth(call):
        result = skills(call)
        cur = {o.id: np.array(env.object_pose(o.id)[:3], dtype=float) for o in env.list_objects()}
        move = "  ".join("%s dz=%+6.1fmm dxy=%5.1fmm" % (
            oid, 1000 * (cur[oid][2] - truth0[oid][2]),
            1000 * float(np.linalg.norm(cur[oid][:2] - truth0[oid][:2])))
            for oid in truth0)
        print("[diag] 真值@%s -> %s" % (call.skill, move))
        return result

    loop = AgentLoop(env, exec_with_truth, vlm, cameras=("overview", "top"),
                     max_steps=args.max_steps, log=print, include_poses=False,
                     pixel_resolver=pixel_resolver)

    t0 = time.time()
    results = loop.run(args.task)
    elapsed = time.time() - t0

    print("\n[diag] ===== 逐步结果 =====")
    for i, r in enumerate(results, 1):
        print("%2d) %-12s ok=%-5s %s" % (i, r.skill, r.ok, str(r.message)[:110]))
        if r.metrics:
            print("      metrics=%s" % json.dumps(r.metrics, ensure_ascii=False)[:220])

    truth1 = {o.id: np.array(env.object_pose(o.id)[:3], dtype=float) for o in env.list_objects()}
    print("\n[diag] ===== 真值位移（调试用） =====")
    for oid in truth0:
        d = truth1[oid] - truth0[oid]
        print("   %-12s %-6s Δ=(%+.3f,%+.3f,%+.3f)  |Δxy|=%.1fmm" % (
            oid, colors.get(oid, "?"), d[0], d[1], d[2],
            1000 * float(np.linalg.norm(d[:2]))))
    print("\n[diag] 用时 %.1fs, 步数 %d" % (elapsed, len(results)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
