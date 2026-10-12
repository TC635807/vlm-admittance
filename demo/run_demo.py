"""VLM 决策驱动的抓取 demo。

用法:
    cd /home/tc63/mujoco/vlm_admittance
    # 离线（MockVLM，无需 API key）
    MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m demo.run_demo --task "把红色方块放进托盘，轻一点"
    # 接真实多模态大模型（OpenAI 兼容）
    VLM_API_KEY=sk-xxx VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 VLM_MODEL=qwen-vl-max-latest \
      MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m demo.run_demo --vlm api --task "把蓝色方块放进托盘"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from agent.loop import AgentLoop
from agent.vlm_client import make_vlm
from core.config import AGENT_MAX_STEPS, DEFAULT_COMPLIANCE, OUTPUT_DIR
from planning.collision import CollisionChecker
from planning.planner import JointPlanner
from control.admittance import AdmittanceController
from control.kinematics import Kinematics
from control.motion import MotionRunner
from sim.env import MuJoCoArmEnv
from skills import SkillExecutor


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="把红色方块放进托盘，轻一点")
    ap.add_argument("--vlm", choices=["auto", "mock", "api"], default="auto")
    ap.add_argument("--with-barrier", action="store_true")
    ap.add_argument("--max-steps", type=int, default=AGENT_MAX_STEPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(OUTPUT_DIR / "demo"))
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args(argv)

    env = MuJoCoArmEnv(with_barrier=args.with_barrier, seed=args.seed)
    kin = Kinematics(env)
    runner = MotionRunner(env, kin, AdmittanceController(DEFAULT_COMPLIANCE, env.dt))
    planner = JointPlanner(env, kin, CollisionChecker(env))
    skills = SkillExecutor(env, kin, runner, planner)

    prefer_api = args.vlm == "api"
    vlm = make_vlm(prefer_api=prefer_api)
    if args.vlm == "api" and not getattr(vlm, "available", False):
        print("[demo] 警告: 未检测到 VLM_API_KEY，回退 MockVLM")
    print(f"[demo] 决策器: {type(vlm).__name__}")
    print(f"[demo] 指令: {args.task}")

    outdir = Path(args.outdir)
    if args.save_images:
        outdir.mkdir(parents=True, exist_ok=True)

    step_counter = {"i": 0}
    original_executor = skills

    def executor(call):
        res = original_executor(call)
        if args.save_images:
            try:
                from PIL import Image
                step_counter["i"] += 1
                for cam in ("overview",):
                    Image.fromarray(env.render(cam)).save(outdir / f"step{step_counter['i']:02d}_{call.skill}.png")
            except Exception:  # noqa: BLE001
                pass
        return res

    loop = AgentLoop(env, executor=executor, vlm=vlm, cameras=("overview",),
                     max_steps=args.max_steps, log=print)
    t0 = time.time()
    results = loop.run(args.task)
    elapsed = time.time() - t0

    print("\n[demo] ==== 执行轨迹 ====")
    for i, r in enumerate(results, 1):
        flag = "OK " if r.ok else "FAIL"
        print(f"  {i:02d} {flag} {r.skill:<16} {r.message}")

    ok = bool(results) and any(r.ok and r.skill == "pick" for r in results) and any(
        r.ok and r.skill == "place" for r in results)
    print(f"[demo] 结果: {'成功' if ok else '未完成'}，用时 {elapsed:.1f}s")
    print("[demo] SUMMARY " + json.dumps(
        {"task": args.task, "vlm": type(vlm).__name__, "ok": ok,
         "results": [r.to_observation() for r in results], "elapsed": round(elapsed, 2)},
        ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
