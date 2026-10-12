"""脚本化端到端验收（不依赖 VLM）：home -> pick -> place，验证 sim/control/planning/skills 闭环。

用法:
    cd /home/tc63/mujoco/vlm_admittance
    MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m demo.scripted_pick_place --target box_red --container tray
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from core.config import DEFAULT_COMPLIANCE, OUTPUT_DIR
from core.types import SkillCall
from planning.collision import CollisionChecker
from planning.planner import JointPlanner
from control.admittance import AdmittanceController
from control.kinematics import Kinematics
from control.motion import MotionRunner
from sim.env import MuJoCoArmEnv
from skills import SkillExecutor


def _save(env, outdir: Path, tag: str) -> None:
    try:
        from PIL import Image
        outdir.mkdir(parents=True, exist_ok=True)
        for cam in env.camera_names:
            img = env.render(cam)
            Image.fromarray(img).save(outdir / f"{tag}_{cam}.png")
    except Exception as exc:  # noqa: BLE001
        print(f"[e2e] 保存图像失败（不影响验收）: {exc}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="box_red")
    ap.add_argument("--container", default="tray")
    ap.add_argument("--with-barrier", action="store_true")
    ap.add_argument("--outdir", default=str(OUTPUT_DIR / "e2e"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    t_start = time.time()
    env = MuJoCoArmEnv(with_barrier=args.with_barrier, seed=args.seed)
    kin = Kinematics(env)
    adm = AdmittanceController(DEFAULT_COMPLIANCE, env.dt)
    runner = MotionRunner(env, kin, adm)
    checker = CollisionChecker(env)
    planner = JointPlanner(env, kin, checker)
    skills = SkillExecutor(env, kin, runner, planner)

    outdir = Path(args.outdir)
    _save(env, outdir, "00_start")

    print(f"[e2e] 物体: {[o.id for o in env.list_objects()]}")
    print(f"[e2e] 初始 TCP: {np.round(env.site_pose('g_pinch')[0], 3)}")

    results = []
    t_home = time.time()
    r = skills(SkillCall("home"))
    results.append(r)
    print(f"[e2e] home  -> ok={r.ok} {r.message} ({time.time()-t_home:.1f}s)")

    t_pick = time.time()
    r = skills(SkillCall("pick", {"target_id": args.target}))
    results.append(r)
    print(f"[e2e] pick  -> ok={r.ok} {r.message} metrics={r.metrics} ({time.time()-t_pick:.1f}s)")
    _save(env, outdir, "01_after_pick")

    t_place = time.time()
    r = skills(SkillCall("place", {"container_id": args.container}))
    results.append(r)
    print(f"[e2e] place -> ok={r.ok} {r.message} metrics={r.metrics} ({time.time()-t_place:.1f}s)")
    _save(env, outdir, "02_after_place")

    summary = {
        "ok": all(x.ok for x in results),
        "results": [x.to_observation() for x in results],
        "elapsed": round(time.time() - t_start, 2),
        "images": str(outdir),
    }
    print("[e2e] SUMMARY " + json.dumps(summary, ensure_ascii=False))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
