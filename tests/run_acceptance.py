"""全链路验收：逐模块 smoke + 端到端抓放。

用法（在工程根）:
    MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m tests.run_acceptance
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

CASES = [
    ("core", [PY, "-c", "from core import SceneState, ComplianceParams; import core.config as c; "
                        "assert c.UR5E_XML.exists() and c.GRIPPER_XML.exists(); print('core OK')"]),
    ("sim", [PY, "-m", "sim.smoke"]),
    ("control", [PY, "-m", "control.smoke"]),
    ("planning", [PY, "-m", "planning.smoke"]),
    ("agent", [PY, "-m", "agent.smoke"]),
    ("e2e_scripted", [PY, "-m", "demo.scripted_pick_place", "--target", "box_red", "--container", "tray"]),
    ("interactive_selftest", [PY, "-m", "demo.interactive", "--headless", "--self-test"]),
    ("interactive_flows", [PY, "-m", "tests.test_interactive_flows"]),
    ("target_overlay", [PY, "-m", "tests.test_target_overlay"]),
    ("place_semantics", [PY, "-m", "tests.test_place_semantics"]),
]


def main() -> int:
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "glfw")
    report = []
    for name, cmd in CASES:
        t0 = time.time()
        print(f"\n{'=' * 70}\n[acceptance] {name}: {' '.join(cmd)}\n{'=' * 70}", flush=True)
        try:
            p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
            out = (p.stdout or "") + (p.stderr or "")
            tail = "\n".join(out.strip().splitlines()[-25:])
            ok = p.returncode == 0
        except subprocess.TimeoutExpired:
            tail, ok = "TIMEOUT after 600s", False
        dt = time.time() - t0
        print(tail, flush=True)
        report.append({"case": name, "ok": bool(ok), "seconds": round(dt, 1)})
        print(f"[acceptance] {name}: {'PASS' if ok else 'FAIL'} ({dt:.1f}s)", flush=True)

    print("\n" + "=" * 70)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    passed = sum(1 for r in report if r["ok"])
    print(f"[acceptance] {passed}/{len(report)} 通过")
    return 0 if passed == len(report) else 1


if __name__ == "__main__":
    sys.exit(main())
