#!/usr/bin/env bash
# =============================================================================
#  vlm_admittance 启动脚本
#  ---------------------------------------------------------------------------
#  用法：  ./run.sh [子命令] [额外参数...]
#  例：    ./run.sh                        # 交互式：MuJoCo 窗口 + 命令台
#          ./run.sh headless               # 无窗口（自动化/自测一律用这个）
#          ./run.sh daily                  # 日常物品场景（水瓶/眼镜盒）
#          ./run.sh task "蓝色方块放到绿色方块上"   # 一条命令跑完就退出
#          ./run.sh panels                 # 浏览器打开画面/曲线面板
#          ./run.sh doctor                 # 打印环境与配置体检
#  ---------------------------------------------------------------------------
#  注意：只有 [demo]/[daily] 这些"人工使用"的子命令会开 MuJoCo 窗口；
#        任何脚本、测试、CI 都必须用 [headless]，否则会给用户桌面弹窗。
# =============================================================================
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJ_DIR"

DEFAULT_PY="/home/tc63/mujoco/.venv/bin/python"
PY="${VLM_PY:-$DEFAULT_PY}"
if [ ! -x "$PY" ]; then PY="$(command -v python3 || true)"; fi
if [ -z "$PY" ]; then echo "[run.sh] 找不到 python 解释器（可用 VLM_PY=... 指定）" >&2; exit 1; fi

export MUJOCO_GL="${MUJOCO_GL:-glfw}"          # 本机 EGL 不可用，必须 glfw
export DISPLAY="${DISPLAY:-:0}"                # 有窗口/离屏渲染都要
export PYTHONPATH="$PROJ_DIR${PYTHONPATH:+:$PYTHONPATH}"
PANELS_DIR="$PROJ_DIR/outputs/panels"

log() { printf '[run.sh] %s\n' "$*"; }
die() { printf '[run.sh] 错误：%s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
vlm_admittance 启动脚本

  ./run.sh [子命令] [额外参数...]

子命令（默认 demo）:
  demo        交互式：MuJoCo 窗口 + 终端命令台（人工使用，默认真模型 + 纯视觉感知）
  daily       同 demo，但用日常物品场景（水瓶 / 眼镜盒 / 红绿方块 / 托盘）
  headless    无窗口纯终端；自动化、自测、CI 一律用这个（面板图照常产出）
  task TXT    跑一条自然语言任务就退出（等于 headless + 自动输入 TXT + :quit）
  e2e         脚本化端到端验收（home -> pick -> place，不走大模型）
  smoke       跑各模块冒烟（sim / perception / agent / skills / control / planning）
  test        跑 tests/ 下的独立测试
  panels      用系统默认浏览器打开 outputs/panels/index.html（画面 + 力矩曲线）
  doctor      打印解释器 / MuJoCo / 模型 / 显示环境 / 面板路径
  help        显示本帮助

透传参数会原样交给 demo.interactive，例如:
  ./run.sh demo --vlm api --perception vlm --planner subgoal --realtime
  ./run.sh headless --vlm mock --perception oracle --max-steps 6
  ./run.sh task "把红色方块放到盘子里，绿色方块放到红方块上"

常用环境变量:
  VLM_PY      指定 python 解释器（默认 /home/tc63/mujoco/.venv/bin/python）
  DISPLAY     默认 :0 ；MUJOCO_GL 默认 glfw
  AI_API_URL / AI_API_KEY / AI_MODEL   真模型端点（也可写 config.local.json）
EOF
}

open_panels() {
  local f="$PANELS_DIR/index.html"
  [ -f "$f" ] || { log "面板还没生成（先跑一次任务）：$PANELS_DIR"; return 0; }
  if   command -v xdg-open >/dev/null 2>&1; then xdg-open "$f" >/dev/null 2>&1 &
  elif command -v wslview  >/dev/null 2>&1; then wslview  "$f" >/dev/null 2>&1 &
  elif [ -x /mnt/c/Windows/explorer.exe ]; then
       local w; w="$(wslpath -w "$f" 2>/dev/null || echo "$f")"
       /mnt/c/Windows/explorer.exe "$w" >/dev/null 2>&1 &
  elif command -v open >/dev/null 2>&1; then open "$f" >/dev/null 2>&1 &
  else log "没找到可用的打开器，请手动打开：$f"; return 0; fi
  log "已请求打开 $f（每 500ms 自动刷新）"
}

doctor() {
  log "项目目录 : $PROJ_DIR"
  log "解释器   : $PY"
  "$PY" - <<'PYEOF'
import sys
print("  python   :", sys.version.split()[0])
for mod in ("numpy", "mujoco", "PIL", "matplotlib"):
    try:
        m = __import__(mod); print("  %-9s:" % mod, getattr(m, "__version__", "ok"))
    except Exception as e:  print("  %-9s: 缺失（%s）" % (mod, type(e).__name__))
try:
    from core.config import VLM_BASE_URL, VLM_MODEL, VLM_TIMEOUT, VLM_API_KEY, LOCAL_CONFIG_SOURCE
    key = (VLM_API_KEY[:4] + "****" + VLM_API_KEY[-4:]) if VLM_API_KEY else "未配置"
    print("  VLM      :", VLM_MODEL, "@", VLM_BASE_URL, "| timeout", VLM_TIMEOUT, "s")
    print("  VLM key  :", key, "| 配置来源", LOCAL_CONFIG_SOURCE)
except Exception as e:
    print("  VLM 配置 : 读取失败（%s）" % e)
PYEOF
  log "MUJOCO_GL : $MUJOCO_GL"
  log "DISPLAY   : $DISPLAY"
  log "面板输出  : $PANELS_DIR（llm_frames.png / curves.png / index.html）"
}

MODE="${1:-demo}"
[ $# -gt 0 ] && shift || true

case "$MODE" in
  demo|"")
    log "交互模式：会打开 MuJoCo 窗口 + 终端命令台（自动化请改用 ./run.sh headless）"
    log "看画面/曲线：命令台敲 :panels open ，或另开终端跑 ./run.sh panels"
    exec "$PY" -u -m demo.interactive "$@"
    ;;
  daily)
    log "交互模式（日常物品场景）：水瓶 / 眼镜盒 / 红绿方块 / 托盘"
    exec "$PY" -u -m demo.interactive --scene daily "$@"
    ;;
  headless)
    exec "$PY" -u -m demo.interactive --headless "$@"
    ;;
  task)
    [ $# -ge 1 ] || die 'task 需要一条任务文本，例如 ./run.sh task "把红色方块放进托盘"'
    TASK_TEXT="$1"; shift || true
    log "任务：$TASK_TEXT"
    printf '%s\n:quit\n' "$TASK_TEXT" | exec "$PY" -u -m demo.interactive --headless "$@"
    ;;
  e2e)
    exec "$PY" -u -m demo.scripted_pick_place "$@"
    ;;
  smoke)
    for m in sim.vision_smoke sim.perception_smoke sim.smoke agent.smoke skills.smoke control.smoke planning.smoke; do
      printf '=== %s ===\n' "$m"
      "$PY" -m "$m" 2>&1 | tail -2
    done
    log "smoke 结束"
    ;;
  test)
    "$PY" -m unittest discover -s tests -p 'test_*.py' -v 2>&1 | tail -5
    ;;
  panels)
    open_panels
    ;;
  doctor)
    doctor
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    die "未知子命令：$MODE（试 ./run.sh help）"
    ;;
esac
