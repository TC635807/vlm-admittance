# vlm_admittance

**多模态大模型(VLM)高层决策 + 路径规划 + 导纳控制** 的 MuJoCo 机械臂抓取框架。
**不需要训练 VLA**：底层是经典控制（DLS IK / 笛卡尔轨迹 / 导纳外环 / RRT-Connect 规划），
上层用支持图片输入的大模型（任意 OpenAI 兼容端点）做任务级决策——选物体、选放置点、
给柔顺参数、失败重试。

参考项目：[momo-qxc/VLM_Grasp_Bio](https://github.com/momo-qxc/VLM_Grasp_Bio)
（MuJoCo + UR5e + VLM + SAM + GraspNet + RRT 的"语言-视觉-动作"闭环）。

---

## 1. 快速开始

```bash
cd /home/tc63/mujoco/vlm_admittance

# 全链路验收（6 项：core/sim/control/planning/agent + 端到端抓放）
MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m tests.run_acceptance

# 端到端抓放（不依赖 VLM，验证底层闭环）
MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m demo.scripted_pick_place --target box_red --container tray

# VLM 决策 demo（离线 MockVLM，无需 API key）
MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m demo.run_demo --task "把红色方块放进托盘，轻一点" --vlm mock --save-images

# 接真实多模态大模型（任意 OpenAI 兼容接口）
#   凭据支持别名：VLM_* / OPENAI_* / AI_API_URL+AI_API_KEY+AI_MODEL
AI_API_URL=https://ollama.com/v1 AI_API_KEY=xxx AI_MODEL=deepseek-v4.1-flash \
  MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m demo.run_demo --vlm api --task "把蓝色方块轻放进托盘"

# 接端点前先探针（连通性 / 图片输入 / function calling 三项能力）
AI_API_URL=... AI_API_KEY=... AI_MODEL=... PYTHONPATH=. /home/tc63/mujoco/.venv/bin/python tools/vlm_probe.py
```

**环境**：Python 3.14（`/home/tc63/mujoco/.venv`）、MuJoCo 3.12、numpy/scipy/pillow。
离屏渲染必须 `MUJOCO_GL=glfw`（本机 EGL 不可用）；依赖见 [requirements.txt](requirements.txt)，无额外第三方依赖。

---

### 1.1 交互式运行（一个 MuJoCo 窗口 + 一个命令台）

> ⚠️ **自动化 / 自测 / CI 一律必须加 `--headless`**。不带它会给用户的桌面**真的弹出一个 MuJoCo 窗口**，
> 跑几十次就是几十个（崩溃后还可能残留黑窗）。面板是**纯文件产出**（`outputs/panels/*.png` +
> `index.html`），headless 下同样出图，因此自动化**没有任何理由**开窗——下面第 1~3 行是给人用的，
> 第 4 行才是给脚本/测试用的。

**最省事的方式：用启动脚本 [run.sh](run.sh)**（自动设好 `MUJOCO_GL=glfw` / `DISPLAY` / `PYTHONPATH` / 解释器）：

```bash
./run.sh                    # 交互式：MuJoCo 窗口 + 命令台（真模型 + 纯视觉感知）
./run.sh headless           # 无窗口（自动化/自测一律用这个）
./run.sh daily              # 日常物品场景（水瓶/眼镜盒）
./run.sh task "把红色方块放到盘子里，绿色方块放到红方块上"   # 跑完一条就退出
./run.sh panels             # 浏览器打开画面 / 力矩曲线面板
./run.sh doctor             # 环境与配置体检
./run.sh help               # 全部子命令
```

不想用脚本时的等价原始命令：

```bash
cd /home/tc63/mujoco/vlm_admittance
MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m demo.interactive            # 【人工】窗口 + 终端命令台
MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m demo.interactive --vlm api  # 【人工】用真实大模型决策
MUJOCO_GL=glfw ... -m demo.interactive --realtime        # 【人工】按真实时间配速（默认全速，约 7× 实时）
MUJOCO_GL=glfw ... -m demo.interactive --headless        # 【脚本/自测】无窗口，纯终端；PNG 照常产出
```

**MuJoCo 窗口**：实时显示机械臂、物体、抓取与接触（passive viewer，主线程 100 Hz 同步）。
**命令台**（当前终端）：直接输入自然语言即为任务；`:命令` 单独调技能。

| 命令 | 作用 |
|---|---|
| 直接输入中文/英文 | 交给 VLM/AgentLoop 执行（例：`把红色方块轻放进托盘`） |
| `:task <文本>` | 显式指定任务 |
| `:pick <id>` / `:place <id>` | 抓取 / 放置（例 `:pick box_red`） |
| `:goto x y z` | 规划后直线移动到坐标 |
| `:push <id> dx dy dz [dist]` | 推动物体 |
| `:gripper open\|close\|<0~1>` | 夹爪控制 |
| `:compliance stiff\|soft\|z_only [k_scale]` | 切换导纳档位 |
| `:home` / `:reset` | 回初始位姿 / 重置场景 |
| `:objects` / `:state` | 列出物体 / 打印机器人+接触状态 |
| `:shot [camera]` | 存一张渲染 png |
| `:vlm mock\|api` | 热切换决策器 |
| `:help` / `:quit` | 帮助 / 退出（关闭窗口同样退出） |

实现要点（踩坑见 §7）：主线程 `viewer.sync()`，工作线程跑命令台与技能；
**`env.step` 与 `env.render` 都用同一把 RLock 保护**，因此跑任务时窗口依旧流畅。

## 2. 架构（双环：慢决策 + 快控制）

```
┌──────────── 慢环 0.5~2 Hz（事件触发）─ agent/ ────────────┐
│  render(overview/wrist/top) + SceneState                  │
│      → build_messages(图像 + 场景JSON + 历史)             │
│      → VLM(OpenAI 兼容, function calling) / MockVLM       │
│      → parse_skill_call → validate_skill_call             │
│      → SkillCall{skill, args, compliance}                 │
└───────────────────────────┬───────────────────────────────┘
                            ▼  skills/
┌──────────── 快环 500 Hz（每步 mj_step）───────────────────┐
│ skills: pick / place / push / home / move_to / set_compliance …
│   规划: planning.RRTConnect + CollisionChecker (私有 MjData)
│   轨迹: control.CartesianTrajectory (min-jerk + slerp)
│   导纳: M·ẍ + D·ẋ + K·(x−x_ref) = −f_ext  → Δx
│   执行: IK(DLS+零空间) → 关节目标 → MuJoCo 位置伺服
│   力源: sim.contact_force_on_ee()（仿真免传感器）
└───────────────────────────────────────────────────────────┘
```

## 3. 目录结构

| 路径 | 内容 | 负责人 |
|---|---|---|
| `core/` | 冻结契约：types（SceneState/SkillCall/ComplianceParams…）、config、Protocol | Lead |
| `sim/` | UR5e + Robotiq 2F85 + 桌面 + 5 物体 + 3 相机；`MuJoCoArmEnv` 实现 `SimEnv`；RGB/深度渲染；接触力 | sim-builder |
| `control/` | `Kinematics`(DLS IK/雅可比)、`CartesianTrajectory`、`AdmittanceController`、`MotionRunner` | ctrl-builder |
| `planning/` | `CollisionChecker`(私有 MjData)、`RRTConnect`+`shortcut`+`densify`、`JointPlanner` | plan-builder |
| `agent/` | 状态序列化/图像编码、TOOLS schema、`VLMClient`+`MockVLM`、`AgentLoop` | agent-builder |
| `skills/` | 技能库：把 SkillCall 变成确定性动作序列（规划+导纳+抓放+校验+重试） | Lead |
| `demo/` | `scripted_pick_place.py`（无 VLM 验收）、`run_demo.py`（VLM 决策） | Lead |
| `tests/` | `run_acceptance.py` 全链路验收 | Lead |
| `assets/` | Menagerie 的 `universal_robots_ur5e` + `robotiq_2f85`（BSD-3） | Lead |
| [INTERFACES.md](INTERFACES.md) | 模块契约（冻结版），并行开发与集成依据 | Lead |

## 4. 实测结果（2026-10-11，冻结版）

`tests/run_acceptance`：**6/6 通过**。

| 用例 | 关键指标 |
|---|---|
| core | 契约导入、资产存在 |
| sim.smoke | nq=49/nv=44/nu=7；3 相机 512×512 RGB+深度；静止力 0.000 N |
| control.smoke | 27 用例；IK 往返 0.38 mm / 4.9e-4 rad；直线终态 0.98 mm；导纳阶跃稳态 = −f/k、不发散 |
| planning.smoke | 5 用例；plan_pose 误差 0.0000 m；绕障 0.13 s、挡板接触 0 |
| agent.smoke | 36 用例；MockVLM 决策确定性；非法调用被拒 |
| e2e_scripted | `box_red` 抓起抬升 119 mm（峰值力 1.2 N）→ 放入托盘（离中心 1 mm） |

抓放覆盖：`box_red / box_blue / box_green` 均通过；`can_yellow`（30 mm 细高圆柱）见"已知限制"。

VLM 决策 demo（MockVLM，自然语言 → pick → place）实测：
```
$ ... -m demo.run_demo --task "把绿色方块放进托盘" --vlm mock
 01 OK  pick   已抓起 box_green（第 1 次尝试 深度+0mm 闭合0.30）
 02 OK  place  已放置 box_green（离容器中心 0 mm, dz 19 mm）
 03 OK  stop   任务结束
```

### 4.1 真实多模态大模型实测（已跑通）

端点：`AI_API_URL=https://ollama.com/v1` + `AI_MODEL=deepseek-v4.1-flash`
（凭据只从环境变量读，**不落盘**）。`tools/vlm_probe.py` 三项能力全通过：

```
[probe] 1) 文本 OK (0.7s) reply='收到'
[probe] 2) 图片 OK (0.8s) reply='红色'      ← 纯红测试图，视觉通道确实生效
[probe] 3) tools OK (0.8s) tool_calls=[{"function":{"name":"home","arguments":"{}"}}]
[probe] 结论: 文本=OK  图片=OK  function_calling=OK
```

真实模型驱动的三个任务全部成功（每步含 1 张 512×512 相机图 + 场景 JSON）：

| 指令 | 模型决策序列 | 结果 |
|---|---|---|
| 把红色方块放进托盘 | `pick(box_red)` → `place(tray)` → `stop` | 成功，11.9 s |
| 把绿色方块放进托盘 | `pick(box_green)` → `place(tray)` → `stop` | 成功，9.1 s |
| 把蓝色方块**轻**放进托盘，别压坏 | `set_compliance(soft)` → `pick(box_blue, compliance=soft)` → `place(tray, compliance=soft)` → `stop` | 成功，10.6 s |

第三条是关键证据：**"轻一点"这类自然语言被模型翻译成了导纳参数**（K/D/axes/force_limit），
经 `skills` 的 `_motion_compliance` 兜底后作用于实际控制——语言 → 柔顺度的通路是通的。

## 5. 与参考项目 VLM_Grasp_Bio 的对照

| 能力 | VLM_Grasp_Bio | 本项目 |
|---|---|---|
| 仿真/机器人 | MuJoCo + UR5e | MuJoCo + UR5e + Robotiq 2F85（Menagerie 模型，MjSpec.attach 组合） |
| 语言指令解析 | Qwen-VL-Max + DeepSeek 润色 | 任意 OpenAI 兼容多模态端点 + MockVLM 离线回退 |
| 视觉 | 多相机 + SAM 分割 + 深度 | 3 相机离屏渲染（可扩展 depth→点云）；**不做分割**，物体清单用仿真真值 |
| 抓取位姿 | GraspNet 生成 | 解析式：按物体几何 + 抓取宽度反算闭合量，多次 re-grasp 尝试 |
| 运动规划 | 改进 RRT + 在线避障 | OMPL 风格的 RRT-Connect（自研，零依赖）+ shortcut/densify |
| 柔顺控制 | —（本项目新增） | **导纳外环**（6 维，可配置 K/D/M/轴）+ 技能级柔顺档 |
| 工程形态 | 单仓 14 万行脚本 + GUI | 分层模块 + 冻结接口 + 每模块 smoke + 全链路验收 |

**复现的核心思想**：语言→视觉→动作闭环、按场景切换规划链路、物体清单交给大模型决策。
**主要简化**：不引入 GraspNet/SAM（重依赖），抓取姿态用几何解析；不引入 Qt GUI。

## 6. 已知限制（诚实清单）

1. **细高圆柱（can_yellow）抓取不稳**：2F85 指垫对 30 mm 圆柱的接触窗口窄，抓取成败对末端下沉量（UR5e 位置伺服无重力补偿，随位形漂移 ~1 cm）敏感。
2. **`--with-barrier` 场景下的抓放不可靠**：规划器只对"臂+夹爪"做碰撞检测，**不建模被夹持物体**，因此带着物体规划时可能让物体撞到挡板/托盘；降级为笛卡尔直线时更容易发生。绕障本身（空手）已由 planning.smoke 验证。
3. **接触力尺度大**：UR5e 位置执行器（kp=2000、forcerange ±150）刚性压入时末端力可达 1e3 N 量级，`force_limit`（8~20 N）会被轻易触发；本文所有接近段都用柔顺/慢速轨迹规避。
4. MockVLM 只做 pick→place 规则决策，不产生 push/move_to 等；真实 VLM 路径未联网实测（无 API key）。
5. 夹爪闭合量按物体宽度线性估算（标定：ratio=1→93 mm，ratio=0→8.4 mm），未做接触力闭环。

## 7. 工程踩坑集锦（都在代码 docstring 里）

1. UR5e 位置伺服**无重力补偿**：HOME 静沉 ~11 mm、任意关节角稳态静差 ~14 mm → 纯 IK→ctrl 到不了 <5 mm，必须加任务空间积分外环 + 速度前馈 + 收尾伺服。
2. 抓取下降段必须**只用 z 向柔顺**（xy/姿态刚性），否则接触反力使末端横移、指垫夹偏。
3. 放置高度**不能取容器中心**：容器有厚度，TCP 下到中心会让被夹物穿刺盘面（实测 18 N 冲击）。应按"容器上表面 + 物体半高 + 余量"计算。
4. 夹爪闭合量必须**按物体现算**：固定 0.35 对 40 mm 方块可行，对 30 mm 圆柱（指间距 38 mm > 30 mm）根本夹不住。
5. 接触失败后**先松爪再撤退**：闭合状态下抬升会把物体/托盘一起拖走，越挣越死。
6. `mj_contactForce` 返回的是施加在 geom2 上的力，`contact_force_on_ee` 需按 EE 是 geom1/geom2 取符号。
7. 规划路径有 50+ 路点，逐点等伺服会耗 15 s 仿真时间 → 用弧长 min-jerk 时间律 + 关节速度前馈流式执行。
8. `MjSpec.from_string` + attach 加 props 会给名字加 `/` 前缀导致 `mj_name2id` 返回 −1；改用 MjSpec API 直接加。
9. 本机离屏渲染需 `MUJOCO_GL=glfw` 且 `visual.global_.offwidth/offheight` 要 ≥ 渲染分辨率。
10. **交互式下 render 必须与 viewer.sync 同锁**：AgentLoop 取图时 `Renderer` 会读 `data`，
    与主线程 `viewer.sync()` 并发是数据竞争（实测 SIGABRT/134）。
11. **passive viewer 快速启停会段错误**：启动后 1 s 内退出时 GLFW 渲染线程尚未初始化完，
    立刻销毁上下文触发 SIGSEGV/139；`viewer.close()` 后留 0.4 s 收尾即可稳定。
12. 本机无 tkinter，因此"命令窗口"用终端实现（不需要额外 GUI 依赖）。

## 8. 下一步

- 用真实 VLM 端点跑通并固化 prompt（`VLM_API_KEY` 已支持）
- 把被夹持物体纳入规划模型（扩展 `CollisionChecker` 的机器人 geom 集合）
- 抓取改为接触力闭环（用 `force_limit` 做 guarded move，而不是位置到点）
- 可选：VLM 输出 `compliance` 参数驱动 K/D（`set_compliance` 已支持 stiff/soft/z_only + k_scale）
