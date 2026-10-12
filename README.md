<div align="center">

# VLM Admittance

**多模态大模型决策 + RRT-Connect 规划 + 六维导纳控制的 MuJoCo 机械臂抓取框架**

UR5e + Robotiq 2F85 · 3 相机离屏渲染 · 任意 OpenAI 兼容多模态端点 · **不需要训练 VLA**

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.12-0E7C86?style=flat-square)
![Robot](https://img.shields.io/badge/UR5e%20%2B%20Robotiq%202F85-Menagerie-4B5563?style=flat-square)
![VLM](https://img.shields.io/badge/VLM-OpenAI%20compatible-8A2BE2?style=flat-square)
![No training](https://img.shields.io/badge/VLA-不需要训练-2E7D32?style=flat-square)

[演示](#演示) · [快速开始](#快速开始) · [实测结果](#实测结果) · [几个值得记的坑](#几个值得记的坑) · [已知问题](#已知问题)

</div>

---

![场景总览：UR5e + Robotiq 2F85 + 5 个物体 + 托盘](docs/assets/hero.png)

## 这是什么

一句话：**底层是经典控制，上层是大模型**。

- 大模型（多相机图 + function calling）只做**任务级决策**：抓哪个、放到哪、多柔、失败了怎么改；
- 其余全是确定性控制：DLS IK → 笛卡尔 min-jerk 轨迹 → 导纳外环 → RRT-Connect 规划 → MuJoCo 位置伺服；
- 因此**没有数据集、没有训练**：换一个支持图片输入的 OpenAI 兼容端点即可；没 key 时用 `MockVLM`
  离线跑同一条代码路径。

参考项目：[momo-qxc/VLM_Grasp_Bio](https://github.com/momo-qxc/VLM_Grasp_Bio)
（MuJoCo + UR5e + VLM + SAM + GraspNet + RRT 的"语言-视觉-动作"闭环）。本项目的取舍见
[与参考项目的对照](#与参考项目-vlm_grasp_bio-的对照)。

## 演示

[![演示：左边终端跑 VLM 任务，右边 MuJoCo 实时跟随](docs/media/vla_arm_demo_preview.gif)](docs/media/vla_arm_demo.mp4)

左边是命令台（`./run.sh`：自然语言 → VLM 决策 → 技能调用），右边是 MuJoCo 窗口（实时跟随）。
完整视频 54 s / 1920×1080 / 24 fps：[docs/media/vla_arm_demo.mp4](docs/media/vla_arm_demo.mp4)
（36 MB，点上面的动图即可播放）。

视频里的一帧（能看到命令台的决策日志与 MuJoCo 侧的实际动作）：

![视频中的一帧：左命令台、右 MuJoCo](docs/assets/video_poster.png)

> [!NOTE]
> README 里放的是 6 s 动图预览。想在 README 内嵌带播放器的视频，可以在 GitHub 网页编辑器里把
> `docs/media/vla_arm_demo.mp4` 拖进去，GitHub 会生成 `user-attachments` 链接，替换上面的链接即可。

一次 `pick → place` 的四个瞬间（本项目实拍渲染，非示意图）：

![抓取流水线：初始 / top 视角 / 抓起 / 放入托盘](docs/assets/grasp_sequence.jpg)

## 架构：慢环决策 + 快环控制

```
┌──────────── 慢环 0.5~2 Hz（事件触发）─ agent/ ────────────┐
│  render(overview/wrist/top) + SceneState                  │
│      → build_messages（图像 + 场景 JSON + 历史）          │
│      → VLM（OpenAI 兼容, function calling）/ MockVLM      │
│      → parse_skill_call → validate_skill_call             │
│      → SkillCall{skill, args, compliance}                 │
└───────────────────────────┬───────────────────────────────┘
                            ▼  skills/
┌──────────── 快环 500 Hz（每步 mj_step）───────────────────┐
│ skills: pick / place / push / home / move_to / set_compliance …
│   规划: planning.RRTConnect + CollisionChecker（私有 MjData）
│   轨迹: control.CartesianTrajectory（min-jerk + slerp）
│   导纳: M·ẍ + D·ẋ + K·(x − x_ref) = −f_ext  → Δx
│   执行: IK（DLS + 零空间）→ 关节目标 → MuJoCo 位置伺服
│   力源: sim.contact_force_on_ee()（仿真免力传感器）
└───────────────────────────────────────────────────────────┘
```

大模型每一步只拿到两样东西：**场景 JSON**（物体 `id`/颜色/类别，**不含真值位姿**）+ **相机图**。

![发给 VLM 的画面：overview + top](docs/assets/vlm_frames.png)

语言不只是选物体——`把蓝色方块轻放进托盘` 这类指令会被模型翻译成导纳参数
（K/D/axes/force_limit）再作用于快环，语言→柔顺度的通路是通的。

交互式运行时还有实时面板（6 关节力矩 + 夹爪力 + 指间距）。面板是**纯文件产出**，
headless 下同样出图：

![力矩与夹爪曲线](docs/assets/force_curves.png)

## 快速开始

```bash
git clone <本仓库地址> && cd vlm_admittance
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # mujoco / numpy / scipy / pillow，无其它第三方依赖

export MUJOCO_GL=glfw                    # 离屏渲染；本机 EGL 不可用时必须
```

开发环境：Python 3.14、MuJoCo 3.12、numpy/scipy/pillow。Windows 上用 WSL + WSLg 跑通，
X socket 在 `/mnt/wslg/.X11-unix/X0`。

最省事的方式是启动脚本 [run.sh](run.sh)（自动设好 `MUJOCO_GL` / `DISPLAY` / `PYTHONPATH` / 解释器）：

```bash
./run.sh                    # 交互式：MuJoCo 窗口 + 命令台（真模型 + 纯视觉感知）
./run.sh headless           # 无窗口（自动化 / 自测 / CI 一律用这个）
./run.sh daily              # 日常物品场景（水瓶 / 眼镜盒）
./run.sh task "把红色方块放到盘子里，绿色方块放到红方块上"   # 跑完一条就退出
./run.sh panels             # 浏览器打开画面 / 力矩曲线面板
./run.sh doctor             # 环境与配置体检
./run.sh help               # 全部子命令
```

> [!WARNING]
> 自动化 / 自测 / CI 一律加 `--headless`。不带它会给桌面**真的弹出一个 MuJoCo 窗口**，
> 跑几十次就是几十个（崩溃后可能残留黑窗）；面板在 headless 下照样出图，脚本没有理由开窗。

等价原始命令：

```bash
MUJOCO_GL=glfw python -m demo.interactive              # 【人工】窗口 + 终端命令台
MUJOCO_GL=glfw python -m demo.interactive --vlm api    # 【人工】用真实大模型决策
MUJOCO_GL=glfw python -m demo.interactive --headless   # 【脚本】无窗口，PNG 照常产出

# 全链路验收（10 项）
MUJOCO_GL=glfw python -m tests.run_acceptance

# 端到端抓放（不依赖 VLM，验证底层闭环）
MUJOCO_GL=glfw python -m demo.scripted_pick_place --target box_red --container tray

# 接真实端点前先探针（连通性 / 图片输入 / function calling 三项能力）
AI_API_URL=... AI_API_KEY=... AI_MODEL=... python tools/vlm_probe.py
```

凭据只从环境变量读，**不落盘**；也支持 `VLM_*` / `OPENAI_*` 别名与本地 `config.local.json`
（已被 `.gitignore` 排除，模板见 [config.local.json.example](config.local.json.example)）。

### 命令台

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
| `:objects` / `:state` | 列出物体 / 打印机器人 + 接触状态 |
| `:shot [camera]` | 存一张渲染 png |
| `:vlm mock\|api` | 热切换决策器 |
| `:help` / `:quit` | 帮助 / 退出（关闭窗口同样退出） |

实现要点：主线程 `viewer.sync()`，工作线程跑命令台与技能；**`env.step` 与 `env.render` 共用同一把
RLock**，所以跑任务时窗口依旧流畅。

## 实测结果

`tests/run_acceptance`（2026-10-12 实跑）：**9/10 通过**，唯一失败是
`tests/test_target_overlay.py::test_injected_force_pure_compliance` 的**导纳让位方向**
（断言 `0.32 not less than 0.0`）——已知未修项，见[已知问题](#已知问题)。

| 验收项 | 关键指标 |
|---|---|
| core | 契约导入、资产存在 |
| sim.smoke | `nq/nv/nu` 与契约一致；3 相机 512×512 RGB + 深度；静止力 0.000 N |
| control.smoke | 44 用例；IK 往返 0.38 mm / 4.9e-4 rad；直线终态 0.98 mm；导纳阶跃稳态 = −f/k、不发散 |
| planning.smoke | 5 用例；`plan_pose` 误差 0.0000 m；绕障 0.12 s、挡板接触 0 |
| agent.smoke | 239 用例；MockVLM 决策确定性；非法调用被拒 |
| e2e_scripted | `box_red` 抓起抬升 **119.3 mm**（峰值力 1.46 N）→ 放入托盘（xy 偏差 0.5 mm，落座 `|F|` 0.00 N） |
| interactive_selftest | `demo.interactive --headless --self-test`：真实 `SkillExecutor` + 编排器 + 独立复核 |
| interactive_flows | 17 用例：意图路由、hold 不变量、强制恢复、pick 子目标复核端到端 |
| target_overlay | **FAIL**：导纳让位方向（既有缺陷） |
| place_semantics | 抓放语义 / 叠放 / 容器判据 |

其它单测：`sim.test_env` 14 / `tests.test_grasp_height` 9 / `tests.test_panels` 12 /
`tests.test_subgoal_loop` 6 / `tests.test_isolation_no_truth` 5（毒丸：物体真值 API 直接抛异常，
纯视觉闭环仍须跑通）等。

抓放覆盖：`box_red / box_blue / box_green` 均通过；`can_yellow`（细高圆柱）见已知问题。

### 真实多模态大模型实测

端点 `AI_API_URL=https://ollama.com/v1` + `AI_MODEL=deepseek-v4.1-flash`，
`tools/vlm_probe.py` 三项能力全通过：

```
[probe] 1) 文本 OK (0.7s) reply='收到'
[probe] 2) 图片 OK (0.8s) reply='红色'      ← 纯红测试图，视觉通道确实生效
[probe] 3) tools OK (0.8s) tool_calls=[{"function":{"name":"home","arguments":"{}"}}]
[probe] 结论: 文本=OK  图片=OK  function_calling=OK
```

真实模型驱动的三个任务全部成功（每步 1~2 张 512×512 相机图 + 场景 JSON）：

| 指令 | 模型决策序列 | 结果 |
|---|---|---|
| 把红色方块放进托盘 | `pick(box_red)` → `place(tray)` → `stop` | 成功，11.9 s |
| 把绿色方块放进托盘 | `pick(box_green)` → `place(tray)` → `stop` | 成功，9.1 s |
| 把蓝色方块**轻**放进托盘，别压坏 | `set_compliance(soft)` → `pick(box_blue, soft)` → `place(tray, soft)` → `stop` | 成功，10.6 s |

## 与参考项目 VLM_Grasp_Bio 的对照

| 能力 | VLM_Grasp_Bio | 本项目 |
|---|---|---|
| 仿真 / 机器人 | MuJoCo + UR5e | MuJoCo + UR5e + Robotiq 2F85（Menagerie 模型，`MjSpec.attach` 组合） |
| 语言指令解析 | Qwen-VL-Max + DeepSeek 润色 | 任意 OpenAI 兼容多模态端点 + `MockVLM` 离线回退 |
| 视觉 | 多相机 + SAM 分割 + 深度 | 3 相机离屏渲染（可扩展 depth→点云）；**不做分割**，物体清单用仿真真值 id/颜色 |
| 抓取位姿 | GraspNet 生成 | 解析式：按物体几何 + 抓取宽度反算闭合量，多次 re-grasp 尝试 |
| 运动规划 | 改进 RRT + 在线避障 | OMPL 风格 RRT-Connect（自研、零依赖）+ shortcut / densify |
| 柔顺控制 | — | **导纳外环**（6 维可配 K/D/M/轴）+ 技能级柔顺档 |
| 工程形态 | 单仓大脚本 + GUI | 分层模块 + 冻结接口 + 每模块 smoke + 全链路验收 |

**复现的核心思想**：语言→视觉→动作闭环、按场景切换规划链路、物体清单交给大模型决策。
**主要简化**：不引入 GraspNet / SAM（重依赖），抓取姿态用几何解析；不引入 Qt GUI。

## 几个值得记的坑

1. **"复核前让开视线"会把刚抓起的物体丢掉。** 每个子目标复核前都会归位；带载时规划失败会走
   `env.home()` 瞬移，旧实现还顺带张开夹爪 → 方块落回原位 → 复核判 `subject_still_present`，
   `g1:pick` 三次尝试全废，而技能层却报"已抓起"。现在 pick 子目标 / 仍持物时**不归位**，
   带载禁止瞬移兜底（返回 `home_blocked`）。
2. **抓取成功不能用代理值判。** 夹爪开合比例按标定换算出的"指间距"比真实指垫间距大 7~10 mm，
   "被挡住"与"没合到位"分不开；更严重的是抬升 `move_to_pose` 的返回值曾被完全忽略
   （实测抬升 46 N 中止、物体只动 1 mm 仍报成功）。现在要求：抬升 ok + 夹爪接触的物体确实离地。
3. **单相机验证两个方向都会错。** 腕部相机只要停在物体上方就能看到上万像素（近似恒真的假阳性）；
   top 前后对比又对"举高但仍可见"漏判（蓝块真抓起后 top 面积 818→827）。现在主判据是 top
   `check_gone`，配物理随动证据兜底，再加"夹的必须是目标物 / 腕部颜色"的反向闸。
4. **`size=(0.02, 0.02, 0.02)` 是半尺寸**，即 40 mm 方块，不是 20 mm。按 20 mm 算闭合量会
   直接把方块捏住不放、抬升超力限。
5. UR5e 位置伺服**无重力补偿**：HOME 静沉 ~11 mm、任意关节角静差 ~14 mm，纯 IK→ctrl 到不了
   <5 mm，必须加任务空间积分外环 + 速度前馈 + 收尾伺服。
6. 抓取下降段**只用 z 向柔顺**（xy/姿态刚性），否则接触反力使末端横移、指垫夹偏；接触失败后
   **先松爪再撤退**，闭合状态抬升会把物体/托盘一起拖走。
7. `mj_contactForce` 返回施加在 geom2 上的力，`contact_force_on_ee` 需按 EE 是 geom1/geom2 取符号。
8. 本机离屏渲染需 `MUJOCO_GL=glfw`，且 `visual.global_.offwidth/offheight` 要不小于渲染分辨率。
9. **交互式下 render 必须与 `viewer.sync` 同锁**：AgentLoop 取图时 `Renderer` 会读 `data`，
   与主线程 `viewer.sync()` 并发是数据竞争（实测 SIGABRT/134）；passive viewer 启动后 1 s 内退出
   也可能 SIGSEGV/139，`close()` 后留 0.4 s 收尾即可稳定。
10. `MjSpec.from_string` + attach 加物体给名字加 `/` 前缀，`mj_name2id` 直接返回 −1；改用
    `MjSpec` API 直接加。

## 目录结构

```
vlm_admittance/
├── run.sh                  # 启动脚本（demo / headless / task / panels / doctor）
├── core/                   # 冻结契约：types（SceneState/SkillCall/ComplianceParams）、config、Protocol
├── sim/                    # UR5e + 2F85 + 桌面 + 5 物体 + 3 相机；MuJoCoArmEnv、RGB/深度、接触力
├── control/                # Kinematics(DLS IK)、CartesianTrajectory、AdmittanceController、MotionRunner
├── planning/               # CollisionChecker（私有 MjData）、RRTConnect+shortcut+densify、JointPlanner
├── agent/                  # 状态序列化/图像编码、TOOLS schema、VLMClient+MockVLM、AgentLoop、子目标编排
├── skills/                 # 技能库：SkillCall → 确定性动作序列（规划 + 导纳 + 抓放 + 校验 + 重试）
├── demo/                   # interactive.py（窗口+命令台）、run_demo.py、scripted_pick_place.py、panels.py
├── tests/                  # run_acceptance.py 全链路验收 + 各专项回归
├── tools/                  # vlm_probe.py（端点三项能力）、vision_accuracy.py、live_vla_baseline.py
├── assets/                 # Menagerie 的 universal_robots_ur5e + robotiq_2f85（BSD-3-Clause）
├── docs/                   # 本 README 的图片与演示视频
├── outputs/                # 运行产物（面板 / 验收日志 / 诊断），已 gitignore
└── INTERFACES.md           # 模块契约（冻结版），并行开发与集成依据
```

## 文档

- [INTERFACES.md](INTERFACES.md)：模块契约（冻结版），并行开发与集成依据
- [README.md](README.md)：本文件（架构 / 快速开始 / 实测 / 坑）
- 运行产物与诊断脚本在 `outputs/`（已 gitignore）：面板 PNG、验收日志、抓取失败诊断与修复报告

## 已知问题

1. **`target_overlay` 的导纳让位方向用例失败**（验收 9/10 的唯一失败）：外力 +z 时参考位移
   应为负，实测 +0.32 —— 方向/符号未修。
2. **细高圆柱（`can_yellow`）抓取不稳**：2F85 指垫对 30 mm 圆柱的接触窗口窄，成败对末端下沉量
   （位置伺服无重力补偿，随位形漂移 ~1 cm）敏感。
3. **`--with-barrier` 场景抓放不可靠**：规划器只对"臂 + 夹爪"做碰撞检测，**不建模被夹持物体**。
4. **接触力尺度大**：UR5e 位置执行器刚性压入时末端力可达 1e3 N 量级，`force_limit`（8~20 N）
   容易被触发；接近段都靠柔顺/慢速轨迹规避。
5. **`env.home()` 是裸瞬移**：低层直接调用它不做"是否持物"判断，持物时会掉件；流水线内已全部
   改为带载禁瞬移。
6. **感知 z 不稳定**：overview 第二像素在某些实现里无形状守门，遮挡下颜色精修会漂（实测 Δ=33 px），
   且 `residual < 15 mm` 不是有效的可信度判据（接受带对应 ±100 mm 深度误差）。
7. 夹爪闭合量按物体宽度线性估算（ratio=1→93 mm，ratio=0→8.4 mm），**未做接触力闭环**。
8. 演示视频 36 MB 直接入库，克隆会慢；若嫌大，可改成 Release 附件或 user-attachments 链接。

## 致谢

- [momo-qxc/VLM_Grasp_Bio](https://github.com/momo-qxc/VLM_Grasp_Bio)：本项目的参考实现
- [MuJoCo](https://github.com/google-deepmind/mujoco) 与
  [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)：物理引擎与 UR5e / Robotiq 2F85 模型
- [Robotiq 2F85](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85) 的夹爪标定与执行器配置

## 许可

本仓库代码目前未附带许可证文件（如需开源建议补 MIT）；`assets/` 下的 UR5e 与 Robotiq 2F85
模型来自 MuJoCo Menagerie，遵循 **BSD-3-Clause**（见各自目录下的 `LICENSE`）。
