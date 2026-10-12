"""交互式 demo：一个 MuJoCo 实时窗口 + 一个命令台。

界面
----
* **MuJoCo 窗口**：`mujoco.viewer.launch_passive`，主线程按 ~100 Hz 同步，全程实时显示机械臂/物体/接触。
* **命令窗口**：当前终端。直接输入自然语言即为任务；也可以用 `:命令` 单独调技能。

用法
----
    cd /home/tc63/mujoco/vlm_admittance
    MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m demo.interactive
    # 可选: --vlm auto|mock|api（默认 auto）  --perception auto|vlm|oracle（默认 auto）
    #       --scene default|daily（默认 default）  --planner subgoal|flat（默认 subgoal）
    #       --panels none|files|all（默认 all；纯文件产出，headless 也照写）
    #       --with-barrier  --realtime  --headless  --self-test

    ⚠️ 自动化 / 自测 / CI 一律必须加 --headless：不加会真的给用户弹出一个 MuJoCo 窗口
       （用户已投诉黑窗）。面板是纯文件产出，headless 下同样出图，例如：
         MUJOCO_GL=glfw ... -m demo.interactive --headless --panels all --vlm mock --perception vlm
       上面那条不带 --headless 的命令只用于「人在终端交互使用」。

命令一览（`:help`）
--------------------
    自然语言              例: 把红色方块放进托盘，轻一点
    :task <文本>          显式交给 VLM/AgentLoop
    :pick <id>            抓取      :place <id>   放置
    :goto x y z           直线移动到坐标
    :push <id> dx dy dz [dist]
    :gripper open|close|<0~1>
    :compliance stiff|soft|z_only [k_scale]
    :home                 回初始位姿    :reset  重置场景
    :objects              列出物体      :state  打印状态
    :shot [camera]        存一张 png    :vlm auto|mock|api  切换决策器
    :target on|off|axes|sphere          显示当前目标点（默认 on）
    :perception auto|vlm|oracle         感知模式（默认 auto）
    :scene default|daily                切换场景（默认 default）
    :plan                打印子目标表    :verify  子目标复核结论与依据
    :panels none|files|all              面板开关（默认 all）
    :panels shot                        立即刷新曲线图
    :panels open                        用浏览器打开面板页（每 500ms 自动刷新）
    画面/曲线去哪儿看：浏览器打开 outputs/panels/index.html（自动刷新），
    或直接看 outputs/panels/llm_frames.png（发给 LLM 的那两帧）与 curves.png（力矩/夹爪力/指间距）
    :history              打印会话历史    :clear-history  清空会话历史
    :recover              应急恢复（硬脱离高接触力卡死）
    :help / :quit

会话历史
--------
* 自然语言任务共用一段 history：上一条命令放下的物体，下一条命令可以直接指代
  （例：「把蓝色方块放回桌子上」→「放到桌子上不是盘子里」里的“它”= 上一条命令的物体）。
* :reset 只重置场景，**不清会话历史**（场景重置不等于对话重置）；要清历史用 :clear-history。

决策器
------
* 默认 --vlm auto：有 VLM_API_KEY / OPENAI_API_KEY / AI_API_KEY 时用真模型（OpenAI 兼容端点 +
  function calling + 相机图像），没有 key 才退化为离线规则 MockVLM。
* 启动日志打印 决策器类型 / model / base_url / 是否带图像（**不打印 key**）；每个任务打印
  「本步由 <模型名|MockVLM> 决策」，并配合 AgentLoop 的 [step] reasoning 摘要。
* 切真模型示例: AI_API_URL=<base_url> AI_API_KEY=<key> AI_MODEL=<model> python -m demo.interactive --vlm api
* 凭据优先级: 环境变量 > vlm_admittance/config.local.json > 内置默认。若最终回退 MockVLM，启动会打印
  醒目多行警告（「当前没有大模型…」），且 banner 与「本步由 … 决策」都带 [无 LLM] 标记。
* :vlm status 查看 决策器来源（env / config.local.json / 无）、base_url、model 与掩码 key（绝不打印明文 key）。

感知模式（INTERFACES.md §4.4，用 :perception 切换）
----------------------------------------------
* oracle（默认给 MockVLM）：状态里带物体真实位姿，视觉/关系判断由真值兜底。
* vlm（默认给真模型）：**不给真值位姿**（include_poses=False）；模型输出目标像素 (u,v)，
  由 sim.vision.pixel_to_table 反投影成世界坐标再交给技能；叠加层画出该世界点（label 带 VLM）。
* vlm 模式下 :objects 不打印物体真值坐标；sim.vision / AgentLoop 契约未就绪时会提示一条并退回 oracle 行为。
* 像素链路（§5.6）：VLM 像素 -> sim.perception.refine_pixel 颜色精修（失败回退粗像素）-> sim.vision.pixel_to_table；
  日志形如「[感知] VLM 像素 (u,v)[cam] → 颜色精修 (u',v') Δ=…px (≈…mm) → 世界 (x,y,z)」
  （top 相机 1px≈1.489mm，overview 为斜视只给 px）。

子目标规划（§5.6，--planner subgoal 为默认）
--------------------------------------------
* subgoal：先 taskplan.plan_subgoals 拆子目标，再逐子目标跑 AgentLoop 并**独立复核**（几何 / VLM 依据），
  每个子目标最多 SUBGOAL_MAX_ATTEMPTS(3) 次尝试；汇总打印
  「g1 完成（几何：…）」/「g2 未完成（3 次尝试）」，子目标没全完成一律判失败。
* flat：单任务直接跑（旧行为）。
* agent.taskplan 未就绪时自动退回 flat 并提示一条；:plan / :verify 也会给出清晰提示，不崩。

原生面板（--panels / :panels，task-64）
------------------------------------
* 面板跑在**独立子进程**（multiprocessing spawn）里：父进程只发帧/遥测，GL 上下文与主进程隔离，
  避免「额外 viewer 窗口 + 离线渲染」在本机 GLFW 下 segfault；子进程挂掉只降级为无面板，不影响仿真。
* cam：两个原生窗口之一显示**实际发给大模型的那两帧**（左 overview | 右 top，按发送时 JPEG q85 解码，
  没有 JPEG 时退回原始 RGB）；控制台每步打一行 caption（相机/分辨率/格式/字节/步号）。
* chart：另一个窗口画 8 条泳道曲线（6 关节力矩 + 夹爪执行器力 & 末端接触力 + 实测指间距 mm），
  10s 滚动窗、20~30Hz 刷新；数据由物理线程写环形缓冲，面板只在主线程 sync（面板线程绝不 env.render）。
* :panels 可运行时切换；--headless 完全不开窗（行为与原先一致）；退出时显式关面板+主窗口并有 os._exit 兜底。

场景（--scene / :scene）
------------------------
* default：方块 + 托盘；daily：日常物品（水瓶 / 眼镜盒，§5.5）。
* sim.scene.DAILY_OBJECTS 未就绪时 daily 回退 default 并提示；:scene 会重建 env 并清空会话历史
  （GUI 下需重启窗口刷新 viewer.model）。
* vlm 模式随附图像为 overview + top（top 是像素瞄准主力，像素->世界约 0.65mm/px）；启动时打印
  「随附图像相机: [...]」，与消息里的「随附图像(按此顺序)」一致。--cameras 可覆盖（oracle 仍只发 overview）。

任务收尾
--------
* 每个任务结束打印一行「结果: 成功 / 成功（重试后）/ 中止（需要你补充信息）/ 失败」，
  并如实给出三段计数「N 步: k OK / m 已纠正 / j FAIL」；被拒后改正成功（invalid_call 等）记「已纠正」，
  目标技能最终没成功、ask_user、不可重试错误仍判失败/中止，不洗白。
* 若夹爪里仍有物体：**异常结束（中止/失败）**才调用 skills.return_held_object() 放回，并打印
  「任务中止：已把 <id> 放回 <位置>」+ metrics 摘要；**成功结束**保留夹持并打印一行提示
  （「只抓不放」类任务如「夹住蓝色方块」不应被自动放回）。接口未就绪/异常时明确警告
  「请手动处理」，不会静默。

目标点叠加（:target 控制，需 sim/overlay；缺接口时只提示一条并照常运行）
    青色球=当前下发目标 cmd（球+三轴始终显示，不受接触影响，才是本需求主角）
    橙色球=轨迹参考 ref    紫红小球=纯导纳输出 adm
    橙色/紫红让位标记的唯一触发条件：导纳外力 force_norm > 0.5N
      （无接触 force_norm 恒为 0.0，0.5N 只是数值余量；自由空间不显示）
    橙-紫距离=柔顺让位量   紫-青距离=重力积分静差补偿量
    姿态让位 > ~2.9° 时紫红点额外画三轴（adm_quat）；缺 force_norm 键则不画让位标记

线程模型
--------
* 主线程：viewer.sync()（MuJoCo 要求同线程）
* 工作线程：读命令 + 执行（技能会连续 env.step 几百步）
* env.data 的读写由一把 RLock 保护（step 与 sync 互斥），因此运行任务时窗口依旧流畅。
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import threading
from collections import deque
import time
from pathlib import Path

import numpy as np

from agent.loop import AgentLoop
from agent.vlm_client import MockVLM, make_vlm
from core.config import (AGENT_MAX_STEPS, DEFAULT_COMPLIANCE, GRIPPER_GAP_CLOSED_MM,
                        GRIPPER_GAP_OPEN_MM, LOCAL_CONFIG_SOURCE, OUTPUT_DIR, TABLE_HEIGHT,
                        VLM_API_KEY, VLM_BASE_URL, VLM_MODEL)
from core.types import SkillCall
from planning.collision import CollisionChecker
from planning.planner import JointPlanner
from control.admittance import AdmittanceController
from control.kinematics import Kinematics
from control.motion import MotionRunner
from sim.env import MuJoCoArmEnv
from skills import SkillExecutor, approach_quat

# 目标点叠加渲染（sim/overlay.py 由并行任务提供）：缺失时整个功能降级为 no-op，
# 保证本文件在接口未就绪时也能独立运行。
try:
    from sim.overlay import clear as _overlay_clear
    from sim.overlay import draw_target as _overlay_draw_target
except ImportError:  # pragma: no cover - 接口未就绪
    _overlay_clear = None
    _overlay_draw_target = None

# 姿态让位量需要四元数工具（control/trajectory.py）；缺失时只跳过“旋转让位”显示。
try:
    from control.trajectory import quat_conj as _quat_conj
    from control.trajectory import quat_mul as _quat_mul
    from control.trajectory import rotvec_from_quat as _rotvec_from_quat
except ImportError:  # pragma: no cover - 接口未就绪
    _quat_mul = _quat_conj = _rotvec_from_quat = None

# 像素 -> 桌面反投影（INTERFACES.md §4.1，sim/vision.py 由 sim-builder 提供）
try:
    from sim.vision import pixel_to_table as _pixel_to_table
    from sim.vision import project_point as _project_point
except ImportError:  # pragma: no cover - T47 未就绪
    _pixel_to_table = _project_point = None

# §5.1 颜色精修（vision-builder）：缺失时像素链路退化为「粗像素直投」，不崩。
try:
    from sim.perception import color_blob as _color_blob
    from sim.perception import refine_pixel as _refine_pixel
except ImportError:  # pragma: no cover - §5.1 未就绪
    _color_blob = _refine_pixel = None

# §5.10 双目三角定位（vision-builder）：缺失时退化为单目平面反投影（source=single_view）。
try:
    from sim.perception import object_point as _object_point
    from sim.perception import triangulate as _triangulate
except ImportError:  # pragma: no cover - §5.10 未就绪
    _object_point = _triangulate = None

# task-64 面板子进程的重启策略（缺失 demo.panels 时给同样的默认值）
try:
    from demo.panels import MAX_RESTARTS as _MAX_PANEL_RESTARTS
    from demo.panels import RESTART_BACKOFF_S as _PANEL_RESTART_BACKOFF_S
except Exception:  # noqa: BLE001
    _MAX_PANEL_RESTARTS, _PANEL_RESTART_BACKOFF_S = 3, 2.0

# §5.2 子目标规划 / 独立复核（agent-builder）：缺失时 --planner subgoal 退回 flat 并提示。
try:
    from agent.taskplan import SUBGOAL_MAX_ATTEMPTS as _SUBGOAL_MAX_ATTEMPTS
    from agent.taskplan import SubGoalOrchestrator as _SubGoalOrchestrator
    from agent.taskplan import SubGoalVerifier as _SubGoalVerifier
    from agent.taskplan import plan_subgoals as _plan_subgoals
except ImportError:  # pragma: no cover - §5.2 未就绪
    _SUBGOAL_MAX_ATTEMPTS = 3
    _SubGoalOrchestrator = _SubGoalVerifier = _plan_subgoals = None

# §5.5 daily 场景对象（scene-builder）：缺失时 --scene daily 回退 default 并提示。
try:
    from sim.scene import DAILY_OBJECTS as _DAILY_OBJECTS
except ImportError:  # pragma: no cover - §5.5 未就绪
    _DAILY_OBJECTS = None

#: 任务文本里的颜色词 -> sim.perception 的颜色类（只用于 refine_pixel 精修，不读物体真值）
#: 面板子进程启动宽限（秒）：期间主循环照常跑（非阻塞轮询），超时即降级
_PANEL_START_GRACE_S = 6.0

#: bbox 高/宽 ≥ 此值才认为目标“竖长”（瓶/筒），才用 bbox 顶边中点当第二像素；
#: 矮方块俯视下近方形（比值小），路径保持不变。
_TALL_BBOX_RATIO = 1.4
#: 最近 locate 记录保留条数
_RECENT_LOCATE_MAX = 4

_COLOR_WORDS = (
    ("红色", "red"), ("红", "red"), ("red", "red"),
    ("蓝色", "blue"), ("蓝", "blue"), ("blue", "blue"),
    ("绿色", "green"), ("绿", "green"), ("green", "green"),
    ("黄色", "yellow"), ("黄", "yellow"), ("yellow", "yellow"),
    ("青色", "cyan"), ("cyan", "cyan"), ("水瓶", "cyan"), ("bottle", "cyan"),
    ("黑色", "black"), ("黑", "black"), ("black", "black"),
    ("眼镜盒", "black"), ("眼镜", "black"), ("case", "black"),
    ("白色", "white"), ("白", "white"), ("white", "white"),
    ("灰色", "gray"), ("灰", "gray"), ("gray", "gray"), ("grey", "gray"),
    ("橙色", "orange"), ("橙", "orange"), ("orange", "orange"),
)

#: 纯导纳输出点 adm_pos 的标记颜色（紫红色，与青色 target / 橙色 ref 区分）
_ADM_MARKER_RGBA = (0.85, 0.30, 0.95, 1.0)
#: 让位类标记（橙色 ref 球+连线、紫红 adm 球）的唯一触发条件：导纳外力 > 0.5N
#: （无接触 force_norm 恒为 0.0，0.5N 只是数值余量；2N 会漏掉 0.66~1.2N 的真实轻接触）
_YIELD_FORCE_N = 0.5
#: 姿态让位可见阈值（rad，约 2.9°）：超过则给 adm 标记补画姿态三轴
_ROT_YIELD_VISIBLE_RAD = 0.05

#: 凭据环境变量名（与 core/config.py 的映射一致）；只用于判断来源，绝不打印值
_ENV_KEY_NAMES = ("VLM_API_KEY", "OPENAI_API_KEY", "AI_API_KEY")
_ENV_BASE_NAMES = ("VLM_BASE_URL", "OPENAI_BASE_URL", "AI_API_URL")
_ENV_MODEL_NAMES = ("VLM_MODEL", "AI_MODEL")
#: 「没有大模型」醒目警告的分隔线
_WARN_LINE = "=" * 68

BANNER = r"""
 ___   ___  __  __   __   ___  ___  _   _  ___
| | | | __||  \/  | / _| /   \|_ _|| | | ||   \
| |_| | _| | |\/| || |_ | | | || | | |_| || | |
 \___/|___||_|  |_| \__||_|_|_|___| \___/ |___|
  VLM 决策 + 路径规划 + 导纳控制 · 交互式仿真台
"""


class InteractiveApp:
    def __init__(self, args) -> None:
        self.args = args
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.vlm_mode = args.vlm         # mock|api（选择结果），auto 会解析成其中之一
        self.vlm_request = args.vlm      # 用户请求：auto|mock|api（仅用于日志）
        self.target_mode = "on"          # 目标点叠加：on|off|axes|sphere
        self._overlay_warned = False     # 接口/渲染异常只提示一次
        self._force_key_warned = False   # legacy 缺 force_norm 只提示一次
        self.history: list[dict] = []    # 跨命令会话历史（每步的 assistant/user 消息）
        self._history_warned = False     # AgentLoop 不支持 history 参数时只提示一次
        # 感知模式（INTERFACES.md §4.4）：oracle=给真值位姿；vlm=只给图像，靠像素瞄准
        self.perception_request = getattr(args, "perception", "auto")   # auto|vlm|oracle
        self.perception_mode = self.perception_request                  # 解析结果 vlm|oracle
        self._perception_warned = False  # 契约接口未就绪只提示一次
        self._last_vlm_point = None      # 模型像素反投影出的世界点（叠加用）
        self._last_pixel = None          # (u, v, camera) 供 label
        self._task_cameras = None        # 本任务实际发给模型的相机（顺序 = 发送顺序）
        # §5.6：场景 / 子目标规划开关，以及最近一次规划/复核结果
        self.scene_mode = getattr(args, "scene", "default")      # default|daily
        self.planner_mode = getattr(args, "planner", "subgoal")   # subgoal|flat
        self._last_plan: list = []
        self._last_verify: list = []
        self._subgoal_results: list = []
        self._subgoal_loops: list = []   # 子目标闭环里创建的 AgentLoop（回捞 SkillResult）
        self._last_task = ""
        self._last_locate_color = None   # 上一次 locate 标定出的 color（自动补 overview 用）
        self._recent_locate: list = []   # 最近几次 locate 记录（color/pixel/bbox_{top,overview}）
        self._auto_ov_reason = ""        # 自动补第二视角失败的原因（日志用）
        self._taskplan_warned = False    # taskplan 未就绪只提示一次
        self._refine_warned = False      # sim.perception 未就绪只提示一次
        self._verifier = None            # SubGoalVerifier 缓存（:scene 重建时清空）
        # task-64 面板 = 纯文件产出（llm_frames.png / curves.png），不开窗、不起子进程；
        # headless 不例外，照样写文件（「无窗口也要有图」）。
        self.panel_request = getattr(args, "panels", None) or "all"
        self.panel_mode = self.panel_request
        self._panels_files = None        # demo.panels.PanelFiles（纯文件，无窗口/子进程）
        self._panel_last = 0.0
        self._gl_warmed = False               # 离屏渲染器是否已在主线程串行预热
        self._gl_ready = threading.Event()    # worker 线程离屏渲染器就绪信号
        self._telemetry = None
        self._panel_frames: dict = {}
        self._panel_meta: dict = {}
        self._llm_hook_installed = False      # agent.loop.build_messages 只包一次
        self._panel_capture = False
        self._panel_step = 0
        self.planner_request = getattr(args, "planner", None)   # None = 未显式指定
        self.planner_mode = "flat"       # 占位；_set_perception() 里按模式解析

        self._setup_runtime()
        self._set_vlm(args.vlm)
        self._set_perception(getattr(args, "perception", "auto"))
        if self.panel_request != "none":      # 纯文件产出：headless 也照写
            self._open_panels()

    # ------------------------------------------------------------------ #
    # 场景与运行时装配（§5.5/§5.6）：--scene / :scene 会重建 env 与依赖
    # ------------------------------------------------------------------ #
    def _make_env(self):
        """按 self.scene_mode 构造环境；daily 未就绪时明确提示并回退 default。"""
        objects = None
        if self.scene_mode == "daily":
            if _DAILY_OBJECTS is None:
                self._log("[app] daily 场景尚未就绪（sim.scene.DAILY_OBJECTS 缺失），回退 default")
                self.scene_mode = "default"
            else:
                objects = list(_DAILY_OBJECTS)
        return MuJoCoArmEnv(objects=objects, with_barrier=self.args.with_barrier,
                            seed=self.args.seed)

    def _setup_runtime(self) -> None:
        """构造/重建 env 及其上的 kin/runner/planner/skills（:scene 切换时复用）。"""
        self.env = self._make_env()
        self._patch_step()
        self.kin = Kinematics(self.env)
        self.runner = MotionRunner(self.env, self.kin,
                                   AdmittanceController(DEFAULT_COMPLIANCE, self.env.dt))
        self.planner = JointPlanner(self.env, self.kin, CollisionChecker(self.env))
        self.skills = SkillExecutor(self.env, self.kin, self.runner, self.planner, log=self._log)
        self._verifier = None            # 定位/复核后端与新 env 绑定，重建时作废

    # ------------------------------------------------------------------ #
    # 统一持锁代理：env 的所有公开 data 访问都走同一把 RLock。
    # 为什么必须这么做：主线程 viewer.sync() 内部 mj_copyDataVisual，而 worker 线程的
    # env.render（离屏）会 mjv_updateScene/mj_forward；任何一条没持锁的 data 访问都会撞出
    # 「mj_copyDataVisual: attempting to copy mjData while stack is in use」。
    # 锁必须可重入（RLock）：MotionRunner 内部会嵌套调用 env.step/render。
    # 只加锁，不改语义/返回值/签名（参数原样透传）。
    # ------------------------------------------------------------------ #
    _LOCKED_ENV_METHODS = ("step", "render", "render_depth", "get_state", "list_objects",
                           "object_pose", "site_pose", "body_pose", "contact_force_on_ee",
                           "get_gripper", "set_gripper", "set_arm_joint_targets",
                           "home", "reset")

    def _patch_step(self) -> None:
        orig_step = self.env.step
        lock, realtime, dt = self.lock, bool(self.args.realtime), self.env.dt

        def locked_step(n: int = 1):
            if not realtime:
                with lock:
                    state = orig_step(n)
                    self._sample_telemetry()      # task-64：每物理步采一次（没开面板时零开销）
                    return state
            state = None
            for _ in range(max(1, int(n))):
                t0 = time.perf_counter()
                with lock:
                    state = orig_step(1)
                    self._sample_telemetry()
                left = dt - (time.perf_counter() - t0)
                if left > 0:
                    time.sleep(left)
            return state

        self.env.step = locked_step  # type: ignore[method-assign]

        # 其余公开方法统一持锁（render 同时记下「发给 LLM 的那一帧」）
        for name in self._LOCKED_ENV_METHODS:
            if name == "step":
                continue
            orig = getattr(self.env, name, None)
            if not callable(orig):
                continue
            def wrapped(*a, _orig=orig, **kw):
                with lock:
                    return _orig(*a, **kw)        # 面板不再在这里抓帧：真源是 build_messages 里的 data URL

            setattr(self.env, name, wrapped)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # task-64 原生面板：相机帧（发给 LLM 的那两帧）+ 力矩/夹爪力/指间距曲线
    # ------------------------------------------------------------------ #
    def _panel_caption(self, step: int) -> str:
        """控制台 caption：本步发给 LLM 的画面（相机/分辨率/格式/字节/步号）。"""
        parts, total, fmt = [], 0, "raw rgb"
        for cam in ("overview", "top"):
            meta = self._panel_meta.get(cam)
            if not meta:
                continue
            w, h, nbytes, mfmt = meta
            parts.append("%s %dx%d" % (cam, w, h))
            total += int(nbytes)
            if "jpeg" in str(mfmt):
                fmt = "jpeg q85"
        return ("[panel] 发给 LLM 的画面: %s | %s | 合计 %.1f KB | step %d"
                % (" + ".join(parts) if parts else "(尚无帧)", fmt, total / 1024.0, step))

    def _sample_telemetry(self) -> None:
        """物理线程（持锁）每步采一次遥测；没开面板时零开销，绝不改 control/skills。"""
        tel = self._telemetry
        if tel is None:
            return
        try:
            af = np.asarray(self.env.data.actuator_force, dtype=float).reshape(-1)
            tau6 = af[:6] if af.size >= 6 else np.zeros(6, dtype=float)
            grip_act = float(af[6]) if af.size > 6 else 0.0
            fee = float(np.linalg.norm(np.asarray(self.env.contact_force_on_ee()[0], dtype=float)))
            ratio = float(np.clip(float(self.env.get_gripper()), 0.0, 1.0))
            gap_mm = GRIPPER_GAP_CLOSED_MM + ratio * (GRIPPER_GAP_OPEN_MM - GRIPPER_GAP_CLOSED_MM)
            tel.append((float(self.env.data.time), *[float(v) for v in tau6],
                        grip_act, fee, gap_mm))
            sink = self._panels_files
            if sink is not None and (time.time() - sink.curves_at) >= sink.curve_interval:
                sink.write_curves(self._telemetry_rows())   # <=1Hz 节流，CPU 增量很小
        except Exception:  # noqa: BLE001 - 遥测异常绝不能影响物理
            pass

    def _install_llm_frame_hook(self) -> None:
        """把 agent.loop.build_messages 包一层：直接取消息里那份 data URL 的字节当唯一真源。

        为什么不在 env.render 路径抓帧：一个请求里同一相机可能被渲染/编码多次，抓到的未必
        是真正发出去的那一份（verifier: overview panel_bytes != sent_bytes）。这里读的就是
        即将随请求发出去的 base64，字节级同一份，md5 自证。
        """
        if self._llm_hook_installed:
            return
        try:
            import agent.loop as _loop
            orig = getattr(_loop, "build_messages", None)
            if orig is None:
                return

            def wrapped(*a, **kw):
                msgs = orig(*a, **kw)
                try:
                    self._on_llm_messages(msgs)
                except Exception:  # noqa: BLE001 - 面板出错绝不影响任务
                    pass
                return msgs

            _loop.build_messages = wrapped
            self._llm_hook_installed = True
        except Exception:  # noqa: BLE001
            pass

    def _on_llm_messages(self, messages) -> None:
        """从构造好的消息里取出图片 data URL 的字节（= 真正发给 LLM 的那份）。"""
        if self._panels_files is None or not messages:
            return
        import base64
        import hashlib
        urls = []
        for m in messages:
            c = m.get("content") if isinstance(m, dict) else None
            if not isinstance(c, list):
                continue
            for part in c:
                if not isinstance(part, dict):
                    continue
                url = (part.get("image_url") or {}).get("url") if isinstance(part.get("image_url"), dict) else None
                if isinstance(url, str) and url.startswith("data:image/"):
                    urls.append(url)
        if not urls:
            return
        cams = sorted(str(c) for c in (self._task_cameras or ()))
        payload, meta, marks = {}, {}, []
        for i, url in enumerate(urls):
            try:
                head, b64 = url.split(",", 1)
                blob = base64.b64decode(b64)
            except Exception:  # noqa: BLE001
                continue
            cam = cams[i] if i < len(cams) else ("cam%d" % i)
            fmt = "jpeg" if "jpeg" in head or "jpg" in head else "png"
            w = h = 0
            try:
                import io
                from PIL import Image
                im = Image.open(io.BytesIO(blob))
                w, h = int(im.width), int(im.height)
            except Exception:  # noqa: BLE001
                pass
            payload[cam] = (fmt, w, h, blob)
            meta[cam] = (w, h, len(blob), "jpeg q85" if fmt == "jpeg" else fmt)
            marks.append("%s sent md5[%s]=%s bytes=%d" % (cam, cam,
                          hashlib.md5(blob).hexdigest()[:16], len(blob)))
        if not payload:
            return
        self._panel_frames.update(payload)
        self._panel_meta.update(meta)
        self._panels_files.write_frames(payload, meta)      # 唯一真源：与上面同一份字节
        for cam, item in payload.items():
            used = self._panels_files._last_frames.get(cam)
            same = bool(used) and used[3] is item[3]
            marks.append("panel md5[%s]=%s %s" % (
                cam, hashlib.md5(used[3]).hexdigest()[:16] if used else "none",
                "相等=True" if same else "相等=False"))
        self._log("[panel] " + " | ".join(marks))

    def _set_panels(self, mode=None) -> None:
        """:panels 切换（none|files|all）；文件产出不涉及窗口，headless 也照写。"""
        if mode:
            m = str(mode).lower()
            if m in ("cam", "chart"):
                m = "all"                      # 兼容旧写法：现在都只写文件
            if m not in ("none", "files", "all"):
                self._log("[app] :panels 只支持 none|files|all（shot=立即出曲线）")
                return
            self.panel_request = m
            self.panel_mode = m
            if m == "none":
                self._panels_files = None
            elif self._panels_files is None:
                self._open_panels()
        if self._panels_files is None:
            self._log("[app] 面板: none（未写文件）")
        else:
            self._log(f"[app] 面板: {self.panel_mode} | " + self._panels_files.status())

    def _open_panels(self) -> None:
        """创建文件产出器（不建窗口、不起进程）。"""
        try:
            from demo.panels import PanelFiles
        except Exception as exc:  # noqa: BLE001
            self._log(f"[app] 面板文件不可用（{type(exc).__name__}: {exc}）；面板降级，不影响任务")
            self.panel_mode = "none"
            return
        try:
            self._panels_files = PanelFiles(log=self._log)
            self._install_llm_frame_hook()          # 帧唯一真源：build_messages 里的 data URL
            if self._telemetry is None:
                self._telemetry = deque(maxlen=12000)   # (t, j0..j5, grip, |F|, gap_mm)
            self._panel_capture = True
            self._log("[panel] 文件产出已启用：" + self._panels_files.status())
            self._log(self._panels_hint())          # 用户不知道去哪看图：启动就把路指清楚
        except Exception as exc:  # noqa: BLE001
            self._panels_files = None
            self._panel_capture = False
            self._log(f"[app] 面板文件初始化失败（{type(exc).__name__}: {exc}）；面板降级，不影响任务")

    def _telemetry_rows(self) -> list:
        """最近 10s 的遥测行（供曲线图）。"""
        rows = list(self._telemetry or [])
        if not rows:
            return []
        t_end = float(rows[-1][0])
        return [r for r in rows if t_end - float(r[0]) <= 10.0]

    def _sync_panels(self) -> None:
        """主线程：曲线写盘（内部 <=1Hz 节流）；不涉及任何窗口/GL。"""
        if self._panels_files is not None:
            self._panels_files.write_curves(self._telemetry_rows())

    def _panels_hint(self) -> str:
        """一行「去哪儿看画面/曲线」的提示（启动、任务开始、:help 共用）。"""
        p = self._panels_files
        if p is None:
            return ("[panel] 想看画面/曲线：先输入 :panels all（纯文件产出，headless 也写文件）")
        return (f"[panel] 想看画面/曲线：浏览器打开 {p.index_path}（每 500ms 自动刷新），"
                f"或直接看 {p.frames_path} / {p.curves_path}；"
                "命令台 :panels 看状态、:panels shot 立即刷新曲线、:panels open 打开浏览器")

    #: explorer.exe 在 WSL 里的固定路径（拉起的是 Windows 默认浏览器）
    _WSL_EXPLORER = "/mnt/c/Windows/explorer.exe"

    @staticmethod
    def _windows_path(path: str) -> str:
        r"""给 explorer.exe 用的 Windows 路径：优先 wslpath -w，失败退回 \\wsl.localhost\<distro>\...。"""
        try:
            out = subprocess.run(["wslpath", "-w", path], capture_output=True,
                                 text=True, timeout=5)
            if out.returncode == 0 and str(out.stdout).strip():
                return str(out.stdout).strip()
        except Exception:  # noqa: BLE001 - 没有 wslpath 就自己拼
            pass
        try:
            distro = os.environ.get("WSL_DISTRO_NAME") or "Ubuntu"
            rel = path.lstrip("/").replace("/", "\\")
            return "\\\\wsl.localhost\\" + distro + "\\" + rel
        except Exception:  # noqa: BLE001
            return path

    def _panels_open(self) -> None:
        """用户显式发起的「打开浏览器看面板」；headless 只打印路径，不拉起浏览器。

        候选顺序：xdg-open -> wslview -> /mnt/c/Windows/explorer.exe（WSL，传 Windows 路径）-> open。
        """
        if self._panels_files is None:
            self._set_panels("all")
        if self._panels_files is None:
            self._log("[panel] 面板未启用；请先 :panels all")
            return
        path = self._panels_files.index_path
        if self.args.headless:
            self._log(f"[panel] headless：不打开浏览器，请手动打开 {path}")
            self._log(self._panels_hint())
            return
        explorer = self._WSL_EXPLORER if os.path.exists(self._WSL_EXPLORER) else None
        opener = (shutil.which("xdg-open") or shutil.which("wslview") or explorer
                  or shutil.which("open"))
        if not opener:
            self._log(f"[panel] 未找到可用的浏览器打开器（xdg-open/wslview/explorer.exe/open）；"
                      f"请手动打开 {path}")
            self._log(self._panels_hint())
            return
        target = self._windows_path(path) if opener.endswith("explorer.exe") else path
        try:
            subprocess.Popen([opener, target], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            self._log(f"[panel] 已用浏览器打开 {target}（每 500ms 自动刷新）"
                      + ("；WSL explorer.exe -> Windows 默认浏览器" if opener.endswith("explorer.exe") else ""))
        except Exception as exc:  # noqa: BLE001 - 打开失败只提示路径
            self._log(f"[panel] 浏览器打开失败（{type(exc).__name__}: {exc}）；请手动打开 {path}")
        self._log(self._panels_hint())

    def _panels_shot(self) -> None:
        """立即出图（:panels shot）：帧 + 曲线都补一张。"""
        if self._panels_files is None:
            self._log("[app] 面板未启用（先 :panels all）")
            return
        if self._panel_frames:
            self._panels_files.write_frames(dict(self._panel_frames), self._panel_meta)
        self._panels_files.write_curves(self._telemetry_rows(), force=True)
        self._log("[panel] 已立即出图：" + self._panels_files.status())

    def _panel_caption(self, step: int) -> str:
        """控制台 caption：本步发给 LLM 的画面（相机/分辨率/格式/字节/步号）。"""
        parts, total, fmt = [], 0, "raw rgb"
        for cam in ("overview", "top"):
            meta = self._panel_meta.get(cam)
            if not meta:
                continue
            w, h, nbytes, mfmt = meta
            parts.append("%s %dx%d" % (cam, w, h))
            total += int(nbytes)
            if "jpeg" in str(mfmt):
                fmt = "jpeg q85"
        return ("[panel] 发给 LLM 的画面: %s | %s | 合计 %.1f KB | step %d"
                % (" + ".join(parts) if parts else "(尚无帧)", fmt, total / 1024.0, step))

    def _log(self, msg: str) -> None:
        print(str(msg), flush=True)

    def _set_vlm(self, mode: str) -> None:
        """auto/api: 有 key 用真模型，没有则回退 MockVLM；mock: 强制离线规则决策器。"""
        mode = (mode or "auto").lower()
        self.vlm_request = mode
        if mode == "mock":
            self.vlm = MockVLM()
        else:  # auto 与 api 同一条选择逻辑（api 只是显式表达“我要真模型”）
            self.vlm = make_vlm(prefer_api=True)
        self.vlm_mode = "api" if not isinstance(self.vlm, MockVLM) else "mock"

    def _vlm_name(self) -> str:
        """给日志用的决策器名字：真模型返回模型名，否则 MockVLM。"""
        if isinstance(self.vlm, MockVLM):
            return "MockVLM"
        return str(getattr(self.vlm, "model", None) or VLM_MODEL)

    @staticmethod
    def _mask_secret(value: str) -> str:
        """api_key 掩码（前 4 + 后 4）；**绝不返回明文**。"""
        s = str(value or "")
        if not s:
            return "（未设置）"
        if len(s) <= 8:
            return "*" * len(s)
        return f"{s[:4]}{'*' * (len(s) - 8)}{s[-4:]}"

    @staticmethod
    def _config_fields() -> set:
        """config.local.json 实际填充的字段（来自 core.config.LOCAL_CONFIG_SOURCE）。"""
        src = LOCAL_CONFIG_SOURCE or ""
        if not src or "环境变量已覆盖全部字段" in src:
            return set()
        return {k for k in ("api_key", "base_url", "model") if k in src}

    @classmethod
    def _field_source(cls, field: str, env_names: tuple) -> str:
        """单个字段的来源：config.local.json / env / 内置默认。"""
        if field in cls._config_fields():
            return "config.local.json"
        if any(os.environ.get(n) for n in env_names):
            return "env"
        return "内置默认"

    @classmethod
    def _credential_source(cls) -> str:
        """凭据（api_key）来源：env / config.local.json / 无。"""
        if "api_key" in cls._config_fields():
            return "config.local.json"
        if any(os.environ.get(n) for n in _ENV_KEY_NAMES):
            return "env"
        return "无"

    def _log_no_llm_warning(self) -> None:
        """醒目多行警告：当前根本没有大模型（纯文本，不加依赖）。"""
        self._log(_WARN_LINE)
        self._log("[警告] 当前没有大模型：所有决策来自离线规则 MockVLM，不是 LLM 判断！")
        self._log("        你看到的每一步都是关键词/规则匹配的结果，不代表模型能力。")
        self._log(f"        凭据来源: {self._credential_source()}"
                  "（未检测到 VLM_API_KEY / OPENAI_API_KEY / AI_API_KEY）")
        self._log("        设置方法（任选其一）:")
        self._log("          1) 环境变量: export AI_API_KEY=<key> AI_MODEL=<model> AI_API_URL=<base_url>")
        self._log('          2) 项目内配置: vlm_admittance/config.local.json '
                  '{"base_url": "...", "api_key": "...", "model": "..."}')
        self._log("          3) 运行时切换: 命令台输入 :vlm api（需先配好上面的 key）")
        self._log(_WARN_LINE)

    def _log_vlm_banner(self) -> None:
        """打印决策器类型/模型名/base_url/是否带图；MockVLM 必带 [无 LLM]；**绝不打印 key**。"""
        images = "带图像" if self.args.with_images else "无图像"
        if isinstance(self.vlm, MockVLM):
            if self.vlm_request == "api":
                self._log("[app] 已请求 --vlm api，但未检测到 API key，回退 MockVLM")
            self._log(f"[app] 决策器: MockVLM（离线规则决策器） [无 LLM] | {images}")
            self._log("[app] 离线规则决策器，仅支持有限的指令模式；"
                      "设置 AI_API_KEY/AI_MODEL 后可用真模型")
            if self.vlm_request in ("auto", "api"):
                self._log_no_llm_warning()
            else:
                self._log("[app] 你显式选择了 --vlm mock（离线规则）；如需真模型请 :vlm api")
            return
        model = getattr(self.vlm, "model", None) or VLM_MODEL
        base_url = getattr(self.vlm, "base_url", None) or VLM_BASE_URL
        self._log(f"[app] 决策器: {type(self.vlm).__name__}（真模型） [LLM] | model={model} | "
                  f"base_url={base_url} | {images}")

    def _log_vlm_status(self) -> None:
        """:vlm status：来源 / base_url / model / 掩码 key（永不打印明文）。"""
        self._log(f"[app] 决策器来源: {self._credential_source()}")
        self._log(f"[app] model: {VLM_MODEL}  (来源: "
                  f"{self._field_source('model', _ENV_MODEL_NAMES)})")
        self._log(f"[app] base_url: {VLM_BASE_URL}  (来源: "
                  f"{self._field_source('base_url', _ENV_BASE_NAMES)})")
        self._log(f"[app] api_key: {self._mask_secret(VLM_API_KEY)}")

    # ------------------------------------------------------------------ #
    # 感知模式（INTERFACES.md §4.4）：oracle 给真值位姿；vlm 只给图像 + 像素瞄准
    # ------------------------------------------------------------------ #
    def _agent_cameras(self) -> tuple:
        """本任务实际发给模型的相机，顺序即发送顺序（task-54）。

        vlm 模式默认 ("overview", "top")：top 是俯视相机，像素->世界约 0.65mm/px，是像素瞄准主力
        （只发 overview 时模型给 pixel_camera=top 会导致同一像素两种解释差 ~37mm）；
        oracle 只发 overview（省渲染）；--cameras 显式指定时尊重之。
        """
        if not getattr(self.args, "with_images", True):
            return ()
        explicit = getattr(self.args, "cameras", None)
        if explicit:
            names = tuple(str(c).strip() for c in str(explicit).split(",") if str(c).strip())
        elif self.perception_mode == "vlm":
            names = ("overview", "top")
        else:
            names = ("overview",)
        known = set(getattr(self.env, "camera_names", ()) or ())
        if known:
            missing = [n for n in names if n not in known]
            if missing:
                self._log(f"[app] 警告: 相机 {missing} 不在场景 {sorted(known)} 中，已忽略")
            names = tuple(n for n in names if n in known)
        if self.perception_mode == "vlm" and names and "top" not in names:
            self._log("[app] 警告: vlm 模式未发送 top 相机，而 pixel_camera 默认 top；"
                      "请在 --cameras 里带上 top，或让模型显式指定 pixel_camera")
        return names

    def _perception_notice(self, exc: "Exception | None", key: str) -> None:
        """契约接口未就绪时只提示一次，不阻塞运行。"""
        if self._perception_warned:
            return
        self._perception_warned = True
        detail = f"{type(exc).__name__}: {exc}" if exc is not None else "接口缺失"
        self._log(f"[app] 感知接口未就绪（{key}，{detail}）；vlm 模式退化为 oracle 行为")

    def _set_perception(self, mode: "str | None" = None) -> None:
        """:perception 切换；auto: 真模型 -> vlm，MockVLM -> oracle。"""
        if mode:
            m = str(mode).lower()
            if m not in ("auto", "vlm", "oracle"):
                self._log("[app] :perception 只支持 auto|vlm|oracle")
                return
            self.perception_request = m
        if self.perception_request == "auto":
            self.perception_mode = "oracle" if isinstance(self.vlm, MockVLM) else "vlm"
        else:
            self.perception_mode = self.perception_request
        # --planner 未显式指定时：vlm 模式默认 subgoal（§5.6），oracle 保持 flat（旧行为）
        if self.planner_request:
            self.planner_mode = self.planner_request
        else:
            self.planner_mode = "subgoal" if self.perception_mode == "vlm" else "flat"
        self._last_vlm_point = None
        self._last_pixel = None
        resolver = "ready" if _pixel_to_table is not None else "missing(sim.vision 未就绪)"
        self._log(f"[app] 感知模式: {self.perception_mode}（请求 {self.perception_request}；"
                  f"include_poses={self.perception_mode != 'vlm'}；pixel_resolver={resolver}）")
        if self.perception_mode == "vlm" and _pixel_to_table is None:
            self._perception_notice(None, "sim.vision.pixel_to_table")

    def _refine_notice(self) -> None:
        """§5.1 未就绪只提示一次（像素链路退化为粗像素直投）。"""
        if self._refine_warned:
            return
        self._refine_warned = True
        self._log("[app] sim.perception 未就绪，像素链路退化为「粗像素直投」（无颜色精修）")

    def _current_subject_color(self) -> "str | None":
        """目标颜色：上一次 locate 的 color > 子目标 subject_color > 任务文本颜色词（不读真值）。"""
        if self._last_locate_color:
            return str(self._last_locate_color)
        for sg in self._last_plan or []:
            c = getattr(sg, "subject_color", None)
            if c:
                return str(c)
        text = (self._last_task or "").lower()
        for key, color in _COLOR_WORDS:
            if key in text:
                return color
        return None

    @staticmethod
    def _delta_desc(delta_px: float, camera: str) -> str:
        """精修位移描述；top 相机 1 px = 1.489 mm（Lead 实测），overview 为斜视只给像素。"""
        if str(camera).lower() == "top":
            return f"Δ={delta_px:.1f}px (≈{delta_px * 1.489:.1f}mm)"
        return f"Δ={delta_px:.1f}px"

    def _resolve_world(self, u, v, cam, overview_uv=None, z_hint=None):
        """§5.10：优先双像素三角定位（residual_mm>15 -> 失败）；单像素回退平面反投影。

        返回 (xyz | None, source:str, raw:dict)。source ∈ triangulate / single_view(z=…) /
        off_table / none。
        """
        if _object_point is not None:
            kw: dict = {}
            if z_hint is not None:
                kw["z_hint"] = float(z_hint)
            if cam == "overview":
                kw["overview_uv"] = (float(u), float(v))
                if overview_uv is not None:
                    kw["top_uv"] = (float(overview_uv[0]), float(overview_uv[1]))
            else:
                kw["top_uv"] = (float(u), float(v))
                if overview_uv is not None:
                    kw["overview_uv"] = (float(overview_uv[0]), float(overview_uv[1]))
            try:
                res = _object_point(self.env, **kw) or {}
            except Exception as exc:  # noqa: BLE001 - 三角失败回退单目
                res = {}
                self._log(f"[感知] object_point 异常（回退单目）: {type(exc).__name__}: {exc}")
            src = str(res.get("source") or "")
            xyz = res.get("xyz")
            if res.get("ok") and xyz is not None:
                arr = np.asarray(xyz, dtype=float).reshape(-1)
                if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                    if src == "triangulate":
                        return arr, "triangulate", res
                    return arr, f"single_view(z={float(z_hint or TABLE_HEIGHT):.3f})", res
            if src == "triangulate":            # 残差过大/不可信 -> 可重试 perception_error
                return None, "triangulate_residual", res
        if _pixel_to_table is None:
            self._perception_notice(None, "sim.vision.pixel_to_table")
            return None, "none", {}
        try:
            xyz = _pixel_to_table(self.env, float(u), float(v), camera=cam)
        except Exception as exc:  # noqa: BLE001 - 越界/异常一律失败，绝不崩
            self._log(f"[感知] 像素反投影异常: {type(exc).__name__}: {exc}")
            return None, "error", {}
        if xyz is None:
            return None, "off_table", {}
        arr = np.asarray(xyz, dtype=float).reshape(-1)
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            return None, "off_table", {}
        return arr, f"single_view(z={TABLE_HEIGHT:.3f})", {}

    def _auto_overview_pixel(self, u, v, cam, color):
        """§5.10 自动补第二视角：top 像素 -> 世界点 -> 投影到 overview -> 找同色块质心。

        找不到/异常一律返回 None 并写 self._auto_ov_reason（gray/白/黑类 color_blob 设计上 found=False）。
        """
        self._auto_ov_reason = ""
        if _color_blob is None:
            self._auto_ov_reason = "sim.perception.color_blob 未就绪"
            return None
        hint = None
        if _pixel_to_table is not None:
            try:
                xyz = _pixel_to_table(self.env, float(u), float(v), camera=cam)
            except Exception as exc:  # noqa: BLE001 - 粗投影失败就用像素当 hint
                xyz = None
                self._log(f"[感知] 自动 overview 粗投影失败: {type(exc).__name__}: {exc}")
            if xyz is not None and _project_point is not None:
                try:
                    p = _project_point(self.env, np.asarray(xyz, dtype=float), "overview")
                    if p is not None:
                        hint = (float(p[0]), float(p[1]))
                except Exception:  # noqa: BLE001
                    hint = None
        if hint is None:
            hint = (float(u), float(v))
        try:
            res = _color_blob(self.env.render("overview"), color,
                              hint_uv=hint, radius_px=160)
        except Exception as exc:  # noqa: BLE001 - 找不到/异常都只回退单目，不报错
            self._auto_ov_reason = f"overview 色块搜索异常（{type(exc).__name__}: {exc}）"
            return None
        found = res.get("found") if isinstance(res, dict) else getattr(res, "found", False)
        uv = res.get("uv") if isinstance(res, dict) else getattr(res, "uv", None)
        if not found or uv is None:
            self._auto_ov_reason = (f"overview 未找到 {color} 色块"
                                    "（gray/白/黑类设计上找不到）")
            return None
        return (float(uv[0]), float(uv[1]))

    def _overview_bbox_top(self, color, top_uv=None):
        """取最近一次**匹配** locate 的 overview bbox **顶边中点**当第二像素。

        Lead 实测：模型在 top 点顶面、在 overview 常点到瓶身中部，两点不是同一物理点（xy 偏 25mm）；
        改用 bbox 顶边中点后偏 4.2mm。仅当 bbox 明显竖长（高/宽 ≥ _TALL_BBOX_RATIO，典型瓶/筒）才启用，
        矮方块路径保持不变。匹配条件：颜色相同，或（颜色未知时）top 像素相距 ≤80px。
        返回 (u_center, v_top, bbox) 或 None。
        """
        rec = None
        for r in reversed(self._recent_locate):
            if not r.get("bbox_overview"):
                continue
            if color and r.get("color") and str(r["color"]) != str(color):
                continue
            if top_uv is not None and r.get("pixel_top") is not None:
                try:
                    px = np.asarray(r["pixel_top"], dtype=float).reshape(-1)[:2]
                    if float(np.linalg.norm(px - np.asarray(top_uv, dtype=float)[:2])) > 80.0:
                        continue
                except Exception:  # noqa: BLE001 - 无法比对时不做距离筛选
                    pass
            rec = r
            break
        if rec is None:
            return None
        try:
            u1, v1, u2, v2 = [float(x) for x in rec["bbox_overview"]]
        except Exception:  # noqa: BLE001
            return None
        w, h = abs(u2 - u1), abs(v2 - v1)
        if w <= 1 or h <= 1 or h / max(w, 1e-6) < _TALL_BBOX_RATIO:
            return None
        w_img, h_img = getattr(self.env, "render_size", (512, 512))
        u_c = float(np.clip((u1 + u2) / 2.0, 0.0, float(w_img) - 1))
        v_t = float(np.clip(min(v1, v2), 0.0, float(h_img) - 1))
        return (u_c, v_t, (u1, v1, u2, v2))

    def _pixel_resolver(self, u, v, camera: str = "top", *extra, **kw):
        """§5.6/§5.10 像素链路：VLM 像素（可含 overview 像素）-> 颜色精修 -> 世界坐标。"""
        cam = str(camera or "top")
        overview_uv = None
        if extra:
            overview_uv = extra[0]
        for key in ("overview_uv", "target_pixel_overview", "pixel_overview", "uv_overview"):
            if overview_uv is None and kw.get(key) is not None:
                overview_uv = kw[key]
        u0, v0 = float(u), float(v)
        u1, v1, refined, delta = u0, v0, False, 0.0
        color = self._current_subject_color()
        if _refine_pixel is None:
            self._refine_notice()
        elif color:
            try:
                rgb = self.env.render(cam)
                res = _refine_pixel(rgb, color, (u0, v0), camera=cam, env=self.env) or {}
                if res.get("refined"):
                    uv = res.get("uv") or (u0, v0)
                    u1, v1 = float(uv[0]), float(uv[1])
                    refined = True
                    delta = float(res.get("delta_px") or 0.0)
            except Exception as exc:  # noqa: BLE001 - 精修失败必须回退粗像素，绝不崩
                self._log(f"[感知] 颜色精修异常（回退粗像素）: {type(exc).__name__}: {exc}")
        # §5.10：第二视角像素的优先级
        #   1) 有匹配的 overview bbox（且竖长）-> 用 bbox 顶边中点（模型常在 overview 点瓶身中部）
        #   2) 模型给了 overview 像素 -> 用它
        #   3) 颜色已知 -> color_blob 在 overview 里自动找同色块
        auto_overview = False
        bbox_overview = False
        ov_uv = overview_uv
        bbox_hit = self._overview_bbox_top(color, (u1, v1))
        if bbox_hit is not None:                 # 有匹配的 overview bbox -> 顶边中点优先
            ov_uv = (bbox_hit[0], bbox_hit[1])
            bbox_overview = True
        elif ov_uv is None and color:
            ov_uv = self._auto_overview_pixel(u1, v1, cam, color)
            auto_overview = ov_uv is not None
        arr, source, raw = self._resolve_world(u1, v1, cam, overview_uv=ov_uv)
        pix_desc = f"({u0:.0f},{v0:.0f})[{cam}]"
        if bbox_overview:
            pix_desc += (f" + overview(bbox-top {float(ov_uv[0]):.0f},{float(ov_uv[1]):.0f}"
                         f" from {bbox_hit[2]})")
        elif overview_uv is not None:
            pix_desc += (f" + overview(model {float(overview_uv[0]):.0f},"
                         f"{float(overview_uv[1]):.0f})")
        elif auto_overview:
            pix_desc += f" + overview(auto {float(ov_uv[0]):.0f},{float(ov_uv[1]):.0f})"
        if arr is None:
            self._last_pixel = (u0, v0, cam)
            self._last_vlm_point = None
            if source == "triangulate_residual":
                why = f"双目三角残差过大 residual={raw.get('residual_mm')}mm > 15mm"
                self._log(f"[感知] VLM 像素 {pix_desc} {why}，返回可重试 perception_error")
                return {"ok": False, "source": "triangulate",
                        "residual_mm": raw.get("residual_mm"),
                        "why": "双目三角残差过大，定位不可信，请重新给两个相机的像素"}
            elif source == "off_table":
                why = "反投影失败（桌外/相机后方）"
            elif source == "none":
                return None
            else:
                why = f"反投影异常（{source}）"
            self._log(f"[感知] VLM 像素 {pix_desc} {why}，返回可重试 perception_error")
            return None
        self._last_pixel = (u0, v0, cam)
        self._last_vlm_point = arr[:3].copy()
        resid = raw.get("residual_mm")
        if source.startswith("triangulate") and source != "triangulate_residual":
            kind = ("bbox-top-overview" if bbox_overview else
                    "auto-overview" if auto_overview else
                    "model-overview" if overview_uv is not None else "unknown")
            src_desc = f"source=triangulate({kind})"
            if isinstance(resid, (int, float)):
                src_desc += f" residual={resid:.1f}mm"
        else:
            src_desc = f"source={source}"
            if (source.startswith("single_view") and overview_uv is None
                    and not bbox_overview and self._auto_ov_reason):
                src_desc += f"（{self._auto_ov_reason}）"
        # 注意：下面是模型自己选的像素（+可选颜色精修）的反投影，不是场景真值坐标
        if refined:
            mid = f"→ 颜色精修 ({u1:.0f},{v1:.0f}) {self._delta_desc(delta, cam)} → {src_desc} → 世界"
        else:
            why = "颜色精修未启用" if color is None else "颜色精修未命中"
            mid = f"→ {why}（回退粗像素） → {src_desc} → 世界"
        self._log(f"[感知] VLM 像素 {pix_desc} {mid} "
                  f"({arr[0]:.3f}, {arr[1]:.3f}, {arr[2]:.3f})")
        if raw:            # §5.10：object_point 结果按 dict 回传（loop 会记录 source/residual_mm）
            return {"ok": True, "xyz": [float(v) for v in arr[:3]],
                    "source": raw.get("source") or source,
                    "residual_mm": raw.get("residual_mm")}
        return arr[:3]

    def _executing_skills(self, call):
        """包一层 executor：每步日志带上当前感知模式（§4.4）+ 相机面板 caption（task-64）。"""
        self._log(f"[app] 感知 {self.perception_mode} | 执行 {call.skill}")
        if self._panel_capture:
            self._panel_step += 1
            self._log(self._panel_caption(self._panel_step))
        return self.skills(call)

    def _pixel_label(self) -> str:
        if not self._last_pixel:
            return "VLM"
        u, v, cam = self._last_pixel
        return f"VLM ({u:.0f},{v:.0f})[{cam}]"

    def _draw_vlm_point(self, scn) -> None:
        """叠加：画出模型像素反投影出的世界点（label 带 VLM），只在 vlm 模式画。"""
        if self.perception_mode != "vlm" or self._last_vlm_point is None:
            return
        try:
            import mujoco
        except ImportError:  # pragma: no cover
            return
        maxgeom = int(getattr(scn, "maxgeom", 0) or 0)
        idx = int(getattr(scn, "ngeom", 0) or 0)
        if idx >= maxgeom:
            return
        try:
            geom = scn.geoms[idx]
            mujoco.mjv_initGeom(geom, int(mujoco.mjtGeom.mjGEOM_SPHERE),
                                np.array([0.012, 0.0, 0.0], dtype=np.float64),
                                np.asarray(self._last_vlm_point, dtype=np.float64),
                                np.eye(3, dtype=np.float64).reshape(9),
                                np.array([1.0, 0.15, 0.15, 1.0], dtype=np.float32))
            geom.label = self._pixel_label()
            try:
                geom.category = int(mujoco.mjtCategory.mjCAT_DECOR)
            except Exception:  # noqa: BLE001
                pass
            scn.ngeom = idx + 1
        except Exception as exc:  # noqa: BLE001 - 叠加失败不能打断渲染
            self._overlay_unavailable_notice(exc)

    # ------------------------------------------------------------------ #
    # 当前目标点叠加：每帧先 clear 再 draw，最后交给 viewer.sync() 合入渲染
    # ------------------------------------------------------------------ #
    def _overlay_unavailable_notice(self, exc: "Exception | None" = None) -> None:
        """接口未就绪/叠加渲染异常时只提示一条，绝不影响其他功能。"""
        if self._overlay_warned:
            return
        self._overlay_warned = True
        if exc is None:
            self._log("[app] 目标点叠加不可用（sim.overlay 或 runner.target_state 未就绪），已跳过")
        else:
            self._log(f"[app] 目标点叠加渲染跳过: {type(exc).__name__}: {exc}")

    def _force_key_notice(self) -> None:
        """legacy target_state 没有 force_norm 键：宁可少画也不误报，只提示一次。"""
        if self._force_key_warned:
            return
        self._force_key_warned = True
        self._log("[app] target_state 缺少 force_norm，保守起见不显示让位标记（只画青色目标点）")

    @staticmethod
    def _pure_rot_yield(state: dict) -> float:
        """纯导纳姿态让位角（rad）= |rotvec(adm_quat * conj(ref_quat))|。

        不能用 |adm_quat - ref_quat|（四元数分量差没有物理意义）。工具缺失或异常时返回 0。
        """
        if _quat_mul is None or _quat_conj is None or _rotvec_from_quat is None:
            return 0.0
        try:
            ref_q, adm_q = state.get("ref_quat"), state.get("adm_quat")
            if ref_q is None or adm_q is None:
                return 0.0
            dq = _quat_mul(np.asarray(adm_q, dtype=float),
                           _quat_conj(np.asarray(ref_q, dtype=float)))
            return float(np.linalg.norm(_rotvec_from_quat(dq)))
        except Exception:  # noqa: BLE001 - 旋转让位只是锦上添花，失败即视为 0
            return 0.0

    def _update_target_overlay(self, scn) -> None:
        """把「当前目标点」画进 viewer.user_scn（必须先于 viewer.sync()）。

        顺序：clear -> runner.target_state() -> draw_target -> 调用方 sync()。
        开关关闭时只 clear（user_scn.ngeom 保持 0）；valid=False 或接口缺失时跳过。
        """
        try:
            if _overlay_clear is not None:
                _overlay_clear(scn)
            if self.target_mode == "off":
                return
            if _overlay_draw_target is None or not hasattr(self.runner, "target_state"):
                self._overlay_unavailable_notice()
                return
            state = self.runner.target_state()
            if not isinstance(state, dict) or not state.get("valid"):
                return
            cmd_pos, cmd_quat = state.get("cmd_pos"), state.get("cmd_quat")
            if cmd_pos is None or cmd_quat is None:
                return
            ref_pos = state.get("ref_pos")
            # 让位类标记（橙色 ref 球+连线、紫红 adm 球）的唯一触发条件 = 导纳外力 > 0.5N。
            # 不能用 |adm_pos-ref_pos| 判让位：导纳稳态跟踪滞后 (D/K)*v 让无接触时也有
            # 中位 5.5mm / 最大 18.7mm 的位移差（验证者实测），必然误报。
            if "force_norm" in state:
                force_norm = float(state.get("force_norm") or 0.0)
            else:  # legacy：没有 force_norm，宁可少画也不误报
                force_norm = 0.0
                self._force_key_notice()
            yielding = force_norm > _YIELD_FORCE_N
            rot_yield = self._pure_rot_yield(state) if yielding else 0.0
            # 模式语义（sim/overlay.draw_target：quat 决定是否画三轴，show_ref 决定是否画橙色
            # reference 点+连线，show_label 决定是否贴文字）：
            #   on     = 球 + 三轴（+接触时橙色 reference + 连线 + 标签）
            #   axes   = 球 + 三轴（+接触时橙色 reference，不画标签）
            #   sphere = 只画青色目标球
            # 青色目标球/三轴始终显示（本需求主角），橙色/紫红让位标记只在接触时出现。
            quat = None if self.target_mode == "sphere" else cmd_quat
            show_ref = yielding
            show_label = self.target_mode == "on"
            _overlay_draw_target(scn, cmd_pos, quat,
                                 ref_pos=ref_pos,
                                 show_ref=show_ref, show_label=show_label)
            # 紫红标记 = 纯导纳输出点 adm_pos：到橙色 ref 的距离=柔顺让位量，
            # 到青色 cmd 的距离=重力积分静差补偿量；仅接触（force_norm>0.5N）时显示。
            # 姿态让位明显时带上 adm_quat，多画三轴把“旋转让位”也显示出来。
            adm_pos = state.get("adm_pos")
            if adm_pos is not None and yielding:
                adm_q = state.get("adm_quat") if rot_yield > _ROT_YIELD_VISIBLE_RAD else None
                _overlay_draw_target(scn, adm_pos, adm_q, show_ref=False, show_label=False,
                                     radius=0.008, rgba=_ADM_MARKER_RGBA)
            # §4.4：vlm 模式把「模型像素反投影出的世界点」画出来（label 带 VLM）
            self._draw_vlm_point(scn)
        except Exception as exc:  # noqa: BLE001 - 叠加渲染异常绝不能打断 viewer 主循环
            self._overlay_unavailable_notice(exc)

    def _set_target_mode(self, mode: "str | None" = None) -> None:
        """:target 切换；无参数只打印当前模式。"""
        modes = ("on", "off", "axes", "sphere")
        if mode:
            m = str(mode).lower()
            if m not in modes:
                self._log(f"[app] :target 只支持 {'|'.join(modes)}")
                return
            self.target_mode = m
        self._log(f"[app] 目标点显示: {self.target_mode}")
        if self.target_mode != "off" and (_overlay_draw_target is None
                                          or not hasattr(self.runner, "target_state")):
            self._overlay_unavailable_notice()

    # ------------------------------------------------------------------ #
    # 应急恢复：skills 可能还没提供 hard_recover（并行开发），缺失时安全跳过
    # ------------------------------------------------------------------ #
    def _hard_recover(self) -> "str | None":
        """调用 skills.hard_recover()，打印并返回它给出的恢复级别。

        skills 尚未提供该方法时只提示并跳过，保证本文件可独立运行。
        """
        if not hasattr(self.skills, "hard_recover"):
            self._log("[app] skills 尚未提供 hard_recover")
            return None
        level = self.skills.hard_recover()
        self._log(f"[app] 应急恢复级别: {level}")
        return level

    # ------------------------------------------------------------------ #
    # 命令分发
    # ------------------------------------------------------------------ #
    def _history_notice(self, exc: Exception) -> None:
        """AgentLoop 还不支持 history 参数：退回旧行为（每条命令独立），只提示一次。"""
        if self._history_warned:
            return
        self._history_warned = True
        self._log(f"[app] AgentLoop 暂不支持 history（{type(exc).__name__}: {exc}），"
                  "本轮按无上下文运行")

    def _build_agent_loop(self):
        """按当前契约能力构造 AgentLoop（history/include_poses/pixel_resolver 逐个探测）。

        返回 (loop, accepted)：accepted 记录真正被构造器接受的可选关键字。
        """
        cameras = getattr(self, "_task_cameras", None)
        if cameras is None:
            cameras = self._agent_cameras()
        base = dict(executor=self._executing_skills, vlm=self.vlm,
                    cameras=cameras,
                    max_steps=self.args.max_steps, log=self._log)
        optional: list[tuple[str, object]] = [("history", self.history), ("locator", self._locator)]
        if self.perception_mode == "vlm":            # §4.4：无真值位姿 + 像素解析器
            optional += [("include_poses", False), ("pixel_resolver", self._pixel_resolver)]
        accepted: dict = {}
        for key, value in optional:                  # 逐个探测契约关键字是否已就绪
            try:
                AgentLoop(self.env, **base, **accepted, **{key: value})
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                if key == "history":
                    self._history_notice(exc)
                else:
                    self._perception_notice(exc, key)
                continue
            accepted[key] = value
        return AgentLoop(self.env, **base, **accepted), accepted

    def _run_agent_loop(self, task: str) -> list:
        """构造 AgentLoop 并执行，透传/回写 self.history（接口未就绪时退化为旧行为）。"""
        loop, accepted = self._build_agent_loop()
        seeded = "history" in accepted
        try:
            results = loop.run(task, history=self.history)
        except TypeError as exc:                     # run 还是旧签名
            if not seeded:                           # 构造已播种则不算退化，不必提示
                self._history_notice(exc)
            results = loop.run(task)
        if hasattr(loop, "history"):                 # 回写完整对话，供下一条命令指代
            self.history = list(loop.history)
        return results

    # ------------------------------------------------------------------ #
    # §5.2 子目标规划 / 独立复核接线（taskplan 未就绪时清晰降级，绝不崩）
    # ------------------------------------------------------------------ #
    def _taskplan_notice(self, why: str = "agent.taskplan 尚未就绪") -> None:
        if self._taskplan_warned:
            return
        self._taskplan_warned = True
        self._log(f"[app] {why}；--planner subgoal 暂按 flat（单任务）执行")

    def _locator(self, target, color=None, camera="top"):
        """§5.3 语义标定后端：接 taskplan.SubGoalVerifier.locate。

        未就绪/失败一律返回 None -> AgentLoop 记为**可重试** perception_error（不空转、不崩）。
        """
        if _SubGoalVerifier is None:
            self._perception_notice(None, "agent.taskplan.SubGoalVerifier.locate")
            return None
        if self._verifier is None:
            try:
                self._verifier = _SubGoalVerifier(self.env, perception=None, vlm=self.vlm,
                                                  renderer=None,  # taskplan: None -> env.render（§5.2）；env.render 无 .rgb
                                                  log=self._log)
            except Exception as exc:  # noqa: BLE001
                self._log(f"[感知] 定位后端构造失败: {type(exc).__name__}: {exc}")
                return None
        try:
            res = self._verifier.locate(str(target), color=color,
                                        camera=str(camera or "top")) or {}
        except Exception as exc:  # noqa: BLE001 - 标定失败必须可重试
            self._log(f"[感知] locate({target!r}) 失败（可重试）: {type(exc).__name__}: {exc}")
            return None
        if not isinstance(res, dict):
            self._log(f"[感知] locate({target!r}) 返回 {type(res).__name__}，应为 dict（可重试）")
            return None
        if res.get("pixel") is None:
            # 原样回传给 AgentLoop：它会记成可重试 perception_error 并带上 why（比返回 None 更有用）
            self._log(f"[感知] locate({target!r}) 未拿到像素（可重试）: "
                      f"{res.get('why') or '未找到目标色块'}")
        else:
            # 记住标定出的颜色：像素链路自动补 overview 第二视角时用它找同色块（§5.10）
            got_color = res.get("color") or color
            if got_color:
                self._last_locate_color = str(got_color)
            # 记下这次 locate 的像素与 bbox：像素链路可用 bbox 顶边中点补第二视角（§5.10）
            self._recent_locate.append({
                "target": str(target), "color": (str(got_color) if got_color else None),
                "pixel_top": res.get("pixel_top") or res.get("pixel"),
                "pixel_overview": res.get("pixel_overview"),
                "bbox_top": res.get("bbox_top"), "bbox_overview": res.get("bbox_overview"),
            })
            del self._recent_locate[:-_RECENT_LOCATE_MAX]
        return res

    def _observe_home(self, sg=None):
        """复核前的「观察位」：走 skills 的 home（带载归位语义），失败也不影响判定。

        taskplan 在每次复核前调用；夹爪杵在叠放点上方会连 VLM 都看不到目标，
        所以先把机械臂挪回安全位。异常自己吞掉，由 taskplan 记日志。

        P0 修复（pick 子目标复核丢件）：**pick 类子目标复核前绝对不挪机械臂**。
        pick 的复核要证明"物体已经被抓走了"，而归位（尤其带载瞬移）会把刚抓起来的
        物体丢掉/丢回原位 —— 实测复核因此永远拿到 subject_still_present，
        3 次尝试全废。同理，只要技能还认为手里有东西（holding 非 None），也不动。
        """
        goal_type = str(getattr(sg, "goal_type", "") or "")
        holding = getattr(self.skills, "holding", None)
        if goal_type == "pick" or holding is not None:
            why = ("pick 子目标" if goal_type == "pick"
                   else f"仍持物(holding={holding})")
            self._log(f"[panel] 复核前观察位：跳过 skills.home（{why}）"
                      "—— 挪动机械臂会把刚抓起的物体丢掉，复核将失去意义")
            return None
        try:
            r = self.skills(SkillCall("home"))
            ok = bool(getattr(r, "ok", False))
            self._log(f"[panel] 复核前观察位：skills.home -> {'OK' if ok else 'FAIL'}"
                      + (f"（{getattr(r, 'message', '')}）" if not ok else ""))
            return r
        except Exception as exc:  # noqa: BLE001 - 观察位失败绝不能影响复核/任务
            self._log(f"[panel] 复核前观察位失败（{type(exc).__name__}: {exc}），由 taskplan 兜底")
            return None

    def _new_agent_loop(self, sg=None, attempt=None):
        """taskplan 的 loop_factory：返回全新 AgentLoop（兼容 (sg, attempt)/(sg)/() 三种调用）。

        同时把每次创建的 loop 记进 self._subgoal_loops，任务结束后可回捞 SkillResult 列表
        （保持 run_task 返回 list[SkillResult] 的既有契约）。
        """
        loop = self._build_agent_loop()[0]
        self._subgoal_loops.append(loop)
        return loop

    def _run_subgoal_plan(self, task: str):
        """逐子目标闭环（§5.2）；未就绪 / 异常 / 空计划 -> None（run_task 回退 flat）。"""
        if _SubGoalOrchestrator is None or _plan_subgoals is None:
            self._taskplan_notice()
            return None
        try:
            state = self.env.get_state()
            plan = list(_plan_subgoals(self.vlm, task, state, {},
                                       log=self._log,
                                       include_poses=self.perception_mode != "vlm") or [])
            if not plan:
                self._taskplan_notice("子目标规划返回空表")
                return None
            self._last_plan = plan
            verifier = None
            if _SubGoalVerifier is not None:
                verifier = _SubGoalVerifier(self.env, perception=None, vlm=self.vlm,
                                            renderer=None,  # taskplan: None -> env.render（§5.2）；env.render 无 .rgb
                                            log=self._log)
            self._subgoal_loops = []
            kw = dict(include_poses=self.perception_mode != "vlm")
            try:      # observe：复核前让开视线（走 skills 的 home，带载归位语义）
                orch = _SubGoalOrchestrator(self.env, self.vlm, self._new_agent_loop,
                                            verifier, self._log,
                                            observe=self._observe_home, **kw)
            except TypeError:   # 旧版 orchestrator 没有 observe 形参 -> 退回默认 env.home()
                orch = _SubGoalOrchestrator(self.env, self.vlm, self._new_agent_loop,
                                            verifier, self._log, **kw)
            run = getattr(orch, "run", None)
            if run is None:
                self._taskplan_notice("SubGoalOrchestrator 没有 run()")
                return None
            # 契约签名：run(task, subgoals=None, *, state=None, images=None)
            try:
                out = run(task, plan, state=state, images={})
            except TypeError as exc:
                self._taskplan_notice(f"SubGoalOrchestrator.run 调用签名不匹配（{exc}）")
                return None
            return list(out) or None
        except Exception as exc:  # noqa: BLE001 - 规划/复核异常绝不能拖垮交互台
            self._taskplan_notice(f"agent.taskplan 调用失败（{type(exc).__name__}: {exc}）")
            return None

    def _subgoal_skill_results(self) -> list:
        """回捞子目标闭环里所有 AgentLoop 的 SkillResult（保持 run_task 返回契约）。"""
        out: list = []
        for loop in getattr(self, "_subgoal_loops", []):
            out.extend(list(getattr(loop, "results", []) or []))
        return out

    @staticmethod
    def _fmt_subgoal_result(item: dict) -> str:
        """子目标结果 -> 「g1 完成（几何：…）」/「g2 未完成（3 次尝试）」（§5.6）。"""
        sg = item.get("subgoal") if isinstance(item, dict) else None
        sid = getattr(sg, "id", None) or (item.get("id") if isinstance(item, dict) else None) or "?"
        done = bool(item.get("done")) if isinstance(item, dict) else False
        attempts = item.get("attempts") if isinstance(item, dict) else None
        why = str(item.get("why") or "") if isinstance(item, dict) else ""
        ev = item.get("evidence") if isinstance(item, dict) else None
        if done:
            detail = f"{ev}：{why}" if (ev and why) else (str(ev) if ev else why)
            return f"{sid} 完成" + (f"（{detail}）" if detail else "")
        tail = f"{attempts} 次尝试" if attempts else f"最多 {_SUBGOAL_MAX_ATTEMPTS} 次尝试"
        return f"{sid} 未完成（{tail}）" + (f"：{why}" if why else "")

    @staticmethod
    def _fmt_verify_item(item: dict, sg=None) -> str:
        sid = getattr(sg, "id", None) or "?"
        done = bool(item.get("done"))
        ev = item.get("evidence") or "none"
        why = str(item.get("why") or "")
        conf = item.get("confidence")
        extra = f" confidence={conf:.2f}" if isinstance(conf, (int, float)) else ""
        return f"{sid} {'完成' if done else '未完成'}（依据: {ev}{extra}）" + (f" {why}" if why else "")

    def _print_plan(self) -> None:
        """:plan 打印子目标表（taskplan 未就绪时给出清晰提示）。"""
        if not self._last_plan and _plan_subgoals is not None and self._last_task:
            try:
                self._last_plan = list(_plan_subgoals(self.vlm, self._last_task,
                                                      self.env.get_state(), {},
                                                      log=self._log) or [])
            except Exception as exc:  # noqa: BLE001
                self._log(f"[app] 子目标规划失败: {type(exc).__name__}: {exc}")
        if not self._last_plan:
            if _plan_subgoals is None:
                self._taskplan_notice(":plan 需要 agent.taskplan")
            else:
                self._log("[app] 暂无子目标表（先跑一条任务，或规划返回空表）")
            return
        self._log(f"[app] 子目标表（{len(self._last_plan)} 条）:")
        for sg in self._last_plan:
            hint = getattr(sg, "verify_hint", "")
            self._log(f"  {getattr(sg, 'id', '?'):<3} {getattr(sg, 'goal_type', ''):<18} "
                      + str(getattr(sg, "instruction", ""))
                      + (f"  | 复核: {hint}" if hint else ""))

    def _print_verify(self) -> None:
        """:verify 打印每个子目标的复核结论与依据（geometry/vlm）。"""
        if self._last_verify:
            self._log(f"[app] 子目标复核（{len(self._last_verify)} 条）:")
            for sg, item in zip(self._last_plan or [], self._last_verify):
                self._log(f"  {self._fmt_verify_item(item, sg)}")
            return
        if self._subgoal_results:
            self._log("[app] 子目标复核（上次执行结果）:")
            for item in self._subgoal_results:
                self._log(f"  {self._fmt_subgoal_result(item)}")
            return
        if _SubGoalVerifier is None:
            self._taskplan_notice(":verify 需要 agent.taskplan")
            return
        if not self._last_plan:
            self._log("[app] 暂无子目标表可复核（先跑一条任务，或用 :plan 生成）")
            return
        try:
            verifier = _SubGoalVerifier(self.env, perception=None, vlm=self.vlm,
                                        renderer=None,  # taskplan: None -> env.render（§5.2）
                                        log=self._log)
            self._last_verify = [verifier.verify(sg) or {} for sg in self._last_plan]
        except Exception as exc:  # noqa: BLE001 - 复核异常不崩，如实报告
            self._log(f"[app] 子目标复核失败: {type(exc).__name__}: {exc}")
            return
        self._log(f"[app] 子目标复核（{len(self._last_verify)} 条）:")
        for sg, item in zip(self._last_plan, self._last_verify):
            self._log(f"  {self._fmt_verify_item(item, sg)}")

    def _set_scene(self, mode: "str | None" = None) -> None:
        """:scene default|daily 切换场景（重建 env 与依赖）。"""
        if mode:
            m = str(mode).lower()
            if m not in ("default", "daily"):
                self._log("[app] :scene 只支持 default|daily")
                return
            if m == self.scene_mode:
                self._log(f"[app] 场景已是 {m}"
                          f"（物体 {[o.id for o in self.env.list_objects()]}）")
                return
            self.scene_mode = m
            self._setup_runtime()
            self.history = []                 # 物体集合变了，旧对话里的物体不再存在
            self._last_plan, self._last_verify, self._subgoal_results = [], [], []
            self._log(f"[app] 场景已切换: {self.scene_mode}"
                      f"（物体 {[o.id for o in self.env.list_objects()]}）；"
                      "GUI 下请重启窗口刷新 viewer.model")
            return
        self._log(f"[app] 当前场景: {self.scene_mode}"
                  f"（物体 {[o.id for o in self.env.list_objects()]}）")

    @classmethod
    def _summarize_results(cls, results: list) -> dict:
        """「最终成功语义」汇总（task-56 / E1）。

        - corrected：某次 FAIL 之后**同类技能又成功**（如 target_id+target_pixel 被拒 invalid_call 后
          改正成纯像素调用并成功）-> 记「已纠正」，不算失败；
        - fail：其余 FAIL（含目标技能最终没成功）-> 如实计失败，绝不洗白；
        - 出现 ask_user 或不可重试错误码 -> 中止（无论前面纠正与否）。
        返回 dict(outcome, n_ok, n_corrected, n_fail)；三段计数之和 == 步数。
        """
        n_ok = 0
        recovered = 0          # 先被拒（FAIL）后同类成功 -> 真正重试
        arg_fixed = 0          # §5.12 validate 自动纠正了冗余参数（metrics.corrected_args）
        failed: list = []
        for i, r in enumerate(results):
            skill = getattr(r, "skill", "")
            if getattr(r, "ok", False):
                if cls._corrected_args(r) and skill != "ask_user":
                    arg_fixed += 1                       # 成功但参数被纠正 -> 计「已纠正」
                else:
                    n_ok += 1
                continue
            later_ok = any(getattr(x, "ok", False) and getattr(x, "skill", "") == skill
                           for x in results[i + 1:])
            if later_ok and skill != "ask_user":
                recovered += 1
            else:
                failed.append(r)
        nonretry = getattr(AgentLoop, "NON_RETRYABLE_ERRORS", ()) or ()
        if any(getattr(r, "skill", "") == "ask_user" for r in results):
            outcome = "中止（需要你补充信息）"
        elif any(getattr(r, "error_code", None) in nonretry for r in results):
            outcome = "中止（不可重试错误）"
        elif results and not failed:
            outcome = "成功（重试后）" if recovered else "成功"
        else:
            outcome = "失败"
        return {"outcome": outcome, "n_ok": n_ok,
                "n_corrected": recovered + arg_fixed, "n_fail": len(failed)}

    @classmethod
    def _corrected_args(cls, r) -> list:
        """§5.12：validate 把「冗余但一致」的参数纠正后会写 metrics.corrected_args。"""
        m = getattr(r, "metrics", None)
        if not isinstance(m, dict):
            return []
        ca = m.get("corrected_args")
        if isinstance(ca, dict):
            return [str(k) for k in ca]           # 只要给了键就代表发生了纠正（值可为 None=被丢弃）
        if isinstance(ca, (list, tuple)):
            return list(ca)
        return [str(ca)] if ca else []

    @classmethod
    def _classify_results(cls, results: list) -> str:
        """任务结局字符串（向后兼容；语义见 _summarize_results）。"""
        return cls._summarize_results(results)["outcome"]

    @staticmethod
    def _placement_desc(metrics: dict, message: str) -> str:
        """从 return_held_object 的 metrics/message 里取一个人类可读的位置描述。"""
        if isinstance(metrics, dict):
            for key in ("surface", "container_id", "location", "target"):
                if metrics.get(key):
                    return str(metrics[key])
            xy = metrics.get("returned_to_xy")          # skills.return_held_object 的落点
            if xy is not None:
                try:
                    return (f"{metrics.get('mode') or 'safe_spot'} "
                            + str(np.round(np.asarray(xy, dtype=float), 3).tolist()))
                except Exception:  # noqa: BLE001 - 描述失败就退回 message
                    pass
            pos = metrics.get("pos")
            if pos is None:
                pos = metrics.get("position")
            if pos is not None:
                try:
                    return "位置 " + str(np.round(np.asarray(pos, dtype=float), 3).tolist())
                except Exception:  # noqa: BLE001 - 描述失败就退回 message
                    pass
        return message or "安全位置"

    def _return_held_object(self) -> None:
        """任务结束后若夹爪里仍有物体，调用 skills.return_held_object() 放回，避免悬空。"""
        held = getattr(self.skills, "holding", None)
        if held is None:
            return
        if not hasattr(self.skills, "return_held_object"):
            self._log(f"[app] 警告: 物体 {held} 仍在夹爪中：skills 尚未提供 return_held_object，"
                      "物体仍在夹爪中，请手动处理")
            return
        try:
            res = self.skills.return_held_object()
        except Exception as exc:  # noqa: BLE001 - 清理失败也必须让人看见
            self._log(f"[app] 警告: 放回 {held} 失败（{type(exc).__name__}: {exc}），"
                      "物体仍在夹爪中，请手动处理")
            return
        ok = getattr(res, "ok", None)
        message = getattr(res, "message", "") or ""
        metrics = getattr(res, "metrics", None)
        if isinstance(res, dict):                      # 也允许返回 dict
            ok = res.get("ok", ok)
            message = res.get("message", message) or ""
            metrics = res.get("metrics", metrics)
        if not isinstance(metrics, dict):
            metrics = {}
        self._log(f"[app] 任务中止：已把 {held} 放回 {self._placement_desc(metrics, message)}")
        if metrics:
            self._log(f"  return_held_object metrics: {metrics}")
        if ok is False:
            self._log(f"[app] 警告: 放回 {held} 未成功（{message}），请确认是否仍在夹爪中")

    def run_task(self, task: str) -> list:
        self._log(f"\n[app] 任务: {task!r}  决策器: {type(self.vlm).__name__}"
                  f"  |  历史 {len(self.history)} 条")
        self._log(f"[app] 本步由 {self._vlm_name()} 决策"
                  + ("  [无 LLM]" if isinstance(self.vlm, MockVLM) else "")
                  + f"  |  感知: {self.perception_mode}")
        if self._panels_files is not None:
            self._log(self._panels_hint())
        self._last_locate_color = None               # 每条任务重新学习目标颜色（§5.10）
        self._recent_locate = []                     # 每条任务重置 locate 记录（bbox 复用）
        self._task_cameras = self._agent_cameras()   # 本任务随附图像（task-54）
        self._log(f"[app] 随附图像相机: {list(self._task_cameras) if self._task_cameras else '(无图像)'}"
                  + ("  |  pixel_camera 默认: top" if self.perception_mode == "vlm" else ""))
        # 上一条命令失败后机械臂可能卡在 400+ N 接触里，直接跑下一条会连锁失败：
        # 先探测末端接触力，超过 5 N 就做一次应急恢复再进入 AgentLoop。
        force, _torque = self.env.contact_force_on_ee()
        fmag = float(np.linalg.norm(np.asarray(force, dtype=float)))
        if fmag > 5.0:
            self._log(f"[app] 末端接触力 {fmag:.1f} N > 5.0 N，任务前先应急恢复…")
            self._hard_recover()
        t0 = time.time()
        self._last_task = task
        self._subgoal_results = []
        subgoal = self._run_subgoal_plan(task) if self.planner_mode == "subgoal" else None
        if subgoal is None:                          # flat（默认或 taskplan 未就绪）
            results = self._run_agent_loop(task)
            self._log(f"[app] 完成 {len(results)} 步，用时 {time.time() - t0:.1f}s")
            for i, r in enumerate(results, 1):
                self._log(f"  {i:02d} {'OK  ' if r.ok else 'FAIL'} {r.skill:<16} {r.message}")
            summary = self._summarize_results(results)
            outcome = summary["outcome"]
            self._log(f"[app] 结果: {outcome}"
                      f"  |  {len(results)} 步: {summary['n_ok']} OK / "
                      f"{summary['n_corrected']} 已纠正 / {summary['n_fail']} FAIL")
        else:                                        # §5.6 子目标维度汇总
            self._subgoal_results = list(subgoal)
            results = self._subgoal_skill_results()  # 保持返回 list[SkillResult] 的契约
            if not results:
                results = []
            n_tot = len(self._subgoal_results)
            self._log(f"[app] 完成 {n_tot} 个子目标（{len(results)} 步技能），"
                      f"用时 {time.time() - t0:.1f}s")
            n_done = 0
            for item in self._subgoal_results:
                self._log(f"  {self._fmt_subgoal_result(item)}")
                n_done += 1 if (isinstance(item, dict) and item.get("done")) else 0
            retried = any(isinstance(x, dict) and (x.get("attempts") or 1) > 1
                          for x in self._subgoal_results)
            if n_tot and n_done == n_tot:
                outcome = "成功（重试后）" if retried else "成功"
            else:
                outcome = "失败"                     # 子目标没全完成绝不谎报成功
            self._log(f"[app] 结果: {outcome}"
                      f"  |  {n_tot} 个子目标: {n_done} 完成 / {n_tot - n_done} 未完成")
        held = getattr(self.skills, "holding", None)
        if held is not None:
            if outcome in ("成功", "成功（重试后）"):
                # “只抓不放”语义（如「夹住蓝色方块」）：成功结束就保持夹持，不擅自放回
                self._log(f"[app] 任务结束，物体 {held} 仍在夹爪中"
                          "（保持夹持；如需放下请说『把它放到桌上』）")
            else:                                      # 中止/失败才清理，避免悬空
                self._return_held_object()
        if self._panels_files is not None:
            self._panels_files.write_curves(self._telemetry_rows(), force=True)   # 任务结束补一张
        return results

    def dispatch(self, line: str) -> bool:
        """返回 False 表示要退出。"""
        line = line.strip()
        if not line:
            return True
        if not line.startswith(":"):
            self.run_task(line)
            return True

        parts = shlex.split(line)
        cmd, rest = parts[0][1:].lower(), parts[1:]

        if cmd in ("quit", "exit", "q"):
            return False
        if cmd in ("help", "h", "?"):
            print(__doc__)
            return True
        if cmd == "task":
            text = " ".join(rest) or input("  任务描述> ").strip()
            if text:
                self.run_task(text)
            return True
        if cmd == "vlm":
            mode = (rest[0] if rest else "").lower()
            if mode in ("auto", "mock", "api"):
                self._set_vlm(mode)
                self._set_perception(self.perception_request)   # auto 跟随决策器重解析
            elif mode and mode != "status":
                self._log(f"[app] :vlm 只支持 status|auto|mock|api，收到 {mode!r}")
            self._log_vlm_banner()
            self._log_vlm_status()
            self._log("[app] :vlm api 需要环境变量 AI_API_URL/AI_API_KEY/AI_MODEL；示例: "
                      "AI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 "
                      "AI_API_KEY=sk-xxx AI_MODEL=qwen-vl-max-latest python -m demo.interactive --vlm api")
            return True
        if cmd == "objects":
            for o in self.env.list_objects():
                if self.perception_mode == "vlm":    # vlm 模式不得打印物体真值坐标
                    self._log(f"  {o.id:<12} {o.category:<9} {o.color:<7} "
                              "（vlm 感知：不显示真值位姿）")
                else:
                    self._log(f"  {o.id:<12} {o.category:<9} {o.color:<7} "
                              f"pos={np.round(o.pose[:3], 3)} "
                              f"container={o.is_container} graspable={o.graspable}")
            return True
        if cmd == "state":
            st = self.env.get_state()
            self._log("  robot: " + str(st.robot.to_json()))
            self._log("  contact: " + str(st.contact.to_json()))
            self._log(f"  holding: {self.skills.holding}")
            return True
        if cmd == "reset":
            self.env.reset(seed=self.args.seed)
            self.skills.holding = None
            self._log("[app] 场景已重置")
            return True
        if cmd == "recover":
            self._log("[app] 执行应急恢复…")
            self._hard_recover()
            return True
        if cmd == "target":
            self._set_target_mode(rest[0] if rest else None)
            return True
        if cmd == "scene":
            self._set_scene(rest[0] if rest else None)
            return True
        if cmd == "panels":
            arg = (rest[0] if rest else "").lower()
            if arg == "shot":
                self._panels_shot()
            elif arg == "open":
                self._panels_open()
            else:
                self._set_panels(arg or None)
            return True
        if cmd == "plan":
            self._print_plan()
            return True
        if cmd == "verify":
            self._print_verify()
            return True
        if cmd == "perception":
            arg = (rest[0] if rest else "").lower()
            if arg:
                self._set_perception(arg)
            else:
                resolver = "ready" if _pixel_to_table is not None else "missing(sim.vision 未就绪)"
                self._log(f"[app] 感知模式: {self.perception_mode}"
                          f"（请求 {self.perception_request}；"
                          f"include_poses={self.perception_mode != 'vlm'}；pixel_resolver={resolver}）")
            return True
        if cmd == "history":
            n = len(self.history)
            turns = sum(1 for m in self.history
                        if isinstance(m, dict) and str(m.get("role", "")) == "user")
            self._log(f"[app] 会话历史: {n} 条消息（user 角色 {turns} 条），前 3 条摘要:")
            if n == 0:
                self._log("  （空）")
            for i, msg in enumerate(self.history[:3], 1):
                if isinstance(msg, dict):
                    role = str(msg.get("role", "?"))
                    text = " ".join(str(msg.get("content", "")).split())
                else:
                    role, text = "?", " ".join(str(msg).split())
                self._log(f"  {i:02d} {role:<9} {text[:100]}")
            return True
        if cmd in ("clear-history", "clear_history"):
            self.history = []
            self._log("[app] 会话历史已清空")
            return True
        if cmd == "shot":
            cam = rest[0] if rest else "overview"
            try:
                from PIL import Image
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                p = OUTPUT_DIR / f"shot_{cam}_{int(time.time())}.png"
                with self.lock:
                    Image.fromarray(self.env.render(cam)).save(p)
                self._log(f"[app] 已保存 {p}")
            except Exception as exc:  # noqa: BLE001
                self._log(f"[app] 截图失败: {exc}")
            return True

        # --- 直接调技能 ------------------------------------------------ #
        def call(skill: str, args: dict) -> None:
            r = self.skills(SkillCall(skill, args))
            self._log(f"  {'OK  ' if r.ok else 'FAIL'} {r.skill}: {r.message} {r.metrics or ''}")

        if cmd == "home":
            call("home", {})
        elif cmd == "pick" and rest:
            call("pick", {"target_id": rest[0]})
        elif cmd == "place" and rest:
            call("place", {"container_id": rest[0]})
        elif cmd == "goto" and len(rest) >= 3:
            call("move_to", {"pos": [float(v) for v in rest[:3]], "planned": True})
        elif cmd == "push" and len(rest) >= 4:
            args = {"target_id": rest[0], "direction": [float(v) for v in rest[1:4]]}
            if len(rest) >= 5:
                args["distance"] = float(rest[4])
            call("push", args)
        elif cmd == "gripper" and rest:
            v = rest[0].lower()
            if v == "open":
                call("open_gripper", {})
            elif v == "close":
                call("close_gripper", {})
            else:
                self.env.set_gripper(float(v))
                self._log(f"  OK   夹爪 -> {float(v):.2f}")
        elif cmd == "compliance" and rest:
            prof = rest[0]
            args = {"profile": prof}
            if len(rest) > 1:
                args["k_scale"] = float(rest[1])
            call("set_compliance", args)
        else:
            self._log("[app] 未知命令，输入 :help 查看用法")
        return True

    # ------------------------------------------------------------------ #
    def console_loop(self) -> None:
        # 【GL 初始化必须先串行完成，且要在「以后真正用它的那个线程」上】
        # 离屏 Renderer 在 worker(console) 线程创建；这里先渲染一帧并通知主线程，
        # 主线程等到 gl_ready 之后才 launch_passive —— 避免两条线程并发 glfwInit
        # （实测 _glfwGrabErrorHandlerX11 断言 / GLXMakeCurrent BadAccess）。
        if not self.args.headless:
            try:
                self.env.render("top")
                self._log("[app] 离屏渲染器已预热（worker 线程，先于 viewer）")
            except Exception as exc:  # noqa: BLE001
                self._log(f"[app] 离屏渲染器预热失败（{type(exc).__name__}: {exc}），继续")
            finally:
                self._gl_warmed = True
                self._gl_ready.set()
        print(BANNER)
        self._log_vlm_banner()
        self._log(f"[app] 相机: {self.env.camera_names}"
                  f"  |  物体: {[o.id for o in self.env.list_objects()]}")
        if self.args.headless:
            self._log("[app] headless 模式（无 MuJoCo 窗口）")
        self._log("[app] 直接输入自然语言即为任务；:help 查看命令；:quit 退出\n")
        while not self.stop.is_set():
            try:
                line = input("vlm_admittance> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            try:
                if not self.dispatch(line):
                    break
            except Exception as exc:  # noqa: BLE001
                self._log(f"[app] 命令执行异常: {type(exc).__name__}: {exc}")
        self.stop.set()

    def self_test(self) -> int:
        cmds = [":objects", ":state", ":vlm mock",
                "把红色方块放进托盘", ":state", ":shot overview", ":home", ":quit"]
        for c in cmds:
            self._log(f"\n>>> {c}")
            if not self.dispatch(c):
                break
        ok = self.skills.holding is None
        self._log(f"[self-test] {'OK' if ok else 'FAIL'}")
        return 0 if ok else 1

    # ------------------------------------------------------------------ #
    def main(self) -> int:
        if self.args.self_test:
            return self.self_test()

        console = threading.Thread(target=self.console_loop, daemon=True)
        console.start()

        if self.args.headless:
            self.stop.wait()
            return 0
        self._gl_ready.wait(timeout=15.0)     # 等 worker 线程把离屏 Renderer 初始化完

        import mujoco.viewer  # 延迟导入：headless 时不需要 GL 窗口
        self._log("[app] 正在打开 MuJoCo 窗口…（关闭窗口即退出）")
        viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        try:
            while viewer.is_running() and not self.stop.is_set():
                with self.lock:
                    self._update_target_overlay(viewer.user_scn)  # 先画目标点
                    viewer.sync()                                 # 再合入渲染
                    self._sync_panels()                           # 面板同样只在主线程 sync
                time.sleep(0.01)
        finally:
            try:
                viewer.close()
            except Exception:  # noqa: BLE001
                pass
            # 快速启停（启动后 1 s 内退出）时 GLFW 的渲染线程还没完成初始化，
            # 立刻销毁上下文会 segfault（实测 exit 139）。留一点时间让它收尾。
            time.sleep(0.4)
            self.stop.set()
            console.join(timeout=1.0)
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="vlm_admittance 交互式仿真台")
    ap.add_argument("--vlm", choices=["auto", "mock", "api"], default="auto",
                    help="决策器: auto=有 VLM/OpenAI/AI key 用真模型，否则 MockVLM（默认）")
    ap.add_argument("--perception", choices=["auto", "vlm", "oracle"], default="auto",
                    help="感知模式: auto=真模型->vlm(只给图像+像素瞄准)、MockVLM->oracle(给真值位姿)")
    ap.add_argument("--panels", choices=["none", "files", "all"], default=None,
                    help="面板产出（纯文件，默认 all）: files/all=每次请求前写 llm_frames.png、"
                         "<=1Hz 写 curves.png（headless 也照写，不建任何窗口/子进程）")
    ap.add_argument("--scene", choices=["default", "daily"], default="default",
                    help="场景: default=方块+托盘、daily=日常物品(水瓶/眼镜盒，§5.5)")
    ap.add_argument("--planner", choices=["subgoal", "flat"], default=None,
                    help="规划: 默认 vlm 模式=subgoal(子目标闭环§5.2)、oracle=flat；"
                         "显式 --planner flat 可回退单任务")
    ap.add_argument("--with-barrier", action="store_true")
    ap.add_argument("--with-images", action="store_true", default=True,
                    help="把相机图发给 VLM（默认开）")
    ap.add_argument("--no-images", dest="with_images", action="store_false")
    ap.add_argument("--cameras", default=None,
                    help="显式指定发给模型的相机（逗号分隔，如 overview,top）；"
                         "默认 vlm=overview,top（top 为像素瞄准主力）、oracle=overview")
    ap.add_argument("--realtime", action="store_true", help="按真实时间配速（默认全速）")
    ap.add_argument("--headless", action="store_true", help="不开 MuJoCo 窗口（终端命令台）")
    ap.add_argument("--self-test", action="store_true", help="非交互自检")
    ap.add_argument("--max-steps", type=int, default=AGENT_MAX_STEPS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    return InteractiveApp(args).main()


if __name__ == "__main__":
    sys.exit(main())
