# INTERFACES.md —— 模块契约（冻结版 v1）

> 本文件是**唯一**的接口约定。任何模块不得擅自更改签名；需要变更时先在共享任务里说明。
> 所有模块通过 `core/` 的类型与 Protocol 交互，**不要**私自定义重复的数据结构。

## 0. 环境事实（已实测，别重复踩坑）

| 项 | 值 |
|---|---|
| Python | `/home/tc63/mujoco/.venv/bin/python`（3.14.4） |
| 工程根 | `/home/tc63/mujoco/vlm_admittance`（所有命令在此目录下执行） |
| MuJoCo | 3.12.0（`mujoco.MjSpec` 可用） |
| 已装依赖 | numpy 2.5.2, scipy 1.18.1, pillow 12.3.0, gymnasium 1.3.0, matplotlib, imageio, trimesh, glfw, PyOpenGL |
| **未装** | mink、ompl、openai（venv 内）→ **不要引入这些依赖** |
| 离屏渲染 | **`MUJOCO_GL=glfw` 可用**（需要 DISPLAY=:0，已具备）；**EGL 在本机报错，不要用** |
| 测试框架 | 用标准库 `unittest`；**不要依赖 pytest** |

运行任何脚本前先 `cd /home/tc63/mujoco/vlm_admittance` 并设 `MUJOCO_GL=glfw`，例如：

```bash
cd /home/tc63/mujoco/vlm_admittance && MUJOCO_GL=glfw /home/tc63/mujoco/.venv/bin/python -m sim.smoke
```

导入方式：`from core.types import SceneState, ComplianceParams, ...`、`from core.config import SIM_DT, ...`（工程根即包根）。

## 1. 机器人模型（已实测的组合方式）

`assets/ur5e/ur5e.xml`（Menagerie，BSD-3）与 `assets/robotiq_2f85/2f85.xml`（Menagerie）通过 **MjSpec.attach** 组合：

```python
import mujoco
from core.config import UR5E_XML, GRIPPER_XML

arm = mujoco.MjSpec.from_file(str(UR5E_XML))
grip = mujoco.MjSpec.from_file(str(GRIPPER_XML))
arm.attach(grip, prefix="g_", site=arm.site("attachment_site"))   # ✅ 已实测可用
m = arm.compile()
# 结果：nq=14, nv=14, nu=7, neq=3
```

编译后：

| 名称 | 值 |
|---|---|
| 机械臂关节 | `shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint`（qpos 下标 0..5） |
| 机械臂执行器 | `shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3`（ctrl 下标 0..5） |
| 夹爪执行器 | `g_fingers_actuator`（ctrl 下标 6，范围 0..255，**闭合方向需实测确认并写进注释**） |
| 夹爪 TCP site | `g_pinch` |
| 法兰 site | `attachment_site`（pos 相对法兰 `0 0.1 0`） |
| 夹爪 body | `g_base_mount` / `g_base` / `g_right_*` / `g_left_*` |

> 注意：attach 时会有 `impratio` / `cone` 冲突警告，可在 scene spec 里设 `spec.option.impratio=10; spec.option.cone=elliptic` 消除。
> 夹爪有 8 个内部自由度 + 3 个 equality 约束；**IK / 规划 / 碰撞只处理 6 个机械臂关节**，夹爪自由度保持不动即可。

## 2. 模块与签名

### 2.1 `sim/`（负责人：sim-builder）
```python
# sim/scene.py
DEFAULT_OBJECTS: list[dict]          # id/category/color/size/pos，含 3 个彩色方块 + 1 个圆柱 + 1 个托盘(container)
def build_spec(objects=None, with_barrier: bool = False) -> mujoco.MjSpec
def write_scene_xml(path) -> str

# sim/env.py
class MuJoCoArmEnv:                  # 满足 core.interfaces.SimEnv
    def __init__(self, objects=None, with_barrier=False, dt=SIM_DT, render_size=(512,512), seed=0)
    # 必须实现（签名见 core/interfaces.py 的 SimEnv）:
    #   reset/step/get_state/render/render_depth/set_arm_joint_targets/set_gripper/get_gripper
    #   contact_force_on_ee/site_pose/body_pose/object_pose/set_object_pose/list_objects/home
    # 属性: model, data, dt, arm_qpos_idx(6,), arm_dof_idx(6,), camera_names

# sim/render.py
class SceneRenderer:                 # 懒加载 mujoco.Renderer；提供 rgb/depth
    def __init__(self, model, width=512, height=512)
    def rgb(self, data, camera) -> np.ndarray
    def depth(self, data, camera) -> np.ndarray
    def close(self) -> None
```
要求：
- 相机名固定为 `core.config.CAMERAS = ("overview", "wrist", "top")`；`wrist` 相机需挂在腕部（gripper base 附近）。
- 物体用 free joint（可被夹取）；托盘 `is_container=True`，`graspable=False`。
- 提供 `sim/smoke.py`（`python -m sim.smoke`）：打印 nq/nu/相机名 → 存 3 张 PNG 到 `outputs/` → 打印无接触时的力读数 → 设定一组关节角并 step → 打印 TCP 位姿。
- `set_gripper(open_ratio)`：0=全闭、1=全开，**实测映射并写注释**。

### 2.2 `control/`（负责人：ctrl-builder）
```python
# control/kinematics.py
class Kinematics:
    def __init__(self, env, tcp_site: str = TCP_SITE)
    def fk(self, q=None) -> tuple[np.ndarray, np.ndarray]        # TCP pos(3), quat wxyz(4)
    def jacobian(self, q=None) -> np.ndarray                      # (6, nv) 世界系 TCP site 雅可比
    def ik(self, target_pos, target_quat=None, q0=None, max_iters=300, tol=1e-3) -> tuple[np.ndarray, bool]
    def clamp(self, q) -> np.ndarray

# control/trajectory.py
def min_jerk(s: float) -> float
def slerp(q0, q1, s) -> np.ndarray
def quat_from_rotvec(rv), rotvec_from_quat(q), quat_mul(a, b)
class CartesianTrajectory:
    def __init__(self, p0, p1, duration, dt, rot0=None, rot1=None)
    duration: float
    def sample(self, t) -> tuple[np.ndarray, np.ndarray]          # pos(3), quat wxyz(4)

# control/admittance.py
class AdmittanceController:
    def __init__(self, params: ComplianceParams, dt: float)
    def reset(self, x_ref: np.ndarray) -> None                    # x_ref: (6,) pos + rotvec
    def update(self, x_ref: np.ndarray, wrench: np.ndarray) -> np.ndarray   # -> x_cmd (6,)
    def set_params(self, params) -> None
    def state(self) -> dict

# control/motion.py
class MotionRunner:                                               # 满足 MotionProtocol
    def __init__(self, env, kin, admittance=None)
    def move_to_pose(self, pos, quat=None, duration=None, compliance=None,
                     timeout=8.0, force_limit=None) -> MoveResult
    def follow(self, traj, compliance=None, timeout=8.0, force_limit=None) -> MoveResult
    def follow_joint_path(self, path, max_step_rad=0.05, timeout=20.0, compliance=None) -> MoveResult
    def hold(self, duration: float) -> MoveResult
```
要求（导纳环关键点，务必写进 docstring）：
1. `M·ẍ + D·ẋ + K·(x − x_ref) = −f_ext` 的**半隐式欧拉**离散，dt 用 `env.dt`；`axes[i]==0` 的轴不做修正。
2. 单步修正量限幅（平动 ≤ 0.05 m、转动 ≤ 0.15 rad），防止接触瞬间弹飞。
3. `move_to_pose`：直线轨迹 + 每步导纳修正 + 差分 IK → `env.set_arm_joint_targets` → `env.step()`；超时/力超限/IK 不可达时返回 `MoveResult(ok=False, error_code=...)`，error_code 用 `"timeout" | "force_limit" | "ik_fail" | "unreachable"`。
4. `follow_joint_path` 按关节空间插值执行（供规划器路径使用），不做笛卡尔导纳。
5. `fk/jacobian` 必须使用**私有的临时 MjData**（`mujoco.MjData(env.model)`）计算，绝不能改坏 `env.data`。
- `control/smoke.py`：IK 往返误差、直线跟踪误差（< 5 mm）、导纳单元测试（给定合成力看修正方向与稳定性）。

### 2.3 `planning/`（负责人：plan-builder）
```python
# planning/collision.py
class CollisionChecker:
    def __init__(self, env, ignore_pairs: set[tuple[str,str]] | None = None)
    def is_state_valid(self, q: np.ndarray) -> bool       # 关节限位 + 无碰撞
    def is_edge_valid(self, q1, q2, steps: int = 12) -> bool

# planning/rrt_connect.py
class RRTConnect:
    def __init__(self, checker, q_min, q_max, step_size=0.15, max_iters=4000, seed=0, goal_bias=0.1)
    def plan(self, q_start, q_goal) -> list[np.ndarray] | None
def shortcut(path, checker, iters=200, seed=0) -> list[np.ndarray]
def densify(path, max_step=0.05) -> list[np.ndarray]

# planning/planner.py
class JointPlanner:                                        # 满足 PlannerProtocol
    def __init__(self, env, kin, checker=None)
    def plan_joint(self, q_goal) -> list[np.ndarray] | None
    def plan_pose(self, pos, quat=None, n_seeds=20) -> list[np.ndarray] | None
```
要求：
- 碰撞检测用**私有临时 MjData**（设置 6 个臂关节 qpos → `mj_forward` → 看 `data.ncon` / `mj_contactForce`），不能污染 `env.data`。
- 允许忽略的接触对：夹爪 finger 之间、夹爪与自身；被规划场景中应忽略"夹爪碰到桌面上的目标物体"吗？**不忽略**——规划时以"不碰任何东西"为默认；抓取贴近阶段由 `control` 的笛卡尔运动负责。
- 提供 `planning/smoke.py`：从 HOME_QPOS 规划到另一组可行关节角，路径经 `shortcut`，断言端点误差与路径有效性；再用 `with_barrier=True` 场景验证能绕障。

### 2.4 `agent/`（负责人：agent-builder）
```python
# agent/state.py
def serialize_state(state: SceneState, task: str, extra: dict | None = None) -> dict
def encode_image(img: np.ndarray, fmt: str = "jpeg", quality: int = 85) -> str   # base64
def build_messages(task: str, state: SceneState, images: dict[str, np.ndarray],
                   history: list[dict] | None = None) -> list[dict]

# agent/schema.py
TOOLS: list[dict]                       # OpenAI function-calling schema
def system_prompt() -> str
def parse_skill_call(raw: str | dict) -> SkillCall        # 非法输入 raise ValueError
def validate_skill_call(call: SkillCall, state: SceneState) -> tuple[bool, str]

# agent/vlm_client.py
class VLMClient:                        # 标准库 urllib 实现，OpenAI 兼容 /chat/completions
    def __init__(self, model=None, base_url=None, api_key=None, timeout=None)
    available: bool
    def decide(self, messages: list[dict]) -> SkillCall
class MockVLM:                          # 规则式，离线可跑，同 decide 接口
    def decide(self, messages: list[dict]) -> SkillCall
def make_vlm(prefer_api: bool = True)

# agent/loop.py
class AgentLoop:
    def __init__(self, env, executor: SkillExecutor, vlm, cameras=("overview",),
                 max_steps=AGENT_MAX_STEPS, log=print)
    def run(self, task: str) -> list[SkillResult]
```
要求：
- `TOOLS` 覆盖技能：`home, move_to, pick, place, push, open_gripper, close_gripper, set_compliance, stop, ask_user`，参数与 `skills/` 对齐（见 §2.5）。
- `MockVLM.decide` 必须能根据 messages 里的场景 JSON 做出**确定性**决策：解析颜色/容器关键词 → `pick(target_id)` → `place(container_id)`，接触时降低刚度（`set_compliance`）。它用于 CI 与无 key 演示。
- `VLMClient` 只依赖标准库（`urllib.request`、`json`、`base64`）；请求体含 `tools` 与 `tool_choice="auto"`，先解析 `tool_calls`，失败则从 `content` 里正则抽取 JSON。
- `encode_image` 优先用 PIL 存 JPEG（内存 BytesIO），PIL 不可用时退化为 PNG。
- `agent/smoke.py`：合成 `SceneState` → `serialize_state`/`build_messages` → `MockVLM.decide` → `validate_skill_call`；断言非法调用被拒；断言 `encode_image` 输出可 base64 解码。

### 2.5 `skills/`（负责人：Lead）
```python
class SkillExecutor:
    def __init__(self, env, kin, runner, planner, cfg=None)
    def __call__(self, call: SkillCall) -> SkillResult
# 技能与参数（TOOLS 必须与此一致）：
#   home()
#   move_to(pos:[x,y,z], quat:[w,x,y,z]|null, compliance:str|object|null)
#   pick(target_id:str, approach_height?:float, close_ratio?:float, compliance?:object)
#   place(container_id:str, offset?:[dx,dy,dz], compliance?:object)
#   push(target_id:str, direction:[dx,dy,dz], distance:float)
#   open_gripper(); close_gripper(ratio?:float)
#   set_compliance(profile:"stiff"|"soft"|"z_only", k_scale?:float)
#   stop(); ask_user(question:str)
# compliance 为对象时形如 {"k":[6], "d":[6], "m":[6], "axes":[6], "force_limit":float}
```

## 3. 协作规则

> **【2026-10-12 硬性规则】** 任何**自动化**运行（测试、验证、探针、验收脚本）启动 `demo.interactive` **必须加 `--headless`**：
> 不带它会给用户的桌面弹出一个 MuJoCo 窗口，跑几十次就是几十个（崩溃后还留黑窗）。
> 面板现在是**纯文件产出**（`outputs/panels/*.png` + `index.html`），headless 下同样产出，所以自动化没有任何理由开窗。
> 需要看窗口的只有用户自己。

1. **只写自己 scope 内的文件**（见共享任务的 write_scopes）。`core/` 由 Lead 维护，发现契约问题先报告。
2. 每个模块自带 `smoke.py`，必须能在无 GUI 环境下通过（`MUJOCO_GL=glfw`）。
3. 不新增第三方依赖（只能用已装的 mujoco/numpy/scipy/pillow/gymnasium/matplotlib）。
4. 写完在共享任务里 `complete`，并回报：创建/修改的文件、smoke 命令与结果、遗留问题。
5. 需要别的模块尚未存在的代码时，**按契约写代码**并在自己的测试里造一个最小 stub，不要越界去实现别人的模块。

## 4. VLM 感知闭环（pixel → world）—— 冻结契约（Lead 2025 增补）

动机：真实场景拿不到目标物体真值坐标，只有相机图像。新增「像素瞄准」通道：大模型输出目标中心在指定相机
图像里的像素 (u,v)，由 `sim/vision.py` 反投影到桌面平面得到世界坐标；技能层只接收世界 xy，**不接收真值位姿**。

### 4.1 `sim/vision.py`（负责人：sim-builder）
```python
class CameraModel:
    def __init__(self, env, camera: str = "top")
    #: 属性: name, width, height, fovy(deg), cam_pos(world 3), cam_mat(world 3x3: 列 = x右, y上, z后)
    def project(self, xyz) -> tuple[float, float] | None          # world -> (u,v)；相机后方返回 None
    def ray(self, u, v) -> tuple[np.ndarray, np.ndarray]          # -> (origin, 单位方向)，world 系
    def pixel_to_plane(self, u, v, z: float) -> np.ndarray | None # 与水平面 z 求交 -> world xyz；无交点 None
def camera_model(env, camera="top") -> CameraModel
def pixel_to_table(env, u, v, camera="top", z=TABLE_HEIGHT) -> np.ndarray | None
def project_point(env, xyz, camera="top") -> tuple[float,float] | None
```
约束：u 向右、v 向下，原点在图像左上角；像素允许越界（模型会数错），实现不得崩溃——越界时
`project` 返回 None，`pixel_to_plane` 仍按针孔模型求交（可能落在图像外，这是允许的），两者都要写进 docstring。
分辨率与 `sim/render.py` 一致（512x512）；标定必须来自 `model.cam_fovy` + `mj_forward` 的相机位姿，禁止硬编码相机位置。
smoke（允许 MUJOCO_GL=glfw）：(1) 桌面已知点 project -> pixel -> pixel_to_table 往返误差 < 1 mm；
(2) `top` 与 `overview` 两个相机都测；(3) 反投影 z 与给定平面一致（< 1e-6）。

### 4.2 `skills/`（负责人：skill-fixer）
```python
# pick 新增「无真值位姿」入口（与 target_id 互斥）：
pick(target_xy: [x, y],                # 世界系桌面坐标（上层把像素反投影得到）
     target_z_hint: float | None,      # 物体中心 z 估计；None -> 桌面 z + 默认半高
     target_size: [sx, sy, sz] | None, # 可选尺寸估计；缺失用默认
     close_ratio?, compliance?)
```
- R-a：`target_xy` 路径的目标点必须来自入参，**不得**读 `scene.objects[...].pose` 当真值（真值只允许出现在诊断 metrics 里）。
- R-b：无真值模式下夹取成功判据改为**夹爪实测**：闭合后抬升阶段读指关节实际间距（用 `GRIPPER_GAP_*` 标定换算 mm），
  实测间距 > 全闭间距 + 阈值（指间有物）即判定抓住；metrics 报 `grasp_gap_mm` / `grasp_evidence="jaw_gap"`。
  带 `target_id` 的旧路径判据不变（必须回归通过）。
- R-c：`_obstruction()` 在无真值模式下用接触对反查物体 id（sim 接触信息可用），拿不到给 None。

### 4.3 `agent/`（负责人：agent-fixer）
- `TOOLS`：`pick` 增加 `target_pixel: [u,v]`、`pixel_camera: "top"|"overview"|"wrist"`（默认 "top"）、`target_z_hint?`；
  `place` 增加 `target_pixel`/`pixel_camera`。`target_pixel` 与 `target_id`/`target_xy` 三选一（互斥校验要给出可读错误）。
- `serialize_state(state, include_poses: bool = True)`：False 时**删除** `objects[*].pose/size` 与全部
  `relations`/支撑字段（等于真值），只留 `{id, color, category}`；并加 `camera_calibration`
  `{camera, width, height, fovy, table_z}`。`build_messages(..., include_poses=True)` 透传。
- no-oracle 的系统提示追加：不会提供物体真实坐标；要抓/放某物体时输出它的中心像素 (u,v)（左上角原点，u 右 v 下）；
  不确定时用 `top` 相机并给估计值。
- `AgentLoop.__init__(..., pixel_resolver=None)`：`pixel_resolver(u, v, camera) -> np.ndarray|None`。
  收到 `target_pixel` 时解析为世界 xy 再以 `target_xy` 调技能；失败返回可重试的 `perception_error`，
  并把原始像素写进 message/metrics 供 LLM 纠错；日志打印「像素 (u,v) [camera] -> 世界 (x,y,z)」。
- MockVLM 路径保持 oracle 不变（不要求它输出像素）。

### 4.4 `demo/`（负责人：ui-fixer）
- 新开关 `--perception auto|vlm|oracle`（默认 auto：真模型 -> vlm，MockVLM -> oracle）与运行时 `:perception`。
- vlm 模式：`include_poses=False`；接 `pixel_resolver = lambda u,v,cam: sim.vision.pixel_to_table(env,u,v,cam)`；
  目标点叠加里画出模型像素反投影出的世界点（label 带 "VLM"）；每步日志显示感知模式；vlm 模式不得打印物体真值坐标。
- oracle 模式维持现状。

### 4.5 验证（负责人：verifier，只测这条新链路，禁止跑全量套件）
(1) 无真值断言：agent 消息与 state JSON 中不存在物体真值位姿（结构键 + 数值双重扫描）；
(2) 标定精度：>=5 个桌面点在 `top`/`overview` 下的往返误差 (mm)；
(3) 一次实机端到端：`--perception vlm` + 真模型，仅凭像素完成一次 pick+place，记录像素误差与成功率；
(4) 反例：像素在桌外/相机后方时不得崩溃，须返回可重试的 `perception_error`。

---

## 5. 纯视觉稳定化 v2（Lead 冻结 2026-10-12）

目标：`--perception vlm`（无任何物体真值）下稳定完成「红色方块放进托盘」「绿色方块叠到红色方块上」，
并可扩展到日常物品（水瓶 / 眼镜盒）。

### 5.0 环境事实（本轮实测，别重复踩）
- **网络当前不通**：`curl` 经 `http://127.0.0.1:7897` 代理一律 `TLS unexpected eof` / 超时（`example.com` 也一样），
  裸连 DNS 解析失败。因此**本轮所有自动化测试一律不得依赖真模型 API**，用 MockVLM 或本契约 5.8 的
  `PixelScriptVLM`（脚本化像素决策器）。真模型实机验收推迟到代理恢复后由 Lead 统一跑。
- `VLM_TIMEOUT` 默认 30 s 偏短（带 2 张图 + 长 prompt 时实测会被掐断）；Lead 已调到 120 s（core/config.py）。
- **相机标定实测（Lead 2026-10-12，纠正旧文档的 0.65 mm/px 说法）**：`top` 512×512、fovy=45°，
  桌面尺度 **1.489 mm/px（x/y 两轴一致）**，视场覆盖桌面 x∈[-0.02, 0.741]、y∈[-0.331, 0.43]（默认 5 个物体全在框内）；
  `overview` 为斜视：du=2.895 mm/px、dv=6.882 mm/px，不要用它做精定位。
  ⇒ **模型给 `top` 像素时，1 px 误差 = 1.489 mm**；40 mm 方块宽 27 px、30 mm 圆柱宽 20 px，
  所以「颜色质心精修」是必需项，而提高 `top` 分辨率（512→1024）可直接把同样像素误差的世界误差砍半。
  渲染成本很低，是否提分辨率由 scene 侧给数据后决定；默认 512 的行为不要改（既有测试依赖）。
- 无 LLM 时 `perception auto` 会退回 oracle；`--perception vlm` + MockVLM 也走无真值链路（用于离线测试）。

### 5.1 新模块 `sim/perception.py`（负责人：vision-builder）—— 纯图像，禁止读真值
只允许输入：渲染出的 RGB、`sim/vision.py` 的相机标定、VLM 给的像素/框。
**禁止**调用 `env.object_pose` / `env.get_state().objects[*].pose`；只允许 `env.get_state().robots` 之外什么都不用。

```python
# 颜色表（HSV 判定用）；scene.py 的 COLOR_RGBA 是渲染真值，这里只做粗分类
# "red" "blue" "green" "yellow" "gray" "black" "white" "cyan"(水瓶) "orange"
@dataclass
class BlobResult:
    found: bool
    uv: tuple[float, float]          # 质心像素
    bbox: tuple[int, int, int, int]  # u1, v1, u2, v2
    area_px: int
    color: str
    score: float                     # 0..1 颜色纯度 * 面积占比

def color_blob(rgb, color, hint_uv=None, *, radius_px=110, min_area_px=25) -> BlobResult
    # HSV 阈值 + 连通域（4 邻域，纯 numpy，标签用 BFS/并查集；不得依赖 cv2/PIL）
    # hint_uv 给定：只在该像素 radius_px 内的候选里选“离 hint 最近且面积最大”者；
    # 找不到返回 found=False（绝不抛异常）。

def refine_pixel(rgb, color, hint_uv, *, camera="top", env=None) -> dict
    # VLM 粗像素 -> 颜色质心精修。
    # -> {"uv": (u,v), "bbox": (...), "refined": bool, "source": "color_blob"|"vlm_raw",
    #     "delta_px": float, "metrics": {...}}
    # refined=False 时 uv 必须原样等于 hint_uv（调用方可安全回退）。

def zoom_window(uv, *, size_px=160, scale=4) -> dict          # {"u0","v0","size","scale","uv_zoom"}
def crop_upscale(rgb, win) -> np.ndarray                        # 双线性放大，越界用边缘填充
def map_zoom_uv(uv_zoom, win) -> tuple[float, float]            # 放大图坐标 -> 原图坐标

def bbox_to_world(env, bbox, *, camera="top", z=TABLE_HEIGHT) -> dict
    # 框四角经 sim.vision.ray 与 z 平面求交 -> 世界 AABB
    # -> {"ok": bool, "center_xy": (x,y), "size_xy": (sx,sy), "corner_xy": [...]}
def estimate_size(env, bbox, *, camera="top") -> dict
    # -> {"ok","half_xy":(hx,hy), "long_side": float, "short_side": float}

def point_in_bbox(uv, bbox, *, shrink_px=0.0) -> bool
def overlap_ratio(b1, b2) -> float                              # 交集 / min(面积)

def check_inside(rgb, subject_color, container_bbox, *, min_area_px=25, shrink_px=None) -> dict
    # 图像空间包含判定：subject 色块质心在 container 框内（内缩 = 容器壁像素宽，缺省 0.06*短边）
    # -> {"ok": bool, "why": str, "metrics": {subject_uv, container_bbox, margin_px}}
def check_stacked(rgb, upper_color, lower_color, *, min_overlap=0.55, min_area_px=25) -> dict
    # 叠放判定：上层色块质心落于下层框内 且 overlap_ratio >= min_overlap
    # -> {"ok","why","metrics"}
def check_gone(rgb_before, rgb_after, color, hint_uv, *, radius_px=60, min_area_px=25) -> dict
    # 抓走后：原位置附近该颜色块消失或面积掉到 1/3 以下
    # -> {"ok","why","metrics"}
```

### 5.2 新模块 `agent/taskplan.py`（负责人：agent-builder）—— 子目标规划 + 独立复核
```python
@dataclass
class SubGoal:
    id: str                      # "g1"
    instruction: str             # 直接喂给 AgentLoop 的自然语言子任务
    goal_type: str               # place_in_container | place_on_top | place_at | pick | free
    subject: str                 # "红色方块"
    subject_color: str | None
    container: str | None        # "托盘"
    container_color: str | None
    anchor_color: str | None     # 叠放/横向锚点颜色
    verify_hint: str             # 给人看的验收说明

def plan_subgoals(vlm, task, state, images, *, max_subgoals=4, log=None) -> list[SubGoal]
    # 1) 先尝试让 VLM 调 plan_subgoals 工具（schema 见 5.3），解析成功即用；
    # 2) 无 LLM / 解析失败 / 空列表 -> fallback_subgoals(task)（单条，instruction=原任务）。
def fallback_subgoals(task) -> list[SubGoal]

SUBGOAL_MAX_ATTEMPTS = 3

class SubGoalVerifier:
    def __init__(self, env, *, perception=None, vlm=None, renderer=None, log=None)
    def locate(self, sg_or_text, *, color=None, camera="top") -> dict   # 经 VLM 的 locate 工具，带缓存
    def verify(self, sg, *, metrics=None, rgb_before=None) -> dict
        # 判定顺序（冻结）：
        #  a) place_in_container -> check_inside(rgb, sg.subject_color, container_bbox)
        #  b) place_on_top       -> check_stacked(rgb, sg.subject_color, sg.anchor_color)
        #  c) pick               -> 技能 metrics 的 grasp_gap_mm/holding 优先；缺失则 check_gone
        #  d) place_at / free   -> 直接问 VLM 复核（带 top 图），confidence>=0.6 才算 done
        # 几何 ok -> {"done": True, "evidence": "geometry", ...}
        # 几何 不确定/失败 -> VLM 复核 -> {"done":bool,"evidence":"vlm","confidence":c}
        # 两者都判否 -> {"done": False, "evidence": "none", "why": "..."}
        # 任何异常都必须转成 done=False，绝不上抛、绝不谎报成功。
```

### 5.3 `agent/schema.py`（负责人：agent-builder）
新增工具（TOOLS 必须与实现一致）：
- `plan_subgoals(subgoals:[{instruction, subject, goal_type, subject_color?, container?, container_color?, anchor_color?}])`
  —— 规划阶段专用；只允许在「规划调用」里出现，执行循环里出现要判 `invalid_call`。
- `locate(target:str, camera:"top"|"overview"=("top"), color?:str)`
  —— 语义标定：返回 `{"pixel":[u,v],"bbox":[u1,v1,u2,v2]}`；由循环记录，供 pick/place 复用（用户要求的「标定目标点 a」）。
- `subgoal_done(note:str)` —— 模型自评当前子目标完成；系统仍独立复核（5.2）。
- `pick` 增加 `grasp_profile: "auto"|"top"|"side"`（默认 auto）。
- `place` 增加 `descend_until_contact: bool|null`（null=自动：`on_top_of` 与容器内为 true）。
- no-oracle 系统提示增加「工作流程」段：① 先 `locate` 标定目标像素 ② 再 `pick`/`place` ③ 完成后 `subgoal_done`；
  并给出日常物品识别要点（水瓶=细长半透明圆柱带盖、眼镜盒=扁长方形硬壳）与「拿不准就再 locate 一次」的重试建议。
- `AgentLoop.__init__(..., locator=None)`：执行循环里收到 `locate` 调用时**不得当技能执行**，而是调
  `locator(target, color, camera) -> dict`，把结果记成 `SkillResult(skill="locate", ok=..., message=..., metrics={"pixel":[u,v],"bbox":[...]})`
  作为观测回灌；`locator=None` 时按可重试的 `perception_error` 处理。`subgoal_done` 记成 `ok=True` 结果并在日志标注。
- `taskplan.SubGoalOrchestrator(env, vlm, loop_factory, verifier, log)`：逐子目标跑 AgentLoop（每个子目标一个新 epoch），
  跑完调 `verifier.verify()`；未通过则带上「上次失败原因 + 请再试一次」重跑同一子目标，最多 `SUBGOAL_MAX_ATTEMPTS` 次；
  返回 `[{"subgoal":..., "attempts":int, "done":bool, "evidence":str, "why":str}]`。demo 在 `--planner subgoal` 下用它。

### 5.4 `skills/library.py`（负责人：skill-builder）
- 新增接触式落放：`place(..., descend_until_contact=True)` —— 到 xy 上方 60 mm 后按 1.5 mm 步长下降，
  接触判据 `contact_force_norm >= 2.5 N`（用现有导纳档位 z_only/soft）；接触即松爪并返回
  `metrics.touchdown_z / contact_force_norm`。超时（下降 > 120 mm）返回**可重试** `place_contact_miss`。
- 容器内落放：落点像素/xy 在容器内时，先用 5.1 的 bbox 把放置点内缩；撞壁（力超阈值且未到位）返回可重试
  `place_blocked`，metrics 带 `obstruction_id=容器 id` 与「换落点像素」提示。
- `pick(grasp_profile="side")`：从侧面水平接近并夹持（圆柱/瓶身；夹持高度取 `target_z_hint`）；
  `close_ratio` 由 `target_size` 推（已有 `_close_ratio_for_size`），未给 size 时按类别先验（方块 0.35 / 圆柱 0.55 / 瓶 0.75）。
- 保持 oracle 路径与既有 metrics/错误码语义不变；`PICK_MAX_ATTEMPTS` 等既有策略不改。

### 5.5 `sim/scene.py`（负责人：scene-builder）
- `COLOR_RGBA` 增 `"cyan"`=`(0.25,0.68,0.88,1.0)`、`"black"`=`(0.10,0.10,0.12,1.0)`。
- 新增 `DAILY_OBJECTS`（**不动** `DEFAULT_OBJECTS` 与 `OBJECT_IDS`）：
  `bottle_water`(cylinder, cyan, size (0.032,0.032,0.100) → 即 半径 32 mm、高 200 mm)、
  `case_black`(box, black, size (0.068,0.026,0.012))、`box_red`/`box_green`(同默认 20 mm 方块)、`tray`。
  `DAILY_OBJECT_IDS = ("box_red","box_green","bottle_water","case_black","tray")`。
  布局：方块 x≈0.35、瓶 x≈0.45 y≈-0.18、盒 x≈0.35 y≈0.20、托盘同默认；物体间距 ≥90 mm、互不重叠。
- 允许为 `top` 相机试 768×768（SceneRenderer 支持自定义 width/height）；**必须给出前后误差数据**，
  数据不显著就回退 512。

### 5.6 `demo/interactive.py`（负责人：Lead）
- `--scene default|daily` 与 `:scene default|daily`；`--planner subgoal|flat`（默认 subgoal）。
- vlm 模式像素链路：VLM 像素 → `perception.refine_pixel` → `sim.vision.pixel_to_table`；
  日志打印 `[感知] VLM 像素 (u,v)[cam] → 颜色精修 (u',v') Δ=…px → 世界 (x,y,z)`；精修失败回退原始像素。
- `:plan` 打印子目标表；`:verify` 打印每个子目标的复核结论与依据（geometry/vlm）。
- 结果汇总增加子目标维度：`g1 完成（几何：红块质心在托盘框内）` / `g2 未完成（3 次尝试）`。

### 5.7 信息隔离守则（强制）
- 运行路径（`AgentLoop` / `taskplan` / `perception` / `skills`）在 vlm 模式下**禁止**读物体真值
  （`env.object_pose` / `set_object_pose` / `state.objects[*].pose`）。
- 真值只允许出现在：oracle 模式、`tests/`、调试脚本 `tools/vision_accuracy.py`（Lead/verifier 维护）。
- verifier 必须做「毒丸测试」：monkeypatch `MuJoCoArmEnv.object_pose` 与 `set_object_pose` 抛异常，
  再跑 vlm 闭环（PixelScriptVLM），必须成功。

### 5.8 验证（负责人：verifier；只测本轮链路，禁止跑全量套件）
1. `sim/perception` 单元：合成图（已知圆形/矩形色块）→ `color_blob`/`refine_pixel` 质心误差 ≤2 px；
   `bbox_to_world` 用已知标定回算，尺寸误差 ≤3 mm；越界/单色/噪声图不崩。
2. 毒丸隔离测试（见 5.7）。
3. 子目标闭环：用 `PixelScriptVLM` 造「g1 前两次失败、第三次成功」→ 最终必须报「成功（重试后）」；
   造「三次全失败」→ 必须报 `subgoal_failed`，不得谎报成功。
4. 接触式落放：sim 中把方块叠到另一方块上，真值仅用于断言，顶部高度差 ≤5 mm；容器内落放成功率 ≥2/3（3 个不同起点）。
5. 日常物品：daily 场景下 `pick` 水瓶/眼镜盒，`close_ratio` 与夹爪实际开口吻合（jaw_gap 与命令差 ≤3 mm）。
6. 端到端（**网络恢复后**由 Lead 跑）：真模型 vlm 模式执行
   「把红色方块放到盘子里，绿色方块放到红方块上」，逐子目标记录结果与用时；网络仍不通则跳过并标注。

### 5.10 双目三角定位（Lead 增补 2026-10-12）—— 解决俯视视差
实测（ctrl-builder，daily 场景）：物体**顶面**像素按 `z=TABLE_HEIGHT` 反投影会有系统性视差：
40 mm 方块 6~9 mm，**水瓶 68.6 mm**（比瓶半径 32 mm 还大），且与分辨率无关。根因是 `top` 相机并非严格正交俯视。
所以对高物体不能只靠单目 + 桌面平面假设。

新增（`sim/perception.py`，vision-builder）：
```python
def triangulate(env, uv_a, cam_a, uv_b, cam_b) -> dict
    # 两条射线最近点；{"ok","xyz","residual_mm","why"}；residual_mm>15 视为不可信 -> ok=False
def object_point(env, *, top_uv=None, overview_uv=None, z_hint=None) -> dict
    # 优先双目三角；只有单目时按 z_hint（缺省 TABLE_HEIGHT）平面反投影，返回 source="triangulate"|"single_view"
def height_from_top_z(z_top) -> float        # 物体高度估计 = z_top - TABLE_HEIGHT（供侧抓高度 / 叠放用）
# bbox_to_world(env, bbox, camera, z=None)：z=None 按 TABLE_HEIGHT 并返回 assumed_z=True
```

`agent/schema.py`（agent-builder）：
- `pick` / `place` 增加 `target_pixel_overview: [u,v]|null`（与 `target_pixel` 配合；只有 overview 时也允许）。
- `locate(target, camera)` 的 `camera` 增加 `"both"`（**默认 both**），返回
  `{"pixel_top":..., "pixel_overview":..., "bbox_top":..., "bbox_overview":...}`。
- no-oracle 提示词追加：**高物体（水瓶、立起来的盒子）必须同时给 top 与 overview 两个像素**，
  否则会有最大 7 cm 的视差误差；矮方块给一个 top 像素即可。

像素解析（ui-fixer + Lead）：优先双像素三角定位（residual>15 mm 判 `perception_error` 可重试），
单像素时回退平面反投影并在日志标注 `source=triangulate|single_view(z=…)`。

验证（verifier）：水瓶 top-only 反投影误差应 ≥60 mm（复现视差）；双像素三角后 ≤5 mm；方块双像素 ≤3 mm。

### 5.12 参数冗余要「纠正」不要「拒绝」（Lead 冻结，实机诊断驱动）
Lead 实机诊断（真模型 + vlm 模式，`outputs/diag_vla_1.log`，2026-10-12）：
- **目标识别其实是准的**：模型给红色方块像素 (267,125) → 距真值 8.1 mm；托盘像素 (228,422) → 距托盘中心 3.3 mm；
  （加颜色精修后预期 ≤2 mm）。所以「识别不准」不是本轮的主因。
- **任务失败的直接原因是校验器把「更具体」的冗余参数当成冲突拒了**：
  1. `pick(target_id="box_red", target_pixel=[267,125])` → 判「三选一冲突」拒绝（模型既点名又给像素）；
  2. `place(container_id="tray", xy=[0.402,0.297])` → 连续两次被拒 → 触发重复守卫 → **任务中止且方块被举在空中**。

**新策略（冻结）：冗余但一致的参数要纠正后继续，只有互相矛盾才拒绝。**
1. vlm（无真值）模式：`pick` 同时给 `target_id` 与 `target_pixel`（或 `target_xy`）→ **丢掉 target_id，以像素为准**，
   日志 `[validate] 冗余 target_id 已忽略（无真值模式以像素为准）`。id 在无真值模式下不含可执行信息。
2. `place` 同时给 `container_id` 与 `xy`/`target_pixel`：落点落在该容器范围**内** → 接受 xy 为落点
   （container 只决定放置高度/越壁策略），日志标注；落点在容器**外** → 仍判矛盾、拒绝并给可读错误。
3. 真矛盾（`surface="table"` + `container_id="tray"`、`on_top_of` + `container_id` 同时给）保持拒绝。
4. oracle 模式：`target_id` + `target_xy` 同时给，且 xy 与该物体真值距离 ≤ 30 mm → 接受 xy；否则拒绝。
5. 所有「纠正」必须在 `SkillResult.metrics` 里留 `corrected_args`（记录被丢/被换的键），UI 计入「已纠正」三分计数。

验证（verifier）：新增用例覆盖 1/2/3/4；其中 2 必须验证「落点在容器内→接受、容器外→拒绝」。

---

## 6. 接触段参考限速（方法 A，Lead 冻结 2026-10-12）

背景：接触动作的单步压入量 = 参考速度 × dt。当前 `move_to_pose` 平均 0.15 m/s（min-jerk 峰值 0.19~0.28 m/s），
dt=2 ms ⇒ **每步前进 0.4~0.6 mm**，力检查在 step 之后 ⇒ 水瓶真机单步冲到 116 N（历史 2910 N）。
方法 A：把参考速度压下来即可把单步压入量按比例压小。

### 6.1 接口（`control/motion.py`，负责人：ctrl-builder）
```python
def move_to_pose(self, pos, quat=None, duration=None, compliance=None,
                 timeout: float = 8.0, force_limit=None, v_max: float|None = None) -> MoveResult
def follow(self, traj, compliance=None, timeout: float = 8.0,
           force_limit=None, v_max: float|None = None) -> MoveResult
```
- `v_max=None`：**行为与现在完全一致**（min-jerk 时间参数化 + 速度前馈 + 收尾 settle），老调用方零改动。
- `v_max>0`：参考点按 `‖p_ref − target‖ ≤ v_max·dt` 逐控制步推进（**忽略 duration**）；
  该模式下**关闭速度前馈**（或对限速后的参考做差分），避免前馈按 0.15 m/s 给关节速度；
  收尾段只允许把误差收到 `settle_tol`，不得重新加速。
- `timeout` 仍是**仿真时间**上限；调用方必须按 `dist/v_max + 1s` 显式放大（否则会 timeout）。
- **错误码集合不变**（timeout/force_limit/ik_fail/unreachable/invalid_path），不新增码。
  力检查仍在每步 step 之后、`fmax > force_limit` 即 `_abort`（走 `_halt()` 停住）；
  `_abort` 返回的 `final_pos` 必须可用——上层要据此取"接触高度"。

### 6.2 使用约定（负责人：skill-fixer）
1. **方块/低矮物体的 pick 下降**：保持现状（或最多 `v_max=0.03`），不要开低力限 —— 方块靠"到指定 z"，
   过早接触会夹到上沿（已实测回归）。
2. **细高物体（圆柱/水瓶）的 pick 下降**：`v_max=0.008`、`timeout≥12`、`force_limit=3.0`；
   若返回 `error_code="force_limit"`，**解释为"到位接触"而不是失败**：取 `final_pos` 作
   `grasp_z_effective`，就地合爪、抬升。这是语义解释，不是放宽安全门（硬限仍 15/200 N 层级不变）。
3. **place 的落放下降**：`v_max=0.01`、`timeout≥12`；接触式落放原样保留。

### 6.3 验证
- `control/smoke` 全绿（默认关闭，老行为不变）。
- 新增：① 对刚性面 60 mm 限速下降，峰值 ≤5 N；② `v_max=None` 与改动前同一条轨迹的 final_pos/duration 一致；
  ③ `v_max>0` 时速度前馈确实被关闭（或按限速参考差分）。
- 端到端：真模型 live 复跑「把水瓶抓起来」（daily、无坐标），判据 = **loop 腕部核验通过** + `peak_contact_force`。

### 5.11 端点行为实测（Lead 2026-10-12，必看）
- 该端点的 `message` 里除 `content` 外还有 **`reasoning`（思维链）**，且它**占用 completion token 预算**：
  `max_tokens=64` 时 `finish_reason=length`、`content` 为空；提到 256 才拿到答案。
  ⇒ **任何需要模型结构化判断的地方都必须走 function calling，不要依赖纯文本 content**（§5.2 的 VLM 复核要改成工具调用）。
- 因此 `VLMResponseError("响应体里没有可解析的工具调用")` 必须视为**可重试**（多为被截断），已由 Lead 改好。
- 网络：走 `http://127.0.0.1:7897` 访问 ollama.com 会**间歇性** `SSL: UNEXPECTED_EOF_WHILE_READING`（5 s 被掐断），
  同一秒重试常常成功；Lead 已给 `VLMClient._post` 加 3 次重试（0.6/1.5/3.0 s 退避）+ 超时 120 s。实测可用。

### 5.9 `PixelScriptVLM`（verifier 写进 `tests/`，Lead 也可在 tools 里用）
一个离线决策器：读 SCENE_JSON 的 `objects[*].id/color/category`（**不含坐标**），从固定脚本表查表得到像素/工具调用，
用于在无网络下驱动完整 vlm 链路（含 locate/refine/verify 的全部代码路径）。


