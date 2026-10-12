"""技能库：VLM 决策与底层控制之间的执行层（Lead 维护）。

每个技能 = 一段「规划 -> 运动 -> 抓取/放置 -> 导纳 -> 校验」的确定性流程。
VLM 只负责选技能和参数；这里负责把它做对，并在失败时给出结构化 error_code，
让上层能带着信息重试。

设计要点
--------
* 自由段用刚性档（几乎无位置修正），接触/抓取段用柔顺档（导纳接管）。
* 抓取方向：UR5e 上 Robotiq 2F85 的 TCP(g_pinch) 局部 +z 即"接近轴"（pinch 相对
  base_mount 沿 +z 偏移 0.145），所以"从上往下抓"== 让 TCP 局部 +z 指向世界 -z。
  用 mujoco.mju_quatZ2Vec 求该姿态，再绕局部 z 转 yaw 选择手指开合方向。
* 抓取成功判定：夹紧并抬升后，物体相对抓取前的 z 位移 > 0.02 m。
"""
from __future__ import annotations

import contextlib
import inspect
import time
from typing import Any, Optional

import mujoco
import numpy as np

from core.config import (
    APPROACH_HEIGHT, ARM_JOINT_LIMITS, DEFAULT_COMPLIANCE, GRASP_COMPLIANCE, GRIPPER_CLOSE_HOLD,
    GRIPPER_GAP_CLOSED_MM, GRIPPER_GAP_OPEN_MM, GRIPPER_OPEN, HOME_QPOS, LIFT_HEIGHT,
    PLACE_COMPLIANCE, SIM_DT, SOFT_COMPLIANCE, STIFF_COMPLIANCE, TABLE_HEIGHT,
)
from core.types import (
    ARM_NDOF, TCP_SITE, ComplianceParams, MoveResult, SceneObject, SkillCall, SkillResult,
)

PROFILES: dict[str, ComplianceParams] = {
    "stiff": STIFF_COMPLIANCE,
    "soft": SOFT_COMPLIANCE,
    "z_only": DEFAULT_COMPLIANCE,
}

#: 放置下降专用（守卫式下降 / guarded move）：力阈值 6 N，轻触即停。
#: 为什么不用几何硬算高度：容器（托盘）有厚度与边缘，几何算出的 tz 差几毫米就会
#: 过度下压——实测用 PLACE_COMPLIANCE(force_limit=20 N) 下降先 21.6 N 中止，随后
#: 机械臂卡死、接触力 400~500 N，多次脱身失败。真实机器人上的放置下降就是 guarded
#: move：不依赖精确几何，接触力一到阈值即认为"碰到容器/桌面 = 到位"。
PLACE_TOUCH_COMPLIANCE: ComplianceParams = ComplianceParams.z_only(
    k=120.0, d=20.0, force_limit=6.0)

#: 抓取接近后允许的物体漂移上限（米）。预抓取位的移动本身可能把物体碰动；
#: 超过该阈值说明目标已被推离（实测圆柱被推 33mm -> 重试又推到 127mm/152mm），
#: 此时必须撤退而不是继续下降/合爪。
PICK_APPROACH_DRIFT_TOL: float = 0.008

#: _skill_home 用 env.home() **瞬移** 兜底时的前置条件：接触力必须接近 0 且夹爪已张开。
#: 否则位置伺服会把"压入/夹持"状态瞬间拉到 HOME，产生极大接触力（实测可把物体抛飞）。
HOME_SAFE_FORCE_N: float = 5.0
#: 判定"夹爪已张开/已松开"的比例阈值（get_gripper: 1=全开、0=全闭）。
GRIPPER_OPEN_MIN: float = 0.8

#: _retreat 笛卡尔抬升的力看门狗。旧实现传 force_limit=1e9（理由："别被现有接触力
#: 中止，否则卡死退不出来"），但一旦指垫卡死，位置伺服 + 导纳积分会正反馈：实测
#: 撤退期间 TCP 反而从 0.412 掉到 0.384，|F| 从进入时的 20 N 一路涨到 8437 N。
#: 现在取"进入力的 HEADROOM 倍"并夹到 [MIN, MAX]：正常脱身时力在下降，不会误触发；
#: 卡死则立刻中止，交由 control 的 halt + 本函数的关节空间退化接管。
RETREAT_FORCE_HEADROOM: float = 1.5
#: task-44/R1：任何撤退/脱身的 force_limit **硬封顶 45 N**。
#: 用户实测日志里 5 方向阶梯把上限抬到 40→43.5→52.4→60.7→68.6 N，全部 force_limit 失败、
#: |F| 反而从 16 N 涨到 59.8 N；同一阶梯还重复 4 轮、状态零变化。此后不允许再出现 60+ N。
RETREAT_FORCE_CAP_N: float = 45.0
RETREAT_MIN_FORCE_N: float = 40.0
#: 兼容旧名（等价于封顶值）
RETREAT_MAX_FORCE_N: float = RETREAT_FORCE_CAP_N
#: task-44/R4：几何脱身判据——松爪后连续这么多个物理步都不与该物体有接触才算脱身
RETREAT_CLEAR_STEPS: int = 20
#: task-44/R2：入口/终态最多做几次温和撤退（不再有多轮升级阶梯）
RETREAT_MAX_GENTLE_TRIES: int = 1
#: 撤退运动的时长（秒）：旧实现 0.08 m 在 duration≈0.6 s 内完成，冲击大、越挣越死。
RETREAT_DURATION_S: float = 2.5
#: 多方向撤退的候选方向个数（-反力 / +z / 组合 / 水平切向 ±）。
RETREAT_MAX_DIRECTIONS: int = 5
#: 撤退专用柔顺档：全 6 维使能、比 SOFT 更软（k 120/20），让末端在卡滞处让位而不是硬顶。
RETREAT_COMPLIANCE: ComplianceParams = ComplianceParams(
    k=np.array([120.0, 120.0, 120.0, 20.0, 20.0, 20.0]),
    d=np.array([25.0, 25.0, 25.0, 5.0, 5.0, 5.0]),
    m=np.array([1.5, 1.5, 1.5, 0.15, 0.15, 0.15]),
    axes=np.ones(6), force_limit=RETREAT_MAX_FORCE_N)

# --------------------------------------------------------------------------- #
# 桌面放置（task-24 要求 1/2）
# --------------------------------------------------------------------------- #
#: 桌面放置时物体底面离桌面的余量（米）
PLACE_TABLE_MARGIN_M: float = 0.010
#: 自选空位时，与其它物体/托盘的最小净距（米）
PLACE_FREE_CLEARANCE_M: float = 0.08
#: 自选空位的候选网格（世界系，落在桌面可达区域内；y 上限避开托盘 y≈0.30）
PLACE_TABLE_X_RANGE: tuple = (0.28, 0.54)
PLACE_TABLE_Y_RANGE: tuple = (-0.42, 0.18)
PLACE_TABLE_GRID_M: float = 0.04
#: 空位偏好锚点（"桌面正前方偏外侧"，离各物体都远）
PLACE_TABLE_ANCHOR: tuple = (0.44, -0.30)

#: 可抓带（task-28 实测）：桌面落点若选在这里，放下后能被稳定再抓起来。
#: 逐点实测（box_blue 放桌面 -> HOME 起步 -> 抓取 -> 抬升，成功 = 抬升 > 20 mm）：
#:   OK  : x∈[0.28,0.40] & y∈[-0.26,0.18]（个别点 (0.32,-0.34)/(0.40,-0.26) 失败）
#:   OK  : (0.44, +0.10)
#:   FAIL: x∈[0.44,0.48] & y∈[-0.34,-0.18] 全部抬升中止（力 8~10 N，物体纹丝不动）
#: 失败点与 IK 残差/关节余量无关（IK 排练反而更好），是抬升执行时位置伺服跟不住、
#: 把物体压在桌面上造成的；所以只能按"实测可抓带"选落点，不能靠 IK 指标推断。
PLACE_GRASPABLE_BAND: tuple = (0.27, 0.42, -0.20, 0.18)
#: 桌面落点距桌面边缘的最小距离（米）
PLACE_EDGE_MARGIN_M: float = 0.06
#: 抓取重试覆盖（task-28 要求 2）：深度分数 × 闭合量偏移 × 偏航变体，按优先级排序。
#: task-44/R3：盲重试网格 10 组 -> 3 组。取舍理由（实测）：
#:   * 深压 (-0.4h/-0.8h) 在"物体已经放在桌面上"时会把指垫顶到桌面，descend 直接 force_limit，
#:     收益极低；真正有效的是"标准深度 / 略浅 + 90° 偏航"（指垫与物体面不贴合时救场）。
#:   * U1 要求技能层不做"换参数盲重试"——3 组用完就把结构化失败观测交回 LLM 重新规划。
#: 组合 = (depth_frac, ratio_offset, yaw)，按序尝试。
PICK_ATTEMPT_PLAN: tuple = (
    (0.0, 0.0, 0.0),                    # 标准深度、基准闭合量、yaw 0
    (0.3, 0.0, 0.0),                    # 略浅（避免顶桌面）
    (0.0, -0.15, 1.5707963267948966),   # 更紧 + 90° 偏航（贴合方向救场）
)
PICK_MAX_ATTEMPTS: int = len(PICK_ATTEMPT_PLAN)
#: task-48/R-b：无真值模式的抓取判据——抬升阶段实测指间距 > GRIPPER_GAP_CLOSED_MM + 该阈值(mm)
#: 即认为指间有物。全闭标定 8.4 mm，取 4 mm 余量（约 12.4 mm 起判"夹到东西"）。
GRASP_JAW_GAP_MIN_MM: float = 4.0
#: 反例护栏：若指间距仍接近全开（>= OPEN - 该余量），说明"合爪"根本没发生（夹爪卡住/指令没生效），
#: 此时大间距不能算"夹到东西"——否则张开着也会被判成抓住。
GRASP_JAW_OPEN_MARGIN_MM: float = 5.0
#: 无真值模式的默认半尺寸估计（米），target_size 缺失时使用
PICK_DEFAULT_HALF: tuple = (0.02, 0.02, 0.02)
#: 无真值抓取成功后 self.holding 的占位（不知道物体 id；place/return 会按"未知 id"降级处理）
HOLDING_UNNAMED: str = "unnamed"

# --------------------------------------------------------------------------- #
# task-59 / INTERFACES §5.4：接触式落放 + 侧面夹持
# --------------------------------------------------------------------------- #
#: 接触式落放：从目标 xy 上方 PLACE_CONTACT_START_M 起，按 PLACE_CONTACT_STEP_M 步长下降，
#: |F| >= PLACE_CONTACT_FORCE_N 视为接触 -> 立即松爪；下降超过 PLACE_CONTACT_MAX_M 未接触
#: 返回可重试的 place_contact_miss。
PLACE_CONTACT_START_M: float = 0.060
PLACE_CONTACT_STEP_M: float = 0.0015
PLACE_CONTACT_MAX_M: float = 0.120
PLACE_CONTACT_FORCE_N: float = 2.5
PLACE_CONTACT_STEP_DURATION_S: float = 0.15
#: 下降每一步的力看门狗：一旦 |F| 超过它，control 立刻中止这一步（不再往下压），
#: 我们随后 _freeze_arm 把 z 保持住并松爪。实测接触是"单步跳变"（0.66 N -> 18.6 N），
#: 只有让 control 在步内中止才能把落放峰值压下来（目标 <=10 N）。
PLACE_CONTACT_STOP_FORCE_N: float = 2.5
#: 接触后"停住保持 z"的步数：太长会在"夹爪还闭着、物体已顶住支撑面"时继续压出峰值，
#: 40 ms 实测能把 |F| 从 17.6 N 顶到 22.5 N；10 ms 足够撤销压入指令又不制造新峰值。
PLACE_CONTACT_HOLD_STEPS: int = 5
#: 容器内撞壁：接触点比目标高度高出 PLACE_BLOCKED_GAP_M 以上 -> 认为顶到内壁/盘沿，
#: 返回可重试的 place_blocked 并带 obstruction_id=容器 id。
PLACE_BLOCKED_GAP_M: float = 0.020
#: 接触式下降的"几何地板"：允许比几何目标再往下探这么多找接触；到地板还没接触就停住释放
#: （contact_miss=True，但不是错误）。超过它继续压只会把物体/托盘顶出峰值力。
PLACE_CONTACT_OVERSHOOT_M: float = 0.010
#: 接近"几何目标上方 PLACE_CONTACT_FINE_WINDOW_M"时改用细步长：粗步 1.5 mm 一旦越过接触面，
#: 单步下压量就直接变成接触力（实测叠放进托盘时 1.5 mm -> 峰值 11.5~12.0 N）；细步 0.5 mm
#: 把单步压入量压到 1/3，峰值随之落到 10 N 以内。
PLACE_CONTACT_FINE_STEP_M: float = 0.0002
PLACE_CONTACT_FINE_WINDOW_M: float = 0.020
#: 最后 20 mm 下降段（以及进入落放区的接近段）改用**更软的柔顺档**：把落放档的 k 按
#: PLACE_DESCENT_SOFT_K_SCALE 缩小（d 按 sqrt 缩），**force_limit 一个字节都不动**——
#: 安全红线（15 N / 6 N 硬限）不放宽，只是让末端在被挡时更多"让位"，把接触峰值压下来。
#: 实测（真机 accept_v2_final2）：硬档下冲 18.95 N 触发 15 N 硬限 -> place 失败。
PLACE_DESCENT_SOFT_K_SCALE: float = 0.35
PLACE_DESCENT_SOFT_WINDOW_M: float = 0.020
PLACE_DESCENT_SOFT_COMPLIANCE: ComplianceParams = PLACE_TOUCH_COMPLIANCE.scaled(
    PLACE_DESCENT_SOFT_K_SCALE)
#: 接近到落放起点用的软档：6 轴全使能更好吸收不确定方向的碰撞，硬限与 STIFF 一致（15 N，不变）
PLACE_APPROACH_SOFT_COMPLIANCE: ComplianceParams = ComplianceParams.soft(
    force_limit=STIFF_COMPLIANCE.force_limit).scaled(PLACE_DESCENT_SOFT_K_SCALE)
#: 侧面夹持的水平接近方向候选（按此顺序用"预抓取位 IK 可解"挑第一个可用的）。
#: 实测：近处物体 (0.40,0.02) 从 +x 接近时预抓取位在 x≈0.30，UR5e 逆解不可达；
#: 而 -x / ±y 都可解，所以不能写死方向。
SIDE_APPROACH_CANDIDATES: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                   (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0))
#: 兼容旧名（默认第一候选）
SIDE_APPROACH_DIR: tuple = (1.0, 0.0, 0.0)
#: grasp_profile 取值。默认 auto = top（自上而下）：**水瓶/圆柱也默认走 top**
#: （2F85 行程 85 mm、水瓶直径 64 mm，顶抓夹住上半身完全可行）。
#: "side"（侧面水平夹持）只在调用方显式要求**且场景有 >= PLACE_SIDE_MIN_CLEARANCE_M 净空**时用；
#: 实测没有净空时横向推进会把物体推走（已知限制，不在此加复杂力控，control 侧另派）。
GRASP_PROFILES: tuple = ("auto", "top", "side")
#: side 抓取要求的目标周围水平净空（米）
PLACE_SIDE_MIN_CLEARANCE_M: float = 0.10
# --------------------------------------------------------------------------- #
# INTERFACES §6.2 方法 A：接触段参考限速（v_max）
# --------------------------------------------------------------------------- #
#: lead 裁决（方法 A 接对地方）：**只有"最后一段接触下降"限速**，转移/抬升/接近一律原速。
#:   两段式接触下降：先按原速走到目标上方 PLACE_CONTACT_LAST10_M，再对最后这段用 CONTACT_VMAX。
#:   实测把 v_max 套到所有移动上会让 e2e 从 8s 涨到 >300s。
PLACE_CONTACT_LAST10_M: float = 0.010
CONTACT_VMAX: float = 0.003                 # ~3.3 mm/s
CONTACT_VMAX_TIMEOUT_MIN_S: float = 6.0
#: 细高物体（圆柱/水瓶）pick 最后 10mm：force_limit=3.0；返回 force_limit 解释为"到位接触"
#: （语义解释，不是放宽安全门——硬限层级不变），随后沿接近方向回退 2mm 再合爪。
PICK_VMAX_SLENDER: float = 0.008
#: 细高物体 pick 的"巡航段"（预抓取位 -> 目标上方 10mm）：模型给的 z 常是**瓶身中部**，
#: 真正的接触（瓶盖/瓶肩）就落在这段里，所以这段也不能全速冲——用中速 + 同一个 3N 看门狗。
#: 实测全速冲会把峰值顶到 25.7N；限速后显著下降，且总耗时仍在几秒级。
PICK_CRUISE_VMAX_SLENDER: float = 0.03
#: §6.2#2 说"细高物体"：用半高判据区分——半高 >= 50mm 才算细高（水瓶 200mm ✓、黄罐 70mm ✗）。
#: 实测把矮圆柱也走限速下降会改掉收尾、夹到上沿（can_yellow 回归 3 条）。
PICK_SLENDER_HALF_HEIGHT_M: float = 0.05
#: 方块/低矮物体的 pick 下降：保持现状（最多 v_max=0.03），且**不开低力限**
#: （方块靠"到指定 z"，过早接触会夹到上沿——实测回归）。
PICK_VMAX_BOX: float = 0.03
PICK_SLENDER_FORCE_LIMIT_N: float = 3.0
#: place 的落放：最后 10mm 限速 PLACE_VMAX
PLACE_VMAX: float = 0.003
#: lead 裁决（旗舰 g2 失败根因）：叠放(on_top_of)与容器内落放的**接近段**原来太快（0.03 m/s 量级），
#: 按 ctrl-builder 标定 0.016->28N / 0.008->13N / 0.003->7N，撞上就是几十牛、2.5N 看门狗在单步内
#: 拦不住 -> 直接被 15N 硬限停住（live 实测 18.66N / 37.99N）。现在整段下降都慢速：
#: 接近段 0.006、最后 10mm 0.003；看门狗(2.5N)与硬限都不动，timeout 按 dist/v_max+5 放大。
PLACE_APPROACH_VMAX: float = 0.006
VMAX_TIMEOUT_MIN_S: float = 6.0
#: 接触中止后主动回退的距离（实测中止后掌部会静压在瓶盖顶，必须退让；lead 定为 10mm）
#: **只用在细高物体分支**，方块/低矮物体路径不动。
PICK_CONTACT_BACKOFF_M: float = 0.010
#: 正面证据（lead 要求）：闭合结束时 |实测指间距 - max(2*半宽,2*半长)| <= 这个值才算"夹住物体"，
#: 明显更大说明只夹到空气或掌部顶住物体（实测水瓶 gap=79mm vs 直径 64mm）。
GRASP_SIZE_GAP_TOL_MM: float = 8.0
#: 方案 B（lead 备选，A 实测不行时的兜底）：细高物体不要"碰到就停"，而是把夹持高度定在
#: **可见顶点下方 PLACE_SLENDER_TOP_OFFSET_M**。顶点由 target_z_hint + 半高推得（模型给的是瓶身中部）：
#: 实测水瓶 hint=0.50、半高 0.10 -> 顶点 0.60（与真值一致），抓 0.57 -> 掌部(≈TCP+54mm)在 0.624，
#: 完全避开瓶盖顶，指垫正好落在瓶身上部。A（接触即停 + 回退 10mm）实测仍会掌部顶盖、合爪卡在 79mm。
PLACE_SLENDER_TOP_OFFSET_M: float = 0.030
#: 细高判据的**几何证据**（lead 要求，无真值路径必须有这条）：模型只给 target_xy / target_z_hint /
#: target_size(2 元) 时，没有 category 也拿不到半高，类别判据永远为 False -> 细高分支不触发
#: （真机 live 实测水瓶就是这么走了老路径、顶到 116.3N）。因此：**可见点高出桌面 >= 80mm 即算细高**，
#: 与类别判据取或。水瓶 0.5218-0.40=122mm ✔；方块 23mm、黄罐 42mm 都不命中（继续走老路径）。
PLACE_SLENDER_HINT_HEIGHT_M: float = 0.080
#: 细高物体夹持高度的下限保护：不低于桌面 + 这个值（兼容 z_hint 偏低的旧语义）
PLACE_SLENDER_FLOOR_MIN_M: float = 0.030
#: 细高物体的"受控合到接触"（lead 批准）：原来的 guarded close 只有 0.6s 预算，
#: 实测夹爪从 93mm 只合到 69mm 就没时间了（指令是 58mm）——这就是"没夹住瓶子"的直接原因。
#: 这里把预算给到 3s，力看门狗收到 18N（碰到瓶身就停，不硬捏）。
PLACE_SLENDER_CLOSE_TOTAL_S: float = 3.0
PLACE_SLENDER_CLOSE_FORCE_N: float = 18.0
#: 正面证据的"偏松"侧：实测 gap 比目标直径还大 3mm 以上 -> 没夹到物体（判 grasp_failed）
GRASP_SIZE_GAP_OVER_MM: float = 3.0
#: lead 最终口径：细高物体成功 = 合爪**被力挡住**（合爪过程见到 >= 这个力）**且未到机械全闭**
#: （实测 gap 仍 > 全闭间距 + GRASP_SIZE_GAP_OVER_MM）。不再拿 size 反算的直径当门槛
#: （live 的 2 元 size 推出 80.4mm vs 真值 64mm，会把真夹住判死）。
PLACE_SLENDER_MIN_CONTACT_N: float = 3.0
#: 方块/低矮物体（非细高）的"被挡住"判据容差：实测空抓时指间距**恰好等于指令值**（差 0），
#: 真夹住时一定合不到指令值（至少差几毫米）。所以用 +1mm 判"被挡住 = 夹住"，
#: 而不是原来的 +GRASP_JAW_GAP_MIN_MM(4mm)——真机方块 40mm 宽、指令 37.2mm，只差 2.8mm 会被误判。
PICK_BLOCKED_GAP_TOL_MM: float = 1.0
#: side 的"推进段"必须用轴对齐软档：GRASP_COMPLIANCE 是 z_only，水平推进时水平方向**完全没有
#: 导纳** -> 硬顶。真机实测单步冲到 **2910.3 N**（安全红线）。这里给 side 专用 6 轴软档，
#: 并给每一步显式 6 N 看门狗 —— 任何单步都不可能顶到千牛级。
PLACE_SIDE_STOP_FORCE_N: float = 6.0
PLACE_SIDE_SOFT_COMPLIANCE: ComplianceParams = ComplianceParams.soft(
    force_limit=PLACE_SIDE_STOP_FORCE_N).scaled(0.6)
#: side 预抓取/推进高度距桌面的最小余量；不足就直接拒绝进入侧向推进（退回 top-down）。
PLACE_SIDE_FLOOR_CLEARANCE_M: float = 0.012
#: 圆柱/水瓶（默认 top 抓上半身）的夹持高度下限（避免贴桌面抓）
PLACE_MIN_GRASP_HEIGHT_M: float = 0.025
#: top-down 抓取的"守卫式接触下降"（P0）：与 place 同一套思路。
#: 真机实测：瓶身中部 z_hint 让夹爪先撞瓶盖/瓶肩，旧实现一路顶到 116.3 N 才靠力限中止。
PLACE_PICK_STOP_FORCE_N: float = 1.0        # 下降看门狗（越小越早中止，单次下降的过冲峰值越小）
#: 兜底硬上限：只有真异常（旧实现那种 116N/2910N）才中止。实测"单次下降撞瓶盖"这一类的
#: 物理过冲约 28N，属于正常接触，按"接触即停"处理而不是判失败（否则瓶类抓取必挂）。
PLACE_PICK_HARD_FORCE_N: float = 60.0
PLACE_PICK_DESCEND_MAX_M: float = 0.25      # 下降行程上限（实测从 pre 位到目标 ~0.10m，留足余量）
#: 两段式下降的窗口：目标上方这么多用"1.5mm 微步 + 2.5N 看门狗"（提前碰到瓶盖/瓶肩就在那里停），
#: 窗口内改用**原来的一次性下降**落到目标 —— 保证合爪位置与旧实现完全一致
#: （实测全程微步会让指垫高 1~2mm，夹到方块上沿，抬升卡死）。
PLACE_PICK_MICRO_WINDOW_M: float = 0.020
#: 抓取下降专用软档（与 side 推进档分开）：**z_only**（水平保持刚性，位置不会漂，
#: 实测 6 轴软档会让指垫在 xy 上漂 -> 合爪位置偏掉、抬升时被卡住），z 向 k 缩一半。
#: 档位硬限仍是 PLACE_TOUCH_COMPLIANCE 的 6N；真正兜底的是每步 2.5N 看门狗，
#: PLACE_PICK_HARD_FORCE_N 只在真异常时中止。
#: 档位刚度与旧实现保持一致（z_only k=200/d=25）：实测把 k 调软会让末端 z 下沉/漂移，
#: 合爪位置偏掉 -> 抬升时被卡住（真实回归 7 条）。真正把峰值压下来的是"每步 2.5N 看门狗 + 1.5mm 小步"，
#: 不是刚度；这里只把档位 force_limit 收紧到 6N。
PLACE_PICK_DESCEND_COMPLIANCE: ComplianceParams = ComplianceParams.z_only(
    k=float(GRASP_COMPLIANCE.k[2]), d=float(GRASP_COMPLIANCE.d[2]),
    force_limit=PLACE_SIDE_STOP_FORCE_N)
#: place 入口清场兜底：无真值时，若只有一个物体与 EE 接触、且夹爪处于"中位开合"
#: （既非全开也非全闭），说明它正卡在指间 = 手里抓着的东西，不该当障碍物清场。
PLACE_HELD_GAP_MIN_MM: float = 12.0     # 全闭 8.4 + ~4
PLACE_HELD_GAP_MAX_MM: float = 88.0     # 全开 93 - 5
#: close_ratio 类别先验（仅在"没有尺寸估计"时使用；有 size/物体时仍按宽度反算，语义不变）
GRASP_CLOSE_PRIOR: dict = {"box": 0.35, "cylinder": 0.55, "bottle": 0.75}
GRASP_CLOSE_PRIOR_DEFAULT: float = 0.35
#: guarded close：合爪过程中接触力超过它就停止继续闭合（避免把物体推走/捏飞）。
#: 阈值必须高于"正常夹住"的合爪峰值（实测四个物体：box_red 13.8 N / box_blue 28.1 N /
#: box_green 19.0 N / can_yellow 34.3 N——执行器一路捏到目标开合度时本来就会这么大力），
#: 又要低于"把物体顶飞"的量级（实测错位合爪 55 N 起）。取 45 N。
PICK_CLOSE_FORCE_MAX_N: float = 45.0
#: 抓取预演里"关节离限位余量"的期望值（米制无关，单位 rad）：低于它按比例罚分
PLACE_REHEARSAL_MIN_MARGIN: float = 0.20

# --------------------------------------------------------------------------- #
# 叠放 on_top_of（task-32）
# --------------------------------------------------------------------------- #
#: 叠放余量：被夹物底面离锚点顶面的目标间隙（米）
PLACE_STACK_MARGIN_M: float = 0.004
#: 叠放判据：被夹物底面与锚点顶面的距离允许区间（米）
PLACE_STACK_GAP_MIN_M: float = -0.008
PLACE_STACK_GAP_MAX_M: float = 0.02
#: xy 必须落在锚点 footprint 内，半宽额外容差（米）
PLACE_STACK_FOOTPRINT_TOL_M: float = 0.02
#: 锚点必须在桌面上：其底面相对桌面的允许误差（米）
PLACE_ANCHOR_TABLE_MIN_M: float = -0.01
PLACE_ANCHOR_TABLE_MAX_M: float = 0.02
#: 释放后的观察时长（秒）与"滑落/翻滚"上限
PLACE_STACK_OBSERVE_S: float = 0.4
PLACE_STACK_SLIDE_MAX_M: float = 0.02
PLACE_STACK_DROP_MAX_M: float = 0.02

# --------------------------------------------------------------------------- #
# 横向关系放置 anchor_id + relation（task-36）
# --------------------------------------------------------------------------- #
#: 方向约定（**俯视图/世界系：x 向右、y 向前**，与场景物体沿 x/y 摆放一致）：
#:   right = +x    left = -x    front = +y    back = -y
RELATION_DIRS: dict = {
    "right": (1.0, 0.0), "left": (-1.0, 0.0),
    "front": (0.0, 1.0), "back": (0.0, -1.0),
}
#: 关系词别名（英文/中文）
RELATION_ALIASES: dict = {
    "right": "right", "right_of": "right", "rightof": "right", "右边": "right", "右侧": "right",
    "left": "left", "left_of": "left", "leftof": "left", "左边": "left", "左侧": "left",
    "front": "front", "in_front_of": "front", "infront": "front", "前面": "front", "前方": "front",
    "back": "back", "behind": "back", "后面": "back", "后方": "back",
    "beside": "beside", "next_to": "beside", "nextto": "beside", "旁边": "beside", "侧边": "beside",
}
#: 并排间距余量（米）：中心距 = 锚点沿该轴半宽 + 被夹物沿该轴半宽 + margin
PLACE_SIDE_MARGIN_M: float = 0.01
#: 实际中心距相对期望值的允许区间（米）：[期望-2cm, 期望+4cm]
PLACE_SIDE_DIST_MIN_M: float = 0.02
PLACE_SIDE_DIST_MAX_M: float = 0.04
#: 桌面上表面外沿（table center 0.10,0 / half 0.60,0.60，留 0.03 余量）
PLACE_TABLE_BOUNDS: tuple = (-0.47, 0.67, -0.57, 0.57)

#: 放置成功判据（task-24 要求 2）：xy 距目标 / 底面与支撑面距离 / 接触力 / 夹爪开合
PLACE_XY_TOL_M: float = 0.06
PLACE_SEAT_MIN_M: float = -0.01
PLACE_SEAT_MAX_M: float = 0.02
PLACE_FORCE_MAX_N: float = 5.0
#: 释放后强制抬升的高度（米）与允许残留力；用于把指垫彻底带离被放物体
PLACE_LIFT_CLEARANCE_M: float = 0.15


def approach_quat(yaw: float = 0.0) -> np.ndarray:
    """返回让 TCP 局部 +z 指向世界 -z 的四元数（wxyz），再绕局部 z 旋转 yaw。"""
    q_down = np.zeros(4)
    mujoco.mju_quatZ2Vec(q_down, np.array([0.0, 0.0, -1.0]))
    if abs(float(yaw)) < 1e-9:
        return q_down
    q_yaw = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_yaw, np.array([0.0, 0.0, 1.0]), float(yaw))
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q_down, q_yaw)
    return out


def approach_quat_dir(direction, roll: float = 0.0) -> np.ndarray:
    """让 TCP 局部 +z 指向 direction（世界系）的四元数（wxyz），再绕**局部 z** 转 roll。

    top-down 就是 direction=[0,0,-1]（与既有 approach_quat 完全等价）；侧面夹持用 [1,0,0]。
    roll 用于选择两指开合方向（对贴合敏感）。
    """
    d = np.asarray(direction, dtype=float).reshape(-1)[:3]
    n = float(np.linalg.norm(d))
    d = d / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])
    q = np.zeros(4)
    mujoco.mju_quatZ2Vec(q, d)
    if abs(float(roll)) < 1e-9:
        return q
    q_roll = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_roll, np.array([0.0, 0.0, 1.0]), float(roll))
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q, q_roll)
    return out


class SkillExecutor:
    """把 SkillCall 映射为具体动作。可直接作为 AgentLoop 的 executor 传入。"""

    def __init__(self, env, kin, runner, planner, log=print):
        self.env = env
        self.kin = kin
        self.runner = runner
        self.planner = planner
        self.log = log
        self.compliance: ComplianceParams = DEFAULT_COMPLIANCE
        self.holding: Optional[str] = None
        #: 最近一次成功 pick 时、物体被抓起前的位姿（供 return_held_object 放回原处）
        self.last_pick_pose: Optional[np.ndarray] = None
        self.history: list[dict] = []
        # 注：规划路径的整定速度由 control.MotionRunner 决定（现行 0.06 s/路点、0.05 rad），
        # 这里不再覆盖，避免与模块默认值打架。

    # ------------------------------------------------------------------ #
    # 入口
    # ------------------------------------------------------------------ #
    def __call__(self, call: SkillCall) -> SkillResult:
        t0 = time.time()
        name = call.skill
        fn = getattr(self, "_skill_" + str(name), None)
        if fn is None:
            res = SkillResult(False, name, f"unknown skill: {name}", "unknown_skill")
        else:
            try:
                res = fn(call)
            except Exception as exc:  # noqa: BLE001 - 技能失败必须变成结构化结果
                res = SkillResult(False, name, f"exception: {type(exc).__name__}: {exc}", "exception")
        res.duration = time.time() - t0
        self.history.append({"call": call.to_json(), "result": res.to_observation()})
        return res

    # ------------------------------------------------------------------ #
    # 辅助
    # ------------------------------------------------------------------ #
    def _settle(self, seconds: float) -> None:
        self.env.step(max(1, int(round(seconds / self.env.dt))))

    def _object(self, obj_id: str) -> Optional[SceneObject]:
        for o in self.env.list_objects():
            if o.id == obj_id:
                return o
        return None

    def _motion_compliance(self, call: SkillCall, base: ComplianceParams) -> ComplianceParams:
        """技能运动用的柔顺档。

        VLM 的柔顺意图只应决定「刚度/阻尼」（软=让位多），不应降低技能自身的安全力阈值
        ——force_limit 是"接触异常中止"的保护，不是柔顺度。否则 VLM 一句"轻一点"会把
        阈值降到 8 N，导致正常贴面放置都判超限。这里对 force_limit 取技能基准作为下界。
        """
        comp = self._compliance_from_call(call, base)
        return ComplianceParams(k=comp.k, d=comp.d, m=comp.m, axes=comp.axes,
                                force_limit=max(float(comp.force_limit), float(base.force_limit)))

    @staticmethod
    def close_ratio_for(obj: SceneObject, margin_mm: float = 6.0) -> float:
        """按物体宽度反算夹爪闭合量。

        sim 实测：open_ratio=1 -> 指间距 93 mm，open_ratio=0 -> 8.4 mm，线性。
        目标指间距 = 物体宽度 - margin_mm（略小于物体，保证真正夹紧）。
        固定用 0.35 对 40 mm 方块勉强可行，但对 30 mm 圆柱（指间距 38 mm > 30 mm）
        根本夹不住——必须按物体现算。
        """
        width_mm = 2000.0 * float(max(obj.size[0], obj.size[1]))
        gap_mm = width_mm - float(margin_mm)
        span = GRIPPER_GAP_OPEN_MM - GRIPPER_GAP_CLOSED_MM
        return float(np.clip((gap_mm - GRIPPER_GAP_CLOSED_MM) / span, 0.0, 1.0))

    def _arm_qpos_now(self) -> np.ndarray:
        """读当前 6 个臂关节角（不假设 env 一定暴露 data / arm_qpos_idx）。"""
        data = getattr(self.env, "data", None)
        idx = getattr(self.env, "arm_qpos_idx", None)
        if data is None or idx is None:
            return np.zeros(ARM_NDOF, dtype=float)
        try:
            return np.array(data.qpos, dtype=float)[np.asarray(idx, dtype=int)]
        except Exception:  # noqa: BLE001
            return np.zeros(ARM_NDOF, dtype=float)

    def _gripper_now(self) -> float:
        """读夹爪实际开合（1=全开）；env 不支持时按全开处理。"""
        try:
            return float(self.env.get_gripper())
        except Exception:  # noqa: BLE001
            return float(GRIPPER_OPEN)

    def _jaw_gap_mm(self) -> float:
        """task-48/R-b：把夹爪开合比例用 GRIPPER_GAP_* 标定换算成实际指间距（mm）。"""
        r = float(np.clip(self._gripper_now(), 0.0, 1.0))
        return float(GRIPPER_GAP_CLOSED_MM + r * (GRIPPER_GAP_OPEN_MM - GRIPPER_GAP_CLOSED_MM))

    @staticmethod
    def _close_ratio_for_size(half_size) -> float:
        """无真值模式：按尺寸估计反算夹爪闭合量（与 close_ratio_for 同一标定）。"""
        s = np.asarray(half_size, float).reshape(-1)
        width_mm = 2000.0 * float(max(s[0], s[1]))
        gap_mm = width_mm - 6.0
        span = GRIPPER_GAP_OPEN_MM - GRIPPER_GAP_CLOSED_MM
        return float(np.clip((gap_mm - GRIPPER_GAP_CLOSED_MM) / span, 0.0, 1.0))

    def _no_oracle_diagnostics(self, target_point) -> dict:
        """诊断用真值（R-a 明确允许真值只出现在 metrics 里）：离目标点最近的物体与误差。"""
        try:
            tp = np.asarray(target_point, float).reshape(-1)[:3]
            best = None
            for o in self.env.list_objects():
                if self.holding is not None and str(o.id) == str(self.holding):
                    continue
                # 只比水平误差（抬升后 z 会差 12 cm，与"目标点选得准不准"无关）
                d = float(np.linalg.norm(np.asarray(o.pose[:2], float) - tp[:2]))
                if best is None or d < best[0]:
                    best = (d, str(o.id))
            if best is None:
                return {"oracle_nearest_id": None, "oracle_xy_error_mm": None}
            return {"oracle_nearest_id": best[1],
                    "oracle_xy_error_mm": round(float(best[0]) * 1000.0, 2)}
        except Exception:  # noqa: BLE001
            return {"oracle_nearest_id": None, "oracle_xy_error_mm": None}

    def _choose_side_dir(self, point, approach: float = 0.10, roll: float = 0.0,
                         exclude_id: Optional[str] = None,
                         use_clearance: bool = False) -> np.ndarray:
        """侧面夹持：挑一个"预抓取位 IK 可解"的水平方向（+x 优先，其次 ±y、-x）。

        为什么不能写死：实测近处物体 (0.40,0.02,0.435) 从 +x 接近时预抓取位 x≈0.30 逆解不可达，
        而 -x / ±y 都可解；远处物体两者都行。
        use_clearance=True（**仅 oracle 路径**：会读物体位姿）时再在其中挑"预抓取位离其它物体
        最远"的方向——实测默认场景里 +y 会让张开 93 mm 的夹爪扫到邻居（box_red），
        换到 -x 就干净了。无真值(vlm)模式必须保持 False（§5.7 禁止读真值）。
        全部不可解时退回第一候选，让上层照常报错。
        """
        pt = np.asarray(point, float).reshape(-1)[:3]
        scored = []
        for idx, d in enumerate(SIDE_APPROACH_CANDIDATES):
            dv = np.asarray(d, float)
            pre = pt - dv * float(approach)
            try:
                _q, ok = self.kin.ik(pre, approach_quat_dir(dv, roll))
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                continue
            clear = 9.9
            if use_clearance:
                try:
                    for o in self.env.list_objects():
                        if exclude_id is not None and str(o.id) == str(exclude_id):
                            continue
                        if self.holding is not None and str(o.id) == str(self.holding):
                            continue
                        clear = min(clear,
                                    float(np.linalg.norm(
                                        np.asarray(o.pose[:2], float) - pre[:2]))
                                    - float(max(o.size[0], o.size[1])))
                except Exception:  # noqa: BLE001
                    clear = 9.9
            scored.append((clear, -idx, dv))
        if not scored:
            return np.asarray(SIDE_APPROACH_CANDIDATES[0], float)
        return max(scored, key=lambda c: (round(float(c[0]), 3), c[1]))[2]

    def _looks_like_holding_object(self) -> bool:
        """无真值兜底：判断"EE 接触的唯一物体就是手里抓着的东西"。

        条件：与 EE 接触的物体恰好 1 个（已排除已知手持 id），且夹爪实测指间距处于中位区间
        （既非全开 93 mm 也非全闭 8.4 mm）——夹爪卡在中位说明指间有物 = 正被夹着。
        """
        touching = self._objects_touching_ee(exclude_held=True)
        if len(touching) != 1:
            return False
        gap = self._jaw_gap_mm()
        return (float(PLACE_HELD_GAP_MIN_MM) <= gap <= float(PLACE_HELD_GAP_MAX_MM))

    def _runner_accepts(self, name: str) -> bool:
        """runner.move_to_pose 是否支持某个新形参（v_max / stop_on_force）。

        ctrl-builder 的新接口是陆续落地的；老 runner 或测试桩没有该形参时自动退回旧行为（零回归）。
        """
        cache = getattr(self, "_runner_accepts_cache", None)
        if cache is None:
            cache = {"v_max": False, "stop_on_force": False}
            try:
                params = inspect.signature(self.runner.move_to_pose).parameters
                has_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                cache = {k: (k in params or has_kw) for k in ("v_max", "stop_on_force")}
            except (TypeError, ValueError):
                pass
            self._runner_accepts_cache = cache
        return bool(cache.get(name, False))

    @staticmethod
    def _is_contact_stop(r) -> bool:
        """ctrl-builder 的 stop_on_force：力到阈值就冻结参考并**正常返回**（ok=True, stop_reason='contact'）。

        上层必须把它当"到位接触"处理（停住 -> 释放/校验），而不是错误。
        """
        if not bool(getattr(r, "ok", False)):
            return False
        if getattr(r, "stopped", False):          # 兼容：ctrl-builder 也可能用 bool 标记
            return True
        return str(getattr(r, "stop_reason", "") or "").strip().lower() in (
            "contact", "contact_stop", "force", "force_stop")

    def _supports_v_max(self) -> bool:
        """§6.1 的 v_max 是 ctr-builder 新加的形参；没有就退回旧行为（不传 v_max），保证零回归。"""
        return self._runner_accepts("v_max")

    @staticmethod
    def _vmax_timeout(dist: float, v_max: float,
                      minimum: float = VMAX_TIMEOUT_MIN_S) -> float:
        """§6.1：限速模式下 timeout 必须按 dist/v_max + 1s 放大，且 §6.2 要求 >=12s。

        实测按 +1s 太紧：限速参考走完 dist/v_max 之后还有收尾 settle，会刚好超时
        （现象：'收尾超时 12.986s > 12.984s' 且报 timeout）。这里留 +3s 余量。
        """
        # 实测 dist/v_max 之后还有 1~2s 收尾 settle：+2s 会刚好超时（"收尾超时 8.668s > 8.667s"），
        # 这里留 +5s 余量。timeout 只是上限，不影响正常耗时。
        return max(float(minimum), abs(float(dist)) / max(float(v_max), 1e-9) + 5.0)

    def _move_vmax(self, pos, quat, compliance, timeout: float, force_limit,
                   v_max: Optional[float] = None, duration: Optional[float] = None,
                   stop_on_force: Optional[bool] = None):
        """带 v_max / stop_on_force 的 move_to_pose 包装：runner 不支持时按旧行为调用（零回归）。

        stop_on_force=True（ctrl-builder 的 contact-stop 模式）：力到阈值就**冻结参考并正常返回**
        （ok=True, stop_reason="contact"），而不是中止报错——叠放/容器落放要的就是"撞到就停住并正常释放"。
        """
        if v_max is not None and self._supports_v_max():
            # §6.1：v_max>0 时忽略 duration，所以这里不传 duration
            kwargs = {"compliance": compliance, "timeout": timeout,
                      "force_limit": force_limit, "v_max": float(v_max)}
            if stop_on_force is not None and self._runner_accepts("stop_on_force"):
                kwargs["stop_on_force"] = bool(stop_on_force)
            return self.runner.move_to_pose(pos, quat, **kwargs)
        kwargs = {"compliance": compliance, "timeout": timeout, "force_limit": force_limit}
        if duration is not None:
            kwargs["duration"] = float(duration)
        return self.runner.move_to_pose(pos, quat, **kwargs)

    def _descend_grasp_until_contact(self, target, quat, timeout: float = 6.0,
                                     slender: bool = False,
                                     base_compliance: Optional[ComplianceParams] = None):
        """top-down 抓取的守卫式接触下降：1.5 mm 小步 + 每步 2.5 N 看门狗 + 软档。

        一旦接触就**停在当前高度**（metrics.grasp_z_effective / touchdown_force），由调用方在这个
        高度合爪再抬升 —— 对"模型给的是瓶身中部、实际先碰到瓶盖/瓶肩"的高物体，这就是正确行为
        （等于抓上半身），比一路顶到力限再失败好得多。
        每一步都立刻查力，任何单步都不得冲到几十/几百牛；PLACE_PICK_HARD_FORCE_N 只做兜底。
        返回 (status, metrics, move_result)，status ∈ {"contact","arrived","error"}。
        """
        tgt = np.asarray(target, float).reshape(-1)[:3]
        z_now = float(np.asarray(self.env.site_pose(TCP_SITE)[0], float)[2])
        if slender:
            # lead 裁决的两段式：巡航段（预抓取位 -> 目标上方 10mm）+ 最后 10mm 限速。
            # 注意：模型给的 z 常是**瓶身中部**，真正的接触（瓶盖/瓶肩）落在巡航段里，
            # 所以巡航段必须同时带同一个 3N 看门狗（否则全速冲 = 25.7N 峰值）。
            comp = PLACE_PICK_DESCEND_COMPLIANCE
            cruise = np.array([float(tgt[0]), float(tgt[1]),
                               float(tgt[2]) + float(PLACE_CONTACT_LAST10_M)])
            _cdist = max(1e-3, abs(z_now - float(cruise[2])))
            r_cruise = self._move_vmax(cruise, quat, comp,
                                       self._vmax_timeout(_cdist, PICK_CRUISE_VMAX_SLENDER,
                                                          CONTACT_VMAX_TIMEOUT_MIN_S),
                                       float(PICK_SLENDER_FORCE_LIMIT_N),
                                       PICK_CRUISE_VMAX_SLENDER)
            if not r_cruise.ok and r_cruise.error_code != "force_limit":
                return "error", {"step_error": r_cruise.error_code or "descend_failed"}, r_cruise
            if not r_cruise.ok:
                r = r_cruise          # 巡航段就碰到：按"到位接触"处理，跳过阶段2
            else:
                r = self._move_vmax(tgt, quat, comp,
                                    self._vmax_timeout(float(PLACE_CONTACT_LAST10_M),
                                                       float(CONTACT_VMAX),
                                                       CONTACT_VMAX_TIMEOUT_MIN_S),
                                    float(PICK_SLENDER_FORCE_LIMIT_N), float(CONTACT_VMAX))
        else:
            # §6.2#1 方块/低矮物体：**完全保持现状**——不传 v_max、不开低力限
            # （实测 v_max=0.03 也会改掉收尾行为、把方块夹到上沿 -> 抬升卡死）
            comp = base_compliance or PLACE_PICK_DESCEND_COMPLIANCE
            r = self._move_vmax(tgt, quat, comp, timeout, None, None)
        # 接触高度优先取 r.final_pos（§6.1：_abort 的 final_pos 必须可用，上层据此取接触高度）
        fp = getattr(r, "final_pos", None)
        tcp_z = (float(np.asarray(fp, float).reshape(-1)[2]) if fp is not None else z_now)
        f_now = self._force_norm()
        peak = max(float(f_now), float(getattr(r, "max_force", 0.0) or 0.0))
        if peak >= float(PLACE_PICK_HARD_FORCE_N):        # 兜底：真异常才中止
            return "error", {"peak_contact_force": round(peak, 3),
                             "step_error": "force_limit"}, r
        if r.ok:
            return "arrived", {"grasp_z_effective": round(tcp_z, 4),
                               "touchdown_force": round(float(f_now), 3),
                               "peak_contact_force": round(peak, 3),
                               "grasp_descend_steps": 1}, r
        if r.error_code == "force_limit":
            # 看门狗中止 = 半路碰到东西：停在该高度合爪（等于抓上半身），并转成成功到点
            self._freeze_arm(int(PLACE_CONTACT_HOLD_STEPS))
            if slender:
                # lead 要求：中止后手臂会静靠在物体上（实测约 10N），必须**沿接近方向回退 2mm**
                # 主动让开，再合爪/抬升。
                back = np.array([float(tgt[0]), float(tgt[1]),
                                 float(tcp_z) + float(PICK_CONTACT_BACKOFF_M)])
                self._move_vmax(back, quat, comp, 6.0, None, None, duration=0.5)
            f_hold = max(float(f_now), self._force_norm(), float(peak))
            return "contact", {"grasp_z_effective": round(tcp_z, 4),
                               "touchdown_force": round(float(f_hold), 3),
                               "peak_contact_force": round(float(max(peak, f_hold)), 3),
                               "grasp_descend_steps": 1}, \
                MoveResult(True, "接触即停（守卫式接触下降）", None, max_force=float(f_hold))
        return "error", {"step_error": r.error_code or "descend_failed"}, r
        # Phase 2：与旧实现同一条轨迹，落到目标
        r = self.runner.move_to_pose(tgt, quat, compliance=comp, timeout=timeout,
                                      force_limit=float(PLACE_PICK_STOP_FORCE_N))
        tcp_z = float(np.asarray(self.env.site_pose(TCP_SITE)[0], float)[2])
        f_now = self._force_norm()
        peak = max(peak, float(f_now), float(getattr(r, "max_force", 0.0) or 0.0))
        if r.ok:
            return "arrived", {"grasp_z_effective": round(tcp_z, 4),
                               "touchdown_force": round(float(f_now), 3),
                               "peak_contact_force": round(peak, 3),
                               "grasp_descend_mm": round((window_top - z) * 1000.0, 1),
                               "grasp_descend_steps": int(steps + 1)}, r
        if r.error_code == "force_limit":
            self._freeze_arm(int(PLACE_CONTACT_HOLD_STEPS))
            f_hold = max(float(f_now), self._force_norm(), float(peak))
            return "contact", {"grasp_z_effective": round(tcp_z, 4),
                               "touchdown_force": round(float(f_hold), 3),
                               "peak_contact_force": round(float(max(peak, f_hold)), 3),
                               "grasp_descend_steps": int(steps + 1)}, \
                MoveResult(True, "接触即停（守卫式接触下降）", None, max_force=float(f_hold))
        return "error", {"step_error": r.error_code or "descend_failed"}, r

    def _side_clearance(self, point, half_self: float = 0.02,
                        exclude_id: Optional[str] = None) -> float:
        """目标点周围的最小水平净空（离最近物体表面的距离，米）。仅 oracle 路径可用。"""
        pt = np.asarray(point, float).reshape(-1)[:3]
        best = 9.9
        try:
            for o in self.env.list_objects():
                if exclude_id is not None and str(o.id) == str(exclude_id):
                    continue
                if self.holding is not None and str(o.id) == str(self.holding):
                    continue
                d = float(np.linalg.norm(np.asarray(o.pose[:2], float) - pt[:2]))
                best = min(best, d - float(half_self) - float(max(o.size[0], o.size[1])))
        except Exception:  # noqa: BLE001
            return 9.9
        return float(best)

    def _descend_until_contact(self, target, quat, container_id: Optional[str] = None):
        """task-59/§5.4 接触式落放：xy 上方 60 mm 起、1.5 mm 步长下降，|F| >= 2.5 N 即接触。

        返回 (status, metrics, move_result)：
          * "contact"：触底（metrics.touchdown_z / contact_force_norm；调用方随即松爪）
          * "miss"   ：下降超过 120 mm 仍未接触 -> 可重试 place_contact_miss
          * "blocked"：容器内落放，在目标上方 >= 20 mm 就接触（顶到内壁/盘沿）
                       -> 可重试 place_blocked，metrics.obstruction_id = 容器 id
          * "error"  ：步进本身失败（timeout/ik_fail/...），沿用既有 descend_failed 语义
        接触判据用"当前力 与 该步 MoveResult.max_force 的较大者"，避免漏掉步内峰值。
        """
        tgt = np.asarray(target, float).reshape(-1)[:3]
        max_travel = float(PLACE_CONTACT_MAX_M)
        start = np.array([tgt[0], tgt[1], tgt[2] + float(PLACE_CONTACT_START_M)])
        # 接近段用软档（硬限仍是 STIFF 的 15 N，不放宽）：真机失败那次就是这一步行进中
        # 被挡到 18.95 N 触发硬限；软档让末端让位而不是硬顶。
        r0 = self._goto_planned(start, quat, PLACE_APPROACH_SOFT_COMPLIANCE)
        if not r0.ok:
            return "error", {"step_error": r0.error_code or "approach_failed",
                             "touched": False}, r0
        floor_z = float(tgt[2]) - float(PLACE_CONTACT_OVERSHOOT_M)      # 几何地板
        phase1_z = float(tgt[2]) + float(PLACE_CONTACT_LAST10_M)        # 阶段2 边界
        total_mm = abs(float(start[2]) - floor_z) * 1000.0
        if abs(float(start[2]) - float(floor_z)) > max_travel:
            # 目标异常深（整个下降超出 PLACE_CONTACT_MAX_M 窗口）-> 可重试 place_contact_miss
            return "miss", {"touched": False, "descend_travel_mm": round(total_mm, 1),
                            "peak_contact_force": 0.0, "soft_descent_steps": 0}, None
        peak = 0.0
        steps = 0

        def _touch(r_cur, f_val, peak_now, steps_now):
            """接触即停 + 容器撞壁判定（阶段1/阶段2 共用）。"""
            self._freeze_arm(int(PLACE_CONTACT_HOLD_STEPS))
            f_hold = max(float(f_val), self._force_norm())
            tcp_z = float(np.asarray(self.env.site_pose(TCP_SITE)[0], float)[2])
            m = {"touchdown_z": round(tcp_z, 4),
                 "contact_force_norm": round(float(f_hold), 3),
                 "peak_contact_force": round(float(max(peak_now, f_hold)), 3),
                 "descend_travel_mm": round(total_mm, 1),
                 "descend_steps": int(steps_now),
                 "soft_descent_steps": 1}
            if (container_id is not None
                    and (tcp_z - float(tgt[2])) > float(PLACE_BLOCKED_GAP_M)):
                m["obstruction_id"] = str(container_id)
                m["blocked_gap_mm"] = round((tcp_z - float(tgt[2])) * 1000.0, 1)
                return "blocked", m, r_cur
            return "contact", m, r_cur

        # ---- 阶段1（限速接近，一次走完）：起点 -> 目标上方 PLACE_CONTACT_LAST10_M ----
        # lead 裁决（旗舰 g2 失败根因）：叠放/容器内落放的接近段原来太快（0.03 m/s 量级），
        # 按 ctrl-builder 标定 0.016->28N / 0.008->13N / 0.003->7N，撞上就是几十牛、2.5N 看门狗
        # 在单步内拦不住，直接被 15N 硬限停住（live 18.66N / 37.99N）。现在接近段 0.006 m/s。
        # 看门狗与硬限都不动；timeout 按 dist/v_max+5 放大。
        _d1 = max(1e-3, abs(float(start[2]) - float(phase1_z)))
        r = self._move_vmax(np.array([tgt[0], tgt[1], phase1_z]), quat,
                            PLACE_TOUCH_COMPLIANCE,
                            self._vmax_timeout(_d1, PLACE_APPROACH_VMAX),
                            None, PLACE_APPROACH_VMAX, duration=1.5,
                            stop_on_force=True)
        steps = 1
        f = max(self._force_norm(), float(getattr(r, "max_force", 0.0) or 0.0))
        peak = max(peak, f)
        if (self._is_contact_stop(r) or (not r.ok and r.error_code == "force_limit")
                or f >= float(PLACE_CONTACT_FORCE_N)):
            # stop_on_force 的 contact-stop（ok=True）与看门狗中止一样都算"到位接触"
            return _touch(r, f, peak, steps)          # 还没到 10mm 边界就碰到了
        if not r.ok:
            return "error", {"step_error": r.error_code or "descend_failed",
                             "touched": False, "descend_travel_mm": round(total_mm, 1)}, r

        # ---- 阶段2（限速 v_max=CONTACT_VMAX）：最后 10mm + 几何地板，一次走完 ----
        dist2 = abs(float(floor_z) - float(phase1_z))
        if dist2 <= 1e-9:
            return "arrived", {"touchdown_z": round(float(phase1_z), 4),
                               "contact_force_norm": round(float(f), 3),
                               "peak_contact_force": round(float(peak), 3),
                               "descend_travel_mm": round(total_mm, 1),
                               "descend_steps": int(steps), "contact_miss": True}, r
        r2 = self._move_vmax(np.array([tgt[0], tgt[1], floor_z]), quat,
                             PLACE_DESCENT_SOFT_COMPLIANCE,
                             self._vmax_timeout(dist2, PLACE_VMAX, CONTACT_VMAX_TIMEOUT_MIN_S),
                             float(PLACE_CONTACT_STOP_FORCE_N), PLACE_VMAX,
                             stop_on_force=True)
        steps += 1
        f2 = max(self._force_norm(), float(getattr(r2, "max_force", 0.0) or 0.0))
        peak = max(peak, f2)
        if (self._is_contact_stop(r2) or (not r2.ok and r2.error_code == "force_limit")
                or f2 >= float(PLACE_CONTACT_FORCE_N)):
            return _touch(r2, f2, peak, steps)
        if not r2.ok:
            return "error", {"step_error": r2.error_code or "descend_failed",
                             "touched": False, "descend_travel_mm": round(total_mm, 1)}, r2
        # 走到几何地板仍无接触：正常释放/校验（不是错误）
        self._freeze_arm(int(PLACE_CONTACT_HOLD_STEPS))
        f_hold = max(float(f2), self._force_norm())
        tcp_z = float(np.asarray(self.env.site_pose(TCP_SITE)[0], float)[2])
        return "arrived", {"touchdown_z": round(tcp_z, 4),
                           "contact_force_norm": round(float(f_hold), 3),
                           "peak_contact_force": round(float(max(peak, f_hold)), 3),
                           "descend_travel_mm": round(total_mm, 1),
                           "descend_steps": int(steps),
                           "soft_descent_steps": 1,
                           "contact_miss": True}, r2

    def _refresh_holding_if_released(self) -> None:
        """N2 holding 卫生：夹爪实际已张开（>= GRIPPER_OPEN_MIN）说明物已松开，清空 holding。

        否则 UI 的 :state 会显示 skills.holding 与实际夹爪不一致（实测松爪后 robot
        holding=None/gripper=0.966，skills.holding 仍是 box_blue）。
        """
        if self.holding is not None and self._gripper_now() >= GRIPPER_OPEN_MIN:
            self.holding = None

    @contextlib.contextmanager
    def _ignore_contacts_with(self, obj_id: Optional[str]):
        """规划期间临时忽略「物体 <-> 夹爪」接触对的上下文，出栈必还原。

        为什么需要：带载/被卡住时，夹爪与手持物体持续接触会让 CollisionChecker 把
        *起始位形* 判为非法，于是 plan_joint 直接返回 None（N1 的 home_blocked、
        N3 的退不出来都是这个原因）。只在规划窗口里置位 mask，finally 还原，不影响
        其它模块的碰撞语义。checker 不支持该内部结构时安全跳过。

        P0 修复（无真值模式 holding='unnamed'）：旧实现按"geom body 名 == obj_id"匹配，
        'unnamed' 永远匹配 0 个 geom（实测日志 `忽略 unnamed <-> 夹爪 接触对 0 个`），
        于是带载归位必然回落瞬移兜底、把物体丢下。现在 obj_id 认不出真实物体时，
        按"当前与夹爪接触的可动物体"反查真实 body 名。
        """
        checker = getattr(self.planner, "checker", None)
        mask = getattr(checker, "_ignore_mask", None)
        names = getattr(checker, "_geom_body_names", None)
        is_grip = getattr(checker, "_is_gripper_geom", None)
        model = getattr(checker, "model", None)
        if obj_id is None or mask is None or names is None or is_grip is None or model is None:
            yield
            return
        saved = mask.copy()
        n = int(getattr(model, "ngeom", len(names)))
        try:
            targets = self._resolve_ignore_targets(obj_id, names)
            if not targets:
                yield
                return
            obj_geoms = [g for g in range(n) if names[g] in targets]
            grip_geoms = [g for g in range(n) if bool(is_grip[g])]
            for g in obj_geoms:
                for h in grip_geoms:
                    mask[g, h] = True
                    mask[h, g] = True
            self.log(f"[skills] 规划临时忽略 {', '.join(sorted(targets))} <-> 夹爪 接触对 "
                     f"{len(obj_geoms) * len(grip_geoms)} 个")
            yield
        finally:
            mask[:] = saved

    def _resolve_ignore_targets(self, obj_id: Any, geom_body_names) -> set:
        """把要忽略的 obj_id 解析成 checker 里真实的 body 名集合。

        真值/oracle 路径直接用 obj_id；认不出来（无真值占位 'unnamed'、id 拼错）时
        退回"当前与夹爪接触的可动物体"，绝不因为名字对不上就放弃忽略。
        """
        wanted = str(obj_id)
        try:
            known = {str(x) for x in geom_body_names}
        except Exception:  # noqa: BLE001
            known = set()
        if wanted in known:
            return {wanted}
        return set(self._gripper_contact_bodies() or ())

    @staticmethod
    def _body_is_free(model, bid: int) -> bool:
        """body 是否带 free joint（= 可被搬动的物体；排除 world/table/机械臂连杆）。"""
        try:
            adr, num = int(model.body_jntadr[bid]), int(model.body_jntnum[bid])
        except Exception:  # noqa: BLE001
            return False
        for j in range(adr, adr + num):
            try:
                if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                    return True
            except Exception:  # noqa: BLE001
                return False
        return False

    def _gripper_contact_bodies(self) -> Optional[list]:
        """当前与夹爪 geom 接触的**可动物体** body 名列表；环境没有接触信息返回 None。

        纯物理接触信息（不读物体真值位姿），用于无真值模式反查"手里到底夹着谁"。
        """
        env = self.env
        model = getattr(env, "model", None)
        data = getattr(env, "data", None)
        if model is None or data is None or getattr(data, "contact", None) is None:
            return None
        checker = getattr(self.planner, "checker", None)
        is_grip = getattr(checker, "_is_gripper_geom", None)
        if is_grip is None:
            return None
        try:
            ncon = int(data.ncon)
        except Exception:  # noqa: BLE001
            return None
        out: list = []
        for c in data.contact[:ncon]:
            for g, other in ((int(c.geom1), int(c.geom2)), (int(c.geom2), int(c.geom1))):
                if not bool(is_grip[g]) or bool(is_grip[other]):
                    continue
                try:
                    bid = int(model.geom_bodyid[other])
                except Exception:  # noqa: BLE001
                    continue
                if not self._body_is_free(model, bid):
                    continue
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
                if name and str(name) not in out:
                    out.append(str(name))
        return out

    def _body_supported_by_static(self, body_name: Any) -> bool:
        """该物体当前是否仍与固定体（world 直接子体且无关节：桌面/地面/挡板）接触。

        "还压在支撑面上" = 没被抬起来。用于抬升后的随动证据。
        """
        env = self.env
        model = getattr(env, "model", None)
        data = getattr(env, "data", None)
        if model is None or data is None or getattr(data, "contact", None) is None:
            return False
        try:
            bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(body_name)))
            ncon = int(data.ncon)
        except Exception:  # noqa: BLE001
            return False
        if bid < 0:
            return False
        for c in data.contact[:ncon]:
            for g, other in ((int(c.geom1), int(c.geom2)), (int(c.geom2), int(c.geom1))):
                if int(model.geom_bodyid[g]) != bid:
                    continue
                try:
                    ob = int(model.geom_bodyid[other])
                    if int(model.body_parentid[ob]) == 0 and int(model.body_jntnum[ob]) == 0:
                        return True
                except Exception:  # noqa: BLE001
                    continue
        return False

    def _carry_evidence(self) -> tuple:
        """抬升后「物体随动」证据（P1-2）：夹爪还抓着可动物体，且它已不被支撑面托着。

        返回 (判定, 夹爪接触的物体名列表)。判定 None = 环境没有接触信息（测试桩），
        调用方**不得**据此否决（保持旧行为）。
        """
        bodies = self._gripper_contact_bodies()
        if bodies is None:
            return None, []
        free_of_support = [b for b in bodies if not self._body_supported_by_static(b)]
        return bool(free_of_support), list(bodies)

    def _freeze_arm(self, steps: int = 20) -> None:
        """【安全闸】停住：把当前 6 关节角写成位置目标并步进若干步。

        为什么必须先做这件事：move_to_pose 因 force_limit 中止时，data.ctrl 仍停在
        最后一条"继续压入"指令上。此时若直接去做下一次运动，而它恰好是出发前 IK
        预检失败（ik_fail，一步都不走），位置伺服会继续按旧 ctrl 往下压——实测
        0.196 s 内 |F| 冲到 7471.6 N，并把无关物体抛离桌面 1.39 m。
        因此任何中止/恢复路径在移动之前，都先把当前关节角设为目标以撤销压入。
        """
        q = self._arm_qpos_now()
        try:
            self.env.set_arm_joint_targets(q)
        except Exception as exc:  # noqa: BLE001
            self.log(f"[skills] 停住失败（set_arm_joint_targets）: {exc}")
            return
        try:
            self.env.step(max(1, int(steps)))
        except Exception as exc:  # noqa: BLE001
            self.log(f"[skills] 停住步进失败: {exc}")

    def _force_norm(self) -> float:
        """当前末端接触力模长（世界系，作用于 TCP）。"""
        return float(np.linalg.norm(self.env.contact_force_on_ee()[0]))

    @staticmethod
    def _unit(vec) -> np.ndarray:
        v = np.asarray(vec, dtype=float).reshape(-1)[:3]
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else np.zeros(3, dtype=float)

    @staticmethod
    def _escape_dir(f, idx: int) -> np.ndarray:
        """第 idx 个撤退候选方向（世界系单位向量），按**当前**反力 f 现算。

        方向符号（重要）：core 的 contact_force_on_ee() 返回**环境施加在末端上的力**
        （sim/env.py：方块压桌面时它给 +z 支撑），环境把末端往接触面外推，所以"退开"
        = 沿 **+f̂**；-f̂ 是顶进接触面，只会越压越死。
        优先级：+f̂ -> 正上方 -> (+f̂ + z) 对角 -> 两个水平切向。|f|≈0 时退化为纯向上。
        每次调用都基于新读到的力，所以"每步 _freeze_arm 后重新读力"是自然满足的。
        """
        up = np.array([0.0, 0.0, 1.0])
        f = np.asarray(f, dtype=float).reshape(-1)[:3]
        away = SkillExecutor._unit(f) if float(np.linalg.norm(f)) > 1e-6 else None
        cands: list[np.ndarray] = []
        if away is not None and float(np.linalg.norm(away)) > 1e-9:
            cands.append(away)
        cands.append(up)
        if away is not None:
            diag = SkillExecutor._unit(away + up)
            if float(np.linalg.norm(diag)) > 1e-9:
                cands.append(diag)
            tang = np.cross(up, away)
            if float(np.linalg.norm(tang)) > 1e-6:
                t = SkillExecutor._unit(tang)
                cands.append(t)
                cands.append(-t)
        uniq: list[np.ndarray] = []
        for d in cands:
            if not any(float(np.linalg.norm(d - e)) < 1e-6 for e in uniq):
                uniq.append(d)
        if not uniq:
            uniq = [up]
        d_out = uniq[min(max(int(idx), 0), len(uniq) - 1)]
        # 硬安全约束（task-68）：撤退方向的 z **不得为负**。
        # 实机日志：|F|=0.6N 的朝下噪声让 _retreat 主动向下顶 8cm -> 0.6N 变 32.3N（自己把臂顶到桌上）。
        if float(d_out[2]) < 0.0:
            flat = np.array([float(d_out[0]), float(d_out[1]), 0.0])
            d_out = (SkillExecutor._unit(flat) if float(np.linalg.norm(flat)) > 1e-9 else up)
        return d_out

    def _objects_touching_ee(self, exclude_held: bool = False,
                             contacts_only: bool = False) -> set:
        """当前与末端(夹爪)有接触的非机器人物体 id 集合（几何真值）。

        contacts_only=True（task-48/R-c 无真值模式）：只允许用**接触对**反查 id；
        拿不到 contact 信息时返回空集，绝不退化成"读物体位姿"。
        

        优先读 env 的接触信息（data.contact；sim 用 _ee_body_ids / _body_to_obj 反查物体）；
        读不到（如 stub、无 data）时退化为"TCP 与物体中心距离 <= 半对角线 + 0.03 m"。
        供 task-44 的几何脱身判据与受阻物上报使用。
        """
        data = getattr(self.env, "data", None)
        model = getattr(self.env, "model", None)
        body_to_obj = getattr(self.env, "_body_to_obj", None)
        if data is not None and model is not None and body_to_obj:
            try:
                ee = getattr(self.env, "_ee_body_ids", None)
                if ee is None:
                    ee = frozenset(
                        i for i in range(int(model.nbody))
                        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "")
                        .startswith("g_"))
                out = set()
                for i in range(int(data.ncon)):
                    c = data.contact[i]
                    b1 = int(model.geom_bodyid[c.geom1])
                    b2 = int(model.geom_bodyid[c.geom2])
                    if b1 in ee and b2 in body_to_obj:
                        out.add(str(body_to_obj[b2]))
                    if b2 in ee and b1 in body_to_obj:
                        out.add(str(body_to_obj[b1]))
                if exclude_held and self.holding:
                    out.discard(str(self.holding))   # 正常持物接触不算"受阻"
                return out
            except Exception:  # noqa: BLE001
                pass
        if contacts_only:
            return set()          # R-c：无真值模式不许退化成"读物体位姿"，拿不到就给空集
        try:
            tcp = np.asarray(self.env.site_pose(TCP_SITE)[0], float)
        except Exception:  # noqa: BLE001
            return set()
        near = set()
        for o in self.env.list_objects():
            half = float(np.linalg.norm(np.asarray(o.size, float)))
            if float(np.linalg.norm(np.asarray(o.pose[:3], float) - tcp)) <= half + 0.03:
                near.add(str(o.id))
        if exclude_held and self.holding:
            near.discard(str(self.holding))
        return near

    def _escaped(self, objs=None, steps: int = RETREAT_CLEAR_STEPS) -> bool:
        """task-44/R4：**几何**脱身判据——连续 steps 个物理步都不与关注物体接触才算成功。

        为什么废弃"瞬时 |F| <= 5 N"：松爪瞬间力会掉到 0，但指垫仍卡在物体上/下方，于是
        日志里出现"关节空间短抬升脱身成功"紧接着又 15.0 N 进入下一轮（假成功）。
        """
        if objs is None:
            # 未指定关注集合：要求"完全无接触"（持物接触也会算脏）
            watch = set(self._objects_touching_ee())
            require_empty = True
        else:
            # 指定了关注集合：只要求这些"受阻物"不再接触（例如持物时的正常接触不算）
            watch = {str(x) for x in objs}
            require_empty = False
        for _ in range(max(1, int(steps))):
            self.env.step()
            now = self._objects_touching_ee()
            if watch & now:
                return False
            if require_empty and now:
                return False
        return True

    def _obstruction(self, contacts_only: bool = False):
        """task-44/R5 + U1：受阻物观测（交回 LLM 重新规划用）。

        contacts_only=True（task-48/R-c 无真值模式）：只用接触对反查 id，拿不到给 None。
        返回 (metrics 片段, 可拼进 message 的说明)。读不到就给 None，不硬编码分支。
        """
        ids = sorted(self._objects_touching_ee(contacts_only=contacts_only))
        obs = {"obstruction_id": (ids[0] if ids else None),
               "obstruction_ids": ids,
               "obstruction_force": round(self._force_norm(), 3),
               "suggest_image": "wrist"}
        note = f"（受阻物: {ids[0]}）" if ids else ""
        return obs, note

    def _retreat(self, quat=None, dz: float = 0.08,
                 force_limit: Optional[float] = None,
                 threshold: float = 5.0) -> Optional[MoveResult]:
        """一次**温和**撤退：沿接触反力反方向(+f̂；力≈0 时向上)慢抬 dz。

        task-44/R1/R2/U1：删掉"5 方向 × 力上限逐级抬高"的暴力阶梯（实测 40→68.6 N、
        同一阶梯重复 4 轮且状态零变化）。现在只走一步，力上限封顶 RETREAT_FORCE_CAP_N=45 N；
        是否真的脱身由调用方用几何判据 _escaped()（连续 >=20 步无接触）确认。
        返回这一步的 MoveResult（可能失败），失败会由上层转成结构化观测交给 LLM。
        """
        self._freeze_arm(20)
        f = self.env.contact_force_on_ee()[0]
        f_norm = float(np.linalg.norm(f))
        if f_norm <= float(threshold):
            self._retreat_no_gain = False           # 已经松了，清零
        elif getattr(self, "_retreat_no_gain", False):
            # lead 要求（F3 安全）：上一次撤退**反而顶得更高** -> 不再重复顶（实测同一序列
            # 36.4 -> 72.9 -> 84.9 N 越退越高）。交给上层走关节抬升 / 结构化失败。
            self.log(f"[skills] 撤退无收益（上次撤退后力未降反升，|F|={f_norm:.1f}N），跳过再顶")
            return None
        # task-68 根因 1：力还没到"有意义接触"的量级（<= threshold）时，方向必须取**纯 +z**，
        # 绝不能照着噪声反力走——实机 |F|=0.6N（threshold=5.0，等于没接触）噪声朝下，
        # 旧代码取 +f̂ 就是朝下，主动下压 8cm：0.6N -> 32.3N，臂卡死，后续复核全废。
        direction = (np.array([0.0, 0.0, 1.0]) if f_norm <= float(threshold)
                     else self._escape_dir(f, 0))
        lim = min(RETREAT_FORCE_CAP_N,
                  max(RETREAT_MIN_FORCE_N, RETREAT_FORCE_HEADROOM * f_norm))
        if force_limit is not None:
            lim = min(lim, float(force_limit))
        pos, cur_quat = self.env.site_pose(TCP_SITE)
        q_keep = np.asarray(quat, float) if quat is not None else np.asarray(cur_quat, float)
        comp = ComplianceParams(k=RETREAT_COMPLIANCE.k, d=RETREAT_COMPLIANCE.d,
                                m=RETREAT_COMPLIANCE.m, axes=RETREAT_COMPLIANCE.axes,
                                force_limit=float(lim))
        self.log(f"[skills] 撤退：|F|={f_norm:.1f}N 方向 {np.round(direction, 3).tolist()} "
                 f"力上限 {lim:.1f}N（封顶 {RETREAT_FORCE_CAP_N:.0f}N）")
        last: Optional[MoveResult] = None
        try:
            last = self.runner.move_to_pose(np.asarray(pos, float) + direction * float(dz),
                                            q_keep, duration=float(RETREAT_DURATION_S),
                                            compliance=comp, timeout=10.0,
                                            force_limit=float(lim))
        except Exception as exc:  # noqa: BLE001
            self.log(f"[skills] 撤退异常: {exc}")
        finally:
            self._freeze_arm(20)
        # lead 要求：**每一步撤退后都查力**；如果没降反升，标记"无收益"，后续不再重复顶
        f_after = self._force_norm()
        self._retreat_no_gain = bool(f_after > f_norm + 1.0)
        if self._retreat_no_gain:
            self.log(f"[skills] 撤退无收益：|F| {f_norm:.1f} -> {f_after:.1f} N（不再重复顶）")
        return last

    def _recover(self, quat=None, threshold: float = 5.0) -> bool:
        """接触恢复：松爪 + **一次**温和撤退，用几何判据确认；不再多轮升级。

        为什么必须松爪：放置失败时被夹物可能卡在托盘边缘，抬升会把托盘一起拽起来
        （free-joint），力反而更大；松爪是唯一可靠的退出方式（实测）。
        task-44/R2：这里不做第二次、也不做方向阶梯——一次不行就交给上层当 structured
        失败观测（含 obstruction_id）交回 LLM，而不是自己换参数盲试。
        """
        # 判"已恢复"必须同时满足两条：
        #   1) 几何上不再与被挡物体接触（_escaped，连续 >=20 步）；**且**
        #   2) 末端接触力确实落回安全值（|F| <= threshold）。
        # 只有几何判据是不够的：压桌面/压托盘这类接触反查不到"场景物体"（_objects_touching_ee
        # 只映射物体），会误判成"干净"，于是 hard_recover 第 1 级假成功、脏状态没清掉，
        # 后续抓取连锁失败（tests/test_interactive_flows.py TestDirtyStateAutoRecover）。
        f_now = self._force_norm()
        obstructions = self._objects_touching_ee(exclude_held=True)
        if not obstructions and f_now <= float(threshold):
            return True                     # 干净（或只有正常持物接触）
        self.env.set_gripper(GRIPPER_OPEN)
        self._settle(0.2)
        self._retreat(quat, dz=0.08)
        f_now = self._force_norm()
        ok = self._escaped(obstructions) and f_now <= float(threshold)
        self.log(f"[skills] 恢复{'成功' if ok else '未清空'}：关注 {sorted(obstructions)}，"
                 f"|F|={f_now:.1f} N（阈值 {float(threshold):.1f}）")
        return ok

    def _escape_contact(self, quat=None, obj_id: Optional[str] = None,
                        threshold: float = 5.0, dz: float = 0.12) -> bool:
        """失败终态清场（task-44/R4 几何判据）：松爪 -> 一次温和撤退 -> 一次短距离关节抬升。

        与旧版的关键差别：
          * 不再有 5 方向 × 力上限逐级抬高的阶梯（force_limit 一律 <= RETREAT_FORCE_CAP_N=45）；
          * 不再用"瞬时 |F| <= 5 N"判成功（松爪瞬间力必然掉到 0，是假成功），
            改用 _escaped()：连续 RETREAT_CLEAR_STEPS(20) 步都不与该物体有接触才算脱身。
        返回是否几何脱身；级别记在 self.last_escape_level（'clear'|'retreat'|'lifted'|'blocked'）。
        """
        self._freeze_arm(20)
        objs = set(self._objects_touching_ee())
        # 只在"目标物体确实在接触里 / 还被夹着"时才把它纳入关注集合；否则不该凭空发起撤退
        if obj_id is not None and str(self.holding or "") == str(obj_id):
            objs.add(str(obj_id))
        if not objs:
            self.last_escape_level = "clear"
            return self._escaped()
        self.log(f"[skills] 终态清场：与 {sorted(objs)} 接触，松爪 + 一次温和撤退")
        self.env.set_gripper(GRIPPER_OPEN)
        self._settle(0.2)
        self._retreat(quat, dz=0.08)
        if self._escaped(objs):
            self.last_escape_level = "retreat"
            return True

        # 仍不干净：一次短距离关节空间抬升（带速度前馈、不被力中止，实测是真能出来的动作）
        self.log(f"[skills] 终态清场：仍与 {sorted(self._objects_touching_ee())} 接触，"
                 f"试一次抬高 {dz * 1000:.0f}mm 的关节位形")
        q_lift = None
        try:
            pos, cur_quat = self.env.site_pose(TCP_SITE)
            q_sol, ok = self.kin.ik(np.asarray(pos, float) + np.array([0.0, 0.0, float(dz)]),
                                    quat if quat is not None else cur_quat)
            if ok:
                q_lift = np.asarray(q_sol, dtype=float).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            self.log(f"[skills] 终态清场 IK 失败: {exc}")
        if q_lift is not None:
            try:
                path = None
                try:
                    with self._ignore_contacts_with(obj_id if obj_id is not None else self.holding):
                        path = self.planner.plan_joint(q_lift)
                except Exception as exc:  # noqa: BLE001
                    self.log(f"[skills] 终态清场规划异常: {exc}")
                if path is not None:
                    self.runner.follow_joint_path(path, max_step_rad=0.12, timeout=40.0,
                                                  compliance=STIFF_COMPLIANCE)
                else:
                    self.log("[skills] 终态清场：plan_joint 失败，直接关节插值抬升")
                    self.runner.follow_joint_path([q_lift], max_step_rad=0.12, timeout=20.0,
                                                  compliance=STIFF_COMPLIANCE)
            except Exception as exc:  # noqa: BLE001
                self.log(f"[skills] 终态清场执行异常: {exc}")
        self._freeze_arm(20)
        ok = self._escaped(objs)
        self.last_escape_level = "lifted" if ok else "blocked"
        if not ok:
            self.log(f"[skills] 终态清场未清空：仍与 {sorted(self._objects_touching_ee())} 接触")
        return ok

    def _finish_failure(self, quat=None, obj_id: Optional[str] = None,
                        contacts_only: bool = False):
        """失败终态的统一收尾：先几何脱身；**确实清不掉才**重置场景，并如实上报原因。

        task-44/R5 + U1：返回值里带 obstruction_id / obstruction_force / suggest_image，
        供上层把结构化观测交回 LLM 重新规划（而不是技能层自己换参数继续试）。
        返回 (extra_metrics, error_code_override, message_suffix)。
        """
        # R5：先在"还卡着"的时刻取受阻物观测——脱身成功之后就什么都看不到了
        obs, note = self._obstruction(contacts_only=contacts_only)
        cleared = self._escape_contact(quat, obj_id)
        level = str(getattr(self, "last_escape_level", "?"))
        extra = {"escaped": bool(cleared), "force": obs["obstruction_force"],
                 "escape_level": level, **obs}
        self.holding = None
        if cleared:
            return extra, None, note
        self.log("[skills] 温和脱身与关节抬升都未清空，最后手段：重置场景")
        self.env.reset(seed=0)
        self.holding = None
        return extra, "scene_reset", f"（脱身未清空{note}，已重置场景）"

    def _ensure_clear(self, quat=None, threshold: float = 5.0,
                      obj_id: Optional[str] = None) -> bool:
        """接触入口检查：不干净时**最多一次温和撤退**；仍不干净立即返回 False。

        task-44/R2：抓取入口遇到 14.8~15.1 N 残留时，旧实现会进入"5 方向 × 多轮"的暴力
        阶梯（力上限 40→68.6 N、同一阶梯重复 4 轮、状态零变化）。现在只允许一次温和撤退；
        失败就把带 obstruction_id 的结构化观测交回上层/LLM，而不是技能层自己继续挣扎。
        """
        obstructions = self._objects_touching_ee(exclude_held=True)
        if not obstructions:
            return True                     # 干净（或只有正常持物接触）
        self.log(f"[skills] 入口不干净：与 {sorted(obstructions)} 接触，做一次温和撤退")
        ok = self._recover(quat, threshold=threshold)
        if not ok:
            self.log("[skills] 温和撤退后仍不干净，立即返回 stuck（不再进入升级阶梯）")
        return ok

    def hard_recover(self, quat=None, reset_if_force_above: float = 30.0) -> str:
        """【应急】三级强制脱困，返回实际使用的级别: 'retreat' | 'home_joint' | 'scene_reset'。

        这是交互式使用中的**最后手段**，不是常规失败恢复路径（常规路径是 _recover /
        _retreat）。只有在机械臂已被压死、笛卡尔脱身无效、接触力达到数百牛时，
        UI/上层才应调用本方法。

        级别
        ----
        1) 'retreat'     常规抬升脱身（_recover，含松爪后二次抬升）成功，力已降到安全阈值；
        2) 'home_joint'  抬不动时用规划器直接回 HOME_QPOS 关节位形（绕开笛卡尔 IK），
                         执行后若末端力 <= reset_if_force_above 则视为已解除；
        3) 'scene_reset' 兜底：env.reset(seed=0) **重置整个场景**（物体回初始位置、
                         机械臂回 HOME_QPOS），并清空 self.holding。此级会丢失当前
                         任务进度，仅用于卡死救不回的情况。

        安全约定：三级在移动/重置前都会先 _freeze_arm（把当前关节角设为目标并步进），
        任何一级失败都不得留下"继续压入"的 ctrl 指令。

        参数
        ----
        quat : 可选，脱身时保持的姿态；None 表示沿用当前姿态。
        reset_if_force_above : 第 2 级之后判定"是否仍需重置场景"的力阈值（N）。
        """
        # 1) 常规脱身
        try:
            if self._recover(quat):
                return "retreat"
        except Exception as exc:  # noqa: BLE001 - 应急入口不允许把异常抛给 UI
            self.log(f"[skills] hard_recover 第1级异常: {exc}")

        # 2) 关节空间回 HOME（关节路径不因既有接触力中止，适合"压死"状态）
        try:
            path = self.planner.plan_joint(HOME_QPOS)
            if path is not None:
                self.log("[skills] hard_recover 第2级：follow_joint_path -> HOME_QPOS")
                self.runner.follow_joint_path(path, max_step_rad=0.12, timeout=40.0,
                                              compliance=STIFF_COMPLIANCE)
                force, _ = self.env.contact_force_on_ee()
                f_norm = float(np.linalg.norm(force))
                if f_norm <= float(reset_if_force_above):
                    return "home_joint"
                self.log(f"[skills] 回 HOME 后仍受 {f_norm:.1f} N "
                         f"(> {float(reset_if_force_above):.1f} N)，转入场景重置")
            else:
                self.log("[skills] hard_recover 第2级：plan_joint 返回 None")
            self._freeze_arm(20)          # 第 2 级未解除：先停住，不留压入指令
        except Exception as exc:  # noqa: BLE001
            self.log(f"[skills] hard_recover 第2级异常: {exc}")
            self._freeze_arm(20)

        # 3) 兜底：先停住，再重置场景（会清空任务进度）
        self._freeze_arm(20)
        self.env.reset(seed=0)
        self.holding = None
        return "scene_reset"

    def _compliance_from_call(self, call: SkillCall, default: ComplianceParams) -> ComplianceParams:
        if call.compliance is not None:
            return call.compliance
        raw = (call.args or {}).get("compliance")
        return self._resolve_profile(raw, default) or default

    def _resolve_profile(self, raw: Any, default: ComplianceParams) -> Optional[ComplianceParams]:
        if raw is None:
            return None
        if isinstance(raw, str):
            return PROFILES.get(raw)
        if isinstance(raw, dict):
            base = PROFILES.get(str(raw.get("profile", "")), None)
            if base is None and "k" in raw:
                return ComplianceParams(
                    k=np.array(raw["k"], float), d=np.array(raw["d"], float),
                    m=np.array(raw["m"], float), axes=np.array(raw["axes"], float),
                    force_limit=float(raw.get("force_limit", default.force_limit)),
                )
            if base is None:
                base = default
            return base.scaled(float(raw.get("k_scale", 1.0)))
        return None

    def _goto_planned(self, pos, quat=None, compliance=None, timeout: float = 40.0) -> MoveResult:
        """先用规划器做无碰撞接近，再用笛卡尔把末端精确拉到目标位姿。"""
        pos = np.asarray(pos, float)
        comp = compliance if compliance is not None else STIFF_COMPLIANCE
        path = None
        try:
            path = self.planner.plan_pose(pos, quat)
        except Exception as exc:  # noqa: BLE001
            self.log(f"[skills] plan_pose 异常，退化到笛卡尔: {exc}")
        if path is not None:
            r = self.runner.follow_joint_path(path, max_step_rad=0.12, timeout=timeout, compliance=comp)
            if r.ok:
                return self.runner.move_to_pose(pos, quat, compliance=comp, timeout=timeout)
            self.log(f"[skills] 关节路径执行失败({r.error_code}: {r.message})，退化到笛卡尔")
        return self.runner.move_to_pose(pos, quat, compliance=comp, timeout=timeout)

    def _check_force(self, res: MoveResult, limit: float) -> Optional[SkillResult]:
        if res.ok:
            return None
        if res.error_code == "force_limit":
            return SkillResult(False, "pick", f"接触力超限 ({res.max_force:.2f} N)", "force_limit",
                               metrics={"max_force": res.max_force})
        return None

    # ------------------------------------------------------------------ #
    # 技能
    # ------------------------------------------------------------------ #
    def _skill_home(self, call: SkillCall) -> SkillResult:
        """回初始位姿（关节路径优先；瞬移兜底有前置条件）。

        S4：plan_joint 失败时不再直接瞬移 env.home()——若此刻还压着/夹着，位置伺服
        会把状态瞬间拉到 HOME，产生极大接触力（实测可抛飞物体）。先停住并 _retreat
        抬起，只有"接触力接近 0 且夹爪已张开"才允许瞬移，否则返回 home_blocked。
        S3：结束时按夹爪实际开合重判 holding（>= 0.8 视为已松开 -> None），避免 UI
        的 :state 显示的 holding 与实际夹爪不一致。
        N1：带载归位——夹爪与手持物体的持续接触会让 CollisionChecker 把起始位形判非法，
        于是 plan_joint 返回 None、home 被 home_blocked 拒绝。此时在规划窗口临时忽略
        「手持物 <-> 夹爪」接触对后重试 plan_joint；仍失败才回退到 home_blocked 安全门。
        """
        path = self.planner.plan_joint(HOME_QPOS)
        if path is None and self.holding is not None:
            self.log(f"[skills] 带载归位：holding={self.holding}，忽略手持物接触后重试规划")
            with self._ignore_contacts_with(self.holding):
                path = self.planner.plan_joint(HOME_QPOS)
            if path is not None:
                self.log("[skills] 带载归位：重试规划成功，走关节路径回 HOME")
        if path is not None:
            r = self.runner.follow_joint_path(path, max_step_rad=0.12, timeout=40.0,
                                              compliance=STIFF_COMPLIANCE)
        else:
            # 瞬移前的前置条件：先撤销压入指令并抬起
            self._freeze_arm(20)
            self._retreat(None)
            force, _ = self.env.contact_force_on_ee()
            f_norm = float(np.linalg.norm(force))
            gripper = self._gripper_now()
            # task-68 根因 2：原门要求 gripper >= GRIPPER_OPEN_MIN(0.80)，但手里夹着 45.5mm 方块时
            # 开合恒为 0.44 -> 带载归位**永远**被拒（与 task-13/14「夹着 box_blue 也应成功归位」冲突）。
            # 现在：无载荷时旧门原样（开合必须 >= 0.80）。
            # P0 修复（带载禁瞬移）：带载（holding 非 None，含 'unnamed'）时**一律禁止**瞬移兜底
            # —— 旧实现"力安全就放行"会让机械臂瞬间跑到 0.6 m 外，手里的物体被拖着走/丢下
            # （实测方块被丢回桌面原位 -> 复核 subject_still_present）。带载只能走关节路径，
            # 走不了就诚实返回 home_blocked，由上层保留持物状态。
            loaded = self.holding is not None and gripper < GRIPPER_OPEN_MIN
            grip_gate_ok = (gripper >= GRIPPER_OPEN_MIN) or loaded
            if loaded:
                why = (f"带载（holding={self.holding}）禁止瞬移归位：瞬移会把手里夹着的物体"
                       f"拖走/丢下，必须走关节路径或先放下物体")
            elif f_norm > HOME_SAFE_FORCE_N or not grip_gate_ok:
                why = (f"接触力 {f_norm:.1f} N（阈值 {HOME_SAFE_FORCE_N:.1f} N）/ "
                       f"夹爪开合 {gripper:.2f}（需 >= {GRIPPER_OPEN_MIN:.2f}）")
            else:
                why = ""
            if why:
                self.log(f"[skills] 归位被阻塞：{why}，拒绝 env.home() 瞬移")
                self._freeze_arm(20)
                return SkillResult(False, "home",
                                   f"无法安全归位：{why}",
                                   "home_blocked",
                                   metrics={"force": f_norm, "gripper": gripper,
                                            "holding": self.holding, "loaded": loaded})
            self.log("[skills] 归位预检通过（力≈0 且夹爪已张开），env.home() 瞬移")
            self.env.home()
            r = MoveResult(True, "env.home()")

        # S3: holding 卫生——按夹爪实际开合重判
        gripper_now = self._gripper_now()
        if gripper_now >= GRIPPER_OPEN_MIN:
            self.holding = None
        return SkillResult(bool(r.ok), "home", r.message or "已回到初始位姿",
                           r.error_code,
                           metrics={"steps": r.steps, "gripper": gripper_now,
                                    "holding": self.holding})

    def _skill_move_to(self, call: SkillCall) -> SkillResult:
        args = call.args or {}
        if "pos" not in args:
            return SkillResult(False, "move_to", "缺少 pos", "bad_args")
        pos = np.asarray(args["pos"], float)
        quat = np.asarray(args["quat"], float) if args.get("quat") else None
        comp = self._compliance_from_call(call, STIFF_COMPLIANCE)
        planned = args.get("planned", False)
        r = self._goto_planned(pos, quat, comp) if planned else \
            self.runner.move_to_pose(pos, quat, compliance=comp, timeout=float(args.get("timeout", 10.0)))
        return SkillResult(bool(r.ok), "move_to", r.message or "到位", r.error_code,
                           metrics={"max_force": r.max_force, "steps": r.steps})

    def _skill_open_gripper(self, call: SkillCall) -> SkillResult:
        self.env.set_gripper(1.0)
        self._settle(0.3)
        self._refresh_holding_if_released()      # N2：松爪后刷新 holding
        return SkillResult(True, "open_gripper", "夹爪已张开",
                           metrics={"holding": self.holding, "gripper": self._gripper_now()})

    def _skill_close_gripper(self, call: SkillCall) -> SkillResult:
        ratio = float((call.args or {}).get("ratio", GRIPPER_CLOSE_HOLD))
        self.env.set_gripper(ratio)
        self._settle(0.6)
        return SkillResult(True, "close_gripper", f"夹爪闭合到 {ratio:.2f}")

    def _skill_set_compliance(self, call: SkillCall) -> SkillResult:
        args = call.args or {}
        profile = str(args.get("profile", "soft"))
        base = PROFILES.get(profile)
        if base is None:
            return SkillResult(False, "set_compliance", f"未知档位: {profile}", "bad_args")
        self.compliance = base.scaled(float(args.get("k_scale", 1.0)))
        return SkillResult(True, "set_compliance", f"已切换到 {profile}", metrics=self.compliance.to_json())

    def _skill_stop(self, call: SkillCall) -> SkillResult:
        return SkillResult(True, "stop", (call.args or {}).get("reason", "任务结束"))

    def _skill_ask_user(self, call: SkillCall) -> SkillResult:
        q = (call.args or {}).get("question", "需要用户确认")
        return SkillResult(False, "ask_user", f"需要确认: {q}", "ask_user")

    # ---------------- 抓取 ---------------- #
    def _guarded_close(self, ratio: float, threshold: float = PICK_CLOSE_FORCE_MAX_N,
                       total: float = 0.6, block: float = 0.02) -> bool:
        """guarded close（task-28 要求 2）：分块合爪并监测接触力，超阈值就停在当前开合度。

        为什么需要：旧的"set_gripper(ratio) + 静置 0.6 s"会一路捏到底，遇到姿态不贴合
        的物体时指垫会把它推走/挤飞（实测重试反而把物体越推越远）。返回是否合到目标。
        """
        self.env.set_gripper(float(ratio))
        n_block = max(1, int(round(float(block) / self.env.dt)))
        elapsed = 0.0
        f_seen = 0.0                      # 合爪过程见到的最大力（正面证据用）
        self.last_close_force = 0.0
        while elapsed < float(total):
            self.env.step(n_block)
            elapsed += n_block * self.env.dt
            f = self._force_norm()
            f_seen = max(f_seen, float(f))
            self.last_close_force = f_seen
            if f > float(threshold):
                held = self._gripper_now()
                self.env.set_gripper(held)          # 停在此处，不再继续捏
                self.log(f"[skills] guarded close：{f:.1f} N 超阈值 {threshold:.1f} N，"
                         f"停在开合 {held:.2f}（目标 {ratio:.2f}）")
                return False
        return True

    def _skill_pick(self, call: SkillCall) -> SkillResult:
        """抓取。两条互斥入口：
          * target_id（oracle 路径，判据=抬升位移 > 20 mm，语义与此前一致）；
          * target_xy（无真值路径，§4.2；判据=夹爪实测指间距，见 R-b）。

        grasp_profile（§5.4）：
          * "auto"（默认）/ "top" —— 自上而下。**水瓶/圆柱也默认走 top**：2F85 行程 85 mm、
            水瓶直径 64 mm，顶抓夹住上半身完全可行；实测顶抓在默认场景里最稳。
          * "side" —— 侧面水平夹持（圆柱/瓶身），夹持高度取 target_z_hint，接近方向按预抓取位
            IK 可解性自动挑。**已知限制**：需要目标周围 >= 10 cm 水平净空，否则横向推进会把物体
            推走（oracle 模式下净空不足会自动退回 top 并在 metrics.side_fallback_reason 说明；
            无真值模式无法测量净空，由调用方负责保证）。这里不做复杂力控。
          * close_ratio 缺省：有 size/物体时按宽度反算；没有时按类别先验
            （方块 0.35 / 圆柱 0.55 / 水瓶 0.75，可选 args.target_category）。
        """
        args = call.args or {}
        tid = args.get("target_id")
        target_xy_arg = args.get("target_xy")
        # ---- 互斥：target_id(旧, oracle) vs target_xy(无真值, INTERFACES §4.2) ----
        if tid is not None and str(tid) != "" and target_xy_arg is not None:
            return SkillResult(False, "pick",
                               "target_id 与 target_xy 互斥，只能给一个", "bad_args")
        no_oracle = target_xy_arg is not None
        target_point = None
        half_size = np.asarray(PICK_DEFAULT_HALF, float)
        if no_oracle:
            # R-a：目标点**只来自入参**，不读 scene.objects[...].pose 当真值
            xy = np.asarray(target_xy_arg, float).reshape(-1)
            if xy.size < 2:
                return SkillResult(False, "pick", "target_xy 需要 [x, y]", "bad_args")
            xy = xy[:2]
            size_arg = args.get("target_size")
            if size_arg is not None:
                size_v = np.asarray(size_arg, float).reshape(-1)
                if size_v.size >= 3:
                    half_size = size_v[:3]
                elif size_v.size == 2:          # 允许 (sx, sy)，高度用默认
                    half_size = np.array([size_v[0], size_v[1], PICK_DEFAULT_HALF[2]])
            z_hint = args.get("target_z_hint")
            if z_hint is None:
                z_center = float(TABLE_HEIGHT) + float(half_size[2])
            elif str(args.get("grasp_profile", "") or "").lower() == "side":
                # side 保持原语义：夹持高度就是可见点（侧面水平夹持，另有 12mm 高度护栏）
                z_center = float(z_hint)
            else:
                # lead 统一口径（真回归修复）：模型/taskplan 给的 z 现在是**顶面**（真值核对过），
                # 所以无真值 **top** 抓取一律 夹持高度 = 顶面下方 PLACE_SLENDER_TOP_OFFSET_M，
                # **方块/圆柱/细高都用这一条**（细高分支后面的公式与此等价，保持不变）。
                # 之前方块分支直接把 z_hint 当夹持高度 -> TCP 停在顶面上方 16mm -> 合在空气里
                # （真机实测：target_z_hint=0.4567 停 0.456，指间距=命令值 38mm，没碰到方块）。
                z_center = max(float(z_hint) - float(PLACE_SLENDER_TOP_OFFSET_M),
                               float(TABLE_HEIGHT) + float(PLACE_SLENDER_FLOOR_MIN_M))
            target_point = np.array([float(xy[0]), float(xy[1]), z_center])
            obj = None
            tid = None
        else:
            if not tid:
                return SkillResult(False, "pick",
                                   "缺少 target_id，或给 target_xy（无真值）", "bad_args")
            obj = self._object(tid)
            if obj is None:
                return SkillResult(False, "pick", f"物体不存在: {tid}", "object_not_found")
            if not obj.graspable:
                return SkillResult(False, "pick", f"物体不可抓取: {tid}", "not_graspable")
            half_size = np.asarray(obj.size, float)

        # task-59/§5.4：grasp_profile。auto 保持既有 top-down 行为（oracle 路径语义不变）；
        # "side" = 从侧面水平接近夹持（圆柱/瓶身），夹持高度取 target_z_hint。
        profile = str(args.get("grasp_profile", "auto") or "auto").strip().lower()
        if profile not in GRASP_PROFILES:
            return SkillResult(False, "pick",
                               f"不支持的 grasp_profile: {profile}（可用 {'/'.join(GRASP_PROFILES)}）",
                               "bad_args")
        side_grasp = (profile == "side")
        side_fallback_reason: Optional[str] = None
        approach_dir = np.asarray(SIDE_APPROACH_DIR if side_grasp else (0.0, 0.0, -1.0), float)

        def _attempt_quat(roll: float) -> np.ndarray:
            """本次尝试的 TCP 姿态：top 沿 -z 俯冲，side 沿 +x 水平接近；roll 绕接近轴。"""
            return (approach_quat_dir(approach_dir, roll) if side_grasp
                    else approach_quat(roll))

        approach = float(args.get("approach_height", APPROACH_HEIGHT))
        base_yaw = float(args.get("yaw", 0.0))
        # 类别（无真值时由调用方给；圆柱/水瓶默认 top 抓上半身）
        cat = str(args.get("target_category", "") or "").strip().lower()
        if side_grasp:
            # 侧面接近方向按"预抓取位 IK 可解"自动挑（不能写死，见 _choose_side_dir）
            side_ref = (np.asarray(target_point, float) if no_oracle
                        else np.asarray(obj.pose[:3], float))
            if args.get("target_z_hint") is not None:
                side_ref = np.asarray(side_ref, float).copy()
                side_ref[2] = float(args["target_z_hint"])
            approach_dir = self._choose_side_dir(
                side_ref, approach=approach, roll=base_yaw,
                exclude_id=(None if no_oracle else (str(tid) if tid else None)),
                use_clearance=(not no_oracle))   # 无真值模式禁止读物体位姿(§5.7)
            if not no_oracle:
                # lead 裁决：side 只在净空 >=10cm 时用，否则退回 top（否则会把物体推走）
                clear = self._side_clearance(
                    side_ref, half_self=float(max(half_size[0], half_size[1])),
                    exclude_id=str(tid) if tid else None)
                if clear < float(PLACE_SIDE_MIN_CLEARANCE_M):
                    side_fallback_reason = (
                        f"side 需要 >= {PLACE_SIDE_MIN_CLEARANCE_M * 1000:.0f} mm 净空，"
                        f"实测仅 {clear * 1000:.0f} mm -> 本次退回 top")
                    self.log(f"[skills] {side_fallback_reason}")
                    side_grasp = False
                    approach_dir = np.asarray((0.0, 0.0, -1.0), float)
            # P0（2910 N 安全红线）：预抓取/推进高度离桌面不足就**拒绝进入侧向推进段**，
            # 退回 top-down。只用到调用方给的高度（无真值模式同样适用），不读任何物体真值。
            z_grasp = float(side_ref[2])
            if side_grasp and z_grasp <= float(TABLE_HEIGHT) + float(PLACE_SIDE_FLOOR_CLEARANCE_M):
                side_fallback_reason = (
                    f"side 夹持高度 z={z_grasp:.3f} 距桌面仅 "
                    f"{(z_grasp - float(TABLE_HEIGHT)) * 1000:.0f} mm（需 >= "
                    f"{PLACE_SIDE_FLOOR_CLEARANCE_M * 1000:.0f} mm）-> 拒绝侧向推进，退回 top-down")
                self.log(f"[skills] {side_fallback_reason}")
                side_grasp = False
                approach_dir = np.asarray((0.0, 0.0, -1.0), float)
            self.log(f"[skills] 侧面夹持方向 {np.round(approach_dir, 2).tolist()}"
                     f"（按预抓取位 IK 可解性选）")
        quat = _attempt_quat(base_yaw)
        # 对准下降只用 z 向柔顺、xy/姿态保持刚性；VLM 的柔顺意图只作用于抬升段。
        intent = self._compliance_from_call(call, GRASP_COMPLIANCE)
        descend = ComplianceParams(k=GRASP_COMPLIANCE.k, d=GRASP_COMPLIANCE.d,
                                   m=GRASP_COMPLIANCE.m, axes=GRASP_COMPLIANCE.axes,
                                   force_limit=max(float(intent.force_limit),
                                                   float(GRASP_COMPLIANCE.force_limit)))
        soft = self._motion_compliance(call, SOFT_COMPLIANCE)

        # 无真值模式用尺寸估计反算闭合量；没有估计就用默认保持值。
        computed_ratio = (self._close_ratio_for_size(half_size) if no_oracle
                          else self.close_ratio_for(obj))
        # 上层（含 VLM）给的闭合量只作为"更紧"的上限：更紧不会夹空，更松会（实测同一个
        # 0.35 对 box_red 可行、对 box_green 夹空）。取两者较小值。
        # task-59：完全没有尺寸信息时（无真值 + 没给 target_size）改用**类别先验**：
        # 方块 0.35 / 圆柱 0.55 / 水瓶 0.75；给了 size 或走 oracle(有物体尺寸) 时仍按宽度反算，
        # 因此既有 oracle 语义不变。
        base_ratio = computed_ratio
        if args.get("close_ratio") is not None:
            base_ratio = min(float(args["close_ratio"]), computed_ratio)
        elif no_oracle and args.get("target_size") is None:
            default_cat = "cylinder" if side_grasp else "box"
            base_ratio = float(GRASP_CLOSE_PRIOR.get(cat or default_cat,
                                                     GRASP_CLOSE_PRIOR_DEFAULT))
            self.log(f"[skills] 闭合量用类别先验 {cat or default_cat} -> {base_ratio:.2f}")
        half_h = float(half_size[2])
        # 多次尝试（re-grasp attempts）：UR5e 位置伺服无重力补偿，末端稳态下沉量随位形
        # 变化可达 ~1 cm，同一套参数在不同摆放位置会时好时坏（实测 box_red 恒成、box_green
        # 需要更深 2 mm、can_yellow 需更浅+更紧）。真实机器人上就是用"抓取深度 + 闭合量"
        # 的组合重试来处理这种不确定性。
        # task-44/R3 + U1：盲重试网格从 10 组压到 PICK_ATTEMPT_PLAN 的 3 组（见常量注释：
        # 深压会顶桌面收益低、真正有效的是"标准深度 / 略浅 / 90° 偏航"）。3 组用完仍失败时，
        # 把结构化失败观测（error_code + obstruction_id + 力 + 建议图像）交回 LLM 重新规划，
        # 而不是在这里继续换参数盲试。
        attempts: list[tuple[float, float, float]] = [
            (frac * half_h, float(np.clip(base_ratio + off, 0.0, 1.0)), base_yaw + yaw)
            for frac, off, yaw in PICK_ATTEMPT_PLAN
        ][:int(PICK_MAX_ATTEMPTS)]
        self.log(f"[skills] 抓取重试覆盖：{len(attempts)} 组 "
                 f"(深度分数 x 闭合量偏移 x 偏航, 上限 {PICK_MAX_ATTEMPTS})")

        if not self._ensure_clear(quat, obj_id=tid):
            extra, override, note = self._finish_failure(quat, tid,
                                                         contacts_only=no_oracle)
            return SkillResult(False, "pick", "机械臂卡在接触中，无法开始抓取" + note,
                               override or "stuck", metrics=extra)
        anchor = obj.pose[:3].copy() if not no_oracle else None   # 旧路径用它检测"被碰跑"
        last_moved, last_force = 0.0, 0.0
        last_gap_mm = None
        last_commanded_gap_mm = None
        last_gap_vs_size_mm = None
        last_gap_over_size_mm = None
        last_close_contact = None
        last_close_force_n = None
        last_gap_positive = None
        # P1：抬升是否成功 / 抬升后随动证据（失败路径也要能观测，见下面 metrics）
        lift_ok = None
        carry_ok = None
        carry_bodies: list = []
        # 重要：这两行必须在**循环外**先赋默认值——循环里每次迭代会重新赋值，但如果每次都在
        # 下降段之前就 continue（例如 side 回退 top-down 后又被挡），循环结束后那个收尾失败的
        # 返回会引用到未绑定的 descend_metrics，抛 UnboundLocalError（verifier F3 实测，
        # 真机拿到的是 "exception" 而不是语义化错误码）。
        descend_metrics: dict = {}
        slender_now = False
        reasons: list[str] = []                      # 每次尝试的失败原因（用于给出明确失败信息）
        for attempt, (dz, ratio, yaw) in enumerate(attempts, 1):
            attempt_quat = _attempt_quat(float(yaw))
            if no_oracle:
                # R-a：始终用入参给的目标点（+ 深度分数），不读任何物体真值位姿
                before = None
                target = np.asarray(target_point, float) - approach_dir * float(dz)
                # 细高判据（与下降段同一条 OR 证据）：类别+半高 **或** 可见点高出桌面 >= 80mm。
                # 注意：live 形状没有 category、size 只有 2 元，只能靠几何证据，所以这里不能再包在
                # "cat in (cylinder,bottle)" 里（上一版就是这么漏掉的）。
                _zh = (float(args["target_z_hint"])
                       if args.get("target_z_hint") is not None else float(target[2]))
                _hh = float(half_size[2])
                _slender_hint = ((cat in ("cylinder", "bottle", "can")
                                  and _hh >= float(PICK_SLENDER_HALF_HEIGHT_M))
                                 or ((_zh - float(TABLE_HEIGHT))
                                     >= float(PLACE_SLENDER_HINT_HEIGHT_M)))
                if _slender_hint:
                    # 方案 B（lead 修正语义）：z_hint 现在是**顶面**（live 证据 0.613951 ≈ 真值顶面 0.60），
                    # 夹持高度 = 顶面下方 PLACE_SLENDER_TOP_OFFSET_M。**不再对 z_hint 加 0.05**：
                    # 旧语义("z 是中部")那版会算到 0.6643 = 瓶顶上方 64mm，两指合在空气里 -> 抓空。
                    # 下限保护：不低于桌面 + PLACE_SLENDER_FLOOR_MIN_M（兼容 hint 偏低的旧语义）。
                    target[2] = max(float(_zh) - float(PLACE_SLENDER_TOP_OFFSET_M),
                                    float(TABLE_HEIGHT) + float(PLACE_SLENDER_FLOOR_MIN_M))
                elif cat in ("cylinder", "bottle"):
                    # 矮圆柱/水瓶：不低于桌面 + PLACE_MIN_GRASP_HEIGHT_M
                    target[2] = max(float(target[2]),
                                    float(TABLE_HEIGHT) + float(PLACE_MIN_GRASP_HEIGHT_M))
            else:
                obj = self._object(tid)              # 每次重试重新读物体位姿
                if obj is None:
                    return SkillResult(False, "pick", f"物体已不在场景中: {tid}",
                                       "object_not_found")
                drift = float(np.linalg.norm(obj.pose[:3] - anchor))
                if attempt > 1 and drift > 0.03:     # 已被碰跑：继续试只会越搞越乱
                    extra, override, note = self._finish_failure(quat, tid)
                    return SkillResult(False, "pick",
                                       f"夹取失败且物体已被推移 {drift * 1000:.0f} mm，停止重试" + note,
                                       override or "object_disturbed",
                                       metrics={"moved": last_moved, "max_force": last_force,
                                                "attempts": attempt - 1, "drift": drift, **extra})
                before = obj.pose.copy()
                base_pt = np.asarray(before[:3], float).copy()
                # 高度取 target_z_hint：side 侧面夹持高度 / 圆柱·水瓶抓上半身（top）
                if args.get("target_z_hint") is not None and (
                        side_grasp or cat in ("cylinder", "bottle")):
                    base_pt[2] = float(args["target_z_hint"])
                target = base_pt - approach_dir * float(dz)

            self.env.set_gripper(GRIPPER_OPEN)
            self._settle(0.25)
            pre = target - approach_dir * float(approach)   # top: 上方；side: 侧向后方
            if side_grasp:
                # 侧面夹持分两段接近：先在"预抓取位正上方 approach"落脚（避开贴桌面横扫），
                # 再竖直到预抓取高度，最后由下面的下降段沿水平轴推进到目标。
                staging = np.array([pre[0], pre[1], pre[2] + float(approach)])
                r = self._goto_planned(staging, attempt_quat, STIFF_COMPLIANCE)
                if r.ok:
                    r = self.runner.move_to_pose(pre, attempt_quat, compliance=STIFF_COMPLIANCE,
                                                 timeout=6.0)
            else:
                r = self._goto_planned(pre, attempt_quat, STIFF_COMPLIANCE)
            if not r.ok:
                if side_grasp:
                    # 无真值模式没法预检净空：side 接近段被挡就直接改走 top-down 重试
                    # （真机实测 can_yellow / bottle_water 都是这样失败的）
                    side_fallback_reason = (f"side 接近受阻（{r.error_code or 'ik'}, "
                                            f"{r.max_force:.1f}N）-> 自动退回 top-down 重试")
                    self.log(f"[skills] {side_fallback_reason}")
                    side_grasp = False
                    approach_dir = np.asarray((0.0, 0.0, -1.0), float)
                    attempts.append((0.0, ratio, 0.0))     # 补一次标准深度的 top 尝试
                self._retreat(attempt_quat)
                last_force = max(last_force, r.max_force)
                reasons.append(f"#{attempt} dz={dz * 1000:+.0f}mm yaw={np.degrees(yaw):.0f}° "
                               f"接近失败({r.error_code or 'ik'})")
                continue

            # 接近段本身可能把目标碰跑（实测圆柱在预抓取位移动中被推移 33 mm，
            # 继续下降/合爪会把它推到 127 mm/152 mm）。下降前复检：超过 8 mm 立刻撤退。
            # 无真值模式没有物体位姿可读（R-a），跳过这项检查。
            obj_now = self._object(tid) if not no_oracle else None
            if not no_oracle and obj_now is None:
                extra, override, note = self._finish_failure(quat, tid)
                return SkillResult(False, "pick", f"物体已不在场景中: {tid}" + note,
                                   override or "object_not_found", metrics=extra)
            drift_approach = (float(np.linalg.norm(obj_now.pose[:3] - before[:3]))
                              if not no_oracle else 0.0)
            if not no_oracle and drift_approach > PICK_APPROACH_DRIFT_TOL:
                extra, override, note = self._finish_failure(attempt_quat, tid)
                return SkillResult(False, "pick",
                                   f"接近过程碰动物体 {drift_approach * 1000:.0f} mm "
                                   f"(> {PICK_APPROACH_DRIFT_TOL * 1000:.0f} mm)，放弃下降/合爪" + note,
                                   override or "object_disturbed",
                                   metrics={"drift": drift_approach, "attempts": attempt, **extra})

            # side 的"下降"其实是**水平推进**：必须用轴对齐软档 + 显式 6N 看门狗，
            # 否则 z_only 档在水平方向零导纳 -> 硬顶（真机 2910.3N）。
            # top-down 则走"守卫式接触下降"：小步 + 2.5N 看门狗 + 软档，碰到瓶盖/瓶肩就停在
            # 那个高度合爪（真机旧实现会一路顶到 116.3N 才中止）。
            descend_metrics: dict = {}
            slender_now = False          # side 分支不适用（只有 top-down 的细高物体才判定正面证据）
            if side_grasp:
                r = self.runner.move_to_pose(target, attempt_quat,
                                             compliance=PLACE_SIDE_SOFT_COMPLIANCE, timeout=6.0,
                                             force_limit=float(PLACE_SIDE_STOP_FORCE_N))
            else:
                cat_now = (str(getattr(obj, "category", "") or "").lower()
                           if not no_oracle else cat)
                half_h_now = (float(obj.size[2]) if not no_oracle else float(half_size[2]))
                # §6.2#2 只对"细高物体"走限速下降；矮物体保持现状。
                # 判据 = 类别+半高（oracle/带 3 元 size）**或** 几何证据（可见点高出桌面 >= 80mm）。
                _zh_now = (float(args["target_z_hint"])
                           if args.get("target_z_hint") is not None else None)
                slender_now = (
                    (cat_now in ("cylinder", "bottle", "can")
                     and half_h_now >= float(PICK_SLENDER_HALF_HEIGHT_M))
                    or (no_oracle and _zh_now is not None
                        and (_zh_now - float(TABLE_HEIGHT)) >= float(PLACE_SLENDER_HINT_HEIGHT_M)))
                status, descend_metrics, r = self._descend_grasp_until_contact(
                    target, attempt_quat,
                    slender=slender_now, base_compliance=(None if slender_now else descend))
                if status == "error":
                    # 真异常（单步力爆表/硬错误）：转成失败 MoveResult，走既有的撤退+失败分支
                    r = MoveResult(False, "抓取接触下降异常中止",
                                   str(descend_metrics.get("step_error") or "force_limit"),
                                   max_force=float(descend_metrics.get("peak_contact_force", 0.0)))
                else:
                    self.log(f"[skills] 抓取接触下降：{status} grasp_z_effective="
                             f"{descend_metrics['grasp_z_effective']:.4f} "
                             f"touchdown_force={descend_metrics['touchdown_force']:.2f} N "
                             f"（{descend_metrics['grasp_descend_steps']} 步）")
                    if status == "contact":
                        # 关键：接触那一步的 MoveResult 是"看门狗中止"(ok=False)，
                        # 那是**正常的接触信号**，不能当成下降失败往下传。
                        r = MoveResult(True, "接触即停（守卫式接触下降）", None,
                                       max_force=float(descend_metrics.get("touchdown_force", 0.0)))
            if not r.ok:
                last_force = max(last_force, r.max_force)
                if side_grasp:
                    side_fallback_reason = (f"side 推进受阻（{r.error_code}, "
                                            f"{r.max_force:.1f}N）-> 自动退回 top-down 重试")
                    self.log(f"[skills] {side_fallback_reason}")
                    side_grasp = False
                    approach_dir = np.asarray((0.0, 0.0, -1.0), float)
                    attempts.append((0.0, ratio, 0.0))     # 补一次标准深度的 top 尝试
                self._retreat(attempt_quat)
                reasons.append(f"#{attempt} dz={dz * 1000:+.0f}mm yaw={np.degrees(yaw):.0f}° "
                               f"下压中止({r.error_code}, {r.max_force:.1f}N)")
                continue

            # task-28 要求 2：guarded close——边合爪边看力，超阈值就停在这个开合度，
            # 不再继续捏（把物体推走/挤飞的头号来源）。
            # lead 批准：细高物体改"受控合到接触"——预算 0.6s -> 3s（实测 0.6s 只合到 69mm
            # 就停了，指令是 58mm），力看门狗 45N -> 18N（贴到瓶身就停，不硬捏）。
            if slender_now:
                _close_ok = self._guarded_close(ratio,
                                                threshold=float(PLACE_SLENDER_CLOSE_FORCE_N),
                                                total=float(PLACE_SLENDER_CLOSE_TOTAL_S))
            else:
                _close_ok = self._guarded_close(ratio)
            # 正面证据：合爪被物体**挡住**（力看门狗触发）才算"真的捏到东西"
            last_close_contact = (not _close_ok)

            # top-down：从"实际接触高度"抬升（可能高于模型给的 z_hint），保证是往上抬
            if not side_grasp:
                lift = np.array([float(target[0]), float(target[1]),
                                 float(max(float(target[2]),
                                           float(descend_metrics.get("grasp_z_effective",
                                                                     target[2]))))
                                 + float(LIFT_HEIGHT)])
            else:
                lift = target - approach_dir * float(LIFT_HEIGHT)   # side: 侧向抽离
            r = self.runner.move_to_pose(lift, attempt_quat, compliance=soft, timeout=8.0)
            last_force = max(last_force, r.max_force)
            # P1-1：抬升失败必须判失败。旧实现只要 gap 判据过了就 return ok=True，**完全忽略**
            # 抬升的 r.ok —— 实测抬升以 force_limit 46 N 中止、方块只动 +0.8 mm 仍报"已抓起"。
            lift_ok = bool(getattr(r, "ok", True))
            # P1-2：抬升后"物体随动"证据（夹爪仍抓着可动物体 且 它已不被支撑面托着）。
            # 判定 None = 环境没有接触信息（测试桩），不据此否决，保持旧行为。
            carry_ok, carry_bodies = self._carry_evidence() if no_oracle else (None, [])

            if no_oracle:
                # R-b：无真值模式用**夹爪实测**判成功——闭合后抬升阶段读指间距，
                # > 全闭间距 + 阈值 即认为指间有物（被抓住）。
                last_gap_mm = self._jaw_gap_mm()
                # 判定"指间有物"必须相对**指令闭合量**看：空桌面时夹爪照样能走到指令开合度
                # （实测 target_xy 指向空处也会读到 33.8 mm），只有"被物体挡住、合不到位"
                # 才是真抓住。所以基线取 max(全闭间距, 指令间距)，并保留全开反例护栏。
                commanded_gap_mm = float(GRIPPER_GAP_CLOSED_MM + ratio * (
                    GRIPPER_GAP_OPEN_MM - GRIPPER_GAP_CLOSED_MM))
                last_commanded_gap_mm = commanded_gap_mm
                gap_baseline = max(float(GRIPPER_GAP_CLOSED_MM), commanded_gap_mm)
                # lead 口径：方块/低矮物体"只要被挡住就算夹住"——容差 4mm -> 1mm（空抓实测差 0，
                # 真夹住至少差几毫米；真机方块 40mm 宽/指令 37.2mm 只差 2.8mm，用 4mm 会误判未夹住）。
                # 细高物体仍用 4mm（它另有"合爪被力挡住 + 未全闭"的正面证据门）。
                _blocked_tol = (float(GRASP_JAW_GAP_MIN_MM) if slender_now
                                else float(PICK_BLOCKED_GAP_TOL_MM))
                grasped = (last_gap_mm > gap_baseline + _blocked_tol
                           and last_gap_mm < GRIPPER_GAP_OPEN_MM - GRASP_JAW_OPEN_MARGIN_MM)
                # P1-1：抬升没成功 = 没真正把物体拿起来，不算抓住。
                grasped = grasped and lift_ok
                # P1-2：抬升后夹爪与物体脱离接触 / 物体仍被桌面支撑 = 没随动，不算抓住
                # （判定为 None 表示环境没有接触信息，不否决）。
                grasped = grasped and (carry_ok is not False)
                # 正面证据（lead 要求）：指间距要贴近目标直径（细高物体才判定）
                size_gap_mm = 2000.0 * float(max(half_size[0], half_size[1]))
                last_gap_vs_size_mm = abs(float(last_gap_mm) - size_gap_mm)
                last_gap_over_size_mm = float(last_gap_mm) - size_gap_mm
                if slender_now:
                    # 诚实判失败（lead 要求）：实测 gap 比直径大 3mm 以上 = 没夹到东西；且合爪必须
                    # **被挡住**（力看门狗触发）才算捏到。为什么不用 ±8mm 当门槛：live 形状只有 2 元
                    # target_size（模型估的直径 80.4mm vs 真值 64mm），±8 会把真夹住判成失败；
                    # 该值仍作为诊断指标 grasp_gap_vs_size_mm 保留。
                    last_close_force_n = float(getattr(self, "last_close_force", 0.0))
                    last_gap_positive = (
                        last_close_force_n >= float(PLACE_SLENDER_MIN_CONTACT_N)
                        and float(last_gap_mm) > (float(GRIPPER_GAP_CLOSED_MM)
                                                  + float(GRASP_SIZE_GAP_OVER_MM)))
                    grasped = grasped and last_gap_positive
                if grasped:
                    self.holding = HOLDING_UNNAMED      # 不知道 id（无真值），用占位
                    self.last_pick_pose = None
                    return SkillResult(True, "pick",
                                       f"已抓起（无真值, 第 {attempt} 次尝试 深度{dz * 1000:+.0f}mm "
                                       f"闭合{ratio:.2f} yaw{np.degrees(yaw):.0f}°；"
                                       f"指间距 {last_gap_mm:.1f} mm；"
                                       f"抬升ok={lift_ok} 随动证据={carry_ok}）",
                                       metrics={"grasp_gap_mm": round(float(last_gap_mm), 2),
                                                "grasp_commanded_gap_mm": round(commanded_gap_mm, 2),
                                                "grasp_evidence": "jaw_gap+lift+carry",
                                                # P1：抬升是否成功 + 抬升后物体是否随动
                                                "grasp_lift_ok": bool(lift_ok),
                                                "grasp_carry_evidence": carry_ok,
                                                "grasp_carry_bodies": list(carry_bodies),
                                                "grasp_gap_vs_size_mm": (None if last_gap_vs_size_mm
                                                                        is None else round(float(last_gap_vs_size_mm), 2)),
                                                "grasp_gap_over_size_mm": (None if last_gap_over_size_mm
                                                                           is None else round(float(last_gap_over_size_mm), 2)),
                                                # lead 要求：实测 gap 就当"量到的物体宽度"
                                                "grasp_width_mm": round(float(last_gap_mm), 2),
                                                "grasp_close_force_n": (None if last_close_force_n is None
                                                                        else round(float(last_close_force_n), 3)),
                                                "grasp_positive_evidence": (None if last_gap_positive is None
                                                                            else bool(last_gap_positive)),
                                                "grasp_profile": profile,
                                                "grasp_profile_effective": ("top" if side_fallback_reason
                                                                            else profile),
                                                "side_fallback_reason": side_fallback_reason,
                                                "max_force": last_force, "attempts": attempt,
                                                "yaw_deg": float(np.degrees(yaw)),
                                                **descend_metrics,
                                                "target_xy": [round(float(target_point[0]), 4),
                                                              round(float(target_point[1]), 4)],
                                                "target_z": round(float(target_point[2]), 4),
                                                **self._no_oracle_diagnostics(target_point)})
            else:
                after = self.env.object_pose(tid)
                last_moved = float(after[2] - before[2])
                if last_moved >= 0.02:
                    self.holding = str(tid)
                    self.last_pick_pose = np.asarray(before, float).copy()   # task-36：归还用
                    return SkillResult(True, "pick",
                                       f"已抓起 {tid}（第 {attempt} 次尝试 深度{dz * 1000:+.0f}mm "
                                       f"闭合{ratio:.2f} yaw{np.degrees(yaw):.0f}°）",
                                       metrics={"moved": last_moved, "max_force": last_force,
                                                "attempts": attempt, "object_z": float(after[2]),
                                                "yaw_deg": float(np.degrees(yaw)),
                                                "grasp_profile": profile,
                                                "grasp_profile_effective": ("top" if side_fallback_reason
                                                                            else profile),
                                                "side_fallback_reason": side_fallback_reason,
                                                **descend_metrics})
            if not r.ok:
                reasons.append(f"#{attempt} dz={dz * 1000:+.0f}mm yaw={np.degrees(yaw):.0f}° "
                               f"抬升中止({r.error_code}, {r.max_force:.1f}N)")
            elif no_oracle:
                why = "（判定未夹住）"
                if not lift_ok:
                    why = "（抬升未完成，判定未夹住）"
                elif carry_ok is False:
                    why = (f"（抬升后夹爪与物体脱离接触或物体仍被桌面支撑，判定未夹住；"
                           f"夹爪接触={carry_bodies or '无'}）"
                           if carry_bodies else "（抬升后夹爪与物体脱离接触，判定未夹住）")
                reasons.append(f"#{attempt} dz={dz * 1000:+.0f}mm 闭合{ratio:.2f} "
                               f"yaw={np.degrees(yaw):.0f}° 指间距仅 {last_gap_mm:.1f} mm {why}")
            else:
                reasons.append(f"#{attempt} dz={dz * 1000:+.0f}mm 闭合{ratio:.2f} "
                               f"yaw={np.degrees(yaw):.0f}° 未夹住(仅 {last_moved * 1000:+.1f}mm)")
            # 先松开夹爪再撤退：闭合状态下直接抬升会把物体一起拖走（实测位移 39 mm）
            self.env.set_gripper(GRIPPER_OPEN)
            self._settle(0.2)
            if not self._recover(attempt_quat):
                extra, override, note = self._finish_failure(attempt_quat, tid,
                                                             contacts_only=no_oracle)
                return SkillResult(False, "pick", "抓取失败后无法脱离接触（stuck）" + note,
                                   override or "stuck",
                                   metrics={"moved": last_moved, "max_force": last_force,
                                            "attempts": attempt,
                                            "grasp_profile_effective": ("top" if side_fallback_reason
                                                                        else profile),
                                            "side_fallback_reason": side_fallback_reason,
                                            "grasp_gap_mm": (None if last_gap_mm is None
                                                             else round(float(last_gap_mm), 2)),
                                            "grasp_commanded_gap_mm": (
                                                None if last_commanded_gap_mm is None
                                                else round(float(last_commanded_gap_mm), 2)),
                                            "grasp_gap_vs_size_mm": (None if last_gap_vs_size_mm is None
                                                                    else round(float(last_gap_vs_size_mm), 2)),
                                            "grasp_positive_evidence": (None if last_gap_vs_size_mm is None
                                                                        else bool(last_gap_vs_size_mm
                                                                                  <= float(GRASP_SIZE_GAP_TOL_MM))),
                                            "grasp_evidence": "jaw_gap" if no_oracle else None,
                                            # P1：失败时也要能观测抬升/随动证据
                                            "grasp_lift_ok": (None if lift_ok is None
                                                              else bool(lift_ok)),
                                            "grasp_carry_evidence": carry_ok,
                                            "grasp_carry_bodies": list(carry_bodies),
                                            **extra})

        extra, override, note = self._finish_failure(quat, tid, contacts_only=no_oracle)
        detail = "；".join(reasons[-4:]) if reasons else "无可用尝试"
        who = "目标点" if no_oracle else str(tid)
        best = ("" if no_oracle
                else f"（最好 {last_moved * 1000:+.1f} mm）")
        return SkillResult(False, "pick",
                           f"夹取失败：{len(attempts)} 组参数均未抓起 {who}{best}。"
                           f"原因：{detail}" + note,
                           override or "grasp_failed",
                           metrics={"moved": last_moved, "max_force": last_force,
                                    "attempts": len(attempts), "reasons": reasons,
                                    "grasp_profile": profile,
                                    "grasp_profile_effective": ("top" if side_fallback_reason
                                                                else profile),
                                    "side_fallback_reason": side_fallback_reason,
                                    **descend_metrics,
                                    "grasp_gap_mm": (None if last_gap_mm is None
                                                     else round(float(last_gap_mm), 2)),
                                    "grasp_commanded_gap_mm": (
                                        None if last_commanded_gap_mm is None
                                        else round(float(last_commanded_gap_mm), 2)),
                                    "grasp_gap_vs_size_mm": (None if last_gap_vs_size_mm is None
                                                                else round(float(last_gap_vs_size_mm), 2)),
                                    "grasp_gap_over_size_mm": (None if last_gap_over_size_mm is None
                                                               else round(float(last_gap_over_size_mm), 2)),
                                    "grasp_width_mm": (None if last_gap_mm is None
                                                       else round(float(last_gap_mm), 2)),
                                    "grasp_close_force_n": (None if last_close_force_n is None
                                                            else round(float(last_close_force_n), 3)),
                                    "grasp_positive_evidence": (None if last_gap_positive is None
                                                                else bool(last_gap_positive)),
                                    "grasp_evidence": "jaw_gap" if no_oracle else None,
                                    # P1：失败时也要能观测抬升/随动证据（便于定位假成功）
                                    "grasp_lift_ok": (None if lift_ok is None else bool(lift_ok)),
                                    "grasp_carry_evidence": carry_ok,
                                    "grasp_carry_bodies": list(carry_bodies),
                                    **extra})

    # ---------------- 放置 ---------------- #
    def _ik_reachable(self, xy, z, quat=None) -> bool:
        """该位姿 IK 是否有解（用于筛选桌面空位）。"""
        try:
            _q, ok = self.kin.ik(np.array([float(xy[0]), float(xy[1]), float(z)], float), quat)
            return bool(ok)
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _quat_angle(qa, qb) -> float:
        """两个 wxyz 四元数的夹角（rad，取最短弧）。"""
        a = np.asarray(qa, dtype=float).reshape(-1)[:4]
        b = np.asarray(qb, dtype=float).reshape(-1)[:4]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return float(np.pi)
        return 2.0 * float(np.arccos(np.clip(abs(float(np.dot(a, b))) / (na * nb), -1.0, 1.0)))

    def _band_distance(self, xy) -> float:
        """点到"实测可抓带"的欧氏距离（带内为 0）。"""
        bx0, bx1, by0, by1 = PLACE_GRASPABLE_BAND
        dx = max(bx0 - float(xy[0]), 0.0, float(xy[0]) - bx1)
        dy = max(by0 - float(xy[1]), 0.0, float(xy[1]) - by1)
        return float(np.hypot(dx, dy))

    def _xy_on_table(self, xy, half: float = 0.02) -> bool:
        """xy（含半宽）是否落在桌面上表面范围内。"""
        x0, x1, y0, y1 = PLACE_TABLE_BOUNDS
        return (x0 + float(half) <= float(xy[0]) <= x1 - float(half)
                and y0 + float(half) <= float(xy[1]) <= y1 - float(half))

    def _spot_clear(self, xy, half_self: float, exclude_id=None, min_clear: float = 0.05) -> bool:
        """xy 与其它物体（可排除某个 id）的水平净距是否都 >= 自身半宽+对方半宽+min_clear。"""
        for o in self.env.list_objects():
            if exclude_id is not None and str(o.id) == str(exclude_id):
                continue
            d = float(np.linalg.norm(np.asarray(xy, float) - np.asarray(o.pose[:2], float)))
            if d < float(half_self) + float(max(o.size[0], o.size[1])) + float(min_clear):
                return False
        return True

    def _choose_beside_direction(self, anchor, held_obj):
        """beside：在四个方向里选"与其它物体净距最大、且不出桌面"的那个。

        返回 (单位方向向量(2,), 依据字符串)；依据会写进 metrics（spot_reason）。
        """
        best = None
        for name, vec in RELATION_DIRS.items():
            d = np.asarray(vec, float)
            axis_x = abs(float(d[0])) > 0.5
            half_anchor = float(anchor.size[0] if axis_x else anchor.size[1])
            half_held = (float(held_obj.size[0] if axis_x else held_obj.size[1])
                         if held_obj is not None else 0.02)
            dist = half_anchor + half_held + PLACE_SIDE_MARGIN_M
            p = np.asarray(anchor.pose[:2], float) + d * dist
            if not self._xy_on_table(p, half_held):
                clear = -1.0                      # 出桌面：直接罚负
            else:
                clear = min((float(np.linalg.norm(p - np.asarray(o.pose[:2], float)))
                             - half_held - float(max(o.size[0], o.size[1]))
                             for o in self.env.list_objects()
                             if str(o.id) not in (str(anchor.id), str(self.holding or ""))),
                            default=9.9)
            if best is None or clear > best[0]:
                best = (clear, d, name)
        return best[1], f"beside 自动选 {best[2]}（净距 {best[0] * 1000:.0f}mm）"

    def _grasp_rehearsal(self, xy, z, quat=None):
        """抓取预演评估（task-28 要求 1）：top-down IK 残差 + 关节限位余量 + 姿态偏差。

        返回 (penalty, note)；penalty 越小越好，IK 无解返回 (inf, 'ik_fail')。
        只做运动学排练（不推进物理），用于排序候选点。
        """
        try:
            q_sol, ok = self.kin.ik(np.array([float(xy[0]), float(xy[1]), float(z)], float), quat)
            if not ok:
                return float("inf"), "ik_fail"
            p, qq = self.kin.fk(q_sol)
            res = float(np.linalg.norm(np.asarray(p, float).reshape(-1)[:3]
                                       - np.array([float(xy[0]), float(xy[1]), float(z)])))
            rot = 0.0 if quat is None else self._quat_angle(quat, qq)
            q_sol = np.asarray(q_sol, float).reshape(-1)
            margin = float(np.min(np.minimum(q_sol - ARM_JOINT_LIMITS[:, 0],
                                             ARM_JOINT_LIMITS[:, 1] - q_sol)))
            pen = res / 0.005 + max(0.0, PLACE_REHEARSAL_MIN_MARGIN - margin) / 0.2 + rot / 0.05
            return float(pen), "res={:.1f}mm rot={:.3f}rad margin={:.2f}rad".format(
                res * 1000, rot, margin)
        except Exception as exc:  # noqa: BLE001
            return float("inf"), f"ik_exc:{exc}"

    def _choose_table_spot(self, held_obj, quat=None):
        """为桌面放置自选一个空位（task-24 要求 1 + task-28 要求 1）。

        硬约束：桌面外沿内（含自身半宽）+ PLACE_EDGE_MARGIN_M；
                与其它物体/托盘水平距离 >= 自身半宽 + 对方半宽 + 净距（0.08 -> 0.06 -> 0.04 逐级放宽）；
                该点 top-down IK 有解。
        排序（越小越优先）：
                score = 3.0 * 到"实测可抓带"的距离      # task-28：落点必须放下后还能再抓起来
                      + 1.0 * 抓取预演惩罚（IK 残差/限位余量/姿态偏差）
                      - 0.5 * 实际净距余量               # 同分时更空旷的优先
        返回 (xy(2,), reason 字符串)，reason 会写进 metrics['spot_reason']。
        """
        half_self = 0.02
        if held_obj is not None:
            half_self = float(max(held_obj.size[0], held_obj.size[1]))
        others = []
        for o in self.env.list_objects():
            if held_obj is not None and o.id == held_obj.id:
                continue
            others.append((np.asarray(o.pose[:2], float),
                           float(max(o.size[0], o.size[1]))))
        z = float(TABLE_HEIGHT + half_self + PLACE_TABLE_MARGIN_M)
        x0, x1, y0, y1 = PLACE_TABLE_BOUNDS
        xs = np.arange(PLACE_TABLE_X_RANGE[0], PLACE_TABLE_X_RANGE[1] + 1e-9, PLACE_TABLE_GRID_M)
        ys = np.arange(PLACE_TABLE_Y_RANGE[0], PLACE_TABLE_Y_RANGE[1] + 1e-9, PLACE_TABLE_GRID_M)

        clearance = float(PLACE_FREE_CLEARANCE_M)
        cands = []
        while clearance >= 0.039 and not cands:
            for x in xs:
                for y in ys:
                    p = np.array([float(x), float(y)])
                    if not (x0 + half_self + PLACE_EDGE_MARGIN_M <= p[0]
                            <= x1 - half_self - PLACE_EDGE_MARGIN_M
                            and y0 + half_self + PLACE_EDGE_MARGIN_M <= p[1]
                            <= y1 - half_self - PLACE_EDGE_MARGIN_M):
                        continue
                    dmin = min((float(np.linalg.norm(p - c)) - half_self - half_o
                                for c, half_o in others), default=9.9)
                    if dmin < clearance:
                        continue
                    cands.append((p, dmin))
            if not cands:
                clearance -= 0.02
        if not cands:
            self.log("[skills] 桌面上找不到满足净距的空位，退回锚点")
            return (np.asarray(PLACE_TABLE_ANCHOR, float),
                    "fallback_anchor(无满足净距的空位)")

        best = None
        best_reason = ""
        n_ik_fail = 0
        for p, dmin in cands:
            pen, note = self._grasp_rehearsal(p, z, quat)
            if not np.isfinite(pen):
                n_ik_fail += 1
                continue
            band = self._band_distance(p)
            score = 3.0 * band + 1.0 * pen - 0.5 * dmin
            if best is None or score < best[0]:
                best = (score, p, band, dmin, pen, note)
        if best is None:
            return (np.asarray(PLACE_TABLE_ANCHOR, float),
                    f"fallback_anchor({len(cands)} 个候选全部 IK 无解)")
        _score, p, band, dmin, pen, note = best
        reason = ("可抓带内" if band <= 1e-9 else "距可抓带 %.0fmm" % (band * 1000))
        reason += "，净距 %.0fmm(要求 %.0fmm)，预演 %s，候选 %d 个(IK 失败 %d)" % (
            dmin * 1000, clearance * 1000, note, len(cands), n_ik_fail)
        self.log(f"[skills] 桌面落点 {np.round(p, 3).tolist()}：{reason}")
        return p, "grasp_aware(" + reason + ")"

    def _skill_place(self, call: SkillCall) -> SkillResult:
        """放置：container_id(容器) / surface='table'(桌面) / on_top_of(叠放) 三选一。

        task-32 叠放：on_top_of=<anchor_id>，可选 xy=[x,y]（相对锚点的水平偏移，默认 0）
          或 offset=[dx,dy,dz]。目标 xy = 锚点 xy + 偏移；z = 锚点顶面 + 被夹物半高
          + PLACE_STACK_MARGIN_M；姿态仍是 top-down。判据在前四条之外再加：
          底面与锚点顶面 ∈ [-8, +20] mm、xy 落在锚点 footprint（半宽+20 mm）、
          释放后再观察 0.4 s 无滑落/翻滚（xy 位移 <20 mm 且 z 下降 <20 mm）。
        container_id / surface='table' 两个分支的行为与判据保持不变。

        要求 1（桌面放置）：surface='table' 时支撑面 = core.config.TABLE_HEIGHT，
          z = TABLE_HEIGHT + 物体半高 + 余量；xy 未给出就自选一个"离现有物体/tray
          至少 PLACE_FREE_CLEARANCE_M、且在桌面内、IK 可解"的空位，最终 xy/dz 写进 metrics。
        要求 2（判据加强）：物体 xy 距目标 < 6 cm、底面离支撑面 ∈ [-1, +2] cm、
          接触力 <= 5 N、夹爪已张开——四条同时满足才算成功，否则 error_code='place_mismatch'。
        要求 4（减少无中生有的卡死）：释放后强制向上抬起 PLACE_LIFT_CLEARANCE_M，
          把指垫彻底带离被放物体；若抬升失败或仍有残压力，立刻做多方向脱身。
        """
        args = call.args or {}
        cid = args.get("container_id")
        surface = args.get("surface")
        surface_l = str(surface).strip().lower() if surface is not None else None
        use_table = surface_l in ("table", "desk", "desktop", "桌面")
        anchor_id = args.get("on_top_of")
        use_stack = anchor_id is not None and str(anchor_id) != ""
        # task-36 第 4 种目标：anchor_id + relation（横向关系并排）
        side_anchor_id = args.get("anchor_id")
        use_side = side_anchor_id is not None and str(side_anchor_id) != ""
        relation_raw = args.get("relation")
        relation = None
        if use_side:
            if relation_raw is None:
                return SkillResult(False, "place",
                                   "anchor_id 需要同时给 relation（right/left/front/back/beside）",
                                   "bad_args")
            relation = RELATION_ALIASES.get(str(relation_raw).strip().lower())
            if relation is None:
                return SkillResult(False, "place",
                                   f"不支持的 relation: {relation_raw}"
                                   "（可用 right/left/front/back/beside）", "bad_args")
        n_modes = (int(use_table) + int(use_stack) + int(use_side)
                   + int(cid is not None and str(cid) != ""))
        if n_modes > 1:
            return SkillResult(False, "place",
                               "container_id / surface / on_top_of / anchor_id 只能四选一", "bad_args")
        cont = None
        if not use_table and not use_stack and cid is not None and str(cid) != "":
            cont = self._object(str(cid))
            if cont is None:
                return SkillResult(False, "place", f"容器不存在: {cid}", "object_not_found")
            if not cont.is_container:
                return SkillResult(False, "place", f"{cid} 不是容器", "not_container")
        if not use_table and not use_stack and not use_side and cont is None:
            return SkillResult(False, "place",
                               "缺少 container_id / surface='table' / on_top_of / "
                               "(anchor_id+relation)", "bad_args")

        # task-32 要求 3：锚点校验（存在 / 可抓 / 没被本机夹着 / 确实在桌面上）
        anchor = None
        if use_stack or use_side:
            _aid = anchor_id if use_stack else side_anchor_id
            anchor = self._object(str(_aid))
            if anchor is None:
                return SkillResult(False, "place", f"锚点不存在: {_aid}", "object_not_found")
            if not anchor.graspable:
                return SkillResult(False, "place", f"锚点不可作为参照物: {_aid}", "not_graspable")
            if self.holding is not None and str(self.holding) == str(_aid):
                return SkillResult(False, "place",
                                   f"锚点 {_aid} 正被本机夹爪夹持，无法以它为参照",
                                   "anchor_held")
        if use_stack:
            anchor_id = _aid
            a_bottom = float(anchor.pose[2] - float(anchor.size[2]))
            a_gap = a_bottom - float(TABLE_HEIGHT)
            if not (PLACE_ANCHOR_TABLE_MIN_M <= a_gap <= PLACE_ANCHOR_TABLE_MAX_M):
                return SkillResult(False, "place",
                                   f"锚点 {anchor_id} 不在桌面上（底面离桌面 {a_gap * 1000:+.0f} mm），"
                                   "拒绝叠放", "anchor_not_on_surface",
                                   metrics={"anchor": str(anchor_id),
                                            "anchor_bottom_gap": round(a_gap, 4)})

        if self.holding is None and not args.get("allow_empty", False):
            return SkillResult(False, "place", "当前未夹持任何物体，拒绝下压", "nothing_held")
        offset = np.asarray(args.get("offset", [0.0, 0.0, 0.0]), float)
        soft = self._motion_compliance(call, PLACE_COMPLIANCE)
        quat = approach_quat(float(args.get("yaw", 0.0)))

        # P0：入口清场不能把"夹爪里正抓着的东西"当障碍物（实测无真值链路上，_ensure_clear 把
        # 手持的 box_red 当阻挡、撤退时把方块弄掉，而 place 仍报 ok）。
        #   1) 上层（loop，模型不填）显式给 holding=True -> 直接跳过清场；
        #   2) 无真值兜底：只有一个物体与 EE 接触且夹爪卡在中位开合 -> 认定它就是手里那个；
        # oracle 路径（holding 已知 id）由 _objects_touching_ee(exclude_held=True) 照旧排除，语义不变。
        holding_arg = args.get("holding")
        held_unknown = self.holding is None or str(self.holding) == HOLDING_UNNAMED
        entry_clear_skipped = None
        if holding_arg is True:
            entry_clear_skipped = "holding_arg"
        elif held_unknown and self._looks_like_holding_object():
            entry_clear_skipped = "held_heuristic"
        if entry_clear_skipped is None:
            if not self._ensure_clear(quat):
                extra, override, note = self._finish_failure(quat, self.holding)
                return SkillResult(False, "place", "机械臂卡在接触中，无法开始放置" + note,
                                   override or "stuck",
                                   metrics={"entry_clear_skipped": entry_clear_skipped, **extra})
        else:
            self.log(f"[skills] 持物中：跳过入口清场检查（手里的物体不算障碍，依据={entry_clear_skipped}）")

        held_obj = self._object(self.holding) if self.holding else None
        half_z = float(held_obj.size[2]) if held_obj is not None else 0.02
        xy_arg = args.get("xy")
        side_dir = None
        side_expected_dist = None
        xy_source: Optional[str] = None
        if use_table:
            # 桌面：支撑面 = 桌面上表面（几何硬约束，与容器无关）
            support_z = float(TABLE_HEIGHT)
            target_kind = "table"
            target_label = "桌面"
            if xy_arg is not None:
                target_xy = np.asarray(xy_arg, float).reshape(-1)[:2]
                spot_note = "explicit_xy"
            else:
                target_xy, spot_note = self._choose_table_spot(held_obj, quat)
            tz = support_z + half_z + PLACE_TABLE_MARGIN_M
        elif use_stack:
            # task-32 要求 1：叠放。xy = 锚点 xy + 水平偏移；z = 锚点顶面 + 被夹物半高 + 余量；
            # 姿态仍是 top-down，保证是"放上去"而不是"推过去"。
            anchor_top_z = float(anchor.pose[2] + float(anchor.size[2]))
            support_z = anchor_top_z
            target_kind = "on_top_of:" + str(anchor_id)
            target_label = f"{anchor_id} 顶面"
            if xy_arg is not None and held_unknown:
                # P0：无真值链路里模型给的 xy 就是"它从图像标定出来的锚点位置"（**绝对**世界坐标），
                # 必须直接拿它当落点，绝不能再叠加 scene 里锚点的位姿——实测会把 0.397 变成 0.794
                # （工作空间边角 → unreachable），g2 必然失败。
                # z 仍然交给"接触式落放"（到该 xy 上方 → 下降到接触 → 松爪）。
                target_xy = np.asarray(xy_arg, float).reshape(-1)[:2]
                xy_offset = target_xy - np.asarray(anchor.pose[:2], float)   # 仅供 metrics 诊断
                xy_source = "model_absolute"
                spot_note = "stack_model_xy(absolute)"
            else:
                # oracle 路径语义不变：xy 仍是"相对锚点的水平偏移"（task-32 冻结接口）
                if xy_arg is not None:
                    xy_offset = np.asarray(xy_arg, float).reshape(-1)[:2]
                else:
                    xy_offset = np.array([float(offset[0]), float(offset[1])])
                target_xy = np.asarray(anchor.pose[:2], float) + xy_offset
                xy_source = "anchor_offset"
                spot_note = "stack_on_" + str(anchor_id)
            tz = anchor_top_z + half_z + PLACE_STACK_MARGIN_M
            if abs(float(offset[2])) > 1e-9:
                tz += float(offset[2])
            spot_note = "stack_on_" + str(anchor_id)
        elif use_side:
            # task-36 要求：横向关系并排。xy = 锚点 xy + 方向单位向量 × 中心距；
            # 方向约定（俯视图/世界系，x 向右、y 向前）：right=+x, left=-x, front=+y, back=-y。
            if relation == "beside":
                side_dir, beside_why = self._choose_beside_direction(anchor, held_obj)
            else:
                side_dir = np.asarray(RELATION_DIRS[relation], float)
                beside_why = f"指定方向 {relation}"
            axis_x = abs(float(side_dir[0])) > 0.5
            half_anchor = float(anchor.size[0] if axis_x else anchor.size[1])
            if held_obj is not None:
                half_held = float(held_obj.size[0] if axis_x else held_obj.size[1])
            else:
                half_held = 0.02
            if args.get("distance") is not None:
                side_expected_dist = float(args["distance"])
            else:
                side_expected_dist = half_anchor + half_held + PLACE_SIDE_MARGIN_M
            target_xy = np.asarray(anchor.pose[:2], float) + side_dir * side_expected_dist
            support_z = float(TABLE_HEIGHT)
            tz = support_z + half_z + PLACE_TABLE_MARGIN_M
            target_kind = f"{relation}_of:{side_anchor_id}"
            target_label = f"{side_anchor_id} 的 {relation} 侧桌面"
            spot_note = f"side_{relation}({beside_why})"
        else:
            support_z = float(cont.pose[2] + cont.size[2])
            target_kind = str(cid)
            target_label = str(cid)
            target_xy = np.array([cont.pose[0] + offset[0], cont.pose[1] + offset[1]])
            if xy_arg is not None:
                target_xy = np.asarray(xy_arg, float).reshape(-1)[:2]
                spot_note = "explicit_xy"
            else:
                spot_note = "container_center"
            tz = support_z + half_z + 0.010
            if abs(float(offset[2])) > 1e-9:      # 显式给了 z 偏移则以调用方为准
                tz = float(cont.pose[2] + offset[2])
        target = np.array([float(target_xy[0]), float(target_xy[1]), float(tz)])
        above = target + np.array([0.0, 0.0, 0.12])
        self.log(f"[skills] 放置目标：{target_label} xy={np.round(target_xy, 3).tolist()} "
                 f"support_z={support_z:.3f} tz={tz:.3f} [{spot_note}]")

        # task-59/§5.4：接触式落放开关。默认 null=自动：on_top_of(叠放) 与容器内 True；
        # surface='table' / 横向关系落放保持原来的"守卫式下降"（语义不变）。
        duc_arg = args.get("descend_until_contact")
        descend_until_contact = (bool(use_stack or (cont is not None))
                                 if duc_arg is None else bool(duc_arg))
        touched = False
        touch_metrics: dict = {}
        r: Optional[MoveResult] = None
        if descend_until_contact:
            status, touch_metrics, rr = self._descend_until_contact(
                target, quat, container_id=(str(cid) if cont is not None else None))
            if status == "contact":
                touched = True
                r = rr
                self.log(f"[skills] 接触式落放到位：touchdown_z={touch_metrics['touchdown_z']:.4f} "
                         f"|F|={touch_metrics['contact_force_norm']:.2f} N "
                         f"（下降 {touch_metrics['descend_travel_mm']:.1f} mm 触到）")
            elif status == "arrived":
                # 到几何地板仍无接触：按正常流程释放并走原有校验（contact_miss 只是观测）
                self.log(f"[skills] 接触式落放未读到接触：下降到几何地板 "
                         f"z={touch_metrics['touchdown_z']:.4f}（|F|="
                         f"{touch_metrics['contact_force_norm']:.2f} N），按正常流程释放/校验")
            elif status == "miss":
                self.log(f"[skills] 接触式落放未触底：下降 {touch_metrics['descend_travel_mm']:.0f} mm "
                         f"仍未达 {PLACE_CONTACT_FORCE_N:.1f} N")
                return SkillResult(False, "place",
                                   f"接触式落放未触底：从上方 {PLACE_CONTACT_START_M * 1000:.0f} mm 起"
                                   f"下降 {touch_metrics['descend_travel_mm']:.0f} mm（上限 "
                                   f"{PLACE_CONTACT_MAX_M * 1000:.0f} mm）仍未达 "
                                   f"{PLACE_CONTACT_FORCE_N:.1f} N，请重试或修正落点 xy/像素",
                                   "place_contact_miss",
                                   metrics={"target_xy": [round(float(v), 4) for v in target_xy],
                                            "target_kind": target_kind,
                                            "support_z": round(float(support_z), 4),
                                            # 还没释放，量不到"物体"偏差；给 None 而不是假 0
                                            "dxy_target": None,
                                            "entry_clear_skipped": entry_clear_skipped,
                                            **touch_metrics})
            elif status == "blocked":
                self.log(f"[skills] 容器内落放撞壁：目标上方 "
                         f"{touch_metrics.get('blocked_gap_mm', 0.0):.0f} mm 就接触"
                         f"（|F|={touch_metrics['contact_force_norm']:.1f} N）")
                return SkillResult(False, "place",
                                   f"容器内落放撞壁（obstruction_id={cid}）：在目标上方 "
                                   f"{touch_metrics.get('blocked_gap_mm', 0.0):.0f} mm 就接触，"
                                   f"请换一个落点像素/xy（当前落点贴到容器内壁或盘沿）",
                                   "place_blocked",
                                   metrics={"target_xy": [round(float(v), 4) for v in target_xy],
                                            "target_kind": target_kind,
                                            "support_z": round(float(support_z), 4),
                                            "dxy_target": None,
                                            "obstruction_id": str(cid), **touch_metrics})
            else:
                rr = rr if rr is not None else MoveResult(False, "descend", "descend_failed")
                extra, override, note = self._finish_failure(quat, self.holding)
                return SkillResult(False, "place", f"下降到放置位失败: {rr.message}" + note,
                                   override or (rr.error_code or "descend_failed"),
                                   metrics={"touched": False,
                                            "target_xy": [round(float(v), 4) for v in target_xy],
                                            "target_kind": target_kind,
                                            "support_z": round(float(support_z), 4),
                                            "dxy_target": None,
                                            **touch_metrics, **extra})
        else:
            r = self._goto_planned(above, quat, STIFF_COMPLIANCE)
            if not r.ok:
                extra, override, note = self._finish_failure(quat, self.holding)
                return SkillResult(False, "place", f"移动到放置位上方失败: {r.message}" + note,
                                   override or (r.error_code or "approach_failed"), metrics=extra)

            # 守卫式下降（guarded move）：几何算出的 target 只是上限，真正"到位"由接触力判定。
            # force_limit=6 N 一到就停，不继续压；容器/托盘厚度与边缘造成的几毫米几何误差由
            # 导纳吸收。实测旧实现（PLACE_COMPLIANCE, 20 N）先 21.6 N 中止，随后卡死到
            # 400~500 N——所以这里把"因 force_limit 中止"视为【轻触到位】，而不是失败。
            # lead 裁决：两段式——先原速到目标上方 10mm，最后 10mm 才限速 CONTACT_VMAX
            _above10 = np.array([float(target[0]), float(target[1]),
                                 float(target[2]) + float(PLACE_CONTACT_LAST10_M)])
            r = self.runner.move_to_pose(_above10, quat, compliance=PLACE_TOUCH_COMPLIANCE,
                                         duration=1.5, timeout=8.0)
            if r.ok:
                r = self._move_vmax(target, quat, PLACE_TOUCH_COMPLIANCE,
                                    self._vmax_timeout(float(PLACE_CONTACT_LAST10_M), PLACE_VMAX,
                                                       CONTACT_VMAX_TIMEOUT_MIN_S),
                                    None, PLACE_VMAX)
            if not r.ok:
                if r.error_code == "force_limit":
                    touched = True
                    # 注：检测阈值是 6 N，但 500Hz 步进下单步可能过冲到更高值（实测深压时
                    # 峰值 ~20 N）；关键是它停住了，不会继续压到 400~500 N 卡死。
                    self.log(f"[skills] 放置下降轻触到位（峰值 {r.max_force:.2f} N 超过 "
                             f"{PLACE_TOUCH_COMPLIANCE.force_limit:.1f} N 阈值，已停），按到位处理")
                else:
                    extra, override, note = self._finish_failure(quat, self.holding)
                    return SkillResult(False, "place", f"下降到放置位失败: {r.message}" + note,
                                       override or (r.error_code or "descend_failed"),
                                       metrics={"touched": False, **extra})

        # task-32 要求 2：记录"刚释放那一刻"的位姿（必须在松爪/落定之前取），
        # 抬升后再观察 PLACE_STACK_OBSERVE_S 秒，用这段时间的 xy 位移 / z 下降判断滑落或翻滚。
        # 放在松爪之前也更贴合语义：松爪之后的任何位移都算"释放后"的运动。
        pose_at_release = None
        if use_stack:
            try:
                pose_at_release = np.asarray(self.env.object_pose(str(self.holding)), float).copy()
            except Exception:  # noqa: BLE001
                pose_at_release = None
        self.env.set_gripper(GRIPPER_OPEN)
        self._settle(0.5)                      # 让物体落下并稳定
        # 要求 4：释放后强制向上抬起，把指垫彻底带离被放物体（旧实现只抬到 above，
        # 一旦这次抬升失败/被挡，指垫就继续压着物体，形成 20~50 N 的"无中生有"接触）。
        lift_target = target + np.array([0.0, 0.0, PLACE_LIFT_CLEARANCE_M])
        r_lift = self.runner.move_to_pose(lift_target, quat, duration=2.0,
                                          compliance=soft, timeout=6.0)
        if (not r_lift.ok) or self._force_norm() > PLACE_FORCE_MAX_N:
            self.log("[skills] 释放后抬升未清空接触，执行多方向脱身")
            self._escape_contact(quat, self.holding)
        if use_stack:
            self._settle(float(PLACE_STACK_OBSERVE_S))   # 叠放：额外静置观察

        held = self.holding
        self._refresh_holding_if_released()      # N2：释放后按实际夹爪刷新
        self.holding = None
        if held is None:
            return SkillResult(True, "place", "已释放（未记录夹持物，跳过校验）",
                               metrics={"target_kind": target_kind, "touched": bool(touched),
                                        "target_xy": [round(float(v), 4) for v in target_xy],
                                        "entry_clear_skipped": entry_clear_skipped,
                                        "xy_source": xy_source, **touch_metrics})
        obj = self._object(held)
        if obj is None:
            return SkillResult(True, "place", f"已释放 {held}（物体已不在场景中）",
                               metrics={"target_kind": target_kind, "touched": bool(touched),
                                        "target_xy": [round(float(v), 4) for v in target_xy],
                                        "entry_clear_skipped": entry_clear_skipped,
                                        "xy_source": xy_source, **touch_metrics})

        # 要求 2：四条判据同时满足才算放置成功（旧实现只看 |dz|<0.12，把"摞在别的物体上"
        # 也算成功——实测 box_red 摞在 tray 里的 box_blue 上，dz=59 mm 却报 ok）。
        half_held = float(obj.size[2])
        bottom_z = float(obj.pose[2] - half_held)
        dxy_target = float(np.linalg.norm(np.asarray(obj.pose[:2], float) - target_xy))
        bottom_gap = bottom_z - float(support_z)
        # 叠放专用：footprint + 释放后稳定性
        slide = 0.0
        drop = 0.0
        stable_ok = True
        footprint_ok = True
        if use_stack:
            dxa = abs(float(obj.pose[0] - float(anchor.pose[0])))
            dya = abs(float(obj.pose[1] - float(anchor.pose[1])))
            footprint_ok = (dxa <= float(anchor.size[0]) + PLACE_STACK_FOOTPRINT_TOL_M
                            and dya <= float(anchor.size[1]) + PLACE_STACK_FOOTPRINT_TOL_M)
            if pose_at_release is not None:
                post = np.asarray(obj.pose, float)
                slide = float(np.linalg.norm(post[:2] - pose_at_release[:2]))
                drop = float(pose_at_release[2] - post[2])
                stable_ok = (slide < PLACE_STACK_SLIDE_MAX_M
                             and drop < PLACE_STACK_DROP_MAX_M)
        else:
            footprint_ok = dxy_target < PLACE_XY_TOL_M

        if use_stack:
            # task-32 要求 2：底面贴住锚点顶面
            seat_ok = PLACE_STACK_GAP_MIN_M <= bottom_gap <= PLACE_STACK_GAP_MAX_M
        elif use_side or target_kind == "table":
            # 桌面：底面必须贴在桌面上表面附近
            seat_ok = PLACE_SEAT_MIN_M <= bottom_gap <= PLACE_SEAT_MAX_M
        else:
            # 容器（托盘）：物体是"落进盘里"的，底面天然低于盘口（support_z=盘口），
            # 所以不能拿盘口当支撑面。判据改为：底面不得穿到桌面以下，也不得高出盘口
            # 2 cm 以上（后者正是"摞在别的物体上/搁在盘沿"的 dz=59 mm 伪成功）。
            seat_ok = (bottom_z >= float(TABLE_HEIGHT) + PLACE_SEAT_MIN_M
                       and bottom_z <= float(support_z) + PLACE_SEAT_MAX_M)
        # task-36：并排还要核对"实际偏移方向与请求方向一致 + 中心距在区间内"
        side_ok = True
        side_dist = None
        side_dot = None
        if use_side:
            delta = np.asarray(obj.pose[:2], float) - np.asarray(anchor.pose[:2], float)
            side_dist = float(np.linalg.norm(delta))
            side_dot = float(np.dot(delta, np.asarray(side_dir, float)))
            side_ok = (side_dot > 0.0
                       and (float(side_expected_dist) - PLACE_SIDE_DIST_MIN_M
                            <= side_dist
                            <= float(side_expected_dist) + PLACE_SIDE_DIST_MAX_M))
        f_now = self._force_norm()
        gripper_now = self._gripper_now()
        ok = (footprint_ok and seat_ok and stable_ok and side_ok
              and f_now <= PLACE_FORCE_MAX_N
              and gripper_now >= GRIPPER_OPEN_MIN)
        code = None if ok else "place_mismatch"
        metrics = {
            "target_kind": target_kind,
            "target_xy": [round(float(v), 4) for v in target_xy],
            "xy": [round(float(v), 4) for v in np.asarray(obj.pose[:2], float)],
            "dxy_target": round(dxy_target, 4),
            "dz": round(float(obj.pose[2] - support_z), 4),
            "support_z": round(float(support_z), 4),
            "bottom_z": round(bottom_z, 4),
            "bottom_gap": round(bottom_gap, 4),
            "bottom_vs_table": round(bottom_z - float(TABLE_HEIGHT), 4),
            "seat_ok": bool(seat_ok),
            "force": round(float(f_now), 3),
            "gripper": round(float(gripper_now), 3),
            "touched": bool(touched),
            "touch_force": round(float(getattr(r, "max_force", 0.0) or 0.0), 3),
            "spot": spot_note,
            "spot_reason": spot_note,
            "descend_until_contact": bool(descend_until_contact),
            "entry_clear_skipped": entry_clear_skipped,
            "xy_source": xy_source,
            **touch_metrics,
        }
        if use_side:
            # task-36：写清 anchor / relation / 方向向量 / 方向依据 / 期望与实际中心距 / 方向点积
            metrics.update({
                "anchor": str(side_anchor_id),
                "relation": str(relation),
                "relation_dir": [float(v) for v in np.asarray(side_dir, float)],
                "relation_reason": str(spot_note),
                "expected_dist": round(float(side_expected_dist), 4),
                "actual_dist": None if side_dist is None else round(float(side_dist), 4),
                "side_dot": None if side_dot is None else round(float(side_dot), 4),
                "side_ok": bool(side_ok),
            })
        if use_stack:
            # task-32 要求 4：写清 anchor / anchor_top_z / stack_z / xy_offset / dz，便于核对
            metrics.update({
                "anchor": str(anchor_id),
                "anchor_top_z": round(float(anchor_top_z), 4),
                "stack_z": round(float(obj.pose[2]), 4),
                "xy_offset": [round(float(v), 4) for v in np.asarray(xy_offset, float)],
                "dz": round(float(obj.pose[2] - float(anchor_top_z)), 4),
                "footprint_ok": bool(footprint_ok),
                "stable_ok": bool(stable_ok),
                "slide": round(float(slide), 4),
                "drop": round(float(drop), 4),
            })
        if use_stack:
            msg = (f"已把 {held} 叠放到 {anchor_id} 上（xy=({obj.pose[0]:.3f},{obj.pose[1]:.3f}) "
                   f"锚点顶面 z={float(anchor_top_z):.3f}，物体中心 z={obj.pose[2]:.3f}，"
                   f"底面间隙 {bottom_gap * 1000:+.0f} mm，|F| {f_now:.2f} N，夹爪 {gripper_now:.2f}，"
                   f"观察 {PLACE_STACK_OBSERVE_S:.1f}s 滑移 {slide * 1000:.0f} mm/下沉 "
                   f"{drop * 1000:.0f} mm）")
        elif use_side:
            msg = (f"已把 {held} 放到 {side_anchor_id} 的 {relation} 侧"
                   f"（xy=({obj.pose[0]:.3f},{obj.pose[1]:.3f})，实际中心距 "
                   f"{side_dist * 1000:.0f} mm / 期望 {float(side_expected_dist) * 1000:.0f} mm，"
                   f"方向点积 {side_dot:.3f}，|F| {f_now:.2f} N，夹爪 {gripper_now:.2f}）")
        else:
            msg = (f"已放置 {held} 到{target_label}（xy 偏差 {dxy_target * 1000:.0f} mm, "
                   f"底面 {bottom_z * 1000:.0f} mm（相对支撑面 {bottom_gap * 1000:+.0f} mm）, "
                   f"|F| {f_now:.2f} N, 夹爪 {gripper_now:.2f}）")
        if not ok:
            if use_side:
                msg = (f"并排偏差过大：实际中心距 {side_dist * 1000:.0f} mm"
                       f"（期望 {float(side_expected_dist) * 1000:.0f} mm，允许 "
                       f"[{(float(side_expected_dist) - PLACE_SIDE_DIST_MIN_M) * 1000:.0f}, "
                       f"{(float(side_expected_dist) + PLACE_SIDE_DIST_MAX_M) * 1000:.0f}] mm）、"
                       f"方向点积 {side_dot:.3f}（需 >0，即确实在锚点 {relation} 侧）、"
                       f"底面离桌面 {bottom_gap * 1000:+.0f} mm、|F| {f_now:.2f} N、"
                       f"夹爪 {gripper_now:.2f}")
            else:
                msg = (f"放置偏差过大：xy 偏差 {dxy_target * 1000:.0f} mm（限 "
                       f"{PLACE_XY_TOL_M * 1000:.0f} mm）、物体底面 z={bottom_z * 1000:.0f} mm"
                       f"（支撑面 {support_z * 1000:.0f} mm，允许区间 "
                       f"[{(support_z + PLACE_SEAT_MIN_M) * 1000:.0f}, "
                       f"{(support_z + PLACE_SEAT_MAX_M) * 1000:.0f}] mm；桌面 {TABLE_HEIGHT * 1000:.0f} mm）、"
                       f"|F| {f_now:.2f} N（限 {PLACE_FORCE_MAX_N:.1f}）、夹爪 {gripper_now:.2f}")
        return SkillResult(ok, "place", msg, code, metrics=metrics)

    def return_held_object(self) -> SkillResult:
        """【安全网】任务中止时把当前夹着的物体放回去，绝不允许留在悬空状态。

        优先级（metrics.mode）：
          1) 'pick_pose'  ：优先放回拾取时的位姿 self.last_pick_pose（要求 xy 在桌面内、
                            与其它物体有净距）；
          2) 'table_spot' ：否则就近选一个桌面空位（复用 _choose_table_spot）；
          3) 'drop'       ：连目标上方/下降都失败时，退化为"原地降到桌面高度并松爪"。
        返回 message 写明放在哪儿；metrics 带 returned_to_xy / mode / bottom_gap / force。
        调用后 self.holding 必为 None。
        """
        held = self.holding
        if held is None:
            return SkillResult(True, "return_held", "当前未夹持物体，无需归还",
                               metrics={"mode": "none", "returned_to_xy": None})
        obj = self._object(held)
        half_z = float(obj.size[2]) if obj is not None else 0.02
        half_xy = float(max(obj.size[0], obj.size[1])) if obj is not None else 0.02
        quat = approach_quat(0.0)
        mode = None
        target_xy = None
        note = ""

        if self.last_pick_pose is not None:
            p = np.asarray(self.last_pick_pose, float).reshape(-1)
            if (self._xy_on_table(p[:2], half_xy)
                    and self._spot_clear(p[:2], half_xy, held, 0.05)):
                target_xy = np.asarray(p[:2], float).copy()
                mode = "pick_pose"
                note = f"回到拾取位姿 {np.round(target_xy, 3).tolist()}"
        if target_xy is None:
            try:
                target_xy, spot_note = self._choose_table_spot(obj, quat)
                mode = "table_spot"
                note = "就近桌面空位 " + str(spot_note)
            except Exception as exc:  # noqa: BLE001
                target_xy = np.asarray(PLACE_TABLE_ANCHOR, float).copy()
                mode = "table_spot"
                note = f"空位选择异常({exc})，退回锚点"

        tz = float(TABLE_HEIGHT) + half_z + PLACE_TABLE_MARGIN_M
        target = np.array([float(target_xy[0]), float(target_xy[1]), float(tz)])
        above = target + np.array([0.0, 0.0, 0.12])
        r = self._goto_planned(above, quat, STIFF_COMPLIANCE, timeout=30.0)
        if not r.ok:
            self.log(f"[skills] 归还：到 {note} 上方失败（{r.error_code}），改为原地下降")
            cur = np.asarray(self.env.site_pose(TCP_SITE)[0], float)
            target = np.array([float(cur[0]), float(cur[1]), float(tz)])
            target_xy = target[:2].copy()
            above = target + np.array([0.0, 0.0, 0.12])
            mode = "drop"
            note = "原地降到桌面高度（目标上方不可达）"
            r = self._goto_planned(above, quat, STIFF_COMPLIANCE, timeout=30.0)

        # 归还/落放下降同样两段式：原速到目标上方 10mm，最后 10mm 限速
        _above10b = np.array([float(target[0]), float(target[1]),
                              float(target[2]) + float(PLACE_CONTACT_LAST10_M)])
        r2 = self.runner.move_to_pose(_above10b, quat, compliance=PLACE_TOUCH_COMPLIANCE,
                                      duration=1.5, timeout=8.0)
        if r2.ok:
            r2 = self._move_vmax(target, quat, PLACE_TOUCH_COMPLIANCE,
                                 self._vmax_timeout(float(PLACE_CONTACT_LAST10_M), PLACE_VMAX,
                                                    CONTACT_VMAX_TIMEOUT_MIN_S),
                                 None, PLACE_VMAX)
        if (not r2.ok) and r2.error_code != "force_limit":
            self.log(f"[skills] 归还：下降未完成（{r2.error_code}: {r2.message}）")
            mode = "drop"
        self.env.set_gripper(GRIPPER_OPEN)
        self._settle(0.5)
        lift_target = target + np.array([0.0, 0.0, PLACE_LIFT_CLEARANCE_M])
        r3 = self.runner.move_to_pose(lift_target, quat, duration=2.0,
                                      compliance=PLACE_COMPLIANCE, timeout=6.0)
        if (not r3.ok) or self._force_norm() > PLACE_FORCE_MAX_N:
            self.log("[skills] 归还：抬升未清空接触，执行多方向脱身")
            self._escape_contact(quat, held)
        self._refresh_holding_if_released()
        self.holding = None

        gap = None
        obj_after = self._object(held)
        if obj_after is not None:
            gap = float(obj_after.pose[2] - float(obj_after.size[2]) - float(TABLE_HEIGHT))
        f_now = self._force_norm()
        suspended = bool(gap is not None and gap > 0.05)
        if suspended:
            mode = "drop"
        ok = (not suspended) and f_now <= PLACE_FORCE_MAX_N
        gap_mm = "?" if gap is None else f"{gap * 1000:+.0f}"
        if suspended:
            msg = (f"归还 {held} 未落到支撑面（底面离桌面 {gap_mm} mm，mode=drop）")
        else:
            msg = (f"已归还 {held}：{note}（mode={mode}，底面离桌面 {gap_mm} mm，"
                   f"|F| {f_now:.2f} N）")
        obs, obs_note = self._obstruction()          # task-44/R5：受阻物观测
        if not ok and obs_note:
            msg = msg + obs_note
        return SkillResult(bool(ok), "return_held", msg,
                           None if ok else "return_incomplete",
                           metrics={"mode": str(mode),
                                    "returned_to_xy": [round(float(v), 4) for v in target_xy],
                                    "bottom_gap": None if gap is None else round(gap, 4),
                                    "force": round(float(f_now), 3),
                                    "suspended": bool(suspended), **obs})

    # ---------------- 推 ---------------- #
    def _skill_push(self, call: SkillCall) -> SkillResult:
        args = call.args or {}
        tid = args.get("target_id")
        obj = self._object(tid) if tid else None
        if obj is None:
            return SkillResult(False, "push", f"物体不存在: {tid}", "object_not_found")
        direction = np.asarray(args.get("direction", [1.0, 0.0, 0.0]), float)
        n = float(np.linalg.norm(direction))
        if n < 1e-6:
            return SkillResult(False, "push", "direction 不能为零向量", "bad_args")
        direction = direction / n
        distance = float(args.get("distance", 0.08))
        soft = self._compliance_from_call(call, SOFT_COMPLIANCE)
        quat = approach_quat(float(args.get("yaw", 0.0)))

        p = obj.pose[:3].copy()
        start = p - direction * 0.08
        r = self._goto_planned(start, quat, STIFF_COMPLIANCE)
        if not r.ok:
            return SkillResult(False, "push", f"接近失败: {r.message}", r.error_code or "approach_failed")
        before = obj.pose[:3].copy()
        r = self.runner.move_to_pose(p + direction * distance, quat, compliance=soft, timeout=8.0)
        after = self.env.object_pose(tid)[:3]
        moved = float(np.linalg.norm(after - before))
        ok = moved > 0.01
        return SkillResult(ok, "push", f"推动 {moved * 1000:.0f} mm",
                           None if ok else "push_failed",
                           metrics={"moved": moved, "max_force": r.max_force})
