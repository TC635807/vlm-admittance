"""关节空间碰撞检测（planning.collision）。

原理
====
MuJoCo 的碰撞是"位置决定"的：给定完整 qpos 后调用 mj_forward，所有接触对
写入 data.contact（共 data.ncon 个）。本模块为每次查询使用一个
**私有的临时 MjData**（mujoco.MjData(env.model)），把 env.data.qpos
快照过来、只改写 6 个机械臂关节，再做 mj_forward。因此：

* 规划过程不会污染 env.data（仿真状态、物体位姿、夹爪自由度都不变）；
* 规划器看到的是"当前场景"（物体已经落在哪儿就按哪儿算），而不是初始场景。

三个必须踩过的坑
================
1) ncon > 0 不等于碰撞。桌面上永远有"物体与桌子"的接触，物体堆叠也有接触；
   如果直接看 ncon，所有位形都会被判非法。本模块只统计"至少一个 geom 属于
   机器人（UR5e 本体 + 挂在法兰上的夹爪）"的接触对，纯环境接触
   （物体压桌子/地板）一律忽略。
2) UR5e 的碰撞 geom 没有名字（XML 里 <geom class="collision"> 无 name，
   只有 visual mesh 才是命名过的），所以 ignore_pairs 同时支持
   **geom 名**和 **body 名**两种匹配；2F85 只有 pad 有名字（g_right_pad1 等）。
   默认忽略对按"body 是否属于夹爪子树"判定（不依赖 g_ 前缀硬编码）：
   夹爪内部 8 自由度 4 杆机构的自接触、以及手指互插，不算机械臂碰撞。
3) 父子自动过滤。MuJoCo 默认 filterparent=true，同一 body 内、以及父子 body
   之间的 geom 不会生成接触；所以只需担心"非相邻连杆"的自碰撞
   （例如 forearm 打到大臂），那正是应该判非法的。

语义约定（与 INTERFACES.md 2.3 一致）
=====================================
* 不忽略"夹爪碰到桌上的目标物体"——规划默认要求不碰任何东西；
  贴近/抓取阶段由 control 的笛卡尔运动负责。
* is_state_valid = 关节限位（core.config.ARM_JOINT_LIMITS，端点含）内且无上述碰撞。
"""
from __future__ import annotations

from typing import Iterable, Optional

import mujoco
import numpy as np

from core.config import ARM_JOINT_LIMITS
from core.types import ARM_JOINT_NAMES

__all__ = ["CollisionChecker"]


class CollisionChecker:
    """关节空间碰撞检测器。

    Parameters
    ----------
    env : MuJoCoArmEnv
        只需满足 core.interfaces.SimEnv（用到 model / data / arm_qpos_idx）。
    ignore_pairs : set[tuple[str, str]] | None
        额外忽略的接触对，元素是 (名字 A, 名字 B)，名字可以是 geom 名或 body 名，
        顺序无关。默认忽略对已包含"夹爪内部自接触"。
    """

    def __init__(self, env, ignore_pairs: Optional[Iterable[tuple[str, str]]] = None):
        self.env = env
        self.model = env.model
        #: 私有临时 MjData —— 全程只读写它，绝不碰 env.data
        self.data = mujoco.MjData(self.model)

        self.q_min = np.asarray(ARM_JOINT_LIMITS[:, 0], dtype=np.float64).copy()
        self.q_max = np.asarray(ARM_JOINT_LIMITS[:, 1], dtype=np.float64).copy()
        self.n_arm = int(self.q_min.size)

        self.qpos_idx = np.asarray(env.arm_qpos_idx, dtype=int).reshape(-1)
        if self.qpos_idx.size != self.n_arm:
            raise ValueError(
                "env.arm_qpos_idx 长度 %d != 机械臂关节数 %d" % (self.qpos_idx.size, self.n_arm)
            )

        #: 接触深度容差（米）：dist >= -min_penetration 的接触忽略不计。
        #: 默认 0，即"只要有接触就算碰撞"；若实测被网格数值噪声误报，
        #: 可在构造后把它调大（例如 1e-4）。
        self.min_penetration = 0.0

        self.ignore_pairs: set = {(str(p[0]), str(p[1])) for p in (ignore_pairs or ())}

        self._build_geometry_index()

    # ------------------------------------------------------------------ #
    # 几何索引
    # ------------------------------------------------------------------ #
    def _build_geometry_index(self) -> None:
        model = self.model
        ngeom = int(model.ngeom)
        self._geom_body = np.array([int(model.geom_bodyid[g]) for g in range(ngeom)], dtype=int)
        self._geom_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(ngeom)
        ]
        self._geom_body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
            for g in range(ngeom)
        ]

        joint_bodies = []
        for jname in ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                joint_bodies.append(int(model.jnt_bodyid[jid]))
        if not joint_bodies:
            raise RuntimeError(
                "模型里找不到任何机械臂关节 %s，无法建立机器人几何集合" % (ARM_JOINT_NAMES,)
            )

        # 机器人根 body：从任一关节 body 沿 parent 上溯到世界的直接子节点。
        root = joint_bodies[0]
        while int(model.body_parentid[root]) != 0:
            root = int(model.body_parentid[root])
        self.robot_root_body = root

        subtree = self._subtree_bodies(root)

        # 机械臂连杆 body 集合 = 所有臂关节 body 的祖先链（含 root）；
        # 机器人子树里剩下的就是夹爪（attach 进来的部分）。
        arm_chain = set()
        for b in joint_bodies:
            cur = b
            while True:
                arm_chain.add(cur)
                if cur == root:
                    break
                cur = int(model.body_parentid[cur])
        self._arm_bodies = arm_chain
        self._gripper_bodies = set(subtree) - arm_chain

        self.robot_body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in subtree
        ]
        subtree_set = set(subtree)
        self.robot_geoms = [g for g in range(ngeom) if self._geom_body[g] in subtree_set]
        self._is_robot_geom = np.isin(self._geom_body, np.array(sorted(subtree), dtype=int))
        self._is_gripper_geom = np.isin(
            self._geom_body, np.array(sorted(self._gripper_bodies), dtype=int)
        )

        self._rebuild_ignore_mask()

    def _subtree_bodies(self, root: int) -> list:
        model = self.model
        out = [root]
        frontier = [root]
        while frontier:
            nxt = []
            for b in frontier:
                for c in range(int(model.nbody)):
                    if int(model.body_parentid[c]) == b:
                        nxt.append(c)
            out.extend(nxt)
            frontier = nxt
        return out

    def _geoms_named(self, name: str) -> list:
        """按 geom 名或 body 名找出所有匹配的 geom id。"""
        hits = []
        for g in range(int(self.model.ngeom)):
            if self._geom_names[g] == name or self._geom_body_names[g] == name:
                hits.append(g)
        return hits

    def _rebuild_ignore_mask(self) -> None:
        ngeom = int(self.model.ngeom)
        mask = self._is_gripper_geom[:, None] & self._is_gripper_geom[None, :]
        for a, b in self.ignore_pairs:
            for g in self._geoms_named(a):
                for h in self._geoms_named(b):
                    mask[g, h] = True
                    mask[h, g] = True
        np.fill_diagonal(mask, False)
        self._ignore_mask = mask

    def ignore_pair(self, name_a: str, name_b: str) -> None:
        """追加一个忽略对（名字可为 geom 名或 body 名），立即生效。"""
        self.ignore_pairs.add((str(name_a), str(name_b)))
        self._rebuild_ignore_mask()

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def is_state_valid(self, q: np.ndarray) -> bool:
        """关节限位内且无碰撞则 True。"""
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size != self.n_arm:
            return False
        if not np.all(np.isfinite(q)):
            return False
        if np.any(q < self.q_min) or np.any(q > self.q_max):
            return False
        return self._collision_free(q)

    def is_edge_valid(self, q1: np.ndarray, q2: np.ndarray, steps: int = 12) -> bool:
        """关节空间直线段离散采样后逐点 is_state_valid。

        steps 是段数（采样 steps+1 个点，含两端）。直线段上的关节值一定在限位盒内
        （凸集），但仍逐点走 is_state_valid 以便统一语义与提前退出。
        """
        q1 = np.asarray(q1, dtype=np.float64).reshape(-1)
        q2 = np.asarray(q2, dtype=np.float64).reshape(-1)
        if q1.size != self.n_arm or q2.size != self.n_arm:
            return False
        steps = max(1, int(steps))
        if not self.is_state_valid(q1) or not self.is_state_valid(q2):
            return False
        for k in range(1, steps):
            s = k / steps
            if not self.is_state_valid(q1 + s * (q2 - q1)):
                return False
        return True

    def _collision_free(self, q: np.ndarray) -> bool:
        data = self.data
        # 1) 快照当前场景（物体位姿 / 夹爪自由度），2) 只覆盖 6 个臂关节
        data.qpos[:] = self.env.data.qpos[:]
        data.qpos[self.qpos_idx] = q
        # 位置决定接触；速度清零让结果可复现（不参与碰撞几何）
        data.qvel[:] = 0.0
        try:
            mujoco.mj_forward(self.model, data)
        except Exception:  # pragma: no cover - 数值异常一律当作非法
            return False

        is_robot = self._is_robot_geom
        ignore = self._ignore_mask
        min_pen = self.min_penetration
        for i in range(int(data.ncon)):
            c = data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            # 坑 1：纯环境接触（物体压桌子）不算
            if not (is_robot[g1] or is_robot[g2]):
                continue
            # 数值噪声容差
            if c.dist >= -min_pen:
                continue
            # 坑 2：夹爪内部自接触 / 用户指定的忽略对
            if ignore[g1, g2]:
                continue
            return False
        return True

    # ------------------------------------------------------------------ #
    # 调试辅助
    # ------------------------------------------------------------------ #
    def debug_contacts(self, q: np.ndarray) -> list:
        """返回 q 处涉及机器人且未被忽略的接触列表，用于排查。"""
        data = self.data
        data.qpos[:] = self.env.data.qpos[:]
        data.qpos[self.qpos_idx] = np.asarray(q, dtype=np.float64).reshape(-1)
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)
        out = []
        for i in range(int(data.ncon)):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if not (self._is_robot_geom[g1] or self._is_robot_geom[g2]):
                continue
            out.append(
                {
                    "geom1": self._geom_names[g1] or self._geom_body_names[g1],
                    "geom2": self._geom_names[g2] or self._geom_body_names[g2],
                    "dist": float(c.dist),
                    "ignored": bool(self._ignore_mask[g1, g2]),
                }
            )
        return out

    def __repr__(self) -> str:  # pragma: no cover - 调试友好
        return "CollisionChecker(n_arm=%d, robot_geoms=%d, gripper_geoms=%d, ignore_pairs=%d)" % (
            self.n_arm,
            int(self._is_robot_geom.sum()),
            int(self._is_gripper_geom.sum()),
            len(self.ignore_pairs),
        )
