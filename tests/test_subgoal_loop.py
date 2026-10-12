"""tests/test_subgoal_loop.py —— 子目标闭环重试语义（INTERFACES.md §5.8-3）。

依赖 agent/taskplan.py（§5.2）。未落地时整模块 skip（pending），并打印原因——不放宽断言。

断言：
  * g1 前两次失败、第三次成功 -> 最终必须报「成功（重试后）」且 done=True、attempts=3；
  * 三次全失败 -> 必须报 subgoal_failed，**不得**谎报成功；
  * 每个子目标使用新的 epoch（history 不串）。

运行：
    MUJOCO_GL=glfw DISPLAY=:0 /home/tc63/mujoco/.venv/bin/python -m tests.test_subgoal_loop
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_IMPORT_ERROR = ""
taskplan = None
try:
    from agent import taskplan as taskplan
except Exception as exc:                                  # noqa: BLE001
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

_PENDING = taskplan is None
_SKIP = f"agent.taskplan 未落地（§5.2 pending: {_IMPORT_ERROR}）"


class _FakeVerifier:
    """按预设序列返回 verify 结果；忽略真实视觉。"""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def verify(self, sg, *, metrics=None, rgb_before=None):
        self.calls += 1
        done = self.outcomes.pop(0) if self.outcomes else False
        return {"done": bool(done), "evidence": "geometry" if done else "none",
                "why": "fake"}


class _FakeLoop:
    """最小 AgentLoop 替身：只记录被喂的 instruction。"""

    def __init__(self, state, instruction):
        self.state = state
        self.instruction = instruction
        self.history: list = []

    def run(self, instruction=None, history=None):
        self.state["runs"] += 1
        self.state["tasks"].append(instruction if instruction is not None else self.instruction)
        return []


def _fake_loop_factory(*, succeed_on_attempt):
    """orch 会按 (sg, attempt) 调用；返回能记录 instruction 的假 loop。"""
    state = {"runs": 0, "tasks": []}

    def factory(sg, attempt=1):
        return _FakeLoop(state, getattr(sg, "instruction", str(sg)))

    return factory, state


@unittest.skipUnless(not _PENDING, _SKIP)
class TestSubGoalRetry(unittest.TestCase):
    def _sg(self):
        return taskplan.SubGoal(id="g1", instruction="把红色方块放到托盘里",
                                goal_type="place_in_container", subject="红色方块",
                                subject_color="red", container="托盘", container_color=None,
                                anchor_color=None, verify_hint="红块在托盘框内")

    def test_two_fails_then_success(self):
        vf = _FakeVerifier([False, False, True])
        factory, state = _fake_loop_factory(succeed_on_attempt=3)
        orch = taskplan.SubGoalOrchestrator(env=None, vlm=None, loop_factory=factory, verifier=vf)
        out = orch.run("把红色方块放到托盘里", [self._sg()])
        print(f"\n[probe] 2 败 1 成 -> {out}")
        self.assertEqual(len(out), 1)
        rec = out[0]
        self.assertTrue(rec["done"])
        self.assertEqual(int(rec["attempts"]), 3)
        self.assertIn("重试", str(rec.get("evidence", "")) + str(rec.get("why", "")))

    def test_three_fails_reports_subgoal_failed(self):
        vf = _FakeVerifier([False, False, False])
        factory, state = _fake_loop_factory(succeed_on_attempt=99)
        orch = taskplan.SubGoalOrchestrator(env=None, vlm=None, loop_factory=factory, verifier=vf)
        out = orch.run("把红色方块放到托盘里", [self._sg()])
        print(f"[probe] 3 败 -> {out}")
        rec = out[0]
        self.assertFalse(rec["done"], "三次全败不得报成功")
        self.assertEqual(int(rec["attempts"]), taskplan.SUBGOAL_MAX_ATTEMPTS)
        self.assertIn("subgoal_failed", str(rec.get("why", "")) + str(rec.get("evidence",
                                                                            "")))



class _ReviewEnv:
    """复核用最小环境：render 返回小图；get_state 故意失败（复核不得依赖真值快照）。"""

    def render(self, cam: str = "top", **kw):
        import numpy as np
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        img[:, :] = 100
        return img

    def get_state(self):
        raise RuntimeError("fake env: no state")


class _ReviewPerception:
    """可编程 check_inside：记录被喂进来的容器框，用来证明「复核用的是缓存框」。"""

    def __init__(self, inside: bool):
        self.inside = inside
        self.seen_bbox = None

    def color_blob(self, rgb, color, hint_uv=None, **kw):
        return {"found": False, "uv": None, "bbox": None}      # 强制走缓存路径

    def refine_pixel(self, rgb, color, hint_uv, **kw):
        return {"uv": hint_uv, "bbox": None, "refined": False, "source": "vlm_raw",
                "delta_px": 0.0}

    def check_inside(self, rgb, subject_color, container_bbox, **kw):
        self.seen_bbox = list(container_bbox)
        return {"ok": bool(self.inside), "why": "inside" if self.inside else "not_inside",
                "metrics": {"subject_uv": [1.0, 2.0], "container_bbox": list(container_bbox)}}


@unittest.skipUnless(not _PENDING, _SKIP)
class TestReviewReusesActionBboxButStillChecksImage(unittest.TestCase):
    """① 复核复用动作阶段容器框：只影响「框从哪来」，不影响「是否完成」的判据。

    核心安全断言：缓存框本身**不能**让「未完成」变成完成 —— 完成与否仍由 check_inside
    在**当前帧**上判定；框缺失/判不了时返回 None（交给 VLM 复核），绝不默认 done。
    """

    def _verifier(self, inside: bool):
        perc = _ReviewPerception(inside)
        v = taskplan.SubGoalVerifier(_ReviewEnv(), perception=perc, vlm=None)
        return v, perc

    def _sg(self):
        return taskplan.SubGoal(id="g1", instruction="把红色方块放到托盘里",
                                goal_type="place_in_container", subject="红色方块",
                                subject_color="red", container="托盘", container_color=None,
                                anchor_color=None, verify_hint="红块在托盘框内")

    def test_cached_bbox_cannot_turn_not_done_into_done(self):
        taskplan.clear_pixel_cache()
        taskplan.remember_pixel(None, [100.0, 200.0], camera="top", bbox=[80, 180, 120, 220],
                                source="action_place", target="tray")
        v, perc = self._verifier(inside=False)          # 当前帧里主体不在框内 -> 未完成
        verdict = v.verify(self._sg())
        print(f"\n[probe] 缓存框+不在框内 -> done={verdict.get('done')} "
              f"evidence={verdict.get('evidence')} bbox={perc.seen_bbox}")
        self.assertFalse(verdict.get("done"), f"未完成不得判完成: {verdict}")
        self.assertEqual(perc.seen_bbox, [80, 180, 120, 220], "应当用缓存下来的容器框")
        # 几何否决不等于最终完成：verify 会把它交给 VLM 复核（本用例 vlm=None -> evidence=none），
        # 关键安全性质是 done=False —— 缓存框绝不能自己把「未完成」变成完成。
        self.assertIn(verdict.get("evidence"), ("geometry", "none"))

    def test_cached_bbox_present_and_inside_marks_done(self):
        taskplan.clear_pixel_cache()
        taskplan.remember_pixel(None, [100.0, 200.0], camera="top", bbox=[80, 180, 120, 220],
                                source="action_place", target="tray")
        v, perc = self._verifier(inside=True)
        verdict = v.verify(self._sg())
        print(f"[probe] 缓存框+在框内 -> done={verdict.get('done')} "
              f"why={verdict.get('why')}")
        self.assertTrue(verdict.get("done"))
        self.assertEqual(verdict.get("evidence"), "geometry")

    def test_no_bbox_means_inconclusive_not_done(self):
        taskplan.clear_pixel_cache()                    # 没有任何缓存 -> 框不出来
        v, _perc = self._verifier(inside=True)
        verdict = v.verify(self._sg())
        print(f"[probe] 无缓存框 -> done={verdict.get('done')} evidence={verdict.get('evidence')}")
        self.assertFalse(verdict.get("done"), f"定位不到容器框时不得判完成: {verdict}")

    def test_recall_prefers_requested_color_over_latest(self):
        taskplan.clear_pixel_cache()
        taskplan.remember_pixel("red", [10.0, 20.0], camera="top", bbox=[0, 10, 20, 30],
                                target="红色方块")
        taskplan.remember_pixel("blue", [200.0, 100.0], camera="top", bbox=[190, 90, 210, 110],
                                target="蓝色方块")
        hit = taskplan.recall_pixel("red", camera="top")
        print(f"[probe] recall(red)={hit.get('uv') if hit else None}")
        self.assertEqual(list(hit["uv"]), [10.0, 20.0], "给了颜色就必须走精确命中，不得取最新")
        self.assertIsNone(taskplan.recall_pixel("green", camera="top"),
                          "颜色对不上且未允许 allow_latest 时必须返回 None（不猜）")

if __name__ == "__main__":
    if _PENDING:
        print(f"[PENDING] {_SKIP} —— 需要 agent/taskplan.py 落地后本模块才会真正执行")
    unittest.main(verbosity=2)
