"""tests/test_vlm_retry.py —— task-67 回归：VLM 空回复不再杀死任务（无人值守、无需窗口）。

背景（2026-10-12 实机）：任务『绿色方块放到红色方块上』第 1 轮 step 4 死于
    [step 4/12] VLM 异常: 响应既没有 tool_calls，content 也为空
AgentLoop 直接 break；当时绿块已在手里、红块像素已标定，只差最后一个 place。

根因：老实现只在传输层（_post -> _post_once）重试，而『空回复/解析失败』是在 decide() 里抛的，
retry 循环根本包不住它；且 _retryable_response 只匹配中文字符串『没有可解析的工具调用』，
与 decide 实际抛的两条消息都不匹配 -> 一次都不重试。

本文件锁死修复后的行为（全部离线，可用 ./run.sh test 跑）：
  1) decide() 内对『空回复/无工具调用』做真实重试，且重试时提高 max_tokens；
  2) 重试耗尽后抛 VLMResponseError，文本里能看出重试过、带 error_code；
  3) retryable 判据是异常属性（稳定），不再依赖中文子串；
  4) AgentLoop 允许连续 2 次 vlm_error 才中止，偶发空回复任务仍能跑完。
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.types import SkillCall, SkillResult                          # noqa: E402
from agent.loop import AgentLoop                                       # noqa: E402
from agent.vlm_client import (VLMClient, VLMResponseError,             # noqa: E402
                              _retryable_response)


def _empty(reasoning: str = "长思维链" * 50):
    return {"choices": [{"message": {"content": "", "reasoning": reasoning}}]}


def _ok(name: str = "home", args: str = "{}"):
    return {"choices": [{"message": {"tool_calls": [
        {"function": {"name": name, "arguments": args}}]}}]}


class _FakePostOnce:
    """替掉 VLMClient._post_once：按脚本回响应，并记录每次请求体。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads: list = []

    def __call__(self, data: bytes):
        self.payloads.append(json.loads(data.decode("utf-8")))
        return self.responses.pop(0) if self.responses else _empty()


class TestDecideLevelRetry(unittest.TestCase):
    def _client(self, responses, retries=3):
        c = VLMClient(api_key="dummy", max_retries=retries, sleep=lambda _s: None)
        fake = _FakePostOnce(responses)
        c._post_once = fake            # type: ignore[method-assign]
        return c, fake

    def test_empty_then_ok_retries_with_grown_token_budget(self):
        c, fake = self._client([_empty(), _ok("home")])
        call = c.decide([{"role": "user", "content": "回家"}])
        self.assertEqual(call.skill, "home")
        self.assertEqual(len(fake.payloads), 2, "空回复必须触发一次真实重试")
        first, second = fake.payloads
        self.assertEqual(first["max_tokens"], c.max_tokens)
        self.assertGreater(second["max_tokens"], first["max_tokens"])
        self.assertEqual(second["max_tokens"], min(first["max_tokens"] * 2, 8192))
        self.assertNotEqual(first["messages"], second["messages"],
                            "重试 payload 要附加强制工具调用提示")
        self.assertEqual(len(first["messages"]), 1, "首次请求不动调用方 messages")
        self.assertEqual(c.last_decide_attempts, 2)

    def test_all_empty_gives_up_with_retry_evidence(self):
        c, fake = self._client([_empty() for _ in range(4)], retries=2)
        with self.assertRaises(VLMResponseError) as cm:
            c.decide([{"role": "user", "content": "x"}])
        self.assertEqual(len(fake.payloads), 3, "retries=2 -> 3 次尝试")
        text = str(cm.exception)
        self.assertIn("重试", text)
        self.assertIn("max_tokens", text)
        self.assertEqual(cm.exception.error_code, "empty_response")
        self.assertFalse(cm.exception.retryable, "耗尽重试的最终异常不再标可重试")

    def test_retryable_criterion_is_attribute_based(self):
        self.assertTrue(_retryable_response(
            VLMResponseError("随便什么文案", retryable=True, error_code="empty_response")))
        self.assertFalse(_retryable_response(
            VLMResponseError("没有可解析的工具调用")))          # 旧字符串判据已废弃
        self.assertFalse(_retryable_response("响应既没有 tool_calls，content 也为空"))
        c, _fake = self._client([], retries=0)
        with self.assertRaises(VLMResponseError) as cm:
            c.decide([{"role": "user", "content": "x"}])
        self.assertTrue(cm.exception.retryable, "decide 抛出的空回复必须标可重试")
        self.assertEqual(cm.exception.error_code, "empty_response")

    def test_network_error_retries_inside_post(self):
        calls = {"n": 0}

        class C(VLMClient):
            def _post_once(self, data):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise __import__("agent.vlm_client", fromlist=["x"]).VLMNetworkError("boom")
                return _ok("stop")

        c = C(api_key="dummy", max_retries=2, sleep=lambda _s: None)
        self.assertEqual(c.decide([{"role": "user", "content": "x"}]).skill, "stop")
        self.assertEqual(calls["n"], 2)


# --------------------------------------------------------------------------- #
# 环内容忍（stub env / executor，无 MuJoCo）
# --------------------------------------------------------------------------- #
class _StubEnv:
    def get_state(self):
        from core.types import ContactState, RobotState, SceneObject, SceneState
        import numpy as np
        obj = SceneObject("box_red", "box", "red", np.array([0.4, 0.1, 0.42, 1, 0, 0, 0]),
                          (0.02, 0.02, 0.02), True, False)
        tray = SceneObject("tray", "tray", "gray", np.array([0.42, 0.02, 0.405, 1, 0, 0, 0]),
                           (0.12, 0.12, 0.01), False, True)
        return SceneState(0.0, RobotState(np.zeros(6), np.zeros(6), np.zeros(3),
                                          np.array([1.0, 0, 0, 0])),
                          [obj, tray], ContactState(np.zeros(3), np.zeros(3), 0.0, False))

    def render(self, camera="overview"):
        import numpy as np
        return np.zeros((16, 16, 3), dtype=np.uint8)


class _RecordingExec:
    def __init__(self):
        self.calls: list = []

    def __call__(self, call):
        self.calls.append(call)
        return SkillResult(ok=True, skill=call.skill, message="stub ok")


class TestLoopToleratesTransientVLMErrors(unittest.TestCase):
    def test_one_transient_empty_response_still_finishes(self):
        class Flaky:
            def __init__(self):
                self.script = [SkillCall("pick", {"target_id": "box_red"}),
                               SkillCall("place", {"container_id": "tray"}),
                               SkillCall("stop", {})]
                self.n = 0

            def decide(self, messages):
                self.n += 1
                if self.n == 2:      # 第 2 步偶发空回复
                    raise VLMResponseError("响应既没有 tool_calls，content 也为空",
                                           retryable=True, error_code="empty_response")
                return self.script.pop(0) if self.script else SkillCall("stop", {})

        ex = _RecordingExec()
        logs: list = []
        loop = AgentLoop(_StubEnv(), ex, Flaky(), cameras=(), max_steps=8, log=logs.append)
        results = loop.run("把红色方块放进托盘")
        self.assertEqual([c.skill for c in ex.calls], ["pick", "place", "stop"])
        self.assertEqual([r.error_code for r in results if r.error_code], ["vlm_error"])
        self.assertTrue(results[-1].ok, "偶发空回复不应打断任务")
        self.assertTrue(any("VLM 连续异常 1/2" in m for m in logs), logs)

    def test_two_consecutive_failures_abort_with_vlm_error(self):
        class Dead:
            def __init__(self):
                self.n = 0

            def decide(self, messages):
                self.n += 1
                raise VLMResponseError("响应既没有 tool_calls，content 也为空",
                                       retryable=True, error_code="empty_response")

        dead = Dead()
        logs: list = []
        loop = AgentLoop(_StubEnv(), _RecordingExec(), dead, cameras=(), max_steps=8,
                         log=logs.append)
        results = loop.run("把红色方块放进托盘")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.error_code == "vlm_error" for r in results))
        self.assertEqual(dead.n, 2)
        self.assertTrue(any("VLM 连续异常 2/2，任务中止" in m for m in logs), logs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
