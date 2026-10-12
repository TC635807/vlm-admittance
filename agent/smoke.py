"""agent.smoke —— agent 模块离线自测（标准库 unittest，无需 GUI / 无需 api_key）。

覆盖：
1. 状态序列化：JSON-ready、字段齐全、extra 进度/失败信息保留；
2. build_messages：system+history+user 结构、图像顺序稳定、data URI 可解码；
3. encode_image：PIL JPEG（base64 以 /9j/ 开头）、纯 Python PNG 兜底、异常格式报错；
4. schema：TOOLS 覆盖 10 个技能且 required 参数对齐 §2.5；
   parse_skill_call 兼容 tool_calls / 纯文本 JSON / 代码块 / name({...})，非法输入 raise ValueError；
   validate_skill_call 拒绝不存在物体、非容器、非法档位、错误类型；
5. MockVLM：颜色/容器关键词 -> pick -> place -> stop，接触力超阈值先 set_compliance(soft)；
5b. 意图路由：回家 / 停止 / 松开夹爪 / 只夹不放（pick 成功后不追加 place）；
6. AgentLoop：用 stub env/executor 跑通全链路，非法调用回灌后继续，VLM 异常不炸环，
   ask_user 立即终止（不再空转到 max_steps）；
7. 不可重试错误码（home_blocked / stuck / object_disturbed / scene_reset）与同一失败
   连续重复 2 次都立即终止，不做 12 步空转；
8. place 支持 surface='table'/'tray' + xy（与 container_id 二选一，validate 拒绝同时给/都不给）；
   MockVLM 无颜色时复用历史最近一次成功 pick 的目标；「放到桌子上不是盘子里」这类指代纠正；
9. AgentLoop 跨命令历史：构造/run 注入 seed，结束后可读回；超 max_history_messages 按整组裁剪；
   不注入时每次 run 仍清空（行为与原来一致）。
10. 任务纪元（task-29）：纪元标记内联在每条命令第一条 assistant 消息里；picked/placed/
   pick_fails 只看当前纪元并按目标归属，跨命令不泄漏（D1 无上下文纠正转 ask_user、
   D2 抓取失败计数不跨目标、D3 历史 placed 不短路新任务）。
"""
from __future__ import annotations

import base64
import json
import unittest
from unittest import mock

import numpy as np

from core.config import TABLE_HEIGHT
from core.types import (
    ComplianceParams, ContactState, RobotState, SceneObject, SceneState, SkillCall, SkillResult,
)

from agent import state as state_mod
from agent.loop import AgentLoop
from agent.schema import (
    ALLOWED_ARGS, GOAL_TYPES, GRASP_PROFILES, LOCATE_CAMERAS, PROFILE_NAMES, SKILL_NAMES, TOOLS,
    compliance_from_profile, parse_skill_call, system_prompt, take_corrections,
    validate_skill_call,
)
from agent.state import (build_messages, decode_data_uri, derive_relations, encode_image,
                         serialize_state)
from agent.taskplan import (SUBGOAL_MAX_ATTEMPTS, SubGoal, SubGoalOrchestrator,
                            SubGoalVerifier, canonical_color, clear_pixel_cache,
                            fallback_subgoals, plan_subgoals, recall_pixel, remember_pixel)
from agent.vlm_client import (
    MockVLM, VLMClient, VLMError, VLMResponseError, _pick_only, _placement_surface,
    _retryable_response, _wants_correction, _wants_home, _wants_open, _wants_stop, make_vlm,
)


# --------------------------------------------------------------------------- #
# 合成场景（id 命名遵循 Lead 约定：box_red / box_blue / box_green / can_yellow / tray）
# --------------------------------------------------------------------------- #
def _obj(oid, cat, color, pos, size, graspable=True, is_container=False):
    return SceneObject(
        id=oid, category=cat, color=color,
        pose=np.array([pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0], dtype=float),
        size=(float(size[0]), float(size[1]), float(size[2])),
        graspable=graspable, is_container=is_container,
    )


def default_objects():
    return [
        _obj("box_red", "box", "red", [0.40, 0.10, 0.42], [0.025, 0.025, 0.025]),
        _obj("box_blue", "box", "blue", [0.45, -0.10, 0.42], [0.025, 0.025, 0.025]),
        _obj("box_green", "box", "green", [0.35, -0.15, 0.42], [0.025, 0.025, 0.025]),
        _obj("can_yellow", "cylinder", "yellow", [0.50, 0.15, 0.45], [0.02, 0.02, 0.05]),
        _obj("tray", "tray", "gray", [0.42, 0.02, 0.405], [0.12, 0.12, 0.01],
             graspable=False, is_container=True),
    ]


def make_state(force=(0.0, 0.0, 0.0), objects=None, holding=None) -> SceneState:
    force = np.asarray(force, dtype=float)
    return SceneState(
        time=1.25,
        robot=RobotState(
            qpos=np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0]),
            qvel=np.zeros(6),
            tcp_pos=np.array([0.4, 0.0, 0.5]),
            tcp_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper=1.0,
            holding=holding,
        ),
        objects=list(objects if objects is not None else default_objects()),
        contact=ContactState(force=force, torque=np.zeros(3),
                             max_force=float(np.linalg.norm(force)),
                             in_contact=bool(np.linalg.norm(force) > 1e-6)),
        joint_names=list(("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")),
        meta={"source": "smoke"},
    )


def make_image(h=16, w=16) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 200
    img[..., 1] = 100
    img[h // 2:, :, 2] = 255
    return img


def _tool_by_name(name: str) -> dict:
    for t in TOOLS:
        if t["function"]["name"] == name:
            return t
    raise AssertionError(f"TOOLS 缺少技能 {name}")


def history_of(*observations) -> list[dict]:
    """把若干 SkillResult 的 observation 拼成 history（MockVLM 只看 observation JSON）。"""
    return [
        {"role": "user", "content": "OBSERVATION_JSON " + json.dumps(obs.to_observation())}
        for obs in observations
    ]


def step_messages(call: SkillCall, result: SkillResult) -> list[dict]:
    """一步历史 = assistant 调用 JSON + user 观察 JSON（与 AgentLoop 写入格式一致）。

    MockVLM 要从中反解「最近一次成功 pick 的 target_id」和「最近一次成功 place 的参数」，
    所以历史里必须保留 assistant 的调用 args，只放 observation 是不够的。
    """
    return [
        {"role": "assistant", "content": json.dumps(call.to_json(), ensure_ascii=False)},
        {"role": "user", "content": "OBSERVATION_JSON " + json.dumps(result.to_observation())},
    ]


# --------------------------------------------------------------------------- #
# stub（不越界实现别人的模块，只在自己测试里造最小桩）
# --------------------------------------------------------------------------- #
class StubEnv:
    """满足 SimEnv 中 AgentLoop 用到的两个方法：get_state / render。"""

    def __init__(self, state=None, render_ok=True):
        self._state = state or make_state()
        self.render_ok = render_ok
        self.render_calls: list[str] = []

    def get_state(self) -> SceneState:
        return self._state

    def render(self, camera="overview"):
        if not self.render_ok:
            raise RuntimeError("stub render failure")
        self.render_calls.append(camera)
        return make_image()


class RecordingExecutor:
    """记录调用顺序，返回 ok=True 的 SkillResult。"""

    def __init__(self):
        self.calls: list[SkillCall] = []

    def __call__(self, call: SkillCall) -> SkillResult:
        self.calls.append(call)
        return SkillResult(ok=True, skill=call.skill, message="stub ok", duration=0.01)


class ScriptedVLM:
    """按脚本依次返回调用；脚本耗尽后返回 stop。"""

    def __init__(self, calls):
        self.calls = list(calls)
        self.seen_messages = []

    def decide(self, messages):
        self.seen_messages.append(messages)
        if self.calls:
            return self.calls.pop(0)
        return SkillCall("stop", {}, reasoning="scripted end")


class StubSkillExecutor:
    """按 skill 返回预设 (ok, error_code, message)；未配置的 skill 返回 ok=True。"""

    def __init__(self, results=None):
        #: skill -> (ok, error_code, message)
        self.results = dict(results or {})
        self.calls: list[SkillCall] = []

    def __call__(self, call: SkillCall) -> SkillResult:
        self.calls.append(call)
        if call.skill in self.results:
            ok, code, msg = self.results[call.skill]
            return SkillResult(ok=bool(ok), skill=call.skill, message=str(msg), error_code=code)
        return SkillResult(ok=True, skill=call.skill, message="stub ok")


# --------------------------------------------------------------------------- #
# 1) 状态序列化 / 图像编码 / 消息构造
# --------------------------------------------------------------------------- #
class TestStateSerialization(unittest.TestCase):

    def test_serialize_state_is_json_ready(self):
        st = make_state(force=(1.0, 2.0, 3.0), holding="box_red")
        payload = serialize_state(st, "把红色方块放进托盘", extra={"progress": "picked", "last_error": None})
        text = json.dumps(payload, ensure_ascii=False)  # 不能抛 TypeError
        self.assertIn("box_red", text)

        for key in ("task", "time", "robot", "objects", "contact", "extra", "summary", "joint_names"):
            self.assertIn(key, payload)
        self.assertEqual(payload["task"], "把红色方块放进托盘")
        self.assertEqual(payload["extra"]["progress"], "picked")
        self.assertEqual(len(payload["robot"]["qpos"]), 6)
        self.assertAlmostEqual(payload["contact"]["force_norm"], float(np.sqrt(14.0)), places=2)

        ids = [o["id"] for o in payload["objects"]]
        self.assertEqual(ids, ["box_red", "box_blue", "box_green", "can_yellow", "tray"])
        first = payload["objects"][0]
        for k in ("id", "category", "color", "pose", "size", "graspable", "is_container"):
            self.assertIn(k, first)
        self.assertEqual(payload["summary"]["containers"], ["tray"])
        self.assertIn("can_yellow", payload["summary"]["graspable"])

    def test_serialize_state_nan_safe(self):
        st = make_state()
        st.contact.force = np.array([float("nan"), 0.0, 0.0])
        payload = serialize_state(st, "t")
        json.dumps(payload)  # NaN 不能漏进 JSON
        self.assertTrue(np.isfinite(payload["contact"]["force"][0]))

    def test_encode_image_jpeg_base64(self):
        b64 = encode_image(make_image(), fmt="jpeg", quality=85)
        self.assertTrue(b64.startswith("/9j/"), f"JPEG base64 应以 /9j/ 开头，实际 {b64[:8]!r}")
        raw = base64.b64decode(b64)
        self.assertEqual(raw[:3], b"\xff\xd8\xff", "JPEG SOI 标记不对")

    def test_encode_image_float_and_grayscale(self):
        fimg = make_image().astype(np.float32) / 255.0     # 0~1 浮点
        self.assertTrue(encode_image(fimg).startswith("/9j/"))
        gimg = np.full((12, 12), 128, dtype=np.uint8)      # 灰度 (H,W)
        self.assertTrue(encode_image(gimg).startswith("/9j/"))

    def test_png_path_and_fallback(self):
        b64 = encode_image(make_image(), fmt="png")
        raw = base64.b64decode(b64)
        self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")
        # 纯标准库 PNG 兜底编码器（PIL 缺失时走这条路）
        py_png = state_mod._encode_png_py(make_image())
        self.assertEqual(py_png[:8], b"\x89PNG\r\n\x1a\n")
        with mock.patch.object(state_mod, "_HAS_PIL", False):
            b64_fallback, mime = state_mod.encode_image_with_mime(make_image(), fmt="jpeg")
            self.assertEqual(mime, "image/png", "PIL 缺失时 jpeg 必须退化为 PNG")
            self.assertEqual(base64.b64decode(b64_fallback)[:8], b"\x89PNG\r\n\x1a\n")

    def test_encode_image_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            encode_image(make_image(), fmt="bmp")
        with self.assertRaises(ValueError):
            encode_image(np.zeros((4,), dtype=np.uint8))

    def test_build_messages_structure_and_stable_image_order(self):
        st = make_state()
        images = {"top": make_image(), "overview": make_image()}  # 故意逆序插入
        msgs = build_messages("把红色方块放进托盘", st, images)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertEqual(msgs[0]["content"], system_prompt())

        content = msgs[1]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("SCENE_JSON_BEGIN", content[0]["text"])
        self.assertIn("SCENE_JSON_END", content[0]["text"])
        self.assertIn("把红色方块放进托盘", content[0]["text"])
        self.assertIn("box_red", content[0]["text"])

        urls = [p["image_url"]["url"] for p in content[1:]]
        self.assertEqual(len(urls), 2)
        self.assertTrue(urls[0].startswith("data:image/jpeg;base64,"))
        # 稳定顺序 = sorted(keys) -> overview 先于 top
        expect_overview = "data:image/jpeg;base64," + encode_image(images["overview"])
        expect_top = "data:image/jpeg;base64," + encode_image(images["top"])
        self.assertEqual(urls[0], expect_overview)
        self.assertEqual(urls[1], expect_top)
        self.assertTrue(decode_data_uri(urls[0]).startswith(b"\xff\xd8\xff"))

        # 同一输入两次构造必须完全一致（可复现）
        self.assertEqual(msgs, build_messages("把红色方块放进托盘", st, images))

    def test_build_messages_with_history(self):
        st = make_state()
        history = [
            {"role": "assistant", "content": json.dumps({"skill": "pick", "args": {"target_id": "box_red"}})},
            {"role": "user", "content": "OBSERVATION_JSON " + json.dumps({"ok": True, "skill": "pick"})},
            {"ok": False, "skill": "place", "message": "非法调用"},   # 裸 observation 也要兼容
        ]
        msgs = build_messages("t", st, {}, history=history)
        # 结构 = system + 3 条 history + 当前 user（当前 user 追加在最后）
        self.assertEqual([m["role"] for m in msgs], ["system", "assistant", "user", "user", "user"])
        self.assertIn("OBSERVATION_JSON", msgs[2]["content"])
        self.assertIn("非法调用", msgs[3]["content"])   # 裸 observation dict 被包成 user 文本
        self.assertEqual(msgs[-1]["content"][0]["type"], "text")  # 无图像时只有文本 part
        self.assertIn("SCENE_JSON_BEGIN", msgs[-1]["content"][0]["text"])


# --------------------------------------------------------------------------- #
# 2) schema：TOOLS / parse / validate
# --------------------------------------------------------------------------- #
class TestSchema(unittest.TestCase):

    def test_tools_cover_all_skills(self):
        names = [t["function"]["name"] for t in TOOLS]
        self.assertEqual(sorted(names), sorted(SKILL_NAMES))
        self.assertEqual(len(names), len(set(names)), "工具名不能重复")
        for t in TOOLS:
            self.assertEqual(t["type"], "function")
            self.assertEqual(t["function"]["parameters"]["type"], "object")

    def test_tool_required_params_match_contract(self):
        self.assertEqual(_tool_by_name("pick")["function"]["parameters"]["required"], ["target_id"])
        self.assertEqual(_tool_by_name("place")["function"]["parameters"]["required"], ["container_id"])
        self.assertEqual(_tool_by_name("move_to")["function"]["parameters"]["required"], ["pos"])
        self.assertEqual(_tool_by_name("push")["function"]["parameters"]["required"],
                         ["target_id", "direction", "distance"])
        self.assertEqual(_tool_by_name("ask_user")["function"]["parameters"]["required"], ["question"])
        props = _tool_by_name("set_compliance")["function"]["parameters"]["properties"]
        self.assertEqual(props["profile"]["enum"], list(PROFILE_NAMES))
        self.assertEqual(_tool_by_name("home")["function"]["parameters"].get("required", []), [])
        self.assertEqual(_tool_by_name("home")["function"]["parameters"]["properties"], {})
        # pick 的 compliance 是对象 schema；参数白名单和 TOOLS 必须一致
        for tool in TOOLS:
            name = tool["function"]["name"]
            declared = set(tool["function"]["parameters"]["properties"].keys())
            self.assertEqual(declared, set(ALLOWED_ARGS[name]), f"{name} 的 TOOLS/白名单不一致")

    def test_parse_openai_tool_call_with_string_arguments(self):
        raw = {"name": "pick", "arguments": json.dumps({"target_id": "box_red", "approach_height": 0.12})}
        call = parse_skill_call(raw)
        self.assertEqual(call.skill, "pick")
        self.assertEqual(call.args["target_id"], "box_red")
        self.assertAlmostEqual(call.args["approach_height"], 0.12)
        self.assertIsNone(call.compliance)

    def test_parse_full_tool_call_and_response_shell(self):
        full = {"id": "call_1", "type": "function",
                "function": {"name": "place", "arguments": json.dumps({"container_id": "tray"})}}
        self.assertEqual(parse_skill_call(full).skill, "place")
        shell = {"choices": [{"message": {"tool_calls": [full]}}]}
        self.assertEqual(parse_skill_call(shell["choices"][0]["message"]).skill, "place")
        self.assertEqual(parse_skill_call([full]).skill, "place")

    def test_parse_text_embedded_json(self):
        text = '好的，我来抓取。\n{"name": "pick", "arguments": {"target_id": "box_blue"}}\n'
        self.assertEqual(parse_skill_call(text).args["target_id"], "box_blue")
        fenced = "思考中...\n" + chr(96) * 3 + 'json\n{"skill": "home", "args": {}}\n' + chr(96) * 3
        self.assertEqual(parse_skill_call(fenced).skill, "home")
        call_text = 'pick({"target_id": "box_green", "close_ratio": 0.4})'
        self.assertEqual(parse_skill_call(call_text).args["target_id"], "box_green")
        # 我们自己的 SkillCall JSON
        own = json.dumps({"skill": "set_compliance", "args": {"profile": "z_only"}, "reasoning": "r"})
        call = parse_skill_call(own)
        self.assertEqual(call.skill, "set_compliance")
        self.assertIsNotNone(call.compliance)
        self.assertEqual(call.reasoning, "r")

    def test_parse_empty_arguments_ok(self):
        self.assertEqual(parse_skill_call({"name": "home", "arguments": ""}).args, {})
        self.assertEqual(parse_skill_call({"name": "stop", "arguments": None}).args, {})

    def test_parse_compliance_string_and_object(self):
        call = parse_skill_call({"name": "move_to", "arguments": {"pos": [0.4, 0, 0.5], "compliance": "soft"}})
        self.assertIsInstance(call.compliance, ComplianceParams)
        self.assertTrue(np.all(call.compliance.axes == 1.0))

        cp = compliance_from_profile("z_only")
        call2 = parse_skill_call({"name": "pick", "arguments": {"target_id": "box_red", "compliance": cp.to_json()}})
        self.assertIsInstance(call2.compliance, ComplianceParams)
        self.assertEqual(float(call2.compliance.axes[2]), 1.0)
        self.assertEqual(float(call2.compliance.axes[0]), 0.0)

    def test_parse_set_compliance_builds_params(self):
        call = parse_skill_call({"name": "set_compliance", "arguments": {"profile": "soft", "k_scale": 0.5}})
        self.assertEqual(call.args["profile"], "soft")
        self.assertIsNotNone(call.compliance)
        base = compliance_from_profile("soft")
        self.assertLess(float(call.compliance.k[0]), float(base.k[0]))

    def test_parse_filters_unknown_args_and_records_reasoning(self):
        call = parse_skill_call({"name": "pick", "arguments": {"target_id": "box_red", "speed": 99}})
        self.assertNotIn("speed", call.args)
        self.assertIn("speed", call.reasoning)

    def test_parse_rejects_invalid(self):
        for bad in (
            {"name": "fly", "arguments": {}},                      # 未知技能
            {"arguments": {"target_id": "x"}},                     # 缺技能名
            {"name": "pick", "arguments": "not-json"},             # arguments 非 JSON
            {"name": "pick", "arguments": [1, 2]},                 # arguments 非对象
            "这里没有任何工具调用",                                 # 文本无 JSON
            "",                                                     # 空串
            12345,                                                  # 类型不支持
            {"name": "move_to", "arguments": {"pos": [0, 0, 0], "compliance": "very_soft"}},
            {"name": "pick", "arguments": {"target_id": "box_red", "compliance": {"k": [1]}}},
        ):
            with self.assertRaises(ValueError, msg=f"应拒绝: {bad!r}"):
                parse_skill_call(bad)

    def test_validate_accepts_legal_calls(self):
        st = make_state()
        ok, msg = validate_skill_call(parse_skill_call({"name": "pick", "arguments": {"target_id": "box_red"}}), st)
        self.assertTrue(ok, msg)
        ok, msg = validate_skill_call(parse_skill_call({"name": "place", "arguments": {"container_id": "tray"}}), st)
        self.assertTrue(ok, msg)
        ok, msg = validate_skill_call(parse_skill_call({"name": "set_compliance", "arguments": {"profile": "soft"}}), st)
        self.assertTrue(ok, msg)
        ok, msg = validate_skill_call(parse_skill_call({"name": "stop", "arguments": {}}), st)
        self.assertTrue(ok, msg)
        ok, msg = validate_skill_call(SkillCall("move_to", {"pos": [0.4, 0.0, 0.5]}), st)
        self.assertTrue(ok, msg)
        ok, msg = validate_skill_call(SkillCall("push", {"target_id": "box_blue",
                                                         "direction": [1, 0, 0], "distance": 0.05}), st)
        self.assertTrue(ok, msg)

    def test_validate_rejects_scene_inconsistencies(self):
        st = make_state()
        cases = [
            (SkillCall("pick", {"target_id": "ghost"}), "不存在"),
            (SkillCall("pick", {}), "target_id"),
            (SkillCall("pick", {"target_id": "tray"}), "graspable"),
            (SkillCall("place", {"container_id": "box_red"}), "is_container"),
            (SkillCall("place", {"container_id": "ghost"}), "不存在"),
            (SkillCall("push", {"target_id": "box_red", "direction": [0, 0, 0], "distance": 0.05}), "零向量"),
            (SkillCall("push", {"target_id": "box_red", "direction": [1, 0, 0], "distance": 5.0}), "distance"),
            (SkillCall("move_to", {"pos": [0.4, 0.0]}), "pos"),
            (SkillCall("move_to", {"pos": [0.4, 0.0, 0.5], "quat": [1, 0, 0]}), "quat"),
            (SkillCall("close_gripper", {"ratio": 3.0}), "ratio"),
            (SkillCall("set_compliance", {"profile": "very_soft"}), "profile"),
            (SkillCall("ask_user", {"question": "  "}), "question"),
            (SkillCall("home", {"pos": [1, 2, 3]}), "不接受参数"),
            (SkillCall("fly", {}), "白名单"),
        ]
        for call, needle in cases:
            ok, msg = validate_skill_call(call, st)
            self.assertFalse(ok, f"应拒绝 {call.skill} {call.args}")
            self.assertIn(needle, msg, f"错误信息应包含 {needle!r}，实际 {msg!r}")

    def test_validate_rejects_wrong_types(self):
        st = make_state()
        ok, msg = validate_skill_call(SkillCall("pick", {"target_id": 123}), st)
        self.assertFalse(ok)
        ok, msg = validate_skill_call(SkillCall("pick", {"target_id": "box_red", "approach_height": "high"}), st)
        self.assertFalse(ok)
        self.assertIn("approach_height", msg)


# --------------------------------------------------------------------------- #
# 3) MockVLM / VLMClient 工厂
# --------------------------------------------------------------------------- #
class TestMockVLM(unittest.TestCase):

    def _messages(self, task, state, history=None):
        return build_messages(task, state, {"overview": make_image()}, history=history)

    def test_no_api_key_falls_back_to_mock(self):
        with mock.patch("agent.vlm_client.VLM_API_KEY", ""):
            vlm = make_vlm(prefer_api=True)
            self.assertIsInstance(vlm, MockVLM)
        self.assertIsInstance(make_vlm(prefer_api=False), MockVLM)
        with mock.patch("agent.vlm_client.VLM_API_KEY", "fake-key"):
            self.assertIsInstance(make_vlm(prefer_api=True), VLMClient)
            self.assertIsInstance(make_vlm(prefer_api=False), MockVLM)

    def test_client_available_and_offline_error(self):
        self.assertTrue(VLMClient(api_key="k").available)
        self.assertFalse(VLMClient(api_key="").available)
        with self.assertRaises(VLMError):
            VLMClient(api_key="").decide([{"role": "user", "content": "hi"}])

    def test_mock_picks_color_target_then_places_then_stops(self):
        vlm = MockVLM()
        st = make_state()
        call1 = vlm.decide(self._messages("把红色的方块放进托盘", st))
        self.assertEqual(call1.skill, "pick")
        self.assertEqual(call1.args["target_id"], "box_red")
        self.assertTrue(validate_skill_call(call1, st)[0])

        obs1 = SkillResult(ok=True, skill="pick", message="picked").to_observation()
        hist = [{"role": "assistant", "content": json.dumps(call1.to_json())},
                {"role": "user", "content": "OBSERVATION_JSON " + json.dumps(obs1)}]
        call2 = vlm.decide(self._messages("把红色的方块放进托盘", st, hist))
        self.assertEqual(call2.skill, "place")
        self.assertEqual(call2.args["container_id"], "tray")
        self.assertTrue(validate_skill_call(call2, st)[0])

        obs2 = SkillResult(ok=True, skill="place", message="placed").to_observation()
        hist += [{"role": "assistant", "content": json.dumps(call2.to_json())},
                 {"role": "user", "content": "OBSERVATION_JSON " + json.dumps(obs2)}]
        call3 = vlm.decide(self._messages("把红色的方块放进托盘", st, hist))
        self.assertEqual(call3.skill, "stop")
        # 决定性：同样输入两次决策一致
        self.assertEqual(call3.to_json(), vlm.decide(self._messages("把红色的方块放进托盘", st, hist)).to_json())

    def test_mock_handles_english_and_yellow(self):
        vlm = MockVLM()
        st = make_state()
        self.assertEqual(vlm.decide(self._messages("put the yellow can into the tray", st)).args["target_id"],
                         "can_yellow")
        self.assertEqual(vlm.decide(self._messages("抓绿色方块", st)).args["target_id"], "box_green")

    def test_mock_no_color_falls_back_to_first_graspable(self):
        vlm = MockVLM()
        st = make_state()
        # box_red 的 id 字典序最小 -> 退化选择它
        self.assertEqual(vlm.decide(self._messages("把东西收好", st)).args["target_id"], "box_red")

    def test_mock_soft_keyword_adds_compliance(self):
        vlm = MockVLM()
        st = make_state()
        call = vlm.decide(self._messages("轻轻地把蓝色方块放进托盘，别压坏", st))
        self.assertEqual(call.skill, "pick")
        self.assertEqual(call.args["target_id"], "box_blue")
        self.assertIsNotNone(call.compliance, "「轻」应触发 soft 导纳")
        self.assertTrue(np.all(call.compliance.axes == 1.0))

    def test_mock_contact_force_triggers_set_compliance_first(self):
        vlm = MockVLM()
        st = make_state(force=(6.0, 3.0, 0.0))   # |f| = 6.7 N
        call = vlm.decide(self._messages("抓红色方块", st))
        self.assertEqual(call.skill, "set_compliance")
        self.assertEqual(call.args["profile"], "soft")
        self.assertTrue(validate_skill_call(call, st)[0])

    def test_mock_retries_with_soft_after_pick_failure(self):
        vlm = MockVLM()
        st = make_state()
        fail = SkillResult(ok=False, skill="pick", message="ik_fail", error_code="ik_fail").to_observation()
        hist = [{"role": "user", "content": "OBSERVATION_JSON " + json.dumps(fail)}]
        call = vlm.decide(self._messages("抓红色方块", st, hist))
        self.assertEqual(call.skill, "set_compliance")

    def test_mock_degrades_gracefully(self):
        vlm = MockVLM()
        # 没有 SCENE JSON -> stop
        self.assertEqual(vlm.decide([{"role": "user", "content": "hello"}]).skill, "stop")
        # 没有可抓取物体 -> ask_user
        st = make_state(objects=[_obj("tray", "tray", "gray", [0.4, 0, 0.4], [0.1, 0.1, 0.01],
                                     graspable=False, is_container=True)])
        call = vlm.decide(self._messages("抓红色方块", st))
        self.assertEqual(call.skill, "ask_user")
        self.assertTrue(validate_skill_call(call, st)[0])


# --------------------------------------------------------------------------- #
# 3b) 意图路由（问题 1/2/3 回归）：回家 / 停止 / 松开夹爪 / 只夹不放
# --------------------------------------------------------------------------- #
class TestIntentRouting(unittest.TestCase):

    def _messages(self, task, state, history=None):
        return build_messages(task, state, {"overview": make_image()}, history=history)

    def test_intent_predicates(self):
        self.assertTrue(_wants_home("回到默认位置"))
        self.assertTrue(_wants_home("请 reset pose"))
        self.assertTrue(_wants_open("松开夹爪"))
        self.assertTrue(_wants_open("open gripper"))
        self.assertTrue(_wants_stop("停下"))
        self.assertTrue(_wants_stop("Stop now"))
        self.assertTrue(_pick_only("夹住蓝色方块"))
        self.assertTrue(_pick_only("夹住黄色圆柱体"))
        self.assertTrue(_pick_only("grasp the red box"))
        # 「放开/松开」含「放」，但必须先归为松开意图，不能算只夹
        self.assertFalse(_pick_only("放开夹爪"))
        self.assertFalse(_pick_only("松开"))
        # 含放置词 -> 不是只夹
        self.assertFalse(_pick_only("抓起红色方块放进托盘"))
        self.assertFalse(_pick_only("把红色方块放进托盘"))

    def test_mock_home_then_stop(self):
        vlm = MockVLM()
        st = make_state()
        call1 = vlm.decide(self._messages("回到默认位置", st))
        self.assertEqual(call1.skill, "home")
        self.assertEqual(call1.args, {})
        self.assertTrue(validate_skill_call(call1, st)[0])
        hist = history_of(SkillResult(ok=True, skill="home", message="homed"))
        call2 = vlm.decide(self._messages("回到默认位置", st, hist))
        self.assertEqual(call2.skill, "stop", "home 成功后必须停止，不得继续 pick/place")

    def test_mock_pick_only_never_places(self):
        vlm = MockVLM()
        st = make_state()
        for task, target in (("夹住蓝色方块", "box_blue"), ("夹住黄色圆柱体", "can_yellow")):
            with self.subTest(task=task):
                call1 = vlm.decide(self._messages(task, st))
                self.assertEqual(call1.skill, "pick")
                self.assertEqual(call1.args["target_id"], target)
                self.assertTrue(validate_skill_call(call1, st)[0])
                hist = history_of(SkillResult(ok=True, skill="pick", message="picked"))
                call2 = vlm.decide(self._messages(task, st, hist))
                self.assertEqual(call2.skill, "stop", f"{task} 抓到后不得追加 place")
                self.assertIn("只要求夹住", call2.reasoning)

    def test_mock_open_gripper_then_stop(self):
        vlm = MockVLM()
        st = make_state()
        call1 = vlm.decide(self._messages("松开夹爪", st))
        self.assertEqual(call1.skill, "open_gripper")
        self.assertEqual(call1.args, {})
        self.assertTrue(validate_skill_call(call1, st)[0])
        hist = history_of(SkillResult(ok=True, skill="open_gripper", message="opened"))
        self.assertEqual(vlm.decide(self._messages("松开夹爪", st, hist)).skill, "stop")

    def test_mock_stop_intent(self):
        vlm = MockVLM()
        st = make_state()
        self.assertEqual(vlm.decide(self._messages("停下", st)).skill, "stop")
        self.assertEqual(vlm.decide(self._messages("cancel", st)).skill, "stop")

    def test_loop_intent_routing_end_to_end(self):
        cases = {
            "回到默认位置": ["home", "stop"],
            "夹住蓝色方块": ["pick", "stop"],
            "松开夹爪": ["open_gripper", "stop"],
            "停下": ["stop"],
        }
        for task, expected in cases.items():
            with self.subTest(task=task):
                env = StubEnv()
                executor = RecordingExecutor()
                loop = AgentLoop(env, executor, MockVLM(), max_steps=8, log=None)
                results = loop.run(task)
                skills = [c.skill for c in executor.calls]
                self.assertEqual(skills, expected, task)
                self.assertNotIn("place", skills, f"{task} 不应出现 place")
                self.assertTrue(all(r.ok for r in results), [r.message for r in results])

    def test_loop_terminates_on_ask_user(self):
        env = StubEnv()
        executor = RecordingExecutor()
        vlm = ScriptedVLM([
            SkillCall("ask_user", {"question": "请确认物体位置"}),
            SkillCall("pick", {"target_id": "box_red"}),   # 不应被执行
        ])
        logs: list[str] = []
        loop = AgentLoop(env, executor, vlm, max_steps=8, log=logs.append)
        results = loop.run("抓一下但需要确认")
        self.assertEqual(len(results), 1, "ask_user 后必须立即终止，不得空转到 max_steps")
        self.assertEqual(results[0].skill, "ask_user")
        self.assertEqual([c.skill for c in executor.calls], ["ask_user"])
        self.assertTrue(any("需要人工确认，任务中止" in m for m in logs), logs)

    def test_loop_single_ask_user_when_no_graspable(self):
        # 问题 3：场景无可抓物时，MockVLM 只问一次就结束，不再连续 ask_user 8 次
        objects = [_obj("tray", "tray", "gray", [0.42, 0.02, 0.405], [0.12, 0.12, 0.01],
                        graspable=False, is_container=True)]
        env = StubEnv(state=make_state(objects=objects))
        loop = AgentLoop(env, RecordingExecutor(), MockVLM(), max_steps=8, log=None)
        results = loop.run("抓红色方块")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].skill, "ask_user")


# --------------------------------------------------------------------------- #
# 3c) 不可重试错误 / 重复失败立即终止（task-11：消除 home_blocked 空转）
# --------------------------------------------------------------------------- #
class TestNonRetryableTermination(unittest.TestCase):

    def _messages(self, task, state, history=None):
        return build_messages(task, state, {"overview": make_image()}, history=history)

    def test_mock_stops_on_all_non_retryable_codes(self):
        vlm = MockVLM()
        st = make_state()
        for code in ("home_blocked", "stuck", "object_disturbed", "scene_reset"):
            with self.subTest(code=code):
                hist = history_of(SkillResult(ok=False, skill="home", message="致命失败",
                                              error_code=code))
                call = vlm.decide(self._messages("回到默认位置", st, hist))
                self.assertEqual(call.skill, "stop", f"{code} 应直接 stop")
                self.assertIn(code, call.reasoning)

    def test_mock_non_fatal_error_does_not_stop(self):
        # ik_fail 不在不可重试集合 -> 仍走原有「先降刚度再重试」逻辑
        vlm = MockVLM()
        st = make_state()
        hist = history_of(SkillResult(ok=False, skill="pick", message="ik fail",
                                      error_code="ik_fail"))
        self.assertEqual(vlm.decide(self._messages("抓红色方块", st, hist)).skill,
                         "set_compliance")

    def test_loop_home_blocked_ends_within_3_steps(self):
        # 夹持中「回到默认位置」：home -> home_blocked -> stop，<= 3 步
        env = StubEnv()
        executor = StubSkillExecutor({"home": (False, "home_blocked", "夹持中禁止回零")})
        logs: list[str] = []
        loop = AgentLoop(env, executor, MockVLM(), max_steps=12, log=logs.append)
        results = loop.run("回到默认位置")
        skills = [c.skill for c in executor.calls]
        self.assertEqual(skills, ["home", "stop"], f"应为 home -> stop，实际 {skills}")
        self.assertLessEqual(len(results), 3, f"必须在 <=3 步内结束，实际 {len(results)}")
        self.assertNotEqual(len(results), 12, "不得空转到 max_steps=12")
        # 终止原因写在 stop 调用的 reasoning 里（MockVLM 判定不可重试）
        self.assertIn("不可重试", loop.history[-2]["content"])
        self.assertEqual([r.skill for r in results], ["home", "stop"])
        self.assertEqual(results[0].error_code, "home_blocked")

    def test_loop_backstop_on_stubborn_vlm(self):
        # 即使 VLM 死活重复 home，AgentLoop 也在同一失败重复 2 次时终止
        class AlwaysHome:
            def decide(self, messages):
                return SkillCall("home", {})

        env = StubEnv()
        executor = StubSkillExecutor({"home": (False, "home_blocked", "夹持中禁止回零")})
        logs: list[str] = []
        loop = AgentLoop(env, executor, AlwaysHome(), max_steps=12, log=logs.append)
        results = loop.run("回到默认位置")
        self.assertEqual(len(results), 2, "同一 skill+error_code 连续 2 次应终止")
        self.assertEqual([r.skill for r in results], ["home", "home"])
        self.assertTrue(any("任务中止" in m and "home_blocked" in m for m in logs), logs)

    def test_loop_fatal_code_across_different_skills(self):
        # 不同 skill 但同一不可重试码第 2 次出现 -> 终止
        class HomeThenPick:
            def __init__(self):
                self.n = 0

            def decide(self, messages):
                self.n += 1
                if self.n == 1:
                    return SkillCall("home", {})
                return SkillCall("pick", {"target_id": "box_red"})

        env = StubEnv()
        executor = StubSkillExecutor({"home": (False, "stuck", "卡住"),
                                      "pick": (False, "stuck", "卡住")})
        logs: list[str] = []
        loop = AgentLoop(env, executor, HomeThenPick(), max_steps=12, log=logs.append)
        results = loop.run("任务")
        self.assertEqual(len(results), 2, "同一不可重试码第 2 次出现应终止")
        self.assertTrue(any("不可重试错误码 stuck" in m for m in logs), logs)

    def test_loop_repeated_nonfatal_error_terminates(self):
        class AlwaysPick:
            def decide(self, messages):
                return SkillCall("pick", {"target_id": "box_red"})

        env = StubEnv()
        executor = StubSkillExecutor({"pick": (False, "ik_fail", "ik fail")})
        logs: list[str] = []
        loop = AgentLoop(env, executor, AlwaysPick(), max_steps=12, log=logs.append)
        results = loop.run("抓红色方块")
        self.assertEqual(len(results), 2, "同一 skill+error_code 连续 2 次应终止")
        self.assertTrue(any("同一调用 pick() 连续 2 次" in m for m in logs), logs)

    def test_loop_success_repeats_do_not_terminate(self):
        # 成功步（error_code 为空）重复不算失败，不得误杀 test_loop_respects_max_steps 语义
        class AlwaysPick:
            def decide(self, messages):
                return SkillCall("pick", {"target_id": "box_red"})

        loop = AgentLoop(StubEnv(), RecordingExecutor(), AlwaysPick(), max_steps=3, log=None)
        self.assertEqual(len(loop.run("抓")), 3)


# --------------------------------------------------------------------------- #
# 3d) place 支持桌面 surface + 指代纠正 + 无颜色复用历史目标（task-23）
# --------------------------------------------------------------------------- #
class TestPlaceSurfaceAndCorrection(unittest.TestCase):

    def _messages(self, task, state, history=None):
        return build_messages(task, state, {"overview": make_image()}, history=history)

    @staticmethod
    def _picked_step(target_id="box_blue"):
        return step_messages(SkillCall("pick", {"target_id": target_id}),
                             SkillResult(ok=True, skill="pick", message="picked"))

    @staticmethod
    def _placed_step(**args):
        return step_messages(SkillCall("place", dict(args)),
                             SkillResult(ok=True, skill="place", message="placed"))

    def test_tools_place_has_surface_and_xy(self):
        params = _tool_by_name("place")["function"]["parameters"]
        self.assertIn("surface", params["properties"])
        self.assertIn("on_top_of", params["properties"])
        self.assertIn("xy", params["properties"])
        self.assertEqual(params["properties"]["surface"]["oneOf"][0]["enum"], ["table", "tray"])
        self.assertEqual(set(params["properties"]), set(ALLOWED_ARGS["place"]))

    def test_surface_and_correction_predicates(self):
        self.assertEqual(_placement_surface("放到桌子上"), "table")
        self.assertEqual(_placement_surface("把蓝色方块放回到桌子上"), "table")
        self.assertEqual(_placement_surface("放到盘子里"), "tray")
        # 被否定的表面词不算目标
        self.assertEqual(_placement_surface("放到桌子上不是盘子里"), "table")
        self.assertEqual(_placement_surface("不是桌子上，放到盘子里"), "tray")
        self.assertIsNone(_placement_surface("不要放桌子上"))
        self.assertTrue(_wants_correction("放到桌子上不是盘子里"))
        self.assertTrue(_wants_correction("放错了，重新放"))
        self.assertFalse(_wants_correction("把蓝色方块放回到桌子上"))

    def test_place_table_surface_args(self):
        # 自测 1)「放到桌子上」-> place args 含 surface='table' 且不含 container_id
        vlm = MockVLM()
        st = make_state()
        call = vlm.decide(self._messages("放到桌子上", st, self._picked_step()))
        self.assertEqual(call.skill, "place")
        self.assertEqual(call.args.get("surface"), "table")
        self.assertNotIn("container_id", call.args)
        self.assertTrue(validate_skill_call(call, st)[0])
        self.assertIn("surface", call.reasoning)

    def test_blue_box_to_table_sequence(self):
        # 自测 2)「把蓝色方块放回到桌子上」-> pick(box_blue) + place(surface='table')
        vlm = MockVLM()
        st = make_state()
        task = "把蓝色方块放回到桌子上"
        call1 = vlm.decide(self._messages(task, st))
        self.assertEqual(call1.skill, "pick")
        self.assertEqual(call1.args["target_id"], "box_blue")
        hist = step_messages(call1, SkillResult(ok=True, skill="pick", message="picked"))
        call2 = vlm.decide(self._messages(task, st, hist))
        self.assertEqual(call2.skill, "place")
        self.assertEqual(call2.args.get("surface"), "table")
        self.assertNotIn("container_id", call2.args)
        hist += step_messages(call2, SkillResult(ok=True, skill="place", message="placed"))
        self.assertEqual(vlm.decide(self._messages(task, st, hist)).skill, "stop")

    def test_reference_correction_repicks_history_target(self):
        # 自测 3)「放到桌子上不是盘子里」+ 历史 place(tray)
        #         -> pick(历史目标 box_blue) -> place(surface='table') -> stop，且不得出现 box_red
        vlm = MockVLM()
        st = make_state()
        task = "放到桌子上不是盘子里"
        hist = self._picked_step("box_blue") + self._placed_step(container_id="tray")

        call1 = vlm.decide(self._messages(task, st, hist))
        self.assertEqual(call1.skill, "pick", f"应先重新抓取刚放下的物体，实际 {call1.skill}")
        self.assertEqual(call1.args["target_id"], "box_blue")
        hist += step_messages(call1, SkillResult(ok=True, skill="pick", message="picked"))

        call2 = vlm.decide(self._messages(task, st, hist))
        self.assertEqual(call2.skill, "place")
        self.assertEqual(call2.args.get("surface"), "table")
        self.assertNotIn("container_id", call2.args)
        self.assertTrue(validate_skill_call(call2, st)[0])
        hist += step_messages(call2, SkillResult(ok=True, skill="place", message="placed"))

        call3 = vlm.decide(self._messages(task, st, hist))
        self.assertEqual(call3.skill, "stop", "纠正完成后应 stop，不得再次 pick/place")
        for call in (call1, call2, call3):
            self.assertNotEqual(call.args.get("target_id"), "box_red")
        self.assertEqual([c.skill for c in (call1, call2, call3)], ["pick", "place", "stop"])

    def test_validate_place_exactly_one_target(self):
        # 自测 4) 四选一（container_id / surface / on_top_of / anchor_id+relation）
        st = make_state()
        for both in ({"container_id": "tray", "surface": "table"},
                     {"container_id": "tray", "on_top_of": "box_red"},
                     {"surface": "table", "on_top_of": "box_red"},
                     {"on_top_of": "box_red", "anchor_id": "box_blue", "relation": "right"}):
            ok, msg = validate_skill_call(SkillCall("place", dict(both)), st)
            self.assertFalse(ok, f"{both} 应被拒绝")
            self.assertIn("四选一", msg)
        ok, msg = validate_skill_call(SkillCall("place", {}), st)
        self.assertFalse(ok)
        self.assertIn("四选一", msg)
        # 合法：surface / surface+xy / surface='tray' / 容器 / on_top_of(+xy) / 横向方位(+distance)
        for args in ({"surface": "table"}, {"surface": "table", "xy": [0.4, 0.1]},
                     {"surface": "tray"}, {"container_id": "tray"},
                     # §5.12-2：container_id + xy 且落点在容器内 -> 纠正后接受（不再判矛盾）
                     {"container_id": "tray", "xy": [0.4, 0.1]},
                     {"on_top_of": "box_red"}, {"on_top_of": "box_blue", "xy": [0.35, 0.1]},
                     {"anchor_id": "box_red", "relation": "right"},
                     {"anchor_id": "box_blue", "relation": "beside", "distance": 0.12}):
            ok, msg = validate_skill_call(SkillCall("place", dict(args)), st)
            self.assertTrue(ok, f"{args} 应通过：{msg}")
        # 非法：未知 surface / xy 形状错 / xy 与 container 同用 / 非容器 / 不存在 / 锚点不可抓
        for args, needle in (
            ({"surface": "floor"}, "surface"),
            ({"surface": "table", "xy": [0.4]}, "xy"),
            ({"surface": "table", "xy": ["a", 0.1]}, "xy"),
            # §5.12-2：xy 落点在容器**外**仍是真矛盾 -> 拒绝
            ({"container_id": "tray", "xy": [0.05, 0.9]}, "不在容器"),
            ({"container_id": "box_red"}, "is_container"),
            ({"container_id": "ghost"}, "不存在"),
            ({"on_top_of": "ghost"}, "不存在"),
            ({"on_top_of": "tray"}, "不可抓"),
            ({"on_top_of": "tray", "xy": [0.1, 0.1]}, "不可抓"),
            ({"anchor_id": "box_red"}, "成对"),                    # 只给锚点不给方向
            ({"relation": "right"}, "成对"),                       # 只给方向不给锚点
            ({"anchor_id": "box_red", "relation": "diagonal"}, "relation"),
            ({"anchor_id": "ghost", "relation": "right"}, "不存在"),
            ({"anchor_id": "tray", "relation": "right"}, "不可抓"),
            ({"surface": "table", "distance": 0.1}, "distance"),   # distance 只在横向方位用
            ({"anchor_id": "box_red", "relation": "right", "distance": 0.0}, "distance"),
        ):
            ok, msg = validate_skill_call(SkillCall("place", dict(args)), st)
            self.assertFalse(ok, f"{args} 应被拒绝")
            self.assertIn(needle, msg, f"{args} 拒绝理由应含 {needle!r}，实际 {msg!r}")

    def test_no_color_reuses_last_pick_target(self):
        # 要求 B：无颜色/类别时优先复用历史最近一次成功 pick 的 target_id，历史为空才退化
        vlm = MockVLM()
        objs = [o.to_json() for o in make_state().objects]
        self.assertEqual(
            vlm._select_target("把它放到桌子上", objs, fallback_id="box_blue")["id"], "box_blue")
        # 任务里给了颜色/类别 -> 仍以文本为准，历史目标不覆盖
        self.assertEqual(
            vlm._select_target("把红色方块放好", objs, fallback_id="box_blue")["id"], "box_red")
        # 历史为空（fallback_id=None）-> 退化到第一个 graspable
        self.assertEqual(vlm._select_target("把它放到桌子上", objs)["id"], "box_red")
        # decide 层面：历史里已成功抓取 -> 直接进入放置阶段（用本次指定的 surface）
        st = make_state()
        call = vlm.decide(self._messages("把它放到桌子上", st, self._picked_step("box_blue")))
        self.assertEqual(call.skill, "place")
        self.assertEqual(call.args.get("surface"), "table")
        self.assertNotIn("container_id", call.args)
        self.assertTrue(validate_skill_call(call, st)[0])

    def test_loop_blue_box_to_table_end_to_end(self):
        env = StubEnv()
        executor = RecordingExecutor()
        loop = AgentLoop(env, executor, MockVLM(), max_steps=8, log=None)
        results = loop.run("把蓝色方块放回到桌子上")
        self.assertEqual([c.skill for c in executor.calls], ["pick", "place", "stop"])
        self.assertEqual(executor.calls[0].args["target_id"], "box_blue")
        self.assertEqual(executor.calls[1].args.get("surface"), "table")
        self.assertNotIn("container_id", executor.calls[1].args)
        self.assertTrue(all(r.ok for r in results), [r.message for r in results])


# --------------------------------------------------------------------------- #
# 3e) AgentLoop 跨命令历史 seed / 回写 / 裁剪（task-25）
# --------------------------------------------------------------------------- #
class TestCrossCommandHistory(unittest.TestCase):

    @staticmethod
    def _messages(task, state, history=None):
        return build_messages(task, state, {"overview": make_image()}, history=history)

    @staticmethod
    def _pick_place_seed(target="box_blue", container="tray"):
        """一步成功 pick + 一步成功 place 的历史（assistant 调用 + user 观察）。"""
        return (step_messages(SkillCall("pick", {"target_id": target}),
                              SkillResult(ok=True, skill="pick", message="picked"))
                + step_messages(SkillCall("place", {"container_id": container}),
                                SkillResult(ok=True, skill="place", message="placed")))

    def test_injected_history_enables_correction(self):
        # 注入历史后 MockVLM 能据此做纠正决策：pick(历史目标) -> place(surface=table) -> stop
        seed = self._pick_place_seed()
        env, executor = StubEnv(), RecordingExecutor()
        loop = AgentLoop(env, executor, MockVLM(), max_steps=6, log=None, history=seed)
        results = loop.run("放到桌子上不是盘子里")
        self.assertEqual([c.skill for c in executor.calls], ["pick", "place", "stop"])
        self.assertEqual(executor.calls[0].args["target_id"], "box_blue")
        self.assertEqual(executor.calls[1].args.get("surface"), "table")
        self.assertNotIn("container_id", executor.calls[1].args)
        self.assertTrue(all(r.ok for r in results))
        # 结束后可读回：注入历史在前，本次新增在后
        self.assertEqual(len(loop.history), len(seed) + 2 * len(results))
        self.assertEqual(loop.history[:len(seed)], seed)

    def test_run_history_overrides_constructor(self):
        ctor = step_messages(SkillCall("pick", {"target_id": "box_red"}),
                             SkillResult(ok=True, skill="pick", message="picked"))
        run_seed = step_messages(SkillCall("pick", {"target_id": "box_blue"}),
                                 SkillResult(ok=True, skill="pick", message="picked"))
        loop = AgentLoop(StubEnv(), RecordingExecutor(), MockVLM(), max_steps=4, log=None,
                         history=ctor)
        loop.run("把它放到桌子上", history=run_seed)
        self.assertIn("box_blue", loop.history[0]["content"])
        self.assertNotIn("box_red", loop.history[0]["content"])

    def test_constructor_history_used_when_run_arg_none(self):
        seed = self._pick_place_seed()
        loop = AgentLoop(StubEnv(), RecordingExecutor(), MockVLM(), max_steps=4, log=None,
                         history=seed)
        self.assertEqual([r.skill for r in loop.run("放到桌子上不是盘子里")],
                         ["pick", "place", "stop"])
        # 第二次不传参仍回到构造种子，而不是接着上一次的历史
        self.assertEqual([r.skill for r in loop.run("放到桌子上不是盘子里")],
                         ["pick", "place", "stop"])
        self.assertEqual(loop.history[:len(seed)], seed)

    def test_no_injection_clears_history(self):
        env, executor = StubEnv(), RecordingExecutor()
        loop = AgentLoop(env, executor, MockVLM(), max_steps=8, log=None)
        loop.run("把红色的方块放进托盘")
        n1 = len(loop.history)
        self.assertEqual([c.skill for c in executor.calls], ["pick", "place", "stop"])
        executor.calls.clear()
        loop.run("把红色的方块放进托盘")
        self.assertEqual([c.skill for c in executor.calls], ["pick", "place", "stop"])
        self.assertEqual(len(loop.history), n1, "不注入历史时每次 run 应清空（与原来一致）")

    def test_history_trimmed_to_limit_in_whole_pairs(self):
        seed = []
        for i in range(40):          # 80 条消息 = 40 组
            seed += step_messages(SkillCall("pick", {"target_id": "box_blue"}),
                                  SkillResult(ok=True, skill="pick", message=f"p{i}"))
        loop = AgentLoop(StubEnv(), RecordingExecutor(), MockVLM(), max_steps=1, log=None,
                         history=seed, max_history_messages=40)
        self.assertEqual(loop.max_history_messages, 40)   # 实例属性，便于调用方调整
        loop.run("放到桌子上")
        hist = loop.history
        self.assertLessEqual(len(hist), 40, f"历史应裁到上限，实际 {len(hist)}")
        self.assertEqual(len(hist) % 2, 0)
        for i in range(0, len(hist), 2):           # 成对完整：assistant 紧跟 user
            self.assertEqual(hist[i]["role"], "assistant")
            self.assertEqual(hist[i + 1]["role"], "user")
        self.assertIn("p0", hist[1]["content"])       # 最早一组保留
        # 最近一组（本次新增的调用）保留，且带纪元标记（标记与后续观测不拆散）
        self.assertIn("TASK_EPOCH_JSON", hist[-2]["content"])

    def test_trim_never_splits_a_pair(self):
        # 单组超长时也整组保留，不把 assistant/user 拆开
        long_pair = [{"role": "assistant", "content": "a" * 5},
                     {"role": "user", "content": "b" * 5}]
        seed = [long_pair[0], long_pair[1]] + self._pick_place_seed()
        loop = AgentLoop(StubEnv(), RecordingExecutor(), MockVLM(), max_steps=2, log=None,
                         history=seed, max_history_messages=3)
        loop.run("把红色的方块放进托盘")
        roles = [m["role"] for m in loop.history]
        self.assertEqual(len(roles) % 2, 0)
        for i in range(0, len(roles), 2):
            self.assertEqual(roles[i], "assistant")
            self.assertEqual(roles[i + 1], "user")

    def test_correction_after_trailing_stop_repicks(self):
        # task-26 FINDING：上一条命令末尾是 stop（已 place 到桌面），纠正时必须先重新 pick，
        # 不能因为 last_ok_skill=="stop" 就直接 place(tray) 得到 nothing_held。
        seed = (step_messages(SkillCall("pick", {"target_id": "box_blue"}),
                              SkillResult(ok=True, skill="pick", message="picked"))
                + step_messages(SkillCall("place", {"surface": "table"}),
                                SkillResult(ok=True, skill="place", message="placed"))
                + step_messages(SkillCall("stop", {}),
                                SkillResult(ok=True, skill="stop", message="任务结束")))
        env, executor = StubEnv(), RecordingExecutor()
        loop = AgentLoop(env, executor, MockVLM(), max_steps=6, log=None, history=seed)
        results = loop.run("放到盘子里不是桌子上")
        self.assertEqual([c.skill for c in executor.calls], ["pick", "place", "stop"])
        self.assertEqual(executor.calls[0].args["target_id"], "box_blue")
        self.assertEqual(executor.calls[1].args.get("container_id"), "tray")
        self.assertTrue(all(r.ok for r in results))

    def test_correction_while_holding_places_directly(self):
        # 反向：历史末尾虽然也是 stop，但 stop 之前已经重新 pick（仍夹着物体）-> 直接 place
        seed = (step_messages(SkillCall("pick", {"target_id": "box_blue"}),
                              SkillResult(ok=True, skill="pick", message="picked"))
                + step_messages(SkillCall("place", {"surface": "table"}),
                                SkillResult(ok=True, skill="place", message="placed"))
                + step_messages(SkillCall("pick", {"target_id": "box_blue"}),
                                SkillResult(ok=True, skill="pick", message="re-picked"))
                + step_messages(SkillCall("stop", {}),
                                SkillResult(ok=True, skill="stop", message="任务结束")))
        env, executor = StubEnv(), RecordingExecutor()
        loop = AgentLoop(env, executor, MockVLM(), max_steps=6, log=None, history=seed)
        loop.run("放到盘子里不是桌子上")
        self.assertEqual([c.skill for c in executor.calls], ["place", "stop"])
        self.assertEqual(executor.calls[0].args.get("container_id"), "tray")

    def test_correction_repick_honors_retry_policy(self):
        # 纠正路径的重新抓取同样遵守「失败先降刚度 / 超限转 ask_user」，不再无限硬抓
        seed = (step_messages(SkillCall("pick", {"target_id": "box_blue"}),
                              SkillResult(ok=True, skill="pick", message="picked"))
                + step_messages(SkillCall("place", {"surface": "table"}),
                                SkillResult(ok=True, skill="place", message="placed"))
                + step_messages(SkillCall("stop", {}),
                                SkillResult(ok=True, skill="stop", message="done")))
        task = "放到盘子里不是桌子上"
        vlm, st = MockVLM(), make_state()
        fail = step_messages(SkillCall("pick", {"target_id": "box_blue"}),
                             SkillResult(ok=False, skill="pick", message="grasp_failed",
                                         error_code="grasp_failed"))
        call = vlm.decide(self._messages(task, st, seed + fail))
        self.assertEqual(call.skill, "set_compliance", "纠正重抓失败 1 次应先降刚度")
        comp = step_messages(SkillCall("set_compliance", {"profile": "soft"}),
                             SkillResult(ok=True, skill="set_compliance", message="soft"))
        call2 = vlm.decide(self._messages(task, st, seed + fail + comp + fail + fail))
        self.assertEqual(call2.skill, "ask_user", "连续抓取失败应转人工确认")


# --------------------------------------------------------------------------- #
# 3f) 任务纪元 / 跨命令泄漏回归（task-29 D1/D2/D3）
# --------------------------------------------------------------------------- #
class TestCrossCommandEpoch(unittest.TestCase):
    """验证者给出的三个可复现缺陷（D1 无上下文纠正、D2 失败计数泄漏、D3 placed 短路）。"""

    def _messages(self, task, state, history=None):
        return build_messages(task, state, {"overview": make_image()}, history=history)

    @staticmethod
    def _step(spec, ok=True, error_code=None):
        call = SkillCall(spec[0], dict(spec[1]))
        return step_messages(call, SkillResult(ok=ok, skill=call.skill, message="ok",
                                               error_code=error_code))

    class _CylinderFails(RecordingExecutor):
        """只让 can_yellow 的 pick 失败，其余动作成功。"""

        def __call__(self, call):
            self.calls.append(call)
            if call.skill == "pick" and call.args.get("target_id") == "can_yellow":
                return SkillResult(ok=False, skill="pick", message="grasp_failed",
                                   error_code="grasp_failed")
            return SkillResult(ok=True, skill=call.skill, message="ok")

    def test_epoch_marker_embedded_once_and_history_stays_paired(self):
        loop = AgentLoop(StubEnv(), RecordingExecutor(), MockVLM(), max_steps=8, log=None)
        loop.run("把红色的方块放进托盘")
        roles = [m["role"] for m in loop.history]
        self.assertEqual(len(roles) % 2, 0, "纪元标记内联后历史仍应严格成对")
        self.assertEqual(roles[0], "assistant")
        markers = [m for m in loop.history if "TASK_EPOCH_JSON" in str(m.get("content"))]
        self.assertEqual(len(markers), 1, "每个 run 只写一次纪元标记")
        self.assertEqual(markers[0]["role"], "assistant")

    def test_cross_command_place_does_not_short_circuit(self):
        # D3：上一条命令 place 过之后，新命令不得被历史 placed 短路（用户命令被静默忽略）
        ex = RecordingExecutor()
        loop = AgentLoop(StubEnv(), ex, MockVLM(), max_steps=8, log=None)
        loop.run("把蓝色方块放回到桌子上")
        ex.calls.clear()
        loop.run("把红色方块放进托盘", history=loop.history)   # 模拟 UI 透传上一条命令历史
        self.assertEqual([c.skill for c in ex.calls], ["pick", "place", "stop"])
        self.assertEqual(ex.calls[0].args["target_id"], "box_red")
        self.assertEqual(ex.calls[1].args.get("container_id"), "tray")

    def test_cross_command_pick_failures_do_not_leak(self):
        # D2：上一条命令抓 can_yellow 连续失败，不得让 box_blue 一上来就 ask_user
        ex = self._CylinderFails()
        loop = AgentLoop(StubEnv(), ex, MockVLM(), max_steps=8, log=None)
        r1 = loop.run("夹住黄色圆柱体")
        self.assertEqual(r1[-1].skill, "ask_user", "第一条命令应因 can_yellow 失败转人工")
        ex.calls.clear()
        r2 = loop.run("夹住蓝色方块", history=loop.history)
        self.assertEqual([r.skill for r in r2], ["pick", "stop"])
        self.assertEqual(ex.calls[0].args["target_id"], "box_blue")

    def test_clear_history_correction_asks_user(self):
        # D1：历史清空（或会话初始）时，纠正/否定指令不得猜物体
        vlm, st = MockVLM(), make_state()
        call = vlm.decide(self._messages("放到桌子上不是盘子里", st, []))
        self.assertEqual(call.skill, "ask_user")
        self.assertIn("你指的是哪个物体", call.args.get("question", ""))
        self.assertIn("box_red", call.args.get("question", ""))   # 给出候选清单
        # 有上文可参考时仍走跨命令指代（不询问）
        hist = (self._step(("pick", {"target_id": "box_blue"}))
                + self._step(("place", {"surface": "table"})))
        self.assertEqual(vlm.decide(self._messages("放到桌子上不是盘子里", st, hist)).skill, "stop")

    def test_fresh_task_with_new_target_not_stopped(self):
        # D3（无纪元标记的历史，与验证者用例一致）：明确点名了别的物体 -> 必须去抓
        hist = (self._step(("pick", {"target_id": "box_blue"}))
                + self._step(("place", {"surface": "table"})))
        call = MockVLM().decide(self._messages("把红色方块放进托盘", make_state(), hist))
        self.assertEqual(call.skill, "pick")
        self.assertEqual(call.args["target_id"], "box_red")

    def test_targetless_failures_do_not_hard_block_new_target(self):
        # D2（无纪元标记的历史）：3 次「目标未知」的失败不得让新目标直接 ask_user
        hist = (self._step(("pick", {}), ok=False, error_code="grasp_failed")
                + self._step(("pick", {}), ok=False, error_code="grasp_failed")
                + self._step(("pick", {}), ok=False, error_code="grasp_failed"))
        call = MockVLM().decide(self._messages("夹住蓝色方块", make_state(), hist))
        self.assertNotEqual(call.skill, "ask_user")
        # 同目标明确归属的失败仍会触发 ask_user（不放松原有上限）
        hist2 = (self._step(("pick", {"target_id": "box_blue"}), ok=False, error_code="grasp_failed")
                 + self._step(("pick", {"target_id": "box_blue"}), ok=False, error_code="grasp_failed")
                 + self._step(("pick", {"target_id": "box_blue"}), ok=False, error_code="grasp_failed"))
        call2 = MockVLM().decide(self._messages("夹住蓝色方块", make_state(), hist2))
        self.assertEqual(call2.skill, "ask_user")


# --------------------------------------------------------------------------- #
# 3g) 叠放/物体关系 + 表达不了就大声报错（task-31）
# --------------------------------------------------------------------------- #
class TestStackingAndLoudFailure(unittest.TestCase):
    """place 支持 on_top_of；关系解析不出来必须 ask_user，绝不静默丢托盘。"""

    def _messages(self, task, state, history=None):
        return build_messages(task, state, {"overview": make_image()}, history=history)

    @staticmethod
    def _picked_step(target_id):
        return step_messages(SkillCall("pick", {"target_id": target_id}),
                             SkillResult(ok=True, skill="pick", message="picked"))

    def test_stack_parse_moved_and_anchor(self):
        # C：「蓝色方块放在红色方块上方」-> pick(第 1 个物体) -> place(on_top_of=锚点)
        vlm, st = MockVLM(), make_state()
        task = "蓝色方块放在红色方块上方"
        call1 = vlm.decide(self._messages(task, st))
        self.assertEqual(call1.skill, "pick")
        self.assertEqual(call1.args["target_id"], "box_blue")
        call2 = vlm.decide(self._messages(task, st, self._picked_step("box_blue")))
        self.assertEqual(call2.skill, "place")
        self.assertEqual(call2.args.get("on_top_of"), "box_red")
        self.assertNotIn("container_id", call2.args)
        self.assertNotIn("surface", call2.args)
        self.assertIn("锚点", call2.reasoning)          # 依据写进 reasoning
        self.assertTrue(validate_skill_call(call2, st)[0])

    def test_stack_swapped_order(self):
        # B：两个物体必须区分开——第 1 个是被移动物，关系词前最近的是锚点
        vlm, st = MockVLM(), make_state()
        task = "红色方块放在蓝色方块上方"
        call1 = vlm.decide(self._messages(task, st))
        self.assertEqual(call1.args["target_id"], "box_red")
        call2 = vlm.decide(self._messages(task, st, self._picked_step("box_red")))
        self.assertEqual(call2.args.get("on_top_of"), "box_blue")

    def test_unresolvable_place_asks_user_not_tray(self):
        # B：有放置语义但解析不出目标 -> ask_user，绝不默认丢托盘
        vlm, st = MockVLM(), make_state()
        for task in ("把蓝色方块放到旁边", "把蓝色方块放好", "把蓝色方块放到上面"):
            with self.subTest(task=task):
                call = vlm.decide(self._messages(task, st, self._picked_step("box_blue")))
                self.assertEqual(call.skill, "ask_user", f"{task} 不得静默放置")
                self.assertNotEqual(call.args.get("container_id"), "tray")
                q = call.args.get("question", "")
                self.assertTrue(q.strip(), "ask_user 必须有具体问题")
                self.assertIn("当前可选", q)

    def test_non_graspable_anchor_asks_user(self):
        # A/B：托盘不能当叠放锚点
        vlm, st = MockVLM(), make_state()
        call = vlm.decide(self._messages("把蓝色方块放到托盘上方", st,
                                         self._picked_step("box_blue")))
        self.assertEqual(call.skill, "ask_user")
        self.assertIn("不可抓", call.args.get("question", ""))

    def test_single_object_stack_is_ambiguous(self):
        vlm, st = MockVLM(), make_state()
        call = vlm.decide(self._messages("把蓝色方块放到上面", st, self._picked_step("box_blue")))
        self.assertEqual(call.skill, "ask_user")

    def test_stack_end_to_end_and_reasoning_in_log(self):
        # C+D：完整链路 + 每步日志带 reasoning 摘要
        env, executor = StubEnv(), RecordingExecutor()
        logs: list[str] = []
        loop = AgentLoop(env, executor, MockVLM(), max_steps=8, log=logs.append)
        results = loop.run("蓝色方块放在红色方块上方")
        self.assertEqual([c.skill for c in executor.calls], ["pick", "place", "stop"])
        self.assertEqual(executor.calls[0].args["target_id"], "box_blue")
        self.assertEqual(executor.calls[1].args.get("on_top_of"), "box_red")
        self.assertTrue(all(r.ok for r in results), [r.message for r in results])
        self.assertTrue(any("依据:" in m for m in logs), logs)
        self.assertTrue(any("依据: MockVLM: 叠放到 box_red 上方" in m for m in logs), logs)


# --------------------------------------------------------------------------- #
# 3h) 横向关系 + 先规划后执行（task-35）
# --------------------------------------------------------------------------- #
class TestLateralRelations(unittest.TestCase):
    """place 支持 anchor_id+relation；未抓取前先规划，失败就 ask_user，不做半执行。"""

    def _messages(self, task, state, history=None):
        return build_messages(task, state, {"overview": make_image()}, history=history)

    @staticmethod
    def _picked_step(target_id):
        return step_messages(SkillCall("pick", {"target_id": target_id}),
                             SkillResult(ok=True, skill="pick", message="picked"))

    def test_english_right_of_two_objects(self):
        # 自测 a) 'set box_green right of box_blue'：第 1 步就规划好放置并去 pick，
        #         第 2 步 place(anchor_id=box_blue, relation=right)，不得先 pick 再 ask_user
        vlm, st = MockVLM(), make_state()
        task = "set box_green right of box_blue"
        call1 = vlm.decide(self._messages(task, st))
        self.assertEqual(call1.skill, "pick", "应先把放置目标规划好再抓")
        self.assertEqual(call1.args["target_id"], "box_green")
        self.assertIn("anchor_id=box_blue", call1.reasoning)
        self.assertIn("relation=right", call1.reasoning)
        call2 = vlm.decide(self._messages(task, st, self._picked_step("box_green")))
        self.assertEqual(call2.skill, "place")
        self.assertEqual(call2.args.get("anchor_id"), "box_blue")
        self.assertEqual(call2.args.get("relation"), "right")
        for k in ("container_id", "surface", "on_top_of"):
            self.assertNotIn(k, call2.args)
        self.assertTrue(validate_skill_call(call2, st)[0])

    def test_lateral_phrasings(self):
        vlm, st = MockVLM(), make_state()
        cases = (("put box_green to the right of box_blue", "right"),
                 ("place box_green beside box_blue", "beside"),
                 ("put box_green behind box_blue", "back"),
                 ("把绿色方块放在蓝色方块右边", "right"),
                 ("把绿色方块放在蓝色方块左边", "left"))
        for task, rel in cases:
            with self.subTest(task=task):
                call = vlm.decide(self._messages(task, st, self._picked_step("box_green")))
                self.assertEqual(call.skill, "place", task)
                self.assertEqual(call.args.get("relation"), rel, task)
                self.assertEqual(call.args.get("anchor_id"), "box_blue", task)

    def test_end_to_end_right_of(self):
        env, executor = StubEnv(), RecordingExecutor()
        loop = AgentLoop(env, executor, MockVLM(), max_steps=8, log=None)
        results = loop.run("set box_green right of box_blue")
        self.assertEqual([c.skill for c in executor.calls], ["pick", "place", "stop"])
        self.assertEqual(executor.calls[0].args["target_id"], "box_green")
        self.assertEqual(executor.calls[1].args.get("anchor_id"), "box_blue")
        self.assertEqual(executor.calls[1].args.get("relation"), "right")
        self.assertTrue(all(r.ok for r in results), [r.message for r in results])

    def test_unsupported_relation_asks_before_any_pick(self):
        # 自测 b) 不支持的方向：第 1 步 ask_user，且没有任何 pick/place；c) 回显锚点
        env, executor = StubEnv(), RecordingExecutor()
        loop = AgentLoop(env, executor, MockVLM(), max_steps=8, log=None)
        results = loop.run("把绿色方块放到蓝色方块斜后方")
        self.assertEqual(results[0].skill, "ask_user")
        self.assertEqual([c.skill for c in executor.calls], ["ask_user"])
        self.assertFalse(any(c.skill in ("pick", "place") for c in executor.calls),
                         "不支持的方向不得先 pick（半执行）")
        q = executor.calls[0].args.get("question", "")
        self.assertIn("box_blue", q, "报错必须回显已识别到的锚点")
        self.assertIn("斜后", q)
        self.assertIn("不支持", q)
        self.assertIn("right", q)

    def test_unsupported_relation_direct_decide(self):
        vlm, st = MockVLM(), make_state()
        for task in ("把绿色方块放到蓝色方块斜后方", "把绿色方块放到蓝色方块中间"):
            with self.subTest(task=task):
                call = vlm.decide(self._messages(task, st))
                self.assertEqual(call.skill, "ask_user", f"{task} 不得先 pick")
                self.assertIn("box_blue", call.args.get("question", ""))

    def test_pick_only_still_not_planned(self):
        # 不能误伤「只抓不放」：没有放置词就不需要放置计划
        vlm, st = MockVLM(), make_state()
        call = vlm.decide(self._messages("夹住蓝色方块", st))
        self.assertEqual(call.skill, "pick")
        self.assertNotIn("已规划放置", call.reasoning)


# --------------------------------------------------------------------------- #
# 3i) 悬挂指示词 + 复合方向（task-40 R1/R2）
# --------------------------------------------------------------------------- #
class TestDeicticAndCompoundDirections(unittest.TestCase):
    """R1：「那边」等指示词不得先 pick；R2：左上/upper-left/diagonal-back 不得执行错方向。"""

    def _messages(self, task, state, history=None):
        return build_messages(task, state, {"overview": make_image()}, history=history)

    @staticmethod
    def _picked_step(target_id):
        return step_messages(SkillCall("pick", {"target_id": target_id}),
                             SkillResult(ok=True, skill="pick", message="picked"))

    def test_deictic_asks_before_any_pick(self):
        # R1：原来「那边」是 ['pick','ask_user']，现在必须第 1 步就 ask_user
        for task in ("那边", "那边去", "这里", "那里", "它", "这个", "那个"):
            with self.subTest(task=task):
                env, executor = StubEnv(), RecordingExecutor()
                loop = AgentLoop(env, executor, MockVLM(), max_steps=6, log=None)
                results = loop.run(task)
                self.assertEqual(results[0].skill, "ask_user", task)
                self.assertFalse(any(c.skill in ("pick", "place") for c in executor.calls),
                                 f"{task} 不得先 pick")
                self.assertIn("当前可选", executor.calls[0].args.get("question", ""))

    def test_deictic_with_resolvable_placement_still_works(self):
        # 不误伤：有明确放置目标时，「它」仍可正常抓放
        vlm, st = MockVLM(), make_state()
        call = vlm.decide(self._messages("把它放到桌子上", st, self._picked_step("box_blue")))
        self.assertEqual(call.skill, "place")
        self.assertEqual(call.args.get("surface"), "table")

    def test_compound_directions_are_unsupported(self):
        # R2：这些复合方向必须 ask_user，并回显锚点，绝不执行一个错方向
        vlm, st = MockVLM(), make_state()
        tasks = ("把绿色方块放在蓝色方块左上", "把绿色方块放在蓝色方块右上",
                 "把绿色方块放在蓝色方块左下", "把绿色方块放在蓝色方块右下",
                 "把绿色方块放到蓝色方块斜后方",
                 "put box_green upper-left of box_blue",
                 "place box_green diagonal-back of box_blue",
                 "put box_green diagonally behind box_blue")
        for task in tasks:
            with self.subTest(task=task):
                call = vlm.decide(self._messages(task, st, self._picked_step("box_green")))
                self.assertEqual(call.skill, "ask_user", f"{task} 不得执行一个错方向")
                q = call.args.get("question", "")
                self.assertIn("box_blue", q)
                self.assertIn("不支持", q)

    def test_compound_direction_no_pick_in_loop(self):
        env, executor = StubEnv(), RecordingExecutor()
        loop = AgentLoop(env, executor, MockVLM(), max_steps=6, log=None)
        results = loop.run("把绿色方块放在蓝色方块左上")
        self.assertEqual(results[0].skill, "ask_user")
        self.assertFalse(any(c.skill in ("pick", "place") for c in executor.calls))

    def test_supported_directions_still_execute(self):
        # 不退化：right/left/front/back/beside 仍正常发出 place
        vlm, st = MockVLM(), make_state()
        for task, rel in (("把绿色方块放在蓝色方块左边", "left"),
                          ("set box_green right of box_blue", "right"),
                          ("place box_green beside box_blue", "beside"),
                          ("put box_green behind box_blue", "back"),
                          ("put box_green in front of box_blue", "front")):
            with self.subTest(task=task):
                call = vlm.decide(self._messages(task, st, self._picked_step("box_green")))
                self.assertEqual(call.skill, "place", task)
                self.assertEqual(call.args.get("relation"), rel, task)

    def test_compound_alias_table_all_ask_user(self):
        # task-42：统一复合方向别名表（中英）——全部不支持，ask_user 并回显锚点
        vlm, st = MockVLM(), make_state()
        cn = ("左上", "右上", "左下", "右下", "斜左", "斜右", "斜前", "斜后",
              "左上方", "右上方", "前上方", "后上方",
              "斜左上方", "斜右上方", "斜前上方", "斜后上方",
              "左前", "右前", "左后", "右后", "对角", "中间", "偏左", "正上")
        en = ("upper-left", "upper-right", "upper-front", "upper-back",
              "top-left", "top-right", "top left", "bottom-left", "bottom-right",
              "lower-left", "lower-right", "lower left", "front-left", "front-right",
              "back-left", "back-right", "diagonal", "diagonally")
        for word in cn:
            with self.subTest(word=word):
                call = vlm.decide(self._messages(f"把绿色方块放在蓝色方块{word}", st,
                                                 self._picked_step("box_green")))
                self.assertEqual(call.skill, "ask_user", word)
                self.assertIn("box_blue", call.args.get("question", ""), word)
                self.assertIn("不支持", call.args.get("question", ""), word)
        for word in en:
            with self.subTest(word=word):
                call = vlm.decide(self._messages(f"put box_green {word} of box_blue", st,
                                                 self._picked_step("box_green")))
                self.assertEqual(call.skill, "ask_user", word)
                self.assertIn("box_blue", call.args.get("question", ""), word)
                self.assertIn("不支持", call.args.get("question", ""), word)

    def test_diagonal_above_not_on_top_of(self):
        # N2：斜右上方既不能是 left/right，也不能是 on_top_of
        vlm, st = MockVLM(), make_state()
        for task in ("把绿色方块放在蓝色方块斜右上方", "把绿色方块放在蓝色方块斜左上方",
                     "把绿色方块放在蓝色方块左上方"):
            with self.subTest(task=task):
                call = vlm.decide(self._messages(task, st, self._picked_step("box_green")))
                self.assertEqual(call.skill, "ask_user", task)
                self.assertNotIn("on_top_of", call.args)
                self.assertNotIn("relation", call.args)

    def test_compound_alias_no_action_in_loop(self):
        # 序列里只有 ask_user，没有任何 pick/place
        for task in ("put box_green top-left of box_blue",
                     "把绿色方块放在蓝色方块斜右上方"):
            with self.subTest(task=task):
                env, executor = StubEnv(), RecordingExecutor()
                loop = AgentLoop(env, executor, MockVLM(), max_steps=6, log=None)
                results = loop.run(task)
                self.assertEqual(results[0].skill, "ask_user", task)
                self.assertEqual([c.skill for c in executor.calls], ["ask_user"], task)
                self.assertFalse(any(c.skill in ("pick", "place") for c in executor.calls), task)

    def test_pure_stack_and_directions_not_broken(self):
        # 要求 3：纯 on top of / 纯方向 / 桌面 / 只抓不放 全部保持
        vlm, st = MockVLM(), make_state()
        cases = (("put box_green on top of box_blue", "on_top_of", "box_blue"),
                 ("把绿色方块放在蓝色方块上方", "on_top_of", "box_blue"),
                 ("put box_green to the right of box_blue", "relation", "right"),
                 ("place box_green beside box_blue", "relation", "beside"))
        for task, key, val in cases:
            with self.subTest(task=task):
                call = vlm.decide(self._messages(task, st, self._picked_step("box_green")))
                self.assertEqual(call.skill, "place", task)
                self.assertEqual(call.args.get(key), val, task)
        call = vlm.decide(self._messages("把蓝色方块放回到桌子上", st,
                                         self._picked_step("box_blue")))
        self.assertEqual(call.skill, "place")
        self.assertEqual(call.args.get("surface"), "table")
        self.assertEqual(vlm.decide(self._messages("夹住蓝色方块", st)).skill, "pick")


# --------------------------------------------------------------------------- #
# 3j) 谁压着谁 / 抓取被遮挡（task-45：只补观测）
# --------------------------------------------------------------------------- #
class TestSceneRelations(unittest.TestCase):
    """纯几何关系观测 + 中文摘要进 VLM message + 失败 obstruction_id 回灌。"""

    @staticmethod
    def _stack_state():
        """box_blue 正好叠在 box_red 上（中心距 = 两个半高之和）。"""
        red = _obj("box_red", "box", "red", [0.40, 0.10, 0.42], [0.025, 0.025, 0.025])
        blue = _obj("box_blue", "box", "blue", [0.40, 0.10, 0.47], [0.025, 0.025, 0.025])
        return make_state(objects=[red, blue])

    def test_stacked_relations(self):
        rel = derive_relations(self._stack_state())
        self.assertEqual(rel["box_blue"]["supported_by"], ["box_red"])
        self.assertEqual(rel["box_red"]["supporting"], ["box_blue"])
        self.assertTrue(rel["box_red"]["grasp_blocked"])
        self.assertEqual(rel["box_red"]["blocked_by"], ["box_blue"])
        self.assertFalse(rel["box_blue"]["grasp_blocked"])
        self.assertEqual(rel["box_blue"]["blocked_by"], [])

    def test_serialized_fields_and_summary(self):
        st = self._stack_state()
        payload = serialize_state(st, "把 box_blue 挪走")
        objs = {o["id"]: o for o in payload["objects"]}
        self.assertTrue(objs["box_red"]["grasp_blocked"])
        self.assertEqual(objs["box_red"]["blocked_by"], ["box_blue"])
        self.assertEqual(objs["box_blue"]["supported_by"], ["box_red"])
        self.assertEqual(payload["relations"]["box_red"]["blocked_by"], ["box_blue"])
        self.assertIn("box_red 上方压着 box_blue（抓取受阻）", payload["relation_summary"])
        self.assertIn("box_blue 叠在 box_red 上", payload["relation_summary"])
        # 摘要随图像一起进每步 VLM message
        msgs = build_messages("把 box_blue 挪走", st, {"overview": make_image()})
        text = msgs[-1]["content"][0]["text"]
        self.assertIn("场景关系摘要", text)
        self.assertIn("box_red 上方压着 box_blue", text)

    def test_side_by_side_not_misjudged(self):
        # 反例：默认场景物体并排，不得误判成支撑/遮挡
        st = make_state()
        rel = derive_relations(st)
        for oid, r in rel.items():
            self.assertFalse(r["grasp_blocked"], oid)
            self.assertEqual(r["blocked_by"], [], oid)
            self.assertEqual(r["supporting"], [], oid)
            self.assertEqual(r["supported_by"], [], oid)
        self.assertIn("没有物体互相压着", serialize_state(st, "t")["relation_summary"])

    def test_mockvlm_ok_with_relations_fields(self):
        # R4：新增字段不得让 schema / MockVLM 输出路径报错
        st = self._stack_state()
        call = MockVLM().decide(build_messages("抓 box_red", st, {"overview": make_image()}))
        self.assertEqual(call.skill, "pick")
        self.assertTrue(validate_skill_call(call, st)[0])

    def test_failed_pick_observation_carries_obstruction(self):
        # U1/R3：抓 box_red 被 box_blue 压住 -> 失败 observation 带 obstruction_id 等
        env = StubEnv(state=self._stack_state())
        executor = StubSkillExecutor({"pick": (False, "grasp_blocked", "抓取被 box_blue 遮挡")})
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red"}), SkillCall("stop", {})])
        loop = AgentLoop(env, executor, vlm, cameras=("overview",), max_steps=4, log=None)
        results = loop.run("把 box_red 抓起来")
        metrics = results[0].to_observation()["metrics"]
        self.assertEqual(metrics.get("obstruction_id"), "box_blue")
        self.assertIn("failure_reason", metrics)
        self.assertIn("contact_force_norm", metrics)
        self.assertEqual(metrics.get("suggested_cameras"), ["overview"])
        # 回灌进历史的 observation JSON，模型下一步能看到
        self.assertIn("obstruction_id", loop.history[1]["content"])

    def test_skills_obstruction_id_not_overwritten(self):
        # skills 侧 R5 已给 obstruction_id 时必须原样保留
        class _Exec:
            def __call__(self, call):
                return SkillResult(ok=False, skill=call.skill, message="x",
                                   error_code="grasp_blocked",
                                   metrics={"obstruction_id": "from_skills"})

        loop = AgentLoop(StubEnv(state=self._stack_state()), _Exec(),
                         ScriptedVLM([SkillCall("pick", {"target_id": "box_red"}),
                                      SkillCall("stop", {})]),
                         max_steps=4, log=None)
        self.assertEqual(loop.run("t")[0].metrics.get("obstruction_id"), "from_skills")


# --------------------------------------------------------------------------- #
# 3k) no-oracle 感知：target_pixel + 真值剥离 + pixel_resolver（task-49 / §4.3）
# --------------------------------------------------------------------------- #
class TestPixelPerception(unittest.TestCase):
    """interfaces §4.3：TOOLS 像素参数、include_poses=False 真值剥离、pixel_resolver 反投影。"""

    def test_serialize_state_include_poses_false_strips_truth(self):
        st = make_state()
        payload = serialize_state(st, "把红色方块放进托盘", include_poses=False)
        text = json.dumps(payload, ensure_ascii=False)
        for o in payload["objects"]:
            self.assertEqual(set(o.keys()), {"id", "color", "category"})
        self.assertNotIn("relations", payload)
        self.assertNotIn("relation_summary", payload)
        self.assertNotIn('"pose"', text)
        self.assertNotIn('"size"', text)
        self.assertNotIn("grasp_blocked", text)
        cal = payload["camera_calibration"]
        for k in ("camera", "width", "height", "fovy", "table_z"):
            self.assertIn(k, cal)
        # oracle 模式仍带真值
        full = serialize_state(st, "把红色方块放进托盘")
        self.assertIn("pose", full["objects"][0])
        self.assertIn("relations", full)

    def test_build_messages_passes_include_poses(self):
        st = make_state()
        self.assertNotIn("无真值感知模式", system_prompt())
        msgs = build_messages("抓红色方块", st, {"top": make_image()}, include_poses=False)
        self.assertIn("无真值感知模式", msgs[0]["content"])
        user_text = msgs[-1]["content"][0]["text"]
        self.assertIn("camera_calibration", user_text)
        self.assertNotIn('"pose"', user_text)
        # oracle（默认）路径不变
        msgs2 = build_messages("抓红色方块", st, {"top": make_image()})
        self.assertEqual(msgs2[0]["content"], system_prompt())
        self.assertIn('"pose"', msgs2[-1]["content"][0]["text"])

    def test_tools_have_pixel_params(self):
        pick = _tool_by_name("pick")["function"]["parameters"]["properties"]
        place = _tool_by_name("place")["function"]["parameters"]["properties"]
        for name in ("target_pixel", "pixel_camera"):
            self.assertIn(name, pick)
            self.assertIn(name, place)
        self.assertIn("target_xy", pick)
        self.assertIn("target_z_hint", pick)
        self.assertEqual(pick["pixel_camera"]["oneOf"][0]["enum"], ["top", "overview", "wrist"])

    def test_validate_pick_three_way_and_place_pixel(self):
        st = make_state()
        # §5.12-1：target_id + target_pixel 不再判冲突——丢掉 target_id，以像素为准
        redun = SkillCall("pick", {"target_id": "box_red", "target_pixel": [1, 2]})
        ok, msg = validate_skill_call(redun, st)
        self.assertTrue(ok, msg)
        self.assertNotIn("target_id", redun.args)
        self.assertEqual(redun.args["target_pixel"], [1, 2])
        # 真矛盾（surface=table + container_id）仍然拒绝，且错误里点名冲突项
        ok, msg = validate_skill_call(SkillCall("place", {"surface": "table", "container_id": "tray"}), st)
        self.assertFalse(ok)
        self.assertIn("四选一", msg)
        ok, msg = validate_skill_call(SkillCall("pick", {}), st)
        self.assertFalse(ok)
        self.assertIn("target_id", msg)
        ok, msg = validate_skill_call(SkillCall("pick", {"target_xy": [0.4, 0.1]}), st)
        self.assertTrue(ok, msg)
        # 像素允许越界：反投影阶段才失败成 perception_error
        ok, msg = validate_skill_call(SkillCall("pick", {"target_pixel": [-5, 9999]}), st)
        self.assertTrue(ok, msg)
        ok, msg = validate_skill_call(SkillCall("pick", {"target_pixel": [1]}), st)
        self.assertFalse(ok)
        self.assertIn("target_pixel", msg)
        ok, msg = validate_skill_call(
            SkillCall("pick", {"target_pixel": [1, 2], "pixel_camera": "left"}), st)
        self.assertFalse(ok)
        self.assertIn("pixel_camera", msg)
        ok, msg = validate_skill_call(
            SkillCall("place", {"surface": "table", "target_pixel": [3, 4]}), st)
        self.assertTrue(ok, msg)

    def test_pixel_resolver_rewrites_call_and_logs(self):
        seen = {}

        def resolver(u, v, camera):
            seen["uv"] = (u, v, camera)
            return np.array([0.41, 0.12, 0.42])

        env, executor = StubEnv(), RecordingExecutor()
        logs: list[str] = []
        vlm = ScriptedVLM([SkillCall("pick", {"target_pixel": [123, 45]}), SkillCall("stop", {})])
        loop = AgentLoop(env, executor, vlm, cameras=(), max_steps=4, log=logs.append,
                         pixel_resolver=resolver)
        loop.run("抓那个东西")
        self.assertEqual(seen["uv"], (123.0, 45.0, "top"))
        self.assertEqual(executor.calls[0].skill, "pick")
        self.assertEqual(executor.calls[0].args.get("target_xy"), [0.41, 0.12])
        self.assertNotIn("target_pixel", executor.calls[0].args)
        self.assertTrue(
            any("像素 (123,45) [top] -> 世界 (0.410, 0.120, 0.420)" in m for m in logs), logs)

    def test_pixel_resolver_none_is_retryable_perception_error(self):
        state = {"n": 0}

        def resolver(u, v, camera):
            state["n"] += 1
            return None if state["n"] == 1 else [0.40, 0.10, 0.42]

        env, executor = StubEnv(), RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_pixel": [900, 900]}),
                           SkillCall("pick", {"target_pixel": [100, 100]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, executor, vlm, cameras=(), max_steps=5, log=None,
                         pixel_resolver=resolver)
        results = loop.run("抓那个东西")
        self.assertEqual(results[0].error_code, "perception_error")
        self.assertEqual(results[0].metrics.get("target_pixel"), [900, 900])
        self.assertEqual(results[0].metrics.get("pixel_camera"), "top")
        self.assertIn("900", results[0].message)
        self.assertIn("perception_error", loop.history[1]["content"])
        self.assertIn("900", loop.history[1]["content"])
        self.assertEqual(executor.calls[0].args.get("target_xy"), [0.4, 0.1])
        self.assertEqual(results[-1].skill, "stop")

    def test_pixel_resolver_exception_is_perception_error(self):
        def boom(u, v, camera):
            raise RuntimeError("no calibration")

        loop = AgentLoop(StubEnv(), RecordingExecutor(),
                         ScriptedVLM([SkillCall("pick", {"target_pixel": [5, 5]}),
                                      SkillCall("stop", {})]),
                         cameras=(), max_steps=3, log=None, pixel_resolver=boom)
        results = loop.run("t")
        self.assertEqual(results[0].error_code, "perception_error")
        self.assertIn("反投影异常", results[0].message)

    def test_no_resolver_configured_is_perception_error(self):
        loop = AgentLoop(StubEnv(), RecordingExecutor(),
                         ScriptedVLM([SkillCall("pick", {"target_pixel": [5, 5]}),
                                      SkillCall("stop", {})]),
                         cameras=(), max_steps=3, log=None)
        results = loop.run("t")
        self.assertEqual(results[0].error_code, "perception_error")
        self.assertIn("pixel_resolver", results[0].message)

    def test_place_target_pixel_resolves_to_xy(self):
        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red"}),
                           SkillCall("place", {"surface": "table", "target_pixel": [256, 300]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(StubEnv(), ex, vlm, cameras=(), max_steps=5, log=None,
                         pixel_resolver=lambda u, v, c: [0.42, 0.05, 0.40])
        loop.run("把红色方块放到桌上")
        self.assertEqual(ex.calls[1].args.get("xy"), [0.42, 0.05])
        self.assertNotIn("target_pixel", ex.calls[1].args)

    def test_mockvlm_oracle_path_unchanged(self):
        def boom(u, v, camera):
            raise AssertionError("MockVLM oracle 路径不应调用 pixel_resolver")

        ex = RecordingExecutor()
        loop = AgentLoop(StubEnv(), ex, MockVLM(), cameras=("overview",), max_steps=8,
                         log=None, pixel_resolver=boom)
        results = loop.run("把红色的方块放进托盘")
        self.assertEqual([c.skill for c in ex.calls], ["pick", "place", "stop"])
        self.assertEqual(ex.calls[0].args.get("target_id"), "box_red")
        self.assertTrue(all(r.ok for r in results), [r.message for r in results])

    def test_agent_loop_include_poses_false_strips_truth(self):
        # task-52 R4：AgentLoop 透传 include_poses=False 时 prompt 里不得有真值
        vlm = ScriptedVLM([SkillCall("stop", {})])
        loop = AgentLoop(StubEnv(), RecordingExecutor(), vlm, cameras=("top",),
                         max_steps=2, log=None, include_poses=False)
        loop.run("抓红色方块")
        msgs = vlm.seen_messages[-1]
        user_text = msgs[-1]["content"][0]["text"]
        self.assertIn("无真值感知模式", msgs[0]["content"])
        self.assertIn("camera_calibration", user_text)
        self.assertNotIn('"pose"', user_text)
        self.assertNotIn('"size"', user_text)
        self.assertNotIn("relations", user_text)

    def test_agent_loop_include_poses_true_keeps_truth(self):
        # task-52 R3：默认 True 行为不变，prompt 里仍有位姿与关系
        vlm = ScriptedVLM([SkillCall("stop", {})])
        loop = AgentLoop(StubEnv(), RecordingExecutor(), vlm, cameras=("top",),
                         max_steps=2, log=None)   # 默认 include_poses=True
        loop.run("抓红色方块")
        msgs = vlm.seen_messages[-1]
        user_text = msgs[-1]["content"][0]["text"]
        self.assertIn('"pose"', user_text)
        self.assertIn('"size"', user_text)
        self.assertIn("relations", user_text)
        self.assertNotIn("无真值感知模式", msgs[0]["content"])


# --------------------------------------------------------------------------- #
# 3l) 重复守卫参数感知（task-53 D1）
# --------------------------------------------------------------------------- #
class TestRepeatGuardArgsAware(unittest.TestCase):
    """守卫 key = (skill, error_code, args 指纹)：参数变了必须放行，完全相同才终止。"""

    class _AlwaysFail:
        def __init__(self):
            self.calls: list[SkillCall] = []

        def __call__(self, call):
            self.calls.append(call)
            return SkillResult(ok=False, skill=call.skill, message="boom", error_code="ik_fail")

    def test_changed_args_are_allowed_through(self):
        ex = self._AlwaysFail()
        vlm = ScriptedVLM([
            SkillCall("pick", {"target_id": "box_red"}),
            SkillCall("pick", {"target_id": "box_red"}),                 # 同参 -> 触发 pending
            SkillCall("pick", {"target_id": "box_red", "approach_height": 0.2}),  # 参数变了 -> 必须执行
            SkillCall("stop", {}),
        ])
        logs: list[str] = []
        loop = AgentLoop(StubEnv(), ex, vlm, max_steps=6, log=logs.append)
        results = loop.run("t")
        picks = [c for c in ex.calls if c.skill == "pick"]
        self.assertEqual(len(picks), 3, "参数改变后的调用必须被执行")
        self.assertEqual(picks[2].args.get("approach_height"), 0.2)
        self.assertTrue(any("放行执行" in m for m in logs), logs)
        self.assertEqual(results[-1].skill, "stop")

    def test_identical_args_still_terminate(self):
        ex = self._AlwaysFail()
        same = {"target_id": "box_red"}
        vlm = ScriptedVLM([SkillCall("pick", dict(same)) for _ in range(4)])
        loop = AgentLoop(StubEnv(), ex, vlm, max_steps=6, log=None)
        results = loop.run("t")
        self.assertEqual(len(ex.calls), 2, "完全相同的失败调用第 3 次不再执行")
        self.assertEqual(len(results), 2)

    def test_invalid_then_corrected_pixel_call_runs(self):
        # D1 原场景（§5.12 后语义调整）：前两次是**真矛盾**被拒，第 3 次改成像素落点 -> 必须执行。
        # （target_id+target_pixel 现在按 §5.12-1 纠正放行，不再当冲突，见 TestParamCorrections）
        ex = RecordingExecutor()
        bad = {"container_id": "tray", "surface": "table"}      # 真矛盾：容器与桌面同给
        vlm = ScriptedVLM([SkillCall("place", dict(bad)),
                           SkillCall("place", dict(bad)),
                           SkillCall("place", {"container_id": "tray", "target_pixel": [228, 422]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(StubEnv(), ex, vlm, cameras=(), max_steps=6, log=None,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.40])
        results = loop.run("把东西放进托盘")
        self.assertEqual([r.error_code for r in results[:2]], ["invalid_call", "invalid_call"])
        self.assertEqual(ex.calls[0].skill, "place")
        self.assertEqual(ex.calls[0].args.get("container_id"), "tray")
        self.assertEqual(ex.calls[0].args.get("xy"), [0.4, 0.1])
        self.assertNotIn("target_pixel", ex.calls[0].args)
        fixes = results[2].metrics.get("corrected_args") or []
        self.assertEqual(fixes[0]["key"], "xy", "落点在容器内 -> 记一条 accepted 纠正")
        self.assertEqual(results[-1].skill, "stop")


# --------------------------------------------------------------------------- #
# 4) AgentLoop 闭环
# --------------------------------------------------------------------------- #
class TestAgentLoop(unittest.TestCase):

    def test_loop_with_mock_vlm_end_to_end(self):
        env = StubEnv()
        executor = RecordingExecutor()
        loop = AgentLoop(env, executor, MockVLM(), cameras=("overview",), max_steps=8, log=print)
        results = loop.run("把红色的方块放进托盘")
        self.assertEqual([c.skill for c in executor.calls], ["pick", "place", "stop"])
        self.assertTrue(all(r.ok for r in results), [r.message for r in results])
        # 3 步 x (assistant 动作 + user 观察) = 6 条历史
        self.assertEqual(len(loop.history), 6)
        # 每步都渲染了 overview
        self.assertEqual(env.render_calls, ["overview"] * 3)
        # 消息里带图像 data URI
        last_user = loop.history[-1]
        self.assertIn("OBSERVATION_JSON", last_user["content"])

    def test_loop_records_invalid_call_and_continues(self):
        env = StubEnv()
        executor = RecordingExecutor()
        vlm = ScriptedVLM([
            SkillCall("pick", {"target_id": "ghost"}),          # 校验应拒绝
            SkillCall("stop", {}),
        ])
        loop = AgentLoop(env, executor, vlm, max_steps=5, log=None)
        results = loop.run("抓幽灵")
        self.assertEqual(results[0].error_code, "invalid_call")
        self.assertEqual(results[-1].skill, "stop")
        self.assertEqual([c.skill for c in executor.calls], ["stop"], "非法调用不应进入执行器")

    def test_loop_survives_vlm_exception(self):
        # task-67：VLM 异常不再第一步就 break —— 允许连续 2 次 vlm_error 才中止
        env = StubEnv()

        class BoomVLM:
            def __init__(self):
                self.calls = 0

            def decide(self, messages):
                self.calls += 1
                raise RuntimeError("network down")

        vlm = BoomVLM()
        loop = AgentLoop(env, RecordingExecutor(), vlm, max_steps=4, log=None)
        results = loop.run("抓红色方块")
        self.assertEqual(len(results), 2, "连续 2 次 vlm_error 才中止")
        self.assertTrue(all(r.error_code == "vlm_error" for r in results))
        self.assertEqual(vlm.calls, 2, "每次都重新决策（会重新截图/重建 messages）")

    def test_loop_survives_executor_exception(self):
        env = StubEnv()

        def bad_executor(call):
            raise ValueError("actuator exploded")

        loop = AgentLoop(env, bad_executor, ScriptedVLM([SkillCall("home", {})]), max_steps=4, log=None)
        results = loop.run("回家")
        self.assertEqual(results[0].error_code, "executor_exception")
        self.assertFalse(results[0].ok)

    def test_loop_respects_max_steps(self):
        env = StubEnv()

        class AlwaysPick:
            def decide(self, messages):
                return SkillCall("pick", {"target_id": "box_red"})

        loop = AgentLoop(env, RecordingExecutor(), AlwaysPick(), max_steps=3, log=None)
        results = loop.run("抓")
        self.assertEqual(len(results), 3)
        self.assertEqual([r.skill for r in results], ["pick"] * 3)

    def test_loop_skips_failing_camera(self):
        env = StubEnv(render_ok=False)
        executor = RecordingExecutor()
        loop = AgentLoop(env, executor, MockVLM(), cameras=("overview", "wrist"), max_steps=4, log=None)
        results = loop.run("把红色方块放进托盘")
        self.assertTrue(results, "渲染失败也必须继续决策")
        self.assertEqual([c.skill for c in executor.calls][:2], ["pick", "place"])


# --------------------------------------------------------------------------- #
# 11) v2 §5.3 控制面工具：plan_subgoals / locate / subgoal_done + pick/place 新参数
# --------------------------------------------------------------------------- #
class TestV2SchemaTools(unittest.TestCase):

    def test_new_tools_registered_and_whitelist_consistent(self):
        for name in ("plan_subgoals", "locate", "subgoal_done"):
            self.assertIn(name, SKILL_NAMES)
            tool = _tool_by_name(name)
            self.assertEqual(set(tool["function"]["parameters"]["properties"]), set(ALLOWED_ARGS[name]))
        self.assertEqual(_tool_by_name("locate")["function"]["parameters"]["required"], ["target"])
        self.assertEqual(_tool_by_name("subgoal_done")["function"]["parameters"]["required"], ["note"])
        self.assertEqual(_tool_by_name("plan_subgoals")["function"]["parameters"]["required"], ["subgoals"])
        items = _tool_by_name("plan_subgoals")["function"]["parameters"]["properties"]["subgoals"]["items"]
        self.assertEqual(items["properties"]["goal_type"]["enum"], list(GOAL_TYPES))

    def test_pick_place_new_params_are_optional(self):
        pp = _tool_by_name("pick")["function"]["parameters"]["properties"]
        self.assertIn("grasp_profile", pp)
        self.assertEqual(pp["grasp_profile"]["oneOf"][0]["enum"], list(GRASP_PROFILES))
        qp = _tool_by_name("place")["function"]["parameters"]["properties"]
        self.assertIn("descend_until_contact", qp)
        # required 必须保持向后兼容（旧调用不受影响）
        self.assertEqual(_tool_by_name("pick")["function"]["parameters"]["required"], ["target_id"])
        self.assertEqual(_tool_by_name("place")["function"]["parameters"]["required"], ["container_id"])

    def test_parse_and_validate_new_tools(self):
        st = make_state()
        call = parse_skill_call({"name": "locate", "arguments": {"target": "红色方块", "color": "red"}})
        self.assertEqual(call.skill, "locate")
        self.assertEqual(call.args["target"], "红色方块")
        self.assertTrue(validate_skill_call(call, st)[0])
        done = parse_skill_call({"name": "subgoal_done", "arguments": {"note": "已在图上确认"}})
        self.assertTrue(validate_skill_call(done, st)[0])
        plan = parse_skill_call({"name": "plan_subgoals",
                                 "arguments": {"subgoals": [{"instruction": "A", "goal_type": "pick"}]}})
        self.assertEqual(len(plan.args["subgoals"]), 1)
        # locate 默认相机 / 缺省参数
        bare = parse_skill_call({"name": "locate", "arguments": {"target": "托盘"}})
        self.assertEqual(bare.args, {"target": "托盘"})

    def test_validate_rejects_bad_new_params(self):
        st = make_state()
        cases = [
            (SkillCall("locate", {}), "target"),
            (SkillCall("locate", {"target": "x", "camera": "side_42"}), "camera"),
            (SkillCall("locate", {"target": "x", "color": ""}), "color"),
            (SkillCall("subgoal_done", {}), "note"),
            (SkillCall("pick", {"target_id": "box_red", "grasp_profile": "diagonal"}), "grasp_profile"),
            (SkillCall("place", {"container_id": "tray", "descend_until_contact": "yes"}),
             "descend_until_contact"),
        ]
        for call, needle in cases:
            ok, msg = validate_skill_call(call, st)
            self.assertFalse(ok, f"{call.skill} 应被拒绝")
            self.assertIn(needle, msg)
        self.assertTrue(validate_skill_call(
            SkillCall("pick", {"target_id": "box_red", "grasp_profile": "side"}), st)[0])
        self.assertTrue(validate_skill_call(
            SkillCall("place", {"container_id": "tray", "descend_until_contact": True}), st)[0])
        self.assertTrue(validate_skill_call(
            SkillCall("place", {"container_id": "tray", "descend_until_contact": None}), st)[0])

    def test_plan_subgoals_rejected_in_execution_loop(self):
        st = make_state()
        call = SkillCall("plan_subgoals", {"subgoals": [{"instruction": "A", "goal_type": "pick"}]})
        ok, msg = validate_skill_call(call, st)
        self.assertFalse(ok, "执行循环里 plan_subgoals 必须判 invalid_call")
        self.assertIn("规划", msg)
        ex = RecordingExecutor()
        loop = AgentLoop(StubEnv(), ex, ScriptedVLM([call, SkillCall("stop", {})]),
                         max_steps=4, log=None)
        results = loop.run("先规划再干活")
        self.assertEqual(results[0].skill, "plan_subgoals")
        self.assertEqual(results[0].error_code, "invalid_call")
        self.assertEqual([c.skill for c in ex.calls], ["stop"], "规划工具绝不能进技能执行器")

    def test_no_oracle_prompt_has_workflow_and_daily_items(self):
        p = system_prompt(include_poses=False)
        for needle in ("工作流程", "locate", "subgoal_done", "水瓶", "眼镜盒", "grasp_profile"):
            self.assertIn(needle, p, f"无真值提示词缺少 {needle!r}")
        oracle = system_prompt(include_poses=True)
        self.assertNotIn("工作流程", oracle)
        self.assertNotIn("眼镜盒", oracle)
        self.assertEqual(system_prompt(), oracle, "默认（oracle）提示词必须保持原样")


# --------------------------------------------------------------------------- #
# 12) AgentLoop：locate / subgoal_done 特判
# --------------------------------------------------------------------------- #
class StubLocator:
    """locator(target, color, camera) -> dict 桩；result 为 Exception 时抛异常。"""

    def __init__(self, result):
        self.result = result
        self.calls: list[tuple] = []

    def __call__(self, target, color, camera):
        self.calls.append((target, color, camera))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class TestLoopControlPlane(unittest.TestCase):

    def setUp(self):
        clear_pixel_cache()

    def test_locate_goes_to_locator_not_executor(self):
        loc = StubLocator({"ok": True, "pixel": [100.5, 120.25], "bbox": [80, 100, 120, 140],
                           "source": "color_blob", "why": "颜色分割找到 red 色块"})
        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("locate", {"target": "红色方块", "camera": "top", "color": "red"}),
                           SkillCall("subgoal_done", {"note": "已在 top 图确认红块在托盘里"})])
        loop = AgentLoop(StubEnv(), ex, vlm, max_steps=5, log=None, locator=loc)
        results = loop.run("把红色方块放进托盘")

        self.assertEqual([r.skill for r in results], ["locate", "subgoal_done"])
        self.assertTrue(results[0].ok)
        self.assertEqual(results[0].metrics["pixel"], [100.5, 120.25])
        self.assertEqual(results[0].metrics["bbox"], [80.0, 100.0, 120.0, 140.0])
        self.assertEqual(results[0].metrics["source"], "color_blob")
        self.assertEqual(loc.calls, [("红色方块", "red", "top")])
        self.assertEqual(ex.calls, [], "locate/subgoal_done 绝不能进技能执行器")
        # subgoal_done：记 ok=True 并结束本 epoch（自评不算验收）
        self.assertTrue(results[1].ok)
        self.assertEqual(results[1].metrics.get("self_reported"), True)
        self.assertIn("复核", results[1].message + "复核")
        self.assertEqual(len(loop.history), 4, "两步 x (调用+观察)")

    def test_locate_without_locator_is_retryable_perception_error(self):
        ex = RecordingExecutor()
        loop = AgentLoop(StubEnv(), ex, ScriptedVLM([SkillCall("locate", {"target": "红色方块"})]),
                         max_steps=4, log=None)
        results = loop.run("标定一下")
        self.assertEqual(results[0].skill, "locate")
        self.assertFalse(results[0].ok)
        self.assertEqual(results[0].error_code, "perception_error", "locator 缺失必须是可重试错误")
        self.assertIn("locator", results[0].message)
        self.assertEqual([c.skill for c in ex.calls], ["stop"], "locate 不执行")

    def test_locate_exception_and_pixel_less_return_are_retryable(self):
        loop = AgentLoop(StubEnv(), RecordingExecutor(),
                         ScriptedVLM([SkillCall("locate", {"target": "红色方块", "color": "red"})]),
                         max_steps=3, log=None, locator=StubLocator(RuntimeError("camera exploded")))
        res = loop.run("标定")[0]
        self.assertFalse(res.ok)
        self.assertEqual(res.error_code, "perception_error")
        self.assertIn("异常", res.message)

        class NoPixel:
            def __call__(self, target, color, camera):
                return {"ok": True}                    # 没有 pixel -> 不算成功

        loop2 = AgentLoop(StubEnv(), RecordingExecutor(),
                          ScriptedVLM([SkillCall("locate", {"target": "红色方块", "color": "red"})]),
                          max_steps=3, log=None, locator=NoPixel())
        r2 = loop2.run("标定")[0]
        self.assertFalse(r2.ok)
        self.assertEqual(r2.error_code, "perception_error")
        self.assertIn("没拿到像素", r2.message)

    def test_locate_missing_target_is_invalid_call(self):
        loc = StubLocator({"ok": True, "pixel": [1, 2]})
        loop = AgentLoop(StubEnv(), RecordingExecutor(),
                         ScriptedVLM([SkillCall("locate", {}), SkillCall("stop", {})]),
                         max_steps=3, log=None, locator=loc)
        results = loop.run("标定")
        self.assertEqual(results[0].error_code, "invalid_call")
        self.assertEqual(loc.calls, [], "非法 locate 不该调用定位后端")


# --------------------------------------------------------------------------- #
# 13) taskplan：规划 / 复核 / 编排
# --------------------------------------------------------------------------- #
class StubBlob:
    """模拟 sim/perception.BlobResult（§5.1）：只做测试桩，不越界实现 perception。"""

    def __init__(self, found=True, uv=(100.0, 120.0), bbox=(80, 100, 120, 140), color="red",
                 area_px=400):
        self.found = bool(found)
        self.uv = None if uv is None else tuple(uv)
        self.bbox = None if bbox is None else tuple(bbox)
        self.area_px = int(area_px)
        self.color = color
        self.score = 0.9


class StubPerception:
    """按 §5.1 签名实现的桩；可注入判定结果与「某判据抛异常」，并记录调用次数。"""

    def __init__(self, blob=None, inside=True, stacked=True, gone=True, raise_on=None,
                 refine=None, blob_seq=None):
        self.blob = blob if blob is not None else StubBlob()
        #: 按调用顺序逐个返回的 blob（测「top 失败换 overview」这类重试路径）
        self.blob_seq = list(blob_seq or [])
        self.inside = bool(inside)
        self.stacked = bool(stacked)
        self.gone = bool(gone)
        self.raise_on = set(raise_on or ())
        #: None 表示「不做精修」（refined=False）——与 §5.1 的「没精修就原样返回」一致
        self.refine_result = dict(refine) if refine is not None else {"refined": False}
        self.blob_calls: list = []
        self.inside_calls: list = []
        self.stacked_calls: list = []
        self.gone_calls: list = []
        self.refine_calls: list = []

    def refine_pixel(self, rgb, color, hint_uv, *, camera="top", env=None):
        self.refine_calls.append((color, tuple(hint_uv), camera))
        return dict(self.refine_result)

    def color_blob(self, rgb, color, hint_uv=None, *, radius_px=110, min_area_px=25):
        self.blob_calls.append((color, hint_uv))
        if self.blob_seq:
            return self.blob_seq.pop(0)
        return self.blob

    def check_inside(self, rgb, subject_color, container_bbox, *, min_area_px=25, shrink_px=None):
        if "inside" in self.raise_on:
            raise RuntimeError("check_inside boom")
        self.inside_calls.append((subject_color, tuple(container_bbox)))
        return {"ok": self.inside, "why": "红块质心在托盘框内" if self.inside else "红块不在托盘框内",
                "metrics": {"margin_px": 12.0}}

    def check_stacked(self, rgb, upper_color, lower_color, *, min_overlap=0.55, min_area_px=25):
        if "stacked" in self.raise_on:
            raise RuntimeError("check_stacked boom")
        self.stacked_calls.append((upper_color, lower_color))
        return {"ok": self.stacked, "why": "绿块叠在红块上" if self.stacked else "绿块不在红块上方",
                "metrics": {"overlap": 0.8}}

    def check_gone(self, rgb_before, rgb_after, color, hint_uv, *, radius_px=60, min_area_px=25):
        if "gone" in self.raise_on:
            raise RuntimeError("check_gone boom")
        self.gone_calls.append((color, tuple(hint_uv)))
        return {"ok": self.gone, "why": "原位置色块消失" if self.gone else "原位置色块还在",
                "metrics": {"area_ratio": 0.1}}


class PlanVLM:
    """返回 plan_subgoals 调用的假 VLM（subgoals 可以是 list 或 JSON 字符串）。"""

    def __init__(self, subgoals):
        self.subgoals = subgoals
        self.seen: list = []

    def decide(self, messages):
        self.seen.append(messages)
        return SkillCall("plan_subgoals", {"subgoals": self.subgoals}, reasoning="拆解任务")


class ReviewVLM:
    """复核用假 VLM：返回 subgoal_done（可带 confidence）或 ask_user。"""

    def __init__(self, skill="subgoal_done", confidence=None, note="看图上已经完成"):
        self.skill = skill
        self.confidence = confidence
        self.note = note
        self.seen: list = []

    def decide(self, messages):
        self.seen.append(messages)
        if self.skill == "subgoal_done":
            args = {"note": self.note}
            if self.confidence is not None:
                args["confidence"] = self.confidence
            return SkillCall("subgoal_done", args, reasoning=self.note)
        return SkillCall("ask_user", {"question": self.note}, reasoning=self.note)


class FakeLoop:
    """AgentLoop 的最小替身：记录 instruction，返回预设结果。"""

    def __init__(self, results=None, name="loop"):
        self.results = list(results or [])
        self.instructions: list[str] = []
        self.name = name

    def run(self, task, history=None):
        self.instructions.append(task)
        return list(self.results)


class LoopFactory:
    """每次调用造一个**新** FakeLoop（模拟「每个子目标一个新 epoch」）。"""

    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls: list = []
        self.loops: list[FakeLoop] = []

    def __call__(self, sg, attempt):
        self.calls.append((sg.id, attempt))
        loop = FakeLoop(self.results, name=f"{sg.id}-{attempt}")
        self.loops.append(loop)
        return loop


class ScriptedVerifier:
    """按脚本依次返回复核结论；记录每次收到的子目标。"""

    def __init__(self, verdicts=None, camera="top"):
        self.verdicts = list(verdicts or [])
        self.calls: list = []
        self.camera = camera

    def verify(self, sg, *, metrics=None, rgb_before=None):
        self.calls.append(sg)
        if self.verdicts:
            return dict(self.verdicts.pop(0))
        return {"done": False, "evidence": "none", "why": "脚本没有更多结论"}


class TestSubGoalPlanning(unittest.TestCase):

    def test_subgoal_contract_fields(self):
        sg = SubGoal(id="g1", instruction="i", goal_type="pick", subject="s")
        for name in ("id", "instruction", "goal_type", "subject", "subject_color",
                     "container", "container_color", "anchor_color", "verify_hint"):
            self.assertTrue(hasattr(sg, name), f"SubGoal 缺字段 {name}")
        self.assertEqual(SUBGOAL_MAX_ATTEMPTS, 3)
        json.dumps(sg.to_json(), ensure_ascii=False)          # 必须可 JSON 序列化

    def test_canonical_color(self):
        self.assertEqual(canonical_color("红色"), "red")
        self.assertEqual(canonical_color("浅蓝"), "cyan")      # 长词优先，不能判成 blue
        self.assertEqual(canonical_color("cyan"), "cyan")
        self.assertIsNone(canonical_color("那个东西"))
        self.assertIsNone(canonical_color(None))

    def test_fallback_single_subgoal_instruction_is_task(self):
        sgs = fallback_subgoals("把红色方块放进托盘")
        self.assertEqual(len(sgs), 1)
        self.assertEqual(sgs[0].id, "g1")
        self.assertEqual(sgs[0].instruction, "把红色方块放进托盘")
        self.assertEqual(sgs[0].goal_type, "place_in_container")
        self.assertEqual(sgs[0].subject_color, "red")
        self.assertEqual(sgs[0].container, "托盘")
        self.assertTrue(sgs[0].verify_hint)

    def test_fallback_infers_stack_and_pick(self):
        sgs = fallback_subgoals("把绿色方块叠到红色方块上")
        self.assertEqual(sgs[0].goal_type, "place_on_top")
        self.assertEqual(sgs[0].subject_color, "green")
        self.assertEqual(sgs[0].anchor_color, "red")
        self.assertEqual(fallback_subgoals("抓起蓝色方块")[0].goal_type, "pick")
        self.assertEqual(fallback_subgoals("随便弄一下")[0].goal_type, "free")

    def test_plan_subgoals_uses_vlm_tool(self):
        vlm = PlanVLM([
            {"instruction": "把红色方块放进托盘", "subject": "红色方块",
             "goal_type": "place_in_container", "subject_color": "红",
             "container": "托盘", "container_color": "gray"},
            {"instruction": "把绿色方块叠到红色方块上", "subject": "绿色方块",
             "goal_type": "place_on_top", "subject_color": "green", "anchor_color": "红"},
        ])
        sgs = plan_subgoals(vlm, "把红块放进托盘，绿块叠上去", make_state(),
                            {"overview": make_image()})
        self.assertEqual([s.id for s in sgs], ["g1", "g2"])
        self.assertEqual(sgs[0].goal_type, "place_in_container")
        self.assertEqual(sgs[0].subject_color, "red")          # 「红」被规范化
        self.assertEqual(sgs[0].container, "托盘")
        self.assertEqual(sgs[0].container_color, "gray")
        self.assertEqual(sgs[1].goal_type, "place_on_top")
        self.assertEqual(sgs[1].anchor_color, "red")
        self.assertTrue(vlm.seen, "必须真的问过 VLM")
        self.assertIn("plan_subgoals", json.dumps(vlm.seen[-1], ensure_ascii=False))

    def test_plan_subgoals_caps_and_skips_invalid(self):
        vlm = PlanVLM([
            {"instruction": "", "goal_type": "pick"},            # 缺 instruction -> 丢
            {"instruction": "A", "goal_type": "bogus_type"},     # 非法类型 -> 降级 free
            {"instruction": "B", "goal_type": "pick"},
            {"instruction": "C", "goal_type": "pick"},
            {"instruction": "D", "goal_type": "pick"},
        ])
        sgs = plan_subgoals(vlm, "t", make_state(), {}, max_subgoals=3)
        self.assertEqual([s.instruction for s in sgs], ["A", "B", "C"])
        self.assertEqual(sgs[0].goal_type, "free")
        self.assertEqual([s.id for s in sgs], ["g1", "g2", "g3"])

    def test_plan_subgoals_accepts_json_string(self):
        vlm = PlanVLM(json.dumps([{"instruction": "把红块放进托盘",
                                   "goal_type": "place_in_container",
                                   "subject_color": "red", "container": "托盘"}]))
        sgs = plan_subgoals(vlm, "t", make_state(), {})
        self.assertEqual(sgs[0].goal_type, "place_in_container")
        self.assertEqual(sgs[0].subject_color, "red")

    def test_plan_subgoals_falls_back_without_plan_tool(self):
        # MockVLM 不会调 plan_subgoals -> 回退单条，oracle 路径行为不变
        sgs = plan_subgoals(MockVLM(), "把红色方块放进托盘", make_state(),
                            {"overview": make_image()})
        self.assertEqual(len(sgs), 1)
        self.assertIsInstance(sgs[0], SubGoal)
        self.assertEqual(sgs[0].instruction, "把红色方块放进托盘")

    def test_plan_subgoals_falls_back_on_exception_and_none(self):
        class Boom:
            def decide(self, messages):
                raise RuntimeError("network down")

        self.assertEqual(plan_subgoals(Boom(), "抓红色方块", make_state(), {})[0].goal_type, "pick")
        self.assertEqual(plan_subgoals(None, "抓红色方块", make_state(), {})[0].goal_type, "pick")
        self.assertEqual(len(plan_subgoals(MockVLM(), "抓红色方块", None, {})), 1)


class TestSubGoalVerifier(unittest.TestCase):

    def setUp(self):
        clear_pixel_cache()      # 共享像素缓存是模块级的，测试之间必须隔离

    def _inside_sg(self):
        return SubGoal(id="g1", instruction="把红色方块放进托盘", goal_type="place_in_container",
                       subject="红色方块", subject_color="red", container="托盘",
                       container_color="gray")

    def _pick_sg(self):
        return SubGoal(id="g1", instruction="抓起红色方块", goal_type="pick",
                       subject="红色方块", subject_color="red")

    def test_locate_uses_color_blob_and_caches(self):
        sp = StubPerception()
        v = SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)
        r1 = v.locate("红色方块", color="red", camera="top")
        self.assertTrue(r1["ok"], r1)
        self.assertEqual(r1["pixel"], [100.0, 120.0])
        self.assertEqual(r1["bbox"], [80.0, 100.0, 120.0, 140.0])
        self.assertEqual(r1["pixel_top"], [100.0, 120.0])
        self.assertEqual(r1["source"], "color_blob")
        r2 = v.locate("红色方块", color="red", camera="top")
        self.assertEqual(r1, r2)
        self.assertEqual(len(sp.blob_calls), 1, "同一 (target,color,camera) 必须命中缓存")
        self.assertEqual(v.locate_calls, 1)
        # §5.10：默认 camera='both' -> 同时跑 top + overview（缓存键含相机，互不干扰）
        both = v.locate("红色方块", color="red")
        self.assertEqual(both["camera"], "both")
        self.assertEqual(both["pixel_top"], [100.0, 120.0])
        self.assertEqual(both["pixel_overview"], [100.0, 120.0])
        self.assertEqual(both["bbox_top"], [80.0, 100.0, 120.0, 140.0])
        self.assertEqual(both["bbox_overview"], [80.0, 100.0, 120.0, 140.0])
        self.assertEqual(both["pixel"], [100.0, 120.0], "兼容别名优先 top")
        self.assertEqual(len(sp.blob_calls), 3, "both 会各跑一次 top/overview 分割")
        self.assertEqual(v.locate_calls, 2)

    def test_locate_refines_pixel_with_refine_pixel(self):
        sp = StubPerception(refine={"refined": True, "uv": [101.5, 119.5],
                                    "bbox": [81, 101, 121, 141], "source": "color_blob",
                                    "delta_px": 1.58})
        v = SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)
        r = v.locate("红色方块", color="red", camera="top")
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["refined"], "已知颜色时必须用 refine_pixel 精修")
        self.assertEqual(r["pixel_top"], [101.5, 119.5])
        self.assertEqual(r["bbox_top"], [81.0, 101.0, 121.0, 141.0])
        self.assertEqual(r["delta_px"], 1.58)
        self.assertEqual(r["pixel"], [101.5, 119.5], "兼容别名也应是精修后的点")
        self.assertEqual(sp.refine_calls[0][0], "red")
        self.assertEqual(sp.refine_calls[0][1], (100.0, 120.0), "精修 hint = color_blob 质心")
        self.assertEqual(sp.refine_calls[0][2], "top")

    def test_locate_keeps_raw_pixel_when_refine_not_refined(self):
        sp = StubPerception(refine={"refined": False, "uv": [999.0, 999.0], "delta_px": 40.0})
        v = SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)
        r = v.locate("红色方块", color="red", camera="top")
        self.assertFalse(r["refined"])
        self.assertEqual(r["pixel_top"], [100.0, 120.0], "refined=False 必须原样返回 hint_uv")
        self.assertEqual(r["pixel"], [100.0, 120.0])
        self.assertEqual(r["source"], "color_blob")

    def test_locate_gray_not_found_is_honest(self):
        # gray/white 颜色规则在修（误检可达几十 cm）——找不到时必须诚实失败，不给错点
        sp = StubPerception(blob=StubBlob(found=False))
        v = SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)
        r = v.locate("托盘", color="gray", camera="both")
        self.assertFalse(r["ok"], "找不到 gray 色块不能返回错点")
        self.assertIsNone(r["pixel"])
        self.assertIsNone(r["pixel_top"])
        self.assertIsNone(r["pixel_overview"])
        self.assertIn("gray", r["why"])

    def test_locate_without_perception_or_color_is_honest(self):
        v = SubGoalVerifier(StubEnv(), perception=None, vlm=None, log=None)
        v._perception_resolved = True                 # 固定成「后端不可用」，与并行开发进度解耦
        v.perception = None
        r = v.locate("红色方块", color="red")
        self.assertFalse(r["ok"])
        self.assertIsNone(r["pixel"])
        self.assertIn("perception", r["why"])

        sp = StubPerception()
        v2 = SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)
        r2 = v2.locate("那个东西", color=None)
        self.assertFalse(r2["ok"])
        self.assertIn("颜色", r2["why"])
        self.assertEqual(sp.blob_calls, [], "没有颜色线索时不该瞎分割")

    def test_place_in_container_geometry_ok(self):
        sp = StubPerception(inside=True)
        v = SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)
        res = v.verify(self._inside_sg(), metrics={}, rgb_before=None)
        self.assertTrue(res["done"], res)
        self.assertEqual(res["evidence"], "geometry")
        self.assertIn("框内", res["why"])
        self.assertEqual(sp.inside_calls[0][0], "red")
        self.assertEqual(sp.inside_calls[0][1], (80.0, 100.0, 120.0, 140.0))

    def test_place_in_container_fail_without_vlm_is_not_done(self):
        sp = StubPerception(inside=False)
        v = SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)
        res = v.verify(self._inside_sg(), metrics={}, rgb_before=None)
        self.assertFalse(res["done"], "几何失败且无 VLM 复核时绝不能报完成")
        self.assertEqual(res["evidence"], "none")
        self.assertIn("不在托盘框内", res["why"])

    def test_geometry_fail_can_be_overridden_by_vlm_review(self):
        sp = StubPerception(inside=False)
        v = SubGoalVerifier(StubEnv(), perception=sp, vlm=ReviewVLM("subgoal_done"), log=None)
        res = v.verify(self._inside_sg())
        self.assertTrue(res["done"])
        self.assertEqual(res["evidence"], "vlm")
        self.assertGreaterEqual(res["confidence"], 0.6)

    def test_vlm_low_confidence_is_not_done(self):
        v = SubGoalVerifier(StubEnv(), perception=StubPerception(inside=False),
                            vlm=ReviewVLM("subgoal_done", confidence=0.4), log=None)
        res = v.verify(self._inside_sg())
        self.assertFalse(res["done"], "confidence < 0.6 不算完成")
        self.assertEqual(res["confidence"], 0.4)

    def test_vlm_says_not_done_is_not_done(self):
        v = SubGoalVerifier(StubEnv(), perception=StubPerception(inside=False),
                            vlm=ReviewVLM("ask_user", note="红块还在桌上"), log=None)
        res = v.verify(self._inside_sg())
        self.assertFalse(res["done"])
        self.assertIn("还在桌上", res["why"])

    def test_place_on_top_stacked_judgement(self):
        sp = StubPerception(stacked=True)
        v = SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)
        sg = SubGoal(id="g2", instruction="把绿色方块叠到红色方块上", goal_type="place_on_top",
                     subject="绿色方块", subject_color="green", anchor_color="red")
        res = v.verify(sg)
        self.assertTrue(res["done"], res)
        self.assertEqual(res["evidence"], "geometry")
        self.assertEqual(sp.stacked_calls[0], ("green", "red"))
        # 缺 anchor_color -> 判不了，又没 VLM -> 不谎报
        sg2 = SubGoal(id="g3", instruction="叠上去", goal_type="place_on_top",
                      subject="绿块", subject_color="green")
        self.assertFalse(v.verify(sg2)["done"])

    def test_pick_metrics_take_priority(self):
        sp = StubPerception(gone=False)
        v = SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)
        sg = self._pick_sg()
        self.assertTrue(v.verify(sg, metrics={"holding": True})["done"])
        gap = v.verify(sg, metrics={"grasp_gap_mm": 8.0, "grasp_closed_mm": 4.0})
        self.assertTrue(gap["done"], gap)
        self.assertIn("指间有物", gap["why"])
        self.assertFalse(v.verify(sg, metrics={"holding": False})["done"])
        self.assertEqual(sp.gone_calls, [], "metrics 有结论时不该再走 check_gone")

    def test_pick_falls_back_to_check_gone(self):
        sp = StubPerception(gone=True)
        v = SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)
        sg = self._pick_sg()
        res = v.verify(sg, metrics={}, rgb_before=make_image())
        self.assertTrue(res["done"], res)
        self.assertEqual(res["evidence"], "geometry")
        self.assertEqual(len(sp.gone_calls), 1)
        self.assertFalse(v.verify(sg, metrics={}, rgb_before=None)["done"], "没有抓前图像不谎报")

    def test_free_goal_without_vlm_never_reports_success(self):
        v = SubGoalVerifier(StubEnv(), perception=StubPerception(), vlm=None, log=None)
        sg = SubGoal(id="g1", instruction="随便弄一下", goal_type="free", subject="目标")
        res = v.verify(sg)
        self.assertFalse(res["done"])
        self.assertEqual(res["evidence"], "none")
        self.assertTrue(res["why"])

    def test_perception_exceptions_are_contained(self):
        sp = StubPerception(raise_on={"inside", "gone"})
        v = SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)
        res = v.verify(self._inside_sg())
        self.assertFalse(res["done"], "感知异常 -> done=False，绝不抛")
        self.assertIn("异常", res["why"])
        res2 = v.verify(self._pick_sg(), metrics={}, rgb_before=make_image())
        self.assertFalse(res2["done"])

    def test_verify_never_raises_on_broken_input(self):
        v = SubGoalVerifier(StubEnv(), perception=StubPerception(), vlm=None, log=None)
        self.assertFalse(v.verify("把红色方块放进托盘")["done"])       # 纯文本也接
        weird = SubGoal(id="g1", instruction="x", goal_type="不存在的类型", subject="x")
        self.assertFalse(v.verify(weird)["done"])
        # 坏 metrics 只影响「证据」不影响几何判定，但绝不允许抛异常
        res = v.verify(self._inside_sg(), metrics="not-a-dict")
        self.assertIsInstance(res, dict)
        self.assertTrue(res["done"], "几何证据足够时，坏 metrics 不该把结论抹掉")

    def test_review_prompt_does_not_leak_truth(self):
        captured = {}

        class PromptSpy:
            def decide(self, messages):
                content = messages[-1].get("content")
                captured["text"] = content[0]["text"] if isinstance(content, list) else str(content)
                return SkillCall("subgoal_done", {"note": "ok"})

        v = SubGoalVerifier(StubEnv(), perception=StubPerception(inside=True), vlm=PromptSpy(), log=None)
        sg = SubGoal(id="g1", instruction="随便", goal_type="free", subject="x")
        v.verify(sg, metrics={"object_pose": [0.4, 0.1, 0.42], "holding": True,
                              "grasp_gap_mm": 6.5})
        self.assertNotIn("0.42", captured["text"], "复核 prompt 不得泄漏真值位姿（§5.7）")
        self.assertNotIn("object_pose", captured["text"])
        self.assertIn("holding", captured["text"])
        self.assertIn("grasp_gap_mm", captured["text"])


class TestSubGoalOrchestrator(unittest.TestCase):

    def setUp(self):
        clear_pixel_cache()

    def _sg(self):
        return SubGoal(id="g1", instruction="把绿色方块叠到红色方块上", goal_type="place_on_top",
                       subject="绿色方块", subject_color="green", anchor_color="red")

    def test_two_failures_then_success_reports_retry_success(self):
        verifier = ScriptedVerifier([
            {"done": False, "evidence": "geometry", "why": "绿块不在红块上方"},
            {"done": False, "evidence": "geometry", "why": "绿块还是没叠上"},
            {"done": True, "evidence": "geometry", "why": "绿块已叠在红块上"},
        ])
        factory = LoopFactory([SkillResult(ok=True, skill="place", message="stub")])
        logs: list[str] = []
        orch = SubGoalOrchestrator(StubEnv(), MockVLM(), factory, verifier, log=logs.append)
        records = orch.run("把绿色方块叠到红色方块上", [self._sg()])

        self.assertEqual(len(records), 1)
        rec = records[0]
        for key in ("subgoal", "attempts", "done", "evidence", "why"):
            self.assertIn(key, rec)
        self.assertTrue(rec["done"])
        self.assertEqual(rec["attempts"], 3)
        self.assertEqual(rec["evidence"], "geometry")
        self.assertIn("成功（重试后）", rec["why"])
        self.assertEqual([a for _i, a in factory.calls], [1, 2, 3])
        self.assertEqual(len({id(loop) for loop in factory.loops}), 3, "每次尝试必须是新 epoch")
        self.assertIn("上一次复核未通过", factory.loops[1].instructions[0])
        self.assertTrue(any("成功（重试后" in m for m in logs), logs)

    def test_all_three_failures_report_subgoal_failed(self):
        verifier = ScriptedVerifier([{"done": False, "evidence": "geometry", "why": f"第 {i} 次没过"}
                                     for i in range(3)])
        factory = LoopFactory()
        logs: list[str] = []
        orch = SubGoalOrchestrator(StubEnv(), MockVLM(), factory, verifier, log=logs.append)
        rec = orch.run("把绿色方块叠到红色方块上", [self._sg()])[0]

        self.assertFalse(rec["done"], "三次全败绝不能报成功")
        self.assertEqual(rec["attempts"], 3)
        self.assertEqual(rec["evidence"], "none")
        self.assertIn("subgoal_failed", rec["why"])
        self.assertNotIn("成功", rec["why"])
        self.assertTrue(any("subgoal_failed" in m for m in logs), logs)
        self.assertEqual(len(factory.loops), 3)

    def test_verifier_exception_is_never_success(self):
        class BoomVerifier:
            camera = "top"

            def verify(self, sg, *, metrics=None, rgb_before=None):
                raise RuntimeError("verifier exploded")

        orch = SubGoalOrchestrator(StubEnv(), MockVLM(), LoopFactory(), BoomVerifier(), log=None)
        rec = orch.run("t", [self._sg()])[0]
        self.assertFalse(rec["done"])
        self.assertIn("subgoal_failed", rec["why"])

    def test_loop_factory_exception_is_never_success(self):
        def bad_factory(sg, attempt):
            raise RuntimeError("no sim")

        orch = SubGoalOrchestrator(StubEnv(), MockVLM(), bad_factory, ScriptedVerifier(), log=None)
        rec = orch.run("t", [self._sg()])[0]
        self.assertFalse(rec["done"])
        self.assertIn("subgoal_failed", rec["why"])

    def test_run_without_subgoals_falls_back_for_mock_vlm(self):
        verifier = ScriptedVerifier()
        factory = LoopFactory()
        orch = SubGoalOrchestrator(StubEnv(), MockVLM(), factory, verifier, log=None)
        records = orch.run("把红色方块放进托盘")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["subgoal"].instruction, "把红色方块放进托盘")
        self.assertFalse(records[0]["done"], "无证据（MockVLM 不规划 + 无复核结论）不得报成功")
        self.assertEqual(len(verifier.calls), 3)

    def test_run_uses_vlm_planning_when_available(self):
        verifier = ScriptedVerifier([
            {"done": True, "evidence": "vlm", "confidence": 0.9, "why": "图上完成"},
            {"done": True, "evidence": "vlm", "confidence": 0.9, "why": "图上完成"},
        ])
        vlm = PlanVLM([{"instruction": "把红块放进托盘", "goal_type": "place_in_container",
                        "subject_color": "red", "container": "托盘", "container_color": "gray"},
                       {"instruction": "把绿块叠到红块上", "goal_type": "place_on_top",
                        "subject_color": "green", "anchor_color": "red"}])
        factory = LoopFactory()
        orch = SubGoalOrchestrator(StubEnv(), vlm, factory, verifier, log=None)
        records = orch.run("两步任务", state=make_state(), images={"overview": make_image()})
        self.assertEqual([r["subgoal"].id for r in records], ["g1", "g2"])
        self.assertTrue(all(r["done"] for r in records))
        self.assertEqual([a for _i, a in factory.calls], [1, 1], "一次通过就不该重试")
        self.assertEqual(factory.loops[0].instructions[0], "把红块放进托盘", "子目标 instruction 直接喂 AgentLoop")

    def test_loop_factory_accepts_fewer_args(self):
        seen: list = []

        def factory(sg):
            seen.append(sg.id)
            return FakeLoop()

        verifier = ScriptedVerifier([{"done": True, "evidence": "geometry", "why": "ok"}])
        orch = SubGoalOrchestrator(StubEnv(), MockVLM(), factory, verifier, log=None)
        rec = orch.run("t", [self._sg()])[0]
        self.assertTrue(rec["done"])
        self.assertEqual(seen, ["g1"])


# --------------------------------------------------------------------------- #
# 14) §5.12 参数冗余要纠正不要拒绝（实机诊断驱动）
# --------------------------------------------------------------------------- #
class TestParamCorrections(unittest.TestCase):
    """§5.12：冗余但一致的参数纠正后继续，只有互相矛盾才拒绝；纠正要落进 metrics。"""

    def setUp(self):
        clear_pixel_cache()

    def test_pick_redundant_target_id_dropped_for_pixels(self):
        st = make_state()
        call = SkillCall("pick", {"target_id": "box_red", "target_pixel": [267, 125]})
        ok, msg = validate_skill_call(call, st)
        self.assertTrue(ok, msg)
        self.assertNotIn("target_id", call.args, "无真值模式以像素为准")
        self.assertEqual(call.args["target_pixel"], [267, 125])
        fixes = take_corrections(call)
        self.assertEqual([f["key"] for f in fixes], ["target_id"])
        self.assertEqual(fixes[0]["action"], "dropped")
        self.assertIn("冗余 target_id 已忽略（无真值模式以像素为准）", fixes[0]["log"])
        # target_xy 冗余同样丢掉（像素更具体）；只给 overview 像素也允许
        call2 = SkillCall("pick", {"target_xy": [0.3, 0.1], "target_pixel": [267, 125]})
        self.assertTrue(validate_skill_call(call2, st)[0])
        self.assertNotIn("target_xy", call2.args)
        self.assertEqual([f["key"] for f in take_corrections(call2)], ["target_xy"])
        call3 = SkillCall("pick", {"target_id": "box_red", "target_pixel_overview": [300, 260]})
        self.assertTrue(validate_skill_call(call3, st)[0])
        self.assertNotIn("target_id", call3.args)

    def test_pick_oracle_xy_consistency_30mm(self):
        st = make_state()                      # box_red 真值 [0.40, 0.10]
        good = SkillCall("pick", {"target_id": "box_red", "target_xy": [0.405, 0.095]})   # 7 mm
        ok, msg = validate_skill_call(good, st)
        self.assertTrue(ok, msg)
        self.assertNotIn("target_id", good.args)
        self.assertEqual(good.args["target_xy"], [0.405, 0.095])
        self.assertIn("一致", take_corrections(good)[0]["detail"])
        bad = SkillCall("pick", {"target_id": "box_red", "target_xy": [0.30, 0.30]})      # 210 mm
        ok, msg = validate_skill_call(bad, st)
        self.assertFalse(ok)
        self.assertIn("30 mm", msg)
        self.assertIn("矛盾", msg)
        # 只给 target_xy（无 id）不受真值校验约束
        self.assertTrue(validate_skill_call(SkillCall("pick", {"target_xy": [0.30, 0.30]}), st)[0])

    def test_place_xy_inside_container_accepted_outside_rejected(self):
        st = make_state()                      # tray 中心 [0.42, 0.02]，半宽 0.12
        inside = SkillCall("place", {"container_id": "tray", "xy": [0.40, 0.10]})
        ok, msg = validate_skill_call(inside, st)
        self.assertTrue(ok, msg)
        self.assertEqual(inside.args["container_id"], "tray", "container 仍决定高度/越壁")
        self.assertEqual(inside.args["xy"], [0.40, 0.10])
        fixes = take_corrections(inside)
        self.assertEqual([f["key"] for f in fixes], ["xy"])
        self.assertEqual(fixes[0]["action"], "accepted")
        outside = SkillCall("place", {"container_id": "tray", "xy": [0.05, 0.90]})
        ok, msg = validate_skill_call(outside, st)
        self.assertFalse(ok)
        self.assertIn("不在容器", msg)
        self.assertIn("矛盾", msg)
        # 未反投影的落点像素：validate 阶段判不了内外 -> 放行，交给 AgentLoop 反投影后再校验
        pixel_only = SkillCall("place", {"container_id": "tray", "target_pixel": [228, 422]})
        self.assertTrue(validate_skill_call(pixel_only, st)[0])

    def test_true_contradictions_still_rejected(self):
        st = make_state()
        for args in ({"surface": "table", "container_id": "tray"},
                     {"on_top_of": "box_red", "container_id": "tray"},
                     {"surface": "table", "on_top_of": "box_red"},
                     {"anchor_id": "box_red", "relation": "right", "container_id": "tray"}):
            ok, msg = validate_skill_call(SkillCall("place", dict(args)), st)
            self.assertFalse(ok, f"{args} 是真矛盾，必须拒绝")
            self.assertIn("四选一", msg)

    def test_loop_attaches_corrected_args_to_metrics(self):
        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [267, 125]}),
                           SkillCall("stop", {})])
        logs: list[str] = []
        loop = AgentLoop(StubEnv(), ex, vlm, cameras=(), max_steps=4, log=logs.append,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        results = loop.run("抓红色方块")
        self.assertEqual(ex.calls[0].skill, "pick")
        self.assertNotIn("target_id", ex.calls[0].args, "执行器拿到的是纠正后的调用")
        self.assertEqual(ex.calls[0].args.get("target_xy"), [0.4, 0.1])
        fixes = results[0].metrics.get("corrected_args")
        self.assertTrue(fixes, results[0].metrics)
        self.assertEqual(fixes[0]["key"], "target_id")
        self.assertTrue(any("冗余 target_id 已忽略（无真值模式以像素为准）" in m for m in logs), logs)
        self.assertEqual([c.skill for c in ex.calls], ["pick", "stop"])

    def test_loop_place_inside_container_records_xy_correction(self):
        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_xy": [0.40, 0.10]}),
                           SkillCall("place", {"container_id": "tray", "xy": [0.40, 0.10]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(StubEnv(), ex, vlm, cameras=(), max_steps=5, log=None)
        results = loop.run("把红色方块放进托盘")
        self.assertEqual([c.skill for c in ex.calls], ["pick", "place", "stop"])
        fixes = results[1].metrics.get("corrected_args") or []
        self.assertEqual(fixes[0]["key"], "xy")
        self.assertEqual(fixes[0]["action"], "accepted")


# --------------------------------------------------------------------------- #
# 15) §5.10 双目三角定位 + §5.11 复核必须走 function calling
# --------------------------------------------------------------------------- #
class ToolReviewVLM:
    """复核假 VLM：返回 verify_subgoal 工具调用（§5.11 首选路径）。"""

    def __init__(self, done=True, confidence=0.8, why="图上看已完成"):
        self.done = done
        self.confidence = confidence
        self.why = why
        self.seen: list = []

    def decide(self, messages):
        self.seen.append(messages)
        return SkillCall("verify_subgoal",
                         {"done": self.done, "confidence": self.confidence, "why": self.why})


class EmptyVLM:
    """模拟 §5.11 的截断回复：没有工具调用，文本也是空的。"""

    def decide(self, messages):
        return SkillCall("stop", {}, reasoning="")


class TestDualPixelAndToolReview(unittest.TestCase):
    """§5.10/§5.11：双像素 tool 字段、locate both、verify_subgoal 与提示词。"""

    def setUp(self):
        clear_pixel_cache()

    def test_tools_registered(self):
        for name in ("pick", "place"):
            props = _tool_by_name(name)["function"]["parameters"]["properties"]
            self.assertIn("target_pixel_overview", props, f"{name} 缺 target_pixel_overview")
        cam = _tool_by_name("locate")["function"]["parameters"]["properties"]["camera"]
        self.assertEqual(cam["oneOf"][0]["enum"], list(LOCATE_CAMERAS))
        self.assertIn("both", cam["oneOf"][0]["enum"], "locate 必须支持 both（双目）")
        vs = _tool_by_name("verify_subgoal")["function"]["parameters"]
        self.assertEqual(vs["required"], ["done", "confidence", "why"])
        self.assertIn("verify_subgoal", SKILL_NAMES)
        # required 仍向后兼容
        self.assertEqual(_tool_by_name("pick")["function"]["parameters"]["required"], ["target_id"])

    def test_validate_dual_pixel_and_review_tool(self):
        st = make_state()
        # 只给 overview 像素也允许（§5.10）
        self.assertTrue(validate_skill_call(SkillCall("pick", {"target_pixel_overview": [300, 260]}), st)[0])
        self.assertTrue(validate_skill_call(
            SkillCall("pick", {"target_pixel": [1, 2], "target_pixel_overview": [3, 4]}), st)[0])
        ok, msg = validate_skill_call(SkillCall("pick", {"target_pixel_overview": [1]}), st)
        self.assertFalse(ok)
        self.assertIn("target_pixel_overview", msg)
        ok, msg = validate_skill_call(
            SkillCall("place", {"container_id": "tray", "target_pixel_overview": "x"}), st)
        self.assertFalse(ok)
        self.assertIn("target_pixel_overview", msg)
        # locate 的 both
        self.assertTrue(validate_skill_call(SkillCall("locate", {"target": "水瓶", "camera": "both"}), st)[0])
        ok, msg = validate_skill_call(SkillCall("locate", {"target": "水瓶", "camera": "side"}), st)
        self.assertFalse(ok)
        self.assertIn("camera", msg)
        # verify_subgoal 执行循环里必须 invalid_call（模型不能自己发「完成」判定）
        ok, msg = validate_skill_call(
            SkillCall("verify_subgoal", {"done": True, "confidence": 0.9, "why": "ok"}), st)
        self.assertFalse(ok)
        self.assertIn("复核", msg)
        ok, msg = validate_skill_call(
            SkillCall("verify_subgoal", {"done": "yes", "confidence": 0.9, "why": "ok"}), st)
        self.assertFalse(ok)
        self.assertIn("布尔", msg)
        ex = RecordingExecutor()
        loop = AgentLoop(StubEnv(), ex, ScriptedVLM([
            SkillCall("verify_subgoal", {"done": True, "confidence": 0.9, "why": "ok"}),
            SkillCall("stop", {})]), max_steps=4, log=None)
        results = loop.run("t")
        self.assertEqual(results[0].error_code, "invalid_call")
        self.assertEqual([c.skill for c in ex.calls], ["stop"])

    def test_no_oracle_prompt_has_parallax_guidance(self):
        p = system_prompt(include_poses=False)
        for needle in ("视差", "target_pixel_overview", "双目", "7 cm"):
            self.assertIn(needle, p, f"无真值提示词缺少 {needle!r}")
        self.assertNotIn("视差", system_prompt(include_poses=True), "oracle 提示词不得被改动")

    def test_no_oracle_prompt_defaults_to_top_grasp(self):
        # 真机 benchmark：模型给圆柱/水瓶选了 side（场景没 10cm 侧向净空）-> 全败。
        # 提示词必须明确「默认 top，只有明确要求 + 周围空旷才 side」。
        p = system_prompt(include_poses=False)
        for needle in ("默认一律用 top", "净空", "93 mm", "推走"):
            self.assertIn(needle, p, f"无真值提示词缺少抓取档位规则 {needle!r}")
        desc = _tool_by_name("pick")["function"]["parameters"]["properties"]["grasp_profile"]["description"]
        self.assertIn("默认", desc)
        self.assertIn("净空", desc)
        self.assertNotIn("水瓶/圆柱建议 side", desc, "旧说法（建议 side）必须删掉")
        self.assertNotIn("默认一律用 top", system_prompt(include_poses=True), "oracle 提示词不变")

    def test_loop_resolves_dual_pixel_with_triangulation(self):
        seen: list = []

        def resolver(u, v, camera, overview_uv=None):
            seen.append((u, v, camera, overview_uv))
            return {"ok": True, "xyz": [0.40, 0.10, 0.43], "source": "triangulate",
                    "residual_mm": 2.5}

        ex, logs = RecordingExecutor(), []
        vlm = ScriptedVLM([SkillCall("pick", {"target_pixel": [267, 125],
                                              "target_pixel_overview": [300, 260]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(StubEnv(), ex, vlm, cameras=(), max_steps=4, log=logs.append,
                         pixel_resolver=resolver)
        results = loop.run("抓高物体")
        self.assertEqual(seen, [(267.0, 125.0, "top", [300.0, 260.0])], "双目像素要一次交给解析器")
        self.assertEqual(ex.calls[0].args.get("target_xy"), [0.4, 0.1])
        self.assertNotIn("target_pixel_overview", ex.calls[0].args)
        self.assertEqual(results[0].metrics.get("source"), "triangulate")
        self.assertEqual(results[0].metrics.get("residual_mm"), 2.5)
        self.assertTrue(any("双目像素" in m and "source=triangulate" in m for m in logs), logs)

    def test_loop_passes_overview_uv_keyword_to_demo_style_resolver(self):
        # demo/interactive._pixel_resolver 的形状：(u, v, camera="top", *extra, **kw)
        seen: list = []

        def resolver(u, v, camera="top", *extra, **kw):
            seen.append((u, v, camera, extra, kw))
            return np.array([0.40, 0.10, 0.43])

        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_pixel": [267, 125],
                                              "target_pixel_overview": [300, 260]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(StubEnv(), ex, vlm, cameras=(), max_steps=4, log=None,
                         pixel_resolver=resolver)
        results = loop.run("抓高物体")
        self.assertEqual(len(seen), 1)
        u, v, cam, extra, kw = seen[0]
        self.assertEqual((u, v, cam), (267.0, 125.0, "top"))
        self.assertEqual(extra, ())
        self.assertEqual(kw.get("overview_uv"), [300.0, 260.0],
                         "§5.10：第二个像素必须用 overview_uv 关键字传进去")
        self.assertEqual(results[0].metrics.get("source"), "triangulate")

    def test_loop_single_view_fallback_when_resolver_lacks_dual(self):
        seen: list = []

        def resolver(u, v, camera):                    # 只有 §4.3 的三参签名
            seen.append((u, v, camera))
            return [0.4, 0.1, 0.42]

        logs: list = []
        vlm = ScriptedVLM([SkillCall("pick", {"target_pixel": [267, 125],
                                              "target_pixel_overview": [300, 260]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(StubEnv(), RecordingExecutor(), vlm, cameras=(), max_steps=4,
                         log=logs.append, pixel_resolver=resolver)
        loop.run("抓高物体")
        self.assertEqual(seen, [(267.0, 125.0, "top")], "解析器不支持双目就退回单目")
        self.assertTrue(any("single_view" in m for m in logs), logs)

    def test_loop_untrusted_triangulation_is_retryable(self):
        def resolver(u, v, camera, overview_uv=None):
            return {"ok": False, "why": "双目三角残差过大", "residual_mm": 22.0,
                    "source": "triangulate"}

        loop = AgentLoop(StubEnv(), RecordingExecutor(),
                         ScriptedVLM([SkillCall("pick", {"target_pixel": [10, 20],
                                                         "target_pixel_overview": [30, 40]}),
                                      SkillCall("stop", {})]),
                         cameras=(), max_steps=3, log=None, pixel_resolver=resolver)
        res = loop.run("抓高物体")[0]
        self.assertEqual(res.error_code, "perception_error", "residual>15mm 必须可重试")
        self.assertIn("残差", res.message)
        self.assertIn("residual=22.0 mm", res.message)
        self.assertEqual(res.metrics.get("residual_mm"), 22.0)

    def test_loop_overview_only_pixel_uses_overview_camera(self):
        seen: list = []

        def resolver(u, v, camera):
            seen.append((u, v, camera))
            return [0.45, -0.18, 0.5]

        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_pixel_overview": [300, 260],
                                              "target_z_hint": 0.45}),
                           SkillCall("stop", {})])
        loop = AgentLoop(StubEnv(), ex, vlm, cameras=(), max_steps=4, log=None,
                         pixel_resolver=resolver)
        loop.run("抓水瓶")
        self.assertEqual(seen, [(300.0, 260.0, "overview")])
        self.assertEqual(ex.calls[0].args.get("target_xy"), [0.45, -0.18])

    def test_loop_locate_defaults_to_both_and_carries_four_keys(self):
        loc = StubLocator({"ok": True, "pixel_top": [100, 120], "pixel_overview": [110, 130],
                           "bbox_top": [80, 100, 120, 140], "bbox_overview": [90, 110, 130, 150],
                           "source": "color_blob", "why": "颜色分割在 top+overview 找到 cyan 色块"})
        vlm = ScriptedVLM([SkillCall("locate", {"target": "水瓶", "color": "cyan"}),
                           SkillCall("stop", {})])
        loop = AgentLoop(StubEnv(), RecordingExecutor(), vlm, max_steps=4, log=None, locator=loc)
        results = loop.run("标定水瓶")
        self.assertEqual(loc.calls, [("水瓶", "cyan", "both")], "locate 默认 both（§5.10）")
        m = results[0].metrics
        self.assertEqual(m["pixel_top"], [100.0, 120.0])
        self.assertEqual(m["pixel_overview"], [110.0, 130.0])
        self.assertEqual(m["bbox_top"], [80.0, 100.0, 120.0, 140.0])
        self.assertEqual(m["bbox_overview"], [90.0, 110.0, 130.0, 150.0])
        self.assertEqual(m["pixel"], [100.0, 120.0], "兼容别名优先 top")

    def test_loop_locate_metrics_carry_refined_flag(self):
        loc = StubLocator({"ok": True, "pixel_top": [101.5, 119.5], "pixel_overview": [110, 130],
                           "bbox_top": [81, 101, 121, 141], "bbox_overview": [90, 110, 130, 150],
                           "refined": True, "delta_px": 1.6, "source": "color_blob"})
        vlm = ScriptedVLM([SkillCall("locate", {"target": "水瓶", "color": "cyan"}),
                           SkillCall("stop", {})])
        loop = AgentLoop(StubEnv(), RecordingExecutor(), vlm, max_steps=4, log=None, locator=loc)
        results = loop.run("标定水瓶")
        m = results[0].metrics
        self.assertTrue(m.get("refined"), m)
        self.assertEqual(m.get("delta_px"), 1.6)
        self.assertEqual(m.get("pixel_top"), [101.5, 119.5])

    def test_vlm_review_prefers_verify_subgoal_tool(self):
        vlm = ToolReviewVLM(done=True, confidence=0.85, why="绿块质心落在红块框内")
        # 几何判据明确说不通过（stacked=False），这样才轮得到 VLM 工具复核生效
        v = SubGoalVerifier(StubEnv(), perception=StubPerception(inside=False, stacked=False),
                            vlm=vlm, log=None)
        sg = SubGoal(id="g1", instruction="把绿色方块叠到红色方块上", goal_type="place_on_top",
                     subject="绿色方块", subject_color="green", anchor_color="red")
        res = v.verify(sg)
        self.assertTrue(res["done"], res)
        self.assertEqual(res["evidence"], "vlm")
        self.assertEqual(res["confidence"], 0.85)
        self.assertEqual(res.get("source"), "vlm_tool")
        prompt = json.dumps(vlm.seen[-1], ensure_ascii=False)
        self.assertIn("verify_subgoal", prompt, "复核 prompt 必须点名要求工具作答")

    def test_verify_subgoal_low_confidence_still_not_done(self):
        vlm = ToolReviewVLM(done=True, confidence=0.5, why="看不太清")
        v = SubGoalVerifier(StubEnv(), perception=StubPerception(inside=False), vlm=vlm, log=None)
        res = v.verify(SubGoal(id="g1", instruction="随便", goal_type="free", subject="x"))
        self.assertFalse(res["done"])
        self.assertEqual(res["confidence"], 0.5)

    def test_empty_reply_is_inconclusive_not_failed(self):
        v = SubGoalVerifier(StubEnv(), perception=StubPerception(inside=False), vlm=EmptyVLM(), log=None)
        res = v.verify(SubGoal(id="g1", instruction="随便", goal_type="free", subject="x"))
        self.assertFalse(res["done"])
        self.assertTrue(res.get("inconclusive"), f"空回复要标「无结论」而不是「未完成」：{res}")
        self.assertIn("无结论", res["why"])
        # 有内容的明确否定不是「无结论」
        v2 = SubGoalVerifier(StubEnv(), perception=StubPerception(inside=False),
                             vlm=ReviewVLM("ask_user", note="红块还在桌上"), log=None)
        res2 = v2.verify(SubGoal(id="g1", instruction="随便", goal_type="free", subject="x"))
        self.assertFalse(res2["done"])
        self.assertFalse(res2.get("inconclusive", False), res2)
        self.assertIn("还在桌上", res2["why"])

    def test_vlm_exception_is_inconclusive(self):
        class BoomVLM:
            def decide(self, messages):
                raise RuntimeError("响应体里没有可解析的工具调用（可能被截断）")

        v = SubGoalVerifier(StubEnv(), perception=StubPerception(inside=False), vlm=BoomVLM(), log=None)
        res = v.verify(SubGoal(id="g1", instruction="随便", goal_type="free", subject="x"))
        self.assertFalse(res["done"])
        self.assertTrue(res.get("inconclusive"), res)

    def test_text_fallback_still_accepts_explicit_completion(self):
        class TextVLM:
            def decide(self, messages):
                return SkillCall("ask_user", {"question": "x"}, reasoning="红块已经在托盘里了")

        v = SubGoalVerifier(StubEnv(), perception=StubPerception(inside=False), vlm=TextVLM(), log=None)
        res = v.verify(SubGoal(id="g1", instruction="随便", goal_type="free", subject="x"))
        self.assertTrue(res["done"], res)
        self.assertEqual(res["source"], "vlm_text")


# --------------------------------------------------------------------------- #
# 16) P0：无真值抓取必须验证 —— 抓空不许报成功
# --------------------------------------------------------------------------- #
def _blob_image(cx: int = 40, cy: int = 40, size: int = 16):
    """合成 top 图：黑底 + 一个红方块（给 sim.perception 真判据用）。"""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[cy - size // 2:cy + size // 2, cx - size // 2:cx + size // 2] = [220, 30, 30]
    return img


def _wrist_image(blob_side: int = 0, cx: int = 64, cy: int = 64):
    """合成腕部图：中央一个实心红方块，blob_side 为边长（0 = 空夹爪，看不到色块）。

    真机实测：抓住 40mm 红方块时 wrist area≈16384px，空夹爪 0px；这里
    blob_side=64 -> 4096px（≥1500 判抓住），blob_side=32 -> 1024px（面积过小），0 -> 找不到。
    """
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    if blob_side > 0:
        h = max(1, blob_side // 2)
        img[cy - h:cy + h, cx - h:cx + h] = [220, 30, 30]
    return img


class PickVerifyEnv:
    """AgentLoop 专用桩（摄像头感知）：

    - render("top")：第 1 次给 before 图，之后给 after 图（top check_gone 主判据/颜色反查用）；
      top_ok=False 时抛异常，用来测「top 判据做不了 -> 腕部兜底」；
    - render("wrist")：按 wrist_seq 顺序给图（腕部现在只作诊断/兜底）；
      wrist_ok=False 时抛异常，用来测「腕部核验做不了 -> 未验证」。
    """

    def __init__(self, before=None, after=None, wrist=None, wrist_seq=None,
                 wrist_ok=True, top_ok=True, state=None, cameras=("top", "wrist")):
        self.before = before if before is not None else _blob_image()
        self.after = after if after is not None else _blob_image()
        self.wrist = wrist if wrist is not None else _wrist_image(0)
        self.wrist_seq = list(wrist_seq or [])
        self.wrist_ok = bool(wrist_ok)
        self.top_ok = bool(top_ok)
        self._state = state or make_state()
        self.cameras = tuple(cameras)
        self.renders: list[str] = []
        self.n = 0

    def get_state(self):
        return self._state

    def render(self, camera="top"):
        self.renders.append(camera)
        if str(camera) == "wrist":
            if not self.wrist_ok:
                raise RuntimeError("stub wrist render failure")
            return self.wrist_seq.pop(0) if self.wrist_seq else self.wrist
        if str(camera) == "top" and not self.top_ok:
            raise RuntimeError("stub top render failure")
        self.n += 1
        return self.before if self.n == 1 else self.after


class LocateThenBoxVLM:
    """返回 locate(...) 工具调用（可带 color / pixel / bbox）的假 VLM。"""

    def __init__(self, target="红色方块", color="red", pixel=None, bbox=None):
        self.target, self.color, self.pixel, self.bbox = target, color, pixel, bbox

    def decide(self, messages):
        args = {"target": self.target, "color": self.color}
        if self.pixel is not None:
            args["pixel"] = list(self.pixel)
        if self.bbox is not None:
            args["bbox"] = list(self.bbox)
        return SkillCall("locate", args)


class TestPickGraspVerification(unittest.TestCase):
    """P1：vlm 模式下技能自称抓取成功后，必须用 **top 前后对比（check_gone）** 独立确认
    「物体真的被拿走了」；物体仍在原位 -> grasp_not_confirmed。腕部只作诊断/兜底。"""

    def setUp(self):
        clear_pixel_cache()

    def _state_with_red(self):
        # box_red 在场景里；target_id 会被 §5.12-1 丢掉，但它的 color 仍用于验证
        st = make_state()
        st.robot.holding = None
        return st

    def test_grasp_air_reports_not_confirmed(self):
        env = PickVerifyEnv(before=_blob_image(), after=_blob_image(),
                            wrist=_wrist_image(0),          # 腕部看不到色块 = 夹爪里没东西
                            state=self._state_with_red())
        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40],
                                              "pixel_camera": "top",
                                              "grasp_profile": "auto"}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, ex, vlm, cameras=(), max_steps=4, log=None,
                         include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        results = loop.run("抓红色方块")
        first = results[0]
        self.assertEqual(first.skill, "pick")
        self.assertFalse(first.ok, "抓空绝不能报成功（实机 diag_vla_2 复现）")
        self.assertEqual(first.error_code, "grasp_not_confirmed")
        self.assertIn("仍留在原位", first.message)
        m = first.metrics
        self.assertFalse(m.get("grasp_verified"))
        self.assertEqual(m.get("grasp_verify_camera"), "top", "主判据必须是 top check_gone")
        self.assertFalse(m.get("grasp_top_check_gone"))
        self.assertEqual(m.get("grasp_verify_color"), "red")
        self.assertEqual(m.get("grasp_verify_color_source"), "scene:box_red")
        self.assertEqual(m.get("grasp_wrist_area_px"), 0, "腕部面积保留为诊断指标")
        self.assertTrue(m.get("skill_claimed_message"), "技能自称的成功要留在证据里")
        # 执行器确实抓到过这次调用（我们只是把结论改写成失败）
        self.assertEqual(ex.calls[0].skill, "pick")

    def test_real_grasp_sets_grasp_verified(self):
        env = PickVerifyEnv(before=_blob_image(), after=np.zeros((128, 128, 3), np.uint8),
                            wrist=_wrist_image(64),          # 腕部 4096px（诊断）
                            state=self._state_with_red())
        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40],
                                              "pixel_camera": "top"}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, ex, vlm, cameras=(), max_steps=4, log=None,
                         include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        results = loop.run("抓那个红块")
        m = results[0].metrics
        self.assertTrue(results[0].ok, results[0].message)
        self.assertTrue(m.get("grasp_verified"), m)
        self.assertEqual(m.get("grasp_verify_camera"), "top")
        self.assertTrue(m.get("grasp_top_check_gone"))
        self.assertGreaterEqual(m.get("grasp_verify_area_px", 0), 1500, m)
        self.assertEqual(m.get("grasp_wrist_area_px"), 4096)
        self.assertEqual(m.get("grasp_verify_color"), "red")
        self.assertEqual(m.get("grasp_verify_color_source"), "scene:box_red")
        self.assertIsNone(results[0].error_code)

    def test_top_still_visible_but_skill_physical_evidence_verifies(self):
        """真抓但 top 没被夹爪挡住（实测蓝块面积 818->827）-> 靠技能物理随动证据判成功。"""
        env = PickVerifyEnv(before=_blob_image(), after=_blob_image(),
                            wrist=_wrist_image(64), state=self._state_with_red())

        class EvidenceExecutor:
            def __call__(self, call):
                if call.skill == "pick":
                    return SkillResult(True, "pick", "已抓起",
                                       metrics={"grasp_lift_ok": True,
                                                "grasp_carry_evidence": True,
                                                "grasp_carry_bodies": ["box_red"]})
                return SkillResult(True, call.skill, "stub ok")

        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, EvidenceExecutor(), vlm, cameras=(), max_steps=4, log=None,
                         include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        results = loop.run("抓红色方块")
        m = results[0].metrics
        self.assertTrue(results[0].ok, results[0].message)
        self.assertTrue(m.get("grasp_verified"), m)
        self.assertEqual(m.get("grasp_verify_camera"), "top")
        self.assertFalse(m.get("grasp_top_check_gone"))
        self.assertIn("物理随动证据", str(m.get("grasp_verify_note")))

    def test_top_still_visible_without_skill_evidence_is_not_confirmed(self):
        """反例：top 仍可见 + 技能没有物理证据 -> 必须判未抓住（不许放行）。"""
        env = PickVerifyEnv(before=_blob_image(), after=_blob_image(),
                            wrist=_wrist_image(64), state=self._state_with_red())
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, RecordingExecutor(), vlm, cameras=(), max_steps=4, log=None,
                         include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        first = loop.run("抓红色方块")[0]
        self.assertFalse(first.ok)
        self.assertEqual(first.error_code, "grasp_not_confirmed")
        self.assertIsNone(first.metrics.get("grasp_skill_carry_evidence"))

    def test_wrist_contradiction_blocks_physical_evidence_exception(self):
        """反例：物理证据说"手里有东西"、夹的也确实是目标物，但腕部看不到**目标颜色**
        -> 不许用物理证据放行。"""
        env = PickVerifyEnv(before=_blob_image(), after=_blob_image(),
                            wrist=_wrist_image(0), state=self._state_with_red())

        class EvidenceExecutor:
            def __call__(self, call):
                if call.skill == "pick":
                    return SkillResult(True, "pick", "已抓起",
                                       metrics={"grasp_lift_ok": True,
                                                "grasp_carry_evidence": True,
                                                "grasp_carry_bodies": ["box_red"]})
                return SkillResult(True, call.skill, "stub ok")

        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, EvidenceExecutor(), vlm, cameras=(), max_steps=4, log=None,
                         include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        first = loop.run("抓红色方块")[0]
        self.assertFalse(first.ok, "腕部看不到目标色不能算抓住")
        self.assertEqual(first.error_code, "grasp_not_confirmed")
        self.assertIn("腕部看不到", first.message)

    def test_physical_evidence_with_wrong_carried_body_is_rejected(self):
        """反例：物理证据说夹着 box_blue，而目标是 box_red -> 不许放行（防夹错物体）。"""
        env = PickVerifyEnv(before=_blob_image(), after=_blob_image(),
                            wrist=_wrist_image(64), state=self._state_with_red())

        class WrongObjectExecutor:
            def __call__(self, call):
                if call.skill == "pick":
                    return SkillResult(True, "pick", "已抓起",
                                       metrics={"grasp_lift_ok": True,
                                                "grasp_carry_evidence": True,
                                                "grasp_carry_bodies": ["box_blue"]})
                return SkillResult(True, call.skill, "stub ok")

        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, WrongObjectExecutor(), vlm, cameras=(), max_steps=4, log=None,
                         include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        first = loop.run("抓红色方块")[0]
        self.assertFalse(first.ok, "夹着别的东西不能算抓住目标")
        self.assertEqual(first.error_code, "grasp_not_confirmed")
        self.assertIn("不是目标 box_red", first.message)

    def test_physical_evidence_with_empty_carry_bodies_is_rejected(self):
        """反例：carry_evidence=True 但 bodies 为空（自相矛盾）-> 不许放行。"""
        env = PickVerifyEnv(before=_blob_image(), after=_blob_image(),
                            wrist=_wrist_image(64), state=self._state_with_red())

        class EmptyBodiesExecutor:
            def __call__(self, call):
                if call.skill == "pick":
                    return SkillResult(True, "pick", "已抓起",
                                       metrics={"grasp_lift_ok": True,
                                                "grasp_carry_evidence": True,
                                                "grasp_carry_bodies": []})
                return SkillResult(True, call.skill, "stub ok")

        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, EmptyBodiesExecutor(), vlm, cameras=(), max_steps=4, log=None,
                         include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        first = loop.run("抓红色方块")[0]
        self.assertFalse(first.ok, "证据自相矛盾不能算抓住")
        self.assertEqual(first.error_code, "grasp_not_confirmed")

    def test_unknown_color_skips_with_explicit_log(self):
        # 只给 target_xy（没有像素、没有 target_id、也没 locate）-> 无从得知颜色
        # -> 明示「抓取未验证（未知颜色）」，但仍按旧行为执行（不判失败）
        env = PickVerifyEnv(before=_blob_image(), after=_blob_image(),
                            state=self._state_with_red())
        logs: list[str] = []
        vlm = ScriptedVLM([SkillCall("pick", {"target_xy": [0.40, 0.10]}), SkillCall("stop", {})])
        loop = AgentLoop(env, RecordingExecutor(), vlm, cameras=(), max_steps=4,
                         log=logs.append, include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        results = loop.run("抓那个东西")
        self.assertTrue(results[0].ok, "未知颜色只能「未验证」，不能因此判失败")
        self.assertIsNone(results[0].metrics.get("grasp_verified"))
        self.assertTrue(any("抓取未验证（未知颜色）" in m for m in logs), logs)

    def test_pixel_inferred_color_catches_air_grasp(self):
        # 模型只给像素（不 locate 不点名 id）-> 用像素反查颜色，抓空仍必须报失败
        env = PickVerifyEnv(before=_blob_image(), after=_blob_image(), wrist=_wrist_image(0),
                            state=self._state_with_red())
        vlm = ScriptedVLM([SkillCall("pick", {"target_pixel": [40, 40]}), SkillCall("stop", {})])
        loop = AgentLoop(env, RecordingExecutor(), vlm, cameras=(), max_steps=4, log=None,
                         include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        results = loop.run("把红色方块拿起来")
        first = results[0]
        self.assertFalse(first.ok, "像素路径抓空也必须报失败")
        self.assertEqual(first.error_code, "grasp_not_confirmed")
        self.assertIn("仍留在原位", first.message)
        self.assertEqual(first.metrics.get("grasp_verify_color"), "red")
        self.assertEqual(first.metrics.get("grasp_verify_color_source"), "pixel")
        self.assertEqual(first.metrics.get("grasp_verify_camera"), "top")

    def test_pixel_inferred_color_marks_real_grasp_verified(self):
        env = PickVerifyEnv(before=_blob_image(), after=np.zeros((128, 128, 3), np.uint8),
                            wrist=_wrist_image(64), state=self._state_with_red())
        vlm = ScriptedVLM([SkillCall("pick", {"target_pixel": [40, 40]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, RecordingExecutor(), vlm, cameras=(), max_steps=4, log=None,
                         include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        results = loop.run("把红色方块拿起来")
        self.assertTrue(results[0].ok, results[0].message)
        self.assertTrue(results[0].metrics.get("grasp_verified"))
        self.assertEqual(results[0].metrics.get("grasp_verify_color_source"), "pixel")
        self.assertEqual(results[0].metrics.get("grasp_verify_camera"), "top")

    def test_wrist_small_blob_is_not_confirmed(self):
        # top 判据做不了（top 渲染失败）-> 退回腕部；腕部只看到 32x32=1024px（< 1500）-> 未抓住
        env = PickVerifyEnv(before=_blob_image(), after=_blob_image(), wrist=_wrist_image(32),
                            top_ok=False, state=self._state_with_red())
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, RecordingExecutor(), vlm, cameras=(), max_steps=4, log=None,
                         include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        first = loop.run("抓红色方块")[0]
        self.assertFalse(first.ok, "腕部面积过小不能算抓住")
        self.assertEqual(first.error_code, "grasp_not_confirmed")
        self.assertIn("腕部色块只有", first.message)
        self.assertEqual(first.metrics.get("grasp_verify_camera"), "wrist")
        self.assertEqual(first.metrics.get("grasp_verify_fallback"), "top_unavailable")
        self.assertLess(int(first.metrics.get("grasp_verify_area_px") or 0), 1500)

    def test_wrist_unavailable_is_unverified_not_success(self):
        # top + 腕部都做不了 -> 结论必须是「未验证」，绝不判成功也不判失败
        env = PickVerifyEnv(before=_blob_image(), after=_blob_image(), wrist_ok=False,
                            top_ok=False, state=self._state_with_red())
        logs: list[str] = []
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, RecordingExecutor(), vlm, cameras=(), max_steps=4,
                         log=logs.append, include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        first = loop.run("抓红色方块")[0]
        self.assertTrue(first.ok, "判据不可用不能把技能的成功改判失败")
        m = first.metrics
        self.assertIsNone(m.get("grasp_verified"), "判据不可用时不得判成功")
        self.assertTrue(m.get("grasp_unverified"))
        self.assertEqual(m.get("grasp_verify_camera"), "none")
        self.assertTrue(any("都不可用" in x for x in logs), logs)

    def test_place_release_reviewed_by_wrist(self):
        # pick 抓稳（腕部 4096px）-> place 后腕部仍看到色块 = 没放下（只记 metrics）
        env = PickVerifyEnv(before=_blob_image(), after=np.zeros((128, 128, 3), np.uint8),
                            wrist_seq=[_wrist_image(64), _wrist_image(64)],
                            state=self._state_with_red())
        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40]}),
                           SkillCall("place", {"container_id": "tray", "xy": [0.40, 0.10]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, ex, vlm, cameras=(), max_steps=5, log=None, include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        results = loop.run("把红色方块放进托盘")
        self.assertEqual([c.skill for c in ex.calls], ["pick", "place", "stop"])
        self.assertTrue(results[0].metrics.get("grasp_verified"))
        place = results[1]
        self.assertTrue(place.ok, "release 复核不得改 ok（skills 语义不变）")
        self.assertFalse(place.metrics.get("release_verified"), place.metrics)
        self.assertEqual(place.metrics.get("release_check_camera"), "wrist")
        self.assertIn("没放下去", str(place.metrics.get("release_note")))

    def test_place_release_verified_when_wrist_is_empty(self):
        env = PickVerifyEnv(before=_blob_image(), after=np.zeros((128, 128, 3), np.uint8),
                            wrist_seq=[_wrist_image(64), _wrist_image(0)],
                            state=self._state_with_red())
        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40]}),
                           SkillCall("place", {"container_id": "tray", "xy": [0.40, 0.10]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, ex, vlm, cameras=(), max_steps=5, log=None, include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        results = loop.run("把红色方块放进托盘")
        place = results[1]
        self.assertTrue(place.ok)
        self.assertTrue(place.metrics.get("release_verified"), place.metrics)
        self.assertEqual(place.metrics.get("release_check_area_px"), 0)

    def test_oracle_mode_keeps_old_semantics(self):
        # include_poses=True（oracle/默认）不做图像复核，行为与旧版一致
        env = PickVerifyEnv(before=_blob_image(), after=_blob_image(),
                            state=self._state_with_red())
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red"}), SkillCall("stop", {})])
        loop = AgentLoop(env, RecordingExecutor(), vlm, cameras=(), max_steps=4, log=None)
        results = loop.run("抓红色方块")
        self.assertTrue(results[0].ok)
        self.assertIsNone(results[0].metrics.get("grasp_verified"))
        self.assertNotIn("wrist", env.renders, "oracle 模式不该做图像复核")

    def test_locate_color_feeds_pick_verification(self):
        # 先 locate 拿到 color，再 pick（只给像素、不点名 id）-> 用 locate 的 color 验证
        env = PickVerifyEnv(before=_blob_image(), after=np.zeros((128, 128, 3), np.uint8),
                            wrist=_wrist_image(64), state=self._state_with_red())
        loc = StubLocator({"ok": True, "pixel_top": [40, 40], "bbox_top": [32, 32, 47, 47],
                           "source": "color_blob", "why": "颜色分割在 top 找到 red 色块"})
        vlm = ScriptedVLM([SkillCall("locate", {"target": "红色方块", "color": "red"}),
                           SkillCall("pick", {"target_pixel": [40, 40], "pixel_camera": "top"}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, RecordingExecutor(), vlm, cameras=(), max_steps=5, log=None,
                         include_poses=False, locator=loc,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        results = loop.run("把红色方块放到桌上")
        self.assertEqual([r.skill for r in results], ["locate", "pick", "stop"])
        pick = results[1]
        self.assertTrue(pick.ok, pick.message)
        self.assertTrue(pick.metrics.get("grasp_verified"), pick.metrics)
        self.assertEqual(pick.metrics.get("grasp_verify_color"), "red")
        self.assertEqual(pick.metrics.get("grasp_verify_color_source"), "locate")
        self.assertEqual(pick.metrics.get("grasp_verify_camera"), "top")


class TestLocateRobustness(unittest.TestCase):
    """P0-4：locate 要稳 —— 缓存复用、换相机重试、模型自报 bbox 取中心。"""

    def setUp(self):
        clear_pixel_cache()

    def _verifier(self, perception, vlm=None):
        return SubGoalVerifier(StubEnv(), perception=perception, vlm=vlm, log=None)

    def test_locate_reuses_cached_pixel_when_color_blob_misses(self):
        remember_pixel("gray", [228.0, 425.0], camera="top", source="pick/place", target="tray")
        sp = StubPerception(blob=StubBlob(found=False))
        v = self._verifier(sp)
        r = v.locate("托盘", color="gray", camera="top")
        self.assertTrue(r["ok"], "颜色分割失败时应复用上一次成功标定的像素")
        self.assertEqual(r["source"], "cache")
        self.assertEqual(r["pixel_top"], [228.0, 425.0])
        self.assertIn("复用", r["why"])

    def test_locate_retries_other_camera(self):
        sp = StubPerception(blob_seq=[StubBlob(found=False),
                                      StubBlob(found=True, uv=(11.0, 22.0), bbox=(5, 5, 20, 30))])
        v = self._verifier(sp)
        r = v.locate("托盘", color="gray", camera="top")
        self.assertTrue(r["ok"])
        self.assertEqual(r["pixel_overview"], [11.0, 22.0], "top 失败要换 overview 再试")
        self.assertEqual(len(sp.blob_calls), 2)
        self.assertIn("换 overview", r["why"])

    def test_locate_uses_model_bbox_center(self):
        sp = StubPerception(blob=StubBlob(found=False))
        vlm = LocateThenBoxVLM(target="托盘", color="gray", bbox=[10, 20, 30, 40])
        v = self._verifier(sp, vlm)
        r = v.locate("托盘", color="gray", camera="top")
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["source"], "vlm_bbox")
        self.assertEqual(r["pixel_top"], [20.0, 30.0], "模型给的 bbox 要取中心当像素")
        self.assertEqual(r["bbox_top"], [10.0, 20.0, 30.0, 40.0])

    def test_locate_schema_accepts_pixel_and_bbox(self):
        st = make_state()
        call = parse_skill_call({"name": "locate",
                                 "arguments": {"target": "托盘", "color": "gray",
                                               "pixel": [228, 425], "bbox": [210, 410, 246, 440]}})
        self.assertEqual(call.args["pixel"], [228, 425])
        ok, msg = validate_skill_call(call, st)
        self.assertTrue(ok, msg)
        bad = SkillCall("locate", {"target": "托盘", "bbox": [1, 2, 3]})
        ok, msg = validate_skill_call(bad, st)
        self.assertFalse(ok)
        self.assertIn("bbox", msg)

    def test_orchestrator_clears_pixel_cache_per_run(self):
        remember_pixel("gray", [1.0, 2.0])
        self.assertIsNotNone(recall_pixel("gray"))
        factory = LoopFactory()
        orch = SubGoalOrchestrator(StubEnv(), MockVLM(), factory, ScriptedVerifier(), log=None)
        orch.run("t", [SubGoal(id="g1", instruction="i", goal_type="free", subject="s")])
        self.assertIsNone(recall_pixel("gray"), "新任务（orchestrator run）开头要清缓存")

    def test_agent_loop_does_not_clear_pixel_cache(self):
        # 关键：动作阶段(place/locate)标定好的框必须活到复核；AgentLoop.run 不得清缓存
        remember_pixel("gray", [10.0, 20.0], camera="top", bbox=[0, 0, 20, 30], target="tray")
        loop = AgentLoop(PickVerifyEnv(state=make_state()), RecordingExecutor(),
                         ScriptedVLM([SkillCall("stop", {})]), cameras=(), max_steps=2, log=None)
        loop.run("停一下")
        self.assertIsNotNone(recall_pixel(None, camera="top", target="tray"),
                             "AgentLoop.run 不得清缓存（act->verify 之间要保留）")

    def test_verify_uses_action_phase_container_bbox_from_cache(self):
        # 实机 accept_v2_final2 的复现：place 成功、动作阶段标到托盘框，但复核时拿不到颜色线索。
        # 复核必须复用缓存 -> evidence=geometry / done=True（不得退化成 VLM 复核或 evidence=none）。
        remember_pixel(None, [230.0, 425.0], camera="top", bbox=[160.0, 380.0, 300.0, 470.0],
                       source="pick/place", target="托盘")
        sp = StubPerception(blob=StubBlob(found=False), inside=True)   # 颜色分割失败，但有缓存
        v = SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)
        sg = SubGoal(id="g1", instruction="把红色方块放进托盘", goal_type="place_in_container",
                     subject="红色方块", subject_color="red", container="托盘", container_color=None)
        res = v.verify(sg)
        print("[probe] 容器框缓存复核: done=%s evidence=%s inside_bbox=%s"
              % (res.get("done"), res.get("evidence"),
                 sp.inside_calls[0][1] if sp.inside_calls else None))
        self.assertTrue(res["done"], res)
        self.assertEqual(res["evidence"], "geometry")
        self.assertEqual(sp.inside_calls[0][1], (160.0, 380.0, 300.0, 470.0),
                         "check_inside 必须用动作阶段缓存的托盘框")

    def test_verify_container_bbox_falls_back_to_latest_cache(self):
        # 目标文本对不上（缓存里是英文 id）时，退回「最近一次 top 带 bbox 的标定」
        remember_pixel("gray", [228.0, 425.0], camera="top", bbox=[150.0, 370.0, 310.0, 480.0],
                       source="pick/place", target="tray")
        sp = StubPerception(blob=StubBlob(found=False), inside=True)
        v = SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)
        sg = SubGoal(id="g1", instruction="把红色方块放进托盘", goal_type="place_in_container",
                     subject="红色方块", subject_color="red", container="托盘", container_color=None)
        res = v.verify(sg)
        self.assertTrue(res["done"], res)
        self.assertEqual(res["evidence"], "geometry")
        self.assertEqual(sp.inside_calls[0][1], (150.0, 370.0, 310.0, 480.0))


class TestResolvedZAndSizeFeed(unittest.TestCase):
    """P0 追加：把三角定位的 z 喂 target_z_hint、把 locate bbox 估的尺寸喂 target_size。"""

    def setUp(self):
        clear_pixel_cache()

    def _loop(self, env, ex, vlm, **kw):
        return AgentLoop(env, ex, vlm, cameras=(), max_steps=4, log=None,
                         include_poses=False, **kw)

    def test_dual_pixel_z_becomes_target_z_hint(self):
        def resolver(u, v, camera, overview_uv=None):
            return {"ok": True, "xyz": [0.45, -0.18, 0.522], "source": "triangulate",
                    "residual_mm": 1.2}

        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_pixel": [300, 260],
                                              "target_pixel_overview": [280, 300],
                                              "grasp_profile": "side"}),
                           SkillCall("stop", {})])
        loop = self._loop(StubEnv(), ex, vlm, pixel_resolver=resolver)
        results = loop.run("抓水瓶")
        pick = ex.calls[0]
        self.assertAlmostEqual(pick.args.get("target_z_hint"), 0.522, places=6,
                               msg="三角定位的 z 必须作为 target_z_hint 传给技能")
        self.assertEqual(results[0].metrics.get("z_hint"), 0.522)

    def test_table_flat_z_is_not_written_as_hint(self):
        # z 与桌面平面一致（single_view 平面反投影）-> 不写 target_z_hint，保持旧行为
        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_pixel": [267, 125]}), SkillCall("stop", {})])
        loop = self._loop(StubEnv(), ex, vlm,
                          pixel_resolver=lambda u, v, c: [0.40, 0.10, float(TABLE_HEIGHT)])
        loop.run("抓红色方块")
        self.assertNotIn("target_z_hint", ex.calls[0].args)

    def test_model_z_hint_kept_when_resolver_is_table_flat(self):
        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_pixel": [267, 125], "target_z_hint": 0.47}),
                           SkillCall("stop", {})])
        loop = self._loop(StubEnv(), ex, vlm,
                          pixel_resolver=lambda u, v, c: [0.40, 0.10, float(TABLE_HEIGHT)])
        loop.run("抓红色方块")
        self.assertEqual(ex.calls[0].args.get("target_z_hint"), 0.47)

    def test_place_does_not_get_z_hint_or_size(self):
        def resolver(u, v, camera, overview_uv=None):
            return {"ok": True, "xyz": [0.40, 0.10, 0.52], "source": "triangulate"}

        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red"}),
                           SkillCall("place", {"container_id": "tray", "target_pixel": [228, 425]}),
                           SkillCall("stop", {})])
        loop = self._loop(StubEnv(), ex, vlm, pixel_resolver=resolver)
        results = loop.run("把红色方块放进托盘")
        self.assertEqual([c.skill for c in ex.calls], ["pick", "place", "stop"], [r.message for r in results])
        place = ex.calls[1]
        self.assertEqual(place.args.get("xy"), [0.4, 0.1])
        self.assertNotIn("target_z_hint", place.args, "place 不写 z_hint")
        self.assertNotIn("target_size", place.args, "place 不写 size")

    def test_locate_bbox_size_feeds_target_size(self):
        remember_pixel("red", [40.0, 40.0], camera="top", bbox=[32.0, 32.0, 48.0, 48.0],
                       source="color_blob")
        with mock.patch("sim.perception.bbox_to_world",
                        return_value={"ok": True, "center_xy": (0.4, 0.1),
                                      "size_xy": (0.05, 0.04), "corner_xy": [], "assumed_z": True}):
            ex = RecordingExecutor()
            vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red",
                                                  "target_pixel": [40, 40]}),
                               SkillCall("stop", {})])
            loop = self._loop(StubEnv(), ex, vlm,
                              pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
            results = loop.run("抓红色方块")
        self.assertEqual(ex.calls[0].args.get("target_size"), [0.025, 0.02],
                         "bbox 全宽 0.05/0.04 -> 半宽 0.025/0.02")
        self.assertEqual(results[0].metrics.get("target_size_est"), [0.025, 0.02])

    def test_size_estimate_guards(self):
        loop = self._loop(StubEnv(), RecordingExecutor(), ScriptedVLM([]),
                          pixel_resolver=lambda u, v, c: [0.4, 0.1, 0.42])
        call = SkillCall("pick", {"target_xy": [0.4, 0.1]})
        st = make_state()
        # gray/white/black 颜色规则不可靠 -> 不估
        remember_pixel("gray", [40.0, 40.0], camera="top", bbox=[10.0, 10.0, 30.0, 30.0])
        loop._locate_color = "gray"
        self.assertIsNone(loop._estimate_target_size(call, st))
        # bbox 面积 > 25% 画面 -> 不估
        remember_pixel("red", [40.0, 40.0], camera="top", bbox=[0.0, 0.0, 511.0, 511.0])
        loop._locate_color = "red"
        self.assertIsNone(loop._estimate_target_size(call, st))
        # 没有 bbox -> 不估
        remember_pixel("blue", [40.0, 40.0], camera="top")
        loop._locate_color = "blue"
        self.assertIsNone(loop._estimate_target_size(call, st))

    def test_pick_target_size_schema(self):
        st = make_state()
        for ts in ([0.025, 0.02], [0.025, 0.02, 0.02]):
            call = parse_skill_call({"name": "pick",
                                     "arguments": {"target_id": "box_red", "target_size": ts}})
            self.assertEqual(len(call.args["target_size"]), len(ts))
            self.assertTrue(validate_skill_call(call, st)[0])
        for bad in ([0.025], [0.02, 0.02, 0.02, 0.02], [-0.02, 0.02], [0.6, 0.2], ["a", 0.02]):
            ok, msg = validate_skill_call(SkillCall("pick", {"target_id": "box_red",
                                                             "target_size": bad}), st)
            self.assertFalse(ok, f"{bad!r} 应被拒绝")
            self.assertIn("target_size", msg)


class TestHoldingInjection(unittest.TestCase):
    """P0：腕部核验过的 pick 之后，place 必须注入 holding=True/holding_color。

    否则 skills 会把手里夹着的物体当入口障碍做「温和撤退」，撤退中把件弄掉
    （实机：无真值 pick 抬起 z=0.5397 -> place 撤退 -> 方块掉回桌面，place 仍报成功）。
    """

    def setUp(self):
        clear_pixel_cache()

    def _state(self):
        st = make_state()
        st.robot.holding = None
        return st

    def test_place_injects_holding_after_verified_pick(self):
        env = PickVerifyEnv(before=_blob_image(), after=np.zeros((128, 128, 3), np.uint8),
                            wrist=_wrist_image(64), state=self._state())
        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40]}),
                           SkillCall("place", {"container_id": "tray", "xy": [0.40, 0.10]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, ex, vlm, cameras=(), max_steps=5, log=None, include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        results = loop.run("把红色方块放进托盘")
        self.assertEqual([c.skill for c in ex.calls], ["pick", "place", "stop"])
        self.assertTrue(results[0].metrics.get("grasp_verified"))
        self.assertNotIn("holding", ex.calls[0].args, "pick 不该被注入 holding")
        place_args = ex.calls[1].args
        self.assertIs(place_args.get("holding"), True, place_args)
        self.assertEqual(place_args.get("holding_color"), "red")

    def test_place_assumes_holding_after_ok_pick_without_color(self):
        # 图像判据都做不了（未验证），但技能自称成功 -> place 仍要按「手里有东西」处理，
        # 否则 skills 会做入口撤退把件弄掉；颜色未知就不带 holding_color。
        env = PickVerifyEnv(wrist_ok=False, top_ok=False, state=self._state())
        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40]}),
                           SkillCall("place", {"container_id": "tray", "xy": [0.40, 0.10]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, ex, vlm, cameras=(), max_steps=5, log=None, include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        results = loop.run("把红色方块放进托盘")
        self.assertTrue(results[0].ok)
        self.assertIsNone(results[0].metrics.get("grasp_verified"))
        self.assertIs(ex.calls[1].args.get("holding"), True, ex.calls[1].args)
        self.assertNotIn("holding_color", ex.calls[1].args)

    def test_no_holding_after_failed_pick(self):
        # pick 被证伪（抓空）-> 不能声称手里有东西
        env = PickVerifyEnv(wrist=_wrist_image(0), state=self._state())
        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40]}),
                           SkillCall("place", {"container_id": "tray", "xy": [0.40, 0.10]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, ex, vlm, cameras=(), max_steps=5, log=None, include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        results = loop.run("把红色方块放进托盘")
        self.assertEqual(results[0].error_code, "grasp_not_confirmed")
        self.assertNotIn("holding", ex.calls[1].args, "抓空证伪后不得注入 holding")

    def test_wrist_lookup_fills_holding_color_for_xy_only_pick(self):
        # target_xy 路径（无像素/无 id/无 locate）抓取时定不了颜色；place 之前用腕部图
        # 正向反查手里是什么颜色（红块 4096px）-> 注入 holding + holding_color
        env = PickVerifyEnv(wrist=_wrist_image(64), state=self._state())
        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_xy": [0.40, 0.10]}),
                           SkillCall("place", {"container_id": "tray", "xy": [0.40, 0.10]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, ex, vlm, cameras=(), max_steps=5, log=None, include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        loop.run("把红色方块放进托盘")
        args = ex.calls[1].args
        self.assertIs(args.get("holding"), True, args)
        self.assertEqual(args.get("holding_color"), "red", "腕部反查应补上颜色")

    def test_release_verified_clears_holding(self):
        # 第一次 place 腕部已看不到色块（放下了）-> 清除持物状态 -> 第二次 place 不注入
        env = PickVerifyEnv(before=_blob_image(), after=np.zeros((128, 128, 3), np.uint8),
                            wrist_seq=[_wrist_image(64), _wrist_image(0)], state=self._state())
        ex = RecordingExecutor()
        place = {"container_id": "tray", "xy": [0.40, 0.10]}
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40]}),
                           SkillCall("place", dict(place)),
                           SkillCall("place", dict(place)),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, ex, vlm, cameras=(), max_steps=6, log=None, include_poses=False,
                         pixel_resolver=lambda u, v, c: [0.40, 0.10, 0.42])
        results = loop.run("把红色方块放进托盘")
        self.assertEqual([c.skill for c in ex.calls], ["pick", "place", "place", "stop"])
        self.assertIs(ex.calls[1].args.get("holding"), True)
        self.assertTrue(results[1].metrics.get("release_verified"))
        place2 = ex.calls[2].args
        self.assertNotIn("holding", place2, "释放核验通过后必须清除持物状态")

    def test_oracle_mode_never_injects_holding(self):
        env = PickVerifyEnv(wrist=_wrist_image(64), state=self._state())
        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red"}),
                           SkillCall("place", {"container_id": "tray"}),
                           SkillCall("stop", {})])
        loop = AgentLoop(env, ex, vlm, cameras=(), max_steps=5, log=None)   # include_poses=True
        loop.run("把红色方块放进托盘")
        self.assertEqual([c.skill for c in ex.calls], ["pick", "place", "stop"])
        self.assertNotIn("holding", ex.calls[1].args, "oracle 模式行为不变")
        self.assertNotIn("wrist", env.renders)


class TestSameFeatureCorrespondence(unittest.TestCase):
    """同一特征对应：双目三角时 overview 像素改用 bbox 顶边中点（=物体顶面中心）。

    实机（水瓶）：模型 top 像素点得完全正确（0.3px），但 overview 点到了瓶身中部（低 45px），
    两视角不是同一物理点 -> 三角 xy 偏 25mm -> 下降撞瓶壁 33.7N。改用 bbox 顶边中点后偏 4.2mm。
    """

    def setUp(self):
        clear_pixel_cache()

    def test_locate_returns_overview_top_midpoint(self):
        sp = StubPerception(blob_seq=[
            StubBlob(found=True, uv=(178.44, 58.47), bbox=(170, 50, 187, 67)),        # top
            StubBlob(found=True, uv=(271.31, 350.79), bbox=(255, 298, 289, 406)),    # overview
        ])
        v = SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)
        r = v.locate("水瓶", color="cyan", camera="both")
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["bbox_overview"], [255.0, 298.0, 289.0, 406.0])
        self.assertEqual(r["pixel_overview"], [271.31, 350.79], "原始 overview 点保留在返回里")
        self.assertEqual(r["pixel_overview_top"], [272.0, 298.0],
                         "顶边中点 = ((255+289)/2, 298)，即 overview 视角的顶面中心")

    def test_locate_without_overview_bbox_has_no_top_midpoint(self):
        sp = StubPerception(blob=StubBlob(found=True, uv=(178.44, 58.47), bbox=None))
        v = SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)
        r = v.locate("水瓶", color="cyan", camera="top")
        self.assertTrue(r["ok"], r)
        self.assertIsNone(r["pixel_overview_top"], "没有 overview bbox 就不给（不瞎猜）")

    def test_dual_pixel_uses_overview_bbox_top_midpoint(self):
        # 动作阶段 locate 缓存的 overview 框（真实瓶身：顶 298 -> 底 406）
        remember_pixel("red", [271.0, 351.0], camera="overview", bbox=[255, 298, 289, 406],
                       source="locate", target="红色方块")
        seen: list = []

        def resolver(u, v, camera, overview_uv=None):
            seen.append((float(u), float(v), camera, overview_uv))
            return {"ok": True, "xyz": [0.4474, -0.1765, 0.614], "source": "triangulate",
                    "residual_mm": 2.0}

        ex = RecordingExecutor()
        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [178.44, 58.47],
                                              "target_pixel_overview": [271.31, 350.79]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(PickVerifyEnv(wrist=_wrist_image(64)), ex, vlm, cameras=(),
                         max_steps=4, log=None, include_poses=False, pixel_resolver=resolver)
        results = loop.run("抓水瓶")
        self.assertEqual(len(seen), 1)
        u, v, cam, ov = seen[0]
        self.assertEqual((u, v, cam), (178.44, 58.47, "top"))
        self.assertEqual(ov, [272.0, 298.0], "overview 必须换成 bbox 顶边中点（同一特征）")
        m = results[0].metrics
        self.assertEqual(m.get("overview_uv_used"), [272.0, 298.0])
        self.assertEqual(m.get("overview_uv_model"), [271.31, 350.79])
        self.assertIn("bbox", str(m.get("overview_uv_source")))
        self.assertAlmostEqual(ex.calls[0].args.get("target_xy", [0])[0], 0.4474, places=4)

    def test_dual_pixel_keeps_model_point_without_bbox(self):
        # 没有 bbox（也渲染不了 overview）-> 沿用模型给的 overview 点：方块/现状行为不变
        class _NoOverviewEnv(PickVerifyEnv):
            def render(self, camera="top"):
                if str(camera) == "overview":
                    raise RuntimeError("no overview camera (stub)")
                return super().render(camera)

        seen: list = []

        def resolver(u, v, camera, overview_uv=None):
            seen.append((float(u), float(v), camera, overview_uv))
            return {"ok": True, "xyz": [0.40, 0.10, 0.43], "source": "triangulate"}

        vlm = ScriptedVLM([SkillCall("pick", {"target_id": "box_red", "target_pixel": [40, 40],
                                              "target_pixel_overview": [50, 60]}),
                           SkillCall("stop", {})])
        loop = AgentLoop(_NoOverviewEnv(wrist=_wrist_image(64)), RecordingExecutor(), vlm,
                         cameras=(), max_steps=4, log=None, include_poses=False,
                         pixel_resolver=resolver)
        loop.run("抓方块")
        self.assertEqual(seen[0][3], [50.0, 60.0], "无 bbox 时保持现状：用模型给的点")
        self.assertNotIn("overview_uv_used", seen[0][3])


class TestStackedOcclusionAndObservation(unittest.TestCase):
    """叠放复核在遮挡下的补证 + 复核前让开视线（实机蓝叠绿案例）。"""

    def setUp(self):
        clear_pixel_cache()

    def _sg(self):
        return SubGoal(id="g1", instruction="把蓝色方块叠到绿色方块上", goal_type="place_on_top",
                       subject="蓝色方块", subject_color="blue", anchor_color="green")

    def _verifier(self, sp):
        return SubGoalVerifier(StubEnv(), perception=sp, vlm=None, log=None)

    def test_stacked_overview_bbox_adjacency(self):
        # top 判不了（下层被上层完全盖住）-> overview 用 bbox 相邻判据补证
        sp = StubPerception(stacked=False, blob_seq=[
            StubBlob(found=True, uv=(100.0, 95.0), bbox=(80.0, 80.0, 120.0, 100.0)),    # 上层
            StubBlob(found=True, uv=(100.0, 120.0), bbox=(80.0, 102.0, 120.0, 140.0)),  # 下层
        ])
        res = self._verifier(sp).verify(self._sg())
        print("[probe] 叠放 overview 相邻判据:", res.get("done"), res.get("why"))
        self.assertTrue(res["done"], res)
        self.assertEqual(res["evidence"], "geometry")
        self.assertIn("相机=overview", res["why"])
        self.assertIn("bbox 相邻", res["why"])

    def test_stacked_overview_occlusion_with_anchor_bbox(self):
        # 下层完全找不到 + 上层质心落在动作阶段标定的锚点框内 -> 遮挡判据
        remember_pixel("green", [100.0, 120.0], camera="top", bbox=[80.0, 90.0, 130.0, 150.0],
                       source="pick/place", target="box_green")
        sp = StubPerception(stacked=False, blob_seq=[
            StubBlob(found=True, uv=(100.0, 95.0), bbox=(80.0, 80.0, 120.0, 100.0)),
            StubBlob(found=False),
        ])
        res = self._verifier(sp).verify(self._sg())
        print("[probe] 叠放遮挡判据:", res.get("done"), res.get("why"))
        self.assertTrue(res["done"], res)
        self.assertEqual(res["evidence"], "geometry")
        self.assertIn("遮挡", res["why"])
        self.assertIn("相机=overview", res["why"])

    def test_stacked_not_widened_when_far_apart(self):
        # 反例：上层与下层在 overview 里离得很远（没叠上）-> 必须 done=False，不许放宽
        sp = StubPerception(stacked=False, blob_seq=[
            StubBlob(found=True, uv=(100.0, 95.0), bbox=(80.0, 80.0, 120.0, 100.0)),
            StubBlob(found=True, uv=(400.0, 400.0), bbox=(380.0, 380.0, 420.0, 420.0)),
        ])
        res = self._verifier(sp).verify(self._sg())
        print("[probe] 叠放反例:", res.get("done"), res.get("why"))
        self.assertFalse(res["done"], "没叠上绝不能判成功")
        self.assertIn("bbox", res["why"])

    def test_top_still_preferred_when_available(self):
        # top 能判就仍用 top（先 top 后 overview），why 里写明相机
        sp = StubPerception(stacked=True)
        res = self._verifier(sp).verify(self._sg())
        self.assertTrue(res["done"], res)
        self.assertIn("相机=top", res["why"])
        self.assertIn("check_stacked", res["why"])

    def test_orchestrator_observes_before_verify(self):
        order: list = []

        class ObsVerifier:
            camera = "top"

            def verify(self, sg, *, metrics=None, rgb_before=None):
                order.append("verify")
                return {"done": True, "evidence": "geometry", "why": "ok"}

        orch = SubGoalOrchestrator(StubEnv(), MockVLM(), LoopFactory(), ObsVerifier(), log=None,
                                   observe=lambda: order.append("observe"))
        rec = orch.run("t", [SubGoal(id="g1", instruction="i", goal_type="free", subject="s")])[0]
        self.assertTrue(rec["done"])
        self.assertEqual(order, ["observe", "verify"], "复核前必须先让开视线")

    def test_observe_hook_receives_subgoal_when_it_accepts_one(self):
        """P0：钩子能吃参数时必须把子目标传进去（demo 靠它识别 pick 子目标、拒绝挪机械臂）。"""
        seen: list = []

        class ObsVerifier:
            camera = "top"

            def verify(self, sg, *, metrics=None, rgb_before=None):
                return {"done": True, "evidence": "geometry", "why": "ok"}

        def hook(sg):
            seen.append(sg)

        orch = SubGoalOrchestrator(StubEnv(), MockVLM(), LoopFactory(), ObsVerifier(), log=None,
                                   observe=hook)
        sg = SubGoal(id="g1", instruction="抓起红块", goal_type="pick", subject="红块")
        rec = orch.run("t", [sg])[0]
        self.assertTrue(rec["done"])
        self.assertEqual(len(seen), 1, "带参钩子必须被调用一次")
        self.assertIs(seen[0], sg, "必须把当前子目标传给钩子")
        self.assertEqual(seen[0].goal_type, "pick")

    def test_observe_failure_does_not_kill_subgoal(self):
        calls: list = []

        def boom():
            calls.append("observe")
            raise RuntimeError("home timeout")

        class OkVerifier:
            camera = "top"

            def verify(self, sg, *, metrics=None, rgb_before=None):
                calls.append("verify")
                return {"done": True, "evidence": "vlm", "why": "ok"}

        orch = SubGoalOrchestrator(StubEnv(), MockVLM(), LoopFactory(), OkVerifier(), log=None,
                                   observe=boom)
        rec = orch.run("t", [SubGoal(id="g1", instruction="i", goal_type="free", subject="s")])[0]
        self.assertTrue(rec["done"], "观察位失败不许把子目标判死")
        self.assertEqual(calls, ["observe", "verify"])

    def test_default_observe_uses_env_home(self):
        class HomeEnv(StubEnv):
            def __init__(self):
                super().__init__()
                self.homes = 0

            def home(self):
                self.homes += 1

        env = HomeEnv()
        orch = SubGoalOrchestrator(env, MockVLM(), LoopFactory(), ScriptedVerifier(
            [{"done": True, "evidence": "geometry", "why": "ok"}]), log=None)
        orch.run("t", [SubGoal(id="g1", instruction="i", goal_type="free", subject="s")])
        self.assertEqual(env.homes, 1, "未注入 observe 钩子时退回 env.home()")

    def test_default_observe_skips_home_for_pick_subgoal(self):
        """P0 绕关口：**没注入钩子**时的默认兜底也不许对 pick 子目标 env.home()。

        实测旧行为：持物时默认兜底瞬移会让物体留在原地掉回桌面（z 0.5393 -> 0.4199），
        pick 子目标复核必然 subject_still_present。"""
        class HomeEnv(StubEnv):
            def __init__(self):
                super().__init__()
                self.homes = 0

            def home(self):
                self.homes += 1

        env = HomeEnv()
        verifier = ScriptedVerifier([{"done": True, "evidence": "geometry", "why": "ok"}])
        orch = SubGoalOrchestrator(env, MockVLM(), LoopFactory(), verifier, log=None)
        rec = orch.run("t", [SubGoal(id="g1", instruction="i", goal_type="pick",
                                     subject="s", subject_color="red")])[0]
        self.assertTrue(rec["done"])
        self.assertEqual(env.homes, 0, "pick 子目标不得走默认 env.home() 观察位")

        env2 = HomeEnv()
        orch2 = SubGoalOrchestrator(env2, MockVLM(), LoopFactory(),
                                    ScriptedVerifier([{"done": True, "evidence": "geometry",
                                                       "why": "ok"}]), log=None)
        orch2.run("t", [SubGoal(id="g1", instruction="i", goal_type="place_in_container",
                                subject="s", subject_color="red")])
        self.assertEqual(env2.homes, 1, "非 pick 子目标仍应回观察位")


# --------------------------------------------------------------------------- #
# 17) task-67：VLM 空回复不再杀死任务（decide 内重试 + token 升级 + 环内容忍）
# --------------------------------------------------------------------------- #
def _empty_response(reasoning="长思维链" * 40):
    return {"choices": [{"message": {"content": "", "reasoning": reasoning}}]}


def _tool_call_response(name="home", arguments="{}"):
    return {"choices": [{"message": {"tool_calls": [
        {"function": {"name": name, "arguments": arguments}}]}}]}


class _FakePostOnce:
    """按脚本返回响应体的 _post_once 替身；记录每次 payload。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads: list[dict] = []

    def __call__(self, data: bytes):
        self.payloads.append(json.loads(data.decode("utf-8")))
        if not self.responses:
            return _empty_response()
        return self.responses.pop(0)


class TestVLMEmptyResponseRetry(unittest.TestCase):
    """空回复/无工具调用必须在 decide() 内重试（老实现只在传输层重试 -> 一次就死）。"""

    def _client(self, responses, *, retries=3):
        c = VLMClient(api_key="dummy", max_retries=retries, sleep=lambda _s: None)
        fake = _FakePostOnce(responses)
        c._post_once = fake          # type: ignore[method-assign]
        return c, fake

    def test_empty_then_ok_retries_and_raises_token_budget(self):
        c, fake = self._client([_empty_response(), _tool_call_response("home")])
        call = c.decide([{"role": "user", "content": "回家"}])
        self.assertEqual(call.skill, "home")
        self.assertEqual(len(fake.payloads), 2, "空回复必须触发一次真实重试")
        p1, p2 = fake.payloads[0], fake.payloads[1]
        self.assertEqual(p1["max_tokens"], c.max_tokens)
        self.assertGreater(p2["max_tokens"], p1["max_tokens"], "重试要升级 max_tokens")
        self.assertEqual(p2["max_tokens"], min(p1["max_tokens"] * 2, 8192))
        # 重试 payload 追加了「必须调用工具」提示，且不动调用方 messages
        self.assertTrue(any("必须调用工具" in json.dumps(m, ensure_ascii=False)
                            for m in p2["messages"]), p2["messages"][-1])
        self.assertEqual(len(p1["messages"]), 1, "首次请求不改动 messages")
        self.assertGreaterEqual(len(p2["messages"]), 2)
        self.assertEqual(c.last_decide_attempts, 2, "decide 层记录了真实的两次尝试")

    def test_all_empty_raises_with_retry_evidence(self):
        c, fake = self._client([_empty_response() for _ in range(5)], retries=2)
        with self.assertRaises(VLMResponseError) as cm:
            c.decide([{"role": "user", "content": "x"}])
        msg = str(cm.exception)
        self.assertEqual(len(fake.payloads), 3, "retries=2 -> 共 3 次尝试")
        self.assertIn("重试", msg)
        self.assertIn("max_tokens", msg)
        self.assertEqual(cm.exception.error_code, "empty_response")
        self.assertFalse(cm.exception.retryable, "重试耗尽的最终异常不再标可重试")

    def test_retryable_criterion_is_attribute_based(self):
        # 稳定判据：读属性，不靠中文子串
        self.assertTrue(_retryable_response(VLMResponseError("任意文案", retryable=True,
                                                              error_code="empty_response")))
        self.assertFalse(_retryable_response(VLMResponseError("没有可解析的工具调用")))
        self.assertFalse(_retryable_response("响应既没有 tool_calls，content 也为空"))
        self.assertTrue(_retryable_response(VLMResponseError("HTTP 429", retryable=True,
                                                              error_code="http_429")))
        # decide 真正抛出的空回复异常必须标 retryable
        c, _fake = self._client([], retries=0)
        with self.assertRaises(VLMResponseError) as cm:
            c.decide([{"role": "user", "content": "x"}])
        self.assertTrue(cm.exception.retryable)
        self.assertEqual(cm.exception.error_code, "empty_response")

    def test_unparseable_content_is_retryable(self):
        bad = {"choices": [{"message": {"content": "我觉得应该先抓红色方块（没有 JSON）"}}]}
        c, fake = self._client([bad, _tool_call_response("stop")])
        self.assertEqual(c.decide([{"role": "user", "content": "x"}]).skill, "stop")
        self.assertEqual(len(fake.payloads), 2)

    def test_no_choices_is_retryable(self):
        c, fake = self._client([{"error": {"message": "boom"}}, _tool_call_response("stop")])
        self.assertEqual(c.decide([{"role": "user", "content": "x"}]).skill, "stop")
        self.assertEqual(len(fake.payloads), 2)


class TestLoopToleratesTransientVLMErrors(unittest.TestCase):
    """环内：偶发空回复/异常不许把只差一步的任务打死。"""

    def test_transient_empty_vlm_still_finishes_task(self):
        class FlakyVLM:
            """第 2、5 次调用抛空回复异常，其余按脚本走 pick -> place -> stop。"""

            def __init__(self):
                self.script = [SkillCall("pick", {"target_id": "box_red"}),
                               SkillCall("place", {"container_id": "tray"}),
                               SkillCall("stop", {})]
                self.calls = 0
                self.seen: list = []

            def decide(self, messages):
                self.calls += 1
                self.seen.append(messages)
                if self.calls in (2, 5):
                    raise VLMResponseError("响应既没有 tool_calls，content 也为空",
                                           retryable=True, error_code="empty_response")
                return self.script.pop(0) if self.script else SkillCall("stop", {})

        vlm = FlakyVLM()
        ex = RecordingExecutor()
        logs: list[str] = []
        loop = AgentLoop(StubEnv(), ex, vlm, cameras=("overview",), max_steps=8, log=logs.append)
        results = loop.run("把红色方块放进托盘")

        # 任务跑完：执行器收到 pick/place/stop；失败步只有 vlm_error，且没有打断后续
        self.assertEqual([c.skill for c in ex.calls], ["pick", "place", "stop"])
        self.assertEqual([r.error_code for r in results if r.error_code], ["vlm_error"])
        self.assertTrue(results[-1].ok)
        self.assertTrue(any("VLM 连续异常 1/2" in m for m in logs), logs)
        # history 没被脏观察污染：只记录真实动作/观察对
        self.assertEqual(len(loop.history) % 2, 0)
        self.assertNotIn("vlm_error", " ".join(str(m.get("content")) for m in loop.history))
        # 重试确实重新截图/重建了 messages：4 次决策（pick / 空回复 / place / stop）
        self.assertEqual(len(vlm.seen), 4, "重试要重新构建 messages，而不是重发同一份")
        self.assertEqual(vlm.calls, 4)

    def test_two_consecutive_failures_abort_cleanly(self):
        class DeadVLM:
            def __init__(self):
                self.calls = 0

            def decide(self, messages):
                self.calls += 1
                raise VLMResponseError("响应既没有 tool_calls，content 也为空",
                                       retryable=True, error_code="empty_response")

        vlm = DeadVLM()
        logs: list[str] = []
        loop = AgentLoop(StubEnv(), RecordingExecutor(), vlm, cameras=(), max_steps=8,
                         log=logs.append)
        results = loop.run("把红色方块放进托盘")
        self.assertEqual(len(results), 2, "连续 2 次失败后中止")
        self.assertTrue(all(r.error_code == "vlm_error" for r in results))
        self.assertEqual(vlm.calls, 2)
        self.assertTrue(any("VLM 连续异常 2/2，任务中止" in m for m in logs), logs)

    def test_success_resets_error_streak(self):
        # 失败 1 次 -> 成功 -> 再失败 1 次：不该因为累计而中止
        class MixedVLM:
            def __init__(self):
                self.n = 0

            def decide(self, messages):
                self.n += 1
                if self.n == 1:
                    raise VLMResponseError("empty", retryable=True, error_code="empty_response")
                if self.n == 3:
                    raise VLMResponseError("empty", retryable=True, error_code="empty_response")
                return SkillCall("home", {}) if self.n < 4 else SkillCall("stop", {})

        ex = RecordingExecutor()
        loop = AgentLoop(StubEnv(), ex, MixedVLM(), cameras=(), max_steps=6, log=None)
        results = loop.run("回家")
        self.assertEqual([c.skill for c in ex.calls], ["home", "stop"])
        self.assertEqual(len([r for r in results if r.error_code == "vlm_error"]), 2)
        self.assertTrue(results[-1].ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
