"""tests/test_panels.py —— 面板「纯文件」架构独立验证（task-65 改版）。

新产出（用户要求：不建任何 MuJoCo 窗口、不要实时画面，只在发消息时更新图片）：
  outputs/panels/llm_frames.png —— 发给模型的 overview+top 并排图 + caption
  outputs/panels/curves.png     —— matplotlib Agg 曲线，<=1Hz
没有任何面板子进程、没有子进程入口、没有 ready 文件。

验证五条：
  1) 帧一致性：PNG 里的两张图 == 本次请求实际发给 LLM 的字节解码结果（逐像素）；
     caption 的字节数/相机/格式与请求一致（F1 的教训：必须真的比对「发出的字节」与「写进 PNG 的像素」）。
  2) 不建窗/不起进程：--panels all 与 --headless --panels all 都不得出现第二个 MuJoCo 窗口、
     不得有新子进程（ps 精确匹配 demo.panels / dsh-panels，且只看本进程后代，避免误判别人的进程）。
  3) 曲线：喂已知遥测序列 -> curves.png 生成、非空、大小稳定；1Hz 节流成立。
  4) 行为不变：test_interactive_flows 12/12；headless 一条 e2e 与 --panels none 结果一致。
  5) 退出：:quit 5s 内退出、无残留进程；跑 60s 无 SIGABRT/SIGSEGV。

文件架构未落地时全部 skip（pending）。

运行：
    MUJOCO_GL=glfw DISPLAY=:0 /home/tc63/mujoco/.venv/bin/python -m tests.test_panels
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import re
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.types import SkillCall                              # noqa: E402

_PANELS_SRC = (_ROOT / "demo" / "panels.py").read_text(encoding="utf-8")
_IA_SRC = (_ROOT / "demo" / "interactive.py").read_text(encoding="utf-8")
# 新（纯文件）架构的唯一标记
_FILE_PANELS = ("llm_frames" in _PANELS_SRC) or ("llm_frames" in _IA_SRC)
_SUBPROC_PANELS = ("PanelProcess" in _PANELS_SRC) or ("panel-child" in _IA_SRC)
_PENDING = ("面板尚未切到「纯文件」架构（llm_frames.png/curves.png 未落地）；"
            f"当前 panels.py 仍是子进程版={_SUBPROC_PANELS}")

_ENV = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "glfw"),
        "DISPLAY": os.environ.get("DISPLAY", ":0")}
_PROMPT = "vlm_admittance>"
_PANEL_DIR = _ROOT / "outputs" / "panels"
_FRAMES_PNG = _PANEL_DIR / "llm_frames.png"
_CURVES_PNG = _PANEL_DIR / "curves.png"


# --------------------------------------------------------------------------- #
# 通用
# --------------------------------------------------------------------------- #
def _run_cli(extra_args, *, stdin_after_prompt=":quit\n", wait_exit_s=5.0,
             boot_timeout_s=40.0, with_images=False):
    # 硬性规则：任何自动化运行都必须 --headless（绝不允许给用户桌面弹窗）
    cmd = [sys.executable, "-u", "-m", "demo.interactive", "--headless", "--vlm", "mock"]
    if not with_images:
        cmd.append("--no-images")        # 没有图像就没有帧，PNG 自然不会更新
    cmd += list(extra_args)
    proc = subprocess.Popen(cmd, cwd=str(_ROOT), env=_ENV, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    chunks: list = []

    def _drain():
        try:
            fd = proc.stdout.fileno()                  # type: ignore[union-attr]
            while True:
                d = os.read(fd, 4096)
                if not d:
                    break
                chunks.append(d.decode("utf-8", "replace"))
        except Exception:                              # noqa: BLE001
            pass

    th = threading.Thread(target=_drain, daemon=True)
    th.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < boot_timeout_s:
        if _PROMPT in "".join(chunks) or proc.poll() is not None:
            break
        time.sleep(0.05)
    boot = time.monotonic() - t0
    if stdin_after_prompt is not None and proc.poll() is None:
        try:
            proc.stdin.write(stdin_after_prompt)       # type: ignore[union-attr]
            proc.stdin.flush()                         # type: ignore[union-attr]
        except Exception:                              # noqa: BLE001
            pass
    t_quit = time.monotonic()
    exit_after = None
    try:
        proc.wait(timeout=wait_exit_s)
        exit_after = time.monotonic() - t_quit
    except subprocess.TimeoutExpired:
        proc.kill()
    th.join(timeout=2.0)
    try:
        if proc.stdout is not None:
            proc.stdout.close()
    except Exception:                                  # noqa: BLE001
        pass
    return proc, "".join(chunks), exit_after, boot


def _descendants_with_args(pid: int) -> list:
    """只看**本进程**的后代（避免把别的 agent 的 demo.interactive 误判成我的残留）。"""
    try:
        out = subprocess.run(["ps", "-o", "pid=,args=", "--ppid", str(pid)],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:                                  # noqa: BLE001
        return []
    res = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        head, _, rest = line.partition(" ")
        if head.isdigit():
            res.append((int(head), rest))
    return res


def _panel_like(args: str) -> bool:
    """旧的面板子进程入口已删除；这里只兜底匹配任何「面板/子进程」痕迹，避免误伤别人。"""
    a = str(args)
    return ("demo.panels" in a) or ("dsh-panels" in a) or ("panel-child" in a)


try:
    from PIL import Image, ImageDraw                            # noqa: E402
except Exception:                                              # noqa: BLE001
    Image = None
    ImageDraw = None


def _load_png(path: Path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _find_subimage(png: np.ndarray, sub: np.ndarray):
    """在 png 里找 sub 的位置（只试几个合理锚点，精确逐像素相等才算命中）。"""
    H, W = png.shape[:2]
    h, w = sub.shape[:2]
    if h > H or w > W:
        return None
    cands = [(0, 0), (W - w, 0), (0, H - h), (W - w, H - h),
             (w, 0), (0, h), ((W - w) // 2, 0), (0, (H - h) // 2)]
    for (x, y) in cands:
        if x < 0 or y < 0:
            continue
        if np.array_equal(png[y:y + h, x:x + w], sub):
            return (x, y)
    return None


def _decode_payload(url: str):
    m = re.match(r"data:(image/\w+);base64,([A-Za-z0-9+/=\s]+)", str(url), re.S)
    if not m:
        return None, None, None
    raw = base64.b64decode(re.sub(r"\s+", "", m.group(2)))
    try:
        from PIL import Image
        arr = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception:                                  # noqa: BLE001
        arr = None
    return m.group(1), raw, arr


class _CapturingVLM:
    """记录「实际发给大模型」的载荷（base64 图像）与当时 app 记下的帧。"""

    def __init__(self, script, snapshot=None):
        self.script = list(script)
        self.i = 0
        self.payloads: list = []
        self.snapshot = snapshot
        self.moment_frames: dict = {}
        self.moment_png: bytes = b""
        self.moment_pngs: list = []          # 每个请求各存一份（第 i 次 decide 的 PNG）

    def decide(self, messages):
        imgs = []
        for m in messages or []:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        u = part.get("image_url")
                        imgs.append(u.get("url") if isinstance(u, dict) else u)
        self.payloads.append(imgs)
        if callable(self.snapshot):
            try:
                got = self.snapshot()
                if isinstance(got, tuple):
                    frames, png_bytes = got
                else:
                    frames, png_bytes = got, b""
                if frames:
                    self.moment_frames = dict(frames)
                if png_bytes:
                    self.moment_png = bytes(png_bytes)
                    self.moment_pngs.append(bytes(png_bytes))
            except Exception:                          # noqa: BLE001
                pass
        if self.i < len(self.script):
            c = self.script[self.i]
            self.i += 1
            return c
        return SkillCall("stop", {})


def _make_app(*, panels="all", headless=True, with_images=True, perception="vlm", **over):
    # 硬性规则：一律 headless（离屏渲染照常工作，PNG 照常写）
    from demo.interactive import InteractiveApp
    kw = dict(vlm="mock", perception=perception, scene="default", planner=None,
              with_barrier=False, with_images=with_images, cameras=None, realtime=False,
              headless=headless, max_steps=3, seed=0, self_test=False, panels=panels)
    kw.update(over)
    args = argparse.Namespace(**kw)
    return InteractiveApp(args)


# --------------------------------------------------------------------------- #
# 1) 帧一致性（F1 的教训：真的比对字节与像素）
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_FILE_PANELS, _PENDING)
class TestLlmFramesFile(unittest.TestCase):
    """① PNG 里的两张图 == 本次请求实际发给 LLM 的字节解码结果（逐像素）。"""

    def test_png_frames_equal_sent_bytes_pixelwise(self):
        from demo.panels import CAPTION_H
        _PANEL_DIR.mkdir(parents=True, exist_ok=True)
        if _FRAMES_PNG.exists():
            _FRAMES_PNG.unlink()
        holder: dict = {}

        def _snap():
            png = _FRAMES_PNG.read_bytes() if _FRAMES_PNG.exists() else b""
            return holder["app"]._panel_frames, png

        vlm = _CapturingVLM([SkillCall("pick", {"target_pixel": [262, 318],
                                               "pixel_camera": "top"}),
                             SkillCall("stop", {})], snapshot=_snap)
        app = _make_app(panels="all", with_images=True)
        holder["app"] = app
        # 新架构在构造时即启用 PanelFiles 并安装 build_messages 钩子（不再有 _apply_panel_mode）
        self.assertTrue(getattr(app, "_panels_files", None) is not None, "PanelFiles 应已创建")
        self.assertTrue(app._panel_capture, "面板应已启用（_panel_capture=True）")
        app.vlm = vlm
        app._run_agent_loop("把红色方块放到桌子上")
        payload = next((p for p in vlm.payloads if p), [])
        # 必须用**同一个请求**的 PNG：第一次 decide 对应 payloads[0] / moment_pngs[0]
        idx0 = vlm.payloads.index(payload)
        png_bytes = vlm.moment_pngs[idx0] if idx0 < len(vlm.moment_pngs) else b""
        print(f"\n[probe] 载荷图数={len(payload)} 请求序号={idx0} 该请求PNG字节={len(png_bytes)}")
        self.assertGreaterEqual(len(payload), 2, "应至少发两张（overview+top）")
        self.assertTrue(png_bytes, "请求时 llm_frames.png 不存在")
        decoded = [_decode_payload(u) for u in payload[:2]]
        cams = ["overview", "top"]                 # 面板约定顺序：左 overview | 右 top
        from demo.panels import CAPTION_H
        with Image.open(io.BytesIO(png_bytes)) as _im:
            png = np.asarray(_im.convert("RGB"))
        print(f"[probe] 请求时 PNG={png.shape} caption_h={CAPTION_H} "
              f"载荷尺寸={[None if d[2] is None else d[2].shape for d in decoded]}")
        self.assertNotEqual(decoded[0][1], decoded[1][1], "两张载荷不应完全相同")
        x = 0
        for idx, cam in enumerate(cams):
            sub = decoded[idx][2]
            self.assertIsNotNone(sub, f"{cam} 载荷需可解码")
            h, w = sub.shape[:2]
            region = png[CAPTION_H:CAPTION_H + h, x:x + w]
            ok = region.shape == sub.shape and np.array_equal(region, sub)
            print(f"[probe] {cam}: PNG 区域逐像素一致={ok} 区域={region.shape}")
            self.assertTrue(ok, f"{cam} 写进 PNG 的像素 != 发给 LLM 的字节（N1）")
            x += w
        self.assertGreater(CAPTION_H, 0, "caption 带高度应 >0")
    def test_caption_matches_request_payload(self):
        from demo.panels import PanelFiles
        import demo.panels as _pn
        seen: list = []
        real_draw = _pn.ImageDraw.Draw

        class _SpyDraw:
            def __init__(self, *a, **k):
                self._d = real_draw(*a, **k)

            def __getattr__(self, name):
                return getattr(self._d, name)

            def text(self, xy, s, *a, **k):
                seen.append(str(s))
                return self._d.text(xy, s, *a, **k)

        _pn.ImageDraw.Draw = _SpyDraw
        try:
            pf = PanelFiles(out_dir=str(_PANEL_DIR))
            a = np.zeros((8, 16, 3), dtype=np.uint8)
            b = np.ones((8, 16, 3), dtype=np.uint8) * 255
            frames = {"overview": ("raw", 16, 8, a), "top": ("raw", 16, 8, b)}
            meta = {"overview": (16, 8, int(a.nbytes), "raw rgb"),
                    "top": (16, 8, int(b.nbytes), "raw rgb")}
            self.assertTrue(pf.write_frames(frames, meta, step=3))
        finally:
            _pn.ImageDraw.Draw = real_draw
        cap = " | ".join(seen)
        total_kb = (int(a.nbytes) + int(b.nbytes)) / 1024.0
        print(f"\n[probe] caption 文本={cap!r} 期望合计={total_kb:.3f}KB")
        for token in ("step 3", "overview 16x8", "top 16x8", "合计"):
            self.assertIn(token, cap)
        # caption 只保留 1 位小数 -> 容差必须 >=0.05
        self.assertAlmostEqual(float(re.search(r"合计 ([\d.]+) KB", cap).group(1)),
                               total_kb, delta=0.06)
# --------------------------------------------------------------------------- #
# 2) 不建窗 / 不起进程
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_FILE_PANELS, _PENDING)
class TestNoWindowNoProcess(unittest.TestCase):
    def test_inprocess_no_viewer_no_child(self):
        for mod in list(sys.modules):
            if mod.startswith("mujoco.viewer"):
                del sys.modules[mod]
        app = _make_app(panels="all")
        kids = _descendants_with_args(os.getpid())
        panellike = [k for k in kids if _panel_like(k[1])]
        print(f"\n[probe] viewer={'mujoco.viewer' in sys.modules} 面板类子进程={panellike} "
              f"panels_files={type(getattr(app, '_panels_files', None)).__name__}")
        self.assertNotIn("mujoco.viewer", sys.modules, "纯文件架构不得开 MuJoCo 窗口")
        self.assertEqual(panellike, [], "不得起面板子进程")

    def test_cli_panels_all_no_window_no_new_process(self):
        proc, out, exit_after, _boot = _run_cli(["--panels", "all"])
        # 只看**面板类**进程（ps 自己/子进程调度不算），避免误判
        leaked = [k for k in _descendants_with_args(os.getpid()) if _panel_like(k[1])]
        print(f"[probe] CLI --panels all: rc={proc.returncode} exit_after={exit_after} "
              f"面板类残留={leaked}")
        self.assertEqual(proc.returncode, 0)
        self.assertNotIn("正在打开 MuJoCo 窗口", out)
        self.assertEqual(leaked, [], "CLI 不得留下面板类子进程")
        self.assertIsNotNone(exit_after, ":quit 5s 内未退出")
        self.assertLessEqual(exit_after, 5.0)

    def test_headless_panels_all_still_writes_files(self):
        """纯文件架构下，--headless --panels all 也应产出 PNG（无窗口、无进程）。
        当前实现若仍强制 panels=none，这里会红——那是新架构的目标偏差，如实记录。"""
        _PANEL_DIR.mkdir(parents=True, exist_ok=True)
        before = _FRAMES_PNG.stat().st_mtime_ns if _FRAMES_PNG.exists() else 0
        proc, out, exit_after, _boot = _run_cli(["--panels", "all", "--perception", "vlm"],
                                                stdin_after_prompt="把红色方块放到桌上\n:quit\n",
                                                wait_exit_s=60.0,
                                                with_images=True)   # 必须有帧才会写图
        after = _FRAMES_PNG.stat().st_mtime_ns if _FRAMES_PNG.exists() else 0
        print(f"\n[probe] --headless --panels all: rc={proc.returncode} png更新={after != before} "
              f"面板行={'[panel]' in out}")
        self.assertNotIn("正在打开 MuJoCo 窗口", out)
        self.assertTrue(after != before, "headless+panels all 也应写出 llm_frames.png"
                                         "（纯文件架构的意义所在）")

    def test_headless_panels_all_same(self):
        proc, out, exit_after, _boot = _run_cli(["--panels", "all"])
        print(f"[probe] CLI --headless --panels all: rc={proc.returncode} "
              f"exit_after={exit_after}")
        self.assertEqual(proc.returncode, 0)
        self.assertNotIn("正在打开 MuJoCo 窗口", out)
        self.assertIsNotNone(exit_after)
        self.assertLessEqual(exit_after, 5.0)


# --------------------------------------------------------------------------- #
# 3) 曲线：生成 / 非空 / 大小稳定 / 1Hz 节流
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_FILE_PANELS, _PENDING)
class TestCurvesFile(unittest.TestCase):
    """③ curves.png：生成/非空/大小稳定，且 <=1Hz 节流成立。"""

    @staticmethod
    def _samples(n=50):
        return [[i * 0.002, 0.10 * i, ~(i % 3), 0.0, 0.0, -0.5, 0.2, 0.3, 1.5,
                 40.0 + (i % 10)] for i in range(n)]

    def test_write_curves_throttled_and_stable(self):
        from demo.panels import PanelFiles
        _PANEL_DIR.mkdir(parents=True, exist_ok=True)
        if _CURVES_PNG.exists():
            _CURVES_PNG.unlink()
        pf = PanelFiles(out_dir=str(_PANEL_DIR), curve_interval=1.0)
        n0 = pf.curves_writes
        ok1 = pf.write_curves(self._samples())        # 第一次：写
        ok2 = pf.write_curves(self._samples())        # 间隔内：节流
        size1 = _CURVES_PNG.stat().st_size if _CURVES_PNG.exists() else 0
        ok3 = pf.write_curves(self._samples(), force=True)   # force：必写
        ok4 = pf.write_curves(self._samples())        # 又节流
        writes = pf.curves_writes - n0
        size2 = _CURVES_PNG.stat().st_size if _CURVES_PNG.exists() else 0
        print(f"\n[probe] curves.png 存在={_CURVES_PNG.exists()} size={size1}->{size2} "
              f"写次数={writes}（ok1={ok1} ok2={ok2} ok3={ok3} ok4={ok4}）")
        self.assertTrue(_CURVES_PNG.exists(), "curves.png 未生成")
        self.assertGreater(size1, 0, "curves.png 为空")
        self.assertEqual(writes, 2, f"1Hz 节流：4 次调用只应真写 2 次（实测 {writes}）")
        self.assertEqual(size1, size2, "同样数据重复写，文件大小应稳定")

    def test_curves_throttled_under_app_steps(self):
        _PANEL_DIR.mkdir(parents=True, exist_ok=True)
        app = _make_app(panels="all", perception="auto")
        pf = getattr(app, "_panels_files", None)
        self.assertIsNotNone(pf, "PanelFiles 应已创建（非 headless）")
        n0 = pf.curves_writes
        t0 = time.monotonic()
        for i in range(120):
            app.env.data.ctrl[6] = float((i % 10) / 10.0)
            app.env.step(1)
        span = time.monotonic() - t0
        writes = pf.curves_writes - n0
        print(f"[probe] app 步进 {span:.2f}s 内 curves 写 {writes} 次")
        self.assertLessEqual(writes, max(3, int(span) + 2),
                             f"{span:.1f}s 内写 {writes} 次，1Hz 节流不成立")
# --------------------------------------------------------------------------- #
# 4) 行为不变
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_FILE_PANELS, _PENDING)
class TestBehaviorUnchanged(unittest.TestCase):
    @staticmethod
    def _steps(out: str):
        return re.findall(r"^\s*\d+ (OK|FAIL)\s+(\w+)", out, re.M)

    def test_headless_e2e_identical_with_and_without_panels(self):
        _p1, out_all, *_ = _run_cli(["--panels", "all", "--perception", "oracle"],
                                    stdin_after_prompt="把红色方块放到桌上\n:quit\n",
                                    wait_exit_s=60.0)
        _p2, out_none, *_ = _run_cli(["--panels", "none", "--perception", "oracle"],
                                     stdin_after_prompt="把红色方块放到桌上\n:quit\n",
                                     wait_exit_s=60.0)
        s_all, s_none = self._steps(out_all), self._steps(out_none)
        print(f"\n[probe] panels=all steps={s_all}\n[probe] panels=none steps={s_none}")
        self.assertTrue(s_all and s_none, "两次都应有步骤输出")
        self.assertEqual(s_all, s_none, "开启面板不得改变执行步骤")


# --------------------------------------------------------------------------- #
# 5) 退出 / 60s 稳定
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_FILE_PANELS, _PENDING)
class TestExitAndStability(unittest.TestCase):
    def test_quit_within_5s_and_no_leftover(self):
        proc, out, exit_after, boot = _run_cli(["--panels", "all"])
        left = [k for k in _descendants_with_args(os.getpid()) if _panel_like(k[1])]
        print(f"\n[probe] :quit boot={boot:.2f}s exit_after={exit_after} rc={proc.returncode} "
              f"面板类残留={left}")
        self.assertIn(_PROMPT, out)
        self.assertIsNotNone(exit_after, ":quit 后 5s 未退出")
        self.assertLessEqual(exit_after, 5.0)
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(left, [], "退出后有残留面板类子进程")

    def test_60s_no_crash(self):
        cmd = [sys.executable, "-u", "-m", "demo.interactive", "--headless", "--vlm", "mock",
               "--panels", "all", "--no-images"]
        proc = subprocess.Popen(cmd, cwd=str(_ROOT), env=_ENV, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        buf = []

        def _drain():
            try:
                fd = proc.stdout.fileno()              # type: ignore[union-attr]
                while True:
                    d = os.read(fd, 4096)
                    if not d:
                        break
                    buf.append(d.decode("utf-8", "replace"))
            except Exception:                          # noqa: BLE001
                pass

        th = threading.Thread(target=_drain, daemon=True)
        th.start()
        time.sleep(60.0)
        alive = proc.poll() is None
        if alive:
            try:
                proc.stdin.write(":quit\n")            # type: ignore[union-attr]
                proc.stdin.flush()                     # type: ignore[union-attr]
                proc.wait(timeout=5)
            except Exception:                          # noqa: BLE001
                proc.kill()
        th.join(timeout=2.0)
        out = "".join(buf)
        print(f"\n[probe] 60s: 存活={alive} rc={proc.returncode}")
        self.assertTrue(alive, f"60s 内提前退出 rc={proc.returncode}: {out[-300:]}")
        self.assertNotIn(proc.returncode, (-6, -11), "SIGABRT/SIGSEGV")
        self.assertNotIn("SIGABRT", out)
        self.assertNotIn("SIGSEGV", out)



class TestNoWindowedInvocations(unittest.TestCase):
    """硬性规则（用户投诉黑色窗口）：tests/ 里任何 `python -m demo.interactive` 命令行都必须带 --headless。"""

    def test_all_demo_invocations_are_headless(self):
        bad = []
        files = sorted((_ROOT / "tests").glob("*.py")) + [_ROOT / "tests" / "run_acceptance.py"]
        for p in files:
            try:
                lines = p.read_text(encoding="utf-8").splitlines()
            except Exception:                              # noqa: BLE001
                continue
            for i, line in enumerate(lines, 1):
                if "demo.interactive" in line and "-m" in line and "--headless" not in line:
                    bad.append(f"{p.name}:{i}: {line.strip()}")
        print(f"\n[probe] 非 headless 的 demo.interactive 调用={bad}")
        self.assertEqual(bad, [], "所有自动化运行 demo.interactive 必须带 --headless（硬性规则）")

if __name__ == "__main__":
    if not _FILE_PANELS:
        print(f"[PENDING] {_PENDING} —— 全部面板用例 skip（等 ui-fixer 封版）")
    unittest.main(verbosity=2)
