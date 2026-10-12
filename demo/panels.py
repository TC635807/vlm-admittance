"""demo/panels.py —— 纯文件「面板」（不建任何 MuJoCo 窗口、不起子进程）。

用户明确要求：不要直接创建 mujoco 窗口（多窗口只有第一个能渲染，其余全黑），
只需要在「给 LLM 发消息的时候」把那一帧写出来即可，不需要实时画面。

产出
----
* outputs/panels/llm_frames.png —— 每次构造 VLM 请求前，把这次实际发给模型的两帧
  （overview + top，按发送时那份 JPEG 字节解码）并排拼图，顶部一条 caption。
  原子写（先写 .tmp 再 os.replace），约 10ms，不阻塞任务。
* outputs/panels/curves.png     —— 6 关节力矩 + 夹爪执行器力 + 末端接触力模长 + 指间距(mm)，
  最近 10s 滚动窗；matplotlib Agg 出图，节流 <=1Hz（任务结束 / :panels shot 强制补一张）。
* outputs/panels/index.html     —— 纯静态页，JS 每 500ms 刷新两张图（可选，不需要服务器）。
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - 无 PIL 时退化为纯 numpy 拼图
    Image = ImageDraw = None

try:
    import matplotlib
    matplotlib.use("Agg")                       # 无交互后端，只保存 PNG
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover
    Figure = None

#: 相机图上方的 caption 高度（像素）
CAPTION_H = 28
#: 曲线节流间隔（秒）
CURVE_INTERVAL_S = 1.0
#: 曲线滚动窗（秒）
CURVE_WINDOW_S = 10.0


class PanelFiles:
    """把发给 LLM 的帧与遥测曲线写成 PNG；不建窗口、不起进程、绝不阻塞物理。"""

    def __init__(self, out_dir: Optional[str] = None, *, log: Any = None,
                 curve_interval: float = CURVE_INTERVAL_S) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dir = str(out_dir or os.path.join(root, "outputs", "panels"))
        self.frames_path = os.path.join(self.dir, "llm_frames.png")
        self.curves_path = os.path.join(self.dir, "curves.png")
        self.index_path = os.path.join(self.dir, "index.html")
        self.log = log if callable(log) else (lambda _m: None)
        self.curve_interval = float(curve_interval)
        self.frames_writes = 0
        self.curves_writes = 0
        self.frames_at = 0.0
        self.curves_at = 0.0
        self.last_error = ""
        self._last_frames: dict = {}
        self._last_meta: dict = {}
        self._step = 0
        self._curve_t: list = []
        self._curve_y: dict = {}
        self._fig = None
        self._ax = None
        try:
            os.makedirs(self.dir, exist_ok=True)
            self.write_index()
        except Exception as exc:                # noqa: BLE001
            self.last_error = str(exc)

    # -- 工具 -------------------------------------------------------------- #
    def _atomic_save_png(self, img, path: str) -> bool:
        tmp = path + ".tmp"
        try:
            img.save(tmp, format="PNG")
            os.replace(tmp, path)
            return True
        except Exception as exc:                # noqa: BLE001
            self.last_error = "%s: %s" % (type(exc).__name__, exc)
            try:
                os.unlink(tmp)
            except Exception:                   # noqa: BLE001
                pass
            return False

    @staticmethod
    def _decode(item):
        """(fmt, w, h, payload) -> uint8 RGB；jpeg 时用 PIL 解码发送时的原始字节。"""
        try:
            fmt, w, h, payload = item
        except Exception:                       # noqa: BLE001
            return None
        if fmt == "jpeg" and Image is not None:
            try:
                import io
                return np.asarray(Image.open(io.BytesIO(bytes(payload))).convert("RGB"))
            except Exception:                   # noqa: BLE001
                return None
        try:
            arr = np.asarray(payload, dtype=np.uint8)
            return arr if arr.ndim == 3 else None
        except Exception:                       # noqa: BLE001
            return None

    # -- 帧：发给 LLM 的那两帧 ------------------------------------------------ #
    def write_frames(self, frames: dict, meta: dict, *, step: Optional[int] = None) -> bool:
        """并排写出这次发给模型的两帧 + caption；原子写，约 10ms。"""
        if not frames:
            return False
        self._last_frames.update(dict(frames))
        self._last_meta.update(dict(meta or {}))
        self._step = int(step) if step is not None else self._step + 1
        imgs = {cam: self._decode(item) for cam, item in self._last_frames.items()}
        imgs = {k: v for k, v in imgs.items() if v is not None}
        if not imgs or Image is None:
            return False
        cams = [c for c in ("overview", "top") if c in imgs] or sorted(imgs)
        h = max(int(imgs[c].shape[0]) for c in cams)
        w = sum(int(imgs[c].shape[1]) for c in cams)
        canvas = np.zeros((h + CAPTION_H, w, 3), dtype=np.uint8)
        x = 0
        for c in cams:
            a = imgs[c]
            canvas[CAPTION_H:CAPTION_H + a.shape[0], x:x + a.shape[1]] = a
            x += a.shape[1]
        bits, total, fmt_seen = [], 0, "raw rgb"
        for c in cams:
            m = self._last_meta.get(c) or ()
            if len(m) == 4:
                bits.append("%s %dx%d" % (c, m[0], m[1]))
                total += int(m[2])
                if "jpeg" in str(m[3]):
                    fmt_seen = "jpeg q85"
        caption = ("step %d | %s | %s | %s | 合计 %.1f KB"
                   % (self._step, time.strftime("%H:%M:%S"),
                      " + ".join(bits) or "(no meta)", fmt_seen, total / 1024.0))
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, w - 1, CAPTION_H - 1], fill=(18, 18, 22))  # 端点含在内，别盖到第 1 行图
        draw.text((6, 7), caption, fill=(235, 235, 240))
        ok = self._atomic_save_png(img, self.frames_path)
        if ok:
            self.frames_writes += 1
            self.frames_at = time.time()
        return ok

    # -- 曲线 --------------------------------------------------------------- #
    def _ensure_fig(self):
        if self._fig is not None or Figure is None:
            return self._fig
        self._fig = Figure(figsize=(10, 6), dpi=90)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlabel("t - now (s)")
        self._ax.set_title("joint torque / gripper force / jaw gap (10s window)")
        self._ax.grid(True, alpha=0.3)
        return self._fig

    def write_curves(self, samples: list, *, force: bool = False) -> bool:
        """最近 10s 的曲线；节流 <=1Hz（force=True 时立即出图）。"""
        now = time.time()
        if not force and (now - self.curves_at) < self.curve_interval:
            return False
        rows = list(samples or [])
        if not rows:
            return False
        fig = self._ensure_fig()
        if fig is None:
            return False
        t_end = float(rows[-1][0])
        win = [r for r in rows if t_end - float(r[0]) <= CURVE_WINDOW_S] or rows
        xs = [float(r[0]) - t_end for r in win]
        series = [("j0..j5 [Nm]", [1, 2, 3, 4, 5, 6]),
                  ("grip act [N]", [7]),
                  ("|F_ee| [N]", [8]),
                  ("jaw gap [mm]", [9])]
        self._ax.clear()
        self._ax.grid(True, alpha=0.3)
        for name, idxs in series:
            for k in idxs:
                ys = [max(1e-9, abs(float(r[k]))) if name.startswith("j") else float(r[k])
                      for r in win]
                self._ax.plot(xs, ys, linewidth=1.0, label=("%s" % name) if k == idxs[0] else None)
        self._ax.set_xlabel("t - now (s)")
        self._ax.set_title("6 joint torques + gripper force + jaw gap (10s)")
        self._ax.legend(loc="upper right", fontsize=8)
        try:
            tmp = self.curves_path + ".tmp"
            fig.savefig(tmp, format="png", bbox_inches="tight")
            os.replace(tmp, self.curves_path)
        except Exception as exc:                # noqa: BLE001
            self.last_error = "%s: %s" % (type(exc).__name__, exc)
            return False
        self.curves_writes += 1
        self.curves_at = now
        return True

    # -- 静态页与状态 ------------------------------------------------------- #
    def write_index(self) -> None:
        html = ("<!doctype html><meta charset=utf-8><title>vlm_admittance panels</title>\n"
                "<body style='margin:0;background:#111;color:#ddd;font:13px sans-serif'>\n"
                "<div style='padding:6px'>llm_frames.png / curves.png（每 500ms 自动刷新）</div>\n"
                "<img id=f style='width:48vw'> <img id=c style='width:48vw'>\n"
                "<script>function r(){var q='?t='+Date.now();"
                "document.getElementById('f').src='llm_frames.png'+q;"
                "document.getElementById('c').src='curves.png'+q;}"
                "r();setInterval(r,500);</script></body>\n")
        try:
            os.makedirs(self.dir, exist_ok=True)
            tmp = self.index_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(html)
            os.replace(tmp, self.index_path)
        except Exception as exc:                # noqa: BLE001
            self.last_error = "%s: %s" % (type(exc).__name__, exc)

    @staticmethod
    def _ago(ts: float) -> str:
        return "从未" if not ts else ("%.1fs 前" % (time.time() - ts))

    def status(self) -> str:
        return ("frames: %s（写 %d 次，%s） | curves: %s（写 %d 次，%s） | index: %s"
                % (self.frames_path, self.frames_writes, self._ago(self.frames_at),
                   self.curves_path, self.curves_writes, self._ago(self.curves_at),
                   self.index_path))
