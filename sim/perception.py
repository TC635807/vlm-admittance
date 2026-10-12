"""sim/perception.py —— 纯图像感知工具（INTERFACES.md §5.1 冻结契约 v2）。

职责
----
给「无物体真值」的 vlm 闭环提供图像侧能力：
    color_blob   —— HSV 阈值 + 连通域，在渲染图里找某颜色的色块（质心/框/面积/置信）
    refine_pixel —— 把 VLM 给的粗像素精修到色块质心（失败必须原样返回粗像素）
    zoom_window / crop_upscale / map_zoom_uv —— 局部放大，便于二次识别或给模型看细节
    bbox_to_world / estimate_size —— 框四角经相机射线交桌面 -> 世界 AABB / 尺寸
    check_inside / check_stacked / check_gone —— 子目标几何复核（放进容器/叠放/已抓走）

**信息隔离红线（强制）**：本模块只允许输入「渲染 RGB + sim/vision.py 的相机标定 + VLM 给的
像素/框」。严禁调用 env.object_pose / env.set_object_pose / state.objects[*].pose，
也不读 DEFAULT_OBJECTS 之类的真值布局。整个文件不出现任何读取物体真值的调用。

原理与实现要点
--------------
1) RGB->HSV 纯 numpy 手写（不做除法告警：分母用 np.where 保护）。H 用度 (0..360)，
   S/V 用 0..1。颜色表见 COLOR_RULES，边界按本机渲染实测调过：
   box_red 渲染 (255,108,98)->H=3.8; box_blue (115,188,255)->H=208.7、(33,77,186)->H=222.7;
   box_green (126,255,167)->H=139、(30,139,49)->H=130.5; can_yellow (211,188,36)->H=52;
   cyan 目标色 (0.25,0.68,0.88) 渲染后 H≈197；黑盒 (0.10,0.10,0.12) -> V≈0.12。
   桌面木色 (0.62,0.48,0.32) 渲染后 H≈30、S≈0.43，因此 orange 的 s_min 取 0.55 把它排除；
   机械臂 urblue 渲染后 H≈203/S≈0.39 会落进 blue（用 hint_uv 局部化规避）；托盘高光接近纯白。
2) 连通域：**逐行 RLE + 并查集**（4 邻域），复杂度 O(像素 + 段数·α)，全程向量化聚合，
   不做任何 O(n^2) 的成对比较，也不依赖 cv2/PIL/scipy。统计（面积/质心/框/纯度）
   直接在「段」上做 bincount，不需要物化 label 图。
3) 反投影只用 sim/vision.py 的公开 API（CameraModel.ray / pixel_to_plane），
   本文件不导入 mujoco 的相机内部量、不硬编码相机参数。

关键约定（调用方按此对齐）
--------------------------
* 像素坐标 u 向右、v 向下，原点左上；bbox = (u1, v1, u2, v2) 为**闭区间**像素索引。
* color_blob 找不到时返回 found=False，uv=(nan,nan)、bbox=(0,0,0,0)、area_px=0、score=0；
  绝不抛异常（形状/类型异常也只会 found=False，只有完全非数组输入才会 ValueError）。
* score = clamp(纯度 * 面积占比, 0, 1)：纯度 = 该连通域像素的 mean(S*V)，
  面积占比 = min(1, area / 搜索窗面积)（有 hint 时窗=π·radius_px²，无 hint 时窗=H*W）。
* 有 hint_uv 时只在「距离 hint <= radius_px 的连通域」里挑，排名 = 面积 / (1 + (距离/半径)^2)
  （既偏心 hints 又偏心大块，避免近处噪声小点压过真实色块）。
* 所有函数对越界像素 / 单色图 / 噪声图 / 空输入都不崩。

双目三角定位（§5.10 —— 解决俯视视差）
--------------------------------------
top 相机**不是正交俯视**：物体顶面的像素若按 z=TABLE_HEIGHT 反投影，会有系统性视差
    Δ_xy ~= |p_top - cam|_xy * (z_top - z_plane) / (cam_z - z_top)
实测：40 mm 方块 6~9 mm；水瓶（高 200 mm、顶面 z=0.60）在 (0.45,-0.18) 处 **68.6 mm**
（比瓶半径 32 mm 还大），且与分辨率无关。所以高物体必须用第二个视角：
* triangulate(env, uv_a, cam_a, uv_b, cam_b)：两条像素射线的最近点，
  residual_mm > 15 视为不可信（ok=False）；
* object_point(env, top_uv=..., overview_uv=..., z_hint=...)：
  双像素优先三角（source="triangulate"），只有单目时按 z_hint（缺省桌面）平面反投影
  （source="single_view"）；双像素但残差过大时**不静默回退**、直接 ok=False 供上层重试；
* height_from_top_z(z_top) = z_top - TABLE_HEIGHT，供侧抓高度 / 叠放高度估计；
* bbox_to_world(..., z=None) 缺省按 TABLE_HEIGHT 反投影并返回 assumed_z=True。

已知限制（Lead 已接受，勿当 bug）
--------------------------------
* check_stacked 的下层框用"与上层框相邻的下层连通块并框"抗遮挡；若上层把下层**完全盖住
  且无任何露出**，下层色块找不到 -> 诚实返回 ok=False（由 §5.2 的 VLM 复核兜底），绝不谎报成功。
* 颜色干扰项（T7 实测与对策）：
  - 桌面木色在 top 图里 H=42~46 / S=0.353~0.384，曾把整块桌面判成"黄"（17030 px）：
    已把 yellow 的 s_min 从 0.35 收紧到 0.42（真黄罐 S=0.44~0.49）。
  - 托盘（灰）在 top 图里大面积高光过曝成纯白，剩下的灰色只有一圈**填充率 0.089** 的薄壁：
    对 gray/white/black 加「致密度 >= 0.40」硬门槛，于是托盘老实返回 found=False
    （refined=False），由上层用 VLM 的 bbox（locate）做几何判定。
  - 机械臂 urblue（H≈203 / S≈0.39）会落进 blue；桌面木色 H≈30 / S≈0.43 已用 orange
    的 s_min=0.55 排除。调用方用 hint_uv 局部化 + 60 px 搜索半径即可。
  - 另外统一忽略「面积 > 25% 画面」的连通块，避免整片背景/桌面被当成目标。
  - gray/white/black 在 **hint_uv=None（全图搜索）时一律 found=False**：无局部化先验时
    必然给错点（实测 tray (264,333) vs 真值 (226.7,422.3) = 99.58 px / 145 mm），宁可诚实拒绝。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from core.config import TABLE_HEIGHT

from .vision import CameraModel

# --------------------------------------------------------------------------- #
# 颜色表
# --------------------------------------------------------------------------- #
#: 颜色 -> HSV 阈值。h 是 [(lo, hi)]，lo > hi 表示跨 0 度；s/v 为上下限（含）。
#: 仅做粗分类，不追求色卡精度。
COLOR_RULES: dict = {
    "red":    {"h": [(345.0, 15.0)],  "s_min": 0.35, "v_min": 0.25},
    "orange": {"h": [(15.0, 42.0)],   "s_min": 0.55, "v_min": 0.45},
    # 黄色 s_min 从 0.35 收到 0.42（T7 实测）：桌面木色在 top 图里 H=42~46 / S=0.353~0.384，
    # 会把整块桌面判成"黄"（最大连通块 17030 px）；真黄罐 H=57~60 / S=0.44~0.49。
    # 0.42 落在两者之间，既排除桌面又保留罐身高光。
    "yellow": {"h": [(42.0, 70.0)],   "s_min": 0.42, "v_min": 0.40},
    "green":  {"h": [(70.0, 170.0)],  "s_min": 0.25, "v_min": 0.25},
    "cyan":   {"h": [(170.0, 200.0)], "s_min": 0.20, "v_min": 0.30},
    "blue":   {"h": [(200.0, 270.0)], "s_min": 0.30, "v_min": 0.20},
    "white":  {"h": [(0.0, 360.0)],   "s_max": 0.16, "v_min": 0.78},
    "gray":   {"h": [(0.0, 360.0)],   "s_max": 0.22, "v_min": 0.18, "v_max": 0.78},
    "black":  {"h": [(0.0, 360.0)],   "v_max": 0.18},
}

#: 支持的颜色的稳定顺序
COLOR_NAMES: tuple = tuple(COLOR_RULES.keys())

#: 双目三角的最大可信残差（mm），超过判 ok=False（§5.10）
MAX_TRIANGULATE_RESIDUAL_MM: float = 15.0

# --------------------------------------------------------------------------- #
# 候选筛选/打分（T7 修正）
# --------------------------------------------------------------------------- #
#: 单个连通块面积超过画面这个比例就直接忽略（多半是桌面/背景整片误检）
MAX_BLOB_AREA_FRAC: float = 0.25
#: 「与桌面/背景同色」的低饱和类：掩码容易被阴影/高光/背景撕成碎片，候选必须足够致密
LOW_SAT_COLORS: tuple = ("gray", "white", "black")
#: 低饱和类的最小致密度 = 面积 / 外接框面积。实测：托盘灰壁 0.089、托盘前缘条带 0.49、
#: 托盘过曝白面 0.999、眼镜盒 0.97、机械臂暗部 0.39 —— 取 0.55 可把托盘的灰色候选全部挡掉
#: （托盘是空心件，其灰壁天然不紧凑），同时保留眼镜盒这类实心块。
MIN_COMPACTNESS_LOW_SAT: float = 0.55
#: 打分里的"理想面积" = 该比例 × 搜索窗面积（hint 半径 60 px 时约 17 px 见方）
AREA_IDEAL_FRAC: float = 0.15


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class BlobResult:
    """一个颜色连通域。bbox 为闭区间像素索引 (u1, v1, u2, v2)。"""

    found: bool
    uv: tuple = (float("nan"), float("nan"))
    bbox: tuple = (0, 0, 0, 0)
    area_px: int = 0
    color: str = ""
    score: float = 0.0

    def to_json(self) -> dict:
        return {
            "found": bool(self.found),
            "uv": [None if not np.isfinite(v) else round(float(v), 2) for v in self.uv],
            "bbox": [int(v) for v in self.bbox],
            "area_px": int(self.area_px),
            "color": self.color,
            "score": round(float(self.score), 4),
        }


_NOT_FOUND = BlobResult(found=False)


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _as_rgb(rgb) -> np.ndarray:
    """统一成 (H, W, 3) float32，取值 0..255。float 输入若 max<=1 视为 0..1。"""
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("rgb 需要 (H, W, 3) 或 (H, W)，收到 shape=%r" % (arr.shape,))
    arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return arr.astype(np.float32, copy=False)
    out = arr.astype(np.float32, copy=False)
    if out.size and float(np.nanmax(out)) <= 1.0 + 1e-6:
        out = out * 255.0
    return out


def _rgb_to_hsv(rgb: np.ndarray):
    """RGB(0..255) -> (h 度 0..360, s 0..1, v 0..1)，纯 numpy。"""
    img = rgb / 255.0
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    safe_diff = np.where(diff > 0.0, diff, 1.0)
    # 色相
    rc = (mx - r) / safe_diff
    gc = (mx - g) / safe_diff
    bc = (mx - b) / safe_diff
    hue = np.where(mx == r, bc - gc, np.where(mx == g, 2.0 + rc - bc, 4.0 + gc - rc)) * 60.0
    hue = np.mod(hue, 360.0)
    hue = np.where(diff > 0.0, hue, 0.0)
    sat = np.where(mx > 0.0, diff / np.where(mx > 0.0, mx, 1.0), 0.0)
    return hue.astype(np.float32), sat.astype(np.float32), mx.astype(np.float32)


def color_mask(rgb, color: str) -> np.ndarray:
    """返回该颜色的布尔掩码 (H, W)。未知颜色返回全 False（不抛异常）。"""
    rule = COLOR_RULES.get(str(color))
    if rule is None:
        return np.zeros(np.asarray(rgb).shape[:2], dtype=bool)
    h, s, v = _rgb_to_hsv(_as_rgb(rgb))
    mask = np.zeros(h.shape, dtype=bool)
    for lo, hi in rule["h"]:
        if lo <= hi:
            mask |= (h >= lo) & (h <= hi)
        else:                                   # 跨 0 度（red）
            mask |= (h >= lo) | (h <= hi)
    if "s_min" in rule:
        mask &= s >= float(rule["s_min"])
    if "s_max" in rule:
        mask &= s <= float(rule["s_max"])
    if "v_min" in rule:
        mask &= v >= float(rule["v_min"])
    if "v_max" in rule:
        mask &= v <= float(rule["v_max"])
    return mask


# --------------------------------------------------------------------------- #
# 连通域：逐行 RLE + 并查集（4 邻域）
# --------------------------------------------------------------------------- #
def _row_runs(mask: np.ndarray):
    """逐行游程编码。返回 (rows, starts, ends, row_ptr)，ends 为开区间。"""
    h, w = mask.shape
    rows_l, starts_l, ends_l, counts = [], [], [], np.zeros(h + 1, dtype=np.int64)
    for v in range(h):
        row = mask[v]
        if not row.any():
            counts[v + 1] = counts[v]
            continue
        padded = np.zeros(w + 2, dtype=np.int8)
        padded[1:w + 1] = row
        d = np.diff(padded)
        s = np.flatnonzero(d == 1)
        e = np.flatnonzero(d == -1)
        rows_l.append(np.full(s.size, v, dtype=np.int64))
        starts_l.append(s.astype(np.int64))
        ends_l.append(e.astype(np.int64))
        counts[v + 1] = counts[v] + s.size
    if not rows_l:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, counts
    return (np.concatenate(rows_l), np.concatenate(starts_l),
            np.concatenate(ends_l), counts)


def _components(mask: np.ndarray):
    """4 邻域连通域标记（段级并查集）。

    返回 (info, stats)：
      info = (rows, starts, ends, comp)  每个段的行/起/止/所属连通域 id
      stats = dict(area, u, v, u1, u2, v1, v2, purity)  逐连通域的统计
    复杂度 O(像素 + 段数·α)，无成对比较。
    """
    rows, starts, ends, row_ptr = _row_runs(mask)
    r = rows.size
    if r == 0:
        return (rows, starts, ends, np.zeros(0, dtype=np.int64)), None

    parent = list(range(r))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for v in range(1, mask.shape[0]):
        # row_ptr[i] = 行 < i 的段数，所以第 i 行的段区间是 [row_ptr[i], row_ptr[i+1])
        a0, a1 = int(row_ptr[v - 1]), int(row_ptr[v])     # 上一行段
        b0, b1 = int(row_ptr[v]), int(row_ptr[v + 1])     # 当前行段
        if a0 == a1 or b0 == b1:
            continue
        j = a0
        for ci in range(b0, b1):
            cs, ce = int(starts[ci]), int(ends[ci])
            while j < a1 and int(ends[j]) <= cs:
                j += 1
            k = j
            while k < a1 and int(starts[k]) < ce:
                ra, rb = find(ci), find(k)
                if ra != rb:
                    parent[rb] = ra
                k += 1

    roots = np.fromiter((find(i) for i in range(r)), dtype=np.int64, count=r)
    _, comp = np.unique(roots, return_inverse=True)
    comp = comp.astype(np.int64)

    n = int(comp.max()) + 1
    length = (ends - starts).astype(np.float64)
    area = np.bincount(comp, weights=length, minlength=n)
    u_sum = ((starts + ends - 1).astype(np.float64) * length) / 2.0
    v_sum = rows.astype(np.float64) * length
    u_tot = np.bincount(comp, weights=u_sum, minlength=n)
    v_tot = np.bincount(comp, weights=v_sum, minlength=n)

    u1 = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    u2 = np.full(n, -1, dtype=np.int64)
    v1 = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    v2 = np.full(n, -1, dtype=np.int64)
    np.minimum.at(u1, comp, starts)
    np.maximum.at(u2, comp, ends - 1)
    np.minimum.at(v1, comp, rows)
    np.maximum.at(v2, comp, rows)

    stats = {
        "area": area,
        "u1": u1, "u2": u2, "v1": v1, "v2": v2,
        "cu": u_tot / np.maximum(area, 1.0),
        "cv": v_tot / np.maximum(area, 1.0),
    }
    return (rows, starts, ends, comp), stats


def _component_purity(rgb: np.ndarray, info, stats) -> np.ndarray:
    """逐连通域的 mean(S*V)（颜色纯度）。用行前缀和按「段」求，避免物化 label 图。"""
    rows, starts, ends, comp = info
    if rows.size == 0 or stats is None:
        return np.zeros(0, dtype=np.float64)
    _, s, v = _rgb_to_hsv(rgb)
    w_sv = s * v
    h, w = w_sv.shape
    csum = np.concatenate([np.zeros((h, 1), dtype=np.float64),
                            np.cumsum(w_sv, axis=1, dtype=np.float64)], axis=1)
    run_sum = csum[rows, ends] - csum[rows, starts]
    n = stats["area"].size
    tot = np.bincount(comp, weights=run_sum, minlength=n)
    return tot / np.maximum(stats["area"], 1.0)


def _blobs_of_color(rgb, color, *, min_area_px: int, hint_uv=None, radius_px: float = 60.0):
    """返回 (candidates, all_stats)。candidates 为 (BlobResult, rank, dist) 列表。

    候选筛选（T7）：
      1) area >= min_area_px；
      2) area <= MAX_BLOB_AREA_FRAC * 画面像素（排除整片桌面/背景误检）；
      3) 低饱和类（gray/white/black）还要求 面积/外接框 >= MIN_COMPACTNESS_LOW_SAT
         （这类颜色与桌面、阴影、高光同色，碎片化的掩码不可信 -> 宁可找不到）；
      4) hint_uv 给定时只保留「有像素距 hint <= radius_px」的连通域。
    打分：rank = area × near × area_score × compact
      near = 1/(1+(d/radius)^2)（离 hint 近）；area_score = 1/(1+|ln(area/ideal)|)
      （面积合理，ideal = AREA_IDEAL_FRAC × 搜索窗面积）；compact = 面积/外接框（致密）。
    """
    mask = color_mask(rgb, color)
    if not mask.any():
        return [], None
    info, stats = _components(mask)
    if stats is None:
        return [], None
    purity = _component_purity(rgb, info, stats)
    area = stats["area"]
    bbox_area = ((stats["u2"] - stats["u1"] + 1).astype(np.float64)
                 * (stats["v2"] - stats["v1"] + 1).astype(np.float64))
    compact = area / np.maximum(bbox_area, 1.0)

    keep = area >= float(min_area_px)
    keep &= area <= float(MAX_BLOB_AREA_FRAC) * float(mask.size)
    if str(color) in LOW_SAT_COLORS:
        keep &= compact >= float(MIN_COMPACTNESS_LOW_SAT)
    if not keep.any():
        return [], stats

    dist = None
    if hint_uv is not None:
        rows, starts, ends, comp = info
        hu, hv = float(hint_uv[0]), float(hint_uv[1])
        dx = np.where((hu >= starts) & (hu < ends), 0.0,
                      np.minimum(np.abs(starts - hu), np.abs(ends - 1 - hu)))
        dy = np.abs(rows - hv)
        run_d = np.sqrt(dx * dx + dy * dy)
        dist = np.full(area.size, np.inf, dtype=np.float64)
        np.minimum.at(dist, comp, run_d)
        keep &= np.isfinite(dist) & (dist <= float(radius_px))
        if str(color) in LOW_SAT_COLORS:
            # 低饱和类再加一条：VLM 点的像素必须落在候选外接框内（VLM 点的是目标身上的点）。
            # 实测作用：daily 里托盘的 hint 会"蹭"到 58 px 外眼镜盒的灰块（85 mm 错点），
            # 眼镜盒自身的黑色又只剩一条 2 px 薄的边缘 -> 两条都会被这条规则挡成 found=False。
            keep &= ((hu >= stats["u1"]) & (hu <= stats["u2"])
                     & (hv >= stats["v1"]) & (hv <= stats["v2"]))
        if not keep.any():
            return [], stats

    window = np.pi * float(radius_px) ** 2 if hint_uv is not None else float(mask.size)
    ideal = max(1.0, float(AREA_IDEAL_FRAC) * float(window))
    out = []
    for cid in np.flatnonzero(keep):
        a = float(area[cid])
        p = float(np.clip(purity[cid], 0.0, 1.0))
        ratio = float(min(1.0, a / max(window, 1.0)))
        d = float(dist[cid]) if dist is not None else 0.0
        near = 1.0 if dist is None else 1.0 / (1.0 + (d / max(float(radius_px), 1e-9)) ** 2)
        area_score = 1.0 if dist is None else 1.0 / (1.0 + abs(float(np.log(max(a, 1.0) / ideal))))
        rank = a * near * area_score * float(compact[cid])
        blob = BlobResult(
            found=True,
            uv=(float(stats["cu"][cid]), float(stats["cv"][cid])),
            bbox=(int(stats["u1"][cid]), int(stats["v1"][cid]),
                  int(stats["u2"][cid]), int(stats["v2"][cid])),
            area_px=int(round(a)),
            color=str(color),
            score=float(np.clip(p * ratio, 0.0, 1.0)),
        )
        out.append((blob, rank, d))
    out.sort(key=lambda item: item[1], reverse=True)
    return out, stats


# --------------------------------------------------------------------------- #
# 冻结接口：色块
# --------------------------------------------------------------------------- #
def color_blob(rgb, color: str, hint_uv=None, *, radius_px: float = 60,
               min_area_px: int = 25) -> BlobResult:
    """在 RGB 图里找指定颜色的连通域，返回 BlobResult。

    hint_uv 给定时：只在该像素 radius_px 内（T7 从 110 px 收紧到 60 px）的候选里选，
    排名 = 面积 ×「离 hint 近」×「面积合理」×「致密」（见 _blobs_of_color）。
    面积 > 25% 画面的连通块直接忽略；gray/white/black 这类与桌面同色的目标还要求
    致密度 >= 0.55 且 hint 落在候选外接框内，碎片化的掩码宁可找不到（返回 found=False，
    由上层用 VLM bbox 做几何判定）。

    **hint_uv=None 时 gray/white/black 一律直接返回 found=False**（Lead 裁决）：这三个颜色
    与桌面/背景/阴影同色，没有局部化先验的全图搜索不可能可靠分割，必然给错点
    （实测 tray 无 hint -> (264,333)，真值 (226.7,422.3)，d_px=99.58 / 144.99 mm）。
    宁可诚实拒绝，也不要给错点。任何情况都不抛异常。

    注：内部函数 _blobs_of_color 不带这条限制（check_stacked 需要无 hint 枚举下层色块），
    本限制只加在公开接口 color_blob 上。
    """
    if hint_uv is not None:
        try:
            hu, hv = float(hint_uv[0]), float(hint_uv[1])
            if not (np.isfinite(hu) and np.isfinite(hv)):
                hint_uv = None
        except Exception:
            hint_uv = None
    if hint_uv is None and str(color) in LOW_SAT_COLORS:
        return BlobResult(found=False, color=str(color))
    try:
        cands, _ = _blobs_of_color(rgb, color, min_area_px=int(min_area_px),
                                   hint_uv=hint_uv, radius_px=float(radius_px))
    except Exception:
        return _NOT_FOUND
    if not cands:
        return BlobResult(found=False, color=str(color))
    return cands[0][0]


def refine_pixel(rgb, color: str, hint_uv, *, camera: str = "top", env=None) -> dict:
    """VLM 粗像素 -> 颜色质心精修。

    返回 {"uv","bbox","refined","source","delta_px","metrics"}。
    **refined=False 时 uv 与 hint_uv 完全相等**（原样返回，调用方可安全回退）。
    """
    try:
        hu, hv = float(hint_uv[0]), float(hint_uv[1])
    except Exception:
        raise ValueError("hint_uv 需要 (u, v)，收到 %r" % (hint_uv,))
    metrics = {"camera": str(camera), "color": str(color), "hint_uv": (hu, hv)}
    if env is not None:
        size = getattr(env, "render_size", None)
        if size:
            metrics["resolution"] = (int(size[0]), int(size[1]))

    blob = color_blob(rgb, color, (hu, hv))
    if not blob.found:
        metrics.update({"area_px": 0, "score": 0.0, "reason": "color_not_found"})
        return {"uv": (hu, hv), "bbox": (0, 0, 0, 0), "refined": False,
                "source": "vlm_raw", "delta_px": 0.0, "metrics": metrics}

    u, v = float(blob.uv[0]), float(blob.uv[1])
    delta = float(np.hypot(u - hu, v - hv))
    metrics.update({"area_px": int(blob.area_px), "score": float(blob.score),
                    "blob_bbox": tuple(int(x) for x in blob.bbox)})
    return {"uv": (u, v), "bbox": tuple(int(x) for x in blob.bbox), "refined": True,
            "source": "color_blob", "delta_px": delta, "metrics": metrics}


# --------------------------------------------------------------------------- #
# 冻结接口：放大
# --------------------------------------------------------------------------- #
def zoom_window(uv, *, size_px: int = 160, scale: int = 4) -> dict:
    """以 uv 为中心取 size_px x size_px 窗口，放大 scale 倍。

    返回 {"u0","v0","size","scale","uv_zoom"}；u0/v0 是窗口左上角在原图的整数坐标
    （可为负，crop_upscale 会做边缘填充）；uv_zoom = ((u-u0)*scale, (v-v0)*scale)。
    """
    size = max(1, int(size_px))
    sc = max(1, int(scale))
    try:
        u, v = float(uv[0]), float(uv[1])
    except Exception:
        raise ValueError("uv 需要 (u, v)，收到 %r" % (uv,))
    u0 = int(np.floor(u - size / 2.0 + 0.5))
    v0 = int(np.floor(v - size / 2.0 + 0.5))
    return {"u0": u0, "v0": v0, "size": size, "scale": sc,
            "uv_zoom": ((u - u0) * sc, (v - v0) * sc)}


def crop_upscale(rgb, win) -> np.ndarray:
    """按 zoom_window 的窗口裁剪并双线性放大；越界用边缘像素填充。"""
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("rgb 需要 (H, W, 3) 或 (H, W)，收到 shape=%r" % (arr.shape,))
    img = arr[:, :, :3]
    h, w = img.shape[:2]
    size = max(1, int(win.get("size", 160)))
    scale = max(1, int(win.get("scale", 4)))
    u0, v0 = int(win.get("u0", 0)), int(win.get("v0", 0))
    out_n = size * scale

    ys = (np.arange(out_n, dtype=np.float64) + 0.5) / scale - 0.5 + v0
    xs = (np.arange(out_n, dtype=np.float64) + 0.5) / scale - 0.5 + u0
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy = ys - y0
    fx = xs - x0
    y0c = np.clip(y0, 0, max(h - 1, 0))
    y1c = np.clip(y0 + 1, 0, max(h - 1, 0))
    x0c = np.clip(x0, 0, max(w - 1, 0))
    x1c = np.clip(x0 + 1, 0, max(w - 1, 0))

    imgf = img.astype(np.float32, copy=False)
    top = imgf[y0c][:, x0c] * (1.0 - fx)[None, :, None] + imgf[y0c][:, x1c] * fx[None, :, None]
    bot = imgf[y1c][:, x0c] * (1.0 - fx)[None, :, None] + imgf[y1c][:, x1c] * fx[None, :, None]
    out = top * (1.0 - fy)[:, None, None] + bot * fy[:, None, None]
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(np.rint(out), 0, 255).astype(arr.dtype)
    return out


def map_zoom_uv(uv_zoom, win) -> tuple:
    """放大图坐标 -> 原图像素坐标。"""
    scale = max(1, int(win.get("scale", 1)))
    u0, v0 = float(win.get("u0", 0)), float(win.get("v0", 0))
    try:
        uz, vz = float(uv_zoom[0]), float(uv_zoom[1])
    except Exception:
        raise ValueError("uv_zoom 需要 (u, v)，收到 %r" % (uv_zoom,))
    return (u0 + uz / scale, v0 + vz / scale)


# --------------------------------------------------------------------------- #
# 冻结接口：像素 -> 世界
# --------------------------------------------------------------------------- #
def _bbox_corners(bbox) -> list:
    u1, v1, u2, v2 = (float(v) for v in bbox)
    lo_u, hi_u = (u1, u2) if u1 <= u2 else (u2, u1)
    lo_v, hi_v = (v1, v2) if v1 <= v2 else (v2, v1)
    return [(lo_u, lo_v), (hi_u, lo_v), (hi_u, hi_v), (lo_u, hi_v)]


def bbox_to_world(env, bbox, *, camera: str = "top", z=None) -> dict:
    """框四角经相机射线与水平面 z 求交，返回世界系 AABB。

    z=None（缺省）按 TABLE_HEIGHT 反投影并返回 assumed_z=True；显式给 z 则 assumed_z=False。
    注意俯视视差（见模块 docstring）：高物体顶面的框按桌面平面反投影会被"外推"到远处，
    此时应配合 triangulate()/object_point() 使用。

    -> {"ok", "center_xy", "size_xy", "corner_xy", "assumed_z"}；任一角无交点则 ok=False
    （center_xy/size_xy 为 None、corner_xy=[]，并附 "why"）。
    只用 sim/vision.py 的公开 API，不读任何物体真值。
    """
    assumed_z = z is None
    plane_z = float(TABLE_HEIGHT if assumed_z else z)
    try:
        corners = _bbox_corners(bbox)
    except Exception:
        return {"ok": False, "center_xy": None, "size_xy": None, "corner_xy": [],
                "assumed_z": assumed_z, "why": "bad_bbox"}
    cam = CameraModel(env, camera)
    pts = []
    for u, v in corners:
        p = cam.pixel_to_plane(u, v, plane_z)
        if p is None:
            return {"ok": False, "center_xy": None, "size_xy": None, "corner_xy": [],
                    "assumed_z": assumed_z, "why": "ray_plane_miss"}
        pts.append([float(p[0]), float(p[1])])
    arr = np.asarray(pts, dtype=np.float64)
    center = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    size = (float(arr[:, 0].max() - arr[:, 0].min()),
            float(arr[:, 1].max() - arr[:, 1].min()))
    return {"ok": True, "center_xy": center, "size_xy": size, "corner_xy": pts,
            "assumed_z": assumed_z}


def estimate_size(env, bbox, *, camera: str = "top") -> dict:
    """由 bbox 反投影估算物体水平尺寸。

    -> {"ok","half_xy","long_side","short_side"}；失败 ok=False 且字段为 None/0。
    """
    world = bbox_to_world(env, bbox, camera=camera)
    if not world["ok"]:
        return {"ok": False, "half_xy": None, "long_side": 0.0, "short_side": 0.0,
                "why": world.get("why", "bbox_to_world_failed")}
    sx, sy = world["size_xy"]
    return {"ok": True, "half_xy": (sx / 2.0, sy / 2.0),
            "long_side": float(max(sx, sy)), "short_side": float(min(sx, sy)),
            "assumed_z": bool(world.get("assumed_z", True))}


# --------------------------------------------------------------------------- #
# 冻结接口：双目三角（§5.10）
# --------------------------------------------------------------------------- #
def triangulate(env, uv_a, cam_a: str, uv_b, cam_b: str,
                *, max_residual_mm: float = MAX_TRIANGULATE_RESIDUAL_MM) -> dict:
    """两条像素射线的最近点（双目三角定位）。

    -> {"ok", "xyz", "residual_mm", "why"}
    失败（xyz=None）的情形：像素/相机非法、两射线近乎平行、交点在相机后方、
    最近点距离 residual_mm > max_residual_mm（默认 15 mm，§5.10 判不可信）。
    只用 sim/vision.py 的 ray()，不读任何物体真值。
    """
    try:
        origin_a, dir_a = CameraModel(env, cam_a).ray(float(uv_a[0]), float(uv_a[1]))
        origin_b, dir_b = CameraModel(env, cam_b).ray(float(uv_b[0]), float(uv_b[1]))
    except Exception as exc:
        return {"ok": False, "xyz": None, "residual_mm": float("nan"),
                "why": "bad_input_%s" % type(exc).__name__}
    w0 = origin_a - origin_b
    b = float(np.dot(dir_a, dir_b))
    d = float(np.dot(dir_a, w0))
    e = float(np.dot(dir_b, w0))
    denom = 1.0 - b * b                      # |dir|=1 -> a=c=1
    if abs(denom) < 1e-12:
        return {"ok": False, "xyz": None, "residual_mm": float("nan"), "why": "parallel_rays"}
    s = (b * e - d) / denom
    t = (e - b * d) / denom
    if s <= 0.0 or t <= 0.0:
        return {"ok": False, "xyz": None, "residual_mm": float("nan"), "why": "behind_camera"}
    pa = origin_a + s * dir_a
    pb = origin_b + t * dir_b
    xyz = 0.5 * (pa + pb)
    residual_mm = float(np.linalg.norm(pa - pb) * 1000.0)
    if residual_mm > float(max_residual_mm):
        return {"ok": False, "xyz": None, "residual_mm": residual_mm,
                "why": "residual_too_large"}
    return {"ok": True, "xyz": xyz, "residual_mm": residual_mm, "why": "ok"}


def object_point(env, *, top_uv=None, overview_uv=None, z_hint=None) -> dict:
    """由像素求物体上的世界点：优先双像素三角，单目回退 z_hint 平面反投影。

    -> {"ok","xyz","source","residual_mm"}；source = "triangulate" | "single_view" | "none"。
    单目时额外给 "z_used"；失败时给 "why"。注意：
      * 双像素但三角残差 > 15 mm 时**不静默回退单目**，直接 ok=False（上层按可重试
        perception_error 处理，见 §5.10 像素解析）；
      * 单目 + 默认 z_hint=TABLE_HEIGHT 对高物体有最大 7 cm 的俯视视差（见 docstring），
        高物体请给 z_hint（可用 height_from_top_z 反推）或补第二视角。
    """
    if top_uv is not None and overview_uv is not None:
        res = triangulate(env, top_uv, "top", overview_uv, "overview")
        if res["ok"]:
            return {"ok": True, "xyz": res["xyz"], "source": "triangulate",
                    "residual_mm": res["residual_mm"], "why": "ok"}
        return {"ok": False, "xyz": None, "source": "triangulate",
                "residual_mm": res["residual_mm"], "why": res["why"]}
    if top_uv is None and overview_uv is None:
        return {"ok": False, "xyz": None, "source": "none", "residual_mm": 0.0,
                "why": "no_pixel"}
    cam = "top" if top_uv is not None else "overview"
    uv = top_uv if top_uv is not None else overview_uv
    z_used = TABLE_HEIGHT if z_hint is None else z_hint
    try:
        z_used = float(z_used)
    except Exception:
        return {"ok": False, "xyz": None, "source": "single_view", "residual_mm": 0.0,
                "why": "bad_z_hint"}
    try:
        xyz = CameraModel(env, cam).pixel_to_plane(float(uv[0]), float(uv[1]), z_used)
    except Exception as exc:
        return {"ok": False, "xyz": None, "source": "single_view", "residual_mm": 0.0,
                "why": "bad_pixel_%s" % type(exc).__name__}
    if xyz is None:
        return {"ok": False, "xyz": None, "source": "single_view", "residual_mm": 0.0,
                "why": "ray_plane_miss"}
    return {"ok": True, "xyz": xyz, "source": "single_view", "residual_mm": 0.0,
            "z_used": z_used, "why": "ok"}


def height_from_top_z(z_top) -> float:
    """由物体顶面高度估计物体高度（米）：z_top - TABLE_HEIGHT（§5.10）。

    供侧抓夹持高度 / 叠放高度估计使用；非法输入返回 nan。
    """
    try:
        value = float(z_top)
    except Exception:
        return float("nan")
    if not np.isfinite(value):
        return float("nan")
    return float(value - TABLE_HEIGHT)


# --------------------------------------------------------------------------- #
# 冻结接口：几何判定
# --------------------------------------------------------------------------- #
def point_in_bbox(uv, bbox, *, shrink_px: float = 0.0) -> bool:
    """uv 是否落在 bbox 内（闭区间）；shrink_px>0 时框四边各内缩。"""
    try:
        u, v = float(uv[0]), float(uv[1])
        u1, v1, u2, v2 = (float(x) for x in bbox)
    except Exception:
        return False
    if not (np.isfinite(u) and np.isfinite(v)):
        return False
    lo_u, hi_u = (u1, u2) if u1 <= u2 else (u2, u1)
    lo_v, hi_v = (v1, v2) if v1 <= v2 else (v2, v1)
    s = max(0.0, float(shrink_px))
    lo_u, hi_u, lo_v, hi_v = lo_u + s, hi_u - s, lo_v + s, hi_v - s
    if hi_u < lo_u or hi_v < lo_v:
        return False
    return bool(lo_u <= u <= hi_u and lo_v <= v <= hi_v)


def _bbox_area(bbox) -> float:
    try:
        u1, v1, u2, v2 = (float(x) for x in bbox)
    except Exception:
        return 0.0
    return max(0.0, abs(u2 - u1) + 1.0) * max(0.0, abs(v2 - v1) + 1.0)


def _bbox_intersection_area(b1, b2) -> float:
    try:
        a_u1, a_v1, a_u2, a_v2 = (float(x) for x in b1)
        b_u1, b_v1, b_u2, b_v2 = (float(x) for x in b2)
    except Exception:
        return 0.0
    w = min(max(a_u1, a_u2), max(b_u1, b_u2)) - max(min(a_u1, a_u2), min(b_u1, b_u2)) + 1.0
    h = min(max(a_v1, a_v2), max(b_v1, b_v2)) - max(min(a_v1, a_v2), min(b_v1, b_v2)) + 1.0
    return max(0.0, w) * max(0.0, h)


def overlap_ratio(b1, b2) -> float:
    """交集面积 / min(两个框面积)，范围 0..1；任一框为空返回 0。"""
    a1, a2 = _bbox_area(b1), _bbox_area(b2)
    if a1 <= 0.0 or a2 <= 0.0:
        return 0.0
    return float(min(1.0, _bbox_intersection_area(b1, b2) / min(a1, a2)))


# --------------------------------------------------------------------------- #
def check_inside(rgb, subject_color: str, container_bbox, *, min_area_px: int = 25,
                 shrink_px=None) -> dict:
    """图像空间包含判定：subject 色块质心是否在 container 框内。

    内缩量缺省 0.06 * 容器短边（近似容器壁像素宽）。-> {"ok","why","metrics"}。
    """
    u1, v1, u2, v2 = (float(x) for x in container_bbox)
    w = abs(u2 - u1) + 1.0
    h = abs(v2 - v1) + 1.0
    shrink = float(shrink_px) if shrink_px is not None else 0.06 * min(w, h)
    center = ((u1 + u2) / 2.0, (v1 + v2) / 2.0)

    blob = color_blob(rgb, subject_color, center,
                      radius_px=max(w, h) * 1.2 + 30.0, min_area_px=int(min_area_px))
    metrics = {
        "subject_uv": [None, None],
        "container_bbox": [int(round(u1)), int(round(v1)), int(round(u2)), int(round(v2))],
        "margin_px": round(float(shrink), 3),
        "shrink_px": round(float(shrink), 3),
        "area_px": 0,
        "score": 0.0,
    }
    if not blob.found:
        metrics["reason"] = "subject_color_not_found"
        return {"ok": False, "why": "subject_color_not_found", "metrics": metrics}
    metrics.update({
        "subject_uv": [round(float(blob.uv[0]), 2), round(float(blob.uv[1]), 2)],
        "area_px": int(blob.area_px),
        "score": round(float(blob.score), 4),
        "subject_bbox": [int(x) for x in blob.bbox],
    })
    if point_in_bbox(blob.uv, container_bbox, shrink_px=shrink):
        return {"ok": True, "why": "subject_centroid_inside_container", "metrics": metrics}
    return {"ok": False, "why": "subject_centroid_outside_container", "metrics": metrics}


def check_stacked(rgb, upper_color: str, lower_color: str, *, min_overlap: float = 0.55,
                  min_area_px: int = 25) -> dict:
    """叠放判定：上层色块质心在下层框内 且 overlap_ratio(upper, lower) >= min_overlap。

    为抵抗遮挡（上层把下层挡成两片/一片月牙），下层框取「与上层框相邻的所有下层连通域的
    并框」：先取面积最大的下层块，再把 bbox 与上层框间隙 <= 25% 上层短边的下层块并入。
    这仍然是纯图像信息，不使用任何物体真值。
    """
    metrics: dict = {}
    upper = color_blob(rgb, upper_color, None, min_area_px=int(min_area_px))
    if not upper.found:
        metrics["reason"] = "upper_color_not_found"
        return {"ok": False, "why": "upper_color_not_found", "metrics": metrics}

    cands, _ = _blobs_of_color(rgb, lower_color, min_area_px=int(min_area_px),
                               hint_uv=None, radius_px=110.0)
    if not cands:
        metrics.update({"upper_uv": [round(float(upper.uv[0]), 2), round(float(upper.uv[1]), 2)],
                        "upper_bbox": [int(x) for x in upper.bbox],
                        "reason": "lower_color_not_found"})
        return {"ok": False, "why": "lower_color_not_found", "metrics": metrics}

    gap = 0.25 * min(abs(upper.bbox[2] - upper.bbox[0]) + 1.0,
                     abs(upper.bbox[3] - upper.bbox[1]) + 1.0)
    merged = [upper.bbox[0] + 0, 0, 0, 0]  # placeholder, 下面重算
    sel_u1 = sel_v1 = np.inf
    sel_u2 = sel_v2 = -np.inf
    n_merged = 0
    for blob, _rank, _d in cands:
        if _bbox_intersection_area(blob.bbox, upper.bbox) > 0.0 or _bbox_gap(blob.bbox, upper.bbox) <= gap:
            sel_u1 = min(sel_u1, blob.bbox[0])
            sel_v1 = min(sel_v1, blob.bbox[1])
            sel_u2 = max(sel_u2, blob.bbox[2])
            sel_v2 = max(sel_v2, blob.bbox[3])
            n_merged += 1
    if n_merged == 0:                                # 兜底：用面积最大的那块
        blob = cands[0][0]
        sel_u1, sel_v1, sel_u2, sel_v2 = blob.bbox
        n_merged = 1
    lower_bbox = (int(sel_u1), int(sel_v1), int(sel_u2), int(sel_v2))
    lower_area = _bbox_area(lower_bbox)
    ratio = overlap_ratio(upper.bbox, lower_bbox)
    centroid_in = point_in_bbox(upper.uv, lower_bbox)

    metrics.update({
        "upper_uv": [round(float(upper.uv[0]), 2), round(float(upper.uv[1]), 2)],
        "lower_uv": None,
        "upper_bbox": [int(x) for x in upper.bbox],
        "lower_bbox": [int(x) for x in lower_bbox],
        "overlap": round(float(ratio), 4),
        "required_overlap": float(min_overlap),
        "upper_area_px": int(upper.area_px),
        "lower_area_px": int(round(lower_area)),
        "lower_merged_components": int(n_merged),
    })
    if not centroid_in:
        return {"ok": False, "why": "upper_centroid_outside_lower_bbox", "metrics": metrics}
    if ratio < float(min_overlap):
        return {"ok": False, "why": "overlap_below_threshold", "metrics": metrics}
    return {"ok": True, "why": "upper_on_lower", "metrics": metrics}


def _bbox_gap(b1, b2) -> float:
    """两个闭区间框之间的最小间隙（相交为 0）。"""
    u1, v1, u2, v2 = (float(x) for x in b1)
    a_u1, a_v1, a_u2, a_v2 = (float(x) for x in b2)
    du = max(min(u1, u2), min(a_u1, a_u2)) - min(max(u1, u2), max(a_u1, a_u2))
    dv = max(min(v1, v2), min(a_v1, a_v2)) - min(max(v1, v2), max(a_v1, a_v2))
    du = max(0.0, du)
    dv = max(0.0, dv)
    return float(np.hypot(du, dv))


def check_gone(rgb_before, rgb_after, color: str, hint_uv, *, radius_px: float = 60,
                min_area_px: int = 25) -> dict:
    """抓走后判定：原位置附近的该颜色块消失，或面积掉到原来的 1/3 以下。

    -> {"ok","why","metrics"}；before 就找不到时 ok=False（不谎报）。
    """
    before = color_blob(rgb_before, color, hint_uv, radius_px=radius_px,
                        min_area_px=int(min_area_px))
    after = color_blob(rgb_after, color, hint_uv, radius_px=radius_px,
                       min_area_px=int(min_area_px))
    metrics = {
        "hint_uv": [round(float(hint_uv[0]), 2), round(float(hint_uv[1]), 2)],
        "radius_px": float(radius_px),
        "area_before_px": int(before.area_px),
        "area_after_px": int(after.area_px),
        "before_bbox": [int(x) for x in before.bbox],
        "after_bbox": [int(x) for x in after.bbox],
        "ratio": round(float(after.area_px / before.area_px), 4) if before.area_px else 0.0,
    }
    if not before.found:
        return {"ok": False, "why": "subject_not_found_before", "metrics": metrics}
    if before.area_px < int(min_area_px):
        return {"ok": False, "why": "subject_too_small_before", "metrics": metrics}
    if not after.found:
        metrics["after_found"] = False
        return {"ok": True, "why": "subject_gone", "metrics": metrics}
    metrics["after_found"] = True
    if after.area_px * 3 < before.area_px:
        return {"ok": True, "why": "subject_area_shrank", "metrics": metrics}
    return {"ok": False, "why": "subject_still_present", "metrics": metrics}
