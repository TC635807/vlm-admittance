"""tools/vision_accuracy.py —— 视觉定位精度调试统计（允许读真值；Lead/verifier 维护）。

统计「VLM 粗像素 -> 颜色精修 -> 世界 xy」与真值的误差，输出 mm 级统计表。

用法：
    cd /home/tc63/mujoco/vlm_admittance
    MUJOCO_GL=glfw DISPLAY=:0 /home/tc63/mujoco/.venv/bin/python tools/vision_accuracy.py \
        [--camera top] [--jitter-px 6] [--scene default|daily]

说明：
  * 真值只用于**统计列**，不参与算法；sim/perception.py 缺失时自动降级为「粗像素直接反投影」并标注 pending。
  * 无网络依赖。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import TABLE_HEIGHT                      # noqa: E402
from sim.env import MuJoCoArmEnv                          # noqa: E402
from sim.vision import CameraModel, pixel_to_table, project_point   # noqa: E402

try:
    from sim import perception as P                       # noqa: E402
except Exception:                                         # noqa: BLE001
    P = None


def _fmt(v, unit="", nd=2):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{nd}f}{unit}"


def _parallax_floor_px(env, obj, camera="top"):
    """俯视视差地板（px）= 可见顶面中心像素 与 体心像素 的距离（§5.10）。"""
    truth = np.asarray(obj.pose[:3], float)
    top = truth.copy()
    top[2] = float(top[2] + float(obj.size[2]))
    p_c = project_point(env, truth, camera=camera)
    p_t = project_point(env, top, camera=camera)
    if p_c is None or p_t is None:
        return None
    return float(np.hypot(float(p_t[0]) - float(p_c[0]), float(p_t[1]) - float(p_c[1])))


def _vlm_pixel(vlm, env, rgb, obj, camera):
    """让真实 VLM 对 top 图给目标像素（真值绝不进 prompt）；失败返回 None。"""
    try:
        from agent.state import build_messages
    except Exception as exc:                              # noqa: BLE001
        print(f"[warn] build_messages 不可用: {exc}")
        return None
    task = (f"标定任务：在 {camera} 图像里找到「{obj.color} {obj.category}」的中心像素，"
            f"只输出一次工具调用，用 target_pixel=[u,v] + pixel_camera='{camera}'；"
            f"不要给 target_id / target_xy，也不要执行抓取。目标物体 id 仅供你确认存在：{obj.id}")
    call = None
    for attempt in (1, 2):
        try:
            state = env.get_state()
            messages = build_messages(task, state, {camera: rgb}, history=[], include_poses=False)
            call = vlm.decide(messages)
            break
        except Exception as exc:                          # noqa: BLE001 - 网络/解析失败重试一次
            print(f"[warn] VLM 调用失败({obj.id}) 第{attempt}次: {type(exc).__name__}: {exc}")
    if call is None:
        return None
    args = getattr(call, "args", None) or {}
    tp = args.get("target_pixel")
    if isinstance(tp, (list, tuple)) and len(tp) == 2:
        try:
            uv = (float(tp[0]), float(tp[1]))
        except (TypeError, ValueError):
            print(f"[warn] {obj.id}: target_pixel 非数值 {tp!r}")
            return None
        print(f"  [vlm] {obj.id}: skill={call.skill} target_pixel={uv} "
              f"pixel_camera={args.get('pixel_camera')}")
        return uv
    if str(getattr(call, "skill", "")) == "locate" and P is not None:
        # §5.3：模型选择 locate -> 由本地 locator 解析像素（颜色质心）
        color = str(args.get("color") or obj.color)
        try:
            blob = P.color_blob(rgb, color)
        except Exception as exc:                          # noqa: BLE001
            print(f"[warn] {obj.id}: locate->color_blob 异常: {exc}")
            return None
        if getattr(blob, "found", False):
            uv = (float(blob.uv[0]), float(blob.uv[1]))
            print(f"  [vlm] {obj.id}: skill=locate(color={color}) -> color_blob 质心 {uv} "
                  f"bbox={blob.bbox}")
            return uv
        print(f"[warn] {obj.id}: locate(color={color}) 在 top 图里没找到色块")
        return None
    print(f"[warn] {obj.id}: VLM 未给 target_pixel（skill={getattr(call, 'skill', None)} "
          f"args={list(args)}）")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="视觉定位精度统计（mm）")
    ap.add_argument("--camera", default="top")
    ap.add_argument("--jitter-px", type=float, default=6.0, help="模拟 VLM 粗像素误差（像素）")
    ap.add_argument("--scene", default="default", choices=["default", "daily"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vlm", action="store_true",
                    help="用真实 VLM 对 top 图给像素（真值只进统计列，绝不进 prompt）")
    ap.add_argument("--objects", default="", help="逗号分隔的物体 id 子集（默认全部）")
    ap.add_argument("--color-only", action="store_true",
                    help="无网络：直接对 top 图跑 color_blob（全图搜索，无 hint），衡量颜色判别本身")
    args = ap.parse_args(argv)

    objects = None
    if args.scene == "daily":
        try:
            from sim.scene import DAILY_OBJECTS as objects      # noqa: N811
        except Exception as exc:                                 # noqa: BLE001
            print(f"[warn] daily 场景未落地（{type(exc).__name__}: {exc}），回退 default")
    env = MuJoCoArmEnv(objects=objects, seed=args.seed)
    env.step(60)

    cam = CameraModel(env, args.camera)
    cu, cv = cam.width / 2.0, cam.height / 2.0
    from sim.vision import CameraModel as _CM
    import mujoco
    _c = _CM(env, args.camera)
    o, d = _c.ray(cu + 50, cv + 50)
    o2, d2 = _c.ray(cu - 50, cv - 50)
    mmpx = None
    try:
        _t = (TABLE_HEIGHT - o[2]) / d[2]
        _t2 = (TABLE_HEIGHT - o2[2]) / d2[2]
        _p1, _p2 = o + _t * d, o2 + _t2 * d2
        mmpx = float(np.linalg.norm(_p1[:2] - _p2[:2])) / float(np.hypot(100, 100))
    except Exception:
        pass

    mode = ("color-only(无 hint 全图搜索；更严的探针)" if args.color_only
            else ("vlm-real(真模型像素；locate 走无 hint，target_pixel 走 hint)" if args.vlm
                  else f"runtime-like(hint=粗像素±{args.jitter_px / 2:.1f}px σ)"))
    print(f"=== vision_accuracy | mode={mode} | scene={args.scene} camera={args.camera} "
          f"{cam.width}x{cam.height} fovy={cam.fovy:.1f}° | 解析尺度 {_fmt(None if mmpx is None else mmpx * 1000, ' mm/px', 3)} ===")
    print("perception.py:", "OK（启用颜色精修）" if P is not None else "pending（降级：粗像素直接反投影）")
    try:
        rgb = env.render(args.camera)
    except Exception as exc:                                  # noqa: BLE001
        print(f"[error] 渲染失败: {exc}")
        return 1

    vlm = None
    if args.vlm:
        try:
            from agent.vlm_client import MockVLM, make_vlm
            vlm = make_vlm(prefer_api=True)
            print("VLM 决策器:", type(vlm).__name__,
                  "(真实像素基线)" if not isinstance(vlm, MockVLM) else "(无 key，退化为 MockVLM)")
        except Exception as exc:                          # noqa: BLE001
            print(f"[error] VLM 初始化失败: {exc}")
            return 1

    want = {s.strip() for s in args.objects.split(",") if s.strip()}
    hdr = ("object", "color", "truth_xy", "coarse_px", "refined_px", "d_px",
           "world_xy", "err_mm", "refined")
    if args.color_only and P is None:
        print("[error] --color-only 需要 sim/perception.py")
        return 1

    rows = []
    rng = np.random.default_rng(args.seed)
    for obj in env.list_objects():
        truth = np.asarray(obj.pose[:3], float)
        px = project_point(env, truth, camera=args.camera)
        if px is None:
            rows.append({"id": obj.id, "color": obj.color, "truth": truth[:2], "coarse": None,
                         "refined": None, "dpx": None, "wxy": None, "err": None,
                         "refined_ok": False, "floor": None, "suspect": False})
            continue
        if want and obj.id not in want:
            continue
        floor_px = _parallax_floor_px(env, obj, args.camera)
        used_hint = False
        if args.color_only:
            try:
                blob = P.color_blob(rgb, obj.color)
            except Exception as exc:                      # noqa: BLE001
                print(f"[warn] color_blob({obj.id}) 异常: {exc}")
                blob = None
            if blob is None or not getattr(blob, "found", False):
                rows.append({"id": obj.id, "color": obj.color, "truth": truth[:2], "coarse": None,
                             "refined": None, "dpx": None, "wxy": None, "err": None,
                             "refined_ok": False, "floor": floor_px, "suspect": False})
                continue
            coarse = (float(blob.uv[0]), float(blob.uv[1]))
        elif vlm is not None:
            coarse = _vlm_pixel(vlm, env, rgb, obj, args.camera)
            if coarse is None:
                rows.append({"id": obj.id, "color": obj.color, "truth": truth[:2], "coarse": None,
                             "refined": None, "dpx": None, "wxy": None, "err": None,
                             "refined_ok": False, "floor": floor_px, "suspect": False})
                continue
        else:
            jx = float(rng.normal(0.0, args.jitter_px / 2.0))
            jy = float(rng.normal(0.0, args.jitter_px / 2.0))
            coarse = (px[0] + jx, px[1] + jy)
            used_hint = True     # 运行时路径：VLM 粗像素 -> refine_pixel(hint, 60px 窗口)
        refined = coarse
        refined_ok = False
        if P is not None:
            try:
                out = P.refine_pixel(rgb, obj.color, coarse, camera=args.camera, env=env)
                refined = tuple(float(v) for v in out["uv"])
                refined_ok = bool(out.get("refined"))
            except Exception as exc:                          # noqa: BLE001
                print(f"[warn] refine_pixel({obj.id}) 异常: {type(exc).__name__}: {exc}")
        world = pixel_to_table(env, refined[0], refined[1], camera=args.camera)
        if world is None:
            err = None
            wxy = None
        else:
            wxy = np.asarray(world[:2], float)
            err = float(np.linalg.norm(wxy - truth[:2])) * 1000.0
        dpx = float(np.hypot(refined[0] - px[0], refined[1] - px[1]))
        suspect = False
        if (vlm is not None or args.color_only) and P is not None:
            try:
                blob = P.color_blob(rgb, obj.color, coarse)
                if getattr(blob, "found", False):
                    u1, v1, u2, v2 = blob.bbox
                    area_frac = float((u2 - u1) * (v2 - v1)) / float(rgb.shape[0] * rgb.shape[1])
                    suspect = area_frac > 0.25          # 色块过大 -> 多半是误检（黄/灰易全图）
                    if suspect:
                        print(f"  [suspect] {obj.id}: color_blob bbox={blob.bbox} "
                              f"占画面 {area_frac * 100:.0f}% -> 疑似误检")
            except Exception:                             # noqa: BLE001
                pass
        rows.append({"id": obj.id, "color": obj.color, "truth": truth[:2], "coarse": coarse,
                     "refined": refined, "dpx": dpx, "wxy": wxy, "err": err,
                     "refined_ok": refined_ok, "floor": floor_px, "suspect": suspect,
                     "hint": used_hint})
    print(f"{'object':<14}{'color':<8}{'d_px':>7}{'floor':>7}{'d-floor':>8}"
          f"{'world_xy':<18}{'err_mm':>8}  refined")
    errs = []
    for row in rows:
        oid, color = row["id"], row["color"]
        truth, coarse, refined = row["truth"], row["coarse"], row["refined"]
        dpx, wxy, err, floor = row["dpx"], row["wxy"], row["err"], row["floor"]
        w = "-" if wxy is None else f"({wxy[0]:+.3f},{wxy[1]:+.3f})"
        dmin = (None if (dpx is None or floor is None) else float(dpx) - float(floor))
        tag = f"{row['refined_ok']}{'/suspect' if row['suspect'] else ''}"
        print(f"{oid:<14}{color:<8}{format(dpx, '7.2f') if dpx is not None else '      -':>7}"
              f"{format(floor, '7.2f') if floor is not None else '      -':>7}"
              f"{format(dmin, '8.2f') if dmin is not None else '       -':>8}"
              f"{w:<18}{format(err, '8.2f') if err is not None else '       -':>8}  {tag}")
        if err is not None and np.isfinite(err):
            errs.append(err)
    if errs:
        a = np.asarray(errs)
        print(f"\n[summary] n={a.size} mean={a.mean():.2f}mm median={np.median(a):.2f}mm "
              f"max={a.max():.2f}mm p90={np.percentile(a, 90):.2f}mm")
        if mmpx:
            print(f"[summary] 1px ≈ {mmpx * 1000:.3f}mm；粗像素 jitter σ={args.jitter_px / 2:.1f}px")

    # ---- 验收口径（Lead 裁决 (a)，**按颜色而非 id**，覆盖 daily 场景）----
    #  非灰类（红/蓝/绿/青水瓶…）：色差 = d_px - floor <= 1px（floor=俯视视差地板）；黄沿用 d_px<=5px
    #  灰/白/黑：refined=False（且不得给错点）或 d_px <= 3px
    #  绝对精度交给双目三角（tests/test_triangulation.py：水瓶<=5mm、方块<=3mm）
    by_id = {r["id"]: r for r in rows}
    print("\n[criteria] 非灰:(d_px-floor)<=1px（黄 d_px<=5px）；灰/白/黑:refined=False 或 d_px<=3px")
    fails = []
    for oid, r in by_id.items():
        color = str(r["color"]).lower()
        dpx, floor, refined = r["dpx"], r["floor"], r["refined_ok"]
        grayish = color in ("gray", "grey", "white", "black")
        if not grayish:
            gap = None if (dpx is None or floor is None) else float(dpx) - float(floor)
            if color == "yellow":
                ok = dpx is not None and float(dpx) <= 5.0
                lim = "d_px<=5.0"
            else:
                ok = gap is not None and gap <= 1.0
                lim = "d-floor<=1.0"
            print(f"  {oid:<14}{color:<7} d_px={format(dpx, '.2f') if dpx is not None else '-':<7}"
                  f"floor={format(floor, '.2f') if floor is not None else '-':<7}"
                  f"d-floor={format(gap, '.2f') if gap is not None else '-':<7}{lim} -> "
                  f"{'PASS' if ok else 'FAIL'}")
            if not ok:
                fails.append(f"{oid}(色差={format(gap, '.2f') if gap is not None else format(dpx, '.2f')}px)")
            continue
        no_point_ok = False
        if not refined:
            if r["coarse"] is None:
                no_point_ok = True          # color-only：无回退点，refined=False 即未给错点
            else:
                uv_same = (r["refined"] is not None
                           and np.allclose(r["coarse"], r["refined"], atol=1e-9))
                truth_px = project_point(env, np.array([r["truth"][0], r["truth"][1], TABLE_HEIGHT]),
                                         camera=args.camera)
                near = (truth_px is not None
                        and float(np.hypot(r["coarse"][0] - truth_px[0],
                                           r["coarse"][1] - truth_px[1])) <= 10.0)
                no_point_ok = bool(uv_same and near)
        ok = (not refined and no_point_ok) or (dpx is not None and float(dpx) <= 3.0)
        print(f"  {oid:<14}{color:<7} refined={refined} "
              f"d_px={format(dpx, '.2f') if dpx is not None else '-'} "
              f"no-wrong-point={no_point_ok} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"{oid}(d_px={format(dpx, '.2f') if dpx is not None else '-'}px,refined={refined})")
    print(f"[criteria] {'ALL PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
