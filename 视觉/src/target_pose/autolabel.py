"""无模型的自动预标注：以中心圆为锚，再由圆定尺度去贴合靶框。

为什么这条路线能成立：
  1. 中心圆是整幅图里最"结构化"的目标（圆 + 强对比），Hough + 径向拟合可以稳定
     定位到 ~1px（见 tests/check_refine.py）。
  2. 有了圆半径 r，靶框边长就落在 k*r（k 由场景标定）附近 —— 尺度和中心都已知，
     四条边的搜索范围只有 ±35%，不像"从零找四边形"那样容易被墙缝/支架带跑。
  3. 每条边单独做 (偏移, 旋转) 局部搜索并按梯度支撑打分：某条边被遮挡/不可见时，
     该边退回标称位置，并把相邻角点标成 v=1（遮挡），而不是整条样本丢弃。

对 C2 这类"靶纸与墙面同色"的样本，靶框本来就看不见 —— 这时输出的是
"按圆比例推出来的标称靶框"（frame_inferred=True, 低置信），交由人工确认。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import N_KPT, SCENES, VIS_OCCLUDED, VIS_VISIBLE
from .labels import Obj, Sample
from .refine import (circle_contrast, fit_circle_from_radial, gradient,
                     line_response, radial_peak_radius, refine_line)


# --------------------------------------------------------------------------- #
# 1. 找圆
# --------------------------------------------------------------------------- #
def _circle_contrast(gray: np.ndarray, center, radius: float) -> float:
    h, w = gray.shape
    pad = int(radius * 1.9) + 4
    x0, y0 = int(center[0]) - pad, int(center[1]) - pad
    x1, y1 = int(center[0]) + pad, int(center[1]) + pad
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return 0.0
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - center[0], yy - center[1])
    inner = roi[d < 0.7 * radius]
    outer = roi[(d > 1.05 * radius) & (d < 1.25 * radius)]
    if inner.size < 12 or outer.size < 12:
        return 0.0
    return float(abs(inner.mean() - outer.mean()))


def find_circles(img: np.ndarray, work_side: int = 820, max_out: int = 6,
                 bands=((0.008, 0.060), (0.050, 0.260))) -> List[Tuple]:
    """小尺度 Hough 粗筛（快且半径准）+ 全分辨率窄窗精修。

    实测大图上直接 Hough 又慢又飘；先缩到 ~800px 让圆只有 20~60px，
    半径反而稳定，再回全分辨率用 ±15% 的窄窗做一次亚像素拟合。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    longest = max(h, w)
    s = min(1.0, work_side / longest)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    small = cv2.GaussianBlur(small, (5, 5), 1.2)
    m = min(small.shape)
    # 分带收集、每带限 6 个，且**小半径带优先**：
    # 之前按半径从大到小去重，大半径的假圆会把真圆挤掉（4K 图上尤其明显）
    cands: List[Tuple[float, float, float]] = []
    for lo, hi, p2 in ((bands[0][0], bands[0][1], 30), (bands[1][0], bands[1][1], 34)):
        min_r, max_r = max(4, int(m * lo)), max(int(m * lo) + 3, int(m * hi))
        got = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=max(min_r * 1.4, 8), param1=110, param2=p2,
                               minRadius=min_r, maxRadius=max_r)
        if got is not None:
            cands += [(cx / s, cy / s, r / s) for cx, cy, r in got[0]][:6]

    uniq: List[Tuple[float, float, float]] = []
    for cx, cy, r in cands:
        if any(np.hypot(cx - u[0], cy - u[1]) < 0.45 * max(r, u[2]) for u in uniq):
            continue
        uniq.append((cx, cy, r))
        if len(uniq) >= 12:
            break

    out = []
    for cx, cy, r in uniq:
        con = circle_contrast(img, (cx, cy), r)
        if con < 9.0:
            continue
        cen, rad, info = fit_circle_from_radial(img, (cx, cy), r * 0.84, r * 1.16)
        if not info.get("circ_ok") or rad <= 2:
            continue
        con = circle_contrast(img, cen, rad)
        if con < 9.0:
            continue
        q = info.get("circ_support", 0.6) * min(con / 45.0, 1.0)
        out.append((cen, rad, {**info, "contrast": con, "quality": q}))
    out.sort(key=lambda t: -t[2]["quality"])
    kept = []
    for c2, r2, info in out:
        if any(np.linalg.norm(c2 - k[0]) < 0.5 * max(r2, k[1]) for k in kept):
            continue
        kept.append((c2, r2, info))
        if len(kept) >= max_out:
            break
    return kept


def refine_circle_by_peak(img, center_guess, r_hint: float):
    """Hough 只提供圆心，半径用"多通道径向峰值"确定（比 Hough 半径可靠得多）。"""
    from .refine import _planes

    lo, hi = max(3.0, 0.35 * r_hint), 2.2 * r_hint
    best = (0.0, None, 0.0)
    for _name, plane in _planes(img):
        r0, peak = radial_peak_radius(plane, center_guess, lo, hi)
        if peak > best[0]:
            best = (peak, r0, peak)
    peak, r0, _ = best
    if r0 is None or r0 <= 2 or peak < 2.0:
        return None, 0.0, {}
    cen, rad, info = fit_circle_from_radial(img, center_guess, r0 * 0.82, r0 * 1.18)
    if not info.get("circ_ok") or rad <= 2 or abs(rad - r0) > 0.3 * r0:
        return None, 0.0, {}
    con = circle_contrast(img, cen, rad)
    if con < 9.0:
        return None, 0.0, {}
    info["circ_contrast"] = con
    info["peak"] = float(peak)
    info["circ_ok"] = True
    return cen, rad, info


# --------------------------------------------------------------------------- #
# 2. 由圆推靶框
# --------------------------------------------------------------------------- #
def _dominant_orientation(gx, gy, c, half: float) -> float:
    """在靶框附近估计主方向（弧度，[0, π/2)）。"""
    mag = np.hypot(gx, gy)
    yy, xx = np.mgrid[0:gx.shape[0], 0:gx.shape[1]]
    d = np.hypot(xx - c[0], yy - c[1])
    m = (d > 0.72 * half) & (d < 1.35 * half) & (mag > 4.0)
    if m.sum() < 60:
        return 0.0
    ang = np.arctan2(gy[m], gx[m]) % (np.pi / 2)
    wgt = mag[m]
    hist, edges = np.histogram(ang, bins=90, range=(0, np.pi / 2), weights=wgt)
    hist = np.convolve(hist, np.ones(5) / 5, mode="same")
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def _interior_uniformity(gray, quad, center, r) -> float:
    h, w = gray.shape
    x1, y1 = np.floor(quad.min(axis=0)).astype(int)
    x2, y2 = np.ceil(quad.max(axis=0)).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 1.0
    roi = gray[y1:y2, x1:x2]
    m = np.zeros(roi.shape, np.uint8)
    cv2.fillPoly(m, [(quad - [x1, y1]).astype(np.int32)], 255)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    d = np.hypot(xx - center[0], yy - center[1])
    m = cv2.erode(m, np.ones((5, 5), np.uint8))
    sel = (m > 0) & (d > r * 1.10)
    if sel.sum() < 40:
        return 0.0
    v = roi[sel].astype(np.float32)
    q25, q75 = np.percentile(v, [25, 75])
    return float((q75 - q25) / 255.0)


def _outside_contrast(gray, quad) -> float:
    h, w = gray.shape
    x1, y1 = np.floor(quad.min(axis=0)).astype(int) - int(0.12 * np.max(quad.max(axis=0) - quad.min(axis=0)))
    x2, y2 = np.ceil(quad.max(axis=0)).astype(int) + int(0.12 * np.max(quad.max(axis=0) - quad.min(axis=0)))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return 0.0
    roi = gray[y1:y2, x1:x2].astype(np.float32)
    m = np.zeros(roi.shape, np.uint8)
    cv2.fillPoly(m, [(quad - [x1, y1]).astype(np.int32)], 255)
    inner = cv2.erode(m, np.ones((7, 7), np.uint8))
    outer = cv2.subtract(cv2.dilate(m, np.ones((7, 7), np.uint8)), m)
    a, b = roi[inner > 0], roi[outer > 0]
    if a.size < 40 or b.size < 40:
        return 0.0
    return float(abs(np.median(a) - np.median(b)))


def board_from_circle(gray: np.ndarray, center: np.ndarray, r: float, k: float,
                      ar: float = 1.0) -> Optional[Dict]:
    """以圆为锚求靶框。k = 靶框边长 / 圆半径。"""
    half = 0.5 * k * r
    if half < 6:
        return None
    # 先裁 ROI 再缩放：原来对整幅图 resize，4K 图上一个靶子要 0.6s
    scale = min(max(64.0 / half, 0.35), 4.0)
    pad = int(half * 1.45) + 4
    hh, ww = gray.shape[:2]
    x0 = int(max(0, round(center[0]) - pad))
    y0 = int(max(0, round(center[1]) - pad))
    x1 = int(min(ww, round(center[0]) + pad + 1))
    y1 = int(min(hh, round(center[1]) + pad + 1))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    crop = gray[y0:y1, x0:x1]
    c = (center - [x0, y0]) * scale
    img = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    gx, gy = gradient(img)
    H = half * scale
    theta = _dominant_orientation(gx, gy, c, H)

    u = np.array([math.cos(theta), math.sin(theta)])
    v = np.array([-u[1], u[0]])

    def rect(hw, hh):
        return np.array([c - u * hw - v * hh, c + u * hw - v * hh,
                         c + u * hw + v * hh, c - u * hw + v * hh])

    q = rect(H, H * ar)
    edges = [(q[i], q[(i + 1) % 4]) for i in range(4)]
    scores: List[float] = []
    strong: List[bool] = []
    for it, (off_r, ang_r) in enumerate(((0.38, 9.0), (0.16, 4.0), (0.06, 1.5))):
        new_edges = []
        scores, strong = [], []
        for a, b in edges:
            a2, b2, sc = refine_line(gx, gy, a, b, off_range=off_r * H, ang_range=ang_r,
                                     off_step=max(0.5, 0.02 * H), ang_step=max(0.5, ang_r / 4))
            new_edges.append((a2, b2))
            scores.append(sc)
        med = float(np.median(scores))
        strong = [s >= max(0.45 * med, 4.0) for s in scores]
        # 弱边退回上一步（避免被背景拉走）
        edges = [new_edges[i] if strong[i] else edges[i] for i in range(4)]
        # 重新求交
        qn = []
        for i in range(4):
            p = _intersect(*edges[(i - 1) % 4], *edges[i])
            if p is None:
                return None
            qn.append(p)
        q = np.asarray(qn, float)
        if not _convex(q):
            return None

    quad = q / scale + np.array([x0, y0], float)
    qc = np.mean(quad, axis=0)
    area = _poly_area(quad)
    nominal = (k * r) ** 2
    uni = _interior_uniformity(gray, quad, center, r)
    ocon = _outside_contrast(gray, quad)
    meta = {
        "edge_scores": [float(x) for x in scores],
        "edge_score": float(np.mean(scores)),
        "strong_edges": strong,
        "weak_edges": int(sum(1 for s in strong if not s)),
        "area_ratio": float(area / max(nominal, 1e-6)),
        "concentric": float(np.linalg.norm(qc - center) / max(half, 1e-6)),
        "uniformity": float(uni),
        "outside_contrast": float(ocon),
        "theta": float(theta),
    }
    meta["frame_ok"] = bool(
        meta["weak_edges"] <= 1
        and 0.55 <= meta["area_ratio"] <= 1.75
        and meta["concentric"] <= 0.18   # 圆必须在靶框中心（窗框那类会偏到一边）
        and uni <= 0.16
    )
    return {"quad": quad, "meta": meta}


def _intersect(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    return np.array([(a * (x3 - x4) - (x1 - x2) * b) / den,
                     (a * (y3 - y4) - (y1 - y2) * b) / den])


def _convex(q) -> bool:
    s = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        s.append(float(np.cross(b - a, c - b)))
    return all(x > 0 for x in s) or all(x < 0 for x in s)


def _poly_area(q) -> float:
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# --------------------------------------------------------------------------- #
# 3. 单图 / 批量
# --------------------------------------------------------------------------- #
# 靶框边长/圆半径 的候选（大中小三种靶子差 3 倍，单一先验不够）
K_CANDIDATES = (1.6, 2.2, 3.0, 4.2, 5.4)


def autolabel_image(img: np.ndarray, scene_key: str, use_k: Optional[float] = None) -> List[Dict]:
    sc = SCENES[scene_key]
    ks = (use_k,) if use_k else K_CANDIDATES
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    res: List[Dict] = []
    for cen, r, cinfo in find_circles(img):
        best = None
        for k in ks:
            bq = board_from_circle(gray, cen, r, k)
            if bq is None:
                continue
            m = bq["meta"]
            s_ = (2.0 if m["frame_ok"] else 0.0) + min(m["edge_score"] / 30.0, 1.5) - 0.4 * m["weak_edges"]
            if best is None or s_ > best[0]:
                best = (s_, bq)
        if best is None:
            continue
        bq = best[1]
        quad, meta = bq["quad"], bq["meta"]
        meta["k"] = float(round(2.0 * np.mean([np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]) / max(r, 1e-6), 2))
        vis = [VIS_VISIBLE] * N_KPT
        # 角点 i 由边 i-1 与边 i 决定
        for i in range(4):
            if not (meta["strong_edges"][(i - 1) % 4] and meta["strong_edges"][i]):
                vis[i] = VIS_OCCLUDED
        obj = Obj(corners=quad, center=cen, vis=vis)
        obj.set_circle(cen, r)
        if obj.radius_rel > 1.02:      # 圆比靶框还大 → 一定抓错了（窗框/阴影）
            continue
        score = (0.55 * min(cinfo["quality"], 1.0)
                 + 0.25 * min(meta["edge_score"] / 30.0, 1.0)
                 + 0.20 * (1.0 if meta["frame_ok"] else 0.0))
        meta.update({"circle_quality": cinfo["quality"], "contrast": cinfo["contrast"]})
        res.append({"obj": obj.sort_corners(), "score": float(score),
                    "frame_ok": meta["frame_ok"], "meta": meta})
    res.sort(key=lambda x: -x["score"])
    return res


def run(args) -> Dict:
    from .datasets import annotate_path, meta_path, scan

    items = scan(args.scene or None)
    todo = []
    for it in items:
        if it.source == "user" and not args.force:
            continue
        if args.skip_seed:
            seed = (Path(args.skip_seed))
            if seed.exists() and it.sid in set(seed.read_text().split()):
                continue
        if it.n_objs > 0 and not args.overwrite:
            continue
        todo.append(it)
    if args.limit:
        todo = todo[: args.limit]
    print(f"自动标注 {len(todo)} 张")

    stat = {"img": 0, "objs": 0, "frame_ok": 0, "occluded": 0, "empty": 0, "lowconf": 0}
    t0 = time.time()
    for k, it in enumerate(todo, 1):
        img = cv2.imread(it.image)
        if img is None:
            continue
        try:
            res = autolabel_image(img, it.scene)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {it.sid}: {e}")
            continue
        if not res:
            stat["empty"] += 1
            # strict 模式下不写空文件：空标注在复核列表里只会是噪音
            if args.strict:
                continue
            lp = annotate_path(it.scene, Path(it.image).stem)
            mp = meta_path(it.scene, Path(it.image).stem)
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text("")
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text(json.dumps({"source": "auto", "score": 0, "reviewed": False,
                                      "n_objs": 0, "note": "未检出"},
                                     ensure_ascii=False, indent=1))
            continue
        exp = SCENES[it.scene].expect_targets
        if exp and len(res) != exp:      # 召回不够 → 不写，留给人工/模型
            stat["lowconf"] += 1
            continue
        if args.strict and not all(r["frame_ok"] for r in res):
            stat["lowconf"] += 1
            continue
        sample = Sample(image=Path(it.image), width=it.width, height=it.height, objs=[],
                        path_hint=it.scene)
        scores = []
        for r in res:
            o = r["obj"].clipped(it.width, it.height)
            sample.objs.append(o)
            scores.append(r["score"])
            stat["frame_ok"] += int(r["frame_ok"])
            stat["occluded"] += int(any(v == VIS_OCCLUDED for v in o.vis))
        lp = annotate_path(it.scene, Path(it.image).stem)
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(sample.to_txt())
        mp = meta_path(it.scene, Path(it.image).stem)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps({
            "source": "auto",
            "score": round(float(np.mean(scores)), 4),
            "reviewed": False,
            "n_objs": len(sample.objs),
            "method": "circle-first",
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False, indent=1))
        stat["img"] += 1
        stat["objs"] += len(sample.objs)
        if args.verbose:
            print(f"  {it.sid}: {len(sample.objs)} 个 " +
                  " ".join(f"{r['score']:.2f}{'*' if not r['frame_ok'] else ''}" for r in res))
        if k % 50 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"  [{k}/{len(todo)}] 图 {stat['img']} 实例 {stat['objs']} "
                  f"框可信 {stat['frame_ok']} 空 {stat['empty']} {el:.0f}s", flush=True)
    print(f"完成：{stat}")
    return stat


def main(argv=None):
    ap = argparse.ArgumentParser(description="无模型自动预标注（圆锚定）")
    ap.add_argument("--scene", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=float, default=None, help="覆盖 靶框边长/圆半径")
    ap.add_argument("--skip-seed", default=None, help="跳过 txt 中列出的 sid（如种子集）")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--force", action="store_true", help="连人工标注也覆盖")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--strict", action="store_true", default=True,
                    help="只写所有靶子都可信的图（默认开）")
    ap.add_argument("--no-strict", dest="strict", action="store_false")
    args = ap.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
