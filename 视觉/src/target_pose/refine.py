"""几何精修：模型给出四角点/圆心初值后，用边缘对齐把坐标压到亚像素。

这里只做「后处理」：
  * refine_quad —— 对四条边各做 (偏移, 旋转) 局部搜索，最大化边上梯度投影，
    再把相邻边求交得到角点（比回归出来的坐标稳定、且不受整图尺度影响）；
  * locate_circle —— 在靶心附近的小 ROI 里做受限 Hough + 径向圆拟合，
    得到高精度圆心（靶心）；
  * refine_target —— 串起来，并给出质量指标供分级复核。
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .geometry import fit_circle_kasa, intersect_lines, is_convex_quad, order_quad_corners, quad_center
from .labels import Obj


# --------------------------------------------------------------------------- #
# 基础：沿直线采样梯度投影
# --------------------------------------------------------------------------- #
def line_response(gx: np.ndarray, gy: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                  n_samp: int = 64) -> float:
    d = p1 - p0
    L = float(np.hypot(*d))
    if L < 2:
        return 0.0
    u = d / L
    t = np.linspace(0.05, 0.95, n_samp)
    mapx = np.clip(p0[0] + t * d[0], 0, gx.shape[1] - 1).astype(np.float32)[None, :]
    mapy = np.clip(p0[1] + t * d[1], 0, gx.shape[0] - 1).astype(np.float32)[None, :]
    vx = cv2.remap(gx, mapx, mapy, cv2.INTER_LINEAR)[0]
    vy = cv2.remap(gy, mapx, mapy, cv2.INTER_LINEAR)[0]
    return float(np.abs(vx * (-u[1]) + vy * u[0]).mean())


def gradient(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0.8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 4.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    return gx, gy


def refine_line(gx, gy, p0, p1, off_range: float = 4.0, ang_range: float = 5.0,
                off_step: float = 0.5, ang_step: float = 1.0):
    """局部搜索一条边的最优 (偏移, 旋转)，返回新端点与得分。"""
    mid = 0.5 * (np.asarray(p0, float) + np.asarray(p1, float))
    d = np.asarray(p1, float) - np.asarray(p0, float)
    L = float(np.hypot(*d))
    u0 = d / (L + 1e-9)
    n0 = np.array([-u0[1], u0[0]])
    best = (-1.0, 0.0, 0.0)
    for off in np.arange(-off_range, off_range + 1e-9, off_step):
        base = mid + n0 * off
        for adeg in np.arange(-ang_range, ang_range + 1e-9, ang_step):
            a = math.radians(adeg)
            ca, sa = math.cos(a), math.sin(a)
            u = np.array([u0[0] * ca - u0[1] * sa, u0[0] * sa + u0[1] * ca])
            sc = line_response(gx, gy, base - u * L / 2, base + u * L / 2)
            if sc > best[0]:
                best = (sc, off, a)
    sc, off, a = best
    ca, sa = math.cos(a), math.sin(a)
    u = np.array([u0[0] * ca - u0[1] * sa, u0[0] * sa + u0[1] * ca])
    base = mid + n0 * off
    return base - u * L / 2, base + u * L / 2, float(sc)


def refine_quad(gray: np.ndarray, corners: np.ndarray, iters: int = 2) -> Tuple[np.ndarray, Dict]:
    """四边逐条精修 → 相邻边求交。返回 (corners, info)。"""
    q = order_quad_corners(np.asarray(corners, float))
    side = float(np.mean([np.linalg.norm(q[(k + 1) % 4] - q[k]) for k in range(4)]))
    h, w = gray.shape
    m = int(max(6.0, min(0.30 * side, 8.0))) + 2
    bx0 = int(max(0, np.floor(q[:, 0].min()) - m))
    by0 = int(max(0, np.floor(q[:, 1].min()) - m))
    bx1 = int(min(w, np.ceil(q[:, 0].max()) + m + 1))
    by1 = int(min(h, np.ceil(q[:, 1].max()) + m + 1))
    if bx1 - bx0 < 8 or by1 - by0 < 8:
        return q, {"edge_score": 0.0, "quad_ok": False}
    gx, gy = gradient(gray[by0:by1, bx0:bx1])
    q = q - np.array([bx0, by0], float)
    off_range = max(2.0, min(0.03 * side, 8.0))
    scores = [0.0] * 4
    ok = True
    for it in range(iters):
        edges = [(q[k], q[(k + 1) % 4]) for k in range(4)]
        new_edges = []
        for a, b in edges:
            a2, b2, sc = refine_line(gx, gy, a, b,
                                     off_range=off_range if it == 0 else off_range * 0.5,
                                     ang_range=5.0 if it == 0 else 2.5,
                                     ang_step=1.0 if it == 0 else 0.5)
            new_edges.append((a2, b2))
            scores.append(sc)
        # 角点 = 相邻边交点（保持环序 a-b, b-c, c-d, d-a）
        q2 = []
        for k in range(4):
            ip = intersect_lines(*new_edges[(k - 1) % 4], *new_edges[k])
            if ip is None or not np.all(np.isfinite(ip)):
                ok = False
                break
            q2.append(ip)
        if not ok:
            break
        q2 = np.asarray(q2, float)
        if not is_convex_quad(q2):
            break
        q = q2
    return order_quad_corners(q + np.array([bx0, by0], float)), {
        "edge_scores": [float(s) for s in scores[-4:]],
        "edge_score": float(np.mean(scores[-4:])),
        "quad_ok": bool(ok and is_convex_quad(q)),
    }


# --------------------------------------------------------------------------- #
# 中心圆（靶心）
# --------------------------------------------------------------------------- #
def radial_gradient(gray: np.ndarray, center, r_lo: float, r_hi: float,
                    n_ang: int = 360, step: float = 0.5):
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    pad = int(max(r_hi, r_lo) * 1.08) + 4
    x0 = int(max(0, round(center[0]) - pad))
    y0 = int(max(0, round(center[1]) - pad))
    x1 = int(min(w, round(center[0]) + pad + 1))
    y1 = int(min(h, round(center[1]) + pad + 1))
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    if crop.size == 0:
        return np.zeros((n_ang, 2), np.float32), np.array([1.0, 2.0])
    blur = cv2.GaussianBlur(crop, (5, 5), 1.1)
    c = np.array([center[0] - x0, center[1] - y0], float)
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    rhos = np.arange(max(2.0, r_lo), max(r_lo + 2, r_hi), step)
    ca, sa = np.cos(ang), np.sin(ang)
    mapx = (c[0] + ca[:, None] * rhos[None, :]).astype(np.float32)
    mapy = (c[1] + sa[:, None] * rhos[None, :]).astype(np.float32)
    prof = cv2.remap(blur, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.abs(np.diff(prof, axis=1)), rhos


def fit_circle_from_radial(gray: np.ndarray, center, r_lo: float, r_hi: float):
    """以 (center, r) 为初值做径向边缘 → 圆拟合，返回 (center, r, info)。"""
    grad, rhos = radial_gradient(gray, center, r_lo, r_hi)
    if grad.shape[1] < 4:
        return np.asarray(center, float), 0.0, {"circ_ok": False}
    idx = np.argmax(grad, axis=1)
    peak = grad[np.arange(len(idx)), idx]
    keep = peak > max(float(np.median(peak)) * 0.55, 2.0)
    f = np.clip(idx, 1, grad.shape[1] - 2)
    ar = np.arange(len(f))
    a, b, c = grad[ar, f - 1], grad[ar, f], grad[ar, f + 1]
    den = a - 2 * b + c
    den = np.where(np.abs(den) > 1e-6, den, 1.0)
    sub = np.clip(0.5 * (a - c) / den, -1.0, 1.0)
    ang = np.linspace(0, 2 * np.pi, len(idx), endpoint=False)
    rho = rhos[f] + sub * 0.5
    if keep.sum() < 24:
        return np.asarray(center, float), 0.0, {"circ_ok": False, "support": float(keep.mean())}
    pts = np.stack([center[0] + np.cos(ang[keep]) * rho[keep],
                    center[1] + np.sin(ang[keep]) * rho[keep]], 1)
    (cx, cy), r = fit_circle_kasa(pts)
    for _ in range(3):
        d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        m = d < max(1.0, 0.08 * r)
        if m.sum() < 20:
            break
        (cx, cy), r = fit_circle_kasa(pts[m])
    resid = float(np.median(np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)))
    support = float((np.abs(rho - r) < max(1.5, 0.08 * r)).mean())
    return np.array([cx, cy]), float(r), {
        "circ_ok": bool(support > 0.6 and r > 2 and resid < 0.1 * r),
        "circ_support": support,
        "circ_resid": resid,
    }


def circle_contrast(gray: np.ndarray, center, radius: float) -> float:
    """圆内 vs 圆外的灰度差。

    Hough 给的半径经常偏 10~20%，所以外环不能贴着 r 取，要在 1.15~1.75r 上取最大差，
    否则"外环还落在圆内"会让对比度变成 0，把真圆误杀。
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if radius < 3:
        return 0.0
    pad = int(radius * 2.0) + 4
    x0, y0 = int(center[0]) - pad, int(center[1]) - pad
    x1, y1 = int(center[0]) + pad, int(center[1]) + pad
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return 0.0
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - center[0], yy - center[1])
    inner = roi[d < 0.62 * radius]
    if inner.size < 12:
        return 0.0
    mi = float(inner.mean())
    best = 0.0
    for f in (1.15, 1.30, 1.50, 1.75):
        ring = roi[(d > f * radius) & (d < (f + 0.15) * radius)]
        if ring.size < 12:
            continue
        best = max(best, abs(mi - float(ring.mean())))
    return best


def locate_circle(gray: np.ndarray, center_guess, r_hint: float, r_lo_f: float = 0.6,
                  r_hi_f: float = 1.6, quad=None) -> Tuple[np.ndarray, float, Dict]:
    """在靶心附近小 ROI 内定位中心圆：受限 Hough 给初值 + 径向拟合精修。

    r_hint：期望半径（可由四边形边长估计，例如 side/2/1.5）。
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cx, cy = float(center_guess[0]), float(center_guess[1])
    if r_hint <= 2:
        return np.array([cx, cy]), 0.0, {"circ_ok": False}
    pad = int(math.ceil(r_hint * r_hi_f * 1.15)) + 6
    x0, y0 = max(0, int(cx) - pad), max(0, int(cy) - pad)
    x1, y1 = min(w, int(cx) + pad + 1), min(h, int(cy) + pad + 1)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return np.array([cx, cy]), 0.0, {"circ_ok": False}
    roi = gray[y0:y1, x0:x1]
    s2 = min(1.0, 360.0 / max(roi.shape))
    small = cv2.resize(roi, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA) if s2 < 1 else roi
    small = cv2.GaussianBlur(small, (5, 5), 1.2)

    best = None
    min_r = max(4, int(r_hint * r_lo_f * 0.85 * s2))
    max_r = max(min_r + 3, int(r_hint * r_hi_f * s2))
    circles = cv2.HoughCircles(
        small, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(min_r, 8),
        param1=110, param2=22, minRadius=min_r, maxRadius=max_r,
    )
    cands = []
    if circles is not None:
        cands = [(float(a / s2 + x0), float(b / s2 + y0), float(r / s2)) for a, b, r in circles[0]]
    cands.append((cx, cy, float(r_hint)))  # 初值本身也作为候选

    for ccx, ccy, rr in cands:
        cen, rad, info = fit_circle_from_radial(gray, (ccx, ccy), rr * 0.75, rr * 1.25)
        if not info.get("circ_ok"):
            continue
        # 圆心必须靠近初值
        dist = float(np.hypot(cen[0] - cx, cen[1] - cy))
        if dist > max(0.5 * r_hint, 0.5 * rad):
            continue
        con = circle_contrast(gray, cen, rad)
        if con < 8.0:          # 圆内外几乎同色 → 是靶板内切圆之类的假圆
            continue
        if quad is not None and not circle_inside_quad(cen, rad, quad):
            continue           # 圆心跑到靶框外 / 圆被靶框切掉 → 判定为干扰
        info["circ_contrast"] = con
        key = (info["circ_support"] * min(con / 40.0, 1.0)
               - 0.5 * info["circ_resid"] / max(rad, 1) - 0.3 * dist / max(rad, 1))
        if best is None or key > best[0]:
            best = (key, cen, rad, info)
    if best is None:
        return np.array([cx, cy]), 0.0, {"circ_ok": False}
    _, cen, rad, info = best
    info["circ_dist"] = float(np.hypot(cen[0] - cx, cen[1] - cy))
    return cen, rad, info


def _planes(img: np.ndarray):
    """多通道表达：灰度 + B/G/R + 饱和度。不同颜色的圆在不同通道里对比度不同
    （例如黑板上的蓝圆在灰度里几乎看不见，但饱和度和蓝通道里有明显阶跃）。"""
    if img.ndim == 2:
        return [("gray", img.astype(np.float32))]
    b, g, r = cv2.split(img.astype(np.uint8))
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
    return [
        ("gray", cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)),
        ("sat", hsv[:, :, 1].astype(np.float32)),
        ("b", b.astype(np.float32)),
        ("g", g.astype(np.float32)),
        ("r", r.astype(np.float32)),
    ]


def radial_peak_radius(gray: np.ndarray, center, r_lo: float, r_hi: float):
    """沿所有角度对径向梯度取平均，圆的边缘会在其半径处形成一个尖峰。

    方形靶框的边缘在不同角度上距离不同 → 被平均抹平；圆的边缘在半径处对齐 → 保留。
    这比 Hough 稳，也更便宜。
    """
    grad, rhos = radial_gradient(gray, center, r_lo, r_hi, n_ang=240, step=0.5)
    if grad.shape[1] < 6:
        return 0.0, 0.0
    prof = grad.mean(axis=0)
    prof = np.convolve(prof, np.ones(3) / 3, mode="same")
    k = int(np.argmax(prof))
    if k <= 1 or k >= len(prof) - 2:
        return float(rhos[k]), float(prof[k])
    a, b, c = prof[k - 1], prof[k], prof[k + 1]
    den = a - 2 * b + c
    sub = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
    return float(rhos[k] + sub * 0.5), float(b)


def _inside_quad(pts, quad) -> np.ndarray:
    """点是否在凸四边形内（向量化）。"""
    pts = np.asarray(pts, float).reshape(-1, 2)
    q = np.asarray(quad, float)
    ok = np.ones(len(pts), bool)
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        e = b - a
        cr = e[0] * (pts[:, 1] - a[1]) - e[1] * (pts[:, 0] - a[0])
        ok &= cr >= 0
    if not ok.all():
        # 换个环绕方向再试一次
        ok2 = np.ones(len(pts), bool)
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            e = b - a
            cr = e[0] * (pts[:, 1] - a[1]) - e[1] * (pts[:, 0] - a[0])
            ok2 &= cr <= 0
        ok = ok2
    return ok


def circle_inside_quad(center, radius: float, quad) -> bool:
    """圆心在靶框内，且整圆不被靶框裁掉（留 4% 余量）。"""
    q = np.asarray(quad, float)
    if not _inside_quad(np.asarray(center, float)[None, :], q)[0]:
        return False
    # 圆心到四条边的最小距离
    dmin = 1e18
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        e = b - a
        L = np.linalg.norm(e)
        if L < 1e-6:
            continue
        dmin = min(dmin, abs(e[0] * (center[1] - a[1]) - e[1] * (center[0] - a[0])) / L)
    return radius <= dmin * 1.04


def refine_circle_near(img: np.ndarray, center_guess, radius: float,
                       tol: float = 0.15) -> Tuple[np.ndarray, float, Dict]:
    """在给定 (center, radius) 附近做局部精修。

    用于「模型已经给了圆心和半径」的场合：靶框被画面裁掉时，全局找圆的尺度先验会崩，
    但模型值通常仍在正确范围附近，所以只在 ±tol 内微调，并用画面内可见弧段拟合。
    """
    if radius is None or radius < 3:
        return np.asarray(center_guess, float), 0.0, {"circ_ok": False}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    c = np.asarray(center_guess, float)
    r_lo, r_hi = radius * (1 - tol), radius * (1 + tol)

    # 只用"采样点仍在画面内"的角度
    ang = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    ok = np.ones(len(ang), bool)
    for k in (r_lo, r_hi):
        xs = c[0] + np.cos(ang) * k
        ys = c[1] + np.sin(ang) * k
        ok &= (xs >= 1) & (xs <= w - 2) & (ys >= 1) & (ys <= h - 2)
    if ok.sum() < 90:                      # 可见弧段太少，直接用模型值
        return c, float(radius), {"circ_ok": False, "visible_arc": int(ok.sum())}

    grad, rhos = radial_gradient(gray, c, r_lo, r_hi, n_ang=360)
    prof = grad[ok].mean(axis=0)
    k = int(np.argmax(prof))
    if k <= 0 or k >= len(prof) - 1:
        return c, float(radius), {"circ_ok": False}
    r0 = float(rhos[k])
    cen, rad, info = fit_circle_from_radial(gray, c, max(3.0, r0 - 3), r0 + 3)
    if not info.get("circ_ok") or rad <= 2:
        return c, float(radius), {"circ_ok": False}
    # 保守策略：模型/人工给的圆已经可信，只接受小幅修正；改动过大就保留原值
    if abs(rad - radius) > 0.10 * radius or np.linalg.norm(cen - c) > 0.25 * radius:
        return c, float(radius), {"circ_ok": False, "rejected": True}
    info["circ_contrast"] = circle_contrast(gray, cen, rad)
    info["visible_arc"] = int(ok.sum())
    info["circ_ok"] = True
    return cen, float(rad), info


def pick_radius_by_profile(img: np.ndarray, center, candidates, band: float = 0.10):
    """在若干候选半径里，挑"径向梯度剖面在该半径处真的鼓起来"的那个。

    判据 = 该半径处的剖面值 / 全范围中位值。靶框被裁、圆只露一部分弧时，
    真实半径仍会在剖面上形成尖峰，而错误的型号不会。
    """
    if not candidates:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lo = max(3.0, min(candidates) * (1 - band))
    hi = max(candidates) * (1 + band)
    grad, rhos = radial_gradient(gray, center, lo, hi, n_ang=240, step=0.5)
    if grad.shape[1] < 6:
        return None
    prof = grad.mean(axis=0)
    prof = np.convolve(prof, np.ones(3) / 3, mode="same")
    med = float(np.median(prof)) + 1e-6
    best = None
    for r in candidates:
        i = int(np.argmin(np.abs(rhos - r)))
        v = float(prof[max(0, i - 3): i + 4].max())
        s = v / med
        if best is None or s > best[0]:
            best = (s, float(r), float(v))
    return best


def locate_circle_robust(gray: np.ndarray, center_guess, r_hint: float,
                         side: float | None = None, quad=None):
    """多先验 + 宽半径范围定位中心圆。

    三个场景里"大/中/小"三种靶子的圆半径能差 3 倍（省赛同屏就是这样），
    所以只靠 side/3.1 这种单点先验会漏掉小圆 —— 给了 side 就用 [0.10, 0.45]*side 的宽范围，
    由"环向支撑度 + 内外对比度 + 圆度残差"综合打分挑最像的那个。
    """
    # 首选：径向剖面的尖峰（对"大中小三种圆"都有效，且不依赖 Hough）
    if center_guess is not None and r_hint > 2:
        if side and side > 6:
            lo, hi = max(3.0, 0.09 * side), 0.47 * side
        else:
            lo, hi = max(3.0, 0.30 * r_hint), 1.9 * r_hint
        best_plane, r0, peak = gray, 0.0, 0.0
        for _name, plane in _planes(gray):
            rr, pp = radial_peak_radius(plane, center_guess, lo, hi)
            if pp > peak:
                best_plane, r0, peak = plane, rr, pp
        if r0 > 2 and peak > 3.0:
            cen, rad, info = fit_circle_from_radial(best_plane, center_guess, r0 * 0.8, r0 * 1.2)
            con = circle_contrast(gray, cen, rad)
            info["circ_contrast"] = con
            inside = True if quad is None else circle_inside_quad(cen, rad, quad)
            info["inside_quad"] = bool(inside)
            # 小/虚焦的圆 support 会偏低，但"各通道都在同一半径出峰 + 内外有对比 + 在框内"
            # 已经是足够强的证据，这里放宽，并打上 weak 标记供复核
            if (rad > 2 and con >= 8.0 and inside and abs(rad - r0) < 0.30 * r0
                    and (info.get("circ_ok") or peak >= 2.5)):
                info["method"] = "radial-peak"
                info["peak"] = float(peak)
                info["circ_weak"] = not info.get("circ_ok", False)
                info["circ_ok"] = True
                return cen, rad, info

    best = None
    r_hi_abs = 0.46 * side if (side and side > 6) else 1.7 * r_hint
    r_lo_abs = max(3.0, 0.10 * side) if (side and side > 6) else 0.35 * r_hint
    tries = [(0.5 * (r_lo_abs + r_hi_abs), 1.0, 1.0),
             (r_hint, 0.55, 1.75),
             (0.6 * r_hint, 0.5, 2.2)]
    for hint, lo_f, hi_f in tries:
        if side and side > 6:
            lo_f = max(0.05, (0.10 * side) / max(hint, 1e-6))
            hi_f = (0.46 * side) / max(hint, 1e-6)
        cen, rad, info = locate_circle(gray, center_guess, hint, r_lo_f=lo_f, r_hi_f=hi_f, quad=quad)
        if not info.get("circ_ok"):
            continue
        if (info.get("circ_support", 0) > 0.85
                and info.get("circ_resid", 1) < 0.05 * max(rad, 1)
                and info.get("circ_contrast", 0) > 20):
            return cen, rad, info
        q = (info.get("circ_support", 0) * min(info.get("circ_contrast", 0) / 40.0, 1.0)
             - 0.5 * info.get("circ_resid", 1.0) / max(rad, 1))
        if best is None or q > best[0]:
            best = (q, cen, rad, info)
    if best is None:
        return np.asarray(center_guess, float), 0.0, {"circ_ok": False}
    return best[1], best[2], best[3]


# --------------------------------------------------------------------------- #
def refine_target(gray: np.ndarray, obj: Obj, do_quad: bool = True) -> Tuple[Obj, Dict]:
    """对一个靶子做完整精修：四角点边缘对齐 + 圆心重定位。"""
    info: Dict = {}
    img = gray
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = obj.corners
    if do_quad:
        corners, qinfo = refine_quad(gray, corners)
        info.update(qinfo)
    side = float(np.mean([np.linalg.norm(corners[(k + 1) % 4] - corners[k]) for k in range(4)]))
    r_model = float(obj.radius) if obj.radius > 2 else 0.0
    r_hint = r_model if r_model > 2 else side / 3.1

    # 1) 优先在模型给的中心/半径附近做局部精修（对"靶框被裁掉"最稳）
    cen, rad, cinfo = refine_circle_near(img, obj.center, r_model if r_model > 2 else r_hint)
    if not cinfo.get("circ_ok"):
        # 2) 局部失败再走全局找圆，但要求与模型值一致，否则回退模型值
        cen2, rad2, cinfo2 = locate_circle_robust(img, obj.center, r_hint,
                                                  side=side, quad=corners)
        consistent = (r_model <= 2 or
                      (cinfo2.get("circ_ok") and abs(rad2 - r_model) < 0.12 * r_model))
        inside = (r_model <= 2 or np.linalg.norm(cen2 - obj.center) < 0.6 * r_model)
        if cinfo2.get("circ_ok") and consistent and inside:
            cen, rad, cinfo = cen2, rad2, cinfo2
        else:
            cen, rad, cinfo = np.asarray(obj.center, float), r_model, {"circ_ok": r_model > 2,
                                                                       "fallback_model": True}
    info.update(cinfo)
    if not cinfo.get("circ_ok"):
        # 找不到圆时用对角线交点兜底（靶框与圆同心）
        cen = quad_center(corners)
        rad = 0.0
        info["circ_fallback"] = True
    out = Obj(corners=np.asarray(corners, float), center=np.asarray(cen, float),
              vis=list(obj.vis), cls=obj.cls)
    if rad > 0:
        out.set_circle(cen, rad)      # 圆的左右极点沿靶框轴放置
    info["side"] = side
    info["radius"] = float(rad)
    info["center_shift"] = float(np.linalg.norm(out.center - obj.center))
    return out.sort_corners(), info
