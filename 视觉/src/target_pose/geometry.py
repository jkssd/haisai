"""纯几何工具：四边形、直线、圆、透视变换。无图像 I/O。"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# 四边形
# --------------------------------------------------------------------------- #
def order_quad_corners(pts: np.ndarray) -> np.ndarray:
    """排序为 [左上, 右上, 右下, 左下]（顺时针）。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    quad = np.stack([pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]])
    if len(np.unique(np.round(quad, 3), axis=0)) < 4:
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        quad = pts[np.argsort(ang)]
        quad = np.roll(quad, -int(np.argmin(quad[:, 0] + quad[:, 1])), axis=0)
    return quad


def quad_area(quad: np.ndarray) -> float:
    q = np.asarray(quad, dtype=np.float64)
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def quad_center(quad: np.ndarray) -> np.ndarray:
    """对角线交点（透视四边形下比顶点均值更接近真实中心）。"""
    q = np.asarray(quad, dtype=np.float64)
    p = intersect_lines(q[0], q[2], q[1], q[3])
    return q.mean(axis=0) if p is None else p


def is_convex_quad(quad: np.ndarray) -> bool:
    q = np.asarray(quad, dtype=np.float64)
    signs = np.array([
        np.cross(q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4]) for i in range(4)
    ])
    return bool(np.all(signs > 0) or np.all(signs < 0))


def quad_edge_lengths(quad: np.ndarray) -> np.ndarray:
    q = np.asarray(quad, dtype=np.float64)
    return np.array([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)])


def quad_side(quad: np.ndarray) -> float:
    return float(np.mean(quad_edge_lengths(quad)))


# --------------------------------------------------------------------------- #
# 直线
# --------------------------------------------------------------------------- #
def intersect_lines(p1, p2, p3, p4) -> Optional[np.ndarray]:
    """两直线 (p1,p2) 与 (p3,p4) 的交点；平行返回 None。"""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    return np.array([(a * (x3 - x4) - (x1 - x2) * b) / den,
                     (a * (y3 - y4) - (y1 - y2) * b) / den], dtype=np.float64)


def fit_line_tls(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """总体最小二乘直线拟合 → (点, 单位方向, 残差中位数)。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    d = vt[0] / (np.linalg.norm(vt[0]) + 1e-12)
    resid = np.abs((pts - mean) @ np.array([-d[1], d[0]]))
    return mean, d, float(np.median(resid))


# --------------------------------------------------------------------------- #
# 圆
# --------------------------------------------------------------------------- #
def fit_circle_kasa(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """Kasa 代数圆拟合 → ((cx,cy), r)。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    return np.array([cx, cy]), r


# --------------------------------------------------------------------------- #
# 透视
# --------------------------------------------------------------------------- #
def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(np.asarray(src, np.float32), np.asarray(dst, np.float32))


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return ph[:, :2] / np.maximum(ph[:, 2:3], 1e-12)


def target_square(n: int = 200) -> np.ndarray:
    """靶面归一化正方形（y 轴向下，与图像一致）。"""
    return np.array([[0, 0], [n, 0], [n, n], [0, n]], np.float64)


def rectify_homography(corners: np.ndarray, n: int = 200) -> np.ndarray:
    """图像 → 靶面正视图 (n x n) 的单应。"""
    return homography(order_quad_corners(corners), target_square(n))


def to_target_coords(corners: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """把图像坐标映射到靶面归一化坐标 [0,1]×[0,1]。"""
    H = rectify_homography(corners, 1.0)
    return apply_h(H, pts)


def scale_quad(quad: np.ndarray, s: float) -> np.ndarray:
    return np.asarray(quad, float) * s


def clip_quad(quad: np.ndarray, w: int, h: int) -> np.ndarray:
    q = np.asarray(quad, np.float64).copy()
    q[:, 0] = np.clip(q[:, 0], 0, w - 1)
    q[:, 1] = np.clip(q[:, 1], 0, h - 1)
    return q


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    """两个凸多边形的 IoU（在二者的并集包围盒内光栅化）。"""
    aa = np.asarray(a, np.float64)
    bb = np.asarray(b, np.float64)
    allp = np.vstack([aa, bb])
    x0, y0 = allp.min(axis=0)
    span = float(np.max(allp.max(axis=0) - [x0, y0]))
    if span <= 1e-6:
        return 0.0
    s = 900.0 / span
    W = int(span * s) + 8
    ha = np.zeros((W, W), np.uint8)
    hb = np.zeros((W, W), np.uint8)
    cv2.fillPoly(ha, [((aa - [x0, y0]) * s).astype(np.int32) + 4], 1)
    cv2.fillPoly(hb, [((bb - [x0, y0]) * s).astype(np.int32) + 4], 2)
    inter = np.count_nonzero((ha == 1) & (hb == 2))
    union = np.count_nonzero((ha == 1) | (hb == 2))
    return float(inter / union) if union else 0.0
