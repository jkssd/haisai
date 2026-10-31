"""YOLO-pose 标签的读写与几何度量。

关键点 7 个：靶框四角 + 圆心 + 圆的左右极点。
    半径 = |r_right - r_left| / 2
    相对半径 = 半径 / (靶框平均边长 / 2)   ← 与拍摄距离无关，用来分大/中/小
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

TARGET_CLS = 0

from .config import CENTER, N_KPT, R_LEFT, R_RIGHT, SIZE_CLASS_NAMES_CN
from .size_class import assign_classes, ratio_to_cls


@dataclass
class Obj:
    """一个靶子实例（像素坐标）。"""

    corners: np.ndarray                       # (4,2) TL,TR,BR,BL
    center: np.ndarray                        # (2,) 靶心（圆心）
    circle_l: Optional[np.ndarray] = None     # (2,) 圆左极点
    circle_r: Optional[np.ndarray] = None     # (2,) 圆右极点
    vis: List[int] = field(default_factory=lambda: [2] * N_KPT)
    cls: int = 0

    # ---------------------------------------------------------------- 派生量
    def __post_init__(self):
        self.corners = np.asarray(self.corners, float).reshape(4, 2)
        self.center = np.asarray(self.center, float).reshape(2)
        if self.circle_l is None or self.circle_r is None:
            r = self.side / 3.1
            u = self.axis
            self.circle_l = self.center - u * r
            self.circle_r = self.center + u * r
        self.circle_l = np.asarray(self.circle_l, float).reshape(2)
        self.circle_r = np.asarray(self.circle_r, float).reshape(2)
        if len(self.vis) < N_KPT:
            self.vis = list(self.vis) + [2] * (N_KPT - len(self.vis))

    @property
    def points(self) -> np.ndarray:
        """(7,2)：四角 + 圆心 + 圆左右极点。"""
        return np.vstack([self.corners, self.center[None, :],
                          self.circle_l[None, :], self.circle_r[None, :]])

    @property
    def bbox(self) -> np.ndarray:
        p = self.points
        return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], float)

    @property
    def area(self) -> float:
        q = self.corners
        x, y = q[:, 0], q[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @property
    def side(self) -> float:
        q = self.corners
        return float(np.mean([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]))

    @property
    def axis(self) -> np.ndarray:
        """靶框的水平轴（上边与下边的平均方向），圆的极点沿它放置。"""
        top = self.corners[1] - self.corners[0]
        bot = self.corners[2] - self.corners[3]
        v = top + bot
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else np.array([1.0, 0.0])

    @property
    def radius(self) -> float:
        # 没有圆（circle_l / circle_r 为 None）时返回 0，调用方据此走"仅框"分支
        if self.circle_l is None or self.circle_r is None:
            return 0.0
        return float(np.linalg.norm(self.circle_r - self.circle_l) / 2.0)

    @property
    def radius_rel(self) -> float:
        """半径 / 靶框半边长。与拍摄距离无关。"""
        half = self.side / 2.0
        return float(self.radius / half) if half > 1e-6 else 0.0

    @property
    def size_cls(self) -> int:
        return ratio_to_cls(self.radius_rel)

    def size_cls_in(self, scene):
        return ratio_to_cls(self.radius_rel, scene)

    @property
    def circle_labeled(self) -> bool:
        """圆心/极性点是否被真正标注（找不到圆时会被置 v=0）。"""
        return min(self.vis[CENTER], self.vis[R_LEFT], self.vis[R_RIGHT]) > 0

    def cls_for_label(self, scene: Optional[str] = None) -> int:
        """写盘用的类别：有圆按比例（按场景阈值）推导，没圆就沿用显式给的 cls。"""
        if self.circle_labeled and self.radius > 1:
            return ratio_to_cls(self.radius_rel, scene)
        return int(self.cls)

    @property
    def size_name(self) -> str:
        return SIZE_CLASS_NAMES_CN[self.size_cls]

    # ---------------------------------------------------------------- 操作
    def sort_corners(self) -> "Obj":
        p = self.corners
        s, d = p[:, 0] + p[:, 1], p[:, 0] - p[:, 1]
        self.corners = np.stack([p[np.argmin(s)], p[np.argmax(d)],
                                 p[np.argmax(s)], p[np.argmin(d)]]).astype(float)
        return self

    def set_circle(self, center, radius: float) -> "Obj":
        """以靶框轴为方向，按半径重新放两个极点。"""
        c = np.asarray(center, float)
        u = self.axis
        self.center = c
        self.circle_l = c - u * radius
        self.circle_r = c + u * radius
        return self

    def clipped(self, w: int, h: int) -> "Obj":
        o = Obj(self.corners.copy(), self.center.copy(),
                self.circle_l.copy(), self.circle_r.copy(), list(self.vis), self.cls)
        for arr in (o.corners, o.center[None, :], o.circle_l[None, :], o.circle_r[None, :]):
            arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
            arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
        return o

    def valid(self, w: int, h: int) -> bool:
        if not np.all(np.isfinite(self.corners)) or not np.all(np.isfinite(self.center)):
            return False
        if self.area < 16:
            return False
        x1, y1, x2, y2 = self.bbox
        if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
            return False
        return True

    def to_row(self, w: int, h: int, scene: Optional[str] = None,
               cls_override: Optional[int] = None) -> str:
        x1, y1, x2, y2 = self.bbox
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        # 尺寸分档（大/中/小）已废弃：训练标签只保留单一 target 类（class 0），
        # 档位改为「同图比例聚类」在推理时算（见 ratio_cluster.py）。
        cls = int(cls_override) if cls_override is not None else TARGET_CLS
        row = [f"{cls}", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
        for (px, py), v in zip(self.points, self.vis):
            row += [f"{px / w:.6f}", f"{py / h:.6f}", f"{int(v)}"]
        return " ".join(row)


@dataclass
class Sample:
    """一张图 + 若干靶子。"""

    image: Path
    width: int
    height: int
    objs: List[Obj] = field(default_factory=list)
    path_hint: str = ""

    @staticmethod
    def from_txt(txt: Path, image: Path, width: int, height: int, path_hint: str = "") -> "Sample":
        objs: List[Obj] = []
        if txt.exists():
            for line in txt.read_text().strip().splitlines():
                parts = line.split()
                legacy = False
                if len(parts) >= 5 + 3 * N_KPT:
                    n_kpt = N_KPT
                elif len(parts) >= 5 + 15:      # 旧版 5 点：四角 + 圆心
                    n_kpt = 5
                    legacy = True
                else:
                    continue
                cls = int(float(parts[0]))
                kp = np.array([float(v) for v in parts[5:5 + 3 * n_kpt]]).reshape(n_kpt, 3)
                pts = kp[:, :2] * [width, height]
                vis = [int(v) for v in kp[:, 2]]
                if legacy:
                    o = Obj(corners=pts[:4], center=pts[CENTER], vis=vis, cls=cls)
                    o.legacy = True          # 圆由经验比例补出，需复核
                    objs.append(o)
                else:
                    objs.append(Obj(corners=pts[:4], center=pts[CENTER],
                                    circle_l=pts[R_LEFT], circle_r=pts[R_RIGHT],
                                    vis=vis, cls=cls))
        return Sample(image=image, width=width, height=height, objs=objs, path_hint=path_hint)

    def to_txt(self) -> str:
        idx = [i for i, o in enumerate(self.objs) if o.circle_labeled and o.radius > 1]
        cls_map = {}
        if idx:
            cs = assign_classes([self.objs[i].radius_rel for i in idx], self.path_hint)
            cls_map = {i: int(c) for i, c in zip(idx, cs)}
        lines = [o.to_row(self.width, self.height, self.path_hint, cls_override=cls_map.get(k))
                 for k, o in enumerate(self.objs)]
        return "\n".join(lines) + ("\n" if lines else "")

    @property
    def n_reviewed(self) -> int:
        return len(self.objs)


# --------------------------------------------------------------------------- #
# 度量
# --------------------------------------------------------------------------- #
def kpt_error_px(a: Obj, b: Obj) -> float:
    return float(np.mean(np.linalg.norm(a.points - b.points, axis=1)))


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_objects(pred: Sequence[Obj], gt: Sequence[Obj], iou_thr: float = 0.3):
    if not pred or not gt:
        return [], list(range(len(pred))), list(range(len(gt)))
    ious = np.zeros((len(pred), len(gt)))
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            ious[i, j] = bbox_iou(p.bbox, g.bbox)
    pairs, used_p, used_g = [], set(), set()
    for _ in range(min(len(pred), len(gt))):
        i, j = np.unravel_index(np.argmax(ious), ious.shape)
        if ious[i, j] < iou_thr:
            break
        pairs.append((i, j))
        used_p.add(i)
        used_g.add(j)
        ious[i, :] = -1
        ious[:, j] = -1
    return pairs, [i for i in range(len(pred)) if i not in used_p], [j for j in range(len(gt)) if j not in used_g]


def bullseye_error(a: Obj, b: Obj) -> float:
    return float(np.linalg.norm(a.center - b.center))


def radius_error(a: Obj, b: Obj) -> float:
    return float(abs(a.radius - b.radius))
