"""靶子类型：由「圆半径 / 靶框半边长」这个与距离无关的比例来分。

为什么用比例而不是外观/像素半径：
  * 像素半径随距离变化，同一块靶子远近能差 3 倍；
  * 外观（白圆/蓝圆/黑底白圆）在同一场景里也会变，不可靠；
  * 比例 = 设计参数，同一型号恒定。

**按场景分别聚类**（重要）：三套靶子是不同实物，实测簇中心分别是

    shengai 0.213 / 0.442 / 0.665
    d2      0.264 / 0.510 / 0.764
    c2      0.375 / 0.554 / 0.756

用一套全局阈值会把 c2 的"小"判成"中"。所以阈值按 scene 存，类别含义是
「该场景里的大/中/小」，这也符合"每个环境一套靶"的实际。

类别不手填，由关键点推导；阈值用 `pose.py radius-stats --apply` 从标注里聚类得到。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import DATA, SIZE_CLASSES, SIZE_THRESHOLDS

CALIB_PATH = DATA / "size_classes.json"


# --------------------------------------------------------------------------- #
def _empty() -> Dict:
    return {"global": {"thresholds": list(SIZE_THRESHOLDS), "centers": [0.67, 0.44, 0.21],
                       "classes": list(SIZE_CLASSES), "source": "config-default"},
            "per_scene": {}}


def load_calibration() -> Dict:
    if CALIB_PATH.exists():
        try:
            raw = json.loads(CALIB_PATH.read_text())
        except json.JSONDecodeError:
            return _empty()
        if "per_scene" in raw or "global" in raw:
            base = _empty()
            base.update({k: v for k, v in raw.items() if k in ("global", "per_scene")})
            return base
        # 兼容旧格式（顶层直接是 thresholds）
        return {"global": raw, "per_scene": {}}
    return _empty()


def thresholds(scene: Optional[str] = None) -> List[float]:
    cal = load_calibration()
    if scene and scene in cal.get("per_scene", {}):
        return list(cal["per_scene"][scene]["thresholds"])
    return list(cal["global"]["thresholds"])


def ratio_to_cls(ratio: float, scene: Optional[str] = None) -> int:
    """0=大 1=中 2=小（比例越大说明圆相对靶框越大）。"""
    th = thresholds(scene)
    if ratio >= th[0]:
        return 0
    if ratio >= th[1]:
        return 1
    return 2


def assign_classes(ratios: List[float], scene: Optional[str] = None,
                   same_tol: float = 0.12) -> List[int]:
    """同一张图里的靶子一起判类：比例相近的必须是同一类。

    先按场景阈值给初值，再把「相对差 < same_tol」的靶子合并成一组、组内用中位比例重算，
    这样阈值附近的两个同型靶子不会被劈成两类；跨场景不比较。
    """
    if not ratios:
        return []
    cls = [ratio_to_cls(r, scene) for r in ratios]
    order = sorted(range(len(ratios)), key=lambda i: ratios[i])
    groups: List[List[int]] = []
    for i in order:
        if groups:
            prev = ratios[groups[-1][-1]]
            if abs(ratios[i] - prev) / max(abs(ratios[i]), 1e-6) < same_tol:
                groups[-1].append(i)
                continue
        groups.append([i])
    for g in groups:
        med = float(np.median([ratios[i] for i in g]))
        c = ratio_to_cls(med, scene)
        for i in g:
            cls[i] = c
    return cls


def radius_candidates(side: float, scene: Optional[str] = None) -> List[float]:
    """由靶框边长反推三种型号的圆半径候选（侧/r = 2 / ratio）。"""
    cal = load_calibration()
    e = cal.get("per_scene", {}).get(scene, cal["global"])
    centers = e.get("centers") or [0.67, 0.44, 0.21]
    # 顺序必须与类别 0/1/2（大/中/小）一致：半径从大到小
    out = sorted({round(side * c / 2.0, 2) for c in centers if c > 0.05}, reverse=True)
    return [r for r in out if r > 2]


def cls_name(cls: int) -> str:
    return SIZE_CLASSES[cls] if 0 <= cls < len(SIZE_CLASSES) else "unknown"


# --------------------------------------------------------------------------- #
def kmeans1d(x: np.ndarray, k: int = 3, iters: int = 80) -> Optional[np.ndarray]:
    xs = np.sort(np.asarray(x, float))
    if len(xs) < k:
        return None
    cs = np.quantile(xs, np.linspace(0.12, 0.88, k))
    for _ in range(iters):
        lab = np.argmin(np.abs(xs[:, None] - cs[None, :]), axis=1)
        new = np.array([xs[lab == j].mean() if (lab == j).any() else cs[j] for j in range(k)])
        if np.allclose(new, cs, atol=1e-6):
            break
        cs = new
    return np.sort(cs)


def _entry(ratios: List[float], k: int = 3) -> Optional[Dict]:
    cs = kmeans1d(np.asarray(ratios, float), k)
    if cs is None:
        return None
    thr = [(cs[i] + cs[i + 1]) / 2 for i in range(len(cs) - 1)]
    return {
        "thresholds": [float(x) for x in thr[::-1]],
        "centers": [float(x) for x in cs[::-1]],
        "classes": list(SIZE_CLASSES),
        "n": len(ratios),
        "k": len(cs),
        "source": "calibrated",
    }


def calibrate(per_scene_ratios: Dict[str, List[float]], k: int = 3, save: bool = True) -> Dict:
    """按场景聚类；样本不足 3 的场景回落到全局。"""
    cal = load_calibration()
    all_rel: List[float] = []
    for scene, ratios in per_scene_ratios.items():
        all_rel += list(ratios)
        e = _entry(ratios, k)
        if e is not None:
            cal["per_scene"][scene] = e
    g = _entry(all_rel, k)
    if g is not None:
        cal["global"] = g
    if save:
        CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALIB_PATH.write_text(json.dumps(cal, ensure_ascii=False, indent=1))
    return cal


def describe() -> str:
    cal = load_calibration()
    lines = ["当前阈值："]
    g = cal["global"]
    lines.append(f"  [全局] 大>={g['thresholds'][0]:.3f}  中>={g['thresholds'][1]:.3f}  簇 {[round(c, 3) for c in g.get('centers', [])]}")
    for scene, e in cal.get("per_scene", {}).items():
        lines.append(f"  [{scene}] 大>={e['thresholds'][0]:.3f}  中>={e['thresholds'][1]:.3f}  "
                     f"簇 {[round(c, 3) for c in e.get('centers', [])]}  n={e.get('n')}")
    return "\n".join(lines)
