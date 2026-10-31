"""按「同一张图内的比例」聚类判定档位 —— 不再用全局 大/中/小 类别。

设计要点（对应需求：取消大中小分类，改为同图比例聚类，不同图互相独立）：

  * ratio = r / ((w + h) / 4)，w/h 是靶框外接框宽高 —— 与距离无关，只由设计决定；
  * **每张图独立聚类**：把这张图里所有靶子的 log(ratio) 排序，
    在"相对间隔"超过阈值的相邻点之间切分，得到该图内部的 1/2/3 档；
  * 图与图之间不共享阈值、不共享簇心 —— 同一个"大"在不同图里可以是不同数值，
    只有图内相对关系才有意义。

所以不需要标定全局阈值，也不需要给训练标签打尺寸类别（一律 class 0 target + class 1 circle）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .config import ANN_DIR, SCENES

MIN_REL_GAP = 0.12      # 相邻 log-ratio 的相对间隔超过它就认为是不同档
K_MAX = 3               # 一张图里最多 3 档（大/中/小）


def ratio_from_box(w_quad: float, h_quad: float, r: float) -> float:
    """r / 靶框平均半边长。"""
    side = max(1e-6, (w_quad + h_quad) / 2.0)
    return float(r) / (side / 2.0)


def quad_rhw(corners: np.ndarray) -> tuple[float, float]:
    x = corners[:, 0]
    y = corners[:, 1]
    return float(x.max() - x.min()), float(y.max() - y.min())


def cluster_ratios(ratios: Sequence[float], min_rel_gap: float = MIN_REL_GAP,
                   k_max: int = K_MAX) -> List[int]:
    """对一张图里的比例做 1D 聚类，返回每项的分档号（0 = 最小）。

    用 log 空间按相对间隔切分：只有相邻两个比例差得足够开（>min_rel_gap）才分成两档，
    否则算同一档。单靶/比例相近的图自然得到全 0。
    """
    rs = np.asarray(list(ratios), float)
    n = len(rs)
    if n == 0:
        return []
    if n == 1 or np.allclose(rs, rs[0]):
        return [0] * n
    order = np.argsort(rs)
    srt = np.sort(rs)
    logs = np.log(np.clip(srt, 1e-6, None))
    gaps = np.diff(logs)
    # 相对间隔：gap / 局部均值，避免小比例处绝对差值被放大
    rel = gaps / np.maximum(1e-6, (logs[1:] + logs[:-1]) / 2.0)
    cuts = [i for i, g in enumerate(rel) if g > min_rel_gap]
    cuts = sorted(cuts, key=lambda i: -rel[i])[: max(0, k_max - 1)]
    cuts = sorted(cuts)
    labels_sorted = np.zeros(n, int)
    for k, c in enumerate(cuts):
        labels_sorted[c + 1:] = k + 1
    out = np.zeros(n, int)
    out[order] = labels_sorted
    return out.tolist()


def image_ratios(label_path: Path, img_w: int, img_h: int) -> List[Dict]:
    """从 7 点标签里读出每个靶子的 ratio（只看 class 0 行）。"""
    items = []
    for line in label_path.read_text().splitlines():
        v = line.split()
        if len(v) < 5 + 21:
            continue
        # 靶框行：4 个角点有可见性标记；派生出来的 circle 行角点 v 全 0，跳过
        if all(float(v[7 + 3 * k]) < 0.5 for k in range(4)):
            continue
        pts = np.array([[float(v[5 + 3 * i]) * img_w, float(v[6 + 3 * i]) * img_h]
                        for i in range(7)], np.float32)
        w, h = quad_rhw(pts[:4])
        r = float(np.linalg.norm(pts[6] - pts[5]) / 2.0)
        items.append({"ratio": ratio_from_box(w, h, r), "center": pts[4].tolist(),
                      "radius": r, "quad": pts[:4].tolist()})
    return items


def cluster_scene(scene: str, min_rel_gap: float = MIN_REL_GAP) -> List[Dict]:
    """对场景内每张图分别聚类（图与图独立）。"""
    sc = SCENES[scene]
    out = []
    for lab in sorted((ANN_DIR / scene).glob("*.txt")):
        img = sc.image_dir / f"{lab.stem}.jpg"
        if not (img.exists() and lab.stat().st_size > 0):
            continue
        from PIL import Image
        with Image.open(img) as im:
            w, h = im.size
        items = image_ratios(lab, w, h)
        if not items:
            continue
        groups = cluster_ratios([it["ratio"] for it in items], min_rel_gap)
        for it, g in zip(items, groups):
            it["group"] = int(g)
        centers = {}
        for it in items:
            centers.setdefault(it["group"], []).append(it["ratio"])
        out.append({
            "stem": lab.stem,
            "n": len(items),
            "items": items,
            "groups": {int(k): {"n": len(v), "mean": float(np.mean(v)),
                                "ratios": [round(float(x), 3) for x in v]}
                       for k, v in sorted(centers.items())},
        })
    return out


def summarize(rows: List[Dict]) -> Dict:
    """全场景统计：图内档位数分布 + 各档比例区间（只用于看数据，不当全局阈值用）。"""
    n_groups: Dict[int, int] = {}
    per_img = []
    for r in rows:
        k = len(r["groups"])
        n_groups[k] = n_groups.get(k, 0) + 1
        per_img.append((r["stem"], k, [round(g["mean"], 3) for g in r["groups"].values()]))
    return {"images": len(rows), "group_count_hist": dict(sorted(n_groups.items())),
            "per_image": per_img}


if __name__ == "__main__":  # pragma: no cover
    import sys
    scene = sys.argv[1] if len(sys.argv) > 1 else "comp"
    rows = cluster_scene(scene)
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=1))
