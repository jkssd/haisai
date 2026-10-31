"""推理：单图/目录 → 靶框四角点 + 靶心（像素坐标 & 靶面归一化坐标）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import CENTER, KPT_NAMES, N_KPT
from .geometry import (is_convex_quad, order_quad_corners, quad_area, quad_center,
                       quad_side, to_target_coords)
from .labels import Obj
from .predict import _load_model, predict_objs, predict_objs_cls
from .refine import refine_target

# 靶心 = 圆心与「角点对角线交点」的加权融合；权重由 45 个复核靶子实测扫出来的
CIRCLE_W = 0.85      # 精修圆心 raw框心 的倒方差定权：σf=3.04px, σc=1.30px
UV_TOL = 0.2          # 圆心相对靶面中心的容差（归一化）；超过就当误检


def solve_image(model, img: np.ndarray, imgsz: int = 1280, conf: float = 0.25,
                refine: bool = True, use_classes=(0, 1)) -> List[Dict]:
    """返回每个靶子的解算结果。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    preds3 = [p for p in predict_objs_cls(model, img, imgsz, conf, 0.55) if p[2] in use_classes]
    # 框 + 圆配对：圆落在框的靶面坐标 [0,1] 内就算同一块靶子，取分数最高的那个圆
    frames = [(o, s) for o, s, c in preds3 if c == 0]
    circles = [(o, s) for o, s, c in preds3 if c == 1]
    preds = []
    for obj, sc in frames:
        best, bs = None, -1.0
        for cobj, csc in circles:
            uv = to_target_coords(obj.corners, cobj.center[None, :])[0]
            # 圆必须在框内、且离靶面中心足够近（设计上圆心就在正中）——
            # 偏得离谱的多半是背景里别的圆形图案，不配对，避免把误检带进来
            if (abs(uv[0] - 0.5) <= UV_TOL and abs(uv[1] - 0.5) <= UV_TOL
                    and -0.05 <= uv[0] <= 1.05 and -0.05 <= uv[1] <= 1.05 and csc > bs):
                best, bs = cobj, csc
        paired = best is not None
        if paired:
            obj.circle_l = best.circle_l
            obj.circle_r = best.circle_r
            obj.center = best.center.copy()
            sc = max(sc, bs)
        else:
            # 框那一行自己回归出来的圆点不可靠（训练时是辅助监督），
            # 没检出 class1 圆就当"没有圆"，靶心退回框心
            obj.circle_l = None
            obj.circle_r = None
        preds.append((obj, sc, paired))
    out: List[Dict] = []
    for obj, sc, paired in preds:
        info: Dict = {}
        # 几何门限：靶框必须是凸四边形、长宽比合理、面积别太小
        # （框是判据的基础，形状都不像框的直接丢，比调置信度阈值更稳）
        if not is_convex_quad(obj.corners):
            continue
        ws = np.linalg.norm(obj.corners[1] - obj.corners[0])
        hs = np.linalg.norm(obj.corners[3] - obj.corners[0])
        if ws < 4 or hs < 4:
            continue
        ar = max(ws, hs) / max(1e-6, min(ws, hs))
        if ar > 4.0 or quad_area(obj.corners) < 300:
            continue
        # 对角线交点用**未精修**的角点算：实测精修把框心误差从 3.0px 拉到 5.0px
        # （框边是软边，精修会往外/往里拉），而精修对圆心是有好处的（1.46→1.30px）
        qc_raw = quad_center(obj.corners)
        if refine:
            obj, info = refine_target(gray, obj)
        obj = obj.sort_corners()
        side = quad_side(obj.corners)
        qc = qc_raw                            # 角点对角线交点 = 靶面中心（透视一致）
        has_circle = bool(paired) and obj.radius > 1.0
        # 圆在就用圆为准，但往「框中心」拉 20%：实测中位误差 1.46px -> 1.03px
        # 圆不可用（没检出/被遮挡/太远）就只用框心，误差 ~3px 但仍然可用
        if has_circle:
            bull = CIRCLE_W * obj.center + (1.0 - CIRCLE_W) * qc
            source = "circle"
        else:
            bull = qc
            source = "frame"
        uv = to_target_coords(obj.corners, bull[None, :])[0]
        out.append({
            "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in obj.corners],
            "bullseye": [round(float(bull[0]), 2), round(float(bull[1]), 2)],
            "bullseye_frame": [round(float(qc[0]), 2), round(float(qc[1]), 2)],
            "bullseye_circle": [round(float(obj.center[0]), 2), round(float(obj.center[1]), 2)],
            "bullseye_source": source,
            "bullseye_uv": [round(float(uv[0]), 4), round(float(uv[1]), 4)],
            "radius": round(float(info.get("radius", 0.0)), 2),
            "radius_rel": round(float(obj.radius_rel), 4),
            "size": obj.size_name,          # 仅供参考：同图比例聚类在 solve_image 末尾重算
            "size_cls": int(obj.cls_for_label()),
            "ratio": round(float(obj.radius_rel), 4),
            "side": round(float(side), 2),
            "conf": round(float(sc), 4),
            "circle_ok": bool(info.get("circ_ok", False)),
            "center_offset_px": round(float(np.linalg.norm(obj.center - qc)) if has_circle else 0.0, 2),
            "center_offset_rel": round(float(np.linalg.norm(obj.center - qc)) / max(side, 1e-6)
                                       if has_circle else 0.0, 4),
            "quad_ok": bool(info.get("quad_ok", False)),
        })
    # 只有圆、没有框（框出画/被挡）：靶心直接用圆心坐标单独输出一条
    if 1 in use_classes:
        paired_centers = [o["bullseye_circle"] for o in out if o["bullseye_source"] == "circle"]
        for cobj, csc in circles:
            if any(np.linalg.norm(cobj.center - np.array(pc)) < 2.0 for pc in paired_centers):
                continue
            out.append({
                "corners": [],
                "bullseye": [round(float(cobj.center[0]), 2), round(float(cobj.center[1]), 2)],
                "bullseye_frame": None,
                "bullseye_circle": [round(float(cobj.center[0]), 2),
                                    round(float(cobj.center[1]), 2)],
                "bullseye_source": "circle_only",
                "bullseye_uv": [0.5, 0.5],
                "radius": round(float(cobj.radius), 2),
                "radius_rel": 0.0,
                "size": "档1",
                "size_cls": 0,
                "ratio": 0.0,
                "side": 0.0,
                "conf": round(float(csc), 4),
                "center_offset_px": 0.0,
                "center_offset_rel": 0.0,
                "circle_ok": True,
                "quad_ok": False,
                "ratio_group": 0,
            })

    # 档位按「同一张图内的比例」聚类（图与图独立），覆盖单实例的 size 字段
    if out:
        from .ratio_cluster import cluster_ratios

        groups = cluster_ratios([o["radius_rel"] for o in out])
        for o, g in zip(out, groups):
            o["ratio_group"] = int(g)
            o["size"] = f"档{g + 1}"
    return out


def draw(img: np.ndarray, sols: List[Dict]) -> np.ndarray:
    v = img.copy()
    for i, s in enumerate(sols):
        if s["corners"]:
            q = np.array(s["corners"], np.int32)
            cv2.polylines(v, [q], True, (0, 255, 0), 2)
        if s["corners"]:
            for p in np.array(s["corners"], np.int32):
                cv2.circle(v, tuple(p), 4, (0, 0, 255), -1)
        c = tuple(int(x) for x in s["bullseye"])
        cv2.drawMarker(v, c, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        if s["radius"] > 0:
            cv2.circle(v, c, int(s["radius"]), (255, 0, 255), 1)
        cv2.putText(v, f"#{i+1} {s['conf']:.2f}", (c[0] + 8, c[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description="靶心解算")
    ap.add_argument("--weights", default="target_pose.pt")
    ap.add_argument("--source", required=True, help="图片或目录")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-refine", dest="refine", action="store_false")
    ap.add_argument("--out", default=None, help="结果 json 路径")
    ap.add_argument("--save-dir", default=None, help="可视化输出目录")
    ap.set_defaults(refine=True)
    args = ap.parse_args(argv)

    src = Path(args.source)
    files = [src] if src.is_file() else sorted(
        [p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    model = _load_model(args.weights)
    results = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        sols = solve_image(model, img, args.imgsz, args.conf, args.refine)
        results.append({"image": str(f), "width": img.shape[1], "height": img.shape[0], "targets": sols})
        print(f"{f.name}: {len(sols)} 个靶子 " + "; ".join(
            f"({s['bullseye'][0]:.0f},{s['bullseye'][1]:.0f})" for s in sols))
        if args.save_dir:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(args.save_dir) / f.name), draw(img, sols))
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print(f"结果 -> {args.out}")


if __name__ == "__main__":
    main()
