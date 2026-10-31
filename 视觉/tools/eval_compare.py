"""多权重 / 多测试集对比：误检、固定测试图、打印靶纸、赛局帧。

用法::

    python tools/eval_compare.py --weights runs/v8/weights/best.pt runs/v10/weights/best.pt
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def dets(model, paths, imgsz=640, conf=0.25):
    res = model.predict([str(p) for p in paths], imgsz=imgsz, conf=conf, verbose=False)
    out = []
    for p, r in zip(paths, res):
        rows = []
        kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((0, 7, 3))
        for i, b in enumerate(r.boxes):
            k = kps[i]
            rad = float(np.linalg.norm(k[5][:2] - k[6][:2]) / 2)
            rows.append((int(b.cls), float(b.conf), float(k[4][0]), float(k[4][1]), rad))
        out.append((str(p), rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    from ultralytics import YOLO

    sets = {
        "横幅负样本": sorted(glob.glob(str(ROOT / "data/neg_banner/images/*.jpg")))[:args.limit],
        "背景负样本": sorted(glob.glob(str(ROOT / "data/neg/images/*.jpg")))[:args.limit],
        "固定测试图": [str(ROOT / "out/t1.png"), str(ROOT / "out/t2.png")],
        "打印靶纸": sorted(glob.glob(str(ROOT / "out/print_targets/target_print_*.png"))),
        "赛局帧": sorted(glob.glob(str(ROOT / "data/new15/img/*.jpg"))),
    }
    for w in args.weights:
        m = YOLO(w)
        print(f"\n=== {w} ===")
        for name, paths in sets.items():
            if not paths:
                continue
            rows = dets(m, paths, args.imgsz, args.conf)
            n_det = sum(len(r[1]) for r in rows)
            n_img_fp = sum(1 for r in rows if r[1])
            cls = Counter(d[0] for r in rows for d in r[1])
            print(f"{name:8s} 图 {len(rows):3d}  检出 {n_det:3d}  (target {cls.get(0,0)} / circle {cls.get(1,0)})"
                  f"  有检出图 {n_img_fp}")
            if name in ("固定测试图", "打印靶纸"):
                for p, rr in rows:
                    s = " ".join(f"{'C' if c else 'T'}{cf:.2f}@({x:.0f},{y:.0f})r{rad:.0f}"
                                 for c, cf, x, y, rad in rr)
                    print(f"    {Path(p).name}: [{s}]")


if __name__ == "__main__":
    main()
