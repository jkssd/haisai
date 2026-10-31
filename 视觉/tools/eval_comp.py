"""在人工复核过的赛局帧上量化 v11/v8/v7：召回、角点误差、靶心误差、半径比例、同图分档。

GT = data/annotations/comp/*.txt（人工复核，7 点）
预测 = 模型 class 0（靶框）的输出，按靶心距离做最近匹配（阈值 = 0.5 × GT 边长）。
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from target_pose.ratio_cluster import cluster_ratios, ratio_from_box  # noqa: E402


def load_gt(stem: str, img_w: int, img_h: int):
    out = []
    p = ROOT / "data" / "annotations" / "comp" / f"{stem}.txt"
    for line in p.read_text().splitlines():
        v = line.split()
        if len(v) < 26:
            continue
        pts = np.array([[float(v[5 + 3 * i]) * img_w, float(v[6 + 3 * i]) * img_h] for i in range(7)], np.float32)
        out.append(pts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", default=["runs/v11/weights/best.pt"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO
    import cv2

    imgs = sorted(glob.glob(str(ROOT / "data" / "new15" / "img" / "*.jpg")))
    for w in args.weights:
        m = YOLO(w)
        tot_gt = tot_hit = 0
        cerr, kerr, rerr = [], [], []
        print(f"\n=== {w} ===")
        for p in imgs:
            im = cv2.imread(p)
            h, wd = im.shape[:2]
            gt = load_gt(Path(p).stem, wd, h)
            r = m.predict([p], imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            preds = []
            kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((0, 7, 3))
            for i, b in enumerate(r.boxes):
                if int(b.cls) != 0:
                    continue
                preds.append(kps[i][:, :2])   # 只留 xy
            used = set()
            hit = 0
            pairs = []
            for g in gt:
                d = [np.linalg.norm(g[4] - q[4]) for q in preds]
                if not d:
                    continue
                j = int(np.argmin(d))
                side = float(np.linalg.norm(g[1] - g[0]) + np.linalg.norm(g[2] - g[1])) / 2
                if d[j] < 0.5 * side and j not in used:
                    used.add(j)
                    hit += 1
                    pairs.append((g, preds[j]))
            for g, q in pairs:
                kerr.append(float(np.median(np.linalg.norm(g[:4] - q[:4], axis=1))))
                cerr.append(float(np.linalg.norm(g[4] - q[4])))
                gw, gh = np.ptp(g[:4], 0)
                qw, qh = np.ptp(q[:4], 0)
                rg = (float(np.linalg.norm(g[6] - g[5])) / 2) / ((gw + gh) / 4)
                rq = (float(np.linalg.norm(q[6] - q[5])) / 2) / ((qw + qh) / 4)
                rerr.append(abs(rq - rg) / max(rg, 1e-6))
            tot_gt += len(gt)
            tot_hit += hit
            # 同图分档
            ratios = [(float(np.linalg.norm(q[6] - q[5])) / 2) /
                      ((np.ptp(q[:4], 0)[0] + np.ptp(q[:4], 0)[1]) / 4) for q in preds]
            gps = cluster_ratios(ratios)
            print(f"  {Path(p).stem}: GT {len(gt)} 召回 {hit}/{len(gt)}  预测分档 {gps}  "
                  f"ratio {[round(x, 3) for x in ratios]}")
        if kerr:
            print(f"  合计: 召回 {tot_hit}/{tot_gt}  角点误差中位 {np.median(kerr):.1f}px  "
                  f"靶心误差中位 {np.median(cerr):.1f}px  比例相对误差中位 {np.median(rerr) * 100:.1f}%")


if __name__ == "__main__":
    main()
