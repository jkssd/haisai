"""对比三种靶心来源的精度（用人工复核的赛局帧做真值）：

  1. circle  —— 只用模型预测的圆心（现在 infer 的做法）
  2. frame   —— 只用 4 角点的对角线交点（框检测到就有，不依赖圆）
  3. fused   —— 两者按一致性/置信度加权融合

输出每个靶子的误差，以及「圆心相对靶面中心的偏移」，用来定融合权重和过滤阈值。
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from target_pose.geometry import quad_center, to_target_coords  # noqa: E402


def load_gt(stem: str, w: int, h: int):
    out = []
    p = ROOT / "data" / "annotations" / "comp" / f"{stem}.txt"
    for line in p.read_text().splitlines():
        v = line.split()
        if len(v) < 26:
            continue
        pts = np.array([[float(v[5 + 3 * i]) * w, float(v[6 + 3 * i]) * h] for i in range(7)], np.float32)
        out.append(pts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/v11/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--w-circle", type=float, default=0.5, help="融合时圆心权重")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    m = YOLO(args.weights)
    e_c, e_f, e_u = [], [], []
    offs = []
    n_match = 0
    for p in sorted(glob.glob(str(ROOT / "data" / "new15" / "img" / "*.jpg"))):
        im = cv2.imread(p)
        h, w = im.shape[:2]
        gt = load_gt(Path(p).stem, w, h)
        r = m.predict([p], imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((0, 7, 3))
        boxes, circles = [], []
        for i, b in enumerate(r.boxes):
            (boxes if int(b.cls) == 0 else circles).append((kps[i][:, :2], float(b.conf)))
        used = set()
        for g in gt:
            # 用真值角点找最近预测框
            best, bd = None, 1e9
            for j, (q, cf) in enumerate(boxes):
                if j in used:
                    continue
                d = np.linalg.norm(quad_center(g[:4]) - quad_center(q[:4]))
                if d < bd:
                    best, bd = j, d
            if best is None:
                continue
            side = float(np.linalg.norm(g[1] - g[0]) + np.linalg.norm(g[3] - g[0])) / 2
            if bd > 0.5 * side:
                continue
            used.add(best)
            q = boxes[best][0]
            pc = quad_center(q)
            # 把真值圆心投影到靶面坐标，看它离靶面中心多远
            uv = to_target_coords(g[:4], g[4][None, :])[0]
            offs.append((float(uv[0]) - 0.5, float(uv[1]) - 0.5))
            # 找这个框里最合适的圆
            cbest, cd = None, 1e9
            for cq, ccf in circles:
                d = np.linalg.norm(cq[4] - pc)
                if d < cd and d < 0.8 * side:
                    cbest, cd = cq, d
            e_f.append(float(np.linalg.norm(pc - g[4])))
            if cbest is not None:
                e_c.append(float(np.linalg.norm(cbest[4] - g[4])))
                wc = args.w_circle
                e_u.append(float(np.linalg.norm(wc * cbest[4] + (1 - wc) * pc - g[4])))
            n_match += 1

    def stat(x, name):
        x = np.array(x)
        print(f"{name:12s} n={len(x):3d} 中位 {np.median(x):5.2f}px  均值 {x.mean():5.2f}px  "
              f"P90 {np.percentile(x, 90):5.2f}px  最大 {x.max():5.2f}px")

    print(f"匹配 {n_match} 个靶子")
    stat(e_f, "frame(角点)")
    stat(e_c, "circle(圆心)")
    stat(e_u, f"fused w={args.w_circle}")
    offs = np.array(offs)
    print(f"真值圆心在靶面坐标里相对中心的偏移: 中位 |u-0.5|={np.median(np.abs(offs[:,0])):.3f} "
          f"|v-0.5|={np.median(np.abs(offs[:,1])):.3f}  (P95 {np.percentile(np.abs(offs[:,1]),95):.3f})")


if __name__ == "__main__":
    main()
