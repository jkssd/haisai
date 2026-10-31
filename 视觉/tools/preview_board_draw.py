"""在 PC 上复刻板端画法，检查「一个靶子只出一个靶心标记」的三种情况：

  B = 框 + 圆 → 0.85*圆心 + 0.15*框心（黄）
  F = 只有框   → 对角线交点（青）
  C = 只有圆   → 圆心坐标（洋红）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from target_pose.geometry import quad_center  # noqa: E402
from target_pose.infer import solve_image  # noqa: E402
from target_pose.predict import _load_model  # noqa: E402

COL = {"B": (0, 255, 255), "F": (255, 255, 0), "C": (255, 0, 255)}


def draw(sol, img, scale=1.0):
    """按照板端规则画：框线 + 角点编号 + 圆环 + 唯一靶心标记。"""
    out = img.copy()
    for s in sol:
        if s["corners"]:
            q = np.array(s["corners"], np.int32)
            cv2.polylines(out, [q.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            for i, p in enumerate(q):
                cv2.circle(out, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
                cv2.putText(out, str(i + 1), (p[0] + 4, p[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if s["bullseye_source"] == "circle":
            cx, cy = s["bullseye_circle"]
            r = int(s["radius"])
            cv2.circle(out, (int(cx), int(cy)), r, (255, 0, 255), 2)
        src = {"circle": "B", "frame": "F"}.get(s["bullseye_source"], "C")
        bx, by = s["bullseye"]
        col = COL[src]
        cv2.drawMarker(out, (int(bx), int(by)), col, cv2.MARKER_CROSS, 22, 3)
        cv2.putText(out, src, (int(bx) + 10, int(by) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if s["radius"] > 1 and s["side"] > 1:
            rel = s["radius"] / (s["side"] / 2)
            cv2.putText(out, f"{rel:.2f}", (int(bx) - 40, int(by) - 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/target_pose_v11.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(ROOT / "out" / "board_draw_demo.jpg"))
    args = ap.parse_args()
    m = _load_model(args.weights)

    img = cv2.imread(str(ROOT / "out" / "t2.png"))
    # 三种情况都用同一张图，只控制"哪一类算可用"：
    #   两类都用 = 现场正常；只留框 = 圆没检出/被挡；只留圆 = 框没检出/出画
    cases = [
        ("框+圆 → B (融合)", img, solve_image(m, img, args.imgsz, args.conf)),
        ("只有框 → F (对角线交点)", img, solve_image(m, img, args.imgsz, args.conf, use_classes=(0,))),
        ("只有圆 → C (圆心)", img, solve_image(m, img, args.imgsz, args.conf, use_classes=(1,))),
    ]

    tiles = []
    for name, img, sol in cases:
        vis = draw(sol, img)
        vis = cv2.resize(vis, (426, 320))
        cv2.putText(vis, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        tiles.append(vis)
        print(name, [(s["bullseye_source"], [round(v) for v in s["bullseye"]], round(s["radius"], 1))
                     for s in sol])
    sheet = np.hstack(tiles)
    cv2.imwrite(args.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("->", args.out)


if __name__ == "__main__":
    main()
