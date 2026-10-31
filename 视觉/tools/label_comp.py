"""给 15 张赛局截图打 7 点标签（圆+靶框），圆靠模型+暗斑检测，靶框用模板比例。

赛局场景 = 一排蓝色立板，每块板中间一个**黑色圆孔**。
  圆孔 = 圆心/半径（模型 v7 的 circle 分支在 0.75~0.86，可用）
  靶框 = 立板外框；立板是同一批印品，物理比例恒定 →
         宽 = W_R * r，高 = H_R * r，孔心在竖直方向 HOLE_DY * 高 处
比例由 n07 实测标定（左板 134x171，孔 r=30）：W_R=4.47 H_R=5.70 HOLE_DY=0.47

只写标签，不裁图；渲染预览用于人工核对（pose.py review 也能改）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

W_R = 4.47
H_R = 5.70
HOLE_DY = 0.47        # 孔心在板高方向的位置（0=上边，1=下边）


def dark_holes(img: np.ndarray) -> list[tuple[float, float, float]]:
    """深色近圆的孔：V 低 + 圆度高，且周围是蓝色板面。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    blue = cv2.inRange(hsv, (95, 60, 25), (145, 255, 235))
    m = ((v < 70) & (cv2.dilate(blue, np.ones((9, 9), np.uint8)) > 0)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 120:
            continue
        per = cv2.arcLength(c, True)
        if per <= 0:
            continue
        circ = 4 * np.pi * a / per ** 2
        (cx, cy), rr = cv2.minEnclosingCircle(c)
        if circ < 0.6 or a / (np.pi * rr * rr) < 0.55:
            continue
        out.append((float(cx), float(cy), float(rr)))
    return out


def model_holes(imgs: list[Path]) -> dict[str, list[tuple[float, float, float]]]:
    from ultralytics import YOLO
    m = YOLO(str(ROOT / "runs" / "v7" / "weights" / "best.pt"))
    res = m.predict([str(p) for p in imgs], imgsz=640, conf=0.2, verbose=False)
    out = {}
    for p, r in zip(imgs, res):
        lst = []
        if r.keypoints is not None:
            for i, b in enumerate(r.boxes):
                if int(b.cls) != 1:
                    continue
                k = r.keypoints.data[i].cpu().numpy()
                c, rl, rr_ = k[4], k[5], k[6]
                rad = float(np.linalg.norm(rl[:2] - rr_[:2]) / 2)
                if rad < 3:
                    continue
                lst.append((float(c[0]), float(c[1]), rad))
        out[str(p)] = lst
    return out


def merge(a, b, tol=0.45):
    out = list(a)
    for x, y, r in b:
        if all(np.hypot(x - u, y - v) > tol * max(r, rr) for u, v, rr in out):
            out.append((x, y, r))
    return out


ASPECT = 1.52          # 板高/板宽（n05 实测 155x235）
BOARD_W_PER_SPACING = 0.93   # 板宽 ≈ 相邻孔间距 * 0.93（板子相邻、中间有窄缝）


def row_spacing(holes, i):
    """第 i 块板到邻居的孔间距（只有一块板时返回 None）。"""
    if len(holes) < 2:
        return None
    if i + 1 < len(holes):
        return holes[i + 1][0] - holes[i][0]
    return holes[i][0] - holes[i - 1][0]


def quad_from_row(cx, cy, r, spacing=None):
    """有邻居就按孔间距定板宽，否则退回 4.47r 模板。"""
    if spacing:
        w = BOARD_W_PER_SPACING * spacing
    else:
        w = W_R * r
    h = ASPECT * w
    top = cy - HOLE_DY * h
    left = cx - w / 2
    return np.array([[left, top], [left + w, top], [left + w, top + h], [left, top + h]], np.float32)


def quad_from_circle(cx, cy, r, tilt_deg=0.0):
    w, h = W_R * r, H_R * r
    top = cy - HOLE_DY * h
    left = cx - w / 2
    pts = np.array([[left, top], [left + w, top], [left + w, top + h], [left, top + h]], np.float32)
    if tilt_deg:
        M = cv2.getRotationMatrix2D((cx, cy), tilt_deg, 1.0)
        pts = cv2.transform(pts[None], M)[0]
    return pts


def write_label(path: Path, cx, cy, r, quad, w_img, h_img):
    q = quad.copy()
    q[:, 0] = np.clip(q[:, 0], 0, w_img - 1)
    q[:, 1] = np.clip(q[:, 1], 0, h_img - 1)
    x1, y1 = q.min(0)
    x2, y2 = q.max(0)
    bx = (x1 + x2) / 2 / w_img
    by = (y1 + y2) / 2 / h_img
    bw = (x2 - x1) / w_img
    bh = (y2 - y1) / h_img
    ratio = r / ((bw * w_img + bh * h_img) / 4)     # r / 半边长（近似）
    row = [f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}"]
    for px, py in q:
        row.append(f"{px / w_img:.6f} {py / h_img:.6f} 2")
    row.append(f"{cx / w_img:.6f} {cy / h_img:.6f} 2")
    for s in (-1, 1):
        row.append(f"{(cx + s * r) / w_img:.6f} {cy / h_img:.6f} 2")
    # 只写 target 行：circle 行由 tools/make_two_class.py 之后派生，
    # 免得复核工具把同一个靶子显示成两个。
    return " ".join(row) + "\n", ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "new15" / "img"))
    ap.add_argument("--out", default=str(ROOT / "data" / "annotations" / "comp"))
    ap.add_argument("--render", default=str(ROOT / "out" / "comp_preview"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    imgs = sorted(list(src.glob("*.jpg")) + list(src.glob("*.png")))
    mh = model_holes(imgs)
    out = Path(args.out)
    rnd = Path(args.render)
    rnd.mkdir(parents=True, exist_ok=True)
    ratios = {}
    for p in imgs:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        holes = merge(dark_holes(img), mh.get(str(p), []))
        holes.sort(key=lambda t: t[0])
        lines, vis = [], img.copy()
        for i, (cx, cy, r) in enumerate(holes):
            q = quad_from_row(cx, cy, r, row_spacing(holes, i))
            lab, ratio = write_label(out / f"{p.stem}.txt", cx, cy, r, q, w, h)
            lines.append(lab)
            ratios.setdefault(p.stem, []).append(round(ratio, 3))
            qi = q.astype(int)
            cv2.polylines(vis, [qi], True, (0, 255, 0), 2)
            for i, (px, py) in enumerate(qi):
                cv2.circle(vis, (px, py), 3, (0, 200, 255), -1)
                cv2.putText(vis, str(i), (px + 4, py - 4), 0, 0.45, (0, 255, 255), 1)
            cv2.drawMarker(vis, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
            cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 0, 255), 1)
            cv2.putText(vis, f"r{r:.1f}", (int(cx) - 12, int(cy) + int(r) + 14), 0, 0.45, (255, 255, 0), 1)
        if args.apply and lines:
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{p.stem}.txt").write_text("".join(lines))
        cv2.imwrite(str(rnd / f"{p.stem}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(p.stem, len(holes), [(round(x), round(y), round(r)) for x, y, r in holes])
    print({k: [float(x) for x in v] for k, v in ratios.items()})


if __name__ == "__main__":
    main()
