"""把中心圆「擦掉」，只留靶框 —— 逼模型学会「没有圆也能定位靶子」。

为什么有用：
  * 现场圆常看不到：反光、距离远、印刷色和板面接近、被遮挡、画面糊；
  * 靶心本来就等于靶框的对角线交点（透视一致），框在就能算靶心，实测中位 3.0px；
  * 现在的模型是靠「圆」把靶子认出来的，圆一糊就整个漏检（板端实测就是这种）。

做法：对每个靶子以概率 p 把圆涂成板面颜色（取圆环周围的板面中位色），
标签里把中心/左右极点的可见性置 0，并删掉派生出来的 circle 行。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from target_pose.config import SCENES  # noqa: E402


def parse(rows_txt: str):
    out = []
    for line in rows_txt.splitlines():
        v = line.split()
        if len(v) >= 5 + 21:
            out.append(v)
    return out


def obj_pts(row):
    return np.array([[float(row[5 + 3 * i]), float(row[6 + 3 * i])] for i in range(7)], np.float32)


def is_circle_row(row):
    return all(float(row[7 + 3 * k]) < 0.5 for k in range(4))


def erase_circle(img, pts, w, h, rng):
    """用环外的板面中位色把圆盖住（带一点噪声，避免纯色块）。"""
    c = np.array([pts[4][0] * w, pts[4][1] * h])
    r = float(np.linalg.norm((pts[6] - pts[5]) * [w, h]) / 2)
    if r < 3:
        return img
    R = int(r * 1.35)
    x0, y0 = max(0, int(c[0] - R)), max(0, int(c[1] - R))
    x1, y1 = min(w, int(c[0] + R)), min(h, int(c[1] + R))
    if x1 <= x0 or y1 <= y0:
        return img
    yy, xx = np.mgrid[y0:y1, x0:x1]
    ring = ((xx - c[0]) ** 2 + (yy - c[1]) ** 2 > (r * 1.15) ** 2) & \
           ((xx - c[0]) ** 2 + (yy - c[1]) ** 2 < (r * 1.34) ** 2)
    patch = img[y0:y1, x0:x1]
    if ring.sum() < 20:
        return img
    col = np.median(patch[ring], axis=0)
    mask = ((xx - c[0]) ** 2 + (yy - c[1]) ** 2) < (r * 1.02) ** 2
    fill = col[None, None, :] + np.random.normal(0, 3, (mask.sum(), 3))
    patch[mask] = np.clip(fill, 0, 255).astype(np.uint8)
    img[y0:y1, x0:x1] = patch
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "frame_only"))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--erase-p", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for f in list((out / "images").glob("*")) + list((out / "labels").glob("*")):
        f.unlink()

    pool = []
    for scene, sc in SCENES.items():
        for lab in sorted((ROOT / "data" / "ann2" / scene).glob("*.txt")):
            img = sc.image_dir / f"{lab.stem}.jpg"
            meta = ROOT / "data" / "meta" / scene / f"{lab.stem}.json"
            if scene == "comp" and meta.exists():
                import json
                try:
                    if not json.loads(meta.read_text()).get("reviewed"):
                        continue
                except Exception:
                    pass
            if img.exists() and lab.stat().st_size > 0:
                pool.append((scene, img, lab))
    print(f"素材 {len(pool)}")
    rng.shuffle(pool)

    made = 0
    i = 0
    while made < args.n and i < len(pool) * 3:
        scene, img_p, lab_p = pool[i % len(pool)]
        i += 1
        rows = parse(lab_p.read_text())
        frames = [r for r in rows if not is_circle_row(r)]
        circles = [r for r in rows if is_circle_row(r)]
        if not frames:
            continue
        img = cv2.imread(str(img_p))
        h, w = img.shape[:2]
        s = 1000.0 / max(w, h)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        keep = []
        for row in frames:
            pts = obj_pts(row)
            keep.append(row)
            if rng.random() < args.erase_p:
                img = erase_circle(img, pts, w, h, rng)
                # 圆心/极点可见性置 0
                for k in (4, 5, 6):
                    row[7 + 3 * k] = "0"
        # 只保留每个框自己的 circle 行（被擦掉的那个框不保留）
        text = "\n".join(" ".join(r) for r in keep) + "\n"
        name = f"fo_{made:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (out / "labels" / f"{name}.txt").write_text(text)
        made += 1
        if made % 100 == 0:
            print(f"  {made}/{args.n}")
    print(f"擦圆样本 {made} -> {out}")


if __name__ == "__main__":
    main()
