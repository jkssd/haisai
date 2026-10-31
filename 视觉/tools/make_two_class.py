"""从现有 7 点标注**自动派生** circle 类（零人工）。

每个靶子原本一行（class 0 target，7 点）。
现在额外输出一行 class 1 circle：
    bbox  = 中心圆的外接框
    kpts  = 4 个角点置 v=0（不参与损失），圆心/左/右 沿用原值
这样模型多学一个"只认圆"的分支，圆形外观稳定、跨场景泛化更好。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from target_pose.labels import Sample  # noqa: E402
from target_pose.config import SCENES    # noqa: E402

NKPT = 7


def circle_row(o, w, h):
    c = o.center; r = o.radius
    if r < 2:
        return None
    x1, y1 = c[0] - r, c[1] - r
    x2, y2 = c[0] + r, c[1] + r
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    row = ["1", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    # 4 个角点：坐标给个合法值，但 v=0（损失忽略）
    for k in range(4):
        px = c[0] + (r * 0.7 if k in (1, 2) else -r * 0.7)
        py = c[1] + (r * 0.7 if k in (2, 3) else -r * 0.7)
        row += [f"{px/w:.6f}", f"{py/h:.6f}", "0"]
    for pt in (c, o.circle_l, o.circle_r):
        row += [f"{pt[0]/w:.6f}", f"{pt[1]/h:.6f}", "2"]
    return " ".join(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "ann2"))
    ap.add_argument("--no-neg", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    n_img = n_obj = 0
    for scene, sc in SCENES.items():
        dst = out / scene
        dst.mkdir(parents=True, exist_ok=True)
        for lab in sorted((ROOT / "data" / "annotations" / scene).glob("*.txt")):
            if lab.stat().st_size == 0:
                continue
            img = sc.image_dir / f"{lab.stem}.jpg"
            if not img.exists():
                continue
            with __import__("PIL.Image", fromlist=["Image"]).open(img) as im:
                w, h = im.size
            s = Sample.from_txt(lab, img, w, h, path_hint=scene)
            lines = []
            for o in s.objs:
                lines.append(o.to_row(w, h, scene))
                cr = circle_row(o, w, h)
                if cr:
                    lines.append(cr)
                    n_obj += 1
            (dst / lab.name).write_text("\n".join(lines) + ("\n" if lines else ""))
            n_img += 1
    print(f"图片 {n_img}，派生 circle 实例 {n_obj} -> {out}")
    # 负样本：直接复用
    if not args.no_neg:
        nd = ROOT / "data" / "neg" / "labels"
        if nd.exists():
            dst = out / "neg"
            dst.mkdir(parents=True, exist_ok=True)
            for f in nd.glob("*.txt"):
                (dst / f.name).write_text("")
            print(f"负样本 {len(list(dst.glob('*.txt')))} 张（空标注）")


if __name__ == "__main__":
    main()
