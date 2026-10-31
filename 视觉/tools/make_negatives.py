"""生成空标注负样本（参考前作 farneg_B / neg_crops 的做法）。

三类负样本：
  1. crop：从有靶子的图里，裁一块**完全避开所有靶框**的区域（背景/桌面/墙体）
  2. zoom：把整图缩小后贴到画布上，但**抹掉靶子区域**（只剩背景）
  3. scene：现场帧（无靶子的画面），由 --scene-dir 指定

前作数据：80 张 farneg + 160 张 neg_crops + 17 张 scene_neg，
训练集里负样本占 17.1%，@0.4 阈值误检 0/80。
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from target_pose.labels import Sample  # noqa: E402
from target_pose.config import SCENES     # noqa: E402


def target_boxes(objs, w, h, pad=0.15):
    out = []
    for o in objs:
        x1, y1, x2, y2 = o.bbox
        pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
        out.append((x1 - pw, y1 - ph, x2 + pw, y2 + ph))
    return out


def overlaps(box, others):
    x1, y1, x2, y2 = box
    for a, b, c, d in others:
        if x1 < c and x2 > a and y1 < d and y2 > b:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "neg"))
    ap.add_argument("--crops", type=int, default=180)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    pool = []
    for scene, sc in SCENES.items():
        for lab in sorted((ROOT / "data" / "annotations" / scene).glob("*.txt")):
            img = sc.image_dir / f"{lab.stem}.jpg"
            if img.exists() and lab.stat().st_size > 0:
                pool.append((scene, img, lab))
    print(f"素材 {len(pool)} 张")
    rng.shuffle(pool)

    made = 0
    size = args.size
    for scene, img_p, lab_p in pool:
        if made >= args.crops:
            break
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        s = Sample.from_txt(lab_p, img_p, w, h, path_hint=scene)
        boxes = target_boxes(s.objs, w, h)
        # 随机试若干次，找一块不碰靶子的裁剪
        ok = False
        for _ in range(30):
            sc = rng.choice([1.0, 0.75, 0.55])
            cw, ch = int(size / sc), int(size / sc)
            if cw >= w or ch >= h:
                cw, ch = min(cw, w), min(ch, h)
            x0 = rng.randint(0, max(0, w - cw))
            y0 = rng.randint(0, max(0, h - ch))
            if overlaps((x0, y0, x0 + cw, y0 + ch), boxes):
                continue
            crop = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (size, size))
            ok = True
            break
        if not ok:
            continue
        name = f"neg_{made:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (out / "labels" / f"{name}.txt").write_text("")     # 空标注 = 负样本
        made += 1
        if made % 60 == 0:
            print(f"  {made}/{args.crops}")

    print(f"生成负样本 {made} 张 -> {out}")


if __name__ == "__main__":
    main()
