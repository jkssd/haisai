"""把打印靶纸「拍成照片」：透视贴到真实/合成背景上，生成带标签的仿真图。

用途：
  * 训练前先验证打印设计能不能被检出（纯平图模型不吃，照片能）；
  * 作为「打印靶纸 → 现场拍摄」这一分布的补充训练数据。

标签同样是 7 点：4 角 = 靶框四角，中心 = 圆心，左右极点 = 圆在靶框水平方向的两端。
"""
from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DPI = 300
MM = DPI / 25.4
A3 = (int(297 * MM), int(420 * MM))
MARGIN_MM = 14
CAP_MM = 46
# 与 make_print_targets.py 保持一致
DESIGNS = [0.30, 0.34, 0.38, 0.42, 0.46]


def board_rect():
    W, H = A3
    mx, my = int(MARGIN_MM * MM), int(MARGIN_MM * MM)
    cap_h = int(CAP_MM * MM)
    return mx, my, W - mx, H - my - cap_h


def background(rng, w, h, bgs):
    if bgs and rng.random() < 0.7:
        bg = cv2.imread(str(rng.choice(bgs)))
        if bg is None:
            bg = None
        else:
            bh, bw = bg.shape[:2]
            s = max(w / bw, h / bh) * rng.uniform(1.0, 1.4)
            bg = cv2.resize(bg, (int(bw * s), int(bh * s)))
            bh, bw = bg.shape[:2]
            x0 = rng.randint(0, max(0, bw - w))
            y0 = rng.randint(0, max(0, bh - h))
            return bg[y0:y0 + h, x0:x0 + w].copy()
    base = np.array([rng.randint(20, 240) for _ in range(3)], np.float32)
    img = np.tile(base, (h, w, 1))
    img += np.random.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "out" / "print_targets" / "sim"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--size", default="1280x960")
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    w, h = (int(v) for v in args.size.split("x"))
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for f in list((out / "images").glob("*")) + list((out / "labels").glob("*")):
        f.unlink()

    bgs = sorted(glob.glob(str(ROOT / "data" / "neg" / "images" / "*.jpg")))[:120]
    px0, py0, px1, py1 = board_rect()
    made = 0
    for i in range(args.n):
        k = i % 5 + 1
        sheet = cv2.imread(str(ROOT / "out" / "print_targets" / f"target_print_{k:02d}.png"))
        if sheet is None:
            continue
        board = sheet[py0:py1, px0:px1].copy()
        bh, bw = board.shape[:2]
        ratio = DESIGNS[k - 1]

        # 目标四边形：随机大小 + 透视
        scale = rng.uniform(0.16, 0.38)
        tw, th = int(w * scale), int(w * scale * bh / bw)
        cx0 = rng.randint(int(tw / 2) + 4, max(int(tw / 2) + 5, w - int(tw / 2) - 4))
        cy0 = rng.randint(int(th / 2) + 4, max(int(th / 2) + 5, h - int(th / 2) - 4))
        x1, y1 = cx0 - tw / 2, cy0 - th / 2
        x2, y2 = cx0 + tw / 2, cy0 + th / 2
        j = 0.05 * min(tw, th)
        dst = np.float32([[x1 + rng.uniform(-j, j), y1 + rng.uniform(-j, j)],
                          [x2 + rng.uniform(-j, j), y1 + rng.uniform(-j, j)],
                          [x2 + rng.uniform(-j, j), y2 + rng.uniform(-j, j)],
                          [x1 + rng.uniform(-j, j), y2 + rng.uniform(-j, j)]])
        src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
        Hm = cv2.getPerspectiveTransform(src, dst)

        canvas = background(rng, w, h, bgs)
        warped = cv2.warpPerspective(board, Hm, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(np.full((bh, bw), 255, np.uint8), Hm, (w, h),
                                   flags=cv2.INTER_NEAREST)
        # 轻微羽化，避免硬边
        m3 = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)[..., None] / 255.0
        canvas = (warped * m3 + canvas * (1 - m3)).astype(np.uint8)

        # 光照/拍摄痕迹
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        gx, gy = rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25)
        shade = 1.0 + gx * (xx / w - 0.5) + gy * (yy / h - 0.5)
        canvas = np.clip(canvas.astype(np.float32) * shade[..., None]
                         * rng.uniform(0.75, 1.25), 0, 255).astype(np.uint8)
        if rng.random() < 0.6:
            canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.4, 1.4))
        canvas = np.clip(canvas.astype(np.float32)
                         + np.random.normal(0, rng.uniform(1, 5), canvas.shape), 0, 255).astype(np.uint8)

        # ---- 标签
        def h2(p):
            v = Hm @ np.array([p[0], p[1], 1.0])
            return v[0] / v[2], v[1] / v[2]

        corners = np.array([h2((0, 0)), h2((bw, 0)), h2((bw, bh)), h2((0, bh))], np.float32)
        ccx, ccy = bw / 2, bh / 2
        c = np.array(h2((ccx, ccy)), np.float32)
        rl = np.array(h2((ccx - ratio * bw / 2, ccy)), np.float32)
        rr = np.array(h2((ccx + ratio * bw / 2, ccy)), np.float32)
        rad = float(np.linalg.norm(rr - rl) / 2)
        allpts = np.vstack([corners, c[None], rl[None], rr[None]])
        x1_, y1_ = allpts.min(0)
        x2_, y2_ = allpts.max(0)
        row = (f"0 {(x1_ + x2_) / 2 / w:.6f} {(y1_ + y2_) / 2 / h:.6f} "
               f"{(x2_ - x1_) / w:.6f} {(y2_ - y1_) / h:.6f} ")
        row += " ".join(f"{p[0] / w:.6f} {p[1] / h:.6f} 2" for p in allpts)
        crow = (f"1 {c[0] / w:.6f} {c[1] / h:.6f} {2 * rad / w:.6f} {2 * rad / h:.6f} ")
        for kk in range(4):
            px = c[0] + (rad * 0.7 if kk in (1, 2) else -rad * 0.7)
            py = c[1] + (rad * 0.7 if kk in (2, 3) else -rad * 0.7)
            crow += f"{px / w:.6f} {py / h:.6f} 0 "
        for p in (c, rl, rr):
            crow += f"{p[0] / w:.6f} {p[1] / h:.6f} 2 "
        name = f"printsim_{made:04d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (out / "labels" / f"{name}.txt").write_text(row + "\n" + crow + "\n")
        made += 1
    print(f"仿真拍图 {made} -> {out}")


if __name__ == "__main__":
    main()
