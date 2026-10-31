"""基于已有标注做「外形/颜色/背景/曝光/干扰物」增强（标签自动跟着变）。

针对现场可能出现的分布做四类扰动（对应需求 1.1~1.4）：
  1. 干扰物：把商标/会徽类图形（圆环、齿轮、星、二维码块、条纹牌、文字块）
     随机贴到背景空白处，不碰靶框 —— 教模型「不是靶子的圆形图案别理」。
  2. 背景换色：靶框以外的区域整体换色（含蓝绿、紫、黑、灰这类相近色），
     也可以换成纯色/渐变的模拟墙面。
  3. 靶子外观：靶框内（圆以外）换色、换明度；中心圆单独填白/黑/红/绿/青/黄，
     有时加一圈对比描边；再加上随机缩放/旋转/透视来模拟远近与角度。
  4. 曝光：gamma、亮度、对比度、饱和、色温（白平衡）抖动。

所有几何变换都把 7 个关键点（4 角 + 圆心 + 圆左右极点）一起变换，
所以标签永远和图像对齐；圆极点沿旋转后的水平轴走，保持一致。
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

WORK = 1000          # 处理时长边缩到这个尺寸（训练只用 640）
CIRCLE_COLORS = [(255, 255, 255), (0, 0, 0), (40, 40, 205), (45, 175, 60),
                 (215, 60, 60), (0, 185, 250), (20, 120, 240), (170, 170, 175)]
BOARD_COLORS = [(30, 30, 34), (20, 40, 120), (40, 90, 30), (60, 30, 90),
                (200, 200, 205), (25, 25, 25), (10, 60, 110), (35, 60, 70)]


# ---------------------------------------------------------------- 载入/写出
def parse_label(p: Path):
    rows = []
    for line in p.read_text().splitlines():
        v = line.split()
        if len(v) < 5 + 21:
            continue
        cls = int(v[0])
        cx, cy, bw, bh = map(float, v[1:5])
        pts = np.array([[float(v[5 + 3 * i]), float(v[6 + 3 * i])] for i in range(7)], np.float32)
        vis = [int(float(v[7 + 3 * i])) for i in range(7)]
        rows.append({"cls": cls, "box": (cx, cy, bw, bh), "pts": pts, "vis": vis})
    return rows


def fmt_label(rows, w, h) -> str:
    out = []
    for r in rows:
        pts = r["pts"].copy()
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        vis = list(r["vis"])
        for i, (x, y) in enumerate(r["pts"]):
            if x < 1 or y < 1 or x > w - 2 or y > h - 2:
                vis[i] = 1
        x1, y1 = pts.min(0)
        x2, y2 = pts.max(0)
        row = [f"{r['cls']} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
               f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"]
        for (x, y), vis_i in zip(pts, vis):
            row.append(f"{x / w:.6f} {y / h:.6f} {vis_i}")
        out.append(" ".join(row))
    return "\n".join(out) + ("\n" if out else "")


def load(p: Path):
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    s = WORK / max(w, h)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------- 干扰物
def draw_logo(img, rng, cx, cy, R):
    """会徽/商标类干扰图形（不画成"靶子"那样干净的同心圆）。"""
    kind = rng.choice(["ring_ticks", "gear", "star", "qr", "bars", "stripes", "ring_dots", "shield"])
    col = tuple(int(c) for c in rng.choice([(255, 255, 255), (200, 220, 255), (60, 60, 60),
                                            (0, 160, 255), (30, 200, 120), (200, 80, 40)]))
    lw = max(1, int(R * rng.uniform(0.08, 0.14)))
    p = (int(cx), int(cy))
    if kind == "ring_ticks":
        cv2.circle(img, p, int(R), col, lw)
        n = rng.choice([10, 12, 16])
        for i in range(n):
            a = 2 * np.pi * i / n
            p1 = (int(cx + R * 0.7 * np.cos(a)), int(cy + R * 0.7 * np.sin(a)))
            p2 = (int(cx + R * 0.95 * np.cos(a)), int(cy + R * 0.95 * np.sin(a)))
            cv2.line(img, p1, p2, col, max(1, lw // 2))
    elif kind == "gear":
        pts = []
        n = rng.choice([10, 12, 14])
        for i in range(n * 2):
            a = 2 * np.pi * i / (n * 2)
            r = R if i % 2 == 0 else R * 0.75
            pts.append([int(cx + r * np.cos(a)), int(cy + r * np.sin(a))])
        cv2.polylines(img, [np.array(pts, np.int32)], True, col, lw)
    elif kind == "star":
        pts = []
        for i in range(10):
            a = -np.pi / 2 + np.pi * i / 5
            r = R * 0.6 if i % 2 == 0 else R * 0.24
            pts.append([int(cx + r * np.cos(a)), int(cy + r * np.sin(a))])
        cv2.fillPoly(img, [np.array(pts, np.int32)], col)
    elif kind == "qr":
        n = rng.choice([5, 7])
        cell = max(2, int(2 * R / n))
        x0, y0 = int(cx - n * cell / 2), int(cy - n * cell / 2)
        for i in range(n):
            for j in range(n):
                if rng.random() < 0.5:
                    cv2.rectangle(img, (x0 + i * cell, y0 + j * cell),
                                  (x0 + (i + 1) * cell - 1, y0 + (j + 1) * cell - 1), col, -1)
    elif kind == "bars":
        n = rng.randint(3, 5)
        bh = max(2, int(R * 0.28))
        w = 2 * R
        for i in range(n):
            y0 = int(cy - n * bh / 2 + i * bh)
            x1 = int(cx - w / 2 + rng.uniform(0, 0.35) * w)
            cv2.rectangle(img, (x1, y0), (int(cx + w / 2), y0 + bh - max(1, bh // 4)), col, -1)
    elif kind == "stripes":
        w, h = int(2 * R), int(1.2 * R)
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (235, 235, 235), -1)
        for i in range(0, w, max(3, int(R * 0.3))):
            cv2.line(img, (x0 + i, y0), (x0 + i + h // 2, y0 + h), col, max(1, lw // 2))
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (90, 40, 30), max(1, lw // 3))
    elif kind == "ring_dots":
        cv2.circle(img, p, int(R), col, lw)
        n = rng.choice([6, 8])
        for i in range(n):
            a = 2 * np.pi * i / n
            q = (int(cx + R * 0.55 * np.cos(a)), int(cy + R * 0.55 * np.sin(a)))
            cv2.circle(img, q, max(1, int(R * 0.08)), col, -1)
    else:
        pts = np.array([[cx, cy - R], [cx + R * 0.8, cy - R * 0.5], [cx + R * 0.8, cy + R * 0.4],
                        [cx, cy + R], [cx - R * 0.8, cy + R * 0.4], [cx - R * 0.8, cy - R * 0.5]],
                       np.int32)
        cv2.polylines(img, [pts], True, col, lw)


def stamp_distractors(img, mask_targets, labels, rng):
    """在背景空白处贴 1~4 个干扰图形（避开靶框）。"""
    h, w = img.shape[:2]
    rads = [np.linalg.norm(r["pts"][5] - r["pts"][6]) / 2 for r in labels]
    rads = [r for r in rads if r > 3] or [min(h, w) * 0.05]
    for _ in range(rng.randint(1, 4)):
        R = rng.uniform(0.5, 2.2) * float(np.median(rads))
        R = max(6.0, min(R, min(h, w) * 0.12))
        for _try in range(30):
            cx = rng.uniform(R, w - R)
            cy = rng.uniform(R, h - R)
            x0, y0 = int(cx - R), int(cy - R)
            x1, y1 = int(cx + R), int(cy + R)
            if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
                continue
            sub = mask_targets[max(0, y0):y1, max(0, x0):x1]
            if sub.size and sub.mean() > 2:
                continue
            draw_logo(img, rng, cx, cy, R)
            break


# ---------------------------------------------------------------- 颜色/曝光
def shift_hsv(img, mask, dh, ds, dv):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + dh) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * ds, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * dv, 0, 255)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if mask is None:
        return out
    return np.where(mask[..., None] > 0, out, img)


def synthetic_bg(img, mask_bg, rng):
    h, w = img.shape[:2]
    base = np.array(rng.choice(BOARD_COLORS + [(235, 238, 240), (18, 18, 20)]), np.float32)
    c2 = np.clip(base + np.random.uniform(-40, 40, 3), 5, 250)
    yy = np.linspace(0, 1, h, np.float32)[:, None, None]
    xx = np.linspace(0, 1, w, np.float32)[None, :, None]
    t = np.clip(rng.uniform(0, 1) * yy + (1 - rng.uniform(0, 1)) * xx, 0, 1)
    bg = base[None, None, :] * (1 - t) + c2[None, None, :] * t
    bg = bg + np.random.normal(0, rng.uniform(1.5, 6), bg.shape)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    return np.where(mask_bg[..., None] > 0, bg, img)


def exposure(img, rng):
    if rng.random() < 0.8:
        g = rng.uniform(0.7, 1.6)
        lut = np.clip(((np.arange(256) / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
        img = cv2.LUT(img, lut)
    if rng.random() < 0.8:
        img = np.clip(img.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25), 0, 255)
        img = img.astype(np.uint8)
    if rng.random() < 0.5:                     # 对比度
        a = rng.uniform(0.75, 1.3)
        img = np.clip((img.astype(np.float32) - 128) * a + 128, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:                     # 白平衡/色温
        b = img.astype(np.float32)
        b[..., 0] *= rng.uniform(0.85, 1.15)
        b[..., 2] *= rng.uniform(0.85, 1.15)
        img = np.clip(b, 0, 255).astype(np.uint8)
    if rng.random() < 0.22:                    # 过曝高光 / 欠曝
        if rng.random() < 0.45:
            img = np.clip(img.astype(np.float32) + rng.uniform(20, 45), 0, 255).astype(np.uint8)
        else:
            img = (img.astype(np.float32) * rng.uniform(0.4, 0.65)).astype(np.uint8)
    return img


# ---------------------------------------------------------------- 几何
def geom(img, rows, rng):
    h, w = img.shape[:2]
    ang = rng.uniform(-8, 8)
    sc = rng.uniform(0.75, 1.45)
    tx = rng.uniform(-0.08, 0.08) * w
    ty = rng.uniform(-0.08, 0.08) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[0, 2] += tx
    M[1, 2] += ty
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    for r in rows:
        pts = r["pts"].copy()
        pts = np.hstack([pts, np.ones((7, 1), np.float32)])
        r["pts"] = (pts @ M.T).astype(np.float32)
    if rng.random() < 0.25:                    # 轻微透视
        k = rng.uniform(0.02, 0.06)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.array([[rng.uniform(-k, k) * w, rng.uniform(-k, k) * h] for _ in range(4)],
                             np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, H, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        for r in rows:
            pts = np.hstack([r["pts"], np.ones((7, 1), np.float32)])
            r["pts"] = (pts @ H.T)[:, :2] / (pts @ H.T)[:, 2:3]
            r["pts"] = r["pts"].astype(np.float32)
    return out, rows


# ---------------------------------------------------------------- 主流程
def build_masks(img, rows, rng):
    h, w = img.shape[:2]
    m_t = np.zeros((h, w), np.uint8)
    m_c = np.zeros((h, w), np.uint8)
    for r in rows:
        if r["cls"] != 0:
            continue
        quad = r["pts"][:4].astype(np.int32)
        cv2.fillPoly(m_t, [quad], 255)
        c, rl = r["pts"][4], r["pts"][5]
        rad = int(np.linalg.norm(r["pts"][6] - rl) / 2)
        if rad > 1:
            cv2.circle(m_c, (int(c[0]), int(c[1])), rad, 255, -1)
    # 边缘收缩/扩张，避免把靶框边界一起改掉
    m_t = cv2.erode(m_t, np.ones((5, 5), np.uint8))
    m_board = cv2.bitwise_and(m_t, cv2.bitwise_not(cv2.dilate(m_c, np.ones((5, 5), np.uint8))))
    m_bg = cv2.bitwise_not(cv2.dilate(cv2.bitwise_or(m_t, m_c), np.ones((9, 9), np.uint8)))
    return m_t, m_board, m_c, m_bg


def aug_one(img, rows, rng):
    h, w = img.shape[:2]
    m_t, m_board, m_c, m_bg = build_masks(img, rows, rng)
    # 1) 背景换色 / 换背景
    if rng.random() < 0.85:
        img = shift_hsv(img, m_bg, rng.uniform(-60, 60), rng.uniform(0.3, 1.8), rng.uniform(0.5, 1.35))
    if rng.random() < 0.45:
        img = synthetic_bg(img, m_bg, rng)
    # 2) 靶框内换色
    if rng.random() < 0.7:
        img = shift_hsv(img, m_board, rng.uniform(-40, 40), rng.uniform(0.2, 1.6), rng.uniform(0.5, 1.4))
    # 3) 中心圆换色（含描边）
    if rng.random() < 0.55:
        for r in rows:
            if r["cls"] != 0:
                continue
            c = r["pts"][4]
            rad = int(np.linalg.norm(r["pts"][6] - r["pts"][5]) / 2)
            if rad < 2:
                continue
            col = tuple(int(v) for v in rng.choice(CIRCLE_COLORS))
            cv2.circle(img, (int(c[0]), int(c[1])), rad, col, -1)
            if rng.random() < 0.4:
                ring = tuple(int(v) for v in rng.choice(CIRCLE_COLORS))
                cv2.circle(img, (int(c[0]), int(c[1])), rad, ring, max(1, rad // 6))
    # 4) 干扰物
    if rng.random() < 0.8:
        stamp_distractors(img, m_t, rows, rng)
    # 5) 曝光
    img = exposure(img, rng)
    # 6) 几何（标签一起变）
    img, rows = geom(img, rows, rng)
    # 7) 模糊/噪声
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.5:
        img = np.clip(img.astype(np.float32) + np.random.normal(0, rng.uniform(1, 6), img.shape),
                      0, 255).astype(np.uint8)
    return img, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "aug_v2"))
    ap.add_argument("--n", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--scenes", nargs="*", default=None)
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
        if args.scenes and scene not in args.scenes:
            continue
        for lab in sorted((ROOT / "data" / "ann2" / scene).glob("*.txt")):
            img_p = sc.image_dir / f"{lab.stem}.jpg"
            if img_p.exists() and lab.stat().st_size > 0:
                pool.append((scene, img_p, lab))
    print(f"素材 {len(pool)} 张")
    rng.shuffle(pool)

    made = 0
    i = 0
    while made < args.n:
        scene, img_p, lab_p = pool[i % len(pool)]
        i += 1
        rows = parse_label(lab_p)
        if not rows:
            continue
        try:
            img = load(img_p)
        except Exception:  # noqa: BLE001
            continue
        rows = [dict(r, pts=r["pts"].copy()) for r in rows]
        h, w = img.shape[:2]
        rows = [dict(r, pts=np.stack([r["pts"][:, 0] * w, r["pts"][:, 1] * h], 1)) for r in rows]
        img2, rows2 = aug_one(img, rows, rng)
        name = f"aug_{made:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img2, [cv2.IMWRITE_JPEG_QUALITY, 88])
        (out / "labels" / f"{name}.txt").write_text(fmt_label(rows2, w, h))
        made += 1
        if made % 200 == 0:
            print(f"  {made}/{args.n}")
    print(f"增强样本 {made} -> {out}")


if __name__ == "__main__":
    main()
