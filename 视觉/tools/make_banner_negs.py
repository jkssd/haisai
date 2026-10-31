"""合成「蓝色横幅 + 白字 + 圆形 logo」负样本（硬负样本，专治横幅误检）。

背景：现场画面顶部有一条蓝色横幅（一排圆形/圆角矩形 logo + 白色中文大标题），
现有模型会把 logo 的圆环认成靶子中心圆（t2 图上 圆 0.68@(408,51) / 0.47@(486,53)）。
比赛现场横幅一定在，所以用合成数据把这个模式教成负样本，比删图/调阈值可靠。

三种产物：
  1. synth  100 张纯合成横幅（蓝底 + logo + 白字），相互独立，直接当负样本
  2. comp   100 张合成横幅贴到**真实背景裁片**上（有场景上下文），也是负样本
  3. overlay 80 张把横幅贴到**已有正样本**的空白处（标签不变），保持"横幅与靶子共存"

约束：绝不读取 out/t1.png、out/t2.png、out/new_scene.png 这三张固定测试图。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from target_pose.config import SCENES  # noqa: E402

FORBIDDEN = {"t1.png", "t2.png", "new_scene.png", "test2.jpg"}

FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]
_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    size = max(8, int(size))
    path = FONTS[0] if bold else FONTS[1]
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


BLUES = [(30, 66, 168), (26, 58, 152), (44, 86, 190), (20, 48, 136),
         (36, 74, 182), (52, 96, 205), (24, 62, 160), (18, 40, 120)]
ACCENTS = [(255, 255, 255), (208, 232, 255), (120, 200, 255), (240, 150, 60),
           (60, 170, 110), (220, 90, 80), (250, 220, 120)]
TEXTS = [
    "第十五届全国海洋航行器设计与制作大赛",
    "全国海洋航行器设计与制作大赛",
    "中国科学技术协会",
    "科普中国",
    "中国造船工程学会",
    "海洋航行器设计与制作",
    "智能航行 挑战未来",
    "第十五届 全国总决赛",
    "CHINA SCIENCE COMMUNICATION",
    "NATIONAL COMPETITION 2026",
    "海洋强国 青年担当",
    "2026 全国青少年科技竞赛",
]
TITLE_CHARS = ("海洋航行器设计制作大赛全国第十五届科学与技术"
               "船舶工程智能控制创新未来挑战杯青少年")


# ------------------------------------------------------------------ 基础绘制
def gradient_blue(w: int, h: int, rng: random.Random) -> Image.Image:
    c1 = np.array(rng.choice(BLUES), np.float32)
    c2 = np.clip(c1 + np.array([rng.randint(-18, 26), rng.randint(-14, 22),
                                rng.randint(-16, 24)], np.float32), 8, 245)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    ang = rng.random()
    t = np.clip(ang * yy + (1 - ang) * xx, 0, 1)
    arr = c1[None, None, :] * (1 - t) + c2[None, None, :] * t
    arr = arr + np.random.normal(0, rng.uniform(1.5, 5.0), arr.shape)
    if rng.random() < 0.5:                     # 轻微的拍摄光斑
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.5, 1.2) * max(w, h)
        yy2, xx2 = np.mgrid[0:h, 0:w].astype(np.float32)
        g = np.exp(-(((xx2 - cx) ** 2 + (yy2 - cy) ** 2) / (2 * r * r)))
        arr = arr + g[:, :, None] * rng.uniform(8, 22)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def draw_logo(d: ImageDraw.ImageDraw, rng: random.Random, cx: float, cy: float, R: float) -> None:
    """一圈 logo 里的一种：尽量像赛事会徽（圆环/齿轮/盾/星/圆角矩形）。"""
    kind = rng.choice(["ring_ticks", "ring_dots", "gear", "star", "shield",
                       "ring_text", "rounded_rect", "ellipse", "two_rings", "arc"])
    col = rng.choice(ACCENTS)
    acc = rng.choice(ACCENTS)
    lw = max(1, int(R * rng.uniform(0.07, 0.13)))
    if kind == "ring_ticks":
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=col, width=lw)
        n = rng.choice([12, 16, 20])
        for i in range(n):
            a = 2 * np.pi * i / n
            r1 = R * 0.72
            r2 = R * 0.95
            d.line([cx + r1 * np.cos(a), cy + r1 * np.sin(a),
                    cx + r2 * np.cos(a), cy + r2 * np.sin(a)], fill=col, width=max(1, lw // 2))
    elif kind == "ring_dots":
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=col, width=lw)
        n = rng.choice([6, 8, 10])
        rr = max(1.5, R * 0.09)
        for i in range(n):
            a = 2 * np.pi * i / n + rng.uniform(0, 0.3)
            px, py = cx + R * 0.55 * np.cos(a), cy + R * 0.55 * np.sin(a)
            d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=acc)
    elif kind == "gear":
        n = rng.choice([12, 16, 18])
        pts = []
        for i in range(n * 2):
            a = 2 * np.pi * i / (n * 2)
            r = R if i % 2 == 0 else R * 0.78
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        d.polygon(pts, outline=col, width=lw)
        d.ellipse([cx - R * 0.45, cy - R * 0.45, cx + R * 0.45, cy + R * 0.45],
                  outline=acc, width=max(1, lw // 2))
    elif kind == "star":
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=col, width=lw)
        pts = []
        for i in range(10):
            a = -np.pi / 2 + np.pi * i / 5
            r = R * 0.62 if i % 2 == 0 else R * 0.26
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        d.polygon(pts, fill=acc)
    elif kind == "shield":
        w0, h0 = R * 0.78, R
        pts = [(cx, cy - h0), (cx + w0, cy - h0 * 0.55), (cx + w0 * 0.85, cy + h0 * 0.35),
               (cx, cy + h0), (cx - w0 * 0.85, cy + h0 * 0.35), (cx - w0, cy - h0 * 0.55)]
        d.polygon(pts, outline=col, width=lw)
        d.ellipse([cx - R * 0.35, cy - R * 0.3, cx + R * 0.35, cy + R * 0.4],
                  outline=acc, width=max(1, lw // 2))
    elif kind == "ring_text":
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=col, width=lw)
        # 圈里塞几笔墨迹，模拟会徽里的图形，但不要做成十字准星
        for _ in range(rng.randint(2, 4)):
            x0 = cx + rng.uniform(-0.5, 0.2) * R
            y0 = cy + rng.uniform(-0.5, 0.4) * R
            d.line([x0, y0, x0 + rng.uniform(0.15, 0.6) * R, y0 + rng.uniform(-0.3, 0.3) * R],
                   fill=acc, width=max(1, int(R * 0.09)))
    elif kind == "rounded_rect":
        w0, h0 = R * rng.uniform(0.7, 0.9), R * rng.uniform(0.9, 1.1)
        d.rounded_rectangle([cx - w0, cy - h0, cx + w0, cy + h0], radius=R * 0.16,
                            outline=col, width=lw)
        g = "王博\n道雅" if rng.random() < 0.5 else "CM\nVC"
        f = font(max(6, R * 0.42))
        d.multiline_text((cx, cy), g, font=f, fill=col, anchor="mm", spacing=max(1, R * 0.1))
    elif kind == "ellipse":
        d.ellipse([cx - R, cy - R * 0.72, cx + R, cy + R * 0.72], outline=col, width=lw)
        d.ellipse([cx - R * 0.6, cy - R * 0.42, cx + R * 0.6, cy + R * 0.42],
                  outline=acc, width=max(1, lw // 2))
    elif kind == "two_rings":
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=col, width=lw)
        d.ellipse([cx - R * 0.72, cy - R * 0.72, cx + R * 0.72, cy + R * 0.72],
                  outline=col, width=max(1, lw // 2))
        d.ellipse([cx - R * 0.45, cy - R * 0.45, cx + R * 0.45, cy + R * 0.45], fill=acc)
    else:  # arc
        d.arc([cx - R, cy - R, cx + R, cy + R], start=rng.uniform(0, 90),
              end=rng.uniform(200, 330), fill=col, width=lw)
        d.arc([cx - R * 0.6, cy - R * 0.6, cx + R * 0.6, cy + R * 0.6], start=rng.uniform(0, 120),
              end=rng.uniform(220, 340), fill=acc, width=max(1, lw // 2))


def draw_text_block(d: ImageDraw.ImageDraw, rng: random.Random, w: int, h: int) -> None:
    """白色中文大标题（横幅下面那行字）。"""
    txt = rng.choice(TEXTS)
    if rng.random() < 0.25:                 # 随机拼字，避免只认固定文案
        n = rng.randint(6, 14)
        txt = "".join(rng.choice(TITLE_CHARS) for _ in range(n))
    size = int(h * rng.uniform(0.24, 0.40))
    f = font(size)
    tw = d.textlength(txt, font=f)
    if tw > w * 0.94:                        # 太长就缩到能放下
        size = max(6, int(size * w * 0.94 / tw))
        f = font(size)
    y = h * rng.uniform(0.60, 0.80)
    d.text((w * 0.5, y), txt, font=f, fill=(255, 255, 255), anchor="mm",
           stroke_width=max(1, size // 24), stroke_fill=(25, 45, 120))
    if rng.random() < 0.45:                  # 副标题 / 英文小字
        sub = rng.choice(["2026 年 8 月 · 中国", "MARINE VEHICLE DESIGN CONTEST",
                          "全国总决赛 · 智能航行组", "www.cmvc-contest.cn"])
        fs = font(max(6, int(size * 0.42)), bold=False)
        d.text((w * 0.5, y + size * 0.85), sub, font=fs, fill=(226, 236, 255), anchor="mm")


def logo_row(d: ImageDraw.ImageDraw, rng: random.Random, w: int, h: int) -> None:
    n = rng.randint(2, 5)
    R = h * rng.uniform(0.10, 0.20)
    y = h * rng.uniform(0.16, 0.30)
    xs = np.linspace(w * 0.10, w * 0.90, n) + np.array([rng.uniform(-0.03, 0.03) * w for _ in range(n)])
    for cx in xs:
        draw_logo(d, rng, float(cx), float(y), R * rng.uniform(0.85, 1.15))


def make_banner(rng: random.Random) -> Image.Image:
    W = rng.randint(520, 1400)
    H = rng.randint(90, 420)
    im = gradient_blue(W, H, rng)
    d = ImageDraw.Draw(im)
    if rng.random() < 0.85:
        logo_row(d, rng, W, H)
    if rng.random() < 0.9:
        draw_text_block(d, rng, W, H)
    if rng.random() < 0.2:                      # 顶部/底部装饰线
        y = rng.choice([2, H - 3])
        d.line([0, y, W, y], fill=rng.choice(ACCENTS), width=max(1, H // 120))
    return im


def warp(im: Image.Image, rng: random.Random, jitter: float = 0.02) -> Image.Image:
    a = np.array(im)
    h, w = a.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.array([[rng.uniform(-jitter, jitter) * w, rng.uniform(-jitter, jitter) * h]
                          for _ in range(4)], np.float32)
    for _ in range(rng.randint(0, 2)):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-2.5, 2.5), 1.0)
        out = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        a = out
    out = cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst), (w, h),
                              borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(out)


def post(im: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.7:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.8)))
    f = np.array(im).astype(np.float32)
    f = f * rng.uniform(0.82, 1.18) + rng.uniform(-12, 12)
    if rng.random() < 0.3:                       # 色彩偏冷/偏暖
        f[:, :, 2] *= rng.uniform(0.92, 1.10)
        f[:, :, 2] += rng.uniform(-8, 8)
    if rng.random() < 0.35:                       # 显示屏翻拍：缩小再放大
        h, w = f.shape[:2]
        s = rng.uniform(0.45, 0.85)
        small = cv2.resize(f, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        f = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    f = f + np.random.normal(0, rng.uniform(0, 4), f.shape)
    return Image.fromarray(np.clip(f, 0, 255).astype(np.uint8))


def fit_square(im: Image.Image, size: int, rng: random.Random, mode: str) -> Image.Image:
    """把（可能很宽的）横幅放进 size×size 画布。"""
    w, h = im.size
    ar = w / h
    if mode == "crop":                        # 直接裁一个正方形窗口
        target_h = rng.uniform(0.55, 1.6) * size
        s = target_h / h
        nw, nh = int(w * s), int(h * s)
        im = im.resize((max(size, nw), max(size, nh)), Image.BICUBIC)
        nw, nh = im.size
        x0 = rng.randint(0, max(0, nw - size))
        y0 = rng.randint(0, max(0, nh - size))
        return im.crop((x0, y0, x0 + size, y0 + size))
    # strip：横幅带贴在纯色/渐变背景上
    bgcol = rng.choice([(238, 240, 244), (210, 214, 220), (18, 20, 26), (60, 64, 72)])
    canvas = Image.fromarray(np.clip(np.random.normal(bgcol, 6, (size, size, 3)), 0, 255).astype(np.uint8))
    if ar < 0.5:                              # 竖条？旋转 90°
        im = im.rotate(90, expand=True)
    bh = int(size * rng.uniform(0.18, 0.45))
    bw = int(bh * im.size[0] / im.size[1])
    if bw > size:
        bw = int(size * rng.uniform(0.75, 1.0))
        bh = int(bw * im.size[1] / im.size[0])
    im2 = im.resize((bw, bh), Image.BICUBIC)
    y = rng.randint(-bh // 4, size - bh // 2)
    x = rng.randint(-bw // 3, size - bw // 2)
    canvas.paste(im2, (x, y))
    return canvas


def load_bg_pool() -> list[Image.Image]:
    pool = []
    for p in sorted((ROOT / "data" / "neg" / "images").glob("*.jpg")):
        if p.name in FORBIDDEN:
            continue
        im = Image.open(p).convert("RGB")
        pool.append(im)
    return pool


# ------------------------------------------------------------------ 三种产物
def synth_one(rng: random.Random, size: int) -> Image.Image:
    b = make_banner(rng)
    if rng.random() < 0.75:
        b = fit_square(b, size, rng, "crop")
        b = Image.fromarray(np.array(b))     # copy
        # 裁完再补一点背景，避免永远满屏蓝
        if rng.random() < 0.3:
            canvas = Image.new("RGB", (size, size),
                               tuple(int(v) for v in np.random.normal(200, 12, 3)))
            canvas.paste(b, (rng.randint(-40, 40), rng.randint(-40, 40)))
            b = canvas
        return b
    return fit_square(b, size, rng, "strip")


def comp_one(rng: random.Random, bgs: list[Image.Image], size: int) -> Image.Image:
    bg = rng.choice(bgs)
    w, h = bg.size
    if w < size or h < size:
        bg = bg.resize((max(size, w), max(size, h)))
        w, h = bg.size
    x0 = rng.randint(0, w - size)
    y0 = rng.randint(0, h - size)
    canvas = bg.crop((x0, y0, x0 + size, y0 + size)).copy()
    for _ in range(rng.choice([1, 1, 2])):
        b = post(warp(make_banner(rng), rng), rng)
        bh = int(size * rng.uniform(0.12, 0.40))
        bw = int(bh * b.size[0] / b.size[1])
        if bw < size * 0.4:
            bw = int(size * rng.uniform(0.5, 1.0))
            bh = max(10, int(bw * b.size[1] / b.size[0]))
        b = b.resize((min(bw, int(size * 1.6)), bh), Image.BICUBIC)
        px = rng.randint(-b.size[0] // 4, size - b.size[0] // 2)
        py = rng.randint(-b.size[1] // 4, size - b.size[1] // 2)
        canvas.paste(b, (px, py))
    return canvas


def overlay_positive(rng: random.Random, img_p: Path, lab_p: Path) -> Image.Image:
    im = Image.open(img_p).convert("RGB")
    W, H = im.size
    if max(W, H) > 1280:                       # 控制体积，训练时还会缩放
        s = 1280.0 / max(W, H)
        im = im.resize((int(W * s), int(H * s)), Image.BICUBIC)
        W, H = im.size
    boxes = read_boxes(lab_p, W, H)
    bw = int(W * rng.uniform(0.45, 0.95))
    b = make_banner(rng)
    b = b.resize((bw, max(12, int(bw * b.size[1] / b.size[0]))), Image.BICUBIC)
    b = post(b, rng)
    bw, bh = b.size
    canvas = im.copy()
    placed = False
    for _ in range(20):
        px = rng.randint(-bw // 6, max(0, W - bw + bw // 6))
        py = rng.randint(-bh // 6, max(0, H - bh + bh // 6))
        if not any(px < x2 + 6 and px + bw > x1 - 6 and py < y2 + 6 and py + bh > y1 - 6
                   for x1, y1, x2, y2 in boxes):
            canvas.paste(b, (px, py))
            placed = True
            break
    if not placed:
        return None
    return canvas


def read_boxes(lab_p: Path, W: int, H: int) -> list[tuple[float, float, float, float]]:
    out = []
    for line in lab_p.read_text().splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        cx, cy, bw, bh = map(float, v[1:5])
        out.append(((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--n-synth", type=int, default=100)
    ap.add_argument("--n-comp", type=int, default=100)
    ap.add_argument("--n-overlay", type=int, default=80)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    neg_img = out / "neg_banner" / "images"
    neg_lab = out / "neg_banner" / "labels"
    pos_img = out / "banner_pos" / "images"
    pos_lab = out / "banner_pos" / "labels"
    for p in (neg_img, neg_lab, pos_img, pos_lab):
        p.mkdir(parents=True, exist_ok=True)
        for f in p.iterdir():
            f.unlink()

    n0 = 0
    for i in range(args.n_synth):
        im = post(warp(synth_one(rng, args.size), rng, 0.01), rng)
        name = f"synbanner_{i:04d}"
        im.save(neg_img / f"{name}.jpg", quality=88)
        (neg_lab / f"{name}.txt").write_text("")
        n0 += 1
    print(f"纯合成横幅负样本 {n0}")

    bgs = load_bg_pool()
    n1 = 0
    for i in range(args.n_comp):
        im = post(comp_one(rng, bgs, args.size), rng)
        name = f"compbanner_{i:04d}"
        im.save(neg_img / f"{name}.jpg", quality=88)
        (neg_lab / f"{name}.txt").write_text("")
        n1 += 1
    print(f"背景合成负样本 {n1}")

    # 已有正样本 + 横幅（标签不变）
    src = []
    for scene, sc in SCENES.items():
        for lab in sorted((ROOT / "data" / "ann2" / scene).glob("*.txt")):
            img = sc.image_dir / f"{lab.stem}.jpg"
            if img.exists() and lab.stat().st_size > 0 and img.name not in FORBIDDEN:
                src.append((scene, img, lab))
    rng.shuffle(src)
    n2 = 0
    for scene, img, lab in src:
        if n2 >= args.n_overlay:
            break
        im = overlay_positive(rng, img, lab)
        if im is None:
            continue
        name = f"{scene}__{img.stem}__bn"
        im.save(pos_img / f"{name}.jpg", quality=92)
        (pos_lab / f"{name}.txt").write_text(lab.read_text())
        n2 += 1
    print(f"带横幅的正样本 {n2}")
    print(f"输出 -> {out/'neg_banner'} , {out/'banner_pos'}")


if __name__ == "__main__":
    main()
