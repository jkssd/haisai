"""生成可打印的靶子图案（A3 / 300dpi），背景色与靶心颜色各不相同。

约定和现场靶一致：**矩形靶框 + 正中心一个圆**，圆半径与靶框半宽的比例
（r/(w/2)）从 0.30 到 0.46 分布，正好覆盖大/中/小三档之间的情况。

每张图除靶子本身外：
  * 靶框外 3mm 一圈对比描边（打印裁切后边框清晰，检测框才有边）
  * A3 四角裁切标记 + 底部说明文字（都在靶框之外，不参与检测）
输出 PNG(300dpi) 与 PDF，可直接送打印店。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "print_targets"
DPI = 300
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

MM = DPI / 25.4
A3 = (int(297 * MM), int(420 * MM))            # 3508 x 4961

# (背景色, 圆色, 描边色, 比例, 名字)
DESIGNS = [
    ((18, 22, 34), (255, 255, 255), (255, 255, 255), 0.30, "哑光黑底 + 白圆"),
    ((26, 46, 128), (255, 255, 255), (250, 250, 250), 0.34, "深蓝底 + 白圆"),
    ((22, 70, 40), (255, 205, 30), (245, 245, 245), 0.38, "深绿底 + 黄圆"),
    ((245, 244, 240), (16, 16, 16), (16, 16, 16), 0.42, "白纸底 + 黑圆"),
    ((120, 24, 30), (245, 245, 245), (250, 250, 250), 0.46, "暗红底 + 白圆"),
]


def font(size_px, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, int(size_px))


def make_one(idx: int, bg, fg, line, ratio, name):
    W, H = A3
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)

    mx, my = int(14 * MM), int(14 * MM)            # 左右上留白
    cap_h = int(46 * MM)                          # 底部说明区
    px0, py0 = mx, my
    px1, py1 = W - mx, H - my - cap_h
    pw, ph = px1 - px0, py1 - py0

    # 靶面
    d.rectangle([px0, py0, px1, py1], fill=bg)
    # 外描边（3mm）
    lw = max(2, int(1.6 * MM))
    d.rectangle([px0, py0, px1, py1], outline=line, width=lw)
    # 内描边（细，帮助压边）
    inset = int(6 * MM)
    d.rectangle([px0 + inset, py0 + inset, px1 - inset, py1 - inset],
                outline=tuple(int(c * 0.55 + 128 * 0.45) for c in line), width=max(1, int(0.6 * MM)))

    # 中心圆
    r = ratio * pw / 2.0
    cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    # 圆外用一圈细线分界（避免与板面同色时糊在一起）
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=line, width=max(1, int(0.5 * MM)))

    # 四角裁切标记（在靶面之外的白边上）
    cm = int(6 * MM)
    for (ax, ay, sx, sy) in [(px0, py0, -1, -1), (px1, py0, 1, -1),
                             (px0, py1, -1, 1), (px1, py1, 1, 1)]:
        d.line([ax + sx * int(2 * MM), ay, ax + sx * cm, ay], fill=(0, 0, 0), width=2)
        d.line([ax, ay + sy * int(2 * MM), ax, ay + sy * cm], fill=(0, 0, 0), width=2)

    # 底部说明（不进靶框）
    ty = py1 + int(10 * MM)
    d.text((mx, ty), f"靶子 #{idx}   {name}", font=font(13 * MM, True), fill=(20, 20, 20))
    r_mm = r / MM
    d.text((mx, ty + int(17 * MM)),
           f"靶框 {pw / MM:.0f} x {ph / MM:.0f} mm   圆 Ø{2 * r_mm:.0f} mm   "
           f"r/(w/2) = {ratio:.2f}",
           font=font(8.5 * MM), fill=(40, 40, 40))
    d.text((mx, ty + int(28 * MM)),
           "打印：A3 / 300dpi / 哑光纸，勿覆膜（反光会拉低检出）；板面要贴平",
           font=font(7.5 * MM), fill=(80, 80, 80))
    d.text((mx, ty + int(37 * MM)),
           "CMVC target set · target_pose dataset",
           font=font(6.5 * MM), fill=(120, 120, 120))
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (bg, fg, line, ratio, name) in enumerate(DESIGNS, 1):
        im = make_one(i, bg, fg, line, ratio, name)
        p = OUT / f"target_print_{i:02d}.png"
        im.save(p, dpi=(DPI, DPI))
        im.save(OUT / f"target_print_{i:02d}.pdf", resolution=DPI)
        paths.append(p)
        print(p, im.size, f"ratio={ratio}")
    # 预览小图
    th = []
    for p in paths:
        im = Image.open(p)
        im.thumbnail((360, 508))
        th.append(im)
    sheet = Image.new("RGB", (360 * len(th), 508), (255, 255, 255))
    for i, im in enumerate(th):
        sheet.paste(im, (i * 360, 0))
    sheet.save(OUT / "preview_sheet.jpg", quality=90)
    print("preview ->", OUT / "preview_sheet.jpg")


if __name__ == "__main__":
    main()
