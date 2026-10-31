"""v12：v11 基础上加「擦圆样本」（data/frame_only）—— 圆被涂掉时也要能框住靶子，
再用角点对角线交点反推靶心。

新增负样本：
  * data/neg_banner  200 张合成横幅（蓝底白字圆 logo + 贴到真实背景上）
新增正样本：
  * data/banner_pos   80 张在已有靶子图上贴横幅（标签不变），让模型学会
    「横幅和靶子同时存在时也只看靶子」
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from target_pose.config import SCENES, KPT_SHAPE, FLIP_IDX  # noqa: E402

OUT = ROOT / "data" / "yolo_v12"


def main():
    random.seed(43)
    for sub in ("images/all", "labels/all"):
        p = OUT / sub
        p.mkdir(parents=True, exist_ok=True)
        for f in p.iterdir():
            f.unlink()

    pos, neg = [], []

    def add(sid, img, lab, is_pos):
        (pos if is_pos else neg).append((sid, img, lab))

    for scene, sc in SCENES.items():
        for lab in sorted((ROOT / "data" / "ann2" / scene).glob("*.txt")):
            img = sc.image_dir / f"{lab.stem}.jpg"
            if not (img.exists() and lab.stat().st_size > 0):
                continue
            # 只收人工复核过的样本（comp 场景是我预标 + 人工改的，没复核的不要）
            meta = ROOT / "data" / "meta" / scene / f"{lab.stem}.json"
            reviewed = None
            if meta.exists():
                try:
                    reviewed = bool(json.loads(meta.read_text()).get("reviewed", False))
                except Exception:
                    reviewed = None
            if scene == "comp":
                if not reviewed:      # 预标未复核的 8 帧不要
                    continue
            elif reviewed is False:
                continue
            add(f"{scene}__{lab.stem}", img, lab, True)

    for lab in sorted((ROOT / "data" / "banner_pos" / "labels").glob("*.txt")):
        img = ROOT / "data" / "banner_pos" / "images" / (lab.stem + ".jpg")
        if img.exists() and lab.stat().st_size > 0:
            add(lab.stem, img, lab, True)

    # 擦圆样本：靶框还在、中心圆被涂掉 —— 教模型别依赖圆
    for lab in sorted((ROOT / "data" / "frame_only" / "labels").glob("*.txt")):
        img = ROOT / "data" / "frame_only" / "images" / (lab.stem + ".jpg")
        if img.exists() and lab.stat().st_size > 0:
            add(f"fo__{lab.stem}", img, lab, True)

    # 打印靶纸仿真拍图：每 4 张留 1 张做评估集（不进训练）
    for lab in sorted((ROOT / "data" / "printsim" / "labels").glob("*.txt")):
        idx = int(lab.stem.split("_")[-1])
        if idx % 4 == 0:
            continue
        img = ROOT / "data" / "printsim" / "images" / (lab.stem + ".jpg")
        if img.exists() and lab.stat().st_size > 0:
            add(f"psim__{lab.stem}", img, lab, True)

    for lab in sorted((ROOT / "data" / "aug_v2" / "labels").glob("*.txt")):
        img = ROOT / "data" / "aug_v2" / "images" / (lab.stem + ".jpg")
        if img.exists() and lab.stat().st_size > 0:
            add(f"aug__{lab.stem}", img, lab, True)

    for lab in sorted((ROOT / "data" / "ann2" / "neg").glob("*.txt")):
        img = ROOT / "data" / "neg" / "images" / (lab.stem + ".jpg")
        if img.exists():
            add(lab.stem, img, lab, False)

    for lab in sorted((ROOT / "data" / "neg_banner" / "labels").glob("*.txt")):
        img = ROOT / "data" / "neg_banner" / "images" / (lab.stem + ".jpg")
        if img.exists():
            add(lab.stem, img, lab, False)

    for sid, img, lab in pos + neg:
        link = OUT / "images" / "all" / (sid + img.suffix)
        if not link.exists():
            os.symlink(img.resolve(), link)
        shutil.copyfile(lab, OUT / "labels" / "all" / f"{sid}.txt")

    random.shuffle(pos)
    n_val = int(round(len(pos) * 0.12))
    val, train = pos[:n_val], pos[n_val:] + neg
    (OUT / "train.txt").write_text(
        "\n".join(str(OUT / "images" / "all" / (s + i.suffix)) for s, i, _ in train) + "\n")
    (OUT / "val.txt").write_text(
        "\n".join(str(OUT / "images" / "all" / (s + i.suffix)) for s, i, _ in val) + "\n")
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: train.txt\nval: val.txt\n"
        f"names:\n  0: target\n  1: circle\n"
        f"kpt_shape: {KPT_SHAPE}\nflip_idx: {FLIP_IDX}\n")
    print(f"pos {len(pos)} (train {len(train)-len(neg)} / val {len(val)}) neg {len(neg)} "
          f"-> {OUT/'data.yaml'}")


if __name__ == "__main__":
    main()
