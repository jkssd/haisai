"""数据集组织：场景扫描、清单、划分、构建 Ultralytics 可用的 YOLO-pose 目录。"""
from __future__ import annotations

import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from .config import (
    ANN_DIR,
    CLASS_NAMES,
    DATA,
    FLIP_IDX,
    KPT_NAMES,
    KPT_SHAPE,
    META_DIR,
    SCENES,
    YOLO_DIR,
    ensure_dirs,
)


@dataclass
class Item:
    """一条样本（一张图）。"""

    sid: str                  # 全局唯一 id: "<scene>__<stem>"
    scene: str
    image: str                # 绝对路径
    width: int
    height: int
    label: str                # 标注 txt 的绝对路径
    source: str = "none"      # none / seed / auto / user
    score: float = 0.0        # 预标注置信度
    reviewed: bool = False
    n_objs: int = 0
    notes: str = ""


def _image_size(p: Path) -> Tuple[int, int]:
    with Image.open(p) as im:
        return im.size


def annotate_path(scene: str, stem: str) -> Path:
    return ANN_DIR / scene / f"{stem}.txt"


def meta_path(scene: str, stem: str) -> Path:
    return META_DIR / scene / f"{stem}.json"


# --------------------------------------------------------------------------- #
def voc_hints(scene_key: str) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """读取已有 Pascal VOC 框标注（只用来给标注工具做初始提示，不作为真值）。"""
    sc = SCENES[scene_key]
    if not sc.voc_dir or not sc.voc_dir.exists():
        return {}
    out: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for xml in sorted(sc.voc_dir.glob("*.xml")):
        try:
            root = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        boxes = []
        for obj in root.findall("object"):
            b = obj.find("bndbox")
            if b is None:
                continue
            try:
                boxes.append(tuple(float(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")))
            except (AttributeError, TypeError):
                continue
        if boxes:
            out[xml.stem] = boxes
    return out


def scan(scene_keys: Optional[Iterable[str]] = None) -> List[Item]:
    """扫描所有场景，生成清单（含已有标注状态）。"""
    ensure_dirs()
    keys = list(scene_keys) if scene_keys else list(SCENES)
    items: List[Item] = []
    for key in keys:
        sc = SCENES[key]
        for img in sc.images():
            w, h = _image_size(img)
            sid = f"{key}__{img.stem}"
            lp = annotate_path(key, img.stem)
            mp = meta_path(key, img.stem)
            meta = json.loads(mp.read_text()) if mp.exists() else {}
            n_objs = 0
            if lp.exists():
                n_objs = sum(1 for line in lp.read_text().splitlines() if line.strip())
            items.append(Item(
                sid=sid, scene=key, image=str(img), width=w, height=h, label=str(lp),
                source=meta.get("source", "user" if lp.exists() else "none"),
                score=float(meta.get("score", 0.0)),
                reviewed=bool(meta.get("reviewed", False)),
                n_objs=n_objs,
                notes=meta.get("notes", ""),
            ))
    return items


def save_manifest(items: List[Item], path: Optional[Path] = None) -> Path:
    path = path or (DATA / "manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(i) for i in items], ensure_ascii=False, indent=1))
    return path


def load_manifest(path: Optional[Path] = None) -> List[Item]:
    path = path or (DATA / "manifest.json")
    raw = json.loads(path.read_text())
    return [Item(**r) for r in raw]


# --------------------------------------------------------------------------- #
def split_items(
    items: List[Item],
    val_ratio: float = 0.15,
    seed: int = 0,
    per_scene: bool = True,
) -> Tuple[List[Item], List[Item]]:
    """按场景分层划分 train/val（每个场景都进入验证集，保证评估有代表性）。"""
    rng = random.Random(seed)
    train: List[Item] = []
    val: List[Item] = []
    if not per_scene:
        pool = [i for i in items if i.n_objs > 0]
        rng.shuffle(pool)
        n_val = max(1, int(round(len(pool) * val_ratio)))
        return pool[n_val:], pool[:n_val]
    # 负样本（n_objs==0）只进训练集，不进验证集（验证集必须是纯正样本，指标才有意义）
    for key in sorted({i.scene for i in items}):
        pool = sorted([i for i in items if i.scene == key and i.n_objs > 0], key=lambda x: x.sid)
        train += [i for i in items if i.scene == key and i.n_objs == 0]
        rng.shuffle(pool)
        if len(pool) <= 1:
            train += pool
            continue
        n_val = max(1, min(int(round(len(pool) * val_ratio)), len(pool) - 1))
        val += pool[:n_val]
        train += pool[n_val:]
    return train, val


def build_yolo(
    val_ratio: float = 0.15,
    seed: int = 0,
    out: Optional[Path] = None,
    only_reviewed: bool = False,
    min_objs: int = 1,
    items: Optional[List[Item]] = None,
) -> Path:
    """构建 Ultralytics YOLO-pose 目录结构（图片用符号链接，避免复制 2.5G）。"""
    ensure_dirs()
    out = out or YOLO_DIR
    for sub in ("images/all", "labels/all"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    # 清理旧的链接，避免残留
    for sub in ("images/all", "labels/all"):
        for f in (out / sub).iterdir():
            if f.is_symlink() or f.suffix in (".jpg", ".txt"):
                f.unlink()

    if items is None:
        items = scan()
    items = [i for i in items if i.n_objs >= min_objs]
    if only_reviewed:
        items = [i for i in items if i.reviewed]
    if not items:
        raise SystemExit("没有可用标注，先跑标注/预标注。")

    for it in items:
        stem = it.sid
        img_link = out / "images" / "all" / f"{stem}{Path(it.image).suffix}"
        lab_link = out / "labels" / "all" / f"{stem}.txt"
        if not img_link.exists():
            os.symlink(Path(it.image).resolve(), img_link)
        shutil.copyfile(it.label, lab_link)

    train, val = split_items(items, val_ratio=val_ratio, seed=seed)
    if not train:
        print("[build] 警告：训练集为空（样本太少或 val_ratio 过大）")
    (out / "train.txt").write_text("\n".join(str(out / "images" / "all" / f"{i.sid}{Path(i.image).suffix}") for i in train) + "\n")
    (out / "val.txt").write_text("\n".join(str(out / "images" / "all" / f"{i.sid}{Path(i.image).suffix}") for i in val) + "\n")

    yaml = out / "data.yaml"
    yaml.write_text(
        f"path: {out}\n"
        f"train: train.txt\n"
        f"val: val.txt\n"
        f"names:\n" + "".join(f"  {k}: {v}\n" for k, v in enumerate(CLASS_NAMES)) +
        f"kpt_shape: {KPT_SHAPE}\n"
        f"flip_idx: {FLIP_IDX}\n"
    )
    (out / "kpt_names.txt").write_text("\n".join(KPT_NAMES) + "\n")
    print(f"[build] train={len(train)} val={len(val)} -> {yaml}")
    return yaml


def stats(items: List[Item]) -> str:
    by_scene: Dict[str, List[Item]] = {}
    for it in items:
        by_scene.setdefault(it.scene, []).append(it)
    lines = []
    tot = tot_lab = tot_rev = 0
    for key, arr in sorted(by_scene.items()):
        lab = sum(1 for i in arr if i.n_objs > 0)
        rev = sum(1 for i in arr if i.reviewed)
        objs = sum(i.n_objs for i in arr)
        lines.append(f"  {SCENES[key].name:<16} 图片 {len(arr):>4}  已标 {lab:>4}  已复核 {rev:>4}  实例 {objs:>5}")
        tot += len(arr); tot_lab += lab; tot_rev += rev
    lines.append(f"  {'合计':<16} 图片 {tot:>4}  已标 {tot_lab:>4}  已复核 {tot_rev:>4}")
    return "\n".join(lines)
