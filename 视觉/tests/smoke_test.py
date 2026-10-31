"""端到端自检：合成数据 → 构建 YOLO 目录 → 训练 3 epoch → 评估。

    python3 tests/smoke_test.py            # 完整跑（约 1-3 分钟）
    python3 tests/smoke_test.py --train 0  # 只校验数据构建
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

import synth  # noqa: E402
from target_pose.datasets import Item, build_yolo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3, help="训练轮数，0 表示跳过训练")
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="pose_smoke_"))
    print(f"[smoke] 临时目录 {tmp}")
    meta = synth.generate(tmp / "images", n=args.n, seed=7)
    synth.write_labels(meta, tmp / "labels")
    print(f"[smoke] 合成 {len(meta)} 张，实例 {sum(len(m['objs']) for m in meta)}")

    items = []
    for m in meta:
        p = Path(m["image"])
        lab = tmp / "labels" / (p.stem + ".txt")
        n = sum(1 for line in lab.read_text().splitlines() if line.strip())
        items.append(Item(sid=f"syn__{p.stem}", scene="syn", image=str(p),
                          width=m["width"], height=m["height"], label=str(lab),
                          source="user", reviewed=True, n_objs=n))
    yaml_path = build_yolo(val_ratio=0.25, seed=0, out=tmp / "yolo", items=items)
    print(f"[smoke] data.yaml:\n{yaml_path.read_text()}")
    train_lines = (tmp / "yolo" / "train.txt").read_text().strip().splitlines()
    assert train_lines, "train.txt 为空"
    first = Path(train_lines[0])
    print(f"[smoke] 示例图片链接 {first} -> {first.resolve()}")

    if args.train > 0:
        from target_pose.evaluate import evaluate, report
        from target_pose.predict import _load_model

        cmd = [sys.executable, str(ROOT / "pose.py"), "train",
               "--data", str(yaml_path), "--model", "yolov8n-pose.pt",
               "--epochs", str(args.train), "--imgsz", "320", "--batch", "6",
               "--name", "smoke", "--patience", "999"]
        print("[smoke] 训练中…")
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print("[smoke] 训练失败")
            return 1
        best = ROOT / "runs" / "smoke" / "weights" / "best.pt"
        print(f"[smoke] best={best.exists()}")

        model = _load_model(str(best))
        val_items = []
        for line in (tmp / "yolo" / "val.txt").read_text().strip().splitlines():
            sid = Path(line).stem
            lab = tmp / "yolo" / "labels" / "all" / f"{sid}.txt"
            src = [i for i in items if i.sid == sid][0]
            val_items.append(Item(sid=sid, scene="syn", image=str(Path(line).resolve()),
                                  width=src.width, height=src.height, label=str(lab),
                                  n_objs=src.n_objs))
        per = evaluate(model, val_items, imgsz=320, conf=0.2, iou_thr=0.5, do_refine=False)
        report(per)

    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
        print("[smoke] 已清理")
    else:
        print(f"[smoke] 保留 {tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
