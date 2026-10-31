"""在训练用的留出集（data/yolo/val.txt）上评估，给出诚实的指标。

用法：
    python3 tests/eval_holdout.py --weights runs/v4/weights/best.pt --refine
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from target_pose.datasets import Item  # noqa: E402
from target_pose.evaluate import evaluate, report  # noqa: E402
from target_pose.predict import _load_model  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/target_pose.pt")
    ap.add_argument("--yolo-dir", default=str(ROOT / "data" / "yolo"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    yolo = Path(args.yolo_dir)
    lines = [ln for ln in (yolo / "val.txt").read_text().splitlines() if ln.strip()]
    items = []
    for ln in lines:
        img = Path(ln)
        sid = img.stem
        lab = yolo / "labels" / "all" / f"{sid}.txt"
        if not lab.exists():
            continue
        with Image.open(img) as im:
            w, h = im.size
        n = sum(1 for x in lab.read_text().splitlines() if x.strip())
        items.append(Item(sid=sid, scene=sid.split("__")[0], image=str(img),
                          width=w, height=h, label=str(lab), n_objs=n, source="user",
                          reviewed=True))
    print(f"留出集 {len(items)} 张 / {sum(i.n_objs for i in items)} 个靶子")
    model = _load_model(args.weights)
    per = evaluate(model, items, args.imgsz, args.conf, 0.5, args.refine)
    rep = report(per)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
