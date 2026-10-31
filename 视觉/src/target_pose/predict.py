"""用已训练的模型给未标注图片生成预标注（含几何精修与置信度分级）。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import ANN_DIR, META_DIR, N_KPT, SCENES, WEIGHTS_DIR
from .datasets import Item, annotate_path, meta_path, scan
from .labels import Obj, Sample
from .refine import refine_target


def _load_model(weights: str):
    from ultralytics import YOLO

    p = Path(weights)
    if not p.exists():
        for cand in (WEIGHTS_DIR / weights, WEIGHTS_DIR / f"{weights}.pt"):
            if cand.exists():
                p = cand
                break
    if not p.exists():
        raise SystemExit(f"找不到模型 {weights}")
    return YOLO(str(p))


def predict_objs(model, image: np.ndarray, imgsz: int, conf: float, iou: float,
                 max_targets: int = 0) -> List[tuple]:
    """返回 [(Obj, score)]。"""
    res = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
        return []
    kxy = res.keypoints.xy.cpu().numpy()          # (n, K, 2)
    kconf = None
    if getattr(res.keypoints, "conf", None) is not None:
        kconf = res.keypoints.conf.cpu().numpy()  # (n, K)
    bconf = res.boxes.conf.cpu().numpy()          # (n,)
    out = []
    for i in range(kxy.shape[0]):
        if kxy.shape[1] < N_KPT:
            continue
        pts = kxy[i][:N_KPT]
        s = float(bconf[i])
        if kconf is not None:
            s = 0.5 * s + 0.5 * float(np.mean(kconf[i][:N_KPT]))
        out.append((Obj(corners=pts[:4].astype(float), center=pts[4].astype(float)), s))
    if max_targets and len(out) > max_targets:
        out.sort(key=lambda t: -t[1])
        out = out[:max_targets]
    return out


def predict_objs_cls(model, image: np.ndarray, imgsz: int, conf: float, iou: float,
                     max_targets: int = 0) -> List[tuple]:
    """返回 [(Obj, score, cls)]：双类模型里 0=靶框、1=中心圆。

    框那一行只填 4 角点 + 圆心（圆心由后面的圆配对覆盖）；圆那一行填圆心和左右极点。
    """
    res = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
        return []
    kxy = res.keypoints.xy.cpu().numpy()
    bconf = res.boxes.conf.cpu().numpy()
    bcls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else None
    out = []
    for i in range(kxy.shape[0]):
        if kxy.shape[1] < N_KPT:
            continue
        pts = kxy[i][:N_KPT]
        cls = int(bcls[i]) if bcls is not None else 0
        s = float(bconf[i])
        if cls == 1:                      # 中心圆：角点无效，圆点有效
            out.append((Obj(corners=pts[:4].astype(float), center=pts[4].astype(float),
                            circle_l=pts[5].astype(float), circle_r=pts[6].astype(float)), s, cls))
        else:
            out.append((Obj(corners=pts[:4].astype(float), center=pts[4].astype(float)), s, cls))
    if max_targets and len(out) > max_targets:
        out.sort(key=lambda t: -t[1])
        out = out[:max_targets]
    return out


def run(args) -> Dict:
    model = _load_model(args.weights)
    items = scan(args.scene or None)
    todo = []
    for it in items:
        if args.only_unlabeled and it.n_objs > 0:
            continue
        if it.source == "user" and not args.force:
            continue  # 保护人工真值
        if args.source and it.source not in args.source.split(","):
            continue
        todo.append(it)
    if args.limit:
        todo = todo[: args.limit]
    print(f"待预标注 {len(todo)} 张（模型 {args.weights}）")

    stat = {"images": 0, "objs": 0, "refined": 0, "circle_ok": 0, "skipped": 0}
    t0 = time.time()
    for k, it in enumerate(todo, 1):
        img = cv2.imread(it.image)
        if img is None:
            continue
        preds = predict_objs(model, img, args.imgsz, args.conf, args.iou,
                             max_targets=SCENES[it.scene].max_targets)
        if not preds:
            stat["skipped"] += 1
            continue
        sample = Sample(image=Path(it.image), width=it.width, height=it.height, objs=[])
        scores = []
        for obj, sc in preds:
            if args.refine:
                obj2, info = refine_target(img, obj)
                if info.get("quad_ok"):
                    stat["refined"] += 1
                if info.get("circ_ok"):
                    stat["circle_ok"] += 1
                obj = obj2
            obj = obj.sort_corners()
            sample.objs.append(obj)
            scores.append(sc)
        if not sample.objs:
            continue
        lp = annotate_path(it.scene, Path(it.image).stem)
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(sample.to_txt())
        mp = meta_path(it.scene, Path(it.image).stem)
        mp.parent.mkdir(parents=True, exist_ok=True)
        score = float(np.mean(scores))
        mp.write_text(json.dumps({
            "source": "auto",
            "score": round(score, 4),
            "reviewed": False,
            "n_objs": len(sample.objs),
            "weights": str(args.weights),
            "refine": bool(args.refine),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False, indent=1))
        stat["images"] += 1
        stat["objs"] += len(sample.objs)
        if k % 20 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"  [{k}/{len(todo)}] 图 {stat['images']} 实例 {stat['objs']} "
                  f"圆心对齐 {stat['circle_ok']} 用时 {el:.0f}s", flush=True)
    print(f"完成：{stat}")
    return stat


def main(argv=None):
    ap = argparse.ArgumentParser(description="模型预标注")
    ap.add_argument("--weights", default="target_pose.pt")
    ap.add_argument("--scene", nargs="*", default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-refine", dest="refine", action="store_false")
    ap.add_argument("--force", action="store_true", help="连人工确认过的标注也覆盖")
    ap.add_argument("--only-unlabeled", action="store_true")
    ap.add_argument("--source", default="", help="只处理指定来源(auto/seed/none)")
    ap.set_defaults(refine=True)
    args = ap.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
