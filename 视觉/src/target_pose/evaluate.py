"""评估：模型预测 vs 人工标注真值（角点误差 / 靶心误差 / 检出率）。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .config import SCENES


def _scene_name(key: str) -> str:
    return SCENES[key].name if key in SCENES else key
from .datasets import scan
from .labels import Obj, Sample, match_objects
from .predict import _load_model, predict_objs
from .refine import refine_target


def _norm4(objs: List[Obj]) -> List[np.ndarray]:
    return [np.asarray(o.corners, float) for o in objs]


def evaluate(model, items, imgsz: int, conf: float, iou_thr: float, do_refine: bool,
             verbose: bool = False) -> Dict:
    per_scene = defaultdict(lambda: {"img": 0, "gt": 0, "pred": 0, "tp": 0,
                                     "corner_err": [], "bull_err": [], "side": []})
    for it in items:
        gt = Sample.from_txt(Path(it.label), Path(it.image), it.width, it.height)
        if not gt.objs:
            continue
        img = cv2.imread(it.image)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        preds = predict_objs(model, img, imgsz, conf, 0.55)
        objs = []
        for obj, sc in preds:
            if do_refine:
                obj, _ = refine_target(gray, obj)
            objs.append(obj.sort_corners())
        pairs, miss_p, miss_g = match_objects(objs, gt.objs, iou_thr=0.5)
        st = per_scene[it.scene]
        st["img"] += 1
        st["gt"] += len(gt.objs)
        st["pred"] += len(objs)
        st["tp"] += len(pairs)
        for i, j in pairs:
            p, g = objs[i], gt.objs[j]
            ce = float(np.mean(np.linalg.norm(p.points - g.points, axis=1)))
            be = float(np.linalg.norm(p.center - g.center))
            side = float(np.mean([np.linalg.norm(g.corners[(k + 1) % 4] - g.corners[k]) for k in range(4)]))
            st["corner_err"].append(ce)
            st["bull_err"].append(be)
            st["side"].append(side)
            if verbose:
                print(f"   {it.sid} corner={ce:.1f}px bull={be:.1f}px side={side:.0f}")
    return per_scene


def report(per_scene: Dict) -> Dict:
    out = {}
    rows = []
    for key, st in sorted(per_scene.items()):
        tp, gt, pred = st["tp"], st["gt"], st["pred"]
        prec = tp / pred if pred else 0.0
        rec = tp / gt if gt else 0.0
        ce = float(np.median(st["corner_err"])) if st["corner_err"] else float("nan")
        be = float(np.median(st["bull_err"])) if st["bull_err"] else float("nan")
        rely = [e / s for e, s in zip(st["corner_err"], st["side"])]
        bely = [e / s for e, s in zip(st["bull_err"], st["side"])]
        out[key] = {
            "name": _scene_name(key), "images": st["img"], "gt": gt, "pred": pred, "tp": tp,
            "precision": round(prec, 3), "recall": round(rec, 3),
            "corner_err_px": round(ce, 2), "bull_err_px": round(be, 2),
            "corner_err_rel": round(float(np.median(rely)), 4) if rely else None,
            "bull_err_rel": round(float(np.median(bely)), 4) if bely else None,
        }
        rows.append(f"  {_scene_name(key):<16} 图{st['img']:>4} GT{gt:>5} 预测{pred:>5} "
                    f"P={prec:.3f} R={rec:.3f} 角点中位误差={ce:6.1f}px 靶心中位误差={be:6.1f}px "
                    f"(相对 {out[key]['bull_err_rel']})")
    all_ce = [e for st in per_scene.values() for e in st["corner_err"]]
    all_be = [e for st in per_scene.values() for e in st["bull_err"]]
    tot_tp = sum(s["tp"] for s in per_scene.values())
    tot_pred = sum(s["pred"] for s in per_scene.values())
    tot_gt = sum(s["gt"] for s in per_scene.values())
    head = (f"合计：P={tot_tp / max(tot_pred, 1):.3f} R={tot_tp / max(tot_gt, 1):.3f} "
            f"角点中位误差={np.median(all_ce) if all_ce else float('nan'):.1f}px "
            f"靶心中位误差={np.median(all_be) if all_be else float('nan'):.1f}px")
    print("\n".join(rows))
    print(head)
    out["_all"] = {"precision": round(tot_tp / max(tot_pred, 1), 3),
                   "recall": round(tot_tp / max(tot_gt, 1), 3),
                   "corner_err_px": round(float(np.median(all_ce)), 2) if all_ce else None,
                   "bull_err_px": round(float(np.median(all_be)), 2) if all_be else None}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="评估关键点模型")
    ap.add_argument("--weights", default="target_pose.pt")
    ap.add_argument("--scene", nargs="*", default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--source", default="user", help="作为真值的标签来源，空=全部")
    args = ap.parse_args(argv)

    items = [i for i in scan(args.scene or None) if i.n_objs > 0]
    if args.source:
        sel = [i for i in items if i.source in args.source.split(",")]
        if sel:
            items = sel
        else:
            print(f"[warn] 没有来源为 {args.source} 的标注，退化为使用全部已标注样本（含 auto，指标会偏高）")
    if args.limit:
        items = items[: args.limit]
    model = _load_model(args.weights)
    per = evaluate(model, items, args.imgsz, args.conf, 0.5, args.refine, args.verbose)
    rep = report(per)
    if args.json:
        Path(args.json).write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
