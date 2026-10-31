"""训练 YOLOv8-pose（自定义 5 关键点）。

第一次训练建议从 COCO-pose 预训练权重迁移；当 kpt_shape 与预训练权重不一致时，
自动退化为「从 yaml 构建 + 加载可匹配的 backbone 权重」。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .config import KPT_SHAPE, RUNS_DIR, WEIGHTS_DIR
from .datasets import build_yolo


def _make_model(base: str, kpt_shape):
    from ultralytics import YOLO

    # 允许把预训练权重放在 weights/ 下，避免每次联网下载
    cand = WEIGHTS_DIR / Path(base).name
    if not Path(base).exists() and cand.exists():
        base = str(cand)

    try:
        m = YOLO(base)
        m.model.yaml["kpt_shape"] = list(kpt_shape)
        return m, f"{base} (kpt_shape 覆盖为 {kpt_shape})"
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 加载 {base} 失败({e})，改用 yolov8n-pose.yaml 从零构建")
        cfg = "yolov8n-pose.yaml" if "n" in base else "yolov8s-pose.yaml"
        m = YOLO(cfg)
        m.model.yaml["kpt_shape"] = list(kpt_shape)
        return m, f"{cfg} (scratch, kpt_shape={kpt_shape})"


def main(argv=None):
    ap = argparse.ArgumentParser(description="训练靶子关键点模型")
    ap.add_argument("--data", default=None, help="data.yaml；缺省则自动构建")
    ap.add_argument("--model", default="yolov8s-pose.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--name", default="pose_v1")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-reviewed", action="store_true")
    args = ap.parse_args(argv)

    # ---- 断点续训 ----
    if args.resume:
        from ultralytics import YOLO
        ck = Path(args.resume)
        if not ck.exists():
            raise SystemExit(f"找不到 checkpoint: {ck}")
        print(f"[train] 从 {ck} 续训（沿用 checkpoint 里的超参）")
        m = YOLO(str(ck))
        m.train(resume=True)
        best = Path(m.trainer.save_dir) / "weights" / "best.pt"
        if best.exists():
            dst = WEIGHTS_DIR / "target_pose.pt"
            shutil.copyfile(best, dst)
            print(f"[train] best -> {dst}")
        return

    data = Path(args.data) if args.data else build_yolo(val_ratio=args.val_ratio, seed=args.seed,
                                                       only_reviewed=args.only_reviewed)
    model, desc = _make_model(args.model, KPT_SHAPE)
    print(f"[train] {desc}\n[train] data={data}")

    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        patience=args.patience,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.0,   # 前作结论：随机擦除会盖住角点/靶心，标签却还在原处
        degrees=8.0,
        translate=0.08,
        scale=0.45,
        shear=2.0,
        perspective=0.0008,
        mosaic=0.6,
        close_mosaic=15,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        pose=12.0,
        kobj=1.0,
        cache=False,
        amp=False,   # 离线环境：跳过 AMP 自检（会尝试联网下载 yolo26n.pt）
        plots=True,
        verbose=True,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if best.exists():
        dst = WEIGHTS_DIR / "target_pose.pt"
        shutil.copyfile(best, dst)
        print(f"[train] best -> {dst}")
    print(f"[train] 输出目录 {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
