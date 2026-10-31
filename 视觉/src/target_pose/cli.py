"""统一命令行入口。

    python -m target_pose.cli stats
    python -m target_pose.cli build
    python -m target_pose.cli review --port 8765
    python -m target_pose.cli train --model yolov8s-pose.pt
    python -m target_pose.cli predict --limit 20
    python -m target_pose.cli eval --weights runs/pose_v1/weights/best.pt
    python -m target_pose.cli infer --source datasets/省赛Dataset --save-dir out/vis
    python -m target_pose.cli refine --source auto
"""
from __future__ import annotations

import argparse
import sys

COMMANDS = {
    "stats":   ("target_pose.tasks", "main_stats", "打印各场景标注进度"),
    "build":   ("target_pose.tasks", "main_build", "构建 YOLO-pose 数据集目录"),
    "refine":  ("target_pose.tasks", "main_refine", "批量几何精修已有标注"),
    "preview": ("target_pose.tasks", "main_preview", "标注可视化联系表"),
    "radius-stats": ("target_pose.tasks", "main_radius_stats", "圆半径比例分布/大小阈值建议"),
    "ratio-cluster": ("target_pose.tasks", "main_ratio_cluster", "按同图比例聚类分档（图间独立）"),
    "migrate": ("target_pose.tasks", "main_migrate", "5点标签→7点（补圆）"),
    "qc": ("target_pose.tasks", "main_qc", "清理不合格的自动标注"),
    "autolabel": ("target_pose.autolabel", "main", "无模型自动预标注（圆锚定+边缘贴合）"),
    "suggest": ("target_pose.tasks", "main_suggest", "挑选 bootstrap 种子样本"),
    "review":  ("target_pose.app", "main", "启动标注/复核 Web 工具"),
    "train":   ("target_pose.train", "main", "训练关键点模型"),
    "predict": ("target_pose.predict", "main", "用模型批量预标注"),
    "eval":    ("target_pose.evaluate", "main", "评估模型精度"),
    "infer":   ("target_pose.infer", "main", "靶心解算"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="python -m target_pose.cli", description="靶子关键点工具链")
    ap.add_argument("cmd", choices=list(COMMANDS), help="子命令")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    mod_name, fn_name, desc = COMMANDS[args.cmd]
    print(f"[{args.cmd}] {desc}", file=sys.stderr)
    mod = __import__(mod_name, fromlist=[fn_name])
    return getattr(mod, fn_name)(args.rest)


if __name__ == "__main__":
    raise SystemExit(main())
