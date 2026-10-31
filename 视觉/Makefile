PY ?= python3
PORT ?= 8765

.PHONY: help stats suggest review train predict eval infer preview build refine smoke clean-runs

help:
	@echo "make stats              统计标注进度"
	@echo "make suggest            挑选 60 张种子样本"
	@echo "make review             启动标注/复核工具 (PORT=8765)"
	@echo "make train              训练 v1 粗模型"
	@echo "make predict            全量预标注(仅未标注)"
	@echo "make eval               评估 weights/target_pose.pt"
	@echo "make infer              解算 datasets/省赛Dataset 并可视化"
	@echo "make preview            生成标注联系表"
	@echo "make refine             批量精修(输出到 data/annotations_refined)"
	@echo "make build              构建 YOLO 训练目录"
	@echo "make smoke              端到端自检"

stats:
	$(PY) pose.py stats

suggest:
	$(PY) pose.py suggest --per-scene 20

review:
	$(PY) pose.py review --port $(PORT)

train:
	$(PY) pose.py train --model yolov8s-pose.pt --epochs 120 --imgsz 960 --batch 4

predict:
	$(PY) pose.py predict --only-unlabeled --imgsz 1280

eval:
	$(PY) pose.py eval --weights weights/target_pose.pt --refine

infer:
	$(PY) pose.py infer --source datasets/省赛Dataset --save-dir out/vis --out out/bullseye.json

preview:
	$(PY) pose.py preview --limit 12

refine:
	$(PY) pose.py refine --source auto

build:
	$(PY) pose.py build

smoke:
	$(PY) tests/smoke_test.py --train 3

clean-runs:
	rm -rf runs/smoke
