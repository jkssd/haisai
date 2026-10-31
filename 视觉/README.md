# 海赛 · 视觉（目标识别）

矩形靶子（靶框 + 中心圆）的 7 点关键点检测，最终输出**靶心坐标**。训练用 YOLOv8-Pose，
部署到 K230（CanMV，448×448 int8，约 22 fps）。

```
[tl, tr, br, bl, center, r_left, r_right]        kpt_shape=[7,3]  flip_idx=[1,0,3,2,4,6,5]
```

---

## 1. 目录

```
视觉/
├── src/target_pose/        核心包（配置 / 标签 / 推理 / 靶心解算 / 比例聚类 / 训练）
│   └── web/                标注复核网页
├── tools/                  数据生成、训练、评测脚本（见第 5 节）
├── tests/                  smoke_test（链路自检）、eval_holdout（留出集评估）
├── k230/                   板端脚本 + 编译工具 + kmodel
├── weights/                模型权重（最终版 v12）
├── data/                   标注、派生标签、预生成样本、划分清单
├── datasets/               原始图片（见下）
├── out/                    效果图与可打印靶纸
└── docs/                   标注规范
```

### 数据集

| 目录 | 内容 | 说明 |
|---|---|---|
| `datasets/省赛Dataset/` | 233 张 4K 手机拍摄 | 黑色靶板 + 白圆，三靶并排 |
| `datasets/C2/` | 647 张 | 白纸靶贴墙，部分靶框不可见 |
| `datasets/Dataset2/` | 610 张 | 室内蓝板，靶子小、底部遮挡 |
| `data/new15/img/` | 15 帧比赛现场 | 蓝立板 + 黑圆孔，全部人工复核 |
| `data/annotations/` | 1358 个 txt | **人工标注真值**（7 点） |
| `data/ann2/` | 两分类训练标签 | 由 annotations + `tools/make_two_class.py` 派生 |
| `data/neg*/` | 负样本（无靶子） | 100 背景 + 200 合成横幅（硬负样本） |
| `data/banner_pos/` | 80 张「靶子 + 横幅同框」 | 正样本，标签不变 |
| `data/frame_only/` | 600 张「圆被涂掉」的标签 | 图片可用脚本重生成 |
| `data/aug_v2/` | 1400 张离线增强标签 | 图片可用脚本重生成 |
| `data/printsim/` | 140 张「打印靶纸仿真拍图」标签 | 图片可用脚本重生成 |
| `data/yolo_v12/` | `train.txt` / `val.txt` / `data.yaml` | 清单路径指向 `images/all/` 软链，需先跑 `tools/train_v12.py` |

## 2. 环境

```bash
python3 -m pip install ultralytics opencv-python numpy pillow     # 训练 / 推理
# K230 编译另需：nncase 2.11 + nncase-kpu + .NET runtime（见 k230/README.md）
```

## 3. 推理：一张图算靶心

```python
import sys, cv2
sys.path.insert(0, "src")
from target_pose.infer import solve_image
from target_pose.predict import _load_model

model = _load_model("weights/target_pose_v12.pt")
img = cv2.imread("sample.jpg")
for s in solve_image(model, img, imgsz=640, conf=0.25):
    print(s["bullseye"], s["bullseye_uv"], s["bullseye_source"],
          round(s["radius"] / (s["side"] / 2), 3))
```

输出字段：

| 字段 | 含义 |
|---|---|
| `corners` | 靶框 4 角（像素） |
| `bullseye` | **最终靶心**（像素） |
| `bullseye_uv` | 靶面归一化坐标，`(0.5,0.5)` 为正中心，可直接算环数 |
| `bullseye_source` | `circle`=框+圆融合 / `frame`=只有框 / `circle_only`=只有圆 |
| `radius` / `side` | 圆半径 / 靶框平均边长 |
| `radius_rel`、`ratio_group`、`size` | 比例 + **同图分档**（见第 4 节） |
| `center_offset_rel` | 圆心离框心的归一化偏移，>0.2 多半是误配 |

### 靶心怎么定

中心圆就画在靶框正中，所以「四角对角线交点」本身就是靶心，和距离、角度无关：

| 靶心来源 | 中位误差 | P90 |
|---|---|---|
| 只用 4 角点对角线交点（`frame`） | 3.04 px | 5.17 px |
| 只用模型圆心（`circle_only`） | 1.24 px | 2.84 px |
| **融合 0.85×圆心 + 0.15×框心** | **1.14 px** | **2.72 px** |

- 框心必须用**未精修**的角点算：精修会把软边往里拉，框心误差从 3.0 涨到 5.0 px；
- 圆心用精修后的（更准）；
- 圆没检出就退回框心，框没检出就只用圆心 —— 两条路都不会「算不出靶心」。

## 4. 靶子档位：同一张图内按比例聚类

`ratio = r / 靶框平均半边长`，与距离无关。**每张图独立**把 `log(ratio)` 排序，
相邻相对间隔 > 0.12 才切一刀，最多 3 档；图与图之间不共享阈值（现场每批靶子型号不同）。

```bash
python3 pose.py ratio-cluster --scene comp
```

实测 16 张图每张都自动切成 3 档，簇心约 `0.12 / 0.25 / 0.37`；换机位后整体偏到
`0.14 / 0.29 / 0.45`，仍然是 3 簇 —— 用全局阈值就会判错。

## 5. 训练与复现（v12）

```bash
# 1) 源标注 -> 两分类训练标签
python3 tools/fix_class_ids.py --dirs data/annotations --apply
python3 tools/make_two_class.py
python3 tools/fix_class_ids.py --apply
# 2) 负样本 / 增强 / 擦圆 / 打印仿真（图片体积大，按需重生成）
python3 tools/make_negatives.py
python3 tools/make_banner_negs.py                 # 横幅硬负样本
python3 tools/augment_v2.py --n 1400              # 换背景/换色/干扰物/曝光
python3 tools/make_frame_only.py --n 600          # 把圆涂掉，教模型别依赖圆
python3 tools/print_photo_sim.py --n 140          # 打印靶纸 -> 仿真拍图
# 3) 组数据 & 训练
python3 tools/train_v12.py                        # -> data/yolo_v12
python3 -m target_pose.cli train --data data/yolo_v12/data.yaml \
        --model runs/v11/weights/best.pt --epochs 30 --imgsz 640 --batch 8 --lr0 0.001
```

### 评测

```bash
python3 tools/eval_compare.py  --weights weights/target_pose_v12.pt  # 负样本/测试图/打印靶纸/赛局帧
python3 tools/eval_comp.py     --weights weights/target_pose_v12.pt  # 赛局帧召回 + 角点/靶心误差
python3 tools/eval_bullseye.py --weights weights/target_pose_v12.pt  # 靶心三种取法对比
python3 tools/preview_board_draw.py                                  # B/F/C 三种标记效果图
```

### 版本对比

| 版本 | 数据 | 负样本误检 | 固定测试图 | 赛局 15 帧 |
|---|---|---|---|---|
| v4 单类 | 1343 张 3 场景 | 4/200 | 2/3 板 | 差 |
| v7 双类 | + 派生 circle | 200/200（横幅） | 3 板 | 只认圆 |
| v8 双类 | + 横幅负样本 | **0** | 0（把赛局场景忘了） | 0 |
| v11 双类 | + 赛局 + 增强 + 打印仿真 | **0** | 3/3 板 + 3/3 圆 | 45/45 |
| **v12** | + 擦圆样本 | **0** | 3/3 + 3/3 | 45/45，靶心 1.14 px |

v12 验证集（425 张）：P 0.984 / R 0.996 / box mAP50 0.986 / pose mAP50-95 0.970。

## 6. K230 部署

```bash
python3 - <<'PY'
from ultralytics import YOLO
YOLO("weights/target_pose_v12.pt").export(format="onnx", imgsz=448, opset=12, simplify=True)
PY
export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH
export NNCASE_PLUGIN_PATH=$(python3 -c "import nncase,os;print(os.path.dirname(os.path.dirname(nncase.__file__)))")/..
python3 k230/build_kmodel.py --onnx weights/target_pose_v12.onnx \
        --out k230/target_pose_v12_448_int8.kmodel --int8 --rgb --size 448 --data data/yolo_v12
```

- `std=[255,255,255]` 把 0–255 归一化到 0–1（ONNX 就是这么训的）；
- 板子经 ai2d 给的是 RGB，所以 `--rgb`（不交换通道）；
- PTQ 校准必须用 uint8。

板端脚本 `k230/tft_live_v11.py`（文件名沿用历史，内容已是 v12）：

- 两类分开解码：框 0.25 / 圆 0.18（int8 量化后圆的分数会掉 0.1~0.2）；
- 候选 40 个再按类 NMS，最多留 12 个 —— 先取 6 个会被同一块板的高分重复框占满，圆永远进不来；
- 多帧一致性（连续 3 帧命中才确认），屏幕上**一个靶子只画一个靶心十字**：B 融合 / F 框心 / C 圆心。

## 7. 坑（都踩过）

1. ultralytics pose 的 ONNX 输出是 `4 + N_cls + 7×3` **三元组**，关键点 stride = 3；
2. kmodel 输入必须 0–1，喂 0–255 会让分数饱和、关键点乱飞；
3. `Sensor(id=2)` 之后必须 `sen.reset()`；`RGBP888` 才是 planar（NCHW），`RGB888` 是 HWC；
4. 面板逻辑分辨率是 **640×480**，不是 800×480；
5. 训练标签里 `class` 只能是 0/1，多出来的类号会被 ultralytics 当 corrupt 整张丢掉（v7 就这样丢过 1310 行小靶子）；
6. CanMV 没有 `traceback` 模块，异常处理要用 `sys.print_exception`；
7. MCP `canmv_write_file` 单次 base64 超过约 2.4 MB 会失败 → 分半上传再在板上拼接（`tools/upload_file.py`）。
