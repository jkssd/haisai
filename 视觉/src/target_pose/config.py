"""全局配置：场景、类别、关键点定义与目录约定。

本工程约定：
  * 每次标注的对象是「一个靶子」，类别只有一类 target；
  * 关键点 7 个：靶框左上、右上、右下、左下 + 圆心 + 圆的左右两个极点；
  * 圆心是靶心；圆的左右极点在"靶框轴"方向上，两者距离的一半 = 圆半径，
    半径用于区分靶子的 大 / 中 / 小（见 SIZE_CLASSES）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- 标签定义
# 类别 = 靶子型号，由「圆/靶框比例」聚类决定（见 size_class.py）。
# 顺序与 SIZE_CLASSES 一致：0=大 1=中 2=小。
CLASS_NAMES: List[str] = ["target_large", "target_medium", "target_small"]
KPT_NAMES: List[str] = ["tl", "tr", "br", "bl", "center", "r_left", "r_right"]
N_KPT: int = len(KPT_NAMES)
KPT_SHAPE: List[int] = [N_KPT, 3]
# 水平翻转后的关键点置换：左右角点互换、圆的左右极点互换，圆心不动
FLIP_IDX: List[int] = [1, 0, 3, 2, 4, 6, 5]

# 关键点索引常量
TL, TR, BR, BL, CENTER, R_LEFT, R_RIGHT = 0, 1, 2, 3, 4, 5, 6

VIS_UNKNOWN, VIS_OCCLUDED, VIS_VISIBLE = 0, 1, 2

# ---------------------------------------------------------------- 大小分类（已废弃为"类别"）
# 现状：训练标签不再区分大/中/小（一律 class 0 target + class 1 circle）。
# 档位改为推理时**按同一张图内的比例聚类**得到（见 ratio_cluster.py），
# 不同图之间互相独立、不共享阈值。下面的阈值只留给旧的 radius-stats 报告用。
# 用「圆半径 / 靶框半边长」这个与距离无关的比值来分大中小：
# 同一型号的靶子无论远近，该比值恒定；不同型号（印的圆大小不同）会分开。
SIZE_CLASSES: List[str] = ["large", "medium", "small"]
SIZE_CLASS_NAMES_CN: List[str] = ["大", "中", "小"]
# 阈值：ratio = r / (side/2)。ratio 越大 → 圆相对靶框越大 → 靶子型号越大。
# 由省赛 5 张图 / 15 个实例的 1D-kmeans 得到（三簇 0.442 / 0.665 / 0.957，分离度很好）：
#   ratio >= 0.81 → 大；>= 0.55 → 中；否则 小。
# 等 d2 / c2 标完再用 `pose.py radius-stats` 复核一次（不同场景可能是同样三种型号）。
SIZE_THRESHOLDS: List[float] = [0.81, 0.55]  # >=0.81 大；>=0.55 中；否则小


def size_class(ratio: float) -> int:
    """0=大 1=中 2=小（阈值以 data/size_classes.json 为准，此处仅兜底）。"""
    from .size_class import ratio_to_cls

    return ratio_to_cls(ratio)


# ---------------------------------------------------------------- 场景
@dataclass
class Scene:
    """一个采集场景 = 一套独立的光照/形态分布。"""

    key: str                      # 内部标识，用于文件名前缀
    name: str                     # 中文名
    image_dir: Path
    pattern: str = "*.jpg"
    voc_dir: Path | None = None   # 若已有 Pascal VOC 框标注，可作为初始提示
    note: str = ""
    # 已有 VOC 框相对"靶框"的比例：框本身就是靶框时为 1.0，
    # 框只是中心圆时约 1.25~1.4。用于标注工具的"按提示生成靶框"。
    hint_scale: float = 1.0
    # 靶框边长 / 中心圆半径（实测先验，供无提示时按圆推框、以及精修估半径）
    board_to_radius: float = 3.0
    # 每张图最多几个靶子（物理上限，用来压误检：人/窗框被当成靶子）。
    max_targets: int = 3
    # 每张图预期的靶子个数（0=不确定）。自动标注时用来兜底召回：
    # 检出数不等于预期就不写标签，避免"少标了 2 个却当成完整标签"污染训练。
    expect_targets: int = 0
    extra: Dict = field(default_factory=dict)

    def images(self) -> List[Path]:
        return sorted(p for p in self.image_dir.glob(self.pattern) if p.is_file())


DATASETS = ROOT / "datasets"

SCENES: Dict[str, Scene] = {
    "shengai": Scene(
        key="shengai",
        name="省赛-室外三靶",
        image_dir=DATASETS / "省赛Dataset",
        note="4K 手机拍摄，黑色靶板+白色圆，板面大、靶框清晰可见",
        board_to_radius=2.8,
        expect_targets=0,   # burst 会推拉镜头，常有只入画 2 块板的帧，不能强求 3
    ),
    "d2": Scene(
        key="d2",
        name="Dataset2-室内蓝板",
        # 线上仓库里图片统一放在 datasets/<场景>/（本地可能叫 JPEGImages）
        image_dir=(DATASETS / "Dataset2" / "JPEGImages") if (DATASETS / "Dataset2" / "JPEGImages").exists()
        else DATASETS / "Dataset2",
        voc_dir=DATASETS / "Dataset2" / "Annotations" if (DATASETS / "Dataset2" / "Annotations").exists() else None,
        note="640x480 室内，深蓝靶板+白圆，靶子很小(20~70px)、底部被桌沿遮挡",
        hint_scale=1.0,
        board_to_radius=2.2,
    ),
    "c2": Scene(
        key="c2",
        name="C2-墙面贴纸靶",
        image_dir=(DATASETS / "C2" / "JPEGImages") if (DATASETS / "C2" / "JPEGImages").exists()
        else DATASETS / "C2",
        voc_dir=DATASETS / "C2" / "Annotations" if (DATASETS / "C2" / "Annotations").exists() else None,
        note="1920x1080 白纸靶贴浅色墙面，部分图靶纸与墙同色、靶框不可见，尺寸差异大",
        hint_scale=1.25,
        board_to_radius=2.66,
    ),
    "comp": Scene(
        key="comp",
        name="赛局-蓝板黑孔（15 帧）",
        image_dir=ROOT / "data" / "new15" / "img",
        pattern="*.jpg",
        note="比赛现场视频截帧：一排蓝色立板，板中心一个黑色圆孔；"
             "孔有大/中/小，靶框=立板外框（板子物理尺寸相同，孔的大小不同）",
        board_to_radius=4.5,
        max_targets=6,
    ),
}


# ---------------------------------------------------------------- 目录
DATA = ROOT / "data"
ANN_DIR = DATA / "annotations"      # 主标注库 data/annotations/<scene>/<stem>.txt
META_DIR = DATA / "meta"            # 每条样本的元信息（来源/置信度/复核状态）
YOLO_DIR = DATA / "yolo"            # 训练用的数据目录
RUNS_DIR = ROOT / "runs"
WEIGHTS_DIR = ROOT / "weights"

ORIG_SUFFIX = ".origin"             # 原始(未精修)预标注的备份后缀


def ensure_dirs() -> None:
    for p in (ANN_DIR, META_DIR, YOLO_DIR, RUNS_DIR, WEIGHTS_DIR):
        p.mkdir(parents=True, exist_ok=True)
