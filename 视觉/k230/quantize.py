"""int8 PTQ 量化：fp32 kmodel 的 ONNX → int8 kmodel。

校准集**按场景分层**，并**排除留出集**（data/yolo/val.txt 里的 202 张），
否则精度回归会虚高。

    export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH
    export NNCASE_PLUGIN_PATH=$HOME/.local/lib/python3.10/site-packages/nncase
    python3 k230/quantize.py --onnx weights/target_pose_v4.onnx --out k230/target_pose_v4_int8.kmodel
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SIZE = 640
# 每场景校准张数（按场景分层，覆盖尺度/光照/背景差异）
PER_SCENE = {"shengai": 24, "d2": 30, "c2": 30, "comp": 10, "aug": 40, "psim": 16}


def letterbox_bgr(img, size=640, color=114):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = res
    return canvas


def build_calib(yolo_dir: Path, seed: int = 0, rgb: bool = False, size: int = 640):
    val_lines = [ln for ln in (yolo_dir / "val.txt").read_text().splitlines() if ln.strip()]
    val_sids = {Path(ln).stem for ln in val_lines}
    train_lines = [ln for ln in (yolo_dir / "train.txt").read_text().splitlines() if ln.strip()]
    rng = random.Random(seed)
    by_scene = {}
    for ln in train_lines:
        sid = Path(ln).stem
        if sid in val_sids:
            continue
        by_scene.setdefault(sid.split("__")[0], []).append(ln)

    imgs = []
    # 兜底：字典里没写的场景各取 12 张，保证新数据（aug/psim/comp）不被漏掉
    for scene in sorted(by_scene):
        PER_SCENE.setdefault(scene, 12)
    for scene, n in PER_SCENE.items():
        pool = by_scene.get(scene, [])
        rng.shuffle(pool)
        picked = pool[:n]
        print(f"  {scene}: 校准 {len(picked)}/{len(pool)} 张（已排除留出集）")
        for ln in picked:
            im = cv2.imread(ln)
            if im is None:
                continue
            lb = letterbox_bgr(im, size)
            imgs.append(lb[:, :, ::-1].copy() if rgb else lb)
    arr = np.stack(imgs, 0).transpose(0, 3, 1, 2)          # (N,3,640,640)
    arr = arr[:, None]                                      # (N,1,3,640,640)
    return arr.astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "weights" / "target_pose_v4.onnx"))
    ap.add_argument("--out", default=str(ROOT / "k230" / "target_pose_v4_int8.kmodel"))
    ap.add_argument("--mtp", action="store_true", help="多线程编译")
    ap.add_argument("--calib-method", default="Kld")
    args = ap.parse_args(argv)

    import nncase

    yolo = ROOT / "data" / "yolo"
    print("构建校准集…")
    calib = build_calib(yolo)
    n = calib.shape[0]
    print(f"校准集 {calib.shape}  ({calib.nbytes/1e6:.0f} MB)")

    c = nncase.CompileOptions()
    c.target = "k230"
    c.output_arrays = ["output0"]
    c.input_shape = [1, 3, SIZE, SIZE]
    c.input_type = "uint8"
    c.input_layout = "NCHW"
    c.input_range = [0, 255]
    c.mean = [0, 0, 0]
    c.std = [1, 1, 1]
    c.preprocess = True
    c.swapRB = True

    ptq = nncase.PTQTensorOptions()
    ptq.samples_count = n
    ptq.calibrate_method = args.calib_method     # Kld 比 NoClip 对 pose 回归头更稳
    ptq.quant_type = "uint8"                     # K230 KPU 对 uint8 激活支持最好
    ptq.w_quant_type = "int8"                    # 权重 int8
    ptq.use_mse_quant_w = True
    # 注意：校准数据必须是 uint8 (0-255)，float 会报 "need u8[...] but get f32[...]"
    ptq.set_tensor_data([calib.astype(np.uint8)])

    comp = nncase.Compiler(c)
    comp.import_onnx(Path(args.onnx).read_bytes(), nncase.ImportOptions())
    comp.use_ptq(ptq)
    comp.compile()
    data = comp.gencode_tobytes()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"OK {out}  {len(data)/1e6:.2f} MB  (fp32 12.60 MB, 压缩 {12.60/(len(data)/1e6):.1f}x)")


if __name__ == "__main__":
    main()
