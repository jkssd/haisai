"""ONNX → kmodel（fp32 / int8）。关键：**输入必须归一化到 0-1**（std=255）。

ultralytics 导出的 ONNX 期望 float32 0-1；nncase 用 input_range=[0,255] + std=[255,255,255]
正好等价于 (x/255)，喂 0-255 会得到完全错误的结果（分数饱和、关键点乱飞）。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SIZE = 640



def make_options(swapRB=True, size=640):
    import nncase
    c = nncase.CompileOptions()
    c.target = "k230"
    c.output_arrays = ["output0"]
    c.input_shape = [1, 3, size, size]
    c.input_type = "uint8"
    c.input_layout = "NCHW"
    c.input_range = [0, 255]
    c.mean = [0, 0, 0]
    c.std = [255, 255, 255]      # ← 归一化到 0-1，必须
    c.preprocess = True
    c.swapRB = swapRB            # 板子给 RGB 时置 False；PC 上 cv2 读的 BGR 时为 True
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "weights" / "target_pose_v4.onnx"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib", type=int, default=160)
    ap.add_argument("--rgb", action="store_true", help="输入已是 RGB（板端），不交换通道")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--data", default=str(ROOT / "data" / "yolo"),
                    help="校准用的 YOLO 目录（含 train.txt/val.txt），要跟训练数据一致")
    args = ap.parse_args(argv)

    import nncase
    import cv2
    global SIZE
    SIZE = args.size
    c = make_options(swapRB=not args.rgb, size=args.size)
    comp = nncase.Compiler(c)
    comp.import_onnx(Path(args.onnx).read_bytes(), nncase.ImportOptions())

    if args.int8:
        sys.path.insert(0, str(ROOT / "k230"))
        from quantize import build_calib  # noqa
        calib = build_calib(Path(args.data), rgb=args.rgb, size=args.size)
        if len(calib) > args.calib:
            calib = calib[: args.calib]
        print("calib", calib.shape)
        ptq = nncase.PTQTensorOptions()
        ptq.samples_count = len(calib)
        ptq.calibrate_method = "Kld"
        ptq.quant_type = "uint8"
        ptq.w_quant_type = "int8"
        ptq.use_mse_quant_w = True
        ptq.set_tensor_data([calib.astype(np.uint8)])
        comp.use_ptq(ptq)

    comp.compile()
    data = comp.gencode_tobytes()
    out = Path(args.out or (ROOT / "k230" / ("target_pose_v4_int8.kmodel" if args.int8 else "target_pose_v4.kmodel")))
    out.write_bytes(data)
    print(f"OK {out.name}  {len(data)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
