"""把 ann2 / banner_pos 的类别号规整成真正的两类：0=target(靶框) 1=circle(中心圆)。

背景：`make_two_class.py` 里派生 circle 行时用了 class 1，但 target 行仍沿用了
原来的**尺寸类别** 0/1/2（大/中/小）。结果 cls1 同时混进了「中号靶子」和「圆」，
cls2（小号靶子）在 class count=2 的 data.yaml 里直接被 ultralytics 当 corrupt 丢掉
——v7 就是这样训的（丢 1310 行小靶子，cls1 混两类）。

判定方式不依赖类别号：circle 行由派生器生成，4 个角点 v=0；target 行角点 v=2。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TARGET, CIRCLE = 0, 1


def remap_line(line: str) -> str | None:
    v = line.split()
    if len(v) < 5 + 21:
        return None
    kp = v[5:5 + 21]
    corner_v = [kp[2], kp[5], kp[8], kp[11]]
    is_circle = all(abs(float(x)) < 1e-6 for x in corner_v)
    cls = CIRCLE if is_circle else TARGET
    return " ".join([str(cls)] + v[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+",
                    default=[str(ROOT / "data" / "ann2"), str(ROOT / "data" / "banner_pos" / "labels")])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup", default=str(ROOT / "data" / "ann2_cls3_bak"))
    args = ap.parse_args()

    tot = {"in": 0, "tgt": 0, "cir": 0, "drop": 0}
    for d in args.dirs:
        d = Path(d)
        for f in sorted(d.rglob("*.txt")):
            lines = [x for x in f.read_text().splitlines() if x.strip()]
            out = []
            for line in lines:
                tot["in"] += 1
                r = remap_line(line)
                if r is None:
                    tot["drop"] += 1
                    continue
                cls = r.split()[0]
                tot["cir" if cls == "1" else "tgt"] += 1
                out.append(r)
            if args.apply:
                bak = Path(args.backup)
                try:
                    rel = f.resolve().relative_to((ROOT / "data").resolve())
                except ValueError:
                    rel = Path(f.name)
                dst = bak / rel
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(f, dst)
                f.write_text("\n".join(out) + ("\n" if out else ""))
    print(tot, "apply" if args.apply else "dry-run")


if __name__ == "__main__":
    main()
