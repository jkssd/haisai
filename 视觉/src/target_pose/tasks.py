"""辅助任务：统计、构建、批量精修。"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from .config import ANN_DIR, DATA, SCENES
from .datasets import annotate_path, build_yolo, scan, stats
from .labels import Obj, Sample
from .refine import refine_target


# --------------------------------------------------------------------------- #
def main_stats(argv=None) -> None:
    ap = argparse.ArgumentParser(description="标注进度统计")
    ap.add_argument("--scene", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    items = scan(args.scene or None)
    print(stats(items))
    if args.json:
        Path(args.json).write_text(json.dumps(
            [{"sid": i.sid, "scene": i.scene, "n_objs": i.n_objs, "reviewed": i.reviewed,
              "source": i.source, "score": i.score} for i in items], ensure_ascii=False, indent=1))


def main_build(argv=None) -> None:
    ap = argparse.ArgumentParser(description="构建 YOLO-pose 数据集")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-reviewed", action="store_true")
    ap.add_argument("--min-objs", type=int, default=1)
    args = ap.parse_args(argv)
    build_yolo(val_ratio=args.val_ratio, seed=args.seed,
               only_reviewed=args.only_reviewed, min_objs=args.min_objs)


def main_refine(argv=None) -> None:
    """对已有标注做批量几何精修（角点边缘对齐 + 圆心重定位）。"""
    ap = argparse.ArgumentParser(description="批量几何精修")
    ap.add_argument("--scene", nargs="*", default=None)
    ap.add_argument("--source", default="", help="只处理指定来源，如 auto")
    ap.add_argument("--in-place", action="store_true", help="覆盖原标注（会备份 .origin）")
    ap.add_argument("--out-dir", default=None, help="输出到新目录（默认 data/annotations_refined）")
    ap.add_argument("--no-quad", action="store_true", help="只精修圆心，不动角点")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    items = scan(args.scene or None)
    if args.source:
        items = [i for i in items if i.source in args.source.split(",")]
    items = [i for i in items if i.n_objs > 0]
    print(f"待精修 {len(items)} 张")

    out_dir = Path(args.out_dir) if args.out_dir else (DATA / "annotations_refined")
    rows = []
    stat = {"objs": 0, "quad_ok": 0, "circ_ok": 0, "d_corner": [], "d_center": []}
    t0 = time.time()
    for k, it in enumerate(items, 1):
        img = cv2.imread(it.image)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = Sample.from_txt(Path(it.label), Path(it.image), it.width, it.height)
        new_objs = []
        for o in s.objs:
            ref, info = refine_target(gray, o, do_quad=not args.no_quad)
            stat["objs"] += 1
            stat["quad_ok"] += int(bool(info.get("quad_ok")))
            stat["circ_ok"] += int(bool(info.get("circ_ok")))
            stat["d_corner"].append(float(np.mean(np.linalg.norm(ref.corners - o.corners, axis=1))))
            stat["d_center"].append(float(np.linalg.norm(ref.center - o.center)))
            new_objs.append(ref)
            rows.append({"sid": it.sid, "shift_corner": round(stat["d_corner"][-1], 2),
                         "shift_center": round(stat["d_center"][-1], 2),
                         "quad_ok": bool(info.get("quad_ok")), "circ_ok": bool(info.get("circ_ok"))})
        s2 = Sample(image=Path(it.image), width=it.width, height=it.height, objs=new_objs)
        if args.in_place:
            lp = Path(it.label)
            if not Path(str(lp) + ".origin").exists():
                shutil.copyfile(lp, str(lp) + ".origin")
            lp.write_text(s2.to_txt())
        else:
            dst = out_dir / it.scene / Path(it.label).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(s2.to_txt())
        if k % 50 == 0:
            print(f"  [{k}/{len(items)}] {time.time() - t0:.0f}s", flush=True)

    if stat["objs"]:
        print(f"对象 {stat['objs']}：quad_ok={stat['quad_ok']} circ_ok={stat['circ_ok']} "
              f"角点平均位移 {np.mean(stat['d_corner']):.2f}px "
              f"圆心平均位移 {np.mean(stat['d_center']):.2f}px")
    if args.report:
        Path(args.report).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    if not args.in_place:
        print(f"输出 -> {out_dir}（与原标注对比后再决定是否替换）")


# --------------------------------------------------------------------------- #
def main_preview(argv=None) -> None:
    """把标注画到图上拼成联系表，用于离线核对（不依赖浏览器）。"""
    ap = argparse.ArgumentParser(description="标注可视化联系表")
    ap.add_argument("--scene", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--thumb", type=int, default=640)
    ap.add_argument("--out", default="out/preview")
    ap.add_argument("--only-labeled", action="store_true", default=True)
    args = ap.parse_args(argv)

    items = [i for i in scan(args.scene or None) if i.n_objs > 0]
    if not items:
        print("没有已标注样本")
        return
    # 每个场景均匀抽样
    by_scene: Dict[str, list] = {}
    for it in items:
        by_scene.setdefault(it.scene, []).append(it)
    chosen = []
    for key, arr in sorted(by_scene.items()):
        step = max(1, len(arr) // max(1, args.limit // max(1, len(by_scene))))
        chosen += arr[::step][: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    thumbs = []
    for it in chosen:
        img = cv2.imread(it.image)
        if img is None:
            continue
        s = Sample.from_txt(Path(it.label), Path(it.image), it.width, it.height)
        for o in s.objs:
            q = o.corners.astype(np.int32)
            cv2.polylines(img, [q], True, (0, 255, 0), 3)
            for k, p in enumerate(q):
                cv2.circle(img, tuple(p), 5, (0, 0, 255), -1)
            c = (int(o.center[0]), int(o.center[1]))
            if o.circle_labeled:
                cv2.circle(img, c, int(o.radius), (255, 0, 255), 3)
                cv2.circle(img, (int(o.circle_l[0]), int(o.circle_l[1])), 5, (255, 0, 255), -1)
                cv2.circle(img, (int(o.circle_r[0]), int(o.circle_r[1])), 5, (255, 0, 255), -1)
            cv2.drawMarker(img, c, (0, 255, 255), cv2.MARKER_CROSS, 24, 3)
            cv2.putText(img, f"{o.size_name} rel={o.radius_rel:.2f}", (c[0] - 40, c[1] - int(o.radius) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        sc = args.thumb / max(img.shape[:2])
        small = cv2.resize(img, None, fx=sc, fy=sc)
        cv2.putText(small, f"{it.scene}/{Path(it.image).stem} n={len(s.objs)}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        thumbs.append(small)
        print(f"  {it.sid}: {len(s.objs)} 个靶子")

    if not thumbs:
        return
    cols = max(1, args.cols)
    rows = (len(thumbs) + cols - 1) // cols
    cw = max(t.shape[1] for t in thumbs)
    ch = max(t.shape[0] for t in thumbs)
    sheet = np.full((rows * ch, cols * cw, 3), 24, np.uint8)
    for k, t in enumerate(thumbs):
        r, c = divmod(k, cols)
        sheet[r * ch:r * ch + t.shape[0], c * cw:c * cw + t.shape[1]] = t
    out = out_dir / "contact_sheet.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"联系表 -> {out}  ({len(thumbs)} 张)")


def main_suggest(argv=None) -> None:
    """挑选每个场景用于 bootstrap 的 N 张（尽量覆盖尺寸/位置/数量分布）。"""
    ap = argparse.ArgumentParser(description="挑选种子标注样本")
    ap.add_argument("--per-scene", type=int, default=20)
    ap.add_argument("--out", default=str(DATA / "splits" / "bootstrap.txt"))
    args = ap.parse_args(argv)

    from .datasets import voc_hints

    items = scan()
    by_scene: Dict[str, list] = {}
    for it in items:
        by_scene.setdefault(it.scene, []).append(it)

    picked: List[str] = []
    for key, arr in sorted(by_scene.items()):
        hints = voc_hints(key)
        scored = []
        for it in arr:
            stem = Path(it.image).stem
            boxes = hints.get(stem, [])
            if boxes:
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                cx = np.mean([(b[0] + b[2]) / 2 for b in boxes]) / max(it.width, 1)
                cy = np.mean([(b[1] + b[3]) / 2 for b in boxes]) / max(it.height, 1)
                size = float(np.sqrt(np.max(areas)))
                key_v = (round(size / 20.0), round(cx * 3), round(cy * 3), len(boxes))
            else:
                # 无先验：按文件名顺序分桶，保证跨连拍段均匀
                key_v = (0, 0, 0, 0)
            scored.append((key_v, it.sid, it))
        if hints:
            # 按 (尺寸桶, 横向位置桶, 纵向位置桶, 目标数) 分层，每层取 1
            buckets: Dict[tuple, list] = {}
            for kv, sid, it in scored:
                buckets.setdefault(kv, []).append(it)
            order = sorted(buckets, key=lambda k: (-k[0], k[1], k[2], k[3]))
            sel: List[str] = []
            i = 0
            while len(sel) < args.per_scene and any(buckets[k] for k in order):
                k = order[i % len(order)]
                if buckets[k]:
                    sel.append(buckets[k].pop(0).sid)
                i += 1
            picked += sel
        else:
            # 无先验：按 "burst 段" 分组，段内等间隔取
            groups: Dict[str, list] = {}
            for it in arr:
                stem = Path(it.image).stem
                g = stem.split("_Burst")[0] if "_Burst" in stem else "shot"
                groups.setdefault(g, []).append(it.sid)
            sel = []
            gi = 0
            keys = sorted(groups)
            while len(sel) < args.per_scene and any(groups[k] for k in keys):
                k = keys[gi % len(keys)]
                if groups[k]:
                    a = groups[k]
                    sel.append(a.pop(len(a) // 2))
                gi += 1
            picked += sorted(set(sel))
        print(f"  {key}: 选 {len([p for p in picked if p.startswith(key)])} 张")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(picked) + "\n")
    print(f"种子清单 -> {out}（共 {len(picked)} 张）")


# --------------------------------------------------------------------------- #
def main_ratio_cluster(argv=None) -> None:
    """按「同一张图内」的比例聚类分档（不同图互相独立，不用全局大中小阈值）。"""
    from .ratio_cluster import cluster_scene, summarize

    ap = argparse.ArgumentParser(description="同图比例聚类")
    ap.add_argument("--scene", nargs="*", default=["comp"])
    ap.add_argument("--min-gap", type=float, default=0.12)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    report = {}
    for scene in args.scene:
        rows = cluster_scene(scene, args.min_gap)
        if not rows:
            print(f"[{scene}] 无标注")
            continue
        report[scene] = summarize(rows)
        print(f"\n=== {scene}：{len(rows)} 张图 ===")
        print("图内档位数分布:", report[scene]["group_count_hist"])
        for stem, k, means in report[scene]["per_image"]:
            if args.quiet and k <= 1:
                continue
            print(f"  {stem}: {k} 档  {means}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print("->", args.json)


def main_radius_stats(argv=None) -> None:
    """统计「圆半径 / 靶框半边长」的分布，用于定 大/中/小 的阈值。"""
    ap = argparse.ArgumentParser(description="圆半径比例分布")
    ap.add_argument("--scene", nargs="*", default=None)
    ap.add_argument("--bins", type=int, default=16)
    ap.add_argument("--json", default=None)
    ap.add_argument("--apply", action="store_true", help="把聚类阈值写进 data/size_classes.json")
    args = ap.parse_args(argv)

    items = [i for i in scan(args.scene or None) if i.n_objs > 0]
    if not items:
        print("还没有标注样本")
        return
    per = {}
    for it in items:
        s = Sample.from_txt(Path(it.label), Path(it.image), it.width, it.height)
        for o in s.objs:
            if not o.circle_labeled:
                continue          # 没标圆的实例不参与比例统计
            per.setdefault(it.scene, {"rel": [], "px": [], "side": []})
            per[it.scene]["rel"].append(o.radius_rel)
            per[it.scene]["px"].append(o.radius)
            per[it.scene]["side"].append(o.side)

    def kmeans1d(x, k=3, iters=60):
        xs = np.sort(np.asarray(x, float))
        if len(xs) < k:
            return None
        cs = np.quantile(xs, np.linspace(0.15, 0.85, k))
        for _ in range(iters):
            lab = np.argmin(np.abs(xs[:, None] - cs[None, :]), axis=1)
            new = np.array([xs[lab == j].mean() if (lab == j).any() else cs[j] for j in range(k)])
            if np.allclose(new, cs):
                break
            cs = new
        order = np.argsort(cs)
        return cs[order]

    report = {}
    for key, d in sorted(per.items()):
        rel = np.array(d["rel"])
        px = np.array(d["px"])
        print(f"\n{SCENES[key].name}  实例 {len(rel)}")
        print(f"  比例 r/半边长: 中位 {np.median(rel):.3f}  min {rel.min():.3f}  max {rel.max():.3f}")
        print(f"  圆半径(px)  : 中位 {np.median(px):.1f}  min {px.min():.1f}  max {px.max():.1f}")
        lo, hi = rel.min(), rel.max()
        hist, edges = np.histogram(rel, bins=args.bins, range=(lo, max(hi, lo + 1e-6)))
        mx = hist.max() or 1
        for h, e in zip(hist, edges):
            bar = "█" * int(round(h / mx * 40))
            print(f"    {e:5.2f} {bar} {h}")
        cs = kmeans1d(rel.tolist(), 3)
        if cs is not None:
            thr = [(cs[i] + cs[i + 1]) / 2 for i in range(len(cs) - 1)]
            print(f"  1D-kmeans 中心 {np.round(cs, 3).tolist()}  → 建议阈值 {np.round(thr[::-1], 3).tolist()}"
                  f"（SIZE_THRESHOLDS = {np.round(thr[::-1], 3).tolist()}，大在前）")
            report[key] = {"kmeans": cs.tolist(), "thresholds": thr[::-1],
                           "rel_median": float(np.median(rel)), "n": int(len(rel))}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1))

    if args.apply:
        from .size_class import calibrate, describe
        cal = calibrate({k: v["rel"] for k, v in per.items()}, k=3)
        print("\n已写入 data/size_classes.json（按场景分别标定）\n" + describe())


# --------------------------------------------------------------------------- #
def main_migrate(argv=None) -> None:
    """把旧的 5 点标签升级成 7 点：保留人工四角，圆心/半径用圆定位补全。"""
    ap = argparse.ArgumentParser(description="5点→7点迁移")
    ap.add_argument("--scene", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--redo-circle", action="store_true", help="已是7点也重新定位圆心/半径")
    args = ap.parse_args(argv)

    from .refine import locate_circle_robust

    items = [i for i in scan(args.scene or None) if i.n_objs > 0]
    n_img = n_obj = n_circ = n_nocirc = 0
    for it in items:
        txt = Path(it.label)
        lines = [ln for ln in txt.read_text().splitlines() if ln.strip()]
        if not lines:
            continue
        n_tok = len(lines[0].split())
        if n_tok >= 5 + 21 and not args.redo_circle:
            continue                      # 已是 7 点
        img = cv2.imread(it.image)
        s = Sample.from_txt(txt, Path(it.image), it.width, it.height)
        for o in s.objs:
            r_hint = o.side / 3.1
            cen, rad, info = locate_circle_robust(img, o.center, r_hint, side=o.side, quad=o.corners)
            if info.get("circ_ok") and rad > 0:
                o.set_circle(cen, rad)
                o.vis[4] = o.vis[5] = o.vis[6] = 2
                n_circ += 1
            else:
                # 靶框内没找到可靠的圆：圆心/极点标为未标注，避免把干扰学进去
                o.vis[4] = o.vis[5] = o.vis[6] = 0
                o.cls = 1
                n_nocirc += 1
            n_obj += 1
        n_img += 1
        if not args.dry_run:
            txt.write_text(s.to_txt())
            mp = Path(str(txt).replace("/annotations/", "/meta/")).with_suffix(".json")
            mp.parent.mkdir(parents=True, exist_ok=True)
            meta = json.loads(mp.read_text()) if mp.exists() else {}
            meta.update({"source": "user", "reviewed": False,
                         "n_objs": len(s.objs),
                         "note": "5点→7点迁移：四角为人工标注，圆心/半径由圆定位补全，待确认"})
            mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        print(f"  {it.sid}: {len(s.objs)} 个靶子")
    print(f"迁移 {n_img} 张 / {n_obj} 实例，{n_circ} 个圆心由算法补全，{n_nocirc} 个未找到圆（已置 v=0）"
          + ("（dry-run 未写盘）" if args.dry_run else ""))


# --------------------------------------------------------------------------- #
def main_qc(argv=None) -> None:
    """清理不合格的自动标注：数量不足预期 / 空标注 / 置信度过低。

    只动 source=auto 的样本，人工标注（source=user）一律不碰。
    """
    ap = argparse.ArgumentParser(description="自动标注质量清理")
    ap.add_argument("--scene", nargs="*", default=None)
    ap.add_argument("--min-score", type=float, default=0.75)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    items = scan(args.scene or None)
    reasons = {"empty": 0, "count": 0, "score": 0}
    for it in items:
        if it.source != "auto":
            continue
        exp = SCENES[it.scene].expect_targets
        bad = None
        if it.n_objs == 0:
            bad = "empty"
        elif exp and it.n_objs != exp:
            bad = "count"
        elif it.score < args.min_score:
            bad = "score"
        if bad is None:
            continue
        reasons[bad] += 1
        if args.dry_run:
            continue
        lp, mp = Path(it.label), Path(str(it.label).replace("/annotations/", "/meta/")).with_suffix(".json")
        if lp.exists():
            lp.unlink()
        if mp.exists():
            mp.unlink()
    print(f"清理: 空 {reasons['empty']} / 数量不符 {reasons['count']} / 低置信 {reasons['score']}"
          + ("（dry-run）" if args.dry_run else ""))
