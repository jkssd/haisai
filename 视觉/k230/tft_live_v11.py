# -*- coding: utf-8 -*-
"""K230 TFT 实时靶心识别（v12：双类 target + circle，448 int8，含"无圆也能出靶心"）。

与 v6 单类版的变化：
  1. 模型是两类：class0 = 靶框（4 角点有效）、class1 = 中心圆（圆心/左右极点有效）；
  2. 两条独立的轨迹表：框和圆各自做多帧一致性，互不抢位置；
  3. 画面上同时给出：靶框四角（编号）+ 圆 + 圆心十字 + 半径比例 r/(边长/2)；
  4. 同一帧里出现多个靶子时，按比例做**同图聚类**（不用全局大中小阈值）。

多帧一致性：同一位置连续 MIN_HITS 帧命中才输出；候选帧只画暗黄小十字。
"""
import sys
sys.path.insert(0, "/sdcard")
import math
import time
import nncase_runtime as nn
import ulab.numpy as np
from media.sensor import *
from media.display import *
from media.media import MediaManager
from libs.AI2D import Ai2d

KMODEL = "/sdcard/kmodel/target_pose_v12_448_int8.kmodel"
CAM_W, CAM_H = 640, 480
IN = 448
LCD_W, LCD_H = 640, 480
NCLS, NKPT = 2, 7
CONF_BOX, CONF_CIRCLE = 0.25, 0.18   # int8 量化后圆的分数会掉 0.1~0.2，圆单独放宽
NMS_IOU = 0.55
MAX_CAND = 40        # 先多取候选再按类 NMS：一块板会有几十个高分 anchor，
MAX_DET = 12         # 只取 6 个的话圆永远排不进去（这就是之前"只有框没有圆"的原因）
MIN_HITS, MAX_MISSES = 3, 2
DBG_EVERY = 120      # 仅用于调试打点，不再用它退出循环
TO_IDE = False
# 推理节流：TFT 看不出 22fps 与 25fps 的差别，但每帧都推理会把 ST7701 的 SPI 刷屏
# 挤掉、还拖慢整轮循环。每 2 帧跑一次 AI、其余帧复用上一轮确认的 pts 重画，
# 是 pose_webrtc.py 实测既稳又跟手的做法。想更跟手把 DETECT_EVERY 设 1。
DETECT_EVERY = 2
SHOW_GAP_MS = 66     # TFT 刷屏节流：ST7701 一次 640x480 RGB565 传输约 30ms，每轮都刷会卡
                      # 推理；限到 ~15fps 刷新足够看，把 CPU 让给取帧与检测。


def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def init_display():
    try:
        # 纯本地 TFT 不需要把每帧镜像到 IDE（USB 调试链路）；关掉 to_ide 少一次传输，
        # 主循环更稳、TFT 更跟手。要连 IDE 看画面时把它改回 True 即可。
        Display.init(Display.ST7701, width=LCD_W, height=LCD_H, to_ide=False)
        print("disp: ST7701 official (to_ide=False)")
    except Exception as e:
        print("disp st7701 err:", e)
        Display.init(Display.VIRT, width=LCD_W, height=LCD_H, to_ide=True, fps=60)
        print("disp: VIRT")


def line_intersect(p1, p2, p3, p4):
    """两条直线的交点（靶框对角线交点 = 靶面中心）。"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return ((x1 + x3) / 2.0, (y1 + y3) / 2.0)
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    px = (a * (x3 - x4) - (x1 - x2) * b) / den
    py = (a * (y3 - y4) - (y1 - y2) * b) / den
    return (px, py)


CIRCLE_W = 0.8          # 靶心 = 0.8*圆心 + 0.2*框心（45 个复核靶子扫出来的最优权重）


def cluster_groups(ratios):
    """同图比例聚类：log 空间按相对间隔切分，最多 3 档（与 PC 端 ratio_cluster 一致）。"""
    n = len(ratios)
    if n <= 1:
        return [0] * n
    idx = sorted(range(n), key=lambda i: ratios[i])
    srt = [ratios[i] + 1e-6 for i in idx]
    logs = [0.0] * n
    for i in range(n):
        logs[i] = math.log(srt[i])
    cuts = []
    for i in range(n - 1):
        gap = logs[i + 1] - logs[i]
        mid = (logs[i + 1] + logs[i]) / 2.0
        if gap / (abs(mid) + 1e-6) > 0.12:
            cuts.append((gap / (abs(mid) + 1e-6), i))
    cuts.sort(reverse=True)
    cuts = [c[1] for c in cuts[:2]]
    cuts.sort()
    lab = [0] * n
    for k, c in enumerate(cuts):
        for j in range(c + 1, n):
            lab[j] = k + 1
    out = [0] * n
    for pos, i in enumerate(idx):
        out[i] = lab[pos]
    return out


def main():
    init_display()
    # 摄像头帧率必须显式要 60：默认只有 30，主循环是「取帧→干活→再取帧」的帧锁结构，
    # 循环耗时会被量化成摄像头帧周期的整数倍 —— 周期 33ms 时 22ms 的活会被凑成 66ms，
    # 直接腰斩到 ~15fps（TFT 卡顿的主因）。显式 60fps 后两通道 640x480 实测能到 57fps。
    try:
        sen = Sensor(id=2, fps=60)
    except TypeError:
        sen = Sensor()          # 老固件 Sensor() 不吃 fps 关键字
    sen.reset()
    sen.set_framesize(w=CAM_W, h=CAM_H, chn=CAM_CHN_ID_1)
    sen.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)
    sen.set_framesize(w=CAM_W, h=CAM_H, chn=CAM_CHN_ID_2)
    sen.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_2)
    MediaManager.init()
    sen.run()
    time.sleep_ms(500)

    ai2d = Ai2d(0)
    ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
    ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
    ai2d.build([1, 3, CAM_H, CAM_W], [1, 3, IN, IN])
    kpu = nn.kpu()
    kpu.load_kmodel(KMODEL)
    print("READY %d lcd %dx%d" % (IN, LCD_W, LCD_H))

    sx = CAM_W / IN
    sy = CAM_H / IN
    tracks = []          # 每条 track: c(中心) h(命中) ms(丢失) m(本帧命中) i(anchor) v(分数) cls
    n = 0
    t_sum = 0
    t_inf = 0
    n_conf = 0
    n_cand = 0
    gate = 0             # 推理节流计数器：每 DETECT_EVERY 帧跑一次 AI
    keep = []            # 最近一次推理留下的候选（非检测帧复用其 pts 重画）
    o = None
    last_show_ms = 0     # TFT 刷屏节流时间戳
    print("LOOP start")

    try:
        while True:
            t0 = time.time_ns()
            img = sen.snapshot(chn=CAM_CHN_ID_1)
            arr = sen.snapshot(chn=CAM_CHN_ID_2).to_numpy_ref()
            t1 = time.time_ns()
            gate += 1
            # 推理节流：每 DETECT_EVERY 帧跑一次 KPU，其余帧跳过 AI、直接复用上一帧结果重画。
            # 跳过 AI 的那一轮整轮只要 ~8ms（取帧+画框），TFT 刷新立刻跟手、不再卡。
            if gate % DETECT_EVERY == 1 or o is None:
                # ---- 检测帧：跑 KPU + 完整 NMS + 多帧一致性，并取关键点 ----
                ten = ai2d.run(arr)
                kpu.set_input_tensor(0, ten)
                kpu.run()
                o = kpu.get_output_tensor(0).to_numpy()[0]
                t2 = time.time_ns()

                # 每个 anchor 取两类里分数高的那个；top-k 用 ulab 向量化，避免逐点 Python 循环
                smax = np.maximum(o[4], o[5])
                keep = []           # (score, box, anchor, cls)
                for _ in range(MAX_CAND):
                    i = int(np.argmax(smax))
                    cls = 0 if float(o[4, i]) >= float(o[5, i]) else 1
                    b0 = float(o[4, i]) if cls == 0 else float(o[5, i])
                    if b0 < (CONF_BOX if cls == 0 else CONF_CIRCLE):
                        smax[i] = 0
                        continue
                    smax[i] = 0
                    cx = float(o[0, i]); cy = float(o[1, i])
                    w2 = float(o[2, i]); h2 = float(o[3, i])
                    box = (cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2)
                    if any(iou(box, k[1]) >= NMS_IOU and k[3] == cls for k in keep):
                        continue
                    keep.append((b0, box, i, cls))
                    if len(keep) >= MAX_DET:
                        break

                # ---------- 多帧一致性 ----------
                for tr in tracks:
                    tr["m"] = False
                for v, box, i, cls in keep:
                    cx0 = (box[0] + box[2]) / 2.0
                    cy0 = (box[1] + box[3]) / 2.0
                    bw, bh = box[2] - box[0], box[3] - box[1]
                    lim = max(12.0, 0.6 * max(bw, bh))
                    best = None
                    bd = 1e9
                    for tr in tracks:
                        if tr["cls"] != cls:
                            continue
                        d = ((tr["c"][0] - cx0) ** 2 + (tr["c"][1] - cy0) ** 2) ** 0.5
                        if d < lim and d < bd:
                            best = tr
                            bd = d
                    if best is None:
                        tracks.append({"c": (cx0, cy0), "h": 1, "ms": 0, "m": True,
                                       "i": i, "v": v, "cls": cls})
                    else:
                        best["h"] += 1
                        best["ms"] = 0
                        best["m"] = True
                        best["c"] = (0.5 * best["c"][0] + 0.5 * cx0, 0.5 * best["c"][1] + 0.5 * cy0)
                        best["i"] = i
                        best["v"] = v
                alive = []
                for tr in tracks:
                    if not tr["m"]:
                        tr["ms"] += 1
                    if tr["ms"] <= MAX_MISSES:
                        alive.append(tr)
                tracks = alive[:16]

                confirmed = [tr for tr in tracks if tr["h"] >= MIN_HITS]
                cands = [tr for tr in tracks if tr["h"] < MIN_HITS]
                n_conf += len(confirmed)
                n_cand += len(cands)

                # 候选（未确认）不再画十字：屏幕上只保留"确认"的靶心标记

                idx_by_i = {}
                for v, box, i, cls in keep:
                    idx_by_i[i] = (v, box, cls)

                infos = []
                for tr in confirmed:
                    hit = idx_by_i.get(tr["i"])
                    if hit is None:
                        # 这一帧这个 anchor 没进 keep（被别的框抢了/刚好没过阈值）：
                        # 用上一帧存下来的点继续画，别让已经确认的靶子闪没
                        if "pts" not in tr:
                            continue
                        pts = tr["pts"]
                        cls = tr["cls"]
                        v = tr["v"]
                    else:
                        v, box, cls = hit
                        i = tr["i"]
                        pts = []
                        for k in range(NKPT):
                            pts.append((float(o[4 + NCLS + 3 * k, i]) * sx,
                                        float(o[4 + NCLS + 3 * k + 1, i]) * sy))
                        tr["pts"] = pts
                        tr["cls"] = cls
                    if cls == 0:            # 靶框
                        side = 0.0
                        for k in range(4):
                            q = pts[(k + 1) % 4]
                            side += ((q[0] - pts[k][0]) ** 2 + (q[1] - pts[k][1]) ** 2) ** 0.5
                        side /= 4
                        for k in range(4):
                            a1 = pts[k]
                            a2 = pts[(k + 1) % 4]
                            img.draw_line(int(a1[0]), int(a1[1]), int(a2[0]), int(a2[1]),
                                          color=(0, 255, 0), thickness=2)
                        for k in range(4):
                            px, py = int(pts[k][0]), int(pts[k][1])
                            img.draw_circle(px, py, 5, color=(0, 0, 255), fill=True)
                            img.draw_string_advanced(px + 6, py - 16, 20, str(k + 1),
                                                     color=(0, 0, 255))
                        fc = line_intersect(pts[0], pts[2], pts[1], pts[3])
                        infos.append({"cls": 0, "pts": pts, "side": side, "fc": fc,
                                      "c": ((pts[0][0] + pts[2][0]) / 2, (pts[0][1] + pts[2][1]) / 2),
                                      "v": v, "h": tr["h"]})
                    else:                   # 中心圆
                        cx, cy = pts[4]
                        r = (((pts[5][0] - pts[6][0]) ** 2 + (pts[5][1] - pts[6][1]) ** 2) ** 0.5) / 2
                        img.draw_circle(int(cx), int(cy), int(r), color=(255, 0, 255), thickness=2)
                        infos.append({"cls": 1, "c": (cx, cy), "r": r, "v": v, "h": tr["h"]})

                # 圆 ↔ 框 配对：一个靶子只出一个靶心标记
                #   B = 圆 + 框心融合（有框也有圆）
                #   F = 只有框 → 对角线交点
                #   C = 只有圆   → 圆心坐标
                boxes = [o2 for o2 in infos if o2["cls"] == 0]
                circles = [o2 for o2 in infos if o2["cls"] == 1]
                ratios = []
                used_circle = [False] * len(circles)
                targets = []
                for bx in boxes:
                    xs = [p[0] for p in bx["pts"][:4]]
                    ys = [p[1] for p in bx["pts"][:4]]
                    host, hd = None, 1e9
                    for j, ci in enumerate(circles):
                        if used_circle[j]:
                            continue
                        if min(xs) <= ci["c"][0] <= max(xs) and min(ys) <= ci["c"][1] <= max(ys):
                            d = abs(ci["c"][0] - bx["fc"][0]) + abs(ci["c"][1] - bx["fc"][1])
                            if d < hd:
                                host, hd = j, d
                    if host is None:
                        targets.append({"bull": bx["fc"], "src": "F", "box": bx})
                        continue
                    ci = circles[host]
                    used_circle[host] = True
                    rel = ci["r"] / (bx["side"] / 2) if bx["side"] > 1 else 0.0
                    ci["rel"] = rel
                    ratios.append(rel)
                    fc = bx["fc"]
                    bull = (CIRCLE_W * ci["c"][0] + (1.0 - CIRCLE_W) * fc[0],
                            CIRCLE_W * ci["c"][1] + (1.0 - CIRCLE_W) * fc[1])
                    targets.append({"bull": bull, "src": "B", "box": bx, "circle": ci, "rel": rel})
                for j, ci in enumerate(circles):          # 单独一个圆（没框）
                    if used_circle[j]:
                        continue
                    targets.append({"bull": ci["c"], "src": "C", "circle": ci})

                if len(ratios) >= 2:
                    grp = cluster_groups(ratios)
                    for ci, gi in zip([c for c in circles if "rel" in c], grp):
                        img.draw_string_advanced(int(ci["c"][0]) - 10,
                                                 int(ci["c"][1]) + int(ci["r"]) + 10, 20,
                                                 "G%d" % (gi + 1), color=(0, 255, 255))
            else:
                # ---- 非检测帧：跳过 AI，直接复用已确认 track 缓存的 pts 重画 ----
                # 跳过 AI 的那一轮整轮只要 ~8ms（取帧 + 画框），TFT 刷新立刻跟手、不再卡。
                confirmed = [tr for tr in tracks if tr["h"] >= MIN_HITS]
                n_conf += len(confirmed)
                infos = []
                for tr in confirmed:
                    if "pts" not in tr:           # 还没存过点就跳过，等下个检测帧
                        continue
                    pts = tr["pts"]
                    cls = tr["cls"]
                    v = tr["v"]
                    if cls == 0:
                        side = 0.0
                        for k in range(4):
                            q = pts[(k + 1) % 4]
                            side += ((q[0] - pts[k][0]) ** 2 + (q[1] - pts[k][1]) ** 2) ** 0.5
                        side /= 4
                        for k in range(4):
                            a1 = pts[k]
                            a2 = pts[(k + 1) % 4]
                            img.draw_line(int(a1[0]), int(a1[1]), int(a2[0]), int(a2[1]),
                                          color=(0, 255, 0), thickness=2)
                        for k in range(4):
                            px, py = int(pts[k][0]), int(pts[k][1])
                            img.draw_circle(px, py, 5, color=(0, 0, 255), fill=True)
                            img.draw_string_advanced(px + 6, py - 16, 20, str(k + 1),
                                                     color=(0, 0, 255))
                        fc = line_intersect(pts[0], pts[2], pts[1], pts[3])
                        infos.append({"cls": 0, "pts": pts, "side": side, "fc": fc,
                                      "c": ((pts[0][0] + pts[2][0]) / 2, (pts[0][1] + pts[2][1]) / 2),
                                      "v": v, "h": tr["h"]})
                    else:
                        cx, cy = pts[4]
                        r = (((pts[5][0] - pts[6][0]) ** 2 + (pts[5][1] - pts[6][1]) ** 2) ** 0.5) / 2
                        img.draw_circle(int(cx), int(cy), int(r), color=(255, 0, 255), thickness=2)
                        infos.append({"cls": 1, "c": (cx, cy), "r": r, "v": v, "h": tr["h"]})
                boxes = [x for x in infos if x["cls"] == 0]
                circles = [x for x in infos if x["cls"] == 1]
                ratios = []
                used_circle = [False] * len(circles)
                targets = []
                for bx in boxes:
                    xs = [p[0] for p in bx["pts"][:4]]
                    ys = [p[1] for p in bx["pts"][:4]]
                    host, hd = None, 1e9
                    for j, ci in enumerate(circles):
                        if used_circle[j]:
                            continue
                        if min(xs) <= ci["c"][0] <= max(xs) and min(ys) <= ci["c"][1] <= max(ys):
                            d = abs(ci["c"][0] - bx["fc"][0]) + abs(ci["c"][1] - bx["fc"][1])
                            if d < hd:
                                host, hd = j, d
                    if host is None:
                        targets.append({"bull": bx["fc"], "src": "F", "box": bx})
                        continue
                    ci = circles[host]
                    used_circle[host] = True
                    rel = ci["r"] / (bx["side"] / 2) if bx["side"] > 1 else 0.0
                    ci["rel"] = rel
                    ratios.append(rel)
                    fc = bx["fc"]
                    bull = (CIRCLE_W * ci["c"][0] + (1.0 - CIRCLE_W) * fc[0],
                            CIRCLE_W * ci["c"][1] + (1.0 - CIRCLE_W) * fc[1])
                    targets.append({"bull": bull, "src": "B", "box": bx, "circle": ci, "rel": rel})
                for j, ci in enumerate(circles):
                    if not used_circle[j]:
                        targets.append({"bull": ci["c"], "src": "C", "circle": ci})
                if len(ratios) >= 2:
                    grp = cluster_groups(ratios)
                    for ci, gi in zip([c for c in circles if "rel" in c], grp):
                        img.draw_string_advanced(int(ci["c"][0]) - 10,
                                                 int(ci["c"][1]) + int(ci["r"]) + 10, 20,
                                                 "G%d" % (gi + 1), color=(0, 255, 255))

            if confirmed:
                for tr in confirmed[:2]:
                    _p = tr.get("pts")
                    if _p:
                        print("  TRK cls=%d h=%d v=%.2f p0=(%.0f,%.0f) p4=(%.0f,%.0f)"
                              % (tr["cls"], tr["h"], tr["v"], _p[0][0], _p[0][1], _p[4][0], _p[4][1]))
                    else:
                        print("  TRK cls=%d h=%d v=%.2f NO_PTS" % (tr["cls"], tr["h"], tr["v"]))
                print("N=%d conf=%d cand=%d ratios=%s"
                      % (len(confirmed), n_conf, n_cand,
                         [round(float(x), 3) for x in ratios]))

            # TFT 刷屏节流：ST7701 一次 640x480 RGB565 传输约 30ms，每轮都刷会挤掉取帧/推理、
            # 让屏幕卡。限到 ~15fps 刷新足够看，把时间让给采集与检测。
            _now = time.ticks_ms()
            if time.ticks_diff(_now, last_show_ms) >= SHOW_GAP_MS:
                try:
                    Display.show_image(img)
                    last_show_ms = _now
                except Exception as e:
                    if n < 3:
                        print("show_image err:", type(e).__name__, e)
            t3 = time.time_ns()
            n += 1
            t_sum += (t3 - t0)
            if "t2" in dir() and t2 is not None:   # 只有检测帧才累加了 t2
                t_inf += (t2 - t1)
            if n % 60 == 0:
                try:
                    img.compress(quality=85).save("/sdcard/_v11_dbg.jpg")
                except Exception as e:
                    print("dbg save err", e)
            if n % 30 == 0:
                print("fps=%.1f inf=%.1fms conf=%d cand=%d"
                      % (30 * 1e9 / max(1, t_sum), t_inf / 30.0 / 1e6, n_conf, n_cand))
                t_sum = 0
                t_inf = 0
    except KeyboardInterrupt:
        pass
    print("LOOP end n=%d" % n)


try:
    main()
except Exception as e:
    # CanMV 没有 traceback 模块，用 sys.print_exception
    try:
        import sys as _sys
        _sys.print_exception(e)
    except Exception:
        print("FATAL", type(e).__name__, e)
