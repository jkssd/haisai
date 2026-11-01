# -*- coding: utf-8 -*-
"""K230 脱机三合一：YOLO 靶心识别 + TFT 本地显示 + WebRTC 无线图传。

关键点（踩过的坑）：
  * `Display.init(..., to_ide=True)` 会和 VENC 抢资源，导致 snapshot 阻塞 ——
    必须用 `to_ide=False`，否则取帧循环直接卡死；
  * 不要用 `MediaManager.link` 把 VENC 绑到传感器，那样多通道 snapshot 会被饿死；
    改用 `snapshot(chn=0, dump_frame=True)` + `encoder.SendFrame()` 手动喂帧；
  * 三个通道分辨率必须一致（都是 640x480），否则传感器侧会出问题。

通道分配：
  chn0 YUV420SP  -> 编码器（WebRTC 推流）
  chn1 RGB565    -> 画检测结果 + 显示到 TFT
  chn2 RGBP888   -> AI 推理输入
"""
import os
import sys
import time

sys.path.insert(0, "/sdcard")

import nncase_runtime as nn
import network
import ulab.numpy as np
import uctypes

from media.sensor import *
from media.display import *
from media.media import *
from media.vencoder import *
from libs.AI2D import Ai2d

import webrtc
import webrtc_stream as ws
from nonai2d import CSC

CFG = ws.CFG

KMODEL = CFG.get("kmodel") or "/sdcard/kmodel/target_pose_v12_448_int8.kmodel"
# 采集/显示/编码统一用同一个分辨率（画框和编码共用同一帧）。
# 低分辨率能显著降低带宽和 CPU 开销，是 2.4G 链路卡顿时最有效的手段。
W = int(CFG.get("width") or 640)
H = int(CFG.get("height") or 480)
W = W // 16 * 16
H = H // 16 * 16
IN = 448
NCLS, NKPT = 2, 7
CONF_BOX, CONF_CIRCLE = 0.25, 0.18
NMS_IOU = 0.55
MAX_CAND = 40
MAX_DET = 12
MIN_HITS, MAX_MISSES = 3, 2
CIRCLE_W = 0.8
BIT_RATE = int(CFG.get("bit_rate") or 1000)
DETECT_EVERY = int(CFG.get("detect_every") or 2)
GOP = int(CFG.get("gop") or 30)
# 码率硬上限。CBR 在复杂场景会超发，2.4G 上一超发就拥塞丢包 -> 花屏。
# 0 = 不限制（保持老行为）。
BIT_RATE_MAX = int(CFG.get("bit_rate_max") or 0)

# --- 自适应码率（STA 模式读 RSSI 做闭环） ---
ADAPTIVE = bool(CFG.get("adaptive", True))
LOW_RATE = int(CFG.get("bit_rate_low") or max(300, BIT_RATE // 4))
RSSI_LOW = int(CFG.get("rssi_low") or -70)     # 平滑后低于此值 -> 降码率
RSSI_HIGH = int(CFG.get("rssi_high") or -58)   # 平滑后高于此值 -> 升回
RSSI_LOW_N = int(CFG.get("rssi_low_n") or 2)
RSSI_HIGH_N = int(CFG.get("rssi_high_n") or 6)


def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def line_intersect(p1, p2, p3, p4):
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


def cluster_groups(ratios):
    import math
    n = len(ratios)
    if n <= 1:
        return [0] * n
    idx = sorted(range(n), key=lambda i: ratios[i])
    srt = [ratios[i] + 1e-6 for i in idx]
    logs = [math.log(x) for x in srt]
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


def gap_for(fps):
    """目标帧间隔（毫秒）。

    摄像头给 60fps，比主循环快得多，所以**帧率完全由这个值决定**：
    循环末尾会补睡到该间隔，送帧判定用它的 -3ms 容差。
    以前用「距离上一帧 >= 1000/fps」做节流，而主循环本身只要 ~21ms，
    于是 27ms 的间隔变成「每隔一轮送一次」= 24fps，白白丢掉一半帧。
    """
    return max(1, int(1000.0 / fps)) if fps > 0 else 0


def detect(kpu, ai2d, arr):
    """跑一次推理，返回 (keep, o)。keep = [(score, box, anchor, cls), ...]。"""
    ten = ai2d.run(arr)
    kpu.set_input_tensor(0, ten)
    kpu.run()
    o = kpu.get_output_tensor(0).to_numpy()[0]

    smax = np.maximum(o[4], o[5])
    keep = []
    for _ in range(MAX_CAND):
        i = int(np.argmax(smax))
        # 早退：smax 是「每次取完就清零」的降序序列，一旦当前最大值低于最低阈值
        # （圆的 0.18），后面不可能再有候选过阈值。以前这里会老老实实跑满 40 次
        # argmax（每次要扫 8400 个 anchor），实测单次推理 113ms 里有一大半耗在这。
        if float(smax[i]) < CONF_CIRCLE:
            break
        cls = 0 if float(o[4, i]) >= float(o[5, i]) else 1
        b0 = float(o[4, i]) if cls == 0 else float(o[5, i])
        if b0 < (CONF_BOX if cls == 0 else CONF_CIRCLE):
            smax[i] = 0
            continue
        smax[i] = 0
        cx = float(o[0, i])
        cy = float(o[1, i])
        w2 = float(o[2, i])
        h2 = float(o[3, i])
        box = (cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2)
        dup = False
        for k in keep:
            if k[3] == cls and iou(box, k[1]) >= NMS_IOU:
                dup = True
                break
        if dup:
            continue
        keep.append((b0, box, i, cls))
        if len(keep) >= MAX_DET:
            break
    return keep, o


def run():
    global DETECT_EVERY
    ip = ws.setup_network()

    # 拿到 STA 网卡句柄（sta / auto 模式连上热点时才有），用来读 RSSI
    link = None
    try:
        _w = network.WLAN(network.STA_IF, ws._wlan_device())
        if _w.isconnected():
            link = _w
            print("RSSI: 已启用自适应码率（rssi=%s）" % (_w.status("rssi"),))
        else:
            print("RSSI: 非 STA 连接，自适应码率关闭（AP 模式读不到客户端信号）")
    except Exception as e:
        print("RSSI: 获取网卡句柄失败 %s" % (e,))

    # 显示：to_ide 必须是 False，否则和 VENC 抢资源
    try:
        Display.init(Display.ST7701, width=640, height=480, to_ide=False)
        print("显示: ST7701 (to_ide=False)")
    except Exception as e:
        print("显示 ST7701 失败: %s，退回 VIRT" % e)
        Display.init(Display.VIRT, width=640, height=480, to_ide=False)

    # 摄像头帧率必须显式要 60：`Sensor()` 的默认帧率只有 30（甚至更低），
    # 而主循环是「取一帧→干活→再取一帧」的帧锁结构，循环耗时会被量化成
    # 摄像头帧周期的整数倍：周期 33ms 时 22ms 的活会被凑成 66ms，直接腰斩。
    # 厂家基线（sdcard_backup/k230_perf.py）也是 Sensor(id=2, fps=60)，
    # 双通道 640x480 实测能到 57fps，所以采集不是瓶颈。
    SENSOR_FPS = int(CFG.get("sensor_fps") or 60)
    try:
        sen = Sensor(id=2, fps=SENSOR_FPS)
    except TypeError:
        sen = Sensor()          # 老固件 Sensor() 不吃 fps 关键字
    sen.reset()
    # 只用两个通道：chn1 画框+显示+编码（同一帧），chn2 给 AI
    sen.set_framesize(w=W, h=H, chn=CAM_CHN_ID_1)
    sen.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_1)
    sen.set_framesize(w=W, h=H, chn=CAM_CHN_ID_2)
    sen.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_2)

    MediaManager.init()

    def make_encoder(rate, fps=30):
        """rate: 目标码率 kbit/s；fps: 实际送帧帧率（同时也是声明的输入帧率）。

        【为什么 src_frame_rate 必须等于真实送帧率】
        VENC 的 CBR 是「每帧预算 = bit_rate / src_frame_rate」再按送进来的帧数累加。
        以前固定写 src_frame_rate=60，而手动 SendFrame 实际只送 ~17 帧/秒，
        于是真实码率只有配置值的 17/60（≈28%）：
            配置 1500 -> 实测 509 kbit/s；配置 1200 -> 实测 305 kbit/s。
        640x480 每帧只剩 3KB 左右，画面全是块状拖影 —— 这就是「花屏」的主因，
        跟丢包没关系（浏览器侧 packetsLost 一直是 0）。
        把 src/dst 都写成真实帧率后，码率才会真的落在配置值附近。
        """
        rate = int(rate)
        if BIT_RATE_MAX > 0 and rate > BIT_RATE_MAX:
            rate = BIT_RATE_MAX
        fps = int(fps) if fps > 0 else 30
        e = Encoder()
        e.SetOutBufs(8, W, H)
        e.Create(ChnAttrStr(Encoder.PAYLOAD_TYPE_H264, Encoder.H264_PROFILE_MAIN, W, H,
                            bit_rate=rate, gopLen=GOP,
                            src_frame_rate=fps, dst_frame_rate=fps))
        e.Start()
        return e

    def swap_encoder(old, rate, fps):
        """换码率/帧率只能重建编码器（约 1 秒毛刺），重建后立刻要一个 IDR。"""
        try:
            old.Stop()
            old.Destroy()
        except Exception:
            pass
        e = make_encoder(rate, fps)
        try:
            e.RequestIDR()
        except Exception:
            pass
        return e

    # 画质档：手机浏览器访问 /near /mid /far /micro 一键切换
    PROFILES = {
        "near": {"rate": BIT_RATE, "fps": 30, "detect": DETECT_EVERY},
        "mid": {"rate": max(800, BIT_RATE // 2), "fps": 20, "detect": max(DETECT_EVERY, 3)},
        "far": {"rate": LOW_RATE, "fps": 15, "detect": max(DETECT_EVERY, 4)},
        "micro": {"rate": max(200, LOW_RATE // 2), "fps": 10, "detect": max(DETECT_EVERY, 5)},
    }
    cur_fps_cap = int(CFG.get("fps") or 25)
    cur_rate = BIT_RATE
    enc = make_encoder(cur_rate, cur_fps_cap)
    # 喂帧节流：编码器每收到一帧就编一帧，主循环跑多快它就编多快，
    # 只有主动控制送帧节奏才能真正限制帧率（否则码率被摊薄、画质变差）。
    frame_gap_ms = gap_for(cur_fps_cap)
    feed_gap_ms = max(1, frame_gap_ms - 3) if frame_gap_ms else 0
    # feed_all=True：不做软件节流，每一轮都喂给编码器。
    # 【实测结论：别开】VENC 的 dst_frame_rate 只在「声明的输入帧率」上参与码率分配，
    # **不会真的丢帧**：喂 45.7 帧/秒它就编 45.7 帧/秒，码率冲到 1900 kbit/s，
    # 2.4G 链路直接拥塞。所以帧率必须由软件节流控制。
    if CFG.get("feed_all"):
        frame_gap_ms = 0
        feed_gap_ms = 0
    last_feed_ms = 0
    last_show_ms = 0

    # RGB565 -> YUV420SP，把画好框的帧转给编码器，这样推流画面带识别框
    csc = CSC(CSC.PIXEL_FORMAT_YUV_SEMIPLANAR_420, max_width=W, max_height=H, buf_num=6)

    ai2d = Ai2d(0)
    ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
    ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
    ai2d.build([1, 3, H, W], [1, 3, IN, IN])

    kpu = None
    detect_on = False
    try:
        os.stat(KMODEL)
        kpu = nn.kpu()
        kpu.load_kmodel(KMODEL)
        detect_on = True
        print("模型: %s" % KMODEL)
    except Exception as e:
        print("模型不可用（%s），先只跑画面+推流" % e)

    server = ws.SignalingServer(webrtc.CODEC_H264)
    sen.run()
    server.start()
    print("图传地址: http://%s:%d/" % (ip, ws.HTTP_PORT))
    print("编码: h264 %dx%d %d kbit/s, 检测: %dx%d(每 %d 帧), 模型: %s" %
          (W, H, BIT_RATE, W, H, DETECT_EVERY, KMODEL))
    ws._self_check(ip, ws.HTTP_PORT)

    wd_state = {"stop": False}
    try:
        import _thread
        _thread.start_new_thread(ws._wifi_watchdog, (wd_state,))
        print("网络: 看门狗已启动")
    except Exception as e:
        print("网络: 看门狗启动失败 %s" % (e,))

    # 主循环心跳（不要叫 alive：检测代码里 alive 是跟踪列表）
    hb = {"t": time.ticks_ms(), "fps": 30.0, "low": 0}

    # 硬件看门狗：能救回多线程在 C 层一起死锁的情况（软件监管线程救不了）
    wdt = None
    if CFG.get("wdt", True):
        try:
            from machine import WDT
            wdt = WDT(1, timeout=30)
            print("看门狗: 硬件 WDT 已启用（30 秒）")
        except Exception as e:
            print("看门狗: WDT 不可用 %s" % (e,))

    def _supervisor():
        import machine
        while not wd_state["stop"]:
            time.sleep(3)
            if time.ticks_diff(time.ticks_ms(), hb["t"]) > 15000:
                print("监管: 主循环卡住超过 15 秒，复位板子")
                try:
                    machine.reset()
                except Exception:
                    pass
                return
            # 卡到几乎不动的状态（ICE 死会话抢 CPU 时就是这样）也要能自愈
            if hb["fps"] < 8:
                hb["low"] += 1
            else:
                hb["low"] = 0
            if hb["low"] >= 5:
                print("监管: 帧率持续低于 8fps，复位板子")
                try:
                    machine.reset()
                except Exception:
                    pass
                return

    try:
        import _thread as _th
        _th.start_new_thread(_supervisor, ())
        print("监管: 主循环看门狗已启动（15 秒）")
    except Exception as e:
        print("监管: 启动失败 %s" % (e,))

    sx = W / IN
    sy = H / IN
    tracks = []
    data = StreamData()
    parameter_sets = None
    frames = 0
    sent_bytes = 0
    t_stat = time.ticks_ms()
    n_conf_total = 0
    cur_fps = 0.0
    n_hdr = 0
    n_i = 0
    T_SNAP = T_AI = T_DRAW = T_CSC = T_SHOW = T_GET = 0
    T_IMG = T_FEED = T_SEND = 0
    pending_feed = False   # 上一轮喂过帧、还没取流
    gated = 0
    gate = 0
    low_streak = 0
    high_streak = 0
    cur_rssi = None
    t_rssi = time.ticks_ms()
    last_states = "-"
    targets_ready = False
    targets = []
    ratios = []
    circles = []

    try:
        while True:
            os.exitpoint()
            hb["t"] = time.ticks_ms()
            if wdt is not None:
                try:
                    wdt.feed()
                except Exception:
                    pass
            if server.stopping:
                raise RuntimeError("signaling stopped")
            if server.reboot_at is not None and \
                    time.ticks_diff(time.ticks_ms(), server.reboot_at) >= 0:
                print("OTA: 收到 /reboot，复位板子")
                import machine
                machine.reset()

            # 0) 先取「上一轮」喂进去那帧的码流。
            #    编码器是异步的：SendFrame 返回时还没编完，如果紧接着 GetStream，
            #    就是干等满 10ms 超时（实测这正是把 25fps 压到 17fps 的元凶）。
            #    把取流挪到下一轮开头，编码器已经有一整轮的时间，几乎立刻返回。
            _t_loop = time.ticks_ms()
            _t5 = time.ticks_ms()
            _get_budget = 3 if pending_feed else 1
            for _k in range(2 if pending_feed else 1):
                if time.ticks_diff(time.ticks_ms(), _t5) > _get_budget:
                    break
                if enc.GetStream(data, timeout=10 if (_k == 0 and pending_feed) else 1) != 0:
                    break
                try:
                    for index in range(data.pack_cnt):
                        payload = uctypes.bytearray_at(data.data[index], data.data_size[index])
                        stype = data.stream_type[index]
                        ts = data.pts[index]
                        if stype == enc.STREAM_TYPE_HEADER:
                            parameter_sets = bytes(payload)
                            n_hdr += 1
                        else:
                            _ts0 = time.ticks_ms()
                            if stype == enc.STREAM_TYPE_I and parameter_sets:
                                server.send_video(parameter_sets, ts)
                                sent_bytes += len(parameter_sets)
                                n_i += 1
                            server.send_video(payload, ts)
                            T_SEND += time.ticks_diff(time.ticks_ms(), _ts0)
                            sent_bytes += len(payload)
                            frames += 1
                finally:
                    enc.ReleaseStream(data)
            pending_feed = False
            T_GET += time.ticks_diff(time.ticks_ms(), _t5)

            # 1) 取一帧（dump_frame 拿到帧对象，to_image 得到可绘制视图，
            #    画完再用 CSC 转成 YUV 送编码器 —— 这样推流画面带识别框）
            _t0 = time.ticks_ms()
            fi = sen.snapshot(chn=CAM_CHN_ID_1, dump_frame=True)
            T_SNAP += time.ticks_diff(time.ticks_ms(), _t0)
            if fi is None:
                continue
            _ti = time.ticks_ms()
            img = fi.v_frame.to_image()
            T_IMG += time.ticks_diff(time.ticks_ms(), _ti)

            # 新会话连上时立刻请求一个关键帧，并补发 SPS/PPS
            if server.need_keyframe():
                try:
                    enc.RequestIDR()
                except Exception:
                    pass
                if parameter_sets and server.connected_count():
                    server.send_video(parameter_sets, time.ticks_ms())
                print("会话就绪：已请求 IDR")

            gate += 1
            # AI 每 DETECT_EVERY 帧跑一次，其余帧复用上次结果重画，避免拖慢帧率
            _t1 = time.ticks_ms()
            if detect_on and (gate % DETECT_EVERY == 1 or not targets_ready):
                gated += 1
                # 2) AI 通道
                arr = sen.snapshot(chn=CAM_CHN_ID_2).to_numpy_ref()
                keep, o = detect(kpu, ai2d, arr) if detect_on else ([], None)

                # --- 多帧一致性 ---
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
                        best["c"] = (0.5 * best["c"][0] + 0.5 * cx0,
                                     0.5 * best["c"][1] + 0.5 * cy0)
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
                n_conf_total = len(confirmed)

                idx_by_i = {}
                for v, box, i, cls in keep:
                    idx_by_i[i] = (v, box, cls)

                infos = []
                for tr in confirmed:
                    hit = idx_by_i.get(tr["i"])
                    if hit is None:
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
                        fc = line_intersect(pts[0], pts[2], pts[1], pts[3])
                        infos.append({"cls": 0, "pts": pts, "side": side, "fc": fc,
                                      "v": v, "h": tr["h"]})
                    else:
                        cx, cy = pts[4]
                        r = (((pts[5][0] - pts[6][0]) ** 2 + (pts[5][1] - pts[6][1]) ** 2) ** 0.5) / 2
                        img.draw_circle(int(cx), int(cy), int(r), color=(255, 0, 255), thickness=2)
                        infos.append({"cls": 1, "c": (cx, cy), "r": r, "v": v, "h": tr["h"]})

                boxes = [x for x in infos if x["cls"] == 0]
                circles = [x for x in infos if x["cls"] == 1]
                ratios = []
                used = [False] * len(circles)
                targets = []
                for bx in boxes:
                    xs = [p[0] for p in bx["pts"][:4]]
                    ys = [p[1] for p in bx["pts"][:4]]
                    host, hd = None, 1e9
                    for j, ci in enumerate(circles):
                        if used[j]:
                            continue
                        if min(xs) <= ci["c"][0] <= max(xs) and min(ys) <= ci["c"][1] <= max(ys):
                            d = abs(ci["c"][0] - bx["fc"][0]) + abs(ci["c"][1] - bx["fc"][1])
                            if d < hd:
                                host, hd = j, d
                    if host is None:
                        targets.append({"bull": bx["fc"], "src": "F"})
                        continue
                    ci = circles[host]
                    used[host] = True
                    rel = ci["r"] / (bx["side"] / 2) if bx["side"] > 1 else 0.0
                    ci["rel"] = rel
                    ratios.append(rel)
                    fc = bx["fc"]
                    bull = (CIRCLE_W * ci["c"][0] + (1.0 - CIRCLE_W) * fc[0],
                            CIRCLE_W * ci["c"][1] + (1.0 - CIRCLE_W) * fc[1])
                    targets.append({"bull": bull, "src": "B", "rel": rel})
                for j, ci in enumerate(circles):
                    if not used[j]:
                        targets.append({"bull": ci["c"], "src": "C"})

                targets_ready = True
            T_AI += time.ticks_diff(time.ticks_ms(), _t1)

            _t2 = time.ticks_ms()
            for t in targets:
                bx0, by0 = t["bull"]
                col = (255, 255, 0) if t["src"] == "B" else (
                    (0, 255, 255) if t["src"] == "F" else (255, 0, 255))
                img.draw_cross(int(bx0), int(by0), color=col, size=18, thickness=3)
                img.draw_string_advanced(int(bx0) + 11, int(by0) + 8, 20, t["src"], color=col)
                if t["src"] == "B":
                    img.draw_string_advanced(int(bx0) - 46, int(by0) - 30, 24,
                                             "%.2f" % t["rel"], color=(255, 255, 0))

            if len(ratios) >= 2:
                grp = cluster_groups(ratios)
                for ci, gi in zip([c for c in circles if "rel" in c], grp):
                    img.draw_string_advanced(int(ci["c"][0]) - 10,
                                             int(ci["c"][1]) + int(ci["r"]) + 10, 20,
                                             "G%d" % (gi + 1), color=(0, 255, 255))

            # 屏幕左上角状态行：帧率 / 目标数 / 会话状态
            try:
                img.draw_string_advanced(4, 4, 20,
                                         "F%.0f n%d %s" % (cur_fps, n_conf_total,
                                                           "ON" if server.connected_count() else "IDLE"),
                                         color=(0, 255, 0))
            except Exception:
                pass

            T_DRAW += time.ticks_diff(time.ticks_ms(), _t2)
            # 3) 画好的这帧转 YUV 后送编码器（推流画面因此带识别框）
            _t3 = time.ticks_ms()
            yuv = None
            fed = False
            try:
                yuv = csc.convert(fi, cvt=False)
            except Exception as e:
                if frames < 3:
                    print("csc 转换失败: %s" % e)
            if yuv is not None:
                # 按目标帧率节流送帧（编码器来一帧编一帧，挡住多余帧才能控制帧率）
                _now_f = time.ticks_ms()
                if frame_gap_ms <= 0 or time.ticks_diff(_now_f, last_feed_ms) >= feed_gap_ms:
                    last_feed_ms = _now_f
                    _tf = time.ticks_ms()
                    try:
                        enc.SendFrame(yuv)
                        fed = True
                    except Exception as e:
                        if frames < 3:
                            print("SendFrame 失败: %s" % e)
                    T_FEED += time.ticks_diff(time.ticks_ms(), _tf)

            T_CSC += time.ticks_diff(time.ticks_ms(), _t3)
            _t4 = time.ticks_ms()
            # 屏幕刷新比编码慢一档（13fps 足够看），把 CPU 让给编码和推流
            _now_s = time.ticks_ms()
            if time.ticks_diff(_now_s, last_show_ms) >= 80:
                last_show_ms = _now_s
                try:
                    Display.show_image(img)
                except Exception:
                    pass

            if yuv is not None:
                try:
                    csc.release_frame(yuv)
                except Exception:
                    pass
            try:
                fi.v_frame.release()
            except Exception:
                pass

            T_SHOW += time.ticks_diff(time.ticks_ms(), _t4)
            # 4) 取码流已挪到下一轮开头（见步骤 0）：这里只记账，下一轮去取
            if fed:
                pending_feed = True

            # 一键切换画质档（手机访问 /near 或 /far）
            _pf = server.requested_profile
            if _pf is not None and _pf != server.current_profile:
                server.requested_profile = None
                _p = PROFILES.get(_pf)
                if _p is not None:
                    print("画质档: %s (码率 %d, 帧率上限 %d, 检测每 %d 帧)" %
                          (_pf, _p["rate"], _p["fps"], _p["detect"]))
                    cur_rate = _p["rate"]
                    cur_fps_cap = _p["fps"]
                    frame_gap_ms = gap_for(cur_fps_cap)
                    feed_gap_ms = max(1, frame_gap_ms - 3) if frame_gap_ms else 0
                    DETECT_EVERY = _p["detect"]
                    server.current_profile = _pf
                    server.current_rate = cur_rate
                    enc = swap_encoder(enc, cur_rate, cur_fps_cap)
                    parameter_sets = None

            # 自适应码率：读 RSSI，越界就换档（换档需重建编码器，约 1 秒毛刺）
            if ADAPTIVE and link is not None and time.ticks_diff(time.ticks_ms(), t_rssi) >= 2000:
                t_rssi = time.ticks_ms()
                try:
                    r = link.status("rssi")
                except Exception:
                    r = None
                if isinstance(r, int):
                    # 指数平滑，抵消 RSSI 的瞬时抖动，避免码率反复横跳
                    cur_rssi = r if cur_rssi is None else (cur_rssi * 3 + r) // 4
                    server.current_rssi = cur_rssi
                    if cur_rssi < RSSI_LOW:
                        low_streak += 1
                        high_streak = 0
                    elif cur_rssi > RSSI_HIGH:
                        high_streak += 1
                        low_streak = 0
                    else:
                        low_streak = 0
                        high_streak = 0
                    if low_streak >= RSSI_LOW_N and cur_rate != LOW_RATE:
                        print("自适应: rssi=%d(平滑) 偏弱，码率 %d -> %d" % (cur_rssi, cur_rate, LOW_RATE))
                        cur_rate = LOW_RATE
                        enc = swap_encoder(enc, cur_rate, cur_fps_cap)
                        parameter_sets = None
                        low_streak = 0
                    elif high_streak >= RSSI_HIGH_N and cur_rate != BIT_RATE:
                        print("自适应: rssi=%d(平滑) 良好，码率 %d -> %d" % (cur_rssi, cur_rate, BIT_RATE))
                        cur_rate = BIT_RATE
                        enc = swap_encoder(enc, cur_rate, cur_fps_cap)
                        parameter_sets = None
                        high_streak = 0

            now = time.ticks_ms()
            _st = server.session_states()
            if _st != last_states:
                ws._slog("STATE %s -> %s" % (last_states, _st))
                last_states = _st
            if time.ticks_diff(now, t_stat) >= 1000:
                dt = time.ticks_diff(now, t_stat) / 1000.0
                cur_fps = frames / dt
                hb["fps"] = cur_fps
                print("统计 fps=%.1f 发送=%.0f 目标=%d 会话=%d 状态=%s H=%d I=%d rssi=%s/%d"
                      " | snap=%d img=%d ai=%d(/%d) draw=%d csc=%d feed=%d show=%d"
                      " get=%d send=%d ms" %
                      (cur_fps, sent_bytes * 8 / dt / 1000.0, n_conf_total,
                       server.connected_count(), server.session_states(), n_hdr, n_i,
                       cur_rssi, cur_rate,
                       T_SNAP, T_IMG, T_AI, gated, T_DRAW, T_CSC, T_FEED, T_SHOW,
                       T_GET, T_SEND))
                # 给 /status 用：不接串口也能远程读这些数
                server.stats = {
                    "fps": cur_fps, "kbps": sent_bytes * 8 / dt / 1000.0,
                    "sess": server.session_states(),
                    "detail": "snap=%d img=%d ai=%d/%d csc=%d feed=%d get=%d send=%d show=%d H%d I%d" %
                              (T_SNAP, T_IMG, T_AI, gated, T_CSC, T_FEED, T_GET, T_SEND,
                               T_SHOW, n_hdr, n_i),
                }
                T_SNAP = T_AI = T_DRAW = T_CSC = T_SHOW = T_GET = 0
                T_IMG = T_FEED = T_SEND = 0
                gated = 0
                t_stat = now
                sent_bytes = 0
                frames = 0
                n_hdr = 0
                n_i = 0

            # 帧节奏：把这一轮补齐到目标帧间隔。
            # 摄像头给 60fps，主循环干完活只要 ~21ms；不补的话「送帧节流」
            # 会落到「每隔一轮送一次」（24fps）。补到 33ms 之后每轮都送，
            # 稳定 30fps，而且 AI/显示/编码的节奏也均匀，不再忽快忽慢。
            if frame_gap_ms > 0:
                _left = frame_gap_ms - time.ticks_diff(time.ticks_ms(), _t_loop)
                if _left > 0:
                    time.sleep_ms(_left)
    except KeyboardInterrupt:
        pass
    finally:
        print("正在关闭…")
        wd_state["stop"] = True
        if wdt is not None:
            try:
                wdt.stop()
            except Exception:
                pass
        try:
            server.close()
        except Exception:
            pass
        try:
            sen.stop()
        except Exception:
            pass
        try:
            enc.Stop()
            enc.Destroy()
        except Exception:
            pass
        try:
            Display.deinit()
        except Exception:
            pass
        print("已停止")


if __name__ == "__main__":
    os.exitpoint(os.EXITPOINT_ENABLE)
    run()
