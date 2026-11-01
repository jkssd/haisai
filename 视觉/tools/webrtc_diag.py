# -*- coding: utf-8 -*-
"""K230 图传「花屏 / 卡顿」定位工具（PC 端，无第三方依赖）。

比 webrtc_probe.py 多抓一层：把 getStats() 里 inbound-rtp(video) 的
**解码侧**字段全打出来 —— 只有这些字段能区分下面三种完全不同的病因：

  A. 网络丢包          packetsLost > 0 / nackCount > 0 / pliCount 涨
  B. 帧到了但被丢掉    framesDropped > 0，且 packetsLost == 0   -> 抖动缓冲/迟到
  C. 位流本身坏了      framesDecoded 正常但 freezeCount 涨 / keyFramesDecoded 异常
  D. 纯画质差          framesDecoded 正常、无丢包、码率却很低   -> 量化太重（看着像花屏）

用法：
  python tools/webrtc_diag.py --url http://192.168.43.188:8080/ --seconds 25
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from webrtc_probe import WS, find_chrome, free_port, js_eval, wait_devtools  # noqa: E402

# 在页面里跑：把 video 的 inbound-rtp 全字段 + candidate-pair 抓出来
JS = r"""
(async () => {
  if (!window.__pc) window.__pc = (typeof pc !== 'undefined') ? pc : null;
  const p = window.__pc;
  if (!p) return JSON.stringify({err: 'no-pc'});
  let rep; try { rep = await p.getStats(); } catch (e) { return JSON.stringify({err: String(e)}); }
  const out = {video: null, pair: null, remote: null, ts: Date.now()};
  let pid = null;
  rep.forEach(r => {
    if (r.type === 'inbound-rtp' && r.kind === 'video') out.video = r;
    if (r.type === 'transport' && r.selectedCandidatePairId) pid = r.selectedCandidatePairId;
    if (r.type === 'remote-inbound-rtp' && r.kind === 'video') out.remote = r;
  });
  if (pid) out.pair = rep.get(pid);
  if (!out.pair) rep.forEach(r => { if (!out.pair && r.type === 'candidate-pair' && r.state === 'succeeded') out.pair = r; });
  const v = document.getElementById('v');
  out.w = v ? v.videoWidth : 0; out.h = v ? v.videoHeight : 0;
  // 白屏/黑屏检测：把当前帧画到 canvas 上采样，粗判是否整片异常（花屏时熵会明显变低）
  try {
    const c = document.createElement('canvas'); c.width = 32; c.height = 24;
    const g = c.getContext('2d'); g.drawImage(v, 0, 0, 32, 24);
    const d = g.getImageData(0, 0, 32, 24).data;
    let sum = 0, sum2 = 0, n = 0;
    for (let i = 0; i < d.length; i += 4) { const y = 0.299*d[i]+0.587*d[i+1]+0.114*d[i+2]; sum += y; sum2 += y*y; n++; }
    const mean = sum/n; out.luma = Math.round(mean); out.luma_std = Math.round(Math.sqrt(Math.max(0, sum2/n - mean*mean)));
  } catch (e) {}
  return JSON.stringify(out);
})()
"""

FIELDS = [
    ("packetsReceived", "收包"), ("packetsLost", "丢包"), ("bytesReceived", "字节"),
    ("framesReceived", "收帧"), ("framesDecoded", "解码"), ("framesDropped", "丢帧"),
    ("keyFramesDecoded", "关键帧"), ("pliCount", "PLI请求"), ("nackCount", "NACK"),
    ("firCount", "FIR请求"), ("freezeCount", "冻结次数"), ("totalFreezesDuration", "冻结总时长"),
    ("jitter", "抖动"), ("jitterBufferDelay", "缓冲延迟"), ("jitterBufferEmittedCount", "缓冲出帧"),
    ("totalDecodeTime", "解码耗时"), ("totalProcessingDelay", "处理延迟"),
    ("totalInterFrameDelay", "帧间隔累计"), ("totalSquaredInterFrameDelay", "帧间隔平方"),
    ("headerBytesReceived", "头部字节"), ("lastPacketReceivedTimestamp", "末包时间"),
]


def js_await(ws, expr):
    """Runtime.evaluate 异步表达式：必须 awaitPromise，否则拿到的是 Promise 对象。"""
    res = ws.call("Runtime.evaluate", {"expression": expr, "returnByValue": True,
                                       "awaitPromise": True})
    return res.get("result", {}).get("value")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--seconds", type=int, default=25)
    ap.add_argument("--chrome", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-pc", action="store_true", help="不新建 RTCPeerConnection，复用页面里的 pc")
    args = ap.parse_args()

    chrome = find_chrome(args.chrome)
    port = free_port()
    profile = tempfile.mkdtemp(prefix="k230diag_")
    cmd = [chrome, "--headless=new", "--remote-debugging-port=%d" % port,
           "--user-data-dir=%s" % profile, "--no-first-run", "--no-default-browser-check",
           "--autoplay-policy=no-user-gesture-required",
           "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
           "--disable-backgrounding-occluded-windows", "--window-size=1280,720",
           "--disable-features=WebRtcHideLocalIpsWithMdns", "about:blank"]
    proc = None
    ws = None
    rows = []
    t_start = None
    first_frame_at = None
    try:
        proc = __import__("subprocess").Popen(cmd, stdout=__import__("subprocess").DEVNULL,
                                              stderr=__import__("subprocess").DEVNULL)
        ws = WS(wait_devtools(port), timeout=30.0)
        js_eval(ws, "location.href=%s" % json.dumps(args.url))
        time.sleep(4.0)
        t_start = time.time()
        print("点击连接…")
        print(js_eval(ws, "(()=>{const b=document.getElementById('go');if(!b)return 'no-button';b.click();return 'ok'})()"))
        # 页面脚本里 pc 是 let 绑定，同一全局词法环境可见；存一份到 window 供后续复用
        js_eval(ws, "window.__hook=setInterval(()=>{try{if(typeof pc!=='undefined'&&pc){window.__pc=pc}}catch(e){}} ,200);0")

        deadline = time.time() + args.seconds
        while time.time() < deadline:
            raw = js_await(ws, JS)
            try:
                s = json.loads(raw) if raw else None
            except Exception:
                s = None
            if s and s.get("video"):
                rows.append(s)
                v = s["video"]
                if first_frame_at is None and v.get("framesDecoded", 0) > 0:
                    first_frame_at = time.time() - t_start
            time.sleep(1.0)
            sys.stdout.write(".")
            sys.stdout.flush()
        print()

        if not rows:
            print("没采到样本（页面没连上？）")
            return 2

        def d(k):
            a, b = rows[0]["video"], rows[-1]["video"]
            return (b.get(k, 0) or 0) - (a.get(k, 0) or 0)

        span = rows[-1]["ts"] - rows[0]["ts"]
        span_s = max(0.001, span / 1000.0)
        v0, v1 = rows[0]["video"], rows[-1]["video"]
        pair = rows[-1].get("pair") or {}
        print("\n===== 接收端解码统计（%.1f 秒） =====" % span_s)
        print("分辨率        %sx%s" % (rows[-1].get("w"), rows[-1].get("h")))
        print("首帧耗时      %.0f ms" % (first_frame_at * 1000 if first_frame_at else -1))
        print("接收帧率      %.1f fps   (解码 %d 帧)" % (d("framesDecoded") / span_s, d("framesDecoded")))
        print("接收码率      %d kbit/s" % (d("bytesReceived") * 8 / span_s / 1000))
        print("RTP 丢包      %d" % d("packetsLost"))
        print("NACK/PLI/FIR  %d / %d / %d" % (d("nackCount"), d("pliCount"), d("firCount")))
        print("解码丢帧      %d   冻结 %.2fs / %d 次" %
              (d("framesDropped"), d("totalFreezesDuration"), d("freezeCount")))
        print("关键帧解码    %d  (%.2f 个/秒)" % (d("keyFramesDecoded"), d("keyFramesDecoded") / span_s))
        jb = (v1.get("jitterBufferDelay", 0) - v0.get("jitterBufferDelay", 0))
        jbe = (v1.get("jitterBufferEmittedCount", 0) - v0.get("jitterBufferEmittedCount", 0))
        print("抖动缓冲      %.0f ms/帧 (%d 帧出缓冲)" % (1000 * jb / jbe if jbe else -1, jbe))
        fi = d("totalInterFrameDelay") / max(1, d("framesDecoded"))
        print("平均帧间隔    %.1f ms  (理论 %.1f ms)" % (fi * 1000, 1000.0 / max(0.01, d("framesDecoded") / span_s)))
        print("RTT / 可用带宽 %s ms / %s kbit/s" % (
            round(pair.get("currentRoundTripTime", -1) * 1000) if pair else "?",
            round(pair.get("availableOutgoingBitrate", 0) / 1000) if pair else "?"))
        print("累计：收包 %d / 丢 %d (%.2f%%)" % (
            v1.get("packetsReceived", 0), v1.get("packetsLost", 0),
            100.0 * (v1.get("packetsLost", 0) or 0) / max(1, (v1.get("packetsReceived", 0) or 0))))
        if rows[-1].get("luma") is not None:
            print("画面亮度均值/标准差  %s / %s（整片灰绿/黑时标准差会很低）" %
                  (rows[-1]["luma"], rows[-1]["luma_std"]))

        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=1)
            print("原始样本 -> %s" % args.out)
        return 0
    finally:
        if ws is not None:
            ws.close()
        if proc is not None:
            try:
                proc.terminate(); proc.wait(timeout=5)
            except Exception:
                proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
