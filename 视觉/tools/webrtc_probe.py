# -*- coding: utf-8 -*-
"""K230 无线图传可靠性压测（PC 端，无第三方依赖）。

做法：
  1. 起一个 headless Chrome（开远程调试）；
  2. 用极简 WebSocket 客户端走 CDP，打开板子的图传页面并点「连接」；
  3. 每秒采样页面里 window.__stats（fps / 码率 / 丢包 / RTT / 抖动）；
  4. 输出统计报告 + 原始样本 JSON。

用法：
  python tools/webrtc_probe.py --url http://192.168.43.188:8080/ --seconds 60
  python tools/webrtc_probe.py --url http://192.168.169.1:8080/ --seconds 30 --chrome "C:/Program Files/Google/Chrome/Application/chrome.exe"
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


# ---------------------------------------------------------------- WebSocket ---
class WS:
    """够用就好的 WebSocket 客户端（只支持 CDP 需要的场景）。"""

    def __init__(self, url, timeout=10.0):
        assert url.startswith("ws://")
        rest = url[5:]
        hostport, _, path = rest.partition("/")
        path = "/" + path
        host, _, port = hostport.partition(":")
        port = int(port or 80)
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        key = base64.b64encode(bytes(random.getrandbits(8) for _ in range(16))).decode()
        req = (
            "GET %s HTTP/1.1\r\nHost: %s:%d\r\nUpgrade: websocket\r\n"
            "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n" % (path, host, port, key)
        )
        self.sock.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise IOError("websocket handshake closed")
            buf += chunk
        head, _, self.rest = buf.partition(b"\r\n\r\n")
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise IOError("websocket handshake failed: %r" % head[:200])
        self.msg_id = 0

    def _send_frame(self, opcode, payload):
        header = bytearray()
        header.append(0x80 | opcode)
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        mask = bytes(random.getrandbits(8) for _ in range(4))
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def _recv_exact(self, n):
        out = b""
        while len(out) < n:
            if self.rest:
                take = self.rest[:n - len(out)]
                self.rest = self.rest[len(take):]
                out += take
                continue
            chunk = self.sock.recv(n - len(out))
            if not chunk:
                raise IOError("socket closed")
            out += chunk
        return out

    def recv_message(self):
        data = b""
        while True:
            b0, b1 = self._recv_exact(2)
            fin = b0 & 0x80
            opcode = b0 & 0x0F
            masked = b1 & 0x80
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._recv_exact(8))[0]
            if masked:
                mask = self._recv_exact(4)
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(self._recv_exact(length)))
            else:
                payload = self._recv_exact(length)
            if opcode == 0x8:
                raise IOError("websocket closed by peer")
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            data += payload
            if fin:
                return data.decode("utf-8", "replace")

    def call(self, method, params=None, timeout=15.0):
        self.msg_id += 1
        mid = self.msg_id
        self._send_frame(0x1, json.dumps({"id": mid, "method": method,
                                          "params": params or {}}).encode())
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = json.loads(self.recv_message())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError("CDP %s -> %s" % (method, msg["error"]))
                return msg.get("result", {})
        raise TimeoutError("CDP %s timed out" % method)

    def close(self):
        try:
            self._send_frame(0x8, b"")
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


# ------------------------------------------------------------------- Chrome ---
def find_chrome(explicit=None):
    if explicit and os.path.exists(explicit):
        return explicit
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    raise SystemExit("找不到 Chrome/Edge，请用 --chrome 指定")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_devtools(port, timeout=25.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/json/list" % port, timeout=1) as r:
                targets = json.load(r)
            for t in targets:
                if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                    return t["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.3)
    raise SystemExit("Chrome DevTools 没起来")


def js_eval(ws, expr):
    res = ws.call("Runtime.evaluate", {"expression": expr, "returnByValue": True,
                                       "awaitPromise": False})
    return res.get("result", {}).get("value")


# -------------------------------------------------------------------- report ---
def summarise(samples, connect_ok):
    if not samples:
        return {"ok": False, "reason": "没有采到样本"}
    kbps = [s.get("kbps", 0) for s in samples if s.get("kbps")]
    fps = [s.get("fps", 0.0) for s in samples if s.get("fps")]
    rtt = [s.get("rtt", -1) for s in samples if s.get("rtt", -1) > 0]
    jit = [s.get("jitter", -1) for s in samples if s.get("jitter", -1) >= 0]
    lost = [s.get("lost", 0) for s in samples]
    frames = [s.get("frames", 0) for s in samples]
    stall = sum(1 for s in samples if (s.get("fps") or 0) <= 0.5)
    return {
        "ok": bool(connect_ok),
        "samples": len(samples),
        "resolution": samples[-1].get("size"),
        "first_frame_ms": samples[-1].get("first"),
        "fps_median": round(statistics.median(fps), 1) if fps else 0,
        "fps_min": min(fps) if fps else 0,
        "fps_max": max(fps) if fps else 0,
        "kbps_median": int(statistics.median(kbps)) if kbps else 0,
        "kbps_min": min(kbps) if kbps else 0,
        "kbps_max": max(kbps) if kbps else 0,
        "frames_decoded": (max(frames) - min(frames)) if len(frames) > 1 else 0,
        "packets_lost": (max(lost) - min(lost)) if len(lost) > 1 else 0,
        "rtt_ms_median": int(statistics.median(rtt)) if rtt else -1,
        "rtt_ms_max": max(rtt) if rtt else -1,
        "jitter_ms_median": int(statistics.median(jit)) if jit else -1,
        "stalled_samples": stall,
        "stall_ratio": round(stall / float(len(samples)), 3),
        "states": sorted(set(s.get("state", "?") for s in samples)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="板子图传页面，例如 http://192.168.43.188:8080/")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--chrome", default=None)
    ap.add_argument("--out", default=None, help="把原始样本写到这个 json")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    chrome = find_chrome(args.chrome)
    port = free_port()
    profile = tempfile.mkdtemp(prefix="k230probe_")
    cmd = [
        chrome,
        "--headless=new",
        "--remote-debugging-port=%d" % port,
        "--user-data-dir=%s" % profile,
        "--no-first-run",
        "--no-default-browser-check",
        "--autoplay-policy=no-user-gesture-required",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
        "--window-size=%d,%d" % (args.width, args.height),
        # 关掉 mDNS 候选混淆，让浏览器直接给出真实局域网 IP（K230 侧好替换/路由）
        "--disable-features=WebRtcHideLocalIpsWithMdns,WebRtcAllowInputVolumeAdjustment",
        "about:blank",
    ]
    print("启动浏览器: %s" % os.path.basename(chrome))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ws = None
    try:
        ws_url = wait_devtools(port)
        ws = WS(ws_url, timeout=30.0)
        # 用 location.href 跳转，避开 Page.navigate 在 headless 下卡住的问题
        print("打开页面: %s" % args.url)
        js_eval(ws, "location.href=%s" % json.dumps(args.url))
        time.sleep(5.0)

        title = js_eval(ws, "document.title")
        print("页面标题: %s" % title)
        if not title:
            print("页面没加载出来")

        clicked = js_eval(ws, """(()=>{const b=document.getElementById('go');
            if(!b)return 'no-button';b.click();return 'clicked'})()""")
        print("触发连接: %s" % clicked)

        samples = []
        connect_ok = False
        t_end = time.time() + args.seconds
        while time.time() < t_end:
            raw = js_eval(ws, "window.__stats?JSON.stringify(window.__stats):''")
            if raw:
                try:
                    s = json.loads(raw)
                    samples.append(s)
                    if s.get("state") == "connected":
                        connect_ok = True
                except Exception:
                    pass
            time.sleep(1.0)

        report = summarise(samples, connect_ok)
        print("\n===== 结果 =====")
        for k, v in report.items():
            print("%-16s %s" % (k, v))

        if args.out:
            payload = {"url": args.url, "report": report, "samples": samples}
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
            print("\n样本已写入 %s" % args.out)
        return 0 if report.get("ok") else 2
    finally:
        if ws is not None:
            ws.close()
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
