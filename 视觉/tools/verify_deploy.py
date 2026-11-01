# -*- coding: utf-8 -*-
"""核对部署到 SD 卡上的脚本/配置是不是和仓库里一致、编码有没有被破坏。

用法：
    py -3 tools/verify_deploy.py F:            # F: 是卡上的 SDCARD 分区

为什么需要它：这块板子的 `/sdcard` 在 Windows 上就是一个 FAT 分区，直接拷文件最可靠，
但**中文 SSID 很容易被编辑器/PowerShell 管道转码，或者被加上 UTF-8 BOM** ——
一旦坏掉，`load_config()` 解析失败会退到内置默认值（`mode=sta` + 空 `ssid`），
板子连不上热点也不开 AP，只能拔卡重来。这个脚本就是部署后的最后一道检查。
"""
import hashlib
import json
import os
import re
import sys

REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "k230")

# 仓库文件 -> 卡上文件名（板端优先读 webrtc_config2.json）
PAIRS = [
    ("pose_webrtc.py", "pose_webrtc.py"),
    ("webrtc_stream.py", "webrtc_stream.py"),
    ("webrtc_config.json", "webrtc_config2.json"),
]


def check(card, pairs=PAIRS):
    ok = True
    for src, dst in pairs:
        with open(os.path.join(REPO, src), "rb") as f:
            a = f.read()
        path = os.path.join(card, dst)
        if not os.path.exists(path):
            print("%-20s -> %-20s MISSING" % (src, dst))
            ok = False
            continue
        with open(path, "rb") as f:
            b = f.read()
        same = a == b
        ok &= same
        print("%-20s -> %-20s %s (%d vs %d bytes)" %
              (src, dst, "SAME" if same else "DIFF", len(a), len(b)))

    raw = open(os.path.join(card, "webrtc_config2.json"), "rb").read()
    m = re.search(rb'"ssid"\s*:\s*"([^"]*)"', raw)
    if m:
        try:
            print("ssid          : %s" % m.group(1).decode("utf-8"))
        except UnicodeDecodeError as e:
            print("ssid          : 不是合法 UTF-8（%s）" % e)
            ok = False
    print("utf-8 BOM     : %s" % (raw[:3] == b"\xef\xbb\xbf"))
    try:
        cfg = json.loads(raw.decode("utf-8").lstrip("\ufeff"))
        print("json ok       : mode=%s sensor_fps=%s fps=%s gop=%s detect_every=%s" %
              (cfg.get("mode"), cfg.get("sensor_fps"), cfg.get("fps"),
               cfg.get("gop"), cfg.get("detect_every")))
    except Exception as e:
        print("json FAIL     : %s" % e)
        ok = False
    return ok


if __name__ == "__main__":
    card = sys.argv[1] if len(sys.argv) > 1 else "F:"
    sys.exit(0 if check(card) else 1)
