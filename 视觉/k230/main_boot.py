# -*- coding: utf-8 -*-
"""开机自启：直接进 TFT 靶心识别。

保留调试出口：
    * 板上存在 /sdcard/offline_off  -> 不启动识别，REPL 空闲（IDE / MCP 直接接管）
    * 运行中在 IDE 里按停止 / 发 Ctrl-C -> 脚本干净退出，回到 REPL

恢复自启：删掉 /sdcard/offline_off 后复位即可。
"""
import os
import sys

sys.path.insert(0, "/sdcard")

OFF_FLAG = "/sdcard/offline_off"
SCRIPT = "/sdcard/tft_live.py"


def _flag_exists():
    try:
        os.stat(OFF_FLAG)
        return True
    except Exception:
        return False


if _flag_exists():
    print("[boot] offline_off 存在：跳过自启，REPL 可用")
else:
    print("[boot] 启动 TFT 靶心识别…")
    try:
        src = open(SCRIPT).read()
        exec(src, {"__name__": "__main__"})
    except KeyboardInterrupt:
        print("[boot] 已被停止，REPL 接管（复位可恢复自启）")
    except Exception as e:
        msg = str(e)
        if "interrupt" in msg.lower():
            # IDE / MCP 一连接就会发 Ctrl-C，这是正常的调试入口，不是故障
            print("[boot] 已被 IDE 中断，REPL 接管（复位可恢复自启）")
        else:
            print("[boot] 启动失败:", type(e).__name__, msg)
            print("[boot] 可写 /sdcard/offline_off 后复位，改为纯手动调试")
