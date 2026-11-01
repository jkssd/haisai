# -*- coding: utf-8 -*-
"""K230 开机入口（替换 Yahboom 出厂 main.py，去掉按键检测）。

出厂 main.py 会检测 YbKey 是否按下，偶尔误判导致 abort_main=1，图传就不启动。
这里直接执行目标脚本，不依赖任何按键状态。

禁用自启：在 /sdcard 建一个空的 noapp 文件。
"""
import os
import sys

sys.path.insert(0, "/sdcard")

OFF_FLAG = "/sdcard/noapp"
CANDIDATES = ("/sdcard/pose_webrtc.py", "/sdcard/webrtc_stream.py")


def _exists(p):
    try:
        os.stat(p)
        return True
    except Exception:
        return False


if _exists(OFF_FLAG):
    print("[boot] noapp 存在，停在 REPL")
else:
    target = None
    for c in CANDIDATES:
        if _exists(c):
            target = c
            break

    if target is None:
        print("[boot] 找不到要运行的脚本")
    else:
        print("[boot] 启动 %s" % target)
        src = None
        f = None
        try:
            f = open(target, "r")
            src = f.read()
        finally:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        try:
            exec(src, {"__name__": "__main__"})
        except KeyboardInterrupt:
            print("[boot] 已手动停止，REPL 接管")
        except Exception as e:
            print("[boot] 启动失败:", e)
            try:
                sys.print_exception(e)
            except Exception:
                pass
