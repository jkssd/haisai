"""通过串口把本地文件上传到 K230（base64 分块，绕开 REPL 多行与转义问题）。

用法：
    python tools/k230_upload.py COM5 <本地文件> <板上路径> [每块字符数]

踩过的坑（都已修）：
  1. 每块 512 字符会把 REPL 输入缓冲冲爆 -> 中途丢字符、a2b_base64 报错、文件截断。
     实测 200 字符稳定，所以默认 200。
  2. 上一轮中断会在 REPL 里留下没关的 f 句柄，再 open 同一个文件会
     OSError [Errno 1] EPERM。写之前先 gc.collect() 释放。
  3. 板子刚 Ctrl-C 完还在打印收尾信息，这时发命令会被吞。
     必须先握手到「连续两次都回 >>>」再动手。
"""
import base64
import os
import sys
import time

import serial

CHUNK_DEFAULT = 200


def drain(s, secs, quiet=0.22):
    buf = b""
    t0 = time.time()
    last = time.time()
    while time.time() - t0 < secs:
        n = s.in_waiting
        if n:
            buf += s.read(n)
            last = time.time()
        else:
            if buf and (time.time() - last) > quiet:
                break
            time.sleep(0.03)
    return buf


def cmd(s, text, wait=0.8):
    s.write(text.encode() + b"\r\n")
    return drain(s, wait).decode("utf-8", "replace")


def ensure_repl(s, tries=20):
    """握手到真正干净的 REPL：连续两次 Ctrl-C 都立刻回 >>>，且空回车也回 >>>。"""
    hits = 0
    for _ in range(tries):
        s.write(b"\x03")
        t = drain(s, 1.4).decode("utf-8", "replace")
        hits = hits + 1 if ">>>" in t else 0
        if hits >= 2:
            s.write(b"\r\n")
            t2 = drain(s, 1.0).decode("utf-8", "replace")
            if ">>>" in t2:
                return True
            hits = 0
    return False


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM5"
    local = sys.argv[2]
    remote = sys.argv[3]
    chunk = int(sys.argv[4]) if len(sys.argv) > 4 else CHUNK_DEFAULT

    data = open(local, "rb").read()
    b64 = base64.b64encode(data).decode()

    s = serial.Serial(port, 115200, timeout=0.25)
    time.sleep(0.4)
    if not ensure_repl(s):
        raise SystemExit("无法让板子回到干净的 REPL（先确认没有脚本在跑）")
    print("REPL 就绪")

    # gc 释放历史遗留的 f 句柄，否则 open 会 EPERM
    cmd(s, "import gc, ubinascii, os; gc.collect()", 1.2)
    out = cmd(s, "f=open('%s','wb'); print('OPENOK')" % remote, 1.0)
    if "OPENOK" not in out:
        raise SystemExit("板上 open 失败（可能文件被占用）：\n" + out[-400:])

    total = len(b64)
    i = 0
    n = 0
    t0 = time.time()
    while i < total:
        piece = b64[i:i + chunk]
        s.write(("f.write(ubinascii.a2b_base64('%s'))" % piece).encode() + b"\r\n")
        time.sleep(0.026)
        out = drain(s, 0.04, quiet=0.02).decode("utf-8", "replace")
        if "Traceback" in out:
            raise SystemExit("写入第 %d 字节出错：\n%s" % (i, out[-400:]))
        i += chunk
        n += 1
        if n % 50 == 0:
            print("  %d%%" % (100 * i // total))

    cmd(s, "f.close()", 0.6)
    out = cmd(s, "print('SZ', os.stat('%s')[6])" % remote, 1.6)
    got = None
    for line in out.replace("\r", "").split("\n"):
        if line.strip().startswith("SZ "):
            try:
                got = int(line.split()[1])
            except Exception:
                pass
    s.close()

    print("local=%d  remote=%s  翻块=%d  %.1fs"
          % (len(data), got, n, time.time() - t0))
    if got != len(data):
        raise SystemExit("大小不一致，上传失败")
    print("OK")


if __name__ == "__main__":
    main()
