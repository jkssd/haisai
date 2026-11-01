# -*- coding: utf-8 -*-
"""跟板上 MicroPython REPL 说话的小工具（COM5 / 115200）。

用法：
  py -3 tools/k230_repl.py -c "print(os.listdir('/sdcard'))"
  py -3 tools/k230_repl.py -f some_board_snippet.py
  py -3 tools/k230_repl.py -f snippet.py --no-int      # 不打断当前脚本
  py -3 tools/k230_repl.py --raw "..."                 # 原样发字节

实现要点：
  * 先用 Ctrl-C 把正在跑的脚本停到 REPL（脚本里有 os.exitpoint，能干净退出）；
  * 多行代码走 **paste mode**（Ctrl-E ... Ctrl-D），避免逐行缩进被吃掉；
  * 读完输出再关串口，超时就打印已有内容。
"""
import argparse
import sys
import time

import serial

PORT = "COM5"
BAUD = 115200


def read_all(s, quiet=0.35, total=6.0):
    out = b""
    t0 = time.time()
    last = time.time()
    while time.time() - t0 < total:
        n = s.in_waiting
        if n:
            out += s.read(n)
            last = time.time()
        else:
            if out and (time.time() - last) > quiet:
                break
            time.sleep(0.03)
    return out.decode("utf-8", "replace")


def open_repl(s, interrupt=True, tries=12):
    s.reset_input_buffer()
    if not interrupt:
        return read_all(s, quiet=0.5, total=1.0)
    buf = ""
    for _ in range(tries):
        s.write(b"\x03")
        time.sleep(0.5)
        buf += read_all(s, quiet=0.25, total=0.9)
        if ">>>" in buf:
            s.write(b"\r\n")
            time.sleep(0.2)
            buf += read_all(s, quiet=0.2, total=0.6)
            return buf
    raise SystemExit("进不了 REPL，最后输出：\n" + buf[-1500:])


def paste(s, code):
    s.write(b"\x05")  # Ctrl-E 进入粘贴模式
    time.sleep(0.35)
    read_all(s, quiet=0.2, total=0.5)
    for line in code.replace("\r\n", "\n").split("\n"):
        s.write(line.encode("utf-8") + b"\r\n")
        time.sleep(0.012)
    time.sleep(0.25)
    s.write(b"\x04")  # Ctrl-D 执行
    time.sleep(0.15)
    return read_all(s, quiet=0.6, total=25.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--code", default=None)
    ap.add_argument("-f", "--file", default=None)
    ap.add_argument("--no-int", action="store_true")
    ap.add_argument("--raw", default=None)
    args = ap.parse_args()

    s = serial.Serial(PORT, BAUD, timeout=0.3)
    time.sleep(0.3)
    try:
        banner = open_repl(s, interrupt=not args.no_int)
        if banner.strip():
            sys.stderr.write("--- 进入 REPL ---\n" + banner[-800:] + "\n")
        if args.raw:
            s.write(args.raw.encode("utf-8"))
            time.sleep(0.2)
            print(read_all(s, quiet=0.4, total=8.0))
            return 0
        code = args.code
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                code = f.read()
        if not code:
            return 0
        print(paste(s, code))
        return 0
    finally:
        s.close()


if __name__ == "__main__":
    sys.exit(main())
