"""通过串口把本地文件上传到 K230（base64 分块，绕开 REPL 多行与转义问题）。

用法：
    python tools/k230_upload.py COM5 <本地文件> <板上路径>
"""
import base64
import os
import sys
import time

import serial


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM5"
    local = sys.argv[2]
    remote = sys.argv[3]
    chunk_chars = int(sys.argv[4]) if len(sys.argv) > 4 else 512

    data = open(local, "rb").read()
    b64 = base64.b64encode(data).decode()

    s = serial.Serial(port, 115200, timeout=0.2)
    time.sleep(0.4)

    # 必须先中断正在运行的脚本并确认回到 REPL，否则写入命令会被忽略
    at_repl = False
    for _ in range(15):
        s.write(b"\x03")
        time.sleep(0.8)
        if ">>>" in s.read_all().decode("utf-8", "replace"):
            at_repl = True
            break
    if not at_repl:
        raise SystemExit("无法让板子回到 REPL，先停下正在运行的脚本再试")
    print("REPL 就绪")

    def send(line, wait=0.02):
        s.write(line.encode() + b"\r\n")
        time.sleep(wait)
        s.read_all()

    # 115200 8N1 ≈ 11520 字符/秒；留 1.8 倍余量，避免输入缓冲被冲爆
    def tx_wait(line):
        return len(line) / 11520.0 * 1.8 + 0.01

    send("import ubinascii; f=open('%s','wb')" % remote, tx_wait("x" * 60))

    total = len(b64)
    n = 0
    t0 = time.time()
    for i in range(0, total, chunk_chars):
        piece = b64[i:i + chunk_chars]
        line = "f.write(ubinascii.a2b_base64('%s'))" % piece
        send(line, tx_wait(line))
        n += 1
        if n % 20 == 0:
            print("  %d/%d chunks  %.0f%%" % (n, (total + chunk_chars - 1) // chunk_chars,
                                              100.0 * min(i + chunk_chars, total) / total))
    send("f.close()", 0.4)

    # 校验大小
    s.write(("print('SIZE', __import__('os').stat('%s')[6])" % remote).encode() + b"\r\n")
    time.sleep(1.5)
    out = s.read_all().decode("utf-8", "replace")
    s.close()

    print(out[-400:])
    print("local=%d bytes  chunks=%d  %.1fs" % (len(data), n, time.time() - t0))


if __name__ == "__main__":
    main()
