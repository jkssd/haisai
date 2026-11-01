"""K230 REPL 串口客户端（备用通道，不依赖 VSCode/CanMV 扩展）。

用法：
    python tools/k230_serial.py exec <本地py文件>      # 在板上执行脚本
    python tools/k230_serial.py cmd "print(1+1)"        # 执行一行/一段代码并返回输出
    python tools/k230_serial.py ls /sdcard              # 列目录
"""
from __future__ import annotations

import serial  # pyserial
import sys
import time
import re

PORT = "COM14"
BAUD = 115200


class Board:
    def __init__(self, port=PORT, baud=BAUD):
        self.s = serial.Serial(port, baud, timeout=0.2)
        time.sleep(0.3)
        self._drain()
        # 进 REPL
        self._interrupt()

    def _drain(self):
        try:
            self.s.reset_input_buffer()
        except Exception:
            pass
        self.s.read_all()

    def _interrupt(self):
        for _ in range(3):
            self.s.write(b"\x03")
            time.sleep(0.15)
        self.s.write(b"\r\n")
        time.sleep(0.3)
        self.s.read_all()

    def read_until(self, timeout=8.0, tail=b">>> "):
        buf = b""
        t0 = time.time()
        while time.time() - t0 < timeout:
            chunk = self.s.read(4096)
            if chunk:
                buf += chunk
                if tail and buf.rstrip().endswith(tail.strip()):
                    break
            else:
                time.sleep(0.02)
                if buf and buf.endswith(b">>> "):
                    break
        return buf.decode("utf-8", "replace")

    def send_and_read(self, code: str, wait=3.0) -> str:
        """把代码包进 exec(...) 一次性执行，收集返回的 traceback 缓冲。"""
        payload = code.replace("\x03", "").replace("\x04", "")
        self.s.write(payload.encode("utf-8") + b"\r\n")
        time.sleep(wait)
        return self.read_until(timeout=5.0)

    def run_file(self, src: str, wait_head=2.0, max_wait=60.0) -> str:
        """用 exec(open(path).read()) 在板上运行文件，等执行结束。"""
        # 写一个小启动器到板上：先切 REPL，再 exec 文件
        path = "/sdcard/_bk_run.py"
        self.write_file(path, src)
        self.s.write(("exec(open('%s').read())\r\n" % path).encode())
        buf = b""
        t0 = time.time()
        marker = b"__BK_DONE__"
        while time.time() - t0 < max_wait:
            chunk = self.s.read(4096)
            if chunk:
                buf += chunk
                if marker in buf:
                    break
            else:
                time.sleep(0.05)
        return buf.decode("utf-8", "replace")

    def write_file(self, remote: str, content: str):
        """分块写入板子文件（base64 规避转义），再用 python 解码。"""
        import base64
        b64 = base64.b64encode(content.encode("utf-8")).decode()
        CH = 1024
        chunks = [b64[i:i + CH] for i in range(0, len(b64), CH)]
        # 第一块 reset
        self._send_quiet("f=open('%s','w')" % remote)
        for i, c in enumerate(chunks):
            self._send_quiet("f.write(__import__('ubinascii').a2b_base64('%s'))" % c)
        self._send_quiet("f.close()")

    def _send_quiet(self, code: str, wait=0.35):
        self.s.write(code.encode("utf-8") + b"\r\n")
        time.sleep(wait)
        self.s.read_all()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    b = Board()
    if cmd == "cmd":
        code = sys.argv[2]
        print(b.send_and_read(code, wait=float(sys.argv[3]) if len(sys.argv) > 3 else 3.0))
    elif cmd == "exec":
        src = open(sys.argv[2], "r", encoding="utf-8").read()
        print(b.run_file(src, max_wait=float(sys.argv[3]) if len(sys.argv) > 3 else 120.0))
    else:
        print(b.send_and_read("print('hello from k230')", wait=2.0))


if __name__ == "__main__":
    main()
