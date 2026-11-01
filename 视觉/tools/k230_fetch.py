"""从板上按偏移分块拉取文件（base64），用法：fetch.py <remote> <local>"""
import base64
import sys
import time

import serial

remote = sys.argv[1]
local = sys.argv[2]

s = serial.Serial("COM5", 115200, timeout=0.5)
time.sleep(0.5)
for i in range(14):
    s.write(b"\x03")
    time.sleep(0.8)
    if ">>>" in s.read_all().decode("utf-8", "replace"):
        print("REPL ok")
        break
s.read_all()

s.write(b"import os, ubinascii\r\n")
time.sleep(0.4)
s.read_all()
s.write(("print('SIZE=', os.stat('%s')[6])\r\n" % remote).encode())
time.sleep(1.2)
out = s.read_all().decode("utf-8", "replace")
size = 0
for l in out.split("\n"):
    st = l.strip()
    if st.startswith("SIZE=") and st[5:].strip().isdigit():
        size = int(st[5:].strip())
print("remote size:", size)
if not size:
    print(out[-300:])
    raise SystemExit(1)

CH = 400
raw = bytearray()
off = 0
while off < size:
    code = ("f=open('%s','rb'); f.seek(%d); print('B64:'+ubinascii.b2a_base64(f.read(%d)).decode().strip()); f.close()"
            % (remote, off, CH))
    s.write(code.encode() + b"\r\n")
    time.sleep(0.55)
    data = s.read_all().decode("utf-8", "replace")
    got = None
    for l in data.replace("\r", "").split("\n"):
        if l.startswith("B64:"):
            got = l[4:].strip()
    if not got:
        print("  重试偏移 %d" % off)
        time.sleep(0.3)
        continue
    try:
        chunk = base64.b64decode(got)
    except Exception:
        continue
    raw.extend(chunk)
    off += len(chunk)
    if off % 4000 < CH:
        print("  %d/%d" % (len(raw), size))
s.close()
open(local, "wb").write(bytes(raw))
print("saved %d bytes -> %s" % (len(raw), local))
