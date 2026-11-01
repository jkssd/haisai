"""可靠上传：传完校验大小，不一致自动重传。用法：k230_put.py <local> <remote> [chunk]"""
import os
import subprocess
import sys
import time

import serial

local = sys.argv[1]
remote = sys.argv[2]
chunk = sys.argv[3] if len(sys.argv) > 3 else "700"

want = os.path.getsize(local)


def remote_size():
    s = serial.Serial("COM5", 115200, timeout=0.4)
    time.sleep(0.4)
    for i in range(12):
        s.write(b"\x03")
        time.sleep(0.7)
        if ">>>" in s.read_all().decode("utf-8", "replace"):
            break
    s.read_all()
    s.write(b"import os\r\n")
    time.sleep(0.4)
    s.read_all()
    s.write(("print('SZ', os.stat('%s')[6])\r\n" % remote).encode())
    time.sleep(1.4)
    out = s.read_all().decode("utf-8", "replace")
    s.close()
    for l in out.replace("\r", "").split("\n"):
        if l.startswith("SZ "):
            try:
                return int(l.split()[1])
            except Exception:
                pass
    return -1


for attempt in range(1, 5):
    subprocess.run([sys.executable, "tools/k230_upload.py", "COM5", local, remote, chunk],
                   capture_output=True, timeout=600)
    got = remote_size()
    print("attempt %d: want=%d got=%d" % (attempt, want, got))
    if got == want:
        print("OK")
        sys.exit(0)
print("FAILED after retries")
sys.exit(1)
