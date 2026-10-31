"""通过 CanMV MCP 分块上传大文件（kmodel 之类 >3MB 的文件）。

MCP 的 canmv_write_file 一次传 3MB 以上会静默失败，所以：
  1. 先在板子上建空文件；
  2. 每块 base64 后用 canmv_run_script 追加写；
  3. 传完用 list_dir 核对字节数。

    python3 tools/upload_file.py k230/target_pose_v11_448_int8.kmodel /sdcard/kmodel/
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canmv_mcp import Mcp  # noqa: E402


def main():
    ap = sys.argv[1:]
    if len(ap) < 2:
        print(__doc__)
        return 1
    local = Path(ap[0])
    remote_dir = ap[1]
    chunk = int(ap[2]) if len(ap) > 2 else 120_000
    name = local.name
    remote = remote_dir.rstrip("/") + "/" + name

    m = Mcp()

    def call(tool, args):
        res = m.rpc("tools/call", {"name": tool, "arguments": args})
        txt = res["result"]["content"][0]["text"]
        try:
            return json.loads(txt)
        except Exception:
            return {"raw": txt}

    data = local.read_bytes()
    total = len(data)
    print(f"{name}  {total/1e6:.2f} MB -> {remote}  （每块 {chunk/1024:.0f} KB）")

    def run(script, wait=1500):
        """写个小脚本到板上执行，返回终端输出（同步等它跑完）。"""
        r = call("canmv_write_and_run_script",
                 {"script": script, "path": "/sdcard/_up.py", "waitMs": wait})
        return r.get("output", "") or ""

    run(f"f=open({remote!r},'wb')\nf.close()\nprint('OK')\n")
    t0 = time.time()
    for off in range(0, total, chunk):
        piece = data[off:off + chunk]
        b64 = base64.b64encode(piece).decode()
        want = min(total, off + chunk)
        script = ("import ubinascii, os\n"
                  f"open({remote!r},'ab').write(ubinascii.a2b_base64({b64!r}))\n"
                  f"print('SZ', os.stat({remote!r})[6])\n")
        got = None
        for attempt in range(3):
            out = run(script, 2500)
            toks = [t for t in out.replace("\r", " ").split() if t.isdigit()]
            got = int(toks[-1]) if toks else None
            if got == want:
                break
            print(f"  重试 @{off}（期望 {want}，得到 {got}）")
        if got != want:
            print(f"  块 @{off} 失败：期望 {want} 实际 {got}")
            return 1
        el = time.time() - t0
        print(f"  {want*100//total:3d}%  {want/1e6:.2f}/{total/1e6:.2f} MB  {el:.0f}s")
    # 核对
    d = call("canmv_list_dir", {"path": remote_dir})
    for e in d.get("entries", []):
        if e.get("name") == name:
            print(f"板上大小 {e['size']}  本地 {total}  {'OK' if e['size'] == total else '不一致!'}")
            return 0 if e["size"] == total else 2
    print("没找到文件")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
