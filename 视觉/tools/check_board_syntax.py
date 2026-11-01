# -*- coding: utf-8 -*-
"""给板端脚本做「PC 侧语法体检」。

板端脚本有两个 CPython 煎不动的点：
  1. `PAGE = b\"\"\"...\"\"\"` 里塞了中文 —— Python 3 的 bytes 字面量不允许非 ASCII，
     但 MicroPython 接受，所以这个文件在 CPython 下必然 SyntaxError；
  2. 顶层 `from media.sensor import *` 之类只有板子上才有。

所以这里做两件事：
  * 把非 ASCII 的 bytes 字面量替换成 b"" 之后再 compile，验证**除它以外**的语法；
  * 把 PAGE 里的 <script> 抽出来喂 `node --check`，验证前端 JS 没写崩。

用法：py -3 tools/check_board_syntax.py k230/pose_webrtc.py k230/webrtc_stream.py
"""
import re
import subprocess
import sys
import tempfile


def strip_nonascii_bytes_literals(src):
    """把含非 ASCII 的 b\"\"\"/b''' 字面量换成 b\"\"，其余原样保留。"""
    out = []
    i = 0
    pat = re.compile(r'b("""|\'\'\')(.*?)\1', re.S)
    for m in pat.finditer(src):
        body = m.group(2)
        out.append(src[i:m.start()])
        out.append('b""' if any(ord(c) > 127 for c in body) else m.group(0))
        i = m.end()
    out.append(src[i:])
    return "".join(out)


def check_py(path):
    src = open(path, encoding="utf-8").read()
    cleaned = strip_nonascii_bytes_literals(src)
    try:
        compile(cleaned, path, "exec")
    except SyntaxError as e:
        print("  [FAIL] %s:%s %s" % (path, e.lineno, e.msg))
        return False
    print("  [ok]   %s (Python 语法)" % path)
    return True


def check_js(path):
    src = open(path, encoding="utf-8").read()
    blocks = re.findall(r"<script>(.*?)</script>", src, re.S)
    if not blocks:
        print("  [--]   %s 没有 <script>" % path)
        return True
    ok = True
    for n, js in enumerate(blocks):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                         encoding="utf-8") as f:
            f.write(js)
            tmp = f.name
        r = subprocess.run(["node", "--check", tmp], capture_output=True, text=True)
        if r.returncode != 0:
            print("  [FAIL] %s <script #%d> JS: %s" % (path, n, r.stderr.strip()[:400]))
            ok = False
        else:
            print("  [ok]   %s <script #%d> JS" % (path, n))
    return ok


def main(paths):
    good = True
    for p in paths:
        good &= check_py(p)
        good &= check_js(p)
    print("ALL OK" if good else "有错")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["k230/pose_webrtc.py", "k230/webrtc_stream.py"]))
