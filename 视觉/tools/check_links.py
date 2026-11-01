# -*- coding: utf-8 -*-
"""检查 Markdown 里的相对链接是否都指向真实文件（发布前用）。

用法：py -3 tools/check_links.py README.md k230/WEBRTC.md
"""
import os
import re
import sys

LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def check(path):
    base = os.path.dirname(os.path.abspath(path))
    bad = []
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        for target in LINK.findall(line):
            target = target.split("#")[0].strip()
            if not target or SCHEME.match(target):
                continue
            full = os.path.normpath(os.path.join(base, target))
            if not os.path.exists(full):
                bad.append((n, target))
    return bad


def main(paths):
    total = 0
    for p in paths:
        bad = check(p)
        for n, t in bad:
            print("  [BROKEN] %s:%d -> %s" % (p, n, t))
        total += len(bad)
        if not bad:
            print("  [ok]     %s" % p)
    print("坏链接 %d 个" % total)
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
