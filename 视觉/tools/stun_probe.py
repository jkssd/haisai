"""Craft a STUN Binding Request to test K230's ICE responder (no reflash needed).

Flow: fetch board /offer -> ice-ufrag/ice-pwd/candidate, build RFC 5389 request,
check whether board replies with Binding Success.
"""
from __future__ import annotations

import binascii
import hashlib
import hmac
import os
import re
import socket
import struct
import sys
import time
import urllib.request

BOARD = sys.argv[1] if len(sys.argv) > 1 else "192.168.43.188"
HTTP_PORT = 8080

MAGIC = 0x2112A442
ATTR_USERNAME = 0x0006
ATTR_MESSAGE_INTEGRITY = 0x0008
ATTR_PRIORITY = 0x0024
ATTR_FINGERPRINT = 0x8028
ATTR_ICE_CONTROLLING = 0x802A


def pad4(b):
    return b + b"\x00" * ((4 - len(b) % 4) % 4)


def attr(t, v):
    return struct.pack(">HH", t, len(v)) + pad4(v)


def parse_sdp(sdp):
    ufrag = re.search(r"a=ice-ufrag:(\S+)", sdp)
    pwd = re.search(r"a=ice-pwd:(\S+)", sdp)
    cands = re.findall(r"a=candidate:(\S+) (\d+) (\S+) (\d+) (\S+) (\d+) typ (\S+)", sdp)
    return (ufrag.group(1) if ufrag else None,
            pwd.group(1) if pwd else None,
            cands)


def build_request(pwd, username, with_fp=True, wrong_pwd=False):
    txid = os.urandom(12)
    body = b""
    body += attr(ATTR_USERNAME, username.encode())
    body += attr(ATTR_PRIORITY, struct.pack(">I", 1845501695))
    body += attr(ATTR_ICE_CONTROLLING, struct.pack(">Q", 0x1234567890ABCDEF))

    key = (pwd + "_wrong") if wrong_pwd else pwd
    mi_input = struct.pack(">HHI", 0x0001, len(body) + 24, MAGIC) + txid + body
    digest = hmac.new(key.encode(), mi_input, hashlib.sha1).digest()
    body += attr(ATTR_MESSAGE_INTEGRITY, digest)

    if with_fp:
        fp_input = struct.pack(">HHI", 0x0001, len(body) + 8, MAGIC) + txid + body
        crc = (binascii.crc32(fp_input) & 0xFFFFFFFF) ^ 0x5354554E
        body += attr(ATTR_FINGERPRINT, struct.pack(">I", crc))

    return struct.pack(">HHI", 0x0001, len(body), MAGIC) + txid + body


def try_request(port, pwd, username, label, with_fp=True, wrong_pwd=False):
    msg = build_request(pwd, username, with_fp=with_fp, wrong_pwd=wrong_pwd)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(3)
    try:
        s.sendto(msg, (BOARD, port))
        data, addr = s.recvfrom(2048)
        mtype = struct.unpack(">H", data[0:2])[0]
        cls = mtype >> 4
        print("  %-30s -> %d bytes, type=0x%04x, class=%d %s"
              % (label, len(data), mtype, cls, "(SUCCESS)" if cls == 1 else "(NOT success)"))
        return True
    except socket.timeout:
        print("  %-30s -> TIMEOUT (no reply)" % label)
        return False
    finally:
        s.close()


def main():
    url = "http://%s:%d/offer" % (BOARD, HTTP_PORT)
    print("fetch offer:", url)
    with urllib.request.urlopen(url, timeout=10) as r:
        sdp = r.read().decode("utf-8", "replace")
    ufrag, pwd, cands = parse_sdp(sdp)
    print("ice-ufrag =", ufrag)
    print("ice-pwd   =", pwd)
    print("candidates:", cands[:4])

    if not cands:
        print("no candidate parsed")
        return

    cand_ip = cands[0][4]
    port = int(cands[0][5])
    our_ufrag = "probe%d" % (int(time.time()) % 100000)
    username = "%s:%s" % (ufrag, our_ufrag)
    print("target: %s:%d  username: %s" % (cand_ip, port, username))

    print("\n=== tests ===")
    try_request(port, pwd, username, "standard (with FINGERPRINT)")
    try_request(port, pwd, username, "without FINGERPRINT", with_fp=False)
    try_request(port, pwd, username, "wrong password (control)", wrong_pwd=True)


if __name__ == "__main__":
    main()
