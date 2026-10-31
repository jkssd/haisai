"""CanMV IDE 的 MCP 客户端（Streamable HTTP）。

端点与 token 来自 VSCode 插件：token 在
~/.config/Code/User/globalStorage/kendryte747.canmv-vscode/mcp-http-token

用法：
    python3 tools/canmv_mcp.py list
    python3 tools/canmv_mcp.py call <tool> '<json args>'
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import requests

URL = "http://127.0.0.1:38985/mcp"
TOKEN = Path.home() / ".config/Code/User/globalStorage/kendryte747.canmv-vscode/mcp-http-token"


class Mcp:
    def __init__(self):
        self.s = requests.Session()
        self.h = {
            "Authorization": "Bearer " + TOKEN.read_text().strip(),
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        self.sid = None
        self._id = 0
        r = self._post({"jsonrpc": "2.0", "id": self._nid(), "method": "initialize",
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                   "clientInfo": {"name": "dsh", "version": "1"}}})
        self.sid = r.headers.get("mcp-session-id") or self.sid
        if self.sid:
            self.h["Mcp-Session-Id"] = self.sid
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _nid(self):
        self._id += 1
        return self._id

    def _post(self, payload):
        return self.s.post(URL, headers=self.h, json=payload, timeout=60)

    def _parse(self, r):
        txt = r.text.strip()
        if txt.startswith("event:") or "\ndata:" in txt:
            for chunk in txt.split("\n\n"):
                for line in chunk.splitlines():
                    if line.startswith("data:"):
                        try:
                            return json.loads(line[5:].strip())
                        except Exception:
                            pass
        try:
            return json.loads(txt)
        except Exception:
            return {"raw": txt[:2000]}

    def rpc(self, method, params=None):
        r = self._post({"jsonrpc": "2.0", "id": self._nid(), "method": method,
                        "params": params or {}})
        return self._parse(r)


def main():
    m = Mcp()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        res = m.rpc("tools/list")
        tools = res.get("result", {}).get("tools", [])
        for t in tools:
            print(f"- {t['name']}: {(t.get('description') or '')[:110]}")
            props = (t.get("inputSchema") or {}).get("properties") or {}
            if props:
                print("    args:", ", ".join(props))
        print(f"\n共 {len(tools)} 个工具")
    elif cmd == "call":
        name = sys.argv[2]
        args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        res = m.rpc("tools/call", {"name": name, "arguments": args})
        print(json.dumps(res, ensure_ascii=False, indent=1)[:4000])
    elif cmd == "resources":
        print(json.dumps(m.rpc("resources/list"), ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
