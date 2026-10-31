"""标注 / 复核 Web 工具（仅依赖标准库 + OpenCV + Pillow）。

用法::

    python -m target_pose.app --port 8765 [--scene d2]

打开 http://127.0.0.1:8765 即可：
  * 左键拖动角点微调，滚轮缩放，右键/空格拖动平移；
  * 「新建靶子」后依次点 左上→右上→右下→左下，圆心会自动落在对角线交点，
    再拖到中心圆的圆心即可（按 L 打开放大镜辅助定位）；
  * Ctrl+S 保存，A 保存并跳到下一张，N/P 切换。
"""
from __future__ import annotations

import argparse
import io
import json
import math
import mimetypes
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from PIL import Image

from .config import DATA, KPT_NAMES, N_KPT, ROOT, SCENES, ensure_dirs
from .datasets import Item, annotate_path, meta_path, scan, voc_hints
from .size_class import thresholds as size_thresholds
from .labels import Obj, Sample

WEB_DIR = Path(__file__).resolve().parent / "web"

STATE: Dict = {
    "items": [],
    "by_id": {},
    "hints": {},
    "img_cache": {},
}


# --------------------------------------------------------------------------- #
# 数据访问
# --------------------------------------------------------------------------- #
def reload_items(scene=None) -> None:
    if isinstance(scene, str):
        scene = [scene]
    items = scan(list(scene) if scene else None)
    STATE["items"] = items
    STATE["by_id"] = {i.sid: i for i in items}
    STATE["img_cache"].clear()


def get_hints(scene: str) -> Dict:
    if scene not in STATE["hints"]:
        STATE["hints"][scene] = voc_hints(scene)
    return STATE["hints"][scene]


def item_payload(it: Item) -> Dict:
    ann = annotate_path(it.scene, Path(it.image).stem)
    hints = get_hints(it.scene).get(Path(it.image).stem, [])
    return {
        "sid": it.sid,
        "scene": it.scene,
        "scene_name": SCENES[it.scene].name,
        "image": f"/img/{it.sid}",
        "width": it.width,
        "height": it.height,
        "source": it.source,
        "score": it.score,
        "reviewed": it.reviewed,
        "n_objs": it.n_objs,
        "notes": it.notes,
        "label": str(ann),
        "hints": hints,
        "hint_scale": SCENES[it.scene].hint_scale,
        "size_thresholds": size_thresholds(it.scene),
        "label_text": ann.read_text() if ann.exists() else "",
    }


def read_image(it: Item) -> bytes:
    if it.sid in STATE["img_cache"]:
        return STATE["img_cache"][it.sid]
    data = Path(it.image).read_bytes()
    if len(STATE["img_cache"]) < 8:
        STATE["img_cache"][it.sid] = data
    return data


def save_item(sid: str, objs: List[Dict], reviewed: bool = True, source: str = "user") -> Dict:
    it = STATE["by_id"].get(sid)
    if it is None:
        raise KeyError(sid)
    sample = Sample(image=Path(it.image), width=it.width, height=it.height, objs=[], path_hint=it.scene)
    for o in objs:
        corners = np.asarray(o["corners"], float).reshape(4, 2)
        center = np.asarray(o["center"], float).reshape(2)
        cl = o.get("circle_l")
        cr = o.get("circle_r")
        vis = [int(v) for v in o.get("vis", [2] * N_KPT)]
        obj = Obj(corners=corners, center=center,
                  circle_l=None if cl is None else np.asarray(cl, float),
                  circle_r=None if cr is None else np.asarray(cr, float),
                  vis=vis).sort_corners()
        # 允许略微出画（靶框被边缘裁掉时，真实角点在画面外）：
        # 硬裁到边界会把整块靶子的几何压扁，训练出来的模型也会"学会"贴边。
        raw = obj.points.copy()
        w, h = it.width, it.height
        mx, my = 0.15 * w, 0.15 * h
        obj.corners[:, 0] = np.clip(obj.corners[:, 0], -mx, w - 1 + mx)
        obj.corners[:, 1] = np.clip(obj.corners[:, 1], -my, h - 1 + my)
        obj.center[0] = float(np.clip(obj.center[0], 0, w - 1))
        obj.center[1] = float(np.clip(obj.center[1], 0, h - 1))
        for arr in (obj.circle_l, obj.circle_r):
            arr[0] = float(np.clip(arr[0], 0, w - 1))
            arr[1] = float(np.clip(arr[1], 0, h - 1))
        for k_i, (px, py) in enumerate(raw):
            if px < 1 or py < 1 or px > w - 2 or py > h - 2:
                obj.vis[k_i] = 1
        sample.objs.append(obj)
    lp = annotate_path(it.scene, Path(it.image).stem)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(sample.to_txt())
    mp = meta_path(it.scene, Path(it.image).stem)
    mp.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": source,
        "reviewed": bool(reviewed),
        "score": it.score,
        "n_objs": len(sample.objs),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    it.n_objs = len(sample.objs)
    it.source = source
    it.reviewed = bool(reviewed)
    return {"ok": True, "n_objs": len(sample.objs)}


def bootstrap_ids() -> set:
    p = DATA / "splits" / "bootstrap.txt"
    if not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}


def stats_payload() -> Dict:
    per: Dict[str, Dict] = {}
    for it in STATE["items"]:
        d = per.setdefault(it.scene, {"scene": it.scene, "name": SCENES[it.scene].name,
                                      "total": 0, "labeled": 0, "reviewed": 0, "objs": 0})
        d["total"] += 1
        d["objs"] += it.n_objs
        if it.n_objs > 0:
            d["labeled"] += 1
        if it.reviewed:
            d["reviewed"] += 1
    tot = {"total": sum(v["total"] for v in per.values()),
           "labeled": sum(v["labeled"] for v in per.values()),
           "reviewed": sum(v["reviewed"] for v in per.values()),
           "objs": sum(v["objs"] for v in per.values())}
    return {"per_scene": list(per.values()), "total": tot}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "TargetPoseLab/1.0"

    def log_message(self, fmt, *args):  # 静音
        pass

    # ---------------------------------------------------------------- utils
    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8",
              extra: Optional[Dict] = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode())

    def _err(self, msg: str, code: int = 400):
        self._json({"ok": False, "error": msg}, code)

    def _static(self, name: str, ctype: str):
        p = WEB_DIR / name
        if not p.exists():
            return self._err(f"missing {name}", 404)
        self._send(200, p.read_bytes(), ctype)

    # ---------------------------------------------------------------- routes
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html", "text/html; charset=utf-8")
            if path == "/app.js":
                return self._static("app.js", "text/javascript; charset=utf-8")
            if path == "/style.css":
                return self._static("style.css", "text/css; charset=utf-8")
            if path == "/api/stats":
                return self._json(stats_payload())
            if path == "/api/items":
                scene = (q.get("scene") or [None])[0]
                only = (q.get("list") or [""])[0]
                items = [i for i in STATE["items"] if not scene or i.scene == scene]
                if only == "bootstrap":
                    ids = bootstrap_ids()
                    items = [i for i in items if i.sid in ids]
                elif only == "todo_auto":
                    items = [i for i in items if i.source == "auto" and not i.reviewed]
                elif only == "todo_manual":
                    items = [i for i in items if i.source == "user" and not i.reviewed]
                if (q.get("sort") or [""])[0] == "priority":
                    # 置信度低 / 检出的靶子少 → 优先纠正
                    items.sort(key=lambda i: (round(i.score, 3), -i.n_objs, i.sid))
                return self._json([{
                    "sid": i.sid, "scene": i.scene, "reviewed": i.reviewed,
                    "n_objs": i.n_objs, "source": i.source, "score": i.score,
                } for i in items])
            if path.startswith("/api/item/"):
                sid = path[len("/api/item/"):]
                it = STATE["by_id"].get(sid)
                if it is None:
                    return self._err("no such item", 404)
                return self._json(item_payload(it))
            if path.startswith("/img/"):
                sid = path[len("/img/"):]
                it = STATE["by_id"].get(sid)
                if it is None:
                    return self._err("no such image", 404)
                data = read_image(it)
                ctype = mimetypes.guess_type(it.image)[0] or "image/jpeg"
                return self._send(200, data, ctype)
            if path.startswith("/thumb/"):
                sid = path[len("/thumb/"):]
                it = STATE["by_id"].get(sid)
                if it is None:
                    return self._err("no such image", 404)
                wid = int((q.get("w") or ["480"])[0])
                im = cv2.imread(it.image)
                h = int(im.shape[0] * wid / im.shape[1])
                im = cv2.resize(im, (wid, h), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return self._send(200, buf.tobytes(), "image/jpeg")
            return self._err("not found", 404)
        except BrokenPipeError:
            return
        except Exception as e:  # noqa: BLE001
            return self._err(f"{type(e).__name__}: {e}", 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:  # noqa: BLE001
            return self._err("bad json")
        try:
            if u.path == "/api/save":
                r = save_item(body["sid"], body.get("objs", []),
                              reviewed=bool(body.get("reviewed", True)),
                              source=body.get("source", "user"))
                return self._json(r)
            if u.path == "/api/refine":
                from .refine import refine_target
                it = STATE["by_id"].get(body["sid"])
                if it is None:
                    return self._err("no such item", 404)
                img = cv2.imread(it.image)
                out = []
                for o in body.get("objs", []):
                    obj = Obj(corners=np.asarray(o["corners"], float).reshape(4, 2),
                              center=np.asarray(o["center"], float).reshape(2),
                              circle_l=None if o.get("circle_l") is None else np.asarray(o["circle_l"], float),
                              circle_r=None if o.get("circle_r") is None else np.asarray(o["circle_r"], float))
                    ref, info = refine_target(gray, obj, do_quad=bool(body.get("quad", True)))
                    out.append({"corners": ref.corners.tolist(),
                                "center": ref.center.tolist(),
                                "circle_l": ref.circle_l.tolist(),
                                "circle_r": ref.circle_r.tolist(),
                                "radius": ref.radius,
                                "radius_rel": ref.radius_rel,
                                "size_cls": ref.size_cls,
                                "info": {k: (float(v) if isinstance(v, (int, float)) else v)
                                         for k, v in info.items() if k != "edge_scores"}})
                return self._json({"ok": True, "objs": out})
            if u.path == "/api/seed":
                # 由四角点自动给圆心与圆半径做初值（人只要微调）
                from .refine import locate_circle_robust
                it = STATE["by_id"].get(body["sid"])
                if it is None:
                    return self._err("no such item", 404)
                img = cv2.imread(it.image)
                out = []
                for o in body.get("objs", []):
                    obj = Obj(corners=np.asarray(o["corners"], float).reshape(4, 2),
                              center=np.asarray(o.get("center"), float).reshape(2))
                    from .refine import circle_contrast, refine_circle_near
                    from .size_class import radius_candidates

                    # 三种型号的半径差 3 倍：分别试，谁在框内最像圆就用谁
                    from .refine import pick_radius_by_profile

                    cands = radius_candidates(obj.side, it.scene)
                    want = o.get("size_cls")
                    if want is not None and 0 <= int(want) < len(cands):
                        cands = [cands[int(want)]]
                    pick = pick_radius_by_profile(img, obj.center, cands)
                    ordered = ([pick[1]] + [c for c in cands if abs(c - pick[1]) > 1]) if pick else cands
                    best = None
                    for r_hint in ordered:
                        cen_i, rad_i, info_i = refine_circle_near(img, obj.center, r_hint, tol=0.22)
                        if not info_i.get("circ_ok"):
                            cen_i, rad_i, info_i = locate_circle_robust(
                                img, obj.center, r_hint, side=obj.side, quad=obj.corners)
                        if not info_i.get("circ_ok"):
                            continue
                        score = (info_i.get("circ_support", 0.5)
                                 * min(circle_contrast(img, cen_i, rad_i) / 40.0, 1.0))
                        if best is None or score > best[0]:
                            best = (score, cen_i, rad_i, info_i)
                    if best is None:
                        cen, rad, info = obj.center, obj.side / 3.1, {"circ_ok": False}
                    else:
                        _, cen, rad, info = best
                    if not info.get("circ_ok") or rad <= 0:
                        cen, rad = obj.center, r_hint
                    obj.set_circle(cen, rad)
                    out.append({"center": obj.center.tolist(),
                                "circle_l": obj.circle_l.tolist(),
                                "circle_r": obj.circle_r.tolist(),
                                "radius": obj.radius, "radius_rel": obj.radius_rel,
                                "size_cls": obj.size_cls, "circ_ok": bool(info.get("circ_ok"))})
                return self._json({"ok": True, "objs": out})
            if u.path == "/api/reload":
                reload_items((body or {}).get("scene"))
                return self._json({"ok": True, "n": len(STATE["items"])})
            return self._err("not found", 404)
        except Exception as e:  # noqa: BLE001
            return self._err(f"{type(e).__name__}: {e}", 500)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="靶子关键点 标注/复核 工具")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--scene", default=None, help="只加载某个场景")
    args = ap.parse_args(argv)

    ensure_dirs()
    reload_items(args.scene)
    print(f"加载 {len(STATE['items'])} 张图片，场景: {args.scene or '全部'}")
    print(f"打开  http://{args.host}:{args.port}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
