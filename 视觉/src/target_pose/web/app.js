'use strict';

/* ============================================================ 状态 */
const S = {
  items: [], queue: [], qidx: 0,
  cur: null, img: null,
  objs: [], sel: -1, newPts: [],
  mode: 'idle',              // idle | new
  view: { s: 1, tx: 0, ty: 0 },
  dirty: false, undo: [],
  mouse: { x: 0, y: 0, in: false },
  spaceDown: false, dragging: null,
  src: null,                 // 离屏画布（取像素做吸附）
  scene: '', status: 'all', boot: new Set(),
};

const $ = (id) => document.getElementById(id);
const cv = $('cv'), ctx = cv.getContext('2d');
const loupe = $('loupe'), lctx = loupe.getContext('2d');
const KPT_NAMES = ['左上', '右上', '右下', '左下', '圆心', '圆左', '圆右'];
const SIZE_CN = ['大', '中', '小'];
let SIZE_TH = [0.553, 0.328];

/* ============================================================ 工具 */
const dpr = () => window.devicePixelRatio || 1;

async function api(path, opt) {
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}

function tip(msg, ms = 2200) {
  const t = $('tip');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(tip._t);
  tip._t = setTimeout(() => t.classList.add('hidden'), ms);
}

function clone(o) { return JSON.parse(JSON.stringify(o)); }

function pushUndo() {
  S.undo.push(clone(S.objs));
  if (S.undo.length > 50) S.undo.shift();
}

/* ============================================================ 坐标 */
function toImg(mx, my) {
  return { x: (mx - S.view.tx) / S.view.s, y: (my - S.view.ty) / S.view.s };
}
function toScreen(ix, iy) {
  return { x: ix * S.view.s + S.view.tx, y: iy * S.view.s + S.view.ty };
}

function resizeCanvas() {
  const r = cv.getBoundingClientRect();
  cv.width = Math.round(r.width * dpr());
  cv.height = Math.round(r.height * dpr());
  const lr = 200;
  loupe.width = lr * dpr();
  loupe.height = lr * dpr();
  render();
}

function fitView() {
  if (!S.img) return;
  const r = cv.getBoundingClientRect();
  const s = Math.min(r.width / S.img.width, r.height / S.img.height) * 0.96;
  S.view.s = s;
  S.view.tx = (r.width - S.img.width * s) / 2;
  S.view.ty = (r.height - S.img.height * s) / 2;
  render();
}

/* ============================================================ 列表 */
async function loadList() {
  const qs = [];
  if (S.scene) qs.push('scene=' + encodeURIComponent(S.scene));
  if (S.status === 'todo_auto' || S.status === 'auto') qs.push('list=todo_auto', 'sort=priority');
  S.items = await api('/api/items' + (qs.length ? '?' + qs.join('&') : ''));
  rebuildQueue();
  updateProgress();
}

function rebuildQueue(keepSid) {
  S.queue = S.items.filter((it) => {
    if (S.status === 'todo') return it.n_objs === 0;
    if (S.status === 'unreviewed') return it.n_objs > 0 && !it.reviewed;
    if (S.status === 'auto') return it.source === 'auto';
    if (S.status === 'todo_auto') return it.source === 'auto' && !it.reviewed;
    if (S.status === 'todo_manual') return it.source === 'user' && !it.reviewed;
    if (S.status === 'bootstrap') return S.boot.has(it.sid);
    return true;
  });
  let i = 0;
  if (keepSid) {
    const j = S.queue.findIndex((x) => x.sid === keepSid);
    i = j >= 0 ? j : 0;
  }
  S.qidx = i;
  updateProgress();
}

function updateProgress() {
  const total = S.queue.length;
  const done = S.queue.filter((x) => x.n_objs > 0).length;
  const rev = S.queue.filter((x) => x.reviewed).length;
  $('progress').textContent =
    `队列 ${total ? S.qidx + 1 : 0}/${total}　已标 ${done}　已复核 ${rev}`;
}

/* ============================================================ 载入 */
function parseLabels(text, W, H) {
  const objs = [];
  for (const line of (text || '').split('\n')) {
    const p = line.trim().split(/\s+/);
    if (p.length < 5 + 21) continue;
    const kp = [];
    for (let k = 0; k < 7; k++) {
      kp.push([parseFloat(p[5 + k * 3]) * W, parseFloat(p[5 + k * 3 + 1]) * H]);
    }
    objs.push({
      corners: kp.slice(0, 4).map((q) => [q[0], q[1]]),
      center: [kp[4][0], kp[4][1]],
      circle_l: [kp[5][0], kp[5][1]],
      circle_r: [kp[6][0], kp[6][1]],
      vis: [2, 2, 2, 2, 2, 2, 2],
    });
  }
  return objs;
}

async function loadItem(qi, { skipDirty = false } = {}) {
  if (!S.queue.length) { tip('队列为空'); return; }
  if (!skipDirty && S.dirty) await save();
  S.qidx = Math.max(0, Math.min(qi, S.queue.length - 1));
  const brief = S.queue[S.qidx];
  const it = await api('/api/item/' + encodeURIComponent(brief.sid));
  S.cur = it;
  if (it.hint_scale) $('inp-hint-scale').value = it.hint_scale;
  if (it.size_thresholds) SIZE_TH = it.size_thresholds;
  S.objs = parseLabels(it.label_text, it.width, it.height);
  S.sel = S.objs.length ? 0 : -1;
  S.newPts = [];
  S.mode = 'idle';
  S.dirty = false;
  S.undo = [];
  if (!S.objs.length) {
    S.mode = 'new';
    tip('依次点击 左上 → 右上 → 右下 → 左下（圆心自动生成，可再拖动）', 4000);
  }

  const img = new Image();
  img.onload = () => {
    S.img = img;
    S.src = document.createElement('canvas');
    S.src.width = img.width; S.src.height = img.height;
    S.src.getContext('2d').drawImage(img, 0, 0);
    fitView();
    updateHud();
    renderObjList();
    updateProgress();
  };
  img.onerror = () => tip('图片加载失败');
  img.src = it.image + '?t=' + Date.now();
  updateHud();
  updateProgress();
}

function updateHud() {
  const it = S.cur;
  if (!it) { $('hud-name').textContent = '—'; return; }
  $('hud-name').textContent =
    `${it.scene_name} · ${it.sid.split('__')[1]} · ${it.width}×${it.height}` +
    (S.dirty ? ' *' : '');
  const parts = [`靶子 ${S.objs.length}`, `来源 ${it.source}`];
  if (it.score) parts.push(`置信 ${it.score.toFixed(2)}`);
  parts.push(it.reviewed ? '已复核' : '未复核');
  $('hud-meta').textContent = parts.join('　');
}

/* ============================================================ 绘制 */
function render() {
  const r = cv.getBoundingClientRect();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#0b0d10';
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (!S.img) return;
  const d = dpr(), s = S.view.s;
  ctx.setTransform(d * s, 0, 0, d * s, d * S.view.tx, d * S.view.ty);
  ctx.imageSmoothingEnabled = s < 2;
  ctx.drawImage(S.img, 0, 0);

  const blind = $('chk-blind').checked;
  const lw = 1.6 / s;

  if (!blind && $('chk-hints').checked && S.cur && S.cur.hints) {
    ctx.save();
    ctx.setLineDash([6 / s, 4 / s]);
    ctx.strokeStyle = 'rgba(255,152,0,.85)';
    ctx.lineWidth = lw;
    for (const b of S.cur.hints) {
      ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
    }
    ctx.restore();
  }

  if ($('chk-grid').checked) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,.25)';
    ctx.lineWidth = 1 / s;
    const W = S.img.width, H = S.img.height;
    for (let i = 1; i <= 2; i++) {
      ctx.beginPath(); ctx.moveTo((W * i) / 3, 0); ctx.lineTo((W * i) / 3, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, (H * i) / 3); ctx.lineTo(W, (H * i) / 3); ctx.stroke();
    }
    ctx.restore();
  }

  if (!blind) {
    S.objs.forEach((o, i) => drawObj(o, i === S.sel, s));
  }

  // 新建中的点
  if (S.mode === 'new' && S.newPts.length) {
    ctx.strokeStyle = '#ffd54a';
    ctx.lineWidth = lw * 1.4;
    ctx.beginPath();
    S.newPts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
    if (S.mouse.in) ctx.lineTo(S.mouse.x, S.mouse.y);
    ctx.stroke();
    S.newPts.forEach((p, i) => dot(p[0], p[1], 4 / s, '#ffd54a'));
  }
}

function drawObj(o, sel, s) {
  const lw = (sel ? 2.4 : 1.5) / s;
  ctx.beginPath();
  o.corners.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
  ctx.closePath();
  ctx.fillStyle = sel ? 'rgba(31,111,235,.10)' : 'rgba(34,224,138,.06)';
  ctx.fill();
  ctx.strokeStyle = sel ? '#4f9dff' : '#22e08a';
  ctx.lineWidth = lw;
  ctx.stroke();

  // 中心圆
  if (o.circle_l && o.circle_r) {
    const rad = Math.hypot(o.circle_r[0] - o.circle_l[0], o.circle_r[1] - o.circle_l[1]) / 2;
    ctx.save();
    ctx.setLineDash([4 / s, 3 / s]);
    ctx.beginPath();
    ctx.arc(o.center[0], o.center[1], Math.max(rad, 1), 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,64,255,.95)';
    ctx.lineWidth = lw;
    ctx.stroke();
    ctx.restore();
    dot(o.circle_l[0], o.circle_l[1], (sel ? 4.5 : 3.5) / s, '#ff40ff', '#0b0d10');
    dot(o.circle_r[0], o.circle_r[1], (sel ? 4.5 : 3.5) / s, '#ff40ff', '#0b0d10');
  }

  // 圆心 / 靶心
  const [cx, cy] = o.center;
  const arm = Math.max(6 / s, Math.min(14 / s, 0.02 * 300));
  ctx.strokeStyle = '#ffd54a';
  ctx.lineWidth = lw;
  ctx.beginPath();
  ctx.moveTo(cx - arm, cy); ctx.lineTo(cx + arm, cy);
  ctx.moveTo(cx, cy - arm); ctx.lineTo(cx, cy + arm);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, arm * 0.45, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255,213,74,.75)';
  ctx.stroke();

  o.corners.forEach((p, i) => {
    dot(p[0], p[1], (sel ? 5 : 4) / s, sel ? '#ffd54a' : '#22e08a', '#0b0d10');
    if (sel) {
      const r = 11 / s;
      ctx.font = `${r}px sans-serif`;
      ctx.fillStyle = '#ffd54a';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(i + 1), p[0], p[1] - r * 1.5);
    }
  });
  if (sel) {
    dot(o.center[0], o.center[1], 5 / s, '#ffd54a', '#0b0d10');
  }
}

function dot(x, y, r, fill, stroke) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.2 / S.view.s; ctx.stroke(); }
}

function drawLoupe() {
  if ($('chk-loupe').checked === false || !S.mouse.in || !S.img) {
    loupe.classList.add('hidden');
    return;
  }
  loupe.classList.remove('hidden');
  const d = dpr();
  const size = 200, zoom = 9;
  const src = size / zoom;                    // 源区域边长(图像像素)
  const sx = S.mouse.x - src / 2, sy = S.mouse.y - src / 2;
  lctx.setTransform(1, 0, 0, 1, 0, 0);
  lctx.fillStyle = '#000';
  lctx.fillRect(0, 0, loupe.width, loupe.height);
  lctx.save();
  lctx.scale(d, d);
  lctx.imageSmoothingEnabled = false;
  lctx.drawImage(S.img, sx, sy, src, src, 0, 0, size, size);
  lctx.restore();

  lctx.setTransform(d, 0, 0, d, 0, 0);
  lctx.strokeStyle = 'rgba(255,255,255,.35)';
  lctx.lineWidth = 1;
  lctx.beginPath();
  lctx.moveTo(size / 2, 0); lctx.lineTo(size / 2, size);
  lctx.moveTo(0, size / 2); lctx.lineTo(size, size / 2);
  lctx.stroke();
  lctx.strokeStyle = 'rgba(31,111,235,.9)';
  lctx.strokeRect(size / 2 - zoom / 2, size / 2 - zoom / 2, zoom, zoom);
  lctx.fillStyle = 'rgba(0,0,0,.6)';
  lctx.fillRect(0, size - 18, size, 18);
  lctx.fillStyle = '#cfd6de';
  lctx.font = '11px monospace';
  lctx.fillText(`x ${S.mouse.x.toFixed(1)}  y ${S.mouse.y.toFixed(1)}`, 6, size - 5);
}

/* ============================================================ 吸附 */
function snapPoint(x, y) {
  if (!$('chk-snap').checked || !S.src) return { x, y };
  const rad = Math.max(2, Math.round(4 / S.view.s));
  const x0 = Math.max(0, Math.round(x) - rad), y0 = Math.max(0, Math.round(y) - rad);
  const w = Math.min(S.src.width - x0, rad * 2 + 1), h = Math.min(S.src.height - y0, rad * 2 + 1);
  if (w < 3 || h < 3) return { x, y };
  let data;
  try {
    data = S.src.getContext('2d').getImageData(x0, y0, w, h).data;
  } catch (e) { return { x, y }; }
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  }
  const cx = Math.round(x) - x0, cy = Math.round(y) - y0;
  let bestX = cx, bx = -1, bestY = cy, by = -1;
  for (let i = 1; i < w - 1; i++) {
    const g = Math.abs(gray[cy * w + i + 1] - gray[cy * w + i - 1]);
    if (g > bx) { bx = g; bestX = i; }
  }
  for (let j = 1; j < h - 1; j++) {
    const g = Math.abs(gray[(j + 1) * w + cx] - gray[(j - 1) * w + cx]);
    if (g > by) { by = g; bestY = j; }
  }
  const TH = 12;
  return {
    x: (bx > TH ? x0 + bestX : x),
    y: (by > TH ? y0 + bestY : y),
  };
}

/* ============================================================ 交互 */
function hitHandle(ix, iy, tol) {
  if (S.sel < 0 || S.sel >= S.objs.length) return null;
  const o = S.objs[S.sel];
  const pts = [...o.corners, o.center, o.circle_l, o.circle_r];
  for (let i = 0; i < 7; i++) {
    if (Math.hypot(pts[i][0] - ix, pts[i][1] - iy) < tol) return i;
  }
  return null;
}

function hitObj(ix, iy) {
  // 点在多边形内
  return S.objs.findIndex((o) => {
    let inside = false;
    const p = o.corners;
    for (let i = 0, j = 3; i < 4; j = i++) {
      if (((p[i][1] > iy) !== (p[j][1] > iy)) &&
          (ix < ((p[j][0] - p[i][0]) * (iy - p[i][1])) / (p[j][1] - p[i][1]) + p[i][0])) {
        inside = !inside;
      }
    }
    return inside;
  });
}

cv.addEventListener('mousedown', (e) => {
  if (!S.img) return;
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const p = toImg(mx, my);

  if (e.button === 2 || S.spaceDown) {
    S.dragging = { type: 'pan', mx, my, tx: S.view.tx, ty: S.view.ty };
    return;
  }
  if (e.button !== 0) return;

  if (S.mode === 'new') {
    const q = snapPoint(p.x, p.y);
    S.newPts.push([q.x, q.y]);
    if (S.newPts.length === 4) finishNew();
    render();
    return;
  }

  const tol = 10 / S.view.s;
  const hi = hitHandle(p.x, p.y, tol);
  if (hi !== null) {
    S.dragging = { type: 'handle', idx: hi };
    pushUndo();
    return;
  }
  const oi = hitObj(p.x, p.y);
  if (oi >= 0) {
    S.sel = oi;
    renderObjList();
    render();
    // 允许直接拖动该靶子最近的点
    return;
  }
  S.dragging = { type: 'pan', mx, my, tx: S.view.tx, ty: S.view.ty };
});

cv.addEventListener('mousemove', (e) => {
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const p = toImg(mx, my);
  S.mouse = { x: p.x, y: p.y, in: true };
  if (S.dragging) {
    if (S.dragging.type === 'pan') {
      S.view.tx = S.dragging.tx + (mx - S.dragging.mx);
      S.view.ty = S.dragging.ty + (my - S.dragging.my);
    } else if (S.dragging.type === 'handle') {
      const q = snapPoint(p.x, p.y);
      const o = S.objs[S.sel];
      if (S.dragging.idx < 4) {
        o.corners[S.dragging.idx] = [q.x, q.y];
        // 圆心跟随（保持与靶框同步，除非用户单独调过圆心/圆）
        if (!o.centerMoved) {
          const c = quadCenter(o.corners);
          const rad = Math.hypot(o.circle_r[0] - o.circle_l[0], o.circle_r[1] - o.circle_l[1]) / 2;
          setCircle(o, c, rad);
        }
      } else if (S.dragging.idx === 4) {
        const rad = Math.hypot(o.circle_r[0] - o.circle_l[0], o.circle_r[1] - o.circle_l[1]) / 2;
        setCircle(o, [q.x, q.y], rad);
        o.centerMoved = true;
      } else {
        // 拖圆左右极点：只改半径，圆心不动
        const rad = Math.hypot(q.x - o.center[0], q.y - o.center[1]);
        let u = [q.x - o.center[0], q.y - o.center[1]];
        const n = Math.hypot(u[0], u[1]) || 1;
        u = [u[0] / n, u[1] / n];
        o.circle_l = [o.center[0] - u[0] * rad, o.center[1] - u[1] * rad];
        o.circle_r = [o.center[0] + u[0] * rad, o.center[1] + u[1] * rad];
        o.centerMoved = true;
      }
      S.dirty = true;
      updateHud();
    }
    render();
  } else if (S.mode === 'new') {
    render();
  }
  drawLoupe();
});

window.addEventListener('mouseup', () => { S.dragging = null; });
cv.addEventListener('mouseleave', () => { S.mouse.in = false; drawLoupe(); render(); });
cv.addEventListener('contextmenu', (e) => e.preventDefault());

cv.addEventListener('wheel', (e) => {
  e.preventDefault();
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const p = toImg(mx, my);
  const k = Math.exp(-e.deltaY * 0.0016);
  const s2 = Math.max(0.02, Math.min(60, S.view.s * k));
  S.view.s = s2;
  S.view.tx = mx - p.x * s2;
  S.view.ty = my - p.y * s2;
  render();
  drawLoupe();
}, { passive: false });

function boardAxis(o) {
  const a = [o.corners[1][0] - o.corners[0][0], o.corners[1][1] - o.corners[0][1]];
  const b = [o.corners[2][0] - o.corners[3][0], o.corners[2][1] - o.corners[3][1]];
  let v = [a[0] + b[0], a[1] + b[1]];
  const n = Math.hypot(v[0], v[1]);
  if (n < 1e-6) return [1, 0];
  return [v[0] / n, v[1] / n];
}

function setCircle(o, center, radius) {
  const u = boardAxis(o);
  o.center = [center[0], center[1]];
  o.circle_l = [center[0] - u[0] * radius, center[1] - u[1] * radius];
  o.circle_r = [center[0] + u[0] * radius, center[1] + u[1] * radius];
}

function quadCenter(q) {
  // 对角线交点
  const [a, b, c, d] = q;
  const d1 = [c[0] - a[0], c[1] - a[1]];
  const d2 = [d[0] - b[0], d[1] - b[1]];
  const den = d1[0] * d2[1] - d1[1] * d2[0];
  if (Math.abs(den) < 1e-9) {
    return [(a[0] + c[0]) / 2, (a[1] + c[1]) / 2];
  }
  const t = ((b[0] - a[0]) * d2[1] - (b[1] - a[1]) * d2[0]) / den;
  return [a[0] + d1[0] * t, a[1] + d1[1] * t];
}

async function finishNew() {
  const corners = S.newPts.slice(0, 4).map((p) => [p[0], p[1]]);
  const c = quadCenter(corners);
  const side = Math.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1]);
  const o = { corners, center: c, vis: [2, 2, 2, 2, 2, 2, 2] };
  const u = boardAxis(o);
  const r0 = side / 3.1;
  o.circle_l = [c[0] - u[0] * r0, c[1] - u[1] * r0];
  o.circle_r = [c[0] + u[0] * r0, c[1] + u[1] * r0];
  S.objs.push(o);
  S.sel = S.objs.length - 1;
  S.newPts = [];
  S.mode = 'idle';
  S.dirty = true;
  renderObjList();
  updateHud();
  render();
  // 服务端定位中心圆，给圆心与半径一个准确初值
  try {
    const r = await api('/api/seed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sid: S.cur.sid, objs: [{ corners, center: c }] }),
    });
    const g = r.objs && r.objs[0];
    if (g) {
      setCircle(o, g.center, Math.hypot(g.circle_r[0] - g.circle_l[0], g.circle_r[1] - g.circle_l[1]) / 2);
      o.centerMoved = true;
      S.dirty = true;
      renderObjList();
      render();
      tip(g.circ_ok ? `圆心已自动对圆｜半径 ${g.radius.toFixed(1)}px（${SIZE_CN[g.size_cls]}）`
                    : '没找到中心圆，请手动拖圆心与圆极点', 3000);
    }
  } catch (e) { /* 忽略 */ }
  updateHud();
}

/* ============================================================ 操作 */
async function save() {
  if (!S.cur) return false;
  const objs = S.objs.map((o) => ({
    corners: o.corners.map((p) => [p[0], p[1]]),
    center: [o.center[0], o.center[1]],
    circle_l: o.circle_l ? [o.circle_l[0], o.circle_l[1]] : null,
    circle_r: o.circle_r ? [o.circle_r[0], o.circle_r[1]] : null,
    vis: o.vis || [2, 2, 2, 2, 2, 2, 2],
  }));
  try {
    await api('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sid: S.cur.sid, objs, reviewed: true }),
    });
    S.dirty = false;
    const it = S.items.find((x) => x.sid === S.cur.sid);
    if (it) { it.n_objs = S.objs.length; it.reviewed = true; it.source = 'user'; }
    updateHud();
    updateProgress();
    return true;
  } catch (e) {
    tip('保存失败: ' + e.message, 4000);
    return false;
  }
}

async function refine() {
  if (!S.objs.length) { tip('没有靶子'); return; }
  const objs = S.objs.map((o) => ({ corners: o.corners, center: o.center,
                                    circle_l: o.circle_l, circle_r: o.circle_r }));
  const r = await api('/api/refine', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sid: S.cur.sid, objs, quad: !$('chk-center-only').checked }),
  });
  pushUndo();
  r.objs.forEach((o, i) => {
    S.objs[i].corners = o.corners;
    S.objs[i].center = o.center;
    if (o.circle_l) S.objs[i].circle_l = o.circle_l;
    if (o.circle_r) S.objs[i].circle_r = o.circle_r;
    S.objs[i].centerMoved = true;
  });
  S.dirty = true;
  updateHud();
  renderObjList();
  render();
  const ok = r.objs.filter((o) => o.info && o.info.circ_ok).length;
  tip(`精修完成：${r.objs.length} 个靶子，${ok} 个圆心已对齐`);
}

function delSel() {
  if (S.sel < 0) { tip('先选中一个靶子'); return; }
  pushUndo();
  S.objs.splice(S.sel, 1);
  S.sel = S.objs.length ? Math.min(S.sel, S.objs.length - 1) : -1;
  S.dirty = true;
  renderObjList();
  updateHud();
  render();
}

function renderObjList() {
  const ul = $('obj-list');
  ul.innerHTML = '';
  S.objs.forEach((o, i) => {
    const li = document.createElement('li');
    if (i === S.sel) li.className = 'sel';
    const side = Math.hypot(o.corners[1][0] - o.corners[0][0], o.corners[1][1] - o.corners[0][1]);
    const rad = (o.circle_l && o.circle_r)
      ? Math.hypot(o.circle_r[0] - o.circle_l[0], o.circle_r[1] - o.circle_l[1]) / 2 : 0;
    const rel = side > 0 ? rad / (side / 2) : 0;
    const th = (S.cur && S.cur.size_thresholds) || SIZE_TH;
    const sc = rel >= th[0] ? 0 : (rel >= th[1] ? 1 : 2);
    li.innerHTML = `<span>#${i + 1} <b style="color:#ff8ff0">${SIZE_CN[sc]}</b></span>
      <span class="muted">边长 ${side.toFixed(0)}px　r=${rad.toFixed(1)}px　r/半边长=${rel.toFixed(2)}</span>`;
    li.onclick = () => { S.sel = i; renderObjList(); render(); };
    ul.appendChild(li);
  });
  if (!S.objs.length) {
    const li = document.createElement('li');
    li.className = 'muted';
    li.textContent = '（无靶子：点「新建靶子」后依次点四个角点）';
    ul.appendChild(li);
  }
}

/* ============================================================ 事件绑定 */
$('btn-prev').onclick = () => loadItem(S.qidx - 1);
$('btn-next').onclick = () => loadItem(S.qidx + 1);
$('btn-accept').onclick = async () => { if (await save()) loadItem(S.qidx + 1, { skipDirty: true }); };
$('btn-save').onclick = async () => { if (await save()) tip('已保存'); };
$('btn-refine').onclick = () => refine().catch((e) => tip('精修失败: ' + e.message, 4000));
$('btn-del').onclick = delSel;

async function seedByClass(cls) {
  if (S.sel < 0 || !S.cur) { tip('先选中一个靶子'); return; }
  const o = S.objs[S.sel];
  try {
    const r = await api('/api/seed', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sid: S.cur.sid, objs: [{ corners: o.corners, center: o.center, size_cls: cls }] }),
    });
    const g = r.objs && r.objs[0];
    if (!g) return;
    setCircle(o, g.center, Math.hypot(g.circle_r[0] - g.circle_l[0], g.circle_r[1] - g.circle_l[1]) / 2);
    o.centerMoved = true;
    S.dirty = true;
    renderObjList(); render();
    tip(`已按「${SIZE_CN[cls]}」生成圆：r=${g.radius.toFixed(1)}px (r/半边长=${g.radius_rel.toFixed(2)})`);
  } catch (e) { tip('生成失败: ' + e.message, 3000); }
}
$('btn-c0').onclick = () => seedByClass(0);
$('btn-c1').onclick = () => seedByClass(1);
$('btn-c2').onclick = () => seedByClass(2);
$('btn-fit').onclick = fitView;

$('btn-new').onclick = () => {
  S.mode = 'new';
  S.newPts = [];
  tip('依次点击 左上 → 右上 → 右下 → 左下（圆心自动生成）', 4000);
  render();
};
$('btn-hint').onclick = () => {
  if (!S.cur || !S.cur.hints || !S.cur.hints.length) { tip('这张图没有提示框'); return; }
  const k = parseFloat($('inp-hint-scale').value) || 1.45;
  pushUndo();
  const made = [];
  S.cur.hints.forEach((b) => {
    const cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2;
    const hw = ((b[2] - b[0]) / 2) * k, hh = ((b[3] - b[1]) / 2) * k;
    const corners = [[cx - hw, cy - hh], [cx + hw, cy - hh], [cx + hw, cy + hh], [cx - hw, cy + hh]];
    const o = { corners, center: [cx, cy], vis: [2, 2, 2, 2, 2, 2, 2], centerMoved: true };
    const u = boardAxis(o);
    const r0 = (hw * 2) / 3.1;
    o.circle_l = [cx - u[0] * r0, cy - u[1] * r0];
    o.circle_r = [cx + u[0] * r0, cy + u[1] * r0];
    S.objs.push(o);
    made.push(o);
  });
  S.dirty = true;
  S.sel = S.objs.length - 1;
  renderObjList();
  updateHud();
  render();
  // 用服务端圆定位修正每个靶子的圆心/半径
  api('/api/seed', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sid: S.cur.sid, objs: made.map((o) => ({ corners: o.corners, center: o.center })) }),
  }).then((r) => {
    (r.objs || []).forEach((g, i) => {
      if (!made[i] || !g) return;
      setCircle(made[i], g.center, Math.hypot(g.circle_r[0] - g.circle_l[0], g.circle_r[1] - g.circle_l[1]) / 2);
    });
    renderObjList(); render();
    tip('已按提示生成，并自动对齐了中心圆；请核对角点与圆');
  }).catch(() => tip('已按提示生成，请逐个拖动角点对齐靶框'));
};
$('btn-reload').onclick = async () => {
  await api('/api/reload', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  await loadList();
  rebuildQueue(S.cur && S.cur.sid);
  if (S.queue.length) loadItem(S.qidx); else { S.img = null; render(); }
  tip(`已刷新：当前筛选 ${S.queue.length} 张`);
};
$('btn-jump').onclick = () => {
  const j = S.queue.findIndex((x) => x.n_objs === 0 || !x.reviewed);
  if (j >= 0) loadItem(j); else tip('全部处理完了 🎉');
};

$('sel-scene').onchange = async (e) => {
  S.scene = e.target.value;
  await loadList();
  if (S.queue.length) loadItem(0); else { S.img = null; render(); updateHud(); }
};
$('sel-filter').onchange = (e) => {
  S.status = e.target.value;
  rebuildQueue(S.cur && S.cur.sid);
  if (S.queue.length) loadItem(S.qidx); 
};

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const k = e.key.toLowerCase();
  if (e.code === 'Space') { S.spaceDown = true; cv.style.cursor = 'grab'; }
  if ((e.ctrlKey || e.metaKey) && k === 's') { e.preventDefault(); save().then((ok) => ok && tip('已保存')); return; }
  if ((e.ctrlKey || e.metaKey) && k === 'z') { e.preventDefault(); if (S.undo.length) { S.objs = S.undo.pop(); S.sel = Math.min(S.sel, S.objs.length - 1); S.dirty = true; renderObjList(); render(); } return; }
  if (e.ctrlKey || e.metaKey) return;
  if (k === 'n') loadItem(S.qidx + 1);
  else if (k === 'p') loadItem(S.qidx - 1);
  else if (k === 'a') $('btn-accept').onclick();
  else if (k === 'r') refine().catch((err) => tip('精修失败: ' + err.message, 4000));
  else if (k === 'f') fitView();
  else if (k === 'z') { if (S.undo.length) { S.objs = S.undo.pop(); S.dirty = true; renderObjList(); render(); } }
  else if (k === 'escape') { S.mode = 'idle'; S.newPts = []; render(); }
  else if (k === 'delete' || k === 'backspace') { e.preventDefault(); delSel(); }
  else if (/^[1-9]$/.test(k)) { const i = parseInt(k, 10) - 1; if (i < S.objs.length) { S.sel = i; renderObjList(); render(); } }
});
window.addEventListener('keyup', (e) => {
  if (e.code === 'Space') { S.spaceDown = false; cv.style.cursor = 'crosshair'; }
});

window.addEventListener('resize', resizeCanvas);

/* ============================================================ 启动 */
(async function main() {
  try {
    const ids = await api('/api/items?list=bootstrap');
    S.boot = new Set(ids.map((x) => x.sid));
  } catch (e) { S.boot = new Set(); }
  const st = await api('/api/stats');
  const sel = $('sel-scene');
  sel.innerHTML = '<option value="">全部场景</option>';
  st.per_scene.forEach((p) => {
    const o = document.createElement('option');
    o.value = p.scene;
    o.textContent = `${p.name} (${p.reviewed}/${p.total})`;
    sel.appendChild(o);
  });
  if (st.per_scene.length === 1) { sel.value = st.per_scene[0].scene; S.scene = sel.value; }
  resizeCanvas();
  await loadList();
  rebuildQueue();
  if (S.queue.length) await loadItem(0);
  else tip('没有图片，检查 data 目录');
})();
