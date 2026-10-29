"""評価用プラークの edge に手で線を引くローカルツール（2026-10-27）。

PCD-plaque-eval の plaques.csv の各行（病変）について、EID1024 と PCD1024 の crop をブラウザに出し、
石灰の edge を垂直に横切る線をクリック 2 回で引く。線は引くたびに JSON へ自動保存する。

起動（ホストの mac でそのまま。Docker 不要。numpy と Pillow だけ使う）:
    python3 eval/annotate_lines.py --data_dir ~/Downloads/PCD-plaque-eval --tag day1
    → ブラウザで http://127.0.0.1:8765 を開く

保存先: <data_dir>/lines/lines_<tag>.json（既にあれば読み込んで続きから）。別の日に引き直すときは --tag を変える。
座標: EID も PCD も 1024 画素の連続座標（画素中心が整数。x = 列、y = 行）。EID は EID1024（実 EID512 の ×2 bicubic）上の座標。
画素値: stored = HU + 1400（16bit PNG）。
外部には何も送らない（127.0.0.1 だけで待ち受ける）。
"""

import argparse
import csv
import datetime
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

HU_OFFSET = 1400


def load_rows(data_dir):
    path = os.path.join(data_dir, "plaques.csv")
    if not os.path.isfile(path):
        sys.exit(f"plaques.csv が無い: {path}")
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            pcd_name = f"{r['pcd_case']}-{int(r['pcd_slice']):03d}"
            eid_name = f"{r['eid_case']}-{int(r['eid_slice']):03d}"
            row = {
                "row": int(r["row"]),
                "pcd_name": pcd_name,
                "eid_name": eid_name,
                "pcd_image": os.path.join("PCD1024", r["pcd_case"], pcd_name + ".png"),
                "eid_image": os.path.join("EID1024", r["eid_case"], eid_name + ".png"),
                "pcd_center": [float(r["pcd_x_1024"]), float(r["pcd_y_1024"])],
                "eid_center": [float(r["eid_x_1024"]), float(r["eid_y_1024"])],
            }
            for key in ("pcd_image", "eid_image"):
                full = os.path.join(data_dir, row[key])
                if not os.path.isfile(full):
                    sys.exit(f"画像が無い（row {row['row']}）: {full}")
            rows.append(row)
    if not rows:
        sys.exit("plaques.csv に行が無い")
    return rows


def load_or_init(out_path, rows, data_dir, tag):
    if os.path.isfile(out_path):
        with open(out_path) as f:
            saved = json.load(f)
        print(f"既存の注釈を読み込んだ: {out_path}")
        return saved
    now = datetime.datetime.now().isoformat(timespec="seconds")
    return {
        "tag": tag,
        "created": now,
        "updated": now,
        "data_dir": os.path.abspath(data_dir),
        "coord_note": "x = 列, y = 行。1024 画素の連続座標で画素中心が整数。EID は EID1024（EID512 の x2 bicubic）上。"
                      "線は始点 = 外側（内腔・軟部）、終点 = 石灰側",
        "hu_offset": HU_OFFSET,
        "rows": {
            str(r["row"]): {
                "pcd_name": r["pcd_name"], "eid_name": r["eid_name"],
                "pcd_image": r["pcd_image"], "eid_image": r["eid_image"],
                "pcd_center": r["pcd_center"], "eid_center": r["eid_center"],
                "eid_lines": [], "pcd_lines": [], "note": "",
            }
            for r in rows
        },
    }


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def make_handler(data_dir, rows, state, out_path):
    row_by_id = {r["row"]: r for r in rows}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/state":
                body = json.dumps({"rows": rows, "saved": state, "out_path": out_path}, ensure_ascii=False)
                self._send(200, body.encode("utf-8"), "application/json")
            elif u.path == "/img":
                q = parse_qs(u.query)
                try:
                    r = row_by_id[int(q["row"][0])]
                    which = q["which"][0]
                    rel = {"eid": r["eid_image"], "pcd": r["pcd_image"]}[which]
                except (KeyError, ValueError):
                    self._send(400, b"bad request", "text/plain")
                    return
                a = np.array(Image.open(os.path.join(data_dir, rel)))
                if a.dtype != np.uint16 or a.ndim != 2:
                    self._send(500, f"16bit 1ch ではない: {rel}".encode(), "text/plain")
                    return
                h, w = a.shape
                header = np.array([w, h], dtype="<u4").tobytes()
                self._send(200, header + a.astype("<u2").tobytes(), "application/octet-stream")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/save":
                self._send(404, b"not found", "text/plain")
                return
            n = int(self.headers.get("Content-Length", "0"))
            incoming = json.loads(self.rfile.read(n).decode("utf-8"))
            for rid, rv in incoming["rows"].items():
                if rid not in state["rows"]:
                    continue
                dst = state["rows"][rid]
                dst["eid_lines"] = rv["eid_lines"]
                dst["pcd_lines"] = rv["pcd_lines"]
                dst["note"] = rv["note"]
            state["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
            atomic_write_json(out_path, state)
            self._send(200, json.dumps({"ok": True, "updated": state["updated"]}).encode(), "application/json")

    return Handler


PAGE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>Plaque edge lines</title>
<style>
:root { --bg:#111; --fg:#ddd; --mut:#888; --acc:#ff4d4d; --acc2:#4dc3ff; --ok:#5fd35f; }
body { background:var(--bg); color:var(--fg); font:14px -apple-system, sans-serif; margin:12px; }
button { background:#2a2a2a; color:var(--fg); border:1px solid #444; padding:4px 10px; border-radius:4px; cursor:pointer; }
button.cur { border-color:var(--acc2); color:var(--acc2); }
button.done::after { content:" ✓"; color:var(--ok); }
input, select { background:#222; color:var(--fg); border:1px solid #444; padding:2px 4px; }
.bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:10px; }
.panels { display:flex; flex-wrap:wrap; gap:18px; }
.panel h3 { margin:4px 0; font-size:15px; }
canvas.img { cursor:crosshair; image-rendering:pixelated; border:1px solid #333; }
canvas.prof { border:1px solid #333; margin-top:6px; }
.lines { font:12px ui-monospace, monospace; margin-top:4px; }
.lines div { display:flex; gap:8px; align-items:center; }
.mut { color:var(--mut); font-size:12px; }
#note { width:520px; }
</style></head><body>
<div class="bar" id="rowbar"></div>
<div class="bar">
  WL <input id="wl" type="number" value="600" style="width:70px"> HU
  WW <input id="ww" type="number" value="2000" style="width:70px"> HU
  crop <select id="crop"><option>48</option><option>72</option><option selected>144</option><option>288</option></select> 画素
  <span class="mut">クリック 2 回で線（始点 = 外側の内腔・軟部、終点 = 石灰側）。<b>線をドラッグで移動</b>、<b>端の丸をドラッグで伸縮・向き変更</b>（Shift を押しながらだと向きを保って伸び縮みだけ）。<b>線の上で右クリック</b>で消す。Backspace で最後の線を消す。Esc で引きかけを取り消し。コピーは中心座標からの相対位置で貼る（既存の線に追加）。変更は自動保存。</span>
</div>
<div class="panels">
  <div class="panel"><h3 id="t_eid">EID1024</h3><div class="bar"><button onclick="delLast('eid')">最後の線を消す</button><button onclick="delAll('eid')">この画像の線を全部消す</button><button onclick="copyLines('eid','pcd')">EID の線 → PCD へコピー</button></div><canvas class="img" id="c_eid" width="576" height="576"></canvas>
    <br><canvas class="prof" id="p_eid" width="576" height="200"></canvas><div class="lines" id="l_eid"></div></div>
  <div class="panel"><h3 id="t_pcd">PCD1024</h3><div class="bar"><button onclick="delLast('pcd')">最後の線を消す</button><button onclick="delAll('pcd')">この画像の線を全部消す</button><button onclick="copyLines('pcd','eid')">PCD の線 → EID へコピー</button></div><canvas class="img" id="c_pcd" width="576" height="576"></canvas>
    <br><canvas class="prof" id="p_pcd" width="576" height="200"></canvas><div class="lines" id="l_pcd"></div></div>
</div>
<div class="bar" style="margin-top:10px">メモ <input id="note" placeholder="この病変についてのメモ（任意）"> <span id="status" class="mut"></span></div>
<div class="mut" id="outpath"></div>
<script>
const HU_OFF = 1400, VIEW = 576;
let ROWS = [], SAVED = null, cur = null;
const imgs = {};            // key "row:which" -> {w,h,data:Uint16Array}
const pending = {eid:null, pcd:null};  // 1 点目
const hover = {eid:null, pcd:null};
const COLORS = ['#ff4d4d','#ffd84d','#4dff88','#4dc3ff','#d34dff'];

async function init() {
  const s = await (await fetch('/state')).json();
  ROWS = s.rows; SAVED = s.saved;
  document.getElementById('outpath').textContent = '保存先: ' + s.out_path;
  const bar = document.getElementById('rowbar');
  ROWS.forEach(r => {
    const b = document.createElement('button');
    b.textContent = '病変 ' + r.row; b.id = 'rb' + r.row;
    b.onclick = () => select(r.row); bar.appendChild(b);
  });
  ['wl','ww','crop'].forEach(id => document.getElementById(id).onchange = drawAll);
  document.getElementById('note').oninput = () => { rec().note = document.getElementById('note').value; save(); };
  for (const w of ['eid','pcd']) setupCanvas(w);
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'Escape') { pending.eid = pending.pcd = null; drawAll(); }
    if (e.key === 'Backspace' || e.key === 'Delete') { e.preventDefault(); delLast(lastPanel); }
  });
  refreshRowButtons();
  select(ROWS[0].row);
}

function rec() { return SAVED.rows[String(cur)]; }
function rowInfo() { return ROWS.find(r => r.row === cur); }

async function getImg(row, which) {
  const k = row + ':' + which;
  if (!imgs[k]) {
    const buf = await (await fetch(`/img?row=${row}&which=${which}`)).arrayBuffer();
    const hdr = new Uint32Array(buf, 0, 2);
    imgs[k] = {w: hdr[0], h: hdr[1], data: new Uint16Array(buf, 8)};
  }
  return imgs[k];
}

async function select(row) {
  cur = row; pending.eid = pending.pcd = null;
  ROWS.forEach(r => document.getElementById('rb' + r.row).classList.toggle('cur', r.row === row));
  const r = rowInfo();
  document.getElementById('t_eid').textContent = `EID1024  ${r.eid_name}  中心 (${r.eid_center})`;
  document.getElementById('t_pcd').textContent = `PCD1024  ${r.pcd_name}  中心 (${r.pcd_center})`;
  document.getElementById('note').value = rec().note || '';
  await getImg(row, 'eid'); await getImg(row, 'pcd');
  drawAll();
}

function view(which) {
  const crop = +document.getElementById('crop').value;
  const c = rowInfo()[which + '_center'];
  const left = Math.round(c[0]) - crop / 2, top = Math.round(c[1]) - crop / 2;
  return {crop, left, top, zoom: VIEW / crop};
}
// 画像座標（画素中心が整数）<-> キャンバス座標
function toImg(v, u, w) { return [v.left + u / v.zoom - 0.5, v.top + w / v.zoom - 0.5]; }
function toCan(v, x, y) { return [(x - v.left + 0.5) * v.zoom, (y - v.top + 0.5) * v.zoom]; }

function sample(im, x, y) {  // 双線形。stored 値を返す
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const g = (xx, yy) => im.data[Math.min(im.h - 1, Math.max(0, yy)) * im.w + Math.min(im.w - 1, Math.max(0, xx))];
  return (g(x0,y0)*(1-fx) + g(x0+1,y0)*fx)*(1-fy) + (g(x0,y0+1)*(1-fx) + g(x0+1,y0+1)*fx)*fy;
}

const drag = {eid: null, pcd: null};   // {n, mode: 'p0'|'p1'|'move', su, sw, orig, moved}
let suppressClick = false;
const r2 = a => Math.round(a * 100) / 100;
function hitTest(which, u, w) {  // 端点（半径 9）を先に、次に線の本体（6 以内）
  const v = view(which), L = rec()[which + '_lines'];
  for (let n = L.length - 1; n >= 0; n--) {
    const p0 = toCan(v, L[n].x0, L[n].y0), p1 = toCan(v, L[n].x1, L[n].y1);
    if (Math.hypot(u - p1[0], w - p1[1]) < 9) return {n, mode: 'p1'};
    if (Math.hypot(u - p0[0], w - p0[1]) < 9) return {n, mode: 'p0'};
  }
  for (let n = L.length - 1; n >= 0; n--) {
    const l = L[n];
    if (segDist(u, w, toCan(v, l.x0, l.y0), toCan(v, l.x1, l.y1)) < 6) return {n, mode: 'move'};
  }
  return null;
}
function setupCanvas(which) {
  const cv = document.getElementById('c_' + which);
  cv.addEventListener('mousedown', e => {
    if (e.button !== 0 || pending[which]) return;
    const h = hitTest(which, e.offsetX, e.offsetY);
    if (h) drag[which] = {...h, su: e.offsetX, sw: e.offsetY, orig: {...rec()[which + '_lines'][h.n]}, moved: false};
  });
  cv.addEventListener('mousemove', e => {
    const v = view(which); hover[which] = toImg(v, e.offsetX, e.offsetY);
    const d = drag[which];
    if (d) {
      let dx = (e.offsetX - d.su) / v.zoom, dy = (e.offsetY - d.sw) / v.zoom;
      if (Math.abs(e.offsetX - d.su) + Math.abs(e.offsetY - d.sw) > 2) d.moved = true;
      const o = d.orig, l = rec()[which + '_lines'][d.n];
      if (d.mode !== 'move' && e.shiftKey) {  // Shift: 向きを保って伸び縮みだけ
        const ux = o.x1 - o.x0, uy = o.y1 - o.y0, len = Math.hypot(ux, uy) || 1;
        const t = (dx * ux + dy * uy) / len; dx = t * ux / len; dy = t * uy / len;
      }
      if (d.mode === 'move' || d.mode === 'p0') { l.x0 = o.x0 + dx; l.y0 = o.y0 + dy; }
      if (d.mode === 'move' || d.mode === 'p1') { l.x1 = o.x1 + dx; l.y1 = o.y1 + dy; }
      drawPanel(which); return;
    }
    if (pending[which]) { drawPanel(which); return; }
    const h = hitTest(which, e.offsetX, e.offsetY);
    cv.style.cursor = !h ? 'crosshair' : h.mode === 'move' ? 'move' : 'grab';
  });
  const endDrag = (fromUp) => {
    const d = drag[which]; if (!d) return;
    drag[which] = null;
    if (d.moved) {
      const l = rec()[which + '_lines'][d.n];
      l.x0 = r2(l.x0); l.y0 = r2(l.y0); l.x1 = r2(l.x1); l.y1 = r2(l.y1);
      suppressClick = fromUp; save(); drawPanel(which);
    }
  };
  cv.addEventListener('mouseup', () => endDrag(true));
  cv.addEventListener('mouseleave', () => endDrag(false));
  cv.addEventListener('mouseenter', () => { lastPanel = which; });
  cv.addEventListener('contextmenu', e => {
    e.preventDefault();
    if (pending[which]) { pending[which] = null; drawPanel(which); return; }
    const h = hitTest(which, e.offsetX, e.offsetY);
    if (h) { rec()[which + '_lines'].splice(h.n, 1); save(); drawPanel(which); }
  });
  cv.addEventListener('click', e => {
    if (suppressClick) { suppressClick = false; return; }
    const v = view(which); const p = toImg(v, e.offsetX, e.offsetY);
    if (!pending[which]) {
      if (hitTest(which, e.offsetX, e.offsetY)) return;  // 既存の線の上のクリックでは新しい線を始めない
      pending[which] = p;
    } else {
      const q = pending[which]; pending[which] = null;
      rec()[which + '_lines'].push({x0: r2(q[0]), y0: r2(q[1]), x1: r2(p[0]), y1: r2(p[1])});
      save();
    }
    drawPanel(which);
  });
}

// 線のコピー: 各画像の中心（plaques.csv）からの相対位置を保って貼る。既存の線には追加
function copyLines(from, to) {
  const src = rec()[from + '_lines']; if (!src.length) { alert('コピー元に線が無い'); return; }
  const cf = rowInfo()[from + '_center'], ct = rowInfo()[to + '_center'];
  const dx = ct[0] - cf[0], dy = ct[1] - cf[1];
  src.forEach(l => rec()[to + '_lines'].push({x0: r2(l.x0 + dx), y0: r2(l.y0 + dy), x1: r2(l.x1 + dx), y1: r2(l.y1 + dy)}));
  save(); drawPanel(to);
}

let lastPanel = 'eid';
function delLast(which) { const L = rec()[which + '_lines']; if (L.length) { L.pop(); save(); drawPanel(which); } }
function delAll(which) {
  const L = rec()[which + '_lines']; if (!L.length) return;
  if (!confirm(`この画像の線 ${L.length} 本を全部消しますか`)) return;
  L.length = 0; save(); drawPanel(which);
}
function segDist(px, py, a, b) {
  const dx = b[0]-a[0], dy = b[1]-a[1], l2 = dx*dx + dy*dy;
  let t = l2 ? ((px-a[0])*dx + (py-a[1])*dy) / l2 : 0; t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (a[0]+t*dx), py - (a[1]+t*dy));
}
function drawAll() { drawPanel('eid'); drawPanel('pcd'); }

function drawPanel(which) {
  const im = imgs[cur + ':' + which]; if (!im) return;
  const v = view(which);
  const wl = +document.getElementById('wl').value, ww = +document.getElementById('ww').value;
  const lo = wl - ww / 2 + HU_OFF, hi = wl + ww / 2 + HU_OFF;
  // crop を等倍で描いてから拡大（pixelated）
  const off = document.createElement('canvas'); off.width = off.height = v.crop;
  const octx = off.getContext('2d'); const id = octx.createImageData(v.crop, v.crop);
  for (let j = 0; j < v.crop; j++) for (let i = 0; i < v.crop; i++) {
    const x = v.left + i, y = v.top + j;
    let val = (x >= 0 && y >= 0 && x < im.w && y < im.h) ? im.data[y * im.w + x] : 0;
    let g = Math.round(255 * (val - lo) / (hi - lo)); g = g < 0 ? 0 : g > 255 ? 255 : g;
    const k = 4 * (j * v.crop + i); id.data[k] = id.data[k+1] = id.data[k+2] = g; id.data[k+3] = 255;
  }
  octx.putImageData(id, 0, 0);
  const cv = document.getElementById('c_' + which), ctx = cv.getContext('2d');
  ctx.imageSmoothingEnabled = false; ctx.drawImage(off, 0, 0, VIEW, VIEW);
  // 中心の十字
  const c = toCan(v, ...rowInfo()[which + '_center']);
  ctx.strokeStyle = 'rgba(77,195,255,0.6)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(c[0]-8, c[1]); ctx.lineTo(c[0]+8, c[1]); ctx.moveTo(c[0], c[1]-8); ctx.lineTo(c[0], c[1]+8); ctx.stroke();
  // 線
  const lines = rec()[which + '_lines'];
  lines.forEach((L, n) => drawLine(ctx, v, L, COLORS[n % COLORS.length], n + 1));
  if (pending[which] && hover[which]) {
    const L = {x0: pending[which][0], y0: pending[which][1], x1: hover[which][0], y1: hover[which][1]};
    drawLine(ctx, v, L, 'rgba(255,255,255,0.7)', '');
  } else if (pending[which]) {
    const p = toCan(v, ...pending[which]); ctx.fillStyle = '#fff'; ctx.fillRect(p[0]-2, p[1]-2, 4, 4);
  }
  drawProfile(which, im, lines, pending[which] && hover[which] ? {x0: pending[which][0], y0: pending[which][1], x1: hover[which][0], y1: hover[which][1]} : null);
  drawLineList(which, lines);
}

function drawLine(ctx, v, L, color, label) {
  const a = toCan(v, L.x0, L.y0), b = toCan(v, L.x1, L.y1);
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  // 終点（石灰側）に矢じり
  const ang = Math.atan2(b[1]-a[1], b[0]-a[0]), h = 9;
  ctx.beginPath(); ctx.moveTo(b[0], b[1]);
  ctx.lineTo(b[0] - h*Math.cos(ang-0.4), b[1] - h*Math.sin(ang-0.4));
  ctx.lineTo(b[0] - h*Math.cos(ang+0.4), b[1] - h*Math.sin(ang+0.4)); ctx.closePath();
  ctx.fillStyle = color; ctx.fill();
  if (label !== '') {
    ctx.font = '13px sans-serif'; ctx.fillText(label, a[0] + 4, a[1] - 4);
    ctx.lineWidth = 1.5; ctx.strokeStyle = color;
    ctx.beginPath(); ctx.arc(a[0], a[1], 4, 0, 2 * Math.PI); ctx.stroke();   // 始点のハンドル
    ctx.beginPath(); ctx.arc(b[0], b[1], 5, 0, 2 * Math.PI); ctx.stroke();   // 終点のハンドル
  }
}

function profileOf(im, L) {
  const len = Math.hypot(L.x1 - L.x0, L.y1 - L.y0), step = 0.1, n = Math.max(2, Math.floor(len / step) + 1);
  const d = [], hu = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    d.push(t * len); hu.push(sample(im, L.x0 + t * (L.x1 - L.x0), L.y0 + t * (L.y1 - L.y0)) - HU_OFF);
  }
  return {d, hu, len};
}

function rise1090(p) {  // 目安: 始点側の最小と終点側の最大の間で、10 % と 90 % を最初に越える距離の差
  const lo = Math.min(...p.hu), hi = Math.max(...p.hu), iMax = p.hu.indexOf(hi);
  const t10 = lo + 0.1 * (hi - lo), t90 = lo + 0.9 * (hi - lo);
  let d10 = null, d90 = null;
  for (let i = iMax; i > 0; i--) {
    if (d90 === null && p.hu[i-1] < t90) d90 = p.d[i];
    if (d10 === null && p.hu[i-1] < t10) { d10 = p.d[i]; break; }
  }
  return (d10 !== null && d90 !== null) ? {d: d90 - d10, d10, d90, lo, hi} : null;
}

function drawProfile(which, im, lines, live) {
  const cv = document.getElementById('p_' + which), ctx = cv.getContext('2d');
  ctx.fillStyle = '#181818'; ctx.fillRect(0, 0, cv.width, cv.height);
  const items = lines.map((L, n) => ({p: profileOf(im, L), color: COLORS[n % COLORS.length]}));
  if (live) items.push({p: profileOf(im, live), color: 'rgba(255,255,255,0.8)'});
  if (!items.length) { ctx.fillStyle = '#666'; ctx.fillText('線を引くとここに HU プロファイル（横軸 = 始点からの距離 [1024 画素]）', 10, 20); return; }
  let maxD = 0, lo = Infinity, hi = -Infinity;
  items.forEach(it => { maxD = Math.max(maxD, it.p.len); lo = Math.min(lo, ...it.p.hu); hi = Math.max(hi, ...it.p.hu); });
  const pad = 30, W = cv.width - pad - 10, H = cv.height - 30;
  const X = d => pad + W * d / Math.max(maxD, 1e-6), Y = h => 10 + H * (1 - (h - lo) / Math.max(hi - lo, 1));
  ctx.strokeStyle = '#333'; ctx.fillStyle = '#777'; ctx.font = '11px sans-serif';
  for (let k = 0; k <= 4; k++) { const h = lo + (hi - lo) * k / 4; ctx.beginPath(); ctx.moveTo(pad, Y(h)); ctx.lineTo(pad + W, Y(h)); ctx.stroke(); ctx.fillText(Math.round(h), 2, Y(h) + 4); }
  for (let k = 0; k <= Math.floor(maxD); k += Math.max(1, Math.round(maxD / 8))) ctx.fillText(k, X(k) - 3, cv.height - 6);
  items.forEach(it => {
    ctx.strokeStyle = it.color; ctx.lineWidth = 1.5; ctx.beginPath();
    it.p.d.forEach((d, i) => { const x = X(d), y = Y(it.p.hu[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
    const r = rise1090(it.p);
    if (r) { ctx.setLineDash([3,3]); ctx.beginPath(); ctx.moveTo(X(r.d10), 10); ctx.lineTo(X(r.d10), 10 + H); ctx.moveTo(X(r.d90), 10); ctx.lineTo(X(r.d90), 10 + H); ctx.stroke(); ctx.setLineDash([]); }
  });
}

function drawLineList(which, lines) {
  const el = document.getElementById('l_' + which); el.innerHTML = '';
  const im = imgs[cur + ':' + which];
  lines.forEach((L, n) => {
    const p = profileOf(im, L), r = rise1090(p);
    const div = document.createElement('div');
    div.innerHTML = `<span style="color:${COLORS[n % COLORS.length]}">線 ${n+1}</span>
      (${L.x0.toFixed(1)}, ${L.y0.toFixed(1)}) → (${L.x1.toFixed(1)}, ${L.y1.toFixed(1)})
      長さ ${p.len.toFixed(1)} px ・ 10–90% 目安 ${r ? r.d.toFixed(2) + ' px' : '—'} ・ ${Math.round(Math.min(...p.hu))}→${Math.round(Math.max(...p.hu))} HU`;
    const b = document.createElement('button'); b.textContent = '消す';
    b.onclick = () => { lines.splice(n, 1); save(); drawPanel(which); };
    div.appendChild(b); el.appendChild(div);
  });
}

function refreshRowButtons() {
  ROWS.forEach(r => {
    const s = SAVED.rows[String(r.row)];
    document.getElementById('rb' + r.row).classList.toggle('done', s.eid_lines.length > 0 && s.pcd_lines.length > 0);
  });
}

let saveTimer = null;
function save() {
  refreshRowButtons();
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    const st = document.getElementById('status');
    try {
      const res = await (await fetch('/save', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({rows: SAVED.rows})})).json();
      st.textContent = '保存済み ' + res.updated;
    } catch (e) { st.textContent = '保存に失敗: ' + e; st.style.color = '#ff4d4d'; }
  }, 200);
}
init();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="評価用プラークの edge に線を引くローカルツール")
    ap.add_argument("--data_dir", required=True, help="PCD-plaque-eval のフォルダ（plaques.csv と PCD1024/ EID1024/ がある所）")
    ap.add_argument("--tag", required=True, help="注釈の名前（例 day1）。<data_dir>/lines/lines_<tag>.json に保存")
    ap.add_argument("--port", type=int, required=True, help="待ち受けポート（127.0.0.1 のみ）")
    args = ap.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    rows = load_rows(data_dir)
    os.makedirs(os.path.join(data_dir, "lines"), exist_ok=True)
    out_path = os.path.join(data_dir, "lines", f"lines_{args.tag}.json")
    state = load_or_init(out_path, rows, data_dir, args.tag)
    if not os.path.isfile(out_path):
        atomic_write_json(out_path, state)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(data_dir, rows, state, out_path))
    print(f"病変 {len(rows)} 件。保存先: {out_path}")
    print(f"ブラウザで http://127.0.0.1:{args.port} を開く（Ctrl+C で終了）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n終了")


if __name__ == "__main__":
    main()
