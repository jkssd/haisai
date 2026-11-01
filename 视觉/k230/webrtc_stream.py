# -*- coding: utf-8 -*-
"""K230 无线图传（WebRTC + 硬件 H.264）。

基于 CanMV 官方 examples/02-Media/webrtc_camera.py，改成可配置的正式版本：

  * 网络：支持 sta（板子连热点/路由器）和 ap（板子自己开热点）两种模式；
  * 参数：从 /sdcard/webrtc_config.json 读，文件不存在就用下面的默认值；
  * 观测：每秒打印 fps / 码率 / 会话数 / 编码耗时，便于压测；
  * 浏览器端：登录 http://<板子IP>:8080/ 即可看实时画面。

比赛建议用 ap 模式：现场网络不可控，板子自建热点是唯一稳定的链路。

配置示例（/sdcard/webrtc_config.json）：
{
  "mode": "ap",                      # "ap" 或 "sta"
  "ap_ssid": "K230-CAM",
  "ap_key": "k230cam88",
  "ssid": "<你的热点名>",             # mode=sta 时用
  "password": "<热点密码>",
  "width": 1280, "height": 720,
  "bit_rate": 3000,                  # Kbit/s
  "codec": "h264",                   # "h264" 或 "h265"
  "port": 8080
}
"""
from media.vencoder import *
from media.sensor import *
from media.media import *
import _thread
import json
import os
import select
import socket
import sys
import time
import ubinascii
import uctypes
import webrtc
import network

CONFIG_PATH = "/sdcard/webrtc_config.json"

DEFAULTS = {
    "mode": "sta",
    "ap_ssid": "K230-CAM",
    "ap_key": "k230cam88",
    "ssid": "",
    "password": "",
    "wlan_device": "auto",
    "width": 1280,
    "height": 720,
    "bit_rate": 3000,
    "bit_rate_max": 0,
    "codec": "h264",
    "fps": 30,
    "gop": 30,
    "sensor_fps": 60,
    "feed_all": False,
    "wdt": True,
    "ota": True,          # 允许 POST /upload 换板端脚本 + GET /reboot 重启（免串口）
    "port": 8080,
    "token": "",
    "kmodel": "",
    "detect_every": 2,
    "adaptive": True,
    "bit_rate_low": 600,
    "rssi_low": -72,
    "rssi_high": -62,
    "rssi_low_n": 2,
    "rssi_high_n": 6,
}

NEGOTIATION_TIMEOUT_MS = 10000
HTTP_REQUEST_TIMEOUT_MS = 2000
HTTP_MAX_CONNECTIONS = 8
HTTP_MAX_HEADER_BYTES = 16384
HTTP_MAX_BODY_BYTES = 65536
STATS_EVERY_MS = 1000

# --- 推流「突然掉到 0」的防护 ---
# 病因（板上实测）：客户端异常消失后 ICE 仍停在 COMPLETED，
# peer.is_connected() 一直为 True，但它的发送缓冲永远排不空，
# 于是 peer.send_video() 阻塞约 1 秒（日志里 get≈send≈966ms），
# 主循环被整个卡住 -> 接收端码率/帧率瞬间归零，然后自己恢复。
# 对策：单次发送超过 SLOW 就把该会话记为「卡住」，连续 STRIKES 次就隔离
# （隔离期内不发，每 PROBE 探一次活），彻底不再阻塞主循环；
# 超过 IDLE_DROP 都没有一次成功发送，就判定是僵尸会话直接踢掉。
PEER_SEND_SLOW_MS = 250       # 单次 send_video 超过这个时长才算「病态」
                              # （正常 2~5ms，抖动顶多几十 ms；真正的阻塞是 ~966ms，
                              #   阈值设 80 会把普通抖动误判成僵死，反而制造卡顿）
PEER_SEND_STRIKES = 1         # 出现一次病态阻塞就立刻隔离，不再让它拖第二下
PEER_QUARANTINE_MS = 1000     # 隔离时长（隔离期内不发帧，避免继续阻塞主循环）
PEER_PROBE_MS = 500           # 隔离期间探活间隔：一探成功就立刻恢复
PEER_IDLE_DROP_MS = 20000     # 这么久没成功发送过就踢掉（僵尸会话）
PEER_REAP_EVERY_MS = 2000     # 主循环里多久跑一次 reap


def load_config():
    cfg = dict(DEFAULTS)
    # 板上的 webrtc_config.json 可能被解释器占用（EBUSY/EPERM）改不动，
    # 所以优先读 config2，方便在不刷机的情况下调整参数。
    for path in ("/sdcard/webrtc_config2.json", CONFIG_PATH):
        try:
            with open(path, "r") as f:
                raw = f.read()
        except Exception:
            continue
        # 有些编辑器会写入 UTF-8 BOM，json.loads 会直接报语法错误
        if raw and raw[0] == "\ufeff":
            raw = raw[1:]
        try:
            user = json.loads(raw)
        except Exception as e:
            print("config: %s 解析失败（%s）" % (path, e))
            continue
        if isinstance(user, dict):
            for k, v in user.items():
                if k in cfg:
                    cfg[k] = v
                else:
                    print("config: 未知键 %s（已忽略）" % k)
            print("config: 已加载 %s" % path)
        break
    return cfg


CFG = load_config()
HTTP_PORT = int(CFG["port"])
ACCESS_TOKEN = CFG.get("token") or ""
MAX_CLIENTS = webrtc.MAX_PEERS
WIDTH = int(CFG["width"])
HEIGHT = int(CFG["height"])
BIT_RATE = int(CFG["bit_rate"])
VIDEO_CODEC = CFG["codec"]
TARGET_FPS = int(CFG.get("fps") or 0)
GOP_LEN = int(CFG.get("gop") or 30)
frame_interval_ms = int(1000.0 / TARGET_FPS) if TARGET_FPS > 0 else 0

PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>K230 无线图传</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b0b0c;color:#eee;font:14px/1.4 -apple-system,system-ui,"Segoe UI",sans-serif;height:100vh;display:flex;flex-direction:column}
header{height:48px;flex:0 0 48px;display:flex;align-items:center;gap:12px;padding:0 14px;background:#1a1a1c;border-bottom:1px solid #2c2c30}
header strong{margin-right:auto;font-weight:600;letter-spacing:.02em}
button{min-width:84px;height:32px;border:0;border-radius:6px;padding:0 14px;background:#1668e3;color:#fff;cursor:pointer;font-size:13px}
button:disabled{cursor:wait;opacity:.55}
#state{color:#9aa0a6;font-size:13px;white-space:nowrap}#state.ok{color:#4ade80}#state.warn{color:#fbbf24}#state.bad{color:#f87171}
.bar{display:flex;gap:1px;flex:0 0 auto;background:#2c2c30;border-bottom:1px solid #2c2c30;overflow-x:auto}
.m{flex:1 0 96px;padding:6px 10px;background:#151517}
.m span{display:block;color:#7c8388;font-size:10px;letter-spacing:.07em;text-transform:uppercase}
.m b{display:block;color:#e5e7eb;font-size:13px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.view{position:relative;flex:1;min-height:0;background:#000}
video{width:100%;height:100%;object-fit:contain;display:block}
.msg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#9aa0a6;font-size:15px;text-align:center;padding:24px;background:rgba(0,0,0,.6)}
.msg.hide{display:none}
</style></head><body>
<header><strong>K230 无线图传</strong><span id="state">未连接</span><button id="go">连接</button></header>
<div class="bar">
<div class="m"><span>分辨率</span><b id="s-size">--</b></div>
<div class="m"><span>接收码率</span><b id="s-kbps">--</b></div>
<div class="m"><span>帧率</span><b id="s-fps">--</b></div>
<div class="m"><span>累计丢包</span><b id="s-lost">--</b></div>
<div class="m"><span>往返时延</span><b id="s-rtt">--</b></div>
<div class="m"><span>抖动</span><b id="s-jit">--</b></div>
<div class="m"><span>首帧耗时</span><b id="s-first">--</b></div>
</div>
<main class="view"><video id="v" autoplay playsinline muted></video><div id="msg" class="msg">点击「连接」开始接收画面</div></main>
<script>
let pc=null,n=0,timer=null,last=null,t0=0,gotFirst=false;
const $=id=>document.querySelector(id.startsWith('#')?id:'#'+id);
const st=$('state'),msg=$('msg'),btn=$('go'),vid=$('v');
const setState=(t,k)=>{st.textContent=t;st.className=k||''};
const tok=new URLSearchParams(location.search).get('token')||'';
const api=p=>tok?p+(p.includes('?')?'&':'?')+'token='+encodeURIComponent(tok):p;
function reset(){last=null;gotFirst=false;t0=performance.now();['s-size','s-kbps','s-fps','s-lost','s-rtt','s-jit','s-first'].forEach(i=>$(i).textContent='--')}
async function stats(){
  if(!pc||n!==pc._n)return;
  let rep;try{rep=await pc.getStats()}catch(e){return}
  let inb=null,pair=null,pid=null;
  rep.forEach(r=>{if(r.type==='inbound-rtp'&&!r.isRemote&&r.kind==='video')inb=r;if(r.type==='transport'&&r.selectedCandidatePairId)pid=r.selectedCandidatePairId});
  if(pid)pair=rep.get(pid);
  if(!pair)rep.forEach(r=>{if(!pair&&r.type==='candidate-pair'&&r.state==='succeeded')pair=r});
  if(inb){
    const now=performance.now(),cur={t:now,b:inb.bytesReceived||0,f:inb.framesDecoded||0};
    if(last){
      const dt=(now-last.t)/1000;
      if(dt>0){
        $('s-kbps').textContent=Math.max(0,Math.round((cur.b-last.b)*8/dt/1000))+' kbit/s';
        $('s-fps').textContent=Math.max(0,((cur.f-last.f)/dt)).toFixed(1)+' fps';
      }
    }
    last=cur;
    $('s-lost').textContent=(inb.packetsLost||0);
    $('s-jit').textContent=inb.jitter!==undefined?Math.round(inb.jitter*1000)+' ms':'--';
    if(vid.videoWidth)$('s-size').textContent=vid.videoWidth+'x'+vid.videoHeight;
    if(!gotFirst&&inb.framesDecoded>0){gotFirst=true;$('s-first').textContent=Math.round(performance.now()-t0)+' ms'}
  }
  if(pair&&pair.currentRoundTripTime!==undefined)$('s-rtt').textContent=Math.round(pair.currentRoundTripTime*1000)+' ms';
  window.__stats={t:Math.round(performance.now()),state:pc?pc.connectionState:'none',
    kbps:parseInt($('s-kbps').textContent)||0,fps:parseFloat($('s-fps').textContent)||0,
    lost:parseInt($('s-lost').textContent)||0,rtt:parseInt($('s-rtt').textContent)||-1,
    jitter:parseInt($('s-jit').textContent)||-1,size:$('s-size').textContent,
    first:parseInt($('s-first').textContent)||-1,frames:inb?inb.framesDecoded:0,
    bytes:inb?inb.bytesReceived:0};
}
async function connect(){
  const id=++n;reset();btn.disabled=true;btn.textContent='连接中';setState('协商中','warn');msg.classList.remove('hide');msg.textContent='正在与板子交换 SDP…';
  try{
    const offer=await fetch(api('/offer'));
    if(!offer.ok)throw new Error('offer HTTP '+offer.status);
    const sid=offer.headers.get('X-WebRTC-Session');
    const sdp=await offer.text();
    if(id!==n)return;
    const p=new RTCPeerConnection({iceServers:[]});pc=p;p._n=id;
    p.ontrack=e=>{vid.srcObject=e.streams&&e.streams[0]||new MediaStream([e.track])};
    p.onconnectionstatechange=()=>{
      if(id!==p._n)return;
      const s=p.connectionState;
      if(s==='connected'){setState('已连接','ok');msg.classList.add('hide');btn.disabled=false;btn.textContent='重连';
        if(!timer)timer=setInterval(stats,1000);stats();}
      else if(s==='failed'){setState('连接失败','bad');msg.classList.remove('hide');msg.textContent='协商失败，请点「重连」';btn.disabled=false;btn.textContent='重连'}
      else if(s==='disconnected'){setState('链路中断','warn')}
    };
    await p.setRemoteDescription({type:'offer',sdp});
    const ans=await p.createAnswer();
    await p.setLocalDescription(ans);
    // 等候选：以前等 iceGatheringState==='complete'，本机有多个网卡时要等到 2.5s 超时，
    // 首帧全耗在这。其实拿到第一个候选 + 300ms 就够（板子只连一个网段）。
    await new Promise(r=>{
      if(p.iceGatheringState==='complete')return r();
      let done=false,t1=null,t2=null;
      const fin=()=>{if(done)return;done=true;clearTimeout(t1);clearTimeout(t2);
        p.removeEventListener('icecandidate',on);p.removeEventListener('icegatheringstatechange',hs);r()};
      const on=e=>{if(e&&e.candidate&&!t1)t1=setTimeout(fin,300)};
      const hs=()=>{if(p.iceGatheringState==='complete')fin()};
      p.addEventListener('icecandidate',on);p.addEventListener('icegatheringstatechange',hs);
      t2=setTimeout(fin,2000);
    });
    if(id!==n)return;
    const r2=await fetch(api('/answer?session='+encodeURIComponent(sid)),{method:'POST',body:p.localDescription.sdp});
    if(!r2.ok)throw new Error('answer HTTP '+r2.status);
    setState('等待首帧','warn');msg.textContent='已建立，等待编码帧…';
  }catch(e){
    if(id!==n)return;
    setState('失败','bad');msg.classList.remove('hide');msg.textContent='连接失败：'+e.message;btn.disabled=false;btn.textContent='重试';
  }
}
vid.addEventListener('playing',()=>{if(pc)setState('播放中','ok');msg.classList.add('hide')});
btn.onclick=()=>{if(pc){try{pc.close()}catch(e){}pc=null;clearInterval(timer);timer=null}connect()};
</script></body></html>"""


def send_all(sock, data):
    offset = 0
    while offset < len(data):
        sent = sock.send(data[offset:])
        if sent <= 0:
            raise OSError("socket send failed")
        offset += sent


def send_response(client, status, content_type, body, session_id=None):
    session_header = "X-WebRTC-Session: %s\r\n" % session_id if session_id else ""
    header = ("HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %d\r\n"
              "Cache-Control: no-store\r\nConnection: close\r\n%s\r\n" %
              (status, content_type, len(body), session_header)).encode()
    send_all(client, header)
    send_all(client, body)


def parse_request(data):
    split = data.find(b"\r\n\r\n")
    if split < 0:
        return False if len(data) >= HTTP_MAX_HEADER_BYTES else None
    try:
        header = data[:split].decode()
    except UnicodeError:
        return False
    body = data[split + 4:]
    lines = header.split("\r\n")
    request_line = lines[0].split(" ", 2)
    if len(request_line) != 3:
        return False
    method, path, _ = request_line
    content_length = 0
    for line in lines[1:]:
        if line.lower().startswith("content-length:"):
            try:
                content_length = int(line.split(":", 1)[1].strip())
            except ValueError:
                return False
            break
    if content_length < 0 or content_length > HTTP_MAX_BODY_BYTES:
        return False
    if len(body) < content_length:
        return None
    return method, path, body[:content_length]


def replace_mdns_candidates(sdp, client_ip):
    lines = sdp.split("\r\n")
    for index in range(len(lines)):
        if not lines[index].startswith("a=candidate:"):
            continue
        fields = lines[index].split(" ")
        if len(fields) > 4 and fields[4].endswith(".local"):
            fields[4] = client_ip
            lines[index] = " ".join(fields)
    return "\r\n".join(lines)


def _slog(msg):
    """会话事件日志（默认关闭）。

    注意：绝对不能在 HTTP 处理流程里同步写 SD 卡——媒体管线同时在跑时，
    文件 I/O 会把 HTTP 线程卡死（实测表现为页面永远停在「交换 SDP」，
    随后看门狗复位）。只有排查问题时才打开 CFG["debug_log"]。
    """
    if not CFG.get("debug_log"):
        return
    try:
        import os
        p = "/sdcard/_sess.log"
        try:
            if os.stat(p)[6] > 20000:
                os.remove(p)
        except Exception:
            pass
        f = open(p, "a")
        try:
            f.write("%d %s\n" % (time.ticks_ms(), msg))
        finally:
            f.close()
    except Exception:
        pass


def random_hex(byte_count=16):
    return ubinascii.hexlify(webrtc.random_bytes(byte_count)).decode()


def parse_request_target(path):
    route, _, query = path.partition("?")
    params = {}
    for item in query.split("&"):
        if not item:
            continue
        key, separator, value = item.partition("=")
        if separator and key not in params:
            params[key] = value
    return route, params


def _apply_default_dev(itf, tag):
    """把默认网卡指向对应接口。

    RT-Smart 上发包要用「默认网卡」：如果它是 None，服务能 bind 成功、本机自连
    也正常，但发给外部客户端的数据包发不出去（浏览器一直转圈、ping 不通）。

    不能盲目设置：早期版本在 AP 模式下调 set_default_dev('wlan0ap') 失败，
    反而把默认网卡清成 None。所以设置后必须回读校验，失败就回退。
    """
    cur = None
    try:
        cur = network.get_default_dev()
    except Exception as e:
        print("网络: get_default_dev 失败 %s" % (e,))

    name = None
    try:
        name = network.get_netdev_name(itf)
    except Exception as e:
        print("网络: get_netdev_name 失败 %s" % (e,))
    if not name:
        name = "wlan0ap" if itf == network.AP_IF else "wlan0"

    if cur == name:
        print("网络: 默认网卡已是 %s [%s]" % (name, tag))
        return name

    ok = False
    try:
        ok = network.set_default_dev(name)
    except Exception as e:
        print("网络: set_default_dev(%s) 异常 %s" % (name, e))

    now = None
    try:
        now = network.get_default_dev()
    except Exception:
        pass
    print("网络: 默认网卡 %s -> %s (set 返回 %s, 当前 %s) [%s]"
          % (cur, name, ok, now, tag))

    if now is None and cur is not None:
        try:
            network.set_default_dev(cur)
            print("网络: 设置失败，已回退默认网卡到 %s" % (cur,))
        except Exception as e:
            print("网络: 回退默认网卡失败 %s" % (e,))
    return now or cur


def _wlan_device():
    """把配置里的 wlan_device 映射成驱动常量（auto/usb/sdio/spi）。"""
    name = str(CFG.get("wlan_device") or "auto").lower()
    table = {
        "auto": getattr(network, "WLAN_AUTO", 0),
        "usb": getattr(network, "WLAN_USB", 1),
        "sdio": getattr(network, "WLAN_SDIO", 2),
        "spi": getattr(network, "WLAN_SPI", 3),
    }
    return table.get(name, table["auto"])


def _start_ap():
    """开热点并返回 IP。"""
    dev = _wlan_device()
    ap = network.WLAN(network.AP_IF, dev)
    ap.active(True)
    time.sleep(1)
    ap.config(ssid=CFG["ap_ssid"], key=CFG["ap_key"])
    time.sleep(2)
    ip = ap.ifconfig()[0]
    print("网络: AP 模式 ssid=%s ip=%s" % (CFG["ap_ssid"], ip))
    # 必须设置默认网卡为 wlan0ap，否则发包没有可用的出口网卡
    _apply_default_dev(network.AP_IF, "ap")
    return ip


def setup_network():
    """返回 ip。ap 自己开热点；sta 连外部 AP；auto 先试 sta，失败自动退 AP。"""
    mode = CFG["mode"]
    dev = _wlan_device()
    if mode == "ap":
        return _start_ap()

    w = network.WLAN(network.STA_IF, dev)
    w.active(True)
    time.sleep(1)
    ssid = CFG["ssid"]
    pwd = CFG["password"]
    if w.isconnected():
        pass
    else:
        # 扫描一次，用驱动返回的原始 SSID（避免中文 SSID 编码不匹配）
        target = ssid
        try:
            scanned = w.scan()
            exact = None
            loose = None
            for ap in scanned:
                try:
                    s = ap.ssid
                except Exception:
                    continue
                # 不同固件版本的 scan 结果可能是 bytes 也可能是 str，统一成 str 再比
                if isinstance(s, bytes):
                    s = s.decode("utf-8", "replace")
                if not s:
                    continue
                if s == ssid:
                    exact = s
                    break
                if loose is None and ssid and (ssid in s or s in ssid):
                    loose = s
            if exact is not None:
                target = exact
            elif loose is not None:
                target = loose
                print("网络: SSID 精确匹配失败，用相近的 %r" % (loose,))
        except Exception as e:
            print("网络: 扫描失败 %s" % (e,))
        if not target:
            if mode == "auto":
                print("网络: 没配置 ssid，自动退到 AP 模式")
                return _start_ap()
            raise RuntimeError("没有配置 ssid，无法连接")
        print("网络: 连接 %r ..." % (target,))
        try:
            w.disconnect()
        except Exception:
            pass
        time.sleep_ms(300)
        try:
            w.connect(target, pwd)
        except Exception as e:
            print("网络: connect 异常 %s" % (e,))
        # auto 模式少等一会儿，尽快回退到 AP
        _wait = 30 if mode == "auto" else 80
        for _ in range(_wait):
            if w.isconnected():
                break
            time.sleep_ms(500)
    if not w.isconnected():
        if mode == "auto":
            print("网络: STA 连接超时，自动退到 AP 模式")
            # 注意：这里不要调用 w.active(False)，RT-Smart 上关网卡再开 AP 会死锁
            return _start_ap()
        raise RuntimeError("WiFi 连接超时")
    ifconf = w.ifconfig()
    print("网络: STA 模式 ip=%s gw=%s rssi=%s" % (ifconf[0], ifconf[2], w.status("rssi") if hasattr(w, "status") else "?"))
    # 和 AP 模式保持一致：不要动默认网卡。
    # 实测 STA 模式下调用 set_default_dev('wlan0') 后，TCP 还能通（HTTP 页面正常），
    # 但 libpeer 的 UDP/ICE 会打不通，协商一直停在 checking 然后 failed。
    if CFG.get("set_default_dev"):
        _apply_default_dev(network.STA_IF, "sta")
    else:
        print("网络: STA 模式跳过默认网卡设置（与 AP 模式一致）")
    return ifconf[0]


def _state_name(peer):
    try:
        return peer.state_name()
    except Exception:
        try:
            return str(peer.state())
        except Exception:
            return "?"


class SignalingServer:
    def __init__(self, video_codec):
        self.video_codec = video_codec
        self.sessions = {}
        self.access_token = ACCESS_TOKEN
        self.lock = _thread.allocate_lock()
        self.done = _thread.allocate_lock()
        self.stopping = False
        self.started = False
        self.listeners = {}
        self.clients = {}
        self.poll = select.poll()
        self.local_ip = None
        # 画质档切换（手机访问 /near 或 /far 触发）
        self.requested_profile = None
        self.current_profile = None
        self.current_rate = 0
        self.current_rssi = None
        self.offer_time = {}
        # 主循环每秒刷一次，/status 里带出来（不接串口也能看帧率/码率/各段耗时）
        self.stats = {}
        # /reboot 用：主循环看到这个时间戳到了就 machine.reset()
        self.reboot_at = None

    def start(self):
        try:
            for ip in self.local_ips():
                listener = socket.socket()
                try:
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    listener.bind((ip, HTTP_PORT))
                    listener.listen(MAX_CLIENTS)
                    listener.setblocking(False)
                    self.poll.register(listener, select.POLLIN)
                except OSError as error:
                    listener.close()
                    print("HTTP bind 失败 %s: %s" % (ip, error))
                    continue
                self.listeners[listener] = ip
                if self.access_token:
                    print("图传地址: http://%s:%d/?token=%s" % (ip, HTTP_PORT, self.access_token))
                else:
                    print("图传地址: http://%s:%d/" % (ip, HTTP_PORT))
            if not self.listeners:
                raise RuntimeError("没有可用的 HTTP 监听地址")
            self.done.acquire()
            try:
                _thread.start_new_thread(self.serve, ())
            except BaseException:
                self.done.release()
                raise
            self.started = True
        except BaseException:
            self.close()
            raise

    def local_ips(self):
        seen = []
        out = []
        # 只取真正有地址的 WLAN 接口，别去动默认网卡（那会让 UDP 走错口）
        for itf in (network.STA_IF, network.AP_IF):
            try:
                wl = network.WLAN(itf)
                cfg = wl.ifconfig()
                if cfg and cfg[0] and cfg[0] != "0.0.0.0" and cfg[0] not in seen:
                    seen.append(cfg[0])
                    out.append(cfg[0])
            except Exception:
                continue
        if not out:
            out.append("0.0.0.0")
        return out

    def drop(self, session_id):
        session = self.sessions.pop(session_id, None)
        if session is None:
            return
        try:
            session["peer"].close()
        except Exception:
            pass

    def close_client(self, client):
        try:
            self.poll.unregister(client)
        except OSError:
            pass
        self.clients.pop(client, None)
        client.close()

    def accept_client(self, listener):
        try:
            client, address = listener.accept()
        except OSError:
            return
        if len(self.clients) >= HTTP_MAX_CONNECTIONS:
            client.close()
            return
        client.setblocking(False)
        self.clients[client] = {
            "address": address[0],
            "local": self.listeners[listener],
            "data": b"",
            "deadline": time.ticks_add(time.ticks_ms(), HTTP_REQUEST_TIMEOUT_MS),
        }
        events = select.POLLIN
        if hasattr(select, "POLLERR"):
            events |= select.POLLERR
        if hasattr(select, "POLLHUP"):
            events |= select.POLLHUP
        self.poll.register(client, events)

    def service_client(self, client):
        connection = self.clients.get(client)
        if connection is None:
            return
        try:
            chunk = client.recv(2048)
        except OSError:
            return
        if not chunk:
            self.close_client(client)
            return
        connection["data"] += chunk
        connection["deadline"] = time.ticks_add(time.ticks_ms(), HTTP_REQUEST_TIMEOUT_MS)
        request = parse_request(connection["data"])
        if request is None:
            return
        try:
            client.settimeout(2)
            if request is False:
                send_response(client, "400 Bad Request", "text/plain", b"Bad Request")
            else:
                response = self.handle(*request, connection["address"], connection["local"])
                send_response(client, *response)
        except OSError as error:
            if not error.args or error.args[0] not in (32, 104):
                sys.print_exception(error)
        except BaseException as error:
            sys.print_exception(error)
            try:
                send_response(client, "500 Internal Server Error", "text/plain", b"signaling failed")
            except OSError:
                pass
        finally:
            self.close_client(client)

    def reap(self):
        now = time.ticks_ms()
        for session_id, session in list(self.sessions.items()):
            state = session["peer"].state()
            if state in (webrtc.STATE_CLOSED, webrtc.STATE_FAILED, webrtc.STATE_DISCONNECTED):
                _slog("REAP %s state=%s (closed/failed)" % (session_id[:8], _state_name(session["peer"])))
                self.drop(session_id)
            elif state != webrtc.STATE_COMPLETED and time.ticks_diff(now, session["activity"]) > NEGOTIATION_TIMEOUT_MS:
                _slog("REAP %s state=%s (negotiation timeout)" % (session_id[:8], _state_name(session["peer"])))
                self.drop(session_id)
            else:
                # 老代码只按 state 判断，COMPLETED 的会话永远不回收。
                # 但客户端异常消失（关页面/掉 WiFi/手机休眠）时 ICE 会长期停在
                # COMPLETED，is_connected() 仍是 True，发送缓冲排不空 -> 堵死主循环。
                # 这里用「最近一次成功发送」当心跳，太久没成功就认定是僵尸会话。
                last_ok = session.get("last_ok")
                if last_ok is None:
                    last_ok = session.get("activity", now)
                if time.ticks_diff(now, last_ok) > PEER_IDLE_DROP_MS:
                    _slog("REAP %s state=%s (僵尸会话：%d ms 没有发送成功)"
                          % (session_id[:8], _state_name(session["peer"]),
                             time.ticks_diff(now, last_ok)))
                    self.drop(session_id)

    def reap_safe(self):
        """主循环用：拿不到锁就直接跳过，绝不阻塞画面。"""
        if not self.lock.acquire(False):
            return
        try:
            self.reap()
        except Exception:
            pass
        finally:
            self.lock.release()

    def handle(self, method, path, body, client_ip, local_ip):
        route, params = parse_request_target(path)
        if self.access_token and params.get("token") != self.access_token:
            return "403 Forbidden", "text/plain", b"Forbidden", None
        if method == "OPTIONS":
            return "204 No Content", "text/plain", b"", None
        if method == "GET" and route == "/":
            print("页面请求来自 %s" % (client_ip,))
            _slog("PAGE %s" % (client_ip,))
            return "200 OK", "text/html", PAGE, None
        if method == "GET" and route in ("/near", "/mid", "/far", "/micro"):
            # 一键切换画质档：手机浏览器直接访问即可
            self.requested_profile = route[1:]
            print("画质档切换请求: %s" % self.requested_profile)
            return ("200 OK", "text/plain",
                    ("切换到 %s 档\n" % self.requested_profile).encode(), None)
        if method == "GET" and route == "/status":
            st = self.stats or {}
            return ("200 OK", "application/json",
                    ('{"mode":"%s","profile":"%s","rate":%d,"rssi":%s,"clients":%d,'
                     '"fps":%.1f,"kbps":%.0f,"sess":"%s","st":"%s"}'
                     % (CFG["mode"], self.current_profile or "?", self.current_rate or 0,
                        self.current_rssi if self.current_rssi is not None else "null",
                        self.connected_count(), st.get("fps", 0.0), st.get("kbps", 0.0),
                        st.get("sess", "-"), st.get("detail", ""))).encode(), None)
        if method == "POST" and route == "/upload":
            # 不开串口也能换板端脚本：curl --data-binary @pose_webrtc.py
            #   "http://<板子>:8080/upload?path=/sdcard/pose_webrtc.py"
            # 先 compile() 过一遍语法，能过才落盘 —— 防止传上去一个跑不起来的脚本
            # 把板子搞成哑巴（那时只能拔卡）。
            if not CFG.get("ota"):
                return "403 Forbidden", "text/plain", b"ota disabled", None
            target = params.get("path") or ""
            if not (target.startswith("/sdcard/") and
                    (target.endswith(".py") or target.endswith(".json"))):
                return "400 Bad Request", "text/plain", b"path must be /sdcard/*.py|*.json", None
            if not body:
                return "400 Bad Request", "text/plain", b"empty body", None
            if target.endswith(".py"):
                try:
                    compile(body.decode("utf-8"), target, "exec")
                except BaseException as error:
                    return ("400 Bad Request", "text/plain",
                            ("syntax error: %s\n" % (error,)).encode(), None)
            tmp = target + ".new"
            try:
                f = open(tmp, "wb")
                try:
                    f.write(body)
                finally:
                    f.close()
                try:
                    os.remove(target)
                except Exception:
                    pass
                os.rename(tmp, target)
            except Exception as error:
                return ("500 Internal Server Error", "text/plain",
                        ("write failed: %s\n" % (error,)).encode(), None)
            print("OTA: %s 已更新 %d 字节" % (target, len(body)))
            return ("200 OK", "text/plain",
                    ("%s %d bytes\n" % (target, len(body))).encode(), None)
        if method == "GET" and route == "/reboot":
            if not CFG.get("ota"):
                return "403 Forbidden", "text/plain", b"ota disabled", None
            # 留 400ms 让这个响应先发出去，主循环到点再复位
            self.reboot_at = time.ticks_add(time.ticks_ms(), 400)
            return "200 OK", "text/plain", b"rebooting\n", None
        if method == "GET" and route == "/offer":
            # 限流：同一个客户端 3 秒内只能建一次会话。
            # 否则旧标签页反复重连会留下多个连不通的 ICE 会话，探测线程把 CPU 吃光，
            # 表现为整机卡顿（实测 fps 从 33 掉到 5）。
            _now = time.ticks_ms()
            _last = self.offer_time.get(client_ip)
            if _last is not None and time.ticks_diff(_now, _last) < 3000:
                _slog("OFFER-RATE-LIMIT %s" % (client_ip,))
                return "429 Too Many Requests", "text/plain", b"slow down", None
            self.offer_time[client_ip] = _now
            with self.lock:
                self.reap()
                if len(self.sessions) >= MAX_CLIENTS:
                    return "503 Service Unavailable", "text/plain", b"too many clients", None
                # 同一个客户端重新打开页面时，它上一次的会话（COMPLETED 僵尸）
                # 会一直占着发送缓冲。新 offer 进来就把它踢掉，避免越积越多
                # 到最后每个会话都堵一下主循环。只动同一个 client_ip 且已经
                # 停止发送超过 3 秒的会话，不影响正常的多设备同时观看。
                _tnow = time.ticks_ms()
                for _sid, _s in list(self.sessions.items()):
                    if _s.get("client") != client_ip:
                        continue
                    _ok = _s.get("last_ok") or _s.get("activity", _tnow)
                    if time.ticks_diff(_tnow, _ok) > 3000:
                        _slog("REAP %s（同 IP 旧会话，新 offer 顶掉）" % _sid[:8])
                        self.drop(_sid)
                session_id = random_hex()
                while session_id in self.sessions:
                    session_id = random_hex()

            peer = webrtc.PeerConnection(video_codec=self.video_codec,
                                         audio_codec=webrtc.CODEC_NONE, local_ip=local_ip)
            try:
                offer_sdp = peer.create_offer()
                _dump_candidates(offer_sdp, "offer")
                offer = offer_sdp.encode()
                with self.lock:
                    self.sessions[session_id] = {"peer": peer, "client": client_ip,
                        "local": local_ip, "activity": time.ticks_ms(), "ready": False,
                        "last_ok": time.ticks_ms(), "strikes": 0, "stalled_until": 0}
            except BaseException:
                peer.close()
                raise
            print("会话 %s: %s -> %s (offer %d 字节, 状态 %s)"
                  % (session_id[:8], local_ip, client_ip, len(offer), _state_name(peer)))
            _slog("OFFER %s local=%s client=%s %d bytes" %
                  (session_id[:8], local_ip, client_ip, len(offer)))
            for _l in offer_sdp.split("\n"):
                if _l.startswith("a=candidate") or _l.startswith("a=ice-ufrag") or _l.startswith("a=setup"):
                    _slog("  O| " + _l.strip()[:110])
            return "200 OK", "application/sdp", offer, session_id

        peer = None
        with self.lock:
            self.reap()
            if method != "POST" or route not in ("/answer", "/disconnect"):
                return "404 Not Found", "text/plain", b"Not Found", None
            session_id = params.get("session")
            if not session_id or len(session_id) != 32:
                return "400 Bad Request", "text/plain", b"missing session", None
            session = self.sessions.get(session_id)
            if session is None:
                return "404 Not Found", "text/plain", b"session not found", None
            if session["client"] != client_ip or session["local"] != local_ip:
                return "403 Forbidden", "text/plain", b"session mismatch", None
            if route == "/disconnect":
                self.drop(session_id)
                return "200 OK", "text/plain", b"OK", None
            if not body:
                return "400 Bad Request", "text/plain", b"missing sdp", None
            peer = session["peer"]

        # 关键：set_remote_description 会做 ICE 初始化，可能阻塞数百毫秒甚至更久。
        # 必须在锁外调用，否则主循环的 send_video 会被拖死（表现为整机卡住）。
        answer = replace_mdns_candidates(body.decode(), client_ip)
        n_mdns = body.decode().count(".local")
        _dump_candidates(body.decode(), "answer")
        peer.set_remote_description(answer)
        st = peer.state()
        print("会话 %s: 收到 answer %d 字节 (mDNS 候选 %d 个) -> 状态 %s"
              % (session_id[:8], len(body), n_mdns, _state_name(peer)))
        _slog("ANSWER %s %d bytes mdns=%d state=%s ip=%s" %
              (session_id[:8], len(body), n_mdns, _state_name(peer), client_ip))
        for _l in body.decode().split("\n"):
            if _l.startswith("a=candidate") or _l.startswith("a=setup"):
                _slog("  A| " + _l.strip()[:110])
        if st == webrtc.STATE_FAILED:
            with self.lock:
                self.drop(session_id)
            return "400 Bad Request", "text/plain", b"bad sdp", None
        with self.lock:
            session = self.sessions.get(session_id)
            if session is not None:
                session["activity"] = time.ticks_ms()
            return "200 OK", "text/plain", b"OK", None

    def serve(self):
        try:
            while not self.stopping:
                with self.lock:
                    self.reap()
                for item, event in self.poll.poll(50):
                    if self.stopping:
                        break
                    if item in self.listeners:
                        if event & select.POLLIN:
                            self.accept_client(item)
                    elif item in self.clients:
                        if event & select.POLLIN:
                            self.service_client(item)
                        elif event:
                            self.close_client(item)
                now = time.ticks_ms()
                for client, connection in list(self.clients.items()):
                    if time.ticks_diff(now, connection["deadline"]) >= 0:
                        self.close_client(client)
        finally:
            self.stopping = True
            for client in list(self.clients):
                self.close_client(client)
            self.done.release()

    def need_keyframe(self):
        # 主循环每帧都会调用，绝不能被 HTTP 线程的长操作拖住：
        # 拿不到锁就直接返回 False，下一帧再判断。
        if not self.lock.acquire(False):
            return False
        try:
            needed = False
            for session in self.sessions.values():
                if session["peer"].is_connected() and not session["ready"]:
                    session["ready"] = True
                    needed = True
            return needed
        finally:
            self.lock.release()

    def connected_count(self):
        if not self.lock.acquire(False):
            return 0
        try:
            n = 0
            for session in self.sessions.values():
                if session["peer"].is_connected():
                    n += 1
            return n
        finally:
            self.lock.release()

    def session_states(self):
        if not self.lock.acquire(False):
            return "busy"
        try:
            if not self.sessions:
                return "-"
            return ",".join(_state_name(s["peer"]) for s in self.sessions.values())
        finally:
            self.lock.release()

    def send_video(self, data, timestamp):
        # 不能被阻塞，但也不能随手丢帧：丢一帧就是一个 RTP 序号空洞，
        # 接收端参考帧断链 -> 直到下一个 IDR 之前都是花屏。
        # 所以给一个很短的退避重试窗口（最长 6ms），实在拿不到锁才放弃。
        got = self.lock.acquire(False)
        for _ in range(3):
            if got:
                break
            time.sleep_ms(2)
            got = self.lock.acquire(False)
        if not got:
            return False
        try:
            now = time.ticks_ms()
            sent = False
            for session_id, session in list(self.sessions.items()):
                peer = session["peer"]
                try:
                    if not peer.is_connected():
                        continue
                except Exception:
                    continue
                # 隔离期：跳过这个会话，只按 PROBE 间隔试一次，避免再次长时间阻塞
                until = session.get("stalled_until", 0)
                if until and time.ticks_diff(until, now) > 0:
                    if time.ticks_diff(now, session.get("probe_at", 0)) < 0:
                        continue
                session["probe_at"] = time.ticks_add(now, PEER_PROBE_MS)
                t_send = time.ticks_ms()
                try:
                    peer.send_video(data, timestamp)
                except Exception:
                    _slog("SEND ERR %s -> 踢掉" % session_id[:8])
                    self.drop(session_id)
                    continue
                cost = time.ticks_diff(time.ticks_ms(), t_send)
                if cost <= PEER_SEND_SLOW_MS:
                    # 正常：清掉卡顿计数；如果之前在隔离，说明链路恢复了，要补一个 IDR
                    if session.get("stalled_until"):
                        session["ready"] = False
                        _slog("RECOVER %s cost=%dms -> 请求 IDR" % (session_id[:8], cost))
                    session["stalled_until"] = 0
                    session["strikes"] = 0
                    session["last_ok"] = time.ticks_ms()
                else:
                    session["strikes"] = session.get("strikes", 0) + 1
                    if session["strikes"] >= PEER_SEND_STRIKES:
                        session["stalled_until"] = time.ticks_add(
                            time.ticks_ms(), PEER_QUARANTINE_MS)
                        _slog("ISOLATE %s cost=%dms strikes=%d（暂停给该会话发帧）"
                              % (session_id[:8], cost, session["strikes"]))
                sent = True
            return sent
        finally:
            self.lock.release()

    def close(self):
        self.stopping = True
        if self.started:
            self.done.acquire()
            self.done.release()
            self.started = False
        for listener in self.listeners:
            try:
                self.poll.unregister(listener)
            except OSError:
                pass
            listener.close()
        self.listeners.clear()
        with self.lock:
            for session_id in list(self.sessions):
                self.drop(session_id)


def _dump_candidates(sdp, tag):
    """打印 SDP 里的 ICE 候选地址，用来判断对端发的是哪个网段的地址。"""
    ips = []
    for line in sdp.split("\r\n"):
        if line.startswith("a=candidate:"):
            f = line.split(" ")
            if len(f) > 5:
                ip = f[4]
                proto = f[2] if len(f) > 2 else "?"
                if "%s/%s" % (ip, proto) not in ips:
                    ips.append("%s/%s" % (ip, proto))
    print("%s 候选(%d): %s" % (tag, len(ips), ",".join(ips[:8]) if ips else "无"))


def _self_check(ip, port):
    """板子自己访问自己的页面：先排除“服务没起来”，再怀疑链路。"""
    try:
        s = socket.socket()
        s.settimeout(5)
        s.connect((ip, port))
        s.send(b"GET / HTTP/1.0\r\nHost: k230\r\n\r\n")
        head = s.recv(64)
        s.close()
        print("自检: http://%s:%d/ -> %r" % (ip, port, head[:24]))
    except Exception as e:
        print("自检失败: %s %s（说明服务没在 %s:%d 上监听）" % (type(e).__name__, e, ip, port))


def _pick_ssid(w, want):
    """用扫描结果里的原始 SSID（驱动可能返回 bytes），避免中文编码不匹配。"""
    target = want
    try:
        exact = None
        loose = None
        for ap in w.scan():
            try:
                s = ap.ssid
            except Exception:
                continue
            if isinstance(s, bytes):
                s = s.decode("utf-8", "replace")
            if not s:
                continue
            if s == want:
                exact = s
                break
            if loose is None and want and (want in s or s in want):
                loose = s
        if exact is not None:
            target = exact
        elif loose is not None:
            target = loose
    except Exception as e:
        print("网络: 扫描失败 %s" % (e,))
    return target


def _wifi_watchdog(state):
    """后台盯着 STA 连接，掉了就重连。

    注意：网卡对象必须只创建一次。每轮循环都调用 network.WLAN(...) 会重新初始化
    网卡对象，导致无线驱动周期性停顿——实测表现为每 4~5 秒出现一次 600ms 的 RTT
    尖峰，画面跟着卡顿花屏。这里改成循环外创建、循环内只读状态。
    """
    w = None
    try:
        w = network.WLAN(network.STA_IF, _wlan_device())
    except Exception as e:
        print("网络: 看门狗获取网卡失败 %s" % (e,))
        return

    while not state["stop"]:
        try:
            if CFG["mode"] in ("sta", "auto") and not w.isconnected():
                print("网络: 检测到掉线，开始重连…")
                target = _pick_ssid(w, CFG["ssid"])
                try:
                    w.disconnect()
                except Exception:
                    pass
                time.sleep_ms(300)
                try:
                    w.connect(target, CFG["password"])
                except Exception as e:
                    print("网络: connect 异常 %s" % (e,))
                for _ in range(40):
                    if w.isconnected():
                        break
                    time.sleep_ms(500)
                if w.isconnected():
                    print("网络: 重连成功 ip=%s" % (w.ifconfig()[0],))
                else:
                    print("网络: 重连失败，5 秒后再试")
        except Exception as e:
            print("网络: watchdog 异常 %s" % (e,))
        time.sleep(5)


def run():
    width = ALIGN_UP(WIDTH, 16)
    sensor = None
    encoder = None
    link = None
    server = None
    sent_bytes = 0
    frames = 0
    t_stat = time.ticks_ms()
    last_sent_ms = 0

    ip = setup_network()

    try:
        if VIDEO_CODEC == "h265":
            peer_video_codec = webrtc.CODEC_H265
            payload_type = Encoder.PAYLOAD_TYPE_H265
            profile = Encoder.H265_PROFILE_MAIN
        else:
            peer_video_codec = webrtc.CODEC_H264
            payload_type = Encoder.PAYLOAD_TYPE_H264
            profile = Encoder.H264_PROFILE_MAIN

        server = SignalingServer(peer_video_codec)
        sensor = Sensor()
        sensor.reset()
        sensor.set_framesize(width=width, height=HEIGHT, alignment=12)
        sensor.set_pixformat(Sensor.YUV420SP)

        encoder = Encoder()
        encoder.SetOutBufs(8, width, HEIGHT)
        # 硬件限帧：把编码器输出帧率直接压到目标值（比在 Python 里丢帧干净得多）
        dst_fps = TARGET_FPS if TARGET_FPS > 0 else 30
        # src_frame_rate 必须和 dst 一致：VENC 的 CBR 用「bit_rate / src_frame_rate」
        # 算每帧预算，声明 60 而实际只送 30 帧就只有一半码率（pose_webrtc 里这个坑
        # 解释了实测码率只有配置值 28% 的现象）。
        attributes = ChnAttrStr(payload_type, profile, width, HEIGHT,
                                bit_rate=BIT_RATE, gopLen=GOP_LEN,
                                src_frame_rate=dst_fps, dst_frame_rate=dst_fps)
        encoder.Create(attributes)
        link = MediaManager.link(sensor.bind_info()["src"],
                                 (VIDEO_ENCODE_MOD_ID, VENC_DEV_ID, encoder.chn))
        encoder.Start()
        sensor.run()
        server.start()
        print("编码: %s %dx%d %d kbit/s" % (VIDEO_CODEC, width, HEIGHT, BIT_RATE))
        _self_check(ip, HTTP_PORT)

        # 后台看门狗：热点重启/信号抖动时自动重连
        wd_state = {"stop": False}
        try:
            _thread.start_new_thread(_wifi_watchdog, (wd_state,))
            print("网络: 看门狗已启动")
        except Exception as e:
            print("网络: 看门狗启动失败 %s" % (e,))

        stream = StreamData()
        parameter_sets = None
        while True:
            os.exitpoint()
            if server.stopping:
                raise RuntimeError("signaling stopped")
            if server.reboot_at is not None and \
                    time.ticks_diff(time.ticks_ms(), server.reboot_at) >= 0:
                print("OTA: 收到 /reboot，复位板子")
                import machine
                machine.reset()
            if server.need_keyframe():
                encoder.RequestIDR()
            if encoder.GetStream(stream, timeout=100) != 0:
                continue

            # 帧率已由编码器 dst_frame_rate 在硬件层限制，这里不再软件丢帧。
            try:
                for index in range(stream.pack_cnt):
                    data = uctypes.bytearray_at(stream.data[index], stream.data_size[index])
                    stream_type = stream.stream_type[index]
                    timestamp = stream.pts[index]
                    if stream_type == encoder.STREAM_TYPE_HEADER:
                        parameter_sets = bytes(data)
                    else:
                        if stream_type == encoder.STREAM_TYPE_I and parameter_sets:
                            server.send_video(parameter_sets, timestamp)
                            sent_bytes += len(parameter_sets)
                        server.send_video(data, timestamp)
                        sent_bytes += len(data)
                        frames += 1
            finally:
                encoder.ReleaseStream(stream)

            now = time.ticks_ms()
            if time.ticks_diff(now, t_stat) >= STATS_EVERY_MS:
                dt = time.ticks_diff(now, t_stat) / 1000.0
                print("统计 fps=%.1f 发送=%.0f kbit/s 会话=%d 状态=%s" %
                      (frames / dt, sent_bytes * 8 / dt / 1000.0,
                       server.connected_count(), server.session_states()))
                server.stats = {"fps": frames / dt,
                                "kbps": sent_bytes * 8 / dt / 1000.0,
                                "sess": server.session_states(), "detail": "stream"}
                t_stat = now
                sent_bytes = 0
                frames = 0
    except KeyboardInterrupt:
        pass
    finally:
        print("正在关闭…")
        if server is not None:
            server.close()
        if sensor is not None:
            sensor.stop()
        if link is not None:
            link.destroy()
        if encoder is not None and encoder.chn >= 0:
            encoder.Stop()
            encoder.Destroy()
        print("已停止")


if __name__ == "__main__":
    os.exitpoint(os.EXITPOINT_ENABLE)
    run()
