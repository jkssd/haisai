# K230 无线图传（WebRTC）

K230（Yahboom 版，CanMV）通过板载 2.4G Wi-Fi 直接把摄像头画面以 **WebRTC + H.264 硬编码**推到浏览器。
延迟低、无需服务器、不依赖场馆网络。

---

## 1. 为什么需要自己编固件

官方 CanMV 固件（含 v1.8 的所有 Release）**没有编译 `webrtc` 模块**：

| 检查 | 结果 |
|---|---|
| 原厂 v1.8 镜像 `import webrtc` | `ImportError: no module named 'webrtc'` |
| v1.7 / v1.8 源码 `Kconfig.micropython` | 没有 `ENABLE_MODULE_WEBRTC` 这一项 |
| 当前 `canmv_k230` 分支 | `ENABLE_MODULE_WEBRTC` 默认 `y`，且 `select ... LIBPEER` |
| 官方 Release / CI 产物 | 最新 tag 就是 v1.8，Actions 产物为 0 |

所以本项目**自带一版编好的固件**，`webrtc` 和 `libpeer` 都已启用。

- 固件：**镜像不进仓库**（单个 240MB 以上，超过 GitHub 单文件上限），
  编译方法和原来的文件名见 [`k230/firmware/README.md`](firmware/README.md)
- 版本串：`CanMV v1.8-76-gf6ca68a`
- 板级配置：`k230_canmv_yahboom_defconfig`
- 必带的 5 个 libpeer 补丁：[`libpeer_patch/libpeer.diff`](libpeer_patch/libpeer.diff)

---

## 2. 烧录

镜像按扇区写入 SD 卡（不是拷文件）。

1. 读卡器插卡，确认盘符（**务必确认是没有重要数据的那张卡**）
2. 解压 `.img.gz` 得到 `.img`
3. 用 Rufus / balenaEtcher / Win32DiskImager 写入
4. 卡插回板子，上电

> 写入途中有个坑：Windows 会在写入过程中自动挂载新冒出来的分区，导致后半段写入被拒绝（`访问被拒绝`）。
> 解决办法是先把磁盘设为**离线**再写，写完再设回在线。

原 `main.py`（TFT 识别启动器）已备份为 `/sdcard/main.tft.py`。
想换回去：`cp /sdcard/main.tft.py /sdcard/main.py`。

---

## 3. 开机自动启动

`/sdcard/main.py` 现在是图传启动器（[`k230/main_webrtc.py`](main_webrtc.py)），
上电后自动执行 `/sdcard/webrtc_stream.py`。

临时关闭自动启动：

```python
# 板上执行
open('/sdcard/webrtc_off', 'w').close()
```

---

## 4. 配置

编辑 `/sdcard/webrtc_config.json`：

```json
{
  "mode": "sta",
  "ssid": "<你手机热点的名字>",
  "password": "<热点密码>",
  "width": 640,
  "height": 480,
  "bit_rate": 1400,
  "codec": "h264",
  "sensor_fps": 60,
  "fps": 30,
  "gop": 20,
  "detect_every": 8,
  "port": 8080,
  "token": ""
}
```

> 仓库里的 [`webrtc_config.json`](webrtc_config.json) 是**模板**，SSID/密码留的是占位符。
> 真实配置只放在板子的 `/sdcard/webrtc_config2.json`（优先于 `webrtc_config.json`），
> 别提交到仓库。`mode` 用 `auto`：连不上热点自动退成 AP，避免现场彻底失联。

下面的字段表按旧版 1280×720 的例子写的，数值以模板为准：

| 键 | 说明 |
|---|---|
| `mode` | `ap` = 板子自己开热点；`sta` = 板子连外部热点；`auto` = 先试 sta，失败自动退 AP（**推荐**） |
| `ap_ssid` / `ap_key` | AP 模式的热点名和密码（密码至少 8 位） |
| `ssid` / `password` | STA 模式要连的热点，SSID 支持中文（驱动返回的是 bytes，代码里已做解码） |
| `width` / `height` | 编码分辨率，宽度会向上对齐到 16 的倍数 |
| `bit_rate` | 目标码率，单位 Kbit/s（100–20000）；**实际码率 ≈ `fps实际 × bit_rate / fps`** |
| `bit_rate_max` | 硬上限（0 = 不限），防 CBR 超发把 2.4G 打爆 |
| `codec` | `h264`（兼容性最好）或 `h265`（省带宽，部分浏览器不支持） |
| `sensor_fps` | 摄像头帧率，**必须 60**（见 7.6） |
| `fps` | 目标帧率（实际落在 23–25） |
| `gop` | IDR 间隔；`gop/fps` 就是丢包后最长花屏时间 |
| `detect_every` | 每 N 帧跑一次 AI，帧率的最大旋钮 |
| `ota` | 允许 `/upload` + `/reboot`（免串口改脚本） |
| `token` | 非空时访问页面/接口必须带 `?token=` |

---

## 5. 观看

1. 电脑连 `K230-CAM`（密码 `k230cam88`）
2. 浏览器打开 **http://192.168.169.1:8080/**
3. 点「连接」

页面上会实时显示：分辨率、接收码率、帧率、累计丢包、往返时延、抖动、首帧耗时。

> **注意：本机只有一块无线网卡。**
> 连了 K230-CAM 就断开 OUC 校园网，反之亦然。这是硬性约束，测试时要预留一段断网时间。

---

## 6. 实测数据（端到端跑通）

**WebRTC 画面已跑通**，1280×720 H.264，浏览器稳定收流。

| 指标 | 数值 |
|---|---|
| 分辨率 | **1280×720** |
| 首帧耗时 | **2.3 秒** |
| 接收帧率 | 中位 **46.5 fps**（27–62） |
| 接收码率 | 中位 **5.6 Mbit/s**（4.8–6.6） |
| 累计解码帧 | 1050 帧 / 25 秒 |
| 丢包 | 147 |
| RTT | 中位 **33 ms**（最大 162 ms） |
| 抖动 | 中位 **5 ms** |
| 卡顿比例 | 8.7% |
| 连接状态 | `connected` 全程保持 |

板端编码 57 fps，日志中持续出现 `recv non-STUN data 70/86 bytes`（DTLS 加密媒体流）。

## 6.1 修好的四个 libpeer bug（完整 diff 见 [`libpeer_patch/libpeer.diff`](libpeer_patch/libpeer.diff)）

| # | 文件 | 问题 | 修复 |
|---|---|---|---|
| 1 | `agent.c` | `agent_recv()` 在 `if (!any_active) return -1;` 之后，候选对全失败后 agent 彻底"失聪" | 提到判断之前 |
| 2 | `agent.c` | **ICE 成功了却进不了 CONNECTED**：成功判断在 `any_active` 提前返回之后，只要应答落在两次检查之间（`inprog` 归零）就永远卡在 checking | 删掉提前返回，让成功判断无条件执行 |
| 3 | `peer_connection.c` | 只有 `COMPLETED` 状态才收包 | `CHECKING` 分支补 `agent_recv()` |
| 4 | `utils.h` + `modwebrtc.c` | **RT-Smart 下裸 `printf` 到不了串口**，所有 libpeer 日志凭空消失（排查时最大的坑） | 改走 `mp_printf`，`-DLOG_REDIRECT=1` |

## 6.2 DTLS 握手：改用 client 角色

server 角色下 mbedtls 的 hello-verify cookie 反复校验失败（`cookie verification failed`，
server state 1↔2 死循环，最终 `MBEDTLS_ERR_SSL_UNEXPECTED_MESSAGE`）。

**解法**：offer 里把板子设为 **DTLS client**（`a=setup:active`），让 Chrome 当 server。
client 端没有 hello-verify 机制，握手一次通过。

```c
case SDP_TYPE_OFFER:
  role = DTLS_SRTP_ROLE_CLIENT;   /* 原为 SERVER */
```

## 6.3 启动入口（Yahboom 固件特有）

### 已解决：开机自启不稳定

Yahboom 出厂 `/sdcard/main.py` 里有按键检测：

```python
key = YbKey()
if key.is_pressed():
    if key.is_pressed():
        abort_main = 1
if not abort_main:
    from ybMain.main import *; start()
```

`YbKey` 偶尔误判成"按下"，`start()` 就不执行，板子停在 REPL——上电后没人敲命令，直接卡死。

**解决**：用 [`main_stream.py`](main_stream.py) 覆盖 `/sdcard/main.py`，完全去掉按键逻辑。
实测连续 3 次 `machine.reset()` + 1 次冷启动，**4/4 全部自动拉起**。

禁用自启：`/sdcard` 下建空文件 `noapp`。

### 已解决：TFT 显示 + WebRTC 推流共存

想同时做「TFT 本地显示（含 AI 叠加）+ WebRTC 推流」时，一开始所有组合都卡死。
逐操作计时排查后定位到**两个真凶**：

| 真凶 | 现象 | 解决 |
|---|---|---|
| **`Display.init(..., to_ide=True)`** | 一进来 snapshot 就阻塞 | 必须 `to_ide=False` |
| **`MediaManager.link(sensor → VENC)`** | 绑死后多通道 snapshot 被饿死 | 改用 `snapshot(chn=0, dump_frame=True)` + `encoder.SendFrame()` 手动喂帧 |

改完这两点，三通道（chn0 YUV420SP / chn1 RGB565 / chn2 RGBP888）稳定共存：

```
display: ST7701 (to_ide=False)
summary: stream=40 fails=0  snapshot0=21ms SendFrame=23ms GetStream=34ms snapshot1=114ms
统计 fps=38.4 发送=630 kbit/s 会话=0 状态=-
```

> 关键对比：**同样的代码，只把 `to_ide` 从 True 改成 False，就从"完全卡死"变成"40/40 帧全部成功"**。
> IDE 预览层和 VENC 会抢同一份缓冲/带宽。

三通道分辨率必须一致（都用 640×480），否则传感器侧同样会出问题。

实现见 [`pose_webrtc.py`](pose_webrtc.py)。

## 6.4 刷机注意（可移动媒体）

Windows **不允许把 SD 卡置为离线**（`可移动媒体不支持此操作`），
而写入途中自动挂载新分区会导致 `访问被拒绝`。可靠做法：

```powershell
wsl --mount \\.\PHYSICALDRIVE1 --bare   # 会失败，但会把磁盘置为离线
# 然后直接写 \\.\PhysicalDrive1
```



---

## 7. 已踩的坑

### 6.4.1 实测带宽上限：这套 2.4G 组合只有 1.0–1.5 Mbit/s

这是**所有卡顿/花屏问题的总根源**，先给结论：

| 配置码率 | 实际发出 | 丢包(40s) | 卡顿比例 | RTT 中位/最大 | 结论 |
|---|---|---|---|---|---|
| 1000 | 839 kbps | 5 | 8.7% | 17 / 688ms | 稳 |
| 1500 | 1540 kbps | 8 | 10% | 17 / 365ms | 可用 |
| 3000 | 1826 kbps | 5~18 | 6% | 15 / 1061ms | 开始尖峰 |
| 4000 | 2243 kbps | **100** | 13% | 43 / 466ms | 花屏严重 |

**超过约 1.8 Mbit/s 就会周期性拥塞**：RTT 从 20ms 飙到 600–900ms，
期间整片丢包 → H.264 参考帧错乱 → 画面花屏。

调整码率/分辨率对帧率**没有**帮助（512×384 和 640×480 都是 17 fps），
说明瓶颈是链路带宽和主循环本身，不是像素处理。

> **更正（2026-11）**：这张表的「实际发出」并不是链路跑满的结果，而是 `src_frame_rate=60`
> 把编码器码率砍到 28% 之后的数字（见 7.3）。真实链路容量比表里高，
> 但「1 Mbit/s 稳、1.8 Mbit/s 开始尖峰」这个量级仍然可以参考 —— 现在的默认配置按
> 1000 kbit/s 给。帧率 17 fps 的瓶颈也不是像素处理，而是 `GetStream` 空等（见 7.4）。

### 6.5 为什么会"莫名其妙卡死"（已定位并修复）

现象：板子随机假死，串口无输出、HTTP 无响应，只能断电。

**主因：连不通的 ICE 会话在烧 CPU。** 实测同一块板子、同一份代码：

```
会话建立前:  fps=33.6   AI 单次推理 40 ms
会话建立后:  fps=26.5 → 7.9 → 5.3   AI 单次推理 318 ms
```

libpeer 为每个会话起一个 ICE 探测线程，**连不通就会持续重试**（旧标签页反复重连尤其严重），
把 CPU 抢光 → 摄像头/AI/编码全面变慢 → 从用户视角就是"卡死"。
日志里 `REAP state=failed` 之后帧率立刻恢复，印证了这个因果关系。

**次因：HTTP 线程持锁做重活。** `set_remote_description()` 在持锁状态下执行 ICE 初始化，
而主循环每帧调 `send_video()` 也要这把锁 → 主循环被钉住。

### 6.6 WiFi 看门狗导致周期性 600ms 卡顿（已修复）

`_wifi_watchdog` 原来每 5 秒调用一次 `network.WLAN(...)` 重新创建网卡对象，
在 RT-Smart 上会让无线驱动周期性停顿：

```
修复前: RTT 25 → 13 → 15 → 607 → 688 → 67 → 17   (每4~5秒一次尖峰)
修复后: RTT 17ms 稳定，最大 365ms，丢包 12 → 5
```

改成**循环外创建一次、循环内只读状态**即可。


**修复：**

| 改动 | 作用 |
|---|---|
| 协商超时 30s → **10s** | 死会话快速回收 |
| **限流**：同 IP 3 秒内只能建一次会话 | 旧标签页重连不再堆会话 |
| `send_video`/`need_keyframe` 改**非阻塞拿锁** | 主循环永不被 HTTP 线程拖住 |
| `set_remote_description` 移到**锁外** | 消除最大阻塞点 |
| **监管线程**：主循环卡 15s 或持续 <8fps → `machine.reset()` | 自动恢复 |
| **硬件 WDT**（30s，主循环喂狗） | 救回多线程在 C 层死锁的情况 |

**另一个坑（操作层面）**：脚本运行中通过串口上传文件，Ctrl-C 打断后解释器回不到 REPL，
后续写入落到半截 → 必须断电。正确顺序：先建 `/sdcard/noapp` 停车 → 复位 → 再上传 → 解停。

### 7.0 远距离断线（链路预算问题，不是软件问题）

实测结论：**近距离流畅，远距离直接断线**。

原因：
- K230 板载 2.4G 模块发射功率小、PCB 天线增益低；
- 板子开 AP 时，手机能"看到"热点不代表板子能收到手机上行，链路是**不对称**的；
- 距离一远，重传率飙升、吞吐断崖式下跌，最终 association 直接丢失。

对策（按效果排序）：

| 方案 | 效果 | 成本 |
|---|---|---|
| **换带外置天线的 USB WiFi 模块**（固件支持 `wlan_device: "usb"`） | 最有效，发射功率和天线增益都高一个档次 | 几十元 |
| 板子放在靠靶区的位置，操作台在可视近距离内 | 立竿见影 | 需布置 |
| "远距离模式"降规格：640×480 @ 800–1000 kbit/s | 可用距离延长 2–3 倍 | 画质下降 |
| 换 5GHz | 不可行，板载模块只支持 2.4G | — |

摆位：天线竖直、抬高、远离金属与电池，避免人体遮挡。

### 7.1 码率超发（这个结论后来被推翻了，见 7.3）

配置写 3000/4000 kbit/s 时，编码器实际发出 5.6–9.2 Mbit/s（近 2 倍），2.4G 直接被压垮。
降到 2500 后实测 2.3–2.4 Mbit/s，640×480@1000 实测约 0.9 Mbit/s，均命中目标。

> 上面这组数是在 `pose_webrtc.py`（手动 SendFrame）上量的。**同一份配置在
> `webrtc_stream.py`（MediaManager.link 硬件直连）上会超发，在 pose_webrtc 上反而只有
> 配置值的 28%** —— 原因见 7.3。别再用这两组数互相推断链路容量。

### 7.3 花屏的真正原因：编码器码率模型算错了（已修）

**症状**：640×480 @ 17 fps，浏览器显示 **累计丢包 0、RTT 17ms**，但画面一堆块状拖影、
时不时整片糊掉 1 秒。

先把接收端拆开看（`tools/webrtc_diag.py` 抓 `getStats()` 全字段）：

| 指标 | 实测 | 说明 |
|---|---|---|
| packetsLost | **0** | 网络没丢包，花屏不是丢包造成的 |
| framesReceived / framesDecoded | 269 / 253 | 另有 15 帧收到了但解不出来 |
| pliCount | 5 / 17s | 解码器在反复向板子要关键帧 |
| freezeCount | 1 次 / 0.9s | 整整冻了 0.9 秒 |
| keyFramesDecoded | 19 / 17s | 每 0.9 秒就有一个 IDR |
| jitterBufferDelay | 80 ms/帧 | 抖动缓冲约 1.3 个帧间隔 |

板端同一时刻打印 `fps=17.2 发送=509 kbit/s`，而配置写的是 **1500**。

**根因**：`ChnAttrStr(..., bit_rate=1500, src_frame_rate=60, dst_frame_rate=30)`。
VENC 的 CBR 是**按声明的输入帧率分摊码率**的（海思 MPP 的语义：
「src/dst_frame_rate 用于配合 bit_rate 分配」），即每帧预算 = `bit_rate / src_frame_rate`。
以前固定写 60，而手动 `SendFrame` 实际只有 17 帧/秒，于是真实码率 ≈ 配置值 × 17/60：

| 配置 | 实测发送 | 按 17/60 推算 |
|---|---|---|
| 1500 | 509 kbit/s | 425 |
| 1200 | 305 kbit/s | 340 |
| 2500 @34fps | 1360 | 1417 |
| 1500 @38fps | 905 | 950 |

640×480 每帧只分到 **3 KB 左右**（约 0.09 bit/像素），量化到顶 → 块效应、拖影，
看着就是「花屏」。**这不是链路问题，把码率调大也没用，因为编码器自己把码率砍了 72%。**

**修法**：`src_frame_rate` 必须写成真实送帧帧率，并且等于 `dst_frame_rate`：

```python
e.Create(ChnAttrStr(..., bit_rate=rate, gopLen=GOP,
                    src_frame_rate=fps, dst_frame_rate=fps))   # 以前写死 60
```

改完配置 1000 kbit/s、声明 20 fps、实际 17 fps → 每帧预算 `1000/20 = 50 kbit = 6.2 KB`，
是原来的 **1.8–2 倍**，且链路负载仍在实测「稳」的 1 Mbit/s 以内（见 6.4.1 那张表的
1000/1500 两行）。

### 7.4 帧率 17 fps 的来源：GetStream 每轮空等 10ms（已修）

板端实测每秒耗时分配（`fps=17.2` 时）：`snap=293 img=? ai=251(6次) draw=18 csc=66 show=35`，
合计只有 663 ms —— 有 **337 ms/秒** 不在任何计时器里。

最大的那块是 `GetStream(data, timeout=10)`：编码器是异步的，`SendFrame()` 返回时还没编完，
紧接着取流就会**干等满 10 ms 超时**。按 17 帧/秒算，每秒白扔 170 ms。

> 代码里原来就有一句注释说「只有在确实喂过帧之后才去取流，否则每轮空等 10ms，
> 把帧率从 25 拉到 17」—— 但代码里并没有实现这个判断。

**修法（两步）**：
1. 把取流挪到**下一轮循环开头**（`pending_feed` 标记）。上一轮喂的帧，到那时已经编完，
   `GetStream` 几乎立刻返回；
2. 喂帧后把积压的码流**一次抽干**（最多 4 次），避免输出缓冲被撑爆后整帧丢弃
   （整帧丢弃 = RTP 序号空洞 = 直到下一个 IDR 都是花屏）。

同时把「拿不到锁就丢帧」改成**最多退避 6 ms 再放弃**（`send_video`），
丢一帧的代价是 0.75 秒花屏，不值得省这 6 ms。

### 7.5 免串口改板端脚本（`/upload` + `/reboot`）

调参最痛的环节是「改一行代码要拔卡/串口上传」。现在板子的 HTTP 服务多了两个口子
（`ota: true` 时生效）：

```bash
# 1) 直接把本地文件 POST 上去（板子会先 compile() 过语法，语法错就拒收，不会把自己搞哑）
curl --data-binary @k230/pose_webrtc.py \
     "http://192.168.43.188:8080/upload?path=/sdcard/pose_webrtc.py"
# 2) 复位，2 秒后新脚本就跑起来了
curl "http://192.168.43.188:8080/reboot"
# 3) 远程看状态（帧率/码率/各段耗时/会话状态）
curl "http://192.168.43.188:8080/status"
```

`/status` 现在长这样：

```json
{"mode":"auto","profile":"?","rate":0,"rssi":-54,"clients":1,
 "fps":24.9,"kbps":1181.0,"sess":"completed",
 "st":"snap=27 img=7 ai=151/4 csc=46 feed=7 get=50 send=3 show=27 H1 I1"}
```

`st` 里就是主循环各段的毫秒数/秒（`ai=耗时/次数`，`H/I` 是每秒的 SPS-PPS 包和 I 帧数）
—— **不接串口也能定位瓶颈**。

### 7.6 帧率 14 → 23 fps：把主循环拆开看

同一块板、同一个手机热点，`tools/webrtc_diag.py` 前后对比：

| 指标 | 改前 | 改后 | 说明 |
|---|---|---|---|
| 接收帧率 | 14.1 fps | **23.4 fps** | 板端计数 25 fps |
| 接收码率 | 418 kbit/s | **1199 kbit/s** | 链路仍有余量 |
| 每帧字节 | 3.5 KB | **5.8 KB** | 画质直接受益 |
| PLI 关键帧请求 | 5 次/17s | **0** | 解码器不再报错 |
| 冻结 | 0.9 s | **0** | — |
| 抖动缓冲 | 83–105 ms | **29 ms** | 迟到帧几乎没了 |
| 平均帧间隔 | 71 ms（不均） | **42.7 ms**（理论 42.8） | 节奏稳定 |

#### 关键发现 1：帧率是被「帧锁」量化掉的

主循环是「取一帧 → 干活 → 喂编码器」的串行结构，所以循环周期 T 会被摄像头
帧周期 P **量化成整数倍**：`T = P × ceil(W / P)`（W = 一轮的净耗时）。

```
P=33.3ms(30fps) 时：W=41ms -> T=66ms -> 15fps     ← 改前就卡在这个台阶上
P=16.7ms(60fps) 时：W=22ms -> T=33ms -> 30fps     ← 台阶变细，损失变小
```

所以「提高摄像头帧率」不是为了多送帧，而是为了让量化台阶变细。
`Sensor()` 不传 `fps` 时默认不是 60，必须显式写 `Sensor(id=2, fps=60)`
（厂家基线 `sdcard_backup/k230_perf.py` 也是这么写的，双通道 640×480 能到 57fps）。

改完之后 `snap`（等摄像头出帧）从 24 ms/帧掉到 1 ms/帧，证明采集不再是瓶颈。

#### 关键发现 2：软件节流本身会把帧率砍半

`frame_gap` 的做法是「距离上一帧超过 N ms 才喂」。主循环只要 22 ms，而 30fps 的
间隔是 33 ms，于是变成**每隔一轮才喂一次** → 22.9 fps。这就是板端报 24、页面报 23 的来源。

#### 两次失败尝试（都别再试）

| 尝试 | 结果 |
|---|---|
| `Sensor(id=2, fps=30)` | **卡死在摄像头初始化**：网卡起来了、8080 端口拒连、串口全黑，只能拔卡断电。双通道 640×480 只能用 60fps。 |
| `feed_all=true`（不节流，指望 VENC 的 `dst_frame_rate` 丢帧） | **VENC 不丢帧**：喂 45.7 帧/秒就编 45.7 帧/秒，码率冲到 1900 kbit/s，2.4G 直接压垮。`dst_frame_rate` 只参与码率分配，不是输出帧率。 |
| 循环末尾 `time.sleep_ms()` 补齐到 33 ms | RT-System 的 tick 是 10 ms：补 11 ms 实际睡 20 ms，周期变 42 ms。**亚 10ms 的睡眠做不到**。 |

#### 为什么停在 23–25 fps 而不是 30

要把 22 ms 的循环撑到 33 ms 只剩「忙等」一条路（`while ticks_diff(...) < 33: pass`）。
忙等会占掉约 1/3 个核且不释放 GIL，HTTP 线程/编码线程可能被饿死，风险大于收益，
所以没做。想再往上走，只能从「减少每轮净耗时」下手（见下表），或者上 5GHz WiFi。

#### 可调旋钮（`/sdcard/webrtc_config2.json`，改完 `GET /reboot` 生效）

| 键 | 当前 | 作用 |
|---|---|---|
| `sensor_fps` | 60 | **必须 60**，改 30 会卡死 |
| `fps` | 30 | 目标帧率（实际落在 23–25） |
| `bit_rate` | 1400 | 目标码率；实际 ≈ `实际fps × bit_rate / fps`（现在 ~1170） |
| `gop` | 20 | IDR 间隔 = 20/fps ≈ 0.85 s，这是丢包后最长花屏时间 |
| `detect_every` | 8 | 每 N 帧跑一次 AI；**这是帧率的最大旋钮**（单次推理 ~38 ms） |
| `feed_all` | false | **别开**（见上表） |
| `bit_rate_max` | 1400 | 硬上限，防 CBR 超发把链路打爆 |

> 首帧 2 秒的问题还在：实测 2057 ms，和改前一样。页面侧「等 ICE 收集」已经从
> 2.5 s 优化到 300 ms，说明瓶颈在板端的 ICE + DTLS 握手（板子是 DTLS client，
> mbedtls 握手在这颗 RISC-V 上要 1 s 量级）。要再压只能减少候选对或换更快的密码套件。

### 7.2 其它坑

1. **`webrtc` 模块缺失**：只有 v1.8 之后的源码才有，官方镜像都没带 → 自己编。
2. **`main.py` 会抢占**：脚本跑完软重启后 `/sdcard/main.py` 自动执行，会抢摄像头和屏幕。用 `/sdcard/webrtc_off` 关掉。
3. **脚本依赖串口会话**：通过 REPL `exec()` 启动的脚本，串口一关就被中断，AP 也跟着消失。
   所以必须写进 `main.py` 让它开机自启。
4. **Wi-Fi 不跨脚本保持**：每次脚本重启都会断网，联网必须在脚本内部完成。
5. **扫描返回的 SSID 可能是 bytes**：新固件里 `ap.ssid` 是 `bytes`，和配置里的 `str` 比较会报错，必须先解码。
6. **`media.rtspserver` 可用但 `media.mjpeg` 不可用**：这块固件有 RTSP，没有 MJPEG 模块。
7. **校园网双向隔离**：PC→板子 ICMP/TCP 全丢，板子→PC 也 `ETIMEDOUT`。**现场别依赖场馆网络**。
8. **AP 模式下不要调 `set_default_dev`**：`set_default_dev('wlan0ap')` 会失败并把默认网卡清成
   `None`，板子收得到请求却发不回去，网页一直转圈。AP 模式保持驱动默认即可；STA 模式设
   `wlan0` 是有必要且有效的。
9. **热点重启会让板子卡死**：串口零字节、只能断电重启。已加 WiFi 看门狗（掉线自动重连），
   但仍建议比赛时固定热点、避免重启。
10. **脚本会锁住自身文件**：`main.py` 里 `exec(open(...).read())` 不关文件句柄，之后远程改不了
   该文件（`OSError: 16 EBUSY` / `EPERM`）。已改为显式 `close()`；若又被锁，建 `/sdcard/webrtc_off`
    断电重启后再改。
11. **串口是唯一的"生命线"**：CanMV 的 MCP 只在 VSCode 打开时可用，VSCode 一关就没了。
    项目内自带 [`tools/k230_serial.py`](../tools/k230_serial.py)、`k230_upload.py` 可直接走 REPL。
12. **`webrtc_config2.json` 写坏了 = 板子失联**：`load_config` 解析失败会退到 `DEFAULTS`
    （`mode: "sta"` + 空 `ssid`），连不上热点又不退 AP，表现是「网卡在、8080 拒连、串口全黑」。
    两个坑都踩过：中文 SSID 被编辑器/`Set-Content -Encoding UTF8` 转码，以及 UTF-8 **BOM**。
    改配置一律用字节级方式（写文件 / `curl --data-binary @file`），别用 PowerShell 文本管道。
13. **不要用 `machine.reset()` 代替断电**：从 REPL 软复位后 CSI/ISP 没被真正复位，
    下次启动会卡在 `Sensor()`/`Display.init()`（同样只能拔电）。要重启板端脚本用
    `GET /reboot`（主循环里调 `machine.reset()`，是正常关机路径）。
14. **`Ctrl-C` 打断媒体脚本大概率回不到 REPL**：VM 死在 C 层，串口零字节、只能拔电。
    要停脚本就走 `/reboot` 或拔卡，别按 Ctrl-C。

---

## 8. 文件

| 文件 | 作用 |
|---|---|
| [`k230/pose_webrtc.py`](pose_webrtc.py) | **脱机三合一**：YOLO 检测 + TFT 显示 + WebRTC 推流 |
| [`k230/main_stream.py`](main_stream.py) | 开机入口（替换出厂 `main.py`，去掉按键检测） |
| [`k230/webrtc_stream.py`](webrtc_stream.py) | 纯图传版本（无检测，57fps，备用） |
| [`k230/tft_live_v11.py`](tft_live_v11.py) | 纯本地检测版本（无推流，23fps，备用） |
| [`k230/webrtc_config.json`](webrtc_config.json) | 配置（板上优先读 `webrtc_config2.json`） |
| [`k230/rtsp_stream.py`](rtsp_stream.py) | RTSP 版本（备用，固件自带支持） |
| [`k230/libpeer_patch/libpeer.diff`](libpeer_patch/libpeer.diff) | 自己编固件时要带的 5 个 libpeer 补丁 |
| [`k230/firmware/README.md`](firmware/README.md) | 固件镜像说明（镜像本身 240MB+，不进仓库） |
| [`tools/webrtc_diag.py`](../tools/webrtc_diag.py) | **接收端解码统计**：PLI/NACK/丢帧/冻结/抖动缓冲，定位花屏的第一工具 |
| [`tools/webrtc_probe.py`](../tools/webrtc_probe.py) | PC 端压测（headless Chrome + CDP，无第三方依赖） |
| [`tools/k230_upload.py`](../tools/k230_upload.py) | 串口上传（REPL，base64 分块，自带大小校验） |
| [`tools/k230_repl.py`](../tools/k230_repl.py) | 串口执行一段板端代码（paste mode） |
| [`tools/check_board_syntax.py`](../tools/check_board_syntax.py) | 板端脚本的 PC 侧语法体检（含 `PAGE` 里的 JS） |
| [`tools/verify_deploy.py`](../tools/verify_deploy.py) | 拔卡部署后的校验（编码/BOM/JSON/中文 SSID） |

> `k230/sdcard_backup/`（刷机前的 SD 卡全量备份）和 `k230/_*.py`（一次性探针脚本）
> 只在本地保留，已写进 `.gitignore`。

### 三种运行形态的取舍

| 形态 | 帧率 | 用途 |
|---|---|---|
| `webrtc_stream.py` | 57 fps | 只要远程画面 |
| `tft_live_v11.py` | 23 fps | 只要本地检测显示 |
| `pose_webrtc.py` | **23–25 fps** | 本地显示 + 远程推流同时 |

`main_stream.py` 会**优先启动 `pose_webrtc.py`**，找不到才退回 `webrtc_stream.py`。

### 压测用法

```bash
python tools/webrtc_probe.py --url http://192.168.169.1:8080/ --seconds 60 --out probe.json
```

输出帧率中位数/最小/最大、码率区间、累计丢包、RTT、抖动、卡顿样本比例。
