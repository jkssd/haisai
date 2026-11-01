# -*- coding: utf-8 -*-
"""K230 无线图传（RTSP + 硬件 H.264/H.265）。

为什么有这个脚本：CanMV K230 的 `webrtc` 模块需要固件编译时打开
CONFIG_ENABLE_MODULE_WEBRTC（依赖 libpeer），部分厂商固件（例如 Yahboom
当前 v1.8-0 的镜像）没有编进去，`import webrtc` 会直接失败。而 `multimedia`
里的 RTSP 服务器是同一份固件里就有的，可以先跑起来验证无线链路。

播法（PC 端）：
  ffplay -fflags nobuffer -flags low_delay -framedrop \
         -probesize 32 -analyzeduration 0 rtsp://<板子IP>:8554/live

  VLC: 媒体 -> 打开网络串流 -> 上面的地址；再把「网络缓存」调到 100~200 ms。

配置沿用 /sdcard/webrtc_config.json，多两个可选键：
  "rtsp_port": 8554, "rtsp_path": "live", "gop": 30
"""
from media.vencoder import *
from media.sensor import *
from media.media import *
import json
import os
import sys
import time
import _thread
import uctypes
import multimedia as mm
import network

CONFIG_PATH = "/sdcard/webrtc_config.json"

DEFAULTS = {
    "mode": "sta",
    "ap_ssid": "K230-CAM",
    "ap_key": "k230cam88",
    "ssid": "",
    "password": "",
    "width": 1280,
    "height": 720,
    "bit_rate": 3000,
    "codec": "h264",
    "rtsp_port": 8554,
    "rtsp_path": "live",
    "gop": 30,
}


def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH, "r") as f:
            user = json.load(f)
        if isinstance(user, dict):
            for k, v in user.items():
                if k in cfg:
                    cfg[k] = v
    except Exception as e:
        print("config: 用默认值（%s）" % e)
    return cfg


def setup_network(cfg):
    if cfg["mode"] == "ap":
        ap = network.WLAN(network.AP_IF)
        ap.active(True)
        time.sleep(1)
        ap.config(ssid=cfg["ap_ssid"], key=cfg["ap_key"])
        time.sleep(2)
        ip = ap.ifconfig()[0]
        print("网络: AP 模式 ssid=%s ip=%s" % (cfg["ap_ssid"], ip))
        return ip
    w = network.WLAN(network.STA_IF)
    w.active(True)
    time.sleep(1)
    if not w.isconnected():
        target = cfg["ssid"]
        try:
            exact = None
            loose = None
            for ap_ in w.scan():
                try:
                    s = ap_.ssid
                except Exception:
                    continue
                if not s:
                    continue
                if s == target:
                    exact = s
                    break
                if loose is None and target and (target in s or s in target):
                    loose = s
            if exact is not None:
                target = exact
            elif loose is not None:
                target = loose
                print("网络: SSID 用相近匹配 %r" % (loose,))
        except Exception as e:
            print("网络: 扫描失败 %s" % (e,))
        try:
            w.disconnect()
        except Exception:
            pass
        time.sleep_ms(300)
        w.connect(target, cfg["password"])
        for _ in range(80):
            if w.isconnected():
                break
            time.sleep_ms(500)
    if not w.isconnected():
        raise RuntimeError("WiFi 连接超时")
    print("网络: STA ip=%s" % (w.ifconfig()[0],))
    return w.ifconfig()[0]


class RtspServer:
    def __init__(self, cfg, ip):
        self.session_name = str(cfg["rtsp_path"]).strip("/") or "live"
        self.port = int(cfg["rtsp_port"])
        self.video_type = mm.multi_media_type.media_h265 if cfg["codec"] == "h265" \
            else mm.multi_media_type.media_h264
        self.enable_audio = False
        self.width = ALIGN_UP(int(cfg["width"]), 16)
        self.height = int(cfg["height"])
        self.bit_rate = int(cfg["bit_rate"])
        self.gop_len = int(cfg["gop"])
        self.ip = ip
        self.rtspserver = mm.rtsp_server()
        self.start_stream = False
        self.runthread_over = False
        if self.bit_rate < 100 or self.bit_rate > 20000:
            raise ValueError("bit_rate 必须在 100~20000 Kbit/s")
        if self.video_type == mm.multi_media_type.media_h265:
            self.payload_type = Encoder.PAYLOAD_TYPE_H265
            self.profile = Encoder.H265_PROFILE_MAIN
        else:
            self.payload_type = Encoder.PAYLOAD_TYPE_H264
            self.profile = Encoder.H264_PROFILE_MAIN

    def start(self):
        if self.rtspserver.rtspserver_init(self.port) != 0:
            raise RuntimeError("RTSP 端口 %d 绑定失败" % self.port)
        try:
            if self.rtspserver.rtspserver_createsession(
                    self.session_name, self.video_type, self.enable_audio) != 0:
                raise RuntimeError("RTSP session 创建失败")
            self._init_stream()
            self.rtspserver.rtspserver_start()
            self._start_stream()
        except BaseException:
            self.rtspserver.rtspserver_deinit()
            raise
        self.start_stream = True
        _thread.start_new_thread(self._do_rtsp_stream, ())

    def stop(self):
        if not self.start_stream:
            return
        self.start_stream = False
        while not self.runthread_over:
            time.sleep(0.1)
        self.runthread_over = False
        self._stop_stream()
        self.rtspserver.rtspserver_stop()
        self.rtspserver.rtspserver_deinit()

    def url(self):
        return "rtsp://%s:%d/%s" % (self.ip, self.port, self.session_name)

    def _init_stream(self):
        self.sensor = Sensor()
        self.sensor.reset()
        self.sensor.set_framesize(width=self.width, height=self.height, alignment=12)
        self.sensor.set_pixformat(Sensor.YUV420SP)
        self.encoder = Encoder()
        self.encoder.SetOutBufs(8, self.width, self.height)
        chn = ChnAttrStr(self.payload_type, self.profile, self.width, self.height,
                         bit_rate=self.bit_rate, gopLen=self.gop_len)
        self.encoder.Create(chn)
        self.link = MediaManager.link(self.sensor.bind_info()["src"],
                                      (VIDEO_ENCODE_MOD_ID, VENC_DEV_ID, self.encoder.chn))

    def _start_stream(self):
        self.encoder.Start()
        self.sensor.run()

    def _stop_stream(self):
        self.sensor.stop()
        self.link.destroy()
        self.encoder.Stop()
        self.encoder.Destroy()

    def _do_rtsp_stream(self):
        try:
            data = StreamData()
            while self.start_stream:
                os.exitpoint()
                self.encoder.GetStream(data)
                for i in range(data.pack_cnt):
                    raw = bytes(uctypes.bytearray_at(data.data[i], data.data_size[i]))
                    self.rtspserver.rtspserver_sendvideodata(
                        self.session_name, raw, data.data_size[i], 1000)
                self.encoder.ReleaseStream(data)
        except KeyboardInterrupt:
            pass
        except BaseException as e:
            sys.print_exception(e)
        finally:
            self.runthread_over = True


def main():
    cfg = load_config()
    ip = setup_network(cfg)
    srv = RtspServer(cfg, ip)
    srv.start()
    codec = "H265" if cfg["codec"] == "h265" else "H264"
    print("RTSP 已启动: %s (%s %dx%d, %d kbit/s)" %
          (srv.url(), codec, ALIGN_UP(int(cfg["width"]), 16), int(cfg["height"]),
           int(cfg["bit_rate"])))
    print("提示: 客户端连上后再看串口输出，能看到码流")
    try:
        n = 0
        while True:
            time.sleep(1)
            n += 1
            if n % 10 == 0:
                print("RTSP 运行中 %ds" % n)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    print("已停止")


if __name__ == "__main__":
    os.exitpoint(os.EXITPOINT_ENABLE)
    main()
