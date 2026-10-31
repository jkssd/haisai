# K230 部署（CanMV v1.8 / 亚伯 K230 豪华版）

## 结论速览

| 项 | 实测结果 |
|---|---|
| 板子 | CanMV K230 YAHBOOM - 1G，`v1.8-0-gc2d1f5cc` |
| 相机 | **sensor id=2**（`id=0/1` 报 `Can not found sensor`），GC2093 CSI2，最大 1920x1080@60 |
| 屏幕 | **ST7701 800×480**，`Display.show_image(img, x, y, Display.LAYER_OSD3)` 约 1.2 ms |
| 推理 | int8 kmodel：640 输入 70 ms / **512 输入 46 ms / 448 输入 37 ms** |
| 帧率 | 448 输入 + TFT 直显：**24.1 fps（41 ms/帧）** |
| 烧录 | 走 MCP：3.6 MB kmodel **3 秒**传完（115200 串口要 6 分钟） |
| 上电自启 | `/sdcard/main.py` = `import run_tft` |

## 五个必踩的坑

| # | 现象 | 原因 / 修法 |
|---|---|---|
| 1 | `'ai2d' object has no attribute 'run'` / `'Ai2d' has no attribute 'set_resize_param'` | `nn.ai2d()` 没有 `run`；要用 **`libs.AI2D.Ai2d`**，方法名是 **`resize()`**、`set_ai2d_dtype()`、`build()`、`run()` |
| 2 | `AssertionError: should call reset() first` | `Sensor(id=2)` 后必须 `sen.reset()` 再 `set_framesize` |
| 3 | 喂 `RGB888` 时模型输出接近 0（`maxcls=0.0003`） | ai2d 是 NCHW，`RGB888` 给的是 HWC；必须用 **`RGBP888`**（平面格式，`to_numpy_ref()` 得到 `(3,H,W)`） |
| 4 | kmodel 分数饱和成 1.0、关键点乱飞 | ultralytics ONNX 要 **0–1 归一化输入**；nncase 编译时必须 `std=[255,255,255]`，否则等于喂 0–255 |
| 5 | 关键点索引全错位 | ultralytics pose 输出是 **`(x,y,conf)` 三元组**：`28 = 4 box + 3 cls + 7×3`，步长是 3 不是 2 |

另外两个运行时坑：

* `sensor.run()` 后不 `stop()` → 连续采集阻塞解释器，REPL 无响应（看着像死机），只能按复位键
* `to_ide=True` 向 IDE 推流时，USB 没人取帧会反压卡住 `show_image`（20s 看门狗复位）；上 TFT 时设 `to_ide=False`

## 编译 kmodel

```bash
export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH          # nncase 需要 .NET
export NNCASE_PLUGIN_PATH=$HOME/.local/lib/python3.10/site-packages/nncase

python3 k230/build_kmodel.py --int8 --rgb --size 448 \
        --onnx weights/target_pose_v4_448.onnx \
        --out  k230/target_pose_v4_448_int8.kmodel
```

- `--rgb`：板端输入已是 RGB，**不要**再 swapRB
- `--size`：448 → 37ms（24fps）；512 → 46ms（17fps）；640 → 70ms（12fps）
- PTQ 校准集：按场景分层 160 张（省赛 40 / d2 60 / c2 60），**排除留出集**
- 校准数据必须是 **uint8(0–255)**，传 float 会崩：`need u8[...] but get f32[...]`

## 烧录与运行（MCP，不用插拔 SD 卡）

```bash
python3 tools/canmv_mcp.py list          # 40 个工具
python3 tools/canmv_mcp.py call canmv_write_file '{"path":"/sdcard/x.py","content":"..."}'
```

MCP 端点 `http://127.0.0.1:38985/mcp`，token 在
`~/.config/Code/User/globalStorage/kendryte747.canmv-vscode/mcp-http-token`。

常用：`canmv_write_file` / `canmv_write_and_run_script` / `canmv_execute_file` /
`canmv_terminal_output` / `canmv_stop_script` / `canmv_download_file_to_host` /
`canmv_list_dir` / `canmv_stat_file`。

## 上板脚本

| 文件 | 用途 |
|---|---|
| `k230/tft_live.py` | **最终版**：448 输入 + TFT 直显 + 24fps，每 60 帧存 `/sdcard/pose_preview_dbg.jpg` |
| `k230/shot_v2.py` | 跑 40 帧测速 + 最优帧画框存图（调试） |
| `k230/bench_k230.py` | int8/fp32 回归（需 `bench_bin/` 里的裸 RGB 数据） |
| `k230/diag*.py` | 排查用最小脚本（相机、格式、通道、尺寸） |

## 现场调参经验

- 靶子直径 **≥100 px** 才稳；< 60 px 时置信度会掉到 0.2 以下
- 画面整体偏暗（均值 < 100）时几乎检不出 —— 先把环境光提起来
- 屏幕反光/摩尔纹明显掉分，尽量正对、避免直射灯
