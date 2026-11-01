# 固件镜像（不进仓库）

`webrtc` 模块只有 CanMV v1.8 之后的源码才编译，官方 Release 镜像都没有，所以这个项目
自带一版编好的固件。**镜像文件本身不在仓库里**（单个 240MB 以上，超过 GitHub 单文件
100MB 上限），只保留这份说明。

本目录在本地原来是：

| 文件 | 说明 |
|---|---|
| `CanMV_K230_YAHBOOM_webrtc_patched.img.gz` | 正式固件：`webrtc` + `libpeer` 都启用，日志级别 info |
| `CanMV_K230_YAHBOOM_webrtc_dbg.img.gz` | 调试固件：`LOG_LEVEL=DEBUG`，会把每个收到的包打到串口（**很吵，且串口阻塞会拖慢媒体循环**，只在排查时用） |
| `CanMV_K230_YAHBOOM_micropython_local_nncase_v2.11.0.img[.gz]` | 原厂基准镜像，用于回滚 |

## 自己编

见 [`../README.md`](../README.md) 的「为什么需要自己编固件」一节，要点：

1. 用 `canmv_k230` 分支（`ENABLE_MODULE_WEBRTC` 默认 `y`，并 `select LIBPEER`）；
2. 板级配置 `k230_canmv_yahboom_defconfig`；
3. 必须打上 [`../libpeer_patch/libpeer.diff`](../libpeer_patch/libpeer.diff) 里的 5 个补丁
   （`agent_recv` 提前返回、CHECKING 状态收包、`printf` 重定向到 `mp_printf`、DTLS 改 client 角色等）；
4. `IMG` 是按扇区写入 SD 卡的（不是拷文件），写之前先确认卡里没有要留的数据。

## 烧录

Windows 上把 SD 卡写 `.img` 的坑（写入中途自动挂载导致「访问被拒绝」）见
[`../WEBRTC.md`](../WEBRTC.md) 第 2 节和第 6.4 节。
