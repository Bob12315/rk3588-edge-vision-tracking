# 本地 VLM 与 RK3588 NPU

## 选型

电脑端和 RK3588 统一选用 `Qwen3-VL-2B-Instruct`。

- 电脑端：Ollama `qwen3-vl:2b`，2.13B 参数，Q4_K_M，模型约 1.9 GB。
- RK3588：Rockchip 官方示例的 `qwen3-vl-2b_vision_rk3588.rknn` + `qwen3-vl-2b-instruct_w8a8_rk3588.rkllm`。
- 使用方式：仅对关键帧低频调用，上下文限制为 4096，最多加载一个模型且单并发。

这是“同模型规模和同任务”的可迁移基线，不是“同硬件速度”的模拟。电脑的 AMD RX 6650 XT 与 RK3588 NPU 架构、量化和内存带宽都不同，板端速度必须实测。

## 本地部署

```bash
make vlm-install
```

安装脚本固定 Ollama 0.32.5，不使用 `sudo`，不建立 systemd 服务，运行时位于 `artifacts/ollama-runtime/`。

启动本机服务：

```bash
make vlm-serve
```

服务只监听 `127.0.0.1:11434`，设置 `OLLAMA_NO_CLOUD=1`，按 `Ctrl-C` 停止。另开终端首次拉取模型：

```bash
make vlm-pull
```

模型位于 `artifacts/ollama-models/`。运行时、模型、视频和生成结果均不提交 Git。

执行真实链路：

```bash
make track-white-person-local-vlm \
  VIDEO=/home/level6/视频/37abaa4512176295e7462a589a64674a.mp4
```

## RK3588 NPU 能否跑 VLM

可以。Rockchip 官方 `rknn-llm` 已提供多模态示例，完整链路分成：

1. 视觉编码器转换为 `.rknn`，使用 RKNN Runtime 在 NPU 上运行。
2. 语言模型量化为 `.rkllm`，使用 RKLLM Runtime 在 NPU 上运行。
3. 多模态 demo 把视觉 token 交给语言部分，生成最终回答。

Rockchip 官方已列出 Qwen3-VL-2B 的 RK3588 预转换文件和命令，因此这不是理论上的“可能”，而是有官方工具链和 demo 的部署路径。

## RK3588 官方性能参考

Rockchip 官方基准是在 CPU/NPU 最高频率下测得，不是本项目板卡的保证值：

- Qwen3-VL-2B 语言部分 W8A8：TTFT 383.62 ms，14.98 token/s，内存 1868.98 MB（序列 128、生成 64 token）。
- 完整多模态：448×448 图像编码 2.08 s，prefill 649 ms，decode 14.91 token/s。
- 官方注明图像编码器以 FP16 在全部 NPU 核心上测试。

按当前电脑端结构化回答约 151 token 粗略估算，板端一次冷关键帧分析可能在 12–13 秒量级；这是基于官方分阶段数据的推算，必须用实际板卡和本项目提示词验证。因此板端应继续压缩输出 token，并只在搜索开始、悬停消歧或丢失重捕获时调用。

## 板端迁移顺序

1. 确认 RK3588 板卡内存、系统镜像、NPU 驱动和 RKLLM/RKNN Runtime 版本。
2. 先运行 Rockchip 预转换 Qwen3-VL-2B demo，记录首 token 延迟、token/s、峰值内存和温度。
3. 封装 RKLLM C API 或本机服务适配器，输出与电脑端相同的 `VlmSceneAnalysis`。
4. 将 YOLO/YOLO-World 的板端 RKNN 适配器与 ByteTrack 接在同一契约后。
5. 设置资源仲裁：VLM 只在关键帧触发，不与实时检测长时间争用 NPU 核心。

官方参考：

- [Rockchip RKNN-LLM](https://github.com/airockchip/rknn-llm)
- [Rockchip RKLLM 官方基准](https://github.com/airockchip/rknn-llm/blob/main/benchmark.md)
- [Rockchip RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
- [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Ollama qwen3-vl:2b](https://ollama.com/library/qwen3-vl:2b)
