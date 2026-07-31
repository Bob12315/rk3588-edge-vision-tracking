# 统一视觉任务主程序

## 已打通的主循环

`edge_vision.realtime` 已把此前分散的模块放进同一条持续运行链路：

```text
文件 / 摄像头 / RTSP
        |
        +--> Qwen3-VL-2B（任务开始、目标丢失事件）
        |          |
        |          +--> 场景对象 + YOLO-World 短提示词 + 必要属性
        |
        +--> YOLO-World(person) --> HSV 白衣复核 --> ByteTrack IDs
                                                        |
                                                        v
                                             持久单目标选择器
                                                        |
                                                        v
                                    SEARCH / LOCK / TRACK / LOST / HOLD
                                                        |
                                                        v
                                             安全仲裁后的高级动作
```

VLM 不逐帧运行，也不直接控制飞行。ByteTrack 没有独立的逐轨迹置信度，因此程序不会伪造该数据：存在稳定 `track_id` 且当前检测分数达到阈值时，状态机认为 tracking-by-detection 健康。白衣 HSV 复核通过后，观测才会标记为属性已验证。

## 启动

先启动仅监听本机的 Ollama 服务：

```bash
make vlm-serve
```

另一个终端运行统一程序：

```bash
make run-white-person \
  VIDEO=/home/level6/视频/37abaa4512176295e7462a589a64674a.mp4
```

摄像头和 RTSP 使用相同入口：

```bash
PYTHONPATH=src .venv/bin/python -m edge_vision.realtime 0 --vlm-provider ollama

PYTHONPATH=src .venv/bin/python -m edge_vision.realtime \
  rtsp://user:password@camera/stream --vlm-provider ollama
```

可用 `--display` 显示窗口并按 `q` 或 Esc 退出。没有 VLM 服务时，`--vlm-plan configs/vlm_white_clothing_example.json` 可做确定性回放；`--no-vlm` 则直接使用 `person` 提示词，仅用于诊断。

VLM 默认 HTTP 超时为 60 秒，可用 `--vlm-timeout-seconds` 调整。初始任务解析失败会中止启动；运行中目标丢失触发的 VLM 如果失败，会把错误写入事件日志并继续执行确定性重捕获，不会让逐帧主循环直接崩溃。

默认状态阈值在 `configs/white_person_runtime.json`。它们是针对当前短视频的首轮值，换相机、视角或飞行高度后必须重新标定。

## 目标选择规则

- 第一次从带 ByteTrack ID 的白衣候选中按检测分数、白色比例、框面积和画面中心度选一个目标。
- 当前 ID 仍在时始终保留，即使出现分数更高的新候选。
- 当前 ID 暂时消失时进入宽限窗口，不立即跳到另一个人。
- 超过 `--selector-miss-tolerance` 后才允许选择新 ID，并在事件日志中记录 `switched`。

该策略减少了多人画面中的无理由跳人，但跨越长遮挡的身份一致性仍需 ReID 或专用数据验证。

## 输出与审计

默认写入 `outputs/white_person_mission_runtime/`：

- `annotated.mp4`：候选绿框、当前目标紫框、状态与安全动作。
- `events.jsonl`：逐帧候选、选择事件、状态决策、VLM 事件和耗时。
- `summary.json`：状态占用、状态变化、ID 选择、VLM 调用和延迟分位数。
- `preview.jpg`：代表帧。

当前 57 帧视频的真实本地运行结果：

- Qwen3-VL-2B 首帧识别人、奖牌、水瓶和运动场景，为目标任务生成 `person` 及 `white clothing` 约束。
- 57/57 帧处理完成，白衣复核产生 45 个候选观测。
- 状态经历 `SEARCH → LOCK → SEARCH → LOCK → TRACK`，最终状态为 `TRACK`。
- 目标 ID 在宽限超时后按 `12 → 13 → 24` 重选，未发生即时抢占。
- YOLO-World + ByteTrack + 属性复核约 10.17 FPS，逐帧推理均值约 90.7 ms；初始 VLM 调用约 5.43 秒。

短视频证明主循环和审计链路可运行，不等于身份跟踪精度已经合格。后续必须用遮挡、交叉、离画重入和无人机运动素材统计 ID switch、重捕获耗时、漏检与误检。

## 当前安全边界

本程序只输出 `SEARCH`、`LOCK`、`TRACK`、`HOLD`、`RTL`、`LAND` 等高级决策记录，没有向真实 PX4/ArduPilot 发送指令。接入飞控前还需要：

1. 摄像头采集线程和只保留最新帧的有界队列，避免慢模型积压。
2. 视频冻结、链路断开和时间戳乱序检测，以及板端各模型的独立 deadline。
3. MAVLink 网关、SITL 故障注入、人工接管和心跳 failsafe。
4. RK3588 上的 YOLO-World RKNN、Qwen3-VL 视觉 RKNN + 语言 RKLLM 适配及实机压力测试。
