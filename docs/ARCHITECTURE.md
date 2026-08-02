# 系统架构与边界

## 目标

系统接受自然语言或预定义任务目标，完成搜索、候选确认、持续跟踪、周期重检、丢失重捕获，并只向 PX4/ArduPilot 输出经过安全审查的高级动作。

```text
Camera ──> YOLO-World + ByteTrack ──> target candidates ┐
Keyframe ─> small VLM ──────────────> task grounding ───┤
Candidates -> attribute check -> persistent selector ───┤
GPS / IMU / battery / range ────────> telemetry ───────┤
                                                       v
                                      World-state snapshot
                                                       v
                                      Mission state machine
                                                       v
                                        Safety arbiter
                                                       v
                                  high-level MAVLink gateway
                                                       v
                                          PX4 / ArduPilot
```

## 硬性边界

1. 检测、跟踪、语义验证只能产生观测，不得直接产生电机 PWM 或姿态内环控制量。
2. VLM 低频或事件触发运行；任何 VLM 建议都必须通过确定性规则和安全仲裁。
3. 飞控继续负责姿态、速度、位置、高度和 failsafe 闭环。
4. 模型 SDK 只存在于适配器层，核心状态机不依赖 RKNN、Ultralytics 或特定跟踪库。
5. 录制回放、SITL 和故障注入通过后，才允许进入系留或低风险实飞。

## 运行频率建议

| 模块 | 初始目标频率 | 触发方式 |
|---|---:|---|
| ByteTrack / 轻量跟踪器 | 20–30 FPS | 每帧 |
| 专用 YOLO | 5–10 FPS 或每 0.5–2 秒 | 周期重检、置信度下降 |
| YOLO-World | 低频 | 开放类别任务、重捕获 |
| 小型 VLM | 事件触发 | 任务解析、候选消歧、异常解释 |
| MobileSAM | 按需 | 需要精确轮廓时 |

频率是第一轮预算目标，不是硬件性能承诺；最终以目标板、分辨率和实际 RKNN 模型实测为准。

## 数据契约

所有框使用归一化 `xyxy`；所有置信度限定在 `[0, 1]`。核心观测至少包含：

- `label`、`box`、`detector_confidence`
- 可选 `tracker_confidence`、`track_id`
- `relation_verified`，用于“拿蓝色箱子的人”等关系目标
- 可选 `color_label`、`color_confidence`、`distance_m`
- `frame_id`、单调时间戳

统一契约能让电脑端 ONNX/模拟适配器与板端 RKNN 适配器互换，而不改状态机。

ByteTrack 提供 `track_id`，但不提供独立的逐轨迹置信度；本项目用“当前 ID 存在且当前检测分数合格”作为 tracking-by-detection 健康信号，并保留 `tracker_confidence` 给真正输出该量的跟踪器。

## 状态机

```text
SEARCH --candidate--> LOCK --confirmed--> TRACK
  ^                     |                   |
  |                     v                   v
  +----lock timeout-- SEARCH <---reacquire-- LOST
                                              |
                                              +--timeout--> HOLD

low battery / critical safety event ---------------------> RTL_LAND
```

- `SEARCH`：低速搜索并运行检测。
- `LOCK`：连续多帧确认类别、颜色、关系和距离。
- `TRACK`：跟踪器高频更新，检测器周期验证。
- `LOST`：减速/悬停并扩大搜索，候选重新进入 `LOCK`。
- `HOLD`：需要人工复核或显式重置。
- `RTL_LAND`：飞行安全终止状态。

## 线程与进程建议

后续接入真实视频时建议使用有界队列：采集线程只保留最新帧；检测与 VLM 丢弃过期任务；跟踪器优先实时性；状态管理器按时间戳拒绝乱序结果。不要让慢速模型造成无限缓存和控制延迟。
