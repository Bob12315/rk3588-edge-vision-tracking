# VLM → YOLO-World → ByteTrack 链路

## 职责分工

```text
关键帧 + 任务语句
        |
        v
VLM（低频）：画面摘要、对象目录、属性/关系、短提示词
        |
        v
YOLO-World（每帧）：用 person / blue box 等短词产生候选框
        |
        v
确定性或 VLM 候选复核：颜色、关系、任务约束
        |
        v
ByteTrack（每帧）：数据关联、持续 track_id
        |
        v
状态机 → 安全仲裁 → 高级飞控动作
```

VLM 不直接画框，也不逐帧运行。它把语义结果收敛为结构化任务；YOLO-World 负责开放词汇定位；ByteTrack 只负责跨帧数据关联，不理解“白衣”或其他语义。

## 已实现

- `VlmSceneAnalysis` 严格契约：画面摘要、对象目录、YOLO-World 提示词、必要属性和可选关系。
- JSON 校验与离线回放：可以把任意本地/云端 VLM 的输出固化后重复测试。
- YOLO-World V2 的运行时词汇和 ByteTrack `persist=True` 逐帧跟踪。
- 统一输出中的 `track_id`、带 ID 标注视频、逐帧 JSONL 和轨迹统计。
- 白衣任务使用 `person` 定位后的 HSV 上半身复核；结构化 VLM 结果声明 `white clothing` 是必要属性。

## 当前视频实测

输入：`/home/level6/视频/37abaa4512176295e7462a589a64674a.mp4`，1280×960，57 帧。

```bash
make track-white-person \
  VIDEO=/home/level6/视频/37abaa4512176295e7462a589a64674a.mp4
```

结果：

- 57/57 帧完整解码，无帧数警告。
- 白衣复核后有 45 个帧级观测，全部含 `track_id`。
- 共 3 个白衣相关 ID；各 ID 出现 5、13、27 帧。
- CPU 端到端处理约 10.15 FPS，平均模型+跟踪时间约 90.3 ms/帧。
- 本地输出位于 `outputs/white_clothes_yolo_world_bytetrack/`，大视频与逐帧结果不提交 Git。

这段素材只有约 1.89 秒，而且不是专门的跟踪评测集。它证明了工程链路和 ID 数据可用，不证明已通过遮挡、人员交叉、离画再入和 ID-switch 指标。白衣复核位于 ByteTrack 之后，某些帧的跟踪框如果包含较多背景或旁人，可能不通过颜色阈值；因此不能把“该 ID 本帧未输出”解读为跟踪器丢失。

## 真实本地 VLM

电脑端已使用 Ollama 0.32.5 和 `Qwen3-VL-2B` Q4_K_M 量化模型实现 `VisionLanguageModel.analyze()`。适配器通过本机 REST API 发送 JPEG 关键帧，用 JSON Schema 约束结果，再经本地契约复验。

在当前视频首帧上，真实 VLM 识别到人物、奖牌、水瓶和运动场景，并为白衣任务输出 `person` 和 `white clothing`。AMD RX 6650 XT Vulkan 实测：

- VLM 关键帧调用：模型已加载时 1.61 秒；含约 2.35 秒冷加载时为 5.68 秒。
- 生成速度约 122.8 token/s，上下文 4096，单模型单并发。
- 后续 57 帧 YOLO-World + ByteTrack 约 10.0 FPS，45 个白衣观测全部含 ID。
- 完整输出位于 `outputs/white_clothes_local_vlm_yolo_world_bytetrack/`。

VLM 只在任务开始、多候选消歧或丢失重捕获时运行，不占用每帧实时链路。RK3588 迁移细节见 [本地 VLM 与 RK3588 NPU](LOCAL_VLM_RK3588.md)。

统一运行入口已进一步接上持久单目标选择、任务状态机和安全仲裁，见[统一视觉任务主程序](UNIFIED_RUNTIME.md)。
