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

## 真实 VLM 尚待接入

当前电脑没有本地 VLM 服务/推理库，也没有可用的云端 API 环境变量。`configs/vlm_white_clothing_example.json` 是契约回放样例，不代表真实 VLM 已执行。

下一步需要在两条路径中选择一条：

1. 电脑端云 API：最快完成真实画面目录和候选复核，需要配置凭据，但不是断网方案。
2. 电脑端本地 VLM：先用 CPU/GPU 可运行的小模型完成适配，后续再换成 RK3588 能承载的模型和运行时。

无论选哪条，新适配器只需实现 `VisionLanguageModel.analyze()` 并输出 `VlmSceneAnalysis`，后续 YOLO-World、颜色复核、ByteTrack 和状态机都不需改。
