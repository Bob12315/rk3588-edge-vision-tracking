# 本地网页视觉控制台

## 功能

网页把电脑端视觉链路整理成三个操作步骤：

1. 连接 USB 摄像头，或从浏览器上传本地视频。
2. 点击“分析当前画面”，由本地 Qwen3-VL 用英文输出场景摘要、可见物体和最多四个适合画框的目标类别。
3. 直接点击一个英文目标，或手动输入其他目标，再用 YOLO-World 定位、ByteTrack 分配持续 ID。

也可以点击“扫描并选择人物”。该模式先用 `person` 连续检测18帧，只把稳定
ByteTrack轨迹交给VLM生成英文人物卡片；选择卡片后继续原跟踪会话并只显示该ID。
详细说明见[逐人物扫描、属性目录与指定人物跟踪](PERSON_CATALOG.md)。

界面实时显示带框画面、检测词、当前 track ID、检测数量、处理 FPS 和平均推理延迟。它只做视觉观察，不包含飞控接口。

## 启动

先确认本地 VLM 已安装并启动：

```bash
make vlm-install

# 终端 1
make vlm-serve
```

再启动网页：

```bash
# 终端 2
make web-ui
```

浏览器会打开 <http://127.0.0.1:8765/>。如果没有自动打开，手动访问该地址即可。

也可直接运行并指定端口或推理设备：

```bash
PYTHONPATH=src .venv/bin/python -m edge_vision.web_ui \
  --host 127.0.0.1 \
  --port 8765 \
  --device cpu
```

默认只监听 `127.0.0.1`，没有登录功能，不会暴露到局域网。只有在可信私网中才应改为 `--host 0.0.0.0`。

## USB 摄像头

选择“USB 摄像头”，输入设备编号后连接：

- `0` 通常是第一个摄像头。
- `1` 通常是第二个摄像头。
- 后端通过 OpenCV 打开设备；Linux 上通常对应 `/dev/video0`、`/dev/video1`。

连接后网页持续显示最新画面。VLM 分析使用点击时的当前帧；开始检测后，YOLO-World 与 ByteTrack 处理后续画面。

## 上传视频

支持 `.mp4`、`.mov`、`.avi`、`.mkv`、`.webm` 和 `.m4v`。上传完成后先停在首帧，便于执行 VLM 分析；点击开始检测后从第 0 帧重新播放和处理。

上传视频保存在 `artifacts/web-ui/uploads/`，该目录已经由仓库的 `artifacts/*` 规则忽略，不会提交到 GitHub。默认单文件限制为 2048 MB，可用 `--max-upload-mb` 修改。

## 目标描述与检测词

VLM 分析结果中的“可框选目标”来自结构化 `yolo_world_prompts`，例如 `person`、`water bottle`。点击后会自动填入输入框并关闭二次 VLM grounding，点击“开始检测跟踪”即可直接交给 YOLO-World。

如果列表中没有需要的目标，也可以手动输入。默认勾选“使用 VLM 生成检测词”，适合输入中文或包含属性的目标，例如：

- `白衣服的人` → YOLO-World 提示词 `person`，并自动启用白色衣服分类与 Track ID 多帧稳定。
- `红色汽车` → 短英文目标词与颜色属性计划；当前普通物体只使用 YOLO-World 框，颜色复核仍需后续扩展。
- `water bottle` → 直接生成适合开放词汇检测的短词。

取消勾选后，输入内容会直接成为 YOLO-World 提示词；可以用英文逗号或中文逗号输入多个类别。直接模式不会启用属性复核，建议只用于简短英文类别词。

VLM 负责理解和生成 grounding 计划，不负责逐帧画框。YOLO-World 负责定位，ByteTrack 负责跨帧 ID。衣服颜色会经过通用颜色分类和按 Track ID 的三帧稳定，支持 12 个标准颜色；其他关系属性目前只展示在计划中，不会被错误地宣称为已验证。详见[衣服颜色识别与跟踪](CLOTHING_COLOR_TRACKING.md)。

## 当前视频实测

上传 `/home/level6/视频/37abaa4512176295e7462a589a64674a.mp4` 后：

- Qwen3-VL 用英文输出球场多人场景，并列出 `person`、`stadium seat`、`track`、`bottle` 四个可点击目标。
- 点击 `person` 后没有再次调用 VLM，直接把 `person` 交给 YOLO-World。
- 57/57 帧处理完成，实时档页面会话累计观察到 26 个 ByteTrack ID。
- 原 640 + 低阈值配置约为 9.7–11 FPS；默认实时档的批处理对照约为 20.5 FPS。不同目标词和视频会有不同速度与召回。

网页现提供三档检测尺寸：实时 384、平衡 512、清晰 640。实时档使用折中 ByteTrack 参数；小目标或远距离目标应改选 512/640。详细对照见 [YOLO-World 与 ByteTrack 优化记录](TRACKING_OPTIMIZATION.md)。

网页状态将完整检测、ByteTrack、画框和 JPEG 编码后的速度显示为“处理速度”；纯模型速度单独保留为 API 字段 `model_fps`。当前视频的网页实测分别约为 20.58 FPS 和 25.63 FPS。

## 运行边界

- 网页服务是单机、单视觉会话；新的视频源会替换旧的视频源。
- 切换目标会新建 YOLO-World/ByteTrack 实例，避免沿用上一个目标的轨迹状态。
- 本地 VLM 默认超时 60 秒；错误会显示在网页。
- USB 断流连续发生时会停止检测并提示重新连接。
- 当前 MJPEG 预览适合本地开发，不是低延迟生产流媒体协议。
- 暂未接飞控、MAVLink、云台或任何运动控制。
