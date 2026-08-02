# 本地人物身份标注台

## 用途

该网页用于生成跟踪评测所需的逐帧人物身份真值。它是与视觉控制台分离的本地
工具，默认使用 <http://127.0.0.1:8766/>，不启动 VLM、YOLO 或飞控功能。

它保存两份文件：

- `annotations.json`：可继续编辑的项目，包含明确的 `reviewed_frames`。
- `gt.txt`：MOTChallenge 格式人物真值，可直接交给本项目的评测命令。

`datasets/` 已被 Git 忽略，人工标注和视频不会被误上传。

## 启动

可以空白开始：

```bash
make annotate-identities \
  VIDEO=/absolute/path/to/video.mp4
```

也可把 ByteTrack JSONL 当作人工审核的建议框：

```bash
make annotate-identities \
  VIDEO=/absolute/path/to/video.mp4 \
  IDENTITY_PREDICTIONS=outputs/tracker_comparison/bytetrack/detections.jsonl \
  IDENTITY_ANNOTATION_DIR=datasets/identity_eval/my_video
```

直接命令等价于：

```bash
PYTHONPATH=src .venv/bin/python -m edge_vision.annotation_ui \
  /absolute/path/to/video.mp4 \
  --predictions outputs/tracker_comparison/bytetrack/detections.jsonl \
  --output-dir datasets/identity_eval/my_video \
  --port 8766
```

## 界面规则

- 橙色虚线是模型建议，不是真值。
- 点“USE MODEL SUGGESTIONS”只会把建议复制到当前浏览器页面；仍需人工检查人员
  数量、框和 ID。
- 实线框是当前准备保存或已保存的真值。在画面上拖动可新建框。
- 同一个真实人在整个标注片段中必须使用同一个正整数 `identity_id`。
- 同一帧不允许两个框使用同一 ID，超出画面的框和非法 ID 会被后端拒绝。
- 目标被遮挡但仍可定位时，用 Visibility 记录可见比例。完全无法定位时不画框。
- 画面中没有人也要点“SAVE AS REVIEWED”；这样评测器才知道该帧是人工确认的
  空帧，而不是漏标。

页面在前后帧导航时自动保存已修改的当前帧。也可用 `Ctrl+S`、方向键和
`Delete`。关闭有未保存修改的页面时，浏览器会警告。

## 部分标注的正确评测

只标了部分帧时，必须把 `annotations.json` 传给评测器。评测器会同时过滤真值与
预测，只在 `reviewed_frames` 上计分：

```bash
PYTHONPATH=src python3 -m edge_vision.tracking_evaluation evaluate \
  datasets/identity_eval/my_video/gt.txt \
  outputs/tracker_comparison/bytetrack/mot.txt \
  --annotation-project datasets/identity_eval/my_video/annotations.json \
  --output datasets/identity_eval/my_video/bytetrack_metrics.json
```

不传 `--annotation-project` 表示输入的 MOT 文件已经完整覆盖评测范围。对稀疏或尚未完成的
标注这样做会把未标帧的预测错算成误报。

## 当前已启动的项目

开发机当前使用：

- 视频：`04fc81fbf08451011829cc6a084fad27.mp4`
- 建议：ByteTrack 前 800 帧 JSONL
- 输出：`datasets/identity_eval/brown_cyclist/`
- 地址：<http://127.0.0.1:8766/>

初始状态是 0 帧已审核、0 个真值框；程序没有自动把预测写入真值。
