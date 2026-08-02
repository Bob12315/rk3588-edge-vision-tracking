# 跟踪身份评测

## 目标

跟踪器输出的 ID 数量、画面观感或某个 ID 持续了多少帧，都不能单独证明身份跟踪
更好。本项目增加了两层评测：

1. 无身份真值时，比较速度、轨迹长度、短轨迹和相邻帧空间连续性。这些只是工程
   诊断，报告中显式标记为 proxy。
2. 有 MOTChallenge 格式的逐帧身份真值时，计算 IDF1、MOTA、MOTP、precision、recall 和
   ID switches，并导出可交给官方 TrackEval 的 MOT 文本。

本地快速评测不把自己的算法冒充为 HOTA。需要发布级 HOTA 或 MOTChallenge 对齐结果时，
使用[TrackEval 官方实现](https://github.com/JonathonLuiten/TrackEval)。

## 同参数双模式基准

```bash
make benchmark-trackers \
  VIDEO=/absolute/path/to/video.mp4 \
  TRACKER_BENCHMARK_FRAMES=800
```

命令固定为同一个 `yolov8s-worldv2.pt`、`person`、置信度 `0.05` 和 384 px 输入，
依次运行：

- `configs/bytetrack_balanced.yaml`
- `configs/botsort_reid.yaml`

结果默认位于 `outputs/tracker_comparison/`，包含两份带框视频、JSONL、运行摘要和
`comparison.json`。`outputs/` 已被 Git 忽略。

也可对既有运行单独诊断或比较：

```bash
PYTHONPATH=src python3 -m edge_vision.tracking_evaluation diagnose \
  outputs/run/detections.jsonl

PYTHONPATH=src python3 -m edge_vision.tracking_evaluation compare \
  --bytetrack outputs/byte/detections.jsonl \
  --reid outputs/reid/detections.jsonl \
  --output outputs/comparison.json
```

## 无真值诊断的边界

`spatial_continuity_proxy.approximate_id_switches` 会用 IoU 全局匹配相邻帧的框，统计空间上
连续的框是否换了 ID。它不知道真实人物身份，因此：

- 两人交叉时可能匹配错人。
- 遮挡期间没有框，相邻帧方法无法评价重现后的身份。
- 多出一个 ID 可能是新人、误检或轨迹碎片，不能直接计为 ID switch。

所以报告同时列出每个 ID 的首尾帧、观测帧数、分段数和平均置信度。少于 5 帧
的轨迹单独列为 `short_track_ids`，不与稳定轨迹混为一谈。

## 带真值的评测

导出预测文本：

```bash
PYTHONPATH=src python3 -m edge_vision.tracking_evaluation export-mot \
  outputs/tracker/detections.jsonl \
  datasets/identity_eval/predictions.txt
```

预测使用 MOTChallenge 的 1-based 帧号与十列格式：

```text
frame,id,left,top,width,height,confidence,-1,-1,-1
```

将人工标注的 `gt.txt` 与预测进行本地快速评测：

```bash
PYTHONPATH=src python3 -m edge_vision.tracking_evaluation evaluate \
  datasets/identity_eval/gt.txt \
  datasets/identity_eval/predictions.txt \
  --iou-threshold 0.5 \
  --output datasets/identity_eval/metrics.json
```

默认只接受真值中 `class_id=1` 的行，并忽略 mark/confidence 为 0 的区域。可用
`--gt-class-ids` 和 `--minimum-visibility` 调整。导出的 MOT 预测可进一步按 TrackEval
要求放入数据集目录，计算 HOTA 和官方实现的其他指标。

## 当前 800 帧对照

`04fc81fbf08451011829cc6a084fad27.mp4` 前 800 帧：

| 指标 | ByteTrack | BoT-SORT + ReID |
| --- | ---: | ---: |
| 处理速度 | 25.46 FPS | 18.07 FPS |
| 平均检测跟踪时间 | 31.96 ms | 49.33 ms |
| 稳定 ID（≥5 帧） | 2 | 2 |
| 主目标最长轨迹 | 800 帧 | 800 帧 |
| 少于 5 帧的短 ID | 0 | 4 |
| 相邻帧空间切换代理 | 0 | 0 |

ReID 速度是 ByteTrack 的 70.96%，即低约 29%。两种模式的主目标都持续 800 帧；
ReID 多出的 4 个 ID 只有 1–3 帧，平均置信度约 0.12–0.19，出现在画面边缘的额外
低置信检测上。不能在无真值的情况下判定它们是新人、误检还是轨迹碎片。

这段视频的结论是：继续保持 ByteTrack 为默认；ReID 仍是遮挡专项实验模式，当前没有
足够证据默认启用。下一步必须标注人员交叉、完全遮挡和离画重入片段。
