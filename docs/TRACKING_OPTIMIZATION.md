# YOLO-World 与 ByteTrack 优化记录

## 当前到底是谁在每帧画框

是 YOLO-World。网页每处理一帧都会调用一次 Ultralytics `YOLOWorld.track()`：

```text
当前帧
  -> YOLO-World 检测框和分数
  -> ByteTrack 卡尔曼预测与 IoU/分数关联
  -> 带 track_id 的框
```

ByteTrack 不是光流或相关滤波单目标跟踪器；没有当前帧检测框时，它只能预测轨迹状态，不能可靠地从像素中重新找出目标。因此减少 YOLO-World 调用频率不能单靠调 ByteTrack 完成，需要另加 KCF/CSRT、光流或其他视觉 tracker，并保留周期性检测纠偏。

## 瓶颈测量

输入为 1280×960、57 帧的当前测试视频，目标词为 `person`，电脑端 CPU 推理。

384 输入下的同视频结果：

| 模式 | 平均推理 | 吞吐 |
|---|---:|---:|
| YOLO-World 检测，不使用 ByteTrack | 38.09 ms | 21.48 FPS |
| YOLO-World + ByteTrack | 39.72 ms | 20.53 FPS |

ByteTrack 增加约 1.63 ms，只占检测跟踪推理时间约 4%。因此原先约 10 FPS 的主要原因是 640 输入下每帧运行 YOLO-World，而不是 ByteTrack 本身太慢。

## 原配置的问题

原 `configs/bytetrack.yaml` 为了白衣弱目标召回，把 `track_high_thresh`、`track_low_thresh`、`new_track_thresh` 降到 `0.05 / 0.01 / 0.05`。同视频中产生 24 个 ID，其中 10 个只出现 1–2 帧；很多首帧弱框或重复框直接建立了短轨迹。

新增 `configs/bytetrack_balanced.yaml`：

```yaml
track_high_thresh: 0.12
track_low_thresh: 0.05
new_track_thresh: 0.12
track_buffer: 30
match_thresh: 0.8
fuse_score: true
```

含义是：至少 0.12 才允许建立新 ID，但 0.05–0.12 的检测仍可在第二阶段维持已有轨迹。

## 同视频对照

| 配置 | FPS | 平均每帧框 | ID 数 | ≤5 帧短 ID | ≥30 帧长 ID | 相邻帧近似 ID-switch |
|---|---:|---:|---:|---:|---:|---:|
| 原低阈值，640 | 9.71 | 10.8 | 24 | 11 | 10 | 1 |
| 官方默认，640 | 9.67 | 11.7 | 14 | 1 | 12 | 0 |
| 折中，640 | 9.97 | 11.4 | 18 | 3 | 11 | 0 |
| 折中，512 | 13.75 | 19.1 | 27 | 4 | 18 | 0 |
| 折中，416 | 18.53 | 16.3 | 24 | 2 | 16 | 1 |
| 折中，384 | 20.53 | 17.6 | 26 | 6 | 19 | 0 |

近似 ID-switch 使用相邻帧 IoU ≥ 0.5 的空间匹配统计，只用于同视频工程对照，不是带人工身份真值的 MOT 指标。当前镜头约有二十多人，不能把“ID 数少”直接等同于更好；官方默认配置较稳，但明显少框。

## 网页采用的方案

- ByteTrack 默认改为折中配置。
- 网页增加三档 YOLO-World 输入尺寸：实时 384、平衡 512、清晰 640。
- 默认实时档在当前视频上把批处理吞吐从约 9.7 FPS 提高到约 20.5 FPS。
- 页面“处理速度”统计完整的检测、ByteTrack、画框和 JPEG 编码循环；`model_fps` 另在状态 API 中保留，避免把纯推理速度当成端到端速度。
- USB 摄像头请求 `CAP_PROP_BUFFERSIZE=1`，尽量减少后端缓冲；不同 V4L2 驱动可能忽略该属性。

小目标、远距离目标或复杂开放词汇应选 512/640。384 是当前视频和电脑的速度优先选择，不是所有场景的精度最优值。

## 后续真正提升身份追踪质量

1. 采集带人物身份标注的遮挡、交叉和离画重入视频，计算 IDF1、HOTA、ID switches，而不是只看框是否存在。
2. USB/RTSP 使用独立采集线程和单元素最新帧队列，避免驱动不支持 `CAP_PROP_BUFFERSIZE` 时积压旧帧。
3. 比较 BoT-SORT + ReID；它通常比纯 ByteTrack 更能处理遮挡后身份恢复，但 CPU 成本更高，RK3588 也需要额外 ReID 模型预算。
4. 若最终只追一个用户选中的实例，可用 YOLO-World 周期重检 + 轻量单目标 tracker，换取更高帧率；这不是 ByteTrack 参数调整，而是管线模式变化。
