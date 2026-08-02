# 电脑端 YOLO-World 人物检测基线

## 输入与开放词汇配置

| 项目 | 值 |
|---|---|
| 测试视频 | `37abaa4512176295e7462a589a64674a.mp4` |
| 视频 SHA-256 | `dcf3f683111349479927bd962223bfbc9fca325d1bb25da8efb74f489b641209` |
| 分辨率 | 1280×960 |
| 帧率/帧数 | 30.105 FPS / 57 帧 |
| 时长 | 1.893 秒 |
| 模型 | `yolov8s-worldv2.pt` |
| 运行时文本提示 | `person` |
| 输入尺寸 | 640 |
| 置信度阈值 | 0.05 |
| 设备 | Intel Core i5-12400F，CPU 推理 |

模型文件的本次 SHA-256 为 `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792`。

## 结果

| 指标 | YOLO-World | 普通 YOLO11n 对照 |
|---|---:|---:|
| 成功处理 | 57 / 57 帧 | 57 / 57 帧 |
| 总检测框 | 1136 | 1032 |
| 每帧人物框 | 15–25，平均 19.93 | 16–20，平均 18.11 |
| 模型热身 | 118.76 ms | 779.00 ms |
| 稳态推理 P50 | 90.87 ms | 32.84 ms |
| 稳态推理 P95 | 96.25 ms | 37.08 ms |
| 稳态推理 P99 | 112.28 ms | 37.64 ms |
| 含解码、绘制、写盘吞吐 | 9.94 FPS | 23.85 FPS |

YOLO-World 在 6 帧中的框数少于普通 YOLO、5 帧相同、46 帧更多。代表帧人工检查未发现明显的非人物误框；但该结论只来自少量目视抽查，低阈值可能增加重复框或困难背景误检，仍需人工标注确认。

两组结果使用不同模型和阈值，检测框数量只能用于工程对照，不能替代 precision、recall 或 mAP。当前视频没有人工真值标注。

## 复现

```bash
make detect-person VIDEO=/absolute/path/to/37abaa4512176295e7462a589a64674a.mp4
```

也可以用任意文本类别替换 `person`：

```bash
PYTHONPATH=src .venv/bin/python -m edge_vision.video_detection input.mp4 \
  --backend yolo-world \
  --prompts "person carrying a blue box"
```

默认结果位于 `outputs/person_yolo_world/`。原视频、模型权重、CLIP 权重和生成结果均不提交 Git。

## RK3588 迁移边界

Ultralytics 官方将 `yolov8s-worldv2.pt` 标记为支持导出，但导出后需要把任务词汇固化；Rockchip 官方 RKNN Model Zoo 当前没有直接列出 YOLO-World 示例。因此不能把电脑端 PyTorch 成功等同于 RK3588 NPU 已支持。

上板前需要单独完成：

1. 用比赛任务的少量提示词生成并保存离线词汇模型。
2. 导出 ONNX，逐帧对比 PyTorch 输出。
3. 验证 ONNX→RKNN 转换、算子支持和后处理格式。
4. 使用同一视频测量量化前后召回、置信度漂移和 NPU 时延。

参考：[Ultralytics YOLO-World 文档](https://docs.ultralytics.com/models/yolo-world)、[Rockchip RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)。
