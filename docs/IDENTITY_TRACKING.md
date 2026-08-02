# 身份增强跟踪：BoT-SORT + ReID

## 为什么不只调 ByteTrack

ByteTrack 用检测分数、运动预测和框的空间重合做数据关联，本身没有人物外观
ReID。多人交叉、短时遮挡或相机运动时，只改 `track_buffer` 和 IoU 阈值不能提供
身份信息。

电脑端网页因此提供两种显式模式：

| 模式 | 跟踪器 | 外观特征 | 用途 |
| --- | --- | --- | --- |
| `bytetrack` | ByteTrack | 无 | 默认，速度优先 |
| `reid` | BoT-SORT | 有，`model: auto` | 遮挡和多人交叉的身份保持实验 |

实现使用 Ultralytics 的官方跟踪接口和配置字段，见
[Ultralytics Track 文档](https://docs.ultralytics.com/modes/track/)。

## 当前配置

`configs/botsort_reid.yaml` 是保守身份模式：

- `with_reid: true` 启用外观特征关联。
- `model: auto` 优先复用当前 YOLO 模型的原生特征，避免再加载一个分类模型。
- `appearance_thresh: 0.92` 只接受较严格的外观匹配，降低画面中有其他人时的误绑定风险。
- `gmc_method: sparseOptFlow` 用稀疏光流估计相机运动。
- `track_buffer: 60` 为短时遮挡保留轨迹，不把长时离画后的人盲目视为同一身份。

本机 Ultralytics `8.4.113` 的运行时检查结果是 `BOTSORT` 已创建，并已挂载模型
原生特征编码函数；当前不需要额外下载 ReID 权重。升级 Ultralytics 或更换
YOLO 模型后必须重新检查，不能假定 `auto` 始终走同一特征路径。

## 网页使用

在“指定目标”区域选择跟踪模式，再点“扫描并选择人物”或“开始检测跟踪”。
人物扫描、卡片选择和后续跟踪使用同一个跟踪器会话，中途不更换算法。页面
YOLO-World 提示词旁会显示实际跟踪器。

命令行可替换两个配置路径：

```bash
PYTHONPATH=src .venv/bin/python -m edge_vision.web_ui \
  --tracker-config configs/bytetrack_balanced.yaml \
  --reid-tracker-config configs/botsort_reid.yaml
```

## 实测结果

`04fc81fbf08451011829cc6a084fad27.mp4`、384 px、CPU：

- BoT-SORT + ReID 完成 18 帧扫描，获得 2 张稳定人物卡片，两人均有 18 个样本。
- 扫描阶段处理速度约 20.0 FPS，平均检测跟踪约 42.1 ms。
- 选择骑手后继续到第 268 帧，250 个后续帧都匹配 Track ID 2，未输出旁人 ID；
  这段处理速度约 19.0 FPS。
- 同视频 ByteTrack 的完整指定人物流程约 26.8 FPS；因此 ReID 当前代价约为
  25% 的帧率。

严格同参数的前 800 帧批处理对照中，ByteTrack 为 25.46 FPS，ReID 为 18.07 FPS，
即慢约 29%。两者的主目标都保持同一 ID 800 帧；ReID 没有在这段无真值视频上
证明身份收益，还多出 4 条 1–3 帧的低置信短轨迹。完整对照见
[跟踪身份评测](TRACKING_EVALUATION.md)。

这证明模式、人物卡片和不误切换链路可运行，不等于 ReID 精度已通过，也不支持
将 ReID 设为默认。该视频
没有逐帧身份真值，还需专用遮挡、交叉、离画重入数据集计算 IDF1、HOTA 和
ID switches。

## RK3588 路径

当前是电脑端 PyTorch/Ultralytics 实现，不能把 `model: auto` 的 Python 跟踪路径直接当成
RKNN 部署。Rockchip 的 [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
目前没有本项目可直接复用的人物 ReID 示例。上板前需在两条路线中实测选一条：

1. 导出或修改检测器，同时输出可用的外观 embedding，NPU 运行检测与特征。
2. 使用独立轻量 ReID 模型，例如
   [Torchreid OSNet x0.25](https://github.com/KaiyangZhou/deep-person-reid)，先导出 ONNX，
   再做 RKNN 算子兼容、量化精度和延迟验证。

目标板上仍保留 ByteTrack 作为实时基线，只在 ReID 对 IDF1/ID-switch 的收益大于 NPU
时间和内存成本时启用身份增强。
