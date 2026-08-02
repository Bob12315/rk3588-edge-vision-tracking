# 衣服颜色识别与跟踪

## 当前链路

颜色任务不再把 `person wearing red clothes` 这类复合文本直接交给
YOLO-World。网页把任务拆成：

```text
VLM / 用户描述
        ↓
标准颜色 + prompt "person"
        ↓
YOLO-World 人物框 + ByteTrack ID
        ↓
人物框中央上身区域的加权颜色分类
        ↓
按 Track ID 指数平滑，连续 3 帧后确认
```

支持 `white`、`black`、`gray`、`red`、`orange`、`yellow`、`green`、
`cyan`、`blue`、`purple`、`pink` 和 `brown`。英文或中文衣服描述都会被规范化，
但 VLM 的结构化输出继续使用英文。

## 为什么比原版本稳定

- 原版本只支持白色，直接统计整个矩形 ROI 内的亮、低饱和像素。
- 新版本对上身区域使用椭圆中心权重，降低人物框边缘背景的影响。
- 使用 HSV 规则与多亮度 Lab 颜色原型的混合得分；白/灰/黑先按亮度和饱和度分离，有色类别同时参考色相和感知色差。HSV 结果足够明确时跳过 Lab 比较，以控制实时开销。
- 每个 ByteTrack ID 维护独立的颜色状态；单帧反光、遮挡或压缩噪声不会立即改色。
- 只有颜色判断连续通过三帧才输出，不能把“轨迹存在三帧”误当成“颜色确认三帧”。
- 目标颜色除达到最低占比外，还必须接近框内主色，减少相邻人物颜色污染造成的误判。
- 同时扫描标准上身区和更高的上身区，适应骑车、弯腰等非直立姿态；颜色复核只接受人物检测置信度不低于 0.10 的框。
- JSON、视频标签和网页列表同时输出 `color_label` 与 `color_score`，便于检查。

## 当前视频回归

测试源：`/home/level6/视频/37abaa4512176295e7462a589a64674a.mp4`，CPU，
YOLO-World 384 实时档，折中 ByteTrack 参数。

| 任务 | 处理帧 | 匹配 Track ID | 吞吐 |
|---|---:|---:|---:|
| 仅人物检测跟踪 | 57 | 26 | 20.29 FPS |
| 白衣验证 | 57 | 2 | 17.80 FPS |
| 红衣验证 | 57 | 16 | 17.70 FPS |

自适应双区域会增加约一次轻量颜色取样，吞吐低于初版单区域，但 YOLO-World 仍是主要开销。白衣稳定为
ID 12 和 16。输出保存在 Git 忽略的
`outputs/color_validation_*` 中。

这些数字证明代码路径和时序状态正常，不等于颜色精度指标。视频还没有逐人、逐帧的
颜色真值，因此暂时不能报告 precision、recall 或 F1。

## 棕衣骑车视频回归

视频 `/home/level6/视频/04fc81fbf08451011829cc6a084fad27.mp4` 中的衣服在不同
光照下会落在棕色与铁锈橙之间，而且骑车姿态让原固定区域包含大量黑色裤子。当前实现
为 `brown` 加入受主色比例约束的铁锈橙相似度，并在两个上身区域中选择证据更强的一处；
纯橙色仍不能仅凭橙色分数直接通过棕色主色约束。

最终配置处理前 800 帧的结果：

- 单一 ByteTrack ID。
- 752/800 帧输出棕衣目标，命中帧率 94.0%。
- 24.31 FPS，平均颜色和检测总耗时 34.06 ms。
- 带框预览位于 `outputs/brown_debug/fixed_800_confirmed/preview.jpg`。

整段视频最后人物已经离开画面，因此最后一帧没有检测框。网页现在会额外显示累计匹配
帧数和历史 Track ID，避免把“最后一帧为空”误解为“整段视频从未识别”。

## 运行

网页中输入 `person wearing red clothes`、`白色衣服的人` 等描述，并保持
“使用 VLM 生成检测词”开启；也可关闭 VLM 后直接输入单一衣服颜色目标，程序会自动
把 YOLO-World 检测词改为 `person`。

命令行示例：

```bash
PYTHONPATH=src .venv/bin/python -m edge_vision.video_detection input.mp4 \
  --backend yolo-world \
  --prompts person \
  --clothing-color red \
  --tracker bytetrack \
  --tracker-config configs/bytetrack_balanced.yaml \
  --image-size 384 \
  --output-dir outputs/red_clothing
```

## 仍然存在的边界

当前实现是适合 CPU 实时运行的可解释 Lab/HSV 颜色分类器，没有额外运行人体解析神经网络。
中央上身取样可以降低背景污染，但无法彻底解决严重遮挡、彩色灯光、同色背景、花纹衣服
和极小人物。下一阶段应采集目标相机数据并标注衣服颜色，训练 MobileNetV3 小分类器；
板端将其转换为 RKNN，在 RK3588 NPU 上批量复核人物框。若任务要求像素级区分上衣和
裤子，再增加人体解析模型，不应在电脑端临时加入一个无法迁移的重型分割模型。
