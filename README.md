# RK3588 边缘机载视觉

面向无人机的目标搜索、锁定、持续跟踪与丢失重捕获项目。当前仓库先建立可测试的控制骨架，把感知模型、任务状态机、安全仲裁和飞控边界分开；YOLO-World、普通 YOLO 和 ByteTrack 已完成电脑端视频接入，RKNN 与 MAVLink 适配器将在后续阶段逐项加入。

原始方案见 [RK3588_VLM_YOLO_Tracking_Summary.docx](RK3588_VLM_YOLO_Tracking_Summary.docx)。工程化解读和边界见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 第一版范围

- 专用 YOLO 负责固定比赛类别，YOLO-World 负责运行时指定的开放类别。
- 轻量跟踪器逐帧运行，检测器周期重检并纠正漂移。
- VLM 只在任务解析、多候选消歧或目标丢失时低频触发。
- SAM 暂不进入第一版；只有像素级轮廓确有收益时再评估 MobileSAM。
- 所有视觉输出先经过状态机与安全仲裁器，禁止模型直接控制电机、PWM 或姿态内环。

## 当前已具备

- `SEARCH → LOCK → TRACK → LOST → HOLD` 的确定性状态机。
- 低电量、定位异常、近障和人工保持的安全覆盖逻辑。
- 模型无关的数据契约与可替换适配器接口。
- YOLO-World 运行时文本提示，以及普通 YOLO 对照基线。
- ByteTrack 多目标 ID、逐帧 JSONL 记录和轨迹统计。
- 与供应商无关的 VLM 结构化输出契约及离线回放入口。
- 无第三方依赖的模拟回放 CLI 和单元测试。
- RK3588 环境检查、数据采集清单和分阶段实施计划。

## 快速开始

要求 Python 3.10 或更高版本。当前模拟与测试不需要安装模型或第三方包。

```bash
make check
make demo
PYTHONPATH=src python3 -m edge_vision.cli --scenario lost
```

也可安装为开发包：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
edge-vision-demo --scenario nominal
```

### 电脑端 YOLO-World 开放词汇检测

主检测链路使用 `yolov8s-worldv2.pt`，运行时可传入文字类别。依赖放在项目虚拟环境，不安装到系统 Python：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-pc.txt
make detect-person VIDEO=/absolute/path/to/input.mp4
```

`make detect-person` 使用 YOLO-World、文本提示 `person` 和针对当前密集人群视频选取的 `0.05` 初始阈值，默认输出到 `outputs/person_yolo_world/`。首次设置文字类别时会额外下载 CLIP 文本编码器权重。其他场景必须重新验证阈值。

任意英文提示可直接通过 CLI 指定：

```bash
PYTHONPATH=src .venv/bin/python -m edge_vision.video_detection input.mp4 \
  --backend yolo-world \
  --prompts "person" "blue box" "red vehicle" \
  --output-dir outputs/open_vocabulary
```

输出包括：

- `annotated.mp4`：带人物框和置信度的视频。
- `detections.jsonl`：逐帧时间戳、归一化框、像素框和置信度。
- `summary.json`：检测数量、吞吐率和 P50/P95/P99 推理延迟。
- `preview.jpg`：检测人数最多的代表帧。

原视频、输出视频、模型和权重均不提交 Git。

保留普通 YOLO11n 作为固定类别速度与召回对照：

```bash
make detect-person-yolo VIDEO=/absolute/path/to/input.mp4
```

识别白衣人物使用“YOLO-World 找人 + 上半身白色验证”的两级流程：

```bash
make detect-white-person VIDEO=/absolute/path/to/input.mp4
```

直接把 `person wearing white clothes` 作为单一开放词汇提示，在当前视频中会误框多名红衣人物，因此不作为最终颜色判定。两级流程和本次结果见 [docs/WHITE_CLOTHING_DETECTION.md](docs/WHITE_CLOTHING_DETECTION.md)。

加入 ByteTrack 持续 ID，并回放结构化 VLM 任务：

```bash
make track-white-person VIDEO=/absolute/path/to/input.mp4
```

该命令使用 `configs/vlm_white_clothing_example.json` 中已验证的 VLM 输出契约，将短类别词 `person` 交给 YOLO-World，再做白衣属性复核和 ByteTrack 数据关联。

电脑端已选定并接入真实本地 `Qwen3-VL-2B`：

```bash
make vlm-install
# 终端 1
make vlm-serve
# 终端 2，首次执行
make vlm-pull
make track-white-person-local-vlm VIDEO=/absolute/path/to/input.mp4
```

Ollama 运行时和 1.9 GB 量化模型都存在 `artifacts/` 中并被 Git 忽略；服务只监听 `127.0.0.1`、关闭云功能、单模型单并发。完整链路与实测见 [VLM → YOLO-World → ByteTrack](docs/VLM_YOLO_WORLD_BYTETRACK.md)，RK3588 NPU 部署路径见 [本地 VLM 与 RK3588 NPU](docs/LOCAL_VLM_RK3588.md)。

当前测试视频的实测结果见 [YOLO-World 基线](docs/BASELINE_YOLO_WORLD_PERSON.md)和[普通 YOLO11n 对照](docs/BASELINE_PERSON.md)。

在 RK3588 板端执行基础盘点：

```bash
./scripts/check_rk3588.sh
```

## 仓库结构

```text
configs/                 状态机和安全阈值
docs/                    架构、路线图、硬件与数据准备清单
src/edge_vision/         核心数据契约、状态机、安全仲裁和管线
tests/                   不依赖硬件的单元测试
artifacts/               本地模型产物目录（大文件不入 Git）
datasets/                本地数据集目录（数据不入 Git）
scripts/                 环境检查脚本
```

## 接下来的里程碑

1. 用更多录制视频验证 YOLO-World 与 YOLO11n 人物检测，补充人工标注并计算精度。
2. 用专用长视频测试 ByteTrack 的遮挡、交叉、离画重捕获和 ID 切换率。
3. 采集实际相机、飞行高度和场地光照数据，训练固定类别 YOLO，并加入框内 HSV 颜色复核。
4. 转换 RKNN，在 RK3588 上测量预处理、NPU 推理、后处理和端到端 FPS。
5. 接入 MAVLink 高级动作前，先完成 SITL、录制回放和安全故障注入。
6. 把 Qwen3-VL-2B 从电脑端 Ollama 适配器迁移到 RK3588 的 RKNN + RKLLM 运行时。

详见 [docs/ROADMAP.md](docs/ROADMAP.md)。
