# RK3588 边缘机载视觉

面向无人机的目标搜索、锁定、持续跟踪与丢失重捕获项目。当前仓库先建立可测试的控制骨架，把感知模型、任务状态机、安全仲裁和飞控边界分开；真实 YOLO、YOLO-World、跟踪器、RKNN 与 MAVLink 适配器将在后续阶段逐项接入。

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

1. 使用录制视频接入普通 YOLO，固定输入输出契约并建立精度/时延基线。
2. 接入单目标跟踪器与周期重检，完成遮挡、交叉和离画重捕获测试。
3. 采集实际相机、飞行高度和场地光照数据，训练固定类别 YOLO，并加入框内 HSV 颜色复核。
4. 转换 RKNN，在 RK3588 上测量预处理、NPU 推理、后处理和端到端 FPS。
5. 接入 MAVLink 高级动作前，先完成 SITL、录制回放和安全故障注入。
6. 最后评估 YOLO-World 与轻量 VLM，仅处理开放目标和复杂语义。

详见 [docs/ROADMAP.md](docs/ROADMAP.md)。
