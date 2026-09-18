---
name: cv-codebase-onboarding
description: 分析陌生的计算机视觉代码库，梳理任务、数据流水线、模型结构、训练与推理链路、配置、实验产物及扩展点，并生成上手指南或项目级 AGENTS.md。适用于 YOLO、MMDetection、OpenMMLab 和自定义 PyTorch CV 项目；不用于与视觉模型无关的通用 Web 项目导览。
metadata:
  origin: cantonice/agent
---

# 计算机视觉代码库上手

用最少的文件读取建立可验证的项目地图，重点回答：数据如何进入模型、模型如何训练和评估、权重如何用于推理，以及新增数据集、模型组件或部署后端应从哪里改。

## 边界

- 适用于检测、分割、分类、姿态估计、旋转框、跟踪、深度估计等视觉任务。
- 覆盖 Ultralytics YOLO、MMDetection/OpenMMLab 和自定义 PyTorch 项目，也可按同一证据链分析其他 CV 框架。
- 默认只做静态调查和轻量只读检查。不要安装依赖、下载数据集或权重、启动完整训练、修改代码或生成 `AGENTS.md`，除非用户明确要求相应操作。
- 不要把目录名、配置名或依赖声明当成事实；以实际导入、调用路径和最终生效配置交叉验证。

## 依据与框架差异

当识别出 Ultralytics、MMDetection、MMEngine 或自定义 PyTorch 时，读取 [references/framework-evidence.md](references/framework-evidence.md) 中对应章节。该文件记录本 skill 各项框架规则的官方依据和维护日期；框架行为发生变化时，先更新依据，再更新本文件。

## 工作流

### 1. 低成本勘察

优先使用 `rg --files` 和 `rg`，并行收集以下信号；只读取能消除歧义的文件。

```text
环境与依赖
  pyproject.toml、setup.py、setup.cfg、requirements*.txt、
  environment*.yml、conda*.yml、poetry.lock、uv.lock、Dockerfile

框架特征
  ultralytics 导入、yolo CLI、模型/数据 YAML
  mmdet、mmengine、mmcv、configs/、tools/train.py、Registry
  torch.nn.Module、Dataset、DataLoader、自定义训练循环

生命周期入口
  train、val、test、predict、infer、demo、export、track、benchmark
  CLI、Python API、脚本、notebook、服务入口

视觉资产与配置
  模型配置、数据集配置、类别定义、标注转换、transform/augmentation、
  checkpoint、预训练权重引用、评估器、导出与部署配置

质量与自动化
  tests/、pytest 配置、lint/type-check、CI、最小示例、文档命令
```

输出两层目录快照。忽略 `.git`、缓存、构建目录以及大型实验产物目录的内容，但保留这些目录是否存在及其用途；不要递归枚举 `data/`、`datasets/`、`runs/`、`work_dirs/`、`outputs/`、权重和媒体文件。

先判断：

- 项目解决什么视觉任务，输入和输出分别是什么。
- 它是框架源码、研究复现、训练工程、推理应用、部署工程，还是多者组合。
- 主框架及版本约束是什么；PyTorch、CUDA、torchvision、mmcv/mmengine/mmdet 或 ultralytics 是否存在兼容约束。
- 用户最可能需要修改的是配置、数据、模型组件、训练策略、评估、推理还是部署。

### 2. 找到配置真相源

区分“默认值、继承配置、命令行覆盖、运行时派生值”，说明最终值从哪里来。

**Ultralytics 项目**

- 分开识别视觉任务与运行模式。
- 找到模型 YAML/权重、数据集 YAML，以及 Python API 或 CLI 的参数覆盖。
- 记录 `model`、`data`、`imgsz`、`batch`、`device`、`workers`、`pretrained`、`resume`、`amp`、`seed` 和输出目录等关键运行参数；只报告项目实际设置或继承的值。

**MMDetection/OpenMMLab 项目**

- 从被执行的实验配置出发，沿 `_base_`、`custom_imports` 和命令行 `--cfg-options` 追踪最终配置。
- 映射 `model`、train/val/test dataloader、pipeline、evaluator、train/val/test loop、optim wrapper、parameter scheduler、hooks、launcher、`load_from`、`resume` 和 `work_dir`。
- 环境已经可用时，可运行仓库自带的配置打印命令获取展开配置；否则静态追踪，并明确哪些覆盖关系尚未验证。

**自定义 PyTorch 项目**

- 找到参数解析、配置加载和默认值合并位置。
- 区分 `Dataset`、sampler、`DataLoader`、transform/collate、模型构建、loss、optimizer、scheduler、训练循环、评估和 checkpoint 逻辑。

### 3. 追踪两条核心链路

不要使用 Web 项目的“请求生命周期”。计算机视觉项目至少追踪以下两条链路，并写出真实符号和文件路径。

```text
训练链路
配置/CLI
  → 数据集与标注
  → transform / augmentation
  → sampler / DataLoader / collate
  → 数据预处理
  → 模型前向与 loss
  → optimizer / scheduler
  → runner / loop / hooks
  → checkpoint、日志与 evaluator

推理链路
配置 + 权重 + 输入
  → 解码与预处理
  → 模型构建和权重加载
  → 前向计算
  → 解码、阈值过滤、NMS 等后处理
  → 结果结构
  → 可视化、评估、导出或部署
```

如果项目只实现其中一条，明确另一条“不在此仓库范围内”，不要补造流程。

### 4. 建立 CV 专用地图

**数据契约**

- 数据集划分、图像/视频位置、标注格式、类别名与类别 ID。
- 坐标和尺寸约定：像素或归一化、`xyxy`/`xywh`、mask、关键点、旋转框等。
- train/val/test transform 的差异、随机增强发生位置、batch 组装方式。
- 类别数在数据配置、模型 head 和 evaluator 之间是否一致。
- 只记录路径和格式，不读取或展示不必要的真实数据样本。

**模型与扩展点**

- 从实际配置和构建代码中列出组件，例如 backbone、neck、head、loss、assigner、postprocessor；不要假设每个项目都具有这些层级。
- 对 OpenMMLab，追踪配置中的 `type` 如何通过 registry 定位到实现，并确认注册代码如何被导入。
- 标出新增模型、数据集、transform、metric、hook 或 exporter 的最小扩展路径。

**实验与权重契约**

- checkpoint 保存的是仅模型状态还是还包括 optimizer、scheduler、epoch/iteration、EMA 等恢复状态。
- 说明成功加载权重还依赖哪些模型配置、类别映射和预处理参数。
- 区分预训练、微调、断点续训和仅推理；不要把 `load_from` 与 `resume` 混为一谈。
- 定位日志、指标、可视化、checkpoint 和导出产物目录，并说明哪些属于生成物，不应手工编辑或提交。

**运行与复现条件**

- Python/PyTorch/CUDA 和关键框架版本。
- CPU、单 GPU、多 GPU/多机入口，以及 launcher 或分布式配置。
- batch size、梯度累积、混合精度、DataLoader workers、随机种子和确定性选项。
- 从仓库文档、CI 或脚本提取命令；不凭经验编造可运行命令。

### 5. 识别项目约定

只记录能由代码或 Git 历史证明的约定：

- 配置文件命名、继承层级和覆盖方式。
- 模型、数据集、transform、metric、hook 的注册或工厂模式。
- 实验目录、日志、checkpoint 和模型导出的命名方式。
- 测试分层：配置构建、数据流水线、模型前向、训练 smoke test、评估或导出测试。
- 最近提交和分支足够时，记录提交与 PR 约定；历史过浅时明确跳过。

## 输出 1：CV 上手指南

保持可快速浏览，但每个结论都附真实路径或符号。

```markdown
# CV 项目上手指南：[项目名]

## 项目定位
[任务、输入、输出、项目类型]

## 技术栈与兼容约束
| 层级 | 技术/版本 | 证据位置 |

## 关键入口
| 目的 | 命令或 API | 配置 | 实现入口 |

## 配置真相源
[默认值、继承、CLI 覆盖和最终配置的关系]

## 训练链路
[真实路径和关键符号]

## 推理链路
[真实路径和关键符号]

## 数据契约
[划分、标注、类别、坐标、增强、评估]

## 模型结构与扩展点
[组件构建方式；新增组件应修改的位置]

## 实验、权重与部署
[输出目录、checkpoint 语义、导出格式和部署入口]

## 常用命令
[仅列出仓库已定义且证据明确的环境、训练、验证、推理、测试命令]

## 修改导航
| 我想要…… | 首先查看 | 还需联动检查 |

## 已确认 / 尚未确认
[事实、推断及最小验证动作]
```

## 输出 2：项目级 AGENTS.md

仅在用户要求时创建或更新。先检查从仓库根目录到当前目录的 `AGENTS.override.md` 和 `AGENTS.md`，保留生效中的现有指令，并遵守 Codex 的目录层级覆盖规则。

内容只保留会影响后续开发的稳定事实：

```markdown
# 项目说明

## 环境与兼容性
[Python、PyTorch、CUDA、关键框架及安装入口]

## 配置与入口
[配置真相源；训练、验证、推理和测试命令]

## 数据与模型约定
[标注/类别/坐标约定；组件注册或构建方式]

## 实验产物
[runs/work_dirs/checkpoints/exports 等生成物规则]

## 修改后的最小验证
[配置构建、数据样本、模型前向、相关测试；不要默认执行完整训练]

## 安全边界
[不得擅自下载数据/权重、覆盖实验目录或启动昂贵训练]
```

## 停止条件与质量要求

- 能用较浅层证据回答问题时停止，不要为了“完整”阅读整个仓库。
- 不运行耗时训练来证明静态架构判断；需要动态验证时，提出最小 smoke test。
- 区分事实、合理推断和未知项。每个关键结论至少给出一个文件、符号、配置键或命令来源。
- 不将高 mAP、低 loss 或 checkpoint 可加载直接等同于模型正确；分别报告数据、训练、评估和推理证据。
- 不复制 README；补充入口之间、配置与实现之间、数据与模型之间的连接关系。
