# CV 代码库上手规则的设计依据

最后核对：2026-09-18。

本文件说明 `cv-codebase-onboarding` 中框架专用规则为何存在。维护 skill 时，应先核对官方文档，再改变相应规则。这里只记录会影响调查决策的依据，不复制完整框架手册。

## 变更依据对照

| 从通用上手规则改为什么 | 原因与调查含义 | 官方依据 |
|---|---|---|
| 将 Web 的请求生命周期替换为训练和推理两条链路 | CV 项目的核心行为由数据加载、预处理、模型、loss/预测、评估和运行循环构成；训练与验证/推理的数据流不同 | [MMEngine Runner：基本数据流](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html#basic-dataflow) |
| 分开识别视觉任务和运行模式 | Ultralytics 将 detect/segment/classify/pose/OBB 等定义为任务，将 train/val/predict/export/track/benchmark 定义为模式；两者不能混为一类入口 | [Ultralytics 配置：Tasks 与 Modes](https://docs.ultralytics.com/usage/cfg/) |
| 把模型配置、数据配置和运行参数列为 YOLO 真相源 | Ultralytics 的训练入口分别接受模型文件、数据集 YAML 和覆盖默认值的参数；数据 YAML 包含训练/验证路径与类别信息 | [Ultralytics 配置：命令语法和训练参数](https://docs.ultralytics.com/usage/cfg/)、[检测数据集格式](https://docs.ultralytics.com/datasets/detect/) |
| 将验证、预测、导出、跟踪和 benchmark 纳入生命周期入口 | 这些模式分别覆盖评估、实际输入推理、部署格式转换、视频跟踪和速度/精度比较 | [Ultralytics Modes](https://docs.ultralytics.com/modes/) |
| MMDetection 必须从实验配置追踪最终生效值 | MMDetection 配置涵盖 model、dataloader、evaluator、loop、hooks、scheduler、optimizer、load/resume，并支持模块化继承与命令行覆盖 | [MMDetection：配置文件](https://mmdetection.readthedocs.io/en/v3.2.0/user_guides/config.html)、[训练与测试](https://mmdetection.readthedocs.io/en/v3.3.0/user_guides/train.html) |
| 检查类别定义与模型类别数的一致性 | MMDetection 自定义数据集要求在 dataloader 数据集配置中给出类别元信息，并同步覆盖模型中的 `num_classes` | [MMDetection：自定义数据集](https://mmdetection.readthedocs.io/en/latest/advanced_guides/customize_dataset.html) |
| OpenMMLab 项目要追踪 `type`、registry 和导入链 | MMEngine registry 管理同类模块，并让配置中的名字映射到构建实现；MMDetection 等算法库采用该机制 | [MMEngine Registry](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html) |
| 把 Runner、loop、optimizer、scheduler、hooks 和运行环境作为整体分析 | MMEngine 将 Runner 定义为组织和调度各模块的集成器，其配置包含数据加载器、训练循环、优化器、参数调度、hooks、分布式环境、加载与恢复 | [MMEngine Runner](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html) |
| 自定义 PyTorch 项目要分别定位 Dataset、DataLoader 和 transform/collate | PyTorch 将样本与标签存储定义在 `Dataset`，由 `DataLoader` 提供批处理、采样、多进程加载和内存固定等能力 | [PyTorch Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial)、[torch.utils.data](https://docs.pytorch.org/docs/main/data.html) |
| checkpoint 调查要区分模型状态和训练恢复状态 | `nn.Module.state_dict` 包含参数和持久 buffer；优化器也有自己的 state，因此“可推理权重”和“可恢复训练 checkpoint”不是同一契约 | [PyTorch Module.state_dict](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html)、[保存与加载模型](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) |
| 推理链路要核对 eval 状态 | PyTorch 官方说明推理前应调用 `model.eval()`，以正确设置 dropout 和 batch normalization 的行为 | [PyTorch：保存、加载与运行模型](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html) |
| 把随机种子、DataLoader workers 和确定性设置列入复现条件 | PyTorch 数据加载支持多进程 workers，官方文档同时提示多进程随机性和加载顺序会影响复现 | [torch.utils.data](https://docs.pytorch.org/docs/main/data.html) |
| `AGENTS.md` 必须按目录层级和 override 规则维护 | Codex 从仓库根目录向当前目录构建指令链；每层优先使用 `AGENTS.override.md`，更深目录的指令后加载并覆盖前面的规则 | [OpenAI：Custom instructions with AGENTS.md](https://developers.openai.com/codex/guides/agents-md) |

## 版本与适用范围

- Ultralytics 文档会随当前发布版本更新。skill 只依赖稳定概念：任务、模式、模型/数据配置和参数覆盖；不要固化某个 YOLO 代际名称或默认值。
- MMDetection 规则以 3.x 和 MMEngine 架构为主。遇到 2.x 仓库时，应根据其实际配置字段调整，不要套用 3.x 的 dataloader/runner 键名。
- 自定义 PyTorch 项目不一定使用 OpenMMLab 的 registry 或 Runner。只有检测到相关导入和调用时才应用对应规则。
- 其他 CV 框架应沿用“配置 → 数据 → 模型 → 训练/推理 → 评估/产物”的证据链，但不得假定其具体组件名称。
