# Ascend Benchmark 流水与看板现状总览

> 版本：2026-09-23 介绍版  
> 用途：向项目负责人介绍当前模型看护范围、流水执行方式，以及后续与 PyTorch 社区 106 个稳定模型对齐的规划。

## 1. 结论摘要

当前流水已经具备较完整的 NPU benchmark 看护能力，内部 daily-full 模型池规模约为 **161 个模型**。

后续建议采用“双层模型池、单套执行框架”的方向：

1. **社区对齐池：106 个模型**，作为稳定、可对比、可对外发布的标准口径。
2. **内部扩展池：约 161 个模型**，用于覆盖更多算子、模型结构和后端问题，允许存在实验性模型。
3. 两个模型池复用同一套 runner、环境和数据缓存，只通过模型清单投影结果，不重复跑相同模型。
4. 社区已经明确的特殊配置，例如 inference-only、train-only、容差分类和 batch size 规则，优先跟随社区定义。
5. 能归因于 Triton Experimental 或 torch_npu 的问题，形成最小复现后优先向上游提 PR；纯环境、网络、数据和资源问题不应通过修改模型代码掩盖。

## 2. 当前看护范围

### 2.1 模型池

| 口径 | 模型数量 | 用途 |
| --- | ---: | --- |
| PyTorch 社区稳定看护集合 | 106 | 对外对齐、版本比较、稳定看板 |
| 内部 daily-full 模型集合 | 约 161 | 扩展覆盖、问题发现、后端优化 |
| 内部额外模型 | 约 55 | 不要求全部进入外部看板，可用于专项验证 |

社区 106 个模型来自三个 benchmark 套件：

| 套件 | 数量 | 主要覆盖 |
| --- | ---: | --- |
| TorchBench | 58 | 真实 PyTorch 模型、科学计算、推荐、强化学习、GNN、多模态和视觉任务 |
| HuggingFace | 30 | NLP、Transformer、LLM、seq2seq、语音和文档理解 |
| TIMM | 18 | CNN、ViT、Swin、ConvNeXt、EfficientNet 等视觉 backbone |

### 2.2 后端与执行模式

当前重点看护两个后端：

| 后端 | 作用 |
| --- | --- |
| DVM | 作为现有 NPU 编译后端基线，观察模型可运行性、精度和 E2E 性能 |
| Triton Experimental | 作为重点优化后端，观察算子覆盖、编译性能、算子耗时和 E2E 性能 |

当前 daily-full 的主要执行维度为：

| 维度 | 当前口径 |
| --- | --- |
| Backend | DVM、Triton Experimental |
| Dynamic | `off`、`on` |
| Accuracy | 记录 eager/compile 精度结果 |
| Performance | 记录 E2E 时间、算子时间、compile time |
| Profiler | 使用轻量 profile 采集可用于算子统计的数据 |
| ACLGraph | 应作为独立元数据记录，不能与 dynamic 模式混为一谈 |

因此，161 个模型在两个后端、两种 dynamic 模式下形成约：

`161 × 2 × 2 = 644 个 case`

这 644 个 case 是执行矩阵规模，不代表模型数量。

## 3. 模型分类与测试目标

下面的分类比“Bench/HuggingFace/TIMM”更接近性能分析目标。一个模型可以同时属于多个测试类别，因此这些类别不做简单求和。

| 测试类别 | 典型模型或模型族 | 重点观察内容 |
| --- | --- | --- |
| CNN/视觉 backbone | ResNet、DenseNet、MobileNet、VGG、AlexNet、ShuffleNet、TIMM CNN | 卷积、BN、激活、layout、算子融合和基础 E2E 性能 |
| Transformer/LLM | BERT、T5、OPT、GPT、Qwen、Gemma、Llama、XLNet | attention、linear、KV/cache、长序列、动态 shape 和显存占用 |
| 视觉 Transformer | ViT、DeiT、BEiT、Swin、DINOv2、SigLIP、Visformer | attention、patch embedding、reshape/view、compile 稳定性 |
| 检测/分割/后处理 | Detectron2、YOLO、MaskRCNN、FCOS、SAM | NMS、ROIAlign、proposal、动态后处理、CPU fallback 和精度 |
| NLP/文档理解 | BERT 系列、LayoutLM、ConvBert、Longformer、Pegasus | embedding、mask、长序列和模型特化容差 |
| 语音/音频 | Whisper、Demucs、Speech Transformer、TTS | 长序列卷积、频域计算、动态输入和模型权重下载 |
| 多模态 | LLaVA、CLIP、Moondream、torch_multimodal | 图像/文本双路径、预处理、跨模态融合和依赖完整性 |
| GNN/不规则计算 | EdgeCNN、GCN、GIN、SAGE | scatter/gather、稀疏访问、不规则 shape 和依赖兼容性 |
| 推荐/Embedding | DLRM、DeepRecommender | embedding、稀疏特征、MLP 和内存带宽 |
| 生成模型 | DCGAN、StarGAN、UNet、Super SloMo、LearningToPaint | 反卷积、上采样、生成器/判别器和训练路径 |
| 强化学习 | SAC、DRQ、Opacus CIFAR 等相关训练模型 | 小 batch、优化器、随机性和训练稳定性 |
| 科学计算/HPC | PyHPC、Lennard-Jones、等式状态模型 | elementwise、广播、归约、数值稳定性和算子融合 |
| Microbenchmark | `microbench_unbacked_tolist_sum` | 针对特定 PyTorch/Inductor 行为的回归验证 |
| 数据/依赖压力模型 | Detectron2、Doctr、SAM、HF 权重模型 | 数据集、外网、缓存、第三方库和容器环境完整性 |

## 4. 未来流水方向

### 4.1 模型集合治理

建立两份清单，但只维护一套执行逻辑：

```text
community-106.yaml  -> 外部稳定看板
daily-full.yaml     -> 内部全量优化和问题发现
```

`daily-full` 包含社区 106 个模型，因此社区模型不会因为内部扩展而丢失；外部看板通过模型集合投影，只展示 106 个模型的数据。

### 4.2 配置治理

未来 YAML 应明确记录以下字段：

- `suite`：TorchBench、HuggingFace、TIMM
- `mode`：train、inference、train+inference
- `dynamic`：on、off
- `aclgraph`：on、off
- `accuracy`：是否执行精度校验
- `iterations`：warmup 和 steady-state 次数
- `timeout`：单模型上限
- `batch_size` 和 divisor
- `tolerance` 分类
- `requires_data`、`requires_weights`、`requires_auth`
- `expected_failure_reason`：已知环境/依赖问题

这样可以把“模型特殊配置”和“后端代码逻辑”分离，减少在脚本中堆积临时判断。

### 4.3 结果与看板

每个 run 统一保留：

- 模型、suite、后端、dynamic、ACLGraph、CANN、torch_npu commit
- eager/compile 精度结果
- E2E 时间和算子平均时间
- compile time、cold-start 和 steady-state 时间
- 失败分类与原始日志路径

网站提供两种视图：

1. 社区 106 模型：稳定对外口径，支持日期对比。
2. 内部 daily-full：展示全部模型和失败归因，用于后端优化。

## 5. 后续看板与汇报口径

后续看板和阶段汇报建议围绕以下稳定指标展开：

- 社区 106 模型：计划数、eager pass、compile pass、精度 pass
- 内部 daily-full：计划数、完成数、可运行数、失败分类
- DVM 与 Triton Experimental：分别统计通过率、平均 E2E、平均算子时间
- dynamic-on/off：分别统计通过率和性能变化
- 版本信息：CANN、torch、torch_npu、torchbench commit

通过统一模型集合、执行配置和结果字段，保证不同日期之间可以进行可追溯的性能对比，同时让外部看板保持社区 106 个模型的稳定口径。
