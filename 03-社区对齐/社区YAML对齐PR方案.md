# PTA Benchmark 社区 YAML 对齐 PR 方案

## 1. 背景与目标

当前 PTA 的 TorchBench runner 可以运行社区 benchmark，但模型特化配置没有完整按照社区 YAML 自动加载，造成社区和 NPU 对同一个模型的训练/推理模式、batch size、精度容忍度和 accuracy check 次数不一致。

社区目前将模型行为配置集中在：

- https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/torchbench.yaml
- https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/huggingface.yaml
- https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/timm_models.yaml

本 PR 的目标是让 PTA benchmark 自动发现并加载对应 YAML，同时保留现有命令、流水和 NPU 后端选择逻辑。

## 2. 不在本 PR 范围内

1. 不修改模型源码，不通过 monkey patch 强行让模型 pass。
2. 不把 CUDA 专属优化直接声明为 NPU 行为。
3. 不改变现有 backend、npu-backend、training、inference、only 等命令行参数。
4. 不引入 AscendForge 的内部 YAML、日常流水路径或服务器路径。
5. 不将内部 154/161 模型清单写入 PTA 主仓。
6. 不在主仓中引入网站、rawdata 和看板生成逻辑。

## 3. 配置分类

### 3.1 可以直接复用

这些配置描述模型本身或正确性检查方式，通常与设备无关：

- only_training
- only_inference
- batch_size.training 和 batch_size.inference
- dont_change_batch_size
- tolerance、higher、even_higher
- higher_fp16、higher_bf16
- cosine
- accuracy_check_runs
- use_iou_for_bool_masks
- iou_thresholds
- slow 和 very_slow

这些字段应进入统一配置对象，由 runner 在构造模型和执行 accuracy check 时读取。

### 3.2 需要 NPU 适配后才能复用

以下配置不能按名称直接照搬：

- disable_cudagraph：CUDA cudagraph 与 NPU ACLGraph/NPUGraph 不是同一个机制，只能由 NPU adapter 映射。
- non_deterministic：必须确认 NPU eager 和 compile 是否存在同类非确定性，不能仅凭 CUDA 标记关闭 NPU accuracy。
- force_amp_for_fp16_bf16_models：需要确认 PTA autocast、torch_npu AMP 和 Triton-Experimental 对应 dtype 的支持。
- force_fp16_for_bf16_models：需要先经过 NPU 精度验证。
- trt_not_yet_working：这是 TensorRT 后端名单，不应影响 DVM、Triton-Experimental 或默认 Inductor。
- skip：必须解析 skip 原因，按 backend、device、mode 分类。CUDA OOM 不应自动变成 NPU skip。

## 4. 配置优先级

建议从高到低采用以下优先级：

1. 命令行显式参数；
2. 设备和后端适配层的安全限制；
3. PTA/NPU backend override；
4. 社区 YAML；
5. runner 默认值。

例如，用户显式传 inference 时优先于 YAML 默认 mode；社区的 disable_cudagraph 只对 CUDA backend 生效；NPU 的 graph 开关由 NPU adapter 决定。

## 5. 配置路径发现

不能依赖当前工作目录，也不能写死服务器路径。建议使用 runner 自身路径定位配置：

    BENCHMARK_ROOT = Path(__file__).resolve().parent
    CONFIG_DIR = BENCHMARK_ROOT / "dynamo"

建议的加载顺序：

1. 与 runner 同仓、同版本的 YAML；
2. 显式环境变量指定的配置目录；
3. 找不到时返回空配置并打印一次 warning，继续使用旧 runner 默认逻辑。

建议增加仅用于调试和回滚的环境变量：

    TORCHBENCH_DISABLE_COMMUNITY_YAML=1

正常用户无需设置该变量，默认行为应自动加载社区 YAML。

## 6. 统一配置接口

建议在 common.py 或独立轻量模块中增加只读配置接口，不让 TorchBench、HuggingFace 和 TIMM runner 分别解析 YAML：

    class BenchmarkConfig:
        def model_modes(self, model_name): ...
        def batch_size(self, model_name, mode): ...
        def tolerance(self, model_name, dtype, backend, device): ...
        def accuracy_check_runs(self, model_name): ...
        def use_iou_for_bool_masks(self, model_name): ...
        def backend_flags(self, model_name, backend, device): ...

实现要求：

- YAML 只解析一次；
- 解析失败时给出文件名、字段名和模型名；
- 返回值使用明确默认值；
- 不直接暴露原始 YAML dict；
- 记录配置来源和最终生效值；
- 不修改已有命令行接口。

## 7. 三个 runner 的接入

### TorchBench

在模型加载和 accuracy check 前统一查询 mode、batch size、tolerance、accuracy check 次数、IoU 和社区 skip。

### HuggingFace

同步 train-only、inference-only、generation 相关 batch size、tolerance 和 cosine。网络下载失败要记录为 missing dependency or weight，不能永久写成模型 skip。

### TIMM

同步 inference-only、batch size、AMP dtype、cudagraph 相关配置、非确定性和 tolerance。NPU 遇到 CUDA-only 配置时必须经过 adapter 转换。

## 8. 建议修改文件

1. common.py：增加 YAML 加载、标准化配置和优先级处理。
2. torchbench.py：接入 TorchBench YAML，保持旧版 benchmark commit 导入兼容。
3. HuggingFace/TIMM runner：接入各自 YAML，只读取通用配置，backend/device 专属项交给 adapter。
4. 三个社区 YAML：与选定的 pytorch/benchmark commit 一起固定，并在 README 中注明来源 commit。
5. 测试文件：增加 loader、路径独立性、优先级、缺失 YAML 和 NPU 映射测试。
6. README：说明 YAML 来源、加载路径、配置优先级和回滚方式。

## 9. 分阶段实施

### 第一阶段：只读加载和观测

- 加载三个 YAML；
- 打印每个模型的配置来源和最终模式；
- 暂时不改变执行行为；
- 验证任意工作目录运行；
- 验证 YAML 缺失时旧命令仍可运行。

### 第二阶段：接入安全配置

先接入不涉及设备语义的配置：

- only-training/inference-only；
- batch size；
- tolerance；
- cosine/IoU；
- accuracy check runs。

### 第三阶段：接入 dtype 和 graph 配置

- 引入 NPU adapter；
- 逐项验证 AMP、BF16、FP16；
- 将 CUDA cudagraph 配置映射为 NPU graph 配置；
- 没有明确适配依据的配置保持不生效并记录 warning。

### 第四阶段：清理重复配置

对比 PTA 本地配置和社区 YAML，删除完全重复的本地配置，只保留 NPU 特有 override 和回滚开关。

## 10. 验证方案

静态验证：

    python -m py_compile benchmarks/torchbench/common.py
    python -m py_compile benchmarks/torchbench/torchbench.py
    python torchbench.py --help

路径验证：从仓库根目录、临时目录和 benchmark 目录分别执行相同命令，确保 YAML 都能找到。

模型验证至少覆盖：

- alexnet
- resnet50
- drq
- BERT_pytorch
- 一个 HuggingFace 模型
- 一个 TIMM 模型
- 一个 train-only 模型
- 一个 inference-only 模型
- 一个社区 skip 模型

NPU 后端验证：

    python torchbench.py --devices npu --backend inductor --npu-backend dvm --training --only alexnet --iterations 1
    python torchbench.py --devices npu --backend inductor --npu-backend triton_experimental --inference --only alexnet --iterations 1

检查 backend、mode、batch size、tolerance、YAML 来源，并确认 CUDA-only 配置没有误作用于 NPU。

## 11. PR 拆分建议

### PR 1：引入社区 YAML 和标准加载器

建议标题：

    feat: load upstream benchmark metadata from yaml

内容：

- 加入 YAML 文件；
- 增加路径独立 loader；
- 增加标准化配置对象；
- 保持默认执行行为不变；
- 增加单元测试和 README。

### PR 2：应用社区特化配置

建议标题：

    feat: apply upstream benchmark metadata to NPU runners

内容：

- 应用 mode、batch size、tolerance、IoU 和 accuracy runs；
- 增加 NPU backend adapter；
- 增加 NPU 验证结果；
- 保留 CUDA-only 配置隔离。

如果必须合成一个 PR，PR 描述中必须明确区分配置加载和配置生效两部分。

## 12. 验收标准

1. 三个社区 YAML 能随 benchmark commit 一起获取。
2. 从任意工作目录运行都能自动找到 YAML。
3. 旧版 README 命令无需修改即可运行。
4. only、training、inference 等 CLI 优先级不变。
5. 社区 mode、batch size、tolerance 在验证模型上与预期一致。
6. CUDA-only 配置不会误作用于 NPU。
7. YAML 缺失或字段异常时有清晰 warning，并可回退到旧逻辑。
8. TorchBench、HuggingFace、TIMM 各完成至少一个 NPU smoke。
9. 结果记录 YAML 来源 commit、配置文件和最终生效配置。
10. 不引入服务器路径、内部流水 YAML 和网站逻辑。

## 13. 推荐结论

推荐先提交“加载器 + 安全配置”版本，再提交 backend adapter 版本。这样 reviewer 可以先确认社区 YAML 的来源、路径和配置优先级，再单独审查 NPU 对 CUDA 特化项的处理，不会把“同步社区配置”和“修改 NPU 行为”混在一个难以验证的改动中。
