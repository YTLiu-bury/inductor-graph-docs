# PTA Benchmark 模型看护机制分析

> 分析日期：2026-10-08  
> 分析对象：Ascend/pytorch 当前 master 分支 benchmarks/torchbench 相关代码，以及 BenchBoard 外层流水配置。  
> 配套文档：[PyTorch 社区三套 Benchmark YAML 分析](pytorch_community_three_suite_yaml_analysis_20261008.md)。

## 1. 结论先行

当前 PTA 的 Benchmark 看护不是以 YAML 为中心，而是由以下部分共同决定：

1. torchbench.py：模型发现、模型列表过滤、NPU batch、NPU tolerance、learning rate、training/inference 限制。
2. npu_support.py：按模型注册 Python patch，处理 NPU 初始化、数据路径、模型 forward、算子分解和特殊依赖。
3. common.py：统一处理命令行、eager/compile、accuracy/performance、动态 shape、ACLGraph、profiler、timeout 和结果状态。
4. 外部 pytorch/benchmark 仓库：提供 torchbenchmark 模型注册表、模型实现、数据和部分社区通用元数据。
5. torchbench_models_list.txt：PTA 侧维护的可看护模型范围。
6. BenchBoard 的 run.yaml 或 daily pipeline：在仓外决定具体跑哪些模型、suite、backend、卡和日期。

整体调用关系：

~~~~text
BenchBoard run.yaml / daily pipeline
        |
        v
PTA benchmarks/torchbench/torchbench.py
        |
        |-- 发现外部 benchmark 模型
        |-- 读取 PTA 模型列表
        |-- 合并 PTA 自定义模型
        |-- 应用 skip 和环境过滤
        |-- 应用 NPU 模型策略
        v
PTA benchmarks/torchbench/common.py
        |
        |-- eager
        |-- torch.compile / Inductor
        |-- accuracy / performance
        |-- profiler / timeout / 结果状态
        v
pytorch/benchmark + torch_npu + CANN + NPU backend
~~~~

因此，当前 PTA 的看护模型不能只看一个文件，也不能只看外层 run.yaml。必须同时确认 benchmark commit、PTA runner、npu_support.py、模型列表和外层流水命令。

## 2. 官方源码入口

| 组件 | 作用 | 参考源码 |
| --- | --- | --- |
| PTA TorchBench runner | 模型发现、NPU 特化、加载模型 | [torchbench.py](https://raw.githubusercontent.com/Ascend/pytorch/master/benchmarks/torchbench/torchbench.py) |
| PTA 公共 runner | 参数、执行、精度、性能、profiler | [common.py](https://raw.githubusercontent.com/Ascend/pytorch/master/benchmarks/torchbench/common.py) |
| NPU 模型 patch | 模型级 Python 适配 | [npu_support.py](https://raw.githubusercontent.com/Ascend/pytorch/master/benchmarks/torchbench/npu_support.py) |
| PTA 模型列表 | PTA 侧允许进入 TorchBench 的模型 | [torchbench_models_list.txt](https://raw.githubusercontent.com/Ascend/pytorch/master/benchmarks/torchbench/torchbench_models_list.txt) |
| 社区 TorchBench | 上游模型实现和模型注册表 | [pytorch/benchmark](https://github.com/pytorch/benchmark) |
| 社区标准配置 | 社区通用 skip、batch、tolerance | [PyTorch torchbench.yaml](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/torchbench.yaml) |

## 3. 模型进入 PTA 流水的过程

### 3.1 定位外部 benchmark 代码

torchbench.py 的 setup_torchbench_cwd 会根据相对路径搜索：

~~~~text
./benchmark
./torchbenchmark
../torchbenchmark
../torchbench
../benchmark
../../torchbenchmark
../../torchbench
../../benchmark
~~~~

找到后会切换当前工作目录，并把该目录加入 sys.path。后续会从这里导入 torchbenchmark 和 benchmark。

因此外部 pytorch/benchmark 仓库不是附件，而是 PTA TorchBench 的模型代码和注册表来源。路径或版本不对时，可能出现：

- 找不到 torchbenchmark；
- 找不到 benchmark.userbenchmark；
- 模型列表为空；
- 导入了错误版本的 benchmark；
- 模型接口与 PTA 当前 torch/torch_npu 不匹配。

### 3.2 读取 PTA 模型列表

PTA 当前维护：

~~~~text
benchmarks/torchbench/torchbench_models_list.txt
~~~~

当前 master 文件包含 56 个模型条目。这个文件不是完整的外部 benchmark 注册表，而是 PTA 选择出来的模型范围。

### 3.3 与外部模型注册表求交集

iter_model_names 的逻辑可以概括为：

~~~~text
external_models = torchbenchmark._list_model_paths()
canary_models = selected torchbenchmark canary models
external_names = basename(external_models + canary_models)

pta_names = torchbench_models_list.txt
custom_names = benchmarks/torchbench/models 下的子目录

final_names = (external_names ∩ pta_names) ∪ custom_names
final_names = numpy_version_filter(final_names)
final_names = final_names - SKIP
sort(final_names)
~~~~

这会导致：

- 列表文件中有，但外部 benchmark commit 没有的模型被跳过；
- 外部 benchmark 有，但 PTA 列表没有的模型不进入 PTA TorchBench；
- PTA 自定义模型可以绕过外部注册表加入；
- canary 模型需要显式加入；
- NumPy 版本可能进一步过滤；
- SKIP 集合最终会排除模型。

所以实际模型集合不是列表文件行数，而是 PTA 列表、外部 benchmark 可用模型、自定义模型、canary、环境过滤和 skip 的组合。

### 3.4 模型加载和 mode 决定

PTA 会尝试从以下路径加载模型：

~~~~text
torchbenchmark.models.<model_name>
torchbenchmark.canary_models.<model_name>
torchbenchmark.models.fb.<model_name>
~~~~

然后根据 training/inference 选择 test=train 或 test=eval。

部分模型被标记为 inference-only。如果外层命令传入 training，runner 可能打印 warning 并改成 eval。结果必须记录 effective_mode，不能只看命令行参数。

## 4. PTA 当前的模型特化

### 4.1 Python 常量和集合

当前 torchbench.py 直接维护多类集合。

基础 SKIP 包括：

- detectron2_maskrcnn
- fambench_xlmr
- tacotron2
- maml

这些可能是资源过大、运行过慢或长期不稳定，不等价于 PTA 后端完全不支持。

当前 inference-only 包括：

- cm3leon_generate
- hf_distil_whisper
- pyhpc_equation_of_state
- pyhpc_isoneutral_mixing
- pyhpc_turbulent_kinetic_energy
- sam
- yolov3

### 4.2 NPU batch

当前源码内置的 NPU batch：

| 模型 | NPU batch |
| --- | ---: |
| alexnet | 64 |
| dcgan | 64 |
| lennard_jones | 4096 |
| nvidia_deeprecommender | 256 |

这些是 PTA/NPU 运行策略，不能直接当成社区 A100 默认 batch。

### 4.3 NPU accuracy tolerance

当前源码中存在 NPU 专属 tolerance：

| 分类 | 模型 |
| --- | --- |
| NPU_REQUIRE_HIGHER_TOLERANCE | dcgan、mobilenet_v2、timm_vovnet、phlippe_resnet |
| NPU_REQUIRE_HIGHER_TOLERANCE_ON_A5 | densenet121、resnet50、resnet18、resnext50_32x4d、resnet152 |
| NPU_REQUIRE_EVEN_HIGHER_TOLERANCE | shufflenet_v2_x1_0 |
| NPU_REQUIRE_HIGHER_FP16_TOLERANCE | timm_vision_transformer、functorch_dp_cifar10、moco、speech_transformer、timm_vovnet |

源码中 NPU 默认 accuracy tolerance 为 1e-3，部分模型会放宽到 1e-2、2e-2；Ascend 950 还有单独的 2e-3 分类。

这些规则与社区 YAML 中 GPU/XPU tolerance 的语义不同，不能直接混用。

### 4.4 NPU learning rate

PTA 还会按模型调整 training learning rate。

使用 1e-3：

- alexnet
- dcgan
- nvidia_deeprecommender
- BERT_pytorch

使用 1e-4：

- phlippe_densenet
- phlippe_resnet
- resnet50
- resnet152
- resnext50_32x4d
- densenet121
- vgg16

使用 1e-5：

- mobilenet_v2
- resnet18
- shufflenet_v2_x1_0
- timm_vovnet
- drq
- functorch_maml_omniglot

这会影响 training accuracy 和性能可比性，必须写入结果 metadata。

### 4.5 ACLGraph 特化

当前 PTA 标记为 ACLGraph 可能失败的模型包括：

- cm3leon_generate
- hf_BigBird
- yolov3

相关集合会被复用到 DVM、AKG、MLIR 等 NPU backend 策略中。

需要区分：

- 模型级 ACLGraph 捕获失败；
- backend 本身未启用；
- 外层统一传入 disable-aclgraph；
- 模型 compile 失败。

这些不能在结果中合并成一个状态。

## 5. npu_support.py 的 patch 机制

### 5.1 注册方式

PTA 使用模型名注册表：

~~~~python
_patch_table = {}

@register_patch("model_name")
def patch_function():
    ...
~~~~

运行时根据模型名查找并应用 patch。这样可以不修改外部 benchmark 模型源码，只在 PTA 适配层增加 NPU 行为。

### 5.2 patch 类型

当前 patch 主要包括：

- CUDA 判断和设备初始化适配；
- CPU 构造后迁移到 NPU；
- 模型输入和本地数据路径适配；
- HF T5、DistilBert 等 forward 重写；
- optimizer、batch、loss 和模型初始化顺序调整；
- decomposition、GENERATE_LIST、fallback kernel、NPU IR 调整；
- Triton-Experimental 和 ascend_npu_ir backend 的条件隔离。

例如 torch_multimodal_clip 会依次尝试从 TORCHBENCH_DATA_PATH、torchbenchmark.DATA_PATH 和模型自身 .data 目录查找 pizza.jpg。

### 5.3 patch 与 backend 的关系

并不是每个 patch 对所有 backend 都有效：

- ascend_npu_ir 相关配置主要服务 DVM、MLIR 等 backend；
- Triton-Experimental 不一定消费这些配置；
- 当前 npu_support.py 已经有 backend 判断，避免把 ascend_npu_ir 专属修改错误应用到 Triton-Experimental；
- 同一个模型可能需要通用 patch，也可能只需要某个 backend 的 patch。

### 5.4 patch 不是社区标准

npu_support.py 属于 PTA/NPU 平台适配层。除非问题也存在于社区通用 runner，并且有 upstream issue 或通用实现依据，否则不应直接把 NPU patch 提交到 PyTorch 社区 benchmark 配置中。

## 6. PTA 的执行模式

### 6.1 关键命令维度

| 维度 | 典型参数 |
| --- | --- |
| 设备 | --devices npu、cuda、cpu |
| backend | --backend inductor |
| NPU backend | --npu-backend default、dvm、triton_experimental 等 |
| mode | --training、--inference |
| 测试类型 | --accuracy、--performance、--precision-checker |
| dtype | --amp、--float16、--bfloat16、--float32 |
| iteration | --iterations、--iterations-per-run |
| 模型选择 | --only |
| dynamic | --dynamic-shapes、--dynamic-batch-only |
| ACLGraph | --disable-aclgraph |
| profiler | --enable-profiler、--export-profiler-trace |
| 调试 | --precision-checker-* |
| 资源保护 | --timeout |

### 6.2 accuracy

accuracy 流程大致为：

~~~~text
加载模型
    |
    v
构造 eager 输入和模型副本
    |
    v
运行 eager baseline
    |
    v
运行 eager rerun
    |
    v
运行 compile
    |
    v
比较输出、loss、梯度和必要的 FP64 reference
    |
    v
pass_accuracy / fail_accuracy / fail_to_run
~~~~

需要区分：

- eager 本身报错；
- eager 两次输出不同；
- compile 报错；
- compile 能运行但精度不满足；
- reference 或 profiler 失败；
- 数据或网络导致模型没有加载。

### 6.3 performance

performance 一般包含：

1. eager warmup；
2. compile；
3. compile warmup；
4. 多次稳定 iteration；
5. eager/compile step time；
6. speedup；
7. 可选 profiler 和算子数据。

编译时间、模型加载时间、下载时间和稳定 step time 不应混成一个性能耗时。

### 6.4 precision-checker

precision-checker 更偏向问题定位，支持：

- capture input；
- single pass；
- dump graph；
- 指定 graph 目录；
- 指定 cast dtype；
- 指定 module、module type；
- 忽略指定 module。

它和日常 accuracy 的关系是：

- accuracy：判断模型是否满足验收口径；
- precision-checker：定位哪个 module 或子图引入误差。

### 6.5 profiler

common.py 会根据 torch_npu 是否可用选择 NPUProfiler 或 CUDAProfiler。NPU profiler level 由 enable-profiler 参数控制。

开启 profiler 不等于最终一定有完整 OP 数据，因为还依赖：

- CANN/msprof；
- profiler level；
- CANN、torch_npu、triton-ascend 版本；
- 是否发生嵌套 profiler；
- trace/database 是否成功生成；
- 数据处理脚本是否成功解析。

所以 profiler 状态应单独记录，不能直接作为模型 pass/fail。

## 7. PTA、社区 YAML 和 BenchBoard 的职责

| 内容 | PyTorch 社区 | PTA 仓 | BenchBoard |
| --- | --- | --- | --- |
| 模型清单 | benchmark registry、HF/TIMM list | torchbench_models_list.txt + 自定义模型 | run.yaml 运行集合 |
| 通用 batch | YAML | Python 常量或外部 benchmark metadata | 外层 args/override |
| NPU batch | 不涉及 | torchbench.py | 可覆盖，但必须记录 |
| tolerance | YAML | torchbench.py NPU 集合 | 不应静默覆盖 |
| 模型 patch | 通用 runner 或模型仓 | npu_support.py | 不负责实现 patch |
| backend | inductor 等 | torch_npu/NPU backend | 指定本次流水 backend |
| dynamic | workflow/profile | common.py 参数支持 | run.yaml/脚本决定 |
| ACLGraph | cudagraph profile | PTA/NPU 特化 | 外层流水决定 |
| profiler | workflow 和 runner | common.py + profiler.py | 开启和产物整理 |
| 网站 JSON | 社区 dashboard 产物 | PTA 不直接负责 | rawdata/process/publish |

当前没有一个统一 PTA YAML，配置分散在：

~~~~text
模型集合：
  torchbench_models_list.txt

社区元数据：
  外部 benchmark commit 中的 Python 常量或 YAML

NPU 模型规则：
  torchbench.py

模型代码 patch：
  npu_support.py

公共执行机制：
  common.py

日常流水：
  BenchBoard run.yaml / daily pipeline

网站数据：
  BenchBoard rawdata/process/publish 脚本
~~~~

## 8. 当前架构的优点和问题

### 8.1 优点

- 能快速为 NPU 模型增加适配；
- 不必直接修改外部 benchmark 模型源码；
- 可针对 DVM、MLIR、Triton-Experimental 做不同处理；
- 旧 benchmark commit 上容易稳定运行；
- 部分数据路径支持环境变量覆盖；
- 公共执行逻辑集中在 common.py；
- patch 可以按模型注册，不影响所有模型。

### 8.2 主要问题

#### 配置分散

模型集合、batch、tolerance、ACLGraph、learning rate 和 patch 分布在多个 Python 文件中，review 和追溯成本较高。

#### 社区 commit 耦合

torchbench.py 直接导入外部 benchmark 的 Python 常量。上游把配置迁移到 YAML 后，如果 PTA 没有兼容层，会出现 import error。

#### mode 可能被模型覆盖

外层传入 training，模型内部可能因为 inference-only 切换成 eval。若结果不记录最终 mode，就会产生误判。

#### backend 语义不透明

部分 patch 只对 ascend_npu_ir 有意义，而 Triton-Experimental 不消费这些配置。没有 backend 判断时可能出现误 patch。

#### profiler 产物和业务结果耦合

profiler 目录可能很大，而且 profiler 失败不一定代表模型失败。应把 profiler 状态和模型 accuracy/performance 状态分开。

#### 模型列表和 benchmark 版本耦合

同一份 torchbench_models_list.txt 在不同 benchmark commit 上可能得到不同有效模型数量。

## 9. PTA 看护结果建议如何定义

### 9.1 是否进入流水

~~~~text
not_in_list
missing_from_external_benchmark
skipped_by_pta
skipped_by_numpy_version
selected
~~~~

### 9.2 eager 阶段

~~~~text
eager_pass
eager_error
eager_two_runs_differ
data_missing
dependency_missing
network_download_failed
~~~~

### 9.3 compile 阶段

~~~~text
compile_pass
compile_error
timeout
oom
process_killed
backend_not_enabled
~~~~

### 9.4 accuracy/performance 阶段

~~~~text
pass_accuracy
fail_accuracy
pass_performance
performance_regression
profiler_missing
profiler_parse_failed
~~~~

“模型跑通”至少要明确是 eager 跑通、compile 跑通、accuracy 通过、performance 有效，还是 profiler 数据完整。不能只看进程 exit code。

## 10. 推荐的演进方式

### 10.1 配置分层

建议逐步形成四层：

~~~~text
community_policy
  对齐 PyTorch 社区 batch、mode、tolerance、skip

pta_npu_policy
  NPU batch、NPU tolerance、ACLGraph 和 NPU-only 策略

pta_model_patch
  必须修改 Python 行为的模型 patch

pipeline_profile
  日期、torch/torch_npu/CANN/benchmark commit、卡、backend、dynamic、profiler、timeout
~~~~

YAML 适合承载 model、suite、mode、batch、dtype、tolerance、skip reason、disable ACLGraph、timeout、数据和依赖要求；复杂的 forward、optimizer、decomposition 和 NPU IR 修改仍应保留在 Python patch 中。

### 10.2 每次结果写入运行口径

建议每条模型结果至少记录：

~~~~text
suite
model
requested_mode
effective_mode
backend
dynamic
aclgraph
dtype
batch_size
learning_rate
tolerance
accuracy_status
performance_status
model_wall_time_s
eager_step_time_ms
compile_step_time_ms
profiler_status
error_category
torch_version
torch_npu_version_or_commit
cann_version
benchmark_commit
pta_commit
~~~~

requested_mode 和 effective_mode 必须同时存在，以识别 inference-only 模型对命令行的覆盖。

### 10.3 启动时输出模型过滤过程

建议流水启动时输出：

~~~~text
listed_models
available_external_models
intersection_models
custom_models
skipped_by_pta
skipped_by_numpy
final_models
~~~~

这样可以区分模型没有运行的原因：

- 没加入 PTA list；
- 外部 benchmark commit 没有；
- 被 PTA skip；
- 被环境版本过滤；
- 被外层 run.yaml 排除。

## 11. 与当前社区 106 和内部 full 的关系

建议长期保持三类集合：

~~~~text
community_snapshot_106
  外部看板固定口径

pta_supported_models
  PTA 当前已经有模型实现或适配的集合

internal_daily_full
  内部日常流水，包括社区模型和额外优化模型
~~~~

同时保持两类特化：

~~~~text
community_specialization
  社区已有的 mode、batch、tolerance、skip

npu_specialization
  NPU backend 的 batch、ACLGraph、依赖和 patch
~~~~

社区标准可以逐步通过兼容层接入；NPU 特化继续保留在 PTA policy 和 npu_support.py。

## 12. 最终判断

当前 PTA 的 Benchmark 看护逻辑可以概括为：

~~~~text
外部 benchmark 仓库提供模型实现和基础注册表
+
PTA torchbench_models_list.txt 选择看护范围
+
torchbench.py 提供 NPU 运行策略
+
npu_support.py 提供模型级代码适配
+
common.py 提供统一执行和统计
+
BenchBoard 提供外层流水编排和网站数据处理
~~~~

所以 PTA 仓里没有社区那种 YAML 是正常的。PTA 不是没有配置，而是配置分散在 Python 常量、模型列表文本、NPU patch 注册表、外部 benchmark 仓库、命令行参数和 BenchBoard 外层 YAML 中。

与社区 YAML 对照后，最需要改善的不是立即把所有代码改成 YAML，而是先建立清晰的配置边界和运行 metadata，避免社区标准、PTA NPU 特化和内部流水调度继续混在一起。

