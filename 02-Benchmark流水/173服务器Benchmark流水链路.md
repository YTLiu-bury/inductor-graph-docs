# 173 服务器 Benchmark 流水链路

本文记录 173 服务器上当前 Benchmark 流水的主要调用链和路径关系，方便排查“脚本启动了但实际没有使用最新代码”这类问题。

## 1. 运行环境

流水运行在容器 `ascendforge_ci` 的 `lyt310_ci` Conda 环境中。宿主机上的脚本负责选择卡、进入容器和传递环境变量；模型实际执行发生在容器内。

本文不记录服务器密码、代理认证信息或其他敏感配置。

## 2. 推荐入口

新流水架构的工作目录是：

```text
<WORKSPACE_ROOT>/workspace_benchmark
```

典型调用关系：

```text
run_daily.sh
  -> run_daily_container.sh
      -> docker exec ascendforge_ci
          -> benchctl.py
              -> benchpipe/流水组件
                  -> Benchmark runner
                      -> torchbench.py
                          -> common.py
                              -> 模型实现与 profiler
```

环境配置通常由以下文件集中提供：

```text
<WORKSPACE_ROOT>/workspace_benchmark/config/env.local.sh
```

其中最关键的运行目录变量是：

```bash
TORCHBENCH_DIR=<BENCHBOARD_ROOT>/workspace_pr/BenchBoard/third_party/benchmarks/torchbench
```

因此，启动脚本所在目录和实际执行 Benchmark 的目录不是同一个目录。修改了 `workspace_benchmark` 中的调度代码，不代表已经修改了 `TORCHBENCH_DIR` 指向的 `torchbench.py` 或 `common.py`。

## 3. 旧入口

173 上还保留过历史流水入口：

```text
<WORKSPACE_ROOT>/ci/daily_pipeline.sh
```

它主要用于编包、安装和旧版 UT/OP 流程。不要根据文件名判断它就是当前 Daily Benchmark 入口，实际应先检查脚本内部是否调用了 `workspace_benchmark`，以及它最终传入的 `TORCHBENCH_DIR`。

排查入口时建议执行：

```bash
pwd
readlink -f ./run_daily.sh
grep -R "TORCHBENCH_DIR\|docker exec\|benchctl.py" -n \
  <WORKSPACE_ROOT>/workspace_benchmark \
  <WORKSPACE_ROOT>/ci
```

## 4. 实际 Benchmark 代码目录

当前运行时使用的 Benchmark 根目录是：

```text
<BENCHBOARD_ROOT>/workspace_pr/BenchBoard/third_party/benchmarks/torchbench
```

关键文件包括：

```text
torchbench.py       TorchBench runner 入口、模型加载和命令行参数处理
common.py           通用训练/推理、精度检查、耗时统计和 profiler 调用
npu_support.py      NPU 侧模型/算子适配 patch
profiler.py         CPU/NPU profiler 封装和 schedule
benchmark/          官方 pytorch/benchmark 仓库内容
```

该目录中还包含一个嵌套的官方 Benchmark Git 仓库：

```text
<BENCHBOARD_ROOT>/workspace_pr/BenchBoard/third_party/benchmarks/torchbench/benchmark/.git
```

检查实际版本：

```bash
cd <BENCHBOARD_ROOT>/workspace_pr/BenchBoard/third_party/benchmarks/torchbench/benchmark
git rev-parse HEAD
git status --short
```

需要注意：外层 BenchBoard 代码和内层官方 Benchmark 仓库是两个 Git 边界，不能只查看外层仓库的 commit 就判断实际使用的官方 Benchmark 版本。

## 5. 单模型执行链

以一个模型为例，执行过程大致如下：

```text
模型集合配置
  -> 调度器选择 suite / backend / mode / device
  -> 容器内设置环境变量
  -> torchbench.py 解析参数
  -> TorchBenchmarkRunner 加载模型
  -> benchmark/torchbenchmark/models/<model>/ 模型实现
  -> common.py 执行 eager 或 compile
  -> torch.compile(..., backend="inductor")
  -> torch_npu 注册 NPU Inductor backend
  -> dvm 或 triton_experimental 后端编译
  -> warmup / profiler / accuracy / timing
  -> 输出 pass、fail、耗时和 profiler 产物
```

通常会分别执行以下一种或多种模式：

- `train`
- `inference`
- `eager`
- `compile`
- `dvm`
- `triton_experimental`

具体模式由流水配置和命令行参数共同决定。模型的 train-only、inference-only、batch size、tolerance、跳过精度检查等特化配置，不应简单地从脚本文件名推断，必须回到当前使用的模型配置和 runner 参数确认。

## 6. 模型数据依赖

模型运行时的数据来源主要分为以下几类：

### TorchBench 内置数据

```text
<BENCHBOARD_ROOT>/workspace_pr/BenchBoard/third_party/benchmarks/torchbench/benchmark/torchbenchmark/data/.data
```

典型内容包括小型数据集、输入样例和模型运行所需的本地文件。缺失时常见表现是 `FileNotFoundError`，例如某个模型目录下的 `batch.pt` 不存在。

### 模型权重缓存

常见缓存位置包括：

```text
~/.cache/torch/hub/checkpoints
~/.cache/huggingface
~/.cache/huggingface/hub
```

实际位置可能由 `TORCH_HOME`、`HF_HOME`、`TRANSFORMERS_CACHE` 或流水环境配置改变。运行前应在容器内确认这些变量和目录，而不是只检查宿主机。

### TIMM 权重缓存

TIMM 模型可能通过 Hugging Face、GitHub Release 或 TIMM 配置的 URL 获取权重。网络不可用时，即使 Python 依赖齐全，也可能在模型加载阶段失败。因此离线缓存必须和运行环境中的缓存变量保持一致。

### 外部数据集或专用资源

Detectron2、MAML、语音模型、文档/视觉模型等可能依赖额外数据目录。此类路径通常由模型环境变量或模型实现决定，排查时应从日志中的第一个 `FileNotFoundError` 反查配置，不要只看最后的 `fail_accuracy`。

## 7. 产物边界

一次模型运行通常会产生：

```text
模型运行日志       pass/fail、异常栈、E2E step time、accuracy 结果
raw data           流水原始结果，供后处理使用
profiler 目录      算子耗时、设备活动和数据库/文本结果
编译缓存           TorchInductor/Triton 生成的临时编译文件
```

日志和 raw data 是排查必需品；profiler 目录体积可能很大，应通过 `BENCH_PROFILE_WARMUP` 和 `BENCH_PROFILE_ACTIVE` 控制采集窗口。编译缓存不应直接当作可发布产物，清理前要确认没有正在运行的模型复用它。

网站 JSON 生成和网站服务器同步属于后处理链路，不是单个模型执行的必要步骤。运行 Benchmark 时应先确认 raw data 和日志完整，再执行聚合、脱敏和上传。

## 8. 启动前检查清单

```bash
# 1. 确认当前目录和入口
pwd
readlink -f <WORKSPACE_ROOT>/workspace_benchmark/run_daily.sh

# 2. 确认容器和环境
docker ps --format '{{.Names}}\t{{.Status}}' | grep ascendforge_ci
docker exec ascendforge_ci bash -lc \
  'source /opt/conda/etc/profile.d/conda.sh && conda activate lyt310_ci && python -V'

# 3. 确认 CANN
docker exec ascendforge_ci bash -lc \
  'test -f <WORKSPACE_ROOT>/cann-9.2.0/set_env.sh && \
   source <WORKSPACE_ROOT>/cann-9.2.0/set_env.sh && which msprof'

# 4. 确认实际 Benchmark 根目录
docker exec ascendforge_ci bash -lc \
  'test -f <BENCHBOARD_ROOT>/workspace_pr/BenchBoard/third_party/benchmarks/torchbench/torchbench.py'

# 5. 确认官方 Benchmark 版本和本地修改
docker exec ascendforge_ci bash -lc \
  'cd <BENCHBOARD_ROOT>/workspace_pr/BenchBoard/third_party/benchmarks/torchbench/benchmark && \
   git rev-parse HEAD && git status --short'

# 6. 确认缓存和数据
docker exec ascendforge_ci bash -lc \
  'ls -ld <BENCHBOARD_ROOT>/workspace_pr/BenchBoard/third_party/benchmarks/torchbench/benchmark/torchbenchmark/data/.data \
          ~/.cache/torch ~/.cache/huggingface 2>/dev/null || true'
```

## 9. 核心结论

173 上最容易混淆的是“调度仓”和“实际 Benchmark 仓”：

```text
<WORKSPACE_ROOT>/workspace_benchmark
    负责组织流水、容器和模型任务

<BENCHBOARD_ROOT>/workspace_pr/BenchBoard/third_party/benchmarks/torchbench
    负责实际执行 torchbench.py、common.py 和模型代码
```

后续更新 PTA 适配、官方 Benchmark commit 或本地 profiler patch 时，必须先确认 `TORCHBENCH_DIR` 指向的实际目录，再进行备份、修改和 smoke test。否则很容易出现“代码已经更新，但流水仍然运行旧文件”的假更新。
