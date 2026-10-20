# A100 复现 PyTorch 社区 Benchmark Wiki

本文说明如何在只有 NVIDIA A100 的机器上，按照 PyTorch 社区公开仓库和运行方式复现 benchmark，并补采 GPU E2E 与 GPU OP 数据。

本文面向需要复现结果、定位模型问题和生成内部看板数据的开发者。它不替代昇腾 NPU 流水，也不要求使用 AscendForge。

## 1. 数据口径

### 1.1 社区看板主要是什么数据

PyTorch 官方 TorchInductor 看板以模型级 E2E 指标为主，通常包括：

- eager 与 compile 的端到端耗时；
- compile 开销、吞吐或相对性能；
- training 或 inference 模式；
- dtype、batch size、device、commit 和 workflow 信息。

官方文档明确说明，TorchInductor 性能结果使用 NVIDIA A100 运行，因此 A100 适合复现社区流水：

- [PyTorch Dynamo benchmark README](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/README.md)
- [PyTorch OSS benchmark infrastructure](https://github.com/pytorch/pytorch/wiki/PyTorch-OSS-benchmark-infra)

### 1.2 GPU OP 数据需要单独采集

社区模型级 CSV 通常没有每个模型的 eager OP 总耗时和 compile OP 总耗时。不要把 E2E 耗时复制到 OP 字段，也不要用社区的模型级时间推导不存在的算子数据。

GPU OP 数据应来自同一模型、同一模式、同一 batch、同一 dtype 下的 profiler 或 Inductor kernel benchmark，建议单独记录：

~~~text
model
mode                 # training / inference
device               # cuda
dtype
batch_size
eager_e2e_ms
compile_e2e_ms
eager_op_ms
compile_op_ms
profile_commit
torch_commit
benchmark_commit
~~~

E2E 和 OP 的比较必须保持同一设备和同一运行条件。A100 与 H100 的绝对毫秒数不能直接横向比较；跨设备比较应优先使用相对加速比或同一设备上的 commit-to-commit 差异。

## 2. 推荐目录布局

不要把 benchmark、模型缓存、profiler 产物和代理配置混在代码仓中。建议在 A100 机器上使用独立工作目录：

~~~text
/home/<user>/community_benchmark/
├── repos/
│   ├── pytorch/
│   └── benchmark/
├── env/
│   └── benchmark.env
├── cache/
│   ├── torch/
│   └── huggingface/
├── results/
│   ├── e2e/
│   ├── op/
│   └── manifest/
└── logs/
~~~

每个 benchmark 版本建议单独使用缓存目录，避免不同 torch、torchvision 或模型权重互相污染。

## 3. 代理和外部下载

### 3.1 代理原则

只使用公司或实验室批准的代理地址。本文不内置代理 IP、端口或账号，也不建议从网上随便找镜像替代官方源。

在机器上创建仅当前用户可读的配置文件，例如 env/benchmark.env：

~~~bash
export HTTP_PROXY=http://<approved-proxy-host>:<port>
export HTTPS_PROXY=http://<approved-proxy-host>:<port>
export ALL_PROXY=http://<approved-proxy-host>:<port>
export NO_PROXY=localhost,127.0.0.1,::1

export HF_HOME=/home/<user>/community_benchmark/cache/huggingface
export HF_HUB_CACHE=/home/<user>/community_benchmark/cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/<user>/community_benchmark/cache/huggingface/transformers
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
~~~

执行：

~~~bash
chmod 600 /home/<user>/community_benchmark/env/benchmark.env
source /home/<user>/community_benchmark/env/benchmark.env
~~~

不要把包含密码、token 或内部代理地址的文件提交到 Git，也不要把这些变量打印到流水日志。

### 3.2 先验证网络，再启动流水

~~~bash
curl -I --connect-timeout 10 https://github.com/pytorch/benchmark
curl -I --connect-timeout 10 https://huggingface.co
git ls-remote https://github.com/pytorch/benchmark.git HEAD
~~~

如果 GitHub 能访问但 Hugging Face 不稳定，优先把模型权重预下载到缓存，然后离线运行。只有在公司批准了 Hugging Face 镜像时才设置：

~~~bash
export HF_ENDPOINT=https://<approved-huggingface-mirror>
~~~

预下载完成后，模型执行阶段可使用：

~~~bash
export HF_HUB_OFFLINE=1
~~~

这样运行阶段不会因为临时访问 CDN 或外网而卡死。若某个模型确实缺权重，先关闭 HF_HUB_OFFLINE 单独补齐，不要让全量流水边跑边下载。

### 3.3 Git 使用代理

共享机器上不建议永久修改全局 Git 配置。可以只对单次命令传入代理：

~~~bash
git -c http.proxy="$HTTPS_PROXY" \
    -c https.proxy="$HTTPS_PROXY" \
    clone https://github.com/pytorch/benchmark.git repos/benchmark
~~~

## 4. 获取并固定官方仓库

### 4.1 PyTorch benchmark 仓库

官方仓库为 [pytorch/benchmark](https://github.com/pytorch/benchmark)。建议固定 commit，不要直接使用不断变化的 main：

~~~bash
cd /home/<user>/community_benchmark/repos
git clone https://github.com/pytorch/benchmark.git
cd benchmark
git fetch --tags --prune
git checkout <verified-benchmark-commit>
git rev-parse HEAD > /home/<user>/community_benchmark/results/manifest/benchmark.commit
python install.py
~~~

官方仓库的基础单模型入口是：

~~~bash
python run.py <model> --device cuda --test train
python run.py <model> --device cuda --test eval
~~~

参考：[pytorch/benchmark run.py](https://github.com/pytorch/benchmark/blob/main/run.py)。

### 4.2 PyTorch 官方仓库中的社区 runner

如果需要尽量贴近 PyTorch 社区 TorchInductor 看板，使用 PyTorch 仓库中的 benchmarks/dynamo runner：

~~~bash
cd /home/<user>/community_benchmark/repos
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git fetch --prune origin
git checkout <verified-pytorch-commit>
git rev-parse HEAD > /home/<user>/community_benchmark/results/manifest/pytorch.commit
~~~

Torch、torchvision、torchaudio、torchtext 等应来自同一套兼容的构建来源。不要把 conda 中的一部分 torch 家族包和 pip 中的另一套随意混装。

## 5. A100 环境检查

~~~bash
nvidia-smi
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
~~~

建议确认 torch.cuda.is_available() 为 True、设备名称确实是 A100、驱动和 CUDA runtime 兼容，并且单卡能完成一个不依赖外部数据的 smoke 模型。

## 6. 先跑单模型，再跑分组和全量

### 6.1 PyTorch 社区 TorchBench smoke

~~~bash
cd /home/<user>/community_benchmark/repos/pytorch
CUDA_VISIBLE_DEVICES=0 \
python benchmarks/dynamo/torchbench.py \
  --accuracy \
  --training \
  --amp \
  --backend=inductor \
  --device=cuda \
  --disable-cudagraphs \
  --only=resnet18 \
  --output=/home/<user>/community_benchmark/results/e2e/torchbench_resnet18.json
~~~

推理 smoke：

~~~bash
CUDA_VISIBLE_DEVICES=0 \
python benchmarks/dynamo/torchbench.py \
  --performance \
  --inference \
  --bfloat16 \
  --backend=inductor \
  --device=cuda \
  --disable-cudagraphs \
  --only=resnet18 \
  --output=/home/<user>/community_benchmark/results/e2e/torchbench_resnet18_inference.json
~~~

disable cudagraphs 适合第一阶段验证，能减少捕获、dynamic shape 和显存复用造成的干扰；正式性能对比时应按社区目标参数分别跑开启和关闭两种模式，不能把关闭 cudagraph 的结果冒充默认社区结果。

### 6.2 HuggingFace 和 TIMM

PyTorch 官方 README 给出的 runner 形式如下，具体参数以所固定 PyTorch commit 中的 --help 为准：

~~~bash
cd /home/<user>/community_benchmark/repos/pytorch

CUDA_VISIBLE_DEVICES=0 \
python benchmarks/dynamo/huggingface.py \
  --performance --inference --bfloat16 \
  --backend=inductor --device=cuda \
  --only=bert-base-cased \
  --output=/home/<user>/community_benchmark/results/e2e/huggingface_smoke.json

CUDA_VISIBLE_DEVICES=0 \
python benchmarks/dynamo/timm_models.py \
  --performance --inference --bfloat16 \
  --backend=inductor --device=cuda \
  --only=resnet18 \
  --output=/home/<user>/community_benchmark/results/e2e/timm_smoke.json
~~~

如果某个 runner 不接受上述参数，先执行 python <runner> --help，以当前 commit 的帮助信息为准，不要把旧版本参数硬套到新版本。

### 6.3 使用 benchmark 仓库的低层入口

~~~bash
cd /home/<user>/community_benchmark/repos/benchmark
CUDA_VISIBLE_DEVICES=0 \
python run.py resnet18 \
  --device cuda \
  --test train \
  --profile \
  --profile-folder=/home/<user>/community_benchmark/results/op/resnet18
~~~

先用 --test eval 或 --test train 验证 eager，再进入 compile 和 profiler。这样能区分模型自身、依赖、网络和 Inductor 问题。

## 7. GPU OP 数据采集方案

### 7.1 轻量级 Inductor kernel benchmark

PyTorch 官方 Inductor profiling 文档建议打开唯一 kernel 名称和 kernel benchmark：

~~~bash
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCHINDUCTOR_BENCHMARK_KERNEL=1
~~~

然后运行单模型，例如：

~~~bash
cd /home/<user>/community_benchmark/repos/pytorch
CUDA_VISIBLE_DEVICES=0 \
python benchmarks/dynamo/timm_models.py \
  --performance --inference --bfloat16 \
  --backend=inductor --device=cuda \
  --only=resnet18 \
  --output=/home/<user>/community_benchmark/results/op/timm_resnet18.json
~~~

这套方式适合得到 Inductor 生成 kernel 的类别和单 kernel 耗时，产物通常比完整 Kineto trace 小很多。参考：[TorchInductor GPU profiling](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_inductor_profiling.html)。

### 7.2 必要时使用轻量 torch.profiler

只有在需要核对算子到 CUDA kernel 的映射时才打开 profiler。第一轮不要开启 Python stack、module hierarchy、shape 和 memory 记录：

~~~python
import torch

activities = [torch.profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)

with torch.profiler.profile(
    activities=activities,
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
    with_flops=False,
    with_modules=False,
) as prof:
    for _ in range(3):
        compiled_model(input_data)
        torch.cuda.synchronize()

print(prof.key_averages().table(
    sort_by="self_cuda_time_total",
    row_limit=100,
))
prof.export_chrome_trace("/home/<user>/community_benchmark/results/op/model.trace.json")
~~~

参考：[PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。

推荐策略：

1. 全量模型只采轻量 E2E 和 Inductor kernel 汇总；
2. 只有失败模型、性能异常模型和代表性模型采 Chrome trace；
3. trace 按模型和日期单独保存，定期清理；
4. 不要在全量流水中默认打开 with_stack=True、profile_memory=True、record_shapes=True。

### 7.3 Eager 与 compile 的 OP 对齐

要生成 eager_op_ms 和 compile_op_ms，必须保证：

- 相同输入和 batch size；
- 相同 warmup 次数和测量次数；
- eager 与 compile 分开预热；
- 每次 GPU 测量使用 CUDA event 或显式 torch.cuda.synchronize()；
- 去掉首次权重下载、首次编译、缓存命中等非稳态时间；
- 记录 profiling 配置和 commit。

不能直接把 profiler 的单个 kernel 总和当成 E2E。kernel 之间可能存在并发、同步和框架调度，OP 汇总应作为独立指标展示。

## 8. 结果目录和 manifest

建议每次运行使用如下目录：

~~~text
results/2026-09-23/
├── manifest.json
├── e2e/
│   ├── torchbench_train.json
│   ├── torchbench_inference.json
│   ├── huggingface_inference.json
│   └── timm_inference.json
├── op/
│   ├── model_summary.json
│   └── selected_models/*.trace.json
└── logs/
~~~

manifest.json 至少记录：

~~~json
{
  "date": "2026-09-23",
  "device": "NVIDIA A100",
  "cuda_visible_devices": "0",
  "torch_commit": "<commit>",
  "benchmark_commit": "<commit>",
  "mode": "training or inference",
  "dtype": "float32, amp, or bfloat16",
  "cudagraphs": "on or off",
  "dynamic": "on or off",
  "proxy_prefetch": true,
  "hf_offline_during_run": true
}
~~~

看板发布时建议保留两类数据：

- community_gpu_e2e：社区口径的 A100 E2E 数据；
- internal_gpu_op：单独采集的 GPU OP 数据。

两者可以在同一个页面展示，但字段和来源必须独立。社区基础结果格式允许扩展自定义 metric 和 extra_info，不过要让官方 HUD 直接显示新字段，通常还需要对应的 API fetcher 和 UI 支持，不能只往 JSON 里塞字段就认为社区看板会自动显示。[PyTorch benchmark infra](https://github.com/pytorch/pytorch/wiki/PyTorch-OSS-benchmark-infra)

## 9. 推荐落地顺序

### 第一阶段：环境和网络

1. 建立独立 Python 环境，不修改 NPU 容器和共享环境；
2. 固定 PyTorch 与 benchmark commit；
3. 配置批准的 HTTP/HTTPS 代理；
4. 验证 GitHub、Hugging Face 和模型权重下载；
5. 把权重预下载到独立缓存。

### 第二阶段：单模型和分组

1. resnet18 完成 eager、compile、accuracy smoke；
2. 选择一个 TorchBench、一个 HuggingFace、一个 TIMM 模型；
3. 分别验证 training/inference、float32/amp/bfloat16；
4. 确认 --help 参数与固定 commit 一致；
5. 再按模型类别分组跑，不要一开始启动全量。

### 第三阶段：全量 E2E

1. 先跑社区稳定看护集合；
2. 每个模型失败只记录失败类型，不在模型代码里临时打补丁；
3. 网络失败通过预下载和离线模式重跑；
4. OOM、timeout、依赖缺失与 backend compile error 分开统计；
5. 生成带 commit、设备、模式和 dtype 的 manifest。

### 第四阶段：选择性 GPU OP

1. 全量只开启轻量 Inductor kernel benchmark；
2. 对异常模型补采 profiler trace；
3. 输出 GPU E2E、GPU OP、compile overhead 三组独立指标；
4. 再把结果映射到内部看板，不覆盖原有 NPU 或社区数据。

## 10. 常见问题

### A100 能不能代表社区 GPU 数据？

可以作为复现社区口径的主要设备，因为 PyTorch 官方 TorchInductor 看板本身使用 A100。但 A100 结果不能直接当作 H100 结果，也不能用于证明某个模型在另一种 GPU 上的绝对性能。

### 为什么社区 CSV 没有 GPU OP？

社区模型级结果通常只上传预定义的 E2E、吞吐和编译相关 metric。算子级数据需要 profiler 或 kernel benchmark 产物，并且涉及更大的存储、解析和展示成本，所以通常不会随模型级 CSV 一起提供。

### 为什么不直接使用第三方 Hugging Face 镜像？

外部镜像会引入来源、完整性和合规风险。优先使用批准的代理访问官方源；如果必须使用镜像，应由网络或安全团队提供明确的域名和校验策略，并在 manifest 中记录。

### 为什么不把 profiler 全部打开？

完整 trace 会显著增加耗时、磁盘占用和内存压力，还可能改变模型执行行为。全量使用轻量 kernel 汇总，异常模型再单独做详细 profile，更适合稳定流水。

## 11. 官方参考

- [PyTorch Dynamo benchmark README](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/README.md)
- [PyTorch benchmark repository](https://github.com/pytorch/benchmark)
- [PyTorch benchmark run.py](https://github.com/pytorch/benchmark/blob/main/run.py)
- [TorchInductor GPU profiling](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_inductor_profiling.html)
- [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [PyTorch operator benchmark README](https://github.com/pytorch/pytorch/blob/main/benchmarks/operator_benchmark/README.md)
- [Hugging Face Hub environment variables](https://huggingface.co/docs/huggingface_hub/v0.16.2/package_reference/environment_variables)

