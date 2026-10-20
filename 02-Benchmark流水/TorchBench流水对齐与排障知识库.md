# TorchBench Pipeline 对齐与排障知识库

> 本文档记录 BenchBoard 日常流水、AscendForge torchbench pipeline、PyTorch upstream benchmark 之间的对齐口径，以及 2026-09-14/15 对 0911 数据的排障结论。

## 1. lyt310_ci 环境状态澄清

`lyt310_ci` 环境本身不是不能用。问题出在某些进入容器的方式会触发 `/root/.profile` -> `/root/.bashrc`，而 `.bashrc` 中包含旧 conda 初始化和自动 source 逻辑：

```bash
<OTHER_HOME>/miniconda3/bin/conda shell.bash hook
source <WORKSPACE_ROOT>/miniconda3/envs/lyt310_ci/bin/activate
source /usr/local/Ascend/cann-9.1.0-beta.1/set_env.sh
```

在 170 机器上验证到：

```bash
docker exec ascendforge_ci /bin/bash -c 'echo ok'
# 正常

docker exec ascendforge_ci /bin/bash --noprofile --norc -lc 'echo ok'
# 正常

docker exec ascendforge_ci /bin/bash -lc 'echo ok'
# 可能超时
```

因此推荐排障或自动化脚本使用：

```bash
docker exec -it ascendforge_ci /bin/bash --noprofile --norc
source <WORKSPACE_ROOT>/miniconda3/envs/lyt310_ci/bin/activate
source /usr/local/Ascend/cann-9.2.0/set_env.sh
cd <BENCHBOARD_ROOT>/workspace_pr/BenchBoard
```

如果要彻底修复“普通 bash 进容器卡住”，建议把 `/root/.profile` 中 source `.bashrc` 的逻辑改成只对交互 shell 生效，或清理 `.bashrc` 里的旧 conda 初始化。不要把这个现象理解成 `lyt310_ci` 环境不可用。

## 2. 当前 BenchBoard 日常流水设定

本地和服务器上的 BenchBoard 主配置是 `auto_board/run.yaml`。主要口径：

| Suite | 入口脚本 | 默认模式 |
| --- | --- | --- |
| `Bench` | `python torchbench.py` | `--accuracy --cold-start-latency --train --dump-compile-time --enable-profiler --disable-aclgraph` |
| `HuggingFace` | `python huggingface.py` | 同上 |
| `Timm` | `python timm_models.py` | 同上 |

公共参数：

- backend: `inductor`
- iterations: `50`
- NPU backend 由运行脚本传入，当前主要是 `dvm` 和 `triton_experimental`
- 默认训练模式，少量模型通过 `overrides` 改成 inference
- 默认禁用 ACLGraph
- 默认打开 profiler，用于生成 E2E 和 OP 统计

## 3. AscendForge pipeline 当前设定

170 上当前 AscendForge pipeline 主要目录：

- `<WORKSPACE_ROOT>/AscendForge/torchbench_pipeline`
- `<OTHER_HOME>/AscendForge/torchbench_pipeline`

这两个目录当前不是 git 工作区，应以文件内容为准。

`<OTHER_HOME>/AscendForge/torchbench_pipeline/scripts/run_pipeline.sh` 的 daily/full 默认口径：

- `--training`
- `--execution-mode both`
- `--aclgraph-modes off`
- `--dynamic-modes both`
- `--iterations 50`
- `--repeat 1`
- `--timeout 3600`
- `--profiler`
- `--profiler-level 1`
- 默认 `NPU_BACKEND=triton_experimental`
- 默认 `TORCHBENCH_REUSE_ENV=1`
- 默认会 source CANN，优先 `CANN_92_HOME` / `CANN_HOME` / `<OTHER_HOME>/cann/ascend-toolkit/latest`

需要注意：`scripts/torchbench_pipeline.sh` 里当前 `TORCHBENCH_BENCHMARK_REF` 默认仍是：

```text
9910b31cc17d175a781412fd9ca6f18a4ee04610
```

这和“只适配新的 PyTorch benchmark commit”的目标不一致。后续如果上游 PR 合入，应把默认 benchmark ref、文档和兼容逻辑一起更新，不再保留旧 commit 的说明和兜底分支。

## 4. PyTorch upstream benchmark 对齐规则

最近验证使用的 upstream benchmark clone：

```text
8053e9a1e084289162c53bbb16292aa539471107
2026-09-10 22:29:28 -0700
```

关键配置来自：

- `userbenchmark/dynamo/dynamobench/torchbench.yaml`
- `userbenchmark/dynamo/dynamobench/huggingface.yaml`
- `userbenchmark/dynamo/dynamobench/timm_models.py`
- `userbenchmark/dynamo/dynamobench/timm_models.yaml`

对齐原则：

1. `hf_*` 老 TorchBench 内置模型不应再作为社区对齐模型跑在 `torchbench` suite 中。upstream `torchbench.yaml` 已在 `skip.all` 中跳过 `hf_Albert`、`hf_Bert`、`hf_T5_base`、`hf_distil_whisper` 等，原因是 HuggingFace suite 已覆盖对应模型。
2. `timm_*` 老 TorchBench 内置模型也不应作为社区对齐模型跑在 `torchbench` suite 中。upstream 已跳过 `timm_efficientnet`、`timm_nfnet`、`timm_vision_transformer` 等，TIMM 应从 `timm_models.py` / `timm_models.yaml` 走。
3. `huggingface.yaml` 中的模型级规则需要同步到我们的 `run.yaml` overrides，尤其是：
   - `only_inference`: `M2M100ForConditionalGeneration`
   - `only_fp32`: `GoogleFnet`
   - upstream skip 里的超大或不稳定模型，例如 `openai/gpt-oss-20b`、`mistralai/Mistral-7B-Instruct-v0.3` 等，不能直接纳入外部 106 看护。
4. `torchbench.yaml` 中的 `only_training`、`dont_change_batch_size`、`batch_size.training`、`tolerance`、`skip.all` 需要作为我们的模型入库和 override 依据。
5. `timm_models.py` 对 `emulate_precision_casts` 有特殊处理，尤其 `mobilenetv2_100` 相关精度差异不能简单当作模型问题，需要对齐 upstream 行为。

## 5. 106 模型与当前 run.yaml 差异

公开 106 模型名单见 `docs/community_dashboard_106_models_wiki.md`。

以当前 `auto_board/run.yaml` 统计：

| Suite | 106 中数量 | 已在 run.yaml | 缺失 |
| --- | ---: | ---: | ---: |
| `huggingface` | 30 | 27 | 3 |
| `timm_models` | 18 | 18 | 0 |
| `torchbench` | 58 | 45 | 13 |

缺失的 HuggingFace 模型：

- `Qwen/Qwen3-0.6B`
- `YituTechConvBert`
- `google/gemma-2-2b`

缺失的 TorchBench 模型：

- `basic_gnn_gin`
- `doctr_det_predictor`
- `doctr_reco_predictor`
- `functorch_dp_cifar10`
- `llama_v2_7b_16h`
- `llava`
- `microbench_unbacked_tolist_sum`
- `moco`
- `moondream`
- `opacus_cifar10`
- `pyhpc_turbulent_kinetic_energy`
- `sam`
- `sam_fast`

当前 `run.yaml` 中还存在一批非 106 或 legacy 模型，主要用于内部后端优化或历史对比：

- detectron2 扩展系列
- `hf_Albert`、`hf_Bert`、`hf_T5_base` 等老 TorchBench 内置 HuggingFace 模型
- `timm_efficientnet`、`timm_nfnet`、`timm_vision_transformer` 等老 TorchBench 内置 TIMM 模型
- quantized QAT 模型
- 其他内部扩展模型

建议未来看板分层：

- 外部/社区对齐视图：只看 106 模型，严格按照 upstream suite 归属跑。
- 内部优化视图：保留 extra 模型，用于扩大 triton_experimental 覆盖和定位后端问题。

## 6. 0911 triton_experimental 失败类型总结

0911 triton_experimental 全量结果：

- 总模型数：154
- `ok`: 70
- `accuracy_failed`: 16
- `partial`: 28
- `missing`: 40

相对历史曾经跑通但 0911 非 ok 的模型：

- `functorch_maml_omniglot`
- `hf_Albert`
- `hf_Bert`
- `hf_Bert_large`
- `hf_DistilBert`
- `maml_omniglot`
- `mobilenet_v2_quantized_qat`
- `resnext50_32x4d`
- `timm_vision_transformer`
- `torch_multimodal_clip`
- `beit_base_patch16_224`
- `mobilenetv2_100`

优先级分类：

| 优先级 | 类型 | 数量 | 典型模型 |
| --- | --- | ---: | --- |
| P0 | backend/codegen crash | 3 | `Super_SloMo`、`timm_nfnet`、`dm_nfnet_f0` |
| P0 | runner/env quick fix | 5 | `Background_Matting`、`maml_omniglot`、`sam`、`tacotron2` |
| P0 | HuggingFace YAML/name mapping | 6 | `DistillGPT2`、`GoogleFnet`、`T5Small` 等 |
| P0 | old `hf_*` wrong suite | 9 | `hf_Bert`、`hf_T5_base` 等 |
| P1 | previous-ok accuracy regression | 5 | `resnext50_32x4d`、`mobilenetv2_100` 等 |
| P1 | device/vector core error | 5 | `LayoutLM*`、`swin_base_patch4_window7_224` 等 |
| P2 | accuracy never stable or new | 11 | 需要逐模型确认 |
| P2 | mode/inference/seed issue | 15 | 多数应通过 override 修 |
| P3 | upstream skip/mode timeout | 9 | 应按 upstream skip 或 inference-only |
| P4 | runtime compile error/timeout | 11 | 需要最小复现 |
| P5 | OOM large model | 3 | 建议跳过或单卡特殊跑 |
| P6 | external auth/data | 1 | 需要数据/权限 |
| P7 | model unsupported | 1 | `moco` |

## 7. 已验证的小修复方向

这些验证不修改现有环境：

1. `maml_omniglot` / `functorch_maml_omniglot` 的 `batch.pt` 加载问题与 `torch.load(..., weights_only=True)` 安全白名单有关。添加 numpy reconstruct / dtype 类白名单后可加载。
2. `M2M100ForConditionalGeneration` 按 upstream `only_inference` 跑单模型 smoke 可 `pass_accuracy`，应在 YAML 里设置 inference override。
3. `mobilenet_v2_quantized_qat` inference 不支持，但 train/no-profiler/1 iter 可 `pass_accuracy`；日常失败主要是重 compile/profiler/timeout 配置问题。
4. `hf_Bert` 等老 `hf_*` 在新 upstream 规则下应从 torchbench suite 移除，不应继续当作社区对齐模型。
5. `DistillGPT2` 等 HuggingFace 新名字需要对齐 `huggingface.yaml`，不能依赖旧 wrapper 里的类名猜测。

## 8. 推荐的 YAML 对齐策略

### 8.1 先做安全无缝入库

优先加入不需要特殊数据或已验证可跑的 106 缺失项：

- `basic_gnn_gin`
- `functorch_dp_cifar10`
- `opacus_cifar10`
- `pyhpc_turbulent_kinetic_energy`
- `microbench_unbacked_tolist_sum`

这些模型仍需按 0911 日志确认是否需要 performance-only 或 profiler 关闭。

### 8.2 按 upstream mode 设置 overrides

应从 upstream YAML 自动或半自动生成以下类型 override：

- `only_inference` -> `flags` 改为 `inference`
- `only_training` -> 保持 `train`
- `skip.all` -> 外部视图不纳入，内部视图可单独实验
- `dtype.force_amp_for_fp16_bf16_models` / `emulate_precision_casts` -> 同步到运行参数或记录为已知精度前提

### 8.3 HuggingFace suite 不再用老 `hf_*`

外部 106 视图里：

- 删除或隐藏 `hf_Albert`、`hf_Bert`、`hf_T5_base` 等老 TorchBench 内置模型。
- 用 HuggingFace suite 的 `AlbertForMaskedLM`、`BertForMaskedLM`、`T5ForConditionalGeneration` 等替代。

### 8.4 TIMM 全部走 `timm_models.py`

外部 106 视图里：

- 保留 `timm_models` 中的 18 个社区模型。
- 老 `timm_*` TorchBench 内置模型只作为内部后端优化补充。

### 8.5 大模型和数据模型分层处理

需要额外数据/权限/显存的模型不应直接阻塞主流水：

- `Qwen/Qwen3-0.6B`
- `google/gemma-2-2b`
- `YituTechConvBert`
- `llava`
- `moondream`
- `sam` / `sam_fast`
- `doctr_det_predictor` / `doctr_reco_predictor`
- `llama_v2_7b_16h`

建议先建 `external_106_pending_data.md` 或 YAML 注释，记录数据路径、权限、预计显存和当前阻塞原因。

## 9. 后续动作建议

1. 更新 AscendForge `TORCHBENCH_BENCHMARK_REF` 默认值到新验证 commit，并删除旧 commit 兼容说明。
2. BenchBoard `run.yaml` 分离：
   - `community_106` 逻辑名单
   - `internal_extra` 扩展名单
3. 写一个小脚本从 upstream `torchbench.yaml` / `huggingface.yaml` / `timm_models.yaml` 生成 override 建议，避免手工漂移。
4. 对 106 缺失的 16 个模型逐个标注：
   - 已加入
   - 需要数据
   - 需要依赖
   - upstream skip
   - 仅 inference
   - 暂不支持
5. 新增 dashboard 过滤：
   - 外部视图只看 106
   - 内部视图看全部 run.yaml 模型

