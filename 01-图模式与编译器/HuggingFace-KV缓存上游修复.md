# HuggingFace Benchmark KV Cache 对齐修复

## 背景

Ascend/pytorch 当前的 HuggingFace benchmark runner 在构造模型后，没有关闭模型的 KV cache。

对于 Bart、T5、OPT、Pegasus 等 decoder 类模型，KV cache 会引入跨次执行状态或改变输出结构，可能导致以下问题：

- eager 两次执行结果不一致；
- eager 与 compile 精度比较失败；
- training accuracy 流程无法稳定完成；
- 相同模型在社区 benchmark 与 Ascend benchmark 中使用不同测试口径。

PyTorch 社区当前的 HuggingFace benchmark runner 会在模型构造完成后设置：

```python
if hasattr(model, "config") and hasattr(model.config, "use_cache"):
    model.config.use_cache = False
```

社区代码位置：

<https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/huggingface.py#L441-L445>

因此，本修复属于对齐 PyTorch 社区 benchmark 行为，不是 NPU 专用精度规避。

## 修改方案

在 `benchmarks/torchbench/huggingface.py` 的 `HuggingfaceRunner.load_model()` 中，模型构造完成并生成输入后，关闭支持 KV cache 的模型配置：

```diff
         for attr in dir(config):
             if "drop" in attr and isinstance(getattr(config, attr), float):
                 setattr(config, attr, 1e-30)
 
+        # Match the upstream PyTorch benchmark's non-cached forward path.
+        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
+            model.config.use_cache = False
+
         if is_training and not (self.args.accuracy and model_name in ONLY_EVAL_MODE):
             model.train()
```

采用属性检查可以兼容不包含 `config` 或 `use_cache` 字段的 HuggingFace 模型，不改变这些模型的原有行为。

## 测试口径

关闭 KV cache 后，benchmark 测量的是普通 forward/training 路径，不代表带 KV cache 的自回归生成性能。

该口径与当前 PyTorch 社区 TorchInductor HuggingFace benchmark 保持一致，适合用于社区看板对齐和 eager/compile 精度比较。

## 流水验证

在 2026-09-16 全量流水中，DVM 和 triton-experimental 后端均使用了该修复。

与 2026-09-15 数据相比，以下 13 个 HuggingFace 模型在两个后端上均由未通过变为 `pass_accuracy`：

1. `BartForCausalLM`
2. `BlenderbotSmallForCausalLM`
3. `GPT2ForSequenceClassification`
4. `GPTNeoForCausalLM`
5. `GPTNeoForSequenceClassification`
6. `MBartForCausalLM`
7. `MT5ForConditionalGeneration`
8. `OPTForCausalLM`
9. `PLBartForCausalLM`
10. `PegasusForCausalLM`
11. `T5ForConditionalGeneration`
12. `TrOCRForCausalLM`
13. `XGLMForCausalLM`

整体通过情况变化如下：

| 后端 | 2026-09-15 | 2026-09-16 | 变化 |
| --- | ---: | ---: | ---: |
| DVM | 82/154 | 103/154 | +21 |
| triton-experimental | 70/154 | 96/154 | +26 |

其中两个后端共同新增通过的 13 个 HuggingFace decoder 模型，与本次 KV cache 修改直接相关。

## 建议验证命令

### 静态检查

```bash
python -m py_compile benchmarks/torchbench/huggingface.py
git diff --check
```

### NPU Smoke 验证

建议分别选择 causal LM、seq2seq LM 和 sequence classification 模型：

```bash
python benchmarks/torchbench/huggingface.py \
  --devices npu \
  --backend inductor \
  --npu-backend triton_experimental \
  --disable-aclgraph \
  --accuracy \
  --training \
  --only BartForCausalLM \
  --iterations 1 \
  --timeout 900

python benchmarks/torchbench/huggingface.py \
  --devices npu \
  --backend inductor \
  --npu-backend triton_experimental \
  --disable-aclgraph \
  --accuracy \
  --training \
  --only T5ForConditionalGeneration \
  --iterations 1 \
  --timeout 900

python benchmarks/torchbench/huggingface.py \
  --devices npu \
  --backend inductor \
  --npu-backend triton_experimental \
  --disable-aclgraph \
  --accuracy \
  --training \
  --only GPT2ForSequenceClassification \
  --iterations 1 \
  --timeout 900
```

## PR 范围

本 PR 仅包含 HuggingFace runner 的 KV cache 行为对齐，不包含以下流水特化：

- TIMM 模型强制使用 inference；
- TIMM 或 TorchBench 模型强制使用 cosine；
- `run.yaml` 中的 NPU 单模型配置；
- `M2M100ForConditionalGeneration` 的临时精度配置；
- HuggingFace/TIMM metadata YAML 结构迁移。

这些内容需要分别确认社区依据后，以独立 PR 处理。

## 建议 PR 标题

```text
fix: align HuggingFace benchmark KV-cache behavior with upstream
```

## Demo 信息

- 基线：Ascend/pytorch `master@1b64dadd89`
- Demo 分支：`codex/hf-disable-kv-cache-demo`
- Demo worktree：`D:\code\pytorch-hf-cache-demo`
- 修改文件：`benchmarks/torchbench/huggingface.py`
- 当前状态：代码修改完成，`py_compile` 和 `git diff --check` 已通过，尚未提交或推送
