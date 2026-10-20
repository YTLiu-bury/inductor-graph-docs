# `npu_support.py` 后端 Patch 说明

## 1. 文档范围

本文说明 `Ascend/pytorch` 仓库中 `benchmarks/torchbench/npu_support.py` 的模型兼容和编译调优 patch，重点回答：

- 哪些 patch 面向 DVM/MLIR（`ascend_npu_ir`）后端；
- 哪些 patch 在 Triton Experimental 后端仍然生效；
- 每个 patch 解决什么问题；
- patch 在什么条件下会被调用，以及使用时需要注意什么。

本文对应的代码路径为：

```text
benchmarks/torchbench/npu_support.py
```

## 2. Patch 的注册和调用方式

文件使用 `register_patch()` 将模型名映射到 patch 函数：

```python
@register_patch("model_name")
def patch_model():
    ...
```

实际执行入口是：

```python
patch_model(model_name)
```

在 TorchBench runner 中，当前实现通常在指定单模型时调用：

```python
if args.only and is_npu_available:
    patch_model(args.only)
```

因此，这些 patch 主要通过以下形式触发：

```bash
python torchbench.py --only <model> ...
```

直接不带 `--only` 跑完整模型列表时，不应假设所有模型 patch 都会自动逐模型执行。日常流水如果是“一模型一进程”，则每个模型的 patch 会在对应进程中生效。

## 3. 后端判定规则

文件通过环境变量判断是否使用 Triton Experimental：

```python
os.environ.get("TORCHINDUCTOR_NPU_BACKEND") == "triton_experimental"
```

因此当前代码的实际语义是：

| 条件 | 编译调优 patch 行为 |
|---|---|
| `TORCHINDUCTOR_NPU_BACKEND=triton_experimental` | 跳过 `ascend_npu_ir` 相关调优 |
| 环境变量未设置 | 进入非 Triton Experimental 分支 |
| 环境变量为 `dvm` 或 `mlir` | 进入非 Triton Experimental 分支 |
| 其他值，包括某些 `aot_eager` 场景 | 也可能进入非 Triton Experimental 分支 |

这里的判断不是严格的“只判断 DVM/MLIR”，而是“只要不是 `triton_experimental` 就执行对应分支”。运行 Triton Experimental 时建议显式设置：

```bash
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
```

## 4. 后端专属编译调优 Patch

### 4.1 `patch_remove_ops_from_generate_list`

代码位置：

```text
npu_support.py:562
```

作用：从 `torch_npu._inductor.ascend_npu_ir` 的 `GENERATE_LIST` 中移除指定算子，使这些算子不再走对应的生成或融合路径。

Triton Experimental 会跳过该 patch，因为 Triton Experimental 不读取 `ascend_npu_ir.GENERATE_LIST`。

涉及模型和算子：

| 模型 | 移除的算子 | 目的 |
|---|---|---|
| `speech_transformer` | `aten.cat`、`aten.full` | 避免相关算子进入不适配的生成路径 |
| `pytorch_unet` | `aten.constant_pad_nd` | 避免当前融合路径导致性能问题 |
| `squeezenet1_1` | `aten.div` | 配合 fallback 处理除法路径 |

### 4.2 `patch_remove_decomposition`

代码位置：

```text
npu_support.py:598
```

作用：从 Inductor 的 decomposition 表中移除指定 decomposition，避免某些算子被拆解成不适合 NPU 后端的形式。

该函数在 Triton Experimental 下会跳过。原因是它会修改全局 Inductor decomposition 表，而 Triton Experimental 依赖标准 decomposition 表，误修改可能影响 Triton 编译。

涉及模型和算子：

| 模型 | 算子 | 目的 |
|---|---|---|
| `resnet50` | `aten._native_batch_norm_legit_no_training`、`aten.threshold_backward`、`aten.native_batch_norm_backward` | 调整 BatchNorm 和 threshold backward 的 decomposition |
| `resnet152` | 同上 | 同上 |
| `resnext50_32x4d` | 同上 | 同上 |
| `densenet121` | 同上 | 同上 |
| `hf_T5_base` | `aten._softmax` | 避免当前 softmax decomposition 造成性能问题 |
| `hf_T5_large` | `aten._softmax` | 同上 |
| `timm_resnest` | `aten.threshold_backward`、`aten.native_batch_norm_backward` | 调整训练反向路径 |

### 4.3 `patch_force_fallback_kernels`

代码位置：

```text
npu_support.py:627
```

作用：将指定的 fused kernel 加入 `ascend_npu_ir` 的 `force_fallback_kernel_names`，强制其绕过当前有问题的融合实现。

该函数只对 DVM/MLIR 的 `ascend_npu_ir` 路径有意义，Triton Experimental 会跳过。

涉及模型和 kernel：

| 模型 | 强制 fallback 的 kernel | 目的 |
|---|---|---|
| `hf_T5_base` | `mlir_fused_add_lt_neg_where_16` | 缓解当前 fused kernel 的性能或正确性问题 |
| `hf_T5_large` | `mlir_fused_add_lt_neg_where_16` | 同上 |
| `squeezenet1_1` | `mlir_fused_relu_threshold_backward_2` | 避免反向融合路径问题 |
| `T5ForConditionalGeneration` | `mlir_fused_add_lt_neg_where_13` | 避免特定 T5 融合路径问题 |

### 4.4 `patch_exclude_decomps_npu`

代码位置：

```text
npu_support.py:654
```

作用：同时更新 NPU decomposition 排除列表和 Inductor decomposition 表，使指定算子不走当前 NPU decomposition 路径。

该函数在 Triton Experimental 下跳过。

涉及模型和算子：

| 模型 | 排除的算子 |
|---|---|
| `BartForCausalLM` | `aten.native_layer_norm`、`aten.native_layer_norm_backward` |

## 5. DVM/MLIR 专属 Patch 汇总

下面这些模型 patch 包含后端编译调优，只对非 Triton Experimental 分支生效：

```text
resnet50
resnet152
resnext50_32x4d
densenet121
speech_transformer
hf_T5_base
hf_T5_large
pytorch_unet
squeezenet1_1
T5ForConditionalGeneration
BartForCausalLM
timm_resnest
```

其中 `hf_T5_base` 同时包含两类逻辑：

1. 重写 T5 Attention forward，这是模型兼容 patch，Triton Experimental 也可能执行；
2. 移除 decomposition 和强制 kernel fallback，这是 DVM/MLIR 专属 patch，Triton Experimental 会跳过。

`T5ForConditionalGeneration` 也类似：

1. `force_fallback_kernel_names` 属于 DVM/MLIR 专属；
2. 注册 `native_dropout` 的 effect 属于 PyTorch/Inductor 图编译行为，不应简单归类为 Triton 专属或 DVM 专属。

## 6. 后端无关的模型兼容 Patch

以下 patch 不直接修改 DVM/MLIR 或 Triton 的 kernel 配置，主要解决模型本身的设备、数据、权重加载或算子调用问题，因此在多个后端下都可能生效。

| 模型 | Patch 内容 | 主要解决的问题 |
|---|---|---|
| `LearningToPaint` | 根据 CUDA/NPU 可用性设置 `USE_CUDA` | 避免模型在 NPU 上错误走 CPU 分支 |
| `hf_T5`、`hf_T5_base` | 重写 `T5Attention.forward` | 避免连续 `_to_copy` 在 eager 和图模式下产生精度差异 |
| `soft_actor_critic` | 对分布参数增加 `.contiguous()`，调整 batch size | 修复非连续内存导致的精度或运行问题 |
| `nvidia_deeprecommender` | 重写训练/推理初始化，迁移模型和输入到 NPU | 原始模型只接受 CPU/CUDA 初始化 |
| `torch_multimodal_clip` | 修正图片数据查找、NPU tensor、文本编码和 optimizer | 适配 NPU 设备和共享数据目录 |
| `DistilBertForMaskedLM` | 重写 `MultiHeadSelfAttention.forward` | 修正 mask 和 NPU device 处理 |
| `fastNLP_Bert` | 使用配置文件构造模型，避免读取空权重文件 | 解决 `EOFError` 或空 checkpoint 问题 |
| `drq` | 从 `TORCHBENCH_DATA_PATH`、TorchBench 数据目录或模型目录查找 `obs.pkl` | 取消对额外 prepare 阶段的强依赖 |
| `yolov3` | 重定向 `coco128` 数据路径，重写 device 选择 | 适配 NPU 数据路径和设备选择 |
| `fambench_dlrm` | 在模型初始化期间临时映射必要的 CUDA API 到 NPU API | 处理模型内部写死 CUDA 的情况，同时避免全进程污染 |
| `vision_maskrcnn` | NPU tensor 使用 `torch.ops.npu.npu_nms_with_mask` 和 `npu_roi_align` | 规避 torchvision NMS/ROI Align 的 NPU fallback 或缺失 kernel |
| `fambench_xlmr` | 模型初始化期间临时将 `torch.load` 设置为 `weights_only=False` | 兼容 fairseq checkpoint 中的非 tensor 对象 |

## 7. `vision_maskrcnn` 的特殊说明

`vision_maskrcnn` 不是 DVM/MLIR 专属 patch。它修改的是 torchvision 函数引用：

```text
torchvision.ops.boxes.nms
torchvision.ops.roi_align
torchvision.ops.poolers.roi_align
```

当输入是 NPU tensor 时，重定向到：

```text
torch.ops.npu.npu_nms_with_mask
torch.ops.npu.npu_roi_align
```

当输入仍然是 CPU tensor 时，继续使用原 torchvision 实现。因此它更适合归类为“模型算子适配”，而不是后端编译调优。

## 8. `fambench_dlrm` 的特殊说明

`fambench_dlrm` 没有直接启用全局 `transfer_to_npu`，而是在模型构造期间：

1. 保存需要替换的 `torch.cuda` 属性；
2. 临时映射到 `torch_npu.npu`；
3. 构造模型；
4. 在 `finally` 中恢复原始属性。

这样可以避免全局替换带来的副作用，尤其是影响同一进程内其他模型的 Triton 编译路径。

## 9. 数据路径和 ACLNN 辅助逻辑

文件中还有两类不属于具体模型后端 patch 的辅助逻辑：

### 9.1 `TORCHBENCH_DATA_PATH`

`_patch_environment_variables()` 支持通过环境变量覆盖 TorchBench 数据目录：

```bash
export TORCHBENCH_DATA_PATH=/path/to/torchbench/data/.data
```

它主要影响数据查找，不区分 DVM、MLIR 或 Triton Experimental。

### 9.2 `use_aclnn()`

该函数设置：

```python
os.environ["USE_ACLOP"] = "0"
```

它表示优先使用 ACLNN，而不是 ACLOP。它不是 Triton/DVM kernel patch，是否生效取决于调用方是否显式调用该函数或设置对应环境变量。

## 10. 运行时检查建议

### 10.1 验证当前后端

```bash
python3 - <<'PY'
import os
print("TORCHINDUCTOR_NPU_BACKEND:", os.environ.get("TORCHINDUCTOR_NPU_BACKEND"))
PY
```

### 10.2 显式运行 Triton Experimental

```bash
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
python3 torchbench.py \
    --devices npu \
    --backend inductor \
    --npu-backend triton_experimental \
    --only <model> \
    --accuracy \
    --inference
```

看到以下日志时，说明 DVM/MLIR 编译调优 patch 被跳过：

```text
[patch] triton_experimental backend: skip GENERATE_LIST tuning (ascend_npu_ir only).
[patch] triton_experimental backend: keep stock inductor decompositions (skip removal).
[patch] triton_experimental backend: skip force_fallback_kernel_names (ascend_npu_ir only).
```

### 10.3 显式运行 DVM/MLIR

```bash
export TORCHINDUCTOR_NPU_BACKEND=dvm
python3 torchbench.py \
    --devices npu \
    --backend inductor \
    --only <model> \
    --accuracy \
    --inference
```

## 11. 维护原则

1. 涉及 `GENERATE_LIST`、decomposition、fallback kernel 的 patch，应明确标注为 DVM/MLIR 专属。
2. Triton Experimental 不应复用 DVM 的全局 decomposition 修改。
3. 模型设备、数据路径、权重加载和 NPU 算子替换类 patch，可以独立于编译后端维护。
4. 使用临时 monkey patch 时，应尽量限制作用域，并在 `finally` 中恢复全局状态。
5. 新增 patch 时应同时记录：模型、触发条件、适用后端、解决的问题、是否影响性能以及预计删除条件。
6. 修改 patch 后，至少分别验证一次 DVM/MLIR 和 Triton Experimental，避免只在单一后端上验证。

