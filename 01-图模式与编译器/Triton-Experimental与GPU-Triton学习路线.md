# Triton Experimental 与 GPU Triton 图模式学习路线

## 目标

以同一个 PyTorch 模型和同一个小算子为入口，对照学习：

```text
torch.compile
  -> TorchDynamo FX Graph
  -> AOTAutograd
  -> Inductor IR
  -> Scheduler / Fusion / Algorithm Selection
  -> Triton Codegen
  -> Triton Compiler
  -> Device Runtime
```

重点回答三个问题：

1. GPU 的 Triton 后端如何从 FX 图生成 Triton kernel。
2. Ascend `triton_experimental` 在哪些层复用上游 Inductor，在哪些层做 NPU 特化。
3. 线性化、索引重写、tile/UB 约束、autotune 和 ACLNN fallback 如何影响正确性与性能。

## 一、源码入口

### 1. PyTorch Inductor 与 GPU Triton

- [PyTorch Inductor 总目录](https://github.com/pytorch/pytorch/tree/main/torch/_inductor)
- [Inductor 编译入口 compile_fx.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/compile_fx.py)
- [图和 GraphLowering](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/graph.py)
- [Inductor IR](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/ir.py)
- [Scheduler 与融合](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/scheduler.py)
- [算法选择与 autotune](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/select_algorithm.py)
- [GPU Triton Codegen](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/triton.py)
- [Triton Template](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/triton_template.py)
- [Inductor wrapper codegen](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/wrapper.py)
- [Inductor benchmark/runtime](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/runtime/benchmarking.py)
- [PyTorch Triton 源码](https://github.com/triton-lang/triton/tree/main/python/triton)

### 2. Ascend `triton_experimental`

- [Ascend PyTorch 仓库](https://github.com/Ascend/pytorch)
- [`torch_npu/_inductor` 总入口](https://github.com/Ascend/pytorch/blob/master/torch_npu/_inductor/__init__.py)
- [`triton_experimental` 总目录](https://github.com/Ascend/pytorch/tree/master/torch_npu/_inductor/triton_experimental)
- [`triton_experimental/__init__.py`](https://github.com/Ascend/pytorch/blob/master/torch_npu/_inductor/triton_experimental/__init__.py)
- [`triton_experimental/config.py`](https://github.com/Ascend/pytorch/blob/master/torch_npu/_inductor/triton_experimental/config.py)
- [`triton_experimental/device.py`](https://github.com/Ascend/pytorch/blob/master/torch_npu/_inductor/triton_experimental/device.py)
- [`triton_experimental/lowering.py`](https://github.com/Ascend/pytorch/blob/master/torch_npu/_inductor/triton_experimental/lowering.py)
- [`triton_experimental/ir.py`](https://github.com/Ascend/pytorch/blob/master/torch_npu/_inductor/triton_experimental/ir.py)
- [`triton_experimental/fx_passes.py`](https://github.com/Ascend/pytorch/blob/master/torch_npu/_inductor/triton_experimental/fx_passes.py)
- [`triton_experimental/codegen`](https://github.com/Ascend/pytorch/tree/master/torch_npu/_inductor/triton_experimental/codegen)
- [NPU 后端选择说明](https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/inductor/TORCHINDUCTOR_NPU_BACKEND.md)

### 3. 参考资料

- [NPU Inductor Linearize 技术报告](https://gitcode.com/AllenGuanC/inductor-meta-worktree/blob/master/docs/docs/linearize_report/NPU_Inductor_Linearize_%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A.md)
- [Ascend PyTorch Inductor 相关 PR](https://github.com/Ascend/pytorch/pulls?q=is%3Apr+inductor+triton_experimental)
- [PyTorch 编译器文档](https://pytorch.org/docs/stable/torch.compiler.html)
- [PyTorch Inductor Triton 教程](https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel.html)

## 二、先建立整体心智模型

### 第 0 阶段：基础概念

先掌握这些概念，不急着读大文件：

- eager、graph capture、graph break；
- FX Graph、FakeTensor、Meta 推导；
- AOTAutograd 和 forward/backward graph；
- Inductor IR、Buffer、Layout、ComputedBuffer；
- fusion、tiling、reduction、stride、contiguous；
- Triton program、grid、block、warp、num_warps；
- dynamic shape、symbolic shape、guard；
- autotune、kernel cache、runtime launch。

建议先回答：

> 一个 `x + y` 从 Python 调用到 NPU kernel 启动，中间经过哪些对象？

### 第 1 阶段：学习 GPU Triton 基线

阅读顺序：

1. `torch/_inductor/compile_fx.py`
2. `torch/_inductor/graph.py`
3. `torch/_inductor/ir.py`
4. `torch/_inductor/scheduler.py`
5. `torch/_inductor/select_algorithm.py`
6. `torch/_inductor/codegen/triton.py`
7. `torch/_inductor/codegen/triton_template.py`
8. `torch/_inductor/runtime/benchmarking.py`

每个文件只关注三个问题：

- 输入是什么，输出是什么；
- 它修改了图、IR、kernel 还是 runtime；
- 它什么时候选择 fallback。

### 第 2 阶段：对照 Ascend 后端入口

先看：

```text
torch_npu/_inductor/__init__.py
  -> _load_triton_backend
  -> _load_triton_experimental_backend
  -> triton_experimental._activate
```

重点比较：

| 问题 | GPU Triton | Ascend `triton_experimental` |
|---|---|---|
| 后端如何注册 | PyTorch 原生 Inductor | TorchNPU 延迟注册 |
| device interface | CUDA/XPU 等 | NPU device interface |
| lowering | 上游 lowering | NPU 专用 lowering/override |
| IR | 上游 Inductor IR | 复用并补充 NPU 行为 |
| Codegen | GPU Triton wrapper | NPU Triton template/wrapper |
| autotune | Triton/CUDA benchmark | NPU event/MSPTI 和增强配置 |
| fallback | CUDA/CPU 等 | ACLNN、eager 或上游路径 |
| 主要优化目标 | GPU kernel 性能 | NPU UB、访存、核数、tile 和编译稳定性 |

## 三、重点学习 `triton_experimental` 的几个核心主题

### 主题 1：线性化与索引树

阅读 `config.py` 中这些开关：

```text
codegen_linearize
collapse_xtree
expand_divmod
fold_dual_decomp
fold_flat_rnode
split_strided_pointwise
scalar_odometer_axes
split_reduction_modulo
```

学习目标：

- 多维逻辑索引如何转成线性地址；
- view/permute/reshape 如何影响 stride；
- 为什么复杂索引会导致 scalar gather；
- 为什么线性化可能降低访存开销，也可能增加寄存器或 UB 压力。

### 主题 2：布局、tile 与 UB

重点阅读：

```text
tile_align
pointwise_tile_max
pad_min_block_to_8
reduce_xblock_by_input_stride
realize_strided_slice_input
realize_tail_bcast
```

结合我们遇到的错误理解：

- UB overflow；
- stride/layout assertion；
- tile 太大；
- reduction 的 block 不满足硬件约束；
- 非连续输入是否需要 materialize。

### 主题 3：算子 lowering 与 fallback

重点看：

- `lowering.py`；
- `decomposition.py`；
- `safe_softmax_aclnn_fallback`；
- `log_softmax_aclnn_fallback`；
- `mm/addmm/bmm/grouped_mm` 注册；
- `flex_attention` 注册。

要理解的原则：

```text
适合 Triton 且收益明确 -> Triton fusion
NPU 已有高质量 ACLNN 实现 -> ACLNN fallback
无法证明安全或收益不明确 -> 保留上游路径
```

### 主题 4：autotune 与性能可信度

重点看：

- candidate config 如何生成；
- 哪些 config 会被预筛选；
- NPU event/MSPTI 如何计时；
- autotune cache 如何读写；
- deterministic accuracy 为什么会限制 benchmark；
- performance 流程和 accuracy 流程为什么要分开。

不要把“compile 成功”直接等同于“性能可信”。需要同时记录：

- backend；
- dynamic on/off；
- ACLGraph on/off；
- dtype、batch size、train/inference；
- autotune/cache 状态；
- E2E 时间和 OP 时间；
- fallback 算子比例。

### 主题 5：动态形状与 graph break

学习：

- symbolic shape 如何进入 Inductor；
- guard 如何生成；
- 哪些图会因为数据依赖或动态索引产生 graph break；
- `triton_experimental` 如何在动态 shape 下选择 tile；
- fallback 是否破坏整个图的性能收益。

## 四、建议的最小实验序列

每次只研究一个问题，先用很小的张量。

### 实验 A：elementwise

```python
def f(x, y):
    return (x + y).relu()
```

观察：

- FX graph；
- 是否融合成一个 kernel；
- Triton kernel 的 grid/block；
- GPU 与 NPU 生成代码的差异。

### 实验 B：reduction

```python
def f(x):
    return x.sum(dim=-1)
```

观察：

- reduction axis；
- block size；
- mask；
- NPU 的 r-split、real block 和 padding。

### 实验 C：非连续布局

```python
def f(x):
    return x.transpose(-1, -2).contiguous()
```

观察：

- stride；
- layout 判断；
- 是否 materialize；
- 是否产生额外 kernel。

### 实验 D：matmul

```python
def f(x, y):
    return x @ y
```

观察：

- 是否使用 Triton template、ACLNN 或专用矩阵路径；
- candidate config；
- autotune；
- 不同 tile 配置对性能的影响。

### 实验 E：动态 shape

```python
def f(x):
    return x.reshape(x.shape[0], -1).sum(dim=-1)
```

使用多个 batch/sequence 长度执行，观察：

- 是否重新编译；
- guard 是否变化；
- 是否产生 graph break；
- kernel 是否保持 shape 泛化。

## 五、调试产物和观察顺序

建议每个实验保留以下信息：

```text
1. 输入 shape / dtype / device
2. torch、torch_npu、triton-ascend、CANN 版本
3. backend 选择
4. FX graph
5. Inductor IR
6. generated output_code.py
7. kernel 编译日志
8. autotune 选择结果
9. eager/compiled 输出对比
10. E2E 与 OP 时间
```

调试环境变量只在单模型实验中使用：

```bash
TORCH_COMPILE_DEBUG=1
TORCH_LOGS="+inductor,graph_code,output_code"
```

NPU 实验示例：

```bash
TORCHINDUCTOR_NPU_BACKEND=triton_experimental \
TORCH_COMPILE_DEBUG=1 \
TORCH_LOGS="+inductor,graph_code,output_code" \
python minimal_case.py
```

GPU 对照实验只去掉 NPU backend 变量：

```bash
TORCH_COMPILE_DEBUG=1 \
TORCH_LOGS="+inductor,graph_code,output_code" \
python minimal_case.py
```

不要直接在全量流水中开启这些变量，容易产生大量 debug/profile 文件。

## 六、推荐的 ChatGPT 学习方式

每次在新的 ChatGPT 对话中先粘贴以下背景：

```text
我正在对照学习 PyTorch GPU Triton 和 Ascend torch_npu 的
triton_experimental 图模式实现。

请遵循这些规则：
1. 优先引用上游 PyTorch 和 Ascend/pytorch 源码，不把推测说成事实。
2. 每次只讲一个小主题。
3. 给出具体文件、类、函数和调用链。
4. 明确区分 GPU 原生逻辑、torch_npu 适配逻辑和 triton_experimental 特有逻辑。
5. 先解释机制，再给最小可运行实验。
6. 说明正确性风险、性能影响和 fallback 行为。
7. 如果当前版本与旧版本不同，指出版本差异。
```

推荐的单轮提问格式：

```text
今天只学习 <主题>。

请按以下结构回答：
1. 这个机制解决什么问题
2. GPU Triton 的实现位置
3. triton_experimental 的对应实现位置
4. 一条完整调用链
5. 一个最小实验
6. 我应该观察哪些日志或生成文件
7. 哪些地方是推测，哪些地方有源码依据
```

## 七、最终学习成果

完成后应能够独立回答：

- 一个 FX 节点如何变成 Inductor IR；
- Inductor 如何决定融合或 graph break；
- 一个 reduction 如何被 tile 和线性化；
- Triton template 如何生成 kernel；
- `triton_experimental` 为什么需要自己的 device/codegen/heuristics；
- NPU UB 和 stride 约束如何影响 kernel；
- autotune 为什么会和 deterministic accuracy 冲突；
- 何时应该走 Triton，何时应该走 ACLNN；
- 如何判断一次性能提升来自真正的 kernel 优化，而不是 fallback、缓存或测试口径变化。

## 八、与我们流水对齐的原则

学习和改动时始终遵循：

1. PyTorch 社区 TorchBench 的 accuracy 语义优先。
2. PyTorch Inductor 的通用行为优先复用，不重复实现。
3. NPU 只增加必要的 device、lowering、codegen、runtime 适配。
4. 优化必须带 shape/layout guard，并保留安全 fallback。
5. accuracy、performance、profile 三种流程分开记录。
6. 看板中同时记录 backend、dynamic、ACLGraph、dtype、模式和版本信息。
7. 不通过修改模型源码或全局关闭 deterministic 来制造“通过”。
