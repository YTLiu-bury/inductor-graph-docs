# 图模式基础（一）：从 Eager 到一个融合 Kernel

> 学习路线：`Triton-Experimental与GPU-Triton学习路线.md` 的第 0 阶段、实验 A（elementwise）。
>
> 本文目标：不用先读大文件，先精确回答“`(x + y).relu()` 为什么有机会变成一次设备 kernel 启动”。

## 1. 先给结论

`torch.compile` 的“图模式”不是把 Python 函数简单地翻译成一段固定代码。它先捕获一段满足特定输入条件的张量计算，把其中的算子及其数据依赖表示为图，再为这段图选择、生成并缓存一个可执行实现。

对纯逐元素函数

```python
def f(x, y):
    return (x + y).relu()
```

`add` 的结果只被紧接着的 `relu` 使用时，编译器通常可以把两者融合：不把中间张量 `s = x + y` 写回全局显存/内存，而由同一个 kernel 的每个 lane 直接计算 `relu(x + y)`。这里的“通常”很重要：实际是否融合仍取决于设备、dtype、布局、编译器版本、图边界和后端支持。

## 2. Eager、捕图与图中断

| 概念 | 直觉 | 对本例的含义 |
| --- | --- | --- |
| Eager（即时执行） | Python 每遇到一个张量算子就立刻调度它 | 先执行 `add`，再执行 `relu`；实现上可能对应两个独立 kernel |
| 图捕获 | 记录一段张量计算的算子、输入输出和依赖 | 得到 `x → add → relu → output` 的候选编译单元 |
| guard（守卫） | 复用已编译结果前要成立的条件 | 例如输入的 dtype、rank、某些 shape/stride 或 Python 分支条件 |
| graph break（图中断） | 无法安全纳入当前捕获图的边界 | 中断处回到 Python/eager 或由另一段图处理；不能假设整函数只有一个 kernel |

图捕获保留的是**计算语义和依赖**，不是某次输入张量中的具体数值。因而同一个已编译版本可以在 guard 仍成立时处理新数值；shape 是否能复用，取决于静态/动态 shape 策略及 guard。

## 3. `torch.compile` 的心智调用链

```text
Python 函数 f
  → TorchDynamo：捕获可编译区段，建立 FX Graph，并生成 guards
  → AOTAutograd：需要求导时拆解/编译 forward 与 backward（推理或无 grad 时不一定出现）
  → Inductor GraphLowering：把 FX 运算降低为 Inductor IR
  → Scheduler：决定依赖顺序、融合和实现候选
  → 设备后端 codegen：生成 Triton / 其他后端的代码与 wrapper
  → 编译、缓存、runtime launch：在 guard 命中时启动已生成的实现
```

这是理解用的主干，不是“每次调用必经的无条件流水线”。例如 Python 副作用、数据依赖控制流或不支持的算子都可能引入 graph break；训练才需要 AOTAutograd 的反向图；矩阵乘、归约和 fallback 的代码路径也与本例不同。

## 4. 用公式验证语义

令 `X, Y ∈ R^(2×3)`，并定义

```text
S[i, j] = X[i, j] + Y[i, j]
Z[i, j] = relu(S[i, j]) = max(S[i, j], 0)
```

取

```text
X = [[ 1, -2,  3],
     [ 4,  5, -6]]

Y = [[10, 20, -30],
     [40,-50,  60]]
```

逐元素计算得到

```text
Z = [[11, 18,  0],
     [44,  0, 54]]
```

这就是 eager 的两步实现和融合 kernel 都必须满足的可观察语义。融合改变的是中间结果 `S` 的存储与调度方式，**不允许**改变 `Z` 的元素值、shape、dtype 规则或异常语义。

### 4.1 一维索引验证：二维元素怎样分给 kernel

假设 `X`、`Y` 是连续的 row-major `2×3` 张量。把二维索引压平为

```text
k = i × 3 + j,       0 ≤ k < 6
i = floor(k / 3),    j = k mod 3
```

因此六个逻辑元素的线性索引是 `0, 1, 2, 3, 4, 5`。用教学用块大小 `BLOCK = 4` 时：

```text
程序数 = ceil(6 / 4) = 2
pid = 0：k = [0, 1, 2, 3]，全部有效
pid = 1：k = [4, 5, 6, 7]，仅 k < 6 的 [4, 5] 有效
```

对应的 Triton 风格伪代码如下。`mask` 是尾块正确性的关键；没有它，第二个 program 会访问逻辑张量范围外的位置。

```python
pid = tl.program_id(0)
offsets = pid * BLOCK + tl.arange(0, BLOCK)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
tl.store(z_ptr + offsets, tl.maximum(x + y, 0.0), mask=mask)
```

这里每个 `pid` 表示一个 Triton program；它不是 Python 循环的一次迭代。真实 block、grid、warp 数和向量化宽度要由后端与硬件决定，本节的 `BLOCK=4` 只用于把索引和 mask 算清楚。

## 5. GPU Triton 与 NPU `triton_experimental`：先分清共性和特化

两条路径的共同目标都是：保持 FX 图描述的语义，并为设备生成可执行代码。GPU Triton 的基线主要使用上游 Inductor 的设备后端实现；NPU `triton_experimental` 则在保留 Inductor 主流程的同时，安装 NPU 的调度、wrapper、device interface、lowering/fallback 与 autotune 适配。

本地源码（工作树提交 `0acc26dde0`）可确认的 NPU 激活链是：

```text
TORCHINDUCTOR_NPU_BACKEND=triton_experimental
  → torch_npu._inductor._load_backend()
  → restore_inductor_baseline()
  → _load_triton_experimental_backend()
  → _register_triton_experimental_decompositions()
  → triton_experimental._activate()
  → 注册 NPU scheduling/wrapper、device override/interface、fallback 与 MSPTI autotune
```

本节在 Ascend 910B2 上实际验证了这条路径：设置 `TORCHINDUCTOR_NPU_BACKEND=triton_experimental` 后，生成的 `output_code.py` 导入 `torch_npu._inductor.triton_experimental.npu_triton_heuristics`，并将 kernel 的 device 标为 `npu`。这把“选了哪个后端”落实到了生成代码证据。

## 6. 本节验证记录

### 已验证

- 使用独立 Python 标量实现执行了第 4 节给定矩阵：结果为 `[[11, 18, 0], [44, 0, 54]]`。
- 用 `N = 6`、`BLOCK = 4` 验算了 grid 数为 2，尾块只有索引 4、5 通过 `mask`。
- 阅读本地 `torch_npu/_inductor/__init__.py` 与 `torch_npu/_inductor/triton_experimental/__init__.py`，确认了上述 `triton_experimental` loader/activation 的注册顺序。
- 2026-10-22 在 Ascend 910B2 容器环境运行了 `torch.compile(..., backend="inductor", fullgraph=True)`，并明确设置 `TORCHINDUCTOR_NPU_BACKEND=triton_experimental`。环境为 `torch 2.13.0+cpu`、`torch_npu 2.13.0+gite94cbac`、CANN 9.2.0；NPU 可用且设备数为 8。
- `2×3` 矩阵的 eager 与 compiled 结果都为 `[[11, 18, 0], [44, 0, 54]]`，与公式手算一致。
- 对一维长度 `1、5、7、17` 的输入重复编译并比较 eager/compiled，全部通过。这些长度包含小于 tile 的输入和非整齐尾块，可以检查生成 mask 是否保护最后一组有效元素。
- FX Graph 中能看到 `aten.add → aten.relu` 两个节点；`ir_post_fusion.txt` 中是一个包含 `add` 和 `relu` 的 scheduler node；生成的 `output_code.py` 只有 `triton_unk_fused_add_relu_0` 一个 kernel，并在 wrapper 中调用一次 `.run(...)`。因此这次实验确实验证了融合，不只是输出相等。
- 生成代码使用 `npu_triton_heuristics.pointwise`，设备元数据为 `Ascend910B2`，并在 kernel 中以 `x0index < x0numel` 生成尾块 mask。

### 本次仍未验证

- GPU Triton 的对应生成代码和 NPU/GPU 性能差异；本课只验证 NPU 正确性与融合结构，没有做性能基准。
- 不同 dtype、非连续 stride、动态 shape、训练反向图，以及更大张量下的 tile 与实际 grid 配置。

编译日志出现一次 CANN/NPU allocator 关于 32-byte padding 的 warning，但本次数值检查通过；本课没有据此推断性能或内存影响。调试产物路径中的机器用户名和绝对目录不写入知识库，归档时统一记为 `<CONTAINER_ROOT>` 与 `<TORCHINDUCTOR_CACHE_DIR>`。

## 7. 最小实验代码与运行要点

本次实际执行的核心代码如下。用 `fullgraph=True` 让无法纳入完整图的操作直接报错，这样成功返回可证明本函数没有静默切成多个 Dynamo 图段：

```python
import torch

def f(x, y):
    return (x + y).relu()

device = "npu"
x = torch.tensor([[1., -2., 3.], [4., 5., -6.]], device=device)
y = torch.tensor([[10., 20., -30.], [40., -50., 60.]], device=device)

compiled_f = torch.compile(f, backend="inductor", fullgraph=True)
eager = f(x, y)
compiled = compiled_f(x, y)
torch.testing.assert_close(compiled, eager)
print(compiled.cpu())
```

本次在容器中 source CANN 9.2.0 环境脚本，并给该 Python 进程设置：

```bash
TORCHINDUCTOR_NPU_BACKEND=triton_experimental
TORCH_COMPILE_DEBUG=1
TORCHINDUCTOR_CACHE_DIR=<TORCHINDUCTOR_CACHE_DIR>
```

复核编译融合时依次看：FX graph 是否有 `add → relu`；`ir_post_fusion.txt` 是否只剩一个调度节点；`output_code.py` 是否只生成一个 fused kernel 并只 launch 一次。数值相等是正确性证据，生成代码与 IR 才是融合证据。

## 8. 自检题

1. 为什么 `pid=1` 需要 `mask`？
2. 融合省掉的是哪个中间张量的哪类开销？
3. 为什么“数学结果正确”还不能证明 NPU backend 编译正确？

答案要点：尾块越界；`S = X + Y` 的全局读写/额外调度；还需验证捕图、lowering、代码生成、设备编译与实际输出。

## 参考

- [PyTorch `torch.compile` 文档](https://pytorch.org/docs/stable/torch.compiler.html)
- [PyTorch Inductor 概览](https://docs.pytorch.org/docs/stable/torch.compiler_inductor_overview.html)
- [Ascend/pytorch：`triton_experimental`](https://github.com/Ascend/pytorch/tree/master/torch_npu/_inductor/triton_experimental)
- [学习路线](Triton-Experimental与GPU-Triton学习路线.md)
