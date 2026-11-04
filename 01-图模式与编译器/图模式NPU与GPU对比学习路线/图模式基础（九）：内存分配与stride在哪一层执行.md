# 图模式基础（九）：内存分配与 stride 在哪一层执行

这一课把之前的 shape、stride、view 和物化连起来，并继续追问：Inductor 最终生成 Triton kernel、Triton-Ascend 编译成 NPU 可执行代码时，究竟是谁分配内存？stride 又在哪个阶段“变化”并影响实际访问？

先给结论：这里不是一个统一的“内存阶段”。至少要分开看 **张量存储分配**、**图里的临时 buffer 规划**、**kernel 的局部资源规划**、**按 stride 算地址**四件事。stride 多数时候不是硬件执行时被改写，而是编译期作为 layout 元数据传到代码生成，kernel 执行时据此算出访问地址。

## 1. 先把问题拆成四问

读完本课，应能回答下面四个问题：

1. `transpose/permute/expand` 何时只改 shape/stride 元数据，何时真的复制并申请一块新存储？
2. Inductor/TE 生成的 wrapper 何时、按什么 shape 和 stride 申请输出或临时 buffer？实际设备存储由谁提供？
3. Triton kernel 里的 `ptr + offset` 如何把逻辑坐标和 stride 变成实际地址？
4. Triton-Ascend 在 TTIR 到目标后端代码的过程中，会负责什么内存规划；又不会替 Inductor 决定什么？

## 2. 一个转置视图：先手算地址，不急着谈硬件

假设底层连续内存按行存储，形状为 `(3, 2)`：

```text
原张量 X = [[0, 1],
            [2, 3],
            [4, 5]]
shape = (3, 2), stride = (2, 1)
底层线性存储 = [0, 1, 2, 3, 4, 5]
```

转置 `T = X.T` 后，逻辑矩阵是：

```text
T = [[0, 2, 4],
     [1, 3, 5]]
shape = (2, 3), stride = (1, 2)
```

在元素大小为 1 个单位、storage offset 为 0 时，逻辑坐标 `(i,j)` 对应的元素偏移为：

```text
offset(i,j) = i * stride[0] + j * stride[1]
            = i + 2*j
```

例如 `T[1,2]` 的 offset 是 `1 + 2*2 = 5`，所以读到底层第 5 个元素 `5`。转置 view 本身只换了 shape/stride 的解释方式，底层 `[0,1,2,3,4,5]` 没有重排。

若执行 `Tc = T.contiguous()`，结果逻辑值仍相同，但连续布局变为 `stride=(3,1)`，新存储按 `[0,2,4,1,3,5]` 排列。这个时候才有一次真实的读旧地址、写新地址的物化过程（若编译器能融合或消除该复制，则最终图可能不产生独立副本）。

## 3. “内存分配”其实有四种含义

| 层次 | 谁决定/生成 | 运行时发生什么 | 是否等于创建 PyTorch 输出 tensor |
| --- | --- | --- | --- |
| 逻辑 view 与 layout | PyTorch eager 元数据，或 Dynamo/Inductor IR 与 lowering | `transpose`/`expand` 可只变 shape、stride、offset 的解释 | 否 |
| 图级 buffer 与生存期规划 | Inductor Graph/Scheduler、后端 lowering/codegen | 确定中间值是否物化、buffer 大小、何时可复用；wrapper 产生对应分配代码 | 通常是图中临时或结果存储，但计划与实际申请分属编译期和运行期 |
| 张量设备存储 | 生成的 wrapper 调用 PyTorch/torch_npu 分配入口 | 运行时拿到 NPU 上的存储，并构造带 shape/stride 的 Tensor | 是 |
| kernel 内部资源 | Triton-Ascend / BiSheng 等 kernel 编译后端及运行环境 | 映射寄存器、片上 buffer、同步/流水资源等 | 否；这是 kernel 执行资源，不是 Python 层的 Tensor 对象 |

因此，“Triton-Ascend 分配内存”这句话要问清指哪种内存。设备 kernel 的局部资源分配确实属于后端编译的重要工作；但一般输入/输出 Tensor 的大块存储，已经由上层 wrapper 和 torch_npu runtime 分配好，再把指针传给 Triton kernel。

## 4. 从 Inductor IR 到 wrapper：什么时候真的申请输出空间？

常见的时序可以概括为：

```text
输入 Tensor 已有 storage 和 stride
        ↓
FX / Inductor IR 表达运算、view 和 layout
        ↓
lowering 决定保留 view，或 realize/materialize 出 buffer
        ↓
scheduler/codegen 决定融合、buffer 生存期与分配需求
        ↓
wrapper 代码写出 empty_strided(...)/NPU 分配调用
        ↓
图函数运行时执行 wrapper，runtime/allocator 提供设备存储
        ↓
把输入/输出指针传给 Triton kernel 并 launch
```

编译阶段通常只是**生成分配语句**，不是那时就把 NPU 物理存储申请出来。执行编译产物时，wrapper 才调用 NPU 分配入口。即使两个中间 tensor 的逻辑生命周期不重叠，Inductor 也可能让它们复用同一片 buffer；这属于图级 storage planning，不代表 Triton-Ascend 把两个 Tensor 合成了一个。

本课程当前 Ascend PyTorch checkout 的 TE `codegen/wrapper.py` 对 NPU buffer 发出 `torch_npu._C._empty_strided_npu(allocation_shape, stride, dtype)`，若 allocation shape 和逻辑 shape 不同，再追加 `as_strided(shape, stride)`。这是**当前代码版本的实现证据**，不是所有 PyTorch 版本或所有后端都必须采用同一个入口。

NPU lowering 还可能主动要求物化。例如当前 TE `lowering.py` 的 NPU permute 路径，会评估 permute 后内层 stride 是否非 1、内层长度是否大于 1，以及 gather 的访问代价；满足条件时创建 Pointwise copy 并 `realize()`，否则保留零拷贝 view，交给消费者 kernel 处理。也就是说，NPU 特化可能改变“是否值得拷贝、拷贝成什么 layout”的算法决策；它发生在进入 Triton-Ascend 最终机器码编译之前。

## 5. Triton kernel 怎样把 stride 变成地址？

延续转置例子，假设某个 kernel 按逻辑坐标遍历 `T`，输入指针访问可以抽象为：

```python
offset = i * stride_i + j * stride_j
value = tl.load(T_ptr + offset)
```

代入 `stride=(1,2)`：

```text
T_ptr + i + 2*j
```

若 `T.contiguous()` 的 `stride=(3,1)`，则变为：

```text
Tc_ptr + 3*i + j
```

关键是：kernel 通常拿到的是 base pointer 和某些已知/传入的 index、stride 参数；每个 program 根据 tile 坐标和 lane/range 算一批 offset，经 mask 控制边界，再 load/store。stride 影响地址算术和访存形态，本身不是“运行时把 Tensor 的 stride 字段改掉”。输出是否连续，取决于 IR/layout 决策和 wrapper 分配；写地址也会按输出 layout 计算。

Triton-Ascend 官方调试材料给出的 TTIR 中有 `tt.addptr`、`tt.load`、`tt.store`：在这一层能看见指针偏移与读写语义，但 TTIR 仍未映射到具体 NPU memory model 或执行单元。后续 Ascend backend 和 BiSheng 编译链才继续把它降到目标相关表示、指令与可执行文件。

## 6. Triton-Ascend 负责什么，不负责什么？

可以把责任边界记成下面这张图：

```text
Inductor / TE（图与 wrapper）
  决定：要算什么；view 保留还是 materialize；哪些 Tensor/buffer 需要分配；请求的 shape/stride/layout
  输出：Triton kernel、wrapper 分配/launch 代码、kernel 参数
                 │ base pointers + scalar/stride 参数
                 ▼
Triton 前端 → TTIR
  表达：program id、向量索引、ptr offset、load/store、mask
                 ▼
Triton-Ascend 适配 → Ascend-oriented IR → BiSheng codegen
  决定/实现：NPU 相关的 kernel lowering、硬件指令、kernel 内资源/调度安排
                 ▼
NPU executable + runtime launch
```

边界的简明说法：

- Inductor/TE 的 wrapper 负责把**张量存储分配要求**变成可执行 host 代码；底层 NPU allocation 由 torch_npu/运行时完成。
- Triton kernel 负责消费指针并计算访问偏移；它不是每次 `tl.load` 就申请一份张量存储。
- Triton-Ascend/BiSheng 把 kernel 的抽象读写和计算变成 Ascend 能执行的表示，并规划 kernel 执行资源。后端编译可能需要工作区或局部资源，但不能据此推断它决定了上层 Tensor 的逻辑 stride 或普通输出 storage。
- TE 可以在 IR/lowering/codegen/wrapper/runtime 接口处做 NPU 适配。某些适配会让访问变快、改变物化策略或引入 workspace；这些是上层后端策略，不能一概归给 Triton-Ascend。

官方 Triton-Ascend 架构文档将 Triton IR → Linalg IR 的转换和后续 BiSheng 编译分开描述；其中 BiSheng 后续阶段会执行硬件相关指令选择、内存分配和调度。因此，“机器指令/设备资源分配在后端”是正确的一半，但“Tensor storage 与 stride 都在 Triton-Ascend 执行时才决定”不正确。

## 7. 怎么从生成产物逐层验证？

不要只看最终 `.o`，按以下顺序对照：

1. **Eager 元数据**：记录 `shape`、`stride()`、`storage_offset()`、`is_contiguous()`。转置 view 预期 stride 为 `(1,2)`。
2. **FX / Inductor IR**：确认 permute 是 view/索引关系，还是已经被 realize 成 buffer；检查消费者是否融合。
3. **生成 wrapper**：搜索 `empty_strided_npu`、`as_strided`、workspace 参数和 kernel launch，区分真正分配的对象。
4. **生成 Triton kernel/TTIR**：检查 kernel 参数里有无 stride scalar；检查 `tt.addptr` 前的 offset 公式是否和手算一致。
5. **TTIR → Ascend adapter IR → 设备代码**：用 Triton-Ascend 的 debug dump 观察变换后 load/store 的表示及后端产物；不要把 TTIR 中一个 `tt.load` 直接等同于一条最终硬件指令。
6. **正确性验证**：先对 eager 与 compiled 的数值，再对 view/物化结果的 shape、stride（若 API 语义要求）和别名关系；最后才做性能对比。

官方 Triton-Ascend 调试指南介绍了用 `TRITON_DEBUG=1` 查看 `ttir.mlir` 和 `ttadapter.mlir`，并提醒 TTIR 仍是平台无关的 Triton 抽象。建议只在最小实验里开 dump，避免把全量流水的 cache/debug 产物混进结果。

本课源码核对依据：当前 TE checkout 的 `codegen/wrapper.py`（`write_header` / `make_allocation`）、`lowering.py`（`npu_permute` / `realize`）和 `codegen/triton.py`（NPU workspace 与 kernel 参数生成）。注意区分 checkout 当前实现和上游/其他版本实现。

## 8. 迷你实验与算术核对

数学预期：转置 `T` 的 `(1,2)` 坐标映射为底层 offset `1 + 2×2 = 5`，读出 `5`；连续化后同坐标映射为 `3×1 + 2 = 5`，值不变，但连续化 buffer 顺序变为 `[0,2,4,1,3,5]`。两边值相同，layout 不同。

可在 NPU 机器上用下面脚本检查 eager 与 `triton_experimental` compiled 的正确性和 wrapper 产物：

```python
import torch
import torch_npu

x = torch.arange(6, device="npu", dtype=torch.float32).reshape(3, 2)
t = x.t()
assert tuple(t.shape) == (2, 3)
assert tuple(t.stride()) == (1, 2)

def f(a):
    return a * 2 + 1

compiled_f = torch.compile(f)
y_eager = f(t)
y_compiled = compiled_f(t)
torch.testing.assert_close(y_compiled, y_eager)

tc = t.contiguous()
assert tuple(tc.stride()) == (3, 1)
torch.testing.assert_close(tc, t)
print("t stride:", t.stride(), "tc stride:", tc.stride())
```

运行时显式选择课程要求的后端：

```bash
TORCHINDUCTOR_NPU_BACKEND=triton_experimental \
TORCH_COMPILE_DEBUG=1 \
TORCH_LOGS="+inductor,output_code" \
python stride_allocation_check.py
```

**验证状态**：上述偏移公式和排列顺序可直接手算核对；本课没有声称已经在 NPU 上运行该脚本。NPU 实测应记录 PyTorch/torch_npu、Triton-Ascend、CANN 版本，并分别检查 eager、TE compiled、wrapper 和 TTIR/adapter IR。性能上“保留 strided view”与“复制成连续 layout”谁更快，取决于 shape、stride、消费者以及 NPU 的访存路径，不能只凭公式判定。

## 本课记忆卡

- view 可能只改逻辑 shape/stride；`contiguous()`/`realize()` 通常意味着需要物化，但优化后未必保留独立 copy。
- Inductor/TE 编译期生成 buffer 规划与 wrapper；wrapper 在图函数执行时调用分配入口，runtime/allocator 才提供实际 NPU storage。
- 输入地址偏移公式是 `Σ index[d] × stride[d]`（还需考虑 storage offset）；kernel 按它算地址，并不会因为一次 load 就新分配 Tensor。
- Triton-Ascend 把 Triton 的索引/load/store 语义继续降低到 Ascend 后端；设备指令与 kernel 局部资源规划和 PyTorch Tensor storage 分配不是同一个概念。
- TE 的 NPU lowering 可以在 Triton-Ascend 编译前决定保留 view 或 materialize，以兼顾正确性和 NPU 访存效率。

## 参考

- [PyTorch Inductor `graph.py`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/graph.py)：buffer layout、stride、offset 与所需 storage size 的关系。
- [Triton-Ascend 架构设计](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/architecture_design_and_core_features.md)：Triton IR、Linalg IR、BiSheng 与设备代码生成的职责分段。
- [Triton-Ascend 调试指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/debug_guide/debugging.md)：TTIR/TTAdapter IR 阶段、`tt.addptr`/`tt.load`/`tt.store` 及 dump 方法。
