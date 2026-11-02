# 图模式基础（二）：FX Graph 到 Inductor IR

> 学习路线：接续“图模式基础（一）”的 `(x + y).relu()`，进入 FX Graph、Inductor IR 与 Triton kernel 三种表示的对照。
>
> 本节目标：看懂同一计算经过编译器时，哪些信息保留、哪些信息被改写，以及如何用实际产物判断融合。

## 1. 同一段计算，三种视角

输入函数仍然是：

```python
def f(x, y):
    return (x + y).relu()
```

用 `X, Y ∈ R^(2×3)` 表示输入：

```text
X = [[ 1, -2,   3],
     [ 4,  5,  -6]]
Y = [[10, 20, -30],
     [40,-50,  60]]
Z[i,j] = max(X[i,j] + Y[i,j], 0)
```

计算结果是：

```text
Z = [[11, 18, 0],
     [44,  0,54]]
```

编译器不会只保留这一张结果表，而会构造逐层变化的表示：

| 层 | 本例中能观察到的内容 | 它主要回答什么 |
| --- | --- | --- |
| FX Graph | 两个 ATen 节点：`add`、`relu`，以及输入输出依赖 | Python 张量程序表达了哪些算子、它们怎样连接？ |
| Inductor IR | 输出 buffer、迭代域、输入读依赖、输出写依赖和逐元素 loop body | 对哪些索引计算？数据如何读写？哪些表达式可以内联或合并？ |
| Triton kernel 与 wrapper | `tl.load`、加法、`maximum`、`tl.store`；wrapper 分配输出并调用 kernel | 设备上实际执行什么代码？怎样 launch？ |

FX 和 Inductor IR 不是“图模式”和“非图模式”的区别；它们是编译流程中抽象层级不同的表示。FX 更接近 PyTorch 算子语义，Inductor IR 开始明确描述循环、布局和内存依赖，Triton 代码则落实到设备程序。

## 2. FX Graph：记录算子与数据依赖

本次 NPU 编译的 `fx_graph_readable.py` 中，核心结构等价于：

```text
arg0_1 ─┐
        ├─ aten.add.Tensor ─ aten.relu.default ─ return
arg1_1 ─┘
```

它也可写成：

```python
add = torch.ops.aten.add.Tensor(arg0_1, arg1_1)
relu = torch.ops.aten.relu.default(add)
return (relu,)
```

FX Graph 保留“`relu` 消费 `add` 的结果”这一依赖。节点是算子调用，不代表每个节点最终必须单独启动一个 kernel。后端可以把多个节点合并执行，只要可观察的计算语义不变。

本例使用 `fullgraph=True` 编译成功，表示这个小函数作为一个完整图交给后端，没有因为不支持的 Python 操作切成多个 Dynamo 图段。这个结论只对本函数和本次输入成立，不代表任意模型都没有 graph break。

PyTorch 文档将 FX 描述为包含 symbolic tracer、IR 和 Python 代码生成的工具；TorchDynamo 会从 Python 执行中抽取 PyTorch 运算序列，生成 FX Graph 并交给后端编译。[FX 文档](https://docs.pytorch.org/docs/stable/fx.html)，[TorchDynamo 概览](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamo_overview.html)

## 3. Inductor IR：把二维张量写成循环和读写关系

本例输入是连续的 `2×3` FP32 张量，元素 stride 为 `(3, 1)`。对于二维下标 `(i,j)`，行优先线性索引为：

```text
p = i × 3 + j
0 ≤ i < 2, 0 ≤ j < 3
0 ≤ p < 6
```

反向还原下标：

```text
i = floor(p / 3)
j = p mod 3
```

FP32 每个元素占 4 bytes，所以输入地址为：

```text
address_X(i,j) = base_X + 4 × (i × 3 + j)
address_Y(i,j) = base_Y + 4 × (i × 3 + j)
```

对任意 stride `(s0,s1)`，元素偏移公式是 `i×s0 + j×s1`；只有连续的 `(3,1)` 才可直接化成 `p`。这就是为什么阅读非连续输入时，不能仅凭 shape 推断内存地址连续。

本次 `ir_post_fusion.txt` 的关键内容可概括为：

```text
写依赖：buf0[c0]，0 ≤ c0 < 6
读依赖：arg0_1[c0] 与 arg1_1[c0]
迭代域：p0 ∈ [0, 6)
loop body：load arg0_1、load arg1_1、add、relu、store buf0
```

也就是说，IR 直接遍历六个元素，并把两个输入的相同线性位置读出来，计算后写入一个输出 buffer。这里没有单独给 `X+Y` 分配中间输出 buffer；这从 IR 的读写依赖就能看出来。

### 3.1 六个索引逐个验算

| `p` | `(i,j)` | `X[p] + Y[p]` | `relu` 后写入 `Z[p]` |
| ---: | ---: | ---: | ---: |
| 0 | `(0,0)` | `1 + 10 = 11` | `11` |
| 1 | `(0,1)` | `-2 + 20 = 18` | `18` |
| 2 | `(0,2)` | `3 - 30 = -27` | `0` |
| 3 | `(1,0)` | `4 + 40 = 44` | `44` |
| 4 | `(1,1)` | `5 - 50 = -45` | `0` |
| 5 | `(1,2)` | `-6 + 60 = 54` | `54` |

用整除和取余还原每个 `p`，能得到原来的六个 `(i,j)`，没有漏元素或重复元素。这是公式层面的完整覆盖检查。

## 4. Triton kernel：表达式如何落到设备程序

生成代码中的核心运算近似为：

```python
x0 = tl.load(in_ptr0 + x0index, x0mask)
x1 = tl.load(in_ptr1 + x0index, x0mask)
sum_value = x0 + x1
result = maximum(0, sum_value)
tl.store(out_ptr0 + x0index, result, x0mask)
```

实际 NPU 生成代码使用 `npu_triton_heuristics.pointwise`，并由 wrapper 通过一次 `triton_unk_fused_add_relu_0.run(...)` 启动。kernel 的有效性 mask 是 `x0index < x0numel`，本次 `x0numel=6`。因此尾部超出六个元素的位置不会被读写。

第一课里的 `BLOCK=4` 只是为了讲清楚 mask 的教学例子，不是本次硬件编译所选的真实 block 配置。真实配置来自后端 heuristic；读生成代码时应以实际代码和元数据为准。

## 5. 一个需要记住的细节：FX 节点数不等于 IR 节点数

本次 FX Graph 明确有 `add`、`relu` 两个算子节点。但命名为 `ir_pre_fusion.txt` 和 `ir_post_fusion.txt` 的产物里，都可以看到一个 scheduler node，其 loop body 已按顺序包含 `add` 与 `relu`。因此不能机械地期待“每个 FX 节点对应一个 IR 节点”，也不应只凭文件名猜测融合发生的准确阶段。

本次能确定的事实是：

1. FX 产物保留了两个 ATen 运算及数据依赖；
2. IR dump 用一个输出 buffer 和一个逐元素 loop 表达这两个运算；
3. 最终生成一个 fused Triton kernel，wrapper 只 launch 一次；
4. eager 与 compiled 结果一致。

这四项分别提供了语义结构、内存与循环结构、设备执行结构和数值正确性证据。它们共同支持“本例被融合并正确执行”的结论。

## 6. 本次 NPU 实测复核

实验环境与第一课相同：Ascend 910B2，CANN 9.2.0，`torch 2.13.0+cpu`、`torch_npu 2.13.0+gite94cbac`；进程设置 `TORCHINDUCTOR_NPU_BACKEND=triton_experimental`。生成的代码导入 `torch_npu._inductor.triton_experimental.npu_triton_heuristics`，kernel device 为 `npu`。

已通过的输入包括：

- 二维 `2×3` 示例矩阵；
- 一维长度 `1、5、7、17`，其中含奇数长度和不规则尾部。

每种输入都用 `torch.testing.assert_close(compiled, eager)` 比对，通过 `fullgraph=True` 编译。测试验证的是正确性和这条后端的融合结构；本次没有测试 GPU 对照、性能差异、动态 shape 泛化或非连续 stride。

## 7. 自检题

1. FX Graph 里有两个算子，为什么最终只看到一个 kernel launch？
2. 为什么 `2×3` 的 shape 还不足以证明数据连续？还要知道什么？
3. 对连续 FP32 输入，`(i,j)=(1,2)` 对应的元素偏移和字节偏移各是多少？
4. 只比较 eager 和 compiled 的数值，能不能证明发生了一次融合？还需要看什么？

答案要点：后端可在保持依赖语义的前提下把逐元素表达式合并；地址还取决于 stride；`p=1×3+2=5`，字节偏移 `20`；还需看 IR 读写依赖、生成 kernel 数量和 wrapper launch。

## 参考

- [PyTorch FX 文档](https://docs.pytorch.org/docs/stable/fx.html)
- [TorchDynamo 概览](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamo_overview.html)
- [图模式基础（一）：从 Eager 到一个融合 Kernel](图模式基础（一）：从Eager到一个融合Kernel.md)
- [学习路线](Triton-Experimental与GPU-Triton学习路线.md)
