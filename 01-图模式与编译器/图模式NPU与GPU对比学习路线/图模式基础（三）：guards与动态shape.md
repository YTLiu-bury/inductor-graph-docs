# 图模式基础（三）：guards 与动态 shape

这一课回答一个很实际的问题：`torch.compile` 编译过一次后，为什么有时可以直接复用，有时又要重新编译？

## 先用直觉理解 guard

可以把编译缓存想成一个按条件领取的“成品程序”。编译器第一次看到输入时，记录哪些条件对这份程序成立，例如张量的 shape、stride、dtype、device，以及 Python 分支依赖的静态值。下一次调用时，Dynamo 先检查这些条件（guards）：条件都满足，就复用缓存；有条件失效，就重新捕获/编译一份适用的程序。

guard 不是“输入内容必须完全一样”。对下面的逐元素函数来说，编译器不需要记住每个元素的值；但如果编译时把长度当作固定值，之后长度变了，原程序中的循环范围、mask 或布局假设可能不再成立，因此必须重新编译或使用动态尺寸程序。

本课讲的是 `torch.compile` 的 Dynamo/Inductor 编译图，不是 `torch.npu.NPUGraph` 的 ACL Graph capture/replay。它们都常被口语叫作“图模式”，但解决的问题和生命周期不同。

## 一个最小数学模型

取 `f(x) = relu(x + 1)`，其中 `x` 是长度为 `N` 的一维 FP32 连续张量。按元素写出来：

```text
y_i = max(x_i + 1, 0),  0 <= i < N
```

线性位置 `i` 对应的字节偏移是：

```text
offset_bytes(i) = i * stride * sizeof(float32)
```

连续一维输入 `stride = 1`，FP32 每元素 4 字节，因此 `offset_bytes(i) = 4i`。Triton 的尾块 mask 对应数学定义域：`i < N`；超出长度的 lane 不应读写有效张量元素。

若 shape 静态，编译图可把 `N` 固定为具体值，guard 就会检查这个长度。若 shape 动态，`N` 可以在图里表示为符号尺寸（如 `s0`），kernel 的索引/mask 使用运行时尺寸；这并不意味着“没有 guard”，而是 guard 从“必须等于 6”变成了可适用范围、布局等更一般的条件。

## NPU 实测：先静态，再自动动态

实验使用 Ascend 910B2，PyTorch `2.13.0+cpu`、CANN `9.2.0`，并显式设置 `TORCHINDUCTOR_NPU_BACKEND=triton_experimental`。函数为：

```python
def f(x):
    return (x + 1).relu()

compiled = torch.compile(f, backend="inductor", fullgraph=True, dynamic=False)
```

每个结果同时与 eager 的 `(x + 1).relu()` 做 `torch.testing.assert_close`。输入为一维 FP32 NPU 张量：

| 长度 N | 所有输入元素 x_i | 公式结果 max(x_i + 1, 0) | 结果 |
|---:|---:|---:|---|
| 6 | 1 | 2 | 6 个 2，正确 |
| 6 | -2 | 0 | 6 个 0，正确 |
| 7 | 3 | 4 | 7 个 4，正确 |

`dynamic=False` 日志显示：第二次长度仍为 6、但元素值改变时复用原图；改成长度 7 时 guard 失败，原因是 `tensor 'x' size mismatch at index 0. expected 6, actual 7`，随后重新编译。数学检查也可以手算：当 `x_i=1` 时，`max(1+1,0)=2`；当 `x_i=-2` 时，`max(-2+1,0)=0`；长度只改变参与计算的下标集合，不改变逐元素公式。

再用默认 `dynamic=None` 对照输入长度 `6 -> 7 -> 5 -> 13`，元素值分别为 `1 -> -2 -> 3 -> -4`。四种输入均通过 eager 对比。日志显示长度 6 首次编译；长度 7 触发一次 guard 失败后重新编译，并将 `x` 的长度标记为 dynamic；之后长度 5、13 未再出现重新编译日志。对于本环境、这个简单算子，自动动态化成功覆盖了后续长度。

结论限定在这次实验：已经实测 `triton_experimental` 后端可正确执行这个动态长度逐元素图；日志证明 Dynamo 从具体长度 guard 转为符号尺寸处理。不同算子、布局、PyTorch/torch_npu 版本可能增加约束或仍触发重新编译，不能由这个例子推导“所有动态 shape 都只编译一次”。

## 看日志时抓住这三件事

1. **首次调用**：通常发生 tracing/编译；这是预热成本，不代表稳态执行时间。
2. **guard 失败**：`TORCH_LOGS=recompiles` 会给出第一个失败条件。本例的失败条件是长度不匹配，而非值不同。
3. **动态符号与其他约束**：默认模式第二次编译出现 `marking L['x'] as dynamic` 和符号 `s77`。同一段日志仍有索引宽度相关 guard（`s77 <= 2147483647`），说明动态尺寸不是无约束尺寸；guard 也可能来自编译器的安全/实现约束，而不只是用户输入契约。

PyTorch 官方文档将默认 `dynamic=None` 描述为：先按静态 shape 编译；检测到 shape 变化并发生重编译后，自动尝试将变化维度标成动态。官方也提醒不要把 `dynamic=True` 当通用性能开关，因为它会尽可能提前动态化所有尺寸，可能影响编译和性能。我们的 NPU 实测与该策略相符，但版本是 2.13，具体行为应以当前安装版本的日志为准。

## 本课记忆卡

- **shape** 决定每维元素数量；**stride** 决定沿每维走一步跨过多少元素；它们都是编译器可能需要保护的张量元数据。
- **数据值变化**通常不会单独导致 Dynamo guard 失败；本例同 shape 不同值实测复用。
- **静态 shape**可能更利于专门化，但不同长度会有编译缓存分裂/重复编译。
- **动态 shape**让一个符号尺寸覆盖更多长度，但不保证适用于所有算子、stride、分支或后端优化。
- 诊断时先看 `TORCH_LOGS=recompiles` 的具体失效条件，再决定是否需要显式标记动态维度；不要仅凭“长度变化”就猜是 graph break。

## 可复现实验骨架

以下片段需在已配置好 Ascend NPU 与 `triton_experimental` 的环境执行。静态和默认动态两个版本分别新建进程运行，以便日志易读：

```python
import torch

def f(x):
    return (x + 1).relu()

compiled = torch.compile(
    f, backend="inductor", fullgraph=True, dynamic=False  # 再改为 dynamic=None 对照
)
for n, value in [(6, 1.0), (6, -2.0), (7, 3.0)]:
    x = torch.full((n,), value, device="npu", dtype=torch.float32)
    torch.testing.assert_close(compiled(x), f(x))
```

运行时设置 `TORCHINDUCTOR_NPU_BACKEND=triton_experimental` 和 `TORCH_LOGS=recompiles`；观察 shape 相同但 value 改变、以及 shape 改变时的日志差异。默认动态实验可再加上 `TORCH_LOGS=dynamic`，检查何时产生符号尺寸。

## 参考

- [PyTorch：Dynamic Shapes](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html)
- [PyTorch：Dealing with Recompilations](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.recompilation.html)
- [PyTorch：torch.compile 参数与 guard 行为](https://docs.pytorch.org/docs/stable/generated/torch.compile)
