# 图模式基础（六）：Shape 不等于 Layout——stride 与地址

上一课我们看到 guard 会决定缓存的编译结果能否复用。本课用一个只有 6 个数的例子继续追问：**shape 一样的两个张量，为什么编译器仍可能需要不同的编译版本？**

答案之一是 stride（步长）：shape 告诉我们逻辑上有几个元素；stride 告诉我们沿每个维度移动一个元素时，底层存储要跳过多少个元素。转置可以只改变这份访问说明，而不搬动底层数据。

## 1. shape 是“怎么看”，stride 是“怎么走”

对二维张量 `x`，假设每个元素占一个存储位置，逻辑索引 `(i, j)` 对应的一维元素偏移为：

```text
offset_elements(i, j) = storage_offset + i * stride(0) + j * stride(1)
```

`storage_offset` 与 stride 都以**元素**为单位。若元素是 FP32（4 字节），相对存储起点的字节偏移是 `4 * offset_elements`。

对一个连续的 2×3 行优先张量：

```text
shape  = (2, 3)
stride = (3, 1)

offset(i, j) = 3i + j
```

内存顺序是 `[x00, x01, x02, x10, x11, x12]`。从一行开始到下一行要跨过 3 个元素；同一行向右走一格跨过 1 个元素。

现在先创建一个连续的 3×2 张量，再转置成 2×3 视图：

```text
底层 3×2:       [[0, 1],
                 [2, 3],
                 [4, 5]]

转置视图 2×3:   [[0, 2, 4],
                 [1, 3, 5]]
```

这个视图的 `shape` 是 `(2, 3)`，但 `stride` 是 `(1, 2)`：

```text
offset(i, j) = i + 2j
```

例如逻辑元素 `x[1, 2]` 的元素偏移是 `1 + 2×2 = 5`，其值为 `5`。两个张量 shape 相同、包含元素个数相同，但逻辑坐标到内存地址的映射不同。PyTorch 的 `Tensor.stride()` 返回各维 stride；stride 表示在指定维度上前进一步所需的跳跃量。[官方说明](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.stride.html)

## 2. 转置视图没有自动重排数据

`t()`/`transpose()` 常常返回一个 view：它共享原存储，只改变 shape/stride 等元数据。上例中底层顺序仍是 `[0, 1, 2, 3, 4, 5]`，转置后的逻辑遍历次序变成 `[0, 2, 4, 1, 3, 5]`，并没有因为写了 `.t()` 就把那 6 个值搬成另一种物理排列。

这也是“shape 不等于 layout”的直观版本：shape 描述逻辑坐标范围，stride 描述这些坐标落在哪些存储位置。调用 `.contiguous()` 会在需要时创建连续布局副本；它不是免费的元数据修改，可能引入一次真实的数据搬运。实际是否值得这样做，需要结合后续算子和整体开销验证。

## 3. 同 shape、不同 stride：验证 guard 与重新编译

我们让同一个逐元素函数依次接收连续张量和转置视图。函数的数学定义为：

```text
y[i, j] = 2 * x[i, j] + 1
```

用输入值直接手算：

| 输入布局 | 输入值 | 逐元素计算 | 预期输出 |
| --- | --- | --- | --- |
| 连续，stride `(3, 1)` | `[[0, 1, 2], [3, 4, 5]]` | 每项乘 2 再加 1 | `[[1, 3, 5], [7, 9, 11]]` |
| 转置视图，stride `(1, 2)` | `[[0, 2, 4], [1, 3, 5]]` | 每项乘 2 再加 1 | `[[1, 5, 9], [3, 7, 11]]` |

最小验证代码如下。`dynamic=False` 是为了让这次实验明确从静态特化开始；通过同一个编译函数调用两种输入，再观察第二次调用是否因 stride 不匹配而重新编译。

```python
import torch
import torch_npu

def f(x):
    return x * 2 + 1

compiled = torch.compile(f, backend="inductor", fullgraph=True, dynamic=False)
contiguous = torch.arange(6, device="npu:0", dtype=torch.float32).reshape(2, 3)
transpose_view = torch.arange(6, device="npu:0", dtype=torch.float32).reshape(3, 2).t()

for x in (contiguous, transpose_view):
    actual = compiled(x)
    expected = f(x)
    torch.testing.assert_close(actual, expected)
    print("shape", tuple(x.shape), "stride", x.stride(), "result", actual.cpu().tolist())
```

运行前显式选择课程要求的 NPU 后端，并打开重新编译原因日志：

```bash
TORCHINDUCTOR_NPU_BACKEND=triton_experimental \
TORCH_LOGS=recompiles \
python stride_layout_check.py
```

## 4. Ascend 实测结果

2026-11-02 在 Ascend 950DT 实测；软件信息为 PyTorch `2.13.0+cpu`、torch_npu `2.13.0+git13057da`、CANN `9.2.0`。进程明确设置了 `TORCHINDUCTOR_NPU_BACKEND=triton_experimental`，输入仅 6 个 FP32 元素，没有做耗时或性能比较。

实测输入输出如下：

```text
contiguous:
  shape=(2, 3), stride=(3, 1)
  input = [[0, 1, 2], [3, 4, 5]]
  output= [[1, 3, 5], [7, 9, 11]]
  output stride=(3, 1)

transpose_view:
  shape=(2, 3), stride=(1, 2)
  input = [[0, 2, 4], [1, 3, 5]]
  output= [[1, 5, 9], [3, 7, 11]]
  output stride=(1, 2)
```

两次输出都通过与 eager 结果的 `torch.testing.assert_close` 比较。第二次调用日志明确指出：

```text
Recompiling function f
tensor 'x' stride mismatch at index 0. expected 3, actual 1
```

这条日志说明此编译版本记录了 `x` 第 0 维 stride 为 3 的假设；转置视图该维 stride 是 1，因此 guard 不再成立，Dynamo 重新捕获并编译适配新布局的版本。PyTorch 对 `torch.compile` 的说明也指出，guard 失败时会重新编译；具体 guard 可以通过 `TORCH_LOGS=recompiles` 观察。[重新编译指南](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.recompilation.html)

在这个例子里，两个输入 shape 和 dtype 相同，变化的是 stride；因此观察到的重新编译原因不是 shape 变化。输出在本次算子及后端实测中保留了对应输入 stride，但不要据此推断任意算子都会保留输入 layout。

## 5. 本课记忆卡

- `shape` 规定逻辑索引范围；`stride` 把逻辑索引映射到底层元素偏移。
- 二维地址公式为 `storage_offset + i*stride(0) + j*stride(1)`；若要字节地址，再乘元素字节数。
- 转置视图可在不搬数据的情况下改变 shape/stride 元数据；逻辑遍历次序可能和底层存储顺序不同。
- 同 shape 不代表同 layout。编译代码若特化了 stride，stride guard 失败会触发新编译版本。
- 本例只验证了小型 FP32 逐元素函数、正确性与重新编译原因；没有验证性能，也没有覆盖所有 NPU 算子。

## 下一步可以验证

1. 对转置视图显式调用 `.contiguous()`：shape 不变，但 stride 会变成 `(3, 1)`；比较编译版本是否可复用，并解释复制发生在哪里。
2. 让两个维度大小相同（例如 2×2），再比较两种 stride；这可以排除仅凭 shape 区分布局的可能。
3. 后续再追踪 Inductor IR 中的 `Layout`、stride 表达式和生成 kernel 如何实现这条索引映射。
