# 图模式基础（四）：Graph Break 与重新编译

这一课把两个常被混为一谈的现象分开：**graph break 是一段 Python 执行被拆成多个 FX 子图；重新编译是某个已捕获代码区域因 guard 不匹配而生成新的编译版本。** 前者改变一次执行的分段结构，后者改变某个区域的缓存版本。

## 用一条执行流水来想

`torch.compile` 并不是把 Python 程序原封不动交给 NPU。Dynamo 逐段跟踪 Python 执行：

```text
可捕获的 Tensor 运算 -> FX 子图 A
不能捕获的 Python 片段 -> 普通 Python 执行
恢复跟踪 -> FX 子图 B
```

因此 graph break 并不等于“整个函数退回 eager”。通常 break 前后的可捕获片段仍可以分别编译。需要注意，这里说的是 Dynamo/Inductor 编译图；与 `torch.npu.NPUGraph` 的 ACL Graph capture/replay 是不同层次的“图”。

## 用公式验证：切开图不应改变数学结果

设输入 `x = [-2, -1, 0, 3]`，定义：

```text
u = relu(x + 1)
y = 2u
```

若在 `u` 和 `y` 之间插入 graph break，数学上仍只是把同一个函数分两段计算：

```text
G1(x) = relu(x + 1)
G2(u) = 2u
f(x)  = G2(G1(x)) = 2 * max(x + 1, 0)
```

手算每个元素：

| `x_i` | `x_i + 1` | `u_i = max(x_i + 1, 0)` | `y_i = 2u_i` |
|---:|---:|---:|---:|
| -2 | -1 | 0 | 0 |
| -1 | 0 | 0 | 0 |
| 0 | 1 | 1 | 2 |
| 3 | 4 | 4 | 8 |

所以 eager、单图编译、分成两段编译的数值结果都应为 `[0, 0, 2, 8]`。graph break 的目标是改变“如何分段执行”，不是改变这个函数的数学语义。

## 和上一课的 guards/重新编译对照

| 现象 | 发生了什么 | 常见观察方式 |
|---|---|---|
| Graph break | 一个 Python frame 被切成多个可编译子图，break 处执行未捕获代码后再继续 | `TORCH_LOGS=graph_breaks` |
| Guard 失败并重新编译 | 某个已编译区域的输入条件不满足，Dynamo 为该区域创建另一个编译版本 | `TORCH_LOGS=recompiles` |

一个函数可以有 graph break，但每个子图仍然不重新编译；也可以没有 graph break，却因为输入 shape 违反 guard 而重新编译。它们是两个不同维度的问题。

显式 graph break 最适合教学，因为位置可控；生产代码里的 break 通常来自 Dynamo 无法跟踪的 Python/C 扩展行为、数据依赖控制流等。要找自然产生的 break 及其原因，可先看 graph-break 日志，也可用 `torch._dynamo.explain` 汇总分析；不要只看到“编译区域变多”就认定是 shape 重编译。

## `fullgraph` 是侦测开关，不是让 break 消失的开关

- `fullgraph=False`（默认）：允许 Dynamo 在 break 前后继续捕获可编译子图。
- `fullgraph=True`：要求整个函数捕获成一个图；如果中间存在 graph break，就应当暴露/报告该问题，而不是悄悄保证“只有一个图”。

这对定位非常有用：先用 `fullgraph=True` 找到无法捕获的位置，再决定是改写代码、接受分段，还是把特定函数排除在编译外。若刻意写入 `torch._dynamo.graph_break()`，fullgraph 模式下报错是本例的预期行为，不等于 NPU kernel 运行时故障。

## 最小 NPU 实验：待设备恢复后执行

当前记录日期（2026-10-27）无法连到之前使用的 NPU 主机：SSH TCP/22 连接超时；本地环境也没有安装 PyTorch。因此下面是待实测脚本，不记录为已验证的 NPU 结果。

```python
import torch

def no_break(x):
    u = (x + 1).relu()
    return u * 2

def with_break(x):
    u = (x + 1).relu()
    torch._dynamo.graph_break()  # 人工设置边界，便于观察
    return u * 2

x = torch.tensor([-2.0, -1.0, 0.0, 3.0], device="npu")
compiled = torch.compile(with_break, backend="inductor", fullgraph=False)
actual = compiled(x)
expected = no_break(x)
torch.testing.assert_close(actual, expected)
print(actual.cpu().tolist())  # 预期 [0.0, 0.0, 2.0, 8.0]
```

使用 `TORCHINDUCTOR_NPU_BACKEND=triton_experimental`，分别设置 `TORCH_LOGS=graph_breaks` 与 `TORCH_LOGS=output_code` 观察。先检查结果是否符合手算；然后检查生成代码/日志确认 graph break 前后是否各形成了编译区域，以及各自实际发射多少 kernel。**不能只凭 break 的个数推断 kernel launch 数或性能损失**：融合、代码生成和运行时行为需要查看实物产物；拆段可能阻断跨边界融合，也可能带来额外调度/launch 开销，但当前环境不可达，尚未测量。

运行 `fullgraph=True` 的对照时，预期会因为人工插入的 break 无法满足“一整个图”的要求而失败；该检查应放在独立进程，并把预期异常与数值正确性实验分开。

## 本课记忆卡

- Graph break = 一次执行分成多个 FX 子图；不意味着整段函数都退回 eager。
- Recompile = 某个子图的 guard 失效，需要再编一个版本；不是把 Python 执行切成更多段。
- 正确性先用公式和 eager 对比；性能再检查子图融合、生成 kernel 与 launch，不能把“compile 成功”当作性能结论。
- NPU 实测仍待连接恢复后补充；本页的计算表是手算验证，代码块是待运行实验，不是运行记录。

## 参考

- [PyTorch：Working with Graph Breaks](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html)
- [PyTorch：Common Graph Breaks](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html)
- [PyTorch：`torch.compile` 参数](https://docs.pytorch.org/docs/stable/generated/torch.compile)
- [PyTorch：TORCH_LOGS / 可观测性](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.observability.html)
