# 图模式基础（五）：AOTAutograd 与前向/反向图

前几课我们一直看前向计算如何进入 FX Graph 和 Inductor。训练还多一半：**梯度怎么算、反向算子如何编译？** AOTAutograd 的关键作用是根据前向计算提前捕获反向图，并把前向与反向交给编译后端处理。

## 直觉：前向先写下“梯度作业单”

Eager Autograd 在运行前向时会记录反向所需的信息；调用 `backward()` 时，由 Autograd engine 沿记录的依赖关系执行梯度规则。

`torch.compile` 的训练路径会在编译阶段对可微的前向区域提前追踪其反向计算。官方文档把负责这一部分的组件称为 AOTDispatcher（也称 AOTAutograd）：捕获前向/反向，再将联合图分成可单独编译的前向图和反向图。调用 `.backward()` 时仍由 Autograd engine 调度，但其中对应的反向区域可以是已编译的函数。

可用这个简化数据流理解：

```text
Python forward
  -> TorchDynamo 捕获一个或多个 FX 前向区域
  -> AOTAutograd 为可微区域追踪联合 forward + backward
  -> partitioner 划分前向图 / 反向图（决定保留哪些中间值）
  -> backend（本课程 NPU 侧为 triton_experimental）编译这些图
  -> runtime forward
  -> loss.backward() 由 Autograd engine 调用相应已编译 backward 区域
```

如果前向里已有 graph break，通常会有多个前向区域；不能假设训练反向一定是一张全局大图。PyTorch 官方 FAQ 描述为：每个 Dynamo 捕获的前向 segment 都可以对应生成一个 backward graph segment。

## 手算一个前向和梯度

定义一维输入 `x`，每个元素先加 1、平方，然后把元素求和：

```text
u_i = x_i + 1
y_i = u_i^2
L   = Σ_i y_i
```

对每个 `x_i` 用链式法则：

```text
∂L/∂x_i = (∂L/∂y_i) · (∂y_i/∂u_i) · (∂u_i/∂x_i)
         = 1 · 2u_i · 1
         = 2(x_i + 1)
```

用 `x = [-1, 0, 2]` 手算：

| `x_i` | `u_i=x_i+1` | `y_i=u_i²` | `∂L/∂x_i=2u_i` |
|---:|---:|---:|---:|
| -1 | 0 | 0 | 0 |
| 0 | 1 | 1 | 2 |
| 2 | 3 | 9 | 6 |

因此 `L=10`，梯度为 `[0, 2, 6]`。这个结果可以同时校验 eager 和编译训练路径；若只检查 loss 而不检查 `.grad`，就没有验证 backward 的正确性。

## 为什么需要 partition：保存与重算的折中

反向计算可能需要前向中间值。上例的导数需要 `u=x+1`，编译器可以在前向保存 `u`，也可能在反向重新计算相关值。保存更占显存，但少做一些计算；重算节省保存的激活空间，但会增加计算量。AOTAutograd 的 partitioner 可以在前后向之间选择切分点；某些策略（如 min-cut partitioning）会利用重算/保存折中降低激活内存。

这是一种编译时图划分与内存—计算折中，不代表所有场景都采用同一策略，也不代表一定会重算 `u`。具体选择要检查实际生成的前向/反向图或编译日志。

## 与 Triton Experimental 的关系

AOTAutograd 负责准备可编译的前向/反向图；它本身不等于 NPU kernel。后端负责把传入的图编译成目标设备代码。我们要验证的路径是：

```text
torch.compile(backend="inductor")
  + TORCHINDUCTOR_NPU_BACKEND=triton_experimental
  -> AOT 生成 forward/backward graph
  -> triton_experimental/Inductor 编译后端处理对应图
```

此处还不能仅凭 `torch.compile()` 返回成功就认定两个方向都已在 NPU 上正确执行；需要跑 forward、backward，对比 eager 的 loss 与输入梯度，并查看 AOT 图日志。推理场景没有梯度需求，不会因为使用 `torch.compile` 就凭空产生有用的 backward 工作。

## NPU 最小实验（本次尚未运行）

当前无法连到实验 NPU 主机，且本地 Python 环境没有 PyTorch，因此以下脚本是待设备恢复后的复现步骤，不是已通过的实测记录。

```python
import torch

def loss_fn(x):
    return ((x + 1) ** 2).sum()

values = [-1.0, 0.0, 2.0]
x_eager = torch.tensor(values, device="npu", dtype=torch.float32, requires_grad=True)
x_compiled = torch.tensor(values, device="npu", dtype=torch.float32, requires_grad=True)

eager_loss = loss_fn(x_eager)
eager_loss.backward()

compiled_loss_fn = torch.compile(loss_fn, backend="inductor", fullgraph=True)
compiled_loss = compiled_loss_fn(x_compiled)
compiled_loss.backward()

torch.testing.assert_close(compiled_loss, eager_loss)
torch.testing.assert_close(x_compiled.grad, x_eager.grad)
torch.testing.assert_close(compiled_loss, torch.tensor(10.0, device="npu"))
torch.testing.assert_close(
    x_compiled.grad, torch.tensor([0.0, 2.0, 6.0], device="npu")
)
```

实验进程显式设置 `TORCHINDUCTOR_NPU_BACKEND=triton_experimental`。为控制日志量，仅对这个小脚本启用 `TORCH_LOGS=aot_joint_graph,aot_graphs,output_code`：

- `aot_joint_graph`：观察划分前联合图；
- `aot_graphs`：观察 AOTDispatch 划分后的 forward/backward 图；
- `output_code`：观察 Inductor 生成的后端代码。

预期的数学结果是 `loss=10`、梯度 `[0, 2, 6]`。实际输出、图数量、保存/重算决策和 kernel 数量仍需在目标 PyTorch/torch_npu/CANN 版本上运行后记录；不要把“预期值”误写成“设备实测值”。

## 本课记忆卡

- 前向图描述怎么产生输出；反向图描述上游梯度如何传回输入/参数。
- AOTAutograd 在编译阶段准备前向与反向图；Autograd engine 在运行时负责触发 backward。
- 反向依赖前向信息，因此图划分还要考虑保存哪些激活、哪些值可重算。
- graph break 是分段；每个可微分段都可能对应 backward 图段，并非一张完整反向图必然覆盖整个 Python 函数。
- 精度验证要比较 loss 和 `.grad`；只看 forward 输出不足以证明训练正确。

## 参考

- [PyTorch：`torch.compile` training FAQ](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_faq.html)
- [PyTorch：`torch.compile` autograd 行为](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_backward.html)
- [PyTorch：AOTAutograd 实现源码](https://github.com/pytorch/pytorch/blob/main/torch/_functorch/aot_autograd.py)
- [PyTorch：AOT graph 日志选项](https://docs.pytorch.org/docs/stable/logging)
