# 图模式基础（八）：广播与零步长 stride

上一课把转置视图连续化；本课回到“先不复制数据，能不能让同一个元素服务多个逻辑位置？”这就是广播（broadcasting）的核心。答案是可以，而 `expand` 常通过 **stride 为 0 的视图**表达这种复用。

## 1. 一个 3×1 加一个 1×4

设：

```text
A = [[10],
     [20],
     [30]]        shape=(3,1)

B = [[1, 2, 3, 4]] shape=(1,4)
```

按广播规则，单例维可以扩展到对方的维度，因此输出逻辑 shape 为 `(3,4)`：

```text
A + B = [[11, 12, 13, 14],
         [21, 22, 23, 24],
         [31, 32, 33, 34]]
```

注意：这里说“扩展”先描述的是逻辑坐标空间，不等于真的把 A 复制四列、把 B 复制三行。PyTorch 的 `expand()` 可用 view 表达广播：扩展的单例维 stride 设为 0，不为重复的值分配新存储。[`Tensor.expand` 官方说明](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html)

## 2. 手算 stride=0 地址公式

假定 A、B 都是普通连续的二维张量：

```text
A 原始 shape=(3,1), stride=(1,1)
B 原始 shape=(1,4), stride=(4,1)
```

广播成 `(3,4)` 后，沿着被扩展的单例维前进，底层地址不变，因此对应 stride 为 0：

| 张量 | 广播后 stride | 逻辑坐标 `(i,j)` 的元素偏移 | 实际读到 |
| --- | --- | --- | --- |
| A | `(1,0)` | `i×1 + j×0 = i` | `A[i,0]` |
| B | `(0,1)` | `i×0 + j×1 = j` | `B[0,j]` |
| C=A+B | 通常新建连续输出 `(4,1)` | `i×4 + j` | 每个输出坐标有独立位置 |

比如输出坐标 `(2,3)`：A 的偏移是 `2`，读到 `30`；B 的偏移是 `3`，读到 `4`；输出偏移是 `2×4+3=11`，写入 `34`。

广播读取公式可以一般化为：对每个输入维度 `d`，若该输入维度是单例并被广播，就令对应坐标贡献为 `0×stride[d]=0`；否则使用实际逻辑坐标。缺失的前导维也按单例维处理。逐元素计算仍遍历输出的逻辑坐标，只是各输入的地址公式不同。

## 3. 三种“扩展”要区分

| 写法 | 逻辑 shape | 是否复制扩展值 | 直觉 |
| --- | --- | --- | --- |
| `a.expand(3,4)` | `(3,4)` | 否，创建 stride-0 view | 多个坐标映射回同一个存储位置 |
| `a.repeat(1,4)` | `(3,4)` | 是，创建重复数据 | 真正做出多份元素 |
| `a + b` 的广播 | 输出按广播规则确定 | 输入不必先显式复制；输出通常需要存结果 | 运算时按各自索引规则取数 |

`expand` 通常省掉输入侧复制，但**不代表整个操作不占额外内存**：加法结果仍要有输出空间；后续算子也可能选择物化布局。具体图中是否保留显式 `expand` 节点，要以 FX/Inductor IR 为准，不能只看 eager Python 写法就猜生成 kernel。

## 4. 读同一地址可以，写同一地址要小心

上例里不同输出坐标会重复读取 A 或 B：例如 `(0,0)` 和 `(0,1)` 都从 A 偏移 `0` 读取。这对只读输入是有意义的复用。

但若多个逻辑坐标映射到同一地址，再让它们并行写回，就会发生重叠写入/数据竞争。因此不能把 stride-0 广播 view 当成普通独立元素数组来原地修改。PyTorch 文档也警告：expanded tensor 的多个元素可能指向同一内存位置，in-place 操作可能错误；若要独立写入，应先 `clone()` 或使用新输出。[官方警告](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html)

这和 GPU/NPU kernel 的索引设计直接相关：**输入地址允许因广播而复用，输出地址通常必须保证各个并行逻辑元素不会意外重叠。** 是否允许重叠写，取决于具体算子定义和同步/归约机制，不能当作普通逐元素 store。

## 5. 再加一层数学：广播反向会累加梯度

令 `C[i,j]=A[i,0]+B[0,j]`，并定义 `L=Σ_i Σ_j C[i,j]`。每个 A 元素被横向复用 4 次，每个 B 元素被纵向复用 3 次：

```text
∂L/∂A[i,0] = Σ(j=0..3) ∂L/∂C[i,j] = 4
∂L/∂B[0,j] = Σ(i=0..2) ∂L/∂C[i,j] = 3
```

对本例的数值再算一次：

```text
sum(A+B) = 4×(10+20+30) + 3×(1+2+3+4)
         = 240 + 30
         = 270

grad_A = [[4], [4], [4]]
grad_B = [[3, 3, 3, 3]]
```

这解释了一个容易漏掉的点：前向广播只是多次读同一输入；反向则要把这些逻辑位置的梯度沿被广播维**求和折回原始 shape**。这也是 AOTAutograd backward 中需要正确处理广播关系的原因之一。

## 6. 公式小脚本验证

本课当前环境没有可用的本地 PyTorch 安装，因此先用纯 Python 对输出、地址偏移、loss 和梯度计数做了独立算术校验；结果分别是输出矩阵如上、总和 `270`、`grad_A=[4,4,4]`、`grad_B=[3,3,3,3]`。这是公式检查，不是 PyTorch 或 NPU 执行结果。

同一计算的 NPU PyTorch 最小用例如下：

```python
import torch
import torch_npu

a0 = torch.tensor([[10.0], [20.0], [30.0]], device="npu", requires_grad=True)
b0 = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="npu", requires_grad=True)
a = a0.expand(3, 4)
b = b0.expand(3, 4)
c = a + b
loss = c.sum()
loss.backward()

torch.testing.assert_close(
    c, torch.tensor([[11., 12., 13., 14.], [21., 22., 23., 24.], [31., 32., 33., 34.]], device="npu")
)
torch.testing.assert_close(a0.grad, torch.full((3, 1), 4., device="npu"))
torch.testing.assert_close(b0.grad, torch.full((1, 4), 3., device="npu"))
assert a.stride() == (1, 0)
assert b.stride() == (0, 1)
```

在 NPU 进程中应显式指定课程后端：

```bash
TORCHINDUCTOR_NPU_BACKEND=triton_experimental \
python broadcast_zero_stride_check.py
```

本课没有声称上述 NPU 用例已通过：本次可用机器的 NPU 健康状态显示 `Warning`，而指定流水镜像拉取还需要先配置 SWR 登录。因此 NPU/eager、TE 编译正确性、生成代码中的零步长索引以及反向图都列为待实测项。机器恢复可用后，按“先 eager 前向/梯度，再 `torch.compile` 前向/梯度，最后检查日志与生成代码”的顺序补测。

## 本课记忆卡

- 广播让不同 shape 的输入按兼容维度逐元素计算；单例维扩展后可用 stride 0 表达重复读取。
- 输出坐标 `(i,j)` 上，A 读偏移 `i`，B 读偏移 `j`，输出写偏移 `4i+j`。
- `expand` 是 view 语义，不复制扩展值；`repeat` 会复制数据；广播运算的结果仍需存储。
- stride-0 输入可安全表达共享读取；多个并行输出写到同一地址则需要算子级别的明确语义与同步。
- 反向传播要沿被广播维求和，将输出梯度折回原输入 shape。
- 本课算术公式已用纯 Python 校验；NPU 与 TE 编译结果尚待机器/镜像可用后验证。

## 参考

- [PyTorch：`Tensor.expand`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html)
- [PyTorch：Tensor Views](https://docs.pytorch.org/docs/main/tensor_view.html)
