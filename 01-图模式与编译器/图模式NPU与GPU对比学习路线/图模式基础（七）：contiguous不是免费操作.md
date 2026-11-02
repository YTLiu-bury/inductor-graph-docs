# 图模式基础（七）：`.contiguous()` 不是免费操作

第六课我们看到：同样的 `(2, 3)` shape，连续张量和转置视图的 stride 不同，坐标到地址的映射也不同。本课继续追问：**如果某个算子更喜欢连续输入，能不能先调用 `.contiguous()`？可以，但要算清它可能带来的数据搬运与重新编译成本。**

## 1. 从转置视图开始：逻辑值不变，物理排列不同

建立一个连续的 3×2 张量，再转置成逻辑上的 2×3：

```text
base（shape=(3, 2), stride=(2, 1)）:
[[0, 1],
 [2, 3],
 [4, 5]]

view = base.t()（shape=(2, 3), stride=(1, 2)）:
[[0, 2, 4],
 [1, 3, 5]]
```

转置 view 沿用 base 的存储；它的逻辑坐标 `(i, j)` 对应的元素偏移是 `i + 2j`。也就是说，逻辑第 0 行中的 `0, 2, 4` 分别位于底层偏移 `0, 2, 4`，这三个值并不连续挨在一起。

## 2. `.contiguous()` 做了什么

默认的 `.contiguous()` 要求返回值符合 `torch.contiguous_format`。对上面的非连续 view，它保留逻辑 shape 和每个逻辑位置的值，但新建连续排列：

```text
view:        shape=(2, 3), stride=(1, 2), offset(i,j)=i+2j
view.contiguous():
             shape=(2, 3), stride=(3, 1), offset(i,j)=3i+j

两者逻辑值都为 [[0, 2, 4],
                [1, 3, 5]]
```

新连续存储按逻辑行优先顺序排列为 `[0, 2, 4, 1, 3, 5]`。因此这不是只改 stride 元数据：非连续输入通常要分配目标存储，并把各逻辑元素读出后写入新布局。若输入已经符合指定 memory format，PyTorch 文档说明 `.contiguous()` 会直接返回该 Tensor；否则返回包含相同数据的连续 Tensor。[官方 API 说明](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html) [Tensor Views 指南](https://docs.pytorch.org/docs/main/tensor_view.html)

### 手算这次小拷贝

这个例子有 `2×3=6` 个 FP32 元素，每个 4 字节，所以源数据负载为 `6×4=24` 字节。非连续转连续至少要读出 24 字节并写入 24 字节，名义数据流量约 `48` 字节，此外还可能有分配器和调度开销。这里的数字是从数据量推导出的访问量估算，不是性能测试结果；实际耗时还取决于设备、缓存、同步和后续算子。

如果一个大张量只做一次很便宜的逐元素操作，额外复制可能得不偿失；如果连续化后有多个算子重复受益，或者某个接口确实要求特定布局，则可能值得。判断单位应是“整个后续计算链路”，而不是仅看 `.contiguous()` 之后的单个 kernel。

## 3. 一个容易忽略的边界：`is_contiguous` 不等于 stride 必然唯一

日常 2×3 例子里，连续布局是 `(3, 1)`。但含有大小为 1 的维度时，该维没有不同坐标可供跳转，因此某些不同 stride 也可能仍满足 `is_contiguous()`。不要把“连续”误解成所有维度的 stride 在所有 shape 下都只有唯一取值；需要精确布局时，同时检查 shape、stride 和 memory format。

## 4. NPU eager 实测：复制与 no-op 两种情况

2026-11-02 在 Ascend 950DT 上执行了 6 个 FP32 元素的 eager 验证，软件为 PyTorch `2.13.0+cpu`、torch_npu `2.13.0+git13057da`、CANN `9.2.0`。运行进程显式设置 `TORCHINDUCTOR_NPU_BACKEND=triton_experimental`（本段验证的是 eager 张量布局 API，后端变量不改变 `.contiguous()` 的 eager 语义）。

验证程序的核心为：

```python
base = torch.arange(6, device="npu:0", dtype=torch.float32).reshape(3, 2)
view = base.t()
packed = view.contiguous()
already_packed = packed.contiguous()

assert view.shape == packed.shape == (2, 3)
assert view.stride() == (1, 2)
assert packed.stride() == (3, 1)
assert view.data_ptr() != packed.data_ptr()
assert packed.data_ptr() == already_packed.data_ptr()
torch.testing.assert_close(view, packed)
```

实测输出：

```text
view       stride=(1, 2)  [[0, 2, 4], [1, 3, 5]]
contiguous stride=(3, 1)  [[0, 2, 4], [1, 3, 5]]
再次 contiguous 后 data_ptr 相同：True
```

这验证了两点：非连续 view 连续化后逻辑值相同、stride 改为 `(3, 1)` 且本次确实使用了不同数据地址；对已经连续的结果再次调用 `.contiguous()`，本次返回了相同数据地址。这里通过 `data_ptr()` 做的是本次运行的存储身份检查，不是通用的性能或内存分配器行为保证。

## 5. 它和 `torch.compile` 的 guard 有什么关系

把上一课验证过的 `f(x)=2x+1` 用同一个静态编译函数依次调用：

```python
def f(x):
    return x * 2 + 1

compiled = torch.compile(f, backend="inductor", fullgraph=True, dynamic=False)
for x in (view, view.contiguous(), view.contiguous()):
    torch.testing.assert_close(compiled(x), f(x))
```

课程上一课在该 NPU 后端实测到：同一个 shape 的 `(3,1)` 与 `(1,2)` 输入会因 stride guard 不匹配而重新编译。因此本例的可验证预期是：若先用 stride `(1,2)` 编译，随后传入 stride `(3,1)` 的连续化结果，旧版本的 stride 假设不成立，需要为新布局重新编译；第三个输入的 shape/stride 相同，即使它来自另一次连续化，也应有机会复用对应布局版本。**这一课实际运行的 NPU 编译验证因机器当时系统负载很高而中断，没有获得该序列的 `TORCH_LOGS=recompiles` 结果；上述编译过程在本课中标为待复测预期，不冒充实测。**

机器低负载时可用以下环境变量和代码复测；要确认的观察点是第二种 stride 是否触发重编译、第三次调用是否复用，而不是仅看函数能否返回：

```bash
TORCHINDUCTOR_NPU_BACKEND=triton_experimental \
TORCH_LOGS=recompiles \
python contiguous_guard_check.py
```

```python
base = torch.arange(6, device="npu:0", dtype=torch.float32).reshape(3, 2)
view = base.t()
compiled = torch.compile(f, backend="inductor", fullgraph=True, dynamic=False)

for x in (view, view.contiguous(), view.contiguous()):
    actual = compiled(x)
    torch.testing.assert_close(actual, f(x))
```

即使上述 guard 结果成立，也不能单独说明 `.contiguous()` 让程序更快。它可能新增一次布局复制和分配；也可能令后续多个 kernel 的访问更规则。性能结论需要分别测量复制成本和端到端链路，并保持 backend、输入、同步方式一致。

## 本课记忆卡

- `.contiguous()` 保留逻辑 shape 和值，目标是满足指定 memory format；非连续输入需要重新物化，已连续输入可直接返回自身。
- 2×3 FP32 示例从 `(1,2)` 转成 `(3,1)`，元素排列变为按逻辑行优先的 `[0,2,4,1,3,5]`。
- 复制的名义数据流量至少包含读和写；“连续”不等于免费，也不等于一定更快。
- layout 改变可能让 `torch.compile` 的 stride guard 失效；编译成本、复制成本和下游访问收益要分开分析。
- 本课已验证 NPU eager 布局和值；NPU compiled guard 序列受当时高负载影响尚未取得日志，等待低负载重测。

## 参考

- [PyTorch：`Tensor.contiguous`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html)
- [PyTorch：Tensor Views](https://docs.pytorch.org/docs/main/tensor_view.html)
- [PyTorch：`Tensor.is_contiguous`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html)
