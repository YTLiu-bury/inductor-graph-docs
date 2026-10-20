# NPU 可重放 RNG State PR 代码解读

## 1. 文档目的

本文结合 PTA PR #47685（https://gitcode.com/Ascend/pytorch/pull/47685）的实际代码，解释它为什么要改、每个文件改了什么、运行时如何串起来，以及当前已经覆盖和仍未覆盖的边界。

本文讨论的能力是：

> 在 NPU 编译训练路径中保存随机数状态，并在后续重新执行随机算子时恢复到同一个位置，从而得到逐元素一致的随机结果。

当前 PR 以 triton_experimental 作为首个验证后端，但修改点主要位于 PTA 的 NPU Inductor 适配层，目标并不只是某一个 Triton kernel。

关联 RFC：NPU 编译训练支持可重放 RNG state
https://gitcode.com/Ascend/pytorch/issues/5427

---

## 2. 为什么训练需要重放 RNG

### 2.1 Dropout 做了什么

Dropout 会为输入生成一个随机 mask。假设：

~~~text
x = [2, 4, 6, 8]
mask = [1, 0, 1, 0]
~~~

保留概率为 0.5 时，输出大致为：

~~~text
y = x * mask / 0.5
  = [4, 0, 12, 0]
~~~

### 2.2 Checkpoint 为什么要求同一个 mask

activation checkpoint 为了节省显存，前向时不保存全部中间结果，反向时重新执行一次前向逻辑：

~~~text
第一次前向：x -> dropout -> y
反向阶段：  x -> dropout -> 重新计算 y
~~~

重新执行 dropout 时，必须得到同一个 mask：

~~~text
第一次前向：mask = [1, 0, 1, 0]
反向重算：  mask = [1, 0, 1, 0]
~~~

如果反向重算生成了另一张 mask，计算路径发生变化，梯度可能不正确。

### 2.3 RNG state 保存什么

本 PR 重点处理两类信息：

~~~text
seed   ：随机序列的起点
offset ：当前随机序列已经使用到的位置
~~~

相同 seed 配合相同 offset，可以生成同一段随机序列。

---

## 3. PR 修改文件

| 文件 | 改动职责 |
| --- | --- |
| torch_npu/utils/_inductor.py | 注册 NPU graphsafe RNG，适配 seed/offset，放行 NPU functionalized RNG |
| torch_npu/_inductor/_npu_meta_registration.py | 处理 NPU GeneratorState 的 Inductor fallback 物化 |
| torch_npu/utils/_dynamo.py | 在 lazy Inductor 初始化阶段安装 RNG wrapper |
| test/_inductor/triton_experimental/test_graphsafe_rng_replay.py | 验证 graphsafe RNG、seed/offset、functionalized RNG、checkpoint/dropout |

PR 没有直接修改：

~~~text
torch/_inductor/ir.py
torch/_inductor/cudagraph_trees.py
torch/_inductor/codegen/triton.py
~~~

它是在 PTA 适配层接入上游已有的 RNG 协议。

---

## 4. 整体调用链

~~~mermaid
flowchart TD
    A[torch.compile + checkpoint + dropout] --> B[AOTAutograd 分析随机算子]
    B --> C[产生 graphsafe RNG state / GeneratorState]
    C --> D{NPU 是否注册 graphsafe RNG}
    D -- 否 --> E[NPU 无法进入 graphsafe RNG 路径]
    D -- 是 --> F[读取 NPU seed 和 offset]
    F --> G[Inductor 物化 GeneratorState]
    G --> H[NPU GeneratorState fallback 适配]
    H --> I[编译函数携带 RNG 参数]
    I --> J[执行 graphsafe RNG]
    J --> K[切换到保存的 state]
    K --> L[执行 dropout / random]
    L --> M[恢复外部 RNG state]
    M --> N[checkpoint 反向重算得到相同 mask]
~~~

可以按四个角色理解：

1. AOTAutograd：识别随机算子，决定需要保存什么状态。
2. Inductor：把 RNG state 作为编译图参数处理。
3. PTA：把上游 CUDA 假设适配成 NPU generator 接口。
4. Triton Experimental：生成并执行最终计算 kernel，不负责 RNG state 生命周期。

---

## 5. torch_npu/utils/_inductor.py 改了什么

### 5.1 注册 NPU graphsafe RNG

函数：

~~~text
torch_npu/utils/_inductor.py::patch_register_graphsafe_run_with_rng_state_op
~~~

核心逻辑：

~~~python
from torch._functorch._aot_autograd.utils import (
    register_graphsafe_rng_device_type,
)

graphsafe_run_with_rng_state = getattr(
    rng_prims, "graphsafe_run_with_rng_state", None
)

if graphsafe_run_with_rng_state is not None:
    register_graphsafe_rng_device_type("npu")
~~~

原问题是：上游 graphsafe RNG 默认围绕 CUDA 建立，遇到 GeneratorState(device="npu") 时，NPU 可能无法进入状态保存和恢复链路。

这段代码把 NPU 注册为 graphsafe RNG 设备：

~~~text
AOTAutograd：NPU 可以参与 graphsafe RNG state 管理
~~~

使用 getattr 检查接口，是为了兼容没有 graphsafe RNG 接口的旧 PyTorch 版本。

### 5.2 适配 NPU seed/offset helper

函数：

~~~text
torch_npu/utils/_inductor.py::patch_rng_state_helper_for_npu
~~~

上游类名是：

~~~text
torch._prims_common.CUDARngStateHelper
~~~

当前上游 helper 仍然使用 CUDA 命名。PR 复用这个 helper 的结构，只替换 NPU 场景下的三个方法。

#### 读取 state

~~~python
def get_torch_state_as_tuple(fake_mode=contextlib.nullcontext()):
    if not _should_use_npu_rng_state():
        return original_get_torch_state_as_tuple(fake_mode)

    with fake_mode:
        seed = torch.tensor(torch.npu.initial_seed())
        offset = torch.tensor(torch.npu._get_rng_state_offset())
        return seed, offset
~~~

调用链：

~~~text
torch.npu.initial_seed()
        ↓
torch.npu._get_rng_state_offset()
        ↓
(seed, offset)
        ↓
AOTAutograd / functionalized RNG 保存
~~~

fake_mode 用于防止 FakeTensor 编译阶段推进真实 NPU RNG。

#### 恢复 state

~~~python
def set_torch_state_tensor(seed, offset):
    if not _should_use_npu_rng_state():
        return original_set_torch_state_tensor(seed, offset)

    seed_portion = seed.reshape([1]).view(torch.uint8)
    offset_portion = offset.reshape([1]).view(torch.uint8)
    new_state = torch.cat([seed_portion, offset_portion])
    torch.npu.set_rng_state(new_state)
~~~

这里把显式 seed 和 offset 重新组合成 NPU 接口能接受的 state，然后恢复 NPU RNG。

#### 推进 offset

~~~python
def align_npu_offset(offset):
    return ((int(offset) + 3) // 4) * 4

def set_new_offset(relative_offset):
    if not _should_use_npu_rng_state():
        return original_set_new_offset(relative_offset)

    torch.npu._set_rng_state_offset(
        align_npu_offset(relative_offset.item())
    )
~~~

随机 kernel 消耗随机数后，需要推进 offset。当前代码按 4 对齐：

~~~text
offset = 5  -> 8
offset = 8  -> 8
offset = 9  -> 12
~~~

### 5.3 放行 NPU functionalized RNG decomposition

函数：

~~~text
torch_npu/utils/_inductor.py::patch_functionalized_rng_decomp_for_npu
~~~

核心逻辑：

~~~python
def throw_on_non_cuda(device):
    if device.type == "npu":
        return
    return original_throw_on_non_cuda(device)
~~~

原来非 CUDA 设备会被拒绝。现在变成：

~~~text
CUDA -> 保持上游行为
NPU  -> 允许进入 functionalized RNG decomposition
CPU/其他设备 -> 保持原有检查
~~~

如果没有这一步，NPU 在 RNG decomposition 入口就会失败，后面的 seed/offset 链路不会执行。

### 5.4 修正 functionalized RNG wrapper 参数顺序

函数：

~~~text
torch_npu/utils/_inductor.py::patch_functionalized_rng_wrapper_for_npu_graphsafe
~~~

它修改上游的：

~~~text
torch._functorch._aot_autograd.runtime_wrappers.FunctionalizedRngRuntimeWrapper
~~~

编译前插入 seed/offset：

~~~python
seed, offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
insert_at = len(flat_args) - fw_metadata.num_graphsafe_rng_states
flat_args[insert_at:insert_at] = [seed, offset]
~~~

参数从：

~~~text
[普通输入, GeneratorState]
~~~

变成：

~~~text
[普通输入, seed, offset, GeneratorState]
~~~

运行时重新调整顺序：

~~~python
seed_offset = runtime_args[-2:]
del runtime_args[-2:]

insert_at = len(runtime_args) - num_graphsafe_rng_states
runtime_args[insert_at:insert_at] = seed_offset

return compiled_fn(runtime_args)
~~~

这保证：

~~~text
编译阶段的参数布局 = 运行时的参数布局
~~~

否则可能出现 RNG 参数错位。

### 5.5 Patch 安装

文件末尾安装：

~~~python
patch_register_run_and_save_rng_state_op()
patch_register_run_with_rng_state_op()
patch_register_graphsafe_run_with_rng_state_op()
patch_rng_state_helper_for_npu()
patch_functionalized_rng_decomp_for_npu()
patch_philox_rand_offset()
patch_register_philox_rand()
patch_rng_prims_device()
~~~

通过 _npu_patched 标志避免重复安装。

---

## 6. torch_npu/_inductor/_npu_meta_registration.py 改了什么

### 6.1 原问题

上游 Inductor 处理 GeneratorState 时默认访问：

~~~python
torch.cuda.default_generators[device_index]
~~~

但 NPU 编译图里可能出现：

~~~python
GeneratorState(device="npu")
~~~

继续走 CUDA 逻辑会触发 CUDA-only 断言或设备不匹配。

### 6.2 _npu_fix_generator_state_fallback

函数：

~~~text
torch_npu/_inductor/_npu_meta_registration.py::_npu_fix_generator_state_fallback
~~~

#### 遍历嵌套参数

~~~python
new_args = pytree.tree_map(replace_npu_generator_state, args)
new_kwargs = pytree.tree_map(replace_npu_generator_state, kwargs)
~~~

GeneratorState 可能嵌套在 list、tuple、dict 等 pytree 结构中，所以使用 pytree 递归处理。

#### clone NPU generator

~~~python
def clone_npu_generator_state(device):
    if device.type != "npu":
        return None

    device_index = (
        torch.npu.current_device()
        if device.index is None
        else device.index
    )
    return (
        device_index,
        get_default_generator(device_index).clone_state(),
    )
~~~

真正的随机状态来源是 NPU default generator，不是 CUDA generator。

#### 临时构造 CUDA sentinel

~~~python
return ir.GeneratorState(
    name=x.name,
    device=torch.device("cuda", device_index),
)
~~~

这里不是把 NPU 随机数改成 CUDA 生成，而是把 NPU state 临时转换为上游 Inductor 能识别的形状，借用已有的参数物化逻辑。

#### 临时替换 CUDA generator 表

~~~python
old_default_generators = torch.cuda.default_generators
default_generators = list(old_default_generators)

for device_index, generator_state in npu_generator_states.items():
    default_generators[device_index] = generator_state

torch.cuda.default_generators = tuple(default_generators)
~~~

上游代码处理临时 CUDA sentinel 时，就能取到对应的 NPU generator clone state。

#### finally 恢复原表

~~~python
try:
    return orig(cls, kernel, *new_args, **new_kwargs)
finally:
    torch.cuda.default_generators = old_default_generators
~~~

无论成功还是异常，都恢复原来的 CUDA generator 表。

### 6.3 这段代码的定位

这是 PTA 当前版本的兼容桥，不是最终的上游解耦形态：

~~~text
NPU GeneratorState
    ↓
临时适配为上游可处理的 state
    ↓
复用 Inductor 参数物化流程
    ↓
finally 恢复全局表
~~~

未来如果 PyTorch 提供 backend RNG hook，就可以删除 CUDA sentinel，直接调用 NPU backend 的 generator state 接口。

---

## 7. torch_npu/utils/_dynamo.py 改了什么

位置：

~~~text
torch_npu/utils/_dynamo.py::_lazy_inductor_setup
~~~

PR 增加：

~~~python
from torch_npu.utils._inductor import (
    patch_functionalized_rng_wrapper_for_npu_graphsafe,
)

patch_functionalized_rng_wrapper_for_npu_graphsafe()
~~~

原因是 Inductor、Dynamo 和 torch_npu 初始化之间存在导入依赖。过早导入 runtime wrapper 容易产生循环导入。

实际时机：

~~~text
用户调用 torch.compile
        ↓
触发 _lazy_inductor_setup
        ↓
安装 functionalized RNG wrapper patch
~~~

---

## 8. 测试文件逐个解释

测试文件：

~~~text
test/_inductor/triton_experimental/test_graphsafe_rng_replay.py
~~~

### 8.1 graphsafe state 切换和恢复

测试函数：

~~~python
test_graphsafe_run_with_rng_state_replays_and_restores_npu_state
~~~

先记录 seed 为 58 时的 state：

~~~python
torch.npu.manual_seed(58)
generator = torch.npu._get_generator(device)
rng_state = generator.clone_state()

expected_first = dropout(x)
expected_second = dropout(x)
~~~

然后切换到另一个状态，再使用之前保存的 rng_state：

~~~python
actual_first = graphsafe_run_with_rng_state(
    dropout, x, rng_state=rng_state
)
actual_second = graphsafe_run_with_rng_state(
    dropout, x, rng_state=rng_state
)
~~~

使用 rtol=0、atol=0 的断言验证：

1. 同一个 state 得到同一段 dropout 结果。
2. graphsafe 执行结束后，外部 ambient RNG state 没有被破坏。

### 8.2 seed/offset helper

测试函数：

~~~python
test_philox_seed_offset_helpers_are_npu_aware
~~~

验证：

~~~python
assert getattr(
    rng_prims.CUDARngStateHelper,
    "_npu_patched",
    False,
)

assert decompositions_for_rng.throw_on_non_cuda(
    torch.device("npu")
) is None
~~~

同时验证：

~~~python
assert int(rng_prims.philox_rand_offset((2, 3, 4)).item()) == 24
assert int(rng_prims.philox_rand_offset((7,)).item()) == 7
~~~

表示随机 kernel 处理多少元素，就需要推进相应 offset。

### 8.3 functionalized RNG 同 seed 重放

测试函数：

~~~python
test_functionalized_seed_offset_replays_in_triton_experimental
~~~

被编译函数连续生成两个随机张量：

~~~python
def fn(t):
    first = torch.rand_like(t)
    second = torch.rand_like(t)
    return first, second
~~~

两次设置 seed 2026，并要求输出逐元素完全一致：

~~~python
torch.testing.assert_close(
    first_a.cpu(), replay_a.cpu(), rtol=0, atol=0
)
torch.testing.assert_close(
    first_b.cpu(), replay_b.cpu(), rtol=0, atol=0
)
~~~

同时要求连续两次随机调用不相同，证明 offset 确实推进。

### 8.4 checkpoint + dropout + backward

测试函数：

~~~python
test_checkpoint_dropout_preserves_rng_state_in_triton_experimental
~~~

checkpoint 函数：

~~~python
def _checkpoint_dropout(x):
    def inner(t):
        return F.dropout(t, p=0.4, training=True)

    return checkpoint(
        inner,
        x,
        use_reentrant=False,
        preserve_rng_state=True,
    )
~~~

编译并反向：

~~~python
opt = torch.compile(
    _checkpoint_dropout,
    backend="inductor",
    options={"npu_backend": "triton_experimental"},
)

out = opt(x)
out.sum().backward()
~~~

测试根据输出和梯度反推出 mask，然后比较：

~~~python
assert torch.equal(mask_from_output, mask_from_grad)
~~~

这说明 forward 和 backward 重算使用了同一张 mask。

测试还用相同 seed 重复运行 8x8 和 4x16 两种 shape，检查 output 和 grad 逐元素一致。

### 8.5 测试边界

实际覆盖：

~~~text
graphsafe RNG state 切换和恢复
seed/offset helper
functionalized RNG 同 seed 重放
checkpoint + dropout + backward
两个不同 shape
~~~

需要注意：

1. PR 正文提到 optimizer.step，但新增测试实际主要覆盖 forward/backward 和 mask 一致性，没有完整覆盖优化器更新。
2. 测试文件名包含 triton_experimental，但它验证的核心是 NPU Inductor RNG 状态协议。
3. 这些测试不等于真实 NPUGraph/ACLGraph capture/replay 测试。

---

## 9. 一次完整运行时过程

### 阶段 1：初始化

入口：

~~~text
torch_npu.utils._inductor
torch_npu.utils._dynamo::_lazy_inductor_setup
~~~

安装：

~~~text
NPU graphsafe RNG 注册
NPU seed/offset helper
NPU RNG decomposition 放行
functionalized RNG wrapper
GeneratorState fallback
~~~

### 阶段 2：AOTAutograd 分析

AOTAutograd 发现 checkpoint 内有随机算子，需要记录：

~~~text
前向执行前的 NPU seed
前向执行前的 NPU offset
GeneratorState(device="npu")
~~~

### 阶段 3：functionalized RNG

随机操作被转换为显式 seed/offset 形式，PTA patch 放行 NPU，并把 seed/offset 插入编译参数。

### 阶段 4：Inductor 物化

遇到 NPU GeneratorState 时：

~~~text
clone NPU generator state
临时适配成上游可处理的 state
调用原始 Inductor process_kernel
finally 恢复全局 generator 表
~~~

### 阶段 5：第一次前向

~~~text
state_before -> dropout -> state_after
~~~

checkpoint 保存重算所需的 state，而不是保存所有中间激活。

### 阶段 6：反向重算

~~~text
保存当前 ambient state
切换到前向保存的 state_before
重新执行 dropout
得到原来的 mask
恢复 ambient state
继续反向计算
~~~

---

## 10. 为什么不直接改 Triton 代码

随机数问题分成两层。

### Triton kernel 层

负责：

~~~text
根据输入 shape 执行计算
读取 seed/offset 或随机状态参数
生成对应的随机结果
~~~

### 编译运行时层

负责：

~~~text
什么时候保存 RNG state
什么时候恢复 RNG state
如何把 state 传入编译图
如何在 checkpoint 重算前切换 state
如何避免污染外部 RNG state
~~~

本 PR 主要解决第二层，因此改动集中在 AOTAutograd RNG helper、Inductor GeneratorState、Dynamo lazy setup 和 NPU generator 接口。

只修改 Triton kernel，并不能解决 checkpoint 反向重算时应该恢复哪个 RNG state。

---

## 11. 当前边界和后续解耦方向

### 11.1 当前已经解决

在非 ACLGraph/NPUGraph 场景下，当前 PR 打通：

~~~text
torch.compile
+ AOTAutograd
+ functionalized RNG
+ NPU seed/offset
+ GeneratorState
+ checkpoint
+ dropout
+ backward 重算
+ triton_experimental
~~~

### 11.2 当前还没有完全解决

当前 PR 没有证明以下场景已经完整支持：

~~~text
真实 NPUGraph/ACLGraph capture
真实 NPUGraph/ACLGraph replay
所有 NPU 随机算子进入设备图
NPU dropout/Bernoulli 的完整 tensor seed/offset ABI
图重放过程中 generator state 的完整登记和推进
~~~

因此测试中的 replay 主要指：

~~~text
编译训练运行时的 RNG state 重放
~~~

不能直接等同于设备图 capture/replay 已经全部支持随机算子。

### 11.3 为什么未来需要上游解耦

当前适配仍依赖上游 CUDA 命名和结构：

~~~text
CUDARngStateHelper
torch.cuda.default_generators
~~~

PTA 通过 patch 和临时 sentinel 把 NPU 接入。未来如果 PyTorch 提供通用 backend hook，理想形态是：

~~~python
BackendRngStateHelper("npu")
BackendGeneratorRegistry("npu")
~~~

届时可以：

1. 删除 NPU 对 CUDARngStateHelper 的 monkey patch。
2. 删除 torch.cuda.default_generators 临时替换逻辑。
3. 直接由 NPU backend 提供 generator state clone、restore 和 offset 推进。
4. 保留通用的 AOTAutograd/Inductor graphsafe RNG 流程。

---

## 12. 问题现象和代码修改对照

| 问题现象 | 直接原因 | PR 修改点 | 修复后的行为 |
| --- | --- | --- | --- |
| NPU 无法进入 graphsafe RNG 路径 | NPU 未注册为 graphsafe RNG 设备 | patch_register_graphsafe_run_with_rng_state_op | AOTAutograd 接受 NPU GeneratorState |
| helper 仍然读取 CUDA seed/offset | 上游 helper 面向 CUDA | patch_rng_state_helper_for_npu | 从 NPU generator 读取和恢复 state |
| functionalized RNG 拒绝 NPU | throw_on_non_cuda 对非 CUDA 报错 | patch_functionalized_rng_decomp_for_npu | NPU 进入 seed/offset decomposition |
| seed/offset 参数位置不匹配 | wrapper 编译前后参数布局不同 | patch_functionalized_rng_wrapper_for_npu_graphsafe | 编译和运行时参数顺序一致 |
| GeneratorState(device=npu) 触发 CUDA 断言 | fallback 只访问 CUDA generators | _npu_fix_generator_state_fallback | clone NPU state 并复用上游物化流程 |
| checkpoint 重算 dropout mask 不一致 | 重算前未恢复 forward RNG state | graphsafe RNG 保存/恢复链路 | forward/backward mask 一致 |

---

## 13. 推荐阅读顺序

1. test/_inductor/triton_experimental/test_graphsafe_rng_replay.py
2. torch_npu/utils/_inductor.py::patch_register_graphsafe_run_with_rng_state_op
3. torch_npu/utils/_inductor.py::patch_rng_state_helper_for_npu
4. torch_npu/utils/_inductor.py::patch_functionalized_rng_wrapper_for_npu_graphsafe
5. torch_npu/_inductor/_npu_meta_registration.py::_npu_fix_generator_state_fallback
6. torch_npu/utils/_dynamo.py::_lazy_inductor_setup
7. PyTorch 上游 torch._prims.rng_prims
8. PyTorch 上游 torch._functorch._aot_autograd.runtime_wrappers

---

## 14. 总结

这个 PR 做的不是“给 NPU 增加一个 dropout 算子”，而是补齐随机状态在编译训练链路中的生命周期：

~~~text
识别 RNG state
    ↓
保存 seed / offset
    ↓
把 GeneratorState 传入编译图
    ↓
前向执行随机算子
    ↓
checkpoint 反向重算前恢复原 state
    ↓
生成相同 dropout mask
    ↓
恢复外部 ambient RNG state
~~~

它解决的是 PTA 当前版本中 NPU 编译训练的 RNG replay 基础能力，当前验证后端是 triton_experimental。真实 NPUGraph/ACLGraph 的随机算子 capture/replay，以及上游 backend RNG 接口解耦，仍然是后续工作。

