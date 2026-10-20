# `torch_npu` 引入 FlexAttention patch 后多进程退出时报 EE1017

## 问题描述

2.7.1 分支新增 `apply_flex_attention_patch()`，在 `import torch_npu` 时提前导入 `torch.nn.attention.flex_attention`，用于绕过原生 FlexAttention 的 CUDA/CPU device 校验，使 eager 模式下 NPU tensor 可以正常运行。

该提前导入改变了多进程 `spawn` 场景下的模块初始化和资源析构时序，导致部分 stream 延迟到 `aclrtDestroyStreamForce`、`aclrtResetDevice`、`aclFinalize` 之后才析构。析构过程中调用 `rtStreamSynchronize` 访问已释放的 stream 句柄，进程退出时报 `EE1017`。

## 修改方案

- `import torch_npu` 时不再主动导入 FlexAttention。
- 通过 import hook 监听 `torch.nn.attention.flex_attention`。
- 仅当业务代码实际导入该模块后，再替换其 `_validate_device`。
- 2.7.1 版本只处理 eager 模式 device validation，不引入 autocast 注册。
- CPU 和 CUDA 设备回退到 PyTorch 原生校验逻辑，避免改变其他设备行为。

## 功能验证

未打 patch 时，eager FlexAttention 的 NPU 用例失败：

```text
ValueError: FlexAttention is only supported on CUDA or CPU devices.
Found input tensors on npu device.
```

修改后需要验证：

1. eager FlexAttention NPU 用例正常通过；
2. CPU/CUDA 原有 device 校验行为不变；
3. 多进程 `spawn` 场景退出不再出现 stream 销毁顺序异常和 EE1017。

目前 170 服务器的 Docker/containerd exec 通道异常，暂未完成远端回归验证。
